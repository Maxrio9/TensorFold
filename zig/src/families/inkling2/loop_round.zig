//! Shared Mini CUDA forward, adapted from TensorFold #4786.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const ops = @import("ops.zig");
const attention = @import("kernels.zig");
const modelopt = @import("modelopt.zig");
const exl3_experts = @import("loop_exl3_experts.zig");
const loop_math = @import("loop_math_cuda.zig");
const engine = @import("loop_engine.zig");
const Model = engine.Model;
const Kernels = engine.Kernels;
const prompt_rows = engine.prompt_rows;
const window_rows = engine.window_rows;
const max_tiles = attention.tileCount(prompt_rows, 4);
const window_tiles: usize = 2; // up to 16 rows at the Mini's three query heads per KV head
pub const tile_storage: usize = @max(2 * max_tiles, engine.max_slots * 2 * window_tiles);

pub const Chain = struct {
    slot: usize,
    position: usize,
    tokens: []const u32,
    record: bool = false,
};

const Part = struct {
    slot: u8 = 0,
    row0: u16 = 0,
    rows: u16 = 0,
    tiles: [2]u16 = @splat(0),
    record: bool = false,
};

// Also used as the graph key. All addresses depend only on this shape and model allocations.
pub const Shape = struct {
    parts: [engine.max_slots]Part = @splat(.{}),
    count: usize,
    rows: usize = 0,
    head0: usize,
    heads: usize,
    taps: bool,
};

fn tileOffset(count: usize, i: usize, kind: usize) usize {
    return (2 * i + kind) * (if (count == 1) max_tiles else window_tiles);
}

fn shapeFor(chains: []const Chain, head0: usize, heads: usize, taps: bool, context: usize, slots: usize, rows_per_pass: usize, groups: [2]u32) !Shape {
    if (chains.len == 0 or chains.len > engine.max_slots or heads > engine.round_rows) return error.BadLoopRows;
    var shape: Shape = .{ .count = chains.len, .head0 = head0, .heads = heads, .taps = taps };
    var seen: std.StaticBitSet(engine.max_slots) = .empty;
    for (chains, shape.parts[0..chains.len]) |chain, *part| {
        const n = chain.tokens.len;
        if (chain.slot >= slots or n == 0 or chain.position > context or n > context - chain.position or n > rows_per_pass - shape.rows)
            return error.BadLoopRows;
        if (seen.isSet(chain.slot)) return error.BadLoopRows;
        seen.set(chain.slot);
        if ((chain.record or chains.len > 1) and n > window_rows) return error.WindowTooWide;
        const tiles = [2]usize{ attention.tileCount(n, groups[0]), attention.tileCount(n, groups[1]) };
        const capacity = if (chains.len == 1) max_tiles else window_tiles;
        if (tiles[0] > capacity or tiles[1] > capacity) return error.TooManyTiles;
        part.* = .{
            .slot = @intCast(chain.slot),
            .row0 = @intCast(shape.rows),
            .rows = @intCast(n),
            .record = chain.record,
            .tiles = .{ @intCast(tiles[0]), @intCast(tiles[1]) },
        };
        shape.rows += n;
    }
    if ((chains.len > 1 and shape.rows > engine.round_rows) or head0 > shape.rows or heads > shape.rows - head0)
        return error.BadLoopRows;
    return shape;
}

/// Stage all inputs before capture or replay. Positions and token IDs do not key a graph.
pub fn prepare(m: *Model, s: cuda.Stream, chains: []const Chain, head0: usize, heads: usize, taps: bool) !Shape {
    const c = m.w.c.text;
    const groups = [2]u32{ c.local.q / c.local.kv, c.global.q / c.global.kv };
    const shape = try shapeFor(chains, head0, heads, taps, m.context, m.n_slots, m.promptRows(), groups);
    var ids: [prompt_rows]i32 = undefined;
    var pos: [prompt_rows]i32 = undefined;
    const seg_of: [prompt_rows]i32 = @splat(0);
    var segs: [engine.max_slots]attention.Seg = undefined;
    for (chains, shape.parts[0..chains.len], 0..) |chain, part, i| {
        const row0: usize = part.row0;
        for (chain.tokens, 0..) |token, j| {
            ids[row0 + j] = @intCast(token);
            pos[row0 + j] = @intCast(chain.position + j);
        }
        segs[i] = .{ .row0 = 0, .n = part.rows, .p0 = @intCast(chain.position), .off = 0 };
        for (groups, 0..) |group, kind| {
            var tiles: [max_tiles]attention.Tile = undefined;
            const count = try attention.tiles(segs[i..][0..1], group, &tiles);
            const at = tileOffset(chains.len, i, kind);
            try m.upload(s, m.s.tiles + at * @sizeOf(attention.Tile), std.mem.sliceAsBytes(tiles[0..count]));
        }
    }
    try m.upload(s, m.s.ids, std.mem.sliceAsBytes(ids[0..shape.rows]));
    try m.upload(s, m.s.pos, std.mem.sliceAsBytes(pos[0..shape.rows]));
    try m.upload(s, m.s.seg_of, std.mem.sliceAsBytes(seg_of[0..shape.rows]));
    try m.upload(s, m.s.segs, std.mem.sliceAsBytes(segs[0..chains.len]));
    for (shape.parts[0..shape.count]) |part| m.slots[part.slot].recorded = if (part.record) part.rows else 0;
    return shape;
}

/// The row-wise kernels share one forward. Each slot runs convolution and attention on its own state.
pub fn execute(m: *Model, k: *Kernels, s: cuda.Stream, shape: Shape) !void {
    return executePrompt(m, k, s, shape, null);
}

pub fn executePrompt(m: *Model, k: *Kernels, s: cuda.Stream, shape: Shape, images: ?@import("mini_vision.zig").Rows) !void {
    const c = m.w.c.text;
    const h: usize = c.hidden;
    const rows = shape.rows;
    const parts = shape.parts[0..shape.count];
    const selected = m.slot;
    defer m.slot = selected;
    const conv_at = m.layout.at(0).conv_offset;
    for (parts) |part| {
        m.slot = part.slot;
        if (part.record) try m.copy(s, m.slots[part.slot].backup.ptr, m.slotBase() + conv_at, m.layout.bytes - conv_at);
    }

    try ops.embed(&k.glue, s, m.s.ids, rows, m.w.embed, m.w.embed_norm, m.s.partial_a, h, c.eps);
    if (images) |p| try p.splice(m.d, s, rows, h, m.s.partial_a);
    var partial = m.s.partial_a;
    var next = m.s.partial_b;
    var n_blocks: usize = 0;
    const lp = c.loop.?;
    const taps = if (shape.taps) m.taps else null;

    for (m.w.p.visits[0..m.w.p.n_visits]) |visit| {
        if (taps) |t| {
            const channel: ?usize = switch (visit.cache_slot) {
                5 => 0,
                23 => 1,
                41 => 2,
                else => null,
            };
            if (channel) |ch| try @import("dspark_ops.zig").tap(t.function, s, partial, t.buffer, rows, h, ch);
        }
        const layer = m.w.p.at(visit.layer);
        const lw = &m.w.layers[visit.layer];
        if (visit.pass >= 2 and visit.layer == lp.core_start) {
            try m.copy(s, partial, m.s.prelude, rows * h * 2);
            n_blocks = 1; // block zero is the embedding; later passes never inherit completed blocks
        }
        const messages = visit.hasMessages();
        if (messages) {
            const core_j = visit.layer - lp.core_start;
            try loop_math.messages(&k.loop, s, m.s.bank0, m.s.bank1, m.w.cross_qm + 2 * core_j * h * 4, m.w.cross_kv + 2 * core_j * h * 4, m.s.messages, rows, 1, h, c.eps);
        }
        const nq: usize = if (layer.kind == .global) 3 else 1;
        const query = m.w.attn_queries[visit.layer];
        const blocks = m.s.blocks[0..n_blocks];
        const raw_stride = rows * h * 4;
        const norm_stride = rows * h * 2;
        if (messages) {
            if (nq == 3) {
                try loop_math.ar(&k.loop, s, m.arBlocks(blocks, partial, query, m.s.raw, 3, true, rows));
            } else {
                // A sliding layer stores one query; each gated role gets its own AR result from that query.
                for (0..3) |role| try loop_math.ar(&k.loop, s, m.arBlocks(blocks, partial, query, m.s.raw + role * raw_stride, 1, true, rows));
            }
            for (0..3) |role| {
                const ctl = lw.controllers[role];
                try loop_math.gate(&k.loop, s, m.s.raw + role * raw_stride, m.s.messages + role * raw_stride, ctl.down, ctl.forget, ctl.gain, m.s.zeta, m.s.gate_inputs, m.s.raw + role * raw_stride, rows, h, c.eps);
                const norms = [3]u64{ lw.attn.norms[role], lw.attn.norms[role], lw.attn.norms[role] };
                try ops.mix(&k.glue, s, m.mixArgs(&.{}, m.s.raw + role * raw_stride, 0, norms, m.s.normalized + role * norm_stride, 1, false, rows));
            }
        } else {
            try ops.mix(&k.glue, s, m.mixArgs(blocks, partial, query, lw.attn.norms, m.s.normalized, nq, true, rows));
        }
        const u_q = m.s.normalized;
        const u_k = if (layer.kind == .global or messages) m.s.normalized + norm_stride else u_q;
        const u_v = if (layer.kind == .global or messages) m.s.normalized + 2 * norm_stride else u_q;
        const q_width: usize = @as(usize, layer.heads.q) * layer.heads.dim;
        const kv_width: usize = @as(usize, layer.heads.kv) * layer.heads.dim;
        if (lw.attn.exl3) |a| {
            try k.projectAttention(s, a, .{ u_q, u_k, u_v }, .{ m.s.q_raw, m.s.k_raw, m.s.v_raw, m.s.gate }, rows, shape.count == 1);
        } else {
            try loop_math.projectQkv(&k.loop, s, u_q, u_k, u_v, lw.attn.wq, lw.attn.wk, lw.attn.wv, m.s.q_raw, m.s.k_raw, m.s.v_raw, rows, h, q_width, kv_width);
            try ops.rowdot(&k.glue, s, u_q, lw.attn.gate, m.s.gate, rows, q_width, h, true, false);
        }
        try ops.rowdot(&k.glue, s, u_q, lw.attn.wr, m.s.rel_raw, rows, layer.heads.q * c.d_rel, h, true, false);
        try ops.rowdot(&k.glue, s, u_q, lw.attn.sink, m.s.sink, rows, layer.heads.q, h, true, false);

        const geometry = m.layerGeometry(visit.layer);
        const dynamic = layer.kind == .global;
        const conv_width = @as(usize, layer.heads.kv) * config.conv_taps;
        if (dynamic) for (0..2) |kind| {
            const dyn_out = if (kind == 0) m.s.k_dynamic else m.s.v_dynamic;
            try ops.rowdot(&k.glue, s, if (kind == 0) u_k else u_v, if (kind == 0) lw.attn.k_dynamic else lw.attn.v_dynamic, dyn_out, rows, conv_width, h, false, false);
            try loop_math.roundBf16(&k.loop, s, dyn_out, dyn_out, rows * conv_width);
        };
        const stride = attention.slotsFor(geometry, m.context);
        const part_capacity = m.promptRows() * layer.heads.q * stride;
        const prep: attention.Prep = .{ .qk_clip = c.qk_clip, .rel_clip = c.rel_clip, .floor = @floatFromInt(c.log_floor), .alpha = c.log_alpha, .eps = c.eps };
        for (parts, 0..) |part, i| {
            m.slot = part.slot;
            const r0: usize = part.row0;
            const q_off = r0 * q_width * 2;
            const kv_off = r0 * kv_width * 2;
            const rel_off = r0 * layer.heads.q * c.d_rel * 2;
            const head_off = r0 * layer.heads.q * 2;
            for (0..2) |kind| try ops.conv(&k.glue, s, .{
                .x = (if (kind == 0) m.s.k_raw else m.s.v_raw) + kv_off,
                .state = m.convOf(visit.cache_slot, kind),
                .w = if (kind == 0) lw.attn.k_conv else lw.attn.v_conv,
                .dyn = if (dynamic) (if (kind == 0) m.s.k_dynamic else m.s.v_dynamic) + r0 * conv_width * 4 else 0,
                .out = (if (kind == 0) m.s.k_conv else m.s.v_conv) + kv_off,
                .hist = if (part.record) m.histOf(visit.cache_slot, kind) else 0,
                .x_bf16 = 1,
                .heads = @intCast(layer.heads.kv),
                .c = @intCast(kv_width),
                .rows = part.rows,
                .mode = 0,
            }, 1);
            const kind: usize = @intFromBool(dynamic);
            const r: attention.Rows = .{
                .rows = part.rows,
                .heads = layer.heads.q,
                .kv_heads = layer.heads.kv,
                .pos = m.s.pos + r0 * 4,
                .seg_of = m.s.seg_of,
                .segs = m.s.segs + i * @sizeOf(attention.Seg),
                .tiles = m.s.tiles + tileOffset(shape.count, i, kind) * @sizeOf(attention.Tile),
                .n_tiles = part.tiles[kind],
                .last_pos = @intCast(m.context - 1),
                .sliding_prompt = lw.attn.exl3 != null and shape.count == 1 and !part.record and
                    part.rows >= attention.sliding_prompt_min_rows,
            };
            const cache = m.cacheOf(visit.cache_slot);
            try attention.prep(&k.attn, s, r, geometry, prep, .{ m.s.q_raw + q_off, m.s.k_conv + kv_off, m.s.v_conv + kv_off, m.s.rel_raw + rel_off }, .{ lw.attn.q_scale, lw.attn.rel_scale }, m.s.q_prepared, m.s.rel_prepared, cache);
            try attention.relTable(&k.attn, s, r, m.s.rel_prepared, lw.attn.rel_proj, m.s.rel_table);
            try attention.attend(&k.attn, s, r, geometry, stride, part_capacity, m.s.q_prepared, cache, m.s.rel_table, m.s.part_o, m.s.part_ml, m.s.sink + head_off, m.s.gate + q_off, m.s.attn_heads + q_off);
        }
        if (lw.attn.exl3) |a|
            try Kernels.project(&k.dense.?, s, a.o, m.s.attn_heads, rows, m.s.attn_out)
        else
            try ops.rowdot(&k.glue, s, m.s.attn_heads, lw.attn.wo, m.s.attn_out, rows, h, q_width, true, false);

        if (layer.boundary) {
            try m.copy(s, m.s.blocks[n_blocks], partial, rows * h * 2);
            n_blocks += 1;
        }
        for (parts) |part| {
            m.slot = part.slot;
            const r0: usize = part.row0;
            try ops.conv(&k.glue, s, .{
                .x = m.s.attn_out + r0 * h * 2,
                .state = m.convOf(visit.cache_slot, 2),
                .w = lw.attn_conv,
                .base = if (layer.boundary) 0 else partial + r0 * h * 2,
                .resid = m.s.t + r0 * h * 4,
                .hist = if (part.record) m.histOf(visit.cache_slot, 2) else 0,
                .x_bf16 = 1,
                .c = @intCast(h),
                .rows = part.rows,
                .mode = 1,
            }, 1);
        }

        const mlp_query = m.w.mlp_queries[visit.layer];
        const mlp_blocks = m.s.blocks[0..n_blocks];
        if (messages) {
            try loop_math.ar(&k.loop, s, m.arBlocks(mlp_blocks, m.s.t, mlp_query, m.s.raw, 1, false, rows));
            const ctl = lw.controllers[3];
            try loop_math.gate(&k.loop, s, m.s.raw, m.s.messages + 3 * raw_stride, ctl.down, ctl.forget, ctl.gain, m.s.zeta, m.s.gate_inputs, m.s.raw, rows, h, c.eps);
            try ops.mix(&k.glue, s, m.mixArgs(&.{}, m.s.raw, 0, .{ lw.mlp_norm, lw.mlp_norm, lw.mlp_norm }, m.s.normalized, 1, false, rows));
        } else {
            try ops.mix(&k.glue, s, m.mixArgs(mlp_blocks, m.s.t, mlp_query, .{ lw.mlp_norm, lw.mlp_norm, lw.mlp_norm }, m.s.normalized, 1, false, rows));
        }
        try ops.rowdot(&k.glue, s, m.s.normalized, lw.router, m.s.route_scores, rows, c.experts, h, false, true);
        try ops.topk(&k.glue, s, m.s.route_scores, lw.router_bias, m.s.route_pick, m.s.route_weights, rows, c.experts, c.topk, c.route_scale);
        switch (lw.experts) {
            .modelopt => |e| try modelopt.run(&k.moe.modelopt, s, e, .{ .gate = c.gate_cap, .up = c.up_cap }, m.s.normalized, h, m.s.route_pick, m.s.route_weights, rows, c.topk, m.s.moe_out, m.s.moe_scratch),
            .exl3 => |e| try exl3_experts.run(&k.moe.exl3, s, e.layer, .{ .gate = c.gate_cap, .up = c.up_cap }, e.bits, m.s.normalized, h, .bf16, m.s.route_pick, m.s.route_weights, rows, m.s.moe_out, m.s.exl3_scratch.?),
        }
        for (parts) |part| {
            m.slot = part.slot;
            const r0: usize = part.row0;
            try ops.conv(&k.glue, s, .{
                .x = m.s.moe_out + r0 * h * (if (m.w.c.exl3) @as(usize, 4) else 2),
                .state = m.convOf(visit.cache_slot, 3),
                .w = lw.mlp_conv,
                .resid = m.s.t + r0 * h * 4,
                .out = next + r0 * h * 2,
                .hist = if (part.record) m.histOf(visit.cache_slot, 3) else 0,
                .x_bf16 = @intFromBool(!m.w.c.exl3),
                .round_in = @intFromBool(m.w.c.exl3),
                .c = @intCast(h),
                .rows = part.rows,
                .mode = 2,
            }, 1);
        }
        std.mem.swap(u64, &partial, &next);

        if (visit.layer + 1 == lp.core_start) try m.copy(s, m.s.prelude, partial, rows * h * 2);
        if (visit.pass > 0 and visit.pass < lp.passes and visit.layer + 1 == lp.core_end) {
            try m.copy(s, m.s.bank0, m.s.blocks[n_blocks - 1], rows * h * 2);
            try m.copy(s, m.s.bank1, partial, rows * h * 2);
        }
    }

    const heads = shape.heads;
    if (heads == 0 and taps == null) return;
    try ops.mix(&k.glue, s, m.mixArgs(m.s.blocks[0..n_blocks], partial, m.w.output_query, .{ m.w.norm, m.w.norm, m.w.norm }, m.s.hidden, 1, true, rows));
    if (taps) |t| try @import("dspark_ops.zig").tap(t.function, s, m.s.hidden, t.buffer, rows, h, 3);
    if (heads == 0) return;
    try ops.rowdot(&k.glue, s, m.s.hidden + shape.head0 * h * 2, m.w.unembed, m.s.logits, heads, c.unpadded_vocab, h, false, false);
    try loop_math.roundBf16(&k.loop, s, m.s.logits, m.s.logits, heads * c.unpadded_vocab);
    try ops.softcap(&k.glue, s, m.s.logits, heads * c.unpadded_vocab, c.logit_cap);
}

test "shared windows retain slot, row offset, rewind and tile shape" {
    var inputs = [_]Chain{
        .{ .slot = 3, .position = 3030, .tokens = &.{11} },
        .{ .slot = 1, .position = 22, .tokens = &.{ 41, 42, 43 }, .record = true },
    };
    const shape = try shapeFor(&inputs, 0, 4, true, 4096, 4, prompt_rows, .{ 3, 3 });
    try std.testing.expectEqual(@as(usize, 4), shape.rows);
    try std.testing.expectEqual(Part{ .slot = 3, .rows = 1, .tiles = .{ 1, 1 } }, shape.parts[0]);
    try std.testing.expectEqual(Part{ .slot = 1, .row0 = 1, .rows = 3, .tiles = .{ 1, 1 }, .record = true }, shape.parts[1]);
    inputs[0].position += 63;
    inputs[1].tokens = &.{ 8, 9, 10 };
    try std.testing.expectEqual(shape, try shapeFor(&inputs, 0, 4, true, 4096, 4, prompt_rows, .{ 3, 3 }));
    inputs[1].slot = 3;
    try std.testing.expectError(error.BadLoopRows, shapeFor(&inputs, 0, 4, true, 4096, 4, prompt_rows, .{ 3, 3 }));
}

test "a 4096-row EXL3 pass fits the prepared attention tiles" {
    const ids: [prompt_rows]u32 = @splat(0);
    const chain: Chain = .{ .slot = 0, .position = 4096, .tokens = &ids };
    const shape = try shapeFor(&.{chain}, ids.len - 1, 1, true, 8192, 1, prompt_rows, .{ 3, 3 });
    try std.testing.expectEqual(@as(u16, 192), shape.parts[0].tiles[0]);
    try std.testing.expectEqual(@as(usize, 4096), shape.rows);
}

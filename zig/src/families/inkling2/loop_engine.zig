//! One-rank, text-only Inkling 2 Mini: the complete 48-visit CUDA forward.
const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const loop = @import("loop.zig");
const weights_ = @import("loop_weights.zig");
const ops = @import("ops.zig");
const attention = @import("kernels.zig");
pub const round = @import("loop_round.zig");
const modelopt = @import("modelopt.zig");
const exl3_experts = @import("loop_exl3_experts.zig");
const loop_math = @import("loop_math_cuda.zig");

pub const prompt_rows: usize = 2048;
pub const window_rows: usize = 16;
pub const round_rows = 128;

pub const Kernels = @import("loop_kernels.zig").Kernels;

const Scratch = @import("loop_scratch.zig").Scratch;

/// The cache, rewind sites, and convolution backup and history are private to each stream.
const Slot = struct {
    cache: cuda.DeviceBuffer,
    sites: cuda.DeviceBuffer,
    backup: cuda.DeviceBuffer,
    hist: cuda.DeviceBuffer,
    recorded: usize = 0,

    pub fn deinit(slot: *Slot) void {
        for ([_]*cuda.DeviceBuffer{ &slot.hist, &slot.backup, &slot.sites, &slot.cache }) |b| b.free();
    }
};
pub const max_slots = 16;

pub fn slotAllocs(layout: *const loop.CacheLayout, n_visits: u32) [4]usize {
    const conv = layout.bytes - layout.at(0).conv_offset;
    return .{ @intCast(layout.bytes), @as(usize, n_visits) * 4 * @sizeOf(ops.RewindSite), conv, conv / 3 * window_rows };
}

pub const Model = struct {
    pub const Chain = round.Chain;

    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    w: *const weights_.Weights,
    context: usize,
    layout: loop.CacheLayout,
    slots: [max_slots]Slot = undefined,
    n_slots: usize = 0,
    slot: usize = 0, // the slot forwards, snapshots and rewinds use; forwards share the scratch
    taps: ?struct { function: cuda.Function, buffer: u64 } = null,
    capture_taps: bool = false,
    lazy_head: ?@import("loop_lazy_head.zig").Head = null,
    s: Scratch = undefined,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,

    pub fn slotBase(m: *const Model) u64 {
        return m.slots[m.slot].cache.ptr;
    }

    /// Allocates slots until `n` exist, each with zeroed conv states and its own rewind sites.
    pub fn addSlots(m: *Model, s: cuda.Stream, n: usize) !void {
        if (n > max_slots) return error.TooManySlots;
        const keep = m.slot;
        defer m.slot = keep;
        const kv: usize = @as(usize, m.w.c.text.local.kv) * m.w.c.text.local.dim;
        var sites: [config.max_visits * 4]ops.RewindSite = undefined;
        while (m.n_slots < n) {
            var slot = try m.newSlot();
            errdefer slot.deinit();
            m.slots[m.n_slots] = slot;
            m.slot = m.n_slots;
            for (0..m.w.p.n_visits) |v| for (0..4) |site| {
                const c = if (site < 2) kv else m.w.c.text.hidden;
                const hist = m.histOf(@intCast(v), site) - slot.hist.ptr;
                sites[v * 4 + site] = .{ .state = m.convOf(@intCast(v), site), .hist = @intCast(hist), .backup = @intCast(m.backupOf(@intCast(v), site)), .c = @intCast(c) };
            };
            try m.upload(s, slot.sites.ptr, std.mem.sliceAsBytes(sites[0 .. m.w.p.n_visits * 4]));
            try m.reset(s);
            try s.synchronize();
            m.n_slots += 1;
        }
    }

    /// Allocate one slot before publishing it to the serving backend.
    pub fn newSlot(m: *const Model) !Slot {
        const sizes = slotAllocs(&m.layout, m.w.p.n_visits);
        var cache = try cuda.DeviceBuffer.alloc(m.d, sizes[0]);
        errdefer cache.free();
        var sites = try cuda.DeviceBuffer.alloc(m.d, sizes[1]);
        errdefer sites.free();
        var backup = try cuda.DeviceBuffer.alloc(m.d, sizes[2]);
        errdefer backup.free();
        return .{ .cache = cache, .sites = sites, .backup = backup, .hist = try cuda.DeviceBuffer.alloc(m.d, sizes[3]) };
    }

    fn scratchAllocation(bytes: usize) usize {
        return @max(bytes, 256);
    }

    fn alloc(m: *Model, bytes: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(m.d, scratchAllocation(bytes));
        errdefer b.free();
        try m.buffers.append(m.gpa, b);
        return b.ptr;
    }

    pub fn upload(m: *const Model, s: cuda.Stream, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(m.d, dst, bytes, s);
    }

    pub fn copy(m: *const Model, s: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        try m.d.check(m.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, s.handle), "cuMemcpyDtoDAsync");
    }

    fn zeros(m: *const Model, s: cuda.Stream, ptr: u64, bytes: usize) !void {
        try m.d.check(m.d.api.cuMemsetD8Async(ptr, 0, bytes, s.handle), "cuMemsetD8Async");
    }

    pub fn layerGeometry(m: *const Model, layer: u32) attention.Layer {
        return if (config.isGlobal(layer))
            attention.Layer.global(@intCast(m.context))
        else
            attention.Layer.sliding(m.w.c.text.window, @intCast(@min(prompt_rows, m.context)));
    }

    pub fn cacheOf(m: *const Model, visit: u32) attention.Cache {
        const slot = m.layout.at(visit);
        const base = m.slotBase() + slot.kv_offset;
        return .{ .k = base, .v = base + slot.kv_bytes / 2, .format = m.layout.format };
    }

    pub fn convOf(m: *const Model, visit: u32, site: usize) u64 {
        const c = m.w.c.text;
        const kv: usize = @as(usize, c.local.kv) * c.local.dim;
        const h: usize = c.hidden;
        const widths = [4]usize{ kv, kv, h, h };
        var at = m.slotBase() + m.layout.at(visit).conv_offset;
        for (0..site) |i| at += 3 * widths[i] * 4;
        return at;
    }

    pub fn histOf(m: *const Model, visit: u32, site: usize) u64 {
        const c = m.w.c.text;
        const kv: usize = @as(usize, c.local.kv) * c.local.dim;
        const h: usize = c.hidden;
        const widths = [4]usize{ kv, kv, h, h };
        var at = m.slots[m.slot].hist.ptr + visit * window_rows * (2 * kv + 2 * h) * 4;
        for (0..site) |i| at += window_rows * widths[i] * 4;
        return at;
    }

    fn backupOf(m: *const Model, visit: u32, site: usize) usize {
        return m.convOf(visit, site) - (m.slotBase() + m.layout.at(0).conv_offset);
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, w: *const weights_.Weights, context: usize, format: loop.CacheFormat) !Model {
        const layout = try loop.CacheLayout.init(&w.c.text, &w.p, @intCast(context), format);
        var m: Model = .{
            .gpa = gpa,
            .d = d,
            .w = w,
            .context = context,
            .layout = layout,
        };
        errdefer m.deinitBuffers();
        try m.initScratch();
        try m.addSlots(s, 1);
        return m;
    }

    pub fn initScratch(m: *Model) !void {
        const expert = switch (m.w.layers[0].experts) {
            .modelopt => null,
            .exl3 => |e| e.layer,
        };
        m.s = try scratchAlloc(m, m.w.c.text, m.context, expert);
    }

    pub fn planScratch(c: config.Text, context: usize, expert: ?cuda.exl3_experts.Layer) !usize {
        const Counter = struct {
            bytes: usize = 0,
            fn alloc(self: *@This(), bytes: usize) !u64 {
                self.bytes += scratchAllocation(bytes);
                return 0;
            }
        };
        var counter: Counter = .{};
        _ = try scratchAlloc(&counter, c, context, expert);
        return counter.bytes;
    }

    fn scratchAlloc(buffers: anytype, c: config.Text, context: usize, expert: ?cuda.exl3_experts.Layer) !Scratch {
        const h: usize = c.hidden;
        const q: usize = @as(usize, c.global.q) * c.global.dim;
        const kv: usize = @as(usize, c.local.kv) * c.local.dim;
        const rows = prompt_rows;
        const rel_rows = rows * c.global.q * attention.extent;
        const local = attention.Layer.sliding(c.window, @intCast(@min(prompt_rows, context)));
        const global = attention.Layer.global(@intCast(context));
        const local_slots = attention.slotsFor(local, context);
        const global_slots = attention.slotsFor(global, context);
        const part_slots = @max(rows * c.local.q * local_slots, rows * c.global.q * global_slots);
        const expert_scratch = try buffers.alloc(if (expert) |e|
            try exl3_experts.scratchBytes(e, rows, c.topk)
        else
            try modelopt.scratchBytes(rows, c.topk, h, c.expert_inter));
        var scratch: Scratch = .{
            .ids = try buffers.alloc(rows * 4),
            .pos = try buffers.alloc(rows * 4),
            .seg_of = try buffers.alloc(rows * 4),
            .segs = try buffers.alloc(max_slots * @sizeOf(attention.Seg)),
            .tiles = try buffers.alloc(round.tile_storage * @sizeOf(attention.Tile)),
            .partial_a = try buffers.alloc(rows * h * 2),
            .partial_b = try buffers.alloc(rows * h * 2),
            .prelude = try buffers.alloc(rows * h * 2),
            .bank0 = try buffers.alloc(rows * h * 2),
            .bank1 = try buffers.alloc(rows * h * 2),
            .raw = try buffers.alloc(3 * rows * h * 4),
            .zeta = try buffers.alloc(rows * loop_math.rank * 2),
            .gate_inputs = if (expert != null) try buffers.alloc(rows * 2 * h * 2) else 0,
            .messages = try buffers.alloc(4 * rows * h * 4),
            .normalized = try buffers.alloc(3 * rows * h * 2),
            .t = try buffers.alloc(rows * h * 4),
            .q_raw = try buffers.alloc(rows * q * 2),
            .rel_raw = try buffers.alloc(rows * c.global.q * c.d_rel * 2),
            .sink = try buffers.alloc(rows * c.global.q * 2),
            .gate = try buffers.alloc(rows * q * 2),
            .k_raw = try buffers.alloc(rows * kv * 2),
            .v_raw = try buffers.alloc(rows * kv * 2),
            .k_conv = try buffers.alloc(rows * kv * 2),
            .v_conv = try buffers.alloc(rows * kv * 2),
            .k_dynamic = try buffers.alloc(rows * c.global.kv * config.conv_taps * 4),
            .v_dynamic = try buffers.alloc(rows * c.global.kv * config.conv_taps * 4),
            .q_prepared = try buffers.alloc(rows * q * 2),
            .rel_prepared = try buffers.alloc(rows * c.global.q * c.d_rel * 2),
            .rel_table = try buffers.alloc(rel_rows * 2),
            .part_o = try buffers.alloc(part_slots * attention.head_dim * 4),
            .part_ml = try buffers.alloc(part_slots * 8),
            .attn_heads = try buffers.alloc(rows * q * 2),
            .attn_out = try buffers.alloc(rows * h * 2),
            .route_scores = try buffers.alloc(rows * c.experts * 4),
            .route_pick = try buffers.alloc(rows * c.experts * 4),
            .route_weights = try buffers.alloc(rows * c.experts * 4),
            .moe_out = try buffers.alloc(rows * h * (if (expert != null) @as(usize, 4) else 2)),
            .moe_scratch = expert_scratch,
            .exl3_scratch = if (expert) |e| try cuda.exl3_experts.scratch(expert_scratch, e, rows, c.topk) else null,
            .hidden = try buffers.alloc(rows * h * 2),
            .logits = try buffers.alloc(round_rows * c.unpadded_vocab * 4),
        };
        for (0..ops.max_blocks) |i| scratch.blocks[i] = try buffers.alloc(rows * h * 2);
        return scratch;
    }

    fn deinitBuffers(m: *Model) void {
        for (m.buffers.items) |*b| b.free();
        m.buffers.deinit(m.gpa);
    }

    pub fn deinit(m: *Model) void {
        if (m.lazy_head) |*head| head.deinit();
        m.deinitBuffers();
        for (m.slots[0..m.n_slots]) |*slot| slot.deinit();
        m.* = undefined;
    }

    pub fn reset(m: *Model, s: cuda.Stream) !void {
        m.slots[m.slot].recorded = 0;
        try m.zeros(s, m.slotBase() + m.layout.at(0).conv_offset, m.layout.bytes - m.layout.at(0).conv_offset);
    }

    pub fn rewind(m: *const Model, k: *const Kernels, s: cuda.Stream, kept: usize) !void {
        const slot = &m.slots[m.slot];
        if (slot.recorded == 0 or kept > slot.recorded) return error.BadRewind;
        try ops.rewind(&k.glue, s, slot.sites.ptr, m.w.p.n_visits * 4, m.w.c.text.hidden, slot.backup.ptr, slot.hist.ptr, 0, 0, kept);
    }

    pub fn arBlocks(m: *const Model, blocks: []const u64, partial: u64, query: u64, out: u64, nq: usize, partial_bf16: bool, rows: usize) loop_math.ArArgs {
        var a: loop_math.ArArgs = .{
            .partial = partial,
            .query = query,
            .out = out,
            .n_blocks = @intCast(blocks.len),
            .partial_bf16 = @intFromBool(partial_bf16),
            .nq = @intCast(nq),
            .rows = @intCast(rows),
            .c = @intCast(m.w.c.text.hidden),
            .eps = m.w.c.text.eps,
        };
        for (blocks, 0..) |b, i| a.blocks[i] = b;
        return a;
    }

    pub fn mixArgs(m: *const Model, blocks: []const u64, partial: u64, query: u64, norms: [3]u64, out: u64, nq: usize, partial_bf16: bool, rows: usize) ops.MixArgs {
        var a: ops.MixArgs = .{
            .partial = partial,
            .query = query,
            .out = out,
            .n_blocks = @intCast(blocks.len),
            .partial_bf16 = @intFromBool(partial_bf16),
            .nq = @intCast(nq),
            .rows = @intCast(rows),
            .c = @intCast(m.w.c.text.hidden),
            .eps = m.w.c.text.eps,
        };
        for (blocks, 0..) |b, i| a.blocks[i] = b;
        a.norm = norms;
        return a;
    }

    pub fn forward(m: *Model, k: *Kernels, s: cuda.Stream, tokens: []const u32, p0: usize, head0: usize, heads: usize, record: bool) !void {
        const chain: round.Chain = .{ .slot = m.slot, .position = p0, .tokens = tokens, .record = record };
        try round.execute(m, k, s, try round.prepare(m, s, &.{chain}, head0, heads, m.capture_taps));
    }

    pub const Saved = struct {
        at: u32,
        visits: u32,
        passes: u32,
        hidden: u32,
        format: loop.CacheFormat,
        buf: cuda.DeviceBuffer,
    };

    /// Compact bytes for all valid K/V rows and every convolution state of all 48 visits.
    pub fn snapshotBytes(m: *const Model, at: usize) u64 {
        var bytes = m.layout.bytes - m.layout.at(0).conv_offset;
        for (0..m.w.p.n_visits) |v| {
            const layer = m.w.p.at(m.w.p.visits[v].layer);
            const global = layer.kind == .global;
            const count = if (global) at else @min(at, m.w.c.text.window - 1);
            const row = layer.heads.kv * m.layout.format.rowBytes(layer.heads.dim);
            bytes += 2 * count * row;
        }
        return bytes;
    }

    fn copyState(m: *const Model, s: cuda.Stream, saved: *const Saved, out: bool) !void {
        const at: usize = saved.at;
        var off: usize = 0;
        for (0..m.w.p.n_visits) |v| {
            const visit = m.w.p.visits[v];
            const layer = m.w.p.at(visit.layer);
            const geometry = m.layerGeometry(visit.layer);
            const global = layer.kind == .global;
            const count = if (global) at else @min(at, m.w.c.text.window - 1);
            const first = at - count;
            const cache = m.cacheOf(visit.cache_slot);
            // Each plane is head-major codes followed by head-major scales in FP8 mode.
            const code_bytes = @as(usize, layer.heads.dim) * (if (cache.format == .fp8) @as(usize, 1) else 2);
            const pieces: []const usize = if (cache.format == .fp8) &.{ code_bytes, @sizeOf(f32) } else &.{code_bytes};
            for ([2]u64{ cache.k, cache.v }) |k_or_v| {
                var plane = k_or_v;
                for (pieces) |rb| {
                    for (0..layer.heads.kv) |head| {
                        const base = plane + head * geometry.ring * rb;
                        var pos = first;
                        while (pos < at) {
                            const row = pos % geometry.ring;
                            const n = @min(at - pos, geometry.ring - row);
                            const dev = base + row * rb;
                            const src = if (out) dev else saved.buf.ptr + off;
                            const dst = if (out) saved.buf.ptr + off else dev;
                            try m.copy(s, dst, src, n * rb);
                            off += n * rb;
                            pos += n;
                        }
                    }
                    plane += layer.heads.kv * geometry.ring * rb;
                }
            }
        }
        const conv = m.slotBase() + m.layout.at(0).conv_offset;
        const bytes = m.layout.bytes - m.layout.at(0).conv_offset;
        try m.copy(s, if (out) saved.buf.ptr + off else conv, if (out) conv else saved.buf.ptr + off, bytes);
    }

    pub fn save(m: *const Model, s: cuda.Stream, at: u32) !*Saved {
        if (at == 0 or at > m.context) return error.NotAtMark;
        const gpa = m.gpa;
        const saved = try gpa.create(Saved);
        errdefer gpa.destroy(saved);
        const lp = m.w.c.text.loop.?;
        saved.* = .{ .at = at, .visits = m.w.p.n_visits, .passes = lp.passes, .hidden = m.w.c.text.hidden, .format = m.layout.format, .buf = try cuda.DeviceBuffer.alloc(m.d, @intCast(@max(snapshotBytes(m, at), 256))) };
        errdefer saved.buf.free();
        try m.copyState(s, saved, true);
        return saved;
    }

    pub fn restore(m: *const Model, s: cuda.Stream, saved: *const Saved) !void {
        const lp = m.w.c.text.loop.?;
        if (saved.visits != m.w.p.n_visits or saved.passes != lp.passes or saved.hidden != m.w.c.text.hidden or saved.format != m.layout.format)
            return error.StateOfAnotherModel;
        if (saved.at == 0 or saved.at > m.context or saved.buf.len < snapshotBytes(m, saved.at)) return error.BadLoopSnapshot;
        try m.copyState(s, saved, false);
    }

    pub fn drop(m: *const Model, s: cuda.Stream, saved: *Saved) void {
        s.synchronize() catch {};
        saved.buf.free();
        m.gpa.destroy(saved);
    }
};

//! A BF16 Mini DFlash2 block on CUDA, using committed target features as its context.
const std = @import("std");
const cuda = @import("cuda");
const w_ = @import("dflash2_weights.zig");
const dspark = @import("dspark_cuda.zig");
const dsp = @import("dspark_ops.zig");
const engine = @import("loop_engine.zig");
const ops = @import("ops.zig");
const fp8 = @import("draft_fp8.zig");

const h = w_.hidden;
const block = w_.block;
const depth = block - 1;
const max_context_rows = 32;
const chunks = (w_.window + block - 2) / dsp.chunk + 2;
const top_words = 43;
const Session = dspark.Session;

fn ring(context: usize) usize {
    const kept: usize = @min(context, w_.window - 1);
    return kept + block;
}

pub fn kvBytes(context: usize) usize {
    return w_.layers * ring(context) * 2 * w_.kv * 2;
}

pub fn partialBytes() usize {
    return chunks * block * (w_.q * 4 + w_.heads * 8);
}

const Scratch = struct {
    taps: u64,
    mixed: u64,
    matrix: u64,
    accum: u64,
    residual: u64,
    normed: u64,
    conv: u64,
    delta: u64,
    q: u64,
    attn: u64,
    act: u64,
    partials: u64,
    maxima: u64,
    logits: u64,
    logits_f32: u64,
    top_parts: u64,
    top: u64,
    projected: u64,
    ids: u64,
    scores: u64,
    position: u64,
    walk: u64,

    fn layout(base: u64, total: *usize) Scratch {
        const b = 2;
        const sizes = .{
            .taps = dspark.shared_tap_bytes,
            .mixed = max_context_rows * h * b,
            .matrix = @max(max_context_rows * 2 * w_.kv, block * 2 * w_.inter) * b,
            .accum = @max(max_context_rows * h, block * 2 * w_.inter) * 4,
            .residual = block * h * b,
            .normed = block * h * b,
            .conv = block * h * b,
            .delta = block * 4 * w_.groups * b,
            .q = block * w_.q * b,
            .attn = block * w_.q * b,
            .act = block * w_.inter * b,
            .partials = chunks * block * w_.q * 4,
            .maxima = chunks * block * w_.heads * 8,
            .logits = depth * w_.vocab * b,
            .logits_f32 = depth * w_.vocab * 4,
            .top_parts = cuda.sample.Draws.topkPartsBytes(w_.vocab, w_.top_k, depth),
            .top = depth * top_words * 4,
            .projected = depth * w_.rank * b,
            .ids = w_.top_k * @sizeOf(i64),
            .scores = w_.top_k * b,
            .position = 4,
            .walk = block * 4,
        };
        var s: Scratch = undefined;
        inline for (@typeInfo(@TypeOf(sizes)).@"struct".field_names) |field| {
            @field(s, field) = base + total.*;
            total.* += std.mem.alignForward(usize, @field(sizes, field), 256);
        }
        return s;
    }
};

pub fn scratchBytes() usize {
    var size: usize = 0;
    _ = Scratch.layout(0, &size);
    return size;
}

const Key = struct { slot: usize, n: usize, rule: u64 };
const GraphCache = cuda.graph_cache.Cache(Key);

pub const Drafter = struct {
    w: w_.Weights,
    k: dsp.Kernels,
    dense: cuda.dense_rows.DenseRows,
    lane: ?cuda.qmmf.Lane,
    qk: cuda.Function,
    attention: cuda.Function,
    convolve: cuda.Function,
    select: cuda.Function,
    memory: cuda.DeviceBuffer,
    scratch: Scratch,
    ring_size: usize,
    sessions: [engine.max_slots]Session = undefined,
    count: usize = 0,
    graph_engine: cuda.graph_cache.CudaEngine,
    graphs: ?GraphCache = null,

    /// Takes ownership of weights on success.
    pub fn init(d: *const cuda.Driver, w: w_.Weights, context: usize) !Drafter {
        var k = try dsp.Kernels.load(d);
        errdefer k.unload();
        var dense = try cuda.dense_rows.DenseRows.load(d);
        errdefer dense.unload();
        var lane = try fp8.loadLane(d, w.fp8_layers or w.base != null);
        errdefer if (lane) |*l| l.unload();
        var memory = try cuda.DeviceBuffer.alloc(d, scratchBytes());
        errdefer memory.free();
        var used: usize = 0;
        var out: Drafter = .{
            .w = w,
            .k = k,
            .dense = dense,
            .lane = lane,
            .qk = try k.mod.function("tf_ink2_df_qk"),
            .attention = try k.mod.function("tf_ink2_df_attn"),
            .convolve = try k.mod.function("tf_ink2_df_conv"),
            .select = try k.mod.function(if (w.predecessor.scales != 0) "tf_ink2_df_select8" else "tf_ink2_df_select"),
            .memory = memory,
            .scratch = Scratch.layout(memory.ptr, &used),
            .ring_size = ring(context),
            .graph_engine = .{ .d = d },
        };
        try out.reserve(1);
        return out;
    }

    pub fn enableGraphs(d: *Drafter, a: std.mem.Allocator) void {
        d.graphs = .init(a, d.graph_engine.engine(), .{});
        d.graphs.?.name = "inkling2 mini dflash2 proposal";
    }

    pub fn reserve(d: *Drafter, count: usize) !void {
        const sizes: [w_.layers]usize = @splat(d.ring_size);
        while (d.count < count) {
            d.sessions[d.count] = try Session.initRings(d.memory.d, &sizes, 0);
            d.count += 1;
        }
    }

    pub fn deinit(d: *Drafter) void {
        if (d.graphs) |*g| g.deinit();
        for (d.sessions[0..d.count]) |*session| session.memory.free();
        d.memory.free();
        if (d.lane) |*l| l.unload();
        d.dense.unload();
        d.k.unload();
        d.w.deinit();
    }

    fn project(d: *const Drafter, k: *const ops.Kernels, s: cuda.Stream, x: u64, w: fp8.Matrix, out: u64, rows: usize, n: usize, c: usize) !void {
        if (w.scales != 0) return d.lane.?.matmul(s, x, c, rows, w.lane(n, c), out, 0);
        const a = d.scratch.accum;
        try d.dense.rows(s, x, c, w.data, a, n, rows, n, c);
        try ops.tobf(k, s, a, rows, n, n, out, n, 0, n);
    }

    fn qkArgs(d: *const Drafter, session: *const Session, i: usize, heads: usize, pos: usize) dsp.QkArgs {
        const l = d.w.layer[i];
        return .{
            .x = d.scratch.matrix,
            .qw = l.q_norm,
            .kw = l.k_norm,
            .q = d.scratch.q,
            .kc = session.k[i],
            .vc = session.v[i],
            .ld = @intCast(heads * w_.dim + 2 * w_.kv),
            .hq = @intCast(heads),
            .hk = w_.kv_heads,
            .ring = @intCast(d.ring_size),
            .pos0 = @intCast(pos),
            .eps = w_.eps,
            .theta = w_.theta,
        };
    }

    pub fn absorb(d: *Drafter, session: *Session, k: *const ops.Kernels, s: cuda.Stream, from: u64, pos: usize, rows: usize) !void {
        if (pos != session.end) return error.DraftContextGap;
        const a = d.scratch;
        var offset = rows -| (w_.window - 1);
        while (offset < rows) {
            const n: usize = @min(rows - offset, max_context_rows);
            try d.project(k, s, from + offset * dspark.tap_stride, d.w.fuse, a.mixed, n, h, 4 * h);
            try dsp.addNorm(&d.k, s, a.mixed, 0, 0, 0, d.w.fuse_norm, a.matrix, n, h, w_.eps);
            // Keep the normalized fusion outside the projection output, which is reused by every layer.
            try d.memory.d.check(d.memory.d.api.cuMemcpyDtoDAsync_v2(a.mixed, a.matrix, n * h * 2, s.handle), "cuMemcpyDtoDAsync");
            for (d.w.layer, 0..) |l, i| {
                const kv_weights = l.qkv.afterRows(w_.q, h);
                try d.project(k, s, a.mixed, kv_weights, a.matrix, n, 2 * w_.kv, h);
                try dsp.qk(&d.k, s, d.qkArgs(session, i, 0, pos + offset), n);
            }
            offset += n;
        }
        session.end += rows;
        session.held_len = 0;
    }

    fn launch(f: cuda.Function, s: cuda.Stream, x: usize, y: usize, threads: u32, args: anytype) !void {
        var a: cuda.Args = .{};
        inline for (args) |arg| a.add(arg);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(x), .y = @intCast(y) }, .block = .{ .x = threads } }, s, &a);
    }

    fn attend(d: *const Drafter, s: cuda.Stream, session: *const Session, i: usize) !void {
        const a = d.scratch;
        try launch(d.qk, s, block, w_.heads + w_.kv_heads, 64, .{ d.qkArgs(session, i, w_.heads, 0), a.position });
        const at: dsp.AttnArgs = .{
            .q = a.q,
            .kc = session.k[i],
            .vc = session.v[i],
            .po = a.partials,
            .pml = a.maxima,
            .T = block,
            .hq = w_.heads,
            .hk = w_.kv_heads,
            .ring = @intCast(d.ring_size),
            .lo = 0,
            .hi = 0,
            .c0 = 0,
            .scale = 1.0 / @sqrt(@as(f32, w_.dim)),
        };
        try launch(d.attention, s, chunks, w_.kv_heads, 256, .{ at, a.position, @as(i32, w_.window) });
        try launch(d.k.merge, s, block * w_.heads, 1, 128, .{
            a.partials, a.maxima, @as(i32, chunks), @as(i32, block * w_.heads), a.attn,
        });
    }

    fn convInput(d: *const Drafter, k: *const ops.Kernels, s: cuda.Stream, norm: u64, cv: w_.Conv) !void {
        const a = d.scratch;
        try dsp.addNorm(&d.k, s, a.residual, 0, 0, 0, norm, a.normed, block, h, w_.eps);
        try d.project(k, s, a.normed, cv.projection, a.delta, block, 4 * w_.groups, h);
        try launch(d.convolve, s, block, 1, 256, .{ a.normed, a.delta, cv.base, a.conv, @as(i32, h), @as(i32, 0) });
    }

    fn convOutput(d: *const Drafter, k: *const ops.Kernels, s: cuda.Stream, x: u64, matrix: fp8.Matrix, width: usize, cv: w_.Conv) !void {
        const a = d.scratch;
        try d.project(k, s, x, matrix, a.conv, block, h, width);
        try launch(d.convolve, s, block, 1, 256, .{ a.conv, a.delta, cv.base, a.residual, @as(i32, h), @as(i32, 1) });
    }

    fn selectStep(d: *const Drafter, s: cuda.Stream, i: usize, vocab: usize) !void {
        const a = d.scratch;
        const p = d.w.predecessor;
        const q = d.w.successor;
        var args: cuda.Args = .{};
        for ([_]u64{ a.top, a.projected, p.data, q.data }) |ptr| args.add(ptr);
        if (p.scales != 0) {
            args.add(p.scales);
            args.add(q.scales);
        }
        for ([_]u64{ a.walk, a.ids, a.scores }) |ptr| args.add(ptr);
        args.add(@as(i32, @intCast(i)));
        args.add(@as(i32, @intCast(vocab)));
        args.add(@as(f32, w_.cap));
        try cuda.launch.launch(d.select, .{ .grid = .{ .x = 1 }, .block = .{ .x = 512 } }, s, &args);
    }

    const Block = struct {
        d: *Drafter,
        m: *const engine.Model,
        k: *const engine.Kernels,
        key: Key,

        fn run(ptr: *anyopaque, s: cuda.Stream) anyerror!void {
            const b: *Block = @ptrCast(@alignCast(ptr));
            const d = b.d;
            const a = d.scratch;
            const k = &b.k.glue;
            const target = b.m.w.c.text;
            const session = &d.sessions[b.key.slot];
            try ops.embed(k, s, a.walk, block, b.m.w.embed, b.m.w.embed_norm, a.residual, h, target.eps);
            for (d.w.layer, 0..) |l, i| {
                try d.convInput(k, s, l.input_norm, l.attention_conv);
                try d.project(k, s, a.conv, l.qkv, a.matrix, block, w_.q + 2 * w_.kv, h);
                try d.attend(s, session, i);
                try d.convOutput(k, s, a.attn, l.out, w_.q, l.attention_conv);

                try d.convInput(k, s, l.post_norm, l.mlp_conv);
                try d.project(k, s, a.conv, l.gate_up, a.matrix, block, 2 * w_.inter, h);
                try dsp.silu(&d.k, s, a.matrix, a.act, block, w_.inter);
                try d.convOutput(k, s, a.act, l.down, w_.inter, l.mlp_conv);
            }
            try dsp.addNorm(&d.k, s, a.residual, 0, 0, 0, d.w.norm, a.normed, block, h, w_.eps);
            const x = a.normed + h * 2;
            const n = b.key.n;
            const v = target.unpadded_vocab;
            // The target's padding rows cannot be offered by the draft.
            if (d.w.base) |w| {
                try d.lane.?.matmul(s, x, h, n, w, a.logits, 0);
            } else {
                try d.dense.rows(s, x, h, b.m.w.unembed, a.logits_f32, v, n, v, h);
                try ops.tobf(k, s, a.logits_f32, n, v, v, a.logits, v, 0, v);
            }
            try b.k.draws.topkRows(s, a.logits, v, w_.top_k, a.top_parts, a.top, n);
            try d.project(k, s, x, d.w.project, a.projected, n, w_.rank, h);
            for (0..n) |i| {
                try d.selectStep(s, i, v);
                if (b.key.rule != 0) {
                    try b.k.draws.draw(s, a.scores, w_.top_k, b.key.rule, a.position, i, a.walk + (i + 1) * 4, 1, a.ids, null);
                }
            }
        }
    };

    pub fn propose(d: *Drafter, slot: usize, m: *const engine.Model, k: *const engine.Kernels, s: cuda.Stream, pending: u32, n: usize, rule: ?u64) !void {
        const session = &d.sessions[slot];
        if (n == 0 or n > depth or session.end == 0) return error.BadDraftDepth;
        const a = d.scratch;
        var ids: [block]u32 = @splat(w_.mask);
        ids[0] = pending;
        const position: i32 = @intCast(session.end);
        try cuda.copy_source.enqueue(d.memory.d, a.walk, std.mem.sliceAsBytes(&ids), s);
        try cuda.copy_source.enqueue(d.memory.d, a.position, std.mem.asBytes(&position), s);
        const key: Key = .{ .slot = slot, .n = n, .rule = rule orelse 0 };
        var b: Block = .{ .d = d, .m = m, .k = k, .key = key };
        if (d.graphs) |*g| {
            _ = try g.run(key, s, 0, .{ .ctx = &b, .run = Block.run });
        } else try Block.run(&b, s);
        try s.synchronize();
        try d.memory.download(@intCast(a.walk - d.memory.ptr + 4), std.mem.sliceAsBytes(session.held[0..n]));
        for (session.held[0..n]) |t| if (t >= m.w.c.text.unpadded_vocab) return error.DraftNonfinite;
        session.held_len = n;
    }
};

//! Mini DSpark on one CUDA stream: commit target features, then propose a masked eight-row block.
const std = @import("std");
const cuda = @import("cuda");
const w_ = @import("dspark_weights.zig");
const dsp = @import("dspark_ops.zig");
const engine = @import("loop_engine.zig");
const loop_math = @import("loop_math_cuda.zig");
const ops = @import("ops.zig");

const h = w_.hidden;
const q = w_.q;
const kv = w_.kv;
const inter = w_.inter;
const rank = w_.rank;
const block = w_.block;
const local_window = 1024;
const local_ring = local_window + block;
const absorb_rows = 64;
pub const depth = block - 1;
pub const tap_stride = 4 * h * 2;
pub const shared_tap_bytes = engine.prompt_rows * tap_stride;

fn partialSizes(context: usize) [2]usize {
    const chunks = (context + block + dsp.chunk - 1) / dsp.chunk;
    return .{ chunks * block * q * 4, chunks * block * 24 * 8 };
}

pub fn partialBytes(context: usize) usize {
    const sizes = partialSizes(context);
    return std.mem.alignForward(usize, sizes[0], 256) + std.mem.alignForward(usize, sizes[1], 256);
}

const Scratch = struct {
    taps: u64,
    mixed: u64,
    matrix_rows: u64,
    residual: u64,
    x: u64,
    query: u64,
    out: u64,
    act: u64,
    f32_rows: u64,
    partial: u64,
    maxima: u64,
    base: u64,
    kappa: u64,
    state: u64,
    latent: u64,
    walk: u64,
    draw: u64,

    fn layout(base: u64, context: usize, total: *usize) Scratch {
        const partials = partialSizes(context);
        const b = 2;
        const sizes = .{
            .taps = shared_tap_bytes,
            .mixed = absorb_rows * h * b,
            .matrix_rows = @max(absorb_rows * 2 * kv, block * 2 * inter) * b,
            .residual = block * h * b,
            .x = block * h * b,
            .query = block * q * b,
            .out = block * q * b,
            .act = block * inter * b,
            .f32_rows = depth * w_.vocab * 4,
            .partial = partials[0],
            .maxima = partials[1],
            .base = depth * w_.vocab * b,
            .kappa = depth * 3 * rank * 4,
            .state = rank * b,
            .latent = rank * b,
            .walk = block * 4,
            .draw = depth * 16,
        };
        var out: Scratch = undefined;
        inline for (@typeInfo(@TypeOf(sizes)).@"struct".field_names) |field| {
            @field(out, field) = base + total.*;
            total.* += std.mem.alignForward(usize, @field(sizes, field), 256);
        }
        return out;
    }
};

/// KV, the last fusion vector, and the target taps waiting to be committed belong to a request.
pub const Session = struct {
    memory: cuda.DeviceBuffer,
    k: [5]u64,
    v: [5]u64,
    newest: u64,
    taps: u64,
    end: usize = 0,
    pending: usize = 0,
    held: [depth]u32 = undefined,
    held_len: usize = 0,

    pub const newest_bytes = h * 2;
    pub const tap_bytes = 2 * engine.window_rows * tap_stride;

    fn kvRings(context: usize) [3]usize {
        return .{ local_ring, context + block, local_ring };
    }

    pub fn kvBytes(context: usize) usize {
        const rings = kvRings(context);
        return (rings[0] + rings[1] + rings[2]) * 2 * kv * 2;
    }

    pub fn bytes(context: usize) usize {
        return kvBytes(context) + newest_bytes + tap_bytes;
    }

    pub fn init(d: *const cuda.Driver, context: usize) !Session {
        return initRings(d, &kvRings(context), newest_bytes);
    }

    pub fn initRings(d: *const cuda.Driver, rings: []const usize, newest_size: usize) !Session {
        var kv_bytes: usize = 0;
        for (rings) |ring| kv_bytes += ring * 2 * kv * 2;
        const memory = try cuda.DeviceBuffer.alloc(d, kv_bytes + newest_size + tap_bytes);
        var session: Session = .{ .memory = memory, .k = undefined, .v = undefined, .newest = undefined, .taps = undefined };
        var at = memory.ptr;
        for (rings, session.k[0..rings.len], session.v[0..rings.len]) |ring, *key, *value| {
            key.* = at;
            at += ring * kv * 2;
            value.* = at;
            at += ring * kv * 2;
        }
        session.newest = at;
        session.taps = at + newest_size;
        return session;
    }

    pub fn reset(session: *Session) void {
        session.end = 0;
        session.pending = 0;
        session.held_len = 0;
    }
};

pub const Drafter = struct {
    w: w_.Weights,
    k: dsp.Kernels,
    dense: cuda.dense_rows.DenseRows,
    memory: cuda.DeviceBuffer,
    scratch: Scratch,
    rings: [3]usize,
    sessions: [engine.max_slots]Session = undefined,
    count: usize = 0,

    pub fn scratchBytes(context: usize) usize {
        var bytes: usize = 0;
        _ = Scratch.layout(0, context, &bytes);
        return bytes;
    }

    pub fn allocScratch(d: *const cuda.Driver, context: usize) !cuda.DeviceBuffer {
        return cuda.DeviceBuffer.alloc(d, scratchBytes(context));
    }

    /// Takes ownership of the weights only on success.
    pub fn init(d: *const cuda.Driver, w: w_.Weights, context: usize) !Drafter {
        var kernels = try dsp.Kernels.load(d);
        errdefer kernels.unload();
        var dense = try cuda.dense_rows.DenseRows.load(d);
        errdefer dense.unload();
        var memory = try allocScratch(d, context);
        errdefer memory.free();
        var bytes: usize = 0;
        var out: Drafter = .{
            .w = w,
            .k = kernels,
            .dense = dense,
            .memory = memory,
            .scratch = Scratch.layout(memory.ptr, context, &bytes),
            .rings = Session.kvRings(context),
        };
        try out.reserve(1);
        return out;
    }

    pub fn reserve(d: *Drafter, count: usize) !void {
        while (d.count < count) {
            d.sessions[d.count] = try Session.init(d.memory.d, d.rings[1] - block);
            d.count += 1;
        }
    }

    pub fn deinit(d: *Drafter) void {
        for (d.sessions[0..d.count]) |*session| session.memory.free();
        d.memory.free();
        d.dense.unload();
        d.k.unload();
        d.w.deinit();
    }

    fn copy(d: *const Drafter, s: cuda.Stream, to: u64, from: u64, bytes: usize) !void {
        const drv = d.memory.d;
        try drv.check(drv.api.cuMemcpyDtoDAsync_v2(to, from, bytes, s.handle), "cuMemcpyDtoDAsync");
    }

    fn project(d: *const Drafter, k: *const ops.Kernels, s: cuda.Stream, x: u64, w: u64, to: u64, rows: usize, n: usize, c: usize) !void {
        const f = d.scratch.f32_rows;
        try d.dense.rows(s, x, c, w, f, n, rows, n, c);
        try ops.tobf(k, s, f, rows, n, n, to, n, 0, n);
    }

    pub fn absorb(d: *Drafter, session: *Session, glue: *const ops.Kernels, s: cuda.Stream, from: u64, pos: usize, rows: usize, eps: f32) !void {
        if (pos != session.end) return error.DraftContextGap;
        const a = d.scratch;
        var offset: usize = 0;
        while (offset < rows) {
            const n = @min(rows - offset, absorb_rows);
            const taps = from + offset * tap_stride;
            for (d.w.layers, 0..) |l, i| {
                try dsp.mix(&d.k, s, taps, d.w.alpha + i * 16, l.dflare, a.mixed, n, w_.taps.len, h, eps);
                try d.project(glue, s, a.mixed, l.ctx_kv, a.matrix_rows, n, 2 * kv, h);
                try dsp.qk(&d.k, s, .{
                    .x = a.matrix_rows,
                    .qw = 0,
                    .kw = l.ctx_norm,
                    .q = 0,
                    .kc = session.k[i],
                    .vc = session.v[i],
                    .ld = 2 * kv,
                    .hq = 0,
                    .hk = 8,
                    .ring = @intCast(d.rings[i]),
                    .pos0 = @intCast(pos + offset),
                    .eps = eps,
                    .theta = 1e6,
                }, n);
            }
            offset += n;
        }
        if (rows > 0) try dsp.mix(&d.k, s, from + (rows - 1) * tap_stride, d.w.alpha + 3 * 16, d.w.markov_norm, session.newest, 1, w_.taps.len, h, eps);
        session.end += rows;
        session.held_len = 0;
    }

    fn residual(d: *const Drafter, k: *const engine.Kernels, s: cuda.Stream, x: u64, weight: u64, matrix: u64, width: usize, eps: f32) !void {
        const a = d.scratch;
        try d.dense.rows(s, x, width, matrix, a.f32_rows, h, block, h, width);
        try loop_math.roundBf16(&k.loop, s, a.f32_rows, a.f32_rows, block * h);
        try dsp.addNorm(&d.k, s, a.residual, a.f32_rows, 1, block * h, weight, a.x, block, h, eps);
    }

    fn encode(d: *const Drafter, session: *const Session, m: *const engine.Model, k: *const engine.Kernels, s: cuda.Stream, n: usize) !void {
        const a = d.scratch;
        const c = m.w.c.text;
        try ops.embed(&k.glue, s, a.walk, 1, m.w.embed, m.w.embed_norm, a.residual, h, c.eps);
        for (1..block) |i| try d.copy(s, a.residual + i * h * 2, d.w.mask, h * 2);
        try dsp.addNorm(&d.k, s, a.residual, 0, 0, 0, d.w.layers[0].input_norm, a.x, block, h, c.eps);
        for (d.w.layers, 0..) |l, i| {
            try d.project(&k.glue, s, a.x, l.qkv, a.matrix_rows, block, q + 2 * kv, h);
            try dsp.qk(&d.k, s, .{
                .x = a.matrix_rows,
                .qw = l.q_norm,
                .kw = l.k_norm,
                .q = a.query,
                .kc = session.k[i],
                .vc = session.v[i],
                .ld = q + 2 * kv,
                .hq = 24,
                .hk = 8,
                .ring = @intCast(d.rings[i]),
                .pos0 = @intCast(session.end),
                .eps = c.eps,
                .theta = 1e6,
            }, block);
            try dsp.attend(&d.k, s, .{
                .q = a.query,
                .kc = session.k[i],
                .vc = session.v[i],
                .po = a.partial,
                .pml = a.maxima,
                .T = block,
                .hq = 24,
                .hk = 8,
                .ring = @intCast(d.rings[i]),
                .lo = @intCast(if (i == 1) 0 else session.end -| local_window),
                .hi = @intCast(session.end + block - 1),
                .c0 = 0,
                .scale = 1.0 / @sqrt(@as(f32, dsp.head_dim)),
            }, a.out);
            try d.residual(k, s, a.out, l.post_norm, l.out, q, c.eps);
            try d.project(&k.glue, s, a.x, l.gate_up, a.matrix_rows, block, 2 * inter, h);
            try dsp.silu(&d.k, s, a.matrix_rows, a.act, block, inter);
            const next = if (i + 1 < d.w.layers.len) d.w.layers[i + 1].input_norm else d.w.norm;
            try d.residual(k, s, a.act, next, l.down, inter, c.eps);
        }
        try d.project(&k.glue, s, a.x + h * 2, m.w.unembed, a.base, n, w_.vocab, h);
        try dsp.ctx(&d.k, s, a.x + h * 2, session.newest, d.w.rnn, d.w.rnn_bias, a.kappa, n, h, rank);
    }

    pub fn propose(d: *Drafter, session: *Session, m: *const engine.Model, k: *const engine.Kernels, s: cuda.Stream, pending: u32, n: usize, rule: ?u64) !void {
        if (n == 0 or n > depth or session.end == 0) return error.BadDraftDepth;
        const c = m.w.c.text;
        const a = d.scratch;
        const drv = d.memory.d;
        try cuda.copy_source.enqueue(drv, a.walk, std.mem.asBytes(&pending), s);
        try drv.check(drv.api.cuMemsetD8Async(a.state, 0, rank * 2, s.handle), "cuMemsetD8Async");
        if (rule != null) {
            var meta: [depth][4]i32 = @splat(@splat(0));
            for (&meta, 0..) |*row, i| row[0] = @intCast(session.end + i);
            try cuda.copy_source.enqueue(drv, a.draw, std.mem.sliceAsBytes(meta[0..n]), s);
        }
        try d.encode(session, m, k, s, n);
        for (0..n) |j| {
            const token = a.walk + (j + 1) * 4;
            try dsp.rnn(&d.k, s, a.kappa + j * 3 * rank * 4, d.w.rnn, d.w.markov_w1, a.walk, j, a.state, a.latent, rank, h);
            try dsp.vocab(&d.k, s, a.base + j * w_.vocab * 2, a.latent, d.w.phi, d.w.beta, c.logit_cap, c.unpadded_vocab, c.unpadded_vocab, a.f32_rows);
            if (rule) |r| {
                try ops.tobf(&k.glue, s, a.f32_rows, 1, c.unpadded_vocab, c.unpadded_vocab, a.base + j * w_.vocab * 2, c.unpadded_vocab, 0, c.unpadded_vocab);
                try k.draws.draw(s, a.base + j * w_.vocab * 2, c.unpadded_vocab, r, a.draw + j * 16, 0, token, 1, null, null);
            } else {
                try ops.top1(&k.glue, s, a.f32_rows, 1, c.unpadded_vocab, c.unpadded_vocab, 0, a.out, token);
            }
        }
        try s.synchronize();
        try d.memory.download(@intCast(a.walk - d.memory.ptr + 4), std.mem.sliceAsBytes(session.held[0..n]));
        for (session.held[0..n]) |t| if (t >= c.unpadded_vocab) return error.DraftNonfinite;
        session.held_len = n;
    }
};

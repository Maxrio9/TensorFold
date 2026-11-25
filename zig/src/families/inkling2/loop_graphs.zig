//! Mini decode graphs keyed by the slot addresses, verification widths and launch shape.

// Shared verify cache adapted from MiaAI-Lab patch 0014. Copyright 2026 MiaAI-Lab.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("loop_engine.zig");

const round = engine.round;
const Cache = cuda.graph_cache.Cache(round.Shape);

pub const Graphs = struct {
    cuda_engine: cuda.graph_cache.CudaEngine,
    cache: Cache,
    verify_cache: ?Cache,

    pub fn init(g: *Graphs, gpa: std.mem.Allocator, d: *const cuda.Driver, separate: bool) void {
        g.cuda_engine = .{ .d = d };
        g.cache = .init(gpa, g.cuda_engine.engine(), if (separate) .{ .max = 16 } else .{});
        g.cache.name = "inkling2 mini decode";
        g.verify_cache = if (separate) Cache.init(gpa, g.cuda_engine.engine(), .{ .max = 32 }) else null;
        if (g.verify_cache) |*verify| verify.name = "inkling2 mini shared verify";
    }

    pub fn deinit(g: *Graphs) void {
        if (g.verify_cache) |*verify| verify.deinit();
        g.cache.deinit();
    }

    pub fn forward(g: *Graphs, m: *engine.Model, k: *engine.Kernels, s: cuda.Stream, chains: []const round.Chain, heads: usize, taps: bool, lazy: bool) !void {
        const Body = struct {
            model: *engine.Model,
            kernels: *engine.Kernels,
            shape: round.Shape,

            fn run(ptr: *anyopaque, stream: cuda.Stream) anyerror!void {
                const b: *@This() = @ptrCast(@alignCast(ptr));
                try round.execute(b.model, b.kernels, stream, b.shape);
            }
        };
        const shape = try round.prepareDecode(m, s, chains, heads, taps, lazy);
        var body: Body = .{ .model = m, .kernels = k, .shape = shape };
        const cache = if (shape.count > 1 and shape.rows > shape.count and g.verify_cache != null) &g.verify_cache.? else &g.cache;
        _ = try cache.run(shape, s, 0, .{ .ctx = &body, .run = Body.run });
    }

    pub fn warm(g: *Graphs, m: *engine.Model, k: *engine.Kernels, s: cuda.Stream, width: usize, record: bool, lazy: bool) !void {
        if (!g.cache.settings.on or width > m.context) return;
        const zeros: [engine.window_rows]u32 = @splat(0);
        try m.reset(s);
        try m.forward(k, s, zeros[0..width], 0, 0, width, record);
        try s.synchronize();
        try m.reset(s);
        const chain: round.Chain = .{ .slot = m.slot, .position = 0, .tokens = zeros[0..width], .record = record };
        try g.forward(m, k, s, &.{chain}, width, m.capture_taps, lazy);
        try s.synchronize();
        try m.reset(s);
    }

    /// Price warmed greedy verify graphs so the first request does not learn their capture time.
    pub fn price(g: *Graphs, m: *engine.Model, k: *engine.Kernels, s: cuda.Stream, io: std.Io, width: usize, lazy: bool, samples: usize) !f64 {
        const zeros: [engine.window_rows]u32 = @splat(0);
        const chain: round.Chain = .{ .slot = m.slot, .position = 0, .tokens = zeros[0..width], .record = width > 1 };
        try g.warm(m, k, s, width, chain.record, lazy);
        var times: [3]f64 = undefined;
        for (times[0..samples]) |*ms| {
            try m.reset(s);
            try s.synchronize();
            const t = std.Io.Clock.awake.now(io);
            try g.forward(m, k, s, &.{chain}, width, m.capture_taps, lazy);
            try s.synchronize();
            ms.* = elapsed(io, t);
        }
        return median(times[0..samples]);
    }
};

pub fn elapsed(io: std.Io, t: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
}

pub fn median(times: []f64) f64 {
    std.mem.sort(f64, times, {}, std.sort.asc(f64));
    return times[times.len / 2];
}

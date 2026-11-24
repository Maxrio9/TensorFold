//! A Mini decode executable for each row count, cache slot, and attention tile shape.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("loop_engine.zig");

const round = engine.round;

pub const Graphs = struct {
    cuda_engine: cuda.graph_cache.CudaEngine,
    cache: cuda.graph_cache.Cache(round.Shape),

    pub fn init(g: *Graphs, gpa: std.mem.Allocator, d: *const cuda.Driver) void {
        g.cuda_engine = .{ .d = d };
        g.cache = .init(gpa, g.cuda_engine.engine(), .{});
        g.cache.name = "inkling2 mini decode";
    }

    pub fn deinit(g: *Graphs) void {
        g.cache.deinit();
    }

    pub fn forward(g: *Graphs, m: *engine.Model, k: *engine.Kernels, s: cuda.Stream, chains: []const round.Chain, heads: usize, taps: bool) !void {
        const Body = struct {
            model: *engine.Model,
            kernels: *engine.Kernels,
            shape: round.Shape,

            fn run(ptr: *anyopaque, stream: cuda.Stream) anyerror!void {
                const b: *@This() = @ptrCast(@alignCast(ptr));
                try round.execute(b.model, b.kernels, stream, b.shape);
            }
        };
        const shape = try round.prepare(m, s, chains, 0, heads, taps);
        var body: Body = .{ .model = m, .kernels = k, .shape = shape };
        _ = try g.cache.run(shape, s, 0, .{ .ctx = &body, .run = Body.run });
    }

    pub fn warm(g: *Graphs, m: *engine.Model, k: *engine.Kernels, s: cuda.Stream, width: usize, record: bool) !void {
        if (!g.cache.settings.on or width > m.context) return;
        const zeros: [engine.window_rows]u32 = @splat(0);
        try m.reset(s);
        try m.forward(k, s, zeros[0..width], 0, 0, width, record);
        try s.synchronize();
        try m.reset(s);
        const chain: round.Chain = .{ .slot = m.slot, .position = 0, .tokens = zeros[0..width], .record = record };
        try g.forward(m, k, s, &.{chain}, width, m.capture_taps);
        try s.synchronize();
        try m.reset(s);
    }
};

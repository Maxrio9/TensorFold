//! A Mini decode executable for each row count, cache slot, and attention tile shape.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("loop_engine.zig");

const Key = struct { rows: usize, slot: usize, tiles: [2]u32, record: bool, taps: bool };

pub const Graphs = struct {
    cuda_engine: cuda.graph_cache.CudaEngine,
    cache: cuda.graph_cache.Cache(Key),

    pub fn init(g: *Graphs, gpa: std.mem.Allocator, d: *const cuda.Driver) void {
        g.cuda_engine = .{ .d = d };
        g.cache = .init(gpa, g.cuda_engine.engine(), .{});
        g.cache.name = "inkling2 mini decode";
    }

    pub fn deinit(g: *Graphs) void {
        g.cache.deinit();
    }

    pub fn forward(g: *Graphs, m: *engine.Model, k: *engine.Kernels, s: cuda.Stream, tokens: []const u32, pos: usize, record: bool) !void {
        const Body = struct {
            model: *engine.Model,
            kernels: *engine.Kernels,
            shape: engine.Model.Shape,

            fn run(ptr: *anyopaque, stream: cuda.Stream) anyerror!void {
                const b: *@This() = @ptrCast(@alignCast(ptr));
                try b.model.execute(b.kernels, stream, b.shape);
            }
        };
        const shape = try m.prepare(s, tokens, pos, 0, tokens.len, record);
        var body: Body = .{ .model = m, .kernels = k, .shape = shape };
        // The cache belongs to this model. Its slots, weights, stream and scratch have immutable addresses.
        const key: Key = .{ .rows = shape.rows, .slot = m.slot, .tiles = shape.tiles, .record = record, .taps = shape.taps };
        _ = try g.cache.run(key, s, 0, .{ .ctx = &body, .run = Body.run });
    }

    pub fn warm(g: *Graphs, m: *engine.Model, k: *engine.Kernels, s: cuda.Stream, width: usize, record: bool) !void {
        if (!g.cache.settings.on or width > m.context) return;
        const zeros: [engine.window_rows]u32 = @splat(0);
        try m.reset(s);
        try m.forward(k, s, zeros[0..width], 0, 0, width, record);
        try s.synchronize();
        try m.reset(s);
        try g.forward(m, k, s, zeros[0..width], 0, record);
        try s.synchronize();
        try m.reset(s);
    }
};

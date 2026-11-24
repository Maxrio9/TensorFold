//! Mini's capture writes one BF16 feature channel into otherwise unchanged interleaved rows.
const std = @import("std");
const core = @import("core");
const dk = @import("inkling2").dspark_ops;
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_minimax_glue_ref.zig");
const c = core.contract.spec;
const g = core.contract.gates;

pub const Tap = struct {
    kern: *const dk.Kernels,
    x: []u16 = &.{},
    before: []u16 = &.{},
    x_dev: u64 = 0,
    out: u64 = 0,
    shape: c.Shape = undefined,

    pub fn spec(_: *const Tap) c.Spec {
        return .{
            .name = "inkling2 dspark tf_ink2_dsp_tap",
            .accum = .none,
            .depth = ref.noSums,
            .out = .bf16,
            .nan = .{ .stated = "source and untouched channels retain their value, including NaN and infinity" },
            .served = &.{ .{ .rows = 1, .n = 2304, .k = 4 }, .{ .rows = 8, .n = 2304, .k = 4 }, .{ .rows = 2048, .n = 2304, .k = 4 } },
        };
    }

    pub fn control(_: *const Tap) ?g.Gate {
        return null;
    }

    pub fn free(t: *Tap, a: std.mem.Allocator) void {
        a.free(t.x);
        a.free(t.before);
        t.x = &.{};
        t.before = &.{};
    }

    pub fn prepare(t: *Tap, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k != 4) return false;
        t.free(a);
        t.shape = shape;
        t.x = try ref.bfs(a, rng, shape.rows * shape.n, -4, 4);
        t.before = try ref.bfs(a, rng, t.count(), -4, 4);
        if (fill == .nonfinite and t.x.len >= 3) for ([_]f32{ ref.nan, ref.inf, -ref.inf }, t.x[0..3]) |v, *x| {
            x.* = c.toBf16(v);
        };
        t.x_dev = try ref.input(rig, t.x);
        t.out = try rig.output(t.count() * 2);
        return true;
    }

    pub fn count(t: *const Tap) usize {
        return t.shape.rows * t.shape.n * 4;
    }

    pub fn run(t: *Tap, rig: *rig_.Rig) !void {
        try ref.restore(rig, t.out, std.mem.sliceAsBytes(t.before));
        try dk.tap(t.kern.tap, rig.s, t.x_dev, t.out, t.shape.rows, t.shape.n, t.shape.rows % 4);
    }

    pub fn solve(t: *const Tap, _: std.mem.Allocator, _: ref.Sums, out: []ref.Iv) !void {
        const n = t.shape.n;
        for (out, 0..) |*o, i| {
            const bits = if (i / n % 4 == t.shape.rows % 4) t.x[i / (4 * n) * n + i % n] else t.before[i];
            o.* = ref.Iv.of(ref.bf(bits));
        }
    }
};

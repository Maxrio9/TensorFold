//! The greedy head keeps exact scores, ties and overflow rows against the original CUDA head.
const std = @import("std");
const cuda = @import("cuda");
const ref = @import("contract_inkling2_ops_ref.zig");
const Rig = @import("contract_rig.zig").Rig;
const ink = @import("inkling2");
const c = @import("core").contract.spec;
const g = @import("core").contract.gates;

const cap: f32 = 30;
const shapes = [_]c.Shape{
    .{ .rows = 4, .n = 1025, .k = 2304 },
    .{ .rows = 1, .n = 1025, .k = 2304 },
    .{ .rows = 2, .n = 257, .k = 128 },
    .{ .rows = 32, .n = 257, .k = 128 },
};

pub const Check = struct {
    kern: *const ink.ops.Kernels,
    shape: c.Shape = undefined,
    head: ?ink.lazy_head.Head = null,
    x: u64 = 0,
    w: u64 = 0,
    output: u64 = 0,
    dense: []u8 = &.{},

    pub fn spec(_: *const Check) c.Spec {
        return .{
            .name = "inkling2 Mini certified greedy head",
            .accum = .none,
            .depth = ref.noSums,
            .out = .raw,
            .nan = .{ .stated = "on nonfinite activations, evaluate every logit in the original BF16 reduction order" },
            .served = &shapes,
        };
    }

    pub fn control(_: *const Check) ?g.Gate {
        return null;
    }

    pub fn free(self: *Check, a: std.mem.Allocator) void {
        if (self.head) |*head| head.deinit();
        self.head = null;
        a.free(self.dense);
        self.dense = &.{};
    }

    pub fn prepare(self: *Check, a: std.mem.Allocator, rig: *Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.rows > ink.lazy_head.max_rows or shape.n < 257 or !ink.lazy_head.eligible(shape.n, shape.k))
            return false;
        self.free(a);
        self.shape = shape;
        const x = try a.alloc(u16, shape.rows * shape.k);
        defer a.free(x);
        const w = try a.alloc(u16, shape.n * shape.k);
        defer a.free(w);
        @memset(x, 0);
        for (w) |*value| value.* = c.toBf16(rng.float(f32) * 0.01);
        w[shape.k] = c.toBf16(1);
        w[3 * shape.k] = c.toBf16(2);
        w[9 * shape.k] = c.toBf16(2);
        for (0..shape.rows) |row| {
            x[row * shape.k] = c.toBf16(switch (row % 4) {
                0 => 1,
                1 => 0, // every token ties. Force full fallback.
                2 => 1024, // three raw scores saturate to an equal final score.
                else => if (fill == .nonfinite) std.math.inf(f32) else -2,
            });
        }
        self.x = try ref.input(rig, x);
        self.w = try ref.input(rig, w);
        self.output = try rig.output(self.count());
        self.head = try ink.lazy_head.Head.init(a, rig.gpu.d, rig.s, self.w, shape.n, shape.k, cap);
        const baseline = try rig.scratch(self.count());
        var lm = try ink.loop_math_cuda.Kernels.load(rig.gpu.d);
        defer lm.unload();
        try ink.ops.rowdot(self.kern, rig.s, self.x, self.w, baseline, shape.rows, shape.n, shape.k, false, false);
        try ink.loop_math_cuda.roundBf16(&lm, rig.s, baseline, baseline, shape.rows * shape.n);
        try ink.ops.softcap(self.kern, rig.s, baseline, shape.rows * shape.n, cap);
        self.dense = try a.dupe(u8, try rig.bytes(baseline, self.count()));
        return true;
    }

    pub fn run(self: *Check, rig: *Rig) !void {
        try self.head.?.forward(rig.s, self.x, self.w, self.shape.rows, self.output);
    }

    pub fn count(self: *const Check) usize {
        return self.shape.rows * self.shape.n * @sizeOf(f32);
    }

    pub fn check(self: *const Check, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const n = self.shape.n;
        for (0..self.shape.rows) |row| {
            var got_id = n;
            var want_id = n;
            var got_max = -std.math.inf(f32);
            var want_max = -std.math.inf(f32);
            var pruned: usize = 0;
            for (0..n) |id| {
                const have = ref.readF32(got, row * n + id);
                const want = ref.readF32(self.dense, row * n + id);
                if (have != -std.math.inf(f32)) {
                    if (!ref.sameF32(have, want)) return try std.fmt.bufPrint(msg, "row {d} id {d} changed an exact logit", .{ row, id });
                    if (have > got_max) {
                        got_max = have;
                        got_id = id;
                    }
                } else pruned += 1;
                if (want > want_max) {
                    want_max = want;
                    want_id = id;
                }
            }
            if (got_id != want_id) return try std.fmt.bufPrint(msg, "row {d} pick {d}, expected {d}", .{ row, got_id, want_id });
            if (row % 4 == 0 and pruned == 0) return "clear winner left every logit active";
            if (row % 4 == 1 and pruned != 0) return "head skipped logits in an overflowing tie";
        }
        return null;
    }
};

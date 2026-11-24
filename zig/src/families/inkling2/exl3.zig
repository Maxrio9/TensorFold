//! Inkling 2's EXL3 tensors (exllamav3 1.x, mul1, an inkling2 architecture added to it): rank cuts, grouped experts.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

const Tensor = core.checkpoint.Tensor;

/// The mul1 codebook marker's value.
const mul1_magic: u32 = 0x83DCD12D;

/// What a rank takes of a projection: all of it, its output columns, or its input rows (each a multiple of 128).
pub const Span = struct {
    lo: usize,
    hi: usize,

    fn len(s: Span) usize {
        return s.hi - s.lo;
    }
};
pub const Cut = union(enum) { whole, cols: Span, rows: Span };

/// The device side of a load: all weights use the checkpoint's device arena.
pub const Sink = struct {
    ctx: *anyopaque,
    put: *const fn (ctx: *anyopaque, bytes: []const u8) anyerror!u64,
};

/// One stored projection's tensors, checked: trellis [k/16][n/16][16 * bits] i16, the mul1 marker, fp16 suh and svh.
pub const Group = struct {
    trellis: Tensor,
    suh: Tensor,
    svh: Tensor,
    halves: u32,

    fn tile(g: Group) usize {
        return g.halves * 16;
    }
};

pub fn group(ck: *core.Checkpoint, base: []const u8, n: usize, k: usize) !Group {
    var buf: [192]u8 = undefined;
    if (n == 0 or k == 0 or n % 128 != 0 or k % 128 != 0) return error.UnexpectedInkling2Tensor;
    const t = try ck.get(try std.fmt.bufPrint(&buf, "{s}.trellis", .{base}));
    if (t.dtype != .i16 or t.rank != 3 or t.dim(0) != k / 16 or t.dim(1) != n / 16 or t.dim(2) % 8 != 0) {
        std.log.err("inkling2: {s}.trellis is not [{d}, {d}, 16 * bits]", .{ base, k / 16, n / 16 });
        return error.UnexpectedInkling2Tensor;
    }
    const halves = t.dim(2) / 8;
    if (halves < 2 or halves > 16) return error.UnsupportedExl3;
    const marker = try ck.get(try std.fmt.bufPrint(&buf, "{s}.mul1", .{base}));
    if (marker.dtype != .i32 or marker.bytes.len != 4 or std.mem.readInt(u32, marker.bytes[0..4], .little) != mul1_magic) return error.UnsupportedExl3;
    const suh = try ck.get(try std.fmt.bufPrint(&buf, "{s}.suh", .{base}));
    const svh = try ck.get(try std.fmt.bufPrint(&buf, "{s}.svh", .{base}));
    if (!suh.is(.f16, &.{k}) or !svh.is(.f16, &.{n})) return error.UnexpectedInkling2Tensor;
    return .{ .trellis = t, .suh = suh, .svh = svh, .halves = @intCast(halves) };
}

/// The cut's shape (k, n) and, appended to `out`, its trellis words, suh and svh in the stored layout.
pub fn cut(gpa: std.mem.Allocator, g: Group, n: usize, k: usize, c: Cut, out: *std.ArrayList(u8), suh: *std.ArrayList(u8), svh: *std.ArrayList(u8)) ![2]usize {
    const tb = g.tile();
    const nt = n / 16;
    const tr = g.trellis.bytes;
    switch (c) {
        .whole => {
            try out.appendSlice(gpa, tr);
            try suh.appendSlice(gpa, g.suh.bytes);
            try svh.appendSlice(gpa, g.svh.bytes);
            return .{ k, n };
        },
        .cols => |s| {
            if (s.lo % 128 != 0 or s.hi % 128 != 0 or s.hi > n or s.hi <= s.lo) return error.BadSplit;
            for (0..k / 16) |kt| try out.appendSlice(gpa, tr[(kt * nt + s.lo / 16) * tb ..][0 .. s.len() / 16 * tb]);
            try suh.appendSlice(gpa, g.suh.bytes);
            try svh.appendSlice(gpa, g.svh.bytes[2 * s.lo .. 2 * s.hi]);
            return .{ k, s.len() };
        },
        .rows => |s| {
            if (s.lo % 128 != 0 or s.hi % 128 != 0 or s.hi > k or s.hi <= s.lo) return error.BadSplit;
            try out.appendSlice(gpa, tr[s.lo / 16 * nt * tb .. s.hi / 16 * nt * tb]);
            try suh.appendSlice(gpa, g.suh.bytes[2 * s.lo .. 2 * s.hi]);
            try svh.appendSlice(gpa, g.svh.bytes);
            return .{ s.len(), n };
        },
    }
}

/// The stored [n][k] projection under `base`, cut to the rank, on the device for cuda.exl3.Dense.
pub fn linear(sink: Sink, g: Group, n: usize, k: usize) !cuda.exl3.Weight {
    if (cuda.exl3.variantIndex(g.halves, .mul1, 4) == null) return error.UnsupportedExl3;
    const st = cuda.exl3.strides(k, n, g.halves, false);
    return .{
        .trellis = try sink.put(sink.ctx, g.trellis.bytes),
        .suh = try sink.put(sink.ctx, g.suh.bytes),
        .svh = try sink.put(sink.ctx, g.svh.bytes),
        .k = @intCast(k),
        .n = @intCast(n),
        .halves = g.halves,
        .codebook = .mul1,
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
}

test "exl3: a cut takes a rank's tile columns of every k row, or its whole tile rows, with the matching scales" {
    const a = std.testing.allocator;
    var words: [16 * 16 * 32]u8 = undefined;
    for (0..16) |kt| for (0..16) |nt| @memset(words[(kt * 16 + nt) * 32 ..][0..32], @intCast(kt * 16 + nt));
    var scales: [512]u8 = undefined;
    for (&scales, 0..) |*s, j| s.* = @intCast(j / 2 % 256);
    const t: Tensor = .{ .dtype = .i16, .rank = 3, .shape = .{ 16, 16, 16, 1, 1 }, .bytes = &words };
    const v: Tensor = .{ .dtype = .f16, .rank = 1, .shape = .{ 256, 1, 1, 1, 1 }, .bytes = &scales };
    const g: Group = .{ .trellis = t, .suh = v, .svh = v, .halves = 2 };
    var tr: std.ArrayList(u8) = .empty;
    defer tr.deinit(a);
    var su: std.ArrayList(u8) = .empty;
    defer su.deinit(a);
    var sv: std.ArrayList(u8) = .empty;
    defer sv.deinit(a);
    try std.testing.expectEqual([2]usize{ 256, 128 }, try cut(a, g, 256, 256, .{ .cols = .{ .lo = 128, .hi = 256 } }, &tr, &su, &sv));
    try std.testing.expectEqual(@as(u8, 8), tr.items[0]);
    try std.testing.expectEqual(@as(u8, 16 + 8), tr.items[8 * 32]);
    try std.testing.expectEqualSlices(u8, scales[256..], sv.items);
    tr.clearRetainingCapacity();
    su.clearRetainingCapacity();
    sv.clearRetainingCapacity();
    try std.testing.expectEqual([2]usize{ 128, 256 }, try cut(a, g, 256, 256, .{ .rows = .{ .lo = 128, .hi = 256 } }, &tr, &su, &sv));
    try std.testing.expectEqual(@as(u8, 8 * 16), tr.items[0]);
    try std.testing.expectEqualSlices(u8, scales[256..], su.items);
    try std.testing.expectError(error.BadSplit, cut(a, g, 256, 256, .{ .cols = .{ .lo = 64, .hi = 192 } }, &tr, &su, &sv));
}

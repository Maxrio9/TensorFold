//! Row-local LoopCD mix against an independent host BF16 reference.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const lm = @import("inkling2").loop_math_cuda;
const bf = @import("core").contract.spec.toBf16;

fn value(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn run(gpu: check.Gpu) !void {
    const n = 3 * 259;
    var host: [4][n]u16 = undefined;
    for (0..n) |i| {
        host[0][i] = bf(@as(f32, @floatFromInt(i % 29)) / 8 - 2);
        host[1][i] = bf(@as(f32, @floatFromInt(i % 17)) / 4 - 3);
        host[2][i] = bf(@as(f32, @floatFromInt(i % 13)) / 8 - 1);
        host[3][i] = bf(@as(f32, @floatFromInt(i % 23)) / 4 - 2);
    }
    var buf: [4]cuda.DeviceBuffer = undefined;
    var count: usize = 0;
    defer for (buf[0..count]) |*b| b.free();
    for (&buf, 0..) |*b, i| {
        b.* = try cuda.DeviceBuffer.alloc(gpu.d, n * 2);
        count += 1;
        try b.upload(0, std.mem.sliceAsBytes(&host[i]));
    }
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var k = try lm.Kernels.load(gpu.d);
    defer k.unload();
    try lm.hiddenMix(&k, stream, buf[0].ptr, buf[1].ptr, buf[2].ptr, buf[3].ptr, n, 0.3);
    try stream.synchronize();
    var got: [2][n]u16 = undefined;
    for (0..2) |role| {
        try buf[role].download(0, std.mem.sliceAsBytes(&got[role]));
        for (0..n) |i| {
            const final = value(host[role][i]);
            const delta = final - value(host[role + 2][i]);
            const scaled = @as(f32, 0.3) * delta;
            const want = bf(final + scaled);
            try check.expect(got[role][i] == want, "LoopCD mix role {d}, element {d}", .{ role, i });
        }
    }
    std.debug.print("LOOPCD MIX {d}/{d} values passed\n", .{ 2 * n, 2 * n });
}

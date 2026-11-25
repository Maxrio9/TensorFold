//! Compare the complete Mini DFlash2 arena from the GPU packer and the on-disk FP8 checkpoint.
const std = @import("std");
const cuda = @import("cuda");
const dflash = @import("inkling2").dflash2_weights;
const check = @import("check.zig");

pub fn run(gpu: check.Gpu, bf16: []const u8, fp8: []const u8) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    const real_vocab = 200058;
    var converted = try dflash.load(gpu.gpa, gpu.io, bf16, real_vocab, .{ .layers = true }, .{
        .stream = stream,
        .source = 0,
        .borrowed = null,
    });
    defer converted.deinit();
    var exported = try dflash.load(gpu.gpa, gpu.io, fp8, real_vocab, .{}, .{
        .stream = stream,
        .source = 0,
        .borrowed = null,
    });
    defer exported.deinit();
    try check.expect(converted.fp8_layers and exported.fp8_layers, "DFlash2 FP8 layer mode", .{});
    const bytes = converted.memory.len;
    try check.expect(bytes == exported.memory.len, "DFlash2 FP8 matched arena size", .{});
    const block = 4 * 1024 * 1024;
    const a = try gpu.gpa.alloc(u8, block);
    defer gpu.gpa.free(a);
    const b = try gpu.gpa.alloc(u8, block);
    defer gpu.gpa.free(b);
    var pos: usize = 0;
    while (pos < bytes) {
        const length = @min(block, bytes - pos);
        const ref = a[0..length];
        const got = b[0..length];
        try converted.memory.download(pos, ref);
        try exported.memory.download(pos, got);
        try check.expect(std.mem.eql(u8, ref, got), "DFlash2 export differs in arena at [{d}, {d})", .{ pos, pos + length });
        pos += length;
    }
    std.debug.print("MINI DFLASH2 EXPORT {d} device bytes identical\n", .{bytes});
}

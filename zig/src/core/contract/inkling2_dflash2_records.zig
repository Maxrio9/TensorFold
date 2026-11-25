//! Mini DFlash2 CUDA launch records for nvcc 13.0 and 13.4.
const registry = @import("registry.zig");
const limits = @import("limits.zig");

fn usage(regs: [limits.sms.len]u32, shared: u32, threads: u32) [limits.sms.len]limits.Usage {
    var out: [limits.sms.len]limits.Usage = undefined;
    for (&out, regs) |*u, r| u.* = .{ .regs = @intCast(r), .static = shared, .bound = threads };
    return out;
}

const prefix = "inkling2 dflash2 ";

fn kernel(comptime symbol: []const u8, threads: u32, registers: [limits.sms.len]u32, shared: u32) registry.CudaRecord {
    return .{
        .launch = .{ .name = prefix ++ symbol, .symbol = symbol, .image = "inkling2_dspark", .threads = threads, .min_sm = 80 },
        .usage = usage(registers, shared, threads),
        .later_pending = true,
    };
}

pub const cuda = [_]registry.CudaRecord{
    kernel("tf_ink2_df_qk", 64, .{ 28, 28, 28, 24, 26 }, 128),
    kernel("tf_ink2_df_attn", 256, .{ 70, 70, 70, 54, 54 }, 48192),
    kernel("tf_ink2_df_conv", 256, .{ 25, 24, 24, 32, 32 }, 0),
    kernel("tf_ink2_df_select", 512, .{ 32, 34, 34, 28, 28 }, 64),
    kernel("tf_ink2_df_select8", 512, .{ 24, 25, 40, 32, 32 }, 64),
};

pub const nvcc_134 = [_]registry.Override{
    .{ .name = prefix ++ "tf_ink2_df_qk", .usage = usage(.{ 31, 31, 31, 28, 28 }, 128, 64) },
    .{ .name = prefix ++ "tf_ink2_df_attn", .usage = usage(.{ 70, 70, 70, 53, 53 }, 48192, 256) },
    .{ .name = prefix ++ "tf_ink2_df_conv", .usage = usage(.{ 25, 24, 24, 32, 32 }, 0, 256) },
    .{ .name = prefix ++ "tf_ink2_df_select", .usage = usage(.{ 32, 34, 34, 26, 26 }, 64, 512) },
    .{ .name = prefix ++ "tf_ink2_df_select8", .usage = usage(.{ 24, 25, 40, 30, 30 }, 64, 512) },
};

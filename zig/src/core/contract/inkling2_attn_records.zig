//! Inkling 2's inkling2_attn, inkling2_pattn images: every SM's compile records (nvcc 13.0, 13.2, 13.3).
const registry = @import("registry.zig");

/// nvcc 13.0's usage, the compiler of the release fatbins.
pub const cuda = [_]registry.CudaRecord{
    .{ .launch = .{ .name = "inkling2_attn tf_ink2_prep", .symbol = "tf_ink2_prep", .image = "inkling2_attn", .threads = 32, .min_sm = 80 }, .usage = .{ .{ .regs = 26, .static = 0 }, .{ .regs = 26, .static = 0 }, .{ .regs = 26, .static = 0 }, .{ .regs = 28, .static = 0 }, .{ .regs = 28, .static = 0 } } },
    .{ .launch = .{ .name = "inkling2_attn tf_ink2_rel_table", .symbol = "tf_ink2_rel_table", .image = "inkling2_attn", .threads = 256, .min_sm = 80 }, .usage = .{ .{ .regs = 31, .static = 0 }, .{ .regs = 38, .static = 0 }, .{ .regs = 38, .static = 0 }, .{ .regs = 39, .static = 0 }, .{ .regs = 39, .static = 0 } } },
    .{ .launch = .{ .name = "inkling2_attn tf_ink2_attn", .symbol = "tf_ink2_attn", .image = "inkling2_attn", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 208, .static = 34816, .bound = 128 }, .{ .regs = 208, .static = 34816, .bound = 128 }, .{ .regs = 208, .static = 34816, .bound = 128 }, .{ .regs = 188, .static = 34816, .bound = 128 }, .{ .regs = 188, .static = 34816, .bound = 128 } } },
    .{ .launch = .{ .name = "inkling2_attn tf_ink2_attn_fold", .symbol = "tf_ink2_attn_fold", .image = "inkling2_attn", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 254, .static = 34816, .bound = 128 }, .{ .regs = 254, .static = 34816, .bound = 128 }, .{ .regs = 254, .static = 34816, .bound = 128 }, .{ .regs = 255, .static = 34816, .bound = 128 }, .{ .regs = 255, .static = 34816, .bound = 128 } } },
    .{ .launch = .{ .name = "inkling2_attn tf_ink2_merge", .symbol = "tf_ink2_merge", .image = "inkling2_attn", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 32, .static = 0 }, .{ .regs = 40, .static = 0 }, .{ .regs = 40, .static = 0 }, .{ .regs = 40, .static = 0 }, .{ .regs = 40, .static = 0 } } },
    .{ .launch = .{ .name = "inkling2_pattn tf_ink2_attn_prompt", .symbol = "tf_ink2_attn_prompt", .image = "inkling2_pattn", .threads = 256, .dynamic = 98304, .opt_in = true, .min_sm = 80 }, .usage = .{ .{ .regs = 255, .static = 0, .bound = 256 }, .{ .regs = 255, .static = 0, .bound = 256 }, .{ .regs = 255, .static = 0, .bound = 256 }, .{ .regs = 255, .static = 0, .bound = 256 }, .{ .regs = 255, .static = 0, .bound = 256 } } },
    // FP8 launches compiled with nvcc 13.4.92 across all five release SMs.
    .{ .launch = .{ .name = "inkling2_attn tf_ink2_prep_fp8", .symbol = "tf_ink2_prep_fp8", .image = "inkling2_attn", .threads = 32, .min_sm = 80 }, .usage = .{ .{ .regs = 27, .static = 0 }, .{ .regs = 29, .static = 0 }, .{ .regs = 29, .static = 0 }, .{ .regs = 30, .static = 0 }, .{ .regs = 30, .static = 0 } } },
    .{ .launch = .{ .name = "inkling2_attn tf_ink2_attn_fp8", .symbol = "tf_ink2_attn_fp8", .image = "inkling2_attn", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 206, .static = 34816, .bound = 128 }, .{ .regs = 212, .static = 34816, .bound = 128 }, .{ .regs = 212, .static = 34816, .bound = 128 }, .{ .regs = 190, .static = 34816, .bound = 128 }, .{ .regs = 190, .static = 34816, .bound = 128 } } },
    .{ .launch = .{ .name = "inkling2_attn tf_ink2_attn_fold_fp8", .symbol = "tf_ink2_attn_fold_fp8", .image = "inkling2_attn", .threads = 128, .min_sm = 80 }, .usage = .{ .{ .regs = 255, .static = 34816, .bound = 128 }, .{ .regs = 255, .static = 34816, .bound = 128 }, .{ .regs = 255, .static = 34816, .bound = 128 }, .{ .regs = 255, .static = 34816, .bound = 128 }, .{ .regs = 255, .static = 34816, .bound = 128 } } },
    .{ .launch = .{ .name = "inkling2_pattn tf_ink2_attn_prompt_fp8", .symbol = "tf_ink2_attn_prompt_fp8", .image = "inkling2_pattn", .threads = 256, .dynamic = 32768, .min_sm = 80 }, .usage = .{ .{ .regs = 255, .static = 0, .bound = 256 }, .{ .regs = 255, .static = 0, .bound = 256 }, .{ .regs = 255, .static = 0, .bound = 256 }, .{ .regs = 255, .static = 0, .bound = 256 }, .{ .regs = 255, .static = 0, .bound = 256 } } },
    .{ .launch = .{ .name = "inkling2_pattn tf_ink2_attn_slide", .symbol = "tf_ink2_attn_slide", .image = "inkling2_pattn", .threads = 256, .dynamic = 98304, .opt_in = true, .min_sm = 80 }, .usage = @splat(.{ .regs = 255, .static = 0, .bound = 256 }) },
    .{ .launch = .{ .name = "inkling2_pattn tf_ink2_attn_slide_fp8", .symbol = "tf_ink2_attn_slide_fp8", .image = "inkling2_pattn", .threads = 256, .dynamic = 32768, .min_sm = 80 }, .usage = @splat(.{ .regs = 255, .static = 0, .bound = 256 }) },
    // Paired prompt projections, nvcc 13.4.92.
    .{
        .launch = .{ .name = "exl3_linear tf_ink2_exl3_prompt_rotate_pair", .symbol = "tf_ink2_exl3_prompt_rotate_pair", .image = "exl3_linear", .threads = 128, .min_sm = 80 },
        .usage = .{
            .{ .regs = 22, .static = 0, .bound = 128 },
            .{ .regs = 24, .static = 0, .bound = 128 },
            .{ .regs = 24, .static = 0, .bound = 128 },
            .{ .regs = 24, .static = 0, .bound = 128 },
            .{ .regs = 24, .static = 0, .bound = 128 },
        },
    },
    .{
        .launch = .{ .name = "exl3_linear tf_ink2_exl3_prompt_pair_1", .symbol = "tf_ink2_exl3_prompt_pair_1", .image = "exl3_linear", .threads = 128, .dynamic = 16384, .min_sm = 80 },
        .usage = .{
            .{ .regs = 166, .static = 128, .bound = 128 },
            .{ .regs = 168, .static = 128, .bound = 128 },
            .{ .regs = 168, .static = 128, .bound = 128 },
            .{ .regs = 165, .static = 128, .bound = 128 },
            .{ .regs = 165, .static = 128, .bound = 128 },
        },
    },
    .{
        .launch = .{ .name = "exl3_linear tf_ink2_exl3_prompt_pair_2", .symbol = "tf_ink2_exl3_prompt_pair_2", .image = "exl3_linear", .threads = 128, .dynamic = 16384, .min_sm = 80 },
        .usage = .{
            .{ .regs = 255, .static = 128, .bound = 128 },
            .{ .regs = 255, .static = 128, .bound = 128 },
            .{ .regs = 255, .static = 128, .bound = 128 },
            .{ .regs = 252, .static = 128, .bound = 128 },
            .{ .regs = 252, .static = 128, .bound = 128 },
        },
    },
};
/// nvcc 13.2's, where it differs: pending, as no 13.2 compiler has measured these images.
pub const cuda_132 = [_]registry.Override{};
/// nvcc 13.3's, where it differs.
pub const cuda_133 = [_]registry.Override{
    .{ .name = "inkling2_attn tf_ink2_attn", .usage = .{ .{ .regs = 208, .static = 34816, .bound = 128 }, .{ .regs = 208, .static = 34816, .bound = 128 }, .{ .regs = 208, .static = 34816, .bound = 128 }, .{ .regs = 202, .static = 34816, .bound = 128 }, .{ .regs = 202, .static = 34816, .bound = 128 } } },
};

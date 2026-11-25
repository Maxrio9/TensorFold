//! Run the Mini EXL3 checks with the serving images and without other families' contract fixtures.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const gates = @import("core").contract.gates;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2 and args.len != 4 and args.len != 5 and args.len != 7) return error.BadArguments;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    if (args.len == 2 and std.mem.eql(u8, args[1], "inkling2-mini-exl3-contract"))
        return @import("contract_inkling2_mini_exl3.zig").run(gpu);
    if (args.len == 2 and std.mem.eql(u8, args[1], "inkling2-mini-regressions"))
        return regressions(gpu);
    if (args.len == 2 and std.mem.eql(u8, args[1], "inkling2-mini-dspark-contract"))
        return dsparkContract(gpu);
    if (args.len == 2 and std.mem.eql(u8, args[1], "inkling2-mini-kv-contract"))
        return cacheContract(gpu);
    if (args.len == 2 and std.mem.eql(u8, args[1], "inkling2-mini-paged-contract"))
        return pagedContract(gpu);
    if (args.len == 2 and std.mem.eql(u8, args[1], "inkling2-mini-slide-contract"))
        return slideContract(gpu);
    if (args.len == 2 and std.mem.eql(u8, args[1], "inkling2-mini-head-contract"))
        return headContract(gpu);
    if (args.len == 4 and std.mem.eql(u8, args[1], "inkling2-mini-exl3-teacher"))
        return @import("inkling2_mini_exl3_teacher.zig").run(gpu, args[2], args[3]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "inkling2-mini-dflash2-export"))
        return @import("inkling2_mini_dflash2_export.zig").run(gpu, args[2], args[3]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "inkling2-mini-kv-fidelity"))
        return @import("inkling2_mini_verify.zig").cacheFidelity(gpu, args[2], args[3]);
    if (args.len == 4 and std.mem.eql(u8, args[1], "inkling2-mini-concurrent"))
        return @import("inkling2_mini_verify.zig").concurrent(gpu, args[2], args[3]);
    if (args.len == 5 and std.mem.eql(u8, args[1], "inkling2-mini-prefill-logits"))
        return @import("inkling2_mini_verify.zig").prefillLogits(gpu, args[2], args[3], args[4]);
    if (args.len == 5 and std.mem.eql(u8, args[1], "inkling2-mini-verify"))
        return @import("inkling2_mini_verify.zig").run(gpu, args[2], args[3], args[4]);
    if (args.len == 5 and std.mem.eql(u8, args[1], "inkling2-mini-draft-acceptance"))
        return @import("inkling2_mini_draft_acceptance.zig").run(gpu, args[2], args[3], args[4]);
    if (args.len == 7 and std.mem.eql(u8, args[1], "inkling2-mini-release-fidelity")) {
        const mode = std.meta.stringToEnum(@import("inkling2_mini_verify.zig").FidelityMode, args[2]) orelse
            return error.BadArguments;
        return @import("inkling2_mini_verify.zig").releaseFidelity(gpu, mode, args[3], args[4], args[5], args[6]);
    }
    return error.BadArguments;
}

fn regressions(gpu: check.Gpu) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var kernels: std.ArrayList(gates.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    const set = try @import("contract_inkling2.zig").Set.init(gpu, stream, gpu.gpa, &kernels);
    defer set.deinit();
    @import("contract_after.zig").keep(&kernels, &.{
        "inkling2 Mini NVFP4 decode",           "inkling2 Mini MXFP8 decode",
        "inkling2 loop tf_ink2_loop_gate_down", "inkling2 loop rank-64 gate",
    });
    var tally: gates.Tally = .{};
    try gates.runAll(gpu.gpa, kernels.items, 0x6b63, &tally);
    std.debug.print("MINI REGRESSIONS {d}/{d} kernels, {d}/{d} controls\n", .{
        tally.passed, tally.kernels, tally.caught, tally.controls,
    });
    try check.expect(tally.kernels == 4 and tally.controls == 3 and tally.problems() == 0, "Mini regression contract", .{});
}

fn dsparkContract(gpu: check.Gpu) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var kernels: std.ArrayList(gates.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    var set: @import("contract_inkling2_dspark.zig").Set = .{};
    try set.init(gpu, stream, gpu.gpa, &kernels);
    defer set.deinit();
    var tally: gates.Tally = .{};
    try gates.runAll(gpu.gpa, kernels.items, 0x6b63, &tally);
    std.debug.print("MINI DSPARK CONTRACT {d}/{d} kernels, {d}/{d} controls, {d} problems\n", .{
        tally.passed, tally.kernels, tally.caught, tally.controls, tally.problems(),
    });
    try check.expect(tally.kernels != 0 and tally.problems() == 0, "Mini DSpark contract", .{});
}

fn headContract(gpu: check.Gpu) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var kernels: std.ArrayList(gates.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    var set: @import("contract_inkling2_ops.zig").Set = .{};
    try set.init(gpu, stream, gpu.gpa, &kernels);
    defer set.deinit();
    @import("contract_after.zig").keep(&kernels, &.{ "inkling2 Mini warp router", "inkling2 Mini certified greedy head" });
    var tally: gates.Tally = .{};
    try gates.runAll(gpu.gpa, kernels.items, 0x6b63, &tally);
    std.debug.print("MINI HEAD CONTRACT {d}/{d} kernels, {d} problems\n", .{ tally.passed, tally.kernels, tally.problems() });
    try check.expect(tally.kernels == 2 and tally.problems() == 0, "Mini head and router contract", .{});
}

fn cacheContract(gpu: check.Gpu) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var kernels: std.ArrayList(gates.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    var set: @import("contract_inkling2_attn.zig").Set = .{};
    try set.init(gpu, stream, gpu.gpa, &kernels);
    defer set.deinit();
    try set.preps[0].checkFp8(.{ .rows = 7, .n = 24 * 128, .k = 3076 }, .random);
    try set.preps[1].checkFp8(.{ .rows = 7, .n = 24 * 128, .k = 128004 }, .nonfinite);
    std.debug.print("MINI FP8 PREP 2/2 code, scale and guard fixtures\n", .{});
    @import("contract_after.zig").keep(&kernels, &.{"inkling2 attn FP8"});
    var tally: gates.Tally = .{};
    try gates.runAll(gpu.gpa, kernels.items, 0x6b63, &tally);
    std.debug.print("MINI FP8 KV CONTRACT {d}/{d} kernels, {d} problems\n", .{ tally.passed, tally.kernels, tally.problems() });
    try check.expect(tally.kernels == 6 and tally.problems() == 0, "Mini FP8 KV contract", .{});
}

fn slideContract(gpu: check.Gpu) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var kernels: std.ArrayList(gates.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    var set: @import("contract_inkling2_attn.zig").Set = .{};
    try set.init(gpu, stream, gpu.gpa, &kernels);
    defer set.deinit();
    const rows = @import("contract_inkling2_attn_rows.zig");
    @import("contract_after.zig").keep(&kernels, &.{ rows.slide_prompt_name, rows.slide_prompt_fp8_name });
    var tally: gates.Tally = .{};
    try gates.runAll(gpu.gpa, kernels.items, 0x6b63, &tally);
    std.debug.print("MINI SLIDING PROMPT CONTRACT {d}/{d} kernels, {d} problems\n", .{ tally.passed, tally.kernels, tally.problems() });
    try check.expect(tally.kernels == 2 and tally.problems() == 0, "Mini sliding prompt contract", .{});
}

fn pagedContract(gpu: check.Gpu) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var kernels: std.ArrayList(gates.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    var set: @import("contract_inkling2_attn.zig").Set = .{};
    try set.init(gpu, stream, gpu.gpa, &kernels);
    defer set.deinit();
    try set.preps[3].checkFp8(.{ .rows = 7, .n = 24 * 128, .k = 261 }, .random);
    try set.preps[3].checkFp8(.{ .rows = 3, .n = 24 * 128, .k = 257 }, .nonfinite);
    std.debug.print("MINI PAGED FP8 PREP 2/2 code, scale and guard fixtures\n", .{});
    @import("contract_after.zig").keep(&kernels, &.{"inkling2 attn paged"});
    var tally: gates.Tally = .{};
    try gates.runAll(gpu.gpa, kernels.items, 0x6b63, &tally);
    std.debug.print("MINI PAGED KV CONTRACT {d}/{d} kernels, {d} problems\n", .{ tally.passed, tally.kernels, tally.problems() });
    try check.expect(tally.kernels == 8 and tally.problems() == 0, "Mini paged KV contract", .{});
}

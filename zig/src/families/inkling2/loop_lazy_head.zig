//! Mini's certified greedy head using an E4M3 proxy, outward score bounds and exact BF16 candidate dots.

// Copyright 2026 MiaAI-Lab
const std = @import("std");
const cuda = @import("cuda");

pub const max_rows = 128;
const candidates = 256;
const min_rank = 0x007f; // ordered BF16 bytes for negative infinity
const max_rank = 0xff80;
const rank_count = max_rank - min_rank + 1;

pub fn enabled() bool {
    return @import("sys").environment.read(bool, "TF_INKLING2_LAZY_HEAD", false) catch false;
}

pub fn eligible(n: usize, k: usize) bool {
    return n != 0 and k != 0 and k <= 4096 and k % 64 == 0;
}

pub fn codeBytes(n: usize, k: usize) usize {
    return cuda.fp8.padded(n) * k;
}

pub fn allocationBytes(n: usize, k: usize) usize {
    if (!eligible(n, k)) return 0;
    return codeBytes(n, k) + n * 4 + rank_count * 2 + max_rows * (3 * 4 + candidates * 4);
}

fn int(value: usize) i32 {
    return @intCast(value);
}

fn go(f: cuda.Function, s: cuda.Stream, x: usize, y: usize, args: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(x), .y = @intCast(y) }, .block = .{ .x = 256 } }, s, args);
}

/// A borrowed FP8 head's codes live in the destination arena supplied by the caller.
pub fn pack(d: *const cuda.Driver, s: cuda.Stream, source: u64, dest: u64, n: usize, k: usize) !cuda.qmmf.Weight {
    var mod = try cuda.Module.load(d, cuda.kernels.inkling2_ops);
    defer mod.unload();
    return packWithModule(mod, d, s, source, dest, n, k);
}

fn packWithModule(mod: cuda.Module, d: *const cuda.Driver, s: cuda.Stream, source: u64, dest: u64, n: usize, k: usize) !cuda.qmmf.Weight {
    if (!eligible(n, k)) return error.Inkling2FP8HeadShape;
    var amax = try cuda.DeviceBuffer.alloc(d, 4);
    defer amax.free();
    errdefer s.synchronize() catch {};
    try amax.fill8(0, s.handle);
    var a: cuda.Args = .{};
    a.add(source);
    a.add(@as(i64, @intCast(n * k)));
    a.add(amax.ptr);
    try go(try mod.function("tf_ink2_lazy_amax"), s, 256, 1, &a);
    try s.synchronize();
    var value: u32 = 0;
    try amax.download(0, std.mem.asBytes(&value));
    if (value > 0x4b80) return error.Inkling2FP8HeadWeights;
    const biggest: f32 = @bitCast(value << 16);
    const scale = @exp2(@ceil(@log2(@max(biggest, 0x1p-16) / 448.0)));
    const total = codeBytes(n, k);
    a = .{};
    a.add(source);
    a.add(dest);
    a.add(int(n));
    a.add(int(k));
    a.add(1.0 / scale);
    a.add(@as(i64, @intCast(total)));
    try go(try mod.function("tf_ink2_lazy_pack"), s, (total + 255) / 256, 1, &a);
    try s.synchronize();
    return .{ .mode = .fp8, .codes = dest, .scales = 0, .scale = scale, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(cuda.fp8.padded(n)), .sk = 1 };
}

/// Possible softcapped values need not be strictly increasing, especially at saturation.
fn makeCutoffs(scores: []const f32, suffix: []f32, cutoffs: []u16) !void {
    var lowest = std.math.inf(f32);
    var i = scores.len;
    while (i != 0) {
        i -= 1;
        const score = scores[i];
        if (!std.math.isFinite(score)) return error.Inkling2FP8HeadCap;
        lowest = @min(lowest, score);
        suffix[i] = lowest;
    }
    var upper: usize = 0;
    var highest = -std.math.inf(f32);
    for (suffix, cutoffs) |floor, *cutoff| {
        while (upper < scores.len) : (upper += 1) {
            const next = @max(highest, scores[upper]);
            if (next >= floor) break;
            highest = next;
        }
        cutoff.* = @intCast(min_rank - 1 + upper);
    }
}

const Device = struct {
    coeff: u64,
    cutoffs: u64,
    norm: u64,
    floor: u64,
    counts: u64,
    picks: u64,

    fn layout(ptr: u64, n: usize, k: usize) Device {
        const coeff = ptr + codeBytes(n, k);
        const cutoffs = coeff + n * 4;
        const norm = cutoffs + rank_count * 2;
        return .{ .coeff = coeff, .cutoffs = cutoffs, .norm = norm, .floor = norm + max_rows * 4, .counts = norm + max_rows * 8, .picks = norm + max_rows * 12 };
    }
};

pub const Head = struct {
    buffer: cuda.DeviceBuffer,
    mod: cuda.Module,
    lane: cuda.qmmf.Lane,
    weight: cuda.qmmf.Weight,
    funcs: [4]cuda.Function,
    device: Device,
    cap: f32,

    pub fn init(a: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, source: u64, n: usize, k: usize, cap: f32) !Head {
        if (!eligible(n, k)) return error.Inkling2FP8HeadShape;
        if (!std.math.isFinite(cap) or cap <= 0) return error.Inkling2FP8HeadCap;
        var buf = try cuda.DeviceBuffer.alloc(d, allocationBytes(n, k));
        errdefer buf.free();
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_ops);
        errdefer mod.unload();
        var lane = try cuda.qmmf.Lane.load(d, (try d.currentCapability()) / 10);
        errdefer lane.unload();
        errdefer s.synchronize() catch {};
        const weight = try packWithModule(mod, d, s, source, buf.ptr, n, k);
        const device = Device.layout(buf.ptr, n, k);
        var args: cuda.Args = .{};
        args.add(source);
        args.add(device.coeff);
        args.add(int(n));
        args.add(int(k));
        args.add(weight.scale);
        args.add(1.0 / weight.scale);
        try go(try mod.function("tf_ink2_lazy_coeff"), s, (n + 7) / 8, 1, &args);
        try uploadCutoffs(a, d, s, mod, buf, device.cutoffs - buf.ptr, cap);
        return .{
            .buffer = buf,
            .mod = mod,
            .lane = lane,
            .weight = weight,
            .device = device,
            .cap = cap,
            .funcs = .{
                try mod.function("tf_ink2_lazy_prepare"),
                try mod.function("tf_ink2_lazy_floor"),
                try mod.function("tf_ink2_lazy_select"),
                try mod.function("tf_ink2_lazy_exact"),
            },
        };
    }

    pub fn deinit(head: *Head) void {
        head.lane.unload();
        head.mod.unload();
        head.buffer.free();
    }

    /// The returned row is for greedy draws only. Every retained logit has the dense head's exact value.
    pub fn forward(head: *const Head, s: cuda.Stream, x: u64, weights: u64, rows: usize, logits: u64) !void {
        if (rows == 0 or rows > max_rows) return error.Inkling2FP8HeadShape;
        const d = head.device;
        const n = int(head.weight.n);
        const k = int(head.weight.k);
        var a: cuda.Args = .{};
        a.add(x);
        a.add(k);
        for ([_]u64{ d.norm, d.floor, d.counts }) |p| a.add(p);
        a.add(int(candidates));
        try go(head.funcs[0], s, rows, 1, &a);
        try head.lane.matmulOut(s, x, head.weight.k, rows, head.weight, logits, .f32, 0);
        a = .{};
        for ([_]u64{ logits, d.coeff, d.norm, d.floor }) |p| a.add(p);
        a.add(n);
        try go(head.funcs[1], s, 96, rows, &a);
        a = .{};
        for ([_]u64{ logits, d.coeff, d.norm, d.floor, d.cutoffs }) |p| a.add(p);
        a.add(n);
        a.add(d.counts);
        a.add(d.picks);
        a.add(int(candidates));
        try go(head.funcs[2], s, 96, rows, &a);
        a = .{};
        for ([_]u64{ x, weights, logits, d.counts, d.picks }) |p| a.add(p);
        for ([_]i32{ int(candidates), n, k }) |value| a.add(value);
        a.add(head.cap);
        try go(head.funcs[3], s, @max(32, 256 / rows), rows, &a);
    }
};

fn uploadCutoffs(a: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, mod: cuda.Module, buf: cuda.DeviceBuffer, at: usize, cap: f32) !void {
    var device_scores = try cuda.DeviceBuffer.alloc(d, rank_count * 4);
    defer device_scores.free();
    errdefer s.synchronize() catch {};
    const scores = try a.alloc(f32, rank_count);
    defer a.free(scores);
    const suffix = try a.alloc(f32, rank_count);
    defer a.free(suffix);
    const cutoffs = try a.alloc(u16, rank_count);
    defer a.free(cutoffs);
    var args: cuda.Args = .{};
    args.add(device_scores.ptr);
    args.add(cap);
    try go(try mod.function("tf_ink2_lazy_caps"), s, (rank_count + 255) / 256, 1, &args);
    try s.synchronize();
    try device_scores.download(0, std.mem.sliceAsBytes(scores));
    try makeCutoffs(scores, suffix, cutoffs);
    try buf.upload(at, std.mem.sliceAsBytes(cutoffs));
}

test "lazy head cutoff retains saturation ties and handles nonmonotone softcaps" {
    const values = [_]f32{ -3, 1, 0, 1, 2, 2 };
    var suffix: [values.len]f32 = undefined;
    var cutoffs: [values.len]u16 = undefined;
    try makeCutoffs(&values, &suffix, &cutoffs);
    try std.testing.expectEqualSlices(u16, &.{ min_rank - 1, min_rank, min_rank, min_rank, min_rank + 3, min_rank + 3 }, &cutoffs);
}

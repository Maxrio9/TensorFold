//! Mini's separate BF16 drafter checkpoint and its folded Markov output matrix.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const names = @import("names.zig");
const fold = @import("metal/dspark_fold.zig");

pub const hidden = fold.hidden;
pub const vocab = fold.vocabulary;
pub const rank = fold.rank;
pub const kv = 1024;
pub const q = 3072;
pub const inter = 9216;
pub const block = 8;
pub const prepared_head_bytes = vocab * (rank + 1) * 2;
pub const taps = [_]u32{ 5, 23, 41, 48 };

const Layer = struct {
    dflare: u64,
    input_norm: u64,
    post_norm: u64,
    q_norm: u64,
    k_norm: u64,
    ctx_norm: u64,
    qkv: u64,
    ctx_kv: u64,
    out: u64,
    gate_up: u64,
    down: u64,
};

pub const Weights = struct {
    memory: cuda.DeviceBuffer,
    layers: [3]Layer,
    alpha: u64,
    mask: u64,
    norm: u64,
    markov_norm: u64,
    markov_w1: u64,
    rnn: u64,
    rnn_bias: u64,
    phi: u64,
    beta: u64,

    pub fn deinit(w: *Weights) void {
        w.memory.free();
    }
};

fn admit(c: *const config.Config) !void {
    const m = c.mtp orelse return error.NoDraftHead;
    if (c.text.hidden != hidden or c.text.vocab != vocab or m.layers != 3 or
        m.q_heads != 24 or m.kv_heads != 8 or m.head_dim != 128 or m.inter != inter or
        m.window != 1024 or m.block != block or m.rank != rank or m.rope_theta != 1e6 or m.taps_n != taps.len or
        !m.local[0] or m.local[1] or !m.local[2] or !std.mem.eql(u32, m.taps[0..taps.len], &taps))
        return error.UnsupportedDSparkConfig;
}

const Upload = struct {
    file: *const core.safetensors.File,
    memory: cuda.DeviceBuffer,
    offset: usize = 0,

    fn bytes(u: *Upload, sources: []const []const u8) !u64 {
        u.offset = std.mem.alignForward(usize, u.offset, 16);
        const ptr = u.memory.ptr + u.offset;
        for (sources) |source| {
            try u.memory.upload(u.offset, source);
            u.offset += source.len;
        }
        return ptr;
    }

    fn get(u: *Upload, name: []const u8) core.safetensors.Tensor {
        return u.file.get(name).?; // names.check validated all of these tensors and their shapes before upload
    }

    fn one(u: *Upload, name: []const u8) !u64 {
        return u.bytes(&.{u.get(name).bytes});
    }

    fn joined(u: *Upload, i: usize, comptime suffixes: []const []const u8) !u64 {
        var parts: [suffixes.len][]const u8 = undefined;
        inline for (suffixes, &parts) |suffix, *part| {
            var name: [128]u8 = undefined;
            part.* = u.get(try std.fmt.bufPrint(&name, "layers.{d}." ++ suffix ++ ".weight", .{i})).bytes;
        }
        return u.bytes(&parts);
    }
};

fn bf16(t: core.safetensors.Tensor) []const u16 {
    return std.mem.bytesAsSlice(u16, @as([]align(2) const u8, @alignCast(t.bytes)));
}

/// The four softmax rows use BF16-rounded coefficients, as in Mini's Metal drafter.
fn coefficients(t: core.safetensors.Tensor) [16]f32 {
    const bf = core.bf16_round;
    var a: [16]f32 = undefined;
    for (0..4) |i| {
        const row = a[i * 4 ..][0..4];
        for (bf16(t)[i * 4 ..][0..4], row) |x, *y| y.* = bf.value(x);
        const high = @reduce(.Max, @as(@Vector(4, f32), row.*));
        var sum: f32 = 0;
        for (row) |*x| {
            x.* = @exp(x.* - high);
            sum += x.*;
        }
        for (row) |*x| x.* = bf.value(bf.bits(x.* / sum));
    }
    return a;
}

pub fn checkpointBytes(gpa: std.mem.Allocator, io: std.Io, c: *const config.Config, dir: []const u8) !usize {
    try admit(c);
    var source = try openSource(gpa, io, dir);
    defer source.close(io);
    return arenaBytes(gpa, c, source.names);
}

fn openSource(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !core.safetensors.File {
    const file = if (std.mem.endsWith(u8, dir, ".safetensors")) dir else try std.fs.path.join(gpa, &.{ dir, "model-mtp.safetensors" });
    defer if (file.ptr != dir.ptr) gpa.free(file);
    return core.safetensors.File.open(gpa, io, file) catch |err| blk: {
        if (err != error.FileNotFound or file.ptr == dir.ptr) return err;
        const legacy = try std.fs.path.join(gpa, &.{ dir, "mtp.safetensors" });
        defer gpa.free(legacy);
        break :blk try core.safetensors.File.open(gpa, io, legacy);
    };
}

fn arenaBytes(gpa: std.mem.Allocator, c: *const config.Config, header: core.safetensors.Header) !usize {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const validation = try names.check(arena.allocator(), c, header, .mtp);
    if (!validation.clean()) {
        std.log.err("Mini DSpark: missing {any}, wrong shape/dtype {any}, extra {any}", .{
            validation.missing.items, validation.wrong.items, validation.unknown.items,
        });
        return error.Inkling2DraftTensors;
    }

    var stored: usize = 0;
    for (header.values()) |entry| stored += entry.end - entry.begin;
    return stored + validation.expected * 16 + prepared_head_bytes;
}

pub fn load(
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    c: *const config.Config,
    dir: []const u8,
    unembed: core.safetensors.Tensor,
) !Weights {
    try admit(c);
    var source = try openSource(gpa, io, dir);
    defer source.close(io);
    var memory = try cuda.DeviceBuffer.alloc(d, try arenaBytes(gpa, c, source.names));
    errdefer memory.free();
    var u: Upload = .{ .file = &source, .memory = memory };
    var layers: [3]Layer = undefined;
    for (&layers, 0..) |*l, i| l.* = .{
        .dflare = try u.joined(i, &.{"dflare_norm"}),
        .input_norm = try u.joined(i, &.{"input_layernorm"}),
        .post_norm = try u.joined(i, &.{"post_attention_layernorm"}),
        .q_norm = try u.joined(i, &.{"self_attn.q_norm"}),
        .k_norm = try u.joined(i, &.{"self_attn.k_norm"}),
        .ctx_norm = try u.joined(i, &.{"self_attn.k_norm_ctx"}),
        .qkv = try u.joined(i, &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" }),
        .ctx_kv = try u.joined(i, &.{ "self_attn.k_proj_ctx", "self_attn.v_proj_ctx" }),
        .out = try u.joined(i, &.{"self_attn.o_proj"}),
        .gate_up = try u.joined(i, &.{ "mlp.gate_proj", "mlp.up_proj" }),
        .down = try u.joined(i, &.{"mlp.down_proj"}),
    };

    const phi = try gpa.alloc(u16, vocab * rank);
    defer gpa.free(phi);
    const beta = try gpa.alloc(u16, vocab);
    defer gpa.free(beta);
    try fold.fold(gpa, bf16(unembed), bf16(u.get("markov_head.hidden_out.weight")), bf16(u.get("markov_head.hidden_out.bias")), phi, beta);
    const alpha = coefficients(u.get("fuse_weight"));
    return .{
        .memory = memory,
        .layers = layers,
        .alpha = try u.bytes(&.{std.mem.asBytes(&alpha)}),
        .mask = try u.one("mask_embed"),
        .norm = try u.one("norm.weight"),
        .markov_norm = try u.one("markov_head.dflare_norm.weight"),
        .markov_w1 = try u.one("markov_head.markov_w1.weight"),
        .rnn = try u.one("markov_head.rnn_proj.weight"),
        .rnn_bias = try u.one("markov_head.rnn_proj.bias"),
        .phi = try u.bytes(&.{std.mem.sliceAsBytes(phi)}),
        .beta = try u.bytes(&.{std.mem.sliceAsBytes(beta)}),
    };
}

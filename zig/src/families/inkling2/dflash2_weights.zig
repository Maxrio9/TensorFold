//! The public Inkling 2 Mini DFlash2 BF16 checkpoint and its exact device layout.
const std = @import("std");
const cuda = @import("cuda");
const tensors = @import("core").safetensors;
const config = @import("config.zig");
const dspark = @import("dspark_weights.zig");

pub const hidden = 2304;
pub const heads = 32;
pub const kv_heads = 8;
pub const dim = 128;
pub const q = heads * dim;
pub const kv = kv_heads * dim;
pub const inter = 6912;
pub const layers = 5;
pub const vocab = 201024;
pub const block = 8;
pub const window = 2048;
pub const groups = hidden / 16;
pub const top_k = 16;
pub const rank = 256;
pub const eps = 1e-6;
pub const theta = 10000.0;
pub const cap = 30.0;
pub const mask = 200056;

const Config = struct {
    architectures: [1][]const u8,
    model_type: []const u8,
    dtype: []const u8,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope_parameters: struct { rope_theta: f32, rope_type: []const u8 },
    vocab_size: usize,
    sliding_window: usize,
    layer_types: [layers][]const u8,
    attention_bias: bool,
    hidden_act: []const u8,
    is_causal: bool,
    dflash_config: struct {
        block_size: usize,
        conv_group_size: usize,
        conv_kernel_size: usize,
        final_logit_softcapping: f32,
        mask_token_id: usize,
        selector_rank: usize,
        selector_softcap_mode: []const u8,
        selector_top_k: usize,
        target_layer_ids: [dspark.taps.len]u32,
    },

    fn check(c: Config, target: *const config.Config) !void {
        const d = c.dflash_config;
        const eql = std.mem.eql;
        if (!target.exl3 or target.text.hidden != hidden or target.text.vocab != vocab or
            target.text.unpadded_vocab <= mask or !eql(u8, c.architectures[0], "DFlash2DraftModel") or
            !eql(u8, c.model_type, "qwen3") or !eql(u8, c.dtype, "bfloat16") or !eql(u8, c.hidden_act, "silu") or
            c.hidden_size != hidden or c.intermediate_size != inter or c.num_hidden_layers != layers or
            c.num_attention_heads != heads or c.num_key_value_heads != kv_heads or c.head_dim != dim or
            c.rms_norm_eps != eps or c.rope_parameters.rope_theta != theta or
            !eql(u8, c.rope_parameters.rope_type, "default") or c.vocab_size != vocab or c.sliding_window != window or
            c.attention_bias or c.is_causal or d.block_size != block or d.conv_group_size != 16 or
            d.conv_kernel_size != 2 or d.final_logit_softcapping != cap or d.mask_token_id != mask or
            d.selector_rank != rank or d.selector_top_k != top_k or !eql(u8, d.selector_softcap_mode, "after_correction") or
            !eql(u32, &d.target_layer_ids, &dspark.taps)) return error.UnsupportedMiniDFlash2Config;
        for (c.layer_types) |layer| {
            if (!eql(u8, layer, "sliding_attention")) return error.UnsupportedMiniDFlash2Config;
        }
    }
};

pub fn isDFlash2(a: std.mem.Allocator, io: std.Io, dir: []const u8, target: *const config.Config) !bool {
    if (std.mem.endsWith(u8, dir, ".safetensors")) return false;
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    defer a.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    defer a.free(bytes);
    const root = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer root.deinit();
    if (root.value != .object or !root.value.object.contains("dflash_config")) return false;
    const cfg = std.json.parseFromValue(Config, a, root.value, .{ .ignore_unknown_fields = true }) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return error.UnsupportedMiniDFlash2Config,
    };
    defer cfg.deinit();
    try cfg.value.check(target);
    return true;
}

pub const Conv = struct { base: u64, projection: u64 };

pub const Layer = struct {
    qkv: u64,
    out: u64,
    gate_up: u64,
    down: u64,
    input_norm: u64,
    post_norm: u64,
    q_norm: u64,
    k_norm: u64,
    attention_conv: Conv,
    mlp_conv: Conv,
};

pub const Weights = struct {
    memory: cuda.DeviceBuffer,
    layer: [layers]Layer,
    fuse: u64,
    fuse_norm: u64,
    norm: u64,
    project: u64,
    predecessor: u64,
    successor: u64,

    pub fn deinit(w: *Weights) void {
        w.memory.free();
    }
};

const Reader = struct {
    file: *const tensors.File,
    memory: ?cuda.DeviceBuffer,
    used: usize = 0,
    count: usize = 0,

    fn put(r: *Reader, name: []const u8, shape: []const usize) !u64 {
        const t = r.file.get(name) orelse return bad(name);
        if (t.dtype != .bf16 or !std.mem.eql(usize, t.shape[0..t.rank], shape)) return bad(name);
        const start = r.used;
        r.used += t.bytes.len;
        r.count += 1;
        if (r.memory) |mem| {
            try mem.upload(start, t.bytes);
            return mem.ptr + start;
        }
        return start;
    }

    fn bad(name: []const u8) error{MiniDFlash2Tensor} {
        std.log.err("Mini DFlash2: missing tensor or incompatible dtype/shape for {s}", .{name});
        return error.MiniDFlash2Tensor;
    }

    fn at(r: *Reader, layer: usize, comptime suffix: []const u8, shape: []const usize) !u64 {
        var name: [96]u8 = undefined;
        return r.put(try std.fmt.bufPrint(&name, "layers.{d}." ++ suffix, .{layer}), shape);
    }

    fn conv(r: *Reader, layer: usize, comptime prefix: []const u8) !Conv {
        return .{
            .base = try r.at(layer, prefix ++ ".base_kernel", &.{ 2, 2, hidden }),
            .projection = try r.at(layer, prefix ++ ".kernel_projection.weight", &.{ 4 * groups, hidden }),
        };
    }

    fn read(r: *Reader) !Weights {
        var ls: [layers]Layer = undefined;
        for (&ls, 0..) |*l, i| {
            const qkv = try r.at(i, "self_attn.q_proj.weight", &.{ q, hidden });
            _ = try r.at(i, "self_attn.k_proj.weight", &.{ kv, hidden });
            _ = try r.at(i, "self_attn.v_proj.weight", &.{ kv, hidden });
            const gate_up = try r.at(i, "mlp.gate_proj.weight", &.{ inter, hidden });
            _ = try r.at(i, "mlp.up_proj.weight", &.{ inter, hidden });
            l.* = .{
                .qkv = qkv,
                .out = try r.at(i, "self_attn.o_proj.weight", &.{ hidden, q }),
                .gate_up = gate_up,
                .down = try r.at(i, "mlp.down_proj.weight", &.{ hidden, inter }),
                .input_norm = try r.at(i, "input_layernorm.weight", &.{hidden}),
                .post_norm = try r.at(i, "post_attention_layernorm.weight", &.{hidden}),
                .q_norm = try r.at(i, "self_attn.q_norm.weight", &.{dim}),
                .k_norm = try r.at(i, "self_attn.k_norm.weight", &.{dim}),
                .attention_conv = try r.conv(i, "attention_conv"),
                .mlp_conv = try r.conv(i, "mlp_conv"),
            };
        }
        const out: Weights = .{
            .memory = r.memory orelse undefined,
            .layer = ls,
            .fuse = try r.put("fc.weight", &.{ hidden, dspark.taps.len * hidden }),
            .fuse_norm = try r.put("hidden_norm.weight", &.{hidden}),
            .norm = try r.put("norm.weight", &.{hidden}),
            .project = try r.put("candidate_selector.hidden_projection.weight", &.{ rank, hidden }),
            .predecessor = try r.put("candidate_selector.predecessor_codebook", &.{ vocab, rank }),
            .successor = try r.put("candidate_selector.successor_codebook", &.{ vocab, rank }),
        };
        if (r.file.names.count() != r.count) return error.MiniDFlash2Tensor;
        return out;
    }
};

fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8) !tensors.File {
    const path = try std.fs.path.join(a, &.{ dir, "model.safetensors" });
    defer a.free(path);
    return tensors.File.open(a, io, path);
}

fn size(file: *const tensors.File) !usize {
    var r: Reader = .{ .file = file, .memory = null };
    _ = try r.read();
    return r.used;
}

pub fn checkpointBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8) !usize {
    var file = try open(a, io, dir);
    defer file.close(io);
    return size(&file);
}

pub fn load(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Weights {
    var file = try open(a, io, dir);
    defer file.close(io);
    var mem = try cuda.DeviceBuffer.alloc(d, try size(&file));
    errdefer mem.free();
    var r: Reader = .{ .file = &file, .memory = mem };
    return r.read();
}

test "explicit draft directory releases its config on fallback and refusal" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const target: config.Config = undefined;
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{}" });
    try std.testing.expect(!try isDFlash2(a, io, dir, &target));
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"dflash_config\":{}}" });
    try std.testing.expectError(error.UnsupportedMiniDFlash2Config, isDFlash2(a, io, dir, &target));
}

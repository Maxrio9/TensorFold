//! Mini DFlash2 checkpoints in upstream BF16 or TensorFold's packed FP8G64 layout.
const std = @import("std");
const cuda = @import("cuda");
const tensors = @import("core").safetensors;
const config = @import("config.zig");
const dspark = @import("dspark_weights.zig");
const fp8 = @import("draft_fp8.zig");

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
            target.text.unpadded_vocab <= mask or target.text.unpadded_vocab > vocab or !eql(u8, c.architectures[0], "DFlash2DraftModel") or
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

pub const Conv = struct { base: u64, projection: fp8.Matrix };

pub const Layer = struct {
    qkv: fp8.Matrix,
    out: fp8.Matrix,
    gate_up: fp8.Matrix,
    down: fp8.Matrix,
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
    fuse: fp8.Matrix,
    fuse_norm: u64,
    norm: u64,
    project: fp8.Matrix,
    predecessor: fp8.Table,
    successor: fp8.Table,
    base: ?cuda.qmmf.Weight,
    fp8_layers: bool,

    pub fn deinit(w: *Weights) void {
        w.memory.free();
    }
};

const Reader = struct {
    file: *const tensors.File,
    arena: fp8.Arena = .{},
    options: fp8.Options,
    packed_layers: bool,
    selector_rows: usize,
    count: usize = 0,

    fn tensor(r: *Reader, name: []const u8, shape: []const usize) ![]const u8 {
        return r.typedTensor(name, .bf16, shape);
    }

    fn typedTensor(r: *Reader, name: []const u8, dtype: tensors.DType, shape: []const usize) ![]const u8 {
        const t = r.file.get(name) orelse return bad(name);
        if (t.dtype != dtype or !std.mem.eql(usize, t.shape[0..t.rank], shape)) return bad(name);
        r.count += 1;
        return t.bytes;
    }

    fn put(r: *Reader, name: []const u8, shape: []const usize) !u64 {
        return r.arena.put(&.{try r.tensor(name, shape)});
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
            .projection = try r.matrix(layer, &.{prefix ++ ".kernel_projection.weight"}, .{4 * groups}, hidden),
        };
    }

    fn layerMatrix(r: *Reader, comptime count: usize, names: [count][]const u8, ns: [count]usize, k: usize) !fp8.Matrix {
        var parts: [count][]const u8 = undefined;
        if (r.packed_layers) {
            var scales: [count][]const u8 = undefined;
            for (names, ns, &parts, &scales) |name, n, *data, *scale| {
                const padded = cuda.modelopt.padded(n);
                if (count > 1 and padded != n) return bad(name);
                const column_groups = k / cuda.qmmf.group;
                const tiles = padded / 64;
                data.* = try r.typedTensor(name, .f8_e4m3, &.{ tiles, column_groups, 8, 32, 16 });
                var key: [128]u8 = undefined;
                const scale_name = try std.fmt.bufPrint(&key, "{s}.tf_fp8g64_scale", .{name});
                scale.* = try r.typedTensor(scale_name, .f32, &.{ tiles, column_groups, 64 });
            }
            return r.arena.packedMatrix(&parts, &scales);
        }
        for (names, ns, &parts) |name, n, *part| part.* = try r.tensor(name, &.{ n, k });
        return r.arena.matrix(&parts, @reduce(.Add, @as(@Vector(count, usize), ns)), k, r.options.layers);
    }

    fn matrix(r: *Reader, layer: usize, comptime suffixes: []const []const u8, ns: [suffixes.len]usize, k: usize) !fp8.Matrix {
        var storage: [suffixes.len][96]u8 = undefined;
        var names: [suffixes.len][]const u8 = undefined;
        inline for (suffixes, &storage, &names) |suffix, *name, *part| {
            part.* = try std.fmt.bufPrint(name, "layers.{d}." ++ suffix, .{layer});
        }
        return r.layerMatrix(suffixes.len, names, ns, k);
    }

    fn codebook(r: *Reader, name: []const u8) !fp8.Table {
        const bytes = try r.tensor(name, &.{ vocab, rank });
        // The selector already bounds both the predecessor and candidate IDs by the real vocabulary.
        return r.arena.table(bytes[0 .. r.selector_rows * rank * @sizeOf(u16)], r.selector_rows, rank, r.options.head);
    }

    fn read(r: *Reader, n_vocab: usize, base: ?fp8.BaseHead) !Weights {
        var ls: [layers]Layer = undefined;
        for (&ls, 0..) |*l, i| {
            const qkv = try r.matrix(i, &.{ "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight" }, .{ q, kv, kv }, hidden);
            const gate_up = try r.matrix(i, &.{ "mlp.gate_proj.weight", "mlp.up_proj.weight" }, .{ inter, inter }, hidden);
            l.* = .{
                .qkv = qkv,
                .out = try r.matrix(i, &.{"self_attn.o_proj.weight"}, .{hidden}, q),
                .gate_up = gate_up,
                .down = try r.matrix(i, &.{"mlp.down_proj.weight"}, .{hidden}, inter),
                .input_norm = try r.at(i, "input_layernorm.weight", &.{hidden}),
                .post_norm = try r.at(i, "post_attention_layernorm.weight", &.{hidden}),
                .q_norm = try r.at(i, "self_attn.q_norm.weight", &.{dim}),
                .k_norm = try r.at(i, "self_attn.k_norm.weight", &.{dim}),
                .attention_conv = try r.conv(i, "attention_conv"),
                .mlp_conv = try r.conv(i, "mlp_conv"),
            };
        }
        const out: Weights = .{
            .memory = r.arena.memory orelse undefined,
            .layer = ls,
            .fuse = try r.layerMatrix(1, .{"fc.weight"}, .{hidden}, dspark.taps.len * hidden),
            .fuse_norm = try r.put("hidden_norm.weight", &.{hidden}),
            .norm = try r.put("norm.weight", &.{hidden}),
            .project = try r.arena.matrix(&.{try r.tensor("candidate_selector.hidden_projection.weight", &.{ rank, hidden })}, rank, hidden, r.options.head),
            .predecessor = try r.codebook("candidate_selector.predecessor_codebook"),
            .successor = try r.codebook("candidate_selector.successor_codebook"),
            .base = try r.arena.base(r.options, base, n_vocab, hidden),
            .fp8_layers = r.packed_layers or r.options.layers,
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

fn selectorRows(n_vocab: usize) usize {
    return if (@import("sys").environment.read(bool, "TF_INKLING2_DFLASH2_TRIM_SELECTOR", true) catch true) n_vocab else vocab;
}

fn packedLayers(file: *const tensors.File) bool {
    const t = file.get("fc.weight") orelse return false;
    return t.dtype == .f8_e4m3;
}

fn size(file: *const tensors.File, n_vocab: usize, options: fp8.Options, selector_rows: usize) !usize {
    var r: Reader = .{ .file = file, .options = options, .packed_layers = packedLayers(file), .selector_rows = selector_rows };
    _ = try r.read(n_vocab, null);
    return r.arena.used;
}

pub fn checkpointBytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, n_vocab: usize, options: fp8.Options) !usize {
    var file = try open(a, io, dir);
    defer file.close(io);
    return size(&file, n_vocab, options, selectorRows(n_vocab));
}

pub fn load(a: std.mem.Allocator, io: std.Io, dir: []const u8, n_vocab: usize, options: fp8.Options, base: fp8.BaseHead) !Weights {
    var file = try open(a, io, dir);
    defer file.close(io);
    const d = base.stream.d;
    const selector_rows = selectorRows(n_vocab);
    const packed_layers = packedLayers(&file);
    var mem = try cuda.DeviceBuffer.alloc(d, try size(&file, n_vocab, options, selector_rows));
    errdefer mem.free();
    const stage = @max(if (options.layers and !packed_layers) @as(usize, 2 * 2 * inter * hidden) else 0, if (options.head) selector_rows * rank * 2 else 0);
    var p = if (stage != 0) try fp8.Packer.init(d, base.stream, stage) else null;
    defer if (p) |*packer| packer.deinit();
    var r: Reader = .{
        .file = &file,
        .arena = .{ .memory = mem, .packer = if (p) |*packer| packer else null },
        .options = options,
        .packed_layers = packed_layers,
        .selector_rows = selector_rows,
    };
    return r.read(n_vocab, base);
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

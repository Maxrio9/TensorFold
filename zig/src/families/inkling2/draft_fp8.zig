//! Optional E4M3 storage for Mini draft projections and selector rows.

// Copyright 2026 MiaAI-Lab, draft-only FP8 head design adapted from patch 0006.
const std = @import("std");
const cuda = @import("cuda");
const lazy_head = @import("loop_lazy_head.zig");

pub const Options = struct {
    layers: bool = false,
    head: bool = false,
    borrow_head: bool = false,

    pub fn fromEnv(borrow_head: bool, default_fp8: bool) Options {
        return .{
            .layers = @import("sys").environment.read(bool, "TF_INKLING2_DRAFT_FP8_LAYERS", default_fp8) catch false,
            .head = @import("sys").environment.read(bool, "TF_INKLING2_DRAFT_FP8_HEAD", default_fp8) catch false,
            .borrow_head = borrow_head,
        };
    }

    pub fn active(o: Options) bool {
        return o.layers or o.head;
    }

    pub fn baseBytes(o: Options, n: usize, k: usize) usize {
        return if (o.head and !o.borrow_head) lazy_head.codeBytes(n, k) else 0;
    }
};

/// A BF16 row-major weight, or a CUDA lane with one f32 scale per group of 64 inputs.
pub const Matrix = struct {
    data: u64,
    scales: u64 = 0,

    pub fn afterRows(m: Matrix, rows: usize, k: usize) Matrix {
        if (m.scales == 0) return .{ .data = m.data + rows * k * 2 };
        std.debug.assert(rows % 128 == 0);
        return .{ .data = m.data + rows * k, .scales = m.scales + rows * (k / cuda.qmmf.group) * 4 };
    }

    pub fn lane(m: Matrix, n: usize, k: usize) cuda.qmmf.Weight {
        std.debug.assert(m.scales != 0);
        return cuda.fp8.weight(m.data, m.scales, n, k);
    }
};

/// A BF16 or row-scaled FP8 rank-256 lookup table.
pub const Table = struct { data: u64, scales: u64 = 0 };

pub const BaseHead = struct {
    stream: cuda.Stream,
    source: u64,
    borrowed: ?cuda.qmmf.Weight,
};

pub const Packer = struct {
    stream: cuda.Stream,
    mod: cuda.Module,
    staging: cuda.DeviceBuffer,
    matrix: cuda.Function,
    rows: cuda.Function,

    pub fn init(d: *const cuda.Driver, s: cuda.Stream, staging_bytes: usize) !Packer {
        var mod = try cuda.Module.load(d, cuda.kernels.inkling2_dspark);
        errdefer mod.unload();
        const matrix = try mod.function("tf_ink2_draft_pack_layers");
        const rows = try mod.function("tf_ink2_draft_pack_rows");
        return .{ .stream = s, .mod = mod, .matrix = matrix, .rows = rows, .staging = try cuda.DeviceBuffer.alloc(d, staging_bytes) };
    }

    pub fn deinit(p: *Packer) void {
        p.staging.free();
        p.mod.unload();
    }

    fn apply(p: *const Packer, sources: []const []const u8, f: cuda.Function, data: u64, scales: u64, n: usize, k: usize, work: usize) !void {
        var offset: usize = 0;
        for (sources) |source| {
            try p.staging.upload(offset, source);
            offset += source.len;
        }
        std.debug.assert(offset == n * k * 2);
        var a: cuda.Args = .{};
        a.add(p.staging.ptr);
        a.add(data);
        a.add(scales);
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, @intCast(k)));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((work + 7) / 8) }, .block = .{ .x = 256 } }, p.stream, &a);
        try p.stream.synchronize();
    }
};

pub fn loadLane(d: *const cuda.Driver, wanted: bool) !?cuda.qmmf.Lane {
    if (!wanted) return null;
    const sm = try d.currentCapability();
    if (sm < 120) return error.MiniDraftFp8RequiresBlackwell;
    return try cuda.qmmf.Lane.load(d, sm / 10);
}

/// Host-only accounting and real device upload walk the same tensor order.
pub const Arena = struct {
    memory: ?cuda.DeviceBuffer = null,
    packer: ?*const Packer = null,
    used: usize = 0,

    fn alloc(a: *Arena, bytes: usize) u64 {
        a.used = std.mem.alignForward(usize, a.used, 16);
        const at = a.used;
        a.used += bytes;
        return (if (a.memory) |mem| mem.ptr else 0) + at;
    }

    pub fn put(a: *Arena, sources: []const []const u8) !u64 {
        var bytes: usize = 0;
        for (sources) |source| bytes += source.len;
        const ptr = a.alloc(bytes);
        if (a.memory) |mem| {
            var offset = ptr - mem.ptr;
            for (sources) |source| {
                try mem.upload(offset, source);
                offset += source.len;
            }
        }
        return ptr;
    }

    pub fn matrix(a: *Arena, sources: []const []const u8, n: usize, k: usize, fp8: bool) !Matrix {
        if (!fp8) return .{ .data = try a.put(sources) };
        if (k % cuda.qmmf.group != 0) return error.Inkling2DraftShape;
        const padded = cuda.modelopt.padded(n);
        const groups = k / cuda.qmmf.group;
        const data = a.alloc(padded * k);
        const scales = a.alloc(padded * groups * 4);
        if (a.packer) |p| try p.apply(sources, p.matrix, data, scales, n, k, padded * groups);
        return .{ .data = data, .scales = scales };
    }

    pub fn packedMatrix(a: *Arena, data: []const []const u8, scales: []const []const u8) !Matrix {
        return .{ .data = try a.put(data), .scales = try a.put(scales) };
    }

    pub fn table(a: *Arena, source: ?[]const u8, n: usize, k: usize, fp8: bool) !Table {
        if (!fp8) return if (source) |bytes| .{ .data = try a.put(&.{bytes}) } else .{ .data = a.alloc(n * k * 2) };
        const data = a.alloc(n * k);
        const scales = a.alloc(n * 4);
        if (a.packer) |p| try p.apply(&.{source.?}, p.rows, data, scales, n, k, n);
        return .{ .data = data, .scales = scales };
    }

    pub fn base(a: *Arena, o: Options, head: ?BaseHead, n: usize, k: usize) !?cuda.qmmf.Weight {
        if (!o.head) return null;
        if (o.borrow_head) return if (head) |h| h.borrowed else null;
        const dest = a.alloc(o.baseBytes(n, k));
        if (head) |h| return try lazy_head.pack(h.stream.d, h.stream, h.source, dest, n, k);
        return null;
    }
};

test "Mini FP8 matrices keep the K/V suffix at its own packed tile and scale" {
    var a: Arena = .{ .used = 256 };
    const projection = try a.matrix(&.{}, 6144, 2304, true);
    const kv = projection.afterRows(4096, 2304);
    try std.testing.expectEqual(projection.data + 4096 * 2304, kv.data);
    try std.testing.expectEqual(projection.scales + 4096 * (2304 / 64) * 4, kv.scales);
    try std.testing.expectEqual(256 + 6144 * (2304 + (2304 / 64) * 4), a.used);
    try std.testing.expectEqual(@as(u32, 2048), kv.lane(2048, 2304).npad);
}

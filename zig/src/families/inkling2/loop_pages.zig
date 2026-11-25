//! Inkling 2 Mini's global KV arena and graph-stable page tables.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const loop = @import("loop.zig");
const engine = @import("loop_engine.zig");

pub const page_tokens: u32 = 256; // cache_rows.cuh; a whole number of 64-key attention tiles

pub fn need(prompt: usize, reply: usize) usize {
    const geometry: kv.Layout = .{ .page_tokens = page_tokens, .fixed_bytes = 0, .token_bytes = 1, .snap_bytes = 0 };
    return geometry.need(prompt, reply, engine.window_rows);
}

pub fn tableBytes(context: usize) usize {
    return need(context, 0) * @sizeOf(u32);
}

pub const Pages = struct {
    pool: kv.Pool,
    memory: cuda.DeviceBuffer,
    tokens: usize,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, count: usize, row_bytes: u64) !Pages {
        const tokens = count * page_tokens;
        var pool = try kv.Pool.init(gpa, count, row_bytes * page_tokens);
        errdefer pool.deinit();
        const memory = try cuda.DeviceBuffer.alloc(d, tokens * row_bytes);
        return .{ .pool = pool, .memory = memory, .tokens = tokens };
    }

    pub fn deinit(p: *Pages) void {
        p.pool.deinit();
        p.memory.free();
    }

    pub fn cache(p: *const Pages, visit: loop.SnapshotSlot, format: loop.CacheFormat) @import("kernels.zig").Cache {
        const k = p.memory.ptr + visit.global_offset * p.tokens;
        return .{ .k = k, .v = k + visit.global_row * p.tokens / 2, .format = format };
    }
};

/// A global K/V plane has a region per head's codes and, in FP8, per head's scales.
/// A page's storage spans every plane. Pack state across its regions in physical order.
const StateGeometry = struct {
    heads: usize,
    codes: usize,
    scales: usize,
    pages: usize,
    page_bytes: usize,

    fn piece(g: StateGeometry, ids: []const u32, offset: usize) struct { offset: usize, len: usize } {
        const physical = ids[offset / g.page_bytes];
        const plane = g.heads * (g.codes + g.scales);
        const row_offset = offset % g.page_bytes;
        const plane_base = row_offset / plane * plane;
        const local = row_offset % plane;
        const scale_base = g.heads * g.codes;
        const scale = local >= scale_base;
        const region = if (scale) g.scales else g.codes;
        const start = if (scale) scale_base else 0;
        const within = (local - start) % region;
        const base = plane_base + local - within;
        return .{ .offset = base * g.pages + physical * region + within, .len = region - within };
    }
};

pub const StateCopy = struct {
    m: *const engine.Model,
    s: cuda.Stream,
    ids: []const u32,
    geometry: StateGeometry,
    out: bool,
    offset: usize = 0,

    pub fn init(m: *const engine.Model, s: cuda.Stream, ids: []const u32, out: bool) StateCopy {
        const p = &m.pages.?;
        const heads = m.w.c.text.global;
        const fp8 = m.layout.format == .fp8;
        return .{
            .m = m,
            .s = s,
            .ids = ids,
            .out = out,
            .geometry = .{
                .heads = heads.kv,
                .codes = @as(usize, heads.dim) * page_tokens * (if (fp8) @as(usize, 1) else 2),
                .scales = if (fp8) page_tokens * @sizeOf(f32) else 0,
                .pages = p.pool.total(),
                .page_bytes = @intCast(p.pool.page_bytes),
            },
        };
    }

    pub fn span(w: *StateCopy, ptr: u64, size: usize) !void {
        const arena = w.m.pages.?.memory.ptr;
        var copied: usize = 0;
        while (copied < size) {
            const piece = w.geometry.piece(w.ids, w.offset);
            const n = @min(size - copied, piece.len);
            const dev = ptr + copied;
            const blob = arena + piece.offset;
            try w.m.copy(w.s, if (w.out) blob else dev, if (w.out) dev else blob, n);
            w.offset += n;
            copied += n;
        }
    }
};

test "Mini snapshot bytes span scattered head planes and respect physical page IDs" {
    const g: StateGeometry = .{ .heads = 2, .codes = 32, .scales = 4, .pages = 4, .page_bytes = 144 };
    const ids = &[_]u32{ 3, 1 };
    try std.testing.expectEqual(@as(usize, 96), g.piece(ids, 0).offset);
    try std.testing.expectEqual(@as(usize, 1), g.piece(ids, 31).len);
    try std.testing.expectEqual(@as(usize, 268), g.piece(ids, 64).offset);
    try std.testing.expectEqual(@as(usize, 284), g.piece(ids, 68).offset);
    try std.testing.expectEqual(@as(usize, 384), g.piece(ids, 72).offset);
    try std.testing.expectEqual(@as(usize, 32), g.piece(ids, 144).offset);
}

pub const Map = struct {
    m: *engine.Model,
    s: cuda.Stream,
    slot: usize,

    pub fn table(m: *Map) kv.Table {
        return .{ .ctx = m, .write = write };
    }

    fn write(ctx: *anyopaque, ids: []const u32, from: usize) anyerror!void {
        const m: *Map = @ptrCast(@alignCast(ctx));
        const ptr = m.m.pageTable(m.slot);
        try m.m.upload(m.s, ptr + from * @sizeOf(u32), std.mem.sliceAsBytes(ids[from..]));
    }
};

pub fn reserve(m: *engine.Model, s: cuda.Stream, slot: usize, lease: *kv.Lease, n: usize) !void {
    const p = if (m.pages) |*p| &p.pool else return;
    try p.promise(lease, n - lease.ids.items.len);
    var map: Map = .{ .m = m, .s = s, .slot = slot };
    try p.cover(lease, n, map.table());
}

pub fn finish(m: *engine.Model, lease: *kv.Lease) void {
    if (m.pages) |*p| p.pool.finish(lease);
}

test "Mini reserves whole pages including the last verify window" {
    try std.testing.expectEqual(@as(usize, 1), need(230, 10));
    try std.testing.expectEqual(@as(usize, 2), need(231, 10));
    try std.testing.expectEqual(@as(usize, 33 * 4), tableBytes(8192));
}

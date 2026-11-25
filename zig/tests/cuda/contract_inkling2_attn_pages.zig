//! Keep the reference in logical order while the CUDA fixture interleaves reversed pages from each lane.
const std = @import("std");
const ink = @import("inkling2");
const Rig = @import("contract_rig.zig").Rig;

pub const page_tokens = ink.native.page_tokens;

pub const Map = struct {
    logical_ring: usize,
    streams: usize,

    pub fn ring(map: Map) usize {
        return map.logical_ring * map.streams;
    }

    fn pageId(map: Map, stream: usize, page: usize) usize {
        const pages = map.logical_ring / page_tokens;
        return (pages - 1 - page) * map.streams + stream;
    }

    pub fn slot(map: Map, stream: usize, position: usize) usize {
        return map.pageId(stream, position / page_tokens) * page_tokens + position % page_tokens;
    }

    pub fn bind(map: Map, rig: *Rig, segs: []ink.kernels.Seg) !void {
        const a = rig.gpu.gpa;
        const ids = try a.alloc(u32, map.logical_ring / page_tokens);
        defer a.free(ids);
        for (segs, 0..) |*seg, stream| {
            for (ids, 0..) |*id, page| id.* = @intCast(map.pageId(stream, page));
            seg.paged = 1;
            seg.off = @bitCast(try rig.input(std.mem.sliceAsBytes(ids)));
        }
    }

    /// Copy from each lane's contiguous planes into the physical, head-major K or V plane.
    pub fn pack(map: Map, dst: []u8, src: []const u8, heads: usize, code: usize, scale: usize) void {
        const input_rows = heads * map.logical_ring;
        const output_rows = heads * map.ring();
        const lane_bytes = input_rows * (code + scale);
        for (0..map.streams) |stream| {
            const lane = src[stream * lane_bytes ..][0..lane_bytes];
            for (0..heads) |head| {
                var position: usize = 0;
                while (position < map.logical_ring) : (position += page_tokens) {
                    const from = head * map.logical_ring + position;
                    const to = head * map.ring() + map.slot(stream, position);
                    copy(dst, lane, to, from, code);
                    if (scale > 0) copy(dst[output_rows * code ..], lane[input_rows * code ..], to, from, scale);
                }
            }
        }
    }

    fn copy(dst: []u8, src: []const u8, to: usize, from: usize, stride: usize) void {
        const len = page_tokens * stride;
        @memcpy(dst[to * stride ..][0..len], src[from * stride ..][0..len]);
    }
};

test "cache fixture interleaves physical pages and packs scales after all codes" {
    const map: Map = .{ .logical_ring = 2 * page_tokens, .streams = 2 };
    const pages = 4;
    const page = page_tokens;
    var src: [pages * page * 2]u8 = undefined;
    var dst: [pages * page * 2]u8 = undefined;
    for (0..pages * 2) |i| @memset(src[i * page ..][0..page], @intCast(i));
    map.pack(&dst, &src, 1, 1, 1);
    for ([_]u8{ 1, 5, 0, 4, 3, 7, 2, 6 }, 0..) |val, i| {
        try std.testing.expect(std.mem.allEqual(u8, dst[i * page ..][0..page], val));
    }
}

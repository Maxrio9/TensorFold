//! Host access to the retained Qwen CUDA feature ring.
const lanes = @import("lanes");
const E = @import("cuda_engine.zig");
const be = lanes.backend;

pub fn featuresFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, start: u64, count: u32) !be.Features {
    const self: *@import("cuda_native.zig").Lanes = @ptrCast(@alignCast(ptr));
    return features(self, stream, taps, start, count);
}

pub fn featuresHostFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, start: u64, count: u32) !be.Features {
    const self: *@import("cuda_native.zig").Lanes = @ptrCast(@alignCast(ptr));
    return featuresHost(self, stream, taps, start, count);
}

pub fn features(self: anytype, stream: *lanes.Stream, taps: []const u32, start: u64, count: u32) !be.Features {
    if (taps.len != self.e.n_taps) return error.TapsMismatch;
    const cache = try self.cacheOf(stream);
    const ring = cache.tap orelse return error.NoTaps;
    const held_from = cache.written -| E.tap_rows;
    if (start + count > cache.written or start + count <= held_from) return error.RowsNotHeld;
    return .{ .buffer = ring.ptr, .offset = start, .rows = count, .row_bytes = @intCast(self.e.n_taps * self.e.g.hidden * 2), .space = .device, .dtype = .bf16 };
}

pub fn featuresHost(self: anytype, stream: *lanes.Stream, taps: []const u32, start: u64, count: u32) !be.Features {
    if (count > E.tap_rows) return error.QwenTapChunkTooLong;
    const f = try features(self, stream, taps, start, count);
    const cache = try self.cacheOf(stream);
    if (start < cache.written -| E.tap_rows) return error.RowsNotHeld;
    const bytes = @as(usize, count) * f.row_bytes;
    if (cache.tap_host.len < bytes) {
        self.gpa.free(cache.tap_host);
        cache.tap_host = &.{};
        cache.tap_host = try self.gpa.alloc(u8, bytes);
    }
    const slot: usize = @intCast(start % E.tap_rows);
    const first = @min(count, E.tap_rows - slot);
    try self.e.stream.synchronize();
    const ring = cache.tap.?;
    const first_bytes = first * f.row_bytes;
    try ring.download(slot * f.row_bytes, cache.tap_host[0..first_bytes]);
    if (first < count) try ring.download(0, cache.tap_host[first_bytes..bytes]);
    return .{ .buffer = @intFromPtr(cache.tap_host.ptr), .offset = 0, .rows = count, .row_bytes = f.row_bytes, .space = .host, .ready = .none, .dtype = f.dtype };
}

//! Host access to the retained Qwen CUDA feature ring.
const std = @import("std");
const lanes = @import("lanes");
const E = @import("cuda_engine.zig");
const be = lanes.backend;

pub fn prefillStepFn(ptr: *anyopaque, stream: *lanes.Stream, max_rows: u32) !bool {
    const self: *@import("cuda_native.zig").Lanes = @ptrCast(@alignCast(ptr));
    if (stream.readout != null) return @import("cuda_readout.zig").step(self, stream, max_rows);
    const before = @max(stream.fill_at.load(.acquire), stream.cached);
    const done = try @import("cuda_prefill.zig").step(self, stream, max_rows);
    if (self.e.n_taps > 0 and stream.fill_at.load(.acquire) - before > E.tap_rows) return error.QwenTapChunkTooLong;
    return done;
}

pub fn prepareFeaturesFn(ptr: *anyopaque, taps: []const u32) !void {
    const self: *@import("cuda_native.zig").Lanes = @ptrCast(@alignCast(ptr));
    if (std.mem.eql(u32, self.e.taps[0..self.e.n_taps], taps)) return;
    if (self.caches.count() != 0 or self.pool.items.len != 0) return error.TapsAfterStreams;
    try self.e.setTaps(taps);
}

pub fn featuresHostFn(ptr: *anyopaque, stream: *lanes.Stream, taps: []const u32, start: u64, count: u32) !be.Features {
    const self: *@import("cuda_native.zig").Lanes = @ptrCast(@alignCast(ptr));
    const f = try @import("cuda_native.zig").Lanes.recordFeatures(ptr, stream, taps, start, count);
    return featuresHost(self, stream, f, start, count);
}

pub fn featuresHost(self: anytype, stream: *lanes.Stream, f: be.Features, start: u64, count: u32) !be.Features {
    if (count > E.tap_rows) return error.QwenTapChunkTooLong;
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

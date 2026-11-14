//! Keep selected verification rows in their committed positions in the tap ring.
const std = @import("std");
pub fn apply(tap: anytype, stream: anytype, start: usize, path: []const u32, capacity: usize, row_bytes: usize) !void {
    if (capacity == 0 or path.len == 0 or path.len > capacity or path[0] != 0 or row_bytes == 0) return error.InvalidTapPath;
    for (path, 0..) |source_row, destination_row| {
        if (source_row >= capacity or (destination_row > 0 and source_row <= path[destination_row - 1])) return error.InvalidTapPath;
    }
    for (path, 0..) |source_row, destination_row| {
        if (source_row == destination_row) continue;
        const src = (start + source_row) % capacity;
        const dst = (start + destination_row) % capacity;
        try tap.copyFrom(dst * row_bytes, tap.ptr + src * row_bytes, row_bytes, stream);
    }
}
const Fake = struct {
    ptr: usize,
    bytes: []u8,
    fail_on: ?usize = null,
    calls: *usize,
    pub fn copyFrom(self: Fake, offset: usize, src: usize, n: usize, _: void) !void {
        if (self.fail_on != null and self.calls.* == self.fail_on.?) return error.InjectedCopyFailure;
        self.calls.* += 1;
        const source = src - self.ptr;
        if (source + n > self.bytes.len or offset + n > self.bytes.len) return error.OutOfBounds;
        @memcpy(self.bytes[offset..][0..n], self.bytes[source..][0..n]);
    }
};
test "selected tree path copies actual rows down in order" {
    var data = [_]u8{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var calls: usize = 0;
    try apply(Fake{ .ptr = 1000, .bytes = &data, .calls = &calls }, {}, 0, &.{ 0, 2, 4 }, 8, 1);
    try std.testing.expectEqualSlices(u8, &.{ 10, 30, 50 }, data[0..3]);
    try std.testing.expectEqual(@as(usize, 2), calls);
}
test "wrapped paths do not overwrite later source rows" {
    var data = [_]u8{ 30, 40, 50, 60, 70, 80, 10, 20 };
    var calls: usize = 0;
    try apply(Fake{ .ptr = 1000, .bytes = &data, .calls = &calls }, {}, 6, &.{ 0, 2, 4 }, 8, 1);
    try std.testing.expectEqualSlices(u8, &.{ 10, 30 }, data[6..8]);
    try std.testing.expectEqual(@as(u8, 50), data[0]);
}
test "invalid paths validate before any copies and failures propagate" {
    var data = [_]u8{ 1, 2, 3, 4 };
    var calls: usize = 0;
    const tap = Fake{ .ptr = 1000, .bytes = &data, .calls = &calls };
    try std.testing.expectError(error.InvalidTapPath, apply(tap, {}, 0, &.{ 0, 2, 2 }, 4, 1));
    try std.testing.expectEqual(@as(usize, 0), calls);
    try std.testing.expectError(error.InjectedCopyFailure, apply(Fake{ .ptr = 1000, .bytes = &data, .fail_on = 0, .calls = &calls }, {}, 0, &.{ 0, 2 }, 4, 1));
}

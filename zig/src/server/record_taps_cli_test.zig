const std = @import("std");
const cli = @import("cli.zig");

test "--record-taps requires --record and preserves the supplied list" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var usage: cli.Usage = .{};
    try std.testing.expectError(error.Usage, cli.parse(arena.allocator(), &.{ "model", "--record-taps", "1,2" }, &usage));
    const parsed = try cli.parse(arena.allocator(), &.{ "model", "--record", "out", "--record-taps", "1,2" }, &usage);
    try std.testing.expectEqualStrings("1,2", parsed.record_taps.?);
}

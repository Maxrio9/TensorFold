//! The Mini EXL3 pack metadata shared by the family config parser and its host tests.
const std = @import("std");

/// Mini release packs omit the version; an explicit version must be exllamav3 1.x.
pub fn admit(q: std.json.ObjectMap) !void {
    const book = q.get("codebook") orelse return error.UnsupportedExl3;
    if (book != .string or !std.mem.eql(u8, book.string, "mul1")) return error.UnsupportedExl3;
    if (q.get("version")) |version| {
        if (version != .string or !std.mem.startsWith(u8, version.string, "1.")) return error.UnsupportedExl3;
    }
}

test "exl3: Mini admits absent or 1.x version on mul1" {
    const a = std.testing.allocator;
    for ([_][]const u8{
        "{\"codebook\":\"mul1\"}",
        "{\"codebook\":\"mul1\",\"version\":\"1.6.6\"}",
        "{\"codebook\":\"mul1\",\"version\":\"0.9\"}",
        "{\"codebook\":\"mcg\"}",
    }, 0..) |json, i| {
        const doc = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer doc.deinit();
        if (i < 2) try admit(doc.value.object) else try std.testing.expectError(error.UnsupportedExl3, admit(doc.value.object));
    }
}

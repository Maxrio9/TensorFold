//! Parsing and validating explicit recording tap indices.
const std = @import("std");

pub fn parse(a: std.mem.Allocator, text: []const u8) ![]u32 {
    var values: std.ArrayList(u32) = .empty;
    errdefer values.deinit(a);
    var parts = std.mem.splitScalar(u8, text, ',');
    while (parts.next()) |part| {
        if (part.len == 0) return error.InvalidRecordTaps;
        const value = std.fmt.parseUnsigned(u32, part, 10) catch return error.InvalidRecordTaps;
        try values.append(a, value);
    }
    return values.toOwnedSlice(a);
}

pub fn validate(values: []const u32, layers: usize) !void {
    if (values.len == 0) return error.InvalidRecordTaps;
    for (values, 0..) |value, i| {
        if (value > layers) return error.InvalidRecordTaps;
        for (values[0..i]) |prior| if (value == prior) return error.InvalidRecordTaps;
    }
}

/// The two contiguous spans of a logical row range in a circular buffer.
pub fn spans(start: u64, count: usize, capacity: usize) !struct { slot: usize, first: usize } {
    if (capacity == 0 or count > capacity) return error.RowsNotHeld;
    const slot: usize = @intCast(start % capacity);
    return .{ .slot = slot, .first = @min(count, capacity - slot) };
}

pub fn defaults(a: std.mem.Allocator, draft: []const u32, layers: u32) ![]u32 {
    if (draft.len > 0) {
        try validate(draft, layers);
        return a.dupe(u32, draft);
    }
    const result = try a.alloc(u32, 1);
    result[0] = layers;
    return result;
}

pub fn configure(a: std.mem.Allocator, loaded: anytype, o: anytype, served: anytype, initial: []const u32, problem: anytype) !?[]const u32 {
    var taps = initial;
    if (comptime @hasField(@TypeOf(loaded), "record_engine")) {
        if (o.record != null) {
            const engine = loaded.record_engine;
            const draft_taps = engine.taps[0..engine.n_taps];
            taps = if (o.record_taps) |list| parse(a, list) catch {
                problem.* = "--record-taps: invalid tap list";
                return null;
            } else try defaults(a, draft_taps, @intCast(engine.g.layers));
            var keep = false;
            defer if (!keep) a.free(taps);
            if (taps.len > engine.taps.len) {
                problem.* = "--record-taps: too many taps";
                return null;
            }
            validate(taps, engine.g.layers) catch {
                problem.* = "--record-taps: invalid tap list for target layers";
                return null;
            };
            if (draft_taps.len > 0 and !std.mem.eql(u32, draft_taps, taps)) {
                problem.* = "--record-taps: loaded drafter uses different taps";
                return null;
            }
            // before the stream pool is reserved, so each stream gets its tap ring
            const had = engine.tapBytes();
            try engine.setTaps(taps);
            served.stream_bytes += engine.tapBytes() - had;
            keep = true;
        }
    } else if (o.record_taps != null) {
        problem.* = "--record-taps: unsupported target";
        return null;
    }
    return taps;
}

test "record defaults retain loaded taps or select only the final output" {
    const a = std.testing.allocator;
    const only_final = try defaults(a, &.{}, 64);
    defer a.free(only_final);
    try std.testing.expectEqualSlices(u32, &.{64}, only_final);
    const with_draft = try defaults(a, &.{ 1, 61 }, 64);
    defer a.free(with_draft);
    try std.testing.expectEqualSlices(u32, &.{ 1, 61 }, with_draft);
    const already_final = try defaults(a, &.{ 1, 64 }, 64);
    defer a.free(already_final);
    try std.testing.expectEqualSlices(u32, &.{ 1, 64 }, already_final);
}

test "record taps parse and reject a planted bad tap" {
    const a = std.testing.allocator;
    const values = try parse(a, "1,16,31,46,61,64");
    defer a.free(values);
    try validate(values, 64);
    try std.testing.expectError(error.InvalidRecordTaps, validate(values, 63));
    for ([_][]const u8{ "", ",1", "1,", "1,,64", "-1", "x", "4294967296" }) |bad| {
        try std.testing.expectError(error.InvalidRecordTaps, parse(a, bad));
    }
    try std.testing.expectError(error.InvalidRecordTaps, validate(&.{ 1, 1 }, 64));
    try validate(&.{ 64, 1 }, 64);
}

test "ring spans preserve order at wrap and refuse overwritten rows" {
    const straight = try spans(2, 3, 8);
    try std.testing.expectEqual(@as(usize, 2), straight.slot);
    try std.testing.expectEqual(@as(usize, 3), straight.first);
    const wrapped = try spans(15, 4, 8);
    try std.testing.expectEqual(@as(usize, 7), wrapped.slot);
    try std.testing.expectEqual(@as(usize, 1), wrapped.first);
    try std.testing.expectError(error.RowsNotHeld, spans(0, 9, 8));
    try std.testing.expectError(error.RowsNotHeld, spans(0, 1, 0));
}

test "configured recording preserves a loaded drafter's exact tap layout" {
    const Engine = struct {
        const Self = @This();
        taps: [8]u32 = .{ 1, 61, 0, 0, 0, 0, 0, 0 },
        n_taps: usize = 2,
        g: struct { layers: usize } = .{ .layers = 64 },
        fn tapBytes(e: *Self) usize {
            return e.n_taps * 8;
        }
        fn setTaps(e: *Self, taps: []const u32) !void {
            @memcpy(e.taps[0..taps.len], taps);
            e.n_taps = taps.len;
        }
    };
    const Options = struct { record: ?[]const u8, record_taps: ?[]const u8 };
    var e: Engine = .{};
    var served: struct { stream_bytes: usize } = .{ .stream_bytes = 16 };
    var problem: []const u8 = "";
    const loaded = .{ .record_engine = &e };
    const taps = (try configure(std.testing.allocator, loaded, Options{ .record = "out", .record_taps = null }, &served, &.{}, &problem)).?;
    defer std.testing.allocator.free(taps);
    try std.testing.expectEqualSlices(u32, &.{ 1, 61 }, taps);
    try std.testing.expectEqual(@as(usize, 16), served.stream_bytes);
    const mismatch = try configure(std.testing.allocator, loaded, Options{ .record = "out", .record_taps = "1,61,64" }, &served, &.{}, &problem);
    try std.testing.expect(mismatch == null);
    try std.testing.expectEqualStrings("--record-taps: loaded drafter uses different taps", problem);
    try std.testing.expectEqualSlices(u32, &.{ 1, 61 }, e.taps[0..e.n_taps]);
}

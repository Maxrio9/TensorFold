//! Opt-in tool-call guard for OpenAI chat replies.

// Copyright 2026 MiaAI-Lab
const std = @import("std");
const json = @import("json");
const Value = json.Value;
const Allocator = std.mem.Allocator;

/// The note closing the prompt of a guarded request.
pub const note = "Tools are not available for this reply. Answer in plain text from what you already have; do not write tool calls.";

var cached: std.atomic.Value(u8) = .init(0); // 0 unread, 1 off, 2 on
/// Whether TF_TOOL_CALL_GUARD is 1 (read once).
pub fn on() bool {
    const value = cached.load(.monotonic);
    if (value != 0) return value == 2;
    const enabled = @import("sys").environment.read(bool, "TF_TOOL_CALL_GUARD", false) catch false;
    cached.store(if (enabled) 2 else 1, .monotonic);
    return enabled;
}

/// Whether `messages` (the normalized list) holds a tool result or an assistant turn with tool calls.
pub fn usedTools(messages: Value) bool {
    if (messages != .array) return false;
    for (messages.array) |m| {
        if (m != .object) continue;
        if (m.object.get("role")) |r| if (r == .string and std.mem.eql(u8, r.string, "tool")) return true;
        if (m.object.get("tool_calls")) |c| if (c.truthy()) return true;
    }
    return false;
}

/// `messages` with the guard's system note appended.
pub fn withNote(a: Allocator, messages: Value) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "role", .{ .string = "system" });
    try o.put(a, "content", .{ .string = note });
    const list = try a.alloc(Value, messages.array.len + 1);
    @memcpy(list[0..messages.array.len], messages.array);
    list[messages.array.len] = .{ .object = o };
    return .{ .array = list };
}

test "usedTools: a tool result or assistant tool_calls, not plain chat" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const plain = (try json.parseText(a, "[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"hello\"}]")).ok;
    try std.testing.expect(!usedTools(plain));
    const called = (try json.parseText(a, "[{\"role\":\"user\",\"content\":\"q\"},{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"id\":\"c1\"}]}]")).ok;
    try std.testing.expect(usedTools(called));
    const result = (try json.parseText(a, "[{\"role\":\"tool\",\"content\":\"r\"}]")).ok;
    try std.testing.expect(usedTools(result));
    const noted = try withNote(a, called);
    try std.testing.expectEqual(@as(usize, 3), noted.array.len);
    try std.testing.expectEqualStrings("system", noted.array[2].object.get("role").?.string);
}

test "Policy.guard: whole and unclosed call envelopes leave the reply, prose stays, streamed or whole" {
    const ts = @import("tool_stream.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const off: ts.Policy = .{};
    const call = "<tool_call>session_search<arg_key>after</arg_key><arg_value>7d</arg_value></tool_call>";
    try std.testing.expectEqualStrings("Done." ++ call, try off.content(a, "Done." ++ call));
    const guarded: ts.Policy = .{ .guard = true };
    try std.testing.expectEqualStrings("Done. More.", try guarded.content(a, "Done." ++ call ++ call ++ " More."));
    try std.testing.expectEqualStrings("Summary", try guarded.content(a, "Summary<tool_call>session_search<arg_key>q"));
    try std.testing.expectEqualStrings("a < b and <b>bold</b>", try guarded.content(a, "a < b and <b>bold</b>"));
    // streamed: split across deltas, the markup never reaches the text
    var p: ts.Policy = .{ .guard = true };
    var shown: std.ArrayList(u8) = .empty;
    for ([_][]const u8{ "Here", " it is <tool", "_call>session_search<arg_key>a</arg_key>", "<arg_value>1</arg_value></tool_", "call><tool_call>x</tool_call>", " end" }) |d| try shown.appendSlice(a, try p.text(a, d));
    try shown.appendSlice(a, p.flush());
    try std.testing.expectEqualStrings("Here it is  end", shown.items);
    try std.testing.expect(p.filter.hid);
}

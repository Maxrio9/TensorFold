//! OpenAI wire compatibility for reasoning text.

// Copyright 2026 MiaAI-Lab
const std = @import("std");
const json = @import("json");

const Field = enum(u8) { unread, reasoning_content, reasoning, both };
var cached: std.atomic.Value(Field) = .init(.unread);

fn field() Field {
    const value = cached.load(.monotonic);
    if (value != .unread) return value;
    const raw = @import("sys").environment.rawText("TF_REASONING_FIELD") orelse "";
    const selected = std.meta.stringToEnum(Field, raw) orelse .reasoning_content;
    const result = if (selected == .unread) Field.reasoning_content else selected;
    cached.store(result, .monotonic);
    return result;
}

pub fn put(a: std.mem.Allocator, object: *json.Object, text: []const u8) std.mem.Allocator.Error!void {
    const selected = field();
    if (selected == .reasoning) {
        _ = object.fetchOrderedRemove("reasoning_content");
    } else try object.put(a, "reasoning_content", .{ .string = text });
    if (selected != .reasoning_content) try object.put(a, "reasoning", .{ .string = text });
}

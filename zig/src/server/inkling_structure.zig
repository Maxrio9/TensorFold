//! Native Inkling 2 reply grammars and the prompt wording for tool_choice.

// Copyright 2026 MiaAI-Lab
const std = @import("std");
const json = @import("json");
const template = @import("template");
const model_text = @import("model_text.zig");
const errors = @import("errors.zig");
const grammar = @import("grammar.zig");
const tool_specs = @import("tool_specs.zig");
const tool_grammar = @import("tool_grammar.zig");
const markers = struct {
    const model = "<|message_model|>";
    const text = "<|content_text|>";
    const thinking = "<|content_thinking|>";
    const tool_json = "<|content_invoke_tool_json|>";
    const tool_text = "<|content_invoke_tool_text|>";
    const tool_error = "<|content_tool_error|>";
    const end_sampling = "<|content_model_end_sampling|>";
    const end = "<|end_message|>";
};
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// The sampling field the chat route records the instruction in (not a request field: plan() writes it).
pub const instruction_field = "inkling_tool_instruction";

/// The most blank characters between a constrained JSON value's tokens (grammar.zig's BLANKS for plain JSON).
const value_blanks = 32;

/// Every framing token (vLLM INKLING_CONTROL_TOKENS): free text in a grammar never spells one.
const control = [_][]const u8{
    "<|endoftext|>",     "<|message_user|>", "<|message_system|>",      "<|message_tool|>",
    "<|content_image|>", "<|content_xml|>",  "<|content_audio_input|>", "<|audio_end|>",
    markers.model,       markers.text,       markers.thinking,          markers.tool_json,
    markers.tool_text,   markers.tool_error, markers.end_sampling,      markers.end,
};

/// vLLM's tool_choice instruction, with function names serialized using ASCII escapes.
pub fn instruction(a: Allocator, choice: ?Value, parallel: bool) Allocator.Error!?[]const u8 {
    const c = choice orelse return if (parallel) null else try a.dupe(u8, auto_single);
    switch (c) {
        .null => return if (parallel) null else try a.dupe(u8, auto_single),
        .string => |s| {
            if (std.mem.eql(u8, s, "required")) return if (parallel)
                try a.dupe(u8, "Call at least one of the provided functions in your next response.")
            else
                try a.dupe(u8, "Call exactly one of the provided functions in your next response.");
            if (std.mem.eql(u8, s, "auto") and !parallel) return try a.dupe(u8, auto_single);
            return null;
        },
        .object => {
            const kind = c.get("type") orelse return null;
            if (kind != .string or !std.mem.eql(u8, kind.string, "function")) return null;
            const function = c.get("function");
            const name_v = if (function) |f| (if (f == .object) f.get("name") else null) else c.get("name");
            const name = if (name_v) |n| (if (n == .string) n.string else return null) else return null;
            const quoted = try json.stringify(a, .{ .string = name }, .{ .ascii = true });
            return try std.fmt.allocPrint(a, "Call the function named {s}{s} in your next response.", .{ quoted, if (parallel) "" else " exactly once" });
        },
        else => return null,
    }
}

const auto_single = "If you call a function, call at most one function in your next response.";

/// The recorded chat instruction or the one /tokenize derives from the request.
pub fn instructionOf(a: Allocator, fields: Value) Allocator.Error!?[]const u8 {
    if (fields != .object) return null;
    if (fields.get(instruction_field)) |v| return if (v == .string) v.string else null;
    if (fields.get("messages") == null) return null; // a sampling object without the field: no instruction
    const p = fields.get("parallel_tool_calls");
    return instruction(a, fields.get("tool_choice"), !(p != null and p.? == .bool and !p.?.bool));
}

/// Add the instruction to a leading system or developer message, or return null.
pub fn withInstruction(a: Allocator, messages: Value, text: []const u8) Allocator.Error!?Value {
    if (messages != .array or messages.array.len == 0) return null;
    const first = messages.array[0];
    if (first != .object) return null;
    const role = first.get("role") orelse return null;
    if (role != .string or !(std.mem.eql(u8, role.string, "system") or std.mem.eql(u8, role.string, "developer"))) return null;
    const copy = try json.newObject(a);
    var it = first.object.iterator();
    while (it.next()) |e| try copy.put(a, e.key_ptr.*, e.value_ptr.*);
    const content = first.get("content");
    const joined: Value = if (content) |cv| switch (cv) {
        .string => |s| .{ .string = if (s.len > 0) try std.mem.concat(a, u8, &.{ s, "\n\n", text }) else text },
        .array => |parts| blk: {
            const part = try json.newObject(a);
            try part.put(a, "type", .{ .string = "text" });
            try part.put(a, "text", .{ .string = text });
            break :blk .{ .array = try std.mem.concat(a, Value, &.{ parts, &.{.{ .object = part }} }) };
        },
        .null => .{ .string = text },
        else => cv,
    } else .{ .string = text };
    try copy.put(a, "content", joined);
    const out = try a.dupe(Value, messages.array);
    out[0] = .{ .object = copy };
    return .{ .array = out };
}

/// Add the tool instruction to the synthetic effort block when no system message precedes the conversation.
pub fn appendToEffort(
    a: Allocator,
    compiled: *const template.Compiled,
    tools: std.json.Value,
    context: std.json.ObjectMap,
    globals: std.json.ObjectMap,
    rendered: []const u8,
    spans: []const model_text.Span,
    starts: ?*[]usize,
    sentence: []const u8,
    problem: *[]const u8,
) model_text.Error!model_text.Rendered {
    var diag: template.Diag = .{};
    const head = template.renderMarked(a, compiled, try @import("hf_text.zig").toStd(a, .{ .array = &.{} }), tools, .{ .object = context }, .{ .object = globals }, false, &diag, null) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        problem.* = diag.msg;
        return error.Template;
    };
    const last = std.mem.lastIndexOf(u8, head.s, "<|message_system|>") orelse head.s.len;
    const effort = std.mem.startsWith(u8, head.s[last..], "<|message_system|><|content_text|>Thinking effort level: ");
    if (!effort or !std.mem.startsWith(u8, rendered, head.s) or !std.mem.endsWith(u8, head.s, markers.end)) {
        problem.* = "tool_choice: the template is missing Inkling 2's thinking effort block";
        return error.Template;
    }
    const at = head.s.len - markers.end.len;
    const add = try std.mem.concat(a, u8, &.{ "\n\n", sentence });
    const joined = try std.mem.concat(a, u8, &.{ rendered[0..at], add, rendered[at..] });
    const moved = try a.dupe(model_text.Span, spans);
    for (moved) |*sp| for (sp) |*x| {
        if (x.* > at) x.* += add.len;
    };
    if (starts) |offsets| for (offsets.*) |*x| {
        if (x.* > at) x.* += add.len;
    };
    return .{ .text = joined, .spans = moved };
}

fn node(a: Allocator, kind: []const u8, entries: []const struct { []const u8, Value }) Allocator.Error!Value {
    return tool_grammar.node(a, kind, entries);
}

/// A framing token by id (the tokenizer's own; a 400 when the checkpoint lacks it).
fn token(srv: *Server, cx: *Cx, spelled: []const u8) errors.Refused!Value {
    const id = srv.text.tokenId(spelled) orelse return cx.fail(.request, "structured output: the tokenizer has no {s} token", .{spelled});
    return node(cx.a, "token", &.{.{ "token", try json.intValue(cx.a, id) }});
}

fn constant(a: Allocator, text: []const u8) Allocator.Error!Value {
    return node(a, "const_string", &.{.{ "value", .{ .string = text } }});
}

fn schemaNode(a: Allocator, schema: Value) Allocator.Error!Value {
    return node(a, "json_schema", &.{ .{ "json_schema", schema }, .{ "style", .{ .string = "json" } }, .{ "max_whitespace_cnt", try json.intValue(a, value_blanks) } });
}

/// Zero or more thinking blocks: `<|message_model|><|content_thinking|>` free text `<|end_message|>`.
fn thinking(srv: *Server, cx: *Cx) errors.Refused!Value {
    const a = cx.a;
    var excludes: std.ArrayList(Value) = .empty;
    for (control) |c| if (srv.text.tokenId(c) != null) try excludes.append(a, .{ .string = c });
    const text = try node(a, "any_text", &.{.{ "excludes", .{ .array = excludes.items } }});
    const block = try node(a, "sequence", &.{.{ "elements", .{ .array = try a.dupe(Value, &.{ try token(srv, cx, markers.model), try token(srv, cx, markers.thinking), text, try token(srv, cx, markers.end) }) } }});
    return node(a, "star", &.{.{ "content", block }});
}

fn tag(a: Allocator, format: Value) Allocator.Error![]const u8 {
    return json.stringify(a, try node(a, "structural_tag", &.{.{ "format", format }}), .{ .ascii = false, .compact = true });
}

/// The inner format a grammar request asks for, as a structural tag node.
fn inner(cx: *Cx, s: grammar.Spec) errors.Refused!Value {
    const a = cx.a;
    return switch (s.kind) {
        .json => schemaNode(a, try node(a, "object", &.{})),
        .json_schema => schemaNode(a, switch (try json.parse(a, s.text)) {
            .ok => |v| v,
            .err => return cx.fail(.request, "{s}: the schema is not JSON", .{s.field}),
        }),
        .regex => node(a, "regex", &.{.{ "pattern", .{ .string = s.text } }}),
        .grammar => node(a, "grammar", &.{.{ "grammar", .{ .string = s.text } }}),
        .choice => blk: {
            const list = switch (try json.parse(a, s.text)) {
                .ok => |v| v,
                .err => return cx.fail(.request, "{s}: the choices are not JSON", .{s.field}),
            };
            const items = list.items() orelse return cx.fail(.request, "{s} must be a list", .{s.field});
            const options = try a.alloc(Value, items.len);
            for (items, options) |it, *o| o.* = try constant(a, it.string);
            if (options.len == 1) break :blk options[0];
            break :blk node(a, "or", &.{.{ "elements", .{ .array = options } }});
        },
        .structural_tag => unreachable,
    };
}

/// A framed schema reply allows thinking blocks before one constrained text block.
pub fn framed(srv: *Server, cx: *Cx, s: grammar.Spec, allow_thinking: bool) errors.Refused![]const u8 {
    const a = cx.a;
    const elements = &.{ try token(srv, cx, markers.model), try token(srv, cx, markers.text), try inner(cx, s), try token(srv, cx, markers.end) };
    return framedBody(srv, cx, try sequence(a, elements), allow_thinking);
}

/// One native tool block, whose arguments use the tool's schema and preserve its root.
fn callBlock(srv: *Server, cx: *Cx, tool: Value) errors.Refused!Value {
    const a = cx.a;
    const name = try tool_specs.toolName(a, tool);
    const quoted = try json.stringify(a, .{ .string = name }, .{ .ascii = false });
    const args = try tool_grammar.parameters(a, tool);
    const elements = try a.dupe(Value, &.{
        try token(srv, cx, markers.model),
        try constant(a, name),
        try token(srv, cx, markers.tool_json),
        try constant(a, try std.mem.concat(a, u8, &.{ "{\"name\":", quoted, ",\"args\":" })),
        try schemaNode(a, args),
        try constant(a, "}"),
        try token(srv, cx, markers.end),
    });
    return node(a, "sequence", &.{.{ "elements", .{ .array = elements } }});
}

/// A forced tool reply allows thinking blocks before one call or multiple calls when parallel is enabled.
pub fn calls(srv: *Server, cx: *Cx, tools: []const Value, parallel: bool, allow_thinking: bool) errors.Refused![]const u8 {
    const a = cx.a;
    const blocks = try a.alloc(Value, tools.len);
    for (tools, blocks) |tool, *b| b.* = try callBlock(srv, cx, tool);
    const one = if (blocks.len == 1) blocks[0] else try node(a, "or", &.{.{ "elements", .{ .array = blocks } }});
    const body = if (parallel) try node(a, "plus", &.{.{ "content", one }}) else one;
    return framedBody(srv, cx, body, allow_thinking);
}

fn sequence(a: Allocator, elements: []const Value) Allocator.Error!Value {
    return node(a, "sequence", &.{.{ "elements", .{ .array = try a.dupe(Value, elements) } }});
}

fn framedBody(srv: *Server, cx: *Cx, body: Value, allow_thinking: bool) errors.Refused![]const u8 {
    if (!allow_thinking) return tag(cx.a, body);
    return tag(cx.a, try sequence(cx.a, &.{ try thinking(srv, cx), body }));
}

const testing = std.testing;

test "tool_choice sentences are vLLM's" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqualStrings("Call at least one of the provided functions in your next response.", (try instruction(a, .{ .string = "required" }, true)).?);
    try testing.expectEqualStrings("Call exactly one of the provided functions in your next response.", (try instruction(a, .{ .string = "required" }, false)).?);
    try testing.expectEqualStrings(auto_single, (try instruction(a, .{ .string = "auto" }, false)).?);
    try testing.expectEqual(@as(?[]const u8, null), try instruction(a, .{ .string = "auto" }, true));
    try testing.expectEqual(@as(?[]const u8, null), try instruction(a, null, true));
    try testing.expectEqual(@as(?[]const u8, null), try instruction(a, .{ .string = "none" }, false));
    const named = (try json.parse(a, "{\"type\":\"function\",\"function\":{\"name\":\"get_w\\u00e9ather\"}}")).ok;
    try testing.expectEqualStrings("Call the function named \"get_w\\u00e9ather\" in your next response.", (try instruction(a, named, true)).?);
    try testing.expectEqualStrings("Call the function named \"get_w\\u00e9ather\" exactly once in your next response.", (try instruction(a, named, false)).?);
}

test "message 0 takes the sentence as vLLM's _with_appended_text adds it" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { []const u8, ?[]const u8 }{
        .{ "[{\"role\":\"system\",\"content\":\"Be brief.\"},{\"role\":\"user\",\"content\":\"hi\"}]", "Be brief.\n\nX" },
        .{ "[{\"role\":\"developer\",\"content\":\"\"},{\"role\":\"user\",\"content\":\"hi\"}]", "X" },
        .{ "[{\"role\":\"system\",\"content\":null}]", "X" },
        .{ "[{\"role\":\"user\",\"content\":\"hi\"}]", null },
    };
    for (cases) |c| {
        const got = try withInstruction(a, (try json.parse(a, c[0])).ok, "X");
        if (c[1]) |want| try testing.expectEqualStrings(want, got.?.array[0].get("content").?.string) else try testing.expectEqual(@as(?Value, null), got);
    }
    const parts = (try withInstruction(a, (try json.parse(a, "[{\"role\":\"system\",\"content\":[{\"type\":\"text\",\"text\":\"A\"}]}]")).ok, "X")).?;
    const list = parts.array[0].get("content").?.array;
    try testing.expectEqual(@as(usize, 2), list.len);
    try testing.expectEqualStrings("X", list[1].get("text").?.string);
}

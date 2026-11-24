//! Command-line inputs and reports for Mini's CUDA capacity planner.
const std = @import("std");
const engines = @import("native_engines");
const budget = engines.budget;
const mini = engines.mini_capacity;
const Allocator = std.mem.Allocator;
const Writer = std.Io.Writer;
const MiB = 1 << 20;
const GiB: f64 = 1 << 30;

/// RTX PRO 4500 warm process memory minus TensorFold's tracked device bytes, rounded up to MiB.
pub const default_driver_mib = 335;

pub const Options = struct {
    device_mib: u64,
    free_mib: ?u64 = null,
    driver_mib: ?u64 = null,
    desktop_reserve_gib: f64 = 0,
    reserve_gib: ?[]const u8 = null,
    limit_gib: ?[]const u8 = null,
    parallel: u32,
    context: ?u32 = null,
    bf16: bool = false,
    unified: bool = false,
    no_drafts: bool = false,
    drafter: ?[]const u8 = null,
    json: bool = false,

    fn total(o: Options) u64 {
        return o.device_mib * MiB;
    }

    fn free(o: Options) u64 {
        if (o.free_mib) |available| return available * MiB;
        const desktop: u64 = @intFromFloat(o.desktop_reserve_gib * GiB);
        return o.total() -| ((o.driver_mib orelse default_driver_mib) * MiB) -| desktop;
    }
};

pub fn parse(args: []const []const u8) ?Options {
    var o: Options = .{ .device_mib = 0, .parallel = 0 };
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        const eql = std.mem.eql;
        if (eql(u8, flag, "--unified")) {
            o.unified = true;
        } else if (eql(u8, flag, "--no-drafts")) {
            o.no_drafts = true;
        } else if (eql(u8, flag, "--json")) {
            o.json = true;
        } else {
            i += 1;
            if (i == args.len) return null;
            const value = args[i];
            if (eql(u8, flag, "--device-mib")) {
                o.device_mib = sizeMiB(value) orelse return null;
            } else if (eql(u8, flag, "--free-mib")) {
                o.free_mib = sizeMiB(value) orelse return null;
            } else if (eql(u8, flag, "--driver-mib")) {
                o.driver_mib = sizeMiB(value) orelse return null;
            } else if (eql(u8, flag, "--desktop-reserve-gib")) {
                o.desktop_reserve_gib = sizeGiB(value) orelse return null;
            } else if (eql(u8, flag, "--reserve-gib")) {
                o.reserve_gib = value;
            } else if (eql(u8, flag, "--limit-gib")) {
                o.limit_gib = value;
            } else if (eql(u8, flag, "--parallel")) {
                o.parallel = std.fmt.parseInt(u32, value, 10) catch return null;
            } else if (eql(u8, flag, "--drafter")) {
                if (value.len == 0) return null;
                o.drafter = value;
            } else if (eql(u8, flag, "--context")) {
                o.context = std.fmt.parseInt(u32, value, 10) catch return null;
            } else if (eql(u8, flag, "--kv-cache")) {
                if (eql(u8, value, "bf16")) o.bf16 = true else if (eql(u8, value, "fp8")) o.bf16 = false else return null;
            } else return null;
        }
    }
    if (o.device_mib == 0 or o.parallel == 0 or (o.free_mib orelse 0) > o.device_mib) return null;
    if (comptime @hasDecl(engines, "mini_capacity")) {
        if (o.parallel > mini.max_streams) return null;
        if (o.reserve_gib) |value| {
            if (std.mem.trim(u8, value, " ").len == 0) return null;
            _ = budget.reserveBytes(value, o.total(), o.unified) catch return null;
        }
        if (o.limit_gib) |value| if ((budget.limitBytes(value) catch return null) == null) return null;
    }
    return o;
}

fn sizeMiB(value: []const u8) ?u64 {
    const n = std.fmt.parseInt(u64, value, 10) catch return null;
    return if (n <= std.math.maxInt(u64) / MiB) n else null;
}

fn sizeGiB(value: []const u8) ?f64 {
    const n = std.fmt.parseFloat(f64, value) catch return null;
    return if (n >= 0 and n * GiB < 0x1p64) n else null;
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / GiB;
}

fn printAmounts(out: *Writer, name: []const u8, group: anytype) !void {
    try out.print("{s}\n", .{name});
    inline for (@typeInfo(@TypeOf(group)).@"struct".field_names) |field| {
        const value = @field(group, field);
        const bytes: ?u64 = value;
        if (bytes) |n| try out.print("  {s} {d} bytes ({d:.4} GiB)\n", .{ field, n, gib(n) });
    }
}

fn jsonAmounts(j: *std.json.Stringify, name: []const u8, group: anytype) !void {
    try j.objectField(name);
    try j.beginObject();
    inline for (@typeInfo(@TypeOf(group)).@"struct".field_names) |field| {
        const bytes: ?u64 = @field(group, field);
        try j.objectField(field);
        if (bytes) |n| try j.write(.{ .bytes = n, .gib = gib(n) }) else try j.write(null);
    }
    try j.endObject();
}

fn jsonResult(j: *std.json.Stringify, r: mini.Result) !void {
    try j.beginObject();
    try j.objectField("context");
    try j.write(r.context);
    try j.objectField("admitted");
    try j.write(r.refused_at == null);
    try j.objectField("refused_at");
    try j.write(r.refused_at);
    try j.objectField("message");
    try j.write(r.message);
    if (r.components) |c| try jsonAmounts(j, "components", c);
    if (r.tracked) |t| try jsonAmounts(j, "tracked", t);
    try jsonAmounts(j, "rooms", r.rooms);
    try j.endObject();
}

pub fn run(a: Allocator, io: std.Io, out: *Writer, model: []const u8, dir: []const u8, o: Options) !u8 {
    const pool: budget.Pool = .{
        .total = o.total(),
        .free = o.free(),
        .unified = o.unified,
        .reserve = try budget.reserveBytes(o.reserve_gib, o.total(), o.unified),
        .limit = try budget.limitBytes(o.limit_gib),
    };
    const q: mini.Query = .{
        .parallel = o.parallel,
        .context = o.context,
        .format = if (o.bf16) .bf16 else .fp8,
        .drafts = !o.no_drafts,
        .drafter = o.drafter,
    };
    var problem: []const u8 = "";
    const r = try mini.plan(a, io, dir, pool, q, &problem) orelse {
        if (o.json) {
            try std.json.Stringify.value(.{ .model = model, .error_message = problem }, .{}, out);
            try out.writeByte('\n');
        } else try out.print("{s}: {s}\n", .{ model, problem });
        return 1;
    };
    const memory = .{
        .total = pool.total,
        .startup_free = pool.free,
        .driver_allowance = if (o.free_mib == null) (o.driver_mib orelse default_driver_mib) * MiB else null,
        .desktop_allowance = if (o.free_mib == null) @as(?u64, @intFromFloat(o.desktop_reserve_gib * GiB)) else null,
        .reserve = pool.reserve,
        .limit = pool.limit,
    };
    if (o.json) {
        var j: std.json.Stringify = .{ .writer = out, .options = .{} };
        try j.beginObject();
        try j.objectField("model");
        try j.write(model);
        try j.objectField("unified");
        try j.write(o.unified);
        try j.objectField("parallel");
        try j.write(o.parallel);
        try j.objectField("kv_cache");
        try j.write(q.format);
        try j.objectField("drafts");
        try j.write(r.drafts);
        try j.objectField("model_cap");
        try j.write(r.model_cap);
        try j.objectField("maximum");
        try j.write(r.maximum);
        try j.objectField("rounded_down");
        try j.write(r.rounded_down);
        try j.objectField("model_cap_binds");
        try j.write(r.model_cap_binds);
        try jsonAmounts(&j, "memory", memory);
        try j.objectField("result");
        try jsonResult(&j, r.result);
        try j.objectField("next_context");
        if (r.next_context) |n| try jsonResult(&j, n) else try j.write(null);
        try j.endObject();
        try out.writeByte('\n');
    } else {
        const drafts = r.drafts;
        try out.print("Mini CUDA capacity for {s}, {s}, {d} streams, {s} KV, drafts {s}\n", .{
            model, if (o.unified) "unified" else "discrete", o.parallel, @tagName(q.format), if (drafts) "on" else "off",
        });
        try printAmounts(out, "Memory", memory);
        if (r.maximum) |n| {
            try out.print("Maximum context per request {d} tokens ({d} rounded down to 1024)\n", .{ n, r.rounded_down.? });
            try out.print("Model cap {d} tokens ({s})\n", .{ r.model_cap, if (r.model_cap_binds.?) "binding" else "available beyond this budget" });
        }
        if (r.result.components) |c| try printAmounts(out, "Allocation components", c);
        if (r.result.tracked) |t| try printAmounts(out, "Tracked totals", t);
        try printAmounts(out, "Admission room", r.result.rooms);
        if (r.result.message) |why| {
            try out.print("Context {d} refused at {s}: {s}\n", .{ r.result.context, @tagName(r.result.refused_at.?), why });
        } else try out.print("Context {d} admitted\n", .{r.result.context});
        if (r.next_context) |n| try out.print("Context {d} refused at {s}: {s}\n", .{ n.context, @tagName(n.refused_at.?), n.message.? });
    }
    return if (r.result.refused_at == null) 0 else 1;
}

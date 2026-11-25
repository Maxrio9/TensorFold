//! Mini's CUDA startup admission with explicit pool readings and checkpoint allocation sizes.
const std = @import("std");
const mini = @import("inkling2");
const api = @import("engine_api");
const memory = @import("cuda_family_memory.zig");
const budget = @import("cuda_memory.zig");
const windows = @import("cuda_window.zig");
const kv_pages = @import("cuda_kv_pool.zig");
const startup = @import("cuda_startup_costs.zig");
const streams = @import("cuda_streams.zig");
const Allocator = std.mem.Allocator;

pub const model_type = mini.native.model_type;
pub const Format = mini.capacity.Format;
pub const max_streams = mini.capacity.max_streams;

pub const Query = struct {
    parallel: u32,
    context: ?u32 = null,
    format: Format = .fp8,
    drafts: bool = true,
    drafter: ?[]const u8 = null,
};

pub const Result = struct {
    context: u32,
    refused_at: ?enum { model_context, weights, first_stream_state, snapshot_staging, calibration, streams } = null,
    message: ?[]const u8 = null,
    components: ?mini.capacity.Components = null,
    rooms: Rooms,
    tracked: ?Tracked = null,
};

const Rooms = struct {
    weights: u64,
    first_stream_state: ?u64 = null,
    after_model: ?u64 = null,
    after_staging: ?u64 = null,
    extra_streams: ?u64 = null,
    after_streams: ?u64 = null,
    prompt_cache_budget: ?u64 = null,
};

const Tracked = struct {
    weights_check: u64,
    first_stream_state: u64,
    model: u64,
    per_stream: u64,
    extra_streams: u64,
    ready: u64,
};

pub const Report = struct {
    model_cap: u32,
    drafts: bool,
    drafter: ?[]const u8,
    maximum: ?u32,
    rounded_down: ?u32,
    model_cap_binds: ?bool,
    result: Result,
    next_context: ?Result,
};

// The single supplied free count is the effective pool reading; project it with a fixed external load.
fn poolAfter(before: budget.Pool, allocated: u64) budget.Pool {
    var after = before;
    after.free -|= allocated;
    return after;
}

fn admit(a: Allocator, ck: *const mini.capacity.Checkpoint, before: budget.Pool, weight_check: u64, q: Query, context: u32) !Result {
    var r: Result = .{ .context = context, .rooms = .{ .weights = before.room(0) } };
    const window = windows.contextWindow(context, ck.text.max_context, mini.native.default_context) catch {
        r.refused_at = .model_context;
        r.message = try windows.pastModel(a, context, ck.text.max_context);
        return r;
    };
    r.context = @intCast(window);
    const c = try ck.allocations(r.context, q.format);
    r.components = c;
    const state = c.state();
    const model = c.weights() + state;
    const stream = c.stream();
    const extra = @as(u64, q.parallel - mini.native.startup_streams) * stream;
    r.tracked = .{ .weights_check = weight_check, .first_stream_state = state, .model = model, .per_stream = stream, .extra_streams = extra, .ready = model + extra };
    if (try memory.weightRefusal(a, weight_check, before, 0)) |why| {
        r.refused_at = .weights;
        r.message = why;
        return r;
    }
    const state_room = poolAfter(before, c.weights()).room(c.weights());
    r.rooms.first_stream_state = state_room;
    mini.native.checkCacheRoom(state, state_room) catch |err| {
        r.refused_at = .first_stream_state;
        r.message = mini.native.explain(null, err);
        return r;
    };
    const after = poolAfter(before, model);
    r.rooms.after_model = after.room(model);
    var problem: []const u8 = "";
    var unused: u8 = 0;
    const loaded = .{ .ctx = @as(*anyopaque, &unused), .free_streams = mini.native.startup_streams, .stream_bytes = stream };
    const o: api.Open = .{ .dir = "", .model_type = model_type, .context = window, .lanes = q.parallel, .lanes_fixed = true, .drafts = q.drafts };
    const staging = try kv_pages.startupRoom(mini.native, a, loaded, o, after.room(model), 0, &problem) orelse {
        r.refused_at = .snapshot_staging;
        r.message = problem;
        return r;
    };
    r.rooms.after_staging = staging;
    const room = try startup.room(a, null, staging, &problem) orelse {
        r.refused_at = .calibration;
        r.message = problem;
        return r;
    };
    r.rooms.extra_streams = room;
    const counted = try streams.count(mini.native, a, loaded, o, .{
        .room = room,
        .model = model,
        .reserve = before.reserve,
        .low = "",
        .window = window,
        .stream_bytes = stream,
        .host_cache = false,
        .host_budget = 0,
        .share = false,
    }, &problem) orelse {
        r.refused_at = .streams;
        r.message = problem;
        return r;
    };
    // Mini's eager slot allocation is complete here. The post-warm-up pool optimizer is specific to Qwen.
    const held = stream * (counted.streams -| loaded.free_streams);
    r.rooms.after_streams = room - held;
    r.rooms.prompt_cache_budget = kv_pages.kept(mini.native, loaded, null, o, room, held, 0, false).copied;
    return r;
}

fn search(a: Allocator, ck: *const mini.capacity.Checkpoint, before: budget.Pool, weight_check: u64, q: Query) !Report {
    const cap = ck.text.max_context;
    const drafter = if (ck.draft_arena > 0) ck.draft_kind.name() else null;
    if (q.context) |context| return .{
        .model_cap = cap,
        .drafts = drafter != null,
        .drafter = drafter,
        .maximum = null,
        .rounded_down = null,
        .model_cap_binds = null,
        .result = try admit(a, ck, before, weight_check, q, context),
        .next_context = null,
    };

    var scratch = std.heap.ArenaAllocator.init(a);
    defer scratch.deinit();
    var lo: u32 = 0;
    var hi = cap;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2 + 1;
        const r = try admit(scratch.allocator(), ck, before, weight_check, q, mid);
        if (r.refused_at == null) lo = mid else hi = mid - 1;
    }
    return .{
        .model_cap = cap,
        .drafts = drafter != null,
        .drafter = drafter,
        .maximum = lo,
        .rounded_down = lo / 1024 * 1024,
        .model_cap_binds = lo == cap,
        .result = try admit(a, ck, before, weight_check, q, @max(lo, 1)),
        .next_context = if (lo > 0 and lo < cap) try admit(a, ck, before, weight_check, q, lo + 1) else null,
    };
}

pub fn plan(a: Allocator, io: std.Io, dir: []const u8, before: budget.Pool, q: Query, problem: *[]const u8) !?Report {
    var why: @import("inkling2").capacity.ConfigWhy = .{};
    const ck = mini.capacity.Checkpoint.read(a, io, dir, q.drafts, q.drafter, &why) catch |err| {
        if (err == error.OutOfMemory) return err;
        problem.* = if (why.text().len > 0) try a.dupe(u8, why.text()) else mini.native.explain(null, err) orelse @errorName(err);
        return null;
    };
    const weight_check = if (ck.exl3) ck.checkpoint_bytes else budget.weightBytes(a, io, dir) catch |err| {
        if (err == error.OutOfMemory) return err;
        problem.* = try std.fmt.allocPrint(a, "cannot estimate checkpoint weight files ({s}); refusing CUDA admission", .{@errorName(err)});
        return null;
    };
    return try search(a, &ck, before, weight_check, q);
}

test "Mini capacity: binary search and fixed-context admission agree at both sides of the boundary" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ck = try mini.capacity.testCheckpoint(a);
    const g: u64 = 1 << 30;
    for ([_]bool{ false, true }) |unified| {
        for ([_]?u64{ null, 18 * g }) |limit| {
            const total: u64 = if (unified) 36 * g else 24 * g;
            const pool: budget.Pool = .{ .total = total, .free = total - g, .reserve = try budget.reserveBytes(null, total, unified), .limit = limit, .unified = unified };
            for ([_]u32{ 1, 4 }) |parallel| {
                const q: Query = .{ .parallel = parallel };
                const r = try search(a, &ck, pool, ck.checkpoint_bytes, q);
                try std.testing.expect(r.maximum.? > 0 and r.maximum.? < ck.text.max_context);
                try std.testing.expectEqual(r.maximum.?, r.result.context);
                try std.testing.expectEqual(null, r.result.refused_at);
                try std.testing.expectEqual(r.maximum.? + 1, r.next_context.?.context);
                const refused_at = r.next_context.?.refused_at.?;
                try std.testing.expectEqual(@as(@TypeOf(refused_at), if (parallel == 1) .first_stream_state else .streams), refused_at);
                var fixed = q;
                fixed.context = r.maximum.? + 64;
                try std.testing.expect((try search(a, &ck, pool, ck.checkpoint_bytes, fixed)).result.refused_at != null);
            }
        }
    }

    const pool: budget.Pool = .{ .free = 124 * g, .total = 128 * g, .reserve = 4 * g, .limit = null, .unified = true };
    const r = try search(a, &ck, pool, ck.checkpoint_bytes, .{ .parallel = 1 });
    try std.testing.expectEqual(1_048_576, r.maximum.?);
    try std.testing.expect(r.model_cap_binds.?);
    const over = try search(a, &ck, pool, ck.checkpoint_bytes, .{ .parallel = 1, .context = r.maximum.? + 1 });
    try std.testing.expectEqual(.model_context, over.result.refused_at.?);
    try std.testing.expectEqualStrings("--context 1048577 exceeds this model's 1048576-token window", over.result.message.?);

    var small = pool;
    small.limit = 8 * g;
    const refused = try search(a, &ck, small, ck.checkpoint_bytes, .{ .parallel = 1 });
    try std.testing.expectEqual(0, refused.maximum.?);
    try std.testing.expectEqual(.weights, refused.result.refused_at.?);
}

test "Mini capacity: four-stream DFlash2 reports its own weight, KV, scratch and ready bytes" {
    const a = std.testing.allocator;
    var ck = try mini.capacity.testCheckpoint(a);
    ck.draft_kind = .dflash2;
    ck.dense_scratch = 7_078_656;
    const gib = 1 << 30;
    const pool: budget.Pool = .{ .total = 32 * gib, .free = 31 * gib, .reserve = 4 * gib, .limit = null, .unified = false };
    const q: Query = .{ .parallel = 4, .context = 8704 };
    for ([_]struct { weights: u64, ready: u64 }{
        .{ .weights = 989_966_848, .ready = 17_102_789_360 },
        .{ .weights = 988_977_664, .ready = 17_101_800_176 },
        .{ .weights = 624_644_608, .ready = 16_737_467_120 },
        .{ .weights = 623_655_424, .ready = 16_736_477_936 },
    }) |row| {
        ck.draft_arena = row.weights;
        const r = try search(a, &ck, pool, ck.checkpoint_bytes, q);
        const c = r.result.components.?;
        try std.testing.expect(r.drafts);
        try std.testing.expectEqualStrings("DFlash2", r.drafter.?);
        try std.testing.expectEqual(row.weights, c.drafter_weights_and_padding);
        try std.testing.expectEqual(0, c.drafter_prepared_head);
        try std.testing.expectEqual(42_086_400, c.drafter_kv);
        try std.testing.expectEqual(47_947_520, c.drafter_shared_other + c.drafter_shared_taps + c.drafter_attention_partials);
        try std.testing.expectEqual(478_182_920, r.result.tracked.?.per_stream);
        try std.testing.expectEqual(row.ready, r.result.tracked.?.ready);
    }
    ck.draft_arena = 0;
    const off = try search(a, &ck, pool, ck.checkpoint_bytes, q);
    try std.testing.expect(!off.drafts and off.drafter == null);
    try std.testing.expectEqual(0, off.result.components.?.drafter_kv);
}

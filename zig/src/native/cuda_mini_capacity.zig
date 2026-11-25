//! Mini's CUDA startup admission with explicit pool readings and checkpoint allocation sizes.
const std = @import("std");
const mini = @import("inkling2");
const api = @import("engine_api");
const memory = @import("cuda_family_memory.zig");
const budget = @import("cuda_memory.zig");
const windows = @import("cuda_window.zig");
const kv = @import("core").kv_pool;
const Allocator = std.mem.Allocator;

pub const model_type = mini.native.model_type;
pub const Format = mini.capacity.Format;
pub const max_streams = mini.capacity.max_streams;

pub const Query = struct {
    parallel: u32,
    parallel_fixed: bool = true,
    context: ?u32 = null,
    capability: u32 = 120,
    format: Format = .fp8,
    drafts: bool = true,
    drafter: ?[]const u8 = null,
    vision: ?mini.native.VisionOptions = null,
    prompt_cache_gib: ?f64 = null,
    cache_entries: usize = kv.most_snaps,
};

pub const Result = struct {
    context: u32,
    refused_at: ?enum { model_context, weights, first_stream_state, streams } = null,
    message: ?[]const u8 = null,
    components: ?mini.capacity.Components = null,
    rooms: Rooms,
    tracked: ?Tracked = null,
    paging: ?mini.native.Paging = null,
};

const Rooms = struct {
    weights: u64,
    first_stream_state: ?u64 = null,
    after_model: ?u64 = null,
    extra_streams: ?u64 = null,
    after_streams: ?u64 = null,
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
    var c = try ck.allocationsMode(r.context, q.format, true);
    r.components = c;
    const state = c.state();
    const model = c.weights() + state;
    const stream = c.stream();
    if (try memory.weightRefusal(a, weight_check, before, 0)) |why| {
        r.refused_at = .weights;
        r.message = why;
        return r;
    }
    const state_room = poolAfter(before, c.weights()).room(c.weights());
    r.rooms.first_stream_state = state_room;
    const after = poolAfter(before, model);
    const room = after.room(model);
    r.rooms.after_model = room;
    r.rooms.extra_streams = room;
    const full: kv.groups.Group = .{ .kind = .full, .row = c.global_kv_per_token / ck.scheduleGlobalVisits(), .layers = ck.scheduleGlobalVisits() };
    var layout = kv.Layout.ofRows(full, mini.native.page_tokens, stream, 0);
    const local_layers = ck.schedule.n_visits - full.layers;
    const local: kv.groups.Group = .{ .kind = .window, .row = 2 * ck.text.local.kv * q.format.rowBytes(ck.text.local.dim), .layers = local_layers };
    layout.window = local;
    const request: kv.Budget = .{
        .room = room,
        .engine = 0,
        .lanes = q.parallel,
        .fixed = q.parallel_fixed,
        .window = @as(u64, r.context) + mini.native.window_rows,
        .engine_paged = true,
    };
    var pool = kv.plan(layout, request) catch |err| {
        r.refused_at = if (err == error.NoStream) .first_stream_state else .streams;
        r.message = if (err == error.NoStream) mini.native.explain(null, error.Inkling2CacheRoom) else try std.fmt.allocPrint(a, "--parallel {d} exceeds the room for Mini CUDA lanes beside one request at --context {d}", .{ q.parallel, r.context });
        return r;
    };
    if (state > state_room) {
        r.refused_at = .first_stream_state;
        r.message = mini.native.explain(null, error.Inkling2CacheRoom);
        return r;
    }
    const cache_limit = if (q.prompt_cache_gib) |v| std.math.lossyCast(u64, v * (1 << 30)) else std.math.maxInt(u64);
    const extras = @as(u64, pool.lanes - 1);
    const keep_pages = @min(cache_limit / layout.pageBytes(), pool.pages);
    const cache_entries = if (keep_pages > 0) q.cache_entries else 0;
    if (cache_entries > 0) pool.kept = keep_pages * layout.pageBytes();
    c.target_global_kv = pool.pages * layout.pageBytes();
    const extra = extras * stream;
    const ready = model + extra + c.target_global_kv;
    r.components = c;
    r.tracked = .{ .weights_check = weight_check, .first_stream_state = state, .model = model, .per_stream = stream, .extra_streams = extra, .ready = ready };
    r.rooms.after_streams = room - extra - c.target_global_kv;
    r.paging = .{ .layout = layout, .pool = pool, .ready_bytes = ready, .cache_entries = cache_entries };
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
    var selected = q;
    var probe = q;
    probe.context = 1;
    const base = try admit(scratch.allocator(), ck, before, weight_check, probe, 1);
    if (base.paging) |pg| selected.parallel = pg.pool.lanes;
    selected.parallel_fixed = true;
    var lo: u32 = 0;
    var hi = cap;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2 + 1;
        const r = try admit(scratch.allocator(), ck, before, weight_check, selected, mid);
        if (r.refused_at == null) lo = mid else hi = mid - 1;
    }
    return .{
        .model_cap = cap,
        .drafts = drafter != null,
        .drafter = drafter,
        .maximum = lo,
        .rounded_down = lo / 1024 * 1024,
        .model_cap_binds = lo == cap,
        .result = try admit(a, ck, before, weight_check, selected, @max(lo, 1)),
        .next_context = if (lo > 0 and lo < cap) try admit(a, ck, before, weight_check, selected, lo + 1) else null,
    };
}

/// Mini's model_max_length is in its text config. Resolve it using the same parser as the planner and loader.
pub fn modelContext(a: Allocator, io: std.Io, dir: []const u8, problem: *[]const u8) !?i64 {
    var why: mini.capacity.ConfigWhy = .{};
    const c = mini.native.readConfig(a, io, dir, &why) catch |err| {
        if (err == error.OutOfMemory) return err;
        problem.* = if (why.text().len > 0) try a.dupe(u8, why.text()) else @errorName(err);
        return null;
    };
    return c.text.max_context;
}

pub fn plan(a: Allocator, io: std.Io, dir: []const u8, before: budget.Pool, q: Query, problem: *[]const u8) !?Report {
    var why: @import("inkling2").capacity.ConfigWhy = .{};
    const ck = mini.capacity.Checkpoint.read(a, io, dir, q.drafts, q.drafter, q.vision, q.capability, &why) catch |err| {
        if (err == error.OutOfMemory) return err;
        problem.* = if (why.text().len > 0) try a.dupe(u8, why.text()) else mini.native.explain(null, err) orelse @errorName(err);
        return null;
    };
    const weight_check = if (ck.exl3) ck.checkpoint_bytes + ck.vision_weights else budget.weightBytes(a, io, dir) catch |err| {
        if (err == error.OutOfMemory) return err;
        problem.* = try std.fmt.allocPrint(a, "cannot estimate checkpoint weight files ({s}); refusing CUDA admission", .{@errorName(err)});
        return null;
    };
    var selected = q;
    selected.cache_entries = @min(kv.most_snaps, try @import("core").kv_prompts.sharingDefault(q.cache_entries));
    return try search(a, &ck, before, weight_check, selected);
}

/// Set the exact same plan used by `plan --device-mib` on the native model opener.
fn runtime(a: Allocator, io: std.Io, o: api.Open, before: budget.Pool, capability: u32, opts: *mini.native.Options, problem: *[]const u8) !bool {
    if (o.lanes_fixed and o.lanes > max_streams) {
        problem.* = mini.native.explain(null, error.Inkling2TooManyStreams).?;
        return false;
    }
    const q: Query = .{
        .parallel = @min(o.lanes, max_streams),
        .parallel_fixed = o.lanes_fixed,
        .context = if (o.context != null) @intCast(opts.context) else null,
        .capability = capability,
        .format = if (opts.kv8) .fp8 else .bf16,
        .drafts = opts.drafts,
        .drafter = opts.drafter,
        .vision = opts.vision,
        .prompt_cache_gib = o.prompt_cache_gib,
        .cache_entries = if (o.keep) |n| @intCast(@max(n, 0)) else kv.most_snaps,
    };
    const r = try plan(a, io, o.dir, before, q, problem) orelse return false;
    opts.paging = r.result.paging orelse {
        problem.* = r.result.message.?;
        return false;
    };
    opts.context = r.result.context;
    return true;
}

pub const Startup = struct {
    a: Allocator,
    io: std.Io,
    request: api.Open,
    capability: u32,
    before: budget.Pool,
    problem: *[]const u8,

    pub fn admit(s: *Startup, opts: *mini.native.Options) !bool {
        opts.replan = .{ .ctx = s, .run = replan };
        return runtime(s.a, s.io, s.request, s.before, s.capability, opts, s.problem);
    }

    fn replan(ctx: *anyopaque, opts: *mini.native.Options, available: u64) anyerror!bool {
        const s: *Startup = @ptrCast(@alignCast(ctx));
        var limited = s.before;
        // The live check counted only CUDA allocations after the driver opened its context.
        limited.free = @min(limited.free, limited.reserve +| available);
        return runtime(s.a, s.io, s.request, limited, s.capability, opts, s.problem);
    }
};

pub fn retention(pg: mini.native.Paging) struct { pages: u64, bytes: u64, marks: usize } {
    return .{ .pages = pg.pool.kept / pg.layout.pageBytes(), .bytes = pg.pool.kept, .marks = pg.cache_entries };
}

pub fn reportStartup(paging: ?mini.native.Paging, model: u64, room: u64, reserve: u64, stream: u64, context: i64) void {
    std.log.info("Mini CUDA startup: model {d} bytes, room after reserve {d} bytes, reserve {d} bytes, stream {d} bytes", .{ model, room, reserve, stream });
    if (paging) |pg| std.log.info("Mini CUDA KV: {d} pool tokens, {d} per request, {d} fixed bytes per stream, {d} snapshot slots, {d} allocated bytes", .{
        pg.pool.pages * pg.layout.page_tokens, context, pg.layout.fixed_bytes, pg.pool.snaps, pg.ready_bytes,
    });
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
                const q: Query = .{ .parallel = parallel, .cache_entries = 0 };
                const r = try search(a, &ck, pool, ck.checkpoint_bytes, q);
                try std.testing.expect(r.maximum.? > 0 and r.maximum.? < ck.text.max_context);
                try std.testing.expectEqual(r.maximum.?, r.result.context);
                try std.testing.expectEqual(null, r.result.refused_at);
                try std.testing.expectEqual(r.maximum.? + 1, r.next_context.?.context);
                const refused_at = r.next_context.?.refused_at.?;
                try std.testing.expect(refused_at == .first_stream_state or refused_at == .streams);
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
    const q: Query = .{ .parallel = 4, .context = 8704, .format = .fp8, .cache_entries = 0 };
    for ([_]struct { weights: u64, ready: u64 }{
        .{ .weights = 989_966_848, .ready = 17_102_791_408 },
        .{ .weights = 988_977_664, .ready = 17_101_802_224 },
        .{ .weights = 624_644_608, .ready = 16_737_469_168 },
        .{ .weights = 623_655_424, .ready = 16_736_479_984 },
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
        try std.testing.expectEqual(331_120_276, r.result.tracked.?.per_stream);
        try std.testing.expectEqual(35 * 4, r.result.paging.?.pool.pages);
        try std.testing.expectEqual(row.ready + 17_302_448, r.result.tracked.?.ready);
    }
    ck.draft_arena = 0;
    const off = try search(a, &ck, pool, ck.checkpoint_bytes, q);
    try std.testing.expect(!off.drafts and off.drafter == null);
    try std.testing.expectEqual(0, off.result.components.?.drafter_kv);
}

test "Mini capacity: prefix retention and caps leave the allocation unchanged" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var ck = try mini.capacity.testCheckpoint(a);
    ck.draft_kind = .dflash2;
    const pool: budget.Pool = .{ .total = 24 << 30, .free = 23 << 30, .reserve = 1 << 30, .limit = null, .unified = false };
    var q: Query = .{ .parallel = 4, .context = 8704 };
    const on = (try search(a, &ck, pool, ck.checkpoint_bytes, q)).result.paging.?;
    try std.testing.expectEqual(kv.most_snaps, on.cache_entries);
    try std.testing.expectEqual(@as(usize, 0), on.pool.snaps);
    try std.testing.expectEqual(@as(u64, 0), on.layout.snap_bytes);
    try std.testing.expectEqual(on.pool.pages * on.layout.pageBytes(), on.pool.kept);
    q.prompt_cache_gib = 0;
    const off = (try search(a, &ck, pool, ck.checkpoint_bytes, q)).result.paging.?;
    try std.testing.expectEqual(@as(usize, 0), off.cache_entries);
    try std.testing.expectEqual(@as(u64, 0), off.pool.kept);
    try std.testing.expectEqual(on.pool.pages, off.pool.pages);
    try std.testing.expectEqual(on.ready_bytes, off.ready_bytes);
    q.prompt_cache_gib = 0.2;
    const capped = (try search(a, &ck, pool, ck.checkpoint_bytes, q)).result.paging.?;
    try std.testing.expectEqual(on.cache_entries, capped.cache_entries);
    try std.testing.expectEqual(on.pool.pages, capped.pool.pages);
    try std.testing.expectEqual(on.ready_bytes, capped.ready_bytes);
    const cap_bytes = (1 << 30) / 5;
    try std.testing.expectEqual(cap_bytes / capped.layout.pageBytes(), retention(capped).pages);
    try std.testing.expectEqual(retention(capped).pages * capped.layout.pageBytes(), retention(capped).bytes);
    try std.testing.expectEqual(0, retention(off).marks);
    try std.testing.expectEqual(0, retention(off).pages);
}

test "Mini capacity: explicit zero requires the model window, omitted context fits what is available" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ck = try mini.capacity.testCheckpoint(a);
    const pool: budget.Pool = .{ .total = 32 << 30, .free = 31 << 30, .reserve = 4 << 30, .limit = null, .unified = false };
    const automatic = try search(a, &ck, pool, ck.checkpoint_bytes, .{ .parallel = 1 });
    try std.testing.expect(automatic.result.refused_at == null and automatic.result.context < ck.text.max_context);
    const full = try search(a, &ck, pool, ck.checkpoint_bytes, .{ .parallel = 1, .context = 0 });
    try std.testing.expectEqual(ck.text.max_context, full.result.context);
    try std.testing.expect(full.maximum == null and full.result.refused_at != null);
    const large: budget.Pool = .{ .total = 124 << 30, .free = 120 << 30, .reserve = 24 << 30, .limit = null, .unified = true };
    const admitted = try search(a, &ck, large, ck.checkpoint_bytes, .{ .parallel = 1, .context = 0 });
    try std.testing.expectEqual(ck.text.max_context, admitted.result.context);
    try std.testing.expect(admitted.maximum == null and admitted.result.refused_at == null);
}

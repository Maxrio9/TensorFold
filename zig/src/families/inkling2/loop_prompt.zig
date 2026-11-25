//! Compact non-paged snapshots and prompt-store hooks for Mini's shared global pages.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const engine = @import("loop_engine.zig");
const backend = @import("loop_backend.zig");
const page = @import("loop_pages.zig");
const draft = @import("loop_drafter.zig");
const kv = core.kv_pool;
const prompts = core.kv_prompts;

pub const State = struct {
    marks: []const u32 = &.{},
    kept: u64 = 0,
};

pub const Checkpoint = struct {
    fn drafter(b: *backend.Backend) ?*draft.Drafter {
        return if (b.m.capture_taps) &b.drafter.? else null;
    }

    pub fn bytes(b: *backend.Backend, at: u32) u64 {
        const draft_bytes = if (drafter(b)) |d| d.snapshotBytes(b.m.slot, at) else 0;
        return b.m.snapshotBytes(at) + draft_bytes;
    }

    pub fn copy(b: *backend.Backend, owner: ?*anyopaque, at: u32, ids: []const u32, size: u64, out: bool) !void {
        const l = try b.stateLane(owner);
        if (at == 0 or at > b.m.context or size != bytes(b, at)) return error.BadLoopSnapshot;
        if (out and (l.fill orelse l.len) != at) return error.NotAtMark;
        const d = drafter(b);
        const session = if (d) |active| active.session(b.m.slot) else null;
        if (session) |state| if (out and (state.end != at or state.pending != 0)) return error.DraftContextGap;

        var writer = page.StateCopy.init(b.m, b.s, ids, out);
        try b.m.eachState(at, &writer);
        if (d) |active| try active.eachSnapshot(b.m.slot, at, &writer);
        std.debug.assert(writer.offset == size);
        if (!out) {
            if (session) |state| {
                state.reset();
                state.end = at;
            }
            l.len = at;
        }
    }

    pub fn fence(b: *backend.Backend) void {
        cuda.completion.drain(b.d, b.s, "Mini prompt state copy failed before its pages could be released");
    }
};

const Query = struct {
    images: [100]prompts.Image = undefined,

    fn of(q: *Query, b: *const backend.Backend, s: *const lanes.Stream) prompts.Query {
        const images = q.images[0..s.images.len];
        for (s.images, images) |im, *id| id.* = .{ .at = im.at, .tokens = im.tokens, .gh = im.gh, .gw = im.gw, .hash = im.hash };
        return .{
            .key = if (s.key.len > 0) s.key else s.prompt(),
            .prompt_len = @intCast(s.prompt().len),
            .images = images,
            .namespace = @intFromBool(b.m.capture_taps),
            .salt = s.cache_salt,
        };
    }
};

pub fn open(b: *backend.Backend, s: *lanes.Stream, l: anytype) !usize {
    const p = &b.m.pages.?.pool;
    const store = b.prompts;
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(b.gpa);
    var q: Query = .{};
    const plan: prompts.Plan = if (store != null and !s.untaught and s.images.len <= q.images.len)
        try store.?.plan(b.gpa, p, q.of(b, s), s.history_len, s.shared_prefixes, s, &adopt)
    else
        .{};
    l.cache.marks = plan.marks;
    try p.adopt(&l.lease, adopt.items);
    if (plan.from != l.len) return error.KeptStateMismatch;
    if (plan.from == 0) try b.m.reset(b.s);
    var map: page.Map = .{ .m = b.m, .s = b.s, .slot = b.m.slot };
    if (adopt.items.len > 0) try map.table().write(&map, adopt.items, 0);
    try page.reserve(b.m, b.s, b.m.slot, &l.lease, page.need(s.prompt().len, s.max_new));
    s.cached = plan.from;
    return plan.from;
}

pub fn mark(b: *backend.Backend, s: *lanes.Stream, l: anytype, at: usize) !void {
    const store = b.prompts orelse return;
    if (s.untaught) return;
    if (std.mem.indexOfScalar(u32, l.cache.marks, @intCast(at)) != null) {
        var q: Query = .{};
        var map: page.Map = .{ .m = b.m, .s = b.s, .slot = b.m.slot };
        if (try store.keepPages(b.gpa, &b.m.pages.?.pool, &l.lease, q.of(b, s), at, s, page.page_tokens, map.table()))
            l.cache.kept += 1;
    }
    if (at == s.prompt().len) store.report(at, s.cached, l.cache.kept);
}

pub fn nextMark(marks: []const u32, at: usize, end: usize) usize {
    for (marks) |mark_at| if (mark_at > at) return mark_at;
    return end;
}

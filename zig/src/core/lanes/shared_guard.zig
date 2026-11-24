//! Admit shared drafts only when the estimated round pays for every stream's progress.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const win = @import("windows.zig");

const Prices = struct {
    streams: usize,
    sampled: bool,
    rounds: u64 = 0,
    plain_ms: ?f64 = null,
    extra_row_ms: ?f64 = null,
    head_ms: ?f64 = null,

    fn observe(p: *Prices, rows: u64, ms: f64) void {
        p.rounds +%= 1;
        if (!std.math.isFinite(ms) or ms <= 0) return;
        if (rows == p.streams) {
            blend(&p.plain_ms, ms);
        } else if (p.plain_ms) |plain| {
            blend(&p.extra_row_ms, @max(0, ms - plain) / @as(f64, @floatFromInt(rows - p.streams)));
        }
    }
};

pub const Guard = struct {
    prices: std.ArrayList(Prices) = .empty,

    pub fn deinit(g: *Guard, allocator: std.mem.Allocator) void {
        g.prices.deinit(allocator);
    }

    fn get(g: *Guard, allocator: std.mem.Allocator, streams: usize, sampled: bool) !*Prices {
        for (g.prices.items) |*p| if (p.streams == streams and p.sampled == sampled) return p;
        try g.prices.append(allocator, .{ .streams = streams, .sampled = sampled });
        return &g.prices.items[g.prices.items.len - 1];
    }

    pub fn observeHeads(g: *Guard, streams: usize, sampled: bool, depths: []const i64, ms: f64) void {
        var proposed: usize = 0;
        for (depths) |d| proposed += @intFromBool(d > 0);
        if (proposed == 0 or !std.math.isFinite(ms) or ms <= 0) return;
        for (g.prices.items) |*p| if (p.streams == streams and p.sampled == sampled) {
            blend(&p.head_ms, ms / @as(f64, @floatFromInt(proposed)));
            return;
        };
    }
};

fn blend(value: *?f64, ms: f64) void {
    value.* = if (value.*) |old| old + 0.15 * (ms - old) else ms;
}

pub fn enabled(e: *const Engine) bool {
    return e.cfg.protect_shared_streams and e.cfg.timed and e.cfg.draft_confidence == null and e.trim == null;
}

pub fn budgets(e: *Engine, plans: []const win.Plan, heads: []const usize, out: []i64, rows: u64, ms: f64) !void {
    @memset(out, 0);
    const prices = try e.shared_guard.get(e.gpa, plans.len, win.anySampled(plans));
    prices.observe(rows, ms);
    if (heads.len == 0 or prices.plain_ms == null) return;
    for (plans) |p| if (p.stream.finished or p.stream.paused or p.kind == .forced) return;

    const a = e.arena.allocator();
    const chances = try a.alloc([]const f64, plans.len);
    @memset(chances, &.{});
    for (heads) |i| {
        const s = plans[i].stream;
        const room: usize = @intCast(@max(0, s.draftRoom() - 1));
        const most = @min(e.cfg.most_drafts, room);
        if (most == 0 or s.force.items.len != 0) continue;
        chances[i] = try @import("allocate.zig").chainProbabilities(a, try e.rule.rates(win.who(s)), most);
    }
    const selected = try a.alloc(i64, plans.len);
    @memset(selected, 0);
    const room = e.cfg.batch_rows -| plans.len;
    const head_ms = @max(e.rule.sharedStepMs(), prices.head_ms orelse 0);
    if (prices.extra_row_ms) |row_ms| try choose(a, chances, room, prices.plain_ms.?, row_ms, head_ms, selected);
    // A sparse joint probe can reopen drafting after the streams' acceptance or the verify price changes.
    if (prices.rounds % 128 == 16 and std.mem.allEqual(i64, selected, 0)) {
        const width = @min(2, room / plans.len);
        for (chances) |p| if (p.len < width) return;
        @memset(selected, @intCast(width));
    }
    for (heads, out) |i, *d| {
        d.* = selected[i];
        plans[i].stream.shared_draft_count = plans.len;
    }
}

fn choose(a: std.mem.Allocator, chances: []const []const f64, room: usize, plain_ms: f64, row_ms: f64, head_ms: f64, out: []i64) !void {
    const taken = try a.alloc(i64, chances.len);
    defer a.free(taken);
    @memset(taken, 0);
    const tokens = try a.alloc(f64, chances.len);
    defer a.free(tokens);
    @memset(tokens, 1);
    var sum: f64 = @floatFromInt(chances.len);
    var heads: usize = 0;
    var best = sum / plain_ms;
    for (1..room + 1) |extra| {
        // Raise the least productive stream first. Among equal gains in progress, take the likeliest next token.
        var next: ?usize = null;
        for (chances, tokens, taken, 0..) |p, progress, n, i| {
            const at: usize = @intCast(n);
            if (at >= p.len or p[at] <= 0) continue;
            if (next) |j| {
                const other = chances[j][@intCast(taken[j])];
                if (progress > tokens[j] or (progress == tokens[j] and p[at] <= other)) continue;
            }
            next = i;
        }
        const pick = next orelse break;
        const chance = chances[pick][@intCast(taken[pick])];
        heads += @intFromBool(taken[pick] == 0);
        taken[pick] += 1;
        tokens[pick] += chance;
        sum += chance;
        const ms = plain_ms + row_ms * @as(f64, @floatFromInt(extra)) + head_ms * @as(f64, @floatFromInt(heads));
        const rate = sum / ms;
        if (rate <= best) continue;
        for (tokens) |expected| {
            if (expected * plain_ms < ms * 1.02) break;
        } else {
            best = rate;
            @memcpy(out, taken);
        }
    }
}

test "shared guard rejects an aggregate gain that slows a stream" {
    const strong = [_]f64{ 0.98, 0.9, 0.8, 0.7 };
    const weak = [_]f64{ 0.1, 0.04, 0.01 };
    var depths = [_]i64{ 0, 0 };
    try choose(std.testing.allocator, &.{ &strong, &weak }, 7, 20, 3, 4, &depths);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0 }, &depths);
    try choose(std.testing.allocator, &.{ &strong, &strong }, 7, 20, 3, 4, &depths);
    try std.testing.expect(depths[0] > 0 and depths[1] > 0);
}

//! Admit shared drafts only when the estimated round pays for every stream's progress.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const win = @import("windows.zig");
const Table = @import("table.zig").Table;

const Prices = struct {
    streams: usize,
    sampled: bool,
    rounds: u64 = 0,
    plain_ms: ?f64 = null,
    extra_row_ms: ?f64 = null,
    head_ms: ?f64 = null,
    head_samples: u64 = 0,
    block_costs: ?Table = null,

    fn observe(p: *Prices, gpa: std.mem.Allocator, rows: u64, ms: f64) !void {
        p.rounds +%= 1;
        if (!std.math.isFinite(ms) or ms <= 0) return;
        if (rows == p.streams) {
            blend(&p.plain_ms, ms);
        } else if (p.plain_ms) |plain| {
            const extra = rows - p.streams;
            if (p.block_costs) |*costs| {
                // First use may capture a graph. Measure replay from the second occurrence on.
                const key: i64 = @intCast(extra);
                const seen = costs.get(key);
                var ratio = if (seen == 0) null else seen;
                blend(&ratio, @max(1, ms / plain));
                try costs.put(gpa, key, if (seen == null) 0 else ratio.?);
            }
            blend(&p.extra_row_ms, @max(0, ms - plain) / @as(f64, @floatFromInt(extra)));
        }
    }

    fn observeHead(p: *Prices, ms: f64) void {
        p.head_samples +|= 1;
        if (p.block_costs != null and p.head_samples <= 2) {
            if (p.head_samples == 2) p.head_ms = ms;
            return;
        }
        blend(&p.head_ms, ms);
    }
};

pub const Guard = struct {
    prices: std.ArrayList(Prices) = .empty,

    pub fn deinit(g: *Guard, allocator: std.mem.Allocator) void {
        for (g.prices.items) |*p| if (p.block_costs) |*costs| costs.deinit(allocator);
        g.prices.deinit(allocator);
    }

    fn get(g: *Guard, allocator: std.mem.Allocator, streams: usize, sampled: bool, block: bool) !*Prices {
        for (g.prices.items) |*p| if (p.streams == streams and p.sampled == sampled) return p;
        try g.prices.append(allocator, .{ .streams = streams, .sampled = sampled, .block_costs = if (block) .{} else null });
        return &g.prices.items[g.prices.items.len - 1];
    }

    pub fn observeHeads(g: *Guard, streams: usize, sampled: bool, depths: []const i64, ms: f64) void {
        var proposed: usize = 0;
        for (depths) |d| proposed += @intFromBool(d > 0);
        if (proposed == 0 or !std.math.isFinite(ms) or ms <= 0) return;
        for (g.prices.items) |*p| if (p.streams == streams and p.sampled == sampled) {
            p.observeHead(ms / @as(f64, @floatFromInt(proposed)));
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
    const block = e.rule.k.fixed_draft_block and plans.len <= e.cfg.shared_block_streams;
    const prices = try e.shared_guard.get(e.gpa, plans.len, win.anySampled(plans), block);
    try prices.observe(e.gpa, rows, ms);
    if (heads.len == 0 or prices.plain_ms == null) return;
    for (plans) |p| if (p.stream.finished or p.stream.paused or p.kind == .forced) return;
    if (block and prices.rounds % 64 == 63) return; // refresh the plain price during sustained drafting

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
    const load_head = e.rule.sharedStepMs();
    const head_ms = if (block) prices.head_ms orelse load_head else @max(load_head, prices.head_ms orelse 0);
    const costs = prices.block_costs orelse Table{};
    const row_ms = prices.extra_row_ms orelse 0;
    if (prices.extra_row_ms != null and (!block or hasMeasured(costs))) {
        try choose(a, chances, room, prices.plain_ms.?, row_ms, head_ms, costs, selected);
    }
    // Adjacent fixed-block probes let an unseen width warm up and replay once.
    const phase = prices.rounds % 128;
    if ((phase == 16 or (block and phase == 17)) and (block or std.mem.allEqual(i64, selected, 0))) {
        const probe: usize = if (block and (prices.rounds / 128) % 2 == 1) 4 else 2;
        const width = @min(probe, room / plans.len);
        for (chances) |p| {
            if (p.len < width) break;
        } else @memset(selected, @intCast(width));
    }
    for (heads, out) |i, *d| {
        d.* = selected[i];
        plans[i].stream.shared_draft_count = plans.len;
    }
}

fn hasMeasured(costs: Table) bool {
    for (costs.slots) |v| if ((v orelse 0) > 0) return true;
    return false;
}

fn forwardMs(costs: Table, extra: usize, plain_ms: f64, row_ms: f64) f64 {
    var lo: usize = 0;
    var low_ms = plain_ms;
    for (costs.slots, 0..) |v, i| {
        const ratio = v orelse continue;
        if (ratio == 0) continue;
        const high_ms = @max(low_ms, ratio * plain_ms);
        if (i >= extra) {
            const fraction = @as(f64, @floatFromInt(extra - lo)) / @as(f64, @floatFromInt(i - lo));
            return low_ms + (high_ms - low_ms) * fraction;
        }
        lo = i;
        low_ms = high_ms;
    }
    return low_ms + row_ms * @as(f64, @floatFromInt(extra - lo));
}

fn choose(a: std.mem.Allocator, chances: []const []const f64, room: usize, plain_ms: f64, row_ms: f64, head_ms: f64, costs: Table, out: []i64) !void {
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
        const ms = forwardMs(costs, extra, plain_ms, row_ms) + head_ms * @as(f64, @floatFromInt(heads));
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
    try choose(std.testing.allocator, &.{ &strong, &weak }, 7, 20, 3, 4, .{}, &depths);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0 }, &depths);
    try choose(std.testing.allocator, &.{ &strong, &strong }, 7, 20, 3, 4, .{}, &depths);
    try std.testing.expect(depths[0] > 0 and depths[1] > 0);
}

test "fixed block pricing learns replay cost by width and retains the per-stream guard" {
    const a = std.testing.allocator;
    var prices: Prices = .{ .streams = 2, .sampled = false, .block_costs = .{} };
    defer prices.block_costs.?.deinit(a);
    try prices.observe(a, 2, 20);
    try prices.observe(a, 6, 100);
    try std.testing.expect(!hasMeasured(prices.block_costs.?));
    try prices.observe(a, 6, 25);
    try std.testing.expectEqual(@as(f64, 22.5), forwardMs(prices.block_costs.?, 2, 20, 7));
    prices.observeHead(90);
    prices.observeHead(4);
    try std.testing.expectEqual(@as(?f64, 4), prices.head_ms);
    const strong = [_]f64{ 0.95, 0.8, 0.7 };
    const weak = [_]f64{ 0.1, 0.04, 0.01 };
    var depths: [2]i64 = @splat(0);
    try choose(a, &.{ &strong, &strong }, 6, 20, 7, 4, prices.block_costs.?, &depths);
    try std.testing.expect(depths[0] > 0 and depths[1] > 0);
    depths = @splat(0);
    try choose(a, &.{ &strong, &weak }, 6, 20, 7, 4, prices.block_costs.?, &depths);
    try std.testing.expectEqualSlices(i64, &.{ 0, 0 }, &depths);
}

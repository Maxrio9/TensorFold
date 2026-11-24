//! Teacher-forced Mini verifier fidelity, using the served graph forward and no-draft target as the oracle.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const check = @import("check.zig");

const max_width = 8;
const Row = struct {
    id: []const u8,
    prompt_ids: []const u32,
    generated_ids: []const u32,
    compared_token: ?u32 = null,
};

const Score = struct {
    n: usize = 0,
    hits: usize = 0,
    kl: f64 = 0,
    max_kl: f64 = 0,
    gap: f32 = 0,

    fn add(a: *Score, p: []const f32, q: []const f32) !void {
        var pi: usize = 0;
        var qi: usize = 0;
        for (p, q, 0..) |x, y, i| {
            if (!std.math.isFinite(x) or !std.math.isFinite(y)) return error.NonfiniteLogit;
            if (x > p[pi]) pi = i;
            if (y > q[qi]) qi = i;
        }
        var ps: f64 = 0;
        var qs: f64 = 0;
        var delta: f64 = 0;
        for (p, q) |x, y| {
            const lp = @as(f64, x) - p[pi];
            const lq = @as(f64, y) - q[qi];
            const prob = @exp(lp);
            ps += prob;
            qs += @exp(lq);
            delta += prob * (lp - lq);
        }
        const kl = @max(0, delta / ps + @log(qs / ps));
        a.n += 1;
        a.hits += @intFromBool(pi == qi);
        a.kl += kl;
        a.max_kl = @max(a.max_kl, kl);
        a.gap = @max(a.gap, p[pi] - p[qi]);
    }
};

fn fill(o: *ink.native.Owned, prompt: []const u32, last: bool) !void {
    try o.model.reset(o.stream);
    var at: usize = 0;
    while (at < prompt.len) {
        const end = @min(at + o.model.promptRows(), prompt.len);
        try o.model.forward(&o.kernels, o.stream, prompt[at..end], at, end - at - 1, @intFromBool(last and end == prompt.len), false);
        at = end;
    }
}

/// Export the last prompt position's complete target distribution for cross-build comparisons.
pub fn prefillLogits(gpu: check.Gpu, model: []const u8, fixture: []const u8, output: []const u8) !void {
    const a = gpu.gpa;
    const data = try std.Io.Dir.cwd().readFileAlloc(gpu.io, fixture, a, .limited(1 << 22));
    defer a.free(data);
    const rows = try std.json.parseFromSlice([]const Row, a, data, .{ .ignore_unknown_fields = true });
    defer rows.deinit();
    var context: usize = 0;
    for (rows.value) |row| {
        if (row.prompt_ids.len == 0) return error.BadTeacherFixture;
        context = @max(context, row.prompt_ids.len);
    }
    if (context == 0) return error.BadTeacherFixture;
    const loaded = try ink.native.open(a, gpu.io, gpu.ctx, model, null, .{ .context = context, .drafts = false, .kv8 = true });
    defer loaded.deinit(loaded.ctx);
    const o = loaded.target;
    const vocab = o.weights.c.text.unpadded_vocab;
    const logits = try a.alloc(f32, vocab);
    defer a.free(logits);
    const live: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = o.model.s.logits, .len = logits.len * @sizeOf(f32) };
    const f = try std.Io.Dir.cwd().createFile(gpu.io, output, .{});
    defer f.close(gpu.io);
    var buffer: [4096]u8 = undefined;
    var writer = f.writer(gpu.io, &buffer);
    try writer.interface.writeInt(u32, vocab, .little);
    for (rows.value) |row| {
        try fill(o, row.prompt_ids, true);
        try o.stream.synchronize();
        try live.download(0, std.mem.sliceAsBytes(logits));
        try writer.interface.writeAll(std.mem.sliceAsBytes(logits));
        std.debug.print("PREFILL LOGITS {s} rows={d} vocab={d}\n", .{ row.id, row.prompt_ids.len, vocab });
    }
    try writer.interface.flush();
}

pub fn run(gpu: check.Gpu, model: []const u8, drafter: []const u8, fixture: []const u8) !void {
    const a = gpu.gpa;
    const data = try std.Io.Dir.cwd().readFileAlloc(gpu.io, fixture, a, .limited(1 << 22));
    defer a.free(data);
    const fixtures = try std.json.parseFromSlice([]const Row, a, data, .{ .ignore_unknown_fields = true });
    defer fixtures.deinit();
    const loaded = try ink.native.open(a, gpu.io, gpu.ctx, model, null, .{ .context = 8704, .drafts = true, .drafter = drafter });
    defer loaded.deinit(loaded.ctx);
    const o = loaded.target;
    if (o.backend.graphs) |*g| std.debug.print("VERIFY GRAPHS AT LOAD held={d} captured={d} eager={d}\n", .{
        g.cache.count(), g.cache.stats.captured, g.cache.stats.eager,
    });
    var caches: [8]cuda.graph_cache.Snapshot = undefined;
    const n_caches = cuda.graph_cache.snapshots(&caches);
    for (caches[0..n_caches]) |*cache| {
        std.debug.print("VERIFY CUDA CACHE {s} held={d} captured={d} replayed={d} eager={d}\n", .{
            cache.name(), cache.counts.held, cache.counts.captured, cache.counts.replayed, cache.counts.eager,
        });
    }
    const v = o.weights.c.text.unpadded_vocab;
    const live: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = o.model.s.logits, .len = max_width * v * 4 };
    const window = try a.alloc(f32, max_width * v);
    defer a.free(window);
    var scores: [max_width]Score = @splat(.{});
    for (fixtures.value) |row| {
        if (row.prompt_ids.len == 0 or row.prompt_ids.len + row.generated_ids.len > o.model.context) return error.BadTeacherFixture;
        if (row.compared_token) |token| {
            if (token >= v) return error.BadTeacherFixture;
            try fill(o, row.prompt_ids, true);
            for (row.generated_ids, 0..) |t, i| try o.backend.decode(&.{t}, row.prompt_ids.len + i, false);
            try o.stream.synchronize();
            const logits = window[0..v];
            try live.download(0, std.mem.sliceAsBytes(logits));
            const high = std.mem.max(f32, logits);
            var more: usize = 0;
            for (logits) |value| more += @intFromBool(value > logits[token]);
            std.debug.print("GREEDY GAP {s} rank={d} margin={d:.6}\n", .{ row.id, 1 + more, high - logits[token] });
            continue;
        }
        const tokens = row.generated_ids;
        if (tokens.len < max_width) return error.BadTeacherFixture;
        try fill(o, row.prompt_ids, false);
        const saved = try o.model.save(o.stream, @intCast(row.prompt_ids.len));
        defer o.model.drop(o.stream, saved);
        const plain = try a.alloc(f32, tokens.len * v);
        defer a.free(plain);
        for (tokens, 0..) |t, i| {
            try o.model.forward(&o.kernels, o.stream, &.{t}, row.prompt_ids.len + i, 0, 1, false);
            try o.stream.synchronize();
            try live.download(0, std.mem.sliceAsBytes(plain[i * v ..][0..v]));
        }
        for (&scores, 1..) |*total, width| {
            try o.model.restore(o.stream, saved);
            o.model.capture_taps = true;
            defer o.model.capture_taps = false;
            var per: Score = .{};
            var i: usize = 0;
            while (i + width <= tokens.len) {
                try o.backend.decode(tokens[i..][0..width], row.prompt_ids.len + i, width > 1);
                try o.stream.synchronize();
                try live.download(0, std.mem.sliceAsBytes(window[0 .. width * v]));
                for (0..width) |j| try per.add(plain[(i + j) * v ..][0..v], window[j * v ..][0..v]);
                // A partially accepted chain rolls the convolution histories back before the next verify.
                const kept = if (width > 1 and (i / width) % 3 == 0) width - 1 else width;
                if (kept < width) try o.model.rewind(&o.kernels, o.stream, kept);
                i += kept;
            }
            total.n += per.n;
            total.hits += per.hits;
            total.kl += per.kl;
            total.max_kl = @max(total.max_kl, per.max_kl);
            total.gap = @max(total.gap, per.gap);
            print(row.id, width, per);
        }
    }
    for (scores, 1..) |score, width| if (score.n != 0) print("TOTAL", width, score);
    if (o.backend.graphs) |*g| std.debug.print("VERIFY GRAPHS AFTER held={d} replayed={d} eager={d}\n", .{
        g.cache.count(), g.cache.stats.replayed, g.cache.stats.eager,
    });
}

/// Compare full-vocabulary next-token distributions from both cache formats on the same history.
pub fn cacheFidelity(gpu: check.Gpu, model: []const u8, fixture: []const u8) !void {
    const a = gpu.gpa;
    const data = try std.Io.Dir.cwd().readFileAlloc(gpu.io, fixture, a, .limited(1 << 22));
    defer a.free(data);
    const fixtures = try std.json.parseFromSlice([]const Row, a, data, .{ .ignore_unknown_fields = true });
    defer fixtures.deinit();
    var context: usize = 0;
    var positions: usize = 0;
    for (fixtures.value) |row| {
        if (row.prompt_ids.len == 0 or row.generated_ids.len == 0) return error.BadTeacherFixture;
        context = @max(context, row.prompt_ids.len + row.generated_ids.len);
        positions += row.generated_ids.len;
    }
    if (positions == 0) return error.BadTeacherFixture;

    var reference: []f32 = &.{};
    defer a.free(reference);
    var candidate: []f32 = &.{};
    defer a.free(candidate);
    for ([_]bool{ false, true }) |kv8| {
        const loaded = try ink.native.open(a, gpu.io, gpu.ctx, model, null, .{ .context = context, .drafts = false, .kv8 = kv8 });
        defer loaded.deinit(loaded.ctx);
        const o = loaded.target;
        const v = o.weights.c.text.unpadded_vocab;
        const live: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = o.model.s.logits, .len = v * 4 };
        if (!kv8) reference = try a.alloc(f32, positions * v) else candidate = try a.alloc(f32, v);
        var total: Score = .{};
        var pos: usize = 0;
        for (fixtures.value) |row| {
            try fill(o, row.prompt_ids, true);
            var per: Score = .{};
            for (row.generated_ids, 0..) |token, i| {
                const expected = reference[pos * v ..][0..v];
                try o.stream.synchronize();
                try live.download(0, std.mem.sliceAsBytes(if (kv8) candidate else expected));
                if (kv8) try per.add(expected, candidate);
                pos += 1;
                if (i + 1 < row.generated_ids.len) try o.backend.decode(&.{token}, row.prompt_ids.len + i, false);
            }
            if (kv8) {
                print(row.id, 1, per);
                total.n += per.n;
                total.hits += per.hits;
                total.kl += per.kl;
                total.max_kl = @max(total.max_kl, per.max_kl);
                total.gap = @max(total.gap, per.gap);
            }
        }
        if (kv8) print("FP8 vs BF16 TOTAL", 1, total);
    }
}

fn print(name: []const u8, width: usize, score: Score) void {
    std.debug.print("VERIFY {s} width={d} top1={d}/{d} mean_kl={e:.7} max_kl={e:.7} max_mismatch_gap={d:.5}\n", .{
        name, width, score.hits, score.n, score.kl / @as(f64, @floatFromInt(score.n)), score.max_kl, score.gap,
    });
}

/// Compare one-token FP8 solo logits with four shared chains, including mixed widths and partial commits.
pub fn concurrent(gpu: check.Gpu, model: []const u8, fixture: []const u8) !void {
    const a = gpu.gpa;
    const data = try std.Io.Dir.cwd().readFileAlloc(gpu.io, fixture, a, .limited(1 << 22));
    defer a.free(data);
    const fixtures = try std.json.parseFromSlice([]const Row, a, data, .{ .ignore_unknown_fields = true });
    defer fixtures.deinit();
    const count = 4;
    const steps = 64;
    if (fixtures.value.len != count) return error.BadTeacherFixture;
    const loaded = try ink.native.open(a, gpu.io, gpu.ctx, model, null, .{ .context = 8704, .drafts = true });
    defer loaded.deinit(loaded.ctx);
    try loaded.reserve(loaded.ctx, count);
    const o = loaded.target;
    const v = o.weights.c.text.unpadded_vocab;
    const live: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = o.model.s.logits, .len = count * max_width * v * 4 };
    const solo = try a.alloc(f32, count * steps * v);
    defer a.free(solo);
    const batch = try a.alloc(f32, count * max_width * v);
    defer a.free(batch);
    const rows = fixtures.value;
    const pinned = [count]usize{ 3, 0, 2, 1 };
    for (rows, 0..) |row, i| {
        if (row.prompt_ids.len == 0 or row.generated_ids.len < steps or row.prompt_ids.len + steps > o.model.context)
            return error.BadTeacherFixture;
        o.model.slot = 0;
        try fill(o, row.prompt_ids, false);
        for (row.generated_ids[0..steps], 0..) |token, j| {
            try o.backend.decode(&.{token}, row.prompt_ids.len + j, false);
            try o.stream.synchronize();
            try live.download(0, std.mem.sliceAsBytes(solo[(i * steps + j) * v ..][0..v]));
        }
    }
    const Chain = @TypeOf(o.model).Chain;
    const width_sets = [_][count]usize{ .{ 1, 1, 1, 1 }, .{ 1, 2, 4, 8 }, .{ 3, 5, 6, 7 }, .{ 8, 1, 1, 1 } };
    for ([_]bool{ false, true }) |mixed| {
        for (rows, pinned) |row, slot| {
            o.model.slot = slot;
            try fill(o, row.prompt_ids, false);
        }
        var progress: [count]usize = @splat(0);
        var scores: [count]Score = @splat(.{});
        var round_no: usize = 0;
        while (std.mem.min(usize, &progress) < steps) : (round_no += 1) {
            var chains: [count]Chain = undefined;
            var map: [count]usize = undefined;
            var active: usize = 0;
            var total: usize = 0;
            const widths = if (mixed) width_sets[round_no % width_sets.len] else width_sets[0];
            for (rows, pinned, widths, progress, 0..) |row, slot, width, at, i| {
                if (at == steps) continue;
                const n = @min(width, steps - at);
                chains[active] = .{ .slot = slot, .position = row.prompt_ids.len + at, .tokens = row.generated_ids[at..][0..n], .record = n > 1 };
                map[active] = i;
                active += 1;
                total += n;
            }
            try o.backend.decodeRound(chains[0..active], total, mixed);
            try o.stream.synchronize();
            try live.download(0, std.mem.sliceAsBytes(batch[0 .. total * v]));
            var offset: usize = 0;
            for (chains[0..active], map[0..active]) |chain, i| {
                for (0..chain.tokens.len) |j| {
                    const ref = (i * steps + progress[i] + j) * v;
                    const got = (offset + j) * v;
                    try scores[i].add(solo[ref..][0..v], batch[got..][0..v]);
                }
                offset += chain.tokens.len;
                const kept = chain.tokens.len - @intFromBool(chain.record and (round_no + i) % 2 == 0);
                if (kept < chain.tokens.len) {
                    o.model.slot = chain.slot;
                    try o.model.rewind(&o.kernels, o.stream, kept);
                }
                progress[i] += kept;
            }
        }
        for (scores, rows) |score, row| {
            const mean = score.kl / @as(f64, @floatFromInt(score.n));
            std.debug.print("SHARED mixed={} {s} top1={d}/{d} mean_kl={d:.8} max_kl={d:.8} max_top1_gap={d:.6}\n", .{
                mixed, row.id, score.hits, score.n, mean, score.max_kl, score.gap,
            });
            if (score.hits * 100 < score.n * 95 or mean > 0.01 or score.max_kl > 0.1 or score.gap > 0.125)
                return error.MiniConcurrentFidelity;
        }
    }
    if (o.backend.graphs) |*g| std.debug.print("SHARED GRAPHS captured={d} replays={d} eager={d}\n", .{
        g.cache.stats.captured, g.cache.stats.replayed, g.cache.stats.eager,
    });
}

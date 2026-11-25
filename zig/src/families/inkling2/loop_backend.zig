//! Inkling 2 Mini lane execution and snapshot adapter.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("loop_engine.zig");
const ops = @import("ops.zig");
const dspark = @import("dspark_cuda.zig");
const draft_runtime = @import("loop_drafter.zig");
const masked_draw = @import("loop_masked_draw.zig");

const Model = engine.Model;
const Kernels = engine.Kernels;
const window_rows = engine.window_rows;
const round = engine.round;
const token_ring = 1024;

const Lane = struct {
    len: usize = 0,
    fill: ?usize = null,
    base: usize = 0,
    rows: std.ArrayList(u32) = .empty,
    last: u64,
    rule: u64,
    sampled: bool = false,
};

/// The bytes a slot's lane buffers ask for: a vocabulary row of last logits and a sampling rule.
pub fn laneAllocs(vocab: u32) [2]usize {
    return .{ @as(usize, vocab) * 4, @sizeOf(cuda.sample.Rule) };
}

/// Shared CUDA token input, BF16 logits and metadata.
pub fn sharedAllocs(vocab: u32) [3]usize {
    return .{ (token_ring + engine.round_rows) * 4, window_rows * vocab * 2, 16 };
}

/// The device bytes each served stream holds: its model slot (engine.slotAllocs) and its lane buffers.
pub fn streamBytes(m: *const Model) u64 {
    const slot = engine.slotAllocs(&m.layout, m.w.p.n_visits);
    const lane = laneAllocs(m.w.c.text.unpadded_vocab);
    return slot[0] + slot[1] + slot[2] + slot[3] + lane[0] + lane[1];
}

pub const Backend = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    m: *Model,
    k: *Kernels,
    s: cuda.Stream,
    lanes_: [engine.max_slots]?Lane = @splat(null),
    streams: [engine.max_slots]?*lanes.Stream = @splat(null), // the stream each model slot holds
    tokens: cuda.DeviceBuffer,
    logits16: cuda.DeviceBuffer,
    meta: cuda.DeviceBuffer,
    slot: [engine.max_slots][2]cuda.DeviceBuffer = undefined, // each slot's last prompt logits and sampling rule
    next: u64 = 0,
    held: usize = 0, // slots with buffers

    graphs: ?@import("loop_graphs.zig").Graphs = null,
    drafter: ?draft_runtime.Drafter = null,
    vision: ?*@import("mini_vision.zig").Images = null,
    image_rows: [engine.max_slots]?cuda.DeviceBuffer = @splat(null),
    costs: [dspark.depth + 1]lanes.config.Cost = undefined,
    shared_costs: [dspark.depth + 1]lanes.config.Cost = undefined,
    propose_ms: f64 = 0,
    priced_width: usize = 0,
    shared_block_streams: u32 = 0,

    const be = lanes.backend;

    pub fn warm(b: *Backend, io: std.Io) !void {
        if (!b.m.w.c.exl3) return;
        const environment = @import("sys").environment;
        const separate = try environment.read(bool, "TF_INKLING2_VERIFY_GRAPHS", true);
        b.graphs = @as(@TypeOf(b.graphs.?), undefined);
        const g = &b.graphs.?;
        g.init(b.gpa, b.d, separate);
        errdefer {
            b.s.synchronize() catch {};
            g.deinit();
            b.graphs = null;
        }
        try g.warm(b.m, b.k, b.s, 1, false);
        const d = if (b.drafter) |*d| d else return;
        if (d.* == .dflash2) {
            const shared = if (try b.d.currentCapability() == 121) @as(u32, 4) else 2;
            b.shared_block_streams = try environment.read(u32, "TF_INKLING2_DFLASH2_SHARED", shared);
        }
        d.enableGraphs(b.gpa);
        b.m.taps = d.taps();
        b.m.capture_taps = true;
        defer b.m.capture_taps = false;
        const max = @min(b.costs.len, b.m.context);
        const zeros: [window_rows]u32 = @splat(0);
        for (1..max + 1) |width| {
            try g.warm(b.m, b.k, b.s, width, width > 1);
            try b.m.reset(b.s);
            try b.s.synchronize();
            const t0 = std.Io.Clock.awake.now(io);
            try b.decode(zeros[0..width], 0, width > 1);
            try b.s.synchronize();
            const verify_ms = elapsed(io, t0);
            var propose_ms: f64 = 0;
            if (width > 1) {
                const session = d.session(0);
                session.reset();
                try d.absorb(0, &b.k.glue, b.s, b.m.taps.?.buffer, 0, width, b.m.w.c.text.eps);
                try b.s.synchronize();
                if (d.* == .dflash2) try d.propose(0, b.m, b.k, b.s, 0, width - 1, null);
                const t1 = std.Io.Clock.awake.now(io);
                try d.propose(0, b.m, b.k, b.s, 0, width - 1, null);
                propose_ms = elapsed(io, t1);
            }
            b.costs[width - 1] = .{ .width = @intCast(width), .ms = verify_ms + propose_ms };
            b.shared_costs[width - 1] = .{ .width = @intCast(width), .ms = verify_ms };
            if (width == 2 or (d.* == .dflash2 and width == max)) b.propose_ms = propose_ms;
        }
        b.priced_width = max;
        d.session(0).reset();
        try b.m.reset(b.s);
        try b.s.synchronize();
    }

    fn elapsed(io: std.Io, t: std.Io.Timestamp) f64 {
        return @as(f64, @floatFromInt(t.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
    }

    pub fn decode(b: *Backend, tokens: []const u32, position: usize, record: bool) !void {
        try b.decodeOne(tokens, position, record, false);
    }

    fn decodeOne(b: *Backend, tokens: []const u32, position: usize, record: bool, lazy: bool) !void {
        const chain: round.Chain = .{ .slot = b.m.slot, .position = position, .tokens = tokens, .record = record };
        try b.decodeRoundMode(&.{chain}, tokens.len, b.m.capture_taps, lazy);
    }

    pub fn decodeRound(b: *Backend, chains: []const round.Chain, rows: usize, taps: bool) !void {
        try b.decodeRoundMode(chains, rows, taps, false);
    }

    fn decodeRoundMode(b: *Backend, chains: []const round.Chain, rows: usize, taps: bool, lazy: bool) !void {
        if (b.graphs) |*g| return g.forward(b.m, b.k, b.s, chains, rows, taps, lazy);
        try round.execute(b.m, b.k, b.s, try round.prepareDecode(b.m, b.s, chains, rows, taps, lazy));
    }

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, m: *Model, k: *Kernels, s: cuda.Stream) !Backend {
        const sizes = sharedAllocs(m.w.c.text.unpadded_vocab);
        var tokens = try cuda.DeviceBuffer.alloc(d, sizes[0]);
        errdefer tokens.free();
        var logits16 = try cuda.DeviceBuffer.alloc(d, sizes[1]);
        errdefer logits16.free();
        var meta = try cuda.DeviceBuffer.alloc(d, sizes[2]);
        errdefer meta.free();
        var b: Backend = .{ .gpa = gpa, .d = d, .m = m, .k = k, .s = s, .tokens = tokens, .logits16 = logits16, .meta = meta };
        try b.slotBuffers(m.n_slots);
        return b;
    }

    /// Each model slot's own last-logits and rule buffers, up to `n`.
    pub fn slotBuffers(b: *Backend, n: usize) !void {
        const sizes = laneAllocs(b.m.w.c.text.unpadded_vocab);
        for (0..n) |i| {
            if (i < b.held) continue;
            var last = try cuda.DeviceBuffer.alloc(b.d, sizes[0]);
            errdefer last.free();
            b.slot[i] = .{ last, try cuda.DeviceBuffer.alloc(b.d, sizes[1]) };
            b.held = i + 1;
        }
    }

    /// The server's stream count: one model slot and its buffers per stream.
    pub fn reserve(b: *Backend, streams: u32) !void {
        if (streams > engine.max_slots) return error.Inkling2TooManyStreams;
        try b.m.addSlots(b.s, streams);
        if (b.drafter) |*d| try d.reserve(streams);
        try b.slotBuffers(streams);
        if (b.vision) |v| for (b.image_rows[0..streams]) |*slot| {
            if (slot.* == null) slot.* = try cuda.DeviceBuffer.alloc(b.d, v.slot_bytes);
        };
    }

    pub fn streamBytesWithDraft(b: *const Backend) u64 {
        const draft_bytes = if (b.drafter) |d| std.meta.activeTag(d).sessionBytes(b.m.context) else 0;
        return streamBytes(b.m) + draft_bytes + if (b.vision) |v| v.slot_bytes else 0;
    }

    pub fn deinit(b: *Backend) void {
        for (&b.lanes_) |*x| if (x.*) |*l| l.rows.deinit(b.gpa);
        for (b.slot[0..b.held]) |*pair| for (pair) |*x| x.free();
        for (&b.image_rows) |*slot| if (slot.*) |*rows| rows.free();
        if (b.graphs) |*g| g.deinit();
        if (b.drafter) |*d| d.deinit();
        b.meta.free();
        b.logits16.free();
        b.tokens.free();
        b.* = undefined;
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &vtable, .draft_grammar = false };
    }

    pub fn facts(b: *const Backend) lanes.Model {
        const drafting = b.drafter != null;
        return .{
            .fixed_draft_block = if (b.drafter) |d| d == .dflash2 else false,
            .exact_width = window_rows,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = if (drafting) @intCast(b.priced_width -| 1) else 0,
            .plain_guard = drafting,
            .protect_shared_streams = drafting,
            .shared_block_streams = b.shared_block_streams,
            .window_costs = b.costs[0..b.priced_width],
            .window_costs_lone = drafting,
            .shared_costs = b.shared_costs[0..b.priced_width],
            .shared_step_ms = if (drafting) b.propose_ms else null,
            .learn_row_costs = drafting,
            .copy_drafts = !drafting,
            .hidden_rows = true,
            .plain_share = true,
            .batch_rows = engine.round_rows,
            .max_streams = engine.max_slots,
            .streams_exact = true,
        };
    }

    pub fn admits(b: *const Backend, prompt: usize, max_new: usize) bool {
        return b.freeSlot() != null and prompt <= b.m.context and max_new <= b.m.context - prompt;
    }

    fn freeSlot(b: *const Backend) ?usize {
        for (b.streams[0..@min(b.m.n_slots, b.held)], 0..) |x, i| if (x == null) return i;
        return null;
    }

    fn slotOf(b: *const Backend, s: *const lanes.Stream) ?usize {
        for (b.streams, 0..) |x, i| if (x == s) return i;
        return null;
    }

    /// The device buffer holding `s`'s last prompt logits.
    pub fn lastOf(b: *const Backend, s: *const lanes.Stream) !u64 {
        const i = b.slotOf(s) orelse return error.UnknownStream;
        return b.lanes_[i].?.last;
    }

    fn self(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }

    fn upload(b: *Backend, dst: u64, bytes: []const u8) !void {
        try cuda.copy_source.enqueue(b.d, dst, bytes, b.s);
    }

    /// `s`'s lane, its model slot selected for the calls that follow.
    fn laneOf(b: *Backend, s: *const lanes.Stream) !*Lane {
        const i = b.slotOf(s) orelse return error.UnknownStream;
        if (i >= b.m.n_slots) return error.UnknownStream;
        b.m.slot = i;
        b.m.capture_taps = s.drafts and b.drafter != null;
        return if (b.lanes_[i]) |*l| l else error.UnknownStream;
    }

    fn resumeSaved(b: *Backend, s: *lanes.Stream) usize {
        const saved = s.reuse.saved orelse return 0;
        const k: *const Model.Saved = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= s.prompt().len) {
            s.reuse_failed = true;
            return 0;
        }
        b.m.restore(b.s, k) catch {
            s.reuse_failed = true;
            return 0;
        };
        s.cached = k.at;
        return k.at;
    }

    fn opening(b: *Backend, s: *lanes.Stream) !*Lane {
        const prompt = s.prompt();
        if (prompt.len == 0) return error.EmptyPrompt;
        if (s.images.len > 0 and b.vision == null) return error.Inkling2LoopTextOnly;
        if (prompt.len > b.m.context or s.max_new > b.m.context - prompt.len) return error.PastContext;
        if (b.slotOf(s) == null) {
            const i = b.freeSlot() orelse return error.NoFreeSlot;
            b.lanes_[i] = .{ .last = b.slot[i][0].ptr, .rule = b.slot[i][1].ptr };
            b.streams[i] = s;
        }
        const l = try b.laneOf(s);
        l.sampled = false;
        if (cuda.sample.ruleFor(s.sampling)) |r| {
            try b.upload(l.rule, std.mem.asBytes(&r));
            l.sampled = true;
        }
        l.len = 0;
        try b.m.reset(b.s);
        const at = if (b.m.capture_taps) blk: {
            b.drafter.?.session(b.m.slot).reset();
            if (s.reuse.saved != null) s.reuse_failed = true; // rebuild the drafter's context along with the target
            break :blk 0;
        } else b.resumeSaved(s);
        if (s.images.len > 0) try b.vision.?.begin(b.s, s, at, b.image_rows[b.m.slot].?.ptr);
        l.len = at;
        l.fill = at;
        s.fill_at.store(@intCast(at), .monotonic);
        return l;
    }

    fn chunk(b: *Backend, s: *lanes.Stream, l: *Lane) !bool {
        const prompt = s.prompt();
        const at = l.fill.?;
        if (s.isCancelled()) return error.Cancelled;
        const rows_per_pass = b.m.promptRows();
        var end = @min(prompt.len, (at / rows_per_pass + 1) * rows_per_pass);
        for (s.reuse.marks) |mark| {
            if (mark > at and mark < end) end = mark;
        }
        const done = end == prompt.len;
        if (s.images.len == 0) {
            try b.m.forward(b.k, b.s, prompt[at..end], at, end - at - 1, @intFromBool(done), false);
        } else {
            const chain: round.Chain = .{ .slot = b.m.slot, .position = at, .tokens = prompt[at..end] };
            const shape = try round.prepare(b.m, b.s, &.{chain}, end - at - 1, @intFromBool(done), b.m.capture_taps);
            try round.executePrompt(b.m, b.k, b.s, shape, .{
                .buffer = b.image_rows[b.m.slot].?.ptr,
                .images = s.images,
                .position = at,
            });
        }
        if (b.m.capture_taps) {
            const d = &b.drafter.?;
            try d.prefill(b.m.slot, &b.k.glue, b.s, at, end - at, prompt.len, b.m.w.c.text.eps);
        }
        if (done) try b.d.check(b.d.api.cuMemcpyDtoDAsync_v2(l.last, b.m.s.logits, b.m.w.c.text.unpadded_vocab * 4, b.s.handle), "cuMemcpyDtoDAsync");
        l.len = end;
        l.fill = if (done) null else end;
        s.fill_at.store(@intCast(end), .monotonic);
        if (s.reuse.hook) |hook| if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(end)) != null)
            hook.at(hook.ptr, s, @intCast(end));
        return done;
    }

    fn handle(b: *Backend) usize {
        const h = b.next;
        b.next += 1;
        return @intCast(h % token_ring);
    }

    fn drawRows(b: *Backend, l: *const Lane, logits: u64, rows: usize, position0: u64, at: usize) !void {
        const v = b.m.w.c.text.unpadded_vocab;
        if (!l.sampled) return ops.top1(&b.k.glue, b.s, logits, rows, v, v, 0, b.m.s.t + at * 4, b.tokens.ptr + at * 4);
        try ops.tobf(&b.k.glue, b.s, logits, rows, v, v, b.logits16.ptr, v, 0, v);
        const metadata = [4]i32{ @intCast(position0 - 1), 0, 0, 0 };
        try b.upload(b.meta.ptr, std.mem.asBytes(&metadata));
        try b.k.draws.draw(b.s, b.logits16.ptr, v, l.rule, b.meta.ptr, 0, b.tokens.ptr + at * 4, rows, null, null);
    }

    fn read(ptr: *anyopaque, h: u64) anyerror!u32 {
        const b = self(ptr);
        if (h >= b.next or b.next - h > token_ring) return error.NoSuchToken;
        var out: [1]u32 = undefined;
        try b.s.synchronize();
        try b.tokens.download(@intCast((h % token_ring) * 4), std.mem.sliceAsBytes(&out));
        return out[0];
    }

    const vtable: be.Backend.VTable = .{
        .prefill = prefill,
        .prefill_step = prefillStep,
        .first = first,
        .first_masked = firstMasked,
        .queue = queue,
        .read = read,
        .verify = verify,
        .keep = keep,
        .draft = draft,
        .release = release,
        .reports_fill = true,
    };

    fn prefill(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const b = self(ptr);
        const l = try b.opening(s);
        errdefer release(ptr, s);
        while (!try b.chunk(s, l)) {}
    }

    fn prefillStep(ptr: *anyopaque, s: *lanes.Stream, _: u32) anyerror!bool {
        const b = self(ptr);
        const going = if (b.slotOf(s)) |i| b.lanes_[i].?.fill != null else false;
        const l = if (going) try b.laneOf(s) else try b.opening(s);
        errdefer release(ptr, s);
        return b.chunk(s, l);
    }

    fn first(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const b = self(ptr);
        const l = try b.laneOf(s);
        if (position != l.len or l.fill != null) return error.PositionMismatch;
        const h = b.next;
        try b.drawRows(l, l.last, 1, position, b.handle());
        return h;
    }

    fn firstMasked(ptr: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) anyerror!u64 {
        const b = self(ptr);
        const l = try b.laneOf(s);
        if (position != l.len or l.fill != null) return error.PositionMismatch;
        const h = b.next;
        const at = b.handle();
        try b.drawMasked(s, l.last, &.{position}, at, mask);
        return h;
    }

    fn drawMasked(b: *Backend, s: *const lanes.Stream, src: u64, positions: []const u64, at: usize, masks: []const u32) !void {
        var picked: [window_rows]u32 = undefined;
        try masked_draw.pick(b.gpa, b.s, b.m.w.c.text.unpadded_vocab, b.m.w.c.text.vocab, src, positions, s.sampling, masks, picked[0..positions.len]);
        try b.upload(b.tokens.ptr + at * 4, std.mem.sliceAsBytes(picked[0..positions.len]));
    }

    fn feed(b: *Backend, value: be.Feed) !u32 {
        return switch (value) {
            .value => |v| v,
            .handle => |h| read(b, h),
        };
    }

    fn queue(ptr: *anyopaque, s: *lanes.Stream, value: be.Feed, position: u64) anyerror!u64 {
        const b = self(ptr);
        const l = try b.laneOf(s);
        const token = try b.feed(value);
        if (position != l.len + 1) return error.PositionMismatch;
        try b.decodeOne(&.{token}, l.len, false, !l.sampled and s.logprobs == null and s.grammar == null);
        l.len += 1;
        const h = b.next;
        try b.drawRows(l, b.m.s.logits, 1, position, b.handle());
        return h;
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const b = self(ptr);
        if (windows.len == 0 or windows.len > engine.max_slots or out.len != windows.len) return error.BadWindows;
        var chains: [engine.max_slots]round.Chain = undefined;
        var rows: usize = 0;
        var taps = false;
        var lazy = true;
        for (windows, chains[0..windows.len]) |w, *chain| {
            if (w.parents != null) return error.ChainsOnly;
            const l = try b.laneOf(w.stream);
            const n = w.rows();
            if (n == 0 or n > window_rows) return error.WindowTooWide;
            l.base = l.len;
            l.rows.clearRetainingCapacity();
            try l.rows.append(b.gpa, w.pending);
            if (w.held > 0) {
                if (!b.m.capture_taps) return error.NoDraftHead;
                const session = b.drafter.?.session(b.m.slot);
                if (w.held > session.held_len) return error.NoHeldDrafts;
                try l.rows.appendSlice(b.gpa, session.held[0..w.held]);
            }
            try l.rows.appendSlice(b.gpa, w.tokens);
            for (w.positions, 0..) |position, i| if (position != l.base + i + 1) return error.PositionMismatch;
            chain.* = .{ .slot = b.m.slot, .position = l.base, .tokens = l.rows.items, .record = n > 1 };
            rows += n;
            taps = taps or b.m.capture_taps;
            lazy = lazy and !l.sampled and w.masks.len == 0 and w.stream.logprobs == null and w.stream.grammar == null;
        }
        try b.decodeRoundMode(chains[0..windows.len], rows, taps, lazy);
        const v: usize = b.m.w.c.text.unpadded_vocab;
        var row0: usize = 0;
        for (windows, chains[0..windows.len]) |w, chain| {
            const l = &b.lanes_[chain.slot].?;
            const n = chain.tokens.len;
            if (w.stream.drafts and b.drafter != null) {
                const d = &b.drafter.?;
                const from = b.m.taps.?.buffer + row0 * dspark.tap_stride;
                const session = d.session(chain.slot);
                try b.m.copy(b.s, session.taps + session.pending * dspark.tap_stride, from, n * dspark.tap_stride);
            }
            const src = b.m.s.logits + row0 * v * 4;
            if (w.masks.len == 0)
                try b.drawRows(l, src, n, w.positions[0], token_ring + row0)
            else
                try b.drawMasked(w.stream, src, w.positions, token_ring + row0, w.masks);
            row0 += n;
        }
        var got: [engine.round_rows]u32 = undefined;
        try b.s.synchronize();
        try b.tokens.download(token_ring * 4, std.mem.sliceAsBytes(got[0..rows]));
        row0 = 0;
        for (chains[0..windows.len], out) |chain, o| {
            const l = &b.lanes_[chain.slot].?;
            const n = chain.tokens.len;
            @memcpy(o.sampled[0..n], got[row0..][0..n]);
            @memcpy(o.drafts[0 .. n - 1], chain.tokens[1..]);
            l.len = l.base + n;
            row0 += n;
        }
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        if (windows.len != paths.len) return error.BadWindows;
        for (windows, paths) |w, path| {
            const l = try b.laneOf(w.stream);
            if (path.len == 0 or path.len > l.rows.items.len) return error.EmptyPath;
            for (path, 0..) |row, i| if (row != i) return error.ChainsOnly;
            if (path.len < l.rows.items.len) try b.m.rewind(b.k, b.s, path.len);
            l.len = l.base + path.len;
        }
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = self(ptr);
        for (requests) |r| try b.draftOne(r);
    }

    fn draftOne(b: *Backend, r: be.DraftRequest) !void {
        if (r.lanes != null) return error.ChainsOnly;
        const l = try b.laneOf(r.stream);
        const d = if (b.m.capture_taps) &b.drafter.? else {
            if (r.depth != 0) return error.NoDraftHead;
            return;
        };
        const session = d.session(b.m.slot);
        const pending = if (r.rows) |rows| blk: {
            if (r.start != l.base or rows.len == 0 or rows.len > l.len - l.base or rows.len > r.follow.len or r.position != r.start + rows.len + 1)
                return error.PositionMismatch;
            for (rows, 0..) |row, i| if (row != i) return error.ChainsOnly;
            if (session.end + session.pending != l.base) return error.DraftContextGap;
            session.pending += rows.len;
            session.held_len = 0;
            if (r.depth != 0 or session.pending >= window_rows or !r.stream.shared_round) {
                try d.absorb(b.m.slot, &b.k.glue, b.s, session.taps, session.end, session.pending, b.m.w.c.text.eps);
                session.pending = 0;
            }
            break :blk r.follow[rows.len - 1];
        } else blk: {
            if (r.position != l.len + 1) return error.PositionMismatch;
            if (r.depth == 0) return;
            break :blk try b.feed(r.first orelse return error.NoFirstToken);
        };
        if (r.depth != 0) try d.propose(b.m.slot, b.m, b.k, b.s, pending, r.depth, if (l.sampled) l.rule else null);
    }

    pub fn saveState(b: *Backend, owner: ?*anyopaque, at: u32) !*Model.Saved {
        const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = try b.laneOf(s);
        if ((l.fill orelse l.len) != at) return error.NotAtMark;
        return b.m.save(b.s, at);
    }

    pub fn restoreState(b: *Backend, owner: ?*anyopaque, saved: *const Model.Saved) !void {
        const s: *const lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
        const l = try b.laneOf(s);
        try b.m.restore(b.s, saved);
        l.len = saved.at;
    }

    pub fn dropState(b: *Backend, saved: *Model.Saved) void {
        b.m.drop(b.s, saved);
    }

    fn release(ptr: *anyopaque, s: *lanes.Stream) void {
        const b = self(ptr);
        const i = b.slotOf(s) orelse return;
        if (b.lanes_[i]) |*l| l.rows.deinit(b.gpa);
        b.lanes_[i] = null;
        b.streams[i] = null;
    }
};

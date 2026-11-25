//! Inkling 2 Mini's loop host layout: 48 private visit caches, whole-state snapshots and one-rank admission.
const std = @import("std");
const config = @import("config.zig");
const plan_ = @import("plan.zig");

pub fn validateLoad(c: *const config.Config, rank: usize, images: bool) !void {
    if (rank != 1) return error.Inkling2LoopOneRankOnly;
    if (images) return error.Inkling2LoopTextOnly;
    const lp = c.text.loop orelse return error.Inkling2LoopRequired;
    if (lp.gate_rank != 64) return error.UnsupportedLoopGateRank;
}

pub const modelopt_prompt_rows: usize = 2048;
pub const exl3_prompt_rows: usize = 4096;

pub fn promptRows(exl3: bool) usize {
    if (!exl3) return modelopt_prompt_rows;
    return @import("sys").environment.read(usize, "TF_INKLING2_PREFILL_ROWS", modelopt_prompt_rows) catch modelopt_prompt_rows;
}

pub const CacheFormat = enum {
    bf16,
    fp8,

    pub fn rowBytes(format: CacheFormat, dim: u32) u64 {
        return switch (format) {
            .bf16 => @as(u64, dim) * 2,
            .fp8 => @as(u64, dim) + @sizeOf(f32),
        };
    }
};

pub const SnapshotSlot = struct {
    kv_offset: u64,
    kv_bytes: u64,
    conv_offset: u64,
    conv_bytes: u64,
    global_offset: u64 = 0,
    global_row: u64 = 0,
};

/// A stream or a saved prompt owns the same complete layout: no visit shares KV or convolution state with another.
pub const CacheLayout = struct {
    slots: [config.max_visits]SnapshotSlot,
    n: u32,
    bytes: u64,
    format: CacheFormat,
    rows_per_pass: usize,

    pub fn init(c: *const config.Text, p: *const plan_.Plan, context: u32, format: CacheFormat, rows_per_pass: usize) !CacheLayout {
        return make(c, p, context, format, rows_per_pass, false);
    }

    pub fn initPaged(c: *const config.Text, p: *const plan_.Plan, context: u32, format: CacheFormat, rows_per_pass: usize) !CacheLayout {
        return make(c, p, context, format, rows_per_pass, true);
    }

    fn make(c: *const config.Text, p: *const plan_.Plan, context: u32, format: CacheFormat, rows_per_pass: usize, paged: bool) !CacheLayout {
        if (context == 0 or context > c.max_context or p.n_visits != c.visitCount() or
            rows_per_pass == 0 or rows_per_pass > exl3_prompt_rows) return error.BadLoopCache;
        var out: CacheLayout = undefined;
        out.n = p.n_visits;
        out.format = format;
        out.rows_per_pass = rows_per_pass;
        out.bytes = 0;
        var global_offset: u64 = 0;
        const conv_values = 3 * (2 * @as(u64, c.local.kv * c.local.dim) + 2 * @as(u64, c.hidden));
        for (p.visits[0..p.n_visits], 0..) |visit, i| {
            const layer = p.layers[visit.layer];
            const rows: u32 = @intCast(@min(rows_per_pass, context));
            const global = paged and layer.kind == .global;
            const tokens = if (layer.kind == .global)
                (if (global) 0 else std.mem.alignForward(u32, context, 64))
            else
                std.mem.alignForward(u32, c.window + rows - 1, 64);
            const row = 2 * layer.heads.kv * format.rowBytes(layer.heads.dim);
            const kv = try std.math.mul(u64, tokens, row);
            // The recurrence rounds each site then widens its complete state to FP32 before the FMAs.
            const conv = try std.math.mul(u64, conv_values, 4);
            out.slots[i] = .{
                .kv_offset = out.bytes,
                .kv_bytes = kv,
                .conv_offset = 0,
                .conv_bytes = conv,
                .global_offset = global_offset,
                .global_row = if (global) row else 0,
            };
            if (global) global_offset += row;
            out.bytes = try std.math.add(u64, out.bytes, kv);
        }
        var conv_offset = out.bytes;
        for (out.slots[0..out.n]) |*slot| {
            slot.conv_offset = conv_offset;
            conv_offset = try std.math.add(u64, conv_offset, slot.conv_bytes);
        }
        out.bytes = conv_offset;
        return out;
    }

    pub fn globalBytesPerToken(p: *const plan_.Plan, format: CacheFormat) u64 {
        var bytes: u64 = 0;
        for (p.visits[0..p.n_visits]) |visit| {
            const layer = p.layers[visit.layer];
            if (layer.kind == .global) bytes += 2 * layer.heads.kv * format.rowBytes(layer.heads.dim);
        }
        return bytes;
    }

    pub fn at(l: *const CacheLayout, visit: u32) SnapshotSlot {
        return l.slots[visit];
    }
};

/// The serial lane adapter's admission rule. One request occupies one complete 48-visit stream.
pub const Admission = struct {
    context: u32,

    pub fn fits(a: Admission, prompt: u32, max_tokens: u32) bool {
        return prompt <= a.context and max_tokens <= a.context - prompt;
    }
};

/// A prompt-cache save or restore copies every byte of this layout, including all four convolution recurrences.
pub fn snapshotBytes(l: *const CacheLayout) u64 {
    return l.bytes;
}

pub const Snapshot = struct {
    bytes: []u8,

    pub fn save(gpa: std.mem.Allocator, state: []const u8, l: *const CacheLayout) !Snapshot {
        if (state.len != l.bytes) return error.BadLoopSnapshot;
        return .{ .bytes = try gpa.dupe(u8, state) };
    }

    pub fn restore(s: *const Snapshot, state: []u8, l: *const CacheLayout) !void {
        if (s.bytes.len != l.bytes or state.len != l.bytes) return error.BadLoopSnapshot;
        @memcpy(state, s.bytes);
    }

    pub fn deinit(s: *Snapshot, gpa: std.mem.Allocator) void {
        gpa.free(s.bytes);
        s.* = undefined;
    }
};

test "a saved prompt restores every visit byte and owns its copy" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var why: config.Why = .{};
    const c = try config.parse(arena.allocator(), @import("config_test.zig").mini, &why);
    var prec: [24]plan_.Precision = @splat(.nvfp4);
    const p = try plan_.build(&c.text, &prec);
    const layout = try CacheLayout.init(&c.text, &p, 1, .bf16, modelopt_prompt_rows);
    const state = try std.testing.allocator.alloc(u8, @intCast(layout.bytes));
    defer std.testing.allocator.free(state);
    for (state, 0..) |*v, i| v.* = @truncate(i * 29 + 7);
    var saved = try Snapshot.save(std.testing.allocator, state, &layout);
    defer saved.deinit(std.testing.allocator);
    @memset(state, 0xa5);
    try saved.restore(state, &layout);
    try std.testing.expectEqualSlices(u8, saved.bytes, state);
    try std.testing.expectError(error.BadLoopSnapshot, Snapshot.save(std.testing.allocator, state[0 .. state.len - 1], &layout));
}

test "Mini has 48 independent KV and convolution slots and a whole-state snapshot" {
    const config_test = @import("config_test.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var why: config.Why = .{};
    const c = try config.parse(a, config_test.mini, &why);
    var prec: [24]plan_.Precision = @splat(.nvfp4);
    const p = try plan_.build(&c.text, &prec);
    const layout = try CacheLayout.init(&c.text, &p, 1_048_576, .bf16, modelopt_prompt_rows);
    try std.testing.expectEqual(@as(u32, 48), layout.n);
    try std.testing.expectEqual(@as(u64, 3072 * 2 * 8 * 128 * 2), layout.at(0).kv_bytes);
    try std.testing.expectEqual(@as(u64, 1_048_576 * 2 * 8 * 128 * 2), layout.at(5).kv_bytes);
    try std.testing.expectEqual(@as(u64, 34_866_888_704), layout.bytes);
    try std.testing.expectEqual(layout.bytes, snapshotBytes(&layout));
    try std.testing.expect((Admission{ .context = 8192 }).fits(8000, 192));
    try std.testing.expectEqual(@as(u64, 79_872), layout.at(0).conv_bytes);
}

test "Mini loop load is one-rank text-only with rank-64 gated controllers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var why: config.Why = .{};
    const c = try config.parse(arena.allocator(), @import("config_test.zig").mini, &why);
    try validateLoad(&c, 1, false);
    try std.testing.expectError(error.Inkling2LoopOneRankOnly, validateLoad(&c, 2, false));
    try std.testing.expectError(error.Inkling2LoopTextOnly, validateLoad(&c, 1, true));
}

test "pass two resets and only completed passes with a successor capture message banks" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var why: config.Why = .{};
    const c = try config.parse(arena.allocator(), @import("config_test.zig").mini, &why);
    var prec: [24]plan_.Precision = @splat(.nvfp4);
    const p = try plan_.build(&c.text, &prec);
    const lp = c.text.loop.?;
    var banks: u32 = 0;
    for (p.visits[0..p.n_visits]) |visit| {
        if (visit.pass == 2 and visit.layer == lp.core_start) try std.testing.expect(visit.hasMessages());
        if (visit.pass > 0 and visit.pass < lp.passes and visit.layer + 1 == lp.core_end) banks += 1;
    }
    try std.testing.expectEqual(lp.passes - 1, banks);
}

test "pass snapshots and rollback layout cover every visit" {
    const config_test = @import("config_test.zig");
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var why: config.Why = .{};
    const c = try config.parse(arena.allocator(), config_test.mini, &why);
    var precisions: [24]plan_.Precision = @splat(.nvfp4);
    const p = try plan_.build(&c.text, &precisions);
    const layout = try CacheLayout.init(&c.text, &p, 4096, .bf16, modelopt_prompt_rows);
    try std.testing.expectEqual(@as(u32, 48), layout.n);
    try std.testing.expect(layout.at(17).kv_offset != layout.at(29).kv_offset);
    try std.testing.expectEqual(@as(u64, layout.at(46).conv_offset + layout.at(46).conv_bytes), layout.at(47).conv_offset);
}

test "FP8 holds one E4M3 code per value and one FP32 scale per head row of every visit" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var why: config.Why = .{};
    const c = try config.parse(arena.allocator(), @import("config_test.zig").mini, &why);
    const p = try plan_.build(&c.text, &(@as([24]plan_.Precision, @splat(.nvfp4))));
    const bf16 = try CacheLayout.init(&c.text, &p, 8705, .bf16, modelopt_prompt_rows);
    const fp8 = try CacheLayout.init(&c.text, &p, 8705, .fp8, modelopt_prompt_rows);
    var end: u64 = 0;
    for (fp8.slots[0..fp8.n], bf16.slots[0..bf16.n]) |small, wide| {
        try std.testing.expectEqual(end, small.kv_offset);
        try std.testing.expectEqual(wide.kv_bytes / 256 * 132, small.kv_bytes);
        try std.testing.expectEqual(wide.conv_bytes, small.conv_bytes);
        end += small.kv_bytes;
    }
    try std.testing.expectEqual(end, fp8.at(0).conv_offset);
    try std.testing.expectEqual(@as(u64, 64 * 16896), (try CacheLayout.init(&c.text, &p, 8769, .fp8, modelopt_prompt_rows)).bytes - fp8.bytes);
}

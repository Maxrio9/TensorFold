//! Mini's two CUDA drafters share target taps, accepted-context bookkeeping and verification.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("loop_engine.zig");
const config = @import("config.zig");
const ops = @import("ops.zig");
const dspark = @import("dspark_cuda.zig");
const dspark_weights = @import("dspark_weights.zig");
const dflash = @import("dflash2_cuda.zig");
const dflash_weights = @import("dflash2_weights.zig");
const core = @import("core");
const fp8 = @import("draft_fp8.zig");

pub const Kind = enum {
    dspark,
    dflash2,

    pub fn name(kind: Kind) []const u8 {
        return switch (kind) {
            .dspark => "DSpark",
            .dflash2 => "DFlash2",
        };
    }

    pub fn sessionBytes(kind: Kind, context: usize) usize {
        return switch (kind) {
            .dspark => dspark.Session.bytes(context),
            .dflash2 => dflash.kvBytes(context) + dspark.Session.tap_bytes,
        };
    }
};

pub const Source = struct {
    kind: Kind,
    path: []const u8,
    owns_path: bool = false,

    pub fn deinit(src: Source, a: std.mem.Allocator) void {
        if (src.owns_path) a.free(src.path);
    }

    fn options(src: Source, capability: u32, borrow_head: bool) fp8.Options {
        if (capability < 120) return .{};
        return fp8.Options.fromEnv(borrow_head, src.kind == .dflash2);
    }

    pub fn checkpointBytes(src: Source, a: std.mem.Allocator, io: std.Io, c: *const config.Config, capability: u32, borrow_head: bool) !usize {
        const opts = src.options(capability, borrow_head);
        return switch (src.kind) {
            .dspark => dspark_weights.checkpointBytes(a, io, c, src.path, opts),
            .dflash2 => dflash_weights.checkpointBytes(a, io, src.path, c.text.unpadded_vocab, opts),
        };
    }

    pub fn preparedHeadBytes(src: Source, c: *const config.Config, capability: u32, borrow_head: bool) usize {
        const opts = src.options(capability, borrow_head);
        const folded = if (src.kind == .dspark) dspark_weights.preparedHeadBytes(opts.head) else 0;
        return folded + opts.baseBytes(c.text.unpadded_vocab, c.text.hidden);
    }

    pub fn load(src: Source, a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, c: *const config.Config, ck: *core.Checkpoint, s: cuda.Stream, unembed: u64, borrowed_head: ?cuda.qmmf.Weight) !Weights {
        std.debug.assert(s.d == d);
        const opts = src.options(try d.currentCapability(), borrowed_head != null);
        const base: fp8.BaseHead = .{ .stream = s, .source = unembed, .borrowed = borrowed_head };
        return switch (src.kind) {
            .dspark => blk: {
                const tensor = try ck.expect("model.llm.unembed.weight", .bf16, &.{ c.text.vocab, c.text.hidden });
                break :blk .{ .dspark = try dspark_weights.load(a, io, c, src.path, tensor, opts, base) };
            },
            .dflash2 => .{ .dflash2 = try dflash_weights.load(a, io, src.path, c.text.unpadded_vocab, opts, base) },
        };
    }
};

test "Ampere and Ada account for full BF16 drafter weights even with a Blackwell head requested" {
    const src: Source = .{ .kind = .dflash2, .path = "" };
    for ([_]u32{ 86, 89 }) |capability| {
        const opts = src.options(capability, true);
        try std.testing.expect(!opts.active());
        try std.testing.expect(!opts.borrow_head);
    }
}

pub const Weights = union(Kind) {
    dspark: dspark_weights.Weights,
    dflash2: dflash_weights.Weights,

    pub fn bytes(w: *const Weights) usize {
        return switch (w.*) {
            inline else => |*x| x.memory.len,
        };
    }

    pub fn deinit(w: *Weights) void {
        switch (w.*) {
            inline else => |*x| x.deinit(),
        }
    }
};

pub const Drafter = union(Kind) {
    dspark: dspark.Drafter,
    dflash2: dflash.Drafter,

    pub fn init(d: *const cuda.Driver, w: Weights, context: usize, rows_per_pass: usize) !Drafter {
        return switch (w) {
            .dspark => |x| .{ .dspark = try dspark.Drafter.init(d, x, context, rows_per_pass) },
            .dflash2 => |x| .{ .dflash2 = try dflash.Drafter.init(d, x, context, rows_per_pass) },
        };
    }

    pub fn enableGraphs(d: *Drafter, a: std.mem.Allocator) void {
        if (d.* == .dflash2) d.dflash2.enableGraphs(a);
    }

    pub fn taps(d: *const Drafter) @FieldType(engine.Model, "taps") {
        return switch (d.*) {
            inline else => |*x| .{ .function = x.k.tap, .buffer = x.scratch.taps },
        };
    }

    pub fn reserve(d: *Drafter, count: usize) !void {
        switch (d.*) {
            inline else => |*x| try x.reserve(count),
        }
    }

    pub fn deinit(d: *Drafter) void {
        switch (d.*) {
            inline else => |*x| x.deinit(),
        }
    }

    pub fn session(d: *Drafter, slot: usize) *dspark.Session {
        return switch (d.*) {
            inline else => |*x| &x.sessions[slot],
        };
    }

    pub fn snapshotBytes(d: *Drafter, slot: usize, at: usize) usize {
        if (d.* == .dspark) return dspark.Session.committedBytes(at);
        const state = d.session(slot);
        return @intCast(state.taps - state.memory.ptr);
    }

    pub fn eachSnapshot(d: *Drafter, slot: usize, at: usize, writer: anytype) !void {
        const state = d.session(slot);
        if (d.* == .dspark) return state.eachCommitted(at, writer);
        try writer.span(state.memory.ptr, @intCast(state.taps - state.memory.ptr));
    }

    pub fn absorb(d: *Drafter, slot: usize, k: *const ops.Kernels, s: cuda.Stream, from: u64, pos: usize, rows: usize, eps: f32) !void {
        switch (d.*) {
            .dspark => |*x| try x.absorb(&x.sessions[slot], k, s, from, pos, rows, eps),
            .dflash2 => |*x| try x.absorb(&x.sessions[slot], k, s, from, pos, rows),
        }
    }

    pub fn prefill(d: *Drafter, slot: usize, k: *const ops.Kernels, s: cuda.Stream, pos: usize, rows: usize, end: usize, eps: f32) !void {
        const state = d.session(slot);
        if (state.end != pos) return error.DraftContextGap;
        const skip = if (d.* == .dflash2) @min(rows, (end -| (dflash_weights.window - 1)) -| pos) else 0;
        const from = d.taps().?.buffer + skip * dspark.tap_stride;
        state.end += skip;
        try d.absorb(slot, k, s, from, pos + skip, rows - skip, eps);
    }

    pub fn propose(d: *Drafter, slot: usize, m: *const engine.Model, k: *const engine.Kernels, s: cuda.Stream, pending: u32, n: usize, rule: ?u64) !void {
        switch (d.*) {
            .dspark => |*x| try x.propose(&x.sessions[slot], m, k, s, pending, n, rule),
            .dflash2 => |*x| try x.propose(slot, m, k, s, pending, n, rule),
        }
    }
};

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

    pub fn checkpointBytes(src: Source, a: std.mem.Allocator, io: std.Io, c: *const config.Config) !usize {
        return switch (src.kind) {
            .dspark => dspark_weights.checkpointBytes(a, io, c, src.path),
            .dflash2 => dflash_weights.checkpointBytes(a, io, src.path),
        };
    }

    pub fn load(src: Source, a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, c: *const config.Config, ck: *core.Checkpoint) !Weights {
        return switch (src.kind) {
            .dspark => .{ .dspark = try dspark_weights.load(a, io, d, c, src.path, try ck.expect("model.llm.unembed.weight", .bf16, &.{ c.text.vocab, c.text.hidden })) },
            .dflash2 => .{ .dflash2 = try dflash_weights.load(a, io, d, src.path) },
        };
    }
};

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

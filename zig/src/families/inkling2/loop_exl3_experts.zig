//! Inkling 2's routed experts on core exl3_experts: core's item plan, grouped kernels and softcapped epilogue.
const std = @import("std");
const cuda = @import("cuda");

const ex = cuda.exl3_experts;
const grouped = cuda.grouped;

/// The two SwiGLU softcaps (config.json mlp_gate_softcap, mlp_up_softcap).
pub const Caps = struct { gate: f32, up: f32 };

/// The least and most half-bit widths (k2) among a layer's expert tensors, as the loader read them.
pub const Bits = struct { lo: u32, hi: u32 };

/// The baseline prompt path takes 32-member items at this width.
pub const wide_from = 512;
/// Large prompt chunks have enough pairs to share decoded weights across four member warps.
const prefill_from = 1024;

/// Core's experts module and the plan kernels of the shared experts image.
pub const Kernels = struct {
    main: ex.Experts,
    mod: cuda.Module,
    batch: ?cuda.Module,
    router: grouped.Router,

    pub fn load(d: *const cuda.Driver) !Kernels {
        var main = try ex.Experts.load(d);
        errdefer main.unload();
        var mod = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer mod.unload();
        var batch: ?cuda.Module = null;
        errdefer if (batch) |*image| image.unload();
        const sm = try d.currentCapability();
        if ((sm == 120 or sm == 121) and cuda.kernels.inkling2_exl3_batch.len != 0) {
            batch = try cuda.Module.load(d, cuda.kernels.inkling2_exl3_batch);
            // The desktop decode and Spark prefill variants are selected independently.
            if (sm == 120) main.short = try batch.?.function("tf_exl3_grouped_short_2");
            if (sm == 121) main.prefill = try batch.?.function("tf_exl3_grouped_prefill_2");
        }
        return .{ .main = main, .mod = mod, .batch = batch, .router = try grouped.Router.resolve(mod) };
    }

    pub fn unload(k: *Kernels) void {
        if (k.batch) |*batch| batch.unload();
        k.mod.unload();
        k.main.unload();
    }
};

fn align256(n: usize) usize {
    return (n + 255) / 256 * 256;
}

/// The plan's buffers past core's scratch: members, items, counts, then the wide plan's rank and hist.
fn planSizes(l: ex.Layer, rows: usize, slots: usize) [5]usize {
    const p = rows * slots;
    const hist = if (p > grouped.small) (p + grouped.small - 1) / grouped.small * (l.experts + 1) else 1;
    const items = @max(grouped.maxItems(p, l.experts + 1, 16), grouped.maxItems(@min(rows, 8) * slots, l.experts + 1, 8));
    return .{ p * 4, items * 12, 8, p * 4, hist * 4 };
}

/// Core's scratch, then the item plan.
pub fn scratchBytes(l: ex.Layer, rows: usize, slots: usize) !usize {
    var n = try ex.scratchBytes(l, rows, slots);
    for (planSizes(l, rows, slots)) |b| n += align256(b);
    return n;
}

/// Where the plan sits in scratch laid out by ex.scratch over scratchBytes().
fn planAt(l: ex.Layer, sc: ex.Scratch) grouped.Plan {
    var at = sc.members + align256(@min(sc.rows * sc.slots, l.experts) * sc.rows * 4);
    var ptr: [5]u64 = undefined;
    for (&ptr, planSizes(l, sc.rows, sc.slots)) |*q, b| {
        q.* = at;
        at += align256(b);
    }
    return .{ .members = ptr[0], .items = ptr[1], .counts = ptr[2], .rank = ptr[3], .hist = ptr[4] };
}

/// exl3_experts.Experts.runWith on core's item plan, with the softcaps as act_mode 3.
pub fn run(e: *const Kernels, s: cuda.Stream, l: ex.Layer, caps: Caps, bits: Bits, x: u64, x_stride: usize, x_dtype: ex.DType, pick: u64, wts: u64, rows: usize, out: u64, sc: ex.Scratch) !void {
    if (rows < 1 or rows > sc.rows or sc.slots < 1 or sc.slots > 32) return error.BadExperts;
    if (l.experts < 1 or l.experts > 512 or l.codebook != .mul1 or l.gu_stride != 0) return error.BadExperts;
    if (bits.lo < 2 or bits.hi > 16 or bits.lo > bits.hi) return error.BadExpertBits; // core decodes 1 to 8 bits
    var c = l;
    c.act_mode = 3;
    c.caps = .{ caps.gate, caps.up };
    const tile: usize = if (rows >= prefill_from and e.main.prefill != null)
        64
    else if (rows >= wide_from)
        32
    else if (rows > 1 and rows <= 8 and e.main.short != null)
        8
    else
        16;
    const it: ex.Items = .{ .router = &e.router, .plan = planAt(l, sc), .tile = tile };
    try e.main.runWith(s, c, x, x_stride, x_dtype, pick, wts, rows, out, sc, 0, it);
}

test "experts: Mini expert shapes have built grouped tiles at 2304 by 1152" {
    try std.testing.expectEqual(@as(u32, 4), try ex.plan(2304, 1152, true));
    try std.testing.expectEqual(@as(u32, 1), try ex.plan(1152, 2304, false));
    try std.testing.expect(1152 % 128 == 0 and 2304 % 128 == 0);
}

test "experts: the item plan sits past core's scratch and holds a 2,048-row call's items" {
    var l = std.mem.zeroes(ex.Layer);
    l.experts = 128;
    l.dims = 2304;
    l.width = 1152;
    l.codebook = .mul1;
    const bytes = try scratchBytes(l, 2048, 8);
    const sc = try ex.scratch(0, l, 2048, 8);
    const pl = planAt(l, sc);
    try std.testing.expectEqual(@as(u64, try ex.scratchBytes(l, 2048, 8)), pl.members);
    try std.testing.expect(pl.items - pl.members >= 16384 * 4);
    try std.testing.expect(pl.counts - pl.items >= (16384 / 16 + 129) * 12);
    try std.testing.expect(pl.hist + 16 * 129 * 4 <= bytes);
}

test "experts: run refuses expert widths outside core's 1 to 8 bits on the host" {
    var l = std.mem.zeroes(ex.Layer);
    l.experts = 128;
    l.dims = 2304;
    l.width = 1152;
    l.codebook = .mul1;
    const sc = try ex.scratch(0, l, 4, 8);
    const caps: Caps = .{ .gate = 0, .up = 0 };
    for ([_]Bits{ .{ .lo = 1, .hi = 8 }, .{ .lo = 2, .hi = 17 }, .{ .lo = 6, .hi = 4 } }) |b| {
        try std.testing.expectError(error.BadExpertBits, run(undefined, undefined, l, caps, b, 0, 0, .f16, 0, 0, 4, 0, sc));
    }
}

test "experts: run refuses a call or layer its kernels do not take, before any launch" {
    var l = std.mem.zeroes(ex.Layer);
    l.experts = 128;
    l.dims = 2304;
    l.width = 1152;
    l.codebook = .mul1;
    const sc = try ex.scratch(0, l, 4, 8);
    const caps: Caps = .{ .gate = 0, .up = 0 };
    const four: Bits = .{ .lo = 8, .hi = 8 };
    try std.testing.expectError(error.BadExperts, run(undefined, undefined, l, caps, four, 0, 0, .f16, 0, 0, 5, 0, sc));
    l.codebook = .mcg;
    try std.testing.expectError(error.BadExperts, run(undefined, undefined, l, caps, four, 0, 0, .f16, 0, 0, 4, 0, sc));
}

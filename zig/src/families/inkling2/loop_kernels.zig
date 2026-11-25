//! The CUDA kernels owned by one Mini model, including its format-specific expert and dense projections.
const cuda = @import("cuda");
const ops = @import("ops.zig");
const attention = @import("kernels.zig");
const modelopt = @import("modelopt.zig");
const loop_math = @import("loop_math_cuda.zig");
const exl3_experts = @import("loop_exl3_experts.zig");
const pack = @import("pack_weights.zig");
const weights = @import("loop_weights.zig");
const prompt_projections = @import("loop_prompt_projections.zig");

pub const Kernels = struct {
    glue: ops.Kernels,
    attn: attention.Kernels,
    moe: union(enum) { modelopt: modelopt.Kernels, exl3: exl3_experts.Kernels },
    dense: ?cuda.exl3.Dense,
    prompt_pair: ?prompt_projections.Kernels = null,
    loop: loop_math.Kernels,
    sample_mod: cuda.Module,
    draws: cuda.sample.Draws,

    pub fn load(ctx: *const cuda.Context, exl3: bool) !Kernels {
        const d = ctx.d;
        var glue = try ops.Kernels.load(d);
        errdefer glue.unload();
        var attn = try attention.Kernels.load(d);
        errdefer attn.unload();
        var moe: @FieldType(Kernels, "moe") = if (exl3)
            .{ .exl3 = try exl3_experts.Kernels.load(d) }
        else
            .{ .modelopt = try modelopt.Kernels.load(d, @intCast(try ctx.attribute(.multiprocessor_count))) };
        errdefer switch (moe) {
            inline else => |*m| m.unload(),
        };
        var dense = if (exl3) try cuda.exl3.Dense.load(d) else null;
        errdefer if (dense) |*m| m.unload();
        var lm = try loop_math.Kernels.load(d);
        errdefer lm.unload();
        var sample_mod = try cuda.Module.load(d, cuda.kernels.sample);
        errdefer sample_mod.unload();
        return .{
            .glue = glue,
            .attn = attn,
            .moe = moe,
            .dense = dense,
            .prompt_pair = if (dense != null and prompt_projections.enabled())
                try prompt_projections.Kernels.load(dense.?.lin.mod)
            else
                null,
            .loop = lm,
            .sample_mod = sample_mod,
            .draws = try cuda.sample.Draws.resolve(sample_mod),
        };
    }

    pub fn reserveDense(k: *Kernels, s: cuda.Stream, layers: []const weights.Layer) !void {
        const dense = if (k.dense) |*d| d else return;
        for (layers) |layer| try reserveProjections(dense, s, layer.attn.exl3.?);
    }

    pub fn project(d: *cuda.exl3.Dense, s: cuda.Stream, w: cuda.exl3.Weight, x: u64, rows: usize, out: u64) !void {
        if (rows <= 16) return d.decode(s, w, x, .bf16, rows, out, .bf16);
        return d.prompt(s, w, x, .bf16, rows, out, .bf16);
    }

    pub fn projectAttention(k: *Kernels, s: cuda.Stream, p: pack.Projections, x: [3]u64, out: [4]u64, rows: usize, prompt: bool) !void {
        const d = &k.dense.?;
        const qg = [2]cuda.exl3.Weight{ p.q, p.gate };
        const kv = [2]cuda.exl3.Weight{ p.k, p.v };
        if (prompt and rows >= 64 and d.implementation == .reference and d.lin.midm and d.lin.min_rows == 17) {
            if (k.prompt_pair) |pair| {
                if (prompt_projections.compatible(qg) and prompt_projections.compatible(kv)) {
                    try pair.run(d, s, qg, .{ x[0], x[0] }, .{ out[0], out[3] }, rows);
                    return pair.run(d, s, kv, .{ x[1], x[2] }, .{ out[1], out[2] }, rows);
                }
            }
        }
        if (rows <= 16 and x[0] == x[1] and x[0] == x[2]) {
            try d.decodeGroup(s, &.{ p.q, p.k, p.v }, x[0], .bf16, rows, out[0..3], .bf16);
        } else {
            for ([_]cuda.exl3.Weight{ p.q, p.k, p.v }, x, out[0..3]) |w, src, dst|
                try project(d, s, w, src, rows, dst);
        }
        try project(d, s, p.gate, x[0], rows, out[3]);
    }

    pub fn deinit(k: *Kernels) void {
        k.sample_mod.unload();
        k.loop.unload();
        if (k.dense) |*d| d.unload();
        switch (k.moe) {
            inline else => |*m| m.unload(),
        }
        k.attn.unload();
        k.glue.unload();
        k.* = undefined;
    }
};

/// The loaded and planned dense scratch visit the same projections in the same order.
pub fn reserveProjections(dense: anytype, s: cuda.Stream, p: pack.Projections) !void {
    for ([_]cuda.exl3.Weight{ p.q, p.k, p.v, p.o, p.gate }) |w| try dense.reserve(s, w);
    try dense.reserveGroup(s, &.{ p.q, p.k, p.v });
    const qg = [2]cuda.exl3.Weight{ p.q, p.gate };
    const kv = [2]cuda.exl3.Weight{ p.k, p.v };
    if (prompt_projections.enabled() and prompt_projections.compatible(qg) and prompt_projections.compatible(kv)) {
        try dense.reservePromptPair(s, qg);
        try dense.reservePromptPair(s, kv);
    }
}

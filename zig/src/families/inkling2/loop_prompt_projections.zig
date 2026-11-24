//! Paired EXL3 attention projections over the Mini prompt rows.
const cuda = @import("cuda");

const exl3 = cuda.exl3;
const Weight = exl3.Weight;

const Rotation = extern struct {
    x: [2]u64,
    scales: [2]u64,
    rotated: [2]u64,
    k: i32,
};

const Projection = extern struct {
    xh: u64,
    t: u64,
    stride_k: i64,
    stride_nb: i64,
    svh: u64,
    bias: u64,
    y: u64,
    z: u64,
    counters: u64,
    y_dtype: i32,
    n: i32,
    sk: i32,
    m: i32,
    k: i32,
};

pub fn enabled() bool {
    return @import("sys").environment.read(bool, "TF_INKLING2_PREFILL_PROJECTIONS", false) catch false;
}

pub fn compatible(ws: [2]Weight) bool {
    for (ws) |w| {
        if (w.k != 2304 or w.n != ws[0].n or w.n % 128 != 0) return false;
        const p = exl3.planOf(w);
        if (w.halves != 12 or w.codebook != .mul1 or w.sk != 0 or p.wk != 4 or (w.k / 16 / p.sk / p.wk) % 2 == 0)
            return false; // the shipped Mini shapes use four-warp MPG, with an odd K tile count per warp
    }
    return true;
}

pub const Kernels = struct {
    rotate: cuda.Function,
    mpg: [2]cuda.Function,

    pub fn load(mod: cuda.Module) !Kernels {
        return .{
            .rotate = try mod.function("tf_ink2_exl3_prompt_rotate_pair"),
            .mpg = .{ try mod.function("tf_ink2_exl3_prompt_pair_1"), try mod.function("tf_ink2_exl3_prompt_pair_2") },
        };
    }

    pub fn run(k: Kernels, d: *exl3.Dense, s: cuda.Stream, ws: [2]Weight, xs: [2]u64, ys: [2]u64, rows: usize) !void {
        const w0 = ws[0];
        const plan = exl3.planOf(w0);
        const scratch = try d.groupScratch(s, &ws, 128);
        var row: usize = 0;
        while (row < rows) : (row += 128) {
            const n = @min(rows - row, 128);
            if (n < 17) {
                for (ws, xs, ys) |w, x, y|
                    try d.prompt(s, w, x + row * w.k * 2, .bf16, n, y + row * w.n * 2, .bf16);
                break;
            }
            var rotation: Rotation = .{ .x = undefined, .scales = undefined, .rotated = undefined, .k = @intCast(w0.k) };
            var projections: [2]Projection = undefined;
            var z = scratch.z;
            var counters = scratch.counters;
            for (ws, xs, ys, 0..) |w, x, y, i| {
                rotation.x[i] = x + row * w.k * 2;
                rotation.scales[i] = w.suh;
                rotation.rotated[i] = scratch.xh + i * n * w.k * 2;
                projections[i] = .{
                    .xh = rotation.rotated[i],
                    .t = w.trellis,
                    .stride_k = @intCast(w.stride_k),
                    .stride_nb = @intCast(w.stride_nb),
                    .svh = w.svh,
                    .bias = w.bias,
                    .y = y + row * w.n * 2,
                    .z = z,
                    .counters = counters,
                    .y_dtype = @backingInt(exl3.DType.bf16),
                    .n = @intCast(w.n),
                    .sk = @intCast(plan.sk),
                    .m = @intCast(n),
                    .k = @intCast(w.k),
                };
                z += exl3.Linear.zBytes(w, n);
                counters += exl3.Linear.counterBytes(w);
            }
            var r: cuda.Args = .{};
            r.add(rotation);
            try cuda.launch.launch(k.rotate, .{
                .grid = .{ .x = (w0.k / 128 + 3) / 4, .y = @intCast(n), .z = 2 },
                .block = .{ .x = 128 },
            }, s, &r);
            const many = n > 48;
            const tile: u32 = if (many) 32 else 16;
            var p: cuda.Args = .{};
            p.add(projections);
            try cuda.launch.launch(k.mpg[@intFromBool(many)], .{
                .grid = .{ .x = @intCast(w0.n / 128 * ((n + tile - 1) / tile)), .y = plan.sk, .z = 2 },
                .block = .{ .x = 128 },
                .shared = exl3.Linear.sharedBytes(w0, n),
            }, s, &p);
        }
    }
};

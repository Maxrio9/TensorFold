//! The Mini release's EXL3 names and fused gate/up layout, using #4841's pack reader.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const config = @import("config.zig");
const exl3 = @import("exl3.zig");
const experts_run = @import("loop_exl3_experts.zig");

pub const sidecars: []const []const u8 = &.{ "model-mtp.safetensors", "model-vision.safetensors" };

pub const Projections = struct { q: Weight, k: Weight, v: Weight, o: Weight, gate: Weight };
const Weight = cuda.exl3.Weight;
pub const Experts = struct { layer: cuda.exl3_experts.Layer, bits: experts_run.Bits };

pub fn readBits(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !std.json.Parsed(std.json.Value) {
    const path = try std.fs.path.join(gpa, &.{ dir, "quantization_config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    return std.json.parseFromSlice(std.json.Value, gpa, bytes, .{ .allocate = .alloc_always });
}

pub fn bitsMap(json: std.json.Value) !std.json.ObjectMap {
    if (json != .object) return error.BadExl3Bits;
    const bits = json.object.get("bits") orelse return error.BadExl3Bits;
    if (bits != .object) return error.BadExl3Bits;
    return bits.object;
}

fn halves(bits: std.json.ObjectMap, base: []const u8) !u32 {
    const value = bits.get(base) orelse {
        std.log.err("Inkling 2 Mini EXL3: missing bit width for {s}", .{base});
        return error.BadExl3Bits;
    };
    const count = 2 * switch (value) {
        .integer => @as(f64, @floatFromInt(value.integer)),
        .float => value.float,
        else => return error.BadExl3Bits,
    };
    if (!std.math.isFinite(count) or count < 2 or count > 16 or count != @trunc(count)) return error.BadExl3Bits;
    return @intFromFloat(count);
}

const Loader = struct {
    ck: *core.Checkpoint,
    sink: exl3.Sink,
    bits: std.json.ObjectMap,
    read_marker: bool = true,

    fn group(l: Loader, base: []const u8, n: usize, k: usize) !exl3.Group {
        const g = try exl3.group(l.ck, base, n, k, l.read_marker);
        if (g.halves != try halves(l.bits, base)) {
            std.log.err("Inkling 2 Mini EXL3: {s} bit width disagrees with its trellis", .{base});
            return error.BadExl3Bits;
        }
        return g;
    }

    fn linear(l: Loader, i: u32, name: []const u8, n: usize, k: usize) !Weight {
        var buf: [96]u8 = undefined;
        const base = try std.fmt.bufPrint(&buf, "{d}.self_attn.{s}", .{ i, name });
        return exl3.linear(l.sink, try l.group(base, n, k), n, k);
    }
};

/// Exact Mini dimensions shared by the EXL3 projections and the grouped expert tiles.
pub fn validate(c: config.Text) !void {
    const heads: config.Heads = .{ .q = 24, .kv = 8, .dim = 128 };
    if (c.hidden != 2304 or c.expert_inter != 1152 or c.experts != 128 or c.topk != 8 or
        !std.meta.eql(c.local, heads) or !std.meta.eql(c.global, heads) or c.gate_cap <= 0 or c.up_cap <= 0)
        return error.Inkling2MiniExl3Shape;
}

pub fn attention(ck: *core.Checkpoint, sink: ?exl3.Sink, bits: std.json.ObjectMap, i: u32, h: usize, q: usize, kv: usize) !Projections {
    const Headers = struct {
        fn put(_: *anyopaque, _: []const u8) !u64 {
            return 0;
        }
    };
    var unused: u8 = 0;
    const l: Loader = .{
        .ck = ck,
        .sink = sink orelse .{ .ctx = &unused, .put = Headers.put },
        .bits = bits,
        .read_marker = sink != null,
    };
    return .{
        .q = try l.linear(i, "q_proj.linear", q, h),
        .k = try l.linear(i, "k_proj.linear", kv, h),
        .v = try l.linear(i, "v_proj.linear", kv, h),
        .o = try l.linear(i, "o_proj.linear", h, q),
        .gate = try l.linear(i, "o_proj.gate_proj", q, h),
    };
}

const Projection = struct {
    words: std.ArrayList(u8) = .empty,
    su: std.ArrayList(u8) = .empty,
    sv: std.ArrayList(u8) = .empty,
    offsets: [128]u64 = undefined,
    widths: [128]i32 = undefined,

    fn deinit(p: *Projection, gpa: std.mem.Allocator) void {
        p.words.deinit(gpa);
        p.su.deinit(gpa);
        p.sv.deinit(gpa);
    }

    fn append(p: *Projection, gpa: std.mem.Allocator, e: usize, g: exl3.Group, n: usize, k: usize, cut: exl3.Cut) !void {
        p.offsets[e] = p.words.items.len;
        p.widths[e] = @intCast(g.halves);
        _ = try exl3.cut(gpa, g, n, k, cut, &p.words, &p.su, &p.sv);
    }

    fn upload(p: *Projection, sink: exl3.Sink, shared_suh: ?u64) !struct { trellis: u64, halves: u64, suh: u64, svh: u64 } {
        const words = try sink.put(sink.ctx, p.words.items);
        for (&p.offsets) |*offset| offset.* += words;
        return .{
            .trellis = try sink.put(sink.ctx, std.mem.asBytes(&p.offsets)),
            .halves = try sink.put(sink.ctx, std.mem.asBytes(&p.widths)),
            .suh = shared_suh orelse try sink.put(sink.ctx, p.su.items),
            .svh = try sink.put(sink.ctx, p.sv.items),
        };
    }
};

pub fn experts(gpa: std.mem.Allocator, ck: *core.Checkpoint, sink: exl3.Sink, bits: std.json.ObjectMap, i: u32, c: config.Text) !Experts {
    const l: Loader = .{ .ck = ck, .sink = sink, .bits = bits };
    const h: usize = c.hidden;
    const inter: usize = c.expert_inter;
    var gate: Projection = .{};
    defer gate.deinit(gpa);
    var up: Projection = .{};
    defer up.deinit(gpa);
    var down: Projection = .{};
    defer down.deinit(gpa);

    var range: experts_run.Bits = .{ .lo = 16, .hi = 0 };
    for (0..c.experts) |e| {
        var buf: [96]u8 = undefined;
        const gu = try l.group(try std.fmt.bufPrint(&buf, "{d}.experts.gate_up_proj.{d}", .{ i, e }), 2 * inter, h);
        try gate.append(gpa, e, gu, 2 * inter, h, .{ .cols = .{ .lo = 0, .hi = inter } });
        try up.append(gpa, e, gu, 2 * inter, h, .{ .cols = .{ .lo = inter, .hi = 2 * inter } });
        const d = try l.group(try std.fmt.bufPrint(&buf, "{d}.experts.down_proj.{d}", .{ i, e }), h, inter);
        try down.append(gpa, e, d, h, inter, .whole);
        range = .{ .lo = @min(range.lo, gu.halves, d.halves), .hi = @max(range.hi, gu.halves, d.halves) };
    }

    const g = try gate.upload(sink, null);
    const u = try up.upload(sink, g.suh);
    const d = try down.upload(sink, null);
    var layer = expertGeometry(c);
    layer.gate = g.trellis;
    layer.up = u.trellis;
    layer.down = d.trellis;
    layer.gate_k2 = g.halves;
    layer.up_k2 = u.halves;
    layer.down_k2 = d.halves;
    layer.suh_g = g.suh;
    layer.suh_u = u.suh;
    layer.svh_g = g.svh;
    layer.svh_u = u.svh;
    layer.suh_d = d.suh;
    layer.svh_d = d.svh;
    return .{ .bits = range, .layer = layer };
}

pub fn expertGeometry(c: config.Text) cuda.exl3_experts.Layer {
    return .{
        .gate = 0,
        .up = 0,
        .down = 0,
        .gate_k2 = 0,
        .up_k2 = 0,
        .down_k2 = 0,
        .suh_g = 0,
        .suh_u = 0,
        .svh_g = 0,
        .svh_u = 0,
        .suh_d = 0,
        .svh_d = 0,
        .experts = c.experts,
        .dims = c.hidden,
        .width = c.expert_inter,
        .codebook = .mul1,
    };
}

test "Mini EXL3 bit table uses projection-specific halves" {
    const a = std.testing.allocator;
    const doc = try std.json.parseFromSlice(std.json.Value, a,
        \\{"bits":{"gate_up":3.5,"down":3,"other":4,"dense":6,"bad":3.2}}
    , .{});
    defer doc.deinit();
    const bits = try bitsMap(doc.value);
    try std.testing.expectEqual(7, try halves(bits, "gate_up"));
    try std.testing.expectEqual(6, try halves(bits, "down"));
    try std.testing.expectEqual(8, try halves(bits, "other"));
    try std.testing.expectEqual(12, try halves(bits, "dense"));
    try std.testing.expectError(error.BadExl3Bits, halves(bits, "bad"));
}

test "Mini EXL3 attention dimensions resolve built mul1 plans" {
    for ([_][2]usize{ .{ 2304, 3072 }, .{ 2304, 1024 }, .{ 3072, 2304 } }) |kn| {
        for ([_]u32{ 12, 16 }) |width| {
            const p = cuda.exl3.plan(kn[0], kn[1], 192, 8);
            try std.testing.expect(cuda.exl3.variantIndex(width, .mul1, p.wk) != null);
            try std.testing.expectEqual(0, kn[0] % (16 * p.sk * p.wk));
        }
    }
}

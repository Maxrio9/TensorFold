//! Image preparation, encoding and prompt row replacement for CUDA Mini.

// Copyright 2026 MiaAI-Lab

const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const lanes = @import("lanes");
const image = @import("image.zig");
const vision = @import("vision.zig");
const config = @import("config.zig");

pub const Limits = struct {
    image_tokens: u32 = 4096,
    max_images: u32 = 4,
    image_share: enum { request, fixed } = .request,

    pub fn single(limits: Limits, context: usize) u32 {
        const budget = switch (limits.image_share) {
            .request => limits.image_tokens,
            .fixed => limits.image_tokens / limits.max_images,
        };
        return @intCast(@min(budget, context));
    }

    pub fn rows(limits: Limits, context: usize, hidden: usize) usize {
        return @min(limits.image_tokens, context) * hidden * 2;
    }
};

pub fn validate(c: config.Vision, limits: Limits) !void {
    if (limits.image_tokens == 0 or limits.max_images == 0 or limits.max_images > 100 or limits.single(1) == 0)
        return error.Inkling2VisionLimits;
    if (c.patch != image.patch or c.merge != image.merge or c.grid != image.grid or c.hidden != vision.width or
        c.heads != vision.heads or c.layers != vision.depth or c.mlp_ratio * vision.width != vision.mlp_inter or
        c.proj_hidden != vision.proj_inter)
        return error.UnsupportedInkling2VisionConfig;
}

const Marks = struct { open: u32 = 200005, image: u32, close: u32 = 200010 };

pub const Images = struct {
    pub const template_wraps = true;

    io: std.Io,
    limits: Limits,
    marks: Marks,
    mark_ids: [3]u32,
    tower: *vision.Tower,
    slot_bytes: usize,

    pub fn init(a: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: config.Config, limits: Limits, context: usize) !Images {
        try validate(c.vision, limits);
        const marks: Marks = .{ .image = c.image_token };
        return .{
            .io = io,
            .limits = limits,
            .marks = marks,
            .mark_ids = .{ marks.open, marks.image, marks.close },
            .tower = try vision.Tower.loadWithWidth(a, io, d, dir, limits.single(context), c.text.hidden),
            .slot_bytes = limits.rows(context, c.text.hidden),
        };
    }

    pub fn prepare(v: *Images, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) !struct {
        pixels: []f32,
        gh: u32,
        gw: u32,
        tokens: u32,
        hash: u64,
    } {
        const p = try image.prepareResized(v.io, a, bytes, max_tokens);
        const grid = [2]u32{ p.rows * image.merge, p.cols * image.merge };
        return .{
            .pixels = @constCast(p.pixels), // prepareResized allocated these rows for the server
            .gh = grid[0],
            .gw = grid[1],
            .tokens = p.tokens(),
            .hash = core.picture.identity.withGrid(core.picture.identity.ofBytes(0x494e4b3256495331, bytes), grid),
        };
    }

    pub fn explain(err: anyerror) ?[]const u8 {
        return image.explain(err);
    }

    pub fn begin(v: *Images, s: cuda.Stream, stream: *const lanes.Stream, from: usize, dst: u64) !void {
        const prompt = stream.prompt();
        if (stream.images.len > v.limits.max_images) return error.TooManyImages;
        const cap = v.tower.max_tokens;
        var total: usize = 0;
        var end: usize = 0;
        for (stream.images) |im| {
            const at: usize = im.at;
            const n: usize = im.tokens;
            if (at < end or at > prompt.len or n == 0 or n > prompt.len - at) return error.ImagePlaceholders;
            if (!std.mem.allEqual(u32, prompt[at..][0..n], v.marks.image)) return error.ImagePlaceholders;
            if (im.gh % image.merge != 0 or im.gw % image.merge != 0 or
                @as(u64, im.gh / image.merge) * (im.gw / image.merge) != n or
                im.pixels.len != n * image.side * image.side * 3) return error.ImageGrid;
            end = at + n;
            total += n;
            if (n > cap or total > v.limits.image_tokens or total * v.tower.output_width * 2 > v.slot_bytes)
                return error.TooManyImageTokens;
        }
        if (std.mem.countScalar(u32, prompt, v.marks.image) != total) return error.ImagePlaceholders;
        var off: usize = 0;
        for (stream.images) |im| {
            if (stream.isCancelled()) return error.Cancelled;
            if (@as(usize, im.at) + im.tokens > from) {
                try v.tower.encode(s, .{ .pixels = im.pixels, .rows = im.gh / image.merge, .cols = im.gw / image.merge }, dst + off);
            }
            off += @as(usize, im.tokens) * v.tower.output_width * 2;
        }
    }
};

/// The projected rows belong exclusively to one prompt prefill and never enter a captured decode graph.
pub const Rows = struct {
    buffer: u64,
    images: []const lanes.stream.Image,
    position: usize,

    pub fn splice(r: Rows, d: *const cuda.Driver, s: cuda.Stream, n: usize, hidden: usize, dst: u64) !void {
        const stride = hidden * 2;
        var row: u32 = 0;
        for (r.images) |im| {
            const span: vision.Span = .{ .at = im.at, .n = im.tokens, .row = row };
            if (vision.cut(span, r.position, r.position + n)) |copy| {
                try d.check(d.api.cuMemcpyDtoDAsync_v2(
                    dst + copy.dst * stride,
                    r.buffer + copy.src * stride,
                    copy.n * stride,
                    s.handle,
                ), "cuMemcpyDtoDAsync");
            }
            row += im.tokens;
        }
    }
};

test "Mini capacity: fixed image shares reduce single-image scratch and preserve the request row budget" {
    const request: Limits = .{ .image_tokens = 4096, .max_images = 100 };
    var fixed = request;
    fixed.image_share = .fixed;
    try std.testing.expectEqual(4096, request.single(8704));
    try std.testing.expectEqual(40, fixed.single(8704));
    try std.testing.expectEqual(32, request.single(32));
    try std.testing.expectEqual(4096 * 2304 * 2, fixed.rows(8704, 2304));
}

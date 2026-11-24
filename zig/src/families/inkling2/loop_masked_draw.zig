//! Grammar-constrained draws from Mini's FP32 CUDA target logits.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");

pub fn pick(a: std.mem.Allocator, stream: cuda.Stream, vocab: usize, mask_vocab: usize, src: u64, positions: []const u64, sampling: ?lanes.Sampling, masks: []const u32, out: []u32) !void {
    const words = (mask_vocab + 31) / 32;
    if (masks.len != out.len * words or positions.len != out.len) return error.InvalidGrammarMasks;
    const row = try a.alloc(f32, vocab);
    defer a.free(row);
    // Read only constrained rows. Plain rows keep the asynchronous CUDA draw.
    try stream.synchronize();
    for (out, positions, 0..) |*token, position, r| {
        try cuda.memory.download(stream.d, src + r * vocab * @sizeOf(f32), std.mem.sliceAsBytes(row));
        token.* = try draw(a, row, masks[r * words ..][0..words], position, sampling);
    }
}

fn draw(a: std.mem.Allocator, row: []const f32, mask: []const u32, position: u64, sampling: ?lanes.Sampling) !u32 {
    const rule = sampling orelse lanes.Sampling{ .seed = 0, .temperature = 0 };
    if (rule.temperature <= 0) {
        var best: f32 = -std.math.inf(f32);
        var token: ?u32 = null;
        for (row, 0..) |logit, id| if (logit > best and std.math.isFinite(logit) and lanes.grammar.allows(mask, id)) {
            best = logit;
            token = @intCast(id);
        };
        return token orelse error.NoAllowedToken;
    }
    var scores: std.ArrayList(f64) = .empty;
    defer scores.deinit(a);
    var ids: std.ArrayList(u64) = .empty;
    defer ids.deinit(a);
    for (row, 0..) |logit, id| if (std.math.isFinite(logit) and lanes.grammar.allows(mask, id)) {
        try scores.append(a, logit);
        try ids.append(a, id);
    };
    return @intCast(try lanes.sampling.choose(a, scores.items, ids.items, position, rule));
}

test "masked draws exclude invalid scores and respect each row's allowed set" {
    const a = std.testing.allocator;
    const scores = [_]f32{ 8, 2, 2, std.math.nan(f32) };
    try std.testing.expectEqual(@as(u32, 1), try draw(a, &scores, &.{0b1110}, 1, null));
    try std.testing.expectEqual(@as(u32, 2), try draw(a, &scores, &.{0b1100}, 1, .{ .seed = 1 }));
    try std.testing.expectError(error.NoAllowedToken, draw(a, &scores, &.{0b1000}, 1, null));
}

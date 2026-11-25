//! Measure greedy draft acceptance on a frozen target history at the same maximum depth.
const std = @import("std");
const ink = @import("inkling2");
const check = @import("check.zig");

const Row = struct {
    id: []const u8,
    prompt_ids: []const u32,
    generated_ids: []const u32,
};

pub fn run(gpu: check.Gpu, model: []const u8, drafter: []const u8, fixture: []const u8) !void {
    const a = gpu.gpa;
    const io = gpu.io;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, fixture, a, .limited(1 << 22));
    defer a.free(bytes);
    const rows = try std.json.parseFromSlice([]const Row, a, bytes, .{});
    defer rows.deinit();
    const loaded = try ink.native.open(a, io, gpu.ctx, model, null, .{ .context = 8704, .drafts = true, .drafter = drafter });
    defer loaded.deinit(loaded.ctx);
    const o = loaded.target;
    const d = if (o.backend.drafter) |*draft| draft else return error.NoDrafter;
    if (d.* != .dflash2) return error.ExpectedDFlash2;
    const m = &o.model;
    const s = o.stream;
    m.capture_taps = true;
    const session = d.session(0);
    for (rows.value) |row| {
        const tokens = row.generated_ids;
        if (row.prompt_ids.len == 0 or tokens.len < 2 or row.prompt_ids.len + tokens.len > m.context) return error.BadDraftFixture;
        try m.reset(s);
        session.reset();
        var at: usize = 0;
        while (at < row.prompt_ids.len) {
            const end = @min(at + ink.native.prompt_rows, row.prompt_ids.len);
            try m.forward(&o.kernels, s, row.prompt_ids[at..end], at, 0, 0, false);
            try d.prefill(0, &o.kernels.glue, s, at, end - at, row.prompt_ids.len, o.weights.c.text.eps);
            at = end;
        }

        var hasher = std.crypto.hash.sha2.Sha256.init(.{});
        var accepted: usize = 0;
        var rounds: usize = 0;
        var i: usize = 0;
        while (i + 1 < tokens.len) {
            const n = @min(7, tokens.len - i - 1);
            try d.propose(0, m, &o.kernels, s, tokens[i], n, null);
            const held = session.held[0..n];
            hasher.update(std.mem.sliceAsBytes(held));
            var keep: usize = 0;
            while (keep < n and held[keep] == tokens[i + 1 + keep]) : (keep += 1) {}
            accepted += keep;
            rounds += 1;
            const end = i + 1 + keep;
            try m.forward(&o.kernels, s, tokens[i..end], at + i, 0, 0, false);
            try d.absorb(0, &o.kernels.glue, s, m.taps.?.buffer, at + i, end - i, o.weights.c.text.eps);
            i = end;
        }
        var digest: [32]u8 = undefined;
        hasher.final(&digest);
        const rate = @as(f64, @floatFromInt(accepted)) / @as(f64, @floatFromInt(rounds));
        std.debug.print("DRAFT ACCEPTANCE {s} accepted={d} rounds={d} per_round={d:.6} proposals={s}\n", .{
            row.id, accepted, rounds, rate, &std.fmt.bytesToHex(digest, .lower),
        });
    }
}

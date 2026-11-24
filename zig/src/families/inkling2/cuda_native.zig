//! Inkling 2 Mini's complete one-rank, text-only CUDA engine and checkpoint/snapshot admission.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const core = @import("core");
const config = @import("config.zig");
const plan_ = @import("plan.zig");
const loop = @import("loop.zig");
const engine = @import("loop_engine.zig");
const backend = @import("loop_backend.zig");
const pack = @import("pack_weights.zig");
const weights = @import("loop_weights.zig");
const draft_runtime = @import("loop_drafter.zig");
const dflash_weights = @import("dflash2_weights.zig");

pub const model_type = "inkling2_loop";
pub const formats: []const []const u8 = &.{ "modelopt-mixed", "exl3" };
pub const default_context: i64 = 1_048_576;
pub const max_segments: u32 = 1;
pub const startup_streams: u32 = 1;
pub const prompt_rows: u32 = engine.prompt_rows;
pub const window_rows: u32 = engine.window_rows;
pub const explains_open = true;
pub const takes_drafter = true;
pub const framed_structures = true;

pub const VisionOptions = struct { image_tokens: u32, max_images: u32 };
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
pub const Options = struct {
    context: usize,
    drafts: bool,
    kv8: bool = true,
    segments: usize = 1,
    slide: bool = false,
    vision: ?VisionOptions = null,
    precision: ?enum { full } = null,
    room: ?Room = null,
    drafter: ?[]const u8 = null,
    tp: usize = 1,
    rank: usize = 0,
    master: ?[]const u8 = null,
    master_port: u16 = 29551,
};

pub const Owned = struct {
    gpa: std.mem.Allocator,
    admission: loop.Admission,
    kernels: engine.Kernels,
    stream: cuda.Stream,
    weights: weights.Weights,
    model: engine.Model,
    backend: backend.Backend,
};

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    prompt_rows: u32,
    stream_bytes: usize,
    free_streams: u32 = startup_streams,
    auto_stream_limit: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    reserve: *const fn (*anyopaque, u32) anyerror!void = reserve,
    target: *Owned,
    draft_kind: ?[]const u8 = null,
};

fn own(ptr: *anyopaque) *Owned {
    return @ptrCast(@alignCast(ptr));
}

fn release(p: *anyopaque) void {
    const o = own(p);
    o.stream.synchronize() catch {};
    o.backend.deinit();
    o.stream.deinit();
    o.model.deinit();
    o.kernels.deinit();
    o.weights.deinit();
    const gpa = o.gpa;
    gpa.destroy(o);
}

/// The server's stream count: a cache slot each, past the one allocated at load.
fn reserve(p: *anyopaque, streams: u32) anyerror!void {
    return own(p).backend.reserve(streams);
}

pub fn fits(p: *anyopaque, prompt: u32, max_tokens: u32) bool {
    const o: *const Owned = @ptrCast(@alignCast(p));
    return o.admission.fits(prompt, max_tokens) and o.backend.admits(prompt, max_tokens);
}

pub fn readConfig(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, why: *config.Why) !config.Config {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    return config.parseCuda(gpa, bytes, why);
}

/// EXL3's text arena is sized from the same shards the loader opens, before the native pool admits the model.
pub fn textWeightBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !?u64 {
    var why: config.Why = .{};
    const c = readConfig(gpa, io, dir, &why) catch |err| return if (err == error.OutOfMemory) err else null;
    if (!c.exl3) return null;
    var ck = try core.Checkpoint.openModelExcluding(gpa, io, dir, pack.sidecars);
    defer ck.close();
    return try weights.checkpointBytes(&ck, true);
}

pub fn drafterDir(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, exl3: bool, o: Options) !?[]const u8 {
    if (!o.drafts) return null;
    if (o.drafter) |path| {
        if (!exl3) return error.Inkling2DraftNeedsExl3;
        return path;
    }
    if (!exl3) return null;
    const bundled = try std.fs.path.join(gpa, &.{ dir, "model-mtp.safetensors" });
    defer gpa.free(bundled);
    _ = std.Io.Dir.cwd().statFile(io, bundled, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    return dir;
}

pub fn drafterSource(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const config.Config, o: Options) !?draft_runtime.Source {
    const path = try drafterDir(gpa, io, dir, c.exl3, o) orelse return null;
    const dflash = o.drafter != null and try dflash_weights.isDFlash2(gpa, io, path, c);
    return .{ .kind = if (dflash) .dflash2 else .dspark, .path = path };
}

pub fn openCheckpoint(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, exl3: bool) !core.Checkpoint {
    var ck = if (exl3)
        try core.Checkpoint.openModelExcluding(gpa, io, dir, pack.sidecars)
    else
        try core.Checkpoint.openModel(gpa, io, dir);
    errdefer ck.close();
    try ck.skip("model.visual.");
    return ck;
}

pub fn checkpointPlan(gpa: std.mem.Allocator, ck: *const core.Checkpoint, c: *const config.Config) !plan_.Plan {
    var precisions: [config.max_layers]plan_.Precision = undefined;
    @memset(&precisions, .nvfp4);
    if (!c.exl3) {
        var header_arena = std.heap.ArenaAllocator.init(gpa);
        defer header_arena.deinit();
        var merged: core.safetensors.Header = .empty;
        const names = @import("names.zig");
        for (ck.files.items) |file| try names.merge(header_arena.allocator(), &merged, file.names);
        try names.precisions(merged, c.text.layers, &precisions);
    }
    return plan_.build(&c.text, precisions[0..c.text.layers]);
}

pub fn checkCacheRoom(state_bytes: u64, room: u64) error{Inkling2CacheRoom}!void {
    if (state_bytes > room) return error.Inkling2CacheRoom;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    if (kernels != null) return error.Inkling2CapturedKernels;
    if (o.segments != 1 or o.tp != 1 or o.rank != 0) return error.Inkling2LoopOneRankOnly;
    if (o.vision != null) return error.Inkling2LoopTextOnly;
    var why: config.Why = .{};
    const c = readConfig(gpa, io, dir, &why) catch |err| {
        if (why.text().len != 0) std.log.err("Inkling 2 Mini config refused: {s}", .{why.text()});
        return err;
    };
    try loop.validateLoad(&c, 1, false);
    if (o.context == 0 or o.context > c.text.max_context or o.context > std.math.maxInt(u32)) return error.BadLoopCache;
    if (c.exl3) try pack.validate(c.text);
    const drafter = try drafterSource(gpa, io, dir, &c, o);
    var ck = try openCheckpoint(gpa, io, dir, c.exl3);
    defer ck.close();
    const p = try checkpointPlan(gpa, &ck, &c);
    var bits = if (c.exl3) try pack.readBits(gpa, io, dir) else null;
    defer if (bits) |*b| b.deinit();
    var w = try weights.load(gpa, ctx.d, &ck, &c, p, if (bits) |b| try pack.bitsMap(b.value) else null);
    errdefer w.deinit();
    if (ck.unused() != 0) return error.UnusedCheckpointTensor;
    var dw = if (drafter) |src| try src.load(gpa, io, ctx.d, &c, &ck) else null;
    errdefer if (dw) |*d| d.deinit();
    ck.release();
    const target_weights = w.bytes;
    const draft_weights_bytes = if (dw) |*d| d.bytes() else 0;
    const before_state = cuda.usage(false).device;
    const state_room = if (o.room) |room| room.read(room.ctx) orelse return error.Inkling2CacheRoom else null;
    var k = try engine.Kernels.load(ctx, c.exl3);
    errdefer k.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    errdefer stream.deinit();
    try k.reserveDense(stream, w.layers);
    var m = try engine.Model.init(gpa, ctx.d, stream, &w, o.context, if (o.kv8) .fp8 else .bf16);
    errdefer m.deinit();
    var be = try backend.Backend.init(gpa, ctx.d, &m, &k, stream);
    errdefer be.deinit();
    if (dw) |d| {
        be.drafter = try draft_runtime.Drafter.init(ctx.d, d, o.context, m.promptRows());
        dw = null;
    }
    if (state_room) |room| try checkCacheRoom(cuda.usage(false).device -| before_state, room);
    const out = try gpa.create(Owned);
    errdefer gpa.destroy(out);
    out.* = .{ .gpa = gpa, .admission = .{ .context = @intCast(o.context) }, .kernels = k, .stream = stream, .weights = w, .model = m, .backend = be };
    // The value copies above own pointers to their old stack homes; retarget the live model/backend now.
    out.model.w = &out.weights;
    out.backend.m = &out.model;
    out.backend.k = &out.kernels;
    try out.backend.warm(io);
    const global_kv = loop.CacheLayout.globalBytesPerToken(&w.p, m.layout.format);
    std.log.info("Mini CUDA memory: target weights {d} bytes, drafter weights {d} bytes, {s} target KV {d} bytes per global token, {d} bytes allocated per stream", .{
        target_weights, draft_weights_bytes, @tagName(m.layout.format), global_kv, out.backend.streamBytesWithDraft(),
    });
    return .{
        .backend = out.backend.backend(),
        .facts = out.backend.facts(),
        .rows = window_rows,
        .prompt_rows = @intCast(out.model.promptRows()),
        .stream_bytes = out.backend.streamBytesWithDraft(),
        .auto_stream_limit = engine.max_slots,
        .ctx = out,
        .deinit = release,
        .target = out,
        .draft_kind = if (drafter) |src| src.kind.name() else null,
    };
}

pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    return own(ptr).model.snapshotBytes(at);
}

pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    return @ptrCast(try own(ptr).backend.saveState(owner, at));
}

pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    return own(ptr).backend.restoreState(owner, @ptrCast(@alignCast(saved)));
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    own(ptr).backend.dropState(@ptrCast(@alignCast(saved)));
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.Inkling2LoopOneRankOnly => "Inkling 2 Mini runs at one CUDA rank with one segment; TP and multi-segment modes are refused",
        error.Inkling2LoopTextOnly => "Inkling 2 Mini is text-only on CUDA: images and --vision are refused",
        error.Inkling2DraftNeedsExl3 => "Mini drafting on CUDA requires the EXL3 text checkpoint",
        error.UnsupportedMiniDFlash2Config => "Mini DFlash2 requires the BF16 Inkling 2 Mini drafter configuration",
        error.MiniDFlash2Tensor => "Mini DFlash2 weights do not match the checkpoint configuration",
        error.Inkling2CacheRoom => "the CUDA memory budget cannot fit Mini’s working state and KV cache: lower --context or free device memory",
        error.Inkling2CapturedKernels => "Inkling 2 Mini runs the engine's own kernels: unset TENSORFOLD_CUDA_KERNELS",
        error.Inkling2TooManyStreams => "Inkling 2 Mini on CUDA holds at most 16 streams: lower --parallel",
        error.ChainsOnly => "Inkling 2 Mini supports only non-branching text drafts",
        else => null,
    };
}

test "Mini native registration and refusal wording are explicit" {
    _ = dflash_weights;
    try std.testing.expectEqualStrings("inkling2_loop", model_type);
    try std.testing.expect(max_segments == 1 and prompt_rows == 4096 and window_rows == 16);
    try std.testing.expectEqual(16, engine.max_slots); // the stream cap explain() names
    for ([_]anyerror{ error.Inkling2LoopOneRankOnly, error.Inkling2LoopTextOnly, error.Inkling2DraftNeedsExl3, error.Inkling2CapturedKernels, error.Inkling2TooManyStreams }) |err| {
        try std.testing.expect(explain(null, err) != null);
    }
}

test "the complete prompt snapshot hooks match the store vtable" {
    const pc: struct {
        bytes: *const fn (*anyopaque, u32) u64,
        save: *const fn (*anyopaque, ?*anyopaque, u32) anyerror!*anyopaque,
        restore: *const fn (*anyopaque, ?*anyopaque, *anyopaque) anyerror!void,
        drop: *const fn (*anyopaque, *anyopaque) void,
    } = .{ .bytes = snap_bytes, .save = snap_save, .restore = snap_restore, .drop = snap_drop };
    try std.testing.expect(@intFromPtr(pc.save) != 0);
}

test "stream_bytes is what an extra stream's slot and lane buffers allocate, 25,087,496 bytes past the cache on Mini" {
    const Record = struct { // a DeviceBuffer hook that hands out addresses and sums what is asked
        bytes: usize = 0,
        next: u64 = 1 << 32,
        fn alloc(ctx: *anyopaque, len: usize) ?u64 {
            const r: *@This() = @ptrCast(@alignCast(ctx));
            r.bytes += len;
            r.next += std.mem.alignForward(u64, len, 512);
            return r.next;
        }
        fn free(_: *anyopaque, _: u64) bool {
            return true;
        }
    };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var why: config.Why = .{};
    var w: @import("loop_weights.zig").Weights = undefined;
    w.c = try config.parse(arena.allocator(), @import("config_test.zig").mini, &why);
    var precisions: [24]plan_.Precision = @splat(.nvfp4);
    w.p = try plan_.build(&w.c.text, &precisions);
    var m: engine.Model = undefined;
    m.d = @ptrFromInt(0x1000); // never called: the hook answers every allocation
    m.w = &w;
    m.layout = try loop.CacheLayout.init(&w.c.text, &w.p, 8192, .fp8, loop.promptRows(w.c.exl3));
    var b: backend.Backend = undefined;
    b.d = m.d;
    b.m = &m;
    b.held = 1;
    var rec: Record = .{};
    const before = cuda.memory.currentHook();
    cuda.memory.setHook(.{ .ctx = &rec, .alloc = Record.alloc, .free = Record.free });
    defer cuda.memory.setHook(before);
    var slot = try m.newSlot();
    defer slot.deinit();
    try b.slotBuffers(2); // the second stream's lane buffers; the first already holds its own
    defer for (&b.slot[1]) |*x| x.free();
    try std.testing.expectEqual(@as(u64, rec.bytes), backend.streamBytes(&m));
    try std.testing.expectEqual(@as(u64, 48 * 4 * 32 + 48 * (3 + engine.window_rows) * 6656 * 4 + 200_058 * 4 + 32), backend.streamBytes(&m) - m.layout.bytes);
    const conv = m.layout.bytes - m.layout.at(0).conv_offset;
    const rows = 8 * 8192 + 40 * 1023;
    try std.testing.expectEqual(conv + rows * 2 * 8 * 132, m.snapshotBytes(8192));
    m.layout = try loop.CacheLayout.init(&w.c.text, &w.p, 8192, .bf16, loop.promptRows(w.c.exl3));
    try std.testing.expectEqual(conv + rows * 2 * 8 * 256, m.snapshotBytes(8192));
}

test "the bundled Mini drafter is picked for EXL3 unless drafts are disabled or a path is explicit" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var o: Options = .{ .context = 8192, .drafts = true };
    try std.testing.expect(o.kv8);
    try std.testing.expectEqual(@as(?[]const u8, null), try drafterDir(a, io, dir, true, o));
    try tmp.dir.writeFile(io, .{ .sub_path = "model-mtp.safetensors", .data = "fixture" });
    try std.testing.expectEqualStrings(dir, (try drafterDir(a, io, dir, true, o)).?);
    try std.testing.expectEqual(@as(?[]const u8, null), try drafterDir(a, io, dir, false, o));
    o.drafter = "explicit";
    try std.testing.expectEqualStrings("explicit", (try drafterDir(a, io, dir, true, o)).?);
    try std.testing.expectError(error.Inkling2DraftNeedsExl3, drafterDir(a, io, dir, false, o));
    o.drafts = false;
    try std.testing.expectEqual(@as(?[]const u8, null), try drafterDir(a, io, dir, true, o));
}

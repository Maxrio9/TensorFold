//! Inkling 2 Mini's one-rank CUDA engine and checkpoint admission.
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
const vision = @import("mini_vision.zig");
const lazy_head = @import("loop_lazy_head.zig");
const pages = @import("loop_pages.zig");

pub const model_type = "inkling2_loop";
pub const formats: []const []const u8 = &.{ "modelopt-mixed", "exl3" };
pub const default_context: i64 = 1_048_576;
pub const max_segments: u32 = 1;
pub const startup_streams: u32 = 1;
pub const prompt_rows: u32 = engine.prompt_rows;
pub const window_rows: u32 = engine.window_rows;
pub const page_tokens = pages.page_tokens;
pub const share_grid = page_tokens;
pub const share_reclaim_lru = true;
pub const snap_lookahead = 0;
pub const explains_open = true;
pub const takes_drafter = true;
pub const framed_structures = true;

pub const VisionOptions = vision.Limits;
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
pub const Replan = struct { ctx: *anyopaque, run: *const fn (*anyopaque, *Options, u64) anyerror!bool };
pub const Paging = struct {
    layout: core.kv_pool.Layout,
    pool: core.kv_pool.Plan,
    ready_bytes: u64,
    cache_entries: usize = 0,
};
pub const Options = struct {
    context: usize,
    drafts: bool,
    kv8: bool = true,
    segments: usize = 1,
    slide: bool = false,
    vision: ?VisionOptions = null,
    precision: ?enum { full } = null,
    room: ?Room = null,
    paging: ?Paging = null,
    replan: ?Replan = null,
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
    vision: ?vision.Images,
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
    vision: ?*vision.Images,
    layout: ?core.kv_pool.Layout = null,
    engine_paged: bool = false,
    paging: ?Paging = null,
    context: usize,
};

fn own(ptr: *anyopaque) *Owned {
    return @ptrCast(@alignCast(ptr));
}

fn release(p: *anyopaque) void {
    const o = own(p);
    o.stream.synchronize() catch {};
    o.backend.deinit();
    if (o.vision) |v| v.tower.deinit();
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

pub fn kvPool(p: *anyopaque) ?core.kv_pool.Source {
    const pool = if (own(p).model.pages) |*x| &x.pool else return null;
    return .{ .pool = pool, .page_tokens = pages.page_tokens };
}

pub fn share_pool(p: *anyopaque) ?*core.kv_pool.Pool {
    return if (kvPool(p)) |src| src.pool else null;
}

pub const share_state = @import("loop_prompt.zig").Checkpoint;

pub fn share_backend(p: *anyopaque) *backend.Backend {
    return &own(p).backend;
}

pub fn share_attach(p: *anyopaque, prompts: ?core.kv_prompts.Prompts) void {
    own(p).backend.prompts = prompts;
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

pub fn visionBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, _: u32, _: u32) !u64 {
    return @import("vision.zig").weightBytes(gpa, io, dir);
}

fn hasDraftFile(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !bool {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound, error.NotDir => return false,
        else => return err,
    };
    return stat.kind == .file;
}

fn bundledDSpark(a: std.mem.Allocator, io: std.Io, dir: []const u8) !?draft_runtime.Source {
    if (!try hasDraftFile(a, io, dir, "model-mtp.safetensors")) return null;
    return .{ .kind = .dspark, .path = dir };
}

fn availableDFlash2(a: std.mem.Allocator, io: std.Io, path: []const u8, c: *const config.Config) !bool {
    return try hasDraftFile(a, io, path, "model.safetensors") and try dflash_weights.isDFlash2(a, io, path, c);
}

fn selectDrafter(a: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const config.Config, o: Options, cache: ?[]const u8) !?draft_runtime.Source {
    if (!o.drafts) return null;
    if (o.drafter) |path| {
        if (std.mem.eql(u8, path, "none")) return null;
        if (!c.exl3) return error.Inkling2DraftNeedsExl3;
        if (std.mem.eql(u8, path, "dspark")) {
            return (try bundledDSpark(a, io, dir)) orelse error.MiniDSparkMissing;
        }
        const dflash = try dflash_weights.isDFlash2(a, io, path, c);
        return .{ .kind = if (dflash) .dflash2 else .dspark, .path = path };
    }
    if (!c.exl3) return null;
    const beside = std.fs.path.dirname(std.mem.trimEnd(u8, dir, std.fs.path.sep_str)) orelse ".";
    for ([_][]const u8{ dir, beside }) |parent| {
        const local = try std.fs.path.join(a, &.{ parent, "dflash2" });
        const available = availableDFlash2(a, io, local, c) catch |err| {
            a.free(local);
            return err;
        };
        if (available) return .{ .kind = .dflash2, .path = local, .owns_path = true };
        a.free(local);
    }
    if (cache) |path| {
        if (path.len != 0 and try availableDFlash2(a, io, path, c)) return .{ .kind = .dflash2, .path = path };
    }
    return bundledDSpark(a, io, dir);
}

pub fn drafterSource(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const config.Config, o: Options) !?draft_runtime.Source {
    const sys = @import("sys");
    const name = "TF_INKLING2_DFLASH2_DIR";
    return selectDrafter(gpa, io, dir, c, o, try sys.environment.path(name, sys.environment.rawText(name)));
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

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, asked: Options) !Loaded {
    var o = asked;
    if (kernels != null) return error.Inkling2CapturedKernels;
    if (o.segments != 1 or o.tp != 1 or o.rank != 0) return error.Inkling2LoopOneRankOnly;
    var why: config.Why = .{};
    const c = readConfig(gpa, io, dir, &why) catch |err| {
        if (why.text().len != 0) std.log.err("Inkling 2 Mini config refused: {s}", .{why.text()});
        return err;
    };
    try loop.validateLoad(&c, 1, false);
    if (o.vision) |limits| try vision.validate(c.vision, limits);
    if (o.context == 0 or o.context > c.text.max_context or o.context > std.math.maxInt(u32)) return error.BadLoopCache;
    if (c.exl3) try pack.validate(c.text);
    const drafter = try drafterSource(gpa, io, dir, &c, o);
    defer if (drafter) |src| src.deinit(gpa);
    var ck = try openCheckpoint(gpa, io, dir, c.exl3);
    defer ck.close();
    const p = try checkpointPlan(gpa, &ck, &c);
    var bits = if (c.exl3) try pack.readBits(gpa, io, dir) else null;
    defer if (bits) |*b| b.deinit();
    var moved = false;
    const before_weights = cuda.usage(false).device;
    var w = try weights.load(gpa, ctx.d, &ck, &c, p, if (bits) |b| try pack.bitsMap(b.value) else null);
    errdefer if (!moved) w.deinit();
    if (ck.unused() != 0) return error.UnusedCheckpointTensor;
    var stream = try cuda.Stream.init(ctx.d, true);
    errdefer if (!moved) stream.deinit();
    const sm = try ctx.capability();
    var lazy: ?lazy_head.Head = if (c.exl3 and sm >= 120 and lazy_head.enabled() and lazy_head.eligible(c.text.unpadded_vocab, c.text.hidden))
        lazy_head.Head.init(gpa, ctx.d, stream, w.unembed, c.text.unpadded_vocab, c.text.hidden, c.text.logit_cap) catch |err| switch (err) {
            error.Inkling2FP8HeadWeights => null,
            else => return err,
        }
    else
        null;
    errdefer if (lazy) |*head| head.deinit();
    var dw = if (drafter) |src| try src.load(gpa, io, ctx.d, &c, &ck, stream, w.unembed, if (lazy) |head| head.weight else null) else null;
    errdefer if (dw) |*d| d.deinit();
    ck.release();
    const target_weights = w.bytes;
    const draft_weights_bytes = if (dw) |*d| d.bytes() else 0;
    const before_state = cuda.usage(false).device;
    const state_room = if (o.room) |room| room.read(room.ctx) orelse return error.Inkling2CacheRoom else null;
    if (o.paging) |pg| if (state_room) |room| {
        const loaded_bytes = before_state - before_weights;
        if (pg.ready_bytes -| loaded_bytes > room) {
            const replan = o.replan orelse return error.Inkling2CacheRoom;
            if (!try replan.run(replan.ctx, &o, room +| loaded_bytes)) return error.Inkling2CacheRoom;
            try checkCacheRoom(o.paging.?.ready_bytes -| loaded_bytes, room);
            std.log.info("Mini CUDA sized the pool from the free memory measured after loading weights", .{});
        }
    };
    var k = try engine.Kernels.load(ctx, c.exl3);
    errdefer if (!moved) k.deinit();
    try k.reserveDense(stream, w.layers);
    var m = try engine.Model.initWithPages(gpa, ctx.d, stream, &w, o.context, if (o.kv8) .fp8 else .bf16, if (o.paging) |pg| pg.pool.pages else 0);
    errdefer if (!moved) m.deinit();
    m.lazy_head = lazy;
    lazy = null;
    var be = try backend.Backend.init(gpa, ctx.d, &m, &k, stream);
    errdefer if (!moved) be.deinit();
    if (dw) |d| {
        be.drafter = try draft_runtime.Drafter.init(ctx.d, d, o.context, m.promptRows());
        dw = null;
    }
    if (state_room) |room| {
        const image_bytes = if (o.vision) |limits| (try visionBytes(gpa, io, dir, 0, 1)) +
            @import("vision.zig").scratchBytes(limits.single(o.context)) + limits.rows(o.context, c.text.hidden) else 0;
        try checkCacheRoom((cuda.usage(false).device -| before_state) + image_bytes, room);
    }
    const images = if (o.vision) |limits| try vision.Images.init(gpa, io, ctx.d, dir, c, limits, o.context) else null;
    errdefer if (!moved) {
        if (images) |v| v.tower.deinit();
    };
    const out = try gpa.create(Owned);
    out.* = .{ .gpa = gpa, .admission = .{ .context = @intCast(o.context) }, .kernels = k, .stream = stream, .weights = w, .model = m, .backend = be, .vision = images };
    moved = true;
    errdefer release(out);
    // The value copies above own pointers to their old stack homes; retarget the live model/backend now.
    out.model.w = &out.weights;
    out.backend.m = &out.model;
    out.backend.k = &out.kernels;
    if (out.vision) |*v| out.backend.vision = v;
    try out.backend.reserve(if (o.paging) |pg| pg.pool.lanes else startup_streams);
    if (o.paging) |pg| {
        const tracked = cuda.usage(false).device - before_weights;
        if (tracked != pg.ready_bytes) {
            std.log.err("Mini CUDA allocation check failed: planned {d}, allocated {d} bytes", .{ pg.ready_bytes, tracked });
            return error.Inkling2AllocationMismatch;
        }
    }
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
        .vision = if (out.vision) |*v| v else null,
        .layout = if (o.paging) |pg| pg.layout else null,
        .engine_paged = o.paging != null,
        .paging = o.paging,
        .context = o.context,
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
    const o = own(ptr);
    o.model.drop(o.stream, @ptrCast(@alignCast(saved)));
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.Inkling2LoopOneRankOnly => "Inkling 2 Mini runs at one CUDA rank with one segment; TP and multi-segment modes are refused",
        error.Inkling2LoopTextOnly => "start Inkling 2 Mini with --vision to accept images",
        error.Inkling2VisionLimits => "--vision needs a positive visual-token budget and 1 to 100 images; a fixed share needs at least one token per image",
        error.UnsupportedInkling2VisionConfig => "--vision needs the Inkling 2 Mini vision tower layout",
        error.NoVisionWeights, error.UnsupportedVisionDType => "--vision needs BF16 model.visual.* weights",
        error.TooManyImages, error.TooManyImageTokens => "images exceed the configured image count or token budget",
        error.ImageGrid, error.ImagePlaceholders => "the image grid or placeholder rows do not match the input",
        error.Inkling2DraftNeedsExl3 => "Mini drafting on CUDA requires the EXL3 text checkpoint",
        error.UnsupportedMiniDFlash2Config => "Mini DFlash2 requires the BF16 Inkling 2 Mini drafter configuration",
        error.MiniDSparkMissing => "the model does not include model-mtp.safetensors for --drafter dspark",
        error.MiniDFlash2Tensor => "Mini DFlash2 weights do not match the checkpoint configuration",
        error.Inkling2CacheRoom => "the CUDA memory budget cannot fit Mini’s working state and KV cache: lower --context or free device memory",
        error.Inkling2AllocationMismatch => "Mini CUDA tracked allocation differs from its capacity plan",
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
    m.pages = null;
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

test "Mini picks an explicit drafter, local DFlash2, cached DFlash2, then bundled DSpark" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const cache = try std.fs.path.join(a, &.{ dir, "cache" });
    defer a.free(cache);
    const local = try std.fs.path.join(a, &.{ dir, "dflash2" });
    defer a.free(local);
    var why: config.Why = .{};
    var c = try config.parse(a, @import("config_test.zig").mini, &why);
    c.exl3 = true;
    var o: Options = .{ .context = 8192, .drafts = true };
    try std.testing.expectEqual(null, try selectDrafter(a, io, dir, &c, o, cache));
    o.drafter = "dspark";
    try std.testing.expectError(error.MiniDSparkMissing, selectDrafter(a, io, dir, &c, o, cache));
    try tmp.dir.writeFile(io, .{ .sub_path = "model-mtp.safetensors", .data = "fixture" });
    o.drafter = null;
    try std.testing.expectEqual(.dspark, (try selectDrafter(a, io, dir, &c, o, cache)).?.kind);
    const fixture =
        \\{"architectures":["DFlash2DraftModel"],"model_type":"qwen3","dtype":"bfloat16",
        \\"hidden_size":2304,"intermediate_size":6912,"num_hidden_layers":5,"num_attention_heads":32,
        \\"num_key_value_heads":8,"head_dim":128,"rms_norm_eps":0.000001,
        \\"rope_parameters":{"rope_theta":10000,"rope_type":"default"},"vocab_size":201024,"sliding_window":2048,
        \\"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
        \\"attention_bias":false,"hidden_act":"silu","is_causal":false,
        \\"dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"final_logit_softcapping":30,
        \\"mask_token_id":200056,"selector_rank":256,"selector_softcap_mode":"after_correction",
        \\"selector_top_k":16,"target_layer_ids":[5,23,41,48]}}
    ;
    try tmp.dir.createDirPath(io, "cache");
    try tmp.dir.writeFile(io, .{ .sub_path = "cache/config.json", .data = fixture });
    try std.testing.expectEqual(.dspark, (try selectDrafter(a, io, dir, &c, o, cache)).?.kind);
    try tmp.dir.writeFile(io, .{ .sub_path = "cache/model.safetensors", .data = "fixture" });
    try std.testing.expectEqual(.dflash2, (try selectDrafter(a, io, dir, &c, o, cache)).?.kind);
    try tmp.dir.createDirPath(io, "dflash2");
    try tmp.dir.writeFile(io, .{ .sub_path = "dflash2/config.json", .data = fixture });
    try tmp.dir.writeFile(io, .{ .sub_path = "dflash2/model.safetensors", .data = "fixture" });
    const sibling = (try selectDrafter(a, io, dir, &c, o, cache)).?;
    defer sibling.deinit(a);
    try std.testing.expectEqual(.dflash2, sibling.kind);
    try std.testing.expectEqualStrings(local, sibling.path);
    try tmp.dir.createDirPath(io, "target");
    const target = try std.fs.path.join(a, &.{ dir, "target" });
    defer a.free(target);
    const adjacent = (try selectDrafter(a, io, target, &c, o, cache)).?;
    defer adjacent.deinit(a);
    try std.testing.expectEqualStrings(local, adjacent.path);
    o.drafter = cache;
    try std.testing.expectEqualStrings(cache, (try selectDrafter(a, io, dir, &c, o, cache)).?.path);
    o.drafter = "dspark";
    try std.testing.expectEqual(.dspark, (try selectDrafter(a, io, dir, &c, o, cache)).?.kind);
    c.exl3 = false;
    try std.testing.expectError(error.Inkling2DraftNeedsExl3, selectDrafter(a, io, dir, &c, o, cache));
    o.drafter = "none";
    try std.testing.expectEqual(null, try selectDrafter(a, io, dir, &c, o, cache));
    c.exl3 = true;
    o.drafter = cache;
    o.drafts = false;
    try std.testing.expectEqual(null, try selectDrafter(a, io, dir, &c, o, cache));
}

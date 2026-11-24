//! CUDA allocation sizes for one text-only Mini and its full-window streams.
const std = @import("std");
const cuda = @import("cuda");
const native = @import("cuda_native.zig");
const config = @import("config.zig");
const loop = @import("loop.zig");
const plan = @import("plan.zig");
const engine = @import("loop_engine.zig");
const backend = @import("loop_backend.zig");
const pack = @import("pack_weights.zig");
const weights = @import("loop_weights.zig");
const draft_weights = @import("dspark_weights.zig");
const draft = @import("dspark_cuda.zig");
const dflash = @import("dflash2_cuda.zig");
const draft_runtime = @import("loop_drafter.zig");
const Dense = cuda.exl3.Dense;

pub const Format = loop.CacheFormat;
pub const ConfigWhy = config.Why;
pub const max_streams = engine.max_slots;

const DensePlan = struct {
    sizes: Dense.ScratchBytes = .{},

    fn grow(self: *DensePlan, requested: Dense.ScratchBytes) void {
        inline for (@typeInfo(Dense.ScratchBytes).@"struct".field_names) |field| {
            @field(self.sizes, field) = @max(@field(self.sizes, field), @field(requested, field));
        }
    }

    pub fn reserve(self: *DensePlan, _: cuda.Stream, w: cuda.exl3.Weight) !void {
        self.grow(Dense.reserveBytes(w));
    }

    pub fn reserveGroup(self: *DensePlan, _: cuda.Stream, ws: []const cuda.exl3.Weight) !void {
        if (cuda.exl3.groupable(ws, 16)) self.grow(Dense.groupBytes(ws, 16));
    }

    fn bytes(self: DensePlan) usize {
        return self.sizes.z + self.sizes.xh + self.sizes.counters;
    }
};

pub const Checkpoint = struct {
    text: config.Text,
    schedule: plan.Plan,
    exl3: bool,
    checkpoint_bytes: u64,
    target_arena: u64,
    draft_arena: u64,
    draft_kind: draft_runtime.Kind = .dspark,
    dense_scratch: u64,

    pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, drafts: bool, drafter_dir: ?[]const u8, why: *config.Why) !Checkpoint {
        const c = try native.readConfig(a, io, dir, why);
        try loop.validateLoad(&c, 1, false);
        if (c.exl3) try pack.validate(c.text);
        var ck = try native.openCheckpoint(a, io, dir, c.exl3);
        defer ck.close();
        const schedule = try native.checkpointPlan(a, &ck, &c);
        const bytes = try weights.checkpointBytes(&ck, c.exl3);
        var dense: DensePlan = .{};
        if (c.exl3) {
            var doc = try pack.readBits(a, io, dir);
            defer doc.deinit();
            const bits = try pack.bitsMap(doc.value);
            const h = c.text.hidden;
            for (0..c.text.layers) |i| {
                const heads = c.text.heads(@intCast(i));
                const p = try pack.attention(&ck, null, bits, @intCast(i), h, heads.q * heads.dim, heads.kv * heads.dim);
                try @import("loop_kernels.zig").reserveProjections(&dense, undefined, p);
            }
        }
        const src = try native.drafterSource(a, io, dir, &c, .{ .drafts = drafts, .context = 1, .drafter = drafter_dir });
        const drafter = if (src) |d| try d.checkpointBytes(a, io, &c) else 0;
        return .{
            .text = c.text,
            .schedule = schedule,
            .exl3 = c.exl3,
            .checkpoint_bytes = bytes,
            .target_arena = weights.arenaBytes(bytes),
            .draft_arena = drafter,
            .draft_kind = if (src) |d| d.kind else .dspark,
            .dense_scratch = dense.bytes(),
        };
    }

    pub fn allocations(ck: *const Checkpoint, context: u32, format: Format) !Components {
        const layout = try loop.CacheLayout.init(&ck.text, &ck.schedule, context, format);
        const slot = engine.slotAllocs(&layout, ck.schedule.n_visits);
        const lane = backend.laneAllocs(ck.text.unpadded_vocab);
        const shared = backend.sharedAllocs(ck.text.unpadded_vocab);
        const conv = layout.bytes - layout.at(0).conv_offset;
        const global = loop.CacheLayout.globalBytesPerToken(&ck.schedule, format);
        const global_kv = global * std.mem.alignForward(u64, context, 64);
        const drafted = ck.draft_arena > 0;
        const shared_taps = if (drafted) draft.shared_tap_bytes else 0;
        const ds = drafted and ck.draft_kind == .dspark;
        const df = drafted and ck.draft_kind == .dflash2;
        const draft_partials = if (ds) draft.partialBytes(context) else if (df) dflash.partialBytes() else 0;
        const draft_scratch = if (ds) draft.Drafter.scratchBytes(context) else if (df) dflash.scratchBytes() else 0;
        const prepared_head: u64 = if (ds) draft_weights.prepared_head_bytes else 0;
        return .{
            .target_weights = ck.target_arena,
            .drafter_weights_and_padding = ck.draft_arena - prepared_head,
            .drafter_prepared_head = prepared_head,
            .dense_scratch = ck.dense_scratch,
            .model_working = try engine.Model.planScratch(ck.text, context, if (ck.exl3) pack.expertGeometry(ck.text) else null),
            .backend_shared = shared[0] + shared[1] + shared[2],
            .drafter_shared_other = draft_scratch - shared_taps - draft_partials,
            .drafter_shared_taps = shared_taps,
            .drafter_attention_partials = draft_partials,
            .target_global_kv = global_kv,
            .target_local_kv = layout.bytes - conv - global_kv,
            .target_conv = conv,
            .target_rewind = slot[1] + slot[2] + slot[3],
            .backend_lane = lane[0] + lane[1],
            .drafter_kv = if (ds) draft.Session.kvBytes(context) else if (df) dflash.kvBytes(context) else 0,
            .drafter_pending_taps = if (drafted) draft.Session.tap_bytes else 0,
            .drafter_newest = if (ds) draft.Session.newest_bytes else 0,
            .global_kv_per_token = global,
        };
    }
};

pub const Components = struct {
    target_weights: u64,
    drafter_weights_and_padding: u64,
    drafter_prepared_head: u64,
    dense_scratch: u64,
    model_working: u64,
    backend_shared: u64,
    drafter_shared_other: u64,
    drafter_shared_taps: u64,
    drafter_attention_partials: u64,
    target_global_kv: u64,
    target_local_kv: u64,
    target_conv: u64,
    target_rewind: u64,
    backend_lane: u64,
    drafter_kv: u64,
    drafter_pending_taps: u64,
    drafter_newest: u64,
    global_kv_per_token: u64,

    pub fn weights(self: Components) u64 {
        return self.target_weights + self.drafter_weights_and_padding + self.drafter_prepared_head;
    }

    pub fn stream(self: Components) u64 {
        return self.target_global_kv + self.target_local_kv + self.target_conv +
            self.target_rewind + self.backend_lane + self.drafterSession();
    }

    fn drafterSession(self: Components) u64 {
        return self.drafter_kv + self.drafter_pending_taps + self.drafter_newest;
    }

    fn drafterShared(self: Components) u64 {
        return self.drafter_shared_other + self.drafter_shared_taps + self.drafter_attention_partials;
    }

    pub fn state(self: Components) u64 {
        return self.dense_scratch + self.model_working + self.backend_shared + self.drafterShared() + self.stream();
    }
};

pub fn testCheckpoint(a: std.mem.Allocator) !Checkpoint {
    var why: config.Why = .{};
    const c = try config.parse(a, @import("config_test.zig").mini, &why);
    const precisions: [24]plan.Precision = @splat(.nvfp4);
    return .{
        .text = c.text,
        .schedule = try plan.build(&c.text, &precisions),
        .exl3 = true,
        .checkpoint_bytes = 12_429_923_328,
        .target_arena = 12_429_923_328,
        .draft_arena = 739_122_944,
        .dense_scratch = 0,
    };
}

test "Mini capacity: published stream and FP8 KV sizes" {
    var ck = try testCheckpoint(std.testing.allocator);
    const drafted = try ck.allocations(131072, .fp8);
    const conv = drafted.target_conv;
    const legacy_stream = drafted.stream() - drafted.drafterSession() - conv - conv / 3 * engine.window_rows;
    try std.testing.expectEqual(2_478_755_336, legacy_stream);
    try std.testing.expectEqual(3_048_988_680, drafted.stream());
    try std.testing.expectEqual(480_239_624, (try ck.allocations(8704, .fp8)).stream());
    try std.testing.expectEqual(16_896, drafted.global_kv_per_token);
    ck.draft_arena = 0;
    const plain = try ck.allocations(8704, .fp8);
    try std.testing.expectEqual(435_506_696, plain.stream());
    try std.testing.expectEqual(16_896, plain.global_kv_per_token);
}

test "Mini DFlash2 capacity keeps a fixed sliding ring and does not charge a DSpark folded head" {
    var ck = try testCheckpoint(std.testing.allocator);
    ck.draft_kind = .dflash2;
    ck.draft_arena = 989_967_360;
    const short = try ck.allocations(128, .fp8);
    const long = try ck.allocations(8704, .fp8);
    const longer = try ck.allocations(131072, .fp8);
    try std.testing.expectEqual(ck.draft_arena, long.drafter_weights_and_padding);
    try std.testing.expectEqual(0, long.drafter_prepared_head);
    try std.testing.expectEqual(0, long.drafter_newest);
    try std.testing.expectEqual(5 * (128 + 8) * 4096, short.drafter_kv);
    try std.testing.expectEqual(5 * (2047 + 8) * 4096, long.drafter_kv);
    try std.testing.expectEqual(long.drafter_kv, longer.drafter_kv);
    try std.testing.expectEqual(long.drafterShared(), longer.drafterShared());
}

const Record = struct {
    bytes: usize = 0,
    next: u64 = 1 << 32,

    fn alloc(ptr: *anyopaque, len: usize) ?u64 {
        const r: *Record = @ptrCast(@alignCast(ptr));
        r.bytes += len;
        r.next += std.mem.alignForward(u64, len, 512);
        return r.next;
    }

    fn free(_: *anyopaque, _: u64) bool {
        return true;
    }

    fn start(r: *Record) ?cuda.memory.Hook {
        const before = cuda.memory.currentHook();
        cuda.memory.setHook(.{ .ctx = r, .alloc = alloc, .free = free });
        return before;
    }

    fn expect(r: *Record, bytes: u64) !void {
        try std.testing.expectEqual(bytes, r.bytes);
        r.bytes = 0;
    }
};

test "Mini capacity: shared, first-slot and additional-slot allocations across context, KV and drafts" {
    const a = std.testing.allocator;
    const ck = try testCheckpoint(a);
    var w: weights.Weights = undefined;
    w.c.text = ck.text;
    w.c.exl3 = true;
    w.p = ck.schedule;
    var layer: weights.Layer = undefined;
    layer.experts = .{ .exl3 = .{ .layer = pack.expertGeometry(ck.text), .bits = .{ .lo = 6, .hi = 8 } } };
    w.layers = (&layer)[0..1];
    const d: *const cuda.Driver = @ptrFromInt(0x1000);
    const s = cuda.Stream.borrowed(d, null);
    var kernels: engine.Kernels = undefined;
    var r: Record = .{};
    const before = r.start();
    defer cuda.memory.setHook(before);
    const held = cuda.usage(false).device;
    for ([_]u32{ 8192, 8704, 131072, 1048576 }) |context| {
        for ([_]Format{ .fp8, .bf16 }) |format| {
            for ([_]bool{ false, true }) |drafts| {
                var request = ck;
                if (!drafts) request.draft_arena = 0;
                const c = try request.allocations(context, format);
                const layout = try loop.CacheLayout.init(&ck.text, &ck.schedule, context, format);
                var m: engine.Model = .{ .gpa = a, .d = d, .w = &w, .context = context, .layout = layout };
                defer m.deinit();
                try m.initScratch();
                try r.expect(c.model_working);

                m.slots[0] = try m.newSlot();
                m.n_slots = 1;
                try std.testing.expectEqual(c.target_global_kv + c.target_local_kv + c.target_conv, m.slots[0].cache.len);
                try r.expect(c.target_global_kv + c.target_local_kv + c.target_conv + c.target_rewind);
                var b = try backend.Backend.init(a, d, &m, &kernels, s);
                defer b.deinit();
                try r.expect(c.backend_shared + c.backend_lane);
                if (drafts) {
                    var shared = try draft.Drafter.allocScratch(d, context);
                    defer shared.free();
                    try r.expect(c.drafterShared());
                    var session = try draft.Session.init(d, context);
                    defer session.memory.free();
                    try r.expect(c.drafterSession());
                }

                m.slots[1] = try m.newSlot();
                m.n_slots = 2;
                try b.slotBuffers(2);
                if (drafts) {
                    var session = try draft.Session.init(d, context);
                    defer session.memory.free();
                }
                try r.expect(c.stream());
            }
        }
    }
    try std.testing.expectEqual(held, cuda.usage(false).device);
}

test "Mini capacity: dense scratch is the resident size after single and grouped reserves" {
    const Driver = struct {
        fn zero(_: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }
        fn weight(k: u32, n: u32, halves: u32) cuda.exl3.Weight {
            return .{ .k = k, .n = n, .halves = halves, .codebook = .mul1, .trellis = 0, .suh = 0, .svh = 0, .stride_k = 0, .stride_nb = 0 };
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuMemsetD8Async = Driver.zero;
    const s = cuda.Stream.borrowed(&driver, @ptrFromInt(1));
    var dense: Dense = .{ .d = &driver, .lin = undefined };
    defer for ([_]*?cuda.DeviceBuffer{ &dense.xh, &dense.z, &dense.counters }) |b| {
        if (b.*) |*x| x.free();
    };
    var r: Record = .{};
    const before = r.start();
    defer cuda.memory.setHook(before);
    const held = cuda.usage(false).device;
    var planned: DensePlan = .{};
    for ([_]u32{ 10, 8 }) |halves| {
        const p: pack.Projections = .{
            .q = Driver.weight(2304, 3072, halves),
            .k = Driver.weight(2304, 1024, halves),
            .v = Driver.weight(2304, 1024, halves),
            .o = Driver.weight(3072, 2304, halves),
            .gate = Driver.weight(2304, 3072, halves),
        };
        const kernels = @import("loop_kernels.zig");
        try kernels.reserveProjections(&dense, s, p);
        try kernels.reserveProjections(&planned, s, p);
        try std.testing.expectEqual(planned.bytes(), cuda.usage(false).device - held);
    }
}

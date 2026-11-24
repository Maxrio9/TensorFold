//! Stable device addresses used by the Mini forward and its decode graphs.
const cuda = @import("cuda");
const ops = @import("ops.zig");

pub const Scratch = struct {
    ids: u64,
    pos: u64,
    seg_of: u64,
    segs: u64,
    tiles: u64,
    blocks: [ops.max_blocks]u64 = @splat(0),
    partial_a: u64,
    partial_b: u64,
    prelude: u64,
    bank0: u64,
    bank1: u64,
    raw: u64, // three FP32 AR/gate rows per token
    zeta: u64, // a gate's 64 BF16 bottleneck values per token, between its down and up halves
    messages: u64, // four FP32 message rows per token
    normalized: u64, // three BF16 rows per token
    t: u64,
    q_raw: u64,
    rel_raw: u64,
    sink: u64,
    gate: u64,
    k_raw: u64,
    v_raw: u64,
    k_conv: u64,
    v_conv: u64,
    k_dynamic: u64,
    v_dynamic: u64,
    q_prepared: u64,
    rel_prepared: u64,
    rel_table: u64,
    part_o: u64,
    part_ml: u64,
    attn_heads: u64,
    attn_out: u64,
    route_scores: u64,
    route_pick: u64,
    route_weights: u64,
    moe_out: u64,
    moe_scratch: u64,
    exl3_scratch: ?cuda.exl3_experts.Scratch,
    hidden: u64,
    logits: u64,
};

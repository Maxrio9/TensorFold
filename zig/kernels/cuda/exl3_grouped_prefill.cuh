// The grouped mul1 projection for long Mini prompt chunks.
#pragma once

#include "exl3_grouped.cuh"

namespace tf_exl3 {

constexpr int PREFILL_NT = 2;
constexpr int PREFILL_WARPS = 4;

// One K tile per warp. Store the already-decoded B fragments in lane order so every
// member warp can read them without shuffles or bank conflicts.
using PrefillWeights = uint32_t[PREFILL_WARPS][PREFILL_NT][4][32];

template <int K2>
__device__ __forceinline__ void stage_weights(const uint32_t* trellis, int nt0, int stride, int kt,
                                              int warp, int lane, PrefillWeights& weights) {
    constexpr int TW = tile_words<K2>();
    const uint32_t* tile = trellis + ((size_t)kt * stride + nt0) * TW;
#pragma unroll
    for (int j = 0; j < PREFILL_NT; ++j) {
        uint32_t words[lane_words<K2>()], lo[2], hi[2];
        ldg_lane_words<K2>(tile + j * TW, lane, words);
        decode_lane<K2, CB_MUL1>(words, lane, lo, hi);
        weights[warp][j][0][lane] = lo[0];
        weights[warp][j][1][lane] = lo[1];
        weights[warp][j][2][lane] = hi[0];
        weights[warp][j][3][lane] = hi[1];
    }
}

__device__ __forceinline__ void stage_weights(const uint32_t* trellis, int nt0, int stride, int kt,
                                              int warp, int lane, int k2, PrefillWeights& weights) {
    switch (k2) {
#define TF_EXL3_PREFILL(K2) case K2: stage_weights<K2>(trellis, nt0, stride, kt, warp, lane, weights); break;
        TF_EXL3_PREFILL(2)
        TF_EXL3_PREFILL(3)
        TF_EXL3_PREFILL(4)
        TF_EXL3_PREFILL(5)
        TF_EXL3_PREFILL(6)
        TF_EXL3_PREFILL(7)
        TF_EXL3_PREFILL(8)
        TF_EXL3_PREFILL(9)
        TF_EXL3_PREFILL(10)
        TF_EXL3_PREFILL(11)
        TF_EXL3_PREFILL(12)
        TF_EXL3_PREFILL(13)
        TF_EXL3_PREFILL(14)
        TF_EXL3_PREFILL(15)
        TF_EXL3_PREFILL(16)
#undef TF_EXL3_PREFILL
        default: __trap();
    }
}

__device__ __forceinline__ void grouped_prefill(const GroupedArgs a, int item, int nblock, int z) {
    if (item >= a.ucount[0]) return;
    const int* plan = a.uids + 3 * item;
    const int expert = plan[0];
    if (expert == a.skip) return;
    const int first = plan[1], count = plan[2];
    const int mat = z / a.sk, split = z % a.sk;
    const auto* trellis = reinterpret_cast<const uint32_t*>(mat ? a.tp1[expert] : a.tp0[expert]);
    const int k2 = mat ? a.k2_1[expert] : a.k2_0[expert];
    const int stride = a.ns > 0 ? a.ns : a.n / 16;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g = lane / 4, t = lane % 4;
    const int per_part = a.k / 16 / a.sk / PREFILL_WARPS;
    const int split_start = split * PREFILL_WARPS * per_part, nt0 = nblock * PREFILL_NT;
    const int m0 = warp * 16 + g, m1 = m0 + 8;
    const int r0 = m0 < count ? a.members[first + m0] : -1;
    const int r1 = m1 < count ? a.members[first + m1] : -1;
    const half* x = mat ? a.x1 : a.x0;
    const half* x0 = x + (size_t)(r0 < 0 ? 0 : r0) * a.k + 2 * t;
    const half* x1 = x + (size_t)(r1 < 0 ? 0 : r1) * a.k + 2 * t;
    __shared__ PrefillWeights weights[2];
    float acc[PREFILL_NT][2][4] = {};
    float sum[PREFILL_NT][2][4];
    int part_end = per_part;

    // Consume contiguous four-tile waves, publishing the next wave into the other buffer.
    // Commit each original K range as its last tile is consumed, keeping the FP32 sum order.
    stage_weights(trellis, nt0, stride, split_start + warp, warp, lane, k2, weights[0]);
    for (int step = 0; step < PREFILL_WARPS * per_part; step += PREFILL_WARPS) {
        __syncthreads();
        const int read = (step / PREFILL_WARPS) & 1;
        if (step + PREFILL_WARPS < PREFILL_WARPS * per_part)
            stage_weights(trellis, nt0, stride, split_start + step + PREFILL_WARPS + warp,
                          warp, lane, k2, weights[read ^ 1]);

        if (warp * 16 < count) {
#pragma unroll
            for (int w = 0; w < PREFILL_WARPS; ++w) {
                const int k = (split_start + step + w) * 16;
                const uint32_t frag[4] = {
                    r0 >= 0 ? *reinterpret_cast<const uint32_t*>(x0 + k) : 0u,
                    r1 >= 0 ? *reinterpret_cast<const uint32_t*>(x1 + k) : 0u,
                    r0 >= 0 ? *reinterpret_cast<const uint32_t*>(x0 + k + 8) : 0u,
                    r1 >= 0 ? *reinterpret_cast<const uint32_t*>(x1 + k + 8) : 0u,
                };
#pragma unroll
                for (int j = 0; j < PREFILL_NT; ++j) {
                    const uint32_t lo[2] = {weights[read][w][j][0][lane], weights[read][w][j][1][lane]};
                    const uint32_t hi[2] = {weights[read][w][j][2][lane], weights[read][w][j][3][lane]};
                    mma16816(acc[j][0], frag, lo);
                    mma16816(acc[j][1], frag, hi);
                }
                if (step + w + 1 == part_end) {
#pragma unroll
                    for (int j = 0; j < PREFILL_NT; ++j)
#pragma unroll
                        for (int h = 0; h < 2; ++h)
#pragma unroll
                            for (int c = 0; c < 4; ++c) {
                                sum[j][h][c] = part_end == per_part ? acc[j][h][c] : sum[j][h][c] + acc[j][h][c];
                                acc[j][h][c] = 0.f;
                            }
                    part_end += per_part;
                }
            }
        }
    }

    float* out = a.z + ((size_t)mat * a.sk + split) * a.p * a.n + nt0 * 16;
#pragma unroll
    for (int j = 0; j < PREFILL_NT; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = j * 16 + h * 8 + t * 2;
            if (r0 >= 0) *reinterpret_cast<float2*>(out + (size_t)r0 * a.n + col) = make_float2(sum[j][h][0], sum[j][h][1]);
            if (r1 >= 0) *reinterpret_cast<float2*>(out + (size_t)r1 * a.n + col) = make_float2(sum[j][h][2], sum[j][h][3]);
        }
}

}  // namespace tf_exl3

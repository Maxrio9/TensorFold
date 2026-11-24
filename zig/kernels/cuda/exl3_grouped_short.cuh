// The grouped mul1 projection for up to eight routed pairs per item.
#pragma once

#include "exl3_grouped.cuh"

namespace tf_exl3 {

constexpr int SHORT_NT = 8;
constexpr int SHORT_ROWS = 8;
constexpr int SHORT_WARPS = 4;

// Transpose the matmul: the expert's 16 columns occupy MMA's M dimension,
// and the eight members occupy N. Each trellis tile requires just one MMA.
template <int K2>
__device__ __forceinline__ void short_tiles(const uint32_t* trellis, int nt0, int stride, int kt0, int nkt,
                                            const half* x, int lane, float (&acc)[SHORT_NT][4]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    const size_t kstride = (size_t)stride * TW;
    const uint32_t* tile = trellis + (size_t)kt0 * kstride + (size_t)nt0 * TW;
    uint32_t next[SHORT_NT][LW];
#pragma unroll
    for (int j = 0; j < SHORT_NT; ++j) ldg_lane_words<K2>(tile + j * TW, lane, next[j]);
    for (int k = 0; k < nkt; ++k, tile += kstride) {
        uint32_t words[SHORT_NT][LW];
#pragma unroll
        for (int j = 0; j < SHORT_NT; ++j)
#pragma unroll
            for (int l = 0; l < LW; ++l) words[j][l] = next[j][l];
        if (k + 1 < nkt)
#pragma unroll
            for (int j = 0; j < SHORT_NT; ++j) ldg_lane_words<K2>(tile + kstride + j * TW, lane, next[j]);
        uint32_t b[2] = {};
        if (x) {
            b[0] = *reinterpret_cast<const uint32_t*>(x + k * 16);
            b[1] = *reinterpret_cast<const uint32_t*>(x + k * 16 + 8);
        }
#pragma unroll
        for (int j = 0; j < SHORT_NT; ++j) {
            uint32_t lo[2], hi[2];
            decode_lane<K2, CB_MUL1>(words[j], lane, lo, hi);
            const uint32_t a[4] = {lo[0], hi[0], lo[1], hi[1]};
            mma16816(acc[j], a, b);
        }
    }
}

__device__ __forceinline__ void grouped_short(const GroupedArgs a, int item, int nblock, int z) {
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
    const int per_warp = a.k / 16 / a.sk / SHORT_WARPS;
    const int kt0 = (split * SHORT_WARPS + warp) * per_warp, nt0 = nblock * SHORT_NT;
    const half* x = nullptr;
    if (g < count) {
        const int row = a.members[first + g];
        x = (mat ? a.x1 : a.x0) + (size_t)row * a.k + kt0 * 16 + 2 * t;
    }
    float acc[SHORT_NT][4] = {};
    switch (k2) {
#define TF_EXL3_SHORT(K2) case K2: short_tiles<K2>(trellis, nt0, stride, kt0, per_warp, x, lane, acc); break;
        TF_EXL3_SHORT(2)
        TF_EXL3_SHORT(3)
        TF_EXL3_SHORT(4)
        TF_EXL3_SHORT(5)
        TF_EXL3_SHORT(6)
        TF_EXL3_SHORT(7)
        TF_EXL3_SHORT(8)
        TF_EXL3_SHORT(9)
        TF_EXL3_SHORT(10)
        TF_EXL3_SHORT(11)
        TF_EXL3_SHORT(12)
        TF_EXL3_SHORT(13)
        TF_EXL3_SHORT(14)
        TF_EXL3_SHORT(15)
        TF_EXL3_SHORT(16)
#undef TF_EXL3_SHORT
        default: __trap();
    }

    constexpr int COLS = SHORT_NT * 16;
    // Pad the member stride so the ordered K reduction can read by column.
    __shared__ float red[SHORT_WARPS][COLS][SHORT_ROWS + 1];
#pragma unroll
    for (int j = 0; j < SHORT_NT; ++j) {
        const int col = j * 16 + g, member = 2 * t;
        red[warp][col][member] = acc[j][0];
        red[warp][col][member + 1] = acc[j][1];
        red[warp][col + 8][member] = acc[j][2];
        red[warp][col + 8][member + 1] = acc[j][3];
    }
    __syncthreads();
    const int col = threadIdx.x;
    for (int m = 0; m < count; ++m) {
        float sum = red[0][col][m];
#pragma unroll
        for (int w = 1; w < SHORT_WARPS; ++w) sum += red[w][col][m];
        const int row = a.members[first + m];
        a.z[(((size_t)mat * a.sk + split) * a.p + row) * a.n + nt0 * 16 + col] = sum;
    }
}

}  // namespace tf_exl3

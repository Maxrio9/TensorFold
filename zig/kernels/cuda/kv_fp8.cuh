// Shared E4M3 cache-row quantization: one thread per value, FP32 row scale, nearest-even saturated codes.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace tf_kv_fp8 {
// A warp owns a head row, with consecutive elements in each lane.
template <int PER_LANE>
__device__ __forceinline__ void quant_warp(const float (&x)[PER_LANE], int lane, unsigned char* q, float* s) {
    float amax = 0.0f;
    for (int i = 0; i < PER_LANE; ++i) amax = fmaxf(amax, fabsf(x[i]));
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float inv = amax > 0.0f ? __fdiv_rn(448.0f, amax) : 1.0f;
    for (int i = 0; i < PER_LANE; ++i)
        q[lane * PER_LANE + i] = __nv_cvt_float_to_fp8(__fmul_rn(x[i], inv), __NV_SATFINITE, __NV_E4M3);
    if (lane == 0) *s = amax > 0.0f ? __fdiv_rn(amax, 448.0f) : 1.0f;
}

// Expand eight contiguous codes to the BF16 fragments used by BF16 tensor-core attention.
__device__ __forceinline__ uint4 bf16x8(const unsigned char* q, float scale) {
    const uint64_t codes = *reinterpret_cast<const uint64_t*>(q);
    uint4 result;
    auto* words = reinterpret_cast<unsigned*>(&result);
    for (int i = 0; i < 4; ++i) {
        const auto pair = static_cast<__nv_fp8x2_storage_t>(codes >> (i * 16));
        const float2 x = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3)));
        const unsigned lo = __bfloat16_as_ushort(__float2bfloat16_rn(x.x * scale));
        const unsigned hi = __bfloat16_as_ushort(__float2bfloat16_rn(x.y * scale));
        words[i] = lo | (hi << 16);
    }
    return result;
}

template <int WIDTH, bool FINITE_SCALE = false>
__device__ __forceinline__ void quant_row(float x, int d, unsigned char* q, float* s, float* red) {
    static_assert(WIDTH % 32 == 0 && WIDTH <= 1024, "one whole-warp block owns a row");
    float amax = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o /= 2) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if ((d & 31) == 0) red[d >> 5] = amax;
    __syncthreads();
    amax = red[0];
#pragma unroll
    for (int w = 1; w < WIDTH / 32; w++) amax = fmaxf(amax, red[w]);
    __syncthreads();
    float inv = amax > 0.0f ? __fdiv_rn(448.0f, amax) : 1.0f;
    float scale = 1.0f;
    if constexpr (FINITE_SCALE) {
        scale = amax > 0.0f ? __fdiv_rn(amax, 448.0f) : 1.0f;
        if (scale < 0x1p-126f) { scale = 0x1p-126f; inv = __fdiv_rn(1.0f, scale); }
    }
    q[d] = __nv_cvt_float_to_fp8(__fmul_rn(x, inv), __NV_SATFINITE, __NV_E4M3);
    if (d == 0) {
        if constexpr (FINITE_SCALE) *s = scale;
        else *s = amax > 0.0f ? __fdiv_rn(amax, 448.0f) : 1.0f;
    }
}
}  // namespace tf_kv_fp8

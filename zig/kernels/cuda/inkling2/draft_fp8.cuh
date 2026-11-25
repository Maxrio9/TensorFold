// Copyright 2026 MiaAI-Lab, draft-only FP8 head design adapted from patch 0006.

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace ink2d {
__device__ __forceinline__ float fp8Value(uint8_t code) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
}

__device__ __forceinline__ float fp8Scale(float max_abs) {
    for (int delta = 16; delta > 0; delta >>= 1)
        max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffffu, max_abs, delta));
    return max_abs == 0.0f || !isfinite(max_abs) ? 1.0f : fmaxf(max_abs / 448.0f, 0x1p-126f);
}

__device__ __forceinline__ uint8_t fp8Code(float value, float scale) {
    return __nv_cvt_float_to_fp8(value / scale, __NV_SATFINITE, __NV_E4M3);
}
}  // namespace ink2d

// A warp packs one row of a 64x64 qmmf FP8G tile, including its FP32 scale.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_draft_pack_layers(
    const uint16_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ scales, int n, int k) {
    using namespace ink2d;
    const int lane = threadIdx.x & 31;
    const size_t item = static_cast<size_t>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    const int groups = k / 64;
    const size_t tile = item / 64;
    const int rr = item % 64;
    const int r = (tile / groups) * 64 + rr;
    if (r >= (n + 127) / 128 * 128) return;
    const size_t at = static_cast<size_t>(r) * k + tile % groups * 64 + lane;
    const float lo = r < n ? bf(src[at]) : 0.0f;
    const float hi = r < n ? bf(src[at + 32]) : 0.0f;
    const float scale = fp8Scale(fmaxf(fabsf(lo), fabsf(hi)));
    // Inverse of modelopt_pack.cu's FP8 fragment mapping.
    const int p_lane = (rr % 8) * 4 + (lane % 8) / 2;
    const int b = (lane / 16) * 4 + (lane & 1) + ((lane / 8) & 1) * 2;
    const size_t dst = ((tile * 8 + rr / 8) * 32 + p_lane) * 16 + b;
    out[dst] = fp8Code(lo, scale);
    out[dst + 8] = fp8Code(hi, scale);
    if (lane == 0) scales[item] = scale;
}

// Rank-256 selector or Markov rows stay row-major, with one FP32 scale per row.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_draft_pack_rows(
    const uint16_t* __restrict__ src, uint8_t* __restrict__ out, float* __restrict__ scales, int n, int k) {
    using namespace ink2d;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= n) return;
    const size_t offset = static_cast<size_t>(row) * k;
    float max_abs = 0.0f;
    for (int j = lane; j < k; j += 32) max_abs = fmaxf(max_abs, fabsf(bf(src[offset + j])));
    const float scale = fp8Scale(max_abs);
    for (int j = lane; j < k; j += 32) out[offset + j] = fp8Code(bf(src[offset + j]), scale);
    if (lane == 0) scales[row] = scale;
}

extern "C" __global__ void __launch_bounds__(256) tf_ink2_dsp_vocab8(
    const uint16_t* __restrict__ base, const uint16_t* __restrict__ latent, const uint8_t* __restrict__ fold,
    const float* __restrict__ scales, const uint16_t* __restrict__ fbias, float cap, int valid, int nv,
    float* __restrict__ out) {
    using namespace ink2d;
    const int v = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (v >= nv) return;
    const uint8_t* f = fold + static_cast<size_t>(v) * 256;
    float sum = 0.0f;
    for (int k = lane; k < 256; k += 32) sum += bf(latent[k]) * fp8Value(f[k]);
    const float dot = tf_common::warp_sum(sum) * scales[v];
    if (lane != 0) return;
    const float b = bfr(dot + bf(fbias[v]));
    const float x = bfr(bf(base[v]) + b);
    out[v] = v < valid ? bfr(tanhf(x / cap) * cap) : -INFINITY;
}

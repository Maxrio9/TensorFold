// Copyright 2026 MiaAI-Lab

// Adapted from MiaAI-Lab patch 0008 for Mini's row-major BF16 head and E4M3 qmmf proxy.
#pragma once
#include <cuda_fp8.h>

// Same selected IDs, pick order, FP32 sum and division as tf_ink2_topk.
extern "C" __global__ void __launch_bounds__(128) tf_ink2_topk_warp(
    const float* __restrict__ scores, const float* __restrict__ bias, int* __restrict__ picks,
    float* __restrict__ weights, int experts, int top, float scale, int rows) {
    const int lane = threadIdx.x & 31;
    const size_t row = static_cast<size_t>(blockIdx.x) * 4 + (threadIdx.x >> 5);
    if (row >= static_cast<size_t>(rows)) return;
    float sel[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int e = lane + i * 32;
        sel[i] = e < experts ? scores[row * experts + e] + bias[e] : -INFINITY;
    }
    float sum = 0.0f, picked_score = 0.0f;
    int picked_id = 0;
    for (int k = 0; k < top; ++k) {
        float best = -INFINITY;
        int id = 0x7fffffff;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int e = lane + i * 32;
            if (e < experts && (sel[i] > best || (sel[i] == best && e < id))) {
                best = sel[i];
                id = e;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, offset);
            const int other_id = __shfl_xor_sync(0xffffffffu, id, offset);
            if (other > best || (other == best && other_id < id)) {
                best = other;
                id = other_id;
            }
        }
        const float score = id < experts ? scores[row * experts + id] : NAN;
        if (lane == k) {
            picked_score = score;
            picked_id = id;
        }
        sum += score;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            if (lane + i * 32 == id) sel[i] = -INFINITY;
    }
    if (lane < top) {
        picks[row * top + lane] = picked_id;
        weights[row * top + lane] = picked_score / (sum + 1e-20f) * scale;
    }
}

namespace ink2h {

constexpr unsigned kNegativeInf = 0x007f, kPositiveInf = 0xff80;
constexpr unsigned kMaximumAbs = 0x4b80; // 2^24: caps products far below FP32 overflow.

__device__ __forceinline__ unsigned rank(uint16_t bits) {
    return bits ^ ((bits & 0x8000u) ? 0xffffu : 0x8000u);
}

__device__ __forceinline__ uint16_t unrank(unsigned i) {
    return static_cast<uint16_t>(i ^ ((i & 0x8000u) ? 0x8000u : 0xffffu));
}

__device__ __forceinline__ unsigned warpMax(unsigned v) {
    for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ double warpSumUp(double v) {
    for (int o = 16; o > 0; o >>= 1) v = __dadd_ru(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ uint8_t quantize(float w, float inv) {
    return __nv_cvt_float_to_fp8(w * inv, __NV_SATFINITE, __NV_E4M3);
}

// The input column of a packed qmmf byte, matching tf_pack_fp8_lane.
__device__ __forceinline__ int fp8Column(int lane, int b) {
    const int p = 16 * (b / 4) + 4 * (lane % 4) + b % 4;
    return 16 * (p / 16) + 2 * ((p % 16) / 4) + p % 2 + 8 * ((p % 4) / 2);
}

__device__ __forceinline__ float error(float coefficient, float norm) {
    // Also allow absolute error from subnormal products/sums under the magnitude clamp.
    return __fadd_ru(__fmul_ru(coefficient, norm), 0x1p-60f);
}

}  // namespace ink2h

// Largest absolute BF16 code, including a nonfinite code if any.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_lazy_amax(
    const uint16_t* __restrict__ weights, long long total, unsigned* __restrict__ result) {
    unsigned hi = 0;
    for (long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x; i < total; i += gridDim.x * 256)
        hi = max(hi, static_cast<unsigned>(weights[i]) & 0x7fffu);
    hi = ink2h::warpMax(hi);
    if ((threadIdx.x & 31) == 0) atomicMax(result, hi);
}

extern "C" __global__ void __launch_bounds__(256) tf_ink2_lazy_pack(
    const uint16_t* __restrict__ weights, uint8_t* __restrict__ codes, int n, int k, float inv, long long total) {
    const long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= total) return;
    const int lane = (i / 16) % 32, kg = k / 64;
    const long long tile = i / 4096;
    const long long row = (tile / kg) * 64 + (i / 512 % 8) * 8 + lane / 4;
    const int col = (tile % kg) * 64 + (i / 8 % 2) * 32 + ink2h::fp8Column(lane, i % 8);
    codes[i] = row < n ? ink2h::quantize(bf(weights[row * k + col]), inv) : 0;
}

// Cauchy-Schwarz coefficient: ||w - proxy|| + scalar dot rounding + tensor-core dot rounding.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_lazy_coeff(
    const uint16_t* __restrict__ weights, float* __restrict__ coeff, int n, int k, float scale, float inv) {
    const int v = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (v >= n) return;
    const uint16_t* row = weights + static_cast<size_t>(v) * k;
    double wn = 0.0, pn = 0.0, dn = 0.0;
    for (int i = lane; i < k; i += 32) {
        const float w = bf(row[i]);
        const auto code = ink2h::quantize(w, inv);
        const double p = static_cast<double>(__half2float(__half(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)))) * scale;
        const double d = __dsub_ru(fmax(static_cast<double>(w), p), fmin(static_cast<double>(w), p));
        wn = __dadd_ru(wn, __dmul_ru(w, w));
        pn = __dadd_ru(pn, __dmul_ru(p, p));
        dn = __dadd_ru(dn, __dmul_ru(d, d));
    }
    wn = ink2h::warpSumUp(wn);
    pn = ink2h::warpSumUp(pn);
    dn = ink2h::warpSumUp(dn);
    if (lane == 0) {
        // k <= 4096: 128 scalar steps + warp tree; proxy allows 4096 FP32 roundings and a BF16 product ulp.
        const double wr = __dmul_ru(0x1p-16, __dsqrt_ru(wn));
        const double pr = __dmul_ru(0x1.2p-7, __dsqrt_ru(pn));
        coeff[v] = __double2float_ru(__dadd_ru(__dadd_ru(wr, pr), __dsqrt_ru(dn)));
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_ink2_lazy_caps(float* scores, float cap) {
    const unsigned i = blockIdx.x * 256 + threadIdx.x + ink2h::kNegativeInf;
    if (i <= ink2h::kPositiveInf) scores[i - ink2h::kNegativeInf] = logitCap(bf(ink2h::unrank(i)), cap);
}

extern "C" __global__ void __launch_bounds__(256) tf_ink2_lazy_prepare(
    const uint16_t* __restrict__ x, int k, float* __restrict__ norm, unsigned* __restrict__ floor,
    int* __restrict__ counts, int limit) {
    __shared__ double sums[8];
    __shared__ unsigned maxima[8];
    const int r = blockIdx.x, lane = threadIdx.x & 31;
    const uint16_t* row = x + static_cast<size_t>(r) * k;
    double ss = 0.0;
    unsigned hi = 0;
    for (int i = threadIdx.x; i < k; i += 256) {
        hi = max(hi, static_cast<unsigned>(row[i]) & 0x7fffu);
        const double value = bf(row[i]);
        ss = __dadd_ru(ss, __dmul_ru(value, value));
    }
    ss = ink2h::warpSumUp(ss);
    hi = ink2h::warpMax(hi);
    if (lane == 0) {
        sums[threadIdx.x >> 5] = ss;
        maxima[threadIdx.x >> 5] = hi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < 8; ++i) {
            ss = __dadd_ru(ss, sums[i]);
            hi = max(hi, maxima[i]);
        }
        const bool valid = hi <= ink2h::kMaximumAbs;
        norm[r] = valid ? __double2float_ru(__dsqrt_ru(ss)) : INFINITY;
        floor[r] = ink2h::kNegativeInf;
        counts[r] = valid ? 0 : limit + 1;
    }
}

extern "C" __global__ void __launch_bounds__(256) tf_ink2_lazy_floor(
    const float* __restrict__ logits, const float* __restrict__ coeff, const float* __restrict__ norm,
    unsigned* __restrict__ floor, int vocab) {
    const int row = blockIdx.y, lane = threadIdx.x & 31;
    const float n = norm[row];
    if (!isfinite(n)) return;
    const float* values = logits + static_cast<size_t>(row) * vocab;
    unsigned lower = ink2h::kNegativeInf;
    for (int v = blockIdx.x * 256 + threadIdx.x; v < vocab; v += gridDim.x * 256) {
        const float z = values[v];
        if (isfinite(z)) lower = max(lower, ink2h::rank(tobf(__fsub_rd(z, ink2h::error(coeff[v], n)))));
    }
    lower = ink2h::warpMax(lower);
    if (lane == 0) atomicMax(floor + row, lower);
}

extern "C" __global__ void __launch_bounds__(256) tf_ink2_lazy_select(
    float* __restrict__ logits, const float* __restrict__ coeff, const float* __restrict__ norm,
    const unsigned* __restrict__ floor, const uint16_t* __restrict__ cutoffs, int vocab,
    int* __restrict__ counts, int* __restrict__ picks, int limit) {
    const int row = blockIdx.y, lane = threadIdx.x & 31;
    const float n = norm[row];
    if (!isfinite(n)) return;
    const unsigned cutoff = cutoffs[floor[row] - ink2h::kNegativeInf];
    float* values = logits + static_cast<size_t>(row) * vocab;
    int* ids = picks + static_cast<size_t>(row) * limit;
    for (int base = blockIdx.x * 256; base < vocab; base += gridDim.x * 256) {
        const int v = base + threadIdx.x;
        bool keep = false;
        if (v < vocab) {
            const float z = values[v];
            keep = !isfinite(z) || ink2h::rank(tobf(__fadd_ru(z, ink2h::error(coeff[v], n)))) > cutoff;
            values[v] = -INFINITY;
        }
        const unsigned mask = __ballot_sync(0xffffffffu, keep);
        if (mask == 0) continue;
        const int lead = __ffs(mask) - 1;
        const int start = lane == lead ? atomicAdd(counts + row, __popc(mask)) : 0;
        const int i = __shfl_sync(0xffffffffu, start, lead) + __popc(mask & ((1u << lane) - 1));
        if (keep && i < limit) ids[i] = v;
    }
}

// Overflow, invalid inputs and an empty shortlist use the same dense BF16 row arithmetic.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_lazy_exact(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ weights, float* __restrict__ logits,
    const int* __restrict__ counts, const int* __restrict__ picks, int limit, int vocab, int k, float cap) {
    const int row = blockIdx.y, lane = threadIdx.x & 31, n = counts[row];
    const int* ids = picks + static_cast<size_t>(row) * limit;
    const bool full = n <= 0 || n > limit;
    const uint16_t* xr = x + static_cast<size_t>(row) * k;
    float* out = logits + static_cast<size_t>(row) * vocab;
    for (int i = blockIdx.x * 8 + (threadIdx.x >> 5); i < (full ? vocab : n); i += gridDim.x * 8) {
        const int v = full ? i : ids[i];
        const float dot = rowDot(xr, weights + static_cast<size_t>(v) * k, k, lane);
        if (lane == 0) out[v] = logitCap(bfr(dot), cap);
    }
}

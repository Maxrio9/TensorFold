// Mini DFlash2 additions to the shared draft attention and BF16 operators.
namespace ink2df {
constexpr int kTop = 16;
constexpr int kRank = 256;
constexpr int kLogprobTop = 20;
constexpr int kLogprobWords = 3 + 2 * kLogprobTop;
}

extern "C" __global__ void __launch_bounds__(64) tf_ink2_df_qk(QkArgs a, const int* position) {
    __shared__ float sh[32];
    ink2DraftQk(a, *position, sh);
}

extern "C" __global__ void __launch_bounds__(256) tf_ink2_df_attn(AttnArgs a, const int* position, int window) {
    ink2DraftAttn<true>(a, *position, window);
}

// Each row's coefficients are shared by groups of 16 channels. The prior row stays inside this block.
extern "C" __global__ void __launch_bounds__(256) tf_ink2_df_conv(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ dyn, const uint16_t* __restrict__ base,
    uint16_t* __restrict__ out, int width, int side) {
    using namespace ink2d;
    const int row = blockIdx.x, groups = width / 16;
    const uint16_t* delta = dyn + (4 * row + 2 * side) * groups;
    const uint16_t* kernel = base + 2 * side * width;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        float sum = (bf(kernel[c]) + bf(delta[c / 16])) * bf(x[row * width + c]);
        if (row > 0)
            sum += (bf(kernel[width + c]) + bf(delta[groups + c / 16])) * bf(x[(row - 1) * width + c]);
        const int at = row * width + c;
        out[at] = side == 0 ? tobf(sum) : tobf(bf(out[at]) + bfr(sum));
    }
}

// Score just the current predecessor's edges. A warp evaluates one of the top-16 candidates.
template <typename Book>
__device__ __forceinline__ void ink2DfSelect(
    const unsigned* __restrict__ top, const uint16_t* __restrict__ projected,
    const Book* __restrict__ predecessor, const Book* __restrict__ successor,
    const float* __restrict__ prev_scales, const float* __restrict__ next_scales,
    unsigned* __restrict__ walk, long long* __restrict__ ids, uint16_t* __restrict__ logits,
    int step, int vocab, float cap) {
    using namespace ink2d;
    using namespace ink2df;
    __shared__ float scores[kTop];
    const int candidate = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned* entries = top + step * kLogprobWords;
    const unsigned id = entries[2 + candidate];
    const unsigned parent = walk[step];
    float score = -INFINITY;
    if (id < vocab && parent < vocab) {
        float correction = 0;
        const float ps = sizeof(Book) == 1 ? prev_scales[parent] : 1.0f;
        const float ss = sizeof(Book) == 1 ? next_scales[id] : 1.0f;
        for (int j = lane; j < kRank; j += 32) {
            const Book a = predecessor[static_cast<size_t>(parent) * kRank + j];
            const Book b = successor[static_cast<size_t>(id) * kRank + j];
            const float p = sizeof(Book) == 1 ? bfr(fp8Value(a) * ps) : bf(a);
            const float q = sizeof(Book) == 1 ? bfr(fp8Value(b) * ss) : bf(b);
            const float h = bf(projected[step * kRank + j]);
            correction += bfr(p * h) * q;
        }
        correction = tf_common::warp_sum(correction);
        const float unary = __uint_as_float(entries[2 + kLogprobTop + candidate]);
        const float corrected = bfr(unary + bfr(correction));
        score = bfr(cap * tanhf(corrected / cap));
    }
    if (lane == 0) {
        scores[candidate] = score;
        ids[candidate] = id;
        logits[candidate] = tobf(score);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int best = 0;
        for (int i = 1; i < kTop; ++i)
            if (scores[i] > scores[best]) best = i;
        walk[step + 1] = isfinite(scores[best]) ? static_cast<unsigned>(ids[best]) : static_cast<unsigned>(vocab);
    }
}

extern "C" __global__ void __launch_bounds__(512) tf_ink2_df_select(
    const unsigned* __restrict__ top, const uint16_t* __restrict__ projected,
    const uint16_t* __restrict__ predecessor, const uint16_t* __restrict__ successor,
    unsigned* __restrict__ walk, long long* __restrict__ ids, uint16_t* __restrict__ logits,
    int step, int vocab, float cap) {
    ink2DfSelect(top, projected, predecessor, successor, nullptr, nullptr, walk, ids, logits, step, vocab, cap);
}

extern "C" __global__ void __launch_bounds__(512) tf_ink2_df_select8(
    const unsigned* __restrict__ top, const uint16_t* __restrict__ projected,
    const uint8_t* __restrict__ predecessor, const uint8_t* __restrict__ successor,
    const float* __restrict__ prev_scales, const float* __restrict__ next_scales,
    unsigned* __restrict__ walk, long long* __restrict__ ids, uint16_t* __restrict__ logits,
    int step, int vocab, float cap) {
    ink2DfSelect(top, projected, predecessor, successor, prev_scales, next_scales, walk, ids, logits, step, vocab, cap);
}

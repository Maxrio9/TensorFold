// Mini's batched EXL3 projections are separate from the single-row CUDA image.
#include "exl3_grouped_prefill.cuh"
#include "exl3_grouped_short.cuh"

extern "C" __global__ void tf_exl3_grouped_short_2(tf_exl3::GroupedArgs a) {
    tf_exl3::grouped_short(a, blockIdx.x, blockIdx.y, blockIdx.z);
}

extern "C" __global__ void tf_exl3_grouped_prefill_2(tf_exl3::GroupedArgs a) {
#pragma unroll 1
    for (int stripe = 0; stripe < 2; ++stripe) {
        tf_exl3::grouped_prefill(a, blockIdx.x, blockIdx.y * 2 + stripe, blockIdx.z);
        __syncthreads();
    }
}

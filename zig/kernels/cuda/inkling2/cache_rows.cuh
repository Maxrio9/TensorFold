#pragma once

#include <cstddef>
#include <cstdint>

// A segment has either a contiguous cache offset or the address of its global KV page table.
struct Ink2Segment {
    int row0;
    int n;
    int p0;
    int paged;
    long long off;

    __device__ __forceinline__ size_t offset() const { return paged ? 0 : off; }
    __device__ __forceinline__ size_t physical(int pos, int ring) const {
        constexpr int page_rows = 256; // loop_pages.zig
        if (!paged) return pos % ring;
        const auto* table = reinterpret_cast<const uint32_t*>(off);
        return static_cast<size_t>(table[pos / page_rows]) * page_rows + pos % page_rows;
    }
};

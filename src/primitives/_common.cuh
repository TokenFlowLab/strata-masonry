// Internal helpers shared across primitive headers (unnumbered, infrastructure).
// Not a numbered primitive; mirrors tests/test_utils.cuh pattern for non-PTX code.
#pragma once

// PTX:    n/a (test-helper aliases for the primitives layer)
//
#include <cuda_runtime.h>
#include <cstdint>

// Generic ptr -> SMEM 32-bit address (PTX SMEM addressing).
__device__ __forceinline__ uint32_t cvta_to_shared_u32(void const* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Lane id within a warp (0..31).
__device__ __forceinline__ uint32_t lane_id() {
    uint32_t l;
    asm("mov.u32 %0, %%laneid;" : "=r"(l));
    return l;
}

// Warp id within the CTA (0..warps_per_cta-1).
__device__ __forceinline__ uint32_t warp_id() {
    return threadIdx.x / 32u;
}

// Warpgroup id within the CTA (0..warpgroups_per_cta-1). A warpgroup = 4 warps = 128 threads.
__device__ __forceinline__ uint32_t warpgroup_id() {
    return threadIdx.x / 128u;
}

// Warp id within the warpgroup (0..3).
__device__ __forceinline__ uint32_t warp_id_in_warpgroup() {
    return (threadIdx.x / 32u) % 4u;
}

// CTA rank within a cluster (only meaningful for cluster launches).
__device__ __forceinline__ uint32_t cluster_cta_rank() {
    uint32_t r;
    asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

// 26_cp_async_ca.cuh -- cp.async.ca.shared.global (L1-cached async copy)
//
// ARCH: sm_90a
//
// Per-thread async GMEM -> SMEM copy, cached at all levels (L1+L2).
// Non-TMA: each thread issues its own copy (4, 8, or 16 bytes).
// Completion tracked by cp.async.commit_group / wait_group (#28), a
// DIFFERENT namespace from the bulk cp.async.bulk.commit_group used by
// TMA (#25). Copy widths require proportional alignment.

#pragma once

// PTX:    9.7.10.28.3.1 (cp.async.ca)
//
#include <cstdint>

// 16-byte copy (most common, maximum throughput)
__device__ __forceinline__
void cp_async_ca_16(uint32_t smem_dst, const void* gmem_src) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_dst), "l"(gmem_src) : "memory");
}

// 8-byte copy
__device__ __forceinline__
void cp_async_ca_8(uint32_t smem_dst, const void* gmem_src) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 8;\n"
        :: "r"(smem_dst), "l"(gmem_src) : "memory");
}

// 4-byte copy
__device__ __forceinline__
void cp_async_ca_4(uint32_t smem_dst, const void* gmem_src) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_dst), "l"(gmem_src) : "memory");
}

// 16-byte copy with partial source (copies src_size bytes, zero-fills rest)
__device__ __forceinline__
void cp_async_ca_16_partial(uint32_t smem_dst, const void* gmem_src,
                            uint32_t src_size) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_dst), "l"(gmem_src), "r"(src_size) : "memory");
}

// 16-byte copy with predicated zero-fill (if pred=true, zero-fills entirely)
__device__ __forceinline__
void cp_async_ca_16_zfill(uint32_t smem_dst, const void* gmem_src,
                          bool zero_fill) {
    uint32_t src_size = zero_fill ? 0 : 16;
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16, %2;\n"
        :: "r"(smem_dst), "l"(gmem_src), "r"(src_size) : "memory");
}

// .L2::cache_hint variants. cache_policy is an opaque u64 built by
// createpolicy.fractional.L2::evict_*.b64 (or cuda::access_property).
// Useful for L2 eviction control on streaming GMEM -> SMEM paths.
__device__ __forceinline__
void cp_async_ca_16_l2hint(uint32_t smem_dst, const void* gmem_src,
                            uint64_t cache_policy) {
    asm volatile(
        "cp.async.ca.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
        :: "r"(smem_dst), "l"(gmem_src), "l"(cache_policy) : "memory");
}

__device__ __forceinline__
void cp_async_ca_16_l2hint_partial(uint32_t smem_dst, const void* gmem_src,
                                    uint32_t src_size, uint64_t cache_policy) {
    asm volatile(
        "cp.async.ca.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
        :: "r"(smem_dst), "l"(gmem_src), "r"(src_size), "l"(cache_policy)
        : "memory");
}

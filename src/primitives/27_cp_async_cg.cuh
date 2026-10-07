// 27_cp_async_cg.cuh -- cp.async.cg.shared.global (L1 bypass, 16 bytes)
//
// ARCH: sm_90a
//
// Per-thread async GMEM -> SMEM copy with L1 bypass (cached at L2 only).
// Always 16 bytes per issue. Faster than .ca for streaming workloads where
// data is consumed once.

#pragma once

// PTX:    9.7.10.28.3.1 (cp.async.cg)
//
#include <cstdint>

// 16-byte streaming copy (L2 only)
__device__ __forceinline__
void cp_async_cg_16(uint32_t smem_dst, const void* gmem_src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :: "r"(smem_dst), "l"(gmem_src) : "memory");
}

// 16-byte copy with explicit source-size cap. PTX supports a 2nd operand
// that caps actual bytes copied, padding the rest with zeros. Useful for
// boundary tiles. Two names provided (the body is identical).
__device__ __forceinline__
void cp_async_cg_16_masked(uint32_t smem_dst, const void* gmem_src,
                           uint32_t src_size) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(smem_dst), "l"(gmem_src), "r"(src_size) : "memory");
}

__device__ __forceinline__
void cp_async_cg_16_partial(uint32_t smem_dst, const void* gmem_src,
                            uint32_t src_size) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(smem_dst), "l"(gmem_src), "r"(src_size) : "memory");
}

// 16-byte copy with predicated zero-fill (if zero_fill, copy 0 bytes).
__device__ __forceinline__
void cp_async_cg_16_zfill(uint32_t smem_dst, const void* gmem_src,
                          bool zero_fill) {
  uint32_t src_size = zero_fill ? 0 : 16;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(smem_dst), "l"(gmem_src), "r"(src_size) : "memory");
}

// .L2::cache_hint variants for L1-bypass async copy.
__device__ __forceinline__
void cp_async_cg_16_l2hint(uint32_t smem_dst, const void* gmem_src,
                            uint64_t cache_policy) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
               :: "r"(smem_dst), "l"(gmem_src), "l"(cache_policy) : "memory");
}

__device__ __forceinline__
void cp_async_cg_16_l2hint_partial(uint32_t smem_dst, const void* gmem_src,
                                    uint32_t src_size, uint64_t cache_policy) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;\n"
               :: "r"(smem_dst), "l"(gmem_src), "r"(src_size), "l"(cache_policy)
               : "memory");
}

// Pure L2 prefetch hint for a global address (no SMEM write, no completion
// to track). Used to warm L2 with gather source rows AHEAD of the cp.async
// load warp, so its cp.async lands at L2 latency instead of DRAM. PTX
// 9.7.12.1 (prefetch.global.L2). Hint only -- may be dropped under pressure.
__device__ __forceinline__
void prefetch_global_l2(const void* gmem_src) {
  asm volatile("prefetch.global.L2 [%0];\n" :: "l"(gmem_src) : "memory");
}

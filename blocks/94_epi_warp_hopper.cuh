#pragma once
#if defined(PL_AGENTIC_SM90A)
// 94_epi_warp_hopper.cuh -- Hopper epilogue warp role
//
// ARCH: sm_90a
//
// Hopper one-warp epilogue role: convert an 8-element f32 fragment to
// 4 packed f16x2/bf16x2 regs, stmatrix.x4 to SMEM, fence.proxy.async,
// then TMA store a 16x16 output tile to GMEM.
//
// Composes #22 (tma_store), #25 (bulk_commit_wait), #34 (fence_proxy_async),
// #37 (bar_sync), #40 (stmatrix), #63 (cvt).
//
// Block functions (per code/PLAN.md "Block function signature contract"):
//
//   __device__ __forceinline__ void
//   epi_warp_hopper_f16_block(const CUtensorMap& tma_d, char* smem_buf,
//                             int tm, int tn,
//                             const float (&d)[8], int T);
//
//   __device__ __forceinline__ void
//   epi_warp_hopper_bf16_block(const CUtensorMap& tma_d, char* smem_buf,
//                              int tm, int tn,
//                              const float (&d)[8], int T);
//
// - tma_d: caller-built 2D output tensormap (matching swizzle).
// - smem_buf: caller-allocated SMEM staging buffer of >= EPI_BYTES bytes
//   (16*16*sizeof(half) = 512 B), aligned to 128.
// - tm/tn: tile coords in (m, n) units of EPI_M/EPI_N.
// - d: per-thread 8-element f32 accumulator.
// - T: lane id within the epilogue warp (0..31).
//
// Caller-collective on the 32-lane epilogue warp.
//
// Source: knowledge/building_blocks/epi_warp.md
// PTX:    9.7.10.24 (cvt), 9.7.16.5.16 (stmatrix), 9.7.10.28.5.3 (TMA store)

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "../primitives/23_tma_tensormap.cuh"
#include "../primitives/37_bar_sync.cuh"
#include "../primitives/34_fence_proxy_async.cuh"
#include "../primitives/40_stmatrix.cuh"
#include "../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../primitives/25_tma_async_group.cuh"
#include "../primitives/22_tma_store.cuh"

// Per-thread SMEM row pointer for stmatrix.x4 of a 16x16 row-major FP16 tile.
// Matrix id = T/8, row within matrix = T%8.
__device__ __forceinline__
uint32_t epi_warp_hopper_stmatrix_row_addr_16x16(uint32_t base,
                                                  uint32_t row_stride_bytes,
                                                  int T) {
  int mid = T / 8;
  int r   = (T % 8) + ((mid >= 2) ? 8 : 0);
  int c_off = (mid & 1) ? 16 : 0;
  return base + (uint32_t)r * row_stride_bytes + (uint32_t)c_off;
}

template <int EPI_M, int EPI_N>
__device__ __forceinline__
void epi_warp_hopper_f16_block(const CUtensorMap& tma_d, char* smem_buf,
                               int tm, int tn,
                               const float (&d)[8], int T) {
  uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem_buf));
  uint32_t r0 = cvt_f32x2_to_f16x2(d[0], d[1]);
  uint32_t r1 = cvt_f32x2_to_f16x2(d[2], d[3]);
  uint32_t r2 = cvt_f32x2_to_f16x2(d[4], d[5]);
  uint32_t r3 = cvt_f32x2_to_f16x2(d[6], d[7]);
  uint32_t row_stride = (uint32_t)EPI_N * 2;
  uint32_t addr = epi_warp_hopper_stmatrix_row_addr_16x16(sa, row_stride, T);
  stmatrix_x4(addr, r0, r1, r2, r3);
  asm volatile("bar.sync 1, 32;\n" ::: "memory");
  fence_proxy_async_shared_cta();
  if (T == 0) {
    tma_store_2d(&tma_d, tn * EPI_N, tm * EPI_M, sa);
    tma_store_commit_group();
    tma_store_wait_group<0>();
  }
}

template <int EPI_M, int EPI_N>
__device__ __forceinline__
void epi_warp_hopper_bf16_block(const CUtensorMap& tma_d, char* smem_buf,
                                int tm, int tn,
                                const float (&d)[8], int T) {
  uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem_buf));
  uint32_t r0 = cvt_f32x2_to_bf16x2(d[0], d[1]);
  uint32_t r1 = cvt_f32x2_to_bf16x2(d[2], d[3]);
  uint32_t r2 = cvt_f32x2_to_bf16x2(d[4], d[5]);
  uint32_t r3 = cvt_f32x2_to_bf16x2(d[6], d[7]);
  uint32_t row_stride = (uint32_t)EPI_N * 2;
  uint32_t addr = epi_warp_hopper_stmatrix_row_addr_16x16(sa, row_stride, T);
  stmatrix_x4(addr, r0, r1, r2, r3);
  asm volatile("bar.sync 1, 32;\n" ::: "memory");
  fence_proxy_async_shared_cta();
  if (T == 0) {
    tma_store_2d(&tma_d, tn * EPI_N, tm * EPI_M, sa);
    tma_store_commit_group();
    tma_store_wait_group<0>();
  }
}

#endif  // PL_AGENTIC_SM90A

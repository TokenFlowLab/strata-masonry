// 116_tma_load_stage.cuh -- arrive.expect_tx + TMA load(s) for one pipeline stage (1SM)
//
// ARCH: sm_90a
//
// One pipeline stage: register expected bytes on the stage's mbarrier, then
// issue one or more TMA loads targeting that mbarrier. The elected thread
// runs all of this.
//
// Two API styles are kept:
//   - cluster-scope variants (smem-first arg order): tma_load_stage_{1,2}tensor
//   - cta-scope variants (tmap-first arg order): tma_load_stage_2d, _ab

#pragma once

// PTX:    9.7.10.28.5.3 (cp.async.bulk.tensor), 9.7.15.16.14 (mbarrier.expect_tx)
//
#include <cstdint>
#include "../primitives/18_tma_load.cuh"
#include "../primitives/31_mbarrier_arrive_tx.cuh"

// -- cluster-scope, smem-first ----------------------------------------------

// Single-tensor variant (the common case: either A or B per stage).
__device__ __forceinline__
void tma_load_stage_1tensor(uint32_t mbar_smem, uint32_t expected_bytes,
                            uint32_t smem_dst, const void* tensormap,
                            int x, int y) {
  mbarrier_arrive_expect_tx(mbar_smem, expected_bytes);
  tma_load_2d(smem_dst, tensormap, mbar_smem, x, y);
}

// Dual-tensor variant (A and B per stage, common in GEMM mainloop).
__device__ __forceinline__
void tma_load_stage_2tensor(uint32_t mbar_smem, uint32_t expected_bytes_total,
                            uint32_t smem_dst_a, const void* tm_a,
                            int xa, int ya,
                            uint32_t smem_dst_b, const void* tm_b,
                            int xb, int yb) {
  mbarrier_arrive_expect_tx(mbar_smem, expected_bytes_total);
  tma_load_2d(smem_dst_a, tm_a, mbar_smem, xa, ya);
  tma_load_2d(smem_dst_b, tm_b, mbar_smem, xb, yb);
}

// -- cta-scope, tmap-first ---------------------------------------------------

// Single 2D TMA load stage (one tile, one mbarrier).
__device__ __forceinline__
void tma_load_stage_2d(const void* tensormap_ptr,
                       uint32_t smem_dst,
                       uint32_t mbar_smem_addr,
                       uint32_t expected_bytes,
                       int coord_x, int coord_y) {
  mbarrier_arrive_expect_tx(mbar_smem_addr, expected_bytes);
  tma_load_2d_cta(tensormap_ptr, smem_dst, mbar_smem_addr, coord_x, coord_y);
}

// Double-tile stage (A + B together, single mbarrier).
__device__ __forceinline__
void tma_load_stage_ab(const void* tma_a, const void* tma_b,
                       uint32_t smem_a, uint32_t smem_b,
                       uint32_t mbar_smem_addr,
                       uint32_t expected_bytes_a, uint32_t expected_bytes_b,
                       int coord_x_a, int coord_y_a,
                       int coord_x_b, int coord_y_b) {
  mbarrier_arrive_expect_tx(mbar_smem_addr,
                            expected_bytes_a + expected_bytes_b);
  tma_load_2d_cta(tma_a, smem_a, mbar_smem_addr, coord_x_a, coord_y_a);
  tma_load_2d_cta(tma_b, smem_b, mbar_smem_addr, coord_x_b, coord_y_b);
}

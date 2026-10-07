#pragma once
#if defined(PL_AGENTIC_SM90A)
// 126_epi_subtile_hopper.cuh -- Hopper epilogue sub-tile (regs -> SMEM)
//
// ARCH: sm_90a
//
// One sub-tile of the Hopper epilogue: the WGMMA accumulator already
// lives in registers (unlike Blackwell where it's in TMEM), so this
// composite skips the tcgen05.ld step. Steps:
//   1. type-convert FP32 accumulator -> output dtype (FP16/BF16/FP8)
//   2. stmatrix to SMEM with B128 swizzle
// The next composite (127_epi_tma_store) handles fence + TMA store.
//
// Constraints:
//   stmatrix expects packed register fragments matching its shape
//   SMEM target address must satisfy the swizzle's alignment
//
// Issuer: warp-group (128 threads), distributed work across warps.
// Source: knowledge/building_blocks/epi_warp.md
// PTX:    9.7.10.24 (cvt), 9.7.16.5.16 (stmatrix)
//
#include <cstdint>
#include "40_stmatrix.cuh"
#include "63_cvt_f32_to_f16_bf16.cuh"
#include "64_cvt_f32_to_fp8.cuh"

// ---------------------------------------------------------------------------
// Hopper epilogue sub-tile: convert F32 accumulator to FP16/BF16/FP8 and
// store to SMEM via stmatrix. The stmatrix target address must be swizzled
// to match the TMA store tensormap.
//
// One warp-collective (32 threads) processes one 8x16 FP16 sub-tile via
// stmatrix.x4 (4 x 8x8 tiles).
// ---------------------------------------------------------------------------

// Convert 8 f32 accum values to 4 packed f16x2 regs, then stmatrix.x4
__device__ __forceinline__
void epi_subtile_f32_to_f16_stmatrix(
    uint32_t smem_addr,
    float d0, float d1, float d2, float d3,
    float d4, float d5, float d6, float d7) {
    uint32_t r0 = cvt_f32x2_to_f16x2(d0, d1);
    uint32_t r1 = cvt_f32x2_to_f16x2(d2, d3);
    uint32_t r2 = cvt_f32x2_to_f16x2(d4, d5);
    uint32_t r3 = cvt_f32x2_to_f16x2(d6, d7);
    stmatrix_x4(smem_addr, r0, r1, r2, r3);
}

// Convert 8 f32 accum values to 4 packed bf16x2 regs, then stmatrix.x4
__device__ __forceinline__
void epi_subtile_f32_to_bf16_stmatrix(
    uint32_t smem_addr,
    float d0, float d1, float d2, float d3,
    float d4, float d5, float d6, float d7) {
    uint32_t r0 = cvt_f32x2_to_bf16x2(d0, d1);
    uint32_t r1 = cvt_f32x2_to_bf16x2(d2, d3);
    uint32_t r2 = cvt_f32x2_to_bf16x2(d4, d5);
    uint32_t r3 = cvt_f32x2_to_bf16x2(d6, d7);
    stmatrix_x4(smem_addr, r0, r1, r2, r3);
}

// Process a full 32-reg accumulator fragment in 4 stmatrix.x4 batches
// (m64n64 WGMMA output -> 4 stmatrix batches)
__device__ __forceinline__
void epi_f32_accum_to_f16_subtiles(
    uint32_t smem_addr_base,
    uint32_t smem_stride_bytes,
    float (&d)[32]) {
    // 4 groups of 8 f32 values -> 4 stmatrix.x4 calls
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        epi_subtile_f32_to_f16_stmatrix(
            smem_addr_base + g * smem_stride_bytes,
            d[g*8+0], d[g*8+1], d[g*8+2], d[g*8+3],
            d[g*8+4], d[g*8+5], d[g*8+6], d[g*8+7]);
    }
}

// BF16 variant: same 4-batch m64n64 accumulator drain, BF16 output.
__device__ __forceinline__
void epi_f32_accum_to_bf16_subtiles(
    uint32_t smem_addr_base,
    uint32_t smem_stride_bytes,
    float (&d)[32]) {
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        epi_subtile_f32_to_bf16_stmatrix(
            smem_addr_base + g * smem_stride_bytes,
            d[g*8+0], d[g*8+1], d[g*8+2], d[g*8+3],
            d[g*8+4], d[g*8+5], d[g*8+6], d[g*8+7]);
    }
}

#endif  // PL_AGENTIC_SM90A

#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 5_tcgen05_mma_fp4.cuh -- tcgen05.mma.cta_group::{1,2}.kind::{mxf4, mxf4nvf4}
//                          .block_scale.scale_vec::{1X,2X,4X}
//                          [d-tmem], a-{desc|tmem}, b-desc, idesc,
//                          [scale_A_tmem], [scale_B_tmem], enable-input-d
//
// ARCH: sm_100a
//
// Block-scaled FP4 MMA. Scale factors (scale_A, scale_B) live in TMEM at
// absolute addresses; the instruction consumes them alongside the A/B SMEM
// (or TMEM) descriptors. Atom K = 128 on sm_100a (K = 96 is sm_103a-only;
// see #6). Idesc uses Table 55 layout via #8 make_idesc_mxf4 /
// make_idesc_mxf4nvf4.
//
// Two scaling schemes (different scale-type / .block valid combinations):
//   .kind::mxf4      MX block scale. UE8M0 scale type only.
//                    Per ptxas: only block size 32 (the equivalent
//                    `.block16` is rejected for this kind).
//   .kind::mxf4nvf4  NVFP4 block scale. Either UE4M3 (idesc bit 23 = 0) or
//                    UE8M0 (bit 23 = 1) scale type.
//                    Both block sizes (16 and 32) are
//                    accepted.
//
// Wrappers provided here:
//   tcgen05_mma_mxf4_ss_1sm_block32 / _ss_2sm_block32        (mxf4 SS)
//   tcgen05_mma_mxf4nvf4_ss_1sm_block16 / _ss_2sm_block16    (nvf4 SS, .block16)
//   tcgen05_mma_mxf4nvf4_ss_1sm_block32 / _ss_2sm_block32    (nvf4 SS, .block32)
//   tcgen05_mma_mxf4_ts_1sm_block32                          (mxf4 TS, A-stat)
//   tcgen05_mma_mxf4nvf4_ts_1sm_block16                      (nvf4 TS, A-stat)
//
// enable-input-d: true -> D = A*B + D, false -> D = A*B.
// disable-output-lane mask is not exposed (block-scaled forms tend to use
// the predicate-only operand list per the canonical PTX syntax).
//
//
// Issuer: one thread per CTA (cta_group::1) or per CTA-pair (cta_group::2).
// PTX:    9.7.18.10.10.1 (mma syntax), 9.7.18.4.2 Table 55 (idesc)
//
#include <cstdint>

// mxf4, SMEM x SMEM, 1SM, .block32.
__device__ __forceinline__ void tcgen05_mma_mxf4_ss_1sm_block32(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X "
    "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

// mxf4, SMEM x SMEM, 2SM, .block32.
__device__ __forceinline__ void tcgen05_mma_mxf4_ss_2sm_block32(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X "
    "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

// mxf4nvf4, SMEM x SMEM, 1SM, .block16.
__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_ss_1sm_block16(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
    "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

// mxf4nvf4, SMEM x SMEM, 2SM, .block16.
__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_ss_2sm_block16(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
    "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

// ---------------------------------------------------------------------------
// Cross-block-size variants. Per ptxas:
//   .kind::mxf4      -- only block size 32 accepted; block16 rejected.
//                       (asm: .scale_vec::2X)
//   .kind::mxf4nvf4  -- both block sizes 16 and 32 accepted.
//                       (asm: .scale_vec::4X for block16, .scale_vec::2X
//                        for block32)
// So only `mxf4nvf4 + block32` is added here.
// ---------------------------------------------------------------------------

// mxf4nvf4, SS, 1SM, .block32.
__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_ss_1sm_block32(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X "
    "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

// mxf4nvf4, SS, 2SM, .block32.
__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_ss_2sm_block32(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X "
    "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

// ---------------------------------------------------------------------------
// TS form (A in TMEM, B in SMEM) for mxf4 / mxf4nvf4.
// ---------------------------------------------------------------------------

// mxf4, TS, 1SM, .block32.
__device__ __forceinline__ void tcgen05_mma_mxf4_ts_1sm_block32(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X "
    "[%0], [%1], %2, %3, [%4], [%5], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

// mxf4nvf4, TS, 1SM, .block16.
__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_ts_1sm_block16(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
    "[%0], [%1], %2, %3, [%4], [%5], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

// ---------------------------------------------------------------------------
// .sp (sparse-A) variants. Per PTX 9.7.18.10.10.2 syntax form 2 (block_scale,
// .kind::{mxf4, mxf4nvf4}): no {disable-output-lane} mask; new
// [sp-meta-tmem] operand inserted between b-desc and idesc.
// Per PTX 9.7.18.10.9.4: sparsity selector is ASSUMED 0 for these kinds.
// ---------------------------------------------------------------------------

// mxf4, SS, 1SM, .block32, sparse.
__device__ __forceinline__ void tcgen05_mma_mxf4_ss_1sm_block32_sparse(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t sp_meta_tmem, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %7, 0;\n\t"
    "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.scale_vec::2X "
    "[%0], %1, %2, [%3], %4, [%5], [%6], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
       "r"(sp_meta_tmem), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

// mxf4nvf4, SS, 1SM, .block16, sparse.
__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_ss_1sm_block16_sparse(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t sp_meta_tmem, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %7, 0;\n\t"
    "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
    "[%0], %1, %2, [%3], %4, [%5], [%6], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
       "r"(sp_meta_tmem), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

// mxf4nvf4, SS, 1SM, .block32, sparse.
__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_ss_1sm_block32_sparse(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t sp_meta_tmem, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %7, 0;\n\t"
    "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X "
    "[%0], %1, %2, [%3], %4, [%5], [%6], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
       "r"(sp_meta_tmem), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

// mxf4, TS, 1SM, .block32, sparse.
__device__ __forceinline__ void tcgen05_mma_mxf4_ts_1sm_block32_sparse(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t sp_meta_tmem, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %7, 0;\n\t"
    "tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.scale_vec::2X "
    "[%0], [%1], %2, [%3], %4, [%5], [%6], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b),
       "r"(sp_meta_tmem), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

// mxf4nvf4, TS, 1SM, .block16, sparse.
__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_ts_1sm_block16_sparse(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t sp_meta_tmem, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %7, 0;\n\t"
    "tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
    "[%0], [%1], %2, [%3], %4, [%5], [%6], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b),
       "r"(sp_meta_tmem), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

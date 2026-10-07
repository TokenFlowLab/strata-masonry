#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 3_tcgen05_mma_f16.cuh -- tcgen05.mma.cta_group::{1,2}.kind::f16
//                          [d-tmem], a-{desc|tmem}, b-desc, idesc,
//                          {disable-output-lane}, enable-input-d
//                          {, scale-input-d}
//
// ARCH: sm_100a
//
// Asynchronous MMA on the Blackwell tensor core. .kind::f16 covers both
// FP16 and BF16 A/B inputs (selected via idesc atype/btype) with FP32 or
// FP16 accumulator (selected via idesc dtype). Atom K = 16. Completion is
// reported via a later tcgen05.commit + mbarrier (see #11, #12); this
// instruction returns immediately.
//
// Operand layout:
//   [d-tmem]                  32-bit TMEM address of the destination /
//                             accumulator tile.
//   a-desc / [a-tmem]         A matrix source. SMEM 64-bit descriptor (SS
//                             form) or 32-bit TMEM address (TS form).
//   b-desc                    B matrix source: SMEM 64-bit descriptor only.
//                             (See #42 smem_desc_blackwell.)
//   idesc                     32-bit instruction descriptor (Table 53 via
//                             #8: M, N, atype, btype, dtype, transpose,
//                             negate).
//   {disable-output-lane}     4xb32 for cta_group::1, 8xb32 for cta_group::2;
//                             a 1-bit suppresses the D write for that lane.
//   enable-input-d            predicate; true -> D = A*B + D (accumulate),
//                             false -> D = A*B (first MMA in a K-loop).
//   scale-input-d (optional)  immediate in [0, 15]; D is scaled by
//                             2^-scale_input_d before the add. .kind::f16
//                             and .kind::tf32 only.
//
// Wrappers provided here:
//   tcgen05_mma_f16_ss_1sm / _ss_2sm                (SMEM x SMEM)
//   tcgen05_mma_f16_ts_1sm / _ts_2sm                (TMEM x SMEM, A-stationary)
//   tcgen05_mma_f16_ss<CTA_GROUP>                   (convenience: zero mask)
//   tcgen05_mma_f16_ss_1sm_scaled<SCALE_INPUT_D>    (SS + scale-input-d)
//   tcgen05_mma_f16_ss_2sm_scaled<SCALE_INPUT_D>
//   tcgen05_mma_f16_ss_1sm_sparse                   (.sp 2:4 sparse-A, SS, 1SM)
//   tcgen05_mma_f16_ts_1sm_sparse                   (.sp 2:4 sparse-A, TS, 1SM)
//
// .ashift (convolution weight-stationary shift) and .collector_usage
// (A-stationary collector buffer) modifiers are not exposed; add a variant
// if a convolution kernel needs them.
//
//
// Issuer: one thread per CTA (cta_group::1) or per CTA-pair (cta_group::2).
// PTX:    9.7.18.10.10.1 (mma syntax), 9.7.18.4.2 Table 53 (idesc)
//
#include <cstdint>

// 1SM (cta_group::1), SMEM x SMEM.
__device__ __forceinline__ void tcgen05_mma_f16_ss_1sm(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "
    "{%5, %6, %7, %8}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}

// 2SM (cta_group::2), SMEM x SMEM. mask is 8xb32.
__device__ __forceinline__ void tcgen05_mma_f16_ss_2sm(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3,
    uint32_t m4, uint32_t m5, uint32_t m6, uint32_t m7) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, "
    "{%5, %6, %7, %8, %9, %10, %11, %12}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3),
       "r"(m4), "r"(m5), "r"(m6), "r"(m7));
}

// Convenience: zero lane-disable mask (the common case).
template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_mma_f16_ss(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_mma_f16_ss: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    tcgen05_mma_f16_ss_1sm(tmem_c, desc_a, desc_b, idesc,
                            enable_input_d, 0, 0, 0, 0);
  } else {
    tcgen05_mma_f16_ss_2sm(tmem_c, desc_a, desc_b, idesc,
                            enable_input_d, 0, 0, 0, 0, 0, 0, 0, 0);
  }
}

// 1SM, A from TMEM, B from SMEM.
__device__ __forceinline__ void tcgen05_mma_f16_ts_1sm(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, "
    "{%5, %6, %7, %8}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}

// 2SM, A from TMEM, B from SMEM. mask is 8xb32.
__device__ __forceinline__ void tcgen05_mma_f16_ts_2sm(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3,
    uint32_t m4, uint32_t m5, uint32_t m6, uint32_t m7) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, "
    "{%5, %6, %7, %8, %9, %10, %11, %12}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3),
       "r"(m4), "r"(m5), "r"(m6), "r"(m7));
}

// ---------------------------------------------------------------------------
// scale-input-d variants. Per PTX 9.7.18.10.10.1: when present, D is scaled
// by 2^-scale_input_d before the add (D = A*B + D >> scale_input_d).
// scale_input_d is an immediate in [0, 15] and is legal only with
// .kind::f16 / .kind::tf32. Selected at compile time via a template parameter
// to keep the immediate operand encoded literally.
// ---------------------------------------------------------------------------

// 1SM SS with scale-input-d.
template <int SCALE_INPUT_D>
__device__ __forceinline__ void tcgen05_mma_f16_ss_1sm_scaled(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3) {
  static_assert(SCALE_INPUT_D >= 0 && SCALE_INPUT_D <= 15,
                "scale_input_d must be in [0, 15]");
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, "
    "{%5, %6, %7, %8}, p, %9;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3),
       "n"(SCALE_INPUT_D));
}

// 2SM SS with scale-input-d.
template <int SCALE_INPUT_D>
__device__ __forceinline__ void tcgen05_mma_f16_ss_2sm_scaled(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3,
    uint32_t m4, uint32_t m5, uint32_t m6, uint32_t m7) {
  static_assert(SCALE_INPUT_D >= 0 && SCALE_INPUT_D <= 15,
                "scale_input_d must be in [0, 15]");
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, "
    "{%5, %6, %7, %8, %9, %10, %11, %12}, p, %13;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3),
       "r"(m4), "r"(m5), "r"(m6), "r"(m7),
       "n"(SCALE_INPUT_D));
}

// ---------------------------------------------------------------------------
// .sp (sparse-A) variants. Per PTX 9.7.18.10.10.2 syntax form 1:
//   tcgen05.mma.sp.cta_group::1.kind::f16 [d-tmem], a-desc, b-desc,
//                                         [sp-meta-tmem], idesc,
//                                         { disable-output-lane },
//                                         enable-input-d{, scale-input-d};
// (and matching TS form with [a-tmem] in place of a-desc.)
// The sp-meta-tmem operand is a TMEM address whose contents specify the 2:4
// sparsity mask for matrix A (one row of metadata per row of A). The 2-bit
// sparsity selector lives in idesc bits 0-1 (set via #8 idesc_set_sparsity).
// Caller is responsible for storing valid mask values (see PTX 9.7.18.10.9.2
// for the legal 4-bit mask set; values outside that set yield UB).
// ---------------------------------------------------------------------------

// 1SM SS sparse, .kind::f16.
__device__ __forceinline__ void tcgen05_mma_f16_ss_1sm_sparse(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t sp_meta_tmem, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %5, 0;\n\t"
    "tcgen05.mma.sp.cta_group::1.kind::f16 [%0], %1, %2, [%3], %4, "
    "{%6, %7, %8, %9}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
       "r"(sp_meta_tmem), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}

// 1SM TS sparse (A in TMEM), .kind::f16.
__device__ __forceinline__ void tcgen05_mma_f16_ts_1sm_sparse(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t sp_meta_tmem, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %5, 0;\n\t"
    "tcgen05.mma.sp.cta_group::1.kind::f16 [%0], [%1], %2, [%3], %4, "
    "{%6, %7, %8, %9}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b),
       "r"(sp_meta_tmem), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}

// 1SM SS, lead-thread-predicated: only lane `lead`!=0 issues the mma (warp-
// specialized kernels elect one thread to drive the MMA).
__device__ __forceinline__ void tcgen05_mma_f16_ss_lead(uint32_t lead,
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p, q;\n\t"
    "setp.ne.b32 q, %0, 0;\n\t"
    "setp.ne.b32 p, %5, 0;\n\t"
    "@q tcgen05.mma.cta_group::1.kind::f16 [%1], %2, %3, %4, {%6, %7, %8, %9}, p;\n\t"
    "}\n"
    :: "r"(lead), "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u), "r"(0u), "r"(0u), "r"(0u), "r"(0u));
}

// 1SM TS (A in TMEM), lead-thread-predicated.
__device__ __forceinline__ void tcgen05_mma_f16_ts_1sm_lead(uint32_t lead,
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p, q;\n\t"
    "setp.ne.b32 q, %0, 0;\n\t"
    "setp.ne.b32 p, %5, 0;\n\t"
    "@q tcgen05.mma.cta_group::1.kind::f16 [%1], [%2], %3, %4, {%6, %7, %8, %9}, p;\n\t"
    "}\n"
    :: "r"(lead), "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u), "r"(0u), "r"(0u), "r"(0u), "r"(0u));
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

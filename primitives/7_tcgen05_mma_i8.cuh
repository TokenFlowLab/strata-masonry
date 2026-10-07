#pragma once
#if defined(PL_AGENTIC_SM100A)
// 7_tcgen05_mma_i8.cuh -- tcgen05.mma.cta_group::{1,2}.kind::i8
//                         [d-tmem], a-{desc|tmem}, b-desc, idesc,
//                         {disable-output-lane}, enable-input-d
//
// ARCH: sm_100a (SM100A-strict)
//
// Integer MMA. A and B are INT8 (signed atype=1 or unsigned atype=0, picked
// in idesc via #8 make_idesc_s8_s32 / _u8_s32); accumulator is INT32 (idesc
// dtype=2). Atom K = 128. Saturating add is available via idesc bit 3.
//
// `tcgen05.mma.kind::i8` was REMOVED on sm_103a (Blackwell Ultra), so this
// file is gated to SM100A only. Builds on GB300 (sm_103a) drop it via
// the Makefile's is_sm100a_only_file heuristic.
//
// Wrappers provided here:
//   tcgen05_mma_i8_ss_1sm / _ss_2sm    (SMEM x SMEM)
//   tcgen05_mma_i8_ts_1sm / _ts_2sm    (TMEM x SMEM, A-stationary)
//   tcgen05_mma_i8_ss<CTA_GROUP>       (convenience: zero mask)
//
// disable-output-lane mask: 4xb32 for cta_group::1, 8xb32 for cta_group::2.
// enable-input-d: true -> D = A*B + D, false -> D = A*B.
//
//
// Issuer: one thread per CTA (cta_group::1) or per CTA-pair (cta_group::2).
// PTX:    9.7.18.10.10.1 (mma syntax), 9.7.18.4.2 Table 53 (idesc)
//
#include <cstdint>

__device__ __forceinline__ void tcgen05_mma_i8_ss_1sm(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, "
    "{%5, %6, %7, %8}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}

__device__ __forceinline__ void tcgen05_mma_i8_ss_2sm(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3,
    uint32_t m4, uint32_t m5, uint32_t m6, uint32_t m7) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::2.kind::i8 [%0], %1, %2, %3, "
    "{%5, %6, %7, %8, %9, %10, %11, %12}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3),
       "r"(m4), "r"(m5), "r"(m6), "r"(m7));
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_mma_i8_ss(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_mma_i8_ss: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    tcgen05_mma_i8_ss_1sm(tmem_c, desc_a, desc_b, idesc,
                           enable_input_d, 0, 0, 0, 0);
  } else {
    tcgen05_mma_i8_ss_2sm(tmem_c, desc_a, desc_b, idesc,
                           enable_input_d, 0, 0, 0, 0, 0, 0, 0, 0);
  }
}

// 1SM, A from TMEM, B from SMEM (TS form).
__device__ __forceinline__ void tcgen05_mma_i8_ts_1sm(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::i8 [%0], [%1], %2, %3, "
    "{%5, %6, %7, %8}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}

// 2SM, A from TMEM, B from SMEM.
__device__ __forceinline__ void tcgen05_mma_i8_ts_2sm(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3,
    uint32_t m4, uint32_t m5, uint32_t m6, uint32_t m7) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::2.kind::i8 [%0], [%1], %2, %3, "
    "{%5, %6, %7, %8, %9, %10, %11, %12}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3),
       "r"(m4), "r"(m5), "r"(m6), "r"(m7));
}

// ---------------------------------------------------------------------------
// .sp (sparse-A) variants. Per PTX 9.7.18.10.10.2 syntax form 5 (.kind::i8):
// keeps {disable-output-lane} mask; new [sp-meta-tmem] operand inserted
// between b-desc and idesc. NB: .kind::i8 has no scale-input-d immediate
// (unlike .kind::f16/.kind::tf32). Sparsity selector MUST be 0 for i8 per
// PTX 9.7.18.10.9.4 (only one sub-column layout is legal).
// ---------------------------------------------------------------------------

// 1SM SS sparse, .kind::i8.
__device__ __forceinline__ void tcgen05_mma_i8_ss_1sm_sparse(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t sp_meta_tmem, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %5, 0;\n\t"
    "tcgen05.mma.sp.cta_group::1.kind::i8 [%0], %1, %2, [%3], %4, "
    "{%6, %7, %8, %9}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
       "r"(sp_meta_tmem), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}

// 1SM TS sparse, .kind::i8.
__device__ __forceinline__ void tcgen05_mma_i8_ts_1sm_sparse(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t sp_meta_tmem, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %5, 0;\n\t"
    "tcgen05.mma.sp.cta_group::1.kind::i8 [%0], [%1], %2, [%3], %4, "
    "{%6, %7, %8, %9}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b),
       "r"(sp_meta_tmem), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}

#endif  // PL_AGENTIC_SM100A

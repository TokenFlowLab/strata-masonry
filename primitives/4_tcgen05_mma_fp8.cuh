#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 4_tcgen05_mma_fp8.cuh -- tcgen05.mma.cta_group::{1,2}.kind::{f8f6f4,
//                          mxf8f6f4} [d-tmem], a-{desc|tmem}, b-desc, idesc
//                          [, {disable-output-lane}, enable-input-d
//                           | [scale_A_tmem], [scale_B_tmem], enable-input-d]
//
// ARCH: sm_100a
//
// Same MMA shape as the FP16 wrapper (#3) but with FP8 / FP6 / FP4 inputs.
// Two .kind families are covered:
//
//   .kind::f8f6f4    Non-scaled FP8/FP6/FP4 -> FP32 accumulator. Atom K = 128.
//                    A/B element type (E4M3=0, E5M2=1, E2M3=3, E3M2=4,
//                    E2M1=5) is encoded in idesc atype/btype (Table 53 via
//                    #8 make_idesc_e4m3_f32 / _e5m2_f32 / _fp4_f32 /
//                    _fp8_mixed_f32).
//
//   .kind::mxf8f6f4  Block-scaled FP8/FP6/FP4 with scale matrices in TMEM.
//                    Idesc uses Table 54 layout via #8 make_idesc_mxf8f6f4.
//                    Per ptxas, only block size 32 is accepted (`.block16`
//                    is rejected for this kind). The asm emits the
//                    `.scale_vec::1X` spelling (sm_100a-strict legal;
//                    `.block32` would require `-arch=sm_100f`). PTX
//                    9.7.18.10.10.1: `.block32` is aliased with
//                    `.scale_vec::1X` for `.kind::mxf8f6f4`. Scale type is
//                    UE8M0 (idesc bit 23 = 1).
//
// Wrappers provided here:
//   tcgen05_mma_fp8_ss_1sm / _ss_2sm                (.kind::f8f6f4 SS)
//   tcgen05_mma_fp8_ts_1sm / _ts_2sm                (.kind::f8f6f4 TS, A-stat)
//   tcgen05_mma_fp8_ss<CTA_GROUP>                   (convenience: zero mask)
//   tcgen05_mma_mxf8f6f4_ss_1sm_block32             (block-scaled SS, 1SM)
//   tcgen05_mma_mxf8f6f4_ss_2sm_block32             (block-scaled SS, 2SM)
//
// disable-output-lane mask: 4xb32 for cta_group::1, 8xb32 for cta_group::2.
// enable-input-d: true -> D = A*B + D, false -> D = A*B.
// .ashift / .collector_usage modifiers are not exposed.
//
//
// Issuer: one thread per CTA (cta_group::1) or per CTA-pair (cta_group::2).
// Source: knowledge/instructions/mma/tcgen05_mma.md
// PTX:    9.7.18.10.10.1 (mma syntax), 9.7.18.4.2 Tables 53 & 54 (idesc)
//
#include <cstdint>

__device__ __forceinline__ void tcgen05_mma_fp8_ss_1sm(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, "
    "{%5, %6, %7, %8}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}

__device__ __forceinline__ void tcgen05_mma_fp8_ss_2sm(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3,
    uint32_t m4, uint32_t m5, uint32_t m6, uint32_t m7) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, "
    "{%5, %6, %7, %8, %9, %10, %11, %12}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3),
       "r"(m4), "r"(m5), "r"(m6), "r"(m7));
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_mma_fp8_ss(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_mma_fp8_ss: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    tcgen05_mma_fp8_ss_1sm(tmem_c, desc_a, desc_b, idesc,
                            enable_input_d, 0, 0, 0, 0);
  } else {
    tcgen05_mma_fp8_ss_2sm(tmem_c, desc_a, desc_b, idesc,
                            enable_input_d, 0, 0, 0, 0, 0, 0, 0, 0);
  }
}

// 1SM, A from TMEM, B from SMEM (TS form).
__device__ __forceinline__ void tcgen05_mma_fp8_ts_1sm(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, "
    "{%5, %6, %7, %8}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}

// 2SM, A from TMEM, B from SMEM.
__device__ __forceinline__ void tcgen05_mma_fp8_ts_2sm(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3,
    uint32_t m4, uint32_t m5, uint32_t m6, uint32_t m7) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], %2, %3, "
    "{%5, %6, %7, %8, %9, %10, %11, %12}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3),
       "r"(m4), "r"(m5), "r"(m6), "r"(m7));
}

// ---------------------------------------------------------------------------
// .kind::mxf8f6f4 -- block-scaled FP8/FP6/FP4 (Table 54 idesc layout).
// Scale factors live in TMEM at scale_A_tmem / scale_B_tmem; the instruction
// consumes them alongside the A/B operands. Use idesc from
// make_idesc_mxf8f6f4 (#8). Per ptxas, `.kind::mxf8f6f4` accepts only
// block size 32 (the equivalent `.block16` is rejected). The asm string
// uses `.scale_vec::1X` (sm_100a-strict legal); per PTX 9.7.18.10.10.1,
// this is aliased with `.block32` for `.kind::mxf8f6f4`. The wrapper
// name keeps the user-facing `_block32` suffix.
// ---------------------------------------------------------------------------

// mxf8f6f4, SS, 1SM, .block32.
__device__ __forceinline__ void tcgen05_mma_mxf8f6f4_ss_1sm_block32(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
    "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

// mxf8f6f4, SS, 2SM, .block32.
__device__ __forceinline__ void tcgen05_mma_mxf8f6f4_ss_2sm_block32(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X "
    "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

// ---------------------------------------------------------------------------
// .sp (sparse-A) variants. Per PTX 9.7.18.10.10.2:
//   form 1 (no block scale, .kind::f8f6f4): keeps {disable-output-lane} mask.
//   form 2 (block_scale, .kind::mxf8f6f4):  drops the mask, keeps scale-A,B.
// New operand: [sp-meta-tmem] inserted after b-desc and before idesc.
// Per PTX 9.7.18.10.9.4: sparsity selector MUST be 0 for .kind::f8f6f4 and
// is ASSUMED 0 for .kind::mxf8f6f4. Caller stages valid 2:4 metadata in
// TMEM (mask values per PTX 9.7.18.10.9.2).
// ---------------------------------------------------------------------------

// 1SM SS sparse, .kind::f8f6f4.
__device__ __forceinline__ void tcgen05_mma_fp8_ss_1sm_sparse(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t sp_meta_tmem, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %5, 0;\n\t"
    "tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [%0], %1, %2, [%3], %4, "
    "{%6, %7, %8, %9}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
       "r"(sp_meta_tmem), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}

// 1SM TS sparse, .kind::f8f6f4.
__device__ __forceinline__ void tcgen05_mma_fp8_ts_1sm_sparse(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t sp_meta_tmem, uint32_t idesc,
    bool enable_input_d,
    uint32_t m0, uint32_t m1, uint32_t m2, uint32_t m3) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %5, 0;\n\t"
    "tcgen05.mma.sp.cta_group::1.kind::f8f6f4 [%0], [%1], %2, [%3], %4, "
    "{%6, %7, %8, %9}, p;\n\t"
    "}\n"
    :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b),
       "r"(sp_meta_tmem), "r"(idesc),
       "r"(enable_input_d ? 1u : 0u),
       "r"(m0), "r"(m1), "r"(m2), "r"(m3));
}

// 1SM SS sparse, .kind::mxf8f6f4.block_scale.scale_vec::1X.
__device__ __forceinline__ void tcgen05_mma_mxf8f6f4_ss_1sm_block32_sparse(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t sp_meta_tmem, uint32_t idesc,
    uint32_t tmem_scale_a, uint32_t tmem_scale_b, bool enable_input_d) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "setp.ne.b32 p, %7, 0;\n\t"
    "tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
    "[%0], %1, %2, [%3], %4, [%5], [%6], p;\n\t"
    "}\n"
    :: "r"(tmem_c), "l"(desc_a), "l"(desc_b),
       "r"(sp_meta_tmem), "r"(idesc),
       "r"(tmem_scale_a), "r"(tmem_scale_b),
       "r"(enable_input_d ? 1u : 0u));
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

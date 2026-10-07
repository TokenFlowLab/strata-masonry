#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 79_tcgen05_mma_ws_f16.cuh -- tcgen05.mma.ws.cta_group::1.kind::f16
//                              [d-tmem], a-{desc|tmem}, b-desc, idesc,
//                              enable-input-d {, zero-column-mask-desc}
//
// ARCH: sm_100a
//
// Weight-stationary tcgen05 MMA. cta_group::1 ONLY (the ISA defines no 2SM
// .ws form) -- do not mix with cta_group::2 tcgen05 in one kernel. Atom
// K = 16 for .kind::f16; M in {32, 64, 128}; N in {64, 128, 256}.
//
// Why we use it (VSA): with M=64, .ws selects the ISA's "2x2" Data Path
// Layout E -- ALL 128 TMEM lanes are active and the D(64xN) accumulator is
// DUAL-PACKED into N/2 TMEM columns:
//   element (row m, col n) -> lane m + 64*(n div (N/2)), tmem col n mod (N/2)
// i.e. lanes 0-63 hold D columns [0, N/2) and lanes 64-127 hold D columns
// [N/2, N), both re-based at the D address's column. (Contrast the plain
// tcgen05.mma M=64 "Layout F", which uses HALF the lanes and N full columns.)
// So m64n256 produces S(64x256) in 128 TMEM cols -- one instruction per
// K=16 step, full-datapath.
//
// TS form: the A operand is read PER DATAPATH HALF -- half h (computing D
// cols for B rows [h*N/2, (h+1)*N/2)) reads its A rows from lanes 64h..64h+63.
// Storing DIFFERENT A halves turns one m64n256k16 issue into two independent
// m64n128k16 GEMMs (FMHA P@V with a dual-packed P).
//
// Operands vs the plain form (#3): NO {disable-output-lane} vector; the
// trailing operand is a zero-column-mask descriptor (0 = disabled). idesc is
// the ordinary Table 53 word (#8) with the .ws max-B-reuse bits 30-31 = 0.
//
// Ref: PTX ISA 9.4 sec 9.7.18.10.10.3 (syntax), 9.7.18.10.5 (Layout E),
// Figures 219/220 (lane maps); flashinfer blk64 utils.h (shipping user).

#include <cstdint>

// SS form: A from SMEM (64-bit descriptor), B from SMEM.
__device__ __forceinline__ void tcgen05_mma_ws_f16_ss_1sm(
    uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, 0;\n\t"
      "}\n"
      :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc),
         "r"(enable_input_d ? 1u : 0u));
}

// Predicated issuer form.  `issue` is normally the result of elect.sync and
// rides on the instruction, avoiding a divergent control-flow wrapper around
// the single-thread tcgen05 issue.
__device__ __forceinline__ void tcgen05_mma_ws_f16_ss_1sm_predicated(
    uint32_t issue, uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, bool enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p, q;\n\t"
      "setp.ne.b32 q, %0, 0;\n\t"
      "setp.ne.b32 p, %5, 0;\n\t"
      "@q tcgen05.mma.ws.cta_group::1.kind::f16 "
      "[%1], %2, %3, %4, p, 0;\n\t"
      "}\n"
      :: "r"(issue), "r"(tmem_d), "l"(desc_a), "l"(desc_b),
         "r"(idesc), "r"(enable_input_d ? 1u : 0u));
}

// TS form: A from TMEM (32-bit address; per-half lane read, see header).
__device__ __forceinline__ void tcgen05_mma_ws_f16_ts_1sm(
    uint32_t tmem_d, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc,
    bool enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [%1], %2, %3, p, 0;\n\t"
      "}\n"
      :: "r"(tmem_d), "r"(tmem_a), "l"(desc_b), "r"(idesc),
         "r"(enable_input_d ? 1u : 0u));
}

// TS form, issue-predicated: the mma is issued only where `issue` != 0 (the
// elected lane), so a whole warp can call it without a branch.
__device__ __forceinline__ void tcgen05_mma_ws_f16_ts_1sm_predicated(
    uint32_t issue, uint32_t tmem_d, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, bool enable_input_d) {
  asm volatile(
      "{\n\t"
      ".reg .pred p, q;\n\t"
      "setp.ne.b32 q, %0, 0;\n\t"
      "setp.ne.b32 p, %5, 0;\n\t"
      "@q tcgen05.mma.ws.cta_group::1.kind::f16 "
      "[%1], [%2], %3, %4, p, 0;\n\t"
      "}\n"
      :: "r"(issue), "r"(tmem_d), "r"(tmem_a), "l"(desc_b),
         "r"(idesc), "r"(enable_input_d ? 1u : 0u));
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

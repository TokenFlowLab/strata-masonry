// 40_stmatrix.cuh -- stmatrix.sync.aligned.{x1,x2,x4}.m8n8.shared.b16[.trans]
//
// ARCH: sm_90a
//
// Warp-collective store of 8x8 b16 matrix fragments from registers to SMEM
// in CuTe / CUTLASS-compatible layout. Inverse of ldmatrix (#39).
//
// Key epilogue building block: accumulator regs -> stmatrix -> fence -> TMA store.
//
// x4 has two input forms:
//   - array-reference form `const uint32_t (&r)[4]`     (compact)
//   - four separate `uint32_t` regs                      (explicit)
// Both overloads emit the same instruction.

#pragma once

// Source: knowledge/instructions/smem/stmatrix.md
// PTX:    9.7.16.5.16 (stmatrix)
//
#include <cstdint>

// -- non-transposed ----------------------------------------------------------

// Store 1 x 8x8 b16 matrix.
__device__ __forceinline__
void stmatrix_x1(uint32_t smem_addr, uint32_t r0) {
  asm volatile(
    "stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n"
    :: "r"(smem_addr), "r"(r0) : "memory");
}

// Store 2 x 8x8 b16 matrices.
__device__ __forceinline__
void stmatrix_x2(uint32_t smem_addr, uint32_t r0, uint32_t r1) {
  asm volatile(
    "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n"
    :: "r"(smem_addr), "r"(r0), "r"(r1) : "memory");
}

// Store 4 x 8x8 b16 matrices (array-reference form).
__device__ __forceinline__
void stmatrix_x4(uint32_t smem_addr, const uint32_t (&r)[4]) {
  asm volatile(
    "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
    "[%0], {%1, %2, %3, %4};\n"
    :: "r"(smem_addr),
       "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}

// Store 4 x 8x8 b16 matrices (four-reg form).
__device__ __forceinline__
void stmatrix_x4(uint32_t smem_addr, uint32_t r0, uint32_t r1,
                 uint32_t r2, uint32_t r3) {
  asm volatile(
    "stmatrix.sync.aligned.x4.m8n8.shared.b16 "
    "[%0], {%1, %2, %3, %4};\n"
    :: "r"(smem_addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

// -- transposed --------------------------------------------------------------

__device__ __forceinline__
void stmatrix_x4_trans(uint32_t smem_addr, const uint32_t (&r)[4]) {
  asm volatile(
    "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 "
    "[%0], {%1, %2, %3, %4};\n"
    :: "r"(smem_addr),
       "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}

// Store 1 x 8x8 b16 matrix transposed (symmetry with ldmatrix_x1_trans).
__device__ __forceinline__
void stmatrix_x1_trans(uint32_t smem_addr, uint32_t r0) {
  asm volatile(
    "stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n"
    :: "r"(smem_addr), "r"(r0) : "memory");
}

// Store 2 x 8x8 b16 matrices transposed.
__device__ __forceinline__
void stmatrix_x2_trans(uint32_t smem_addr, uint32_t r0, uint32_t r1) {
  asm volatile(
    "stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n"
    :: "r"(smem_addr), "r"(r0), "r"(r1) : "memory");
}

// ============================================================================
// .m16n8 .b8 form (Blackwell+, FP8 inference epilogues).
//
// Per PTX 9.7.16.5.16, .m16n8 is valid only with .b8 type, .trans is
// mandatory (col-major store). .num in {.x1, .x2, .x4}.
// Each thread provides 1/2/4 source regs (one per matrix); each register
// packs 4 8-bit elements (e0..e3).
//
// Target: Type .b8 with stmatrix requires sm_100a / sm_103a / sm_120a /
//         family-spec sm_100f or higher.
// ============================================================================
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

// Asm bodies gated on __CUDA_ARCH__ >= 1000 (see #39 for rationale).
__device__ __forceinline__
void stmatrix_x1_trans_b8(uint32_t smem_addr, uint32_t r0) {
#if __CUDA_ARCH__ >= 1000
  asm volatile(
    "stmatrix.sync.aligned.m16n8.x1.trans.shared::cta.b8 [%0], {%1};\n"
    :: "r"(smem_addr), "r"(r0) : "memory");
#else
  (void)smem_addr; (void)r0;
#endif
}

__device__ __forceinline__
void stmatrix_x2_trans_b8(uint32_t smem_addr, uint32_t r0, uint32_t r1) {
#if __CUDA_ARCH__ >= 1000
  asm volatile(
    "stmatrix.sync.aligned.m16n8.x2.trans.shared::cta.b8 [%0], {%1, %2};\n"
    :: "r"(smem_addr), "r"(r0), "r"(r1) : "memory");
#else
  (void)smem_addr; (void)r0; (void)r1;
#endif
}

__device__ __forceinline__
void stmatrix_x4_trans_b8(uint32_t smem_addr, const uint32_t (&r)[4]) {
#if __CUDA_ARCH__ >= 1000
  asm volatile(
    "stmatrix.sync.aligned.m16n8.x4.trans.shared::cta.b8 "
    "[%0], {%1, %2, %3, %4};\n"
    :: "r"(smem_addr),
       "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
#else
  (void)smem_addr; (void)r;
#endif
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

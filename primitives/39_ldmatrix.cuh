// 39_ldmatrix.cuh -- ldmatrix.sync.aligned.{x1,x2,x4}.m8n8.shared.b16[.trans]
//
// ARCH: sm_90a
//
// Warp-collective load of 8x8 b16 matrix fragments from SMEM into registers
// laid out for mma.sync / wgmma operand positions. Each lane provides its
// row's SMEM address.
//
//   .x1 = 1 register/thread, .x2 = 2, .x4 = 4
//   .trans = transpose the 8x8 tile at load time (swap rows and columns).
//
// Two output forms for x4 / x4_trans:
//   - array-reference form `uint32_t (&r)[4]`           (compact)
//   - four separate `uint32_t&` regs                     (explicit)
// Both overloads emit the same instruction.

#pragma once

// Source: knowledge/instructions/smem/ldmatrix.md
// PTX:    9.7.16.5.15 (ldmatrix)
//
#include <cstdint>

// -- non-transposed ----------------------------------------------------------

// Load 1 x 8x8 b16 matrix.
__device__ __forceinline__
void ldmatrix_x1(uint32_t& r, uint32_t smem_addr) {
  asm volatile(
    "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
    : "=r"(r) : "r"(smem_addr));
}

// Load 2 x 8x8 b16 matrices.
__device__ __forceinline__
void ldmatrix_x2(uint32_t& r0, uint32_t& r1, uint32_t smem_addr) {
  asm volatile(
    "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
    : "=r"(r0), "=r"(r1) : "r"(smem_addr));
}

// Load 4 x 8x8 b16 matrices (array-reference form).
__device__ __forceinline__
void ldmatrix_x4(uint32_t (&r)[4], uint32_t smem_addr) {
  asm volatile(
    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr));
}

// Load 4 x 8x8 b16 matrices (four-reg form).
__device__ __forceinline__
void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                 uint32_t smem_addr) {
  asm volatile(
    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr));
}

// -- transposed --------------------------------------------------------------

__device__ __forceinline__
void ldmatrix_x1_trans(uint32_t& r, uint32_t smem_addr) {
  asm volatile(
    "ldmatrix.sync.aligned.x1.m8n8.trans.shared.b16 {%0}, [%1];\n"
    : "=r"(r) : "r"(smem_addr));
}

__device__ __forceinline__
void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1, uint32_t smem_addr) {
  asm volatile(
    "ldmatrix.sync.aligned.x2.m8n8.trans.shared.b16 {%0, %1}, [%2];\n"
    : "=r"(r0), "=r"(r1) : "r"(smem_addr));
}

__device__ __forceinline__
void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t smem_addr) {
  asm volatile(
    "ldmatrix.sync.aligned.x4.m8n8.trans.shared.b16 "
    "{%0, %1, %2, %3}, [%4];\n"
    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr));
}

__device__ __forceinline__
void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                       uint32_t smem_addr) {
  asm volatile(
    "ldmatrix.sync.aligned.x4.m8n8.trans.shared.b16 "
    "{%0, %1, %2, %3}, [%4];\n"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr));
}

// ============================================================================
// .m16n16 .b8 form (Blackwell+, FP8 inference epilogues).
//
// Per PTX 9.7.16.5.15, .b8 is valid only with .m16n16 shape, and .trans is
// mandatory. .num is restricted to {.x1, .x2}.
//   .x1 -> 2 destination regs/thread
//   .x2 -> 4 destination regs/thread
//
// Each thread receives 4 8-bit elements per matrix per dest register
// (PTX 9.7.16.5.15, "For matrix shape 16x16, two destination registers
//  r0 and r1 of type .b32 must be specified and in each register four
//  8-bit elements are loaded.").
//
// Target: Type .b8 with ldmatrix requires sm_100a / sm_103a / sm_120a /
//         family-spec sm_100f or higher.
// ============================================================================
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

// Asm bodies are gated on __CUDA_ARCH__ >= 1000 so dual-gencode passes that
// target sm_90a (where .m16n16 / .b8 don't exist) compile cleanly. The .cuh
// file-level guard above keeps the wrapper out of pure-Hopper builds; the
// __CUDA_ARCH__ guard handles dual-arch passes.
__device__ __forceinline__
void ldmatrix_x1_trans_b8(uint32_t& r0, uint32_t& r1, uint32_t smem_addr) {
#if __CUDA_ARCH__ >= 1000
  asm volatile(
    "ldmatrix.sync.aligned.m16n16.x1.trans.shared::cta.b8 "
    "{%0, %1}, [%2];\n"
    : "=r"(r0), "=r"(r1) : "r"(smem_addr));
#else
  (void)smem_addr; r0 = 0; r1 = 0;
#endif
}

__device__ __forceinline__
void ldmatrix_x2_trans_b8(uint32_t (&r)[4], uint32_t smem_addr) {
#if __CUDA_ARCH__ >= 1000
  asm volatile(
    "ldmatrix.sync.aligned.m16n16.x2.trans.shared::cta.b8 "
    "{%0, %1, %2, %3}, [%4];\n"
    : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr));
#else
  (void)smem_addr; r[0] = 0; r[1] = 0; r[2] = 0; r[3] = 0;
#endif
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

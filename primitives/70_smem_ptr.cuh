// 70_smem_ptr.cuh -- generic ptr -> SMEM 32-bit address.
//
// ARCH: any (sm_80+ has __cvta_generic_to_shared)
//
// Wraps the CUDA built-in `__cvta_generic_to_shared` that converts a
// generic device pointer into the 32-bit shared-memory address that
// PTX SMEM operands (mbarrier, ldmatrix, stmatrix, tma, tcgen05.*)
// require. Not strictly a PTX instruction -- it lowers to the
// `cvta.to.shared.u32` PTX op -- but it's the canonical entry point
// every block / composite / test reaches for, so we house it as a
// numbered primitive instead of duplicating it per file.
//
// PTX:    7.7.6 Address Operands - cvta.to.shared
//
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__
uint32_t smem_ptr_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Vector SMEM load used by register-fragment consumers. Keeping the 32-bit
// shared address explicit lets warp-uniform callers retain the base on the
// uniform datapath while issuing one 128-bit LDS transaction.
__device__ __forceinline__
float4 lds_f32x4(uint32_t smem_addr) {
  float4 out;
  asm volatile("ld.shared::cta.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(out.x), "=f"(out.y), "=f"(out.z), "=f"(out.w)
               : "r"(smem_addr)
               : "memory");
  return out;
}

// Volatile f32 store to a 32-bit SMEM address (from smem_ptr_u32). Two effects:
//   1. ORDERING (always-valid): `volatile` + "memory" pin the store so ptxas cannot sink it
//      past a following bar arrive -- use this wherever you publish-then-signal.
//   2. PIPE choice (NOT a default): it lowers to STS (routes via MIO) instead of a generic
//      `*ptr = val` (ST.E via L1TEX). MIO is faster ONLY where that pipe has slack -- measured
//      +24 mha-long but -31 gqa-mid, so callers GATE it (FMHA uses it under MHA, keeps the
//      generic store for GQA). Do not use blindly; pick per pipe slack.
__device__ __forceinline__
void sts_f32(uint32_t smem_addr, float val) {
  asm volatile("st.shared.f32 [%0], %1;" :: "r"(smem_addr), "f"(val) : "memory");
}

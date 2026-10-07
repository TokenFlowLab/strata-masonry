// 51_atom_shared.cuh -- atom.shared::{cta,cluster}.{add,exch}.{u32,f32,b32}
//
// ARCH: sm_90a
//
// Atomic adds / exchanges on SMEM.
//   .shared::cta     -- CTA-local atomic (within one CTA's SMEM).
//   .shared::cluster -- cross-CTA atomic on cluster-shared mbarriers or
//                       distributed-SMEM counters (requires sm_90a+).

#pragma once

// PTX:    9.7.15.5 (atom.shared)
//
#include <cstdint>

// -- CTA scope ---------------------------------------------------------------

__device__ __forceinline__
uint32_t atom_shared_cta_add_u32(uint32_t smem_addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.shared::cta.add.u32 %0, [%1], %2;\n"
               : "=r"(old) : "r"(smem_addr), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__
float atom_shared_cta_add_f32(uint32_t smem_addr, float v) {
  float old;
  asm volatile("atom.shared::cta.add.f32 %0, [%1], %2;\n"
               : "=f"(old) : "r"(smem_addr), "f"(v) : "memory");
  return old;
}

__device__ __forceinline__
uint32_t atom_shared_cta_exch_b32(uint32_t smem_addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.shared::cta.exch.b32 %0, [%1], %2;\n"
               : "=r"(old) : "r"(smem_addr), "r"(v) : "memory");
  return old;
}

// -- cluster scope -----------------------------------------------------------

__device__ __forceinline__
uint32_t atom_shared_cluster_add_u32(uint32_t smem_addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.shared::cluster.add.u32 %0, [%1], %2;\n"
               : "=r"(old) : "r"(smem_addr), "r"(v) : "memory");
  return old;
}

// -- CTA scope: cas / min / max ---------------------------------------------

__device__ __forceinline__
uint32_t atom_shared_cta_cas_b32(uint32_t smem_addr, uint32_t cmp, uint32_t v) {
  uint32_t old;
  asm volatile("atom.shared::cta.cas.b32 %0, [%1], %2, %3;\n"
               : "=r"(old) : "r"(smem_addr), "r"(cmp), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__
uint32_t atom_shared_cta_min_u32(uint32_t smem_addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.shared::cta.min.u32 %0, [%1], %2;\n"
               : "=r"(old) : "r"(smem_addr), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__
uint32_t atom_shared_cta_max_u32(uint32_t smem_addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.shared::cta.max.u32 %0, [%1], %2;\n"
               : "=r"(old) : "r"(smem_addr), "r"(v) : "memory");
  return old;
}

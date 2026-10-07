// 67_mapa.cuh -- mapa + getctarank: cluster-shared address remap.
//
// ARCH: sm_90a+
//
// `mapa`        : translate a local SMEM address in CTA X to the address
//                 that CTA Y would see for the same offset. Used to
//                 arrive on a peer CTA's mbarrier, address distributed
//                 SMEM, etc.
// `getctarank`  : inverse. Given an SMEM address, return the rank of
//                 the CTA that owns it.
//
// Per-thread instructions; no .sync.aligned.

#pragma once
#if defined(PL_AGENTIC_SM90A) || defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

// PTX:    9.7.10.26 (mapa), 9.7.10.27 (getctarank)
//
#include <cstdint>

// =====================================================================
// mapa
// =====================================================================

// mapa.shared::cluster.u32 d, a, b
//   Translate a 32-bit SMEM address (`local_smem_addr`) in this CTA to
//   the address peer CTA `target_rank` would see for the same offset.
__device__ __forceinline__
uint32_t mapa_shared_cluster_u32(uint32_t local_smem_addr,
                                 uint32_t target_rank) {
  uint32_t remote;
  asm("mapa.shared::cluster.u32 %0, %1, %2;\n"
      : "=r"(remote)
      : "r"(local_smem_addr), "r"(target_rank));
  return remote;
}

// mapa.shared::cluster.u64 d, a, b
//   64-bit pointer variant. Use when the source address is a 64-bit
//   shared register (e.g. a generic-cvted SMEM pointer).
__device__ __forceinline__
uint64_t mapa_shared_cluster_u64(uint64_t local_smem_addr,
                                 uint32_t target_rank) {
  uint64_t remote;
  asm("mapa.shared::cluster.u64 %0, %1, %2;\n"
      : "=l"(remote)
      : "l"(local_smem_addr), "r"(target_rank));
  return remote;
}

// mapa.u64 d, a, b
//   Generic-address variant. Both input and output are generic 64-bit
//   addresses (no .shared::cluster qualifier).
__device__ __forceinline__
uint64_t mapa_u64(uint64_t local_generic_addr,
                  uint32_t target_rank) {
  uint64_t remote;
  asm("mapa.u64 %0, %1, %2;\n"
      : "=l"(remote)
      : "l"(local_generic_addr), "r"(target_rank));
  return remote;
}

// =====================================================================
// getctarank
// =====================================================================

// getctarank.shared::cluster.u32 d, a
//   Return the rank of the CTA that owns the 32-bit SMEM address `a`.
__device__ __forceinline__
uint32_t getctarank_shared_cluster_u32(uint32_t smem_addr) {
  uint32_t rank;
  asm("getctarank.shared::cluster.u32 %0, %1;\n"
      : "=r"(rank)
      : "r"(smem_addr));
  return rank;
}

// getctarank.u64 d, a
//   Return the rank of the CTA that owns the 64-bit generic address `a`.
__device__ __forceinline__
uint32_t getctarank_u64(uint64_t generic_addr) {
  uint32_t rank;
  asm("getctarank.u64 %0, %1;\n"
      : "=r"(rank)
      : "l"(generic_addr));
  return rank;
}

#endif  // PL_AGENTIC_SM90A || PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

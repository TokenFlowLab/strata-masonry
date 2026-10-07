// 18_tma_load.cuh -- cp.async.bulk.tensor.{2d,3d}.shared::{cta,cluster}
//
// ARCH: sm_90a
//
// Asynchronous 2D TMA load of a tile described by a CUtensorMap.
// One thread issues this (the elected load-warp thread). Returns immediately;
// completion is signaled when the mbarrier's tx-count reaches zero.
//
// Before calling: the issuing thread must register the expected byte count
// via mbarrier.arrive.expect_tx (#31).
//
// Two destination scopes:
//   .shared::cta     -- CTA-local SMEM (no cluster required).
//   .shared::cluster -- cluster-wide SMEM (required for multicast / 2SM).
//
// Two argument-ordering conventions are provided (ours / theirs):
//   tma_load_2d(smem_dst, tensormap_ptr, mbar, x, y)         smem-first
//   tma_load_2d_{cta,cluster}(tensormap_ptr, smem_dst, mbar, x, y)
//                                                            tmap-first
//
// coord_x is the fast dim (columns), coord_y is the slow dim (rows).

#pragma once

// Source: knowledge/instructions/tma/tma_load.md
// PTX:    9.7.10.28.5.3 (cp.async.bulk.tensor)
//
#include <cuda.h>
#include <cstdint>

// Bulk copy from CTA-local shared memory to a cluster-shared destination.
// The destination and completion barrier must already be remote DSMEM
// addresses (normally produced by mapa.shared::cluster, primitive #67).
// The caller arms remote_mbar for `bytes` before issuing this operation.
// This is the non-tensor bulk-copy primitive used by FA4's 2-CTA dS relay.
__device__ __forceinline__
void cpasync_bulk_s2cluster(uint32_t cluster_dst, uint32_t smem_src,
                           uint32_t bytes, uint32_t remote_mbar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.shared::cta"
      ".mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
      :: "r"(cluster_dst), "r"(smem_src), "r"(bytes), "r"(remote_mbar)
      : "memory");
}

// Non-tensor bulk copy of `bytes` contiguous bytes from global to CTA shared
// memory, completing (complete_tx) on `mbar_smem`, which the caller armed
// for `bytes`. For flat rows (e.g. per-block LSE / Delta vectors) that do not
// need a tensor map.
__device__ __forceinline__
void cpasync_bulk_load_mbarrier(uint32_t smem_dst, const void* gmem_src,
                                uint32_t bytes, uint32_t mbar_smem) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];\n"
      :: "r"(smem_dst), "l"(gmem_src), "r"(bytes), "r"(mbar_smem)
      : "memory");
}

// -- smem-first convention (cluster destination by default) ------------------

__device__ __forceinline__
void tma_load_2d(uint32_t smem_dst, const void* tensormap_ptr,
                 uint32_t mbar_smem, int coord_x, int coord_y) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes"
    " [%0], [%1, {%3, %4}], [%2];\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem), "r"(coord_x), "r"(coord_y)
    : "memory");
}

// -- 3D smem-first (cluster destination) -------------------------------------
// 3D analog of tma_load_2d. Coords {c0, c1, c2} index the CUtensorMap's dims
// from fastest (c0) to slowest (c2); box extents come from the tensormap.
__device__ __forceinline__
void tma_load_3d(uint32_t smem_dst, const void* tensormap_ptr,
                 uint32_t mbar_smem, int c0, int c1, int c2) {
  asm volatile(
    "cp.async.bulk.tensor.3d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes"
    " [%0], [%1, {%3, %4, %5}], [%2];\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem), "r"(c0), "r"(c1), "r"(c2)
    : "memory");
}

// -- 4D smem-first (cluster destination) -------------------------------------
// 4D analog of tma_load_3d. Coords {c0..c3} index the CUtensorMap's dims from
// fastest (c0) to slowest (c3); box extents come from the tensormap.
__device__ __forceinline__
void tma_load_4d(uint32_t smem_dst, const void* tensormap_ptr,
                 uint32_t mbar_smem, int c0, int c1, int c2, int c3) {
  asm volatile(
    "cp.async.bulk.tensor.4d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes"
    " [%0], [%1, {%3, %4, %5, %6}], [%2];\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem), "r"(c0), "r"(c1), "r"(c2), "r"(c3)
    : "memory");
}

// -- tmap-first convention (explicit cta vs cluster scope) -------------------

// 2D TMA load into CTA-local shared memory (.shared::cta).
__device__ __forceinline__
void tma_load_2d_cta(void const* tensormap_ptr, uint32_t smem_dst,
                     uint32_t mbar_smem_addr,
                     int32_t coord_x, int32_t coord_y) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cta.global.tile"
    ".mbarrier::complete_tx::bytes"
    " [%0], [%1, {%3, %4}], [%2];\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem_addr),
       "r"(coord_x), "r"(coord_y)
    : "memory");
}

// 2D TMA load into cluster-wide shared memory (.shared::cluster).
// Required when using multicast or cta_group::2.
__device__ __forceinline__
void tma_load_2d_cluster(void const* tensormap_ptr, uint32_t smem_dst,
                         uint32_t mbar_smem_addr,
                         int32_t coord_x, int32_t coord_y) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes"
    " [%0], [%1, {%3, %4}], [%2];\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem_addr),
       "r"(coord_x), "r"(coord_y)
    : "memory");
}

// -- L2-cache-hint variants --------------------------------------------------

__device__ __forceinline__
void tma_load_2d_cta_l2hint(void const* tensormap_ptr, uint32_t smem_dst,
                            uint32_t mbar_smem_addr,
                            int32_t coord_x, int32_t coord_y,
                            uint64_t cache_policy) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cta.global.tile"
    ".mbarrier::complete_tx::bytes.L2::cache_hint"
    " [%0], [%1, {%3, %4}], [%2], %5;\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem_addr),
       "r"(coord_x), "r"(coord_y),
       "l"(cache_policy)
    : "memory");
}

__device__ __forceinline__
void tma_load_2d_cluster_l2hint(void const* tensormap_ptr, uint32_t smem_dst,
                                uint32_t mbar_smem_addr,
                                int32_t coord_x, int32_t coord_y,
                                uint64_t cache_policy) {
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.tile"
    ".mbarrier::complete_tx::bytes.L2::cache_hint"
    " [%0], [%1, {%3, %4}], [%2], %5;\n"
    :: "r"(smem_dst), "l"(tensormap_ptr),
       "r"(mbar_smem_addr),
       "r"(coord_x), "r"(coord_y),
       "l"(cache_policy)
    : "memory");
}

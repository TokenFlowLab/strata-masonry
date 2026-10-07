// 52_red_async.cuh -- red.async.relaxed.cluster.shared::cluster.{add,min}.u32
//
// ARCH: sm_90a
//
// Fire-and-forget asynchronous reduction to shared::cluster memory, coupled
// to an mbarrier arrive (mbarrier::complete_tx::bytes mechanism). Commonly
// used to contribute to a per-cluster counter and signal completion with
// one instruction.
//
// Note: the .release.sys.global variant requires sm_100+, so it is not
// exposed here.

#pragma once

// PTX:    9.7.15.7 (red.async.cluster)
//
#include <cstdint>

__device__ __forceinline__
void red_async_cluster_add_u32(uint32_t smem_addr, uint32_t v,
                               uint32_t mbar_smem) {
  asm volatile(
    "red.async.relaxed.cluster.shared::cluster"
    ".mbarrier::complete_tx::bytes.add.u32 [%0], %1, [%2];\n"
    :: "r"(smem_addr), "r"(v), "r"(mbar_smem) : "memory");
}

__device__ __forceinline__
void red_async_cluster_min_u32(uint32_t smem_addr, uint32_t v,
                               uint32_t mbar_smem) {
  asm volatile(
    "red.async.relaxed.cluster.shared::cluster"
    ".mbarrier::complete_tx::bytes.min.u32 [%0], %1, [%2];\n"
    :: "r"(smem_addr), "r"(v), "r"(mbar_smem) : "memory");
}

__device__ __forceinline__
void red_async_cluster_max_u32(uint32_t smem_addr, uint32_t v,
                               uint32_t mbar_smem) {
  asm volatile(
    "red.async.relaxed.cluster.shared::cluster"
    ".mbarrier::complete_tx::bytes.max.u32 [%0], %1, [%2];\n"
    :: "r"(smem_addr), "r"(v), "r"(mbar_smem) : "memory");
}

__device__ __forceinline__
void red_async_cluster_add_s32(uint32_t smem_addr, int32_t v,
                               uint32_t mbar_smem) {
  asm volatile(
    "red.async.relaxed.cluster.shared::cluster"
    ".mbarrier::complete_tx::bytes.add.s32 [%0], %1, [%2];\n"
    :: "r"(smem_addr), "r"(v), "r"(mbar_smem) : "memory");
}

__device__ __forceinline__
void red_async_cluster_min_s32(uint32_t smem_addr, int32_t v,
                               uint32_t mbar_smem) {
  asm volatile(
    "red.async.relaxed.cluster.shared::cluster"
    ".mbarrier::complete_tx::bytes.min.s32 [%0], %1, [%2];\n"
    :: "r"(smem_addr), "r"(v), "r"(mbar_smem) : "memory");
}

__device__ __forceinline__
void red_async_cluster_max_s32(uint32_t smem_addr, int32_t v,
                               uint32_t mbar_smem) {
  asm volatile(
    "red.async.relaxed.cluster.shared::cluster"
    ".mbarrier::complete_tx::bytes.max.s32 [%0], %1, [%2];\n"
    :: "r"(smem_addr), "r"(v), "r"(mbar_smem) : "memory");
}

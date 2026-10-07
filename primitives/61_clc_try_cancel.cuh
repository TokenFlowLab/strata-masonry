#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 61_clc_try_cancel.cuh -- clusterlaunchcontrol.try_cancel.async.shared::cta
//
// ARCH: sm_100a
//
// Blackwell hardware scheduler primitive. Asynchronously attempts to steal
// the next tile from the grid's scheduler into a 16-byte SMEM slot. The
// outcome (success, containing a TileId payload, or failure) is observed
// via clusterlaunchcontrol.query_cancel (#62) after the accompanying
// mbarrier has signaled completion.
//
// Persistent-kernel schedulers loop on this primitive to consume tiles
// without host-side launch overhead.
// Source: knowledge/instructions/clc/clusterlaunchcontrol.md
// PTX:    9.7.15.18 (clusterlaunchcontrol.try_cancel)
//
#include <cstdint>

// Asynchronous try_cancel. Writes a 16-byte tile descriptor into [smem_dst]
// when complete, and arrives on [mbar_smem] via complete_tx.
__device__ __forceinline__ void clc_try_cancel_async(
    uint32_t smem_dst, uint32_t mbar_smem) {
  asm volatile(
    "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128"
    " [%0], [%1];\n"
    :: "r"(smem_dst), "r"(mbar_smem) : "memory");
}

// Multicast variant (broadcast the cancel attempt across the cluster).
__device__ __forceinline__ void clc_try_cancel_async_multicast(
    uint32_t smem_dst, uint32_t mbar_smem, uint16_t ctamask) {
  asm volatile(
    "clusterlaunchcontrol.try_cancel.async.shared::cta.multicast::cluster"
    ".mbarrier::complete_tx::bytes.b128 [%0], [%1], %2;\n"
    :: "r"(smem_dst), "r"(mbar_smem), "h"(ctamask) : "memory");
}

// All-CTAs multicast variant (no ctamask; broadcasts to every CTA in the
// cluster). The 128-bit response is written to every CTA's smem_dst, and
// the mbar at mbar_smem is signaled via complete_tx on every CTA. This is
// the form used by the canonical CLC sched warp (composite 106) and by
// CUTLASS PipelineCLCFetchAsync.
__device__ __forceinline__ void clc_try_cancel_async_multicast_all(
    uint32_t smem_dst, uint32_t mbar_smem) {
  asm volatile(
    "clusterlaunchcontrol.try_cancel.async.shared::cta"
    ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
    " [%0], [%1];\n"
    :: "r"(smem_dst), "r"(mbar_smem) : "memory");
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

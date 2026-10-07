// 80_st_async_cluster.cuh -- st.async.shared::cluster.mbarrier::complete_tx::bytes
//
// ARCH: sm_90a+
//
// Fire-and-forget store into a PEER CTA's shared memory within the cluster,
// coupled to that peer's mbarrier via the mbarrier::complete_tx::bytes
// mechanism. The issuing thread does not wait: the store completes
// asynchronously and the hardware credits the destination mbarrier's
// transaction-byte count when the data lands.
//
// This is the write-side dual of reading peer SMEM with mapa + ld.shared::cluster.
// Prefer this for cluster reductions: a pull-based reduction serializes remote
// LOAD latency on one CTA, while a push-based one issues remote STORES from all
// CTAs concurrently and is latency-hidden.
//
// Contract:
//   - `dsmem_addr` must be a SHARED-window address already remapped to the
//     destination CTA. Build it with mapa (see 67_mapa.cuh):
//         uint32_t dst = mapa_shared_cluster_u32(smem_ptr_u32(&local), rank);
//     Passing a plain local SMEM address writes to THIS CTA, silently.
//   - `remote_mbar` must be the destination CTA's mbarrier, likewise remapped
//     with mapa. It must be the mbarrier the consumer waits on.
//   - The consumer must have set the expected transaction byte count on that
//     mbarrier first (mbarrier.arrive.expect_tx with the total byte count, see
//     31_mbarrier_arrive_tx.cuh). Each store credits sizeof(type) bytes.
//   - Per-thread instruction; no .sync / .aligned. Divergence is fine.
//   - Does NOT order against ordinary shared stores. Use a fence if the same
//     addresses are also written non-async.
//
// PTX:    9.7.10.12 (st.async)
//
#pragma once
#if defined(PL_AGENTIC_SM90A) || defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

#include <cstdint>

// st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [dst], val, [mbar]
//   Store one fp32 to peer SMEM; credits 4 bytes to the peer's mbarrier.
__device__ __forceinline__
void st_async_cluster_f32(uint32_t dsmem_addr, float value,
                          uint32_t remote_mbar) {
  asm volatile(
    "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];\n"
    :: "r"(dsmem_addr), "f"(value), "r"(remote_mbar) : "memory");
}

// st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [dst], val, [mbar]
//   Untyped 32-bit variant; credits 4 bytes to the peer's mbarrier.
__device__ __forceinline__
void st_async_cluster_b32(uint32_t dsmem_addr, uint32_t value,
                          uint32_t remote_mbar) {
  asm volatile(
    "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];\n"
    :: "r"(dsmem_addr), "r"(value), "r"(remote_mbar) : "memory");
}

// mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [mbar]
//   Arrive on a PEER CTA's mbarrier (no transaction bytes, no return token).
//   `remote_mbar` must already be mapa-remapped to the destination CTA.
//   Use when a peer must be told "I am done" without shipping data.
__device__ __forceinline__
void mbarrier_arrive_cluster_relaxed(uint32_t remote_mbar) {
  asm volatile(
    "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];\n"
    :: "r"(remote_mbar) : "memory");
}


// st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 -- credits 8 bytes.
__device__ __forceinline__
void st_async_cluster_v2_b32(uint32_t dsmem_addr, uint32_t v0, uint32_t v1,
                             uint32_t remote_mbar) {
  asm volatile(
    "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1,%2}, [%3];\n"
    :: "r"(dsmem_addr), "r"(v0), "r"(v1), "r"(remote_mbar) : "memory");
}

// st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 -- credits 16 bytes.
__device__ __forceinline__
void st_async_cluster_v4_b32(uint32_t dsmem_addr, uint32_t v0, uint32_t v1,
                             uint32_t v2, uint32_t v3, uint32_t remote_mbar) {
  asm volatile(
    "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1,%2,%3,%4}, [%5];\n"
    :: "r"(dsmem_addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3), "r"(remote_mbar) : "memory");
}

#endif  // PL_AGENTIC_SM90A || PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

// 106_clc_fetch_next_tile.cuh -- cluster-wide CLC dispatch handshake.
//
// ARCH: sm_100a
//
// Warp choreography (cluster-wide; one warp per role unless noted):
//
//   sched warp -- Producer (LEADER CTA ONLY). One elected lane on the
//                 leader CTA's sched warp issues
//                 `clc_try_cancel_multicast_all`; lanes 0/1 of the same
//                 warp issue `clc_arrive_expect_tx_cluster(_, 16)` first so
//                 BOTH peers' clc_full mbars carry the expected tx
//                 count (lane 0 -> CTA 0 via mapa::cluster, lane 1 ->
//                 CTA 1). Drives the persistent dispatch loop and
//                 advances the producer (stage, phase) cursor. The
//                 follower CTA does NOT issue try_cancel; if it has a
//                 sched-warp slot it acts as a consumer (or idle).
//
//   load warp -- Consumer. Waits clc_full[stage], uses
//                (m_tile, n_tile) to compute TMA source addresses for
//                the next tile, then arrives clc_empty[stage].
//
//   mma warp  -- Consumer. Waits clc_full[stage], drives the K-loop's
//                tcgen05.mma.cta_group::2 (tile coords feed output
//                addressing via composite 104), then arrives
//                clc_empty[stage].
//
//   epi warp  -- Consumer (typically multiple). Waits clc_full[stage]
//                for the destination tile, drains TMEM via cvt +
//                stmatrix + TMA-store, then arrives clc_empty[stage].
//
//   idle warp -- Pseudo-consumer. Reserves a register / dispatch slot
//                without doing work, but still waits + arrives so the
//                cluster-wide handshake count stays uniform across the
//                warp count chosen at launch.
//
// Coordination:
//   - The leader CTA's sched warp issues try_cancel.async with
//     multicast::cluster::all, which deposits a 16-byte response on
//     EVERY CTA's `clc_response` slot and signals `clc_full[stage]` on
//     EVERY CTA's mbar (the multicast does the cross-CTA fan-out). The
//     load / mma / epi / idle warps on both peers therefore all see
//     the same (m_tile, n_tile, valid) for a given stage.
//   - Each of the load / mma / epi / idle warps maintains its own
//     (clc_cons_stage, clc_cons_phase) cursor advanced via
//     `clc_fetch_next_tile_advance`.
//   - The arrives on clc_empty[stage] from the load / mma / epi / idle
//     warps (on both CTAs) are peer-bit-masked (SM100_CLC_PEER_MASK)
//     so all arrives land on the leader CTA's bar. The leader CTA's
//     sched warp waits for clc_empty[stage] before reusing the slot --
//     that wait is what enforces the cluster-wide tile boundary.
//   - arrive_count of clc_empty[stage] is set at init to match the
//     caller's gating policy: per-thread (e.g. 512 = 8 warps x 32
//     lanes x 2 CTAs, every thread of every load / mma / epi / idle
//     warp calls with do_release=true) or per-warp-role (e.g. 16 = 8
//     warps x 2 CTAs, gated via elect_one_sync or lane mask so only
//     one thread per warp arrives).
//
// Public API:
//   clc_parse_response  <CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(...)
//   clc_fetch_next_tile <CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(...)
// where ORDER is `ClcRasterOrder::AlongN` or `ClcRasterOrder::AlongM`.
//
// K0's typical instantiation is `<1, 2, AlongN>` (cluster_dims=(2,1,1),
// peers split N, raster x->N) for `tiles_m > tiles_n` shapes; flip to
// `<2, 1, AlongM>` when `tiles_n > tiles_m` per the CUTLASS heuristic
// in sched_warp.md sec 5.5.
//
// What the templated helpers DO support:
//   - Any (CLUSTER_SHAPE_M, CLUSTER_SHAPE_N) values, including non-pow2.
//   - Both raster orders, paired with the matching grid layout (caller
//     reshapes the grid so the FAST raster axis (ctaid.x) carries the
//     fast logical axis).
//   - Clusters that span both axes simultaneously (e.g. (2,2,1)).
//
// Convention contract per ORDER:
//
//   AlongN: ctaid.x -> logical N, ctaid.y -> logical M.
//     Grid:    dim3(CLUSTER_SHAPE_N * n_clusters, m_clusters, 1)
//     Decode:  m_tile = ctaid.y / CLUSTER_SHAPE_M
//              n_tile = ctaid.x / CLUSTER_SHAPE_N
//     Example: cluster_dims=(2,1,1) -> CSM=1, CSN=2 (peers split N).
//
//   AlongM: ctaid.x -> logical M, ctaid.y -> logical N.
//     Grid:    dim3(CLUSTER_SHAPE_M * m_clusters, n_clusters, 1)
//     Decode:  m_tile = ctaid.x / CLUSTER_SHAPE_M
//              n_tile = ctaid.y / CLUSTER_SHAPE_N
//     Example: cluster_dims=(2,1,1) -> CSM=2, CSN=1 (peers split M).
//
// Both orders are symmetric "identity-on-swapped-axes" decodes -- the
// raster choice swaps which grid axis carries which logical axis, AND the
// caller swaps CSM/CSN to reflect the cluster geometry interpretation.
// Mirrors CUTLASS's "major/minor" abstraction in
// `tile_scheduler_params.h` (where AlongM swaps which of M/N is "major").
//
// What they do NOT cover yet:
//   - 2D only. The parser reads ctaid.x (d0) and ctaid.y (d1 low16);
//     ctaid.z (d1 high16) and d3 are discarded. Kernels with
//     cluster_dims.z > 1 would lose the z component.
//   - Caller is responsible for setting up the grid AND the CSM/CSN
//     args consistently with ORDER. The helper does not validate.
//
// Earlier versions of this file had an AlongM branch that did a
// "swap cluster-index parts, preserve remainders" formula intended to
// achieve AlongM without reshaping the grid. That formula is correct
// only for square clusters (CSM == CSN); for asymmetric (1x2 / 2x1)
// clusters it produces "peers in same cluster see different n_tiles,"
// which violates the cta_group::2 MMA contract. Replaced by the
// symmetric two-line formula above.
//
// Proxy fence: try_cancel.async writes via the async proxy; reading the
// response via the generic proxy (ld.shared) requires
// `fence.proxy.async.shared::cta` first. Handled inside
// clc_parse_response.
//
// Source: knowledge/instructions/clc/clusterlaunchcontrol.md sec 8,
//         knowledge/instructions/tmem/tcgen05_tmem.md sec 8.3.
// PTX:    9.7.15.18 (clusterlaunchcontrol.try_cancel.async),
//         9.7.15.16.16 (mbarrier.arrive scope/sem defaults),
//         9.7.15.16.19 (mbarrier.try_wait.parity).

#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

#include <cstdint>
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/31_mbarrier_arrive_tx.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/34_fence_proxy_async.cuh"
#include "../primitives/61_clc_try_cancel.cuh"
#include "../primitives/62_clc_query_cancel.cuh"
#include "../primitives/67_mapa.cuh"
#include "118_mbarrier_phase_tracking.cuh"

// Matches cute::Sm100MmaPeerBitMask in cutlass/include/cute/arch/mma_sm100_umma.hpp.
static constexpr uint32_t SM100_CLC_PEER_MASK = 0xFEFFFFFF;

struct ClcTileInfo {
  int  m_tile;
  int  n_tile;
  bool valid;
};

// Raster order for the cluster-grid sweep (`knowledge/building_blocks/sched_warp.md` sec 5.5):
// the hardware dispatches and cancels CTAs x-fastest, so ORDER decides which logical axis
// consecutive tokens sweep by choosing which grid axis carries it (the convention contract above).
//   AlongN: ctaid.x carries N (grid (CSN * n_clusters, m_clusters)), consecutive tokens sweep N.
//   AlongM: ctaid.x carries M (grid (CSM * m_clusters, n_clusters)), consecutive tokens sweep M;
//           the CUTLASS heuristic picks it when tiles_n > tiles_m.
// Both decodes are identity-on-swapped-axes; the caller shapes the grid and the CSM / CSN
// arguments to match ORDER. A 1D grid uses <1, 1, AlongN>: n_tile is the linear block id.
enum class ClcRasterOrder { AlongN, AlongM };

// 2SM (cluster): both peers get arrive_expect_tx via mapa::cluster routing
// (lane 0 -> peer 0, lane 1 -> peer 1).
//
// Source: knowledge/building_blocks/sched_warp.md sec 7.3 (CLC PTX form
//         + expect_tx arrive form, 2SM column).
__device__ __forceinline__
void clc_arrive_expect_tx_cluster(uint32_t clc_full_local_addr, uint32_t tx_bytes) {
  const int lane_idx = threadIdx.x & 31;
  if (lane_idx < 2) {
    uint32_t remote_addr =
        mapa_shared_cluster_u32(clc_full_local_addr,
                                static_cast<uint32_t>(lane_idx));
    mbarrier_arrive_expect_tx_cluster(remote_addr, tx_bytes);
  }
}

// 1SM (non-cluster): plain .shared arrive_expect_tx on the local mbar.
// Single elected lane issues -- no cluster routing needed.
//
// Source: knowledge/building_blocks/sched_warp.md sec 7.3 (CLC PTX form
//         + expect_tx arrive form, 1SM column).
__device__ __forceinline__
void clc_arrive_expect_tx_cta(uint32_t clc_full_local_addr, uint32_t tx_bytes) {
  if ((threadIdx.x & 31) == 0) {
    mbarrier_arrive_expect_tx(clc_full_local_addr, tx_bytes);
  }
}

__device__ __forceinline__
void clc_try_cancel_multicast_all(uint32_t resp_smem_addr,
                                  uint32_t clc_full_addr) {
  clc_try_cancel_async_multicast_all(resp_smem_addr, clc_full_addr);
}

__device__ __forceinline__
void clc_consumer_release(uint32_t clc_empty_local_addr) {
  uint32_t peer0_addr = clc_empty_local_addr & SM100_CLC_PEER_MASK;
  mbarrier_arrive_cluster_default(peer0_addr);
}

// Non-cluster (1SM) variant: arrive on the local mbar via plain .shared
// scope.
//
// Source: knowledge/building_blocks/sched_warp.md sec 7.3 (consumer
//         release on clc_empty, 1SM column).
__device__ __forceinline__
void clc_consumer_release_cta(uint32_t clc_empty_local_addr) {
  mbarrier_arrive_nostate(clc_empty_local_addr);
}

// Generic rasterized parse. Cluster-shape and raster order are template
// parameters so the divmod is constant-folded at compile time.
template <int CLUSTER_SHAPE_M, int CLUSTER_SHAPE_N, ClcRasterOrder ORDER>
__device__ __forceinline__
ClcTileInfo clc_parse_response(uint32_t resp_smem_addr) {
  uint32_t d0, d1, d2, d3;
  fence_proxy_async_shared_cta();
  clc_load_response(resp_smem_addr, d0, d1, d2, d3);
  const int  ctaid_x = static_cast<int>(d0);
  const int  ctaid_y = static_cast<int>(d1 & 0xFFFFu);
  const bool valid   = (d2 & 1u) != 0u;
  (void)d3;

  ClcTileInfo info;
  info.valid = valid;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    info.m_tile = ctaid_y / CLUSTER_SHAPE_M;
    info.n_tile = ctaid_x / CLUSTER_SHAPE_N;
  } else {  // AlongM
    info.m_tile = ctaid_x / CLUSTER_SHAPE_M;
    info.n_tile = ctaid_y / CLUSTER_SHAPE_N;
  }
  return info;
}

template <int CLUSTER_SHAPE_M, int CLUSTER_SHAPE_N, ClcRasterOrder ORDER,
          int CTA_GROUP = 2, bool SUSPEND = false>
__device__ __forceinline__
ClcTileInfo clc_fetch_next_tile(
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    int clc_cons_stage, uint32_t clc_cons_phase, bool do_release) {
  uint32_t full_addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(&clc_full_bar[clc_cons_stage]));
  if constexpr (SUSPEND) mbarrier_wait_parity_suspend(full_addr, clc_cons_phase);
  else                   mbarrier_wait_parity(full_addr, clc_cons_phase);
  uint32_t resp_addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(&clc_response[clc_cons_stage * 4]));
  ClcTileInfo t = clc_parse_response<
      CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(resp_addr);
  // Gotcha (sm_100a, seen 2026-09-15): the release below can take effect before the
  // ld.shared of the response has completed; the sched then refills the slot and this
  // warp reads the NEXT response (item skew between warps -> hang). Complete the read first,
  // same fence as CUTLASS's sm100_tile_scheduler::fetch_next_work.
  // fence_acq_rel_cta() also works: it completes the ld.shared before the arrive, the half that
  // bites; this fence (MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC.S) adds the generic -> async-proxy WAR
  // edge the PTX model requires, since the slot's next writer is try_cancel (async proxy).
  fence_proxy_async_shared_cta();
  if (do_release) {
    uint32_t empty_local = static_cast<uint32_t>(
        __cvta_generic_to_shared(&clc_empty_bar[clc_cons_stage]));
    if constexpr (CTA_GROUP == 1) {
      clc_consumer_release_cta(empty_local);
    } else {
      clc_consumer_release(empty_local);
    }
  }
  return t;
}

// Advance the (stage, phase) consumer state for a CLC ring of STAGES slots.
// Default STAGES=2 matches K0/skeleton's CLC ring depth.
template <int STAGES = 2>
__device__ __forceinline__
void clc_fetch_next_tile_advance(int& clc_cons_stage,
                                 uint32_t& clc_cons_phase) {
  advance_stage_phase<STAGES>(clc_cons_stage, clc_cons_phase);
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

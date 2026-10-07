// 104_acc_pipeline_2bank_blackwell.cuh -- 2-bank TMEM acc pipeline.
//
// ARCH: sm_100a
//
// Warp choreography (cluster-wide; one warp per role unless noted):
//
//   mma warp -- Producer (LEADER CTA ONLY). Drives the K-loop's
//               tcgen05.mma.cta_group::2 against the next acc bank.
//               All 32 lanes of the leader CTA's mma warp call
//               `acc_pipeline_2bank_producer_acquire` (.sync.aligned)
//               on acc_empty[bank] before reusing the bank. After the
//               K-loop's MMAs are issued, one elected lane issues
//               `acc_pipeline_2bank_producer_commit_cluster` on acc_full[bank]
//               via tcgen05.commit.cta_group::2.multicast::cluster --
//               the commit fires only when this thread's prior MMAs
//               have retired. The follower CTA's mma warp slot does
//               NOT acquire or commit on this composite's bars.
//
//   epi warp -- Consumer (BOTH CTAs, typically 4 epi warps per CTA).
//               Each epi thread waits `acc_pipeline_2bank_consumer_wait`
//               on acc_full[bank], drains the bank (TMEM -> regs via
//               tcgen05.ld -> SMEM via cvt + stmatrix -> TMA store),
//               then arrives `acc_pipeline_2bank_consumer_release` on
//               acc_empty[bank]. Both peers' arrives are peer-bit
//               masked so they all land on the leader CTA's bar.
//
// Per-bar contract:
//
//   acc_empty[2]: init arrive_count = caller's choice (typically 256
//     = 4 epi warps x 32 lanes x 2 CTAs for a full warp-specialized
//     GEMM; smaller for smoke tests). EVERY consumer thread arrives
//     once per tile via `mbarrier.arrive.shared::cluster.b64` on a
//     peer-bit-masked addr so all arrives land on the leader CTA's
//     bar. The leader CTA's mma warp (warp 0, all 32 lanes) waits via
//     `mbarrier.try_wait.parity` using the phase-1 trick (see below).
//
//   acc_full[2]: init arrive_count = 1. One elected lane of the leader
//     CTA's mma warp commits via
//     `tcgen05.commit.cta_group::2.multicast::cluster.b64`; the commit
//     fires only after this thread's prior tcgen05.mma.cta_group::2
//     instructions have retired. Every epi thread on BOTH CTAs waits
//     via `mbarrier.try_wait.parity`; phase starts at 0 and flips
//     every 2 visits.
//
// Phase-1 trick: the leader CTA's mma warp waits every tile -- no
// `if (tile_count >= ACC_STAGES)` skip -- which keeps the producer
// body uniform and avoids a tile counter that would spill or add a
// branch in the hot path. The bar's incomplete phase is 0 at init;
// passing phase=1 on the first visit falls through `try_wait.parity`
// because non-matching parity always passes. After 2 producer visits
// (one per bank), phase tracks normally.
//
// PTX: 9.7.15.16.16 (mbarrier.arrive scope/sem defaults; default = .release/.cta)
//      9.7.15.16.19 (mbarrier.try_wait.parity)
//      9.7.18.10.10.1 (tcgen05.mma.cta_group::2)
//      9.7.18.12.1 (tcgen05.commit.cta_group::2.multicast::cluster)
//      9.7.18.5 (Issue Granularity -- one thread per CTA-pair issues commit)
//
// Designed off V56 from program/pl_gemm_gb200/pl_gemm.cu lines 9468-9961.
//
// V56 reference: gemm_v56(...) in program/pl_gemm_gb200/pl_gemm.cu
//   line 9514-9517 -- mbar init (acc_full = 1, acc_empty = 256)
//   line 9727       -- producer state init (stage=0, phase=1; phase-1 trick)
//   line 9747-9749  -- producer acquire (MBARRIER_TRY_WAIT_PARITY on acc_empty)
//   line 9788       -- producer commit (tcgen05_commit_2sm_multicast_v56 on acc_full)
//   line 9791-9792  -- producer state advance (stage^=1; phase flips when stage wraps to 0)
//   line 9814       -- consumer state init (stage=0, phase=0)
//   line 9867-9868  -- consumer wait (mbarrier_try_wait_parity_nb_v56 on acc_full)
//   line 9902       -- consumer release (mbarrier_arrive_peer_sm0 on acc_empty)
//   line 9939-9940  -- consumer state advance (stage^=1; phase flips when stage wraps to 0)

#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

#include <cstdint>
#include "../primitives/11_tcgen05_commit.cuh"
#include "../primitives/29_mbarrier_init.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"

// AND a local SMEM addr with this to remap to peer 0's SMEM. Matches
// cute::Sm100MmaPeerBitMask in cutlass/include/cute/arch/mma_sm100_umma.hpp.
static constexpr uint32_t SM100_ACC_PIPE_2BANK_PEER_MASK = 0xFEFFFFFF;

struct AccPipeline2BankBars {
  uint64_t* acc_full;   // [2]
  uint64_t* acc_empty;  // [2]
};

struct AccPipeline2BankState {
  int count;
};

__device__ __forceinline__
AccPipeline2BankState acc_pipeline_2bank_state_init() {
  return AccPipeline2BankState{0};
}

__device__ __forceinline__
int acc_pipeline_2bank_state_index(const AccPipeline2BankState& s) {
  return s.count & 1;
}

__device__ __forceinline__
int acc_pipeline_2bank_consumer_phase(const AccPipeline2BankState& s) {
  return (s.count >> 1) & 1;
}

// Phase-1 trick: first visit on each bank uses phase 1.
__device__ __forceinline__
int acc_pipeline_2bank_producer_phase(const AccPipeline2BankState& s) {
  int n = (s.count >> 1);
  return (n + 1) & 1;
}

__device__ __forceinline__
void acc_pipeline_2bank_state_advance(AccPipeline2BankState& s) {
  s.count++;
}

// Call from a single thread; pair with __syncthreads() and
// `fence.mbarrier_init.release.cluster` on the caller's side. No
// pre-arrives needed (phase-1 trick).
//
// `empty_arrive_count` is how many threads arrive on each acc_empty
// per tile. Typical values:
//   256  -- full warp-specialized GEMM (4 epi warps x 32 lanes x 2 CTAs).
//   N    -- smoke / smaller setups where only N threads are consumers.
__device__ __forceinline__
void acc_pipeline_2bank_init(AccPipeline2BankBars bars,
                             uint32_t empty_arrive_count) {
  for (int s = 0; s < 2; ++s) {
    mbarrier_init(static_cast<uint32_t>(__cvta_generic_to_shared(&bars.acc_full[s])),
                  /*arrive_count=*/1);
    mbarrier_init(static_cast<uint32_t>(__cvta_generic_to_shared(&bars.acc_empty[s])),
                  empty_arrive_count);
  }
}

// === PRODUCER (MMA warp, leader CTA only) ===

__device__ __forceinline__
void acc_pipeline_2bank_producer_acquire(AccPipeline2BankBars bars,
                                         const AccPipeline2BankState& s) {
  uint32_t bar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(
          &bars.acc_empty[acc_pipeline_2bank_state_index(s)]));
  mbarrier_wait_parity(bar_addr, acc_pipeline_2bank_producer_phase(s));
}

// Caller elects (e.g. `if (elect_one_sync())`) so the primitive 11
// wrapper is shared with non-acc commits.
//
// Source: knowledge/building_blocks/mma_warp.md sec 7.3 (1SM vs 2SM
//         producer commit scope -- .multicast::cluster form).
//         knowledge/building_blocks/pipeline.md sec 2.2 (acc_full_mbar).
template <int CTA_GROUP>
__device__ __forceinline__
void acc_pipeline_2bank_producer_commit_cluster(AccPipeline2BankBars bars,
                                        const AccPipeline2BankState& s,
                                        uint16_t ctamask) {
  uint32_t bar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(
          &bars.acc_full[acc_pipeline_2bank_state_index(s)]));
  tcgen05_commit_multicast<CTA_GROUP>(bar_addr, ctamask);
}

// Non-multicast variant: arrives on the LOCAL CTA's acc_full[bank] mbar
// only (no cluster, no ctamask). For 1SM / cta_group::1 callers where
// the mbar is plain .shared (not .shared::cluster).
//
// Source: knowledge/building_blocks/mma_warp.md sec 7.3 (1SM producer
//         commit; cta_group::1 local-scope commit on acc-full mbar).
//         knowledge/building_blocks/pipeline.md sec 2.2 (acc_full_mbar).
template <int CTA_GROUP>
__device__ __forceinline__
void acc_pipeline_2bank_producer_commit_cta(AccPipeline2BankBars bars,
                                              const AccPipeline2BankState& s) {
  uint32_t bar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(
          &bars.acc_full[acc_pipeline_2bank_state_index(s)]));
  tcgen05_commit<CTA_GROUP>(bar_addr);
}

// === CONSUMER (epilogue warps, every thread of both CTAs) ===

__device__ __forceinline__
void acc_pipeline_2bank_consumer_wait(AccPipeline2BankBars bars,
                                      const AccPipeline2BankState& s) {
  uint32_t bar_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(
          &bars.acc_full[acc_pipeline_2bank_state_index(s)]));
  mbarrier_wait_parity(bar_addr, acc_pipeline_2bank_consumer_phase(s));
}

// NOT the explicit cluster-scope release form -- that has been observed
// to interact badly with the tcgen05.alloc state machine on this
// hardware.
//
// Caller must `tcgen05.wait::ld` before invoking; otherwise the next
// iteration's MMA may overwrite TMEM while reads are still in flight.
//
// Works for both 1SM (cta_group::1, no cluster) and 2SM (cta_group::2,
// cluster=2) consumers:
//   - 2SM: peer 1's local_addr has bit 24 set; the PEER_MASK AND clears
//     it so both peers' arrives land on the LEADER (CTA 0) acc_empty[bank]
//     mbar (arrive_count = 256 = 4 epi warps x 32 lanes x 2 CTAs).
//   - 1SM: only CTA 0 exists, bit 24 of local SMEM addr is already 0, so
//     the AND is a no-op; all 128 EPI threads arrive on the local mbar.
//   - `mbarrier.arrive.shared::cluster.b64` is legal in non-cluster
//     launches (the implicit cluster has size 1 and resolves to .cta).
//
// TODO: rename to `acc_pipeline_2bank_consumer_release_cluster` to mirror
// `producer_commit_cluster` / `producer_commit_cta` naming. Deferred --
// would require threading a CTA_GROUP template param into block 93's
// `epi_warp_blackwell_1tile_1sm2sm_bf16` to dispatch between _cluster
// and _cta variants at every call site.
__device__ __forceinline__
void acc_pipeline_2bank_consumer_release(AccPipeline2BankBars bars,
                                         const AccPipeline2BankState& s) {
  uint32_t local_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(
          &bars.acc_empty[acc_pipeline_2bank_state_index(s)]));
  // 2SM: route peer 1's arrive to peer 0's mbar (leader-CTA-only).
  // 1SM: bit 24 already 0, AND is no-op.
  uint32_t peer0_addr = local_addr & SM100_ACC_PIPE_2BANK_PEER_MASK;
  // shared::cluster: legal in 1SM too (implicit cluster size = 1 -> .cta).
  mbarrier_arrive_cluster_default(peer0_addr);
}

// Non-cluster (1SM / cta_group::1) variant: plain .shared arrive on the
// local CTA's acc_empty[bank] mbar. No PEER_MASK routing, no cluster
// scope. Used by kernels where every
// EPI thread arrives on the local mbar (arrive_count = 128 = 4 epi
// warps x 32 lanes, single CTA).
//
// Source: knowledge/building_blocks/epi_warp.md sec 6.2 (1SM acc_empty
//         release; plain .shared arrive).
//         knowledge/building_blocks/pipeline.md sec 2.2 (acc_empty_mbar).
__device__ __forceinline__
void acc_pipeline_2bank_consumer_release_cta(AccPipeline2BankBars bars,
                                             const AccPipeline2BankState& s) {
  uint32_t local_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(
          &bars.acc_empty[acc_pipeline_2bank_state_index(s)]));
  mbarrier_arrive_nostate(local_addr);
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

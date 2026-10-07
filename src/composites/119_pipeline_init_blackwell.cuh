// 119_pipeline_init_blackwell.cuh -- canonical Blackwell pipeline mbarrier init.
//
// ARCH: sm_100a
//
// Initializes the FULL warp-specialized Blackwell pipeline barrier suite:
//
//   full[NUM_STAGES]            mainloop: load -> MMA
//   empty[NUM_STAGES]           mainloop: MMA -> load
//   acc_full[2]                 accumulator: MMA -> epi
//   acc_empty[2]                accumulator: epi -> MMA
//   clc_full[2]                 CLC HW -> sched
//   clc_empty[2]                CLC consumers -> CLC HW (consumer release)
//   throttle_full[2]            CLC throttle: load -> sched
//   throttle_empty[2]           CLC throttle: sched -> load
//
// All accumulator / CLC / throttle rings are 2-stage; mainloop ring depth
// is the NUM_STAGES template parameter. Optional rings can be left null:
// pass `nullptr` for any of acc_full / acc_empty / clc_full / clc_empty /
// throttle_full / throttle_empty if your kernel doesn't use them (e.g. a
// load+MMA-only smoke test that skips CLC + throttle).
//
// Caller-owned arrive_counts -- the canonical values are
// (1 / 1 / 1 / 256 / 1 / 10 / 32 / 32) but they're per-design.
//
// PTX:    9.7.15.16.12 (mbarrier.init), 9.7.15.4 (fence.mbarrier_init.release.cluster)
//
#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

#include <cstdint>
#include "../primitives/29_mbarrier_init.cuh"
#include "../primitives/35_fence_mbarrier_init.cuh"
#include "../primitives/38_barrier_cluster.cuh"

// Full Blackwell pipeline mbarrier suite. Optional pointers may be nullptr.
struct BlackwellPipelineBars {
  uint64_t* full;            // [NUM_STAGES]
  uint64_t* empty;           // [NUM_STAGES]
  uint64_t* acc_full;        // [2]   (optional; nullptr to skip)
  uint64_t* acc_empty;       // [2]   (optional; nullptr to skip)
  uint64_t* clc_full;        // [2]   (optional; nullptr to skip)
  uint64_t* clc_empty;       // [2]   (optional; nullptr to skip)
  uint64_t* throttle_full;   // [2]   (optional; nullptr to skip)
  uint64_t* throttle_empty;  // [2]   (optional; nullptr to skip)
};

// Per-mbarrier arrive_counts for `pipeline_init_blackwell`. CTA_GROUP
// template param scales CTA-scaled counts (acc_empty, clc_empty) by the
// cluster size: 2SM = 2 CTAs (default), 1SM = 1 CTA. Other defaults
// match the canonical K0 numbers; override per-kernel. CTA-scaled
// arrival counts shift by cluster size.
template <int CTA_GROUP = 2>
struct BlackwellPipelineArriveCounts {
  uint32_t full           = 1;                // load warp's elected lane (+expect_tx)
  uint32_t empty          = 1;                // MMA tcgen05.commit.multicast::cluster
  uint32_t acc_full       = 1;                // MMA tcgen05.commit
  uint32_t acc_empty      = 128 * CTA_GROUP;  // 4 epi warps x 32 lanes x CTA_GROUP CTAs
  uint32_t clc_full       = 1;                // CLC HW try_cancel.async
  uint32_t clc_empty      = 5   * CTA_GROUP;  // 5 elect-per-warp consumers x CTA_GROUP CTAs
  uint32_t throttle_full  = 32;               // entire load warp
  uint32_t throttle_empty = 32;               // entire sched warp
};

// Callers pass the same CTA_GROUP they use for tcgen05.{alloc,mma,commit};
// it threads through to BlackwellPipelineArriveCounts<CTA_GROUP>'s
// CTA-scaled defaults.
template <int NUM_STAGES, int CTA_GROUP = 2>
__device__ __forceinline__
void pipeline_init_blackwell(const BlackwellPipelineBars& b,
                             const BlackwellPipelineArriveCounts<CTA_GROUP>& a = {}) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < NUM_STAGES; ++i) {
      mbarrier_init(smem_ptr_u32(&b.full[i]),  a.full);
      mbarrier_init(smem_ptr_u32(&b.empty[i]), a.empty);
    }
    for (int i = 0; i < 2; ++i) {
      if (b.acc_full)        mbarrier_init(smem_ptr_u32(&b.acc_full[i]),       a.acc_full);
      if (b.acc_empty)       mbarrier_init(smem_ptr_u32(&b.acc_empty[i]),      a.acc_empty);
      if (b.clc_full)        mbarrier_init(smem_ptr_u32(&b.clc_full[i]),       a.clc_full);
      if (b.clc_empty)       mbarrier_init(smem_ptr_u32(&b.clc_empty[i]),      a.clc_empty);
      if (b.throttle_full)   mbarrier_init(smem_ptr_u32(&b.throttle_full[i]),  a.throttle_full);
      if (b.throttle_empty)  mbarrier_init(smem_ptr_u32(&b.throttle_empty[i]), a.throttle_empty);
    }
    fence_mbarrier_init_release_cluster();
  }
  // Cluster-wide sync so every peer CTA sees the initialized barriers.
  barrier_cluster_arrive();
  barrier_cluster_wait();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

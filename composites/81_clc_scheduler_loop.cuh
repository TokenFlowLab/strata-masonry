#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 81_clc_scheduler_loop.cuh -- CLC persistent-kernel scheduler loop
//
// ARCH: sm_100a
//
// Persistent-kernel pattern: issue try_cancel to fetch the next tile,
// wait on its mbarrier, load the 128-bit handle, query_cancel to decode.
// Loop until the hardware scheduler says no more tiles.
// PTX:    9.7.15.18 (clusterlaunchcontrol)
//
#include <cstdint>
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/61_clc_try_cancel.cuh"
#include "../primitives/62_clc_query_cancel.cuh"
#include "../primitives/31_mbarrier_arrive_tx.cuh"

struct ClcTile {
  uint32_t canceled;  // 1 = valid tile, 0 = no more tiles
  uint32_t ctaid_x, ctaid_y, ctaid_z;
};

// Issue one try_cancel, wait, decode. Caller provides the 16-byte SMEM
// slot and the paired mbarrier.
__device__ __forceinline__ ClcTile clc_fetch_next_tile(
    uint32_t slot_smem, uint32_t mbar_smem, uint32_t phase_parity) {
  if (threadIdx.x == 0) {
    mbarrier_arrive_expect_tx(mbar_smem, 16);
    clc_try_cancel_async(slot_smem, mbar_smem);
  }
  mbarrier_wait_parity(mbar_smem, phase_parity);

  uint32_t r0, r1, r2, r3;
  clc_load_response(slot_smem, r0, r1, r2, r3);
  ClcTile t;
  t.canceled = clc_query_is_canceled(r0, r1, r2, r3);
  t.ctaid_x  = clc_query_first_ctaid_x(r0, r1, r2, r3);
  t.ctaid_y  = clc_query_first_ctaid_y(r0, r1, r2, r3);
  t.ctaid_z  = 0;  // extend if needed
  return t;
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

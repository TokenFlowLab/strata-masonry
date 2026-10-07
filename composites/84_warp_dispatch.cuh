// 84_warp_dispatch.cuh -- warp_id -> role assignment + setmaxnreg split
//
// ARCH: sm_90a
//
// Maps warp ID to a role enum and sets the per-warp register budget.
// A single WarpRole enum is shared across the Blackwell and Hopper code
// paths, with aliases so callers using either vocabulary compile cleanly:
//
//   Load     == Loader        (TMA producer)
//   Mma      == MmaDriver     (WGMMA / tcgen05.mma consumer)
//   Idle     == Unused        (no work assigned)
//   Epilogue, Scheduler, Softmax, Correction       (single name each)

#pragma once

// PTX:    9.7.21.5 (setmaxnreg)
//
#include <cstdint>
#include "../primitives/46_setmaxnreg.cuh"

// =============================================================================
// Shared enum
// =============================================================================

enum class WarpRole : uint32_t {
  Loader     = 0,    // TMA producer
  Load       = 0,    // alias for Loader
  MmaDriver  = 1,    // WGMMA / tcgen05.mma driver
  Mma        = 1,    // alias for MmaDriver
  Scheduler  = 2,    // tile scheduler
  Epilogue   = 3,    // output store
  Softmax    = 4,    // FMHA online softmax
  Correction = 5,    // FMHA accumulator correction
  Unused     = 255,  // no work assigned
  Idle       = 255,  // alias for Unused
};

// =============================================================================
// Warp-id helpers (Hopper-style 4-warp groups)
// =============================================================================

constexpr int WARPS_PER_WG    = 4;
constexpr int THREADS_PER_WG  = 128;

__device__ __forceinline__
int get_warp_id() { return (int)(threadIdx.x / 32); }

__device__ __forceinline__
int get_warpgroup_id() { return get_warp_id() / WARPS_PER_WG; }

__device__ __forceinline__
int get_lane_id() { return (int)(threadIdx.x % 32); }

// =============================================================================
// Blackwell warp-role dispatchers
// =============================================================================

// Simple fixed-dispatch: warp 0 = Loader, warp 1 = MmaDriver, warps 2-3 =
// Epilogue.
__device__ __forceinline__
WarpRole warp_role_default(int warp_id) {
  switch (warp_id) {
    case 0:  return WarpRole::Loader;
    case 1:  return WarpRole::MmaDriver;
    case 2:
    case 3:  return WarpRole::Epilogue;
    default: return WarpRole::Unused;
  }
}

// Canonical Blackwell warp-specialized split (16 warps).
__device__ __forceinline__
WarpRole warp_role_blackwell16(int warp_id) {
  if (warp_id == 0) return WarpRole::Loader;
  if (warp_id == 1) return WarpRole::Scheduler;
  if (warp_id == 2) return WarpRole::MmaDriver;
  if (warp_id >= 4 && warp_id < 8)  return WarpRole::Correction;
  if (warp_id >= 8 && warp_id < 16) return WarpRole::Epilogue;
  return WarpRole::Unused;
}

// Typical register-budget split: light warps decrease, heavy (MMA / epilogue)
// warps increase. Callable from warp-aligned code only.
template <int LIGHT_REGS, int HEAVY_REGS>
__device__ __forceinline__
void warp_setmaxnreg_split(bool is_heavy) {
  if (is_heavy) setmaxnreg_inc<HEAVY_REGS>();
  else          setmaxnreg_dec<LIGHT_REGS>();
}

// =============================================================================
// Hopper warp-role dispatchers (3-WG kernel)
// =============================================================================

// Hopper 3-WG dispatch: 1 load WG + 2 consumer WGs (12 warps = 384 threads).
//   WG 0 (warps 0-3):  load warp-group
//   WG 1 (warps 4-7):  consumer 0 (MMA + epilogue half 0)
//   WG 2 (warps 8-11): consumer 1 (MMA + epilogue half 1)
__device__ __forceinline__
WarpRole hopper_role_dispatch_3wg() {
  int wg = get_warpgroup_id();
  if (wg == 0) return WarpRole::Load;
  if (wg == 1 || wg == 2) return WarpRole::Mma;
  return WarpRole::Idle;
}

// Apply register budget for a 3-WG Hopper kernel (call as the FIRST
// instruction). Load WG: dec to 40 regs; consumer WGs: inc to 232.
__device__ __forceinline__
void hopper_apply_regbudget_3wg() {
  int wg = get_warpgroup_id();
  if (wg == 0) {
    setmaxnreg_dec<40>();
  } else {
    setmaxnreg_inc<232>();
  }
}

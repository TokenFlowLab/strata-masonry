// 46_setmaxnreg.cuh -- setmaxnreg.{inc,dec}.sync.aligned
//
// ARCH: sm_90a
//
// Per-warp register budget control. Warp-specialized pipelines use .dec on
// the "light" warps (loader, scheduler) to free registers for the "heavy"
// warps (MMA / epilogue) which .inc their budget.
//
// Must be the FIRST instruction each warp executes after role dispatch.
// All threads in the warp-group must execute the same instruction.
//
//   .dec: release registers to the CTA pool.
//   .inc: claim registers from the CTA pool (may block).
//
// N must be in [24, 256] and a multiple of 8 (PTX ISA 9.7.21.5).

#pragma once

// Source: knowledge/instructions/warp/setmaxnreg.md
// PTX:    9.7.21.5 (setmaxnreg)
//
#include <cstdint>

template <int N>
__device__ __forceinline__
void setmaxnreg_dec() {
  static_assert(N >= 24 && N <= 256 && N % 8 == 0,
                "setmaxnreg_dec: N must be in [24, 256] and a multiple of 8");
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n"
               :: "n"(N) : "memory");
}

template <int N>
__device__ __forceinline__
void setmaxnreg_inc() {
  static_assert(N >= 24 && N <= 256 && N % 8 == 0,
                "setmaxnreg_inc: N must be in [24, 256] and a multiple of 8");
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n"
               :: "n"(N) : "memory");
}

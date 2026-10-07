// 47_nanosleep.cuh -- nanosleep.u32
//
// ARCH: sm_90a
//
// Cooperative backpressure: the calling thread yields for approximately N
// nanoseconds (hardware rounds). Used on producer paths to slow spin loops
// so the scheduler can advance other warps. The actual sleep is a minimum,
// not exact; max observed on Hopper/Blackwell is ~1 us. N=0 is a no-op.
// Typical backpressure values: 20-200 ns.

#pragma once

// PTX:    9.7.21.2 (nanosleep)
//
#include <cstdint>

__device__ __forceinline__
void nanosleep(uint32_t nanoseconds) {
  asm volatile("nanosleep.u32 %0;\n" :: "r"(nanoseconds));
}

// Convenience: ~100 ns sleep (immediate operand, no register).
__device__ __forceinline__
void nanosleep_short() {
  asm volatile("nanosleep.u32 100;\n" ::);
}

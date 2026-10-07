// 44_elect_sync.cuh -- elect.sync (pick one lane out of the warp)
//
// ARCH: sm_90a
//
// Returns 1 (true) for exactly one lane in the warp (deterministic, the lowest
// active lane), 0 for all others. Used to select a single thread for
// one-thread-issued instructions (tcgen05.mma, TMA load, etc.) without
// serializing on lane 0.

#pragma once

// Source: knowledge/instructions/warp/elect_sync.md
// PTX:    9.7.15.15 (elect.sync)
//
#include <cstdint>

// Elect one lane from all 32 lanes (mask = 0xFFFFFFFF).
__device__ __forceinline__
uint32_t elect_one_sync() {
  uint32_t elected;
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "elect.sync %0|p, 0xffffffff;\n\t"
    "selp.b32 %0, 1, 0, p;\n\t"
    "}\n"
    : "=r"(elected));
  return elected;
}

// Elect one lane from a specified member mask.
__device__ __forceinline__
uint32_t elect_one_sync(uint32_t membermask) {
  uint32_t elected;
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "elect.sync %0|p, %1;\n\t"
    "selp.b32 %0, 1, 0, p;\n\t"
    "}\n"
    : "=r"(elected) : "r"(membermask));
  return elected;
}

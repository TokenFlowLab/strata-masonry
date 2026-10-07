#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 42_smem_desc_blackwell.cuh -- Blackwell 64-bit SMEM descriptor.
//
// ARCH: sm_100a
//
// Packs a 64-bit matrix descriptor consumed by tcgen05.mma and tcgen05.cp.
//
// Layout (PTX 9.7.18.4.1 Table 51):
//   [0:14)  start address >> 4 (14 bits)
//   [14:16) reserved
//   [16:30) leading byte offset >> 4 (relative mode) / next_start_address (absolute)
//   [30:32) reserved
//   [32:46) stride byte offset >> 4
//   [46:49) version (constant 0b001)
//   [49:52) base offset (0 if the swizzle pattern starts on its natural boundary)
//   [52:53) leading-dim mode (0 = relative, 1 = absolute 3xFP4)
//   [53:61) reserved (0)
//   [61:64) swizzle mode (0=none, 2=128B, 4=64B, 6=32B)
//
// Alignment: start_addr / lbo / sbo must all be 16-byte aligned.
// Source: knowledge/instructions/mma/tcgen05_mma.md
// PTX:    9.7.18.4.1 (Blackwell SMEM descriptor)
//
#include <cstdint>

enum class SmemSwizzleBlackwell : uint32_t {
  None = 0,
  B128_32atom = 1,
  B128 = 2,
  B64  = 4,
  B32  = 6,
};

__device__ __host__ __forceinline__ uint64_t build_smem_desc_blackwell(
    uint32_t smem_addr,
    uint32_t stride_byte_offset,
    uint32_t leading_byte_offset,
    SmemSwizzleBlackwell swizzle = SmemSwizzleBlackwell::B128,
    uint32_t base_offset = 0) {
  uint64_t d = 0;
  d |= static_cast<uint64_t>((smem_addr >> 4) & 0x3FFF);
  d |= static_cast<uint64_t>((leading_byte_offset >> 4) & 0x3FFF) << 16;
  d |= static_cast<uint64_t>((stride_byte_offset >> 4) & 0x3FFF) << 32;
  d |= static_cast<uint64_t>(1) << 46;                                  // version
  d |= (static_cast<uint64_t>(base_offset) & 0x7) << 49;
  d |= static_cast<uint64_t>(static_cast<uint32_t>(swizzle) & 0x7) << 61;
  return d;
}

// Advance a descriptor's ADDRESS field in place by `inc` (units of 16 B), i.e. the low 32-bit
// word; the hi word (LBO/SBO/swizzle) is k-invariant. Equivalent to trtllm-gen's
// `incrSmemAddr` (trtllm/dev/SmemTile.h:121), which does `tmp.u32[0] += offset`.
//
// Gotcha: asm VOLATILE is load-bearing. A plain `d += inc` lets nvcc CSE descriptors that
// share a value into parallel base+imm forms, each paying a UMOV to rematerialise the hi
// word -- and it silently corrupts a second MMA group that shares the descriptor.
__device__ __forceinline__ void smem_desc_add_lo(uint64_t& d, uint32_t inc) {
  asm volatile("{\n\t"
      ".reg .b32 lo, hi;\n\t"
      "mov.b64 {lo, hi}, %0;\n\t"
      "add.u32 lo, lo, %1;\n\t"
      "mov.b64 %0, {lo, hi};\n\t"
      "}" : "+l"(d) : "r"(inc));
}

// Absolute-address mode (sm_103a 3xFP4 dual-buffer). Caller must ensure
// next_start_address is in the same SMEM region. Sets bit 52.
__device__ __host__ __forceinline__ uint64_t build_smem_desc_blackwell_abs(
    uint32_t smem_addr,
    uint32_t stride_byte_offset,
    uint32_t next_start_address,
    SmemSwizzleBlackwell swizzle = SmemSwizzleBlackwell::B128,
    uint32_t base_offset = 0) {
  uint64_t d = 0;
  d |= static_cast<uint64_t>((smem_addr >> 4) & 0x3FFF);
  d |= static_cast<uint64_t>((next_start_address >> 4) & 0x3FFF) << 16;
  d |= static_cast<uint64_t>((stride_byte_offset >> 4) & 0x3FFF) << 32;
  d |= static_cast<uint64_t>(1) << 46;
  d |= (static_cast<uint64_t>(base_offset) & 0x7) << 49;
  d |= static_cast<uint64_t>(1) << 52;                                  // absolute
  d |= static_cast<uint64_t>(static_cast<uint32_t>(swizzle) & 0x7) << 61;
  return d;
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

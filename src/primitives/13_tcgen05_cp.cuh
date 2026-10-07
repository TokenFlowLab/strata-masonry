#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 13_tcgen05_cp.cuh -- tcgen05.cp.cta_group::{1,2}.<shape>{.dst_fmt.src_fmt}
//
// ARCH: sm_100a
//
// Single-thread asynchronous copy from SMEM (described by a 64-bit matrix
// descriptor) into TMEM. Used to stage block-scaling factor tiles (.4x256b)
// or operand B (.128x256b) into TMEM before an MMA issue. Completion is
// tracked by tcgen05.commit + mbarrier (see #11).
//
// PTX 9.7.18.9.2 syntax:
//   tcgen05.cp.cta_group.shape{.multicast}{.dst_fmt.src_fmt} [taddr], s-desc;
// shape = .128x256b | .4x256b | .128x128b | .64x128b | .32x128b
//
// The source is a 64-bit SMEM matrix descriptor, not a plain SMEM address.
// Use #42 smem_desc_blackwell to construct it.
// PTX:    9.7.18.9.2 (tcgen05.cp), 9.7.18.9.1 (Optional Decompression)
//
#include <cstdint>

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_cp_4x256b(
    uint32_t tmem_addr, uint64_t smem_desc) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_cp_4x256b: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.cp.cta_group::1.4x256b [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  } else {
    asm volatile("tcgen05.cp.cta_group::2.4x256b [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  }
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_cp_128x256b(
    uint32_t tmem_addr, uint64_t smem_desc) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_cp_128x256b: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  } else {
    asm volatile("tcgen05.cp.cta_group::2.128x256b [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  }
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_cp_128x128b(
    uint32_t tmem_addr, uint64_t smem_desc) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_cp_128x128b: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  } else {
    asm volatile("tcgen05.cp.cta_group::2.128x128b [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  }
}

// .64x128b shape requires intra-warpgroup multicast modifier (PTX 9.7.18.9.2):
// data is staged into a subset of the 4 warps in the warp-group.
//   .warpx2::02_13 -- warps 0 and 2 receive the tile
//   .warpx2::01_23 -- warps 0 and 1 receive the tile
template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_cp_64x128b_warpx2_02_13(
    uint32_t tmem_addr, uint64_t smem_desc) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_cp_64x128b_warpx2_02_13: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  } else {
    asm volatile("tcgen05.cp.cta_group::2.64x128b.warpx2::02_13 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  }
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_cp_64x128b_warpx2_01_23(
    uint32_t tmem_addr, uint64_t smem_desc) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_cp_64x128b_warpx2_01_23: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.cp.cta_group::1.64x128b.warpx2::01_23 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  } else {
    asm volatile("tcgen05.cp.cta_group::2.64x128b.warpx2::01_23 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  }
}

// .32x128b shape requires .warpx4 (all 4 warps in the warp-group receive).
template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4(
    uint32_t tmem_addr, uint64_t smem_desc) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2,
                "tcgen05_cp_32x128b_warpx4: CTA_GROUP must be 1 or 2");
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  } else {
    asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  }
}

// .dst_fmt.src_fmt decompression forms (PTX 9.7.18.9.2)
// .src_fmt = .b6x16_p32 (16 FP6 elts in 96 bits + 32 pad)
//          | .b4x16_p64 (16 FP4 elts in 64 bits + 64 pad)
// .dst_fmt = .b8x16     (16 elts expanded to byte-aligned in TMEM)
//
// Used to stage FP6 / FP4 narrow-precision tensor data into TMEM in
// byte-addressable form so a subsequent tcgen05.mma can consume it via
// the .kind::{f8f6f4, mxf8f6f4, mxf4, mxf4nvf4} forms (#4, #5).

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_cp_4x256b_b8x16_b6x16_p32(
    uint32_t tmem_addr, uint64_t smem_desc) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2);
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.cp.cta_group::1.4x256b.b8x16.b6x16_p32 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  } else {
    asm volatile("tcgen05.cp.cta_group::2.4x256b.b8x16.b6x16_p32 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  }
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_cp_4x256b_b8x16_b4x16_p64(
    uint32_t tmem_addr, uint64_t smem_desc) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2);
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.cp.cta_group::1.4x256b.b8x16.b4x16_p64 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  } else {
    asm volatile("tcgen05.cp.cta_group::2.4x256b.b8x16.b4x16_p64 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  }
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_cp_128x256b_b8x16_b6x16_p32(
    uint32_t tmem_addr, uint64_t smem_desc) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2);
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.cp.cta_group::1.128x256b.b8x16.b6x16_p32 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  } else {
    asm volatile("tcgen05.cp.cta_group::2.128x256b.b8x16.b6x16_p32 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  }
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_cp_128x256b_b8x16_b4x16_p64(
    uint32_t tmem_addr, uint64_t smem_desc) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2);
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.cp.cta_group::1.128x256b.b8x16.b4x16_p64 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  } else {
    asm volatile("tcgen05.cp.cta_group::2.128x256b.b8x16.b4x16_p64 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  }
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_cp_128x128b_b8x16_b6x16_p32(
    uint32_t tmem_addr, uint64_t smem_desc) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2);
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.cp.cta_group::1.128x128b.b8x16.b6x16_p32 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  } else {
    asm volatile("tcgen05.cp.cta_group::2.128x128b.b8x16.b6x16_p32 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  }
}

template <int CTA_GROUP>
__device__ __forceinline__ void tcgen05_cp_128x128b_b8x16_b4x16_p64(
    uint32_t tmem_addr, uint64_t smem_desc) {
  static_assert(CTA_GROUP == 1 || CTA_GROUP == 2);
  if constexpr (CTA_GROUP == 1) {
    asm volatile("tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  } else {
    asm volatile("tcgen05.cp.cta_group::2.128x128b.b8x16.b4x16_p64 [%0], %1;\n"
                 :: "r"(tmem_addr), "l"(smem_desc));
  }
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

#pragma once
#if defined(PL_AGENTIC_SM90A)
// 43_smem_desc_hopper.cuh -- Hopper WGMMA 64-bit SMEM descriptor
//
// ARCH: sm_90a
//
// Builds the 64-bit descriptor that wgmma.mma_async takes as its A/B
// operand. Encodes start address, leading-dim byte offset, stride
// byte offset, base offset, and swizzle mode. Bit layout differs from
// Blackwell tcgen05 SMEM descriptor (#42); the swizzle field lives in
// bits [63:62] for Hopper vs [61:63] for Blackwell.
//
// Constraints (PTX 9.7.17.5.1.2.2):
//   all addresses/offsets must be 16-byte aligned
//   start address encoded as (addr >> 4) & 0x3FFF (14 bits)
//   swizzle mode: 0=none, 1=128B, 2=64B, 3=32B (different encoding from sm_100a)
//
// Issuer: per-thread (compile-time constexpr or runtime).
// PTX:    9.7.17.5.1.2.2 (Hopper Matrix Descriptor Format)
//
#include <cstdint>

// ---------------------------------------------------------------------------
// Hopper WGMMA SMEM descriptor (64 bits).
// PTX ISA 9.7.17.5.1.2.2 Matrix Descriptor Format:
//   [13: 0] (14 bits) matrix start address (encoded = (addr & 0x3FFFF) >> 4)
//   [15:14] reserved
//   [29:16] (14 bits) leading dimension byte offset (encoded: >> 4)
//   [31:30] reserved
//   [45:32] (14 bits) stride dimension byte offset (encoded: >> 4)
//   [48:46] reserved
//   [51:49] (3 bits) matrix base offset
//   [61:52] reserved
//   [63:62] (2 bits) swizzle mode: 0=NONE, 1=B128, 2=B64, 3=B32
//           (NOTE: CUTLASS enum: INTERLEAVE=0, B128=1, B64=2, B32=3.)
// ---------------------------------------------------------------------------

// Build a Hopper WGMMA SMEM descriptor with explicit bit placement.
// smem_addr: 32-bit shared memory address
// leading_byte_offset: LBO in bytes
// stride_byte_offset:  SBO in bytes
// swizzle_mode: 0=NONE, 1=B128, 2=B64, 3=B32
// base_offset: base offset (only valid when swizzle != NONE)
__device__ __forceinline__
uint64_t build_smem_desc_hopper(uint32_t smem_addr,
                                uint32_t leading_byte_offset,
                                uint32_t stride_byte_offset,
                                uint32_t swizzle_mode,
                                uint32_t base_offset = 0) {
    uint64_t desc = 0;
    // [13:0] start address (>> 4)
    desc |= ((uint64_t)((smem_addr >> 4) & 0x3FFF)) << 0;
    // [29:16] leading byte offset (>> 4)
    desc |= ((uint64_t)((leading_byte_offset >> 4) & 0x3FFF)) << 16;
    // [45:32] stride byte offset (>> 4)
    desc |= ((uint64_t)((stride_byte_offset >> 4) & 0x3FFF)) << 32;
    // [51:49] base offset (3 bits)
    desc |= ((uint64_t)(base_offset & 0x7)) << 49;
    // [63:62] swizzle mode (2 bits)
    desc |= ((uint64_t)(swizzle_mode & 0x3)) << 62;
    return desc;
}

// Convenience: build descriptor for B128 swizzle (most common for GEMM).
// stride_bytes is the SBO for the matrix.
__device__ __forceinline__
uint64_t build_smem_desc_hopper_b128(uint32_t smem_addr,
                                      uint32_t stride_bytes) {
    // B128 swizzle = 1 in Hopper encoding
    return build_smem_desc_hopper(smem_addr, 0, stride_bytes, 1, 0);
}

// Update base address in existing descriptor (for K-loop tile advance)
__device__ __forceinline__
uint64_t smem_desc_advance_hopper(uint64_t desc, uint32_t byte_offset) {
    // Clear old start_address [13:0], set new start_address (>> 4)
    uint64_t new_base = (uint64_t)((byte_offset >> 4) & 0x3FFF);
    return (desc & ~0x3FFFull) | new_base;
}

#endif  // PL_AGENTIC_SM90A

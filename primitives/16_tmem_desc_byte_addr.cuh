#pragma once
#if defined(PL_AGENTIC_SM103A)
// 16_tmem_desc_byte_addr.cuh -- 64-bit SMEM descriptor builder (absolute / byte-address mode)
//
// ARCH: sm_103a
//
// SMEM descriptor variant where the leading dimension is encoded as an
// absolute byte-address pair instead of a relative leading-byte-offset.
// PTX ISA 9.7.18.4.1: bit 52 of the descriptor selects the mode --
//   0 = leading_byte_offset_relative  (standard, all archs)
//   1 = leading_byte_address_absolute (sm_103a only).
// The absolute mode lets K=48B (= K=96 FP4) cross 128B SMEM boundaries
// by encoding a SECOND start address in bits [16:29] (the same field
// that holds leading_byte_offset in standard mode). Each of the two
// addresses must lie within its own aligned 128B chunk.
//
// Constraints (PTX ISA 9.7.18.3.1.2 and 9.7.18.3.1.2.1, absolute mode):
//   1. Only B128 swizzle (bits [61:63] = 2) is supported.
//   2. K-Major only -- transpose bits (15, 16) of idesc must be 0.
//   3. matrix base offset (bits [49:51]) must be 0.
//
// Issuer: any thread (this is a host-style helper that runs on device);
// the caller passes the resulting descriptor to a tcgen05.{mma,cp}
// instruction. The file name says "tmem" for naming-scheme symmetry with
// related sm_103a work, but the descriptor itself is SMEM-side -- TMEM
// addresses are 32-bit lane:col, not byte-addressed.
// PTX:    9.7.18.4.1 (descriptor format, bit 52 absolute mode)
//
#include <cstdint>

// matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4
// Only 14 useful bits remain after the >>4. SMEM addresses must fit; the
// canonical pointer is a CVTA'd .shared::cta address (file _common.cuh).
__device__ __forceinline__ uint64_t _smem_addr_encode(uint32_t addr) {
    return static_cast<uint64_t>((addr & 0x3FFFFu) >> 4);
}

// Build a 64-bit SMEM descriptor for the absolute-address (byte-address)
// leading dimension stride mode. Sets bit 52 = 1.
//
//   start_addr        : SMEM address of the first 128B-aligned data chunk
//   next_start_addr   : SMEM address of the second chunk; bits [16:29].
//                       Used by the MMA when the operand crosses a 128B
//                       boundary (3xFP4 K=48B pattern).
//   stride_byte_offset: Stride dim byte offset. Must be 16-byte aligned.
//                       Bits [32:45] hold (offset & 0x3FFFF) >> 4.
//   base_offset       : Must be 0 in absolute mode (per ISA restriction 3).
//
// The descriptor's swizzle is hard-coded to B128 (=2 in bits [61:63]) per
// ISA restriction 1.
__device__ __forceinline__ uint64_t build_smem_desc_byte_addr(
    uint32_t start_addr,
    uint32_t next_start_addr,
    int stride_byte_offset)
{
    uint64_t d = 0;
    d |= _smem_addr_encode(start_addr);                                        // bits  0-13
    d |= _smem_addr_encode(next_start_addr) << 16;                             // bits 16-29 (was LBO)
    d |= (static_cast<uint64_t>(stride_byte_offset & 0x3FFFF) >> 4) << 32;     // bits 32-45
    d |= static_cast<uint64_t>(0b001) << 46;                                   // bits 46-48 (version=1)
    // bits 49-51 (matrix base offset) MUST be 0 per ISA restriction 3.
    d |= static_cast<uint64_t>(1) << 52;                                       // bit  52 (absolute mode)
    // bits 53-60 fixed zero
    d |= static_cast<uint64_t>(2) << 61;                                       // bits 61-63 (swizzle = B128)
    return d;
}

#endif  // PL_AGENTIC_SM103A

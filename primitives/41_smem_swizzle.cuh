// 41_smem_swizzle.cuh -- B32 / B64 / B128 swizzle address math
//
// ARCH: sm_90a
//
// Two equivalent formulations are provided:
//
//   1. (row, col) form: swizzle_col_bytes<S>(row, col_bytes), swizzle_offset<S>
//      XORs col_bytes with (row & mask) << 4. Convenient when callers track
//      row/col explicitly.
//
//   2. byte_offset form: smem_swizzle_b{32,64,128}(byte_offset),
//      generic smem_swizzle<BITS>, smem_swizzled_addr_b128. Matches CuTe
//      Swizzle<B,M,S>: out = in XOR ((in >> S) & MASK) where MASK encodes
//      the swizzle mode. For standard GEMM (S=3, M=4) the two forms are
//      equivalent when row_stride == swizzle_bytes.
//
// Standard CuTe modes:
//   B32  = Swizzle<1,4,3>  (32-byte atom)
//   B64  = Swizzle<2,4,3>  (64-byte atom)
//   B128 = Swizzle<3,4,3>  (128-byte atom)  -- most common for GEMM

#pragma once

// Source: knowledge/building_blocks/smem_layout.md
// PTX:    9.7.18.4.1 (Blackwell descriptor swizzle modes), 9.7.17.5.1.2.2 (Hopper Matrix Descriptor Format)
//
#include <cstdint>

// =============================================================================
// (row, col) form
// =============================================================================

// Returns the swizzled byte-offset within an 8-row atom for a given
// (row, col_bytes) pair.
//
//   B128: swizzled_col = col XOR ((row & 7) << 4)
//   B64:  swizzled_col = col XOR ((row & 3) << 4)
//   B32:  swizzled_col = col XOR ((row & 1) << 4)
template <int SWIZZLE_BYTES>
__device__ __host__ __forceinline__
uint32_t swizzle_col_bytes(uint32_t row, uint32_t col_bytes) {
  static_assert(SWIZZLE_BYTES == 32 || SWIZZLE_BYTES == 64
                || SWIZZLE_BYTES == 128,
                "swizzle: SWIZZLE_BYTES must be 32, 64, or 128");
  uint32_t mask = 0;
  if constexpr (SWIZZLE_BYTES == 128) mask = (row & 0x7) << 4;
  if constexpr (SWIZZLE_BYTES == 64)  mask = (row & 0x3) << 4;
  if constexpr (SWIZZLE_BYTES == 32)  mask = (row & 0x1) << 4;
  return col_bytes ^ mask;
}

// Full SMEM offset: swizzled column within the atom + row * row_bytes.
template <int SWIZZLE_BYTES>
__device__ __host__ __forceinline__
uint32_t swizzle_offset(uint32_t row, uint32_t col_bytes, uint32_t row_bytes) {
  return row * row_bytes + swizzle_col_bytes<SWIZZLE_BYTES>(row, col_bytes);
}

// =============================================================================
// byte_offset form (CuTe Swizzle<B,M,S> with S=3, M=4)
// =============================================================================

// B32 swizzle (32-byte atom).
__device__ __forceinline__
uint32_t smem_swizzle_b32(uint32_t byte_offset) {
  return byte_offset ^ ((byte_offset & 0x10) >> 0);
}

// B64 swizzle (64-byte atom).
__device__ __forceinline__
uint32_t smem_swizzle_b64(uint32_t byte_offset) {
  return byte_offset ^ ((byte_offset & 0x30) >> 0);
}

// B128 swizzle (128-byte atom): XOR bits [4:6] with bits [7:9]. Most common.
__device__ __forceinline__
uint32_t smem_swizzle_b128(uint32_t byte_offset) {
  return byte_offset ^ ((byte_offset & 0x380) >> 3);
}

// Generic CuTe swizzle parameterized by BITS = B, with M=4 and S=3 fixed.
template <int BITS>
__device__ __forceinline__
uint32_t smem_swizzle(uint32_t byte_offset) {
  constexpr uint32_t mask = ((1u << BITS) - 1u) << 4;
  constexpr int shift = 3;
  return byte_offset ^ ((byte_offset & (mask << shift)) >> shift);
}

// Compute swizzled SMEM address from base + row + column (B128 mode).
// row_stride: bytes per row in SMEM (e.g. 128 cols * 2 bytes = 256).
// elem_size:  bytes per element (2 for FP16, 1 for FP8).
__device__ __forceinline__
uint32_t smem_swizzled_addr_b128(uint32_t smem_base, int row, int col,
                                 int row_stride, int elem_size) {
  uint32_t byte_offset = row * row_stride + col * elem_size;
  return smem_base + smem_swizzle_b128(byte_offset);
}

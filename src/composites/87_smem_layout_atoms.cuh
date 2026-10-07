// 87_smem_layout_atoms.cuh -- swizzle atom constants + MMA atom pairing lookup
//
// ARCH: sm_90a
//
// Each MMA atom (tcgen05.mma kind, wgmma shape) requires a specific SMEM
// layout atom that matches the tensor core's hardware expectations. The
// tensormap's swizzle parameter and the MMA descriptor's swizzle bits MUST
// match this atom, or the MMA reads garbage.
//
// Per CUTLASS / CuTe defaults:
//   FP16/BF16 MMA with K=16 atom  pairs with B128 swizzle.
//   FP8       MMA with K=32 atom  pairs with B128 swizzle.
//   INT8      MMA with K=32 atom  pairs with B128 swizzle.
//   TF32      MMA with K=8  atom  pairs with B64  swizzle.
//
// Two API styles are kept:
//   AtomPairing + pairing_default<K, ELEM_BYTES>()
//                       Fully constexpr template lookup that returns a flat
//                       struct {smem_swizzle_bytes, smem_row_bytes, mma_k}.
//   SwizzleMode + SwizzleAtom + MmaType + helpers
//                       Enum-driven lookup with hopper_smem_desc_swizzle_bits
//                       (descriptor encoding) and mma_smem_addr_b128 helper.

#pragma once

// PTX:    9.7.18.4.1 (Blackwell descriptor swizzle), 9.7.17.5.1.2.2 (Hopper Matrix Descriptor Format)
//
#include <cstdint>
#include "../primitives/41_smem_swizzle.cuh"

// =============================================================================
// Template-based pairing
// =============================================================================

struct AtomPairing {
  int smem_swizzle_bytes;   // 32, 64, or 128
  int smem_row_bytes;       // stride between rows in bytes
  int mma_k;                // MMA atom's K dimension
};

// Row bytes = elements-per-row * sizeof(elem). Assumes K-major SMEM tile
// (stride of 1 for K).
template <int K, int ELEM_BYTES>
__device__ __host__ __forceinline__
AtomPairing pairing_default() {
  AtomPairing p{};
  p.mma_k = K;
  if constexpr (ELEM_BYTES == 2) {           // FP16 / BF16
    p.smem_swizzle_bytes = 128;
    p.smem_row_bytes = K * 2;
  } else if constexpr (ELEM_BYTES == 1) {    // FP8 / INT8
    p.smem_swizzle_bytes = 128;
    p.smem_row_bytes = K;
  } else if constexpr (ELEM_BYTES == 4) {    // TF32 stored as 4 bytes
    p.smem_swizzle_bytes = 64;
    p.smem_row_bytes = K * 4;
  } else {
    p.smem_swizzle_bytes = 0;
    p.smem_row_bytes = K * ELEM_BYTES;
  }
  return p;
}

// =============================================================================
// Enum-based atom catalog
// =============================================================================

// Matches CuTe Swizzle<B,M,S> classes.
enum class SwizzleMode : uint32_t {
  None = 0,  // no swizzle
  B32  = 1,  // 32-byte atom  (Swizzle<1,4,3>)
  B64  = 2,  // 64-byte atom  (Swizzle<2,4,3>)
  B128 = 3,  // 128-byte atom (Swizzle<3,4,3>) -- standard GEMM
};

// Swizzle atom dimensions: always 8 rows, width varies.
struct SwizzleAtom {
  int bytes_per_row;
  int rows;
  SwizzleMode mode;
};

constexpr SwizzleAtom ATOM_B32  = { 32, 8, SwizzleMode::B32};
constexpr SwizzleAtom ATOM_B64  = { 64, 8, SwizzleMode::B64};
constexpr SwizzleAtom ATOM_B128 = {128, 8, SwizzleMode::B128};

// MMA type tags for atom pairing lookup.
enum class MmaType {
  Hopper_F16,     // wgmma m64nNk16 f32.f16.f16
  Hopper_BF16,    // wgmma m64nNk16 f32.bf16.bf16
  Hopper_TF32,    // wgmma m64nNk8  f32.tf32.tf32
  Hopper_FP8,     // wgmma m64nNk32 f32.{e4m3,e5m2}.{e4m3,e5m2}
  Hopper_I8,      // wgmma m64nNk32 s32.s8.s8
  Blackwell_F16,  // tcgen05.mma kind::f16
  Blackwell_FP8,  // tcgen05.mma kind::f8f6f4
  Blackwell_FP4,  // tcgen05.mma kind::mxf4 / mxf4nvf4
};

// All standard MMA atoms use B128 swizzle by default.
constexpr SwizzleMode mma_default_swizzle(MmaType t) {
  (void)t;
  return SwizzleMode::B128;
}

// Bytes per row for a given MMA atom (at B128 = 128 regardless of MMA type).
constexpr int mma_atom_bytes_per_row(MmaType t) {
  (void)t;
  return 128;
}

// Compute SMEM swizzled address for a (row, col) within an 8-row atom tile.
__device__ __forceinline__
uint32_t mma_smem_addr_b128(uint32_t smem_base, int row, int col,
                            int elem_bytes) {
  return smem_swizzled_addr_b128(smem_base, row, col, 128, elem_bytes);
}

// CuTe-compatible swizzle descriptor value for the Hopper WGMMA SMEM
// descriptor (0=none, 1=B128, 2=B64, 3=B32).
constexpr uint32_t hopper_smem_desc_swizzle_bits(SwizzleMode m) {
  switch (m) {
    case SwizzleMode::None: return 0;
    case SwizzleMode::B128: return 1;
    case SwizzleMode::B64:  return 2;
    case SwizzleMode::B32:  return 3;
  }
  return 0;
}

// 109_fastdivmod.cuh -- FastDivmod: host-precomputed unsigned-division magic
// (CUTLASS form, no add). Packed: bits[0:31]=multiplier, bits[32:39]=shift_right;
// d==1 -> 0 (identity). Replaces per-tile-decode runtime '/','%'
// (I2F.RP/MUFU.RCP/F2I) with IMAD/SHF, matching FA4's StaticPersistentTileScheduler.

#pragma once

#include <cstdint>

__device__ __forceinline__ unsigned fdiv(unsigned n, unsigned long long pk) {
  unsigned M = (unsigned)pk;
  if (M == 0u) return n;
  return __umulhi(n, M) >> (unsigned)(pk >> 32);
}
__host__ inline unsigned long long make_magic(unsigned d) {
  if (d <= 1u) return 0ULL;
  unsigned l = 0; while ((1u << (l + 1)) <= d) ++l;   // floor(log2(d))
  unsigned p = 31u + l;
  unsigned long long m = ((1ull << p) + (unsigned long long)d - 1ull) / d;
  return (m & 0xffffffffULL) | ((unsigned long long)(p - 32u) << 32);
}

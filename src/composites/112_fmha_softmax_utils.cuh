// 112_fmha_softmax_utils.cuh -- FMHA softmax<->correction helpers.
//
//   full_bar_arrive / full_bar_wait -- per-band softmax<->correction HW named barrier
//   mask_s_row_r2p<IS_CAUSAL, K_TILE> -- R2P-bitmask ragged/causal padding mask for one S row

#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

#include <cstdint>
#include <cmath>
#include "../primitives/37_bar_sync.cuh"   // bar_arrive<N> / bar_sync<N>

// softmax->correction "scale ready" via per-band HW named barrier: id = 1 + m_tile*4 + band (1..8),
// 64 threads (1 softmax + 1 correction warp of the band); softmax arrives, correction waits.
// Protocol: bar.arrive N = arrive (increment count) + don't block; bar.sync N = arrive + block
// until N have arrived. So both contribute to the 64; only sync (correction) blocks.
// The CTA has 16 named barriers (ids 0-15); this kernel uses 0 (__syncthreads) + 1..8, 9-15 free.
__device__ __forceinline__ void full_bar_arrive(int m_tile, int band) {
  switch (1 + m_tile * 4 + band) {
    case 1: bar_arrive<1>(64); break;
    case 2: bar_arrive<2>(64); break;
    case 3: bar_arrive<3>(64); break;
    case 4: bar_arrive<4>(64); break;
    case 5: bar_arrive<5>(64); break;
    case 6: bar_arrive<6>(64); break;
    case 7: bar_arrive<7>(64); break;
    case 8: bar_arrive<8>(64); break;
  }
}
__device__ __forceinline__ void full_bar_wait(int m_tile, int band) {
  switch (1 + m_tile * 4 + band) {
    case 1: bar_sync<1>(64); break;
    case 2: bar_sync<2>(64); break;
    case 3: bar_sync<3>(64); break;
    case 4: bar_sync<4>(64); break;
    case 5: bar_sync<5>(64); break;
    case 6: bar_sync<6>(64); break;
    case 7: bar_sync<7>(64); break;
    case 8: bar_sync<8>(64); break;
  }
}

// FA4-style R2P-bitmask mask for one thread's S row: valid keys form a prefix [0,n_keep); mask
// each 32-col chunk with one shift + predicated select. IS_CAUSAL=false: padding only (keys
// >= seqlen_k masked). IS_CAUSAL=true: also keep keys <= q_pos (causal diagonal). Only the
// ragged last K-tile (k==0) needs it.
// k_offset = first key index this K-tile covers (tile spans keys [k_offset, k_offset+K_TILE));
// from the reversed K-loop k_offset = (K_TILES-1-k)*K_TILE. n_keep = valid keys in this tile.
//
// WHY the shift+bitmask (not a per-key `if (key >= n_keep)`): it lowers BRANCH-FREE, and R2P
// batches away the per-key compares. The `?:` clamp -> SHF/SEL (uniform, no branch); `keep =
// 0xFFFFFFFF >> m` is one shift. ptxas then emits R2P (Register-to-Predicate): it loads predicate
// regs FROM keep's bits in ONE instruction -- at most 7 at a time (P0..P6; P7 is PT, the
// hardwired-true predicate, not writable) -- then one predicated SEL per score keeps it or sets
// -inf. So per 32-key chunk it is ~5 R2P + 32 SEL: no branches, no per-key compares (the naive
// `if` would be 128 ISETP). Only k==0 (the ragged tile) runs this, so it is correctness/idiom,
// not a hot-path lever.
template <bool IS_CAUSAL, int K_TILE>
__device__ __forceinline__ void mask_s_row_r2p(float* scores, int k_offset, int q_pos, int seqlen_k) {
  int n_keep = seqlen_k - k_offset;                              // padding: keep kpos < seqlen_k
  if constexpr (IS_CAUSAL) {                                     // causal: also keep kpos <= q_pos
    const int causal = q_pos - k_offset + 1;
    n_keep = n_keep < causal ? n_keep : causal;
  }
  #pragma unroll
  for (int s = 0; s < K_TILE / 32; ++s) {
    int m = (s + 1) * 32 - n_keep;                               // # high cols to mask in chunk s
    m = m < 0 ? 0 : (m > 32 ? 32 : m);
    const uint32_t keep = (m >= 32) ? 0u : (0xFFFFFFFFu >> m);   // low (32-m) bits = keep
    #pragma unroll
    for (int i = 0; i < 32; ++i)
      if (!(keep & (1u << i))) scores[s * 32 + i] = -INFINITY;
  }
}

// Apply one byte of a 32-bit FA4 R2P mask. Keep the original mask register
// live across all four groups so ptxas selects its byte lanes directly.
template <int BIT>
__device__ __forceinline__ void mask_s_r2p_byte(
    uint32_t* values, uint32_t keep) {
  static_assert(BIT == 0 || BIT == 8 || BIT == 16 || BIT == 24);
  asm("{\n\t"
      ".reg .pred p0, p1, p2, p3, p4, p5, p6, p7;\n\t"
      ".reg .b32  t0, t1, t2, t3, t4, t5, t6, t7;\n\t"
      "and.b32 t0, %8, %9;  setp.ne.b32 p0, t0, 0; selp.b32 %0, %0, 0xFF800000, p0;\n\t"
      "and.b32 t1, %8, %10; setp.ne.b32 p1, t1, 0; selp.b32 %1, %1, 0xFF800000, p1;\n\t"
      "and.b32 t2, %8, %11; setp.ne.b32 p2, t2, 0; selp.b32 %2, %2, 0xFF800000, p2;\n\t"
      "and.b32 t3, %8, %12; setp.ne.b32 p3, t3, 0; selp.b32 %3, %3, 0xFF800000, p3;\n\t"
      "and.b32 t4, %8, %13; setp.ne.b32 p4, t4, 0; selp.b32 %4, %4, 0xFF800000, p4;\n\t"
      "and.b32 t5, %8, %14; setp.ne.b32 p5, t5, 0; selp.b32 %5, %5, 0xFF800000, p5;\n\t"
      "and.b32 t6, %8, %15; setp.ne.b32 p6, t6, 0; selp.b32 %6, %6, 0xFF800000, p6;\n\t"
      "and.b32 t7, %8, %16; setp.ne.b32 p7, t7, 0; selp.b32 %7, %7, 0xFF800000, p7;\n\t"
      "}"
      : "+r"(values[0]), "+r"(values[1]), "+r"(values[2]),
        "+r"(values[3]), "+r"(values[4]), "+r"(values[5]),
        "+r"(values[6]), "+r"(values[7])
      : "r"(keep), "n"(1u << (BIT + 0)), "n"(1u << (BIT + 1)),
        "n"(1u << (BIT + 2)), "n"(1u << (BIT + 3)),
        "n"(1u << (BIT + 4)), "n"(1u << (BIT + 5)),
        "n"(1u << (BIT + 6)), "n"(1u << (BIT + 7)));
}

// FA4 backward stores S transposed: one thread owns a key row while register
// columns are queries.  The causal diagonal therefore keeps a suffix rather
// than the prefix used by mask_s_row_r2p above.
template <int COLS>
__device__ __forceinline__ void mask_s_row_transposed_r2p(
    float* scores, int key_row, int query_col_offset) {
  static_assert(COLS % 32 == 0);
  uint32_t* u = reinterpret_cast<uint32_t*>(scores);
  #pragma unroll
  for (int s = 0; s < COLS / 32; ++s) {
    int lo = key_row - (query_col_offset + s * 32);
    lo = lo < 0 ? 0 : (lo > 32 ? 32 : lo);
    const uint32_t keep =
        lo >= 32 ? 0u : (0xFFFFFFFFu << lo);
    mask_s_r2p_byte<0>(u + s * 32, keep);
    mask_s_r2p_byte<8>(u + s * 32 + 8, keep);
    mask_s_r2p_byte<16>(u + s * 32 + 16, keep);
    mask_s_r2p_byte<24>(u + s * 32 + 24, keep);
  }
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

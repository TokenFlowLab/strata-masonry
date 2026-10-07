// fmha_utils.cuh -- shared device/host helpers for the sm100a FMHA context kernels.
//
// ARCH: sm_100a
//
// Contents:
//   make_magic / fdiv             -- FastDivmod tile-id decode (magic multiply)
//   full_bar_arrive / full_bar_wait -- per-band softmax<->correction HW named barrier
//   mask_s_row_r2p<IS_CAUSAL>     -- R2P-bitmask ragged/causal padding mask for one S row
//
// Self-contained: mask_s_row_r2p takes the tile width K_TILE as a template parameter, so this
// header can be included at the top with the other includes.

#pragma once

#include <cstdint>
#include <cmath>
#include "../../../primitives/37_bar_sync.cuh"   // bar_arrive<N> / bar_sync<N>
#include "../../../composites/109_fastdivmod.cuh" // make_magic / fdiv

// softmax->correction "scale ready" via per-band HW named barrier: id = 1 + m_tile*4 + band (1..8),
// 64 threads (1 softmax + 1 correction warp of the band); softmax arrives, correction waits.
// Protocol: bar.arrive N = arrive (increment count) + don't block; bar.sync N = arrive + block
// until N have arrived. So both contribute to the 64; only sync (correction) blocks.
// The CTA has 16 named barriers (ids 0-15); this kernel uses 0 (__syncthreads) + 1..8, 9-15 free.
// FA4 form: barrier id = compile-time base + runtime band, ONE bar instruction.
// Named-barrier scale handshake: id formula (1 + stage*4) + band; one BAR instruction.
__device__ __forceinline__ void full_bar_arrive(int stage, int band) {
  bar_arrive_dyn((uint32_t)((1 + stage * 4) + band), 64);
}
__device__ __forceinline__ void full_bar_wait(int stage, int band) {
  bar_sync_dyn((uint32_t)((1 + stage * 4) + band), 64);
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
// not a hot-path lever. SASS (one chunk):
//   SHF.R.U32.HI R136, RZ, R136, R137   ; keep = 0xFFFFFFFF >> m
//   SEL R136, R136, RZ, P0              ; (m>=32) ? 0 : keep
//   R2P PR, R136, 0x7e                  ; P1..P6 <- keep bits (0x7e=6 preds; 0x7f=7=P0..P6)
//   SEL R101, R101, 0xff800000, P1      ; score = keep_bit ? score : -inf
//   ...                                 ; one SEL/score; next R2P pulls bits via R136.B1/.B2/.B3
// USE_R2P_ASM=true (default): per-8-element asm blocks that pin the and+setp+selp PTX
// order so ptxas emits R2P (see note in the loop). false: the equivalent plain C++ --
// same semantics, kept for readability/reference; nvcc batches its bit-tests and the
// mask lowers to one LOP3+SEL per element instead.
template <bool IS_CAUSAL, int K_TILE, bool USE_R2P_ASM = true>
__device__ __forceinline__ void mask_s_row_r2p(float* scores, int k_offset, int q_pos, int seqlen_k) {
  int n_keep = seqlen_k - k_offset;                              // padding: keep kpos < seqlen_k
  if constexpr (IS_CAUSAL) {                                     // causal: also keep kpos <= q_pos
    const int causal = q_pos - k_offset + 1;
    n_keep = n_keep < causal ? n_keep : causal;
  }
  uint32_t* u = reinterpret_cast<uint32_t*>(scores);
  #pragma unroll
  for (int s = 0; s < K_TILE / 32; ++s) {
    int m = (s + 1) * 32 - n_keep;                               // # high cols to mask in chunk s
    m = m < 0 ? 0 : (m > 32 ? 32 : m);
    const uint32_t keep = (m >= 32) ? 0u : (0xFFFFFFFFu >> m);   // low (32-m) bits = keep
    if constexpr (USE_R2P_ASM) {
      // Per-8 asm blocks pin the and+setp+selp PER-ELEMENT order in the PTX. nvcc's
      // scheduler otherwise batches all 32 bit-tests ahead of the selects, leaving 32
      // predicates live -- past the 7 predicate registers -- so ptxas falls back to one
      // LOP3 per element. With <= 8 live predicates per block, ptxas emits R2P (verified:
      // hand-reordering the same PTX flips LOP3x128 -> R2Px16 in the non-causal variant;
      // the causal variants already came out interleaved and got R2P without help).
      // Blocks are NON-volatile pure-register ops: no scheduling walls, DCE-safe.
      #pragma unroll
      for (int g = 0; g < 32; g += 8) {
        const uint32_t kg = keep >> g;   // this group's 8 keep-bits in the low byte
        asm("{\n\t"
            ".reg .pred p0, p1, p2, p3, p4, p5, p6, p7;\n\t"
            ".reg .b32  t0, t1, t2, t3, t4, t5, t6, t7;\n\t"
            "and.b32 t0, %8, 1;    setp.ne.b32 p0, t0, 0; selp.b32 %0, %0, 0xFF800000, p0;\n\t"
            "and.b32 t1, %8, 2;    setp.ne.b32 p1, t1, 0; selp.b32 %1, %1, 0xFF800000, p1;\n\t"
            "and.b32 t2, %8, 4;    setp.ne.b32 p2, t2, 0; selp.b32 %2, %2, 0xFF800000, p2;\n\t"
            "and.b32 t3, %8, 8;    setp.ne.b32 p3, t3, 0; selp.b32 %3, %3, 0xFF800000, p3;\n\t"
            "and.b32 t4, %8, 16;   setp.ne.b32 p4, t4, 0; selp.b32 %4, %4, 0xFF800000, p4;\n\t"
            "and.b32 t5, %8, 32;   setp.ne.b32 p5, t5, 0; selp.b32 %5, %5, 0xFF800000, p5;\n\t"
            "and.b32 t6, %8, 64;   setp.ne.b32 p6, t6, 0; selp.b32 %6, %6, 0xFF800000, p6;\n\t"
            "and.b32 t7, %8, 128;  setp.ne.b32 p7, t7, 0; selp.b32 %7, %7, 0xFF800000, p7;\n\t"
            "}"
            : "+r"(u[s * 32 + g + 0]), "+r"(u[s * 32 + g + 1]),
              "+r"(u[s * 32 + g + 2]), "+r"(u[s * 32 + g + 3]),
              "+r"(u[s * 32 + g + 4]), "+r"(u[s * 32 + g + 5]),
              "+r"(u[s * 32 + g + 6]), "+r"(u[s * 32 + g + 7])
            : "r"(kg));
      }
    } else {
      // Reference C++ form (readable; lowers to LOP3+SEL per element under nvcc).
      #pragma unroll
      for (int i = 0; i < 32; ++i)
        if (!(keep & (1u << i))) scores[s * 32 + i] = -INFINITY;
    }
  }
}

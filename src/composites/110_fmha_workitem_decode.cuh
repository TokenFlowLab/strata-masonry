// 110_fmha_workitem_decode.cuh -- FMHA uniform work-item decode (1SM + 2SM):
// a flat tile_id (+ peer for a cta_group::CLUSTER_N cluster) ->
// (sample, kv-head, q_tile_base, K_TILES).
//
//   CLUSTER_N == 1 (default, 1CTA): each CTA owns one packed-M tile ->
//     q_tile_base = idx * q_tile_per_cta; causal cap taken from that tile.
//   CLUSTER_N == 2 (2SM cta_group::2): the cluster pairs two adjacent packed-M
//     tiles across its 2 CTAs (even -> peer 0, odd -> peer 1), so
//     q_tile_base = (2*idx + peer) * q_tile_per_cta; the causal K_TILES cap is
//     taken from the HIGHER (peer CLUSTER_N-1) tile of the pair so BOTH peers
//     keep equal K_TILES (the joint cta_group::2 MMA + cross-CTA barriers run in
//     lockstep; unequal K_TILES would desync the barrier counts -> hang).
//
// tile_id decodes as: sample (magic0 = packed-idx-per-sample), then kv-head +
// packed-idx (magic1/magic2), where "idx" is the packed-M tile (1SM) or the
// packed-M PAIR (2SM) index. Uniform (equal-seqlen) tiling; magic-divide via
// 109_fastdivmod. LPT (CLUSTER_N==1 causal only) reverses the packed-M order
// (heaviest causal tile first).

#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

#include <cstdint>
#include "109_fastdivmod.cuh"

template <int K_TILE, bool Q_RASTER, bool IS_CAUSAL, bool LPT, int CLUSTER_N = 1>
__device__ __forceinline__ void decode_workitem(
    int tile_id, int seqlen_kv, int num_kv_heads,
    int packed_idx_per_seq, int q_tile_per_cta,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    int& sample, int& h_kv, int& q_tile_base, int& k_tiles, int peer = 0) {
  const int packed_idx_per_sample = packed_idx_per_seq * num_kv_heads;
  sample = (int)fdiv((unsigned)tile_id, magic0);
  const int rr = tile_id - sample * packed_idx_per_sample;
  int idx;
  if constexpr (Q_RASTER) {
    h_kv = (int)fdiv((unsigned)rr, magic1);
    idx  = rr - h_kv * packed_idx_per_seq;
  } else {
    idx  = (int)fdiv((unsigned)rr, magic2);
    h_kv = rr - idx * num_kv_heads;
  }
  int base_idx = idx;
  if constexpr (IS_CAUSAL && LPT && CLUSTER_N == 1) base_idx = packed_idx_per_seq - 1 - idx;
  q_tile_base = (CLUSTER_N * base_idx + peer) * q_tile_per_cta;
  k_tiles = (seqlen_kv + K_TILE - 1) / K_TILE;
  if constexpr (IS_CAUSAL) {
    // Cap from the highest tile in the cluster (peer CLUSTER_N-1) so both peers match.
    const int cap_q_base = (CLUSTER_N * base_idx + (CLUSTER_N - 1)) * q_tile_per_cta;
    const int causal_k_tiles_cap = (cap_q_base + q_tile_per_cta - 1) / K_TILE + 1;
    if (causal_k_tiles_cap < k_tiles) k_tiles = causal_k_tiles_cap;
  }
}

// Varlen sibling (fmha_context_bf16_varlen.cu): variable per-sample seqlens, so the q and kv
// lengths come from prefix-sum arrays (cu_seqlens_q diff, seqlens_kv[sample]) instead of one
// scalar seqlen, and it emits an extra seqlen_q (used by the ragged-tail store predicate + the
// short-sample `q_tile_base < seqlen_q` guard). Plain integer divmod (no FastDivmod magics):
// the varlen header argues that is the right trade here (no host prefix-sum, slack tiles cheap).
// 1SM only -- no CLUSTER_N/peer. Shares the raster split, LPT reversal, and causal-cap shape.
template <int K_TILE, bool Q_RASTER, bool IS_CAUSAL, bool LPT>
__device__ __forceinline__ void decode_workitem_varlen(
    int tile_id, int num_kv_heads, int packed_idx_per_seq, int packed_idx_per_sample,
    int q_tile_per_cta, const int* cu_seqlens_q, const int* seqlens_kv,
    int& sample, int& h_kv, int& q_tile_base, int& seqlen_q, int& k_tiles) {
  sample = tile_id / packed_idx_per_sample;
  const int rr = tile_id - sample * packed_idx_per_sample;
  int idx;
  if constexpr (Q_RASTER) {
    idx  = rr % packed_idx_per_seq;
    h_kv = rr / packed_idx_per_seq;
  } else {
    idx  = rr / num_kv_heads;
    h_kv = rr % num_kv_heads;
  }
  const int base_idx = LPT ? packed_idx_per_seq - 1 - idx : idx;
  q_tile_base = base_idx * q_tile_per_cta;
  seqlen_q = cu_seqlens_q[sample + 1] - cu_seqlens_q[sample];
  k_tiles = (seqlens_kv[sample] + K_TILE - 1) / K_TILE;
  if constexpr (IS_CAUSAL) {
    const int causal_k_tiles_cap = (q_tile_base + q_tile_per_cta - 1) / K_TILE + 1;
    if (causal_k_tiles_cap < k_tiles) k_tiles = causal_k_tiles_cap;
  }
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

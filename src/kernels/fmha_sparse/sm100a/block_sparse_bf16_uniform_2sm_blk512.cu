// block_sparse_bf16_uniform_2sm_blk512.cu -- VSA "fine" block-sparse FMHA, bf16, 2SM
// (cta_group::2 cluster), 512-token sparse block, sm_100a.
//
// Shares the VSA kernels' 16-warp warp-specialized body, BMM1-ahead software pipeline,
// softmax<->correction<->MMA barrier contract, CLC persistent scheduling, split-P and TMA-store
// epilogue, plus the cta_group::2 specifics (joint 256-row MMA issued by peer 0, cross-CTA mbarrier
// routing via mapa, cluster-collective TMEM alloc<2>, peer-1 idle CLC consumer, the teardown
// contract below).
//
// WHY blk512 EXISTS -- and WHAT IT MEASURED (read this before "improving" it):
//   K/V traffic scales as 1 / (query rows behind one gather). A 1CTA kernel is stuck at 256 rows per
//   gather at ANY block size (4 M-tiles would need S+O = 1024 TMEM columns vs the 512 budget). A 2SM
//   cluster covers 512 query rows, but at blk256 those 512 rows are TWO 256-blocks with TWO lists, so
//   it must run two interleaved streams and its per-CTA traffic equals 1CTA's -- which is exactly why
//   2SM blk256 is a wash. At blk512 the cluster's 512 rows ARE one 512-block with ONE list, so this
//   kernel runs ONE shared K/V stream and gets 512 rows per gather: half the K/V requests of every
//   other VSA kernel here. That part WORKS and is measured -- ncu L2 read sectors at S=32768:
//   133.8M (1CTA blk256) -> 68.7M here, a 1.95x reduction (1.98x at S=131072).
//   THE TRAFFIC SAVING IS NOT WHY IT WINS, THOUGH: these shapes are compute/pipeline-bound, not
//   bandwidth-bound. DRAM traffic is UNCHANGED (251.6 -> 254.1 MB at S=32768) at ~0.2-0.35 TB/s of
//   an ~8 TB/s HBM = 3-4% utilization, already the compulsory floor -- the requests removed were L2
//   hits that cost nothing. Its speed comes from the tuned pipeline, above all the FA4-shape softmax.
//   It is on par with 1CTA blk256 (ahead at 8K and 131K); pick per shape.
//
// VSA design (fixed):
//   - SPARSE_BLOCK = 512 tokens: the top-k selection granularity AND the query-block granularity
//     (square, R = C = 512). The MMA datapath stays at 128: BLOCK = M_TILE = K_TILE = 128, so a
//     selected 512-block is gathered as KTILES_PER_BLOCK = 4 consecutive 128-token K-tiles and
//     num_k_tiles = nkv * 4. K_TILE cannot be 256: TMEM holds S[0],S[1],O[0],O[1] = 4 x 128 =
//     the full 512 columns, and a 256-wide S would need 768.
//   - THE KEY PROPERTY: a joint cta_group::2 MMA has M = JOINT_M = 256, and the cluster runs 2 joint
//     M-tiles, so it covers 512 query rows = EXACTLY ONE 512-block -> every row in the cluster indexes
//     the SAME top-k list. Zero union inflation, one id cache, ONE K/V stream.
//   - Work item: one CLUSTER per (sample, head, 512-block pidx) -- no block pairing, so nb needs no
//     parity. Joint M-tile m = rows [pidx*512 + m*256, +256); peer p supplies its [p*128, +128) half.
//   - ONE K/V STREAM (the dense blocks/111 discipline): the ring slot is waited once,
//     BOTH joint M-tiles' MMA groups issue against it, and it is freed once after the last group.
//     Produce/consume order: K(0) | per kt: V(kt), K(kt+1) | V(last).
//     GOTCHA: with one slot feeding two groups the MMA warp MUST walk its descriptors with
//     desc_add_lo (see SmemDescPair below), or the two groups' identical descriptor gets CSE'd and
//     the second group reads row 0 of the tile. Gate any change here with DUMP_O +
//     block_sparse_bf16_cpu_verifier.py, not the built-in verify.
//   - K loading: the 2SM K half-box per peer covers 64 tokens; K-tile kt is the (kt % 4)-th 128-token
//     quarter of selected 512-block (kt / 4), so peer p loads coordinate
//     sample*seqlen + block_id*512 + (kt % 4)*128 + peer*64.
//   - V loading: per peer, V_SUBTILES = 2 pieces of 64 tokens at the same token coordinate, V^T row
//     coordinate head*HEAD_DIM + peer*(HEAD_DIM/2) (D-split across the peers).
//   - Ring: slots are 16 KB = THIS peer's half-box only, NUM_KV_STAGES = 6 -> 96 KB of SMEM.
//   - MHA only (HQ == HK), non-causal, D = 128, bf16, uniform seqlen = nb * SPARSE_BLOCK.
//
// TMEM LIFETIME / TEARDOWN (dense-2SM contract -- do not reintroduce a named barrier here): the MMA
//   warp OWNS the TMEM (tcgen05_alloc<2> before the warp dispatch, dealloc after it), as in the GEMM
//   2SM path (blocks/90). *tmem_slot is published by the __syncthreads() + cluster join that already
//   precede the dispatch, so no entry rendezvous is needed. Teardown is wp_flush + __syncthreads()
//   (this CTA is done with TMEM) + cluster join (the peer is too) + dealloc; leftover in-flight
//   mbarrier arrives at kernel exit are harmless and are NOT drained. A 13-warp
//   bar_arrive<9>/bar_sync<9> protocol plus closing primes and closed-form tail drains deadlocks
//   intermittently here (a closing prime racing the MMA's epilogue waits).
//
// Barrier contract, budgets (softmax inc<176> x8 + corr dec<88> x4 + singles dec<72> x4), TMEM
// layout (S[i] at i*128, O[i] at 256 + i*128, 512 cols) and the EX2_EMU hybrid exp2 are as in
// fmha_context_bf16_uniform_2sm_inline.cu -- see its header. The blocks/98 P-publish wait_st race fix
// is inherited: every P tcgen05.st is drained with tcgen05_wait_st() BEFORE the
// cross-CTA empty_bar_spo / full_bar_p_last arrive.
//
// PERF NOTE: blk512 removes the structural objection to 2SM that blk256 had (there, per CTA, the
// two streams loaded 64+64 tokens per K-step -- exactly the 128 tokens 1CTA loads once and shares).
// Here one gather feeds 512 query rows and the K/V requests really do halve (ncu-confirmed) -- but
// that saving is free, not decisive (see the header).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include "npy_io.cuh"
#include "block_sparse_bf16_benchmark.cuh"
#include "../../../../tests/test_utils.cuh"
#include "../../../primitives/0_tcgen05_alloc.cuh"
#include "../../../primitives/1_tcgen05_dealloc.cuh"
#include "../../../primitives/2_tcgen05_relinquish.cuh"
#include "../../../primitives/3_tcgen05_mma_f16.cuh"
#include "../../../primitives/8_tcgen05_mma_idesc.cuh"
#include "../../../primitives/9_tcgen05_ld.cuh"
#include "../../../primitives/10_tcgen05_st.cuh"
#include "../../../primitives/11_tcgen05_commit.cuh"
#include "../../../primitives/12_tcgen05_wait.cuh"
#include "../../../primitives/15_tcgen05_fence.cuh"
#include "../../../primitives/18_tma_load.cuh"
#include "../../../primitives/19_tma_load_2sm.cuh"
#include "../../../primitives/22_tma_store.cuh"
#include "../../../primitives/23_tma_tensormap.cuh"
#include "../../../primitives/25_tma_async_group.cuh"
#include "../../../primitives/29_mbarrier_init.cuh"
#include "../../../primitives/30_mbarrier_arrive.cuh"
#include "../../../primitives/31_mbarrier_arrive_tx.cuh"
#include "../../../primitives/33_mbarrier_try_wait.cuh"
#include "../../../primitives/34_fence_proxy_async.cuh"
#include "../../../primitives/35_fence_mbarrier_init.cuh"
#include "../../../primitives/42_smem_desc_blackwell.cuh"
#include "../../../primitives/44_elect_sync.cuh"
#include "../../../primitives/37_bar_sync.cuh"
#include "../../../primitives/38_barrier_cluster.cuh"
#include "../../../primitives/67_mapa.cuh"
#include "../../../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../../../composites/118_mbarrier_phase_tracking.cuh"
#include "../../../composites/106_clc_fetch_next_tile.cuh"
#include "../../../composites/110_fmha_workitem_decode.cuh"
#include "../../../blocks/88_load_warp_blackwell.cuh"
#include "../../../blocks/93_epi_warp_blackwell.cuh"
#include "../../../blocks/97_sched_warp_clc.cuh"
#include "../../../blocks/98_softmax_warp.cuh"
#include "../../../blocks/99_correction_warp.cuh"
#include "../../../blocks/111_fmha_mma_warp_blackwell.cuh"
#include "../../../primitives/46_setmaxnreg.cuh"
#include "../../../primitives/76_packed_f32x2.cuh"
#include "../../../primitives/77_ex2_approx.cuh"
#include "../../../primitives/78_rcp_approx.cuh"
#include "../../../primitives/_warp_prof_noop.cuh"
#include "../../../composites/109_fastdivmod.cuh"
#include "../../../composites/112_fmha_softmax_utils.cuh"

// The 512-token SPARSE block = the cluster's 2 joint M-tiles, so every row indexes the SAME top-k
// list -> ZERO union inflation, no union / membership-bitmask machinery.
constexpr int SPARSE_BLOCK = 512;                // top-k selection granularity == the CLUSTER's query rows
constexpr int BLOCK  = 128;                      // MMA datapath tile (per-CTA M rows / keys per K-tile)
constexpr int M_TILE = BLOCK;                    // this CTA's 128-row half of a joint 256-row M-tile
constexpr int JOINT_M = 2 * M_TILE;              // 256: one joint cta_group::2 MMA's M-dim
constexpr int M_TILES_PER_CTA = 2;               // 2 joint M-tiles = 512 rows = ONE 512-block (softmax/MMA overlap)
constexpr int K_TILE = BLOCK;                    // 128 keys per MMA (N; TMEM-bound: S0,S1,O0,O1 = 4x128 = 512)
constexpr int KTILES_PER_BLOCK = SPARSE_BLOCK / K_TILE;   // 4 K-tiles per selected 512-block
constexpr int HEAD_DIM = 128;
// B128 swizzle atom = 128 bytes = 64 bf16: all SMEM tiles are laid out in
// 64-wide sub-tiles along the contiguous dim.
constexpr int SUB_COLS_BF16 = 64;
constexpr int SUB_COLS_BYTES = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);   // 128 B (one swizzle atom)
constexpr int Q_SUBTILES = HEAD_DIM / SUB_COLS_BF16;  // 2
constexpr int K_SUBTILES = HEAD_DIM / SUB_COLS_BF16;  // 2 (K tile is K_TILE tokens x head_dim)
constexpr int V_SUBTILES = K_TILE / SUB_COLS_BF16;    // 2 (per-peer half-hd V token atoms)
constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;       // 16 KB
constexpr int Q_TILE_BYTES = Q_SUBTILES * Q_SUB_COLS_BYTES;     // 32 KB
constexpr int K_TILE_BYTES = K_SUBTILES * (K_TILE * SUB_COLS_BYTES);   // 32 KB = both peers' halves
// Ring slot = THIS peer's half-box only (16 KB), as in the dense 2SM kernel: 96 KB of SMEM buys 6 stages.
constexpr int KV_SLOT_BYTES = K_TILE_BYTES / 2;                        // 16 KB
constexpr int V_TILE_BYTES = V_SUBTILES * Q_SUB_COLS_BYTES;     // 32 KB across both peers
constexpr int NUM_KV_STAGES = 6;
constexpr int TMEM_TOTAL = 512;                     // S0,S1(128*2)+O0,O1(128*2)=512
constexpr int SPLIT_P_COL = (K_TILE / 4 * 3) / 2;   // 48 (u32 P cols before the empty_bar_spo signal)
constexpr int EX2_FRG_PAIRS = 16;          // 32 elts / fragment = 16 pairs
constexpr int EX2_FRG_CNT   = K_TILE / 32; // 4
constexpr int EX2_FREQ      = 10;          // FA4-2CTA ex2_emu_freq (more FFMA-emu than 1CTA's 16)
constexpr int EX2_RES       = 4;           // FA4 ex2_emu_res
constexpr int EX2_START_FRG = 1;           // FA4-2CTA: fragment 0 pure-HW EX2
constexpr int W_CORR0 = 8;
// 64-bit SMEM descriptor; the atom walk adds to the LOW word only (the hi/swizzle word is
// walk-invariant). As in the dense fmha_context_bf16_uniform_inline.cu, and LOAD-BEARING FOR
// CORRECTNESS in any kernel where ONE ring slot feeds TWO MMA groups (which is exactly what blk512's
// shared K/V stream does): with plain uint64 descriptor arithmetic the compiler CSEs the identical
// descriptor across the two groups into base+imm forms that rematerialize the hi word, and the second
// group's B operand comes out wrong -- every output column reads row 0 of the tile. asm volatile pins
// the serial chain and blocks that.
union SmemDescPair { uint64_t u64; uint2 w; };

__device__ __forceinline__ void desc_add_lo(SmemDescPair& d, uint32_t inc) {
  asm volatile("{\n\t"
      ".reg .b32 lo, hi;\n\t"
      "mov.b64 {lo, hi}, %0;\n\t"
      "add.u32 lo, lo, %1;\n\t"
      "mov.b64 %0, {lo, hi};\n\t"
      "}" : "+l"(d.u64) : "r"(inc));
}

constexpr int W_MMA = 12, W_EPI = 13, W_LOAD = 14, W_SCHED = 15;
constexpr int N_WARPS = 16;
constexpr int CLC_STAGES = 2;

extern __shared__ __align__(1024) uint8_t fmha_smem[];

// VSA 2SM work-item decode. cluster_workitem_id -> (sample, head, 512-block pidx). THE WHOLE POINT OF
// blk512: the cluster's 2 joint M-tiles are 2 x 256 = 512 query rows = EXACTLY ONE 512 sparse block, so
// both joint M-tiles ride ONE top-k list (q2k_row0, no "+ m") and the kernel runs ONE shared K/V stream.
// That is what blk256 could not do -- there a cluster spanned two different 256-blocks with two lists.
// nkv = q2k_num[q2k_row0] is read by EVERY role on BOTH peers -> identical K-loop trip counts
// (lockstep, like the 1CTA VSA's decode_workitem with q2k_num). Q_RASTER=true: block innermost (concurrent
// clusters sit within one head -> its KV/topk blocks stay hot in L2); false: head innermost.
// BOTH peers decode the SAME rows (nothing here depends on `peer`) -> joint-MMA lockstep.
struct Vsa2Item { int sample, head, pidx, q2k_row0, nkv; };
template <bool Q_RASTER>
__device__ __forceinline__ Vsa2Item vsa2_decode(
    int cluster_workitem_id, int num_heads, int clusters_per_seq, int num_blocks,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    const int* __restrict__ q2k_num) {
  Vsa2Item it;
  const int per_sample = num_heads * clusters_per_seq;
  it.sample = (int)fdiv((unsigned)cluster_workitem_id, magic0);
  const int rem = cluster_workitem_id - it.sample * per_sample;
  if constexpr (Q_RASTER) {
    it.head = (int)fdiv((unsigned)rem, magic2);
    it.pidx = rem - it.head * clusters_per_seq;
  } else {
    it.pidx = (int)fdiv((unsigned)rem, magic1);
    it.head = rem - it.pidx * num_heads;
  }
  // row of the ONE 512-block this cluster covers, in the per-512-block top-k tables
  it.q2k_row0 = (it.sample * num_heads + it.head) * num_blocks + it.pidx;
  it.nkv = q2k_num[it.q2k_row0];
  return it;
}

// Compile-time kernel config (template args, set in run()'s `constexpr` block):
//   S_LD_COLS        : cols per softmax tcgen05.ld of the S row (32/64 compile; 128 aborts ptxas).
//   FULL_NAMED_BAR   : softmax->correction "scale ready" signal -- true = HW named barrier
//                      (per-band), false = mbarrier (full_bar_alpha/full_bar_l).
//   EX2_EMU          : route a fraction of softmax exp2 through FFMA f32x2 emulation (vs MUFU.EX2).
//   SPLIT_P          : softmax publishes P in two chunks (96+32 keys); BMM2 starts on the first,
//                      full_bar_p_last gates the tail atoms.
//   SOFTMAX_THROTTLE : FA4 softmax pacing (correction defers the alpha/l slot release).
//   USE_CLC          : true = cluster CLC work-stealing sched (grid = full problem, 2 CTAs/cluster);
//                      false = static cluster-stride loop (w15 idle, grid=#SMs).
//   Q_RASTER         : see vsa2_decode above.
// (MHA fixed true: q2k lists are per (sample, head, q-block). Non-causal, no LPT.)
template <int S_LD_COLS = 32, bool FULL_NAMED_BAR = false, bool EX2_EMU = false, bool SPLIT_P = true,
          bool SOFTMAX_THROTTLE = false, bool USE_CLC = true, bool Q_RASTER = true, int RESCALE_THRESHOLD = 8>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
fmha_context_bf16_vsa_2sm_kernel(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v_t,
    const __grid_constant__ CUtensorMap tmap_o, int seqlen, int num_heads, float scale_log2,
    int num_samples, int clusters_per_seq, int num_blocks, int max_kv,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    const int* __restrict__ q2k_idx, const int* __restrict__ q2k_num) {
  const int peer           = (int)(blockIdx.x & 1);
  const int total_workitems = num_samples * num_heads * clusters_per_seq;

  uint8_t* sQ0 = fmha_smem;
  uint8_t* sQ1 = sQ0 + Q_TILE_BYTES;
  uint8_t* sQ[2] = { sQ0, sQ1 };
  uint8_t* sKV = sQ1 + Q_TILE_BYTES;
  __nv_bfloat16* sO0 = reinterpret_cast<__nv_bfloat16*>(sKV + NUM_KV_STAGES * KV_SLOT_BYTES);
  __nv_bfloat16* sO1 = sO0 + M_TILE * HEAD_DIM;
  __nv_bfloat16* sO_bufs[2] = { sO0, sO1 };
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(
      reinterpret_cast<uint8_t*>(sO1) + M_TILE * HEAD_DIM * sizeof(__nv_bfloat16));
  uint64_t* empty_bar             = full_bar + NUM_KV_STAGES;
  uint64_t* full_bar_q            = empty_bar + NUM_KV_STAGES;
  uint64_t* empty_bar_q           = full_bar_q + 2;
  uint64_t* full_bar_spo          = empty_bar_q + 2;
  uint64_t* empty_bar_spo         = full_bar_spo + 2;
  uint64_t* full_bar_o_acc        = empty_bar_spo + 2;
  uint64_t* full_bar_alpha        = full_bar_o_acc + 2;
  uint64_t* full_bar_l            = full_bar_alpha + 2;
  uint64_t* full_bar_p_last       = full_bar_l + 2;
  uint64_t* empty_bar_alpha_and_l = full_bar_p_last + 2;
  uint64_t* full_bar_o_epi        = empty_bar_alpha_and_l + 2;
  uint64_t* empty_bar_o_epi       = full_bar_o_epi + 2;
  uint64_t* clc_full              = empty_bar_o_epi + 2;
  uint64_t* clc_empty             = clc_full + CLC_STAGES;
  uint64_t* throttle_full         = clc_empty + CLC_STAGES;
  uint64_t* throttle_empty        = throttle_full + 2;
  uint32_t* clc_response = reinterpret_cast<uint32_t*>(
      (reinterpret_cast<uintptr_t>(throttle_empty + 2) + 15u) & ~uintptr_t(15u));
  uint32_t* tmem_slot        = clc_response + CLC_STAGES * 4;
  uint64_t* tmem_dealloc_bar = reinterpret_cast<uint64_t*>(tmem_slot + 2);
  float*    alpha_and_l_smem = reinterpret_cast<float*>(tmem_dealloc_bar + 2);

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
  WpCtx wpc = wp_ctx_init();

  // The MMA warp OWNS the TMEM lifetime (alloc here, dealloc at the end) -- same ownership as the
  // GEMM 2SM path (blocks/90). Publication needs no extra signal: the alloc happens before the
  // __syncthreads() + cluster join below, which is what makes *tmem_slot visible to every warp.
  if (warp_id == W_MMA) {
    tcgen05_alloc<2>(smem_ptr_u32(tmem_slot), TMEM_TOTAL);
    tcgen05_relinquish_alloc_permit<2>();
  }

  if (tid == 0) {
    #pragma unroll
    for (int s = 0; s < NUM_KV_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar[s]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar[s]), 1);
    }
    for (int i = 0; i < 2; ++i) {
      mbarrier_init(smem_ptr_u32(&full_bar_q[i]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_q[i]), 1);
      mbarrier_init(smem_ptr_u32(&full_bar_l[i]), 128);
      mbarrier_init(smem_ptr_u32(&full_bar_spo[i]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_spo[i]), 512);
      mbarrier_init(smem_ptr_u32(&full_bar_o_acc[i]), 1);
      mbarrier_init(smem_ptr_u32(&full_bar_alpha[i]), 128);
      mbarrier_init(smem_ptr_u32(&empty_bar_alpha_and_l[i]), 128);
      mbarrier_init(smem_ptr_u32(&full_bar_p_last[i]), 256);
      mbarrier_init(smem_ptr_u32(&full_bar_o_epi[i]), 128);
      mbarrier_init(smem_ptr_u32(&empty_bar_o_epi[i]), 1);
    }
    mbarrier_init(smem_ptr_u32(tmem_dealloc_bar), 32);
    if constexpr (USE_CLC) {
      #pragma unroll
      for (int s = 0; s < CLC_STAGES; ++s) {
        mbarrier_init(smem_ptr_u32(&clc_full[s]), 1);
        mbarrier_init(smem_ptr_u32(&clc_empty[s]), N_WARPS * 2);
      }
      for (int s = 0; s < 2; ++s) {
        mbarrier_init(smem_ptr_u32(&throttle_full[s]), 32);
        mbarrier_init(smem_ptr_u32(&throttle_empty[s]), 32);
      }
      #pragma unroll
      for (int i = 0; i < CLC_STAGES * 4; ++i)
        clc_response[i] = 0;
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();
  barrier_cluster_arrive();
  barrier_cluster_wait();

  if (warp_id == W_LOAD) {
    // VSA load warp (inlined blocks/88 2SM variant): the seqlen-walk k_offset becomes a top-k
    // block-id gather, in ONE stream shared by both joint M-tiles (they are the two halves of the
    // same 512-block). Block-id prefetch: 32 ids in a lane-spread register chunk (one coalesced read
    // per 32 positions, reused by K AND V), ONE cache for the cluster; warp-collective (shfl) ->
    // compute OUTSIDE elect_one_sync.
    setmaxnreg_dec<72>();

    EmptyPhaseTracker<NUM_KV_STAGES> kv_empty_ph;
    EmptyPhaseTracker<1> q_empty_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    [[maybe_unused]] int thr_prod_stage = 0;
    [[maybe_unused]] uint32_t thr_prod_phase = 1;
    int cluster_workitem_id = (int)(blockIdx.x >> 1);
    while (true) {
      const Vsa2Item it = vsa2_decode<Q_RASTER>(cluster_workitem_id, num_heads, clusters_per_seq, num_blocks, magic0, magic1, magic2, q2k_num);

      // Throttle producer paces the sched warp; only the leader CTA runs it.
      if constexpr (USE_CLC) {
        if (peer == 0) {
          wp_begin(wpc, WP_LOAD_WAIT_THROTTLE);
          mbarrier_wait_parity_suspend(smem_ptr_u32(&throttle_empty[thr_prod_stage]), thr_prod_phase);
          wp_end(wpc, WP_LOAD_WAIT_THROTTLE);
          mbarrier_arrive_nostate(smem_ptr_u32(&throttle_full[thr_prod_stage]));
          advance_stage_phase<2>(thr_prod_stage, thr_prod_phase);
        }
      }

      const int k_start = it.sample * seqlen;
      const int num_k_tiles = it.nkv * KTILES_PER_BLOCK;     // 4 K-tiles per selected 512-block

      // ONE top-k id cache for the whole cluster: both joint M-tiles are halves of the SAME 512-block,
      // so there is ONE list. 32 ids per coalesced read, reused by K and V; warp-collective (shfl) ->
      // must be called OUTSIDE elect_one_sync.
      int id_chunk_base = -(1 << 30);
      int id_chunk_val  = 0;
      auto sel_block_id = [&](int j) -> int {
        if (j < id_chunk_base || j >= id_chunk_base + 32) {
          id_chunk_base = j & ~31;
          const int idx = min(id_chunk_base + lane, it.nkv - 1);   // clamp: tail lanes stay in-row
          id_chunk_val = q2k_idx[(size_t)it.q2k_row0 * max_kv + idx];
        }
        return __shfl_sync(0xffffffffu, id_chunk_val, j & 31);
      };
      // K-tile kt = the (kt%4)-th 128-token quarter of selected 512-block (kt/4).
      auto kv_token_of = [&](int kt) -> int {
        return k_start + sel_block_id(kt / KTILES_PER_BLOCK) * SPARSE_BLOCK
                       + (kt % KTILES_PER_BLOCK) * K_TILE;
      };

      auto load_k = [&](int kt) {
        const int kv_tok = kv_token_of(kt);                 // warp-collective (shfl inside)
        const int kv_stage = kv_empty_ph.get_stage();
        wp_begin(wpc, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        wp_end(wpc, WP_LOAD_WAIT);

        wp_begin(wpc, WP_LOAD_ISSUE_K);
        const uint32_t kbar = tma_peer_bit_mask(smem_ptr_u32(&full_bar[kv_stage]));
        const uint32_t kdst = smem_ptr_u32(sKV + kv_stage * KV_SLOT_BYTES);
        const int k_token = kv_tok + peer * (K_TILE / 2);   // N-split: peer p takes tokens [.. + p*64, +64)
        const int k_head_atom = it.head * K_SUBTILES;
        if (elect_one_sync()) {
          if (peer == 0) mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), K_TILE_BYTES);
          tma_load_3d_2sm(kdst, &tmap_k, kbar, 0, k_token, k_head_atom);
        }
        wp_end(wpc, WP_LOAD_ISSUE_K);
      };
      auto load_v = [&](int kt) {
        const int kv_tok = kv_token_of(kt);
        const int kv_stage = kv_empty_ph.get_stage();
        wp_begin(wpc, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        wp_end(wpc, WP_LOAD_WAIT);

        wp_begin(wpc, WP_LOAD_ISSUE_V);
        const uint32_t vbar = tma_peer_bit_mask(smem_ptr_u32(&full_bar[kv_stage]));
        const int v_col = it.head * HEAD_DIM + peer * (HEAD_DIM / 2);   // V^T is D-split across peers
        if (elect_one_sync()) {
          if (peer == 0) mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar[kv_stage]), V_TILE_BYTES);
          #pragma unroll
          for (int s = 0; s < V_SUBTILES; ++s) {
            tma_load_2d_2sm(smem_ptr_u32(sKV + kv_stage * KV_SLOT_BYTES + s * (Q_SUB_COLS_BYTES / 2)), &tmap_v_t, vbar,
                            kv_tok + s * SUB_COLS_BF16, v_col);
          }
        }
        wp_end(wpc, WP_LOAD_ISSUE_V);
      };

      // BMM1-ahead produce order, ONE shared stream (both joint M-tiles consume the same tiles):
      //   K(0) | per kt: V(kt), K(kt+1) | V(last)
      load_k(0);

      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        wp_begin(wpc, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
        wp_end(wpc, WP_LOAD_WAIT);

        wp_begin(wpc, WP_LOAD_ISSUE_Q);
        const uint32_t qbar = tma_peer_bit_mask(smem_ptr_u32(&full_bar_q[m]));
        // joint M-tile m = the m-th 256-row half of 512-block pidx; this peer supplies its
        // [peer*128, +128) rows of that half.
        const int q_token = it.pidx * SPARSE_BLOCK + m * JOINT_M + peer * M_TILE;
        if (elect_one_sync()) {
          if (peer == 0) mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_q[m]), 2 * Q_TILE_BYTES);
          #pragma unroll
          for (int s = 0; s < Q_SUBTILES; ++s) {
            tma_load_4d_2sm(smem_ptr_u32(sQ[m] + s * Q_SUB_COLS_BYTES), &tmap_q, qbar,
                            s * SUB_COLS_BF16, it.head, q_token, it.sample);
          }
        }
        wp_end(wpc, WP_LOAD_ISSUE_Q);
      }
      q_empty_ph.advance();

      for (int kt = 0; kt + 1 < num_k_tiles; ++kt) {
        load_v(kt); load_k(kt + 1);
      }
      load_v(num_k_tiles - 1);
      __syncwarp();  // converge after the elect-issued TMA loads

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN, /*CTA_GROUP=*/2, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        cluster_workitem_id = next.n_tile;
      } else {
        cluster_workitem_id += (int)(gridDim.x >> 1);
        if (cluster_workitem_id >= total_workitems) break;
      }
    }
  }
  else if (warp_id == W_MMA) {
    // MMA warp: owns the TMEM lifetime (alloc before the dispatch, dealloc after it) and issues the
    // joint cta_group::2 MMAs on peer 0; peer 1 runs the same loop as an idle CLC consumer for
    // count-balance. The per-work-item body is INLINED below instead of calling blocks/111.
    setmaxnreg_dec<72>();
    constexpr int M_TILE_CLUSTER = JOINT_M;      // joint cta_group::2 MMA M-dim (256) = half a 512-block
    // MMA-local geometry.
    constexpr int K_ATOMS_PER_TILE = SUB_COLS_BF16 / 16;                 // 4
    constexpr int S_COLS = K_TILE;                                       // 128
    constexpr int O_COLS = HEAD_DIM;                                     // 128
    constexpr int SPLIT_P_ATOM = (K_TILE / 4 * 3) / 16;                  // 6
    constexpr uint64_t KV_DESC_DELTA      = (uint64_t)KV_SLOT_BYTES >> 4;            // ring-slot stride (16KB half-box)
    constexpr uint64_t SUB_DESC_DELTA_Q   = (uint64_t)Q_SUB_COLS_BYTES >> 4;         // Q hd subtile (16KB)
    constexpr uint64_t SUB_DESC_DELTA_KV  = (uint64_t)(Q_SUB_COLS_BYTES / 2) >> 4;   // K/V hd subtile (8KB, half-box)
    constexpr uint64_t Q_MTILE_DESC_DELTA = (uint64_t)(Q_SUBTILES * Q_SUB_COLS_BYTES) >> 4;

    // tmem_base is already published: this warp alloc'd it before the __syncthreads() + cluster
    // barrier that precede the warp dispatch, so no entry rendezvous is needed.
    const uint32_t tmem_base = *tmem_slot;
    PhaseTracker<1> spo_ph;   // shared by the peer-0 MMA loop and the tail empty_bar_spo drain

    const bool lead = elect_one_sync();
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    // 2SM: idesc M = M_TILE_CLUSTER (256); N stays 128 -- each CTA supplies its half of B.
    const uint32_t idesc_qk = make_idesc_bf16_f32(M_TILE_CLUSTER, K_TILE, false, false);
    const uint32_t idesc_pv = make_idesc_bf16_f32(M_TILE_CLUSTER, HEAD_DIM, false, false);
    const uint64_t desc_q0  = build_smem_desc_blackwell(smem_ptr_u32(sQ0), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_kv0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);

    PhaseTracker<NUM_KV_STAGES> kv_ph;
    PhaseTracker<1> q_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int cluster_workitem_id = (int)(blockIdx.x >> 1);
    while (true) {
      const Vsa2Item it = vsa2_decode<Q_RASTER>(cluster_workitem_id, num_heads, clusters_per_seq, num_blocks, magic0, magic1, magic2, q2k_num);

      // Only peer 0 (leader) issues the joint cta_group::2 MMA; peer 1 runs the loop for
      // CLC count-balance (clc_empty expects 32 arrives/work-item; a missing peer-1 -> SILENT HANG).
      if (peer == 0) {
        const int num_k_tiles = it.nkv * KTILES_PER_BLOCK;
        // ONE SHARED K/V STREAM (the dense uniform_inline discipline): the cluster's two
        // joint M-tiles are the two 256-row halves of ONE 512-block, so they consume the SAME K/V tiles.
        // The caller waits and frees the ring slot; bmm1/bmm2 take it and do no ring wait/commit.
        // Descriptors are WALKED with desc_add_lo -- with one slot feeding both groups, the plain
        // base+imm form hits the CSE bug (see SmemDescPair above).
        auto bmm1 = [&](int i, int slot) {
          wp_begin(wpc, WP_MMA_ISSUE);
          if (lead) {
            const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
            SmemDescPair da, db;
            da.u64 = desc_q0;  da.w.x += (uint32_t)(i * (int)Q_MTILE_DESC_DELTA);
            db.u64 = desc_kv0; db.w.x += (uint32_t)(slot * (int)KV_DESC_DELTA);
            #pragma unroll
            for (int s = 0; s < Q_SUBTILES; ++s) {
              #pragma unroll
              for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
                const bool enable_d = (s != 0) || (ki != 0);
                tcgen05_mma_f16_ss<2>(s_tmem_addr, da.u64, db.u64, idesc_qk, enable_d);
                desc_add_lo(da, 2); desc_add_lo(db, 2);
              }
              desc_add_lo(da, (uint32_t)(SUB_DESC_DELTA_Q  - 2 * K_ATOMS_PER_TILE));
              desc_add_lo(db, (uint32_t)(SUB_DESC_DELTA_KV - 2 * K_ATOMS_PER_TILE));
            }
          }
          wp_end(wpc, WP_MMA_ISSUE);

          wp_begin(wpc, WP_MMA_COMMIT);
          if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&full_bar_spo[i]), 0x3);
          wp_end(wpc, WP_MMA_COMMIT);
        };
        auto bmm2 = [&](int i, int slot, bool first_ktile, bool last_ktile) {
          wp_begin(wpc, WP_MMA_ISSUE);
          const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
          const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
          SmemDescPair dbV;
          dbV.u64 = desc_kv0; dbV.w.x += (uint32_t)(slot * (int)KV_DESC_DELTA);
          #pragma unroll
          for (int s = 0; s < V_SUBTILES; ++s) {
            #pragma unroll
            for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
              const int a = s * K_ATOMS_PER_TILE + ki;
              if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
                wp_end(wpc, WP_MMA_ISSUE);
                wp_begin(wpc, WP_MMA_WAIT_P);
                mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
                wp_end(wpc, WP_MMA_WAIT_P);
                wp_begin(wpc, WP_MMA_ISSUE);
              }
              const bool accumulate = (!first_ktile) || (a != 0);
              if (lead)
                tcgen05_mma_f16_ts_2sm(o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), dbV.u64,
                                       idesc_pv, accumulate, 0, 0, 0, 0, 0, 0, 0, 0);
              desc_add_lo(dbV, 2);
            }
            desc_add_lo(dbV, (uint32_t)(SUB_DESC_DELTA_KV - 2 * K_ATOMS_PER_TILE));
          }
          wp_end(wpc, WP_MMA_ISSUE);

          wp_begin(wpc, WP_MMA_COMMIT);
          if (lead && last_ktile) tcgen05_commit_multicast<2>(smem_ptr_u32(&full_bar_o_acc[i]), 0x3);
          wp_end(wpc, WP_MMA_COMMIT);
        };

        // prologue: BMM1 of K-tile 0 -- ONE slot shared by both joint M-tiles, freed after both.
        int kv_stage = kv_ph.get_stage();
        wp_begin(wpc, WP_MMA_WAIT_FULL_K);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
        kv_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_FULL_K);
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          wp_begin(wpc, WP_MMA_WAIT_FULL_Q);
          mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_q[i]), q_ph.get_phase());
          wp_end(wpc, WP_MMA_WAIT_FULL_Q);
          bmm1(i, kv_stage);
        }
        wp_begin(wpc, WP_MMA_COMMIT);
        if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar[kv_stage]), 0x3);
        wp_end(wpc, WP_MMA_COMMIT);

        // main loop: BMM2(kt) then BMM1-ahead(kt+1). V(kt) and K(kt+1) are each ONE shared slot:
        // V(kt) is freed after the last M-tile's BMM2, K(kt+1) is waited by i==0 and freed after both.
        for (int kt = 0; kt + 1 < num_k_tiles; ++kt) {
          const int kv_stage_v = kv_ph.get_stage();
          wp_begin(wpc, WP_MMA_WAIT_FULL_V);
          mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage_v]), kv_ph.get_phase());
          kv_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_FULL_V);

          int kv_stage_next = 0;
          #pragma unroll
          for (int i = 0; i < M_TILES_PER_CTA; ++i) {
            wp_begin(wpc, WP_MMA_WAIT_P);
            mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
            wp_end(wpc, WP_MMA_WAIT_P);
            bmm2(i, kv_stage_v, /*first_ktile=*/(kt == 0), /*last_ktile=*/false);

            if (i == 0) {
              kv_stage_next = kv_ph.get_stage();
              wp_begin(wpc, WP_MMA_WAIT_FULL_K);
              mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage_next]), kv_ph.get_phase());
              kv_ph.advance();
              wp_end(wpc, WP_MMA_WAIT_FULL_K);
            }
            if (i == M_TILES_PER_CTA - 1) {
              wp_begin(wpc, WP_MMA_COMMIT);
              if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar[kv_stage_v]), 0x3);
              wp_end(wpc, WP_MMA_COMMIT);
            }
            bmm1(i, kv_stage_next);
          }
          wp_begin(wpc, WP_MMA_COMMIT);
          if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar[kv_stage_next]), 0x3);
          wp_end(wpc, WP_MMA_COMMIT);

          spo_ph.advance();
        }

        wp_begin(wpc, WP_MMA_COMMIT);
        if (lead) {
          #pragma unroll
          for (int i = 0; i < M_TILES_PER_CTA; ++i)
            tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar_q[i]), 0x3);
        }
        wp_end(wpc, WP_MMA_COMMIT);

        // epilogue: BMM2 of the last K-tile -> final O (one shared V slot)
        kv_stage = kv_ph.get_stage();
        wp_begin(wpc, WP_MMA_WAIT_FULL_V);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
        kv_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_FULL_V);
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          wp_begin(wpc, WP_MMA_WAIT_P);
          mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
          wp_end(wpc, WP_MMA_WAIT_P);
          bmm2(i, kv_stage, /*first_ktile=*/(num_k_tiles == 1), /*last_ktile=*/true);
        }
        wp_begin(wpc, WP_MMA_COMMIT);
        if (lead) tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar[kv_stage]), 0x3);
        wp_end(wpc, WP_MMA_COMMIT);

        spo_ph.advance();
        q_ph.advance();
      }   // end if (peer == 0)

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN, /*CTA_GROUP=*/2, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        cluster_workitem_id = next.n_tile;
      } else {
        cluster_workitem_id += (int)(gridDim.x >> 1);
        if (cluster_workitem_id >= total_workitems) break;
      }
    }

  }
  else if (warp_id == W_EPI) {
    // EPI (store) warp: blocks/93 per-work-item body reused verbatim (per-CTA local
    // full/empty_bar_o_epi; 4D per-sample O map; both peers always store). Joint M-tile m stores the
    // m-th 256-row half of 512-block pidx, this peer's [peer*128, +128) rows of it -- hence the
    // per-m stride passed below is JOINT_M (256), NOT SPARSE_BLOCK (512).
    setmaxnreg_dec<72>();

    PhaseTracker<1> full_o_ph;
    // Prime empty_bar_o_epi once: corr's first sO pack must not block (no prior store in flight).
    if (elect_one_sync()) {
      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m)
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[m]));
    }
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int cluster_workitem_id = (int)(blockIdx.x >> 1);
    while (true) {
      const Vsa2Item it = vsa2_decode<Q_RASTER>(cluster_workitem_id, num_heads, clusters_per_seq, num_blocks, magic0, magic1, magic2, q2k_num);
      const int q_tb = it.pidx * SPARSE_BLOCK + peer * M_TILE;

      epi_store_warp_blackwell_1tile_1sm2sm_bf16_fmha<M_TILES_PER_CTA, M_TILE, HEAD_DIM>(
          wpc, &tmap_o, sO_bufs, full_bar_o_epi, empty_bar_o_epi,
          it.sample, it.head, q_tb, /*q_tile_per_mtile=*/JOINT_M, /*gqa_group_size=*/1,
          full_o_ph, /*do_store=*/true);

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN, /*CTA_GROUP=*/2, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        cluster_workitem_id = next.n_tile;
      } else {
        cluster_workitem_id += (int)(gridDim.x >> 1);
        if (cluster_workitem_id >= total_workitems) break;
      }
    }
  }
  else if (warp_id == W_SCHED) {
    setmaxnreg_dec<72>();
    if constexpr (USE_CLC) {
      sched_warp_clc_blackwell_ntiles_2sm_bf16<
          /*USE_GRIDDEP_WAIT=*/false, /*CLUSTER_SHAPE_M=*/1, /*CLUSTER_SHAPE_N=*/2,
          ClcRasterOrder::AlongN, /*CTA_GROUP=*/2, /*SUSPEND=*/true>(wpc, clc_full, clc_empty, clc_response, throttle_full, throttle_empty,
          /*cluster_rank=*/peer, lane);
    }
  }
  else if (warp_id >= W_CORR0 && warp_id < W_MMA) {
    // Correction warps: blocks/99 per-work-item body reused verbatim (USE_2CTA routing). Its
    // K_TILES must be nkv * KTILES_PER_BLOCK -- corr consumes one alpha/l per K-TILE, not per
    // selected 512-block (a mismatch there deadlocks softmax on empty_bar_alpha_and_l).
    setmaxnreg_dec<88>();
    const int corr_warp_id = warp_id - W_CORR0;

    const uint32_t tmem_base = *tmem_slot;   // published before the dispatch (see MMA warp)

    [[maybe_unused]] PhaseTracker<1> alpha_ph;
    PhaseTracker<1> o_acc_ph;
    PhaseTracker<1> o_epi_empty_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;

    // Prime the return barriers once (first BMM2 / first softmax stat write).
    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&empty_bar_spo[i]), 0));
      mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i]));
    }

    int cluster_workitem_id = (int)(blockIdx.x >> 1);
    while (true) {
      const Vsa2Item it = vsa2_decode<Q_RASTER>(cluster_workitem_id, num_heads, clusters_per_seq, num_blocks, magic0, magic1, magic2, q2k_num);

      correction_warp_blackwell_1tile_1sm2sm_bf16_fmha<
          M_TILE, M_TILES_PER_CTA, HEAD_DIM, K_TILE, SUB_COLS_BF16, FULL_NAMED_BAR, SOFTMAX_THROTTLE,
          // one selected 512-block = KTILES_PER_BLOCK K-tiles, and corr consumes one
          // alpha/l per K-tile -- must match the softmax/MMA trip count exactly or the pipeline hangs.
          /*USE_2CTA=*/true>(wpc, tmem_base, corr_warp_id, lane, /*K_TILES=*/it.nkv * KTILES_PER_BLOCK, alpha_and_l_smem, sO_bufs,
          full_bar_alpha, full_bar_l, full_bar_o_acc, full_bar_o_epi,
          empty_bar_spo, empty_bar_alpha_and_l, empty_bar_o_epi,
          alpha_ph, o_acc_ph, o_epi_empty_ph);

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN, /*CTA_GROUP=*/2, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        cluster_workitem_id = next.n_tile;
      } else {
        cluster_workitem_id += (int)(gridDim.x >> 1);
        if (cluster_workitem_id >= total_workitems) break;
      }
    }
  }
  else {
    // Softmax warps (inlined blocks/98 2SM variant). Warp band: m_tile = warp_id / 4, 32 rows at
    // warp_in_group * 32. Every K-tile is a real one: both M-tiles ride the cluster's ONE 512-block
    // list, so there is no membership skip, and no column mask either (selected blocks are full and the
    // attention is non-causal). Trip count is nkv * KTILES_PER_BLOCK.
    setmaxnreg_inc<176>();

    const uint32_t tmem_base = *tmem_slot;   // published before the dispatch (see MMA warp)
    constexpr int S_COLS = K_TILE;
    const int warp_id_u     = __shfl_sync(0xffffffffu, warp_id, 0);   // uniform-register promote
    const int m_tile        = warp_id_u / 4;
    const int warp_in_group = warp_id_u & 3;
    const int row_in_m_tile = warp_in_group * 32 + lane;
    const uint32_t s_tmem_addr = tmem_base + (uint32_t)(m_tile * S_COLS) + ((uint32_t)(warp_in_group * 32) << 16);
    PhaseTracker<1> spo_ph;
    PhaseTracker<1> scale_empty_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;

    int cluster_workitem_id = (int)(blockIdx.x >> 1);
    while (true) {
      const Vsa2Item it = vsa2_decode<Q_RASTER>(cluster_workitem_id, num_heads, clusters_per_seq, num_blocks, magic0, magic1, magic2, q2k_num);

      float m_run = -INFINITY, l_run = 0.f;
      wp_begin(wpc, WP_SM_WAIT_SCALE);
      mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
      scale_empty_ph.advance();
      wp_end(wpc, WP_SM_WAIT_SCALE);
      // FA4-shape softmax, as in fmha_context_bf16_uniform_2sm_inline.cu. Three coupled parts --
      // they only pay TOGETHER:
      //   (a) k==0 is PEELED via is_first_c, so the steady body has no k==0 branch;
      //   (b) exp2 is written IN PLACE into scores2 (aliases s_regs), giving a free read-back;
      //   (c) the row-sum is REMOVED from the exp2 loop and deferred past the P stores AND
      //       wait_scale, so the S->P path that gates BMM2 carries no row-sum FADD2s.
      // (b) requires scores2 to outlive the P store (no nested scope).
      float* const alpha_slot = &alpha_and_l_smem[m_tile * M_TILE + row_in_m_tile];
      const int num_k_tiles = it.nkv * KTILES_PER_BLOCK;
      auto softmax_step = [&](auto is_first_c) {
        constexpr bool IS_FIRST = decltype(is_first_c)::value;
        wp_begin(wpc, WP_SM_WAIT_S);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_spo[m_tile]), spo_ph.get_phase());
        wp_end(wpc, WP_SM_WAIT_S);

        wp_begin(wpc, WP_SM_SOFTMAX);
        float alpha;
        uint32_t s_regs[K_TILE];
        #pragma unroll
        for (int c0 = 0; c0 < K_TILE; c0 += S_LD_COLS) {
          const uint32_t taddr = s_tmem_addr + (uint32_t)c0;
          if      constexpr (S_LD_COLS == 32)  tcgen05_ld_32x32b_x32 (taddr, *reinterpret_cast<uint32_t(*)[32]>(&s_regs[c0]));
          else if constexpr (S_LD_COLS == 64)  tcgen05_ld_32x32b_x64 (taddr, *reinterpret_cast<uint32_t(*)[64]>(&s_regs[c0]));
          else if constexpr (S_LD_COLS == 128) tcgen05_ld_32x32b_x128(taddr, *reinterpret_cast<uint32_t(*)[128]>(&s_regs[c0]));
        }
        // no wait_ld: rmax/exp2 below scoreboard-wait the S LDTM (RAW).

        float* scores = reinterpret_cast<float*>(s_regs);
        float2* scores2 = reinterpret_cast<float2*>(s_regs);
        // order this S read ahead of the later tcgen05.st that overwrites the slot with P
        tcgen05_fence_before_thread_sync();

        // No column mask: every selected block is full (128 tokens), non-causal.

        // rmax via 4 independent FMNMX3 accumulators (4-way ILP), old max FOLDED into
        // accumulator 0 (no extra dependent fmaxf below). K_TILE % 8 == 0.
        float rmax0 = m_run, rmax1 = -INFINITY, rmax2 = -INFINITY, rmax3 = -INFINITY;
        #pragma unroll
        for (int j = 0; j < K_TILE; j += 8) {
          rmax0 = fmaxf(fmaxf(rmax0, scores[j + 0]), scores[j + 1]);
          rmax1 = fmaxf(fmaxf(rmax1, scores[j + 2]), scores[j + 3]);
          rmax2 = fmaxf(fmaxf(rmax2, scores[j + 4]), scores[j + 5]);
          rmax3 = fmaxf(fmaxf(rmax3, scores[j + 6]), scores[j + 7]);
        }
        float new_m = fmaxf(fmaxf(rmax0, rmax1), fmaxf(rmax2, rmax3));
        alpha = 0.0f;
        if constexpr (!IS_FIRST) {
          // FA4 sticky max: below the threshold keep the old max EXACTLY and publish
          // alpha == 1.0 -> corr's __all_sync(alpha == 1.0f) skips the full O rescale.
          const float acc_scale_ = (m_run - new_m) * scale_log2;
          alpha = ex2_approx_f32(acc_scale_);
          if (acc_scale_ >= -(float)RESCALE_THRESHOLD) { new_m = m_run; alpha = 1.0f; }
        }
        *alpha_slot = alpha;
        if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
        else mbarrier_arrive(smem_ptr_u32(&full_bar_alpha[m_tile]));

        // scale_subtract_rowmax + apply_exp2_convert (FA4 form): exp2 IN PLACE into scores2[c]
        // (aliases s_regs -> free read-back), bf16 pack right after. NO row-sum here.
        const float2 scale2 = f32x2_splat(scale_log2);
        const float2 neg_m_scaled2 = f32x2_splat(-new_m * scale_log2);
        uint32_t p_regs[K_TILE / 2];
        #pragma unroll
        for (int c = 0; c < K_TILE / 2; ++c) {
          const float2 a2 = ffma2(scores2[c], scale2, neg_m_scaled2);
          if constexpr (EX2_EMU) {
            const int jj = c / EX2_FRG_PAIRS;
            const int kk = 2 * (c % EX2_FRG_PAIRS);
            const bool use_hw = (kk % EX2_FREQ < EX2_FREQ - EX2_RES) || (jj >= EX2_FRG_CNT - 1) || (jj < EX2_START_FRG);
            scores2[c] = use_hw ? make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y))
                                : ex2_emu_f32x2(a2.x, a2.y);
          } else {
            scores2[c] = make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y));
          }
          p_regs[c] = cvt_f32x2_to_bf16x2(scores2[c].x, scores2[c].y);
        }
        const uint32_t p_tmem_addr = s_tmem_addr;
        wp_end(wpc, WP_SM_SOFTMAX);

        wp_begin(wpc, WP_SM_STORE_P);
        if constexpr (SPLIT_P) {
          tcgen05_st_32x32b_x32(p_tmem_addr,      *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
          tcgen05_st_32x32b_x16(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[32]));
          tcgen05_wait_st();   // RACE FIX: fence orders but does NOT complete the async STTM
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&empty_bar_spo[m_tile]), 0));
          tcgen05_st_32x32b_x16(p_tmem_addr + SPLIT_P_COL, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[SPLIT_P_COL]));
          tcgen05_wait_st();   // RACE FIX
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&full_bar_p_last[m_tile]), 0));
        } else {
          tcgen05_st_32x32b_x32(p_tmem_addr,      *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
          tcgen05_st_32x32b_x32(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[32]));
          tcgen05_wait_st();   // RACE FIX
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive_cluster_default(mapa_shared_cluster_u32(smem_ptr_u32(&empty_bar_spo[m_tile]), 0));
        }
        wp_end(wpc, WP_SM_STORE_P);

        spo_ph.advance();
        wp_begin(wpc, WP_SM_WAIT_SCALE);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile]), scale_empty_ph.get_phase());
        scale_empty_ph.advance();
        wp_end(wpc, WP_SM_WAIT_SCALE);
        // deferred update_row_sum: P is already published, so this FADD2 tree overlaps corr's
        // O-rescale and the MMA warp's BMM2 instead of gating them. l_run*alpha is folded into
        // the lt2a seed (no trailing FFMA).
        {
          float2 lt2a = make_float2(IS_FIRST ? 0.0f : l_run * alpha, 0.0f);
          float2 lt2b = make_float2(0.f, 0.f), lt2c = make_float2(0.f, 0.f), lt2d = make_float2(0.f, 0.f);
          #pragma unroll
          for (int c = 0; c < K_TILE / 2; c += 4) {
            lt2a = fadd2(lt2a, scores2[c + 0]);
            lt2b = fadd2(lt2b, scores2[c + 1]);
            lt2c = fadd2(lt2c, scores2[c + 2]);
            lt2d = fadd2(lt2d, scores2[c + 3]);
          }
          const float2 lt2 = fadd2(fadd2(lt2a, lt2b), fadd2(lt2c, lt2d));
          l_run = lt2.x + lt2.y;
        }
        m_run = new_m;
      };
      if (num_k_tiles > 0) softmax_step(std::true_type{});
      for (int k = 1; k < num_k_tiles; ++k) softmax_step(std::false_type{});

      wp_begin(wpc, WP_SM_READ_L);
      *alpha_slot = l_run;
      if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
      else mbarrier_arrive(smem_ptr_u32(&full_bar_l[m_tile]));
      wp_end(wpc, WP_SM_READ_L);

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 2, ClcRasterOrder::AlongN, /*CTA_GROUP=*/2, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        cluster_workitem_id = next.n_tile;
      } else {
        cluster_workitem_id += (int)(gridDim.x >> 1);
        if (cluster_workitem_id >= total_workitems) break;
      }
    }
  }
  wp_flush(wpc);
  // TMEM teardown, dense-2SM style: __syncthreads() proves every warp of THIS CTA is done with
  // TMEM, the cluster join proves the peer CTA is too, then one warp deallocs. Per-warp bar_sync<9>
  // + closing-prime + tail-drain accounting is fragile: a closing prime racing the epilogue waits deadlocks.
  __syncthreads();
  barrier_cluster_arrive();
  barrier_cluster_wait();
  if (warp_id == W_MMA) tcgen05_dealloc<2>(*tmem_slot, TMEM_TOTAL);
}

// ============================== driver ====================================

// CPU reference: VSA fine block-sparse attention in fp32, over the per-512-block top-k lists
// (q and k blocks are both SPARSE_BLOCK wide).
// Layout: Q/K/V natural
// [token, head, hd]; token = b*S + local. gqb = (b*H+h)*nb + qblk.
static void cpu_vsa_ref(const __nv_bfloat16* hQ, const __nv_bfloat16* hK,
                        const __nv_bfloat16* hV, float* hO,
                        int B, int H, int S, int hd, int nb, int max_kv,
                        const int* q2k_idx, const int* q2k_num) {
  const float scale = 1.0f / sqrtf((float)hd);
  const long Nq = (long)B * S;
  for (long i = 0; i < Nq * H * hd; ++i) hO[i] = 0.f;

  #pragma omp parallel for schedule(dynamic)
  for (int bhq = 0; bhq < B * H * nb; ++bhq) {
    const int qblk = bhq % nb;
    const int bh   = bhq / nb;
    const int h    = bh % H;
    const int b    = bh / H;
    const int gqb  = bhq;                        // (b*H + h)*nb + qblk
    const int nkv  = q2k_num[gqb];

    for (int qi = 0; qi < SPARSE_BLOCK; ++qi) {
      const long qp = (long)b * S + (long)qblk * SPARSE_BLOCK + qi;
      std::vector<float> z((size_t)nkv * SPARSE_BLOCK);
      float row_max = -INFINITY;
      int idx = 0;
      for (int kk = 0; kk < nkv; ++kk) {
        const int blk = q2k_idx[gqb * max_kv + kk];
        for (int kj = 0; kj < SPARSE_BLOCK; ++kj) {
          const long kp = (long)b * S + (long)blk * SPARSE_BLOCK + kj;
          float dot = 0.f;
          for (int e = 0; e < hd; ++e)
            dot += __bfloat162float(hQ[(qp * H + h) * hd + e])
                 * __bfloat162float(hK[(kp * H + h) * hd + e]);
          z[idx++] = dot * scale;
          row_max = fmaxf(row_max, z[idx - 1]);
        }
      }
      float sum = 0.f;
      for (int j = 0; j < nkv * SPARSE_BLOCK; ++j) { z[j] = expf(z[j] - row_max); sum += z[j]; }
      if (sum == 0.f) continue;
      const float inv_sum = 1.f / sum;
      for (int e = 0; e < hd; ++e) {
        float acc = 0.f;
        idx = 0;
        for (int kk = 0; kk < nkv; ++kk) {
          const int blk = q2k_idx[gqb * max_kv + kk];
          for (int kj = 0; kj < SPARSE_BLOCK; ++kj) {
            const long kp = (long)b * S + (long)blk * SPARSE_BLOCK + kj;
            acc += z[idx++] * __bfloat162float(hV[(kp * H + h) * hd + e]);
          }
        }
        hO[(qp * H + h) * hd + e] = acc * inv_sum;
      }
    }
  }
}

// Deterministic fill in [-1, 1). xorshift-mixed hash so the sequence has no short period that
// could alias with H*hd and make every token identical. Same as the 1CTA VSA harness.
static void fillr(__nv_bfloat16* h, long n, unsigned seed) {
  block_sparse_bf16_benchmark::fill(h, n, seed);
}

struct Sh { int B, H, nb, topk, hd; const char* lab; };

static double run(const Sh& sh, bool verify) {
  const int  B = sh.B, H = sh.H, nb = sh.nb, topk = sh.topk, hd = sh.hd;
  const int  S = nb * SPARSE_BLOCK;                // seqlen; nb = number of 512-token sparse blocks
  const int  max_kv = topk;                        // tight: exactly topk selected blocks
  const long tq = (long)B * S;                     // total tokens
  const int  clusters_per_seq = nb;                // a CLUSTER does exactly ONE 512-block (its 2 joint M-tiles)
  const int  total_qblk = B * H * nb;              // one top-k list per 512-block
  const int  nclusters = B * H * clusters_per_seq;
  // No union-inflation knob and no nb parity constraint: a cluster is exactly ONE 512-block (one list).
  if (topk < 1 || topk > nb) { printf("  [%s] SKIP: need 1 <= topk <= nb\n", sh.lab); return 0.0; }

  // ---- device buffers (bf16; V stored transposed as V_T for the BMM2 TMA) ----
  __nv_bfloat16 *dQ, *dK, *dVT, *dO;
  CUDA_CHECK(cudaMalloc(&dQ,  tq * H * hd * 2));
  CUDA_CHECK(cudaMalloc(&dK,  tq * H * hd * 2));
  CUDA_CHECK(cudaMalloc(&dVT, (long)H * hd * tq * 2));
  CUDA_CHECK(cudaMalloc(&dO,  tq * H * hd * 2));

  std::vector<__nv_bfloat16> hQ(tq * H * hd), hK(tq * H * hd), hV(tq * H * hd);
  const char* load_npy = getenv("LOAD_NPY");   // Q/K/V + idx from block_sparse_bf16_gen_inputs.py .npy
  if (load_npy) {
    const std::string d(load_npy);
    auto ld = [&](const char* nm, std::vector<__nv_bfloat16>& h) {
      char p[64]; snprintf(p, sizeof p, "/%s_S%d.npy", nm, S);
      auto bits = npy_load_vec<uint16_t>(d + p);
      if (bits.size() != h.size()) { fprintf(stderr, "LOAD_NPY: %s size %zu != %zu\n", nm, bits.size(), h.size()); exit(1); }
      memcpy(h.data(), bits.data(), h.size() * 2);   // raw bf16 bits
    };
    ld("q", hQ); ld("k", hK); ld("v", hV);
  } else {
    fillr(hQ.data(), hQ.size(), 11);
    fillr(hK.data(), hK.size(), 22);
    fillr(hV.data(), hV.size(), 33);
  }

  // V -> V_T transpose ([tok,head,hd] -> [head,hd,tok])
  std::vector<__nv_bfloat16> hVT((long)H * hd * tq, __float2bfloat16(0.f));
  for (long idx = 0; idx < tq; ++idx)
    for (int h = 0; h < H; ++h)
      for (int d = 0; d < hd; ++d)
        hVT[(h * hd + d) * tq + idx] = hV[(idx * H + h) * hd + d];

  CUDA_CHECK(cudaMemcpy(dQ,  hQ.data(),  hQ.size()  * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK,  hK.data(),  hK.size()  * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dVT, hVT.data(), hVT.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dO, 0, hQ.size() * 2));

  // ---- q2k index: topk DISTINCT 512-block ids per (b,h,512-block), fixed density ----
  // One list per 512 sparse block; the whole cluster reads the SAME list, so there is no
  // union and no membership mask.
  std::vector<int> hq2k_idx((size_t)total_qblk * max_kv, 0);
  std::vector<int> hq2k_num(total_qblk, topk);
  if (load_npy) {   // LOAD_NPY: head-independent [nb, topk] list, broadcast across heads
    char p[64]; snprintf(p, sizeof p, "/idx_S%d_blk%d.npy", S, SPARSE_BLOCK);
    auto idx = npy_load_vec<int32_t>(std::string(load_npy) + p);
    if (idx.size() != (size_t)nb * topk) { fprintf(stderr, "LOAD_NPY: idx size %zu != %d\n", idx.size(), nb * topk); exit(1); }
    for (int gqb = 0; gqb < total_qblk; ++gqb) {
      const int qblk = gqb % nb;
      for (int i = 0; i < topk; ++i) hq2k_idx[(size_t)gqb * max_kv + i] = idx[(size_t)qblk * topk + i];
    }
  } else {
    block_sparse_bf16_benchmark::select_blocks(hq2k_idx, nb, topk);
  }

  int *dq2k_idx, *dq2k_num;
  CUDA_CHECK(cudaMalloc(&dq2k_idx, hq2k_idx.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dq2k_num, hq2k_num.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dq2k_idx, hq2k_idx.data(), hq2k_idx.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dq2k_num, hq2k_num.data(), hq2k_num.size() * sizeof(int), cudaMemcpyHostToDevice));

  // ---- TMA tensor maps ----
  // Q/O: 4D per-sample map over [hd, H, token-IN-SAMPLE, sample]; box [64, 1, 128, 1]
  // (MHA: head box dim 1). The per-sample token dim mirrors the dense 2SM kernel.
  CUtensorMap tq_, tk_, tvt_, to_;
  {
    uint64_t gd[4] = { (uint64_t)hd, (uint64_t)H, (uint64_t)S, (uint64_t)B };
    uint64_t gs[3] = { (uint64_t)hd * 2u, (uint64_t)H * hd * 2u, (uint64_t)S * H * hd * 2u };
    uint32_t bd[4] = { (uint32_t)SUB_COLS_BF16, 1u, (uint32_t)M_TILE, 1u };
    uint32_t es[4] = { 1u, 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tq_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, dQ, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
    r = cuTensorMapEncodeTiled(&to_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, dO, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  // K: dims [atom-col 64, token tq, atom (H*hd)/64]; 2SM half-box token dim (K_TILE/2 = 64):
  // each peer loads its 64-token half of the SAME selected block, both hd-atoms in one TMA.
  {
    uint64_t gd[3] = { (uint64_t)SUB_COLS_BF16, (uint64_t)tq, (uint64_t)((long)H * hd / SUB_COLS_BF16) };
    uint64_t gs[2] = { (uint64_t)((long)H * hd) * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t bd[3] = { (uint32_t)SUB_COLS_BF16, (uint32_t)(K_TILE / 2), (uint32_t)K_SUBTILES };
    uint32_t es[3] = { 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tk_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, dK, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  // V_T: 2D [H*hd rows, tq cols]; 2SM box = hd/2 rows x 64 tokens (D-split half per peer).
  CUDA_CHECK(make_tma_2d_tiled(&tvt_, dVT, (long)H * hd, (int)tq, hd / 2, SUB_COLS_BF16, 2,
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B));

  // ---- shared memory budget (same shape as the dense 2SM kernel) ----
  constexpr bool FULL_NAMED_BAR = false, EX2_EMU = true, SPLIT_P = true,
                 SOFTMAX_THROTTLE = false, USE_CLC = false, Q_RASTER = true;
  const size_t smem =
        (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * KV_SLOT_BYTES   // Q (x2) + shared K/V ring
      + (size_t)2 * M_TILE * HEAD_DIM * sizeof(__nv_bfloat16)     // 2 sO bufs for TMA-O
      + (2 * NUM_KV_STAGES + 26) * 8                              // mbarriers (incl o_epi + throttle x4)
      + (USE_CLC ? (size_t)CLC_STAGES * (2 * 8 + 16) + 16 : 0)    // CLC: full+empty + response (16B aligned)
      + 8                                                         // tmem_slot
      + 16                                                        // tmem_dealloc_bar [1] + pad
      + (size_t)2 * M_TILE * sizeof(float)                        // alpha_and_l_smem [2][M_TILE]
      + 256;                                                      // slack / alignment

  auto kfn = &fmha_context_bf16_vsa_2sm_kernel<32, FULL_NAMED_BAR, EX2_EMU, SPLIT_P, SOFTMAX_THROTTLE, USE_CLC, Q_RASTER>;
  CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  const float scale_log2 = (1.0f / sqrtf((float)hd)) * (float)M_LOG2E;
  int numSM = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0));
  int nblk;
  if (USE_CLC) {
    nblk = nclusters * 2;                       // full problem: one cluster per m-pair
  } else {
    nblk = std::min(nclusters * 2, numSM);
    nblk -= (nblk & 1);                      // grid.x must be a multiple of cluster.x = 2
  }
  dim3 grid(nblk, 1, 1), block(N_WARPS * 32, 1, 1);

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid; cfg.blockDim = block; cfg.dynamicSmemBytes = smem; cfg.stream = 0;
  cudaLaunchAttribute cfgAttr[1];
  cfgAttr[0].id = cudaLaunchAttributeClusterDimension;
  cfgAttr[0].val.clusterDim.x = 2; cfgAttr[0].val.clusterDim.y = 1; cfgAttr[0].val.clusterDim.z = 1;
  cfg.attrs = cfgAttr; cfg.numAttrs = 1;
  const unsigned long long magic0 = make_magic((unsigned)(H * clusters_per_seq));
  const unsigned long long magic1 = make_magic((unsigned)H);
  const unsigned long long magic2 = make_magic((unsigned)clusters_per_seq);
  auto launch = [&](cudaStream_t stream = nullptr) {
    cfg.stream = stream;
    return cudaLaunchKernelEx(&cfg, kfn, tq_, tk_, tvt_, to_, S, H, scale_log2,
                              B, clusters_per_seq, nb, max_kv,
                              magic0, magic1, magic2,
                              (const int*)dq2k_idx, (const int*)dq2k_num);
  };

  const double ms = block_sparse_bf16_benchmark::measure(launch);
#ifdef WARP_PROF
  {
    WpBuffer wp = wp_alloc(grid);
    const unsigned pblk = wp.view_block;
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    wp_readback(wp);
    const char *roles[16] = {"sm0",  "sm0",  "sm0",  "sm0",  "sm1", "sm1", "sm1",  "sm1",
                             "corr", "corr", "corr", "corr", "mma", "epi", "load", "sched"};
    printf("  [%s] WARP_PROF block %u:\n", sh.lab, pblk);
    wp_print_busy(wp, roles, 16, pblk);
    wp_dump_raw(wp, "warp_raw_vsa_2sm.bin.gz", pblk, NUM_KV_STAGES);
    wp_free(wp);
  }
#endif

  // blk512 has NO union inflation (the whole cluster shares one 512-block list),
  // so visited == selected and one TFLOPS number says it all.
  const double sel_pairs = (double)B * H * nb * SPARSE_BLOCK * (double)topk * SPARSE_BLOCK;
  const double tflops_sel = block_sparse_bf16_benchmark::tflops(sel_pairs, hd, ms);
  if (block_sparse_bf16_benchmark::benchmark_enabled()) printf("  [%-9s H%-2d nb%-3d k%-3d S%d] N_q=%ld topk=%d  %.4f ms  %.1f TFLOPS (selected)\n",
         sh.lab, H, nb, topk, S, tq, topk, ms, tflops_sel);

  block_sparse_bf16_benchmark::dump_output(dO, hQ.size());

  if (verify) {
    std::vector<__nv_bfloat16> ho(hQ.size());
    CUDA_CHECK(cudaMemcpy(ho.data(), dO, ho.size() * 2, cudaMemcpyDeviceToHost));
    std::vector<float> ref(hQ.size(), 0.f), out(hQ.size());
    cpu_vsa_ref(hQ.data(), hK.data(), hV.data(), ref.data(), B, H, S, hd, nb, max_kv,
                hq2k_idx.data(), hq2k_num.data());
    for (size_t i = 0; i < out.size(); ++i) out[i] = __bfloat162float(ho[i]);
    const bool ok = check_close_f32(ref.data(), out.data(), (int)out.size(), 0.05f, 0.10f);
    printf("  verify [%s]: %s\n", sh.lab, ok ? "OK" : "FAIL");
    if (!ok) std::exit(EXIT_FAILURE);
  }

  // ---- stress mode: STRESS_N launches, cross-run output consistency ----
  if (const char* s = getenv("STRESS_N")) {
    const int N = atoi(s);
    std::vector<__nv_bfloat16> first(hQ.size()), cur(hQ.size());
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(first.data(), dO, first.size() * 2, cudaMemcpyDeviceToHost));
    int mism = 0, errs = 0;
    for (int it = 1; it < N; ++it) {
      CUDA_CHECK(cudaMemset(dO, 0, hQ.size() * 2));
      launch();
      cudaError_t e = cudaDeviceSynchronize();
      if (e != cudaSuccess) { printf("  STRESS run %d: CUDA ERR %s\n", it, cudaGetErrorString(e)); errs++; continue; }
      CUDA_CHECK(cudaMemcpy(cur.data(), dO, cur.size() * 2, cudaMemcpyDeviceToHost));
      if (memcmp(first.data(), cur.data(), cur.size() * 2) != 0) { printf("  STRESS run %d: OUTPUT MISMATCH vs run0\n", it); mism++; }
    }
    printf("  STRESS [%s] N=%d: mismatches=%d cuda_errs=%d\n", sh.lab, N, mism, errs);
  }

  cudaFree(dQ); cudaFree(dK); cudaFree(dVT); cudaFree(dO);
  cudaFree(dq2k_idx); cudaFree(dq2k_num);
  return tflops_sel;
}

int main() {
  CUDA_CHECK(cudaFree(0));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 64 * 1024 * 1024));
  printf("VSA fine block-sparse FMHA bf16 2SM (cta_group::2, blk512: cluster = 2 joint M-tiles = 512 rows\n"
         "  == ONE 512 sparse block -> ONE shared top-k list, ONE K/V stream) sm_100a\n"
         "=====================================\n");

  // shapes: {B, H, nb, topk, hd, label}. nb = number of 512-token sparse blocks (no parity constraint).
  Sh shapes[] = {
    {1,  4,  8,  4, 128, "small"},
    {1, 16, 32,  8, 128, "fastvideo"},
    {1,  8, 64, 16, 128, "25pct"},
  };
  const bool verify = getenv("NOVERIFY") ? false : true;

  // single-shape override: SHAPE=0..2 plus BATCH/HEADS/NB/TOPK env knobs.
  if (const char* s = getenv("SHAPE")) {
    Sh sh = shapes[atoi(s) % 3];
    if (getenv("BATCH")) sh.B    = atoi(getenv("BATCH"));
    if (getenv("HEADS")) sh.H    = atoi(getenv("HEADS"));
    if (getenv("NB"))    sh.nb   = atoi(getenv("NB"));
    if (getenv("TOPK"))  sh.topk = atoi(getenv("TOPK"));
    sh.lab = "custom";
    run(sh, verify);
    return 0;
  }
  const int B = getenv("BATCH") ? atoi(getenv("BATCH")) : 1;
  for (Sh sh : shapes) { sh.B = B; run(sh, verify); }
  return 0;
}

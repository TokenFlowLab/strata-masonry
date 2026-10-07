// fmha_context_bf16_uniform.cu -- K2 FMHA context BF16, sm_100a.
//    
// ASSUMES (baked in -- the kernel is NOT correct otherwise):
//   1. Full mask -- every q-token attends to all keys (only keys >= seqlen are masked). IS_CAUSAL
//      adds the triangular mask + K-loop cap; still uniform (non-ragged) seqlen.
//   2. ALL seqlens EQUAL (no varlen) -- one `seqlen` for every sample, so K_TILES is uniform.
//
// GEN-shape variant. The warp-specialized 16-warp kernel body + barrier contract are the SAME as
// fmha_context_bf16_gqa_nonpersistent.cu (see that file's header for Terminology, Data Layout,
// Execution flow, Barrier Contract, Memory Layout). Differences from those varlen kernels:
//   - PERSISTENT scheduling, selected by USE_CLC (phase trackers persist across tiles, primed once):
//       USE_CLC=true (default): CLC (clusterlaunchcontrol.try_cancel) HW work-stealing scheduler.
//         w15 is a standalone sched warp issuing try_cancel into a CLC_STAGES-deep tile-id ring;
//         the other 15 warps + the sched consume it (all 16 release clc_empty/tile -> arrive_count
//         = N_WARPS). Ring depth covers the load-ahead/epi-behind skew. grid = full problem.
//       USE_CLC=false: static grid-stride loop (workitem_id += gridDim.x); w15 idle; grid = #SMs.
//   - Equal seqlen: uniform K_TILES + a plain (sample, q_tile_id, kv_head) decode -- no prefix sum,
//     no binary search.
//   - TMA-store epilogue: correction packs O*=1/l into a subtile-split sO, then the epi warp (w13)
//     TMA-stores it (full_bar_o_epi). Valid because equal-seqlen tiles are non-ragged;
//     the varlen kernels use a predicated STG re-tile instead.
//
// Barrier contract (additions to gqa_nonpersistent's -- unique to the TMA-sO epilogue):
//   - empty_bar_o_epi[m] (count 1): epi -> corr, "sO[m]'s TMA store drained, slot reusable".
//     epi arrives per m as its commit group drains; corr waits before packing the next tile's
//     sO[m]. Without it, tiny causal K-loops (K_TILES=1) let corr's repack race the
//     still-reading store.
//
// Budgets:
//   - Registers: per-warp budgets sum to exactly the SM file (65536 = 128*512): softmax inc<192>
//     (x8) + correction dec<80> (x4) + the four single warps dec<48> (x4). WARP_PROF's
//     wp_begin/wp_end markers fit as-is (single warps ~R26; corr ~R77; softmax ~R186).
//   - TMEM: each M-tile needs S + O = K_TILE + HEAD_DIM cols of the 512, so
//     M_TILES_PER_CTA <= 512 / (K_TILE + HEAD_DIM) = 2 for 128/128 (adjacent q-tiles of the
//     SAME sequence).
//
// EX2_EMU hybrid exp2 (FA4 apply_exp2_convert):
//   - Per 32-elt fragment, a fraction of pairs use the f32x2 ALU emulation (ex2_emu_f32x2)
//     instead of MUFU.EX2 to relieve the EX2 pipe.
//   - Gate: HW unless (k%EX2_FREQ >= EX2_FREQ-EX2_RES) AND (fragment < last), where for
//     softmax pair c: fragment j = c/EX2_FRG_PAIRS, in-fragment elt k = 2*(c%EX2_FRG_PAIRS).

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio> 
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <cstring>
#include <cmath>
#include <vector>
#include "../../../tests/test_utils.cuh"
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
#include "../../../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../../../composites/118_mbarrier_phase_tracking.cuh"
#include "../../../composites/106_clc_fetch_next_tile.cuh"
#include "../../../primitives/46_setmaxnreg.cuh"
#include "../../../primitives/76_packed_f32x2.cuh"
#include "../../../primitives/77_ex2_approx.cuh"
#include "../../../primitives/78_rcp_approx.cuh"
#include "../../../primitives/_warp_prof_noop.cuh"
#include "fmha_utils.cuh"

constexpr int M_TILE = 128;
constexpr int M_TILES_PER_CTA = 2;
constexpr int K_TILE = 128;
constexpr int HEAD_DIM = 128;
// B128 swizzle atom = 128 bytes = 64 bf16: all SMEM tiles are laid out in
// 64-wide sub-tiles along the contiguous dim.
constexpr int SUB_COLS_BF16 = 64;
constexpr int SUB_COLS_BYTES = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);   // 128 B (one swizzle atom)
constexpr int Q_SUBTILES = HEAD_DIM / SUB_COLS_BF16;  // 2
constexpr int K_SUBTILES = HEAD_DIM / SUB_COLS_BF16;  // 2 (K tile is K_TILE tokens x head_dim)
constexpr int V_SUBTILES = K_TILE / SUB_COLS_BF16;    // 2 (V_T tile is head_dim x K_TILE tokens)
constexpr int P_SUBTILES = K_TILE / SUB_COLS_BF16;    // 2
constexpr int Q_SUB_COLS_BYTES = M_TILE * SUB_COLS_BYTES;         // 16 KB
constexpr int K_SUB_COLS_BYTES = K_TILE * SUB_COLS_BYTES;         // 16 KB
constexpr int Q_TILE_BYTES = Q_SUBTILES * Q_SUB_COLS_BYTES;     // 32 KB
constexpr int K_TILE_BYTES = K_SUBTILES * K_SUB_COLS_BYTES;     // 32 KB
constexpr int V_TILE_BYTES = V_SUBTILES * Q_SUB_COLS_BYTES;     // 32 KB
constexpr int P_TILE_BYTES = P_SUBTILES * Q_SUB_COLS_BYTES;     // 32 KB
constexpr int K_ATOMS_PER_TILE = SUB_COLS_BF16 / 16;    // 4
constexpr int SPLIT_P_N    = K_TILE / 4 * 3;            // 96
constexpr int SPLIT_P_ATOM = SPLIT_P_N / 16;            // 6 (BMM2 atom at the split)
constexpr int SPLIT_P_COL  = SPLIT_P_N / 2;     // 48 (u32 P cols written before the empty_bar_spo signal)
constexpr int EX2_FRG_PAIRS = 16;          // 32 elts / fragment = 16 pairs
constexpr int EX2_FRG_CNT   = K_TILE / 32; // = 4 for K_TILE=128
// EX2_FREQ (HW-vs-emu cadence) is a flat 16 for all instantiations; defined at the use site.
// TODO: tuning knob -- try per-instantiation values (FA4 used freq 10 for GQA non-causal).
constexpr int EX2_RES       = 4;           // FA4 ex2_emu_res
constexpr int EX2_START_FRG = 1;           // FA4 ex2_emu_start_frg (fragment 0 pure-HW; 2SM pairing)
constexpr int NUM_KV_STAGES = 3;
constexpr int S_COLS = K_TILE;                   // FP32 S, single buffer
constexpr int O_COLS = HEAD_DIM;
constexpr int TMEM_TOTAL = 512;                     // S0,S1(128*2)+O0,O1(128*2)=512
constexpr int W_CORR0 = 8, W_MMA = 12, W_EPI = 13, W_LOAD = 14, W_SCHED = 15;
constexpr int N_WARPS = 16;
constexpr int CLC_STAGES = 4;

extern __shared__ __align__(1024) uint8_t fmha_smem[];

// 64-bit SMEM descriptor; k-loop walks add to the LOW word only (hi/swizzle word is
// k-invariant) -> one 32-bit add/step, not a 64-bit carry pair. (2SM-proven form.)
union SmemDescPair { uint64_t u64; uint2 w; };

// Predicated tcgen05 issue (FA4 shape): the elect predicate guards the instruction,
// not an if(lead) block -> straight-line issue on the mma warp, no per-block BSSY/BSYNC
// (r1 analysis: 4-6 pairs/item vs FA4's zero, on the throughput-limiting warp).

// Walk a descriptor pair IN PLACE by a 32-bit add on the LOW word. asm VOLATILE pins the
// serial chain: non-volatile forms get CSE'd into parallel base+imm descs, each paying a
// UMOV to rematerialize the hi word (75 of them between the MMAs on this kernel).
__device__ __forceinline__ void desc_add_lo(SmemDescPair& d, uint32_t inc) {
  asm volatile("{\n\t"
      ".reg .b32 lo, hi;\n\t"
      "mov.b64 {lo, hi}, %0;\n\t"
      "add.u32 lo, lo, %1;\n\t"
      "mov.b64 %0, {lo, hi};\n\t"
      "}" : "+l"(d.u64) : "r"(inc));
}

template <bool Q_RASTER, bool IS_CAUSAL, bool LPT>
__device__ __forceinline__ void decode_workitem(
    int workitem_id, int seqlen, int num_kv_heads,
    int packed_mtiles_per_seq, int packed_mtiles_per_sample, int q_tile_per_cta,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    int lpt_swz_log2, int lpt_hb_quot, int lpt_hb_rem,
    unsigned long long lpt_major_magic, unsigned long long lpt_rem_magic,
    int& sample, int& h_kv, int& q_tile_base, int& k_tiles) {
  int packed_mtiles_index;
  if constexpr (IS_CAUSAL && LPT) {
    // FA4 SingleTileLPTScheduler mapping (causal).
    //   SECTION = a group of adjacent kv-heads whose combined K/V fits L2. Its size is
    //     2^lpt_swz_log2 (largest power-of-2 head count with that K/V <= L2 budget) -- a
    //     power of 2 so the in-section divmod is a shift+mask (below), not a magic-div,
    //     in this hot decode.
    //   ORDER = BLOCK-MAJOR within a section: same m-block across the section's heads,
    //     then the next block (LPT-reversed = heaviest first). The section's K/V stays
    //     L2-resident -> DRAM-fetched ~once per section.
    // ncu: ~45% fewer causal DRAM reads vs the plain m-innermost raster.
    const int major = (int)fdiv((unsigned)workitem_id, lpt_major_magic);   // section
    const int l2mod = workitem_id - major * (packed_mtiles_per_seq << lpt_swz_log2);
    int block, res;
    if (major < lpt_hb_quot) {
      block = l2mod >> lpt_swz_log2;
      res   = l2mod & ((1 << lpt_swz_log2) - 1);
    } else {                                       // residual (partial) last section
      block = (int)fdiv((unsigned)l2mod, lpt_rem_magic);
      res   = l2mod - block * lpt_hb_rem;
    }
    const int hb = (major << lpt_swz_log2) + res;  // flat (batch*nkh + head)
    sample = (int)fdiv((unsigned)hb, magic2);
    h_kv   = hb - sample * num_kv_heads;
    packed_mtiles_index = packed_mtiles_per_seq - 1 - block;   // LPT reversal
    q_tile_base = packed_mtiles_index * q_tile_per_cta;
  } else {
    sample = (int)fdiv((unsigned)workitem_id, magic0);
    const int rr = workitem_id - sample * packed_mtiles_per_sample;
    if constexpr (Q_RASTER) {
      h_kv                = (int)fdiv((unsigned)rr, magic1);
      packed_mtiles_index = rr - h_kv * packed_mtiles_per_seq;
    } else {
      packed_mtiles_index = (int)fdiv((unsigned)rr, magic2);
      h_kv                = rr - packed_mtiles_index * num_kv_heads;
    }
    q_tile_base = packed_mtiles_index * q_tile_per_cta;
  }
  k_tiles = (seqlen + K_TILE - 1) / K_TILE;
  if constexpr (IS_CAUSAL) {
    const int causal_k_tiles_cap = (q_tile_base + q_tile_per_cta - 1) / K_TILE + 1;
    if (causal_k_tiles_cap < k_tiles) k_tiles = causal_k_tiles_cap;
  }
}

// Compile-time kernel config (template args, set in run()'s `constexpr` block):
//   S_LD_COLS        : cols per softmax tcgen05.ld of the S row (32/64 compile; 128 aborts ptxas).
//   FULL_NAMED_BAR   : softmax->corr "scale ready": true = HW named barrier (per-band), false =
//                      mbarrier (full_bar_alpha/full_bar_l). Both use alpha_and_l_smem.
//   EX2_EMU          : route a fraction of softmax exp2 through FFMA f32x2 emulation (vs MUFU.EX2).
//   SPLIT_P          : softmax publishes P in two chunks (96+32 keys); BMM2 starts on the first,
//                      full_bar_p_last gates the tail atoms.
//   SOFTMAX_THROTTLE : FA4 pacing -- corr defers releasing the alpha/l slot until after it consumes,
//                      holding softmax ~1 stage behind correction.
//   USE_CLC          : true = CLC work-stealing sched (w15 sched warp, grid=full problem; wins long S);
//                      false = static grid-stride loop (w15 idle, grid=#SMs; wins short). Same pipe.
//   Q_RASTER         : true = q-tile-innermost raster (adjacent work-items share K/V -> hot L2);
//                      false = kv-head-innermost (the original order).
//   MHA              : true = HQ==HK (gqa_group folds to 1; M-tile = 128 tok x 1 head); false = GQA
//                      (runtime HQ/HK). Body identical; picked by run<MHA>() from main()'s env knob.
//   IS_CAUSAL        : true = triangular causal mask + K-loop cap (uniform seqlen; still non-ragged).
//   LPT              : heaviest-q-tile-first ordering to balance causal load; GATED to IS_CAUSAL.
//   RESCALE_THRESHOLD: sticky-max threshold in log2 units (default 8). If the running max grew by
//                      <= this, keep the old max -> alpha EXACTLY 1.0 -> corr skips the O-rescale.
//                      Higher skips more but risks fp32 overflow of exp2 (2^threshold); 8 is very
//                      safe.

#ifdef ALPHA_DBG
#include "../../../primitives/33_mbarrier_try_wait.cuh"
#define DBGW(b, p) dbg_wait_((b), (p), __LINE__)
#define DBGW2(b, p, sl, kk) dbg_wait_((b), (p), __LINE__, (sl), (kk))
__device__ int g_dbg_item = -1;
__device__ __forceinline__
void dbg_wait_(uint32_t bar, uint32_t parity, int line, int slot = -1, int kk = -1) {
  for (long long n = 0; n < 20000000LL; ++n)
    if (mbarrier_try_wait_parity(bar, parity)) return;
  if ((threadIdx.x & 31) == 0)
    printf("STUCK line=%d warp=%d parity=%u slot=%d k=%d item=%d\n",
           line, (int)(threadIdx.x >> 5), parity, slot, kk, g_dbg_item);
  __trap();
}
#else
#define DBGW(b, p) mbarrier_wait_parity_suspend((b), (p))
#define DBGW2(b, p, sl, kk) mbarrier_wait_parity_suspend((b), (p))
#endif

template <int S_LD_COLS = 32, bool FULL_NAMED_BAR = false, bool EX2_EMU = false, bool SPLIT_P = true,
          bool SOFTMAX_THROTTLE = false, bool USE_CLC = true, bool Q_RASTER = true, bool MHA = false,
          bool IS_CAUSAL = false, bool LPT = false, int RESCALE_THRESHOLD = 8,
          int ALPHA_STAGES = 1>
__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
fmha_context_bf16_gen_kernel(const __grid_constant__ CUtensorMap tmap_q,
    const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v_t,
    const __grid_constant__ CUtensorMap tmap_o, int seqlen,
    int num_q_heads, int num_kv_heads, float scale_log2,
    int packed_mtiles_per_seq, int num_samples,
    unsigned long long magic0, unsigned long long magic1, unsigned long long magic2,
    int lpt_swz_log2, int lpt_hb_quot, int lpt_hb_rem,
    unsigned long long lpt_major_magic, unsigned long long lpt_rem_magic) {
  const int gqa_group_size = MHA ? 1 : (num_q_heads / num_kv_heads);
  const int q_tile_per_mtile = M_TILE / gqa_group_size;        // GQA(8): 16 / MHA: 128
  const int q_tile_per_cta   = M_TILES_PER_CTA * q_tile_per_mtile;     // GQA(8): 32 / MHA: 256
  const int total_workitems = num_samples * packed_mtiles_per_seq * num_kv_heads;

  uint8_t* sQ0 = fmha_smem;
  uint8_t* sQ1 = sQ0 + Q_TILE_BYTES;
  uint8_t* sQ[2] = { sQ0, sQ1 };
  uint8_t* sKV = sQ1 + Q_TILE_BYTES;
  __nv_bfloat16* sO0 = reinterpret_cast<__nv_bfloat16*>(sKV + NUM_KV_STAGES * K_TILE_BYTES);
  __nv_bfloat16* sO1 = sO0 + M_TILE * HEAD_DIM;
  __nv_bfloat16* sO_bufs[2] = { sO0, sO1 };
  uint64_t* full_bar = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(sO1) + M_TILE * HEAD_DIM * sizeof(__nv_bfloat16));
  uint64_t* empty_bar= full_bar + NUM_KV_STAGES;
  uint64_t* full_bar_q  = empty_bar + NUM_KV_STAGES;
  uint64_t* empty_bar_q   = full_bar_q + 2;
  uint64_t* full_bar_spo  = empty_bar_q + 2;
  uint64_t* empty_bar_spo = full_bar_spo + 2;
  uint64_t* full_bar_o_acc   = empty_bar_spo + 2;
  uint64_t* full_bar_alpha = full_bar_o_acc + 2;
  uint64_t* full_bar_l   = full_bar_alpha + 2 * ALPHA_STAGES;
  uint64_t* full_bar_p_last    = full_bar_l + 2;
  uint64_t* empty_bar_alpha_and_l = full_bar_p_last + 2;
  uint64_t* full_bar_o_epi  = empty_bar_alpha_and_l + 2 * ALPHA_STAGES;
  uint64_t* empty_bar_o_epi = full_bar_o_epi + 2;
  uint64_t* clc_full  = empty_bar_o_epi + 2;
  uint64_t* clc_empty = clc_full + CLC_STAGES;
  uint32_t* clc_response = reinterpret_cast<uint32_t*>(
      (reinterpret_cast<uintptr_t>(clc_empty + CLC_STAGES) + 15u) & ~uintptr_t(15u)); // [CLC_STAGES*4]
  uint32_t* tmem_slot = clc_response + CLC_STAGES * 4;
  float* alpha_and_l_smem = reinterpret_cast<float*>(tmem_slot + 2);   // [2][M_TILE]
  // Isolate wait_scale bars on their own wake granule (128B guard both sides): NANOSLEEP
  // sleepers spuriously re-wake on nearby arrive traffic, and the packed bar block's
  // alpha/l/spo arrive storm re-woke our wait 14.4x/wait vs FA4's 3.5. Gated: throughput
  // rows win (+8 r2); boundary/causal lose (-10 r1, -5 r6) since there the spurious wakes
  // keep the waiter responsive to its own imminent arrive.
  if constexpr (!MHA && !IS_CAUSAL) {
    empty_bar_alpha_and_l = reinterpret_cast<uint64_t*>(
        alpha_and_l_smem + 2 * ALPHA_STAGES * M_TILE) + 16;
  }


  static_assert(ALPHA_STAGES == 1 || !FULL_NAMED_BAR,
                "ALPHA_STAGES > 1 needs the mbarrier scale handshake: a HW named "
                "barrier is a rendezvous and cannot be staged");
  static_assert((ALPHA_STAGES & (ALPHA_STAGES - 1)) == 0,
                "ALPHA_STAGES must be a power of two (stage = k & (N-1))");
  // The alpha ring stage is a PURE FUNCTION OF k, never a running pointer:
  // softmax acquires K_TILES+1 slots per work item (one per k-tile, plus one
  // for the l publish) while corr consumes K_TILES, so a per-side tracker
  // drifts by one stage every work item and the two sides deadlock on item 2.
  // Each stage's parity is toggled independently in a bitmask instead.
  constexpr int ALPHA_MASK = ALPHA_STAGES - 1;

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
  WpCtx wpc = wp_ctx_init();

  if (warp_id == 0) {
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), TMEM_TOTAL);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  // Full-TMEM (512-col) alloc always returns base 0; treating it as a compile-time
  // constant (FA4's form: fa4_gen tmem base = 0) frees a live register the mma warp
  // otherwise spills to local and LDL-reloads after every mbarrier wait. Interleaved
  // A/B both orders, 2 GPUs: GQA +109 r2 / +78 r3 / +6 r1, MHA-causal +6 r6 -- but MHA
  // non-causal -10 r5 / -12 r4 (ptxas register-rebalance collateral), so that
  // instantiation keeps the runtime read.
  // TODO(revisit): the (MHA && !IS_CAUSAL) opt-out is a ptxas register-rebalance ARTIFACT,
  //   not a principle -- base is semantically 0 in all cases. The gate is fragile to
  //   toolchain/body changes (retest confirmed -12/-10 -> -25/-21, but it is schedule
  //   collateral). Re-verify with a fresh SASS-diff A/B after any nvcc/ptxas or hot-loop
  //   change; consider dropping the gate (const 0 everywhere) if a newer ptxas schedules it
  //   cleanly.
  const uint32_t tmem_base_rt = *tmem_slot;
  if (tmem_base_rt != 0u) __trap();
  const uint32_t tmem_base = (MHA && !IS_CAUSAL) ? tmem_base_rt : 0u;

  const int packed_mtiles_per_sample = packed_mtiles_per_seq * num_kv_heads;
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
      mbarrier_init(smem_ptr_u32(&empty_bar_spo[i]), 256);
      mbarrier_init(smem_ptr_u32(&full_bar_o_acc[i]), 1);
      #pragma unroll
      for (int st = 0; st < ALPHA_STAGES; ++st) {
        mbarrier_init(smem_ptr_u32(&full_bar_alpha[i * ALPHA_STAGES + st]), 128);
        mbarrier_init(smem_ptr_u32(&empty_bar_alpha_and_l[i * ALPHA_STAGES + st]), 128);
      }
      mbarrier_init(smem_ptr_u32(&full_bar_p_last[i]), 128);
      mbarrier_init(smem_ptr_u32(&full_bar_o_epi[i]), 128);
      mbarrier_init(smem_ptr_u32(&empty_bar_o_epi[i]), 1);
    }
    if constexpr (USE_CLC) {
      #pragma unroll
      for (int s = 0; s < CLC_STAGES; ++s) {
        mbarrier_init(smem_ptr_u32(&clc_full[s]), 1);
        mbarrier_init(smem_ptr_u32(&clc_empty[s]), N_WARPS);
      }
      #pragma unroll
      for (int i = 0; i < CLC_STAGES * 4; ++i)
        clc_response[i] = 0;
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();

  if (warp_id == W_LOAD) {
    setmaxnreg_dec<48>();

    EmptyPhaseTracker<NUM_KV_STAGES> kv_empty_ph;
    EmptyPhaseTracker<1> q_empty_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      wp_marker(wpc, WP_ITEM, workitem_id);
#ifdef ALPHA_DBG
      if (threadIdx.x == 0) g_dbg_item = workitem_id;
#endif
      int sample, h_kv, q_tile_base, K_TILES;
      decode_workitem<Q_RASTER, IS_CAUSAL, LPT>(workitem_id, seqlen, num_kv_heads,
          packed_mtiles_per_seq, packed_mtiles_per_sample, q_tile_per_cta,
          magic0, magic1, magic2,
          lpt_swz_log2, lpt_hb_quot, lpt_hb_rem, lpt_major_magic, lpt_rem_magic,
          sample, h_kv, q_tile_base, K_TILES);
      const int k_start = sample * seqlen;

      for (int k = 0; k < K_TILES; ++k) {
        wp_marker(wpc, WP_ITER, k);
	const int k_offset = (K_TILES - 1 - k) * K_TILE;
        int kv_stage = kv_empty_ph.get_stage();

        wp_begin(wpc, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        wp_end(wpc, WP_LOAD_WAIT);

        wp_begin(wpc, WP_LOAD_ISSUE_K);
        // FA4 guard granularity (2SM-proven): waits + address/coord math UNGUARDED (straight-line
        // -> ptxas proves warp-uniform, computes into URs); only arrive + TMA sit in the elect block.
        // ONE 3D copy folds both head-dim atoms (vs 2x 2D): coords {atom-col 0, token, atom
        // h_kv*K_SUBTILES}, box [SUB_COLS_BF16, K_TILE, K_SUBTILES].
        const uint32_t kbar = smem_ptr_u32(&full_bar[kv_stage]);
        const uint32_t kdst = smem_ptr_u32(sKV + kv_stage * K_TILE_BYTES);   // arithmetic (no runtime-indexed local array -> no LDL)
        const int      k_token = k_start + k_offset;
        const int      k_head_atom  = h_kv * K_SUBTILES;
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(kbar, K_TILE_BYTES);
          tma_load_3d(kdst, &tmap_k, kbar, 0, k_token, k_head_atom);
        }
        wp_end(wpc, WP_LOAD_ISSUE_K);

        if (k == 0) {
          // 4D Q box (token-in-sample dim): rows past seqlen zero-fill, not the next sample.
          #pragma unroll
          for (int m = 0; m < M_TILES_PER_CTA; ++m) {
            wp_begin(wpc, WP_LOAD_WAIT);
            mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_q[m]), q_empty_ph.get_phase());
            wp_end(wpc, WP_LOAD_WAIT);

            wp_begin(wpc, WP_LOAD_ISSUE_Q);
            const uint32_t qbar = smem_ptr_u32(&full_bar_q[m]);
            const int q_token = q_tile_base + m * q_tile_per_mtile;
            const int q_head  = h_kv * gqa_group_size;
            // box [hd/2 x gqa_group_size x q_tile_per_mtile x 1], qh-inner packed rows.
            if (elect_one_sync()) {
              mbarrier_arrive_expect_tx(qbar, Q_TILE_BYTES);
              #pragma unroll
              for (int s = 0; s < Q_SUBTILES; ++s) {
                tma_load_4d(smem_ptr_u32(sQ[m] + s * Q_SUB_COLS_BYTES), &tmap_q, qbar,
                            s * SUB_COLS_BF16, q_head, q_token, sample);
              }
            }
            wp_end(wpc, WP_LOAD_ISSUE_Q);
          }
          q_empty_ph.advance();
	  }
        kv_stage = kv_empty_ph.get_stage();

        wp_begin(wpc, WP_LOAD_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar[kv_stage]), kv_empty_ph.get_phase());
        kv_empty_ph.advance();
        wp_end(wpc, WP_LOAD_WAIT);

        wp_begin(wpc, WP_LOAD_ISSUE_V);
        const uint32_t vbar = smem_ptr_u32(&full_bar[kv_stage]);
        const uint32_t vdst = smem_ptr_u32(sKV + kv_stage * K_TILE_BYTES);
        const int      v_token = k_start + k_offset;
        const int      v_head_col = h_kv * HEAD_DIM;
        if (elect_one_sync()) {
          mbarrier_arrive_expect_tx(vbar, V_TILE_BYTES);
          #pragma unroll
          for (int s = 0; s < V_SUBTILES; ++s) {
            tma_load_2d(vdst + (uint32_t)(s * Q_SUB_COLS_BYTES), &tmap_v_t, vbar,
                        v_token + s * SUB_COLS_BF16, v_head_col);
          }
        }
        wp_end(wpc, WP_LOAD_ISSUE_V);
      }
      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  else if (warp_id == W_MMA) {
    setmaxnreg_dec<48>();

    const uint32_t lead = elect_one_sync() ? 1u : 0u;
    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    // TMEM layout: S[i] at i*128, O[i] at 256 + i*128 (512 cols total). S is
    // single-buffered; the 2 M-tiles provide the cross-tile overlap instead.
    const uint32_t idesc_qk = make_idesc_bf16_f32(M_TILE, K_TILE, false, false); // N=128
    const uint32_t idesc_pv = make_idesc_bf16_f32(M_TILE, HEAD_DIM,  false, false);
    const uint64_t desc_q0  = build_smem_desc_blackwell(smem_ptr_u32(sQ0), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    const uint64_t desc_kv0 = build_smem_desc_blackwell(smem_ptr_u32(sKV), DESC_SBO, DESC_LBO, SmemSwizzleBlackwell::B128);
    // >> 4: the SMEM descriptor address field is in 16-byte units (addr >> 4).
    constexpr uint64_t KV_DESC_DELTA      = K_TILE_BYTES >> 4;
    constexpr uint64_t SUB_DESC_DELTA     = Q_SUB_COLS_BYTES >> 4;
    constexpr uint64_t Q_MTILE_DESC_DELTA = Q_TILE_BYTES >> 4;

    PhaseTracker<NUM_KV_STAGES> kv_ph;
    PhaseTracker<1> q_ph;
    PhaseTracker<1> spo_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      wp_marker(wpc, WP_ITEM, workitem_id);
#ifdef ALPHA_DBG
      if (threadIdx.x == 0) g_dbg_item = workitem_id;
#endif
      int sample, h_kv, q_tile_base, K_TILES;
      decode_workitem<Q_RASTER, IS_CAUSAL, LPT>(workitem_id, seqlen, num_kv_heads,
          packed_mtiles_per_seq, packed_mtiles_per_sample, q_tile_per_cta,
          magic0, magic1, magic2,
          lpt_swz_log2, lpt_hb_quot, lpt_hb_rem, lpt_major_magic, lpt_rem_magic,
          sample, h_kv, q_tile_base, K_TILES);

      // prologue: BMM1 of K-block 0 for every M-tile. FA4's operand-wait ORDER
      // (fa4_gen.py:1414-1436): Q0 first (its TMA landed ~3us ago -- the check is free
      // and overlaps K0's flight), K0 after. Our old K-then-Q order paid the Q barrier
      // check serially AFTER K arrived: r1 item-boundary trace measured L6+L7 = 416ns
      // vs FA4's 64ns on the tail chain that gates the next item's first S.
      int kv_stage = kv_ph.get_stage();
      // KK: K-first operand wait (unified with 2SM). r1 +9.5 but r2 -6.5 -- kept for
      // cross-kernel consistency (both kernels: load K-first + wait K-first).
      // iter 0: the prologue pass, 2 BMM1 only. Every later pass is 2 BMM2 +
      // 2 BMM1 and the epilogue is 2 BMM2 only, so these two are the half-work
      // ends of the pipeline. Marking only the loop would leave this outside
      // any iteration.
      wp_marker(wpc, WP_ITER, 0);
      wp_begin(wpc, WP_MMA_WAIT_FULL_K);
      mbarrier_wait_parity(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
      kv_ph.advance();
      wp_end(wpc, WP_MMA_WAIT_FULL_K);
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        wp_begin(wpc, WP_MMA_WAIT_FULL_Q);
        mbarrier_wait_parity(smem_ptr_u32(&full_bar_q[i]), q_ph.get_phase());
        wp_end(wpc, WP_MMA_WAIT_FULL_Q);

        wp_begin(wpc, WP_MMA_ISSUE);
        {
          const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
          SmemDescPair da, db;
          da.u64 = desc_q0;  da.w.x += (uint32_t)(i * (int)Q_MTILE_DESC_DELTA);
          db.u64 = desc_kv0; db.w.x += (uint32_t)(kv_stage * (int)KV_DESC_DELTA);

          #pragma unroll
          for (int s = 0; s < Q_SUBTILES; ++s) {
            #pragma unroll
            for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
              const bool enable_d = (s != 0) || (ki != 0);
              tcgen05_mma_f16_ss_lead(lead, s_tmem_addr, da.u64, db.u64, idesc_qk, enable_d);
              desc_add_lo(da, 2); desc_add_lo(db, 2);
            }
            desc_add_lo(da, (uint32_t)(SUB_DESC_DELTA - 2 * K_ATOMS_PER_TILE));
            desc_add_lo(db, (uint32_t)(SUB_DESC_DELTA - 2 * K_ATOMS_PER_TILE));
          }
        }
        wp_end(wpc, WP_MMA_ISSUE);

        wp_marker(wpc, WP_MMA_COMMIT, i);
        tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_spo[i]));
      }

      wp_marker(wpc, WP_MMA_COMMIT, kv_stage);
      tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[kv_stage]));

      // main loop: BMM2(tile) then BMM1(next tile)
      for (int k_tile_id = 0; k_tile_id + 1 < K_TILES; ++k_tile_id) {
        // ring: kv_stage = V(current) for BMM2, kv_stage_next = K(next) for BMM1-ahead.
	      const int kv_stage = kv_ph.get_stage();

        wp_begin(wpc, WP_MMA_WAIT_FULL_V);
        mbarrier_wait_parity(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
        kv_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_FULL_V);

        int kv_stage_next = 0;
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          wp_begin(wpc, WP_MMA_WAIT_P);
          mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
          wp_end(wpc, WP_MMA_WAIT_P);

          wp_begin(wpc, WP_MMA_ISSUE);
          const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
          const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
          SmemDescPair dbv;
	  dbv.u64 = desc_kv0;
	  dbv.w.x += (uint32_t)(kv_stage * (int)KV_DESC_DELTA);

          #pragma unroll
          for (int s = 0; s < V_SUBTILES; ++s) {
            #pragma unroll
            for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
              const int a = s * K_ATOMS_PER_TILE + ki;
              if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
                wp_end(wpc, WP_MMA_ISSUE);
                wp_begin(wpc, WP_MMA_WAIT_P);
                mbarrier_wait_parity(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
                wp_end(wpc, WP_MMA_WAIT_P);
                wp_begin(wpc, WP_MMA_ISSUE);
              }
              const bool accumulate = (k_tile_id != 0) || (a != 0);
              tcgen05_mma_f16_ts_1sm_lead(lead, o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), dbv.u64, idesc_pv, accumulate);
              desc_add_lo(dbv, 2);
            }
            desc_add_lo(dbv, (uint32_t)(SUB_DESC_DELTA - 2 * K_ATOMS_PER_TILE));
          }
          wp_end(wpc, WP_MMA_ISSUE);

          // K(next) is shared by both M-tiles: only i==0 waits + advances the ring.
          if (i == 0) {
            kv_stage_next = kv_ph.get_stage();
            wp_begin(wpc, WP_MMA_WAIT_FULL_K);
            mbarrier_wait_parity(smem_ptr_u32(&full_bar[kv_stage_next]), kv_ph.get_phase());
	    wp_end(wpc, WP_MMA_WAIT_FULL_K);
            kv_ph.advance();
          }

          if (i == M_TILES_PER_CTA - 1) {
            wp_marker(wpc, WP_MMA_COMMIT, kv_stage);
            tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[kv_stage]));
          }

          // BMM1(next): Q@K -> S
          wp_begin(wpc, WP_MMA_ISSUE);
          {
            SmemDescPair da, db;
            da.u64 = desc_q0;  da.w.x += (uint32_t)(i * (int)Q_MTILE_DESC_DELTA);
            db.u64 = desc_kv0; db.w.x += (uint32_t)(kv_stage_next * (int)KV_DESC_DELTA);
            #pragma unroll
            for (int s = 0; s < Q_SUBTILES; ++s) {
              #pragma unroll
              for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
                const bool enable_d = (s != 0) || (ki != 0);
                tcgen05_mma_f16_ss_lead(lead, s_tmem_addr, da.u64, db.u64, idesc_qk, enable_d);
                desc_add_lo(da, 2); desc_add_lo(db, 2);
              }
              desc_add_lo(da, (uint32_t)(SUB_DESC_DELTA - 2 * K_ATOMS_PER_TILE));
              desc_add_lo(db, (uint32_t)(SUB_DESC_DELTA - 2 * K_ATOMS_PER_TILE));
            }
          }
          wp_end(wpc, WP_MMA_ISSUE);

          wp_marker(wpc, WP_MMA_COMMIT, i);
          tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_spo[i]));
        }
	wp_marker(wpc, WP_MMA_COMMIT, kv_stage_next);
        tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[kv_stage_next]));

        spo_ph.advance();
        // One marker per k-tile loop pass, not per m-tile, at the very end of
        // the body so the pass's own empty_bar commit closes it. A pass is a
        // fixed unit of work -- 2 BMM2 + 2 BMM1, both m-tiles -- so passes are
        // directly comparable even though the BMM2 and the BMM1-ahead belong to
        // different k-tiles. iter 0 is the exception: it carries the prologue's
        // 2 BMM1 on top of a full pass.
        wp_marker(wpc, WP_ITER, k_tile_id + 1);
      }

      wp_marker(wpc, WP_MMA_COMMIT, 0);
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar_q[i]));
      }

      // epilogue: BMM2 of the last K-block -> final O. No marker here: the last
      // loop pass already opened this iteration, and marking again would create
      // a span holding only the empty_bar_q release -- an iteration with no MMA
      // work in it.
      kv_stage = kv_ph.get_stage();

      wp_begin(wpc, WP_MMA_WAIT_FULL_V);
      mbarrier_wait_parity(smem_ptr_u32(&full_bar[kv_stage]), kv_ph.get_phase());
      kv_ph.advance();
      wp_end(wpc, WP_MMA_WAIT_FULL_V);

      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        wp_begin(wpc, WP_MMA_WAIT_P);
        mbarrier_wait_parity(smem_ptr_u32(&empty_bar_spo[i]), spo_ph.get_phase());
        wp_end(wpc, WP_MMA_WAIT_P);

	wp_begin(wpc, WP_MMA_ISSUE);
        const uint32_t s_tmem_addr = tmem_base + (uint32_t)(i * S_COLS);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS);
        SmemDescPair dbv;
	dbv.u64 = desc_kv0;
	dbv.w.x += (uint32_t)(kv_stage * (int)KV_DESC_DELTA);

        #pragma unroll
        for (int s = 0; s < V_SUBTILES; ++s) {
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
            const int a = s * K_ATOMS_PER_TILE + ki;   // flat atom (split-P + P addr)
            if constexpr (SPLIT_P) if (a == SPLIT_P_ATOM) {
              wp_end(wpc, WP_MMA_ISSUE);
              wp_begin(wpc, WP_MMA_WAIT_P);
              mbarrier_wait_parity(smem_ptr_u32(&full_bar_p_last[i]), spo_ph.get_phase());
              wp_end(wpc, WP_MMA_WAIT_P);
              wp_begin(wpc, WP_MMA_ISSUE);
            }
            const bool accumulate = (K_TILES != 1) || (a != 0);
            tcgen05_mma_f16_ts_1sm_lead(lead, o_tmem_addr, s_tmem_addr + (uint32_t)(a * 8), dbv.u64, idesc_pv, accumulate);
            desc_add_lo(dbv, 2);
          }
          desc_add_lo(dbv, (uint32_t)(SUB_DESC_DELTA - 2 * K_ATOMS_PER_TILE));
        }
        wp_end(wpc, WP_MMA_ISSUE);

        wp_marker(wpc, WP_MMA_COMMIT, i);
        tcgen05_commit1_lead(lead, smem_ptr_u32(&full_bar_o_acc[i]));
      }
      wp_marker(wpc, WP_MMA_COMMIT, kv_stage);
      tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar[kv_stage]));

      spo_ph.advance();
      q_ph.advance();

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  else if (warp_id == W_EPI) {
    setmaxnreg_dec<48>();

    PhaseTracker<1> full_o_ph;
    // Prime empty_bar_o_epi once: corr's first sO pack must not block (no prior store in flight).
    if (elect_one_sync()) {
      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m)
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[m]));
    }
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      wp_marker(wpc, WP_ITEM, workitem_id);
#ifdef ALPHA_DBG
      if (threadIdx.x == 0) g_dbg_item = workitem_id;
#endif
      int sample, h_kv, q_tile_base, K_TILES;
      decode_workitem<Q_RASTER, IS_CAUSAL, LPT>(workitem_id, seqlen, num_kv_heads,
          packed_mtiles_per_seq, packed_mtiles_per_sample, q_tile_per_cta,
          magic0, magic1, magic2,
          lpt_swz_log2, lpt_hb_quot, lpt_hb_rem, lpt_major_magic, lpt_rem_magic,
          sample, h_kv, q_tile_base, K_TILES);

      #pragma unroll
      for (int m = 0; m < M_TILES_PER_CTA; ++m) {
        wp_marker(wpc, WP_ITER, m);
        wp_begin(wpc, WP_EPI_WAIT_TMEM);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_epi[m]), full_o_ph.get_phase());
        wp_end(wpc, WP_EPI_WAIT_TMEM);

        wp_begin(wpc, WP_EPI_STORE);
        if (elect_one_sync()) {
          const int q_token = q_tile_base + m * q_tile_per_mtile;
          // 4D map (token-in-sample dim): rows past seqlen are clamped away, not
          // written into the next sample's tokens.
          #pragma unroll
          for (int s = 0; s < Q_SUBTILES; ++s) {
            tma_store_4d(&tmap_o, s * SUB_COLS_BF16, h_kv * gqa_group_size, q_token, sample,
                         smem_ptr_u32(reinterpret_cast<const uint8_t*>(sO_bufs[m]) + s * Q_SUB_COLS_BYTES));
          }
          cp_async_bulk_commit_group();
        }
        wp_end(wpc, WP_EPI_STORE);
      }

      wp_begin(wpc, WP_EPI_WAIT_STORE);
      // Drain per-m commit groups; release each sO slot to corr as ITS store completes
      // (without this, corr's next pack races the in-flight store at tiny causal K-loops).
      if (elect_one_sync()) {
        cp_async_bulk_wait_group_read<1>();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[0]));
        cp_async_bulk_wait_group_read<0>();
        mbarrier_arrive(smem_ptr_u32(&empty_bar_o_epi[1]));
      }
      wp_end(wpc, WP_EPI_WAIT_STORE);

      full_o_ph.advance();
      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  else if (warp_id == W_SCHED) {
    setmaxnreg_dec<48>();

    if constexpr (USE_CLC) {
      int prod_stage = 0; uint32_t prod_phase = 1;
      int cons_stage = 0; uint32_t cons_phase = 0;
      while (true) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        clc_arrive_expect_tx_cta(smem_ptr_u32(&clc_full[prod_stage]), /*tx_bytes=*/16);
        if (lane == 0)
          clc_try_cancel_async(smem_ptr_u32(&clc_response[prod_stage * 4]),
                               smem_ptr_u32(&clc_full[prod_stage]));
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, cons_stage, cons_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(cons_stage, cons_phase);
        if (!next.valid) break;
      }
      // Tail drain: absorb the in-flight consumer releases before kernel exit.
      for (int s = 0; s < CLC_STAGES; ++s) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
      }
    }
  }
  else if (warp_id >= W_CORR0 && warp_id < W_MMA) {
    setmaxnreg_dec<80>();

    const int corr_warp_id = warp_id - W_CORR0;
    [[maybe_unused]] PhaseTracker<1> alpha_ph;
    // one parity bit per alpha ring stage; stage index comes from k
    [[maybe_unused]] uint32_t alpha_ring_ph = 0;
    PhaseTracker<1> o_acc_ph;
    PhaseTracker<1> o_epi_empty_ph;
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;

    // Prime the return barriers once (first BMM2 / first softmax stat write).
    #pragma unroll
    for (int i = 0; i < M_TILES_PER_CTA; ++i) {
      mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
      #pragma unroll
      for (int st = 0; st < ALPHA_STAGES; ++st)
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i * ALPHA_STAGES + st]));
    }

    int workitem_id = (int)blockIdx.x;
    while (true) {
      wp_marker(wpc, WP_ITEM, workitem_id);
#ifdef ALPHA_DBG
      if (threadIdx.x == 0) g_dbg_item = workitem_id;
#endif
      int sample, h_kv, q_tile_base, K_TILES;
      decode_workitem<Q_RASTER, IS_CAUSAL, LPT>(workitem_id, seqlen, num_kv_heads,
          packed_mtiles_per_seq, packed_mtiles_per_sample, q_tile_per_cta,
          magic0, magic1, magic2,
          lpt_swz_log2, lpt_hb_quot, lpt_hb_rem, lpt_major_magic, lpt_rem_magic,
          sample, h_kv, q_tile_base, K_TILES);

      // block 0: no rescale (no prior O); consume alpha + release the scale slot.
      // This IS corr's k-tile 0 -- the loop below starts at k=1 -- so it needs
      // its own marker, or k=0 would fall outside every iteration.
      wp_marker(wpc, WP_ITER, 0);
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        wp_begin(wpc, WP_CORR_WAIT);
        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else DBGW(smem_ptr_u32(&full_bar_alpha[i * ALPHA_STAGES + 0]),
                  (alpha_ring_ph >> 0) & 1u);
        // FA4: cross-stage deferral: prologue releases only slot 0 -> softmax runs ~1 K-block behind, S always ready (wait_s ~0).
        if constexpr (SOFTMAX_THROTTLE) {
          // Depth 1: corr's release at k serves softmax's step k+1, so only
          // m-tile 0 needs the prologue credit and m-tile 1's shortfall is paid
          // by the post-loop rebalance. Depth >1: the release at k serves step
          // k+2, so BOTH bands need it during ramp-up -- without it band 1
          // stalls one step in and corr then blocks on its missing alpha.
          if (ALPHA_STAGES > 1 || i == 0)
            mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i * ALPHA_STAGES + 0]));
        } else {
          mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i * ALPHA_STAGES + 0]));
        }
        wp_end(wpc, WP_CORR_WAIT);
      }
      if constexpr (!FULL_NAMED_BAR) { alpha_ph.advance(); alpha_ring_ph ^= 1u; }

      for (int k = 1; k < K_TILES; ++k) {
        wp_marker(wpc, WP_ITER, k);
        #pragma unroll
        for (int i = 0; i < M_TILES_PER_CTA; ++i) {
          const int k_st = k & ALPHA_MASK;
          wp_begin(wpc, WP_CORR_WAIT);
          if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
          else DBGW2(smem_ptr_u32(&full_bar_alpha[i * ALPHA_STAGES + k_st]),
                    (alpha_ring_ph >> k_st) & 1u, i * ALPHA_STAGES + k_st, k);
          wp_end(wpc, WP_CORR_WAIT);

          wp_begin(wpc, WP_CORR_READ_ALPHA);
          float alpha = alpha_and_l_smem[(i * ALPHA_STAGES + k_st) * M_TILE + corr_warp_id * 32 + lane];
          if constexpr (!SOFTMAX_THROTTLE)
            mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[i * ALPHA_STAGES + k_st]));
          wp_end(wpc, WP_CORR_READ_ALPHA);

          wp_begin(wpc, WP_CORR_O_SCALE);
          bool skip = __all_sync(0xffffffffu, alpha == 1.0f);
          if (!skip) {
            // O(g-1) is done: BMM1(g) trails BMM2(g-1) in the in-order tcgen05 pipe.
            const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
            // x16 chunks: x64 spills the 80-reg corr budget; x32 now neutral -- keep x16.
            const float2 alpha2 = f32x2_splat(alpha);
            // per-chunk LDTM -> FMUL2 -> STTM, no per-chunk waits; one trailing wait::st drains.
            #pragma unroll
            for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
              uint32_t o_regs[16];
              tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
              float2* o2 = reinterpret_cast<float2*>(o_regs);
              #pragma unroll
              for (int e = 0; e < 8; ++e) o2[e] = fmul2(o2[e], alpha2);
              tcgen05_st_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
            }
            tcgen05_wait_st();
            // publish rescaled O for BMM2
            tcgen05_fence_before_thread_sync();
          }
          // FA4: release the OTHER tile's scale slot (cross-stage deferred release).
          if constexpr (SOFTMAX_THROTTLE)
            mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[
                (M_TILES_PER_CTA - 1 - i) * ALPHA_STAGES + k_st]));
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));
          wp_end(wpc, WP_CORR_O_SCALE);
        }
        if constexpr (!FULL_NAMED_BAR) {
          alpha_ph.advance();
          alpha_ring_ph ^= (1u << (k & ALPHA_MASK));
        }
      }
      // FA4 post-loop rebalance: deferral left slot 1 one release short; pay it so l-publish can proceed.
      const int l_st = K_TILES & ALPHA_MASK;
      // Only needed when the prologue credited m-tile 0 alone (see above).
      if constexpr (SOFTMAX_THROTTLE && ALPHA_STAGES == 1)
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[
            (M_TILES_PER_CTA - 1) * ALPHA_STAGES + l_st]));

      // epilogue: O *= 1/l -> bf16 -> sO[i] -> signal W_EPI (full_bar_o_epi).
      #pragma unroll
      for (int i = 0; i < M_TILES_PER_CTA; ++i) {
        wp_begin(wpc, WP_CORR_WAIT);
        mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_o_acc[i]), o_acc_ph.get_phase());
        // full_bar_wait = HW named barrier (bar_sync<id>(64)): a TWO-SIDED rendezvous -- softmax
        // arrives when l is published + corr waits here; both sides release together (not a one-way
        // mbarrier). Mirror of the alpha wait at 845. Only means l is READY: corr consumes it at
        // the read (896), slot release is 898.
        if constexpr (FULL_NAMED_BAR) full_bar_wait(i, corr_warp_id);
        else mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_l[i]), o_acc_ph.get_phase());
        wp_end(wpc, WP_CORR_WAIT);

        wp_begin(wpc, WP_CORR_EPI);
        const int corr_tid = corr_warp_id * 32 + lane;
        const int l_slot_i = i * ALPHA_STAGES + l_st;
        float l = alpha_and_l_smem[l_slot_i * M_TILE + corr_tid];
        mbarrier_arrive(smem_ptr_u32(&empty_bar_alpha_and_l[l_slot_i]));
        float inv_l = (l > 0.f) ? rcp_approx_ftz_f32(l) : 0.f;
        const float2 inv_l2 = f32x2_splat(inv_l);
        const uint32_t o_tmem_addr = tmem_base + (uint32_t)(2 * S_COLS + i * O_COLS) + ((uint32_t)(corr_warp_id * 32) << 16);
        #pragma unroll
        for (int c0 = 0; c0 < HEAD_DIM; c0 += 16) {
          uint32_t o_regs[16];
          tcgen05_ld_32x32b_x16(o_tmem_addr + (uint32_t)c0, o_regs);
          if (c0 == 0) mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_o_epi[i]), o_epi_empty_ph.get_phase());
          float2* o2 = reinterpret_cast<float2*>(o_regs);
          const int s = c0 / SUB_COLS_BF16;
          const int v_base = (c0 % SUB_COLS_BF16) / 8;
          __nv_bfloat16* so_sub = sO_bufs[i] + s * (M_TILE * SUB_COLS_BF16);
          #pragma unroll
          for (int vv = 0; vv < 2; ++vv) {
            const int v = v_base + vv;
            const float2 r0 = fmul2(o2[vv * 4 + 0], inv_l2);
            const float2 r1 = fmul2(o2[vv * 4 + 1], inv_l2);
            const float2 r2 = fmul2(o2[vv * 4 + 2], inv_l2);
            const float2 r3 = fmul2(o2[vv * 4 + 3], inv_l2);
            uint4 packed;
            packed.x = cvt_f32x2_to_bf16x2(r0.x, r0.y);
            packed.y = cvt_f32x2_to_bf16x2(r1.x, r1.y);
            packed.z = cvt_f32x2_to_bf16x2(r2.x, r2.y);
            packed.w = cvt_f32x2_to_bf16x2(r3.x, r3.y);
	    *reinterpret_cast<uint4*>(&so_sub[corr_tid * SUB_COLS_BF16 + (v ^ (corr_tid & 7)) * 8]) = packed;
          }
        }
        // O ld done -> MMA may reuse the O slot
        tcgen05_fence_before_thread_sync();

        mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[i]));

        // order this thread's st.shared writes (generic proxy) before TMA store (async proxy)
        fence_proxy_async_shared();

        mbarrier_arrive(smem_ptr_u32(&full_bar_o_epi[i]));
        wp_end(wpc, WP_CORR_EPI);
      }
      o_acc_ph.advance();
      o_epi_empty_ph.advance();

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  else {
    setmaxnreg_inc<192>();

    // warp-uniform hint (FA4 make_warp_uniform): R2UR.BROADCAST -> promotes m_tile + derived to URs.
    const int warp_id_u = __shfl_sync(0xffffffffu, warp_id, 0);
    const int m_tile = warp_id_u < 4 ? 0 : 1;
    const int warp_in_group = warp_id_u & 3;
    const int row_in_m_tile = warp_in_group * 32 + lane;
    const uint32_t s_tmem_addr = tmem_base + (uint32_t)(m_tile * S_COLS) + ((uint32_t)(warp_in_group * 32) << 16);
    PhaseTracker<1> spo_ph;
    PhaseTracker<1> scale_empty_ph;
    uint32_t scale_ring_ph = 0;   // one parity bit per alpha ring stage
    [[maybe_unused]] int clc_stage = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;
    int workitem_id = (int)blockIdx.x;
    while (true) {
      wp_marker(wpc, WP_ITEM, workitem_id);
#ifdef ALPHA_DBG
      if (threadIdx.x == 0) g_dbg_item = workitem_id;
#endif
      int sample, h_kv, q_tile_base, K_TILES;
      decode_workitem<Q_RASTER, IS_CAUSAL, LPT>(workitem_id, seqlen, num_kv_heads,
          packed_mtiles_per_seq, packed_mtiles_per_sample, q_tile_per_cta,
          magic0, magic1, magic2,
          lpt_swz_log2, lpt_hb_quot, lpt_hb_rem, lpt_major_magic, lpt_rem_magic,
          sample, h_kv, q_tile_base, K_TILES);

      // causal: this row's query-token position (pack-GQA qh-inner); unused when IS_CAUSAL=false.
      const int q_pos = q_tile_base + m_tile * q_tile_per_mtile + row_in_m_tile / gqa_group_size;
      float m_run = -INFINITY, l_run = 0.f;
      wp_begin(wpc, WP_SM_WAIT_SCALE);
      if constexpr (!MHA && !IS_CAUSAL) {
        DBGW(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile * ALPHA_STAGES + 0]), scale_ring_ph & 1u);
      } else {
        DBGW(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile * ALPHA_STAGES + 0]), scale_ring_ph & 1u);
      }
      scale_ring_ph ^= 1u;
      scale_empty_ph.advance();
      wp_end(wpc, WP_SM_WAIT_SCALE);
      int alpha_stage = 0;
      float* alpha_slot = &alpha_and_l_smem[(m_tile * ALPHA_STAGES + alpha_stage) * M_TILE + row_in_m_tile];
      uint32_t alpha_slot_u32 = smem_ptr_u32(alpha_slot);
      // k==0 (masked, alpha-less first block) PEELED (masked_c/is_first_c = std::true_type); the
      // steady body stays branch-free. MASKED/IS_FIRST are constexpr via the integral_constant args.
      auto softmax_step = [&](auto masked_c, auto is_first_c, int k) {
        constexpr bool MASKED   = decltype(masked_c)::value;
        constexpr bool IS_FIRST = decltype(is_first_c)::value;
        const int k_offset = (K_TILES - 1 - k) * K_TILE;
        wp_marker(wpc, WP_ITER, k);
        wp_begin(wpc, WP_SM_WAIT_S);
        if constexpr (!MHA && !IS_CAUSAL) {
          mbarrier_wait_parity(smem_ptr_u32(&full_bar_spo[m_tile]), spo_ph.get_phase());
        } else {
          mbarrier_wait_parity_suspend(smem_ptr_u32(&full_bar_spo[m_tile]), spo_ph.get_phase());
        }
        wp_end(wpc, WP_SM_WAIT_S);

        wp_begin(wpc, WP_SM_SOFTMAX);
        uint32_t s_regs[K_TILE];
        #pragma unroll
        for (int c0 = 0; c0 < K_TILE; c0 += S_LD_COLS) {
          const uint32_t taddr = s_tmem_addr + (uint32_t)c0;
          if      constexpr (S_LD_COLS == 32)  tcgen05_ld_32x32b_x32 (taddr, *reinterpret_cast<uint32_t(*)[32]>(&s_regs[c0]));
          else if constexpr (S_LD_COLS == 64)  tcgen05_ld_32x32b_x64 (taddr, *reinterpret_cast<uint32_t(*)[64]>(&s_regs[c0]));
          else if constexpr (S_LD_COLS == 128) tcgen05_ld_32x32b_x128(taddr, *reinterpret_cast<uint32_t(*)[128]>(&s_regs[c0]));
        }
        float* scores = reinterpret_cast<float*>(s_regs);
        float2* scores2 = reinterpret_cast<float2*>(s_regs);

        // order the S read ahead of the P store that overwrites the slot
        tcgen05_fence_before_thread_sync();

        // k==0 (diagonal) is the only masked block in non-causal; causal masks the near-diagonal band.
        if constexpr (MASKED) mask_s_row_r2p<IS_CAUSAL, K_TILE>(scores, k_offset, q_pos, seqlen);

        float rmax0 = m_run, rmax1 = -INFINITY, rmax2 = -INFINITY, rmax3 = -INFINITY;
        #pragma unroll
        for (int j = 0; j < K_TILE; j += 8) {
          rmax0 = fmaxf(fmaxf(rmax0, scores[j + 0]), scores[j + 1]);
          rmax1 = fmaxf(fmaxf(rmax1, scores[j + 2]), scores[j + 3]);
          rmax2 = fmaxf(fmaxf(rmax2, scores[j + 4]), scores[j + 5]);
	  rmax3 = fmaxf(fmaxf(rmax3, scores[j + 6]), scores[j + 7]);
        }
        float new_m = fmaxf(fmaxf(rmax0, rmax1), fmaxf(rmax2, rmax3));
        // Fully-masked tile guard: if every key is masked, new_m stays -inf. Feeding that into the
        // exp below gives exp2(score - new_m) = exp2(-inf - (-inf)) = exp2(NaN) = NaN. Substituting 0
        // makes it exp2(-inf) = 0 -> a correctly all-zero P row (contributes nothing), no NaN.
        float row_max_safe = (new_m == -INFINITY) ? 0.0f : new_m;
        float alpha = 0.0f;
        if constexpr (!IS_FIRST) {
          const float acc_scale_ = (m_run - row_max_safe) * scale_log2;
          alpha = ex2_approx_f32(acc_scale_);
          if (acc_scale_ >= -(float)RESCALE_THRESHOLD) {
            new_m = m_run;
            row_max_safe = m_run;
            alpha = 1.0f;
          }
          // Publish alpha. Pin the store for causal + MHA with volatile STS -- else ptxas sinks it
          // to ~66% of the body, starving corr of alpha. GQA keeps the generic store (faster where
          // MIO is the busy pipe, Q11).
          if constexpr (MHA || IS_CAUSAL) sts_f32(alpha_slot_u32, alpha);
          else                            *alpha_slot = alpha;
        }
        if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
        else mbarrier_arrive(smem_ptr_u32(&full_bar_alpha[m_tile * ALPHA_STAGES + alpha_stage]));

        // fused ffma2(scale) + exp2 + bf16 pack; row-sum DEFERRED to the wait_scale shadow (in place).
        const float2 scale2        = f32x2_splat(scale_log2);
        const float2 neg_m_scaled2 = f32x2_splat(-row_max_safe * scale_log2);
        uint32_t p_regs[K_TILE / 2];
        // exp2 written straight into scores2[c] (in place) -- consumed by the deferred row-sum
        // and the bf16 pack. scores2 aliases the s_regs registers, so the read-back is free.
        #pragma unroll
	for (int c = 0; c < K_TILE / 2; ++c) {
          const float2 a2 = ffma2(scores2[c], scale2, neg_m_scaled2);
          if constexpr (EX2_EMU) {
            const int jj = c / EX2_FRG_PAIRS;
            const int kk = 2 * (c % EX2_FRG_PAIRS);
            constexpr int EX2_FREQ = 16;
            const bool use_hw = (kk % EX2_FREQ < EX2_FREQ - EX2_RES) || (jj >= EX2_FRG_CNT - 1) || (jj < EX2_START_FRG);
            scores2[c] = use_hw ? make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y)) : ex2_emu_f32x2(a2.x, a2.y);
          } else {
            scores2[c] = make_float2(ex2_approx_f32(a2.x), ex2_approx_f32(a2.y));
          }
          p_regs[c] = cvt_f32x2_to_bf16x2(scores2[c].x, scores2[c].y);
        }
        const uint32_t p_tmem_addr = s_tmem_addr;
        wp_end(wpc, WP_SM_SOFTMAX);

        wp_begin(wpc, WP_SM_STORE_P);
        if constexpr (SPLIT_P) {
          tcgen05_st_32x32b_x32(p_tmem_addr, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
          tcgen05_st_32x32b_x16(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[32]));
          // fence orders the async STTM but does NOT wait; wait::st stops a fast consumer reading pre-store garbage.
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
          tcgen05_st_32x32b_x16(p_tmem_addr + SPLIT_P_COL, *reinterpret_cast<uint32_t(*)[16]>(&p_regs[SPLIT_P_COL]));
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&full_bar_p_last[m_tile]));
        } else {
          tcgen05_st_32x32b_x32(p_tmem_addr, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[0]));
          tcgen05_st_32x32b_x32(p_tmem_addr + 32, *reinterpret_cast<uint32_t(*)[32]>(&p_regs[32]));
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          mbarrier_arrive(smem_ptr_u32(&empty_bar_spo[m_tile]));
        }
        wp_end(wpc, WP_SM_STORE_P);
	spo_ph.advance();
        wp_begin(wpc, WP_SM_WAIT_SCALE);
        alpha_stage = (k + 1) & ALPHA_MASK;
        if constexpr (!MHA && !IS_CAUSAL) {
          DBGW2(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile * ALPHA_STAGES + alpha_stage]), (scale_ring_ph >> alpha_stage) & 1u, m_tile * ALPHA_STAGES + alpha_stage, k);
        } else {
          DBGW2(smem_ptr_u32(&empty_bar_alpha_and_l[m_tile * ALPHA_STAGES + alpha_stage]), (scale_ring_ph >> alpha_stage) & 1u, m_tile * ALPHA_STAGES + alpha_stage, k);
        }
        scale_ring_ph ^= (1u << alpha_stage);
        scale_empty_ph.advance();
        wp_end(wpc, WP_SM_WAIT_SCALE);
        alpha_slot = &alpha_and_l_smem[(m_tile * ALPHA_STAGES + alpha_stage) * M_TILE + row_in_m_tile];
        alpha_slot_u32 = smem_ptr_u32(alpha_slot);

        wp_begin(wpc, WP_SM_ROWSUM);
        // deferred update_row_sum. Intent: overlap this chain --
        // corr's O-rescale + wait_scale + this row-sum -- with BMM2 on the MMA warp.
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
        m_run = new_m;
        wp_end(wpc, WP_SM_ROWSUM);
      };
      softmax_step(std::true_type{}, std::true_type{}, 0);
      int k = 1;
      if constexpr (IS_CAUSAL) {
        const int m_tile_q_token = q_tile_base + m_tile * q_tile_per_mtile;
        const int masked_steps = K_TILES - (m_tile_q_token >> 7);
        for (; k < masked_steps; ++k) softmax_step(std::true_type{}, std::false_type{}, k);
      }
      for (; k < K_TILES; ++k) softmax_step(std::false_type{}, std::false_type{}, k);

      wp_begin(wpc, WP_SM_READ_L);
      if constexpr (MHA && !IS_CAUSAL) sts_f32(alpha_slot_u32, l_run);
      else                             *alpha_slot = l_run;
      if constexpr (FULL_NAMED_BAR) full_bar_arrive(m_tile, warp_in_group);
      else mbarrier_arrive(smem_ptr_u32(&full_bar_l[m_tile]));
      wp_end(wpc, WP_SM_READ_L);

      if constexpr (USE_CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, /*CTA_GROUP=*/1, /*SUSPEND=*/true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, /*do_release=*/elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        if (!next.valid) break;
        workitem_id = next.n_tile;
      } else {
        workitem_id += gridDim.x;
        if (workitem_id >= total_workitems) break;
      }
    }
  }
  wp_flush(wpc);
  __syncthreads();
  if (warp_id == 0) tcgen05_dealloc<1>(tmem_base, TMEM_TOTAL);
}

// ============================== driver ====================================

#include "fmha_cpu_ref.cuh"
#include "fmha_context_bf16_benchmark.cuh"

// Deterministic fill in [-1, 1) so host and device see identical inputs.
static void fillr(__nv_bfloat16* h, long n, unsigned seed) {
  const char* e = getenv("FILL");
  int mode = e ? atoi(e) : 2;
  for (long i = 0; i < n; ++i) {
    uint32_t x = (uint32_t)i * 2654435761u + seed;
    float v;
    switch (mode) {
      case 1: v = ((x % 256) / 256.0f) - 0.5f; break;
      case 3: v = 1.0f; break;
      case 4: v = (float)((int)(x % 7) - 3); break;
      default: v = (x % 2048) / 1024.0f - 1.0f; break;
    }
    h[i] = __float2bfloat16(v);
  }
}

// One benchmark shape. sl[s] = seqlen of sample s (all equal here); nqh/nkh =
// q/kv head counts; hd = head dim; causal toggles the mask; lab is for printing.
struct Sh {
  std::vector<int> sl;
  int  nqh, nkh, hd;
  bool causal;
  const char* lab;
};

template <bool MHA = false, bool IS_CAUSAL = false, bool LPT = false>
static double run(const Sh& sh, bool verify) {
  const int  ns     = (int)sh.sl.size();
  const int  seqlen = sh.sl[0];
  const long tq     = (long)ns * seqlen;      // total q-tokens
  const long tk     = tq;                     // total k-tokens (== tq here)

  // ---- device buffers (bf16; V stored transposed as V_T for the BMM2 TMA) ----
  __nv_bfloat16 *dQ, *dK, *dVT, *dO;
  CUDA_CHECK(cudaMalloc(&dQ,  tq * sh.nqh * sh.hd * 2));
  CUDA_CHECK(cudaMalloc(&dK,  tk * sh.nkh * sh.hd * 2));
  CUDA_CHECK(cudaMalloc(&dVT, (long)sh.nkh * sh.hd * tk * 2));
  CUDA_CHECK(cudaMalloc(&dO,  tq * sh.nqh * sh.hd * 2));

  // ---- host inputs + V -> V_T transpose ([tok,head,hd] -> [head,hd,tok]) ----
  std::vector<__nv_bfloat16> hQ(tq * sh.nqh * sh.hd),
                             hK(tk * sh.nkh * sh.hd),
                             hV(tk * sh.nkh * sh.hd);
  fillr(hQ.data(), hQ.size(), 11);
  fillr(hK.data(), hK.size(), 22);
  fillr(hV.data(), hV.size(), 33);

  std::vector<__nv_bfloat16> hVT((long)sh.nkh * sh.hd * tk, __float2bfloat16(0.f));
  for (long idx = 0; idx < tk; ++idx)
    for (int h = 0; h < sh.nkh; ++h)
      for (int d = 0; d < sh.hd; ++d)
        hVT[(h * sh.hd + d) * tk + idx] = hV[(idx * sh.nkh + h) * sh.hd + d];

  CUDA_CHECK(cudaMemcpy(dQ,  hQ.data(),  hQ.size()  * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK,  hK.data(),  hK.size()  * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dVT, hVT.data(), hVT.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dO, 0, hQ.size() * 2));

  // ---- TMA tensor maps ----
  // pack-GQA Q/O: 4D TMA over [hd, nqh, token-IN-SAMPLE, sample]; box [hd-subtile x
  // gqa_group_size x q_tile_per_mtile x 1] -> 128 packed rows (qh-inner). The per-sample token
  // dim makes the HW clamp the box at each sample's seqlen boundary: when seqlen is not a
  // multiple of q_tile_per_cta, the last packed M-tile's overrun rows would otherwise land in
  // the NEXT sample's tokens (a global-token 3D map stores them -- cross-sample clobber, racy).
  // With the 4D map the overrun rows read as zero-fill (Q) and are simply not written (O).
  const int gqa = sh.nqh / sh.nkh;            // q-heads per kv-head
  const int tpi = M_TILE / gqa;               // q-tokens per M-tile
  const int tpc = 2 * tpi;                    // q-tokens per CTA (2 M-tiles)
  CUtensorMap tq_, tk_, tvt_, to_;
  {
    uint64_t gd[4] = { (uint64_t)sh.hd, (uint64_t)sh.nqh, (uint64_t)seqlen, (uint64_t)ns };
    uint64_t gs[3] = { (uint64_t)sh.hd * 2u, (uint64_t)sh.nqh * sh.hd * 2u,
                       (uint64_t)seqlen * sh.nqh * sh.hd * 2u };
    uint32_t bd[4] = { (uint32_t)SUB_COLS_BF16, (uint32_t)gqa, (uint32_t)tpi, 1u };
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
  // K: ONE 3D TMA copy folds the 2 head-dim swizzle atoms (HEAD_DIM = 2 x SUB_COLS_BF16) into the box
  // (vs looping 2 x 2D copies). dims [atom-col SUB_COLS_BF16, token tk, atom (nkh*hd)/SUB_COLS_BF16]; box
  // [SUB_COLS_BF16, K_TILE, K_SUBTILES]; strides token=(nkh*hd)*2B, atom=SUB_COLS_BF16*2B. The box dim order
  // (atom outermost) reproduces the atom-outer smem layout the MMA reads (atom0 then atom1).
  {
    uint64_t gd[3] = { (uint64_t)SUB_COLS_BF16, (uint64_t)tk, (uint64_t)(sh.nkh * sh.hd / SUB_COLS_BF16) };
    uint64_t gs[2] = { (uint64_t)(sh.nkh * sh.hd) * 2u, (uint64_t)SUB_COLS_BF16 * 2u };
    uint32_t bd[3] = { (uint32_t)SUB_COLS_BF16, (uint32_t)K_TILE, (uint32_t)K_SUBTILES };
    uint32_t es[3] = { 1u, 1u, 1u };
    CUresult r = cuTensorMapEncodeTiled(&tk_, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, dK, gd, gs, bd, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUDA_CHECK(r == CUDA_SUCCESS ? cudaSuccess : cudaErrorInvalidValue);
  }
  CUDA_CHECK(make_tma_2d_tiled(&tvt_, dVT, (long)sh.nkh * sh.hd, tk, sh.hd, SUB_COLS_BF16, 2,
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B));

  // ---- shared memory budget ----
  const int packed_mtiles_per_seq = (seqlen + tpc - 1) / tpc;   // packed-M tiles per (sample, kv-head)
  // FastDivmod magics for decode_workitem's divides:
  //   magic0 = mtiles_per_sample (workitem_id -> sample), magic1 = mtiles_per_seq (rr -> kv_head),
  //   magic2 = num_kv_heads (swizzle path + non-Q_RASTER rr -> tile_index).
  const unsigned long long magic0 = make_magic((unsigned)(packed_mtiles_per_seq * sh.nkh));
  const unsigned long long magic1 = make_magic((unsigned)packed_mtiles_per_seq);
  const unsigned long long magic2 = make_magic((unsigned)sh.nkh);
  // FA4 SingleTileLPTScheduler params (causal): sections of 2^k kv-heads whose K+V fit L2.
  int lpt_swz_log2 = 0, lpt_hb_quot = 0, lpt_hb_rem = 1;
  unsigned long long lpt_major_magic = 1, lpt_rem_magic = 1;
  if constexpr (IS_CAUSAL) {
    const long kv_head_bytes = (long)seqlen * (sh.hd + sh.hd) * 2;   // K + V per kv-head
    const long size_l2 = 100L << 20;                                 // GB200 L2 ~126MB; leave headroom
    int swz = 1;
    while (((long)swz << 1) * kv_head_bytes <= size_l2) swz <<= 1;
    const int hb_total = ns * sh.nkh;
    while (swz > hb_total && swz > 1) swz >>= 1;                     // clamp to problem
    lpt_swz_log2   = 0; while ((1 << (lpt_swz_log2 + 1)) <= swz) ++lpt_swz_log2;
    lpt_hb_quot    = hb_total >> lpt_swz_log2;
    lpt_hb_rem     = hb_total - (lpt_hb_quot << lpt_swz_log2); if (lpt_hb_rem == 0) lpt_hb_rem = 1;
    lpt_major_magic = make_magic((unsigned)(packed_mtiles_per_seq << lpt_swz_log2));
    lpt_rem_magic   = make_magic((unsigned)lpt_hb_rem);
  }
  constexpr int ALPHA_STAGES = 1;   // softmax->corr alpha ring depth (1 = single slot)
  const size_t smem =
        (size_t)2 * Q_TILE_BYTES + NUM_KV_STAGES * K_TILE_BYTES  // Q (x2) + shared K/V ring
      + (size_t)2 * M_TILE * HEAD_DIM * sizeof(__nv_bfloat16)     // 2 sO bufs for TMA-O
      + (2 * NUM_KV_STAGES + 22 + 4 * (ALPHA_STAGES - 1)) * 8     // mbarriers (incl full/empty_bar_o_epi)
      + (size_t)CLC_STAGES * (2 * 8 + 16) + 16                // CLC: clc_full+clc_empty + response (16B aligned)
      + 8                                                         // tmem_slot
      + (size_t)2 * ALPHA_STAGES * M_TILE * sizeof(float)         // alpha_and_l_smem [2][ALPHA_STAGES][M_TILE]
      + 512;                                                      // slack / alignment + isolated wait_scale bar granule

  // Compile-time kernel config (see the knob docs near the top of this file).
  // FA4-matched config: static persistent sched (USE_CLC=false), softmax throttle,
  // ex2_emu, split_P, named-barrier scale handshake -- mirrors fa4_gen.py's GEN config.
  constexpr bool FULL_NAMED_BAR = (ALPHA_STAGES == 1), EX2_EMU = true, SPLIT_P = true,
                 SOFTMAX_THROTTLE = (ALPHA_STAGES == 1), Q_RASTER = true;
  // Named-bar + throttle restored after FULL softmax-body convergence to the 2SM's proven
  // form (paid-l over the bar, no first-step slot write, sticky max + -inf guard, deferred
  // row-sum, multi-diagonal causal masking). Gated: gqa-short F4 x8, mha-mid F4 stress 30x3,
  // 156-point matrix.
  // Named-bar safety (2SM protocol, ported 2026-07-11): the l publish rides the SAME
  // named bar as the alphas and the corr epilogue does the matching bar.sync -- arrives
  // and syncs are 1:1 per work item, every scale-slot release paid by a same-band sync,
  // so at most one arrive is ever outstanding on a bar id (the bare counter cannot be
  // double-arrived at item boundaries). Gated by STRESS_N=30 x3 at mha-mid FILL=4 +
  // gqa-short FILL=4 x8 + the 156-point verify matrix.
  // SOFTMAX_THROTTLE=false: the throttle's shifted release order desyncs the NAMED-BAR
  // alpha/l pairing at work-item boundaries (bare counter, no phase identity) -- corr
  // reads a one-slot-stale alpha/l stream; intermittent at many-items-per-CTA shapes
  // (gqa-short FILL=4: ~40% fail rate). mbarrier handshake with throttle is clean but
  // costs -13/-39/-26 (mha/gqa/causal); throttle-off costs -1/-10/-16 and is clean 5/5.
  // Scheduler per mask: causal = CLC + LPT (dynamic stealing over heaviest-first order --
  // variable K-loop lengths; +30 and +36 at row-6 FILL=3). Non-causal = static grid-stride
  // (uniform work; static beat CLC here).
  constexpr bool USE_CLC = false;   // PROBE: static + swizzle (FA4's causal config)
  auto kfn = &fmha_context_bf16_gen_kernel<32, FULL_NAMED_BAR, EX2_EMU, SPLIT_P, SOFTMAX_THROTTLE, USE_CLC, Q_RASTER, MHA, IS_CAUSAL, LPT, 8, ALPHA_STAGES>;
  CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));

  // ---- launch geometry: CLC persistent. Launch the FULL problem grid (one CTA per work tile);
  // clusterlaunchcontrol keeps only ~#SMs CTAs resident and hands the rest of the CTA-ids out via
  // try_cancel (HW work-stealing scheduler), so the grid-size is the tile count, not #SMs. ----
  const float scale_log2 = (1.0f / sqrtf((float)sh.hd)) * (float)M_LOG2E;
  int numSM = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0));
  const int total_workitems_host = ns * packed_mtiles_per_seq * sh.nkh;   // one CTA per (sample, packed-M tile, kv-head)
  const int nblk = USE_CLC ? total_workitems_host : std::min(total_workitems_host, numSM);
  (void)numSM;
  dim3 grid(nblk, 1, 1), block(N_WARPS * 32, 1, 1);

  // CLC must be launched via cudaLaunchKernelEx with a cluster-dimension attribute -- a plain
  // <<<grid,block>>> launch does NOT enable clusterlaunchcontrol (try_cancel silently misbehaves
  // and tiles get skipped). cluster {1,1,1} (matches __cluster_dims__(1,1,1)); no PSS attribute
  // (we don't drive griddepcontrol, so leave the dependent-launch serialization off).
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = 0;
  cudaLaunchAttribute cfgAttr[1];
  cfgAttr[0].id = cudaLaunchAttributeClusterDimension;
  cfgAttr[0].val.clusterDim.x = 1;
  cfgAttr[0].val.clusterDim.y = 1;
  cfgAttr[0].val.clusterDim.z = 1;
  cfg.attrs = cfgAttr;
  cfg.numAttrs = 1;
  auto launch = [&](cudaStream_t st = 0) {
    cfg.stream = st;
    if (USE_CLC)
      return cudaLaunchKernelEx(&cfg, kfn, tq_, tk_, tvt_, to_, seqlen, sh.nqh, sh.nkh,
                                scale_log2, packed_mtiles_per_seq, ns, magic0, magic1, magic2,
                                lpt_swz_log2, lpt_hb_quot, lpt_hb_rem, lpt_major_magic, lpt_rem_magic);
    kfn<<<grid, block, smem, st>>>(tq_, tk_, tvt_, to_, seqlen, sh.nqh, sh.nkh,
                               scale_log2, packed_mtiles_per_seq, ns, magic0, magic1, magic2,
                               lpt_swz_log2, lpt_hb_quot, lpt_hb_rem, lpt_major_magic, lpt_rem_magic);
    return cudaGetLastError();
  };

  const double ms = fmha_context_bf16_benchmark::measure(launch);
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
    wp_dump_raw(wp, "warp_raw_fmha_pgen.bin.gz", pblk, NUM_KV_STAGES);
    wp_free(wp);
  }
#endif

  const uint64_t eff = (uint64_t)ns * fmha_context_bf16_benchmark::attended_pairs(
      seqlen, seqlen, sh.causal);
  const double tflops = fmha_context_bf16_benchmark::report(
      sh.lab, tq, sh.causal, sh.hd, sh.nqh, eff, ms);

  // ---- correctness check against the fp32 CPU reference ----
  if (verify) {
    std::vector<__nv_bfloat16> ho(hQ.size());
    CUDA_CHECK(cudaMemcpy(ho.data(), dO, ho.size() * 2, cudaMemcpyDeviceToHost));

    std::vector<int> cq(ns + 1, 0);
    for (int i = 0; i < ns; ++i) cq[i + 1] = cq[i] + seqlen;

    std::vector<float> ref(hQ.size(), 0.f), out(hQ.size());
    {   // disk-cached CPU reference (pure function of shape+fill; key bumps if fills change)
      const char* fe = getenv("FILL");
      char key[128];
      snprintf(key, sizeof key, "B%d_S%d_hq%d_hk%d_hd%d_c%d_f%d",
               ns, seqlen, sh.nqh, sh.nkh, sh.hd, (int)sh.causal, fe ? atoi(fe) : 2);
      cached_ref_f32(key, ref.data(), ref.size(), [&] {
        cpu_fmha_ref(hQ.data(), hK.data(), hV.data(), ref.data(),
                     cq, cq, sh.nqh, sh.nkh, sh.hd, sh.causal);
      });
    }
    for (size_t i = 0; i < out.size(); ++i) out[i] = __bfloat162float(ho[i]);

    if (const char* dp = getenv("DUMP_O")) {   // debug: dump ref+out f32 for offline analysis
      std::string base(dp);
      FILE* f1 = fopen((base + ".ref").c_str(), "wb"); fwrite(ref.data(), 4, ref.size(), f1); fclose(f1);
      FILE* f2 = fopen((base + ".out").c_str(), "wb"); fwrite(out.data(), 4, out.size(), f2); fclose(f2);
    }
    const bool ok = check_close_f32(ref.data(), out.data(), (int)out.size(), 0.05f, 0.10f);
    printf("  verify [%s]: %s\n", sh.lab, ok ? "OK" : "FAIL");
  }

  // ---- stress mode: STRESS_N kernel launches, cross-run output consistency ----
  // Catches intermittent races (output drifts from run 0) and per-launch CUDA errors,
  // WITHOUT paying the CPU fp32 verify each time -- the single verify above (deterministic
  // inputs) already certifies run 0; here we just memcmp every run's O against run 0.
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

  cudaFree(dQ);
  cudaFree(dK);
  cudaFree(dVT);
  cudaFree(dO);
  return tflops;
}

int main() {
  CUDA_CHECK(cudaFree(0));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 64 * 1024 * 1024));
  printf("K2 fmha_context_bf16 GEN (warp-spec, 2 M-tiles) sm_100a\n"
         "=====================================\n");

  Sh s{};
  const int B = getenv("BATCH")  ? atoi(getenv("BATCH"))  : 128;
  const int S = getenv("SEQLEN") ? atoi(getenv("SEQLEN")) : 240;
  const int HQ = getenv("HEADS") ? atoi(getenv("HEADS")) : 32;
  // MHA=1 -> HK==HQ (gqa_group=1); default GQA -> HK=4 (gqa_group=HQ/4=8 at HQ=32).
  const bool mha = getenv("MHA") ? (atoi(getenv("MHA")) != 0) : false;
  // CAUSAL=1 -> triangular mask (uniform seqlen; every query still emits a full row, so the
  // O store stays non-ragged and the TMA-O epilogue is unchanged). Varlen lives in its own file.
  const bool causal = getenv("CAUSAL") ? (atoi(getenv("CAUSAL")) != 0) : false;
  s.sl     = std::vector<int>(B, S);
  s.nqh    = HQ;
  s.nkh    = mha ? HQ : 4;
  s.hd     = 128;
  s.causal = causal;
  s.lab    = causal ? (mha ? "mha-causal" : "causal") : (mha ? "mha" : "gen");
  // CPU fp32 reference is O(B*HQ*S^2*D) -- infeasible at long S; verify only for small S
  // (override with NOVERIFY=1).
  const bool verify = getenv("NOVERIFY") ? false : (S <= 1024 || getenv("VERIFY") != nullptr);
  constexpr bool LPT = true;    // heaviest-first causal balance (FA4 lpt=is_causal); the old
                                // 'neutral' verdict was measured under CLC -- static sched needs it
  if      (mha && causal) run</*MHA=*/true,  /*IS_CAUSAL=*/true,  /*LPT=*/LPT  >(s, verify);
  else if (mha)           run</*MHA=*/true,  /*IS_CAUSAL=*/false, /*LPT=*/false>(s, verify);
  else if (causal)        run</*MHA=*/false, /*IS_CAUSAL=*/true,  /*LPT=*/LPT  >(s, verify);
  else                    run</*MHA=*/false, /*IS_CAUSAL=*/false, /*LPT=*/false>(s, verify);
  return 0;
}

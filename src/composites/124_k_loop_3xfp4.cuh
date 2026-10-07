#pragma once
#if defined(PL_AGENTIC_SM103A)
// 124_k_loop_3xfp4.cuh -- 8-phase K-loop for block-scaled FP4 with K=96 atom (3xFP4 pattern)
//
// ARCH: sm_103a
//
// One K-tile holds 768 FP4 elements -> 8 MMA phases (768 / 96). The
// loop interleaves UTCCP (tcgen05.cp SMEM -> TMEM) of scale factors on
// even phases with MMA issues on every phase so the copy hides behind
// the next MMA.
//
//   even phases (0, 2, 4, 6) : tcgen05.cp SF_A + SF_B -> TMEM, then MMA
//   odd phases  (1, 3, 5, 7) : MMA only
//
// Pipeline depths: AB operand = 6 stages, scale-factor = 8 stages (the
// 3-block circular SMEM, see file 17), accumulator = 2 (ping-pong).
//
// Composes file 6 (K=96 MMA wrapper), file 17 (circular SMEM desc), and
// shared primitives 11 (tcgen05_commit/_multicast), 13 (tcgen05_cp_4x256b),
// 15 (tcgen05_fence_before_thread_sync), 33 (mbarrier_wait_parity).
// The body is gated by __has_include on those four headers so the file
// degrades to an empty include when they are missing.
//
// Issuer: a single MMA-warp thread per CTA (cta_group::1) or per CTA-pair
// (cta_group::2). Caller supplies pre-built SMEM descriptors via file 17
// for both AB and scale-factor sides plus the K=96 idesc from file 6.
// PTX:    9.7.18.10.10.1 (block_scale K=96), 9.7.18.10.7.2.4 (block32 K=96 SF A), 9.7.18.10.7.3.4 (block32 K=96 SF B), 9.7.18.4.1 (absolute desc)
//
#include "../primitives/_common.cuh"
#include "../primitives/6_tcgen05_mma_fp4_k96.cuh"
#include "../primitives/17_smem_desc_circular.cuh"

#if __has_include("../primitives/11_tcgen05_commit.cuh") && \
    __has_include("../primitives/13_tcgen05_cp.cuh") && \
    __has_include("../primitives/15_tcgen05_fence.cuh") && \
    __has_include("../primitives/33_mbarrier_try_wait.cuh")
  #include "../primitives/11_tcgen05_commit.cuh"
  #include "../primitives/13_tcgen05_cp.cuh"
  #include "../primitives/15_tcgen05_fence.cuh"
  #include "../primitives/33_mbarrier_try_wait.cuh"
  #define K_LOOP_3XFP4_DEPS_AVAILABLE 1
#else
  #define K_LOOP_3XFP4_DEPS_AVAILABLE 0
#endif

constexpr int K_LOOP_3XFP4_MMA_PHASES_PER_TILE = 8;
constexpr int K_LOOP_3XFP4_AB_STAGES = 6;
constexpr int K_LOOP_3XFP4_SF_STAGES = 8;
constexpr int K_LOOP_3XFP4_ACC_STAGES = 2;

#if K_LOOP_3XFP4_DEPS_AVAILABLE

// State carried across iterations of the outer K-tile loop. Caller owns
// these; we update them in place.
struct KLoop3xFp4State {
    int full_phase;       // mbarrier parity for full_mbar[]
    int sf_phase;         // mbarrier parity for sf_full_mbar[]
    int ab_stage;         // current AB-operand stage in [0, K_LOOP_3XFP4_AB_STAGES)
    int sf_stage;         // current scale-factor stage in [0, K_LOOP_3XFP4_SF_STAGES)
    bool first_tile;      // true on the first call -> enable_input_d=false
};

// Internal helper: emit a (multicast or plain) tcgen05.commit for ctagroup 1 or 2.
template <int CtaGroup>
__device__ __forceinline__ void k_loop_3xfp4_commit_(uint32_t mbar_smem,
                                                     uint16_t ctamask) {
    if constexpr (CtaGroup == 2) {
        tcgen05_commit_multicast<CtaGroup>(mbar_smem, ctamask);
    } else {
        tcgen05_commit<CtaGroup>(mbar_smem);
    }
}

// Issue one K-tile's worth of work: 8 MMA phases plus UTCCP scale-factor
// copies on even phases. Caller must have invoked the producer side
// (TMA load + scale-factor stage) so that full_mbar[ab_stage] and
// sf_full_mbar[sf_stage] are signalled for this tile.
//
//   tmem_acc          : 32-bit TMEM addr of accumulator (depth-2 ping-pong)
//   sf_a_tmem_base    : 32-bit TMEM addr of scale-factor A region
//   sf_b_tmem_base    : 32-bit TMEM addr of scale-factor B region
//   a_descs[3]        : pre-built circular SMEM descs for A (file 17)
//   b_descs[3]        : pre-built circular SMEM descs for B
//   sf_a_descs[3]     : pre-built 64-bit SMEM descs for scale_A (3-block ring;
//                       see file 17 for the circular descriptor builder)
//   sf_b_descs[3]     : pre-built SMEM descs for scale_B
//   idesc             : K=96 idesc (file 6: build_idesc_mxf4_k96)
//   full_mbar[stage]  : per-AB-stage mbarrier (uint64_t in SMEM) signalled by producer
//   sf_full_mbar[s]   : per-SF-stage mbarrier signalled by SF producer
//   acc_done_mbar     : mbarrier to signal when this K-tile's MMAs all retire
//   ctamask           : 16-bit cluster mask for cta_group::2 commit (.cluster
//                       scope); ignored on cta_group::1
//   state             : updated in place
template <int CtaGroup, MmaMxf4Variant V>
__device__ __forceinline__ void k_loop_3xfp4_one_tile(
    uint32_t tmem_acc,
    uint32_t sf_a_tmem_base,
    uint32_t sf_b_tmem_base,
    uint64_t const a_descs[3],
    uint64_t const b_descs[3],
    uint64_t const sf_a_descs[3],
    uint64_t const sf_b_descs[3],
    uint32_t idesc,
    uint64_t* full_mbar,
    uint64_t* empty_mbar,
    uint64_t* sf_full_mbar,
    uint64_t* sf_empty_mbar,
    uint64_t* acc_done_mbar,
    uint16_t ctamask,
    KLoop3xFp4State& state)
{
    // Wait for AB operands for this stage. Main's mbarrier API takes a
    // 32-bit .shared address; convert from the generic SMEM pointer.
    uint32_t full_addr = cvta_to_shared_u32(&full_mbar[state.ab_stage]);
    mbarrier_wait_parity(full_addr, (uint32_t)state.full_phase);

    // Wait for the scale-factor stage (depth 8, drives all 8 phases).
    uint32_t sf_full_addr = cvta_to_shared_u32(&sf_full_mbar[state.sf_stage]);
    mbarrier_wait_parity(sf_full_addr, (uint32_t)state.sf_phase);

    // Order pending tcgen05 ops (in case the prior K-tile's commit raced).
    tcgen05_fence_before_thread_sync();

    bool enable_d = !state.first_tile;
    state.first_tile = false;

    #pragma unroll
    for (int phase = 0; phase < K_LOOP_3XFP4_MMA_PHASES_PER_TILE; ++phase) {
        // Even phases: kick off UTCCP for the next scale-factor block.
        if ((phase & 1) == 0) {
            int blk = phase / 2;          // 4 SF blocks per K-tile, but we use 3-block circular
            int desc_idx = blk % 3;
            tcgen05_cp_4x256b<CtaGroup>(sf_a_tmem_base + (uint32_t)blk * 32u,
                                        sf_a_descs[desc_idx]);
            tcgen05_cp_4x256b<CtaGroup>(sf_b_tmem_base + (uint32_t)blk * 32u,
                                        sf_b_descs[desc_idx]);
        }

        // MMA against the circular descriptor for this phase.
        int desc_idx = circular_phase_to_desc_index(phase);
        tcgen05_mma_fp4_k96<CtaGroup, V>(
            tmem_acc,
            a_descs[desc_idx],
            b_descs[desc_idx],
            idesc,
            sf_a_tmem_base,
            sf_b_tmem_base,
            enable_d);
        enable_d = true;
    }

    // Signal the empty mbarriers so the producer can refill this stage.
    uint32_t empty_addr    = cvta_to_shared_u32(&empty_mbar[state.ab_stage]);
    uint32_t sf_empty_addr = cvta_to_shared_u32(&sf_empty_mbar[state.sf_stage]);
    k_loop_3xfp4_commit_<CtaGroup>(empty_addr,    ctamask);
    k_loop_3xfp4_commit_<CtaGroup>(sf_empty_addr, ctamask);

    // Advance ring indices.
    state.ab_stage = (state.ab_stage + 1) % K_LOOP_3XFP4_AB_STAGES;
    if (state.ab_stage == 0) state.full_phase ^= 1;
    state.sf_stage = (state.sf_stage + 1) % K_LOOP_3XFP4_SF_STAGES;
    if (state.sf_stage == 0) state.sf_phase ^= 1;
}

// Drive the full K dimension: iterate `num_k_tiles` calls of one_tile and
// emit a final commit that signals `acc_done_mbar` once all MMAs retire.
template <int CtaGroup, MmaMxf4Variant V>
__device__ __forceinline__ void k_loop_3xfp4(
    uint32_t tmem_acc,
    uint32_t sf_a_tmem_base,
    uint32_t sf_b_tmem_base,
    uint64_t const a_descs[3],
    uint64_t const b_descs[3],
    uint64_t const sf_a_descs[3],
    uint64_t const sf_b_descs[3],
    uint32_t idesc,
    uint64_t* full_mbar,
    uint64_t* empty_mbar,
    uint64_t* sf_full_mbar,
    uint64_t* sf_empty_mbar,
    uint64_t* acc_done_mbar,
    uint16_t ctamask,
    int num_k_tiles)
{
    KLoop3xFp4State state{0, 0, 0, 0, true};
    for (int t = 0; t < num_k_tiles; ++t) {
        k_loop_3xfp4_one_tile<CtaGroup, V>(
            tmem_acc, sf_a_tmem_base, sf_b_tmem_base,
            a_descs, b_descs, sf_a_descs, sf_b_descs,
            idesc, full_mbar, empty_mbar, sf_full_mbar, sf_empty_mbar,
            acc_done_mbar, ctamask, state);
    }
    // Final accumulator-ready commit.
    uint32_t acc_done_addr = cvta_to_shared_u32(acc_done_mbar);
    k_loop_3xfp4_commit_<CtaGroup>(acc_done_addr, ctamask);
}

#endif  // K_LOOP_3XFP4_DEPS_AVAILABLE

#endif  // PL_AGENTIC_SM103A

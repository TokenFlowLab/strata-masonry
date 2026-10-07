#pragma once
#if defined(PL_AGENTIC_SM103A)
// 91_mma_warp_3xfp4.cuh -- MMA warp role for the SM103a 3xFP4 pattern.
//
// ARCH: sm_103a
//
// Header-only block. The MMA-warp body for a warp-specialized FP4 GEMM
// on GB300:
//   - tcgen05.alloc (warp-collective TMEM allocation)
//   - tcgen05.relinquish_alloc_permit
//   - K-loop: per tile -> wait full_mbar -> MMA x 8 phases (with SF cp)
//             -> tcgen05.commit (or .multicast for cta_group::2)
//             -> arrive on empty_mbar so producer can refill
//   - barrier.cluster sync (cta_group::2 only) before dealloc
//   - tcgen05.dealloc
//
// Composes file 6 (K=96 MMA), file 17 (circular SMEM desc), composite
// 75 (K-loop), plus shared primitives 0 (alloc), 1 (dealloc),
// 2 (relinquish), 38 (barrier_cluster).
//
// Block function (per code/PLAN.md "Block function signature contract"):
//
//   template <int CtaGroup, MmaMxf4Variant V, int M, int N>
//   __device__ __forceinline__ void
//   mma_warp_3xfp4_block(int num_k_tiles,
//                        uint32_t* tmem_alloc_dst,
//                        uint64_t* full_mbar, uint64_t* empty_mbar,
//                        uint64_t* sf_full_mbar, uint64_t* sf_empty_mbar,
//                        uint64_t* acc_done_mbar,
//                        uint32_t* a_buf_smem, uint32_t* b_buf_smem,
//                        uint32_t* sf_a_smem, uint32_t* sf_b_smem,
//                        int stride_bytes, uint16_t ctamask);
//
// Caller-collective on the 32-thread MMA warp. Caller is responsible for
// the producer + epilogue sides; test 91 wires synthetic mbarriers and
// uses a `dont_run` host flag in its __global__ wrapper to instantiate
// the kernel without engaging the K-loop.
//
// Source: knowledge/building_blocks/mma_warp.md
// PTX:    9.7.18.10.10.1 (block_scale K=96), 9.7.18.10.7.2.4 (block32 K=96 SF A),
//         9.7.18.10.7.3.4 (block32 K=96 SF B), 9.7.18.4.1 (absolute desc)

#include "../primitives/_common.cuh"
#include "../primitives/0_tcgen05_alloc.cuh"
#include "../primitives/1_tcgen05_dealloc.cuh"
#include "../primitives/2_tcgen05_relinquish.cuh"
#include "../primitives/6_tcgen05_mma_fp4_k96.cuh"
#include "../primitives/17_smem_desc_circular.cuh"
#include "../primitives/38_barrier_cluster.cuh"
#include "../composites/124_k_loop_3xfp4.cuh"

#if K_LOOP_3XFP4_DEPS_AVAILABLE

template <int CtaGroup, MmaMxf4Variant V, int M, int N>
__device__ __forceinline__
void mma_warp_3xfp4_block(
    int num_k_tiles,
    uint32_t* tmem_alloc_dst,
    uint64_t* full_mbar,
    uint64_t* empty_mbar,
    uint64_t* sf_full_mbar,
    uint64_t* sf_empty_mbar,
    uint64_t* acc_done_mbar,
    uint32_t* a_buf_smem,
    uint32_t* b_buf_smem,
    uint32_t* sf_a_smem,
    uint32_t* sf_b_smem,
    int       stride_bytes,
    uint16_t  ctamask) {
  extern __shared__ uint8_t smem[];
  uint32_t* tmem_addr_ptr = reinterpret_cast<uint32_t*>(smem);

  if (lane_id() == 0) *tmem_addr_ptr = 0;
  __syncwarp();
  uint32_t tmem_addr_smem = cvta_to_shared_u32(tmem_addr_ptr);
  tcgen05_alloc<CtaGroup>(tmem_addr_smem, 256);
  tcgen05_relinquish_alloc_permit<CtaGroup>();
  uint32_t tmem_acc = *tmem_addr_ptr;
  uint32_t sf_a_tmem = tmem_acc + (256u - 32u);
  uint32_t sf_b_tmem = tmem_acc + (256u - 16u);

  CircularSmemBuffers a_bufs  {{a_buf_smem[0], a_buf_smem[1], a_buf_smem[2]}, stride_bytes};
  CircularSmemBuffers b_bufs  {{b_buf_smem[0], b_buf_smem[1], b_buf_smem[2]}, stride_bytes};
  CircularSmemBuffers sfa_bufs{{sf_a_smem[0],  sf_a_smem[1],  sf_a_smem[2]},  stride_bytes};
  CircularSmemBuffers sfb_bufs{{sf_b_smem[0],  sf_b_smem[1],  sf_b_smem[2]},  stride_bytes};
  uint64_t a_descs[3], b_descs[3], sf_a_descs[3], sf_b_descs[3];
  build_circular_smem_desc_set(a_bufs,   a_descs);
  build_circular_smem_desc_set(b_bufs,   b_descs);
  build_circular_smem_desc_set(sfa_bufs, sf_a_descs);
  build_circular_smem_desc_set(sfb_bufs, sf_b_descs);

  uint32_t idesc = build_idesc_mxf4_k96<M, N>(/*sf_a_id=*/0, /*sf_b_id=*/0,
                                              /*ue8m0=*/true);

  k_loop_3xfp4<CtaGroup, V>(
      tmem_acc, sf_a_tmem, sf_b_tmem,
      a_descs, b_descs, sf_a_descs, sf_b_descs, idesc,
      full_mbar, empty_mbar, sf_full_mbar, sf_empty_mbar,
      acc_done_mbar, ctamask, num_k_tiles);

  if constexpr (CtaGroup == 2) {
    barrier_cluster_arrive();
    barrier_cluster_wait();
  }
  tcgen05_dealloc<CtaGroup>(tmem_acc, 256);
}

#endif  // K_LOOP_3XFP4_DEPS_AVAILABLE

#endif  // PL_AGENTIC_SM103A

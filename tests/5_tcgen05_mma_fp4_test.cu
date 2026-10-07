#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 5_tcgen05_mma_fp4_test.cu -- compile + smoke test for tcgen05.mma.kind::
// {mxf4, mxf4nvf4} (block-scaled FP4).
//
// Verifies (1) the mxf4 / mxf4nvf4 SS wrappers compile (block_scale +
// .block16/.block32 modifiers, scale-A and scale-B TMEM operands, idesc
// from Tables 55), and (2) a kernel allocating TMEM, staging zero
// scale factors, issuing the MMA with zero descriptors, committing,
// waiting, and deallocating completes without hang. End-to-end FP4 GEMM
// correctness lives at the block level (#90, #100, #91 for the K=96
// 3xFP4 variant). Cross-block-size and TS variants added in e40352e are
// separately exercised by _extended_coverage_probe_test.
//
// PTX sniff: `cuobjdump --dump-ptx build/5_tcgen05_mma_fp4_test |
// grep -E 'tcgen05.mma.cta_group::1.kind::mxf4(nvf4)?'` should show the
// issues.

#include "test_utils.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include "../src/primitives/0_tcgen05_alloc.cuh"
#include "../src/primitives/1_tcgen05_dealloc.cuh"
#include "../src/primitives/2_tcgen05_relinquish.cuh"
#include "../src/primitives/5_tcgen05_mma_fp4.cuh"
#include "../src/primitives/8_tcgen05_mma_idesc.cuh"
#include "../src/primitives/11_tcgen05_commit.cuh"

__global__ void k_fp4() {
  __shared__ __align__(16)   uint32_t tmem_base_slot;
  __shared__ __align__(16)   uint64_t mbar;
  __shared__ __align__(128)  uint8_t  smA[128];
  __shared__ __align__(128)  uint8_t  smB[128];
  if (threadIdx.x == 0) {
    tmem_base_slot = 0;
    mbarrier_init_helper(smem_ptr_u32(&mbar), 1);
    for (int i = 0; i < 128; ++i) { smA[i] = 0; smB[i] = 0; }
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&tmem_base_slot), 128);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  uint32_t tmem_base = tmem_base_slot;
  if (threadIdx.x == 0) {
    uint64_t da = 0; da |= static_cast<uint64_t>(1) << 46;
    uint64_t db = 0; db |= static_cast<uint64_t>(1) << 46;
    uint32_t idesc = make_idesc_mxf4(128, 8);
    tcgen05_mma_mxf4_ss_1sm_block32(tmem_base, da, db, idesc,
                                     tmem_base, tmem_base, false);
    tcgen05_commit<1>(smem_ptr_u32(&mbar));
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tmem_base, 128);
}

// Sparse smoke: exercises all 5 .sp 1SM wrappers added for #5
// (mxf4 SS+TS .block32, mxf4nvf4 SS .block16+.block32, mxf4nvf4 TS .block16).
// Per PTX 9.7.18.10.9.4 the sparsity selector is ASSUMED 0 for these kinds.
// Block-scaled forms drop {disable-output-lane}; new operand order is
// [d], a-{desc|tmem}, b-desc, [sp-meta], idesc, [scale-A], [scale-B], p.
enum class SparseFp4Form {
  MXF4_SS_B32, MXF4_TS_B32,
  MXF4NVF4_SS_B16, MXF4NVF4_SS_B32,
  MXF4NVF4_TS_B16
};

template <SparseFp4Form V>
__global__ void k_fp4_sparse() {
  __shared__ __align__(16)   uint32_t tmem_base_slot;
  __shared__ __align__(16)   uint64_t mbar;
  __shared__ __align__(128)  uint8_t  smA[128];
  __shared__ __align__(128)  uint8_t  smB[128];
  if (threadIdx.x == 0) {
    tmem_base_slot = 0;
    mbarrier_init_helper(smem_ptr_u32(&mbar), 1);
    for (int i = 0; i < 128; ++i) { smA[i] = 0; smB[i] = 0; }
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&tmem_base_slot), 128);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  uint32_t tmem_base = tmem_base_slot;
  uint32_t sp_meta   = tmem_base + 120;
  uint32_t scale_a   = tmem_base + 116;
  uint32_t scale_b   = tmem_base + 117;
  uint32_t tmem_a_op = tmem_base + 64;

  if (threadIdx.x < 32) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n"
                 :: "r"(sp_meta), "r"(0x44444444u));
  }
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::);

  if (threadIdx.x == 0) {
    uint64_t da = 0; da |= static_cast<uint64_t>(1) << 46;
    uint64_t db = 0; db |= static_cast<uint64_t>(1) << 46;
    if constexpr (V == SparseFp4Form::MXF4_SS_B32) {
      uint32_t idesc = make_idesc_mxf4(128, 8);
      idesc |= 1u << 2;  // sparsity bit
      tcgen05_mma_mxf4_ss_1sm_block32_sparse(tmem_base, da, db, sp_meta, idesc,
                                              scale_a, scale_b, false);
    } else if constexpr (V == SparseFp4Form::MXF4_TS_B32) {
      uint32_t idesc = make_idesc_mxf4(128, 8);
      idesc |= 1u << 2;
      tcgen05_mma_mxf4_ts_1sm_block32_sparse(tmem_base, tmem_a_op, db, sp_meta,
                                              idesc, scale_a, scale_b, false);
    } else if constexpr (V == SparseFp4Form::MXF4NVF4_SS_B16) {
      uint32_t idesc = make_idesc_mxf4nvf4(128, 8, false, false, false, false,
                                            0, 0, /*ue8m0*/false,
                                            /*sparse*/true);
      tcgen05_mma_mxf4nvf4_ss_1sm_block16_sparse(tmem_base, da, db, sp_meta,
                                                 idesc, scale_a, scale_b, false);
    } else if constexpr (V == SparseFp4Form::MXF4NVF4_SS_B32) {
      uint32_t idesc = make_idesc_mxf4nvf4(128, 8, false, false, false, false,
                                            0, 0, /*ue8m0*/true,
                                            /*sparse*/true);
      tcgen05_mma_mxf4nvf4_ss_1sm_block32_sparse(tmem_base, da, db, sp_meta,
                                                 idesc, scale_a, scale_b, false);
    } else {  // MXF4NVF4_TS_B16
      uint32_t idesc = make_idesc_mxf4nvf4(128, 8, false, false, false, false,
                                            0, 0, /*ue8m0*/false,
                                            /*sparse*/true);
      tcgen05_mma_mxf4nvf4_ts_1sm_block16_sparse(tmem_base, tmem_a_op, db,
                                                 sp_meta, idesc, scale_a,
                                                 scale_b, false);
    }
    tcgen05_commit<1>(smem_ptr_u32(&mbar));
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tmem_base, 128);
}

// Non-sparse extra-forms smoke (round 4c probe-graduation): mirror the
// sparse template structure, exercises the 5 non-sparse 1SM block-scaled
// wrappers currently probe-only.
enum class Fp4ExtraForm {
  MXF4_SS_B32, MXF4_TS_B32,
  MXF4NVF4_SS_B16, MXF4NVF4_SS_B32,
  MXF4NVF4_TS_B16
};

template <Fp4ExtraForm V>
__global__ void k_fp4_extra() {
  __shared__ __align__(16)   uint32_t tmem_base_slot;
  __shared__ __align__(16)   uint64_t mbar;
  __shared__ __align__(128)  uint8_t  smA[128];
  __shared__ __align__(128)  uint8_t  smB[128];
  if (threadIdx.x == 0) {
    tmem_base_slot = 0;
    mbarrier_init_helper(smem_ptr_u32(&mbar), 1);
    for (int i = 0; i < 128; ++i) { smA[i] = 0; smB[i] = 0; }
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&tmem_base_slot), 128);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  uint32_t tmem_base = tmem_base_slot;
  uint32_t scale_a   = tmem_base + 116;
  uint32_t scale_b   = tmem_base + 117;
  uint32_t tmem_a_op = tmem_base + 64;

  if (threadIdx.x == 0) {
    uint64_t da = 0; da |= static_cast<uint64_t>(1) << 46;
    uint64_t db = 0; db |= static_cast<uint64_t>(1) << 46;
    if constexpr (V == Fp4ExtraForm::MXF4_SS_B32) {
      uint32_t idesc = make_idesc_mxf4(128, 8);
      tcgen05_mma_mxf4_ss_1sm_block32(tmem_base, da, db, idesc,
                                       scale_a, scale_b, false);
    } else if constexpr (V == Fp4ExtraForm::MXF4_TS_B32) {
      uint32_t idesc = make_idesc_mxf4(128, 8);
      tcgen05_mma_mxf4_ts_1sm_block32(tmem_base, tmem_a_op, db, idesc,
                                       scale_a, scale_b, false);
    } else if constexpr (V == Fp4ExtraForm::MXF4NVF4_SS_B16) {
      uint32_t idesc = make_idesc_mxf4nvf4(128, 8, false, false, false, false,
                                            0, 0, /*ue8m0*/false);
      tcgen05_mma_mxf4nvf4_ss_1sm_block16(tmem_base, da, db, idesc,
                                           scale_a, scale_b, false);
    } else if constexpr (V == Fp4ExtraForm::MXF4NVF4_SS_B32) {
      uint32_t idesc = make_idesc_mxf4nvf4(128, 8, false, false, false, false,
                                            0, 0, /*ue8m0*/true);
      tcgen05_mma_mxf4nvf4_ss_1sm_block32(tmem_base, da, db, idesc,
                                           scale_a, scale_b, false);
    } else {  // MXF4NVF4_TS_B16
      uint32_t idesc = make_idesc_mxf4nvf4(128, 8, false, false, false, false,
                                            0, 0, /*ue8m0*/false);
      tcgen05_mma_mxf4nvf4_ts_1sm_block16(tmem_base, tmem_a_op, db, idesc,
                                           scale_a, scale_b, false);
    }
    tcgen05_commit<1>(smem_ptr_u32(&mbar));
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tmem_base, 128);
}

int main() {
  auto run = [](const char* lbl, cudaError_t e) {
    printf("%-58s : %s\n", lbl, e == cudaSuccess ? "no hang" : cudaGetErrorString(e));
    (void)cudaGetLastError();
  };

  k_fp4<<<1, 128>>>();
  run("tcgen05.mma.kind::mxf4 smoke", cudaDeviceSynchronize());

  k_fp4_extra<Fp4ExtraForm::MXF4_SS_B32><<<1, 128>>>();
  run("tcgen05.mma.kind::mxf4 (.block32, SS, 1SM)", cudaDeviceSynchronize());
  k_fp4_extra<Fp4ExtraForm::MXF4_TS_B32><<<1, 128>>>();
  run("tcgen05.mma.kind::mxf4 (.block32, TS, 1SM)", cudaDeviceSynchronize());
  k_fp4_extra<Fp4ExtraForm::MXF4NVF4_SS_B16><<<1, 128>>>();
  run("tcgen05.mma.kind::mxf4nvf4 (.block16, SS, 1SM)", cudaDeviceSynchronize());
  k_fp4_extra<Fp4ExtraForm::MXF4NVF4_SS_B32><<<1, 128>>>();
  run("tcgen05.mma.kind::mxf4nvf4 (.block32, SS, 1SM)", cudaDeviceSynchronize());
  k_fp4_extra<Fp4ExtraForm::MXF4NVF4_TS_B16><<<1, 128>>>();
  run("tcgen05.mma.kind::mxf4nvf4 (.block16, TS, 1SM)", cudaDeviceSynchronize());

  k_fp4_sparse<SparseFp4Form::MXF4_SS_B32><<<1, 128>>>();
  run("tcgen05.mma.sp.kind::mxf4 (.block32, SS, 1SM)", cudaDeviceSynchronize());
  k_fp4_sparse<SparseFp4Form::MXF4_TS_B32><<<1, 128>>>();
  run("tcgen05.mma.sp.kind::mxf4 (.block32, TS, 1SM)", cudaDeviceSynchronize());
  k_fp4_sparse<SparseFp4Form::MXF4NVF4_SS_B16><<<1, 128>>>();
  run("tcgen05.mma.sp.kind::mxf4nvf4 (.block16, SS, 1SM)", cudaDeviceSynchronize());
  k_fp4_sparse<SparseFp4Form::MXF4NVF4_SS_B32><<<1, 128>>>();
  run("tcgen05.mma.sp.kind::mxf4nvf4 (.block32, SS, 1SM)", cudaDeviceSynchronize());
  k_fp4_sparse<SparseFp4Form::MXF4NVF4_TS_B16><<<1, 128>>>();
  run("tcgen05.mma.sp.kind::mxf4nvf4 (.block16, TS, 1SM)", cudaDeviceSynchronize());
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

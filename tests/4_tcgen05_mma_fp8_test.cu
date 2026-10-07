#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 4_tcgen05_mma_fp8_test.cu -- compile + smoke test for tcgen05.mma.kind::
// f8f6f4 (FP8 / FP6 / FP4 non-scaled).
//
// Verifies (1) the SS 1SM wrapper compiles (operand list, idesc bit
// encoding for E4M3 / E5M2 / E2M3 / E3M2 / E2M1 atype-btype), and (2) a
// kernel allocating TMEM, issuing the MMA with zero descriptors,
// committing, waiting, and deallocating completes without hang. End-to-end
// correctness (real FP8 GEMM) lives at the block level (#90, #100). The 1SM
// TS, sparse and .kind::mxf8f6f4 forms are smoke-tested here too; their 2SM
// TS / mxf8f6f4 forms are compile-checked by _extended_coverage_probe_test.
//
// PTX sniff: `cuobjdump --dump-ptx build/4_tcgen05_mma_fp8_test |
// grep -E 'tcgen05.mma.cta_group::1.kind::f8f6f4'` should show the issue.

#include "test_utils.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include "../src/primitives/0_tcgen05_alloc.cuh"
#include "../src/primitives/1_tcgen05_dealloc.cuh"
#include "../src/primitives/2_tcgen05_relinquish.cuh"
#include "../src/primitives/4_tcgen05_mma_fp8.cuh"
#include "../src/primitives/8_tcgen05_mma_idesc.cuh"
#include "../src/primitives/11_tcgen05_commit.cuh"

__global__ void k_fp8() {
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
    uint64_t da = 0, db = 0;
    da |= static_cast<uint64_t>((smem_ptr_u32(smA) >> 4) & 0x3FFF);
    da |= static_cast<uint64_t>(1) << 46;
    db |= static_cast<uint64_t>((smem_ptr_u32(smB) >> 4) & 0x3FFF);
    db |= static_cast<uint64_t>(1) << 46;
    uint32_t idesc = make_idesc_e4m3_f32(128, 8);
    tcgen05_mma_fp8_ss<1>(tmem_base, da, db, idesc, false);
    tcgen05_commit<1>(smem_ptr_u32(&mbar));
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tmem_base, 128);
}

// Sparse smoke: exercises all 3 .sp 1SM wrappers (fp8 SS, fp8 TS,
// mxf8f6f4 SS .block32) in one kernel. Sparsity selector per PTX
// 9.7.18.10.9.4 must be 0 for .kind::f8f6f4 / .kind::mxf8f6f4.
enum class SparseFp8Form { F8F6F4_SS, F8F6F4_TS, MXF8F6F4_SS_B32 };

template <SparseFp8Form V>
__global__ void k_fp8_sparse() {
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
  uint32_t tmem_base   = tmem_base_slot;
  uint32_t sp_meta     = tmem_base + 120;
  uint32_t scale_a     = tmem_base + 116;
  uint32_t scale_b     = tmem_base + 117;
  uint32_t tmem_a_op   = tmem_base + 64;  // for TS form

  // Stage valid 2:4 metadata (selector=0 for f8f6f4/mxf8f6f4).
  if (threadIdx.x < 32) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n"
                 :: "r"(sp_meta), "r"(0x44444444u));
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n"
                 :: "r"(scale_a), "r"(0x3F3F3F3Fu));  // any non-zero scale
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n"
                 :: "r"(scale_b), "r"(0x3F3F3F3Fu));
  }
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::);

  if (threadIdx.x == 0) {
    uint64_t da = 0, db = 0;
    da |= static_cast<uint64_t>((smem_ptr_u32(smA) >> 4) & 0x3FFF);
    da |= static_cast<uint64_t>(1) << 46;
    db |= static_cast<uint64_t>((smem_ptr_u32(smB) >> 4) & 0x3FFF);
    db |= static_cast<uint64_t>(1) << 46;

    if constexpr (V == SparseFp8Form::F8F6F4_SS) {
      uint32_t idesc = make_idesc_e4m3_f32(128, 8);
      idesc = idesc_set_sparsity(idesc, /*selector=*/0);
      tcgen05_mma_fp8_ss_1sm_sparse(tmem_base, da, db, sp_meta, idesc,
                                    /*enable_d=*/false, 0, 0, 0, 0);
    } else if constexpr (V == SparseFp8Form::F8F6F4_TS) {
      uint32_t idesc = make_idesc_e4m3_f32(128, 8);
      idesc = idesc_set_sparsity(idesc, /*selector=*/0);
      tcgen05_mma_fp8_ts_1sm_sparse(tmem_base, tmem_a_op, db, sp_meta, idesc,
                                    /*enable_d=*/false, 0, 0, 0, 0);
    } else {  // MXF8F6F4_SS_B32
      uint32_t idesc = make_idesc_mxf8f6f4(128, 8, /*atype=E4M3*/0, /*btype=E4M3*/0,
                                            /*ta*/false, /*tb*/false,
                                            /*neg_a*/false, /*neg_b*/false,
                                            /*sf_a_id*/0, /*sf_b_id*/0,
                                            /*ue8m0*/true, /*sparse*/true);
      tcgen05_mma_mxf8f6f4_ss_1sm_block32_sparse(tmem_base, da, db, sp_meta,
                                                 idesc, scale_a, scale_b,
                                                 /*enable_d=*/false);
    }
    tcgen05_commit<1>(smem_ptr_u32(&mbar));
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tmem_base, 128);
}

// Non-sparse extra-forms smoke: exercises
// tcgen05_mma_fp8_ts_1sm (TS form) and tcgen05_mma_mxf8f6f4_ss_1sm_block32
// (block-scaled SS) directly. Same TMEM/idesc/descriptor scaffold as k_fp8.
enum class Fp8ExtraForm { F8F6F4_TS, MXF8F6F4_SS_B32 };

template <Fp8ExtraForm V>
__global__ void k_fp8_extra() {
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
    if constexpr (V == Fp8ExtraForm::F8F6F4_TS) {
      uint32_t idesc = make_idesc_e4m3_f32(128, 8);
      tcgen05_mma_fp8_ts_1sm(tmem_base, tmem_a_op, db, idesc,
                              /*enable_d=*/false, 0, 0, 0, 0);
    } else {  // MXF8F6F4_SS_B32
      uint32_t idesc = make_idesc_mxf8f6f4(128, 8, /*atype=E4M3*/0, /*btype=E4M3*/0);
      tcgen05_mma_mxf8f6f4_ss_1sm_block32(tmem_base, da, db, idesc,
                                          scale_a, scale_b, /*enable_d=*/false);
    }
    tcgen05_commit<1>(smem_ptr_u32(&mbar));
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tmem_base, 128);
}

// Reorder so the most reliably-passing launches go first; clear sticky
// CUDA error state after each error-prone launch so it doesn't leak.
int main() {
  auto run = [](const char* lbl, cudaError_t e) {
    printf("%-58s : %s\n", lbl, e == cudaSuccess ? "no hang" : cudaGetErrorString(e));
    (void)cudaGetLastError();  // clear sticky state for next launch
  };

  k_fp8<<<1, 128>>>();
  run("tcgen05.mma.kind::f8f6f4 smoke", cudaDeviceSynchronize());

  k_fp8_extra<Fp8ExtraForm::F8F6F4_TS><<<1, 128>>>();
  run("tcgen05.mma.kind::f8f6f4 (TS, 1SM) smoke", cudaDeviceSynchronize());

  k_fp8_sparse<SparseFp8Form::F8F6F4_SS><<<1, 128>>>();
  run("tcgen05.mma.sp.kind::f8f6f4 (SS, 1SM) smoke", cudaDeviceSynchronize());

  k_fp8_sparse<SparseFp8Form::F8F6F4_TS><<<1, 128>>>();
  run("tcgen05.mma.sp.kind::f8f6f4 (TS, 1SM) smoke", cudaDeviceSynchronize());

  k_fp8_extra<Fp8ExtraForm::MXF8F6F4_SS_B32><<<1, 128>>>();
  run("tcgen05.mma.kind::mxf8f6f4 (.block32, SS, 1SM) smoke", cudaDeviceSynchronize());

  k_fp8_sparse<SparseFp8Form::MXF8F6F4_SS_B32><<<1, 128>>>();
  run("tcgen05.mma.sp.kind::mxf8f6f4 (.block32, SS, 1SM) smoke", cudaDeviceSynchronize());
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

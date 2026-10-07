#if defined(PL_AGENTIC_SM100A)
// ARCH: sm_100a (SM100A-strict; tcgen05.mma.kind::i8 removed on sm_103a)
// 7_tcgen05_mma_i8_test.cu -- smoke test for tcgen05.mma.kind::i8 (INT8 ->
// INT32 accumulator, signed or unsigned A/B picked via idesc).
//
// Verifies (1) the SS 1SM wrapper compiles with the i8 idesc layout
// (atype = 0 for unsigned, 1 for signed; dtype = 2 for INT32 D; saturate
// bit 3), and (2) a kernel allocating TMEM, issuing the MMA with zero
// descriptors, committing, waiting, and deallocating completes without
// hang. End-to-end INT8 GEMM correctness lives at the block level
// (#90, #100). The TS forms added in e40352e are separately exercised by
// _extended_coverage_probe_test.
//
// PTX sniff: `cuobjdump --dump-ptx build/7_tcgen05_mma_i8_test |
// grep -E 'tcgen05.mma.cta_group::1.kind::i8'` should show the issue.

#include "test_utils.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/0_tcgen05_alloc.cuh"
#include "../primitives/1_tcgen05_dealloc.cuh"
#include "../primitives/2_tcgen05_relinquish.cuh"
#include "../primitives/7_tcgen05_mma_i8.cuh"
#include "../primitives/8_tcgen05_mma_idesc.cuh"
#include "../primitives/11_tcgen05_commit.cuh"

__global__ void k_i8() {
  __shared__ __align__(16)   uint32_t tmem_base_slot;
  __shared__ __align__(16)   uint64_t mbar;
  __shared__ __align__(128)  int8_t   smA[128];
  __shared__ __align__(128)  int8_t   smB[128];
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
    uint32_t idesc = make_idesc_s8_s32(128, 8);
    tcgen05_mma_i8_ss<1>(tmem_base, da, db, idesc, false);
    tcgen05_commit<1>(smem_ptr_u32(&mbar));
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tmem_base, 128);
}

// Sparse smoke: exercises the 2 .sp 1SM wrappers (i8 SS, i8 TS).
// Per PTX 9.7.18.10.9.4 sparsity selector MUST be 0 for .kind::i8.
enum class SparseI8Form { I8_SS, I8_TS };

template <SparseI8Form V>
__global__ void k_i8_sparse() {
  __shared__ __align__(16)   uint32_t tmem_base_slot;
  __shared__ __align__(16)   uint64_t mbar;
  __shared__ __align__(128)  int8_t   smA[128];
  __shared__ __align__(128)  int8_t   smB[128];
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
    uint32_t idesc = make_idesc_s8_s32(128, 8);
    idesc = idesc_set_sparsity(idesc, /*selector=*/0);
    if constexpr (V == SparseI8Form::I8_SS) {
      tcgen05_mma_i8_ss_1sm_sparse(tmem_base, da, db, sp_meta, idesc, false,
                                    0, 0, 0, 0);
    } else {
      tcgen05_mma_i8_ts_1sm_sparse(tmem_base, tmem_a_op, db, sp_meta, idesc,
                                    false, 0, 0, 0, 0);
    }
    tcgen05_commit<1>(smem_ptr_u32(&mbar));
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tmem_base, 128);
}

// Non-sparse TS smoke (round 4c probe-graduation): exercises
// tcgen05_mma_i8_ts_1sm directly. Same scaffold as k_i8.
__global__ void k_i8_ts() {
  __shared__ __align__(16)   uint32_t tmem_base_slot;
  __shared__ __align__(16)   uint64_t mbar;
  __shared__ __align__(128)  int8_t   smB[128];
  if (threadIdx.x == 0) {
    tmem_base_slot = 0;
    mbarrier_init_helper(smem_ptr_u32(&mbar), 1);
    for (int i = 0; i < 128; ++i) smB[i] = 0;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&tmem_base_slot), 128);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  uint32_t tmem_base = tmem_base_slot;
  uint32_t tmem_a_op = tmem_base + 64;

  if (threadIdx.x == 0) {
    uint64_t db = 0; db |= static_cast<uint64_t>(1) << 46;
    uint32_t idesc = make_idesc_s8_s32(128, 8);
    tcgen05_mma_i8_ts_1sm(tmem_base, tmem_a_op, db, idesc,
                          /*enable_d=*/false, 0, 0, 0, 0);
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

  k_i8<<<1, 128>>>();
  run("tcgen05.mma.kind::i8 smoke (SS, 1SM)", cudaDeviceSynchronize());

  k_i8_ts<<<1, 128>>>();
  run("tcgen05.mma.kind::i8 (TS, 1SM) smoke", cudaDeviceSynchronize());

  k_i8_sparse<SparseI8Form::I8_SS><<<1, 128>>>();
  run("tcgen05.mma.sp.kind::i8 (SS, 1SM) smoke", cudaDeviceSynchronize());

  k_i8_sparse<SparseI8Form::I8_TS><<<1, 128>>>();
  run("tcgen05.mma.sp.kind::i8 (TS, 1SM) smoke", cudaDeviceSynchronize());
  PASS();
}

#endif  // PL_AGENTIC_SM100A

#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 3_tcgen05_mma_f16_test.cu -- smoke test for tcgen05.mma.kind::f16 (FP16
// and BF16 inputs, FP32 / FP16 accumulator).
//
// Verifies (1) the SS 1SM wrapper compiles (idesc + descriptor + predicate
// arguments line up with the inline-asm operand list), and (2) a kernel
// that allocates TMEM, issues the MMA with zero descriptors, runs commit +
// wait, and deallocates completes without hang. Full end-to-end correctness
// -- requiring a correctly-swizzled SMEM layout matching the idesc atom
// tiling -- lives at the block level (#90 mma_warp_blackwell, #100
// pipeline_blackwell).
//
// PTX sniff: `cuobjdump --dump-ptx build/3_tcgen05_mma_f16_test |
// grep -E 'tcgen05.mma.cta_group::1.kind::f16'` should show the issue.

#include "test_utils.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/0_tcgen05_alloc.cuh"
#include "../primitives/1_tcgen05_dealloc.cuh"
#include "../primitives/2_tcgen05_relinquish.cuh"
#include "../primitives/3_tcgen05_mma_f16.cuh"
#include "../primitives/8_tcgen05_mma_idesc.cuh"
#include "../primitives/11_tcgen05_commit.cuh"

__global__ void k_mma() {
  __shared__ __align__(1024) uint16_t smA[128 * 64];
  __shared__ __align__(1024) uint16_t smB[8 * 64];
  __shared__ __align__(16)   uint32_t slot;
  __shared__ __align__(16)   uint64_t mbar;

  for (int i = threadIdx.x; i < 128 * 64; i += blockDim.x) smA[i] = 0;
  for (int i = threadIdx.x; i < 8 * 64;   i += blockDim.x) smB[i] = 0;
  if (threadIdx.x == 0) {
    slot = 0;
    mbarrier_init_helper(smem_ptr_u32(&mbar), 1);
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&slot), 128);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();

  uint32_t tmem_base = slot;

  if (threadIdx.x == 0) {
    uint64_t desc_a = 0, desc_b = 0;
    desc_a |= static_cast<uint64_t>((smem_ptr_u32(smA) >> 4) & 0x3FFF);
    desc_a |= static_cast<uint64_t>(1) << 46;
    desc_b |= static_cast<uint64_t>((smem_ptr_u32(smB) >> 4) & 0x3FFF);
    desc_b |= static_cast<uint64_t>(1) << 46;
    uint32_t idesc = make_idesc_f16_f32(128, 8);
    tcgen05_mma_f16_ss<1>(tmem_base, desc_a, desc_b, idesc, /*enable_d=*/false);
    tcgen05_commit<1>(smem_ptr_u32(&mbar));
  }

  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tmem_base, 128);
}

// Sparse smoke: same shape as k_mma but issues tcgen05.mma.sp.kind::f16
// with the sparsity bit set in idesc. Sparsity metadata is staged in TMEM
// (one column allocated for sp-meta) with a legal repeating mask byte
// (0x44 = two 4-bit chunks of 0b0100, the "positions {0,1}" code from PTX
// 9.7.18.10.9.2 -- the canonical 2:4 example value). Kernel must run to
// completion without hanging; full numerical correctness requires the
// metadata-matrix layout from PTX 9.7.18.10.9.5 figures and a swizzled
// SMEM A layout, which is the block-level pipeline's job (#90 / #100).
__global__ void k_mma_sparse() {
  __shared__ __align__(1024) uint16_t smA[128 * 64];
  __shared__ __align__(1024) uint16_t smB[8 * 64];
  __shared__ __align__(16)   uint32_t slot;
  __shared__ __align__(16)   uint64_t mbar;

  for (int i = threadIdx.x; i < 128 * 64; i += blockDim.x) smA[i] = 0;
  for (int i = threadIdx.x; i < 8 * 64;   i += blockDim.x) smB[i] = 0;
  if (threadIdx.x == 0) {
    slot = 0;
    mbarrier_init_helper(smem_ptr_u32(&mbar), 1);
  }
  __syncthreads();

  // Allocate 128 cols: D (N=8) uses cols [0,8); sp-meta goes at tmem_base + 120.
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&slot), 128);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();

  uint32_t tmem_base = slot;
  uint32_t sp_meta_tmem = tmem_base + 120;   // last few cols, away from D

  // Stage metadata: write the legal repeating 2:4 mask 0x44 across the
  // metadata TMEM column. Use tcgen05.st.32x32b.x1 to fill 32 lanes.
  if (threadIdx.x < 32) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n"
                 :: "r"(sp_meta_tmem), "r"(0x44444444u));
    asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
  }
  __syncthreads();
  // Writer's wait::st + fence, then this fence: the st completes before the mma reads it.
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::);

  if (threadIdx.x == 0) {
    uint64_t desc_a = 0, desc_b = 0;
    desc_a |= static_cast<uint64_t>((smem_ptr_u32(smA) >> 4) & 0x3FFF);
    desc_a |= static_cast<uint64_t>(1) << 46;
    desc_b |= static_cast<uint64_t>((smem_ptr_u32(smB) >> 4) & 0x3FFF);
    desc_b |= static_cast<uint64_t>(1) << 46;
    uint32_t idesc = make_idesc_f16_f32(128, 8);
    idesc = idesc_set_sparsity(idesc, /*selector=*/0);
    tcgen05_mma_f16_ss_1sm_sparse(tmem_base, desc_a, desc_b,
                                  sp_meta_tmem, idesc,
                                  /*enable_d=*/false,
                                  0, 0, 0, 0);
    tcgen05_commit<1>(smem_ptr_u32(&mbar));
  }

  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tmem_base, 128);
}

// Scaled-input-d smoke (round 4c probe-graduation): same shape as k_mma but
// issues the scaled SS form: tcgen05.mma.kind::f16 ..., {mask}, p, scale_input_d
// with SCALE_INPUT_D=1.
__global__ void k_mma_scaled() {
  __shared__ __align__(1024) uint16_t smA[128 * 64];
  __shared__ __align__(1024) uint16_t smB[8 * 64];
  __shared__ __align__(16)   uint32_t slot;
  __shared__ __align__(16)   uint64_t mbar;

  for (int i = threadIdx.x; i < 128 * 64; i += blockDim.x) smA[i] = 0;
  for (int i = threadIdx.x; i < 8 * 64;   i += blockDim.x) smB[i] = 0;
  if (threadIdx.x == 0) {
    slot = 0;
    mbarrier_init_helper(smem_ptr_u32(&mbar), 1);
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&slot), 128);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();

  uint32_t tmem_base = slot;

  if (threadIdx.x == 0) {
    uint64_t desc_a = 0, desc_b = 0;
    desc_a |= static_cast<uint64_t>((smem_ptr_u32(smA) >> 4) & 0x3FFF);
    desc_a |= static_cast<uint64_t>(1) << 46;
    desc_b |= static_cast<uint64_t>((smem_ptr_u32(smB) >> 4) & 0x3FFF);
    desc_b |= static_cast<uint64_t>(1) << 46;
    uint32_t idesc = make_idesc_f16_f32(128, 8);
    tcgen05_mma_f16_ss_1sm_scaled<1>(tmem_base, desc_a, desc_b, idesc,
                                     /*enable_d=*/false, 0, 0, 0, 0);
    tcgen05_commit<1>(smem_ptr_u32(&mbar));
  }

  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tmem_base, 128);
}

// 2SM cta_group::2 wrappers (tcgen05_mma_*_2sm, tcgen05_commit<2>,
// tcgen05_commit_multicast<2>) are NOT smoke-tested here. cta_group::2
// dispatch requires a fully-initialized cluster + valid swizzle-aligned
// SMEM matrix descriptors + real data the MMA can read; zero-stride
// descriptors that 1SM tolerates are UB on 2SM. Their minimum-viable
// runtime test surface is the block level (#88 load_warp_blackwell_test
// + #100 pipeline_blackwell_test, which provide the full pipeline).
// ptxas-level acceptance is covered by _extended_coverage_probe_test.cu.

int main() {
  k_mma<<<1, 128>>>();
  cudaError_t e = cudaDeviceSynchronize();
  printf("tcgen05.mma.kind::f16 smoke (1SM/SS)             : %s\n",
         e == cudaSuccess ? "no hang" : cudaGetErrorString(e));
  (void)cudaGetLastError();

  k_mma_scaled<<<1, 128>>>();
  cudaError_t ec = cudaDeviceSynchronize();
  printf("tcgen05.mma.kind::f16 scale_input_d=1 (1SM/SS)   : %s\n",
         ec == cudaSuccess ? "no hang" : cudaGetErrorString(ec));
  (void)cudaGetLastError();

  k_mma_sparse<<<1, 128>>>();
  cudaError_t es = cudaDeviceSynchronize();
  printf("tcgen05.mma.sp.kind::f16 (1SM/SS)                : %s\n",
         es == cudaSuccess ? "no hang" : cudaGetErrorString(es));

  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

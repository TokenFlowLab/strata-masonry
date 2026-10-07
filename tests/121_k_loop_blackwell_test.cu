#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 121_k_loop_blackwell_test.cu -- compile smoke (body is a no-op MMA placeholder).
// Wiring a real MMA chain is covered by the block kernels (#90).

#include "test_utils.cuh"
#include "../primitives/30_mbarrier_arrive.cuh"
#include "../composites/121_k_loop_blackwell.cuh"

__global__ void __cluster_dims__(1, 1, 1) k() {
  __shared__ __align__(16) uint64_t full[2];
  __shared__ __align__(16) uint64_t empty[2];
  __shared__ __align__(16) uint64_t acc;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                 :: "r"(smem_ptr_u32(&full[0])));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                 :: "r"(smem_ptr_u32(&full[1])));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                 :: "r"(smem_ptr_u32(&acc)));
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    // Pre-arrive so try_wait returns immediately.
    mbarrier_arrive(smem_ptr_u32(&full[0]));
    mbarrier_arrive(smem_ptr_u32(&full[1]));
  }
  __syncthreads();

  MbarrierPhaseTracker<2> ph; ph.init();
  if (threadIdx.x == 0) {
    k_loop_blackwell<2, 1>(
        full, empty, smem_ptr_u32(&acc), /*ctamask=*/0x1, /*k_blocks=*/2,
        ph, [] __device__ (int) {});
  }
  __syncthreads();
}

int main() {
  k<<<1, 128>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("k_loop_blackwell : compile + run OK\n");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

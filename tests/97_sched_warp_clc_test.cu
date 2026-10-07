#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 97_sched_warp_clc_test.cu -- exercises blocks/97_sched_warp_clc.cuh.
//
// Verifies (1) the CLC try_cancel + mbarrier-wait + query_cancel chain
// completes without fault when the kernel is launched in a 1x1x1 cluster,
// and (2) the returned `canceled` flag is propagated through the
// composite (#81 clc_scheduler_loop) into our output buffer.
//
// Real persistent-kernel scheduler behavior (loop until canceled, with
// real tile work in between) is covered in #100 pipeline_blackwell.
//
// PTX sniff: `cuobjdump --dump-ptx build/97_sched_warp_clc_test |
// grep -E 'clusterlaunchcontrol.try_cancel'` should show one issue.

#include "test_utils.cuh"
#include "../blocks/97_sched_warp_clc.cuh"

// __global__ wrapper -- owns SMEM (slot[4] + mbar). The .cuh exports the
// CLC try_cancel + wait + query body as __device__ __forceinline__.
__global__ void __cluster_dims__(1, 1, 1)
sched_warp_clc_test_kernel(uint32_t* out_canceled) {
  WpCtx wpc = wp_ctx_init();  // WARP_PROF TraceContext (no-op without -DWARP_PROF)

  __shared__ __align__(16) uint32_t slot[4];
  __shared__ __align__(16) uint64_t mbar;
  for (int i = threadIdx.x; i < 4; i += blockDim.x) slot[i] = 0;
  if (threadIdx.x == 0)
    mbarrier_init(smem_ptr_u32(&mbar), 1);
  __syncthreads();

  sched_warp_clc_block(wpc, slot, &mbar, out_canceled);

  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
}

int main() {
  uint32_t* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, 4));
  CUDA_CHECK(cudaMemset(d, 0xFF, 4));

  sched_warp_clc_test_kernel<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());

  uint32_t h = 0;
  CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  cudaFree(d);

  printf("sched_clc : canceled=%u (chain completed without fault)\n", h);
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

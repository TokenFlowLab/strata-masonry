#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 100_pipeline_blackwell_test.cu -- exercises blocks/100_pipeline_blackwell.cuh.
//
// Smoke-launches the canonical Blackwell pipeline skeleton kernel:
// 8 warps per CTA, 2-CTA cluster, full mbarrier suite,
// CLC scheduler delivering N tiles. The kernel has no real bodies (no MMA,
// no TMA loads, no EPI work) -- success criterion is "kernel completes
// without hanging across N tiles + cluster sync + tcgen05 alloc/dealloc."

#include "test_utils.cuh"
#include "../blocks/100_pipeline_blackwell.cuh"

int main() {
  // 4 mainloop stages; CLC dispatches NUM_TILES tiles to each cluster.
  // Grid = (2 * NUM_TILES, 1, 1) so CLC has enough launchable tiles to
  // satisfy each cluster's persistent loop.
  constexpr int NUM_STAGES = 4;
  constexpr int NUM_TILES  = 4;

  // SMEM = barriers + clc_response.
  constexpr int BAR_BYTES =
      (2 * NUM_STAGES + 2 * 2 + 2 * 2 + 2 * 2) * sizeof(uint64_t)  // full/empty + acc + clc + throttle
      + 2 * 4 * sizeof(uint32_t);                                  // clc_response
  constexpr int SMEM_BYTES = ((BAR_BYTES + 127) / 128) * 128 + 256;

  CUDA_CHECK(cudaFuncSetAttribute(pipeline_blackwell_skeleton_kernel<NUM_STAGES>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES));
  CUDA_CHECK(cudaFuncSetAttribute(pipeline_blackwell_skeleton_kernel<NUM_STAGES>,
      cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(2 * NUM_TILES, 1, 1);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.dynamicSmemBytes = SMEM_BYTES;
  cfg.stream = nullptr;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {2, 1, 1};
  cfg.attrs = attrs;
  cfg.numAttrs = 1;

  cudaError_t err = cudaLaunchKernelEx(&cfg,
      pipeline_blackwell_skeleton_kernel<NUM_STAGES>);
  if (err != cudaSuccess) FAIL("kernel launch failed");
  CUDA_CHECK(cudaDeviceSynchronize());

  printf("pipeline_blackwell skeleton : %d tiles cycled OK (NUM_STAGES=%d)\n",
         NUM_TILES, NUM_STAGES);
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

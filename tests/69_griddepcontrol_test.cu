// ARCH: sm_100a
// 69_griddepcontrol_test.cu -- prerequisite/dependent grid pair with
// programmatic-stream-serialization. The prerequisite writes a sentinel
// value to gmem, fences, calls launch_dependents. The dependent calls
// wait first, then reads the sentinel and writes a confirmation. Host
// verifies the dependent saw the prerequisite's write.
//
// This test does not prove wait actually blocks (the runtime may
// schedule the dependent after the prerequisite completes anyway), but
// it does verify:
//   - both PTX ops assemble and run without error.
//   - the wait + read pattern produces the correct sentinel.
//   - the host-side ProgrammaticStreamSerialization attribute is
//     accepted by cudaLaunchKernelEx.
// The full eviction-overlap behavior is observable only via NCU /
// nsys trace; functional correctness is what this smoke covers.

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../primitives/69_griddepcontrol.cuh"

__global__ void prerequisite_kernel(uint32_t* sentinel) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    sentinel[0] = 0xCAFEBABE;
    // Release fence so the dependent's wait-acquire sees the write.
    asm volatile("fence.release.gpu;\n" ::: "memory");
  }
  __syncthreads();
  griddepcontrol_launch_dependents();
}

__global__ void dependent_kernel(const uint32_t* sentinel,
                                  uint32_t* observed) {
  // Wait for all prerequisite grids in this stream to drain + commit.
  griddepcontrol_wait();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    observed[0] = sentinel[0];
  }
}

int main() {
  CUDA_CHECK(cudaFree(0));
  printf("69_griddepcontrol: prerequisite/dependent pair via "
         "ProgrammaticStreamSerialization\n");

  uint32_t* d_sentinel = nullptr;
  uint32_t* d_observed = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sentinel, sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_observed, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_sentinel, 0, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_observed, 0, sizeof(uint32_t)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cudaLaunchAttribute attr = {};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;

  cudaLaunchConfig_t cfg_pre = {};
  cfg_pre.gridDim          = dim3(1, 1, 1);
  cfg_pre.blockDim         = dim3(32, 1, 1);
  cfg_pre.dynamicSmemBytes = 0;
  cfg_pre.stream           = stream;
  cfg_pre.attrs            = &attr;
  cfg_pre.numAttrs         = 1;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg_pre, prerequisite_kernel, d_sentinel));

  cudaLaunchConfig_t cfg_dep = cfg_pre;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg_dep, dependent_kernel,
                                static_cast<const uint32_t*>(d_sentinel),
                                d_observed));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  uint32_t observed = 0;
  CUDA_CHECK(cudaMemcpy(&observed, d_observed, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  cudaFree(d_sentinel); cudaFree(d_observed);
  cudaStreamDestroy(stream);

  printf("  observed sentinel: 0x%08x (expected 0xCAFEBABE)\n", observed);
  if (observed != 0xCAFEBABE) {
    FAIL("dependent kernel observed wrong sentinel");
  }
  printf("griddepcontrol.{wait,launch_dependents} : OK\n");
  PASS();
}

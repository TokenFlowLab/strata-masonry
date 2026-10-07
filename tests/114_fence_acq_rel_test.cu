// ARCH: sm_90a
// 114_fence_acq_rel_test.cu -- compile smoke + one observable ordering check.
//
// A fence has no standalone effect to print; the kernel below at least runs the
// pattern the primitive exists for: a thread reads a shared slot, fences, then
// releases the slot with an mbarrier arrive, and a second warp overwrites the
// slot only after that arrive. The value read must be the pre-release one.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../src/primitives/29_mbarrier_init.cuh"
#include "../src/primitives/30_mbarrier_arrive.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include "../src/primitives/35_fence_mbarrier_init.cuh"
#include "../src/primitives/114_fence_acq_rel.cuh"

__global__ void k_fence_compile() {
  fence_acq_rel_cta();
  fence_acq_rel_cluster();
  fence_acq_rel_gpu();
}

// Warp 0 (reader) waits for the slot to be filled, reads it, fences, arrives on `freed`.
// Warp 1 (writer) fills the slot, arrives on `filled`, waits on `freed`, overwrites the slot.
__global__ void k_read_then_release(int* out, int rounds) {
  __shared__ uint64_t filled, freed;
  __shared__ uint32_t slot;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&filled), 1);
    mbarrier_init(smem_ptr_u32(&freed), 32);
    fence_mbarrier_init_release_cluster();
  }
  __syncthreads();
  uint32_t phase = 0;
  int bad = 0;
  for (int r = 0; r < rounds; ++r) {
    if (warp == 1) {
      if (lane == 0) {
        slot = 1000 + r;
        mbarrier_arrive(smem_ptr_u32(&filled));
      }
      mbarrier_wait_parity(smem_ptr_u32(&freed), phase);
    } else {
      mbarrier_wait_parity(smem_ptr_u32(&filled), phase);
      const uint32_t v = *reinterpret_cast<volatile uint32_t*>(&slot);
      fence_acq_rel_cta();
      mbarrier_arrive(smem_ptr_u32(&freed));
      if (v != 1000u + r) ++bad;
    }
    phase ^= 1;
  }
  if (warp == 0 && lane == 0) *out = bad;
}

int main() {
  k_fence_compile<<<1, 32>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  int* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
  k_read_then_release<<<1, 64>>>(d_out, 4096);
  CUDA_CHECK(cudaDeviceSynchronize());
  int bad = -1;
  CUDA_CHECK(cudaMemcpy(&bad, d_out, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out));
  printf("fence.acq_rel : compile OK; read-then-release rounds with stale value: %d\n", bad);
  if (bad != 0) { printf("FAIL\n"); return 1; }
  PASS();
}

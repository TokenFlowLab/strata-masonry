#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 12_tcgen05_wait_test.cu -- tcgen05.wait::{ld,st} compile and block until
// prior tcgen05.ld/.st from the thread have completed. Covered in combination
// by the ld/st round-trip (#9, #10); this is the standalone compile smoke.

#include "test_utils.cuh"
#include "../src/primitives/12_tcgen05_wait.cuh"

__global__ void k_wait() {
  // No prior ld/st -- waits are vacuous and return immediately.
  tcgen05_wait_ld();
  tcgen05_wait_st();
}

int main() {
  k_wait<<<1, 32>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("tcgen05.wait::{ld,st} : compile + vacuous execute OK\n");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

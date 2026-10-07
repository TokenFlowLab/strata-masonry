// ARCH: sm_100a
// 68_l2cache_policy_test.cu -- build cache-policy tokens via createpolicy
// wrappers and confirm each primitive emits a distinct, nonzero token.
//
// createpolicy is a pure-register PTX op (no memory side effect), so the
// minimum-viable runtime test is: have the device build N tokens via the
// N wrappers, write them to gmem, and verify on the host that they are
// nonzero AND that the four "canonical" forms (evict_last,
// evict_normal, evict_first, evict_unchanged) produce 4 distinct
// values. End-to-end producer-consumer coverage (createpolicy ->
// tma_load_2d_2sm_l2hint) is exercised by the load_warp block tests in
// step 3.

#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../primitives/68_l2cache_policy.cuh"

__global__ void k_build_policies(uint64_t* out) {
  if (threadIdx.x != 0) return;
  out[0] = make_l2cache_policy_evict_last_full();
  out[1] = make_l2cache_policy_evict_normal_full();
  out[2] = make_l2cache_policy_evict_first_full();
  out[3] = make_l2cache_policy_evict_unchanged_full();
  out[4] = make_l2cache_policy_fractional_evict_last_unchanged(0.5f);
  out[5] = make_l2cache_policy_fractional_evict_first_unchanged(0.25f);
  // make_l2cache_policy_from_access_property(...) is exercised when the host
  // CUDA runtime supplies an access-property handle; we skip it here
  // since constructing one without the runtime side is brittle.
}

int main() {
  CUDA_CHECK(cudaFree(0));
  printf("68_l2cache_policy: build L2 cache-hint tokens and verify\n");

  uint64_t* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, 6 * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(d, 0, 6 * sizeof(uint64_t)));

  k_build_policies<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());

  uint64_t h[6] = {};
  CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  cudaFree(d);

  const char* names[6] = {
      "evict_last_full",
      "evict_normal_full",
      "evict_first_full",
      "evict_unchanged_full",
      "fractional_evict_last_unchanged(0.5)",
      "fractional_evict_first_unchanged(0.25)",
  };
  for (int i = 0; i < 6; ++i)
    printf("  %-44s -> 0x%016llx\n", names[i],
           (unsigned long long)h[i]);

  bool ok = true;
  for (int i = 0; i < 6; ++i)
    if (h[i] == 0ull) {
      fprintf(stderr, "FAIL: %s produced zero policy\n", names[i]);
      ok = false;
    }

  // The four canonical evict_* full-fraction forms must produce 4
  // distinct tokens (they encode different primary priorities into the
  // same 64-bit container).
  uint64_t canon[4] = {h[0], h[1], h[2], h[3]};
  for (int i = 0; i < 4; ++i)
    for (int j = i + 1; j < 4; ++j)
      if (canon[i] == canon[j]) {
        fprintf(stderr, "FAIL: %s == %s (0x%016llx)\n",
                names[i], names[j], (unsigned long long)canon[i]);
        ok = false;
      }

  if (!ok) FAIL("createpolicy primitives produced unexpected tokens");
  printf("createpolicy primitives : OK (6 tokens, 4 canonical distinct)\n");
  PASS();
}

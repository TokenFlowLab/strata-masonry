#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 49_redux_sync_f32_test.cu -- max[0..31] via redux.sync.max.f32.

#include "test_utils.cuh"
#include "../src/primitives/49_redux_sync_f32.cuh"

__global__ void k_f32(float* out) {
  float v = (float)threadIdx.x * 0.5f;
  float m = redux_sync_max_f32(v);
  if (threadIdx.x == 0) *out = m;
}

// .NaN cross-product test: lane 7 holds NaN; .NaN-flavored reductions must
// propagate NaN; the non-.NaN reductions return the real-numeric result.
__global__ void k_nan_cross(float* out) {
  float v;
  if (threadIdx.x == 7) v = nanf("");
  else                  v = -1.0f * (float)threadIdx.x;  // {-0,-1,...,-31}
  out[0] = redux_sync_min_f32(v);          // ignores NaN -> -31
  out[1] = redux_sync_max_f32(v);          // ignores NaN -> 0
  out[2] = redux_sync_min_abs_f32(v);      // ignores NaN -> 0 (|0|=0)
  out[3] = redux_sync_max_abs_f32(v);      // ignores NaN -> 31 (|-31|)
  out[4] = redux_sync_max_nan_f32(v);      // NaN-propagating
  out[5] = redux_sync_min_nan_f32(v);      // NaN-propagating
  out[6] = redux_sync_min_abs_nan_f32(v);  // NaN-propagating
  out[7] = redux_sync_max_abs_nan_f32(v);  // NaN-propagating
}

int main() {
  float* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
  k_f32<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  float h = 0.f; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h != 15.5f) FAIL("redux.sync.max.f32 wrong");
  printf("redux.sync.max.f32 lane=31*0.5 = %g\n", h);

  // .NaN cross-product subtest
  float* dn = nullptr; CUDA_CHECK(cudaMalloc(&dn, 32));
  k_nan_cross<<<1, 32>>>(dn);
  CUDA_CHECK(cudaDeviceSynchronize());
  float n[8];
  CUDA_CHECK(cudaMemcpy(n, dn, 32, cudaMemcpyDeviceToHost));
  cudaFree(dn);
  const char* labels[8] = {
    "min.f32        (no NaN prop)", "max.f32        (no NaN prop)",
    "min.abs.f32    (no NaN prop)", "max.abs.f32    (no NaN prop)",
    "max.NaN.f32    (NaN prop)   ", "min.NaN.f32    (NaN prop)   ",
    "min.abs.NaN.f32 (NaN prop)  ", "max.abs.NaN.f32 (NaN prop)  ",
  };
  for (int i = 0; i < 8; ++i)
    printf("  %s = %g\n", labels[i], n[i]);
  // Non-NaN forms must NOT be NaN; .NaN forms MUST be NaN (since lane 7 is NaN).
  for (int i = 0; i < 4; ++i) if (isnan(n[i])) FAIL("non-NaN form propagated NaN");
  for (int i = 4; i < 8; ++i) if (!isnan(n[i])) FAIL(".NaN form did not propagate NaN");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

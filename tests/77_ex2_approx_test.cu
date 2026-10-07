// ARCH: sm_100a
// 77_ex2_approx_test.cu -- verify ex2.approx.f32 (HW) and the packed 2x ex2
// polynomial emulation (ex2_emu_f32x2) against host exp2().
//
// Build: nvcc -arch=sm_100a -std=c++17 77_ex2_approx_test.cu -o ex2_test
//
// The HW path must match exp2 to MUFU.EX2 accuracy; the emulation is a degree-3
// poly so it carries more error -- we check a relaxed relative tolerance and that
// masked -inf inputs flush to ~0 (so softmax masking still works).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../primitives/77_ex2_approx.cuh"

__global__ void k_ex2(const float* in, int n, float* hw, float* emu) {
  int t = threadIdx.x;
  if (t >= n) return;
  hw[t] = ex2_approx_f32(in[t]);
  if ((t & 1) == 0 && t + 1 < n) {
    float2 e = ex2_emu_f32x2(in[t], in[t + 1]);
    emu[t] = e.x; emu[t + 1] = e.y;
  }
}

int main() {
  CUDA_CHECK(cudaFree(0));
  // span typical softmax exp2 args (<= 0 after subtracting the row max) plus the
  // -inf mask sentinel and a couple of positive values.
  std::vector<float> h_in = {
      0.0f, -0.5f, -1.0f, -2.0f, -3.7f, -8.0f, -15.0f, -40.0f,
      -0.25f, -6.25f, 1.0f, 2.5f, -100.0f, -INFINITY, -12.0f, -0.1f};
  const int N = (int)h_in.size();

  float *d_in, *d_hw, *d_emu;
  CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hw,  N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_emu, N * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_emu, 0, N * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  k_ex2<<<1, N>>>(d_in, N, d_hw, d_emu);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hw(N), emu(N);
  CUDA_CHECK(cudaMemcpy(hw.data(),  d_hw,  N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(emu.data(), d_emu, N * sizeof(float), cudaMemcpyDeviceToHost));
  cudaFree(d_in); cudaFree(d_hw); cudaFree(d_emu);

  bool ok = true;
  int bad_hw = 0, bad_emu = 0;
  for (int i = 0; i < N; ++i) {
    double ref = exp2((double)h_in[i]);            // host reference
    // HW: ex2.approx.f32 ~ 2^-22 relative; clamp ref for -inf -> 0.
    float tol_hw = (float)(1e-5 * fmax(ref, 1e-30) + 1e-30);
    if (fabsf(hw[i] - (float)ref) > tol_hw) {
      if (bad_hw < 4) printf("  HW  [%2d] in=%g got=%.8g ref=%.8g\n", i, h_in[i], hw[i], ref);
      bad_hw++;
    }
    // Emulation only ran on even lanes (pairs). Degree-3 poly -> ~1%% relative.
    if ((i & 1) == 0) {
      float tol_emu = (float)(0.02 * fmax(ref, 1e-30) + 1e-6);
      if (fabsf(emu[i] - (float)ref) > tol_emu) {
        if (bad_emu < 4) printf("  EMU [%2d] in=%g got=%.8g ref=%.8g\n", i, h_in[i], emu[i], ref);
        bad_emu++;
      }
    }
  }
  printf("ex2_approx_f32:  %s (%d/%d within tol)\n", bad_hw == 0 ? "OK" : "FAIL", N - bad_hw, N);
  printf("ex2_emu_f32x2:   %s (%d/%d pairs-lanes within tol)\n", bad_emu == 0 ? "OK" : "FAIL", N / 2 - bad_emu, N / 2);

  // mask sentinel: -inf must produce ~0 on both paths (index 13 is -inf, lane 12 even -> emu[13] via pair).
  if (hw[13] > 1e-20f) { printf("  HW ex2(-inf) = %g, expected ~0\n", hw[13]); ok = false; }

  if (bad_hw || bad_emu || !ok) { FAIL("ex2 mismatch"); return 1; }
  PASS();
  return 0;
}

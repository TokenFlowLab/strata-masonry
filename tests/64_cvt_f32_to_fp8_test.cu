// ARCH: sm_90a
// 64_cvt_f32_to_fp8_test.cu -- compile smoke + non-zero output check.
//
// Two test sets in one binary: run_ours() and run_theirs().

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../src/primitives/64_cvt_f32_to_fp8.cuh"
#include <cuda_fp8.h>
#include "64_cvt_f32_to_fp8.cuh"

// =============================================================================
// ours
// =============================================================================

// ARCH: sm_90a


__global__ void k_cvt(uint32_t* out) {
  out[0] = cvt_pack_f32_to_e4m3x2(1.0f, -1.0f);
  out[1] = cvt_pack_f32_to_e5m2x2(1.0f, -1.0f);
  out[2] = cvt_f32_to_e4m3(1.5f);
  out[3] = cvt_f32_to_e5m2(-2.25f);
  // .relu packed: hi byte (negative input) clamps to +0
  out[4] = cvt_pack_f32_to_e4m3x2_relu(1.0f, -1.0f);
  out[5] = cvt_pack_f32_to_e5m2x2_relu(1.0f, -1.0f);
}

static int run_ours() {
  uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 24));
  k_cvt<<<1, 1>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h[6]; CUDA_CHECK(cudaMemcpy(h, d, 24, cudaMemcpyDeviceToHost));
  cudaFree(d);
  printf("cvt fp8 : e4m3x2=0x%04x e5m2x2=0x%04x e4m3=0x%02x e5m2=0x%02x\n",
         h[0], h[1], h[2], h[3]);
  printf("cvt fp8 .relu: e4m3x2_relu=0x%04x e5m2x2_relu=0x%04x\n",
         h[4], h[5]);
  if (h[0] == 0 && h[1] == 0) FAIL("fp8 cvt produced all zero");
  // .relu hi byte (which encodes the negative input) must be 0.
  if (((h[4] >> 8) & 0xFF) != 0) FAIL("e4m3x2_relu hi byte not clamped to 0");
  if (((h[5] >> 8) & 0xFF) != 0) FAIL("e5m2x2_relu hi byte not clamped to 0");
  PASS();
}

// =============================================================================
// theirs
// =============================================================================

// Runtime test: cvt.rn.satfinite.{e4m3,e5m2}.f32 -- FP32 -> FP8 round-trip

// Kernel: convert inputs to e4m3 and e5m2 FP8, both single-lane and packed
__global__ void cvt_fp8_kernel(const float* in, int n,
                                uint8_t*  out_e4m3,
                                uint8_t*  out_e5m2,
                                uint16_t* out_e4m3x2,
                                uint16_t* out_e5m2x2) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    float v = in[tid];
    out_e4m3[tid] = (uint8_t)cvt_f32_to_e4m3(v);
    out_e5m2[tid] = (uint8_t)cvt_f32_to_e5m2(v);
    if ((tid & 1) == 0 && tid + 1 < n) {
        float a = in[tid], b = in[tid + 1];
        out_e4m3x2[tid / 2] = cvt_f32x2_to_e4m3x2(a, b);
        out_e5m2x2[tid / 2] = cvt_f32x2_to_e5m2x2(a, b);
    }
}

__global__ void cvt_fp8_perf_kernel(const float* in, uint16_t* out, int iters) {
    int tid = threadIdx.x;
    float a = in[tid];
    float b = in[tid + 1];
    uint16_t acc = 0;
    for (int i = 0; i < iters; i++) {
        acc ^= cvt_f32x2_to_e4m3x2(a + i, b + i);
    }
    if (tid == 0) out[0] = acc;
}

static int run_theirs() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    // Stay in E4M3's range (max ~448). Include values that test saturation too.
    const float h_in[] = {
        0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 100.0f,
        3.14159f, -3.14159f, 0.125f, 0.25f, 42.0f, -42.0f, 200.0f, -200.0f
    };
    const int N = sizeof(h_in) / sizeof(h_in[0]);  // 16

    float    *d_in;       CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    uint8_t  *d_e4m3;     CUDA_CHECK(cudaMalloc(&d_e4m3, N));
    uint8_t  *d_e5m2;     CUDA_CHECK(cudaMalloc(&d_e5m2, N));
    uint16_t *d_e4m3x2;   CUDA_CHECK(cudaMalloc(&d_e4m3x2, (N / 2) * sizeof(uint16_t)));
    uint16_t *d_e5m2x2;   CUDA_CHECK(cudaMalloc(&d_e5m2x2, (N / 2) * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    cvt_fp8_kernel<<<1, N>>>(d_in, N, d_e4m3, d_e5m2, d_e4m3x2, d_e5m2x2);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint8_t  h_e4m3[N],  h_e5m2[N];
    uint16_t h_e4m3x2[N / 2], h_e5m2x2[N / 2];
    CUDA_CHECK(cudaMemcpy(h_e4m3,    d_e4m3,   N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_e5m2,    d_e5m2,   N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_e4m3x2,  d_e4m3x2, (N / 2) * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_e5m2x2,  d_e5m2x2, (N / 2) * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    // Helper: unpack a byte as E4M3 or E5M2 and convert back to float on host
    auto e4m3_to_f32 = [](uint8_t b) -> float {
        __nv_fp8_e4m3 v; memcpy(&v, &b, 1);
        return float(v);
    };
    auto e5m2_to_f32 = [](uint8_t b) -> float {
        __nv_fp8_e5m2 v; memcpy(&v, &b, 1);
        return float(v);
    };

    // E4M3: 4-bit exp, 3-bit mantissa -- ~12% precision (2^-3), max 448
    int bad_e4m3 = 0;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_in[i]) > 448.0f) continue;  // saturation region, skip
        float round = e4m3_to_f32(h_e4m3[i]);
        if (!std::isfinite(round)) { bad_e4m3++; continue; }
        float tol = 0.1f + 0.15f * fabsf(h_in[i]);  // loose per-task spec
        if (fabsf(round - h_in[i]) > tol) {
            if (bad_e4m3 < 3) printf("    e4m3 mismatch [%d]: in=%g round=%g\n",
                                    i, h_in[i], round);
            bad_e4m3++;
        }
    }
    if (bad_e4m3 == 0) printf("  cvt_f32_to_e4m3 (single): OK\n");
    else               { printf("  cvt_f32_to_e4m3: FAIL (%d)\n", bad_e4m3); all_pass = false; }

    // E5M2: 5-bit exp, 2-bit mantissa -- ~25% precision, much larger range
    int bad_e5m2 = 0;
    for (int i = 0; i < N; i++) {
        float round = e5m2_to_f32(h_e5m2[i]);
        if (!std::isfinite(round)) { bad_e5m2++; continue; }
        float tol = 0.2f + 0.30f * fabsf(h_in[i]);
        if (fabsf(round - h_in[i]) > tol) {
            if (bad_e5m2 < 3) printf("    e5m2 mismatch [%d]: in=%g round=%g\n",
                                    i, h_in[i], round);
            bad_e5m2++;
        }
    }
    if (bad_e5m2 == 0) printf("  cvt_f32_to_e5m2 (single): OK\n");
    else               { printf("  cvt_f32_to_e5m2: FAIL (%d)\n", bad_e5m2); all_pass = false; }

    // Packed E4M3x2
    int bad_e4m3x2 = 0;
    for (int i = 0; i < N / 2; i++) {
        uint16_t packed = h_e4m3x2[i];
        uint8_t lo = packed & 0xFF;
        uint8_t hi = (packed >> 8) & 0xFF;
        float rlo = e4m3_to_f32(lo), rhi = e4m3_to_f32(hi);
        if (!std::isfinite(rlo) || !std::isfinite(rhi)) { bad_e4m3x2++; continue; }
        float tol_a = 0.1f + 0.15f * fabsf(h_in[2 * i]);
        float tol_b = 0.1f + 0.15f * fabsf(h_in[2 * i + 1]);
        if (fabsf(h_in[2 * i]) <= 448.0f && fabsf(rlo - h_in[2 * i])     > tol_a) bad_e4m3x2++;
        if (fabsf(h_in[2 * i + 1]) <= 448.0f && fabsf(rhi - h_in[2 * i + 1]) > tol_b) bad_e4m3x2++;
    }
    if (bad_e4m3x2 == 0) printf("  cvt_f32x2_to_e4m3x2: OK\n");
    else                 { printf("  cvt_f32x2_to_e4m3x2: FAIL (%d)\n", bad_e4m3x2); all_pass = false; }

    // Packed E5M2x2
    int bad_e5m2x2 = 0;
    for (int i = 0; i < N / 2; i++) {
        uint16_t packed = h_e5m2x2[i];
        uint8_t lo = packed & 0xFF;
        uint8_t hi = (packed >> 8) & 0xFF;
        float rlo = e5m2_to_f32(lo), rhi = e5m2_to_f32(hi);
        if (!std::isfinite(rlo) || !std::isfinite(rhi)) { bad_e5m2x2++; continue; }
        float tol_a = 0.2f + 0.30f * fabsf(h_in[2 * i]);
        float tol_b = 0.2f + 0.30f * fabsf(h_in[2 * i + 1]);
        if (fabsf(rlo - h_in[2 * i])     > tol_a) bad_e5m2x2++;
        if (fabsf(rhi - h_in[2 * i + 1]) > tol_b) bad_e5m2x2++;
    }
    if (bad_e5m2x2 == 0) printf("  cvt_f32x2_to_e5m2x2: OK\n");
    else                 { printf("  cvt_f32x2_to_e5m2x2: FAIL (%d)\n", bad_e5m2x2); all_pass = false; }

    // No NaN/Inf from satfinite (covers all values we tried)
    int bad_nan = 0;
    for (int i = 0; i < N; i++) {
        float v = e4m3_to_f32(h_e4m3[i]);
        if (!std::isfinite(v)) bad_nan++;
        v = e5m2_to_f32(h_e5m2[i]);
        if (!std::isfinite(v)) bad_nan++;
    }
    if (bad_nan == 0) printf("  no NaN/Inf (satfinite): OK\n");
    else              { printf("  satfinite: FAIL (%d NaN/Inf)\n", bad_nan); all_pass = false; }

    // Perf
    uint16_t *d_perf; CUDA_CHECK(cudaMalloc(&d_perf, 4));
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 1000; i++) cvt_fp8_perf_kernel<<<1, 32>>>(d_in, d_perf, 100);
    t.end();
    printf("  perf: %.2f us/launch (1000 launches, 100 cvts each)\n",
           t.elapsed_ms() * 1000.0f / 1000);

    cudaFree(d_in); cudaFree(d_e4m3); cudaFree(d_e5m2);
    cudaFree(d_e4m3x2); cudaFree(d_e5m2x2); cudaFree(d_perf);
    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}

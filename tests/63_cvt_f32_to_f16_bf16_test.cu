// ARCH: sm_90a
// 63_cvt_f32_to_f16_bf16_test.cu -- verify pair-pack cvt matches __float2half.
//
// Combined test: both ours' and theirs' coverage is exercised
// in a single binary (each side's main() became run_ours/run_theirs).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../src/primitives/63_cvt_f32_to_f16_bf16.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "63_cvt_f32_to_f16_bf16.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k_cvt(uint32_t* out) {
  float a = 1.5f, b = -2.25f;
  out[0] = cvt_pack_f32_to_f16x2(a, b);
  out[1] = cvt_pack_f32_to_bf16x2(a, b);
}

// .relu / .ftz modifier subtest. `.relu` clamps negatives to +0;
// `.ftz` flushes a denormal-result to +0. PTX rejects `.ftz` on packed
// (f16x2/bf16x2) forms, so only `.relu` is exercised on packed.
__global__ void k_cvt_modifiers(uint32_t* out16, uint16_t* outs) {
  float pos = 1.5f, neg = -2.5f, denorm = 1e-7f;
  // Packed .relu: hi (negative) clamps to +0.
  out16[0] = cvt_pack_f32_to_f16x2_relu(pos, neg);
  out16[1] = cvt_pack_f32_to_bf16x2_relu(pos, neg);
  // Scalar .relu / .ftz outputs.
  outs[0] = cvt_f32_to_f16_relu(neg);     // expected 0
  outs[1] = cvt_f32_to_f16_ftz(denorm);   // expected 0 (FP16 denormal flushed)
  outs[2] = cvt_f32_to_bf16_relu(neg);    // expected 0
  outs[3] = cvt_f32_to_bf16_ftz(denorm);  // expected 0 (BF16 denormal flushed)
}

static int run_ours() {
  /* (orig args dropped) */
  uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 8));
  k_cvt<<<1, 1>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h[2]; CUDA_CHECK(cudaMemcpy(h, d, 8, cudaMemcpyDeviceToHost));
  cudaFree(d);
  __half a = __float2half(1.5f), b = __float2half(-2.25f);
  uint32_t want_f16 =
      ((uint32_t)*reinterpret_cast<uint16_t*>(&b) << 16) |
       *reinterpret_cast<uint16_t*>(&a);
  printf("cvt.rn.f16x2  got=0x%08x want=0x%08x\n", h[0], want_f16);
  if (h[0] != want_f16) FAIL("f16x2 pack mismatch");
  printf("cvt.rn.bf16x2 got=0x%08x\n", h[1]);
  if (h[1] == 0) FAIL("bf16x2 produced zero");

  // .relu / .ftz modifier subtest
  uint32_t* dm  = nullptr; CUDA_CHECK(cudaMalloc(&dm,  8));   // 2 packed
  uint16_t* dms = nullptr; CUDA_CHECK(cudaMalloc(&dms, 8));   // 4 scalar
  k_cvt_modifiers<<<1, 1>>>(dm, dms);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t hm[2];  CUDA_CHECK(cudaMemcpy(hm,  dm,  8, cudaMemcpyDeviceToHost));
  uint16_t hms[4]; CUDA_CHECK(cudaMemcpy(hms, dms, 8, cudaMemcpyDeviceToHost));
  cudaFree(dm); cudaFree(dms);
  // Packed: hi halfword should be +0 after .relu clamp.
  for (int i = 0; i < 2; ++i) {
    uint16_t hi = (hm[i] >> 16) & 0xFFFF;
    const char* label = (i == 0) ? "f16x2_relu  hi(neg->0)"
                                 : "bf16x2_relu hi(neg->0)";
    printf("cvt.%s = 0x%04x (lo=0x%04x)\n", label, hi, hm[i] & 0xFFFF);
    if (hi != 0) FAIL("packed .relu did not clamp hi to zero");
  }
  // Scalar: .relu must clamp to +0; .ftz semantics depend on source
  // denormal-ness, so we just check it executed (smoke).
  const char* sl[4] = {"f16_relu(neg)", "f16_ftz(denorm)",
                       "bf16_relu(neg)", "bf16_ftz(denorm)"};
  for (int i = 0; i < 4; ++i) {
    printf("cvt.%s = 0x%04x\n", sl[i], hms[i]);
  }
  if (hms[0] != 0) FAIL("scalar f16_relu did not clamp to +0");
  if (hms[2] != 0) FAIL("scalar bf16_relu did not clamp to +0");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: cvt.rn.f16/bf16.f32 -- FP32 -> FP16/BF16 round-trip

// Kernel: convert N float inputs to f16 and bf16.
// Layout: for each input i
//   out_f16x2[i / 2] packs (input[2i], input[2i+1]) as f16x2
//   out_bf16x2[i / 2] packs the same as bf16x2
//   out_f16[i]  single-lane cvt result
//   out_bf16[i] single-lane cvt result
__global__ void cvt_kernel(const float* in, int n,
                            uint32_t* out_f16x2, uint32_t* out_bf16x2,
                            uint16_t* out_f16,   uint16_t* out_bf16) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    float v = in[tid];
    out_f16[tid]  = cvt_f32_to_f16(v);
    out_bf16[tid] = cvt_f32_to_bf16(v);
    if ((tid & 1) == 0 && tid + 1 < n) {
        float a = in[tid], b = in[tid + 1];
        out_f16x2[tid / 2]  = cvt_f32x2_to_f16x2(a, b);
        out_bf16x2[tid / 2] = cvt_f32x2_to_bf16x2(a, b);
    }
}

__global__ void cvt_perf_kernel(const float* in, uint32_t* out, int iters) {
    int tid = threadIdx.x;
    float a = in[tid];
    float b = in[tid + 1];
    uint32_t acc = 0;
    for (int i = 0; i < iters; i++) {
        acc ^= cvt_f32x2_to_f16x2(a + i, b + i);
    }
    if (tid == 0) out[0] = acc;
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    const float h_in[] = {
        0.0f, 1.0f, -1.0f, 2.5f, -2.5f, 0.1f, -0.1f, 65504.0f,  // last is fp16 max
        3.14159f, -100.0f, 0.001f, 1e-6f, 42.0f, -42.0f, 1024.0f, -1024.0f
    };
    const int N = sizeof(h_in) / sizeof(h_in[0]);  // 16

    float    *d_in;     CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    uint32_t *d_f16x2;  CUDA_CHECK(cudaMalloc(&d_f16x2,  (N / 2) * sizeof(uint32_t)));
    uint32_t *d_bf16x2; CUDA_CHECK(cudaMalloc(&d_bf16x2, (N / 2) * sizeof(uint32_t)));
    uint16_t *d_f16;    CUDA_CHECK(cudaMalloc(&d_f16,   N * sizeof(uint16_t)));
    uint16_t *d_bf16;   CUDA_CHECK(cudaMalloc(&d_bf16,  N * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    cvt_kernel<<<1, N>>>(d_in, N, d_f16x2, d_bf16x2, d_f16, d_bf16);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_f16x2[N / 2], h_bf16x2[N / 2];
    uint16_t h_f16[N], h_bf16[N];
    CUDA_CHECK(cudaMemcpy(h_f16x2,  d_f16x2,  (N / 2) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bf16x2, d_bf16x2, (N / 2) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_f16,    d_f16,    N * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bf16,   d_bf16,   N * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    // Verify single-lane FP16 round-trip
    int bad_f16 = 0;
    for (int i = 0; i < N; i++) {
        __half hv; memcpy(&hv, &h_f16[i], 2);
        float round = __half2float(hv);
        // fp16 max is 65504, so atol of 0.01 too tight there -- use relative
        float diff = fabsf(round - h_in[i]);
        float tol = 0.01f + 0.001f * fabsf(h_in[i]);
        if (diff > tol) {
            if (bad_f16 < 3) printf("    f16 mismatch [%d]: in=%g round=%g diff=%g\n",
                                     i, h_in[i], round, diff);
            bad_f16++;
        }
    }
    if (bad_f16 == 0) printf("  cvt_f32_to_f16 (single): OK (%d values)\n", N);
    else              { printf("  cvt_f32_to_f16: FAIL (%d mismatches)\n", bad_f16); all_pass = false; }

    // Verify single-lane BF16 round-trip (wider tolerance: BF16 has 7-bit mantissa)
    int bad_bf16 = 0;
    for (int i = 0; i < N; i++) {
        __nv_bfloat16 bv; memcpy(&bv, &h_bf16[i], 2);
        float round = __bfloat162float(bv);
        float diff = fabsf(round - h_in[i]);
        float tol = 0.01f + 0.01f * fabsf(h_in[i]);  // BF16 is ~1% precision
        if (diff > tol) {
            if (bad_bf16 < 3) printf("    bf16 mismatch [%d]: in=%g round=%g diff=%g\n",
                                     i, h_in[i], round, diff);
            bad_bf16++;
        }
    }
    if (bad_bf16 == 0) printf("  cvt_f32_to_bf16 (single): OK (%d values)\n", N);
    else               { printf("  cvt_f32_to_bf16: FAIL (%d mismatches)\n", bad_bf16); all_pass = false; }

    // Verify packed f16x2: low bits = input[2i], high bits = input[2i+1]
    int bad_f16x2 = 0;
    for (int i = 0; i < N / 2; i++) {
        uint32_t packed = h_f16x2[i];
        __half lo, hi;
        uint16_t lo_bits = packed & 0xFFFF;
        uint16_t hi_bits = (packed >> 16) & 0xFFFF;
        memcpy(&lo, &lo_bits, 2);
        memcpy(&hi, &hi_bits, 2);
        float rlo = __half2float(lo), rhi = __half2float(hi);
        float tol_a = 0.01f + 0.001f * fabsf(h_in[2 * i]);
        float tol_b = 0.01f + 0.001f * fabsf(h_in[2 * i + 1]);
        if (fabsf(rlo - h_in[2 * i])     > tol_a) bad_f16x2++;
        if (fabsf(rhi - h_in[2 * i + 1]) > tol_b) bad_f16x2++;
    }
    if (bad_f16x2 == 0) printf("  cvt_f32x2_to_f16x2: OK (%d pairs)\n", N / 2);
    else                { printf("  cvt_f32x2_to_f16x2: FAIL (%d mismatches)\n", bad_f16x2); all_pass = false; }

    // Verify packed bf16x2
    int bad_bf16x2 = 0;
    for (int i = 0; i < N / 2; i++) {
        uint32_t packed = h_bf16x2[i];
        __nv_bfloat16 lo, hi;
        uint16_t lo_bits = packed & 0xFFFF;
        uint16_t hi_bits = (packed >> 16) & 0xFFFF;
        memcpy(&lo, &lo_bits, 2);
        memcpy(&hi, &hi_bits, 2);
        float rlo = __bfloat162float(lo), rhi = __bfloat162float(hi);
        float tol_a = 0.01f + 0.01f * fabsf(h_in[2 * i]);
        float tol_b = 0.01f + 0.01f * fabsf(h_in[2 * i + 1]);
        if (fabsf(rlo - h_in[2 * i])     > tol_a) bad_bf16x2++;
        if (fabsf(rhi - h_in[2 * i + 1]) > tol_b) bad_bf16x2++;
    }
    if (bad_bf16x2 == 0) printf("  cvt_f32x2_to_bf16x2: OK (%d pairs)\n", N / 2);
    else                 { printf("  cvt_f32x2_to_bf16x2: FAIL (%d mismatches)\n", bad_bf16x2); all_pass = false; }

    // Perf
    uint32_t *d_perf; CUDA_CHECK(cudaMalloc(&d_perf, 4));
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 1000; i++) cvt_perf_kernel<<<1, 32>>>(d_in, d_perf, 100);
    t.end();
    printf("  perf: %.2f us/launch (1000 launches, 100 cvts each)\n",
           t.elapsed_ms() * 1000.0f / 1000);

    cudaFree(d_in); cudaFree(d_f16x2); cudaFree(d_bf16x2); cudaFree(d_f16); cudaFree(d_bf16); cudaFree(d_perf);
    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}

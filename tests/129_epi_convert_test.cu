// ARCH: sm_90a
// 129_epi_convert_test.cu -- exercise each dtype branch of epi_pack4.
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
#include "../composites/129_epi_convert.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "129_epi_convert.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k(uint32_t* out) {
  uint32_t r0, r1;
  epi_pack4<EpiOutDtype::FP16>(1.f, 2.f, 3.f, 4.f, r0, r1);
  out[0] = r0; out[1] = r1;
  epi_pack4<EpiOutDtype::BF16>(1.f, 2.f, 3.f, 4.f, r0, r1);
  out[2] = r0; out[3] = r1;
  epi_pack4<EpiOutDtype::E4M3>(1.f, 2.f, 3.f, 4.f, r0, r1);
  out[4] = r0; out[5] = r1;
  epi_pack4<EpiOutDtype::E5M2>(1.f, 2.f, 3.f, 4.f, r0, r1);
  out[6] = r0; out[7] = r1;
}

static int run_ours() {
  /* (orig args dropped) */
  uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 32));
  k<<<1, 1>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h[8];
  CUDA_CHECK(cudaMemcpy(h, d, 32, cudaMemcpyDeviceToHost));
  cudaFree(d);
  int nz = 0; for (int i = 0; i < 8; ++i) if (h[i]) ++nz;
  printf("epi_convert : non-zero regs = %d / 8\n", nz);
  if (nz < 4) FAIL("epi_convert produced mostly zero");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: 129_epi_convert -- pure compute conversion verification.
// Tests:
//   - cvt_f32x2_pack<F16>  : FP32 -> FP16 packed
//   - cvt_f32x2_pack<BF16> : FP32 -> BF16 packed
//   - cvt_f32x2_pack_fp8<E4M3> and <E5M2>
//   - cvt_f32x8_to_4u32<F16> : 8 f32 -> 4 u32 (f16x2 pairs)
//   - cvt_f32_to_i8_sat    : saturated s8 conversion with scale

// Kernel: per-lane conversion. Each lane tid supplies 2 f32 inputs
// (in[2*tid], in[2*tid+1]) and stores converted results.
__global__ void kernel_cvt_pairs(const float* in, int n,
                                  uint32_t* out_f16,  uint32_t* out_bf16,
                                  uint16_t* out_e4m3, uint16_t* out_e5m2) {
    int tid = threadIdx.x;
    if (2 * tid + 1 >= n) return;
    float a = in[2 * tid];
    float b = in[2 * tid + 1];
    out_f16[tid]  = cvt_f32x2_pack<OutDtype::F16>(a, b);
    out_bf16[tid] = cvt_f32x2_pack<OutDtype::BF16>(a, b);
    out_e4m3[tid] = cvt_f32x2_pack_fp8<OutDtype::E4M3>(a, b);
    out_e5m2[tid] = cvt_f32x2_pack_fp8<OutDtype::E5M2>(a, b);
}

// Kernel: test cvt_f32x8_to_4u32<F16> (emits 4 u32 per lane).
__global__ void kernel_cvt_x8(const float* in_pattern, uint32_t* out_f16_4,
                               uint32_t* out_bf16_4) {
    int tid = threadIdx.x;
    float x[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) x[i] = in_pattern[8 * tid + i];
    uint32_t yf[4], yb[4];
    cvt_f32x8_to_4u32<OutDtype::F16>(x, yf);
    cvt_f32x8_to_4u32<OutDtype::BF16>(x, yb);
    #pragma unroll
    for (int i = 0; i < 4; i++) out_f16_4[tid * 4 + i]  = yf[i];
    #pragma unroll
    for (int i = 0; i < 4; i++) out_bf16_4[tid * 4 + i] = yb[i];
}

// Kernel: cvt_f32_to_i8_sat with a scale.
__global__ void kernel_cvt_i8(const float* in, int n, int8_t* out_i8, float scale) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    out_i8[tid] = cvt_f32_to_i8_sat(in[tid], scale);
}

__global__ void kernel_cvt_perf(const float* in, uint32_t* out, int iters) {
    int tid = threadIdx.x;
    float a = in[tid];
    float b = in[(tid + 1) % 32];
    uint32_t acc = 0;
    for (int i = 0; i < iters; i++) {
        acc ^= cvt_f32x2_pack<OutDtype::F16>(a + i, b + i);
    }
    if (tid == 0) out[0] = acc;
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    // Inputs: 32 floats (16 pairs).
    const float h_in[] = {
        0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 4.0f,
        -4.0f, 8.0f, -8.0f, 16.0f, -16.0f, 32.0f, -32.0f, 100.0f,
        0.25f, -0.25f, 0.125f, -0.125f, 3.14f, -3.14f, 42.0f, -42.0f,
        0.1f, -0.1f, 0.01f, 10.0f, -10.0f, 1024.0f, -1024.0f, 0.0f
    };
    const int N = sizeof(h_in) / sizeof(h_in[0]);
    const int NPAIR = N / 2;

    float *d_in;
    uint32_t *d_f16, *d_bf16;
    uint16_t *d_e4m3, *d_e5m2;
    CUDA_CHECK(cudaMalloc(&d_in,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_f16,  NPAIR * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bf16, NPAIR * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_e4m3, NPAIR * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_e5m2, NPAIR * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    kernel_cvt_pairs<<<1, NPAIR>>>(d_in, N, d_f16, d_bf16, d_e4m3, d_e5m2);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_f16[NPAIR], h_bf16[NPAIR];
    uint16_t h_e4m3[NPAIR], h_e5m2[NPAIR];
    CUDA_CHECK(cudaMemcpy(h_f16,  d_f16,  NPAIR * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bf16, d_bf16, NPAIR * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_e4m3, d_e4m3, NPAIR * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_e5m2, d_e5m2, NPAIR * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    // ---- F16 pack check ----
    int bad = 0;
    for (int i = 0; i < NPAIR; i++) {
        float a = h_in[2 * i], b = h_in[2 * i + 1];
        uint16_t lo = h_f16[i] & 0xFFFF, hi = (h_f16[i] >> 16) & 0xFFFF;
        __half ah; memcpy(&ah, &lo, 2);
        __half bh; memcpy(&bh, &hi, 2);
        float ra = __half2float(ah), rb = __half2float(bh);
        float tol_a = 0.01f + 0.001f * fabsf(a);
        float tol_b = 0.01f + 0.001f * fabsf(b);
        if (fabsf(ra - a) > tol_a || fabsf(rb - b) > tol_b) {
            if (bad < 3) printf("    f16 [%d]: (%g,%g) -> (%g,%g)\n", i, a, b, ra, rb);
            bad++;
        }
    }
    if (bad == 0) printf("  cvt_f32x2_pack<F16>: OK (%d pairs)\n", NPAIR);
    else          { printf("  cvt_f32x2_pack<F16>: FAIL (%d)\n", bad); all_pass = false; }

    // ---- BF16 pack check ----
    bad = 0;
    for (int i = 0; i < NPAIR; i++) {
        float a = h_in[2 * i], b = h_in[2 * i + 1];
        uint16_t lo = h_bf16[i] & 0xFFFF, hi = (h_bf16[i] >> 16) & 0xFFFF;
        __nv_bfloat16 ab; memcpy(&ab, &lo, 2);
        __nv_bfloat16 bb; memcpy(&bb, &hi, 2);
        float ra = __bfloat162float(ab), rb = __bfloat162float(bb);
        float tol_a = 0.02f + 0.01f * fabsf(a);
        float tol_b = 0.02f + 0.01f * fabsf(b);
        if (fabsf(ra - a) > tol_a || fabsf(rb - b) > tol_b) {
            if (bad < 3) printf("    bf16 [%d]: (%g,%g) -> (%g,%g)\n", i, a, b, ra, rb);
            bad++;
        }
    }
    if (bad == 0) printf("  cvt_f32x2_pack<BF16>: OK (%d pairs)\n", NPAIR);
    else          { printf("  cvt_f32x2_pack<BF16>: FAIL (%d)\n", bad); all_pass = false; }

    // ---- E4M3 (FP8): mantissa 3 bits, limited precision; max ~= 448 ----
    bad = 0;
    for (int i = 0; i < NPAIR; i++) {
        float a = h_in[2 * i], b = h_in[2 * i + 1];
        // Only check values within E4M3 range
        if (fabsf(a) > 440.0f || fabsf(b) > 440.0f) continue;
        uint8_t lo = (uint8_t)(h_e4m3[i] & 0xFF);
        uint8_t hi = (uint8_t)((h_e4m3[i] >> 8) & 0xFF);
        __nv_fp8_e4m3 af, bf;
        memcpy(&af, &lo, 1);
        memcpy(&bf, &hi, 1);
        float ra = (float)af;
        float rb = (float)bf;
        // E4M3 has ~10% worst-case relative error for small mantissa
        float tol_a = 0.06f + 0.13f * fabsf(a);
        float tol_b = 0.06f + 0.13f * fabsf(b);
        if (fabsf(ra - a) > tol_a || fabsf(rb - b) > tol_b) {
            if (bad < 3) printf("    e4m3 [%d]: (%g,%g) -> (%g,%g)\n", i, a, b, ra, rb);
            bad++;
        }
    }
    if (bad == 0) printf("  cvt_f32x2_pack_fp8<E4M3>: OK\n");
    else          { printf("  cvt_f32x2_pack_fp8<E4M3>: FAIL (%d)\n", bad); all_pass = false; }

    // ---- E5M2 (FP8): exponent 5 bits, 2 mantissa bits; larger range, worse precision ----
    bad = 0;
    for (int i = 0; i < NPAIR; i++) {
        float a = h_in[2 * i], b = h_in[2 * i + 1];
        if (fabsf(a) > 57000.0f || fabsf(b) > 57000.0f) continue;
        uint8_t lo = (uint8_t)(h_e5m2[i] & 0xFF);
        uint8_t hi = (uint8_t)((h_e5m2[i] >> 8) & 0xFF);
        __nv_fp8_e5m2 af, bf;
        memcpy(&af, &lo, 1);
        memcpy(&bf, &hi, 1);
        float ra = (float)af;
        float rb = (float)bf;
        // E5M2 has 25% worst-case relative error
        float tol_a = 0.15f + 0.30f * fabsf(a);
        float tol_b = 0.15f + 0.30f * fabsf(b);
        if (fabsf(ra - a) > tol_a || fabsf(rb - b) > tol_b) {
            if (bad < 3) printf("    e5m2 [%d]: (%g,%g) -> (%g,%g)\n", i, a, b, ra, rb);
            bad++;
        }
    }
    if (bad == 0) printf("  cvt_f32x2_pack_fp8<E5M2>: OK\n");
    else          { printf("  cvt_f32x2_pack_fp8<E5M2>: FAIL (%d)\n", bad); all_pass = false; }

    // ---- cvt_f32x8_to_4u32<F16> ----
    // One lane, 8 f32 inputs
    float h_x8[8] = {0.0f, 1.0f, 2.0f, 3.5f, -3.5f, 7.0f, -7.0f, 15.0f};
    float* d_x8;
    uint32_t* d_f16_4;
    uint32_t* d_bf16_4;
    CUDA_CHECK(cudaMalloc(&d_x8,      8 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_f16_4,   4 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bf16_4,  4 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_x8, h_x8, 8 * sizeof(float), cudaMemcpyHostToDevice));
    kernel_cvt_x8<<<1, 1>>>(d_x8, d_f16_4, d_bf16_4);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h_f16_4[4], h_bf16_4[4];
    CUDA_CHECK(cudaMemcpy(h_f16_4,  d_f16_4,  4 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bf16_4, d_bf16_4, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    bad = 0;
    for (int i = 0; i < 4; i++) {
        float a = h_x8[2 * i], b = h_x8[2 * i + 1];
        uint16_t lo = h_f16_4[i] & 0xFFFF, hi = (h_f16_4[i] >> 16) & 0xFFFF;
        __half ah, bh; memcpy(&ah, &lo, 2); memcpy(&bh, &hi, 2);
        if (fabsf(__half2float(ah) - a) > 0.01f + 0.001f * fabsf(a)) bad++;
        if (fabsf(__half2float(bh) - b) > 0.01f + 0.001f * fabsf(b)) bad++;
    }
    for (int i = 0; i < 4; i++) {
        float a = h_x8[2 * i], b = h_x8[2 * i + 1];
        uint16_t lo = h_bf16_4[i] & 0xFFFF, hi = (h_bf16_4[i] >> 16) & 0xFFFF;
        __nv_bfloat16 ab, bb; memcpy(&ab, &lo, 2); memcpy(&bb, &hi, 2);
        if (fabsf(__bfloat162float(ab) - a) > 0.02f + 0.01f * fabsf(a)) bad++;
        if (fabsf(__bfloat162float(bb) - b) > 0.02f + 0.01f * fabsf(b)) bad++;
    }
    if (bad == 0) printf("  cvt_f32x8_to_4u32<F16/BF16>: OK (8 values)\n");
    else          { printf("  cvt_f32x8_to_4u32: FAIL (%d)\n", bad); all_pass = false; }

    // ---- cvt_f32_to_i8_sat ----
    // Test values: out-of-range should saturate to +/-127
    const float h_i8_in[] = {0.0f, 1.0f, -1.0f, 127.0f, -128.0f,
                              200.0f, -200.0f, 126.4f, -126.4f, 42.7f,
                              1.5f, -1.5f, 127.6f, -128.6f, 64.0f};
    const int NI = sizeof(h_i8_in) / sizeof(h_i8_in[0]);
    float* d_i8_in;
    int8_t* d_i8_out;
    CUDA_CHECK(cudaMalloc(&d_i8_in,  NI * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_i8_out, NI * sizeof(int8_t)));
    CUDA_CHECK(cudaMemcpy(d_i8_in, h_i8_in, NI * sizeof(float), cudaMemcpyHostToDevice));
    kernel_cvt_i8<<<1, NI>>>(d_i8_in, NI, d_i8_out, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    int8_t h_i8_out[NI];
    CUDA_CHECK(cudaMemcpy(h_i8_out, d_i8_out, NI * sizeof(int8_t), cudaMemcpyDeviceToHost));
    bad = 0;
    for (int i = 0; i < NI; i++) {
        float v = h_i8_in[i];
        int expected;
        if (v >= 127.0f)       expected = 127;
        else if (v <= -128.0f) expected = -128;
        else                   expected = (int)rintf(v);  // rni
        if ((int)h_i8_out[i] != expected) {
            if (bad < 5) printf("    i8_sat [%d]: in=%g got=%d exp=%d\n",
                                 i, v, (int)h_i8_out[i], expected);
            bad++;
        }
    }
    // With scale=2: 64.0 * 2 = 128 should saturate to 127
    int8_t* d_i8_out2;
    CUDA_CHECK(cudaMalloc(&d_i8_out2, NI * sizeof(int8_t)));
    kernel_cvt_i8<<<1, NI>>>(d_i8_in, NI, d_i8_out2, 2.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    int8_t h_i8_out2[NI];
    CUDA_CHECK(cudaMemcpy(h_i8_out2, d_i8_out2, NI * sizeof(int8_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < NI; i++) {
        float v = h_i8_in[i] * 2.0f;
        int expected;
        if (v >= 127.0f)       expected = 127;
        else if (v <= -128.0f) expected = -128;
        else                   expected = (int)rintf(v);
        if ((int)h_i8_out2[i] != expected) {
            if (bad < 5) printf("    i8_sat scale=2 [%d]: in=%g got=%d exp=%d\n",
                                 i, h_i8_in[i], (int)h_i8_out2[i], expected);
            bad++;
        }
    }
    if (bad == 0) printf("  cvt_f32_to_i8_sat: OK (%d values, 2 scales)\n", NI);
    else          { printf("  cvt_f32_to_i8_sat: FAIL (%d)\n", bad); all_pass = false; }
    cudaFree(d_i8_in); cudaFree(d_i8_out); cudaFree(d_i8_out2);

    // ---- Perf ----
    uint32_t *d_perf; CUDA_CHECK(cudaMalloc(&d_perf, 4));
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 1000; i++) kernel_cvt_perf<<<1, 32>>>(d_in, d_perf, 100);
    t.end();
    printf("  perf: %.2f us/launch (1000 launches, 100 cvts/thread)\n",
           t.elapsed_ms() * 1000.0f / 1000);

    cudaFree(d_in); cudaFree(d_f16); cudaFree(d_bf16);
    cudaFree(d_e4m3); cudaFree(d_e5m2);
    cudaFree(d_x8); cudaFree(d_f16_4); cudaFree(d_bf16_4); cudaFree(d_perf);

    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}

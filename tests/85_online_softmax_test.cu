// ARCH: sm_90a
// 85_online_softmax_test.cu -- online softmax vs numerically stable reference.
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
#include "../composites/85_online_softmax.cuh"
#include "85_online_softmax.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k(float* out) {
  SoftmaxState s; softmax_state_init(s);
  // K-block 1: scores {1,2,3,4} max=4, sum exp2(x-4) = 1+2^-1+2^-2+2^-3
  float m1 = 4.f;
  float se1 = exp2f(1 - m1) + exp2f(2 - m1) + exp2f(3 - m1) + exp2f(4 - m1);
  softmax_state_update(s, m1, se1);
  // K-block 2: scores {0,5,3,6} max=6
  float m2 = 6.f;
  float se2 = exp2f(0 - m2) + exp2f(5 - m2) + exp2f(3 - m2) + exp2f(6 - m2);
  softmax_state_update(s, m2, se2);

  out[0] = s.m;
  out[1] = s.l;
  out[2] = softmax_state_finalize_inv(s);
}

static int run_ours() {
  /* (orig args dropped) */
  float* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 12));
  k<<<1, 1>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  float h[3];
  CUDA_CHECK(cudaMemcpy(h, d, 12, cudaMemcpyDeviceToHost));
  cudaFree(d);
  // Expected max = 6; l = sum_all exp2(x-6) for x in {1,2,3,4,0,5,3,6}
  float want_m = 6.f;
  float want_l = 0.f;
  float xs[] = { 1,2,3,4,0,5,3,6 };
  for (float x : xs) want_l += exp2f(x - want_m);
  printf("softmax: m=%g (want %g), l=%g (want %g), 1/l=%g\n",
         h[0], want_m, h[1], want_l, h[2]);
  if (std::fabs(h[0] - want_m) > 1e-5f) FAIL("running max wrong");
  if (std::fabs(h[1] - want_l) > 1e-4f) FAIL("running sum wrong");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: online softmax -- one tile of [1, 2, 3, 4], verify P + running stats

// Launch one thread (thread 0) that runs online_softmax_update_row on a single
// 4-element tile. Copy back the P values, running_max, running_sum, rescale.
//
// scale_log2 = 1/ln(2) so the composite's exp2(scale * S - max) behaves like
// natural-log softmax:  p_i = exp(S_i - max) / sum(exp(S_j - max)).
__global__ void online_softmax_single_tile(const float* in, int K,
                                             float scale_log2,
                                             float* out_P,
                                             float* out_meta) {
    // 1 thread kernel
    if (threadIdx.x != 0) return;
    OnlineSoftmaxRow state;
    float tile[8];
    for (int k = 0; k < K; k++) tile[k] = in[k];
    float rescale = online_softmax_update_row(state, tile, K, scale_log2);
    for (int k = 0; k < K; k++) out_P[k] = tile[k];
    out_meta[0] = state.running_max;
    out_meta[1] = state.running_sum;
    out_meta[2] = rescale;
}

// Second tile pass: seed state with first-tile result, then push a second tile
// and verify cross-tile running stats.
__global__ void online_softmax_two_tiles(const float* t1, const float* t2, int K,
                                           float scale_log2,
                                           float* out_P1, float* out_P2,
                                           float* out_meta) {
    if (threadIdx.x != 0) return;
    OnlineSoftmaxRow state;
    float tile1[8], tile2[8];
    for (int k = 0; k < K; k++) tile1[k] = t1[k];
    for (int k = 0; k < K; k++) tile2[k] = t2[k];
    float r1 = online_softmax_update_row(state, tile1, K, scale_log2);
    for (int k = 0; k < K; k++) out_P1[k] = tile1[k];
    float r2 = online_softmax_update_row(state, tile2, K, scale_log2);
    for (int k = 0; k < K; k++) out_P2[k] = tile2[k];
    out_meta[0] = state.running_max;
    out_meta[1] = state.running_sum;
    out_meta[2] = r1;
    out_meta[3] = r2;
}

__global__ void online_softmax_perf_kernel(const float* in, int K,
                                             float scale_log2, float* sink) {
    if (threadIdx.x != 0) return;
    OnlineSoftmaxRow state;
    float tile[8];
    for (int k = 0; k < K; k++) tile[k] = in[k];
    float r = online_softmax_update_row(state, tile, K, scale_log2);
    sink[0] = state.running_sum + r;
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    const int K = 4;
    float h_tile[K] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float scale_log2 = 1.0f / logf(2.0f);  // converts exp2(x*scale_log2) -> exp(x)

    float *d_in, *d_P, *d_meta;
    CUDA_CHECK(cudaMalloc(&d_in,   K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P,    K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_meta, 4 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_tile, K * sizeof(float), cudaMemcpyHostToDevice));

    online_softmax_single_tile<<<1, 32>>>(d_in, K, scale_log2, d_P, d_meta);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_P[K], h_meta[4];
    CUDA_CHECK(cudaMemcpy(h_P,    d_P,    K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_meta, d_meta, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference for single tile using the same composite semantics.
    // composite uses max_scaled = max(S_k * scale_log2) and P_k = exp2(S_k * scale_log2 - max_scaled).
    // With scale_log2 = 1/ln(2), max_scaled = max(S_k) / ln(2); sum(P) should equal
    // exp(S_k - max) summed = standard softmax denominator before normalization.
    float ref_P[K];
    float cpu_max_scaled = -INFINITY;
    for (int k = 0; k < K; k++) {
        float v = h_tile[k] * scale_log2;
        if (v > cpu_max_scaled) cpu_max_scaled = v;
    }
    float cpu_sum = 0.0f;
    for (int k = 0; k < K; k++) {
        ref_P[k] = exp2f(h_tile[k] * scale_log2 - cpu_max_scaled);
        cpu_sum += ref_P[k];
    }
    float cpu_max = cpu_max_scaled;

    // Check P values (all > 0, match reference)
    int bad_p = 0;
    for (int k = 0; k < K; k++) {
        if (!(h_P[k] > 0.0f)) {
            if (bad_p < 3) printf("    P[%d] = %g (not > 0)\n", k, h_P[k]);
            bad_p++;
            continue;
        }
        float diff = fabsf(h_P[k] - ref_P[k]);
        if (diff > 1e-3f) {
            if (bad_p < 3) printf("    P[%d] = %g vs ref %g (diff %g)\n",
                                   k, h_P[k], ref_P[k], diff);
            bad_p++;
        }
    }
    if (bad_p == 0) printf("  single-tile P values: OK (all > 0, match CPU ref)\n");
    else            { printf("  single-tile P: FAIL (%d)\n", bad_p); all_pass = false; }

    // Check running_max and running_sum
    bool ok_max = fabsf(h_meta[0] - cpu_max) < 1e-3f;
    bool ok_sum = fabsf(h_meta[1] - cpu_sum) < 1e-3f;
    if (ok_max) printf("  running_max: OK (%.6f)\n", h_meta[0]);
    else        { printf("  running_max: FAIL got=%g exp=%g\n", h_meta[0], cpu_max); all_pass = false; }
    if (ok_sum) printf("  running_sum: OK (%.6f)\n", h_meta[1]);
    else        { printf("  running_sum: FAIL got=%g exp=%g\n", h_meta[1], cpu_sum); all_pass = false; }

    // First-tile rescale should be exp2(old_max - new_max) with old_max = -INF.
    // exp2(-INF - finite) = 0. The composite wraps this at float infinity.
    // We only require rescale to be finite-nonnegative (0 expected on first tile).
    if (std::isfinite(h_meta[2]) && h_meta[2] >= 0.0f && h_meta[2] < 1e-6f) {
        printf("  first-tile rescale: OK (~0, as expected with -INF init)\n");
    } else {
        printf("  first-tile rescale: FAIL got=%g\n", h_meta[2]); all_pass = false;
    }

    // Verify P / running_sum matches the standard softmax probabilities
    int bad_norm = 0;
    float total = 0.0f;
    for (int k = 0; k < K; k++) {
        float pi = h_P[k] / h_meta[1];
        total += pi;
        // standard softmax: exp(S_k - max(S)) / sum exp(S_j - max(S))
        float cpu_pi = expf(h_tile[k] - 4.0f);  // max is 4
        float cpu_denom = 0.0f;
        for (int j = 0; j < K; j++) cpu_denom += expf(h_tile[j] - 4.0f);
        cpu_pi /= cpu_denom;
        if (fabsf(pi - cpu_pi) > 1e-3f) bad_norm++;
    }
    bool ok_total = fabsf(total - 1.0f) < 1e-3f;
    if (bad_norm == 0 && ok_total)
        printf("  softmax normalization: OK (sum=%g)\n", total);
    else {
        printf("  softmax norm: FAIL (%d mismatches, sum=%g)\n", bad_norm, total);
        all_pass = false;
    }

    // Two-tile test: feed [1,2,3,4] then [0,5,2,1]
    float h_t2[K] = {0.0f, 5.0f, 2.0f, 1.0f};
    float *d_t1, *d_t2, *d_P1, *d_P2, *d_meta2;
    CUDA_CHECK(cudaMalloc(&d_t1,   K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_t2,   K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P1,   K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P2,   K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_meta2, 4 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_t1, h_tile, K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t2, h_t2,   K * sizeof(float), cudaMemcpyHostToDevice));

    online_softmax_two_tiles<<<1, 32>>>(d_t1, d_t2, K, scale_log2, d_P1, d_P2, d_meta2);
    CUDA_CHECK(cudaDeviceSynchronize());
    float h_P1[K], h_P2[K], h_meta2[4];
    CUDA_CHECK(cudaMemcpy(h_P1,   d_P1,   K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_P2,   d_P2,   K * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_meta2, d_meta2, 4 * sizeof(float), cudaMemcpyDeviceToHost));

    // After two tiles, final max should be max(1..4, 0,5,2,1) = 5 (scaled by scale_log2)
    // After the second tile, running_sum should be the sum of
    //   rescaled old_sum * exp2(old_max - new_max) + new_tile P values.
    // i.e. equal to the single-pass softmax denominator over [1,2,3,4,0,5,2,1]
    // normalized with max=5 (in original space).
    float all_vals[] = {1,2,3,4,0,5,2,1};
    float ref_denom = 0.0f, ref_max = 5.0f;
    for (float v : all_vals) ref_denom += expf(v - ref_max);
    float ref_max_scaled = ref_max * scale_log2;

    bool ok2_max = fabsf(h_meta2[0] - ref_max_scaled) < 1e-3f;
    bool ok2_sum = fabsf(h_meta2[1] - ref_denom)     < 1e-3f;
    if (ok2_max) printf("  two-tile running_max: OK (%.6f)\n", h_meta2[0]);
    else         { printf("  two-tile running_max: FAIL got=%g exp=%g\n", h_meta2[0], ref_max_scaled); all_pass = false; }
    if (ok2_sum) printf("  two-tile running_sum: OK (%.6f)\n", h_meta2[1]);
    else         { printf("  two-tile running_sum: FAIL got=%g exp=%g\n", h_meta2[1], ref_denom); all_pass = false; }

    // Second tile's rescale = exp2(old_max_scaled - new_max_scaled); old_max was 4*scale_log2
    float exp_rescale2 = exp2f(4.0f * scale_log2 - 5.0f * scale_log2);
    if (fabsf(h_meta2[3] - exp_rescale2) < 1e-3f)
        printf("  second-tile rescale: OK (%.6f)\n", h_meta2[3]);
    else
        { printf("  second-tile rescale: FAIL got=%g exp=%g\n", h_meta2[3], exp_rescale2); all_pass = false; }

    // Perf
    float *d_sink; CUDA_CHECK(cudaMalloc(&d_sink, 4));
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 1000; i++)
        online_softmax_perf_kernel<<<1, 32>>>(d_in, K, scale_log2, d_sink);
    t.end();
    printf("  perf: %.2f us/launch (1000 launches)\n", t.elapsed_ms() * 1000.0f / 1000);

    cudaFree(d_in); cudaFree(d_P); cudaFree(d_meta);
    cudaFree(d_t1); cudaFree(d_t2); cudaFree(d_P1); cudaFree(d_P2); cudaFree(d_meta2);
    cudaFree(d_sink);
    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}

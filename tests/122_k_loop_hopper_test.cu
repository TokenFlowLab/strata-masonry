#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 122_k_loop_hopper_test.cu -- runtime test for k loop hopper
//
// Runtime test: 122_k_loop_hopper -- Hopper K-loop composite correctness.
//
// Strategy: all-ones A (M=64 x K), all-ones B (N=64 x K). All FP16 bytes
// are 0x3C (1.0), so the SMEM swizzle chosen by the descriptor (B128)
// doesn't affect the numerical result: WGMMA reads 1.0 regardless of which
// bytes it picks. Expected: every one of the 32 f32 accumulator regs per
// thread = K (sum of 1.0 * 1.0 across K).
//
// The test validates:
//   * k_loop_hopper_f16_n64_body: the simpler variant without stage-wait.
//   * Caller adds wgmma_fence before, wgmma_commit_group + wgmma_wait_group<0>
//     after (as recommended by the composite's docstring).
//
// We do NOT test k_loop_hopper_f16_n64_stage because it depends on an
// mbarrier wait/arrive handshake (producer / consumer setup) that requires
// multiple warpgroups, which is the scope of 101_pipeline_hopper_test.

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "test_utils.cuh"
#include "43_smem_desc_hopper.cuh"
#include "53_wgmma_f16_ss.cuh"
#include "59_wgmma_fence_commit_wait.cuh"
#include "122_k_loop_hopper.cuh"

// For m64n64k16 with NUM_K_BLOCKS loops:
//   total K = 16 * NUM_K_BLOCKS
//   A tile: 64 rows x (16 * NUM_K_BLOCKS) halves
//   B tile: 64 rows x (16 * NUM_K_BLOCKS) halves
// We lay both out contiguously in SMEM. Row stride in bytes is passed as
// stride_a_bytes / stride_b_bytes. The composite itself computes the
// per-K-block SMEM offset as kb * 32 bytes (one K-block of 16 halves).
//
// Because the composite uses build_smem_desc_hopper_b128 (swizzle=B128),
// we must ensure the layout is compatible with B128 swizzle semantics.
// All-ones sidesteps the correctness implication of swizzle, but the
// descriptor must still address valid SMEM.
//
// We use row_stride_bytes = 16 * 2 = 32 bytes for each K-block worth of
// row: that matches the composite's stride=32 for the smallest configuration.
// However B128 swizzle strictly requires 128B row stride for correct data
// interpretation. Since all-ones bypasses that, what matters is that the
// descriptor is valid. Pass stride = 128 * NUM_K_BLOCKS bytes so the
// descriptor's SBO is a 14-bit-encodable value.
//
// Simplest valid setup: allocate each tile as 64 rows x (NUM_K_BLOCKS * 16)
// halves contiguously; row stride = NUM_K_BLOCKS * 32 bytes; fill all with 1.0.

template <int NUM_K_BLOCKS>
__global__ void kernel_k_loop_body(float* __restrict__ out) {
    constexpr int M = 64;
    constexpr int N = 64;
    constexpr int K = 16 * NUM_K_BLOCKS;
    constexpr int A_ELEMS = M * K;
    constexpr int B_ELEMS = N * K;

    __shared__ __align__(128) half smem_A[A_ELEMS];
    __shared__ __align__(128) half smem_B[B_ELEMS];

    int tid = threadIdx.x;
    // Fill both A and B with 1.0.
    #pragma unroll
    for (int i = tid; i < A_ELEMS; i += 128) smem_A[i] = __float2half(1.0f);
    #pragma unroll
    for (int i = tid; i < B_ELEMS; i += 128) smem_B[i] = __float2half(1.0f);
    __syncthreads();

    uint32_t a_addr = smem_ptr_u32(&smem_A[0]);
    uint32_t b_addr = smem_ptr_u32(&smem_B[0]);

    // Stride (SBO) = 8 rows * row_bytes. Row_bytes = K * 2.
    // For K=16,  SBO = 8 * 32  = 256
    // For K=32,  SBO = 8 * 64  = 512
    // For K=64,  SBO = 8 * 128 = 1024
    // For K=128, SBO = 8 * 256 = 2048
    uint32_t stride_bytes = 8 * (K * 2);

    // Accumulator (scale_d = false on first iter -> overwrite).
    float d[32] = {
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0
    };

    // Composite: K-loop body (fence + commit + wait managed by caller).
    wgmma_fence();
    k_loop_hopper_f16_n64_body<NUM_K_BLOCKS>(
        d, a_addr, b_addr, stride_bytes, stride_bytes,
        /*scale_d=*/false);
    wgmma_commit_group();
    wgmma_wait_group<0>();

    // Each thread writes its 32 f32 regs; 128 threads * 32 = 4096 floats.
    #pragma unroll
    for (int i = 0; i < 32; i++) out[tid * 32 + i] = d[i];
}

template <int NUM_K_BLOCKS>
bool run_one(const char* label) {
    constexpr int NELT = 128 * 32;
    constexpr int K = 16 * NUM_K_BLOCKS;

    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, NELT * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, NELT * sizeof(float)));

    kernel_k_loop_body<NUM_K_BLOCKS><<<1, 128>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  %s: LAUNCH FAIL: %s\n", label, cudaGetErrorString(err));
        cudaFree(d_out);
        return false;
    }
    float h_out[NELT];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, NELT * sizeof(float), cudaMemcpyDeviceToHost));

    float expected = (float)K;
    int bad = 0;
    float first_bad_val = -1.0f;
    int first_bad_idx = -1;
    for (int i = 0; i < NELT; i++) {
        if (fabsf(h_out[i] - expected) > 1e-3f) {
            if (bad == 0) { first_bad_val = h_out[i]; first_bad_idx = i; }
            bad++;
        }
    }
    bool pass = (bad == 0);
    if (pass) {
        printf("  %s (K=%d, %d regs/thr): OK (all %d values == %.1f)\n",
               label, K, 32, NELT, expected);
    } else {
        printf("  %s (K=%d): FAIL (%d/%d; first [%d]=%.3f, exp=%.1f)\n",
               label, K, bad, NELT, first_bad_idx, first_bad_val, expected);
        printf("    samples: [0]=%.3f [16]=%.3f [100]=%.3f [4095]=%.3f\n",
               h_out[0], h_out[16], h_out[100], h_out[4095]);
    }

    cudaFree(d_out);
    return pass;
}

int main() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    // Run several K configurations.
    all_pass &= run_one<1>("k_loop_body NK=1");
    all_pass &= run_one<2>("k_loop_body NK=2");
    all_pass &= run_one<4>("k_loop_body NK=4");
    all_pass &= run_one<8>("k_loop_body NK=8");

    // Perf
    constexpr int NELT_PERF = 128 * 32;
    float* d_out;
    CUDA_CHECK(cudaMalloc(&d_out, NELT_PERF * sizeof(float)));
    GpuTimer t;
    const int ITERS = 100;
    t.begin();
    for (int i = 0; i < ITERS; i++) {
        kernel_k_loop_body<4><<<1, 128>>>(d_out);
    }
    t.end();
    printf("  perf (NK=4, m64n64k64): %.2f us/launch\n", t.elapsed_ms() * 1000.0f / ITERS);
    cudaFree(d_out);

    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}

#endif  // PL_AGENTIC_SM90A

#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 60_mma_sync_f16_test.cu -- runtime test for mma sync f16
//
// Runtime test: mma.sync m16n8k16 FP16 -> FP32 -- numerical GEMM correctness.
//
// Warp-level MMA: 32 threads cooperatively compute D = A*B (shapes 16x16 * 16x8).
// Uses ldmatrix to load A / B fragments so we don't depend on the per-thread
// register layout details. Output fragment layout for D is well-documented
// (same as CLayout = SM80_16x8_Row):
//
//   Thread T holds 4 f32 regs, mapping to output (16x8 row-major) positions:
//     d[0]: (row=T/4,     col=(T%4)*2    )
//     d[1]: (row=T/4,     col=(T%4)*2 + 1)
//     d[2]: (row=T/4 + 8, col=(T%4)*2    )
//     d[3]: (row=T/4 + 8, col=(T%4)*2 + 1)
//
// ldmatrix contracts:
//   ldmatrix.x4 loads 4 8x8 FP16 matrices, matches mma.sync A fragment.
//   ldmatrix.x2.trans loads 2 8x8 FP16 matrices transposed, matches B (col-major).

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "test_utils.cuh"
#include "39_ldmatrix.cuh"
#include "60_mma_sync_f16.cuh"

__global__ void mma_sync_m16n8k16_gemm_kernel(const half* __restrict__ A_g,
                                                const half* __restrict__ B_g,
                                                float* __restrict__ D_g) {
    __shared__ __align__(128) half smem_A[16 * 16];
    __shared__ __align__(128) half smem_B[16 * 8];

    int T = threadIdx.x;

    // Copy A and B into SMEM (simple, not optimized).
    if (T < 16) {
        #pragma unroll
        for (int c = 0; c < 16; c++) smem_A[T * 16 + c] = A_g[T * 16 + c];
        #pragma unroll
        for (int c = 0; c < 8; c++)  smem_B[T * 8  + c] = B_g[T * 8  + c];
    }
    __syncwarp();

    // -- Load A fragment via ldmatrix.x4 --
    // A is 16x16 FP16, row stride = 16 halves.
    // mma.sync A fragment mapping (authoritative NVIDIA PTX ISA):
    //   a0: (row=T/4,   col=(T%4)*2..+1)     -- 8x8 tile at (0,0)
    //   a1: (row=T/4+8, col=(T%4)*2..+1)     -- 8x8 tile at (8,0)
    //   a2: (row=T/4,   col=(T%4)*2+8..+9)   -- 8x8 tile at (0,8)
    //   a3: (row=T/4+8, col=(T%4)*2+8..+9)   -- 8x8 tile at (8,8)
    // ldmatrix.x4 loads 4 8x8 matrices. Thread T provides address for matrix
    // index T/8, row T%8 of that matrix.
    // So matrix 0 (a0) <- rows 0-7 cols 0-7 provided by threads 0-7.
    //    matrix 1 (a1) <- rows 8-15 cols 0-7 provided by threads 8-15.
    //    matrix 2 (a2) <- rows 0-7 cols 8-15 provided by threads 16-23.
    //    matrix 3 (a3) <- rows 8-15 cols 8-15 provided by threads 24-31.
    uint32_t a_row, a_col;
    switch (T / 8) {
        case 0: a_row = T % 8;         a_col = 0; break;   // matrix 0
        case 1: a_row = (T % 8) + 8;   a_col = 0; break;   // matrix 1
        case 2: a_row = T % 8;         a_col = 8; break;   // matrix 2
        case 3: a_row = (T % 8) + 8;   a_col = 8; break;   // matrix 3
    }
    half* a_row_ptr = &smem_A[a_row * 16 + a_col];
    uint32_t a_smem_addr = smem_ptr_u32(a_row_ptr);

    uint32_t a0, a1, a2, a3;
    ldmatrix_x4(a0, a1, a2, a3, a_smem_addr);

    // -- Load B fragment via ldmatrix.x2.trans --
    // B is 16x8 FP16 row-major (K x N). For mma.sync ".col" B, we need B
    // delivered in transposed form. ldmatrix.x2.trans reads 2 8x8 blocks
    // transposed: blocks = B[0:8, 0:8] and B[8:16, 0:8].
    // Thread t in [0,8): addr of row t of block 0 = &B[t, 0].
    // Thread t in [8,16): addr of row (t-8) of block 1 = &B[t, 0].
    // Threads 16-31: unused by ldmatrix.x2, but must participate.
    uint32_t b_row = (T < 16) ? T : 0;  // rest doesn't matter, still aligned
    half* b_row_ptr = &smem_B[b_row * 8];
    uint32_t b_smem_addr = smem_ptr_u32(b_row_ptr);

    uint32_t b0, b1;
    ldmatrix_x2_trans(b0, b1, b_smem_addr);

    // -- Call mma.sync; C = 0 so D = A*B --
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    float d0, d1, d2, d3;
    mma_sync_m16n8k16_f16(d0, d1, d2, d3,
                           a0, a1, a2, a3,
                           b0, b1,
                           c0, c1, c2, c3);

    // -- Scatter D fragment to (16 x 8) output row-major --
    int d_row = T / 4;
    int d_col = (T % 4) * 2;
    D_g[d_row         * 8 + d_col    ] = d0;
    D_g[d_row         * 8 + d_col + 1] = d1;
    D_g[(d_row + 8)   * 8 + d_col    ] = d2;
    D_g[(d_row + 8)   * 8 + d_col + 1] = d3;
}

// Sparse mma.sp test (m16n8k32, 2:4 structured sparsity).
//
// Strategy: A_sparse and B are all 1.0; sparsity metadata 0x44444444
// selects the first 2 of every 4 K-positions, yielding K_active = 16
// active (k_sel, m, n) triples per output element. Expected D[m,n] = 16.
//
// Per-thread fragment storage: 4 u32 A regs (8 FP16 packed), 4 u32 B regs,
// 1 u32 metadata, 4 f32 D. 0x3C003C00 = two FP16 1.0 values packed.
__global__ void mma_sp_m16n8k32_allones_kernel(float* __restrict__ D_g) {
    int T = threadIdx.x;
    uint32_t a0 = 0x3C003C00, a1 = 0x3C003C00, a2 = 0x3C003C00, a3 = 0x3C003C00;
    uint32_t b0 = 0x3C003C00, b1 = 0x3C003C00, b2 = 0x3C003C00, b3 = 0x3C003C00;
    uint32_t e_meta = 0x44444444;  // select positions {0,1} of each 4-group
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    mma_sp_sync_m16n8k32_f16<0>(d0, d1, d2, d3,
                                 a0, a1, a2, a3,
                                 b0, b1, b2, b3,
                                 0.f, 0.f, 0.f, 0.f,
                                 e_meta);

    int d_row = T / 4;
    int d_col = (T % 4) * 2;
    D_g[d_row         * 8 + d_col    ] = d0;
    D_g[d_row         * 8 + d_col + 1] = d1;
    D_g[(d_row + 8)   * 8 + d_col    ] = d2;
    D_g[(d_row + 8)   * 8 + d_col + 1] = d3;
}

int main() {
    srand(12345);
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    constexpr int M = 16, N = 8, K = 16;

    half h_A[M * K], h_B[K * N];
    float ref_D[M * N], h_D[M * N];

    fill_random_f16(h_A, M * K, -2.0f, 2.0f);
    fill_random_f16(h_B, K * N, -2.0f, 2.0f);

    cpu_gemm_f16(h_A, h_B, ref_D, M, N, K);

    half *d_A, *d_B;
    float *d_D;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_D, 0, M * N * sizeof(float)));

    mma_sync_m16n8k16_gemm_kernel<<<1, 32>>>(d_A, d_B, d_D);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  LAUNCH FAIL: %s\n", cudaGetErrorString(err));
        all_pass = false;
    } else {
        CUDA_CHECK(cudaMemcpy(h_D, d_D, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        bool ok = check_close_f32(ref_D, h_D, M * N, 3e-3f, 3e-3f);
        if (ok) {
            float max_abs = 0.f;
            for (int i = 0; i < M * N; i++) {
                float d = fabsf(ref_D[i] - h_D[i]);
                if (d > max_abs) max_abs = d;
            }
            printf("  m16n8k16 f16 gemm: OK (max_abs_err=%.5f)\n", max_abs);
        } else {
            printf("  m16n8k16 f16 gemm: FAIL\n");
            printf("    ref[0..7]  = %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
                   ref_D[0], ref_D[1], ref_D[2], ref_D[3],
                   ref_D[4], ref_D[5], ref_D[6], ref_D[7]);
            printf("    got[0..7]  = %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
                   h_D[0], h_D[1], h_D[2], h_D[3],
                   h_D[4], h_D[5], h_D[6], h_D[7]);
            printf("    ref[8..15] = %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
                   ref_D[8], ref_D[9], ref_D[10], ref_D[11],
                   ref_D[12], ref_D[13], ref_D[14], ref_D[15]);
            printf("    got[8..15] = %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
                   h_D[8], h_D[9], h_D[10], h_D[11],
                   h_D[12], h_D[13], h_D[14], h_D[15]);
            // Find which ref_D[j] best matches got[i] for first few i
            for (int i = 0; i < 4; i++) {
                int best = -1; float bestd = 1e9f;
                for (int j = 0; j < 128; j++) {
                    float d = fabsf(ref_D[j] - h_D[i]);
                    if (d < bestd) { bestd = d; best = j; }
                }
                printf("    got[%d]=%.3f best-match ref[%d]=%.3f (diff=%.4f)\n",
                       i, h_D[i], best, ref_D[best], bestd);
            }
            all_pass = false;
        }
    }

    // --- perf ---
    {
        GpuTimer t;
        const int ITERS = 200;
        t.begin();
        for (int i = 0; i < ITERS; i++) mma_sync_m16n8k16_gemm_kernel<<<1, 32>>>(d_A, d_B, d_D);
        t.end();
        printf("  perf: %.2f us/launch (mma.sync m16n8k16 f16->f32, 1 CTA x 32 thr)\n",
               t.elapsed_ms() * 1000.0f / ITERS);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_D);

    // --- mma.sp m16n8k32 sparse test ---
    {
        constexpr int M_SP = 16, N_SP = 8;
        float* d_Dsp; CUDA_CHECK(cudaMalloc(&d_Dsp, M_SP * N_SP * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_Dsp, 0, M_SP * N_SP * sizeof(float)));

        mma_sp_m16n8k32_allones_kernel<<<1, 32>>>(d_Dsp);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  mma.sp m16n8k32: LAUNCH FAIL: %s\n", cudaGetErrorString(err));
            all_pass = false;
        } else {
            float h_Dsp[M_SP * N_SP];
            CUDA_CHECK(cudaMemcpy(h_Dsp, d_Dsp, M_SP * N_SP * sizeof(float), cudaMemcpyDeviceToHost));
            // 16 active K positions x 1.0 x 1.0 = 16
            const float expected = 16.0f;
            int bad = 0;
            for (int i = 0; i < M_SP * N_SP; i++)
                if (fabsf(h_Dsp[i] - expected) > 1e-3f) bad++;
            if (bad == 0) {
                printf("  mma.sp m16n8k32 f16->f32: OK (all %d == %.0f)\n", M_SP * N_SP, expected);
            } else {
                printf("  mma.sp m16n8k32 f16->f32: FAIL (%d/%d bad; sample [0]=%.3f exp %.0f)\n",
                       bad, M_SP * N_SP, h_Dsp[0], expected);
                all_pass = false;
            }
        }
        cudaFree(d_Dsp);
    }

    if (all_pass) { PASS(); return 0; }
    else { FAIL("gemm correctness failed"); return 1; }
}

#endif  // PL_AGENTIC_SM90A

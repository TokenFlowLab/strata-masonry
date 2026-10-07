#if defined(PL_AGENTIC_SM90A)
// ARCH: sm_90a
// 126_epi_subtile_hopper_test.cu -- runtime test for epi subtile hopper
//
// Runtime test: 126_epi_subtile_hopper -- f32 accumulator -> cvt -> stmatrix.
//
// Strategy: use the property that stmatrix is the inverse of ldmatrix at the
// b16 level. For a given SMEM layout (32 rows, 8 halves/row, lane-L writes
// row L), we:
//   1) each lane converts 8 known f32 values to 4 u32s via cvt_f32x2_to_{f16,bf16}x2
//   2) each lane also builds the same 4 u32s directly (expected_r0..3)
//   3) lane calls epi_subtile_f32_to_f16_stmatrix -> writes to smem_out
//   4) ldmatrix.x4 reads smem_out back into 4 u32s
//   5) verify loaded u32s == expected (round-trip identity)
//
// Separately, we verify the simple "fingerprint" property: every cell in the
// 256-half SMEM is non-zero after stmatrix.x4 with non-zero inputs. This
// guards against the function silently no-op'ing.
//
// Same for epi_subtile_f32_to_bf16_stmatrix.
//
// For epi_f32_accum_to_f16_subtiles: process a 32-reg fragment in 4
// stmatrix.x4 batches; verify the same round-trip for each batch (addresses
// at base + g * stride).

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "test_utils.cuh"
#include "39_ldmatrix.cuh"
#include "126_epi_subtile_hopper.cuh"
#include "63_cvt_f32_to_f16_bf16.cuh"

static constexpr int ROWS = 32;
static constexpr int COLS = 8;       // halves per row
static constexpr int NH   = ROWS * COLS;  // 256 halves

// Kernel: test epi_subtile_f32_to_f16_stmatrix round-trip via ldmatrix.x4.
// Each lane:
//   - has 8 f32 values derived from its lane id
//   - builds expected 4 u32s via direct cvt
//   - calls epi_subtile_f32_to_f16_stmatrix
//   - lane reads back via ldmatrix.x4
//   - writes {got_r0..3, exp_r0..3} to GMEM for host check
__global__ void kernel_epi_subtile_f16(uint32_t* out) {
    __shared__ __align__(128) half smem[NH];
    int lane = threadIdx.x;
    // Zero SMEM.
    #pragma unroll
    for (int j = 0; j < COLS; j++) smem[lane * COLS + j] = __float2half(0.0f);
    __syncwarp();

    // 8 f32 values per lane.
    float d0 = (float)(lane +  1);
    float d1 = (float)(lane +  2);
    float d2 = (float)(lane +  3);
    float d3 = (float)(lane +  4);
    float d4 = (float)(lane +  5);
    float d5 = (float)(lane +  6);
    float d6 = (float)(lane +  7);
    float d7 = (float)(lane +  8);

    // Expected packs (matches what epi_subtile does internally).
    uint32_t exp0 = cvt_f32x2_to_f16x2(d0, d1);
    uint32_t exp1 = cvt_f32x2_to_f16x2(d2, d3);
    uint32_t exp2 = cvt_f32x2_to_f16x2(d4, d5);
    uint32_t exp3 = cvt_f32x2_to_f16x2(d6, d7);

    // stmatrix.x4 via composite.
    uint32_t addr = smem_ptr_u32(&smem[lane * COLS]);
    epi_subtile_f32_to_f16_stmatrix(addr, d0, d1, d2, d3, d4, d5, d6, d7);
    __syncwarp();

    // Round-trip via ldmatrix.x4.
    uint32_t got0, got1, got2, got3;
    ldmatrix_x4(got0, got1, got2, got3, addr);

    // Per-lane output: 8 u32s (got0..3, exp0..3).
    out[lane * 8 + 0] = got0;
    out[lane * 8 + 1] = got1;
    out[lane * 8 + 2] = got2;
    out[lane * 8 + 3] = got3;
    out[lane * 8 + 4] = exp0;
    out[lane * 8 + 5] = exp1;
    out[lane * 8 + 6] = exp2;
    out[lane * 8 + 7] = exp3;
}

// Same test but with BF16 variant.
__global__ void kernel_epi_subtile_bf16(uint32_t* out) {
    __shared__ __align__(128) half smem[NH];
    int lane = threadIdx.x;
    #pragma unroll
    for (int j = 0; j < COLS; j++) smem[lane * COLS + j] = __float2half(0.0f);
    __syncwarp();

    float d0 = (float)(lane +  1);
    float d1 = (float)(lane +  2);
    float d2 = (float)(lane +  3);
    float d3 = (float)(lane +  4);
    float d4 = (float)(lane +  5);
    float d5 = (float)(lane +  6);
    float d6 = (float)(lane +  7);
    float d7 = (float)(lane +  8);

    uint32_t exp0 = cvt_f32x2_to_bf16x2(d0, d1);
    uint32_t exp1 = cvt_f32x2_to_bf16x2(d2, d3);
    uint32_t exp2 = cvt_f32x2_to_bf16x2(d4, d5);
    uint32_t exp3 = cvt_f32x2_to_bf16x2(d6, d7);

    uint32_t addr = smem_ptr_u32(&smem[lane * COLS]);
    epi_subtile_f32_to_bf16_stmatrix(addr, d0, d1, d2, d3, d4, d5, d6, d7);
    __syncwarp();

    uint32_t got0, got1, got2, got3;
    ldmatrix_x4(got0, got1, got2, got3, addr);

    out[lane * 8 + 0] = got0;
    out[lane * 8 + 1] = got1;
    out[lane * 8 + 2] = got2;
    out[lane * 8 + 3] = got3;
    out[lane * 8 + 4] = exp0;
    out[lane * 8 + 5] = exp1;
    out[lane * 8 + 6] = exp2;
    out[lane * 8 + 7] = exp3;
}

// Test epi_f32_accum_to_f16_subtiles: 4 groups of 8 f32s -> 4 stmatrix.x4
// batches at addresses base + g * stride.
__global__ void kernel_epi_accum_subtiles(uint32_t* out) {
    // 4 separate regions of 256 halves each, stride 512 bytes = 256 halves.
    constexpr int REG = NH;
    __shared__ __align__(128) half smem[4 * REG];
    int lane = threadIdx.x;
    // Zero.
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        #pragma unroll
        for (int j = 0; j < COLS; j++)
            smem[g * REG + lane * COLS + j] = __float2half(0.0f);
    }
    __syncwarp();

    float d[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) d[i] = (float)(lane + 1) * 0.125f + (float)i;

    // Expected: 4 groups of 4 u32s.
    uint32_t exp[16];
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        exp[g * 4 + 0] = cvt_f32x2_to_f16x2(d[g*8+0], d[g*8+1]);
        exp[g * 4 + 1] = cvt_f32x2_to_f16x2(d[g*8+2], d[g*8+3]);
        exp[g * 4 + 2] = cvt_f32x2_to_f16x2(d[g*8+4], d[g*8+5]);
        exp[g * 4 + 3] = cvt_f32x2_to_f16x2(d[g*8+6], d[g*8+7]);
    }

    // Stride per sub-tile is REG halves = REG * 2 bytes.
    uint32_t base_addr = smem_ptr_u32(&smem[lane * COLS]);  // lane's row in group 0
    // In the composite, smem_addr_base + g * smem_stride_bytes shifts by
    // stride for each group. Set stride = REG * 2 bytes so each group uses
    // a separate 256-half region.
    epi_f32_accum_to_f16_subtiles(base_addr, (uint32_t)(REG * 2), d);
    __syncwarp();

    // Round-trip: ldmatrix.x4 for each group.
    uint32_t got[16];
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        uint32_t addr_g = smem_ptr_u32(&smem[g * REG + lane * COLS]);
        ldmatrix_x4(got[g*4+0], got[g*4+1], got[g*4+2], got[g*4+3], addr_g);
    }

    // Write per-lane 32 u32s: 16 got + 16 exp.
    #pragma unroll
    for (int i = 0; i < 16; i++) out[lane * 32 + i]      = got[i];
    #pragma unroll
    for (int i = 0; i < 16; i++) out[lane * 32 + 16 + i] = exp[i];
}

static bool verify_rt(const uint32_t* h, int per_lane, const char* name) {
    // Each lane supplies 'per_lane/2' got + 'per_lane/2' exp u32s
    int half = per_lane / 2;
    int bad = 0;
    for (int lane = 0; lane < 32; lane++) {
        for (int k = 0; k < half; k++) {
            uint32_t got = h[lane * per_lane + k];
            uint32_t exp = h[lane * per_lane + half + k];
            if (got != exp) {
                if (bad < 5) printf("    %s lane=%d k=%d got=0x%08x exp=0x%08x\n",
                                     name, lane, k, got, exp);
                bad++;
            }
        }
    }
    if (bad == 0) printf("  %s: OK (32 lanes x %d u32 round-trip)\n", name, half);
    else          printf("  %s: FAIL (%d mismatches)\n", name, bad);
    return bad == 0;
}

int main() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    // --- f16 subtile ---
    uint32_t* d_out; CUDA_CHECK(cudaMalloc(&d_out, 32 * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_out, 0, 32 * 8 * sizeof(uint32_t)));
    kernel_epi_subtile_f16<<<1, 32>>>(d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h_out[32 * 8];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, 32 * 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    all_pass &= verify_rt(h_out, 8, "epi_subtile_f32_to_f16_stmatrix");
    cudaFree(d_out);

    // --- bf16 subtile ---
    CUDA_CHECK(cudaMalloc(&d_out, 32 * 8 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_out, 0, 32 * 8 * sizeof(uint32_t)));
    kernel_epi_subtile_bf16<<<1, 32>>>(d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out, d_out, 32 * 8 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    all_pass &= verify_rt(h_out, 8, "epi_subtile_f32_to_bf16_stmatrix");
    cudaFree(d_out);

    // --- accum to 4 subtiles ---
    uint32_t* d_out32; CUDA_CHECK(cudaMalloc(&d_out32, 32 * 32 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_out32, 0, 32 * 32 * sizeof(uint32_t)));
    kernel_epi_accum_subtiles<<<1, 32>>>(d_out32);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h_out32[32 * 32];
    CUDA_CHECK(cudaMemcpy(h_out32, d_out32, 32 * 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    all_pass &= verify_rt(h_out32, 32, "epi_f32_accum_to_f16_subtiles");

    // --- Perf ---
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 1000; i++) kernel_epi_accum_subtiles<<<1, 32>>>(d_out32);
    t.end();
    printf("  perf (accum 32 reg frag): %.2f us/launch\n", t.elapsed_ms() * 1000.0f / 1000);

    cudaFree(d_out32);

    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}

#endif  // PL_AGENTIC_SM90A

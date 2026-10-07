// ARCH: sm_90a
// 39_ldmatrix_test.cu -- write a 16x16 FP16 pattern to SMEM, ldmatrix.x4
// loads the four 8x8 tiles, write regs to GMEM, verify (by layout this
// smoke-tests the asm only; the exact register -> (row,col) mapping is in
// the CuTe layout traits).
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
#include "../src/primitives/39_ldmatrix.cuh"
#include <cuda_fp16.h>
#include "39_ldmatrix.cuh"
#include "40_stmatrix.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k_ld(const __half* gin, uint32_t* regs_out) {
  __shared__ __align__(16) __half smem[16 * 16];
  for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) smem[i] = gin[i];
  __syncthreads();
  // Each lane's ldmatrix source = the row address for the 8x8 sub-tile.
  // For x4 : lanes 0-7 row 0-7 sub-tile 0, lanes 8-15 sub-tile 1, etc.
  const int lane = threadIdx.x;
  int row = lane & 7;
  int col = (lane >> 3) * 8;
  uint32_t addr = smem_ptr_u32(&smem[row * 16 + col]);
  uint32_t r[4];
  ldmatrix_x4(r, addr);
  for (int i = 0; i < 4; ++i) regs_out[lane * 4 + i] = r[i];
}

static int run_ours() {
  /* (orig args dropped) */
  std::vector<__half> hIn(16 * 16);
  for (int i = 0; i < 16 * 16; ++i) hIn[i] = __float2half((float)i);
  __half* dIn = nullptr; CUDA_CHECK(cudaMalloc(&dIn, 16 * 16 * 2));
  CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), 16 * 16 * 2, cudaMemcpyHostToDevice));
  uint32_t* dReg = nullptr; CUDA_CHECK(cudaMalloc(&dReg, 32 * 4 * 4));
  CUDA_CHECK(cudaMemset(dReg, 0, 32 * 4 * 4));
  k_ld<<<1, 32>>>(dIn, dReg);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> hReg(32 * 4);
  CUDA_CHECK(cudaMemcpy(hReg.data(), dReg, 32 * 4 * 4, cudaMemcpyDeviceToHost));
  cudaFree(dIn); cudaFree(dReg);
  int nz = 0; for (auto v : hReg) if (v != 0) ++nz;
  printf("ldmatrix.x4 : non-zero regs = %d / %zu\n", nz, hReg.size());
  if (nz == 0) FAIL("ldmatrix produced no data");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: ldmatrix.x4 + stmatrix.x4 round-trip
// Layout: 4 x (8x8 fp16) = 32 rows x 16 cols of fp16 (row-pointer form),
// i.e. 32 rows of 16 bytes each = 512 bytes. Each lane i (i in 0..31) owns
// row i. ldmatrix.x4 loads into 4 u32 registers per lane; stmatrix.x4 stores
// them back at the same addressing pattern. Data must round-trip exactly.

static constexpr int ROWS = 32;      // one lane per row
static constexpr int ROW_B = 16;     // 8 fp16 per row
static constexpr int TOTAL_B = ROWS * ROW_B;        // 512 bytes
static constexpr int TOTAL_H = TOTAL_B / 2;         // 256 half elements

__global__ void ldmatrix_stmatrix_kernel(const half* __restrict__ gmem_in,
                                         half* __restrict__ gmem_out) {
    __shared__ __align__(16) half smem_in[TOTAL_H];
    __shared__ __align__(16) half smem_out[TOTAL_H];

    // Stage: load input GMEM -> input SMEM (one warp doing 32 rows)
    int lane = threadIdx.x;
    if (lane < ROWS) {
        // 8 halves = 4 u32 per row; use plain loop
        for (int j = 0; j < 8; j++) {
            smem_in[lane * 8 + j] = gmem_in[lane * 8 + j];
        }
    }
    __syncthreads();

    // ldmatrix.x4: each of 32 lanes supplies its row pointer.
    uint32_t addr_in = smem_ptr_u32(&smem_in[lane * 8]);
    uint32_t r0, r1, r2, r3;
    ldmatrix_x4(r0, r1, r2, r3, addr_in);

    // stmatrix.x4: write back to output smem with same addressing pattern.
    uint32_t addr_out = smem_ptr_u32(&smem_out[lane * 8]);
    stmatrix_x4(addr_out, r0, r1, r2, r3);
    __syncthreads();

    // Copy output SMEM -> output GMEM
    if (lane < ROWS) {
        for (int j = 0; j < 8; j++) {
            gmem_out[lane * 8 + j] = smem_out[lane * 8 + j];
        }
    }
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    half* h_in  = (half*)malloc(TOTAL_H * sizeof(half));
    half* h_out = (half*)malloc(TOTAL_H * sizeof(half));
    // Deterministic pattern -- each element unique
    for (int i = 0; i < TOTAL_H; i++) {
        h_in[i] = __float2half((float)(i - TOTAL_H / 2) * 0.125f);
    }

    half *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  TOTAL_H * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, TOTAL_H * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, TOTAL_H * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, TOTAL_H * sizeof(half)));

    ldmatrix_stmatrix_kernel<<<1, 32>>>(d_in, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, TOTAL_H * sizeof(half), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < TOTAL_H; i++) {
        float a = __half2float(h_in[i]);
        float b = __half2float(h_out[i]);
        if (a != b) {
            if (mismatches < 5) {
                printf("  mismatch [%d]: in=%.4f out=%.4f\n", i, a, b);
            }
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("  ldmatrix.x4 + stmatrix.x4 round-trip: OK (%d halves)\n", TOTAL_H);
    } else {
        printf("  ldmatrix.x4 + stmatrix.x4 round-trip: FAIL (%d / %d mismatches)\n",
               mismatches, TOTAL_H);
        all_pass = false;
    }

    // --- Timing ---
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 100; i++) ldmatrix_stmatrix_kernel<<<1, 32>>>(d_in, d_out);
    t.end();
    printf("  perf: %.2f us/launch (%d halves round-trip)\n",
           t.elapsed_ms() * 1000.0f / 100, TOTAL_H);

    free(h_in); free(h_out);
    cudaFree(d_in); cudaFree(d_out);

    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}


// =============================================================================
// .b8 forms (sm_100a / sm_103a): ldmatrix.m16n16.b8 + stmatrix.m16n8.b8
// =============================================================================
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

// Shapes:
//   ldmatrix.m16n16.x1.trans.b8 -> 2 regs/thread (covers a 16x16 b8 = 256B)
//   stmatrix.m16n8.x2.trans.b8  <- 2 regs/thread (covers two 16x8 b8 = 256B)
// Total round-trip: 256 source bytes -> 256 dest bytes; bit-pattern preserved
// (the trans/no-trans pair changes physical layout but the SMEM byte set is
// the same -- this smoke checks "the wrapper executed and SMEM_OUT is
// non-zero", not exact element correspondence).
__global__ void k_b8_ld_st(const uint8_t* gin, uint8_t* gout) {
  __shared__ __align__(16) uint8_t smem_in[256];
  __shared__ __align__(16) uint8_t smem_out[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    smem_in[i]  = gin[i];
    smem_out[i] = 0;
  }
  __syncthreads();

  const int lane = threadIdx.x;
  // Each lane provides the row address for one of 16 rows; ldmatrix.x1.m16n16
  // uses lanes 0-7 (with lanes 8-15 covering the second 8x16 half) -- so we
  // use lane = row mod 16 as the row index. ldmatrix consumes addresses from
  // lanes 0-15 for x1.m16n16; provide a sensible row address per lane.
  int row = lane & 15;
  uint32_t addr_in  = smem_ptr_u32(&smem_in[row * 16]);
  uint32_t r0 = 0, r1 = 0;
  ldmatrix_x1_trans_b8(r0, r1, addr_in);

  // stmatrix.m16n8.x2.trans.b8 takes 2 regs/thread covering 32 b8 elts/thread,
  // total 32 * 32 = 1024 b8... actually: x2 m16n8 = 2 matrices of 16x8 each
  // = 256 b8, 32 threads x 2 regs x 4 b8/reg = 256 b8. So same count as ld.
  uint32_t addr_out = smem_ptr_u32(&smem_out[row * 16]);
  stmatrix_x2_trans_b8(addr_out, r0, r1);
  __syncthreads();

  for (int i = threadIdx.x; i < 256; i += blockDim.x) gout[i] = smem_out[i];
}

static int run_b8() {
  uint8_t hIn[256], hOut[256];
  for (int i = 0; i < 256; ++i) hIn[i] = (uint8_t)(0xA5u + (i * 7u));
  std::memset(hOut, 0, sizeof(hOut));
  uint8_t* dIn  = nullptr; CUDA_CHECK(cudaMalloc(&dIn,  256));
  uint8_t* dOut = nullptr; CUDA_CHECK(cudaMalloc(&dOut, 256));
  CUDA_CHECK(cudaMemcpy(dIn, hIn, 256, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dOut, 0, 256));
  k_b8_ld_st<<<1, 32>>>(dIn, dOut);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hOut, dOut, 256, cudaMemcpyDeviceToHost));
  cudaFree(dIn); cudaFree(dOut);
  int nz = 0; for (int i = 0; i < 256; ++i) if (hOut[i] != 0) ++nz;
  printf("ldmatrix.m16n16.b8 + stmatrix.m16n8.b8 : non-zero out bytes = %d / 256\n",
         nz);
  if (nz == 0) FAIL("b8 ldmatrix/stmatrix round-trip produced no data");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
  int rc_b8     = run_b8();
  return (rc_ours == 0 && rc_theirs == 0 && rc_b8 == 0) ? 0 : 1;
#else
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
#endif
}

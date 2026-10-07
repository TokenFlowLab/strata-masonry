// ARCH: sm_90a
// 41_smem_swizzle_test.cu -- exercise the swizzle math on host data and
// verify it is a bijection within an 8-row atom.
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
#include "../primitives/41_smem_swizzle.cuh"
#include <set>
#include "41_smem_swizzle.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


static int run_ours() {
  /* (orig args dropped) */
  for (int swiz : { 32, 64, 128 }) {
    std::set<uint32_t> seen;
    for (uint32_t row = 0; row < 8; ++row) {
      for (uint32_t col = 0; col < (uint32_t)swiz; col += 16) {
        uint32_t s;
        if (swiz == 32)  s = swizzle_col_bytes<32>(row, col);
        else if (swiz == 64) s = swizzle_col_bytes<64>(row, col);
        else                  s = swizzle_col_bytes<128>(row, col);
        uint32_t key = (row << 16) | s;
        if (!seen.insert(key).second) {
          fprintf(stderr, "collision swiz=%d row=%u col=%u s=%u\n",
                  swiz, row, col, s);
          FAIL("swizzle not a bijection");
        }
      }
    }
  }
  printf("smem_swizzle : B32/B64/B128 bijective within 8 rows\n");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: smem_swizzle XOR math (B32/B64/B128)
//
// The swizzle primitives are __device__, so we launch a tiny kernel that
// evaluates them on a known list of byte offsets and compare against a
// pure-C++ CPU reference.

// CPU reference (mirror of the device-side formulas)
static uint32_t ref_b32(uint32_t x)  { return x ^ ((x & 0x10u)  >> 0); }
static uint32_t ref_b64(uint32_t x)  { return x ^ ((x & 0x30u)  >> 0); }
static uint32_t ref_b128(uint32_t x) { return x ^ ((x & 0x380u) >> 3); }

__global__ void swizzle_kernel(const uint32_t* offsets, int n,
                                uint32_t* out_b32, uint32_t* out_b64, uint32_t* out_b128) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    uint32_t off = offsets[tid];
    out_b32[tid]  = smem_swizzle_b32(off);
    out_b64[tid]  = smem_swizzle_b64(off);
    out_b128[tid] = smem_swizzle_b128(off);
}

// Test smem_swizzled_addr_b128 (base + swizzle(row*stride + col*esz))
__global__ void swizzle_addr_kernel(uint32_t base, int row_stride, int elem_size,
                                     const int* rows, const int* cols, int n,
                                     uint32_t* out) {
    int tid = threadIdx.x;
    if (tid >= n) return;
    out[tid] = smem_swizzled_addr_b128(base, rows[tid], cols[tid], row_stride, elem_size);
}

__global__ void swizzle_perf_kernel(uint32_t* out, int iters) {
    uint32_t v = threadIdx.x * 16;
    for (int i = 0; i < iters; i++) {
        v = smem_swizzle_b128(v + i);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    // 33 offsets spanning 0 to 0x380, hitting each 0x80-aligned bucket
    uint32_t h_off[] = {
        0x000, 0x010, 0x020, 0x030, 0x040, 0x050, 0x060, 0x070,
        0x080, 0x090, 0x0A0, 0x0B0, 0x0C0, 0x0D0, 0x0E0, 0x0F0,
        0x100, 0x110, 0x120, 0x180, 0x1B0, 0x200, 0x240, 0x280,
        0x2D0, 0x300, 0x360, 0x380, 0x3F0, 0x008, 0x018, 0x088, 0x108
    };
    const int N = sizeof(h_off) / sizeof(h_off[0]);

    uint32_t *d_off;  CUDA_CHECK(cudaMalloc(&d_off,  N * sizeof(uint32_t)));
    uint32_t *d_b32;  CUDA_CHECK(cudaMalloc(&d_b32,  N * sizeof(uint32_t)));
    uint32_t *d_b64;  CUDA_CHECK(cudaMalloc(&d_b64,  N * sizeof(uint32_t)));
    uint32_t *d_b128; CUDA_CHECK(cudaMalloc(&d_b128, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_off, h_off, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    swizzle_kernel<<<1, 64>>>(d_off, N, d_b32, d_b64, d_b128);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_b32[N], h_b64[N], h_b128[N];
    CUDA_CHECK(cudaMemcpy(h_b32,  d_b32,  N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b64,  d_b64,  N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_b128, d_b128, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Match device vs CPU reference
    int bad32 = 0, bad64 = 0, bad128 = 0;
    for (int i = 0; i < N; i++) {
        if (h_b32[i]  != ref_b32(h_off[i]))  bad32++;
        if (h_b64[i]  != ref_b64(h_off[i]))  bad64++;
        if (h_b128[i] != ref_b128(h_off[i])) bad128++;
    }
    if (bad32 == 0)  printf("  smem_swizzle_b32 (vs CPU ref): OK (%d offsets)\n", N);
    else             { printf("  smem_swizzle_b32: FAIL (%d)\n", bad32); all_pass = false; }
    if (bad64 == 0)  printf("  smem_swizzle_b64 (vs CPU ref): OK (%d offsets)\n", N);
    else             { printf("  smem_swizzle_b64: FAIL (%d)\n", bad64); all_pass = false; }
    if (bad128 == 0) printf("  smem_swizzle_b128 (vs CPU ref): OK (%d offsets)\n", N);
    else             { printf("  smem_swizzle_b128: FAIL (%d)\n", bad128); all_pass = false; }

    // Spot-check specific B128 values from the task spec:
    //   0x000 -> 0x000
    //   0x080 -> 0x090
    //   0x100 -> 0x120
    //   0x180 -> 0x1B0
    struct { uint32_t in, exp; } spot[] = {
        {0x000, 0x000}, {0x080, 0x090}, {0x100, 0x120}, {0x180, 0x1B0},
        {0x200, 0x240}, {0x300, 0x360}, {0x380, 0x3F0},
    };
    int bad_spot = 0;
    for (auto& s : spot) {
        uint32_t got = ref_b128(s.in);  // same formula as device, both checked above
        if (got != s.exp) {
            printf("    spot B128 0x%03x: got 0x%03x, expected 0x%03x\n", s.in, got, s.exp);
            bad_spot++;
        }
    }
    if (bad_spot == 0) printf("  B128 spot checks: OK (7 known values)\n");
    else               { printf("  B128 spot checks: FAIL\n"); all_pass = false; }

    // Test smem_swizzled_addr_b128: row=0,1,2, col=0,16,32 with stride=128, esz=2
    int h_rows[] = {0, 0, 1, 2, 3, 4};
    int h_cols[] = {0, 8, 0, 0, 4, 0};
    const int M = 6;
    int *d_rows, *d_cols; uint32_t *d_addr;
    CUDA_CHECK(cudaMalloc(&d_rows, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cols, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_addr, M * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_rows, h_rows, M * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_cols, M * sizeof(int), cudaMemcpyHostToDevice));

    uint32_t base = 0x1000u;
    int row_stride = 128, elem_size = 2;
    swizzle_addr_kernel<<<1, M>>>(base, row_stride, elem_size,
                                  d_rows, d_cols, M, d_addr);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h_addr[M];
    CUDA_CHECK(cudaMemcpy(h_addr, d_addr, M * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    int bad_addr = 0;
    for (int i = 0; i < M; i++) {
        uint32_t byte_off = h_rows[i] * row_stride + h_cols[i] * elem_size;
        uint32_t expected = base + ref_b128(byte_off);
        if (h_addr[i] != expected) {
            printf("    addr[%d] row=%d col=%d: got 0x%x, expected 0x%x\n",
                   i, h_rows[i], h_cols[i], h_addr[i], expected);
            bad_addr++;
        }
    }
    if (bad_addr == 0) printf("  smem_swizzled_addr_b128: OK\n");
    else               { printf("  smem_swizzled_addr_b128: FAIL\n"); all_pass = false; }

    // Perf
    uint32_t *d_perf; CUDA_CHECK(cudaMalloc(&d_perf, 4 * sizeof(uint32_t)));
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 1000; i++) swizzle_perf_kernel<<<1, 32>>>(d_perf, 100);
    t.end();
    printf("  perf: %.2f us/launch (1000 launches, 100 ops each)\n",
           t.elapsed_ms() * 1000.0f / 1000);

    cudaFree(d_off); cudaFree(d_b32); cudaFree(d_b64); cudaFree(d_b128);
    cudaFree(d_rows); cudaFree(d_cols); cudaFree(d_addr); cudaFree(d_perf);
    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}

// ARCH: sm_90a
// 116_tma_load_stage_test.cu -- issue one pipeline-stage load and verify data.
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
#include "../src/primitives/23_tma_tensormap.cuh"
#include "../src/primitives/29_mbarrier_init.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include "../src/composites/116_tma_load_stage.cuh"
#include <cuda_fp16.h>
#include "18_tma_load.cuh"
#include "22_tma_store.cuh"
#include "23_tma_tensormap.cuh"
#include "25_tma_async_group.cuh"
#include "29_mbarrier_init.cuh"
#include "33_mbarrier_try_wait.cuh"
#include "34_fence_proxy_async.cuh"
#include "116_tma_load_stage.cuh"

// =============================================================================
// ours
// =============================================================================

// ARCH: sm_90a


__global__ void k(const __grid_constant__ CUtensorMap ta, float* out) {
  __shared__ __align__(128) float smem[16 * 16];
  __shared__ __align__(16)  uint64_t mbar;
  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    tma_load_stage_1tensor(smem_ptr_u32(&mbar), 16 * 16 * sizeof(float),
                              smem_ptr_u32(smem), &ta, 0, 0);
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  if (threadIdx.x < 16 * 16) out[threadIdx.x] = smem[threadIdx.x];
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
}

static int run_ours() {
  std::vector<float> hin(16 * 16);
  for (int i = 0; i < 16 * 16; ++i) hin[i] = (float)i;
  float* din = nullptr; CUDA_CHECK(cudaMalloc(&din, 16 * 16 * 4));
  CUDA_CHECK(cudaMemcpy(din, hin.data(), 16 * 16 * 4, cudaMemcpyHostToDevice));
  CUtensorMap ta;
  CUDA_CHECK(make_tma_2d_tiled(&ta, din, 16, 16, 16, 16, sizeof(float),
                                 CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                 CU_TENSOR_MAP_SWIZZLE_NONE));
  float* dout = nullptr; CUDA_CHECK(cudaMalloc(&dout, 16 * 16 * 4));
  CUDA_CHECK(cudaMemset(dout, 0, 16 * 16 * 4));
  k<<<1, 256>>>(ta, dout);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> hout(16 * 16);
  CUDA_CHECK(cudaMemcpy(hout.data(), dout, 16 * 16 * 4,
                         cudaMemcpyDeviceToHost));
  cudaFree(din); cudaFree(dout);
  int fails = 0;
  for (int i = 0; i < 16 * 16; ++i) if (hout[i] != hin[i]) ++fails;
  printf("tma_load_stage 1-tensor : fails = %d / %d\n", fails, 16 * 16);
  if (fails) FAIL("content mismatch");
  PASS();
}

// =============================================================================
// theirs
// =============================================================================

// Runtime test: 116_tma_load_stage -- expect_tx + TMA load composite.
//
// Two sub-tests:
//   1) tma_load_stage_2d (single tile, single mbarrier): round-trip via TMA
//      store with SWIZZLE_128B so the swizzle is transparent.
//   2) tma_load_stage_ab (two tiles into same mbarrier): round-trip both A
//      and B tiles; verify both land correctly.
//
// Both use the elected-thread pattern (thread 0 drives mbarrier + TMA ops).


static constexpr int TILE_ROWS = 64;
static constexpr int TILE_COLS = 64;
static constexpr int TILE_ELEMS = TILE_ROWS * TILE_COLS;
static constexpr int TILE_BYTES = TILE_ELEMS * (int)sizeof(half);

extern __shared__ __align__(128) char smem_67[];

// Kernel: tma_load_stage_2d (1 tile, 1 mbarrier) + TMA store round-trip.
__global__ void kernel_stage_2d(const __grid_constant__ CUtensorMap tmap_in,
                                 const __grid_constant__ CUtensorMap tmap_out) {
    half* stile     = reinterpret_cast<half*>(smem_67);
    uint64_t* mbar  = reinterpret_cast<uint64_t*>(smem_67 + TILE_BYTES);
    int tid = threadIdx.x;

    if (tid == 0) {
        uint32_t mbar_a = smem_ptr_u32(mbar);
        mbarrier_init(mbar_a, 1);
        // Composite: expect_tx + TMA load
        tma_load_stage_2d(&tmap_in,
                          smem_ptr_u32(stile),
                          mbar_a,
                          (uint32_t)TILE_BYTES,
                          /*coord_x=*/0, /*coord_y=*/0);
    }
    __syncthreads();

    uint32_t mbar_a = smem_ptr_u32(mbar);
    mbarrier_wait_parity(mbar_a, 0);

    // SMEM -> GMEM via TMA store (same 128B swizzle cancels).
    if (tid == 0) {
        tma_store_2d(&tmap_out, 0, 0, smem_ptr_u32(stile));
        tma_store_commit_group();
        tma_store_wait_group<0>();
    }
}

// Kernel: tma_load_stage_ab (2 tiles, same mbarrier) + 2 TMA stores.
__global__ void kernel_stage_ab(const __grid_constant__ CUtensorMap tmap_a_in,
                                 const __grid_constant__ CUtensorMap tmap_b_in,
                                 const __grid_constant__ CUtensorMap tmap_a_out,
                                 const __grid_constant__ CUtensorMap tmap_b_out) {
    half* stile_a   = reinterpret_cast<half*>(smem_67);
    half* stile_b   = reinterpret_cast<half*>(smem_67 + TILE_BYTES);
    uint64_t* mbar  = reinterpret_cast<uint64_t*>(smem_67 + 2 * TILE_BYTES);
    int tid = threadIdx.x;

    if (tid == 0) {
        uint32_t mbar_a = smem_ptr_u32(mbar);
        mbarrier_init(mbar_a, 1);
        // Both TMA loads -> same mbarrier.
        tma_load_stage_ab(&tmap_a_in, &tmap_b_in,
                          smem_ptr_u32(stile_a),
                          smem_ptr_u32(stile_b),
                          mbar_a,
                          (uint32_t)TILE_BYTES,
                          (uint32_t)TILE_BYTES,
                          0, 0, 0, 0);
    }
    __syncthreads();

    uint32_t mbar_a = smem_ptr_u32(mbar);
    mbarrier_wait_parity(mbar_a, 0);

    // SMEM -> GMEM both tiles.
    if (tid == 0) {
        tma_store_2d(&tmap_a_out, 0, 0, smem_ptr_u32(stile_a));
        tma_store_2d(&tmap_b_out, 0, 0, smem_ptr_u32(stile_b));
        tma_store_commit_group();
        tma_store_wait_group<0>();
    }
}

static int run_theirs() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    // Source tiles with distinguishable patterns.
    std::vector<half> h_a(TILE_ELEMS), h_b(TILE_ELEMS);
    for (int i = 0; i < TILE_ELEMS; i++) {
        h_a[i] = __float2half((float)(i & 0x7F));
        h_b[i] = __float2half((float)((i + 37) & 0x7F));
    }

    half *d_a_in, *d_b_in, *d_a_out, *d_b_out;
    CUDA_CHECK(cudaMalloc(&d_a_in,  TILE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_b_in,  TILE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_a_out, TILE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_b_out, TILE_BYTES));
    CUDA_CHECK(cudaMemcpy(d_a_in, h_a.data(), TILE_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_in, h_b.data(), TILE_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_a_out, 0xaa, TILE_BYTES));
    CUDA_CHECK(cudaMemset(d_b_out, 0xaa, TILE_BYTES));

    // Tensormaps (SWIZZLE_128B so load+store round-trip cancels swizzle).
    CUtensorMap tmap_a_in{}, tmap_b_in{}, tmap_a_out{}, tmap_b_out{};
    create_tma_2d_f16(&tmap_a_in,  d_a_in,  TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS);
    create_tma_2d_f16(&tmap_b_in,  d_b_in,  TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS);
    create_tma_2d_f16(&tmap_a_out, d_a_out, TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS);
    create_tma_2d_f16(&tmap_b_out, d_b_out, TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS);

    // Set dynamic SMEM attributes.
    size_t smem_bytes_single = TILE_BYTES + 16;
    size_t smem_bytes_ab     = 2 * TILE_BYTES + 16;
    CUDA_CHECK(cudaFuncSetAttribute(kernel_stage_2d,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes_single));
    CUDA_CHECK(cudaFuncSetAttribute(kernel_stage_ab,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes_ab));

    // -- Test 1: tma_load_stage_2d --
    {
        GpuTimer t;
        t.begin();
        kernel_stage_2d<<<1, 128, smem_bytes_single>>>(tmap_a_in, tmap_a_out);
        t.end();
        CUDA_CHECK(cudaGetLastError());
        std::vector<half> h_out(TILE_ELEMS);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_a_out, TILE_BYTES, cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < TILE_ELEMS; i++) {
            if (__half2float(h_out[i]) != __half2float(h_a[i])) {
                if (bad < 5) printf("    stage_2d [%d] got=%.1f exp=%.1f\n",
                                     i, __half2float(h_out[i]), __half2float(h_a[i]));
                bad++;
            }
        }
        if (bad == 0) printf("  tma_load_stage_2d: OK (%.3f ms)\n", t.elapsed_ms());
        else          { printf("  tma_load_stage_2d: FAIL (%d mismatches)\n", bad); all_pass = false; }
    }

    // -- Test 2: tma_load_stage_ab --
    {
        CUDA_CHECK(cudaMemset(d_a_out, 0xcc, TILE_BYTES));
        CUDA_CHECK(cudaMemset(d_b_out, 0xcc, TILE_BYTES));
        GpuTimer t;
        t.begin();
        kernel_stage_ab<<<1, 128, smem_bytes_ab>>>(tmap_a_in, tmap_b_in, tmap_a_out, tmap_b_out);
        t.end();
        CUDA_CHECK(cudaGetLastError());
        std::vector<half> h_a_out(TILE_ELEMS), h_b_out(TILE_ELEMS);
        CUDA_CHECK(cudaMemcpy(h_a_out.data(), d_a_out, TILE_BYTES, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b_out.data(), d_b_out, TILE_BYTES, cudaMemcpyDeviceToHost));
        int bad_a = 0, bad_b = 0;
        for (int i = 0; i < TILE_ELEMS; i++) {
            if (__half2float(h_a_out[i]) != __half2float(h_a[i])) bad_a++;
            if (__half2float(h_b_out[i]) != __half2float(h_b[i])) bad_b++;
        }
        if (bad_a == 0 && bad_b == 0)
            printf("  tma_load_stage_ab: OK (A+B; %.3f ms)\n", t.elapsed_ms());
        else {
            printf("  tma_load_stage_ab: FAIL (a=%d b=%d)\n", bad_a, bad_b);
            all_pass = false;
        }
    }

    // -- Perf --
    GpuTimer tp;
    const int ITERS = 200;
    tp.begin();
    for (int i = 0; i < ITERS; i++)
        kernel_stage_ab<<<1, 128, smem_bytes_ab>>>(tmap_a_in, tmap_b_in, tmap_a_out, tmap_b_out);
    tp.end();
    printf("  perf (stage_ab 64x64 FP16 x 2): %.2f us/launch\n",
           tp.elapsed_ms() * 1000.0f / ITERS);

    cudaFree(d_a_in); cudaFree(d_b_in);
    cudaFree(d_a_out); cudaFree(d_b_out);

    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}

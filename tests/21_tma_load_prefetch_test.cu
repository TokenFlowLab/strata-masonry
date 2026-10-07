// ARCH: sm_90a
// 21_tma_load_prefetch_test.cu -- prefetch is a hint; compile smoke.
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
#include "../src/primitives/21_tma_load_prefetch.cuh"
#include "../src/primitives/23_tma_tensormap.cuh"
#include <cuda_fp16.h>
#include "18_tma_load.cuh"
#include "21_tma_load_prefetch.cuh"
#include "22_tma_store.cuh"
#include "23_tma_tensormap.cuh"
#include "25_tma_async_group.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k_prefetch(const __grid_constant__ CUtensorMap desc) {
  if (threadIdx.x == 0) tma_prefetch_2d(&desc, 0, 0);
}

static int run_ours() {
  /* (orig args dropped) */
  std::vector<float> hIn(16 * 16, 1.f);
  float* dIn = nullptr; CUDA_CHECK(cudaMalloc(&dIn, 16 * 16 * 4));
  CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), 16 * 16 * 4, cudaMemcpyHostToDevice));
  CUtensorMap desc;
  CUDA_CHECK(make_tma_2d_tiled(&desc, dIn, 16, 16, 16, 16, sizeof(float),
                                 CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                 CU_TENSOR_MAP_SWIZZLE_NONE));
  k_prefetch<<<1, 32>>>(desc);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(dIn);
  printf("tma_prefetch_2d : compile + run OK\n");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: TMA L2 prefetch (cp.async.bulk.prefetch.tensor.2d)
// Build: nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 --expt-relaxed-constexpr
//        -DNDEBUG -lineinfo -I primitives -I tests -lcuda -o build/21_test tests/21_tma_load_prefetch_test.cu
//
// Prefetch is purely advisory -- it has no architectural output we can
// correctness-test. The best we can do at runtime is:
//  (a) prove the PTX executes without error (the kernel returns),
//  (b) measure its effect on a subsequent TMA load and report the delta.
//
// We compare two scenarios with L2 flushed via cudaMemsetAsync of a large
// scratch buffer between launches:
//   - Baseline:  plain TMA load + store round-trip.
//   - Prefetch:  tma_prefetch_2d first, then the same TMA load + store.
//
// Prefetch latency impact is small (and depends on L2 state); we print the
// numbers but don't gate the test on them.

static constexpr int TILE_ROWS = 64;
static constexpr int TILE_COLS = 64;
static constexpr int TILE_ELEMS = TILE_ROWS * TILE_COLS;
static constexpr int TILE_BYTES = TILE_ELEMS * (int)sizeof(half);

extern __shared__ __align__(128) char smem_buf[];

// Kernel that just issues the two prefetch variants -- sanity check that
// the PTX is accepted at runtime.
__global__ void prefetch_only_kernel(const __grid_constant__ CUtensorMap tmap) {
    if (threadIdx.x == 0) {
        tma_prefetch_2d(&tmap, 0, 0);
        tma_prefetch_2d_l2hint(&tmap, 0, 0, /*cache_policy=*/0ull);
    }
}

// Round-trip that does prefetch first, then the actual load+store.
__global__ void prefetch_then_load_kernel(
    const __grid_constant__ CUtensorMap tmap_in,
    const __grid_constant__ CUtensorMap tmap_out)
{
    half*    smem_tile = reinterpret_cast<half*>(smem_buf);
    uint64_t* mbar     = reinterpret_cast<uint64_t*>(smem_buf + TILE_BYTES);
    int tid = threadIdx.x;

    if (tid == 0) {
        // L2 hint issued up-front.
        tma_prefetch_2d(&tmap_in, 0, 0);
        uint32_t mbar_addr = smem_ptr_u32(mbar);
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                     :: "r"(mbar_addr), "r"(1) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                     :: "r"(mbar_addr), "r"((uint32_t)TILE_BYTES) : "memory");
        tma_load_2d_cta(&tmap_in, smem_ptr_u32(smem_tile), mbar_addr, 0, 0);
    }
    __syncthreads();
    uint32_t mbar_addr = smem_ptr_u32(mbar);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_LOOP_21A:\n"
        "  mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_LOOP_21A;\n"
        "}\n"
        :: "r"(mbar_addr), "r"(0u));
    if (tid == 0) {
        tma_store_2d(&tmap_out, 0, 0, smem_ptr_u32(smem_tile));
        tma_store_commit_group();
        tma_store_wait_group<0>();
    }
}

// Same round-trip, no prefetch.
__global__ void plain_load_kernel(
    const __grid_constant__ CUtensorMap tmap_in,
    const __grid_constant__ CUtensorMap tmap_out)
{
    half*    smem_tile = reinterpret_cast<half*>(smem_buf);
    uint64_t* mbar     = reinterpret_cast<uint64_t*>(smem_buf + TILE_BYTES);
    int tid = threadIdx.x;

    if (tid == 0) {
        uint32_t mbar_addr = smem_ptr_u32(mbar);
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                     :: "r"(mbar_addr), "r"(1) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                     :: "r"(mbar_addr), "r"((uint32_t)TILE_BYTES) : "memory");
        tma_load_2d_cta(&tmap_in, smem_ptr_u32(smem_tile), mbar_addr, 0, 0);
    }
    __syncthreads();
    uint32_t mbar_addr = smem_ptr_u32(mbar);
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAIT_LOOP_21B:\n"
        "  mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_LOOP_21B;\n"
        "}\n"
        :: "r"(mbar_addr), "r"(0u));
    if (tid == 0) {
        tma_store_2d(&tmap_out, 0, 0, smem_ptr_u32(smem_tile));
        tma_store_commit_group();
        tma_store_wait_group<0>();
    }
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    half *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, TILE_BYTES));
    CUDA_CHECK(cudaMalloc(&d_dst, TILE_BYTES));

    std::vector<half> h_src(TILE_ELEMS);
    for (int i = 0; i < TILE_ELEMS; i++)
        h_src[i] = __float2half((float)(i & 0xFF));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), TILE_BYTES, cudaMemcpyHostToDevice));

    CUtensorMap tmap_in{}, tmap_out{};
    create_tma_2d_f16(&tmap_in,  d_src, TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS);
    create_tma_2d_f16(&tmap_out, d_dst, TILE_ROWS, TILE_COLS, TILE_ROWS, TILE_COLS);

    size_t smem_bytes = TILE_BYTES + 16;
    CUDA_CHECK(cudaFuncSetAttribute(prefetch_only_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(prefetch_then_load_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(plain_load_kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)smem_bytes));

    // --- Test 1: prefetch-only kernel launches cleanly -------------------
    prefetch_only_kernel<<<1, 32>>>(tmap_in);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  prefetch-only kernel: OK (ran without error)\n");

    // --- Test 2: correctness of prefetch + load round-trip ---------------
    CUDA_CHECK(cudaMemset(d_dst, 0, TILE_BYTES));
    prefetch_then_load_kernel<<<1, 128, smem_bytes>>>(tmap_in, tmap_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> h_dst(TILE_ELEMS);
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, TILE_BYTES, cudaMemcpyDeviceToHost));
    int mism = 0;
    for (int i = 0; i < TILE_ELEMS; i++) {
        if (__half2float(h_dst[i]) != __half2float(h_src[i])) mism++;
    }
    if (mism == 0) printf("  prefetch + load correctness: OK\n");
    else { printf("  prefetch + load correctness: FAIL (%d mismatches)\n", mism); all_pass = false; }

    // --- Test 3: timing comparison ---------------------------------------
    //
    // Small scratch buffer to flush L2 between iterations.  H100 L2 is ~50 MB.
    // We scrub it by streaming a 96 MB buffer.
    size_t scratch_bytes = 96ull * 1024 * 1024;
    uint8_t* d_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scratch, scratch_bytes));
    CUDA_CHECK(cudaMemset(d_scratch, 0, scratch_bytes));

    const int ITERS = 50;
    auto measure = [&](bool with_prefetch) {
        float total_ms = 0.0f;
        for (int i = 0; i < ITERS; i++) {
            // Flush L2 before each iteration.
            CUDA_CHECK(cudaMemsetAsync(d_scratch, (int)(i & 0xFF), scratch_bytes));
            GpuTimer gt;
            gt.begin();
            if (with_prefetch)
                prefetch_then_load_kernel<<<1, 128, smem_bytes>>>(tmap_in, tmap_out);
            else
                plain_load_kernel       <<<1, 128, smem_bytes>>>(tmap_in, tmap_out);
            gt.end();
            total_ms += gt.elapsed_ms();
        }
        return (total_ms / ITERS) * 1000.0f;  // us per launch
    };

    float us_plain    = measure(false);
    float us_prefetch = measure(true);
    printf("  plain   load (cold L2): %.2f us/launch\n", us_plain);
    printf("  prefetch+load (cold L2): %.2f us/launch\n", us_prefetch);
    float delta = us_plain - us_prefetch;
    if (delta > 0) {
        printf("  prefetch delta: -%.2f us (%.1f%% faster)\n",
               delta, 100.0f * delta / us_plain);
    } else {
        printf("  prefetch delta: +%.2f us (prefetch is hw hint; timing is noisy)\n", -delta);
    }

    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_scratch);
    if (all_pass) { PASS(); return 0; }
    else          { FAIL("subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}

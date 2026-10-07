// ARCH: sm_90a
// 51_atom_shared_test.cu -- SMEM counter accumulates 32 adds.
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
#include "../src/primitives/51_atom_shared.cuh"
#include "51_atom_shared.cuh"

// =============================================================================
// ours
// =============================================================================

// ARCH: sm_90a


__global__ void k_atom(uint32_t* gout) {
  __shared__ uint32_t cnt;
  if (threadIdx.x == 0) cnt = 0;
  __syncthreads();
  atom_shared_cta_add_u32(smem_ptr_u32(&cnt), 1);
  __syncthreads();
  if (threadIdx.x == 0) *gout = cnt;
}

static int run_ours() {
  uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
  CUDA_CHECK(cudaMemset(d, 0, 4));
  k_atom<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h != 32u) FAIL("atom.shared.add wrong");
  printf("atom.shared::cta.add.u32 32x1 = %u\n", h);
  PASS();
}

// =============================================================================
// theirs
// =============================================================================

// Runtime test: atom.shared atomics (CTA-scope) -- verify correctness + timing
// The cluster-scope variant requires cluster launch and is intentionally kept
// compile-only (the primitive header still compiles it).

// Each of THREADS threads adds 1 to a CTA-shared counter; final value must
// equal THREADS.
__global__ void atom_shared_cta_add_u32_kernel(uint32_t* out) {
    __shared__ uint32_t counter;
    if (threadIdx.x == 0) counter = 0;
    __syncthreads();

    uint32_t addr = smem_ptr_u32(&counter);
    atom_shared_cta_add_u32(addr, 1u);

    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = counter;
}

// Each of THREADS threads adds 1.0f to a CTA-shared float accumulator.
__global__ void atom_shared_cta_add_f32_kernel(float* out) {
    __shared__ float accum;
    if (threadIdx.x == 0) accum = 0.0f;
    __syncthreads();

    uint32_t addr = smem_ptr_u32(&accum);
    atom_shared_cta_add_f32(addr, 1.0f);

    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = accum;
}

// Compile-only reference to cluster-scope variant -- never called at runtime.
__global__ void atom_shared_cluster_compile_only(uint32_t* out) {
    __shared__ uint32_t counter;
    if (threadIdx.x == 0) counter = 0;
    __syncthreads();
    uint32_t addr = smem_ptr_u32(&counter);
    uint32_t old = atom_shared_cluster_add_u32(addr, 1u);
    __syncthreads();
    if (threadIdx.x == 0) out[0] = counter + old;
}

static int run_theirs() {
    CUDA_CHECK(cudaFree(0));
    const int BLOCKS = 4;
    const int THREADS = 256;
    bool all_pass = true;

    // --- CTA-scope u32 add ---
    uint32_t* d_u32; CUDA_CHECK(cudaMalloc(&d_u32, BLOCKS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_u32, 0, BLOCKS * sizeof(uint32_t)));
    atom_shared_cta_add_u32_kernel<<<BLOCKS, THREADS>>>(d_u32);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h_u32[BLOCKS];
    CUDA_CHECK(cudaMemcpy(h_u32, d_u32, BLOCKS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    bool u32_ok = true;
    for (int i = 0; i < BLOCKS; i++) if (h_u32[i] != (uint32_t)THREADS) u32_ok = false;
    if (u32_ok) printf("  cta.add.u32: OK (all %d blocks = %d)\n", BLOCKS, THREADS);
    else {
        printf("  cta.add.u32: FAIL (block 0 = %u, expected %d)\n", h_u32[0], THREADS);
        all_pass = false;
    }

    // --- CTA-scope f32 add ---
    float* d_f32; CUDA_CHECK(cudaMalloc(&d_f32, BLOCKS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_f32, 0, BLOCKS * sizeof(float)));
    atom_shared_cta_add_f32_kernel<<<BLOCKS, THREADS>>>(d_f32);
    CUDA_CHECK(cudaDeviceSynchronize());
    float h_f32[BLOCKS];
    CUDA_CHECK(cudaMemcpy(h_f32, d_f32, BLOCKS * sizeof(float), cudaMemcpyDeviceToHost));
    bool f32_ok = true;
    for (int i = 0; i < BLOCKS; i++) if ((int)h_f32[i] != THREADS) f32_ok = false;
    if (f32_ok) printf("  cta.add.f32: OK (all %d blocks = %d)\n", BLOCKS, THREADS);
    else {
        printf("  cta.add.f32: FAIL (block 0 = %.1f, expected %d)\n", h_f32[0], THREADS);
        all_pass = false;
    }

    // --- Sanity: ensure cluster-scope kernel compiled (not launched) ---
    (void)atom_shared_cluster_compile_only;
    printf("  cluster.add.u32: compile-only (not run)\n");

    // --- Timing: 100 iterations of CTA add u32 ---
    GpuTimer t;
    CUDA_CHECK(cudaMemset(d_u32, 0, BLOCKS * sizeof(uint32_t)));
    t.begin();
    for (int i = 0; i < 100; i++) atom_shared_cta_add_u32_kernel<<<BLOCKS, THREADS>>>(d_u32);
    t.end();
    printf("  perf: %.2f us/launch (%d atomics each)\n",
           t.elapsed_ms() * 1000.0f / 100, BLOCKS * THREADS);

    cudaFree(d_u32); cudaFree(d_f32);
    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}


// =============================================================================
// shared cas.b32, min/max u32 on shared
// =============================================================================

__global__ void k_atom_shared_cas(uint32_t* out) {
  __shared__ uint32_t s;
  if (threadIdx.x == 0) s = 100u;
  __syncthreads();
  // Single thread does cas: replace 100 -> 999
  if (threadIdx.x == 0) {
    uint32_t old = atom_shared_cta_cas_b32(smem_ptr_u32(&s), 100u, 999u);
    out[0] = old;       // expect 100
    out[1] = s;         // expect 999
    // Mismatched cas: should NOT replace
    uint32_t old2 = atom_shared_cta_cas_b32(smem_ptr_u32(&s), 100u, 0xDEADu);
    out[2] = old2;      // expect 999
    out[3] = s;         // still 999
  }
}

__global__ void k_atom_shared_min(uint32_t* out) {
  __shared__ uint32_t s;
  if (threadIdx.x == 0) s = 0xFFFFFFFFu;  // start at max
  __syncthreads();
  // Each thread races to min with its tid -- final must be 0.
  atom_shared_cta_min_u32(smem_ptr_u32(&s), threadIdx.x);
  __syncthreads();
  if (threadIdx.x == 0) out[0] = s;       // expect 0
}

__global__ void k_atom_shared_max(uint32_t* out) {
  __shared__ uint32_t s;
  if (threadIdx.x == 0) s = 0u;
  __syncthreads();
  // Each thread races to max with its tid -- final must be blockDim.x-1.
  atom_shared_cta_max_u32(smem_ptr_u32(&s), threadIdx.x);
  __syncthreads();
  if (threadIdx.x == 0) out[0] = s;       // expect blockDim.x - 1
}

static int run_med() {
  bool ok = true;

  // cas
  {
    uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d, 0, 4 * sizeof(uint32_t)));
    k_atom_shared_cas<<<1, 32>>>(d);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h[4];
    CUDA_CHECK(cudaMemcpy(h, d, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d);
    if (h[0] != 100u || h[1] != 999u || h[2] != 999u || h[3] != 999u) {
      printf("  shared cas.b32: FAIL old1=%u s1=%u old2=%u s2=%u\n", h[0],h[1],h[2],h[3]);
      ok = false;
    } else {
      printf("  shared cas.b32: OK (100->999, mismatch preserves)\n");
    }
  }

  // min
  {
    constexpr int T = 128;
    uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
    CUDA_CHECK(cudaMemset(d, 0xFF, 4));
    k_atom_shared_min<<<1, T>>>(d);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
    cudaFree(d);
    if (h != 0u) { printf("  shared min.u32: FAIL got %u\n", h); ok = false; }
    else printf("  shared min.u32: OK (final=%u)\n", h);
  }

  // max
  {
    constexpr int T = 128;
    uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
    CUDA_CHECK(cudaMemset(d, 0, 4));
    k_atom_shared_max<<<1, T>>>(d);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
    cudaFree(d);
    if (h != (uint32_t)(T - 1)) { printf("  shared max.u32: FAIL got %u expected %u\n", h, T-1); ok = false; }
    else printf("  shared max.u32: OK (final=%u)\n", h);
  }

  if (ok) { PASS(); return 0; }
  else { FAIL("shared cas/min/max subtest failed"); return 1; }
}

int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  int rc_med    = run_med();
  return (rc_ours == 0 && rc_theirs == 0 && rc_med == 0) ? 0 : 1;
}

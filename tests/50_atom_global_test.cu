// ARCH: sm_90a
// 50_atom_global_test.cu -- 32 threads each add 1 -> counter == 32.
//
// Combined test: both ours' and theirs' coverage is exercised
// in a single binary (each side's main() became run_ours/run_theirs).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <climits>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../primitives/50_atom_global.cuh"
#include "50_atom_global.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


__global__ void k_atom(uint32_t* out) {
  atom_global_add_u32(out, 1u);
}

static int run_ours() {
  /* (orig args dropped) */
  uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 4));
  CUDA_CHECK(cudaMemset(d, 0, 4));
  k_atom<<<1, 32>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h = 0; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
  cudaFree(d);
  if (h != 32u) FAIL("atom.global.add counter wrong");
  printf("atom.global.add.u32 32x1 = %u\n", h);
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: atom.global atomics -- verify correctness + timing

__global__ void atomic_add_u32_kernel(uint32_t* counter) {
    atom_global_add_u32(counter, 1u);
}

__global__ void atomic_add_f32_kernel(float* sum) {
    atom_global_add_f32(sum, 1.0f);
}

__global__ void atomic_max_kernel(uint32_t* out, uint32_t val) {
    atom_global_max_u32(out, val);
}

__global__ void atomic_cas_kernel(uint32_t* addr, uint32_t compare, uint32_t val) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        atom_global_cas_u32(addr, compare, val);
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    const int BLOCKS = 128, THREADS = 256, EXPECTED = BLOCKS * THREADS;
    bool all_pass = true;

    uint32_t *d_counter; CUDA_CHECK(cudaMalloc(&d_counter, 4));
    CUDA_CHECK(cudaMemset(d_counter, 0, 4));
    atomic_add_u32_kernel<<<BLOCKS, THREADS>>>(d_counter);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h_counter;
    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, 4, cudaMemcpyDeviceToHost));
    if (h_counter == (uint32_t)EXPECTED) printf("  add.u32: OK (%u)\n", h_counter);
    else { printf("  add.u32: FAIL (got %u, expected %d)\n", h_counter, EXPECTED); all_pass = false; }

    float *d_sum; CUDA_CHECK(cudaMalloc(&d_sum, 4));
    CUDA_CHECK(cudaMemset(d_sum, 0, 4));
    atomic_add_f32_kernel<<<BLOCKS, THREADS>>>(d_sum);
    CUDA_CHECK(cudaDeviceSynchronize());
    float h_sum;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, 4, cudaMemcpyDeviceToHost));
    if ((int)h_sum == EXPECTED) printf("  add.f32: OK (%.0f)\n", h_sum);
    else { printf("  add.f32: FAIL (got %.0f)\n", h_sum); all_pass = false; }

    uint32_t *d_max; CUDA_CHECK(cudaMalloc(&d_max, 4));
    CUDA_CHECK(cudaMemset(d_max, 0, 4));
    atomic_max_kernel<<<1, 256>>>(d_max, 42u);
    atomic_max_kernel<<<1, 256>>>(d_max, 100u);
    atomic_max_kernel<<<1, 256>>>(d_max, 50u);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h_max;
    CUDA_CHECK(cudaMemcpy(&h_max, d_max, 4, cudaMemcpyDeviceToHost));
    if (h_max == 100u) printf("  max.u32: OK (%u)\n", h_max);
    else { printf("  max.u32: FAIL (got %u)\n", h_max); all_pass = false; }

    uint32_t *d_cas; CUDA_CHECK(cudaMalloc(&d_cas, 4));
    uint32_t init = 42;
    CUDA_CHECK(cudaMemcpy(d_cas, &init, 4, cudaMemcpyHostToDevice));
    atomic_cas_kernel<<<1, 1>>>(d_cas, 42, 99);
    atomic_cas_kernel<<<1, 1>>>(d_cas, 42, 77);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h_cas;
    CUDA_CHECK(cudaMemcpy(&h_cas, d_cas, 4, cudaMemcpyDeviceToHost));
    if (h_cas == 99u) printf("  cas.b32: OK (%u)\n", h_cas);
    else { printf("  cas.b32: FAIL (got %u)\n", h_cas); all_pass = false; }

    GpuTimer t;
    CUDA_CHECK(cudaMemset(d_counter, 0, 4));
    t.begin();
    for (int i = 0; i < 100; i++) atomic_add_u32_kernel<<<BLOCKS, THREADS>>>(d_counter);
    t.end();
    printf("  perf: %.2f us/launch (%d atomics each)\n",
           t.elapsed_ms() * 1000.0f / 100, EXPECTED);

    cudaFree(d_counter); cudaFree(d_sum); cudaFree(d_max); cudaFree(d_cas);
    if (all_pass) { PASS(); return 0; }
    else { FAIL("some subtests failed"); return 1; }
}


// =============================================================================
// sem / scope plumbing (split-K reader/writer pattern)
// =============================================================================

__global__ void k_atom_release_gpu(uint32_t* ctr) {
    atom_global_add_u32_release_gpu(ctr, 1u);
}
__global__ void k_atom_acquire_gpu(uint32_t* ctr) {
    atom_global_add_u32_acquire_gpu(ctr, 1u);
}
__global__ void k_atom_acq_rel_gpu(uint32_t* ctr) {
    atom_global_add_u32_acq_rel_gpu(ctr, 1u);
}
__global__ void k_atom_relaxed_cta(uint32_t* ctr) {
    atom_global_add_u32_relaxed_cta(ctr, 1u);
}
__global__ void k_atom_relaxed_sys(uint32_t* ctr) {
    atom_global_add_u32_relaxed_sys(ctr, 1u);
}
__global__ void k_atom_cas_release(uint32_t* addr, uint32_t cmp, uint32_t v) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        atom_global_cas_u32_release_gpu(addr, cmp, v);
}

static int run_sem_scope() {
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;
    const int BLOCKS = 32, THREADS = 64, EXPECTED = BLOCKS * THREADS;

    auto check = [&](const char* name, void (*kernel)(uint32_t*)) {
        uint32_t* d; CUDA_CHECK(cudaMalloc(&d, 4));
        CUDA_CHECK(cudaMemset(d, 0, 4));
        kernel<<<BLOCKS, THREADS>>>(d);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t h; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
        cudaFree(d);
        if (h == (uint32_t)EXPECTED) {
            printf("  %s: OK (%u)\n", name, h);
        } else {
            printf("  %s: FAIL (got %u, expected %d)\n", name, h, EXPECTED);
            all_pass = false;
        }
    };

    check("add.release.gpu",  k_atom_release_gpu);
    check("add.acquire.gpu",  k_atom_acquire_gpu);
    check("add.acq_rel.gpu",  k_atom_acq_rel_gpu);
    check("add.relaxed.cta",  k_atom_relaxed_cta);
    check("add.relaxed.sys",  k_atom_relaxed_sys);

    // CAS release: split-K writer pattern
    uint32_t* d_cas; CUDA_CHECK(cudaMalloc(&d_cas, 4));
    uint32_t init = 7u;
    CUDA_CHECK(cudaMemcpy(d_cas, &init, 4, cudaMemcpyHostToDevice));
    k_atom_cas_release<<<1, 1>>>(d_cas, 7u, 42u);
    CUDA_CHECK(cudaDeviceSynchronize());
    uint32_t h_cas; CUDA_CHECK(cudaMemcpy(&h_cas, d_cas, 4, cudaMemcpyDeviceToHost));
    cudaFree(d_cas);
    if (h_cas == 42u) printf("  cas.release.gpu: OK (%u)\n", h_cas);
    else { printf("  cas.release.gpu: FAIL (got %u)\n", h_cas); all_pass = false; }

    if (all_pass) { PASS(); return 0; }
    else { FAIL("some sem/scope subtests failed"); return 1; }
}

// MED items: signed min/max, bitwise, cas.b64
__global__ void k_atom_min_s32(int32_t* addr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    atom_global_min_s32(addr, -tid);  // min over -[0..n-1] = -(n-1)
}
__global__ void k_atom_max_s32(int32_t* addr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    atom_global_max_s32(addr, tid);   // max = n-1
}
__global__ void k_atom_or_b32(uint32_t* addr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    atom_global_or_b32(addr, 1u << (tid & 31));
}
__global__ void k_atom_and_b32(uint32_t* addr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    atom_global_and_b32(addr, ~(1u << (tid & 31)));  // clear bit tid mod 32
}
__global__ void k_atom_xor_b32(uint32_t* addr) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    atom_global_xor_b32(addr, 1u << (tid & 31));
}
__global__ void k_atom_cas_u64(uint64_t* addr) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        atom_global_cas_u64(addr, 0ull, 0xDEADBEEFCAFEBABEull);
}

static int run_med() {
    bool all_pass = true;
    const int B = 16, T = 64, N = B * T;

    // signed min: should converge to -(N-1)
    {
        int32_t* d; CUDA_CHECK(cudaMalloc(&d, 4));
        int32_t init = INT32_MAX; CUDA_CHECK(cudaMemcpy(d, &init, 4, cudaMemcpyHostToDevice));
        k_atom_min_s32<<<B, T>>>(d);
        CUDA_CHECK(cudaDeviceSynchronize());
        int32_t h; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
        if (h == -(N - 1)) printf("  min.s32: OK (%d)\n", h);
        else { printf("  min.s32: FAIL (%d, expected %d)\n", h, -(N - 1)); all_pass = false; }
        cudaFree(d);
    }
    // signed max: should converge to N-1
    {
        int32_t* d; CUDA_CHECK(cudaMalloc(&d, 4));
        int32_t init = INT32_MIN; CUDA_CHECK(cudaMemcpy(d, &init, 4, cudaMemcpyHostToDevice));
        k_atom_max_s32<<<B, T>>>(d);
        CUDA_CHECK(cudaDeviceSynchronize());
        int32_t h; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
        if (h == N - 1) printf("  max.s32: OK (%d)\n", h);
        else { printf("  max.s32: FAIL (%d, expected %d)\n", h, N - 1); all_pass = false; }
        cudaFree(d);
    }
    // OR: every bit (0..31) set by some thread (tid & 31). Result = 0xFFFFFFFF.
    {
        uint32_t* d; CUDA_CHECK(cudaMalloc(&d, 4));
        CUDA_CHECK(cudaMemset(d, 0, 4));
        k_atom_or_b32<<<B, T>>>(d);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t h; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
        if (h == 0xFFFFFFFFu) printf("  or.b32: OK (0x%x)\n", h);
        else { printf("  or.b32: FAIL (0x%x)\n", h); all_pass = false; }
        cudaFree(d);
    }
    // AND: clear bits 0..31 -> result = 0.
    {
        uint32_t* d; CUDA_CHECK(cudaMalloc(&d, 4));
        uint32_t init = 0xFFFFFFFFu; CUDA_CHECK(cudaMemcpy(d, &init, 4, cudaMemcpyHostToDevice));
        k_atom_and_b32<<<B, T>>>(d);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t h; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
        if (h == 0u) printf("  and.b32: OK (0x%x)\n", h);
        else { printf("  and.b32: FAIL (0x%x, expected 0)\n", h); all_pass = false; }
        cudaFree(d);
    }
    // XOR: each bit toxored an even number of times (B*T/32 = 32 threads per
    // bit, even) -> all zeroes regardless of init.
    {
        uint32_t* d; CUDA_CHECK(cudaMalloc(&d, 4));
        CUDA_CHECK(cudaMemset(d, 0, 4));
        k_atom_xor_b32<<<B, T>>>(d);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t h; CUDA_CHECK(cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost));
        // 32 threads per bit (each pair cancels): result == 0
        if (h == 0u) printf("  xor.b32: OK (0x%x)\n", h);
        else { printf("  xor.b32: FAIL (0x%x)\n", h); all_pass = false; }
        cudaFree(d);
    }
    // cas.b64
    {
        uint64_t* d; CUDA_CHECK(cudaMalloc(&d, 8));
        uint64_t init = 0ull; CUDA_CHECK(cudaMemcpy(d, &init, 8, cudaMemcpyHostToDevice));
        k_atom_cas_u64<<<1, 1>>>(d);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t h; CUDA_CHECK(cudaMemcpy(&h, d, 8, cudaMemcpyDeviceToHost));
        if (h == 0xDEADBEEFCAFEBABEull) printf("  cas.b64: OK (0x%llx)\n", (unsigned long long)h);
        else { printf("  cas.b64: FAIL (0x%llx)\n", (unsigned long long)h); all_pass = false; }
        cudaFree(d);
    }

    if (all_pass) { PASS(); return 0; }
    else { FAIL("MED subtests failed"); return 1; }
}

int main() {
  int rc_ours       = run_ours();
  int rc_theirs     = run_theirs();
  int rc_sem_scope  = run_sem_scope();
  int rc_med        = run_med();
  return (rc_ours == 0 && rc_theirs == 0 && rc_sem_scope == 0 && rc_med == 0) ? 0 : 1;
}

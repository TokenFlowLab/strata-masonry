// test_helpers.cuh -- shared test infrastructure. Not a primitive; lives
// outside the numbered dependency chain.
//
// Helpers are hoisted out of arch macro guards so the same names work for
// tests built under -DPL_AGENTIC_SM90A, -DPL_AGENTIC_SM100A, or
// -DPL_AGENTIC_SM103A.
//
// Contents:
//   Host-side
//     CUDA_CHECK(stmt)              cudaError_t assertion (exit 1 on fail)
//     DRIVER_CHECK(stmt)            CUresult assertion (exit 1 on fail)
//     PASS()                        printf "PASS <file>"; return 0
//     FAIL(fmt, ...)                printf "FAIL <file>: <fmt>"; return 1
//     GpuTimer                      cudaEvent timer with stream support
//     fill_random_{f32,f16,u8,i8}   in-place random fill
//     fill_zeros_f32                in-place zero
//     cpu_gemm_{f32,f16}            host reference GEMM (row-major)
//     check_close_f32               relative-tol compare with mismatch report
//     check_equal_u32               exact compare with mismatch report
//   Device-side
//     smem_ptr_u32(ptr)             generic ptr -> .shared .b32 address
//     mbarrier_init_helper          mbarrier.init.shared::cta.b64
//     mbarrier_try_wait_parity      spin until phase flips

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <sys/stat.h>


// -----------------------------------------------------------------------------
// Host-side: error checks
// -----------------------------------------------------------------------------

#define CUDA_CHECK(stmt) do {                                                 \
    cudaError_t _e = (stmt);                                                  \
    if (_e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d: %s -> %s\n",                         \
              __FILE__, __LINE__, #stmt, cudaGetErrorString(_e));             \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define DRIVER_CHECK(stmt) do {                                               \
    CUresult _e = (stmt);                                                     \
    if (_e != CUDA_SUCCESS) {                                                 \
      const char* _msg = nullptr;                                             \
      cuGetErrorString(_e, &_msg);                                            \
      fprintf(stderr, "CUDA driver error %s:%d: %s -> %s\n",                  \
              __FILE__, __LINE__, #stmt, _msg ? _msg : "?");                  \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

// -----------------------------------------------------------------------------
// Host-side: PASS / FAIL (return 0 or 1 from the calling main)
// -----------------------------------------------------------------------------

#define PASS() do { printf("PASS %s\n", __FILE__); return 0; } while (0)

#define FAIL(...) do {                                                        \
    fprintf(stderr, "FAIL %s: ", __FILE__);                                   \
    fprintf(stderr, __VA_ARGS__);                                             \
    fprintf(stderr, "\n");                                                    \
    return 1;                                                                 \
  } while (0)

// -----------------------------------------------------------------------------
// Host-side: GPU timer (cudaEvent-based, stream-aware)
// -----------------------------------------------------------------------------

struct GpuTimer {
  cudaEvent_t start, stop;
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  void begin(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start, s)); }
  void end(cudaStream_t s = 0) {
    CUDA_CHECK(cudaEventRecord(stop, s));
    CUDA_CHECK(cudaEventSynchronize(stop));
  }
  float elapsed_ms() {
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

// CUDA-graph bench: capture one launch(stream) into a graph, replay `iters` times on a
// dedicated stream, return avg ms. Removes host launch overhead -> pure kernel time
// (symmetric with a graph-timed reference). `launch` must issue all work on the passed stream.
template <typename LaunchFn>
inline double bench_graph_ms(LaunchFn launch, int iters = 20, int warmup = 5) {
  cudaStream_t st;
  CUDA_CHECK(cudaStreamCreate(&st));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
  launch(st);
  CUDA_CHECK(cudaStreamEndCapture(st, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  for (int i = 0; i < warmup; ++i) CUDA_CHECK(cudaGraphLaunch(exec, st));
  CUDA_CHECK(cudaStreamSynchronize(st));
  GpuTimer t;
  t.begin(st);
  for (int i = 0; i < iters; ++i) CUDA_CHECK(cudaGraphLaunch(exec, st));
  t.end(st);
  double ms = t.elapsed_ms() / iters;
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(st);
  return ms;
}

// -----------------------------------------------------------------------------
// Host-side: random / zero fill
// -----------------------------------------------------------------------------

inline void fill_random_f32(float* p, int n, float lo = -1.f, float hi = 1.f) {
  for (int i = 0; i < n; ++i)
    p[i] = lo + (hi - lo) * ((float)rand() / RAND_MAX);
}

inline void fill_random_f16(half* p, int n, float lo = -1.f, float hi = 1.f) {
  for (int i = 0; i < n; ++i)
    p[i] = __float2half(lo + (hi - lo) * ((float)rand() / RAND_MAX));
}

inline void fill_random_u8(uint8_t* p, int n) {
  for (int i = 0; i < n; ++i) p[i] = (uint8_t)(rand() & 0xFF);
}

inline void fill_random_i8(int8_t* p, int n) {
  for (int i = 0; i < n; ++i) p[i] = (int8_t)((rand() % 256) - 128);
}

inline void fill_zeros_f32(float* p, int n) {
  for (int i = 0; i < n; ++i) p[i] = 0.f;
}

// -----------------------------------------------------------------------------
// Host-side: CPU reference GEMM (row-major, C = A * B)
// -----------------------------------------------------------------------------

inline void cpu_gemm_f32(const float* A, const float* B, float* C,
                         int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += A[m * K + k] * B[k * N + n];
      C[m * N + n] = acc;
    }
  }
}

inline void cpu_gemm_f16(const half* A, const half* B, float* C,
                         int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k)
        acc += __half2float(A[m * K + k]) * __half2float(B[k * N + n]);
      C[m * N + n] = acc;
    }
  }
}

// -----------------------------------------------------------------------------
// Host-side: comparison with mismatch reporting
// -----------------------------------------------------------------------------

inline bool check_close_f32(const float* ref, const float* test, int n,
                            float atol = 1e-3f, float rtol = 1e-3f) {
  int mismatches = 0;
  for (int i = 0; i < n; ++i) {
    float diff = fabsf(ref[i] - test[i]);
    float tol  = atol + rtol * fabsf(ref[i]);
    if (diff > tol) {
      if (mismatches < 5)
        printf("  mismatch [%d]: ref=%.6f test=%.6f diff=%.6f tol=%.6f\n",
               i, ref[i], test[i], diff, tol);
      ++mismatches;
    }
  }
  if (mismatches > 0) printf("  total mismatches: %d / %d\n", mismatches, n);
  return mismatches == 0;
}

// -----------------------------------------------------------------------------
// CPU-reference disk cache
// -----------------------------------------------------------------------------
// The CPU reference does the same O(B*H*S^2*D) math as the kernel, so it costs
// minutes at large seqlen. When inputs are a pure function of shape (fixed-seed
// fills), the fp32 reference depends only on that shape -- so we cache it to
// disk keyed by a shape string and skip recompute on later runs.
//
//   key    : a string uniquely identifying the shape+inputs (caller builds it,
//            e.g. "B128_S240_hq32_hk4_hd128_c0"). Bump it if inputs change.
//   ref/n  : destination buffer (n fp32 elements) filled on a cache miss.
//   compute: callable that fills ref[0..n) with the reference when we miss.
//
// Returns true if loaded from cache (compute skipped), false if it computed.
// Cache lives under $FMHA_REF_CACHE, else PL_REF_CACHE_DIR (set by the build).
template <class ComputeFn>
inline bool cached_ref_f32(const std::string& key, float* ref, size_t n,
                           ComputeFn&& compute) {
  const char* dirc = getenv("FMHA_REF_CACHE");
#ifdef PL_REF_CACHE_DIR
  std::string dir  = dirc ? dirc : PL_REF_CACHE_DIR;
#else
  std::string dir  = dirc ? dirc : "verify_cache";
#endif
  std::string path = dir + "/ref_" + key + ".f32";

  const uint32_t kMagic = 0x46524546u;  // "FREF"
  if (FILE* f = fopen(path.c_str(), "rb")) {
    uint32_t magic = 0; uint64_t stored_n = 0;
    if (fread(&magic, sizeof magic, 1, f) == 1 && magic == kMagic &&
        fread(&stored_n, sizeof stored_n, 1, f) == 1 && stored_n == n &&
        fread(ref, sizeof(float), n, f) == n) {
      fclose(f);
      printf("  [ref-cache] hit  %s\n", path.c_str());
      return true;
    }
    fclose(f);  // stale/corrupt header -> fall through and recompute
  }

  printf("  [ref-cache] miss %s (computing CPU reference...)\n", path.c_str());
  compute();

  mkdir(dir.c_str(), 0777);  // ignore EEXIST
  if (FILE* f = fopen(path.c_str(), "wb")) {
    uint64_t nn = n;
    fwrite(&kMagic, sizeof kMagic, 1, f);
    fwrite(&nn, sizeof nn, 1, f);
    fwrite(ref, sizeof(float), n, f);
    fclose(f);
  } else {
    printf("  [ref-cache] WARN could not write %s\n", path.c_str());
  }
  return false;
}

inline bool check_equal_u32(const uint32_t* ref, const uint32_t* test, int n) {
  int mismatches = 0;
  for (int i = 0; i < n; ++i) {
    if (ref[i] != test[i]) {
      if (mismatches < 5)
        printf("  mismatch [%d]: ref=%u test=%u\n", i, ref[i], test[i]);
      ++mismatches;
    }
  }
  if (mismatches > 0) printf("  total mismatches: %d / %d\n", mismatches, n);
  return mismatches == 0;
}

// -----------------------------------------------------------------------------
// Device-side helpers
// -----------------------------------------------------------------------------

__device__ __forceinline__ void mbarrier_init_helper(
    uint32_t mbar_smem, uint32_t arrive_count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               :: "r"(mbar_smem), "r"(arrive_count));
}

// (mbarrier_try_wait_parity helpers live in primitives/33_mbarrier_try_wait.cuh:
//   bool mbarrier_try_wait_parity(addr, parity)        single non-blocking check
//   void mbarrier_wait_parity(addr, parity)   blocking spin loop)


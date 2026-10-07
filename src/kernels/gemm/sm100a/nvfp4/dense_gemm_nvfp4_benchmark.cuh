#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

// Shared host-side benchmarking for dense NVFP4 GEMM implementations.
// The caller owns packed inputs, scales, preallocated buffers, launch
// configuration, and correctness checks. No clock or tensor-allocation changes.
namespace dense_gemm_nvfp4_benchmark {

inline void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "NVFP4 benchmark: %s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

// launch(iteration) must enqueue one kernel on stream and return cudaError_t.
// Iteration indices continue across warmup and timing; a caller may use them to
// select preallocated buffer sets. The current kernel uses one buffer set.
// The result is batch-mean latency in milliseconds, not a median.
template <class Launch>
double measure(cudaStream_t stream, Launch&& launch,
               int warmup_iterations = 10, int timed_iterations = 100) {
  if (warmup_iterations < 0 || timed_iterations <= 0) {
    std::fprintf(stderr, "NVFP4 benchmark: require warmup >= 0 and timed > 0\n");
    std::exit(EXIT_FAILURE);
  }

  cudaEvent_t begin, end;
  check_cuda(cudaEventCreate(&begin));
  check_cuda(cudaEventCreate(&end));
  for (int i = 0; i < warmup_iterations; ++i) {
    check_cuda(launch(static_cast<std::size_t>(i)));
  }
  check_cuda(cudaEventRecord(begin, stream));
  for (int i = 0; i < timed_iterations; ++i) {
    check_cuda(launch(static_cast<std::size_t>(warmup_iterations) + i));
  }
  check_cuda(cudaEventRecord(end, stream));
  check_cuda(cudaStreamSynchronize(stream));
  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, begin, end));
  check_cuda(cudaEventDestroy(begin));
  check_cuda(cudaEventDestroy(end));
  return static_cast<double>(elapsed_ms) / timed_iterations;
}

// Sweep mode keeps its own format; single-case mode prints the standard one-line format.
inline double report(int M, int N, int K, double mean_ms, bool static_schedule,
                     int block_m, int block_n, int num_clusters, int total_tiles,
                     bool timed = true, bool standard_output = false) {
  if (!timed) {
    std::printf("  shape (%6d, %6d, %4d): untimed\n", M, N, K);
    return 0.0;
  }
  const double tflops = 2.0 * M * N * K / (mean_ms * 1e-3) / 1e12;
  if (standard_output) {
    std::printf("  shape (%6d, %6d, %4d): mean=%.6f ms  %7.1f NVFP4_TFLOPS\n",
                M, N, K, mean_ms, tflops);
  } else if (static_schedule) {
    std::printf("  (%6d,%6d,%4d) static bm%d bn%d: %6.3f ms  %7.1f NVFP4_TFLOPS"
                " [%d/%d clusters]\n",
                M, N, K, block_m, block_n, mean_ms, tflops, num_clusters, total_tiles);
  } else {
    std::printf("  (%6d,%6d,%4d): %6.3f ms  %7.1f NVFP4_TFLOPS\n",
                M, N, K, mean_ms, tflops);
  }
  return tflops;
}

}  // namespace dense_gemm_nvfp4_benchmark

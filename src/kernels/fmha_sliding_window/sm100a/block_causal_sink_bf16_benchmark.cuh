#pragma once

#include <algorithm>
#include "../../fmha/sm100a/fmha_context_bf16_benchmark.cuh"

// Host-only timing for all block-causal/sink implementations. The launch closure
// owns its stream and existing launch policy. No input preparation is timed.
namespace block_causal_sink_bf16_benchmark {
using fmha_context_bf16_benchmark::check_cuda;
using fmha_context_bf16_benchmark::env_count;

template <class Launch>
double measure(Launch&& launch) {
  const int warmup = env_count("WARMUP", 3, 0);
  const int iterations = env_count("ITERS", 20, 1);
  if (env_count("NO_BENCHMARK", 0, 0, 1)) {
    launch();
    check_cuda(cudaDeviceSynchronize());
    return 0.0;
  }
  for (int i = 0; i < warmup; ++i) launch();
  check_cuda(cudaDeviceSynchronize());
  cudaEvent_t begin, end;
  check_cuda(cudaEventCreate(&begin));
  check_cuda(cudaEventCreate(&end));
  check_cuda(cudaEventRecord(begin));
  for (int i = 0; i < iterations; ++i) launch();
  check_cuda(cudaEventRecord(end));
  check_cuda(cudaEventSynchronize(end));
  float elapsed = 0;
  check_cuda(cudaEventElapsedTime(&elapsed, begin, end));
  check_cuda(cudaEventDestroy(begin));
  check_cuda(cudaEventDestroy(end));
  std::printf("  benchmark: mode=direct warmup=%d iters=%d buffers=1 cache=L2-warm"
              " statistic=batch-mean\n", warmup, iterations);
  return elapsed / static_cast<double>(iterations);
}

// Count the union of sink and rolling window, never their overlap twice.
inline std::uint64_t context_keys(long end, long window_start, long sink) {
  const long sink_end = std::min(end, sink);
  return sink_end + std::max(0L, end - std::max(sink_end, window_start));
}

inline std::uint64_t attended_pairs(long length, bool bcs, bool causal, long block,
                                    long sink, long rolling, long half = 0) {
  if (!bcs) return fmha_context_bf16_benchmark::attended_pairs(length, length, causal);
  if (block <= 0) {
    std::fprintf(stderr, "BCS benchmark: block size must be positive\n");
    std::exit(EXIT_FAILURE);
  }
  std::uint64_t pairs = 0;
  const long limit = half ? half : length;
  for (long start = 0; start < limit; start += block) {
    const long end = std::min(start + block, limit);
    const long rows = end - start;
    const long window_start = std::max(0L, start + block - rolling);
    pairs += rows * context_keys(end, window_start, sink);
    if (half) pairs += rows * (context_keys(start, window_start, sink) + rows);
  }
  return pairs;
}
}  // namespace block_causal_sink_bf16_benchmark

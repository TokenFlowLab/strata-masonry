#pragma once

#include "../../../fmha/sm100a/fmha_context_bf16_benchmark.cuh"
#include <algorithm>
#include <vector>

// Host-only backward timing. The caller owns all buffers, launch arguments,
// forward-state preparation and correctness checks. No device algorithm changes.
namespace block_sparse_bwd_bf16_benchmark {

namespace common = fmha_context_bf16_benchmark;

struct Options {
  int warmup_iterations = 10;
  int timed_iterations = 50;
};

inline Options options_from_env() {
  return {common::env_count("BENCH_WARMUP", 10, 0),
          common::env_count("BENCH_ITERS", 50, 1)};
}

inline bool enabled() {
  return common::env_count("NO_BENCHMARK", 0, 0, 1) == 0;
}

// launch() enqueues one complete backward invocation on the default stream and
// checks its launch errors. Preserve the existing per-invocation CUDA-event
// upper median (sorted samples[iters/2]), including for even iteration counts.
// Same buffers are reused: L2-warm, not cold-buffer or batch-mean measurements.
template <class Launch>
double measure(Launch&& launch, const Options& options = options_from_env()) {
  if (options.warmup_iterations < 0 || options.timed_iterations <= 0) {
    std::fprintf(stderr, "Backward benchmark: require warmup >= 0 and timed > 0\n");
    std::exit(EXIT_FAILURE);
  }
  for (int i = 0; i < options.warmup_iterations; ++i) launch();
  common::check_cuda(cudaDeviceSynchronize());
  cudaEvent_t begin, end;
  common::check_cuda(cudaEventCreate(&begin));
  common::check_cuda(cudaEventCreate(&end));
  std::vector<float> samples(options.timed_iterations);
  for (int i = 0; i < options.timed_iterations; ++i) {
    common::check_cuda(cudaEventRecord(begin));
    launch();
    common::check_cuda(cudaEventRecord(end));
    common::check_cuda(cudaEventSynchronize(end));
    common::check_cuda(cudaEventElapsedTime(&samples[i], begin, end));
  }
  common::check_cuda(cudaEventDestroy(begin));
  common::check_cuda(cudaEventDestroy(end));
  std::sort(samples.begin(), samples.end());
  std::printf("  benchmark: mode=direct warmup=%d iters=%d buffers=1 cache=L2-warm"
              " statistic=upper-median\n", options.warmup_iterations, options.timed_iterations);
  return samples[options.timed_iterations / 2];
}

inline double tflops(int head_dim, double selected_pairs, double milliseconds) {
  if (!std::isfinite(milliseconds) || milliseconds <= 0) {
    std::fprintf(stderr, "Backward benchmark: latency must be finite and positive\n");
    std::exit(EXIT_FAILURE);
  }
  return 2.5 * 4.0 * head_dim * selected_pairs / (milliseconds * 1e-3) / 1e12;
}

}  // namespace block_sparse_bwd_bf16_benchmark

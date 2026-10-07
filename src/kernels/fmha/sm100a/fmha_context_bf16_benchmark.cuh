#pragma once

#include <cuda_runtime.h>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Shared host-side timing and logical FLOP accounting for dense BF16 context FMHA.
// Callers own input generation, preallocated tensors, launch geometry and verification.
// No clock changes, tensor allocation, copies or correctness work in the timed loop.
namespace fmha_context_bf16_benchmark {

inline void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "FMHA benchmark: %s\n", cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}

inline int env_count(const char* name, int fallback, int minimum, int maximum = INT_MAX) {
  const char* text = std::getenv(name);
  if (!text) return fallback;
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(text, &end, 10);
  if (errno || end == text || *end || value < minimum || value > maximum) {
    std::fprintf(stderr, "FMHA benchmark: %s must be an integer in [%d, %d]\n",
                 name, minimum, maximum);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

struct Options {
  int warmup_iterations = 3;
  int timed_iterations = 20;
  bool graph = false;
};

inline Options options_from_env() {
  Options options;
  options.graph = env_count("GRAPH", 0, 0, 1) != 0;
  // Preserve the old direct (3/20) and graph (5/20) defaults.
  options.warmup_iterations = env_count("WARMUP", options.graph ? 5 : 3, 0);
  options.timed_iterations = env_count("ITERS", 20, 1);
  return options;
}

// launch(stream) enqueues one kernel on the supplied stream and returns cudaError_t.
// GRAPH=1 captures that launch once and replays it; capture is outside timing.
// The same tensor buffers are reused: these are L2-warm, batch-mean timings, not
// cold-buffer measurements or per-launch medians. Graph timing can still include
// host submission gaps; it is not guaranteed to be pure device execution time.
template <class Launch>
double measure(Launch&& launch, const Options& options = options_from_env()) {
  if (options.warmup_iterations < 0 || options.timed_iterations <= 0) {
    std::fprintf(stderr, "FMHA benchmark: require warmup >= 0 and timed > 0\n");
    std::exit(EXIT_FAILURE);
  }

  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  cudaEvent_t begin, end;
  check_cuda(cudaEventCreate(&begin));
  check_cuda(cudaEventCreate(&end));
  if (options.graph) {
    check_cuda(cudaStreamCreate(&stream));
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    check_cuda(launch(stream));
    check_cuda(cudaStreamEndCapture(stream, &graph));
    check_cuda(cudaGraphInstantiate(&executable, graph, 0));
  }
  auto enqueue = [&]() {
    return options.graph ? cudaGraphLaunch(executable, stream) : launch(stream);
  };
  for (int i = 0; i < options.warmup_iterations; ++i) check_cuda(enqueue());
  check_cuda(cudaStreamSynchronize(stream));

  check_cuda(cudaEventRecord(begin, stream));
  for (int i = 0; i < options.timed_iterations; ++i) check_cuda(enqueue());
  check_cuda(cudaEventRecord(end, stream));
  check_cuda(cudaEventSynchronize(end));
  float elapsed_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&elapsed_ms, begin, end));

  check_cuda(cudaEventDestroy(begin));
  check_cuda(cudaEventDestroy(end));
  if (options.graph) {
    check_cuda(cudaGraphExecDestroy(executable));
    check_cuda(cudaGraphDestroy(graph));
    check_cuda(cudaStreamDestroy(stream));
  }
  std::printf("  benchmark: mode=%s warmup=%d iters=%d buffers=1 cache=L2-warm"
              " statistic=batch-mean\n",
              options.graph ? "graph" : "direct", options.warmup_iterations,
              options.timed_iterations);
  return static_cast<double>(elapsed_ms) / options.timed_iterations;
}

// Top-left causal: query i attends keys j <= i, also for unequal Q/KV lengths.
inline std::uint64_t attended_pairs(std::uint64_t sq, std::uint64_t sk, bool causal) {
  if (!causal) return sq * sk;
  const std::uint64_t m = sq < sk ? sq : sk;
  return m * (m + 1) / 2 + (sq > sk ? (sq - sk) * sk : 0);
}

// QK and PV multiply-adds only; excludes softmax and padding. Preserve the existing
// result line consumed by benchmark scripts, and return useful TFLOPS.
inline double report(const char* label, long query_tokens, bool causal, int head_dim,
                     int query_heads, std::uint64_t pairs, double mean_ms) {
  if (!std::isfinite(mean_ms) || mean_ms <= 0.0) {
    std::fprintf(stderr, "FMHA benchmark: latency must be finite and positive\n");
    std::exit(EXIT_FAILURE);
  }
  const double tflops = 4.0 * head_dim * query_heads * static_cast<double>(pairs)
                       / (mean_ms * 1e-3) / 1e12;
  std::printf("  [%s] N_q=%ld causal=%d  %.4f ms  %.1f TFLOPS\n",
              label, query_tokens, static_cast<int>(causal), mean_ms, tflops);
  return tflops;
}

}  // namespace fmha_context_bf16_benchmark

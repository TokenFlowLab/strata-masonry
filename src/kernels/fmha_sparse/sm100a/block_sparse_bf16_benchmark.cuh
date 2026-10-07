#pragma once

#include "../../fmha/sm100a/fmha_context_bf16_benchmark.cuh"
#include <cuda_bf16.h>
#include <algorithm>
#include <string>
#include <vector>

// Shared host interface. The kernel owns tensors, sparse metadata and launch geometry.
// Reuse the dense FMHA event/graph timer; never alter clocks or allocate inside its timed loop.
namespace block_sparse_bf16_benchmark {

using fmha_context_bf16_benchmark::check_cuda;
using fmha_context_bf16_benchmark::env_count;

// Host-side deterministic inputs, reproduced by block_sparse_bf16_cpu_verifier.py.
// Arithmetic is explicitly uint32, FP32, then round-to-nearest-even BF16.
inline unsigned input_seed() { return static_cast<unsigned>(env_count("INPUT_SEED", 0, 0)); }

inline void fill(__nv_bfloat16* values, long count, unsigned tensor_seed) {
  const unsigned seed = tensor_seed + input_seed();
  for (long i = 0; i < count; ++i) {
    uint32_t x = static_cast<uint32_t>(i) * 2654435761u + seed * 40503u + 0x9e3779b9u;
    x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
    values[i] = __float2bfloat16(static_cast<float>(x % 2039u) / 1019.5f - 1.0f);
  }
}

inline void select_blocks(std::vector<int>& indices, int num_blocks, int topk) {
  if (num_blocks <= 0 || topk <= 0 || topk > num_blocks ||
      indices.size() % (static_cast<size_t>(num_blocks) * topk)) {
    std::fprintf(stderr, "Block-sparse inputs: invalid block-list dimensions\n");
    std::exit(EXIT_FAILURE);
  }
  std::vector<int> permutation(num_blocks), selected(topk);
  for (int qb = 0; qb < num_blocks; ++qb) {
    for (int i = 0; i < num_blocks; ++i) permutation[i] = i;
    uint32_t state = static_cast<uint32_t>(qb) * 2654435761u + 12345u + input_seed();
    for (int i = 0; i < topk; ++i) {
      state ^= state << 13; state ^= state >> 17; state ^= state << 5;
      const int j = i + state % static_cast<uint32_t>(num_blocks - i);
      std::swap(permutation[i], permutation[j]);
      selected[i] = permutation[i];
    }
    std::sort(selected.begin(), selected.end());
    for (size_t row = qb; row < indices.size() / topk; row += num_blocks)
      std::copy(selected.begin(), selected.end(), indices.begin() + row * topk);
  }
}

inline bool benchmark_enabled() { return env_count("NO_BENCHMARK", 0, 0, 1) == 0; }

template <class Launch>
double measure(Launch&& launch) {
  if (const char* directory = std::getenv("LOAD_NPY"))
    std::printf("  inputs: mode=files directory=%s\n", directory);
  else
    std::printf("  inputs: mode=seeded seed=%u generator=hash32-fy-v1\n", input_seed());
  if (!benchmark_enabled()) {
    check_cuda(launch(nullptr));
    check_cuda(cudaDeviceSynchronize());
    std::printf("  benchmark: disabled (one untimed launch)\n");
    return 0.0;
  }
  auto options = fmha_context_bf16_benchmark::options_from_env();
  // Sparse drivers default to WARMUP=3, for both the direct and the graph path.
  options.warmup_iterations = env_count("WARMUP", 3, 0);
  return fmha_context_bf16_benchmark::measure(launch, options);
}

inline double tflops(double pairs, int head_dim, double mean_ms) {
  if (!benchmark_enabled()) return 0.0;
  if (!std::isfinite(pairs) || pairs <= 0 || !std::isfinite(mean_ms) || mean_ms <= 0) {
    std::fprintf(stderr, "Block-sparse benchmark: invalid work count or latency\n");
    std::exit(EXIT_FAILURE);
  }
  return 4.0 * head_dim * pairs / (mean_ms * 1e9);
}

// DUMP_O=<prefix> -> <prefix>.out, raw little-endian FP32 [S,H,D].
// Every BF16 output element is copied and expanded; no sampling or reference work here.
inline void dump_output(const __nv_bfloat16* output, size_t elements) {
  const char* prefix = std::getenv("DUMP_O");
  if (!prefix) return;
  const std::string path = std::string(prefix) + ".out";
  std::vector<__nv_bfloat16> bits(elements);
  check_cuda(cudaMemcpy(bits.data(), output, elements * sizeof(*output), cudaMemcpyDeviceToHost));
  std::vector<float> values(elements);
  for (size_t i = 0; i < elements; ++i) values[i] = __bfloat162float(bits[i]);
  FILE* file = std::fopen(path.c_str(), "wb");
  if (!file) { std::perror(path.c_str()); std::exit(EXIT_FAILURE); }
  const bool written = std::fwrite(values.data(), sizeof(float), elements, file) == elements;
  const int closed = std::fclose(file);
  if (!written || closed != 0) {
    std::fprintf(stderr, "Block-sparse benchmark: failed to write %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace block_sparse_bf16_benchmark

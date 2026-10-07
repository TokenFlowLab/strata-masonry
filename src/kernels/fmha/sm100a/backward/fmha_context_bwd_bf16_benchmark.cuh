#pragma once

#include <cuda_bf16.h>
#include <algorithm>
#include <cstring>
#include <regex>
#include <string>
#include <vector>
#include "../fmha_context_bf16_benchmark.cuh"

namespace fmha_context_bwd_bf16_benchmark {
using fmha_context_bf16_benchmark::check_cuda;
using fmha_context_bf16_benchmark::env_count;

inline void fail(const std::string& message) {
  std::fprintf(stderr, "FMHA backward: %s\n", message.c_str());
  std::exit(EXIT_FAILURE);
}

inline int argument(const char* text) {
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(text, &end, 10);
  if (errno || end == text || *end || value < 0 || value > INT_MAX)
    fail(std::string("expected a nonnegative integer, got: ") + text);
  return static_cast<int>(value);
}

inline std::string suffix(int batch, int heads, int seqlen, int dim, bool causal) {
  return "_B" + std::to_string(batch) + "_H" + std::to_string(heads) +
         "_S" + std::to_string(seqlen) + "_D" + std::to_string(dim) +
         "_c" + std::to_string(static_cast<int>(causal)) + ".npy";
}

// Strict reader for NumPy's C-order little-endian v1/v2 arrays. Check dimensions,
// dtype and exact payload length; a matching byte count alone is not a contract.
template <class T>
std::vector<T> load(const std::string& path, const char* dtype,
                    const std::vector<int>& shape) {
  FILE* file = std::fopen(path.c_str(), "rb");
  if (!file) fail("cannot open " + path);
  unsigned char prefix[12]{};
  if (std::fread(prefix, 1, 8, file) != 8 || std::memcmp(prefix, "\x93NUMPY", 6) ||
      (prefix[6] != 1 && prefix[6] != 2) || prefix[7] != 0)
    fail("invalid NumPy header: " + path);
  const int length_bytes = prefix[6] == 1 ? 2 : 4;
  if (std::fread(prefix + 8, 1, length_bytes, file) != static_cast<size_t>(length_bytes))
    fail("truncated header: " + path);
  unsigned length = 0;
  for (int i = 0; i < length_bytes; ++i) length |= unsigned(prefix[8 + i]) << (8 * i);
  if (!length || length > 65536) fail("invalid header size: " + path);
  std::string header(length, ' ');
  if (std::fread(header.data(), 1, length, file) != length) fail("short header: " + path);
  header = std::regex_replace(header, std::regex("\\s+"), "");
  std::string dimensions = "'shape':(";
  size_t count = 1;
  for (int dim : shape) {
    if (dim <= 0 || count > SIZE_MAX / static_cast<size_t>(dim)) fail("invalid dimensions");
    count *= dim;
    dimensions += std::to_string(dim) + ",";
  }
  dimensions.pop_back();
  dimensions += ")";
  if (header.find(std::string("'descr':'") + dtype + "'") == std::string::npos ||
      header.find("'fortran_order':False") == std::string::npos ||
      header.find(dimensions) == std::string::npos)
    fail("wrong dtype, shape or order: " + path);
  std::vector<T> result(count);
  if (std::fread(result.data(), sizeof(T), count, file) != count || std::fgetc(file) != EOF)
    fail("wrong payload size: " + path);
  std::fclose(file);
  return result;
}

struct Inputs {
  std::vector<__nv_bfloat16> q, k, v, output, dout;
  std::vector<float> lse;
};

inline Inputs inputs(int batch, int heads, int seqlen, int dim, bool causal) {
  const char* directory = std::getenv("LOAD_NPY");
  if (!directory || !*directory)
    fail("--bench requires LOAD_NPY with real Q/K/V/dO/O/LSE. Generate it using "
         "fmha_context_bwd_bf16_cpu_verifier.py generate; --verify keeps its built-in reference.");
  const std::string base = std::string(directory) + "/";
  const auto tail = suffix(batch, heads, seqlen, dim, causal);
  const std::vector<int> shape{batch, heads, seqlen, dim};
  auto bf16 = [&](const char* name) {
    auto result = load<__nv_bfloat16>(base + name + tail, "<u2", shape);
    for (auto value : result)
      if (!std::isfinite(__bfloat162float(value))) fail(std::string(name) + ": nonfinite input");
    return result;
  };
  Inputs result{bf16("q"), bf16("k"), bf16("v"), bf16("o"), bf16("do"),
                load<float>(base + "lse" + tail, "<f4", {batch, heads, seqlen})};
  for (float value : result.lse)
    if (!std::isfinite(value)) fail("nonfinite LSE");
  return result;
}

// Default-stream, L2-warm batch-mean measurement.
// The closure includes preprocess + main + dQ postprocess, including accumulator clear.
template <class Launch>
double measure(Launch&& launch, int warmup, int iterations) {
  if (env_count("NO_BENCHMARK", 0, 0, 1)) {
    launch();
    check_cuda(cudaDeviceSynchronize());
    return 0;
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
  std::printf("benchmark: scope=preprocess+main+postprocess cache=L2-warm"
              " buffers=1 statistic=batch-mean\n");
  return elapsed / static_cast<double>(iterations);
}

inline void dump(const char* name, const __nv_bfloat16* device, size_t elements) {
  const char* prefix = std::getenv("DUMP_BWD_GPU");
  if (!prefix) return;
  std::vector<__nv_bfloat16> host(elements);
  check_cuda(cudaMemcpy(host.data(), device, elements * sizeof(host[0]), cudaMemcpyDeviceToHost));
  const std::string path = std::string(prefix) + "_" + name + ".bf16";
  FILE* file = std::fopen(path.c_str(), "wb");
  if (!file) fail("cannot write " + path);
  if (std::fwrite(host.data(), sizeof(host[0]), elements, file) != elements)
    fail("short output write: " + path);
  if (std::fclose(file)) fail("failed output close: " + path);
}
}  // namespace fmha_context_bwd_bf16_benchmark

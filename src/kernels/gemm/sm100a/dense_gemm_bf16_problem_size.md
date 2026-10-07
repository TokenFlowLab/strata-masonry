# Dense GEMM BF16 (SM100a)

Implementation: [dense_gemm_bf16.cu](dense_gemm_bf16.cu).
Case definitions and CPU reference: [dense_gemm_bf16_cpu_verifier.py](dense_gemm_bf16_cpu_verifier.py).

## Computation

```text
D[m, n] = BF16_RNE(sum_k FP32(A[m, k]) * FP32(B[k, n]))
```

| Tensor | Logical shape | Type | Layout |
|---|---|---|---|
| A | M x K | BF16 | Row-major |
| B | K x N | BF16 | Row-major (the driver transposes it to `BT[N, K]` on the host) |
| D | M x N | BF16 | Row-major |

FP32 accumulation, round-to-nearest-even output. No bias, activation or accumulation into D.

## Cases

| Case | Workload M | Executed M | K | N | Purpose |
|---|---:|---:|---:|---:|---|
| smoke | 256 | 256 | 64 | 128 | Small correctness case |
| und-gate | 14046 | 14080 | 2048 | 128 | UND router |
| und-qkv | 14046 | 14080 | 2048 | 5120 | UND fused QKV |
| und-o | 14046 | 14080 | 4096 | 2048 | UND output projection |
| gen-gate | 30720 | 30720 | 2048 | 128 | GEN router |
| gen-qkv | 30720 | 30720 | 2048 | 5120 | GEN fused QKV |
| gen-o | 30720 | 30720 | 4096 | 2048 | GEN output projection |

The six UND/GEN cases are the performance grid. UND executes M padded to 14080, so it
does not exercise unpadded tail handling; FLOPs use the executed M.

## Inputs

For row-major flat index `i`, with unsigned 32-bit wraparound:

```text
x = (i * 2654435761 + seed) mod 2^32
A: i = m * K + k, seed = 1
B: i = k * N + n, seed = 2
```

| Fill | Value before BF16 rounding |
|---|---|
| 1 (default) | (x % 256) / 256 - 0.5 |
| 2 | (x % 2048) / 1024 - 1.0 |
| 3 | 1.0 |
| 4 | (x % 7) - 3 |

## CPU verification

The verifier reads `M * N` little-endian BF16 values, regenerates A/B and computes an FP32
reference. Defaults: fill 1, 64 sampled outputs, sample seed 20260827. A sampled pass is not a
full pass. An element passes if it is finite and
`abs(observed - reference) <= max(0.05, 0.05 * abs(reference))`.
Exit codes: 0 pass, 1 mismatch, 2 invalid input.

## Benchmark

Work is `2 * M * N * K` FLOPs (executed dimensions); `TFLOPS = FLOPs / (seconds * 1e12)`.
[dense_gemm_bf16_benchmark.cuh](dense_gemm_bf16_benchmark.cuh) times 10 warmups and 100 launches
on one buffer set with CUDA events and reports the batch mean (not a median). Allocation, input
generation, the B transpose and verification are outside the timed loop. Results are L2-warm.

```cpp
auto launch = [&](std::size_t iteration) -> cudaError_t {
  return cudaLaunchKernelEx(&config, kernel, /* kernel arguments */);
};
double mean_ms = dense_gemm_bf16_benchmark::measure(stream, launch, 10, 100);
double tflops = dense_gemm_bf16_benchmark::report(M, N, K, mean_ms);
```

Clocks are not locked: use matched inputs, an exclusive GPU, settled warmup and interleaved
A/B runs, and record the fill and timing settings.

## Commands

From the repository root:

```bash
cmake -S . -B build && cmake --build build -j --target kernel_sm100a_dense_gemm_bf16
python3 src/kernels/gemm/sm100a/dense_gemm_bf16_cpu_verifier.py cases
build/bin/kernel_sm100a_dense_gemm_bf16 \
  --shape=256,64,128 --fill=1 --no-benchmark --no-gpu-verify --dump-output=gemm-smoke.bf16
python3 src/kernels/gemm/sm100a/dense_gemm_bf16_cpu_verifier.py verify \
  --case smoke --fill 1 --actual gemm-smoke.bf16
build/bin/kernel_sm100a_dense_gemm_bf16 --fill=2     # performance grid
```

`--shape` is ordered M,K,N; the program prints shapes as M,N,K.

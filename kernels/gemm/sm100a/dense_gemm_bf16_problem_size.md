# Dense GEMM BF16 problem sizes

## Scope

The active SM100a implementation is [dense_gemm_bf16.cu](dense_gemm_bf16.cu).
Files under `stale/` are not part of this subfamily's migration.

This page defines the shared workload contract. Case definitions currently
live in [dense_gemm_bf16_cpu_verifier.py](dense_gemm_bf16_cpu_verifier.py).
Production shapes come from the [repository inventory](../../problem_size.md)
(Cosmos3 rank 0, iteration 5000). No existing case is renamed or resized.

## Computation and tensors

```text
D[m, n] = BF16_RNE(sum_k FP32(A[m, k]) * FP32(B[k, n]))
```

There is no bias, activation, or addition of an existing output matrix.

| Tensor | Logical shape | Type | Layout |
|---|---|---|---|
| A | M x K | BF16 | Row-major |
| B | K x N | BF16 | Row-major before host transpose |
| D | M x N | BF16 | Row-major |

The current host driver transposes B into a row-major `BT[N, K]` device
buffer. This is representation, not a different mathematical operation.
Accumulation is FP32. `BF16_RNE` means round to nearest, ties to even.

## Cases

| Case | Production M | Executed M | K | N | Purpose |
|---|---:|---:|---:|---:|---|
| smoke | 256 | 256 | 64 | 128 | Small correctness case |
| und-gate | 14046 | 14080 | 2048 | 128 | UND router |
| und-qkv | 14046 | 14080 | 2048 | 5120 | UND fused QKV |
| und-o | 14046 | 14080 | 4096 | 2048 | UND output projection |
| gen-gate | 30720 | 30720 | 2048 | 128 | GEN router |
| gen-qkv | 30720 | 30720 | 2048 | 5120 | GEN fused QKV |
| gen-o | 30720 | 30720 | 4096 | 2048 | GEN output projection |

All seven cases have CPU-reference definitions. The six UND/GEN cases are
the existing performance grid; smoke is not a substitute for that grid.

UND executes padded rows. Do not report M=14080 results as evidence of
unpadded M=14046 tail handling. Performance FLOPs use the executed M.

## Input generation

For logical row-major flat index `i`, use unsigned 32-bit wraparound:

```text
x = (i * 2654435761 + seed) mod 2^32
A: i = m * K + k, seed = 1
B: i = k * N + n, seed = 2
```

| Fill | Value before BF16 rounding | Use |
|---|---|---|
| 1 | (x % 256) / 256 - 0.5 | Existing default and historical benchmarks |
| 2 | (x % 2048) / 1024 - 1.0 | Uniform-range fixed-power benchmark input |
| 3 | 1.0 | Constant-input correctness case |
| 4 | (x % 7) - 3 | Signed-integer correctness case |

Round each value to BF16 before computation. All compared implementations
must use identical logical inputs, including any executed padding rows.

## CPU verification

The verifier reads exactly `M * N` little-endian BF16 values from an output
dump, independently regenerates A/B, and computes FP32 reference values.
Current defaults are fill 1, 64 sampled outputs, and sample seed 20260827.

An observed element passes only if it is finite and:

```text
abs(observed - reference) <= max(0.05, 0.05 * abs(reference))
```

These are the existing tolerances, not newly justified accuracy bounds.
The current BF16 verifier compares against the FP32 reference before output
rounding. Exit codes are 0 for pass, 1 for mismatch, and 2 for invalid input.

Full and sampled coverage are separate from case size. Explicit full mode,
chunked computation, and coverage reporting belong to phase 2; they are not
implemented by this documentation change. A sampled pass is not a full pass.

## Performance measurement

Useful GEMM work is `2 * M * N * K` FLOPs, using executed dimensions.
`TFLOPS = FLOPs / (latency_seconds * 1e12)`.

Current timing uses 10 warmups followed by 100 launches on one buffer set.
CUDA-event elapsed time divided by 100 is a mean latency, not a median.
Allocation, input generation, B transpose, and verification are outside it.

The timing loops, CUDA events, mean calculation, and result printing live in
[`dense_gemm_bf16_benchmark.cuh`](dense_gemm_bf16_benchmark.cuh). Implementations
provide their own launch callback; allocation, input preparation, and correctness
stay in the driver. The header has no dependency on `code/tests/`.

```cpp
auto launch = [&](std::size_t iteration) -> cudaError_t {
  // Select already-prepared buffers here, then launch on stream.
  return cudaLaunchKernelEx(&config, kernel, /* kernel arguments */);
};
double mean_ms = dense_gemm_bf16_benchmark::measure(stream, launch, 10, 100);
double tflops = dense_gemm_bf16_benchmark::report(M, N, K, mean_ms);
```

The header accepts independent warmup/timed counts; the current driver retains
10/100. This extraction preserves existing measurements. Remaining methodology
work follows [benchmarking_methodology.md](../../../../knowledge/benchmarking_methodology.md):
existing power limit and DVFS (no clock locking), matched fill 2 inputs,
independent warmup/timed settings, buffer rotation, and exclusive GPU use.
Record the actual fill/cache/timing settings; do not relabel historical
single-buffer fill-1 measurements as methodology-compliant cold-cache runs.

## Existing commands

From `books/`:

```bash
make -C code NATIVE_ARCH=sm_100a kernel_sm100a_dense_gemm_bf16_build
python3 code/kernels/gemm/sm100a/dense_gemm_bf16_cpu_verifier.py cases
code/build/kernel_sm100a_dense_gemm_bf16 \
  --shape=256,64,128 --fill=1 --no-benchmark --no-gpu-verify \
  --dump-output=/tmp/dense-gemm-smoke.bf16
python3 code/kernels/gemm/sm100a/dense_gemm_bf16_cpu_verifier.py verify \
  --case smoke --fill 1 --actual /tmp/dense-gemm-smoke.bf16
```

`--shape` is ordered M,K,N; the program's printed shape is M,N,K. The
current performance grid runs without `--shape`; `--fill=2` selects fill 2.

# Dense GEMM NVFP4 problem sizes

## Scope

The active implementation is [dense_gemm_nvfp4.cu](dense_gemm_nvfp4.cu).
The independent reference is
[dense_gemm_nvfp4_cpu_verifier.py](dense_gemm_nvfp4_cpu_verifier.py).
All implementations of this subfamily must share this workload contract.

This page preserves both the existing manual kernel's benchmark grid and
the KB campaign cases. Single-case execution uses the common KB interface;
the Python verifier currently registers only smoke and flagship.

## Computation and tensors

This describes the repository's existing E2M1 + UE8M0 block32 variant.
The label NVFP4 must not be taken to imply a different scale format or
block size. This phase does not change the kernel's numerical format.

```text
kb = floor(k / 32)
D[m, n] = BF16_RNE(sum_k (A[m, k] * SFA[m, kb])
                       * (B[n, k] * SFB[n, kb]))
```

| Tensor | Logical shape | Type and layout |
|---|---|---|
| A | M x K | Row-major E2M1, two values per byte |
| B | N x K | Row-major E2M1, two values per byte |
| SFA | M x (K / 32) | UE8M0, one scale per row and 32 K elements |
| SFB | N x (K / 32) | UE8M0, one scale per row and 32 K elements |
| D | M x N | Row-major BF16 |

Even K indices occupy the low nibble; odd indices occupy the high nibble.
Scale packing for TMA/TMEM is implementation-specific; logical scale values
must agree. Accumulation is FP32; output uses round-to-nearest, ties-to-even.
There is no bias, activation, or addition of an existing output matrix.

## Cases

### Existing verifier and campaign cases

| Case | M | K | N | Purpose |
|---|---:|---:|---:|---|
| smoke | 256 | 256 | 128 | Small CPU-reference case |
| flagship | 30720 | 8192 | 4096 | Campaign correctness and performance |

`flagship` is the same shape as the manual driver's `2K+2N` case below.
Preserve both existing names as aliases, not different workloads.

### Existing manual kernel performance grid

These are the unchanged `v0_shapes`, `large_shapes`, and `push_shapes`
arrays in `dense_gemm_nvfp4.cu`. Only smoke and flagship are currently
accepted by the standalone Python verifier; extending it is phase 2.

| Case | M | K | N | Grid |
|---|---:|---:|---:|---|
| und-gate | 14080 | 2048 | 128 | V0 |
| und-qkv | 14080 | 2048 | 5120 | V0 |
| und-o | 14080 | 4096 | 2048 | V0 |
| gen-gate | 30720 | 2048 | 128 | V0 |
| gen-qkv | 30720 | 2048 | 5120 | V0 |
| gen-o | 30720 | 4096 | 2048 | V0 |
| gen-o-2K | 30720 | 8192 | 2048 | V0 |
| gen-o-4K | 30720 | 16384 | 2048 | V0 |
| 2K+2N | 30720 | 8192 | 4096 | V0 |
| 4K+2N | 30720 | 16384 | 4096 | V0 |
| 2M+4K+2N | 61440 | 16384 | 4096 | V0 |
| gen-o-2M | 61440 | 4096 | 2048 | Scaling |
| gen-o-4M | 122880 | 4096 | 2048 | Scaling |
| gen-o-2N | 30720 | 4096 | 4096 | Scaling |
| gen-qkv-2M | 61440 | 2048 | 5120 | Scaling |
| gen-qkv-2N | 30720 | 2048 | 10240 | Scaling |
| 16384^3 | 16384 | 16384 | 16384 | Scaling |
| 30720^3 | 30720 | 30720 | 30720 | Scaling |
| 32k-mixed | 32768 | 16384 | 32768 | Scaling |
| 2M+2N | 61440 | 4096 | 4096 | Extended |
| 2M+4N | 61440 | 4096 | 8192 | Extended |
| 2M+8N | 61440 | 4096 | 16384 | Extended |
| 4M+4K+2N | 122880 | 16384 | 4096 | Extended |
| 2M+4K+4N | 61440 | 16384 | 8192 | Extended |
| 4M+4K+4N | 122880 | 16384 | 8192 | Extended |
| 2M+4K+8N | 61440 | 16384 | 16384 | Extended |

UND production M=14046 is padded to executed M=14080, as in the
[repository inventory](../../../problem_size.md). This does not establish
unpadded tail support. Keep scaling studies separate from production cases.

## Input generation

The host driver uses `FILL`, default 1, unless single-case `--fill` overrides it.
The Python verifier uses `--fill`, default 2. Match them explicitly.
All integer arithmetic in the following hash wraps to 32 bits:

```text
hash(x):
    x *= 2654435761
    x ^= x >> 16
    x *= 2246822519
    x ^= x >> 13

x_A = hash(2 * (m * K + k) + 0x9E3779B9)
x_B = hash(2 * (n * K + k) + 0x7F4A7C15)
e_A = hash(m * 131071 + kb * 8191) % 5 - 2
e_B = hash(n * 131071 + kb * 8191 + 977) % 5 - 2
```

| Fill | E2M1 data | SFA / SFB exponents |
|---|---|---|
| 1 | Constant 4.0 | m % 5 - 2 / n % 3 - 1 |
| 2 | Decode x_A & 15 / x_B & 15 | e_A / e_B |
| 3 | Constant 1.0 | m % 5 - 2 / n % 3 - 1 |
| 4 | Signed integers selected as below | e_A / e_B |

For fill 4, the magnitude code is `[0, 2, 4, 5][x & 3]` and the sign bit
is `(x >> 5) & 1`. E2M1 codes 0..7 decode to
`[0, 0.5, 1, 1.5, 2, 3, 4, 6]`; bit 3 supplies the sign.
Each scale is `2^exponent`, encoded as UE8M0 byte `127 + exponent`.
Diagnostic overrides such as `K1_TMP_*` are not part of the shared contract.

## CPU verification

The verifier reads exactly `M * N` little-endian BF16 values, regenerates
data and scales independently, and rounds its FP32 sum to BF16 before
comparison. Defaults: fill 2, 2048 sampled outputs, sample seed 20260827.

An observed value passes only if finite and:

```text
abs(observed - reference) <= max(16.0, 0.008 * abs(reference))
```

These are existing tolerances, not newly justified accuracy bounds.
Exit codes are 0 for pass, 1 for mismatch, and 2 for invalid input.

The manual driver's fill-1/fill-3 closed-form checks cover its entire output;
fill-2/fill-4 checks are sampled. A constant-input full check does not replace
full verification on varied inputs. Explicit full/sampled Python modes,
chunked computation, and support for the manual grid belong to phase 2.

## Performance measurement

Work is `2 * M * N * K` FLOPs using executed dimensions, without credit for
sparsity. `TFLOPS = FLOPs / (latency_seconds * 1e12)`.

Current host timing: 10 warmups, 100 timed launches, one buffer set, CUDA
events, elapsed time divided by 100 (mean, not median). Host preparation and
verification are outside the timed loop. Default fill 1 is constant data.

The timing loops, CUDA events, mean calculation, and result printing live in
[`dense_gemm_nvfp4_benchmark.cuh`](dense_gemm_nvfp4_benchmark.cuh). Implementations
provide a `launch(std::size_t iteration)` callback returning `cudaError_t` on the
supplied stream. Allocation, input/scale preparation, and correctness stay in the
driver. The header has no dependency on `code/tests/`.

```cpp
double mean_ms = dense_gemm_nvfp4_benchmark::measure(stream, launch, 10, 100);
double tflops = dense_gemm_nvfp4_benchmark::report(
    M, N, K, mean_ms, static_schedule, block_m, block_n, num_clusters, total_tiles);
```

The header accepts independent warmup/timed counts; the current driver retains
10/100. This extraction preserves the single-buffer measurement, not a cold-cache
benchmark. Remaining buffer-rotation and steady-state timing work must follow
[benchmarking_methodology.md](../../../../../knowledge/benchmarking_methodology.md)
under the existing power limit and DVFS, without clock locking. Constant
fill 1 is not the recommended uniform-input fixed-power benchmark; uniform
E2M1 codes in fill 2 are also not uniformly distributed real values.
Define the quantized performance-input distribution before adopting it;
preserve the historical fills and label their results accurately.

## Build and single-case interface

From `books/`:

```bash
make -C code NATIVE_ARCH=sm_100a kernel_sm100a_dense_gemm_nvfp4_build
python3 code/kernels/gemm/sm100a/nvfp4/dense_gemm_nvfp4_cpu_verifier.py cases
python3 code/kernels/gemm/sm100a/nvfp4/dense_gemm_nvfp4_cpu_verifier.py self-test
```

The binary supports the common KB CLI:

```text
--shape=M,K,N --fill=MODE --no-benchmark --no-gpu-verify --dump-output=PATH
```

`--shape` selects exactly one run of the existing AlongN CLC, NTC=128 dispatch;
it does not sweep variants or select the fastest configuration. Supported M is
a positive multiple of 256, N a positive multiple of 128, and K is one of
256, 512, 1024, 2048, 4096, 8192, 16384, or 30720. These include smoke and flagship.

`--fill=1..4` overrides `FILL` for the selected case. `--no-benchmark` performs
one untimed launch. `--no-gpu-verify` disables the built-in reference check (which
is CPU-based in this kernel); KB runs the independent Python verifier instead.
`--dump-output` writes all M*N BF16 values, row-major, outside the timed region.
Invalid options, failed dumps, and failed single-case reference checks exit nonzero.

```bash
code/build/kernel_sm100a_dense_gemm_nvfp4 \
  --shape=256,256,128 --fill=2 --no-benchmark --no-gpu-verify \
  --dump-output=/tmp/dense-gemm-nvfp4-smoke.bf16
python3 code/kernels/gemm/sm100a/nvfp4/dense_gemm_nvfp4_cpu_verifier.py verify \
  --case smoke --fill 2 --actual /tmp/dense-gemm-nvfp4-smoke.bf16
code/build/kernel_sm100a_dense_gemm_nvfp4 \
  --shape=30720,8192,4096 --fill=1 --no-gpu-verify
```

Timed single-case output is `shape (M, N, K): mean=X ms ... NVFP4_TFLOPS`.
This is a batch mean, not a median; KB computes medians across repeated processes.
Use `kb harness benchmark-ab --profile gemm-sm100a-nvfp4-v1` with the baseline
and candidate materializations, selected case, and GPU UUID. No separate Python
benchmarker is needed. `kb harness evaluate` also runs correctness verification.

With no arguments, the legacy sweep, probe selection, `FILL` handling, and output
format are unchanged. Single-case options require `--shape`; `--help` describes
the interface. A Python verifier self-test does not execute the GPU kernel.

This aligns correctness and benchmarking only. Profiling a multi-instantiation
manual binary still needs the appropriate function selection and WARP_PROF export;
the existing generated-program profiling contract is not changed here.

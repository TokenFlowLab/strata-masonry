# Dense GEMM NVFP4 (SM100a)

Implementation: [dense_gemm_nvfp4.cu](dense_gemm_nvfp4.cu).
CPU reference: [dense_gemm_nvfp4_cpu_verifier.py](dense_gemm_nvfp4_cpu_verifier.py).

## Computation

E2M1 data with UE8M0 scales, one scale per row per 32 K elements:

```text
kb = floor(k / 32)
D[m, n] = BF16_RNE(sum_k (A[m, k] * SFA[m, kb]) * (B[n, k] * SFB[n, kb]))
```

| Tensor | Logical shape | Type and layout |
|---|---|---|
| A | M x K | Row-major E2M1, two values per byte |
| B | N x K | Row-major E2M1, two values per byte |
| SFA | M x (K / 32) | UE8M0 |
| SFB | N x (K / 32) | UE8M0 |
| D | M x N | Row-major BF16 |

Even K indices occupy the low nibble, odd indices the high nibble. Scale packing for
TMA/TMEM is implementation-specific; logical scale values must agree. FP32 accumulation,
round-to-nearest-even output. No bias, activation or accumulation into D.

## Cases

Verifier cases:

| Case | M | K | N |
|---|---:|---:|---:|
| smoke | 256 | 256 | 128 |
| flagship | 30720 | 8192 | 4096 |

Performance grid of the driver (`flagship` equals `2K+2N`):

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

UND rows run M=14046 padded to 14080; they do not exercise unpadded tails.

## Inputs

The driver uses `FILL` (default 1) unless `--fill` overrides it; the verifier's `--fill`
defaults to 2, so set both explicitly. All hash arithmetic wraps to 32 bits:

```text
hash(x): x *= 2654435761; x ^= x >> 16; x *= 2246822519; x ^= x >> 13
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
| 4 | Signed integers (below) | e_A / e_B |

Fill 4 uses magnitude code `[0, 2, 4, 5][x & 3]` and sign bit `(x >> 5) & 1`. E2M1 codes 0..7
decode to `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`; bit 3 is the sign. Each scale is `2^exponent`,
stored as UE8M0 byte `127 + exponent`.

## CPU verification

The verifier reads `M * N` little-endian BF16 values, regenerates data and scales, and rounds
its FP32 sum to BF16. Defaults: fill 2, 2048 sampled outputs, sample seed 20260827. An element
passes if finite and `abs(observed - reference) <= max(16.0, 0.008 * abs(reference))`.
Exit codes: 0 pass, 1 mismatch, 2 invalid input. The driver's built-in fill-1/3 checks cover
the whole output; its fill-2/4 checks are sampled.

## Benchmark

Work is `2 * M * N * K` FLOPs; `TFLOPS = FLOPs / (seconds * 1e12)`.
[dense_gemm_nvfp4_benchmark.cuh](dense_gemm_nvfp4_benchmark.cuh) runs 10 warmups and 100 timed
launches on one buffer set and reports the batch mean (not a median). Host preparation and
verification are outside the timed loop; results are L2-warm. Fill 1 is constant data, so
use fill 2 or 4 for representative timing, and record the fill.

## Commands

From the repository root:

```bash
cmake -S . -B build && cmake --build build -j --target kernel_sm100a_dense_gemm_nvfp4
python3 src/kernels/gemm/sm100a/nvfp4/dense_gemm_nvfp4_cpu_verifier.py self-test
build/bin/kernel_sm100a_dense_gemm_nvfp4 \
  --shape=256,256,128 --fill=2 --no-benchmark --no-gpu-verify --dump-output=nvfp4-smoke.bf16
python3 src/kernels/gemm/sm100a/nvfp4/dense_gemm_nvfp4_cpu_verifier.py verify \
  --case smoke --fill 2 --actual nvfp4-smoke.bf16
build/bin/kernel_sm100a_dense_gemm_nvfp4 --shape=30720,8192,4096 --fill=2 --no-gpu-verify
build/bin/kernel_sm100a_dense_gemm_nvfp4      # full performance sweep
```

`--shape=M,K,N` runs one case with the default AlongN CLC, NTC=128 dispatch. M must be a multiple
of 256, N a multiple of 128, and K one of 256, 512, 1024, 2048, 4096, 8192, 16384, 30720.
`--no-benchmark` performs one untimed launch; `--no-gpu-verify` skips the built-in check;
`--dump-output` writes all M*N BF16 outputs row-major. Timed single-case output is
`shape (M, N, K): mean=X ms ... NVFP4_TFLOPS`.

# Block-sparse BF16 FMHA backward (SM100a)

| Implementation | Block size |
|---|---|
| [block_sparse_bwd_bf16_blk64.cu](block_sparse_bwd_bf16_blk64.cu) | 64, one pass |
| [block_sparse_bwd_bf16_blk64_2pv.cu](block_sparse_bwd_bf16_blk64_2pv.cu) | 64, two passes |
| [block_sparse_bwd_bf16_blk128.cu](block_sparse_bwd_bf16_blk128.cu) | 128 |
| [block_sparse_bwd_bf16_blk256.cu](block_sparse_bwd_bf16_blk256.cu) | 256 |

CPU reference: [block_sparse_bwd_bf16_cpu_verifier.py](block_sparse_bwd_bf16_cpu_verifier.py).
Inputs: [../block_sparse_bf16_gen_inputs.py](../block_sparse_bf16_gen_inputs.py).

## Computation

- `B = 1`, `Hq = Hkv = 8`, `D = 128`, non-causal MHA, `Sq = Skv = S`, scale `1 / sqrt(D)`.
- BF16 Q/K/V, dO, saved forward O and output gradients; FP32 saved LSE (log2, defined below).
- Each query block selects `topk = S / (4 * block_size)` distinct, full KV blocks, shared
  across heads. Outputs are dQ, dK, dV, each BF16 `[S, 8, 128]`.

## Cases

24 cases (forward grid without blk512). Entries are `topk`; `num_blocks = S / block_size`.

| S | blk64 | blk128 | blk256 |
|---:|---:|---:|---:|
| 4096 | 16 | 8 | 4 |
| 8192 | 32 | 16 | 8 |
| 16384 | 64 | 32 | 16 |
| 32768 | 128 | 64 | 32 |
| 65536 | 256 | 128 | 64 |
| 131072 | 512 | 256 | 128 |
| 262144 | 1024 | 512 | 256 |
| 524288 | 2048 | 1024 | 512 |

Small correctness cases: S=1024 blk64 and S=2048 blk128 (25% density); each driver's no-argument
run also covers built-in smoke shapes, including zero-count KV blocks.

## Inputs

`block_sparse_bf16_gen_inputs.py --outdir inputs` writes, per S (and per block size where noted):

```text
inputs/{q,k,v,do}_S{S}.npy           uint16 BF16 bits [S, 8, 128]
inputs/idx_S{S}_blk{block_size}.npy  int32 [num_blocks, topk], sorted
inputs/o_S{S}_blk{block_size}.npy    uint16 BF16 bits [S, 8, 128], real forward output
inputs/lse_S{S}_blk{block_size}.npy  float32 [8, S]
```

For head h and query token i, with j over the tokens of the selected KV blocks only:

```text
z_j = dot(Q[i,h], K[j,h]) / sqrt(D) * log2(e)
LSE[h,i] = max_j(z_j) + log2(sum_j exp2(z_j - max_j(z_j)))
```

Under `LOAD_NPY=<dir>` every driver uses these O/LSE files: `CPU_REF=0` times with them, and
`CPU_REF=1` checks them against the CPU forward before verifying the GPU against a reference built
from them. `CPU_REF=0` without `LOAD_NPY` is rejected. External comparisons must load the same
O/LSE (converting the log base or layout outside timing).

## CPU verification

`verify` reads the same inputs and the GPU dumps (`DUMP_BWD_GPU=<prefix>` writes
`<prefix>_{dq,dk,dv}.npy`, FP32-expanded) and checks every gradient element by default:
FP32 accumulation, Delta from the saved BF16 O, BF16-rounded P for dV and BF16-rounded dS for
dQ/dK. Default tolerance `abs(actual - expected) <= 0.002 + 0.02 * abs(expected)`; dQ/dK/dV are
reported separately; NaN/Inf fails. `--mode sampled --samples 64` checks 64 `(token, head)` rows
and is reported as sampled. Exit codes: 0 pass, 1 mismatch, 2 invalid input.

## Benchmark

[block_sparse_bwd_bf16_benchmark.cuh](block_sparse_bwd_bf16_benchmark.cuh) times each complete
backward invocation (preprocess, main including both passes for 2pv, postprocess) with CUDA events
and reports the upper median. `BENCH_WARMUP` (default 10) and `BENCH_ITERS` (default 50) set the
counts; `NO_BENCHMARK=1` skips timing. Buffers are reused (L2-warm).

```text
selected_pairs = B * H * num_blocks * topk * block_size^2
TFLOPS = 10 * D * selected_pairs / (time_ms * 1e9)
```

## Commands

From the repository root (blk128, S=4096):

```bash
cmake -S . -B build && cmake --build build -j --target kernel_sm100a_block_sparse_bwd_bf16_blk128
python3 src/kernels/fmha_sparse/sm100a/block_sparse_bf16_gen_inputs.py --outdir inputs --seqlens 4096
LOAD_NPY=inputs SHAPE=0 BATCH=1 HEADS=8 NB=32 TOPK=8 CPU_REF=0 NO_BENCHMARK=1 DUMP_BWD_GPU=bwd \
  build/bin/kernel_sm100a_block_sparse_bwd_bf16_blk128
python3 src/kernels/fmha_sparse/sm100a/backward/block_sparse_bwd_bf16_cpu_verifier.py verify \
  --inputs inputs --seqlen 4096 --block-size 128 --actual-prefix bwd
LOAD_NPY=inputs SHAPE=0 BATCH=1 HEADS=8 NB=32 TOPK=8 CPU_REF=0 BENCH_WARMUP=10 BENCH_ITERS=50 \
  build/bin/kernel_sm100a_block_sparse_bwd_bf16_blk128
```

For other cases set `NB = S / block_size`, `TOPK = NB / 4` and pick the matching binary
(`BLOCK=64` for the 2pv driver).

# Block-sparse BF16 FMHA forward (SM100a)

| Implementation | Block size |
|---|---|
| [block_sparse_bf16_uniform.cu](block_sparse_bf16_uniform.cu) | 64 (128 with `-DVSA_BLK128=true`) |
| [block_sparse_bf16_uniform_blk256.cu](block_sparse_bf16_uniform_blk256.cu) | 256 |
| [block_sparse_bf16_uniform_2sm_blk512.cu](block_sparse_bf16_uniform_2sm_blk512.cu) | 512, 2SM |
| [block_sparse_bf16_varlen.cu](block_sparse_bf16_varlen.cu) | 64 (128 with `-DVSA_BLK128=true`), variable block sizes |

CPU reference: [block_sparse_bf16_cpu_verifier.py](block_sparse_bf16_cpu_verifier.py).
Input generator: [block_sparse_bf16_gen_inputs.py](block_sparse_bf16_gen_inputs.py).

## Computation

- `B = 1`, `Hq = Hkv = 8`, `D = 128`, BF16, non-causal MHA, `Sq = Skv = S`.
- Query and KV blocks have the same size. Each query block attends all tokens of exactly
  `topk = S / (4 * block_size)` distinct KV blocks (25% density), with scale `1 / sqrt(D)`.
- Selections are shared across heads.

## Cases

32 cases: eight sequence lengths times four block sizes. Entries are `topk`;
`num_blocks = S / block_size`.

| S | blk64 | blk128 | blk256 | blk512 |
|---:|---:|---:|---:|---:|
| 4096 | 16 | 8 | 4 | 2 |
| 8192 | 32 | 16 | 8 | 4 |
| 16384 | 64 | 32 | 16 | 8 |
| 32768 | 128 | 64 | 32 | 16 |
| 65536 | 256 | 128 | 64 | 32 |
| 131072 | 512 | 256 | 128 | 64 |
| 262144 | 1024 | 512 | 256 | 128 |
| 524288 | 2048 | 1024 | 512 | 256 |

Varlen is run here with full blocks only (`VBS_MIN=block_size`).

## Inputs

Two modes; never compare performance across modes or seeds:

- Seeded (default): `INPUT_SEED=<n>` in the driver, `--seed <n>` in the verifier. Both use the same
  uint32 hash, uniform values in [-1,1), BF16 round-to-nearest-even and sorted Fisher-Yates
  block selection. No files needed.
- Saved files: `LOAD_NPY=<dir>` in the driver, `--inputs <dir>` in the verifier. Generate them with
  `block_sparse_bf16_gen_inputs.py` (Gaussian Q/K/V):

```text
<dir>/{q,k,v}_S{S}.npy               uint16 BF16 bits [S, 8, 128]
<dir>/idx_S{S}_blk{block_size}.npy   int32 [num_blocks, topk], sorted, shared across heads
```

Q/K/V are shared across block sizes; index files are per block size.

## CPU verification

The verifier recomputes the output with chunked FP32 QK, stable online softmax and PV, and checks
every output element by default (`--mode sampled --samples 64` checks 64 rows and is reported as
sampled). `--actual` is the driver's `DUMP_O` file: raw FP32 `[S, H, D]`. An element passes if finite
and `abs(error) <= 0.002 + 0.02 * abs(reference)`. `reference` writes the full CPU result;
`self-test` checks the verifier. Nonzero exit on failure.

## Benchmark

[block_sparse_bf16_benchmark.cuh](block_sparse_bf16_benchmark.cuh): `WARMUP` (default 3),
`ITERS` (default 20), `GRAPH=0|1`. Timing is the CUDA-event batch mean over reused (L2-warm)
buffers; allocation, input setup, capture, output copies and verification are excluded.
TFLOPS count the selected pairs: `4 * D * H * num_blocks * topk * block_size^2 / seconds / 1e12`.
`NO_BENCHMARK=1` launches once untimed; `NOVERIFY=1` skips the in-driver CPU check.

## Commands

From the repository root (blk64, S=4096):

```bash
cmake -S . -B build && cmake --build build -j --target kernel_sm100a_block_sparse_bf16_uniform
INPUT_SEED=0 SHAPE=2 BATCH=1 HEADS=8 NB=64 TOPK=16 NOVERIFY=1 NO_BENCHMARK=1 DUMP_O=sparse-out \
  build/bin/kernel_sm100a_block_sparse_bf16_uniform
python3 src/kernels/fmha_sparse/sm100a/block_sparse_bf16_cpu_verifier.py verify \
  --seed 0 --seqlen 4096 --block-size 64 --actual sparse-out.out
INPUT_SEED=0 SHAPE=2 BATCH=1 HEADS=8 NB=64 TOPK=16 NOVERIFY=1 WARMUP=20 ITERS=100 \
  build/bin/kernel_sm100a_block_sparse_bf16_uniform
```

For other cases set `NB = S / block_size`, `TOPK = NB / 4` and pick the matching binary. For the
blk128 builds of `uniform` and `varlen`, configure a separate build directory with
`-DCMAKE_CUDA_FLAGS=-DVSA_BLK128=true`. For saved inputs run
`python3 src/kernels/fmha_sparse/sm100a/block_sparse_bf16_gen_inputs.py --outdir inputs` and add
`LOAD_NPY=inputs` / `--inputs inputs`.

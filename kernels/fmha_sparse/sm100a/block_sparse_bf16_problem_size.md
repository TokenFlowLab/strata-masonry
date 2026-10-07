# Block-sparse BF16 FMHA problem sizes (SM100a)

Shared workload grid for forward block-sparse attention. Only the cases below
are in scope; no additional smoke, batch, ragged, or variable-length cases.

## Fixed settings

- `B = 1`, `Hq = Hkv = 8`, `D = 128`.
- BF16 Q/K/V, non-causal MHA.
- Q and KV sequence lengths are equal: `Sq = Skv = S`.
- Query and KV blocks have the same size: `block_size = 64, 128, 256, or 512`.
- Sparse density is 25%: each query block selects exactly
  `topk = S / (4 * block_size)` distinct KV blocks.
- Attention uses all tokens in those selected blocks, with scale `1 / sqrt(D)`.

## Problem sizes

There are 32 cases: eight sequence lengths times four block sizes.
The table entries are the selected KV-block counts per query block (`topk`).
The total number of blocks is `num_blocks = S / block_size`.

| S | blk64 topk | blk128 topk | blk256 topk | blk512 topk |
|---:|---:|---:|---:|---:|
| 4096 | 16 | 8 | 4 | 2 |
| 8192 | 32 | 16 | 8 | 4 |
| 16384 | 64 | 32 | 16 | 8 |
| 32768 | 128 | 64 | 32 | 16 |
| 65536 | 256 | 128 | 64 | 32 |
| 131072 | 512 | 256 | 128 | 64 |
| 262144 | 1024 | 512 | 256 | 128 |
| 524288 | 2048 | 1024 | 512 | 256 |

This specifies the workload, not verification or performance results for every case.

## Shared inputs

For each S, every implementation and block size uses the same Q/K/V tensors.
Selection indices are separate for each block size; implementations of the same
case must use identical indices as well as identical Q/K/V values.

Two input modes are supported for both benchmarking and CPU verification:

- Seeded generation (default): `INPUT_SEED=0` in the kernel, `--seed 0` in the CPU verifier.
  Both reproduce the same uint32 hash, uniform values in [-1,1), BF16 round-to-nearest-even,
  and sorted Fisher-Yates block selection. No input files are needed.
- Saved inputs: `LOAD_NPY=<directory>` in the kernel, `--inputs <directory>` in the verifier.
  Reuse the existing Gaussian Q/K/V and block indices from
  [bench/gen_inputs.py](bench/gen_inputs.py) for historical comparisons.

Do not compare performance between different input modes or seeds. Seeded generation
and historical files need not produce the same values. Input setup is outside timing.

Optional saved files are relative to this directory:

```text
inputs/q_S{S}.npy
inputs/k_S{S}.npy
inputs/v_S{S}.npy
inputs/idx_S{S}_blk{block_size}.npy
```

Q/K/V files store `[S, H, D]` arrays as uint16 BF16 bit patterns (`B = 1`).
Index files store int32 `[num_blocks, topk]` arrays, shared across heads.
Selection uses the existing query-block-seeded xorshift32 Fisher-Yates algorithm,
with distinct selected block IDs sorted in ascending order.

Preserve existing input files. In file mode, complete any missing inputs before running a case;
do not silently substitute a different shape, density, or selection pattern.

The original six-size grid and historical results are recorded in
[bench/README.md](bench/README.md) and [fmha_vsa_log.md](fmha_vsa_log.md).
This grid additionally includes S=262144 and S=524288.

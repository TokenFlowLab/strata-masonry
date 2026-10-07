# Block-sparse BF16 FMHA backward problem sizes (SM100a)

Shared workload contract for the existing backward implementations in this directory.
This file specifies cases and required inputs/outputs, not completed verification or
performance results. Historical tables remain in [problem_sizes.md](problem_sizes.md).

Kernel filenames and build targets use `block_sparse_bwd_bf16*`: the combined legacy
driver, `_blk64`, `_blk64_2pv`, `_blk128`, `_blk128_2sm`, and `_blk256` variants.
The matching warp-profile contracts and this family's companion files use the same prefix.

## Fixed settings

- `B = 1`, `Hq = Hkv = 8`, `D = 128`; non-causal MHA.
- Query and KV sequence lengths are equal: `Sq = Skv = S`.
- Q/K/V, incoming output gradient dO, saved forward O, and output gradients are BF16.
- Saved forward log-sum-exp is FP32, in the log2 convention defined below.
- Query and KV blocks have the same size: `block_size = 64, 128, or 256`.
- Sparse density is 25%: each query block selects exactly
  `topk = S / (4 * block_size)` distinct, full KV blocks.
- Attention scale is `1 / sqrt(D)`. Selections are shared across heads.

There is no blk512 backward implementation in this scope. Two-pass and two-CTA
implementations are alternative implementations of a case, not additional problem sizes.

## Problem sizes

The main grid shares the forward suite's eight sequence lengths: 24 cases across three
block sizes. Entries below are selected KV-block counts per query block (`topk`);
the total block count is `num_blocks = S / block_size`.

| S | blk64 topk | blk128 topk | blk256 topk |
|---:|---:|---:|---:|
| 4096 | 16 | 8 | 4 |
| 8192 | 32 | 16 | 8 |
| 16384 | 64 | 32 | 16 |
| 32768 | 128 | 64 | 32 |
| 65536 | 256 | 128 | 64 |
| 131072 | 512 | 256 | 128 |
| 262144 | 1024 | 512 | 256 |
| 524288 | 2048 | 1024 | 512 |

The active performance grid stops at S=524288. Dated larger-size results remain in
the historical tables and logs; they are not additional active cases. Listing a case
does not claim every implementation has passed it.

## Shared backward inputs

Use the existing [../inputs/](../inputs/) directory. For each S, all implementations
and block sizes share Q/K/V/dO. Selections and forward O/LSE are specific to each
block size, because changing the selected tokens changes the attention result.

```text
../inputs/q_S{S}.npy
../inputs/k_S{S}.npy
../inputs/v_S{S}.npy
../inputs/do_S{S}.npy
../inputs/idx_S{S}_blk{block_size}.npy
../inputs/o_S{S}_blk{block_size}.npy
../inputs/lse_S{S}_blk{block_size}.npy
```

- Q/K/V/dO/O: C-order `[S, 8, 128]`, uint16 storing raw BF16 bits.
- Indices: C-order `[num_blocks, topk]`, int32; distinct sorted KV-block IDs per row.
- LSE: C-order `[8, S]`, FP32. For each head h and query token i:

  ```text
  z_j = dot(Q[i,h], K[j,h]) / sqrt(D) * log2(e)
  LSE[h,i] = max_j(z_j) + log2(sum_j(exp2(z_j - max_j(z_j))))
  ```

  Here j ranges only over tokens in that query block's selected KV blocks.
  O is the corresponding attention output rounded to BF16, not a placeholder.

[../bench/gen_inputs.py](../bench/gen_inputs.py) prepares saved inputs and forward state.
Generation and loading are outside backward timing. Preserve existing files; complete
missing data before benchmarking. Never substitute `O=0` or a constant LSE.

For controlled comparisons, ours, Triton, and FA4 must consume this same saved state.
Backend adapters may transpose layouts or convert the LSE logarithm base; record those
conversions. A base conversion need not preserve bits, but must preserve the represented
normalizer. The shared-state loaders in the drivers and Triton/FA4 runners implement
this contract; file preparation remains outside their backward timing loops.

## Outputs and verification contract

Native backward implementations produce dQ, dK, and dV, each BF16 `[S, 8, 128]`.
GPU dumps store their BF16 values expanded to FP32 in `<prefix>_dq.npy`,
`_dk.npy`, and `_dv.npy`. The older union implementation
`block_sparse_bwd_bf16.cu` instead exports FP32 dQ decoded from its
accumulation buffer on the host; its dK/dV dumps are BF16 expanded to FP32.
Delta (`sum(O * dO)` per query row) and dQ accumulation buffers are intermediate state.

The CPU verifier must use the same Q/K/V/dO, selection indices, and saved O/LSE as the
GPU run. Full verification checks all `3 * S * 8 * 128` gradient elements by default;
sampled verification must be explicitly requested and reported as sampled, not full.
Report dQ/dK/dV errors separately, state tolerances, and fail on non-finite outputs.
Disabling CPU verification must not change the inputs used by the timed GPU computation.

Existing small correctness cases remain available separately from the performance grid:
S=1024 blk64 and S=2048 blk128 (B=1, H=8, D=128, 25% density), plus the driver's
built-in smoke/topk edge cases documented in [problem_sizes.md](problem_sizes.md).
The blk256 no-argument smoke case uses S=1024, B=1, H=8, D=128, and topk=1;
it generates its inputs and computes real forward state with the built-in CPU reference.
They do not replace full verification of a reported performance case.

## CPU verifier

Run from this directory; NumPy is the only dependency and no GPU is used:

```bash
python3 block_sparse_bwd_bf16_cpu_verifier.py self-test
python3 block_sparse_bwd_bf16_cpu_verifier.py cases
OPENBLAS_NUM_THREADS=4 python3 block_sparse_bwd_bf16_cpu_verifier.py verify \
  --inputs ../inputs --seqlen 4096 --block-size 128 \
  --actual-prefix /tmp/bwd
```

The last command reads `/tmp/bwd_{dq,dk,dv}.npy`, checks every gradient element,
and prints a JSON report with separate dQ/dK/dV errors and checked-element counts.
Use actual GPU dumps from a run consuming these same saved inputs. A driver run that
recomputes O/LSE instead of loading them is not an exact match to this verifier's inputs.

For example, build and produce those GPU dumps without the in-driver CPU calculation
or timed benchmark (commands below run from `books/code`):

```bash
make kernel_sm100a_block_sparse_bwd_bf16_blk128_build
CUDA_VISIBLE_DEVICES=0 LOAD_NPY=kernels/fmha/sm100a/sparse/inputs \
  SHAPE=0 BATCH=1 HEADS=8 NB=32 TOPK=8 CPU_REF=0 NO_BENCHMARK=1 \
  DUMP_BWD_GPU=/tmp/bwd \
  ./build/kernel_sm100a_block_sparse_bwd_bf16_blk128
```

All six root backward drivers support `DUMP_BWD_GPU` independently of `CPU_REF`.
Acquire exclusive access to the selected GPU before running a case.

The CPU calculation uses FP32 accumulation, Delta from saved BF16 O, BF16-rounded P
for dV, and BF16-rounded dS for dQ/dK, matching the documented backward arithmetic.
It verifies backward given O/LSE; validating forward-state generation is a separate check.
Default elementwise tolerance is `abs(actual - expected) <= 0.002 + 0.02 * abs(expected)`;
`--atol` and `--rtol` are explicit overrides recorded in the report.

For an explicitly sampled check, add `--mode sampled --samples 64 --sample-seed 0`.
This selects 64 `(token, head)` rows of each gradient and checks every D value in each
row. Each sampled dK/dV row includes contributions from all selecting query blocks.
Full remains the default. Full CPU checking at large S is expensive: chunking bounds
memory, not the quadratic amount of attention computation.

`--query-chunk` and `--key-chunk` bound temporary score tiles; inputs and GPU dumps
are memory-mapped and gradients are processed one head at a time. `--result <path>`
optionally saves the report without overwriting a file. Exit codes are 0 for a pass,
1 for gradient mismatches, and 2 for invalid/missing inputs or a computation error.

## Benchmark contract

- Time the complete backward path: preprocess, main computation (both passes for 2pv),
  and postprocess. Exclude forward-state generation, input loading, allocation,
  CPU verification, and dumps. Report index-inversion timing separately if included.
- The legacy union driver still times memset + preprocess + main only; its host dQ
  decoding is outside timing. Do not present its latency as the same end-to-end scope
  as the native drivers. The shared timing header preserves this existing distinction.
- Preserve the existing CUDA-event per-invocation median convention, with independent
  `BENCH_WARMUP` and `BENCH_ITERS` controls (current driver defaults: 10 and 50).
- All six drivers include `block_sparse_bwd_bf16_benchmark.cuh`. It uses the existing
  upper median for even iteration counts, validates counts, and releases timing events.
  `NO_BENCHMARK=1` skips the warmup/timing loops but retains the initial execution and
  requested correctness/dumps. Existing result lines and the 2pv CLI remain supported.
- Existing drivers reuse buffers: label results L2-warm, not cold-buffer measurements.
- Use one GPU workload at a time per GPU. Do not require administrator clock locking;
  record clock/power conditions and compare variants under matched conditions.
- For the existing selected-work convention, with time in milliseconds:

  ```text
  selected_pairs = B * H * num_blocks * topk * block_size^2
  TFLOPS = 10 * D * selected_pairs / (time_ms * 1e9)
  ```

Equal density gives equal selected-pair counts across block sizes, but different masks
are different attention workloads. Compare implementations of the same block size and
selection pattern when claiming a speedup for the same computation.

Follow the shared
[benchmarking methodology](../../../../../../knowledge/benchmarking_methodology.md).
Historical results predate the shared-forward-state fix; they remain historical records,
not evidence that all inputs matched the current contract.

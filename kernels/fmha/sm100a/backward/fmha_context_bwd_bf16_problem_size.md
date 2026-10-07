# Dense BF16 FMHA backward: shared cases, verification, and timing

Scope: `fmha_context_bwd_bf16.cu` and the existing experimental
`fmha_context_bwd_bf16_d128_1cta_tile64.cu`. They share this host contract; the experimental
variant is not promoted to the canonical implementation. `backward/sparse/` and
`fa4_bwd_extract/` are outside this refactor.

Kernel filenames, Makefile targets, and companion files use `fmha_context_bwd_bf16*`.

## Existing performance grid

Reuse the recipe's 24 logical cases. Both head dimensions and both masks apply
to every row; H*D=2048, BF16 MHA, uniform lengths, and scale `1/sqrt(D)`.

| B | S | H,D choices | Masks |
|---:|---:|---|---|
| 32 | 512 | 32,64 or 16,128 | Full and causal |
| 16 | 1024 | 32,64 or 16,128 | Full and causal |
| 8 | 2048 | 32,64 or 16,128 | Full and causal |
| 4 | 4096 | 32,64 or 16,128 | Full and causal |
| 2 | 8192 | 32,64 or 16,128 | Full and causal |
| 1 | 16384 | 32,64 or 16,128 | Full and causal |

D64 supports 1CTA. D128 supports 1CTA and 2CTA, giving 36 raw measurements per
implementation. Choose the variant explicitly; no new auto-dispatch is introduced.
Causal means key `j <= query i`, including the diagonal.

Retain the built-in `--verify` cases: D64 S=31/129/512/1024; D128 S=33/130/256/512;
both masks, batch/head checks, preprocess/postprocess, and 2CTA protocol checks.
Those cases are correctness checks, not the performance grid. The Python verifier
also accepts explicit positive B/H/S with D64 or D128, including non-tile tails.

## Shared input and output contract

- Q/K/V/dO/O: uint16 BF16 bits in C-order NumPy `[B,H,S,D]`.
- LSE: FP32 **natural-log** NumPy `[B,H,S]`. Unlike sparse backward, this is not log2.
- GPU dQ/dK/dV dumps: raw BF16 `[B,H,S,D]`, named `<prefix>_{dq,dk,dv}.bf16`.
- Input names: `<q|k|v|do|o|lse>_B<B>_H<H>_S<S>_D<D>_c<0|1>.npy`.

`fmha_context_bwd_bf16_cpu_verifier.py generate` creates Q/K/V/dO using a fixed NumPy
seed, computes real forward O/LSE on CPU in chunks, and writes one new input
directory. It refuses to overwrite an existing directory. Generate once and let
all implementations read the same files. A random seed alone does not ensure
different libraries generate identical values. External comparisons must also
load this O/LSE, with any required layout/log-base conversion outside timing.

The old benchmark used identical random Q/K/V/O/dO and constant LSE. That was
not valid forward state. `--bench` now requires `LOAD_NPY`; it fails rather than
benchmarking placeholders. Historical performance tables are retained but must
be remeasured on shared real state before making new cross-party comparisons.
The built-in `--verify` still uses its existing C++ generator and CPU reference.

## Full CPU verification

`fmha_context_bwd_bf16_cpu_verifier.py verify` reads the exact saved state and checks
**every dQ, dK, and dV element**. There is no implicit 256-element smoke sample.
It uses CPU FP32 products/sums, BF16 P/dS operands, `delta=sum(dO*O)`, and applies
the attention scale once to dQ/dK. Saved natural-log LSE determines P; the verifier
does not replace saved state with a different forward run. NaN/Inf fails.
Default tolerances retain the recipe's absolute 0.05 / relative 0.02 gate.

Forward generation and backward verification have quadratic compute cost, but
score storage is bounded by query/key chunks. Large full-grid CPU checks can be
slow. The generated O/LSE are inputs; the expected gradients are computed by the
verifier, never read from the GPU output.

From `books/code`, a small full check:

```bash
make kernel_sm100a_fmha_context_bwd_bf16_build
python3 kernels/fmha/sm100a/backward/fmha_context_bwd_bf16_cpu_verifier.py generate \
  --batch 1 --seqlen 512 --head-dim 128 --output /tmp/dense-bwd-inputs
CUDA_VISIBLE_DEVICES=0 LOAD_NPY=/tmp/dense-bwd-inputs NO_BENCHMARK=1 \
  DUMP_BWD_GPU=/tmp/dense-bwd ./build/kernel_sm100a_fmha_context_bwd_bf16 --bench 1 512 128 0
python3 kernels/fmha/sm100a/backward/fmha_context_bwd_bf16_cpu_verifier.py verify \
  --batch 1 --seqlen 512 --head-dim 128 --inputs /tmp/dense-bwd-inputs \
  --actual-prefix /tmp/dense-bwd
```

For causal mode add `--causal` to both Python commands and use mask `1` in the
binary command. For D128 2CTA use `--bench-2cta B S CAUSAL [ITERS]`. `cases` lists
the 24 logical performance cases; `self-test` checks the CPU math without a GPU.

## Shared benchmark

Both root drivers include `fmha_context_bwd_bf16_benchmark.cuh`. The `.cu` retains launch
geometry and kernel selection; the header owns input loading, timing, and dumps.
Defaults remain 5 warmups and 20 timed iterations. Optional CLI ITERS is retained;
`BENCH_WARMUP` and `BENCH_ITERS` override the counts. `NO_BENCHMARK=1` executes one
complete backward operation for verification without a timing result.

The CUDA-event interval contains preprocess (including dQ accumulator clear), main,
and dQ postprocess. Input loading, CPU work, copies, and output dumps are excluded.
Reused buffers are L2-warm; the result is a batch mean, not a per-launch median.
`BENCH ... ms=... tflops=...` stays compatible with existing parsers. FLOP accounting
retains `10*B*H*D*S*S`, halved for causal, and the existing 2500-TFLOPS reference.
The causal FLOP convention approximates the exact diagonal-inclusive pair count.

Use exclusive GPUs, alternate A/B order, and repeat small differences. Record input
files, source revision, GPU, CTA group, mask, and counts. Do not assume clock-lock
privileges. This refactor does not add profiling or change device algorithms.

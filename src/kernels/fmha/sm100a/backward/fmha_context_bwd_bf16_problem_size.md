# Dense BF16 FMHA backward (SM100a)

Implementation: [fmha_context_bwd_bf16.cu](fmha_context_bwd_bf16.cu).
Input generator and CPU reference: [fmha_context_bwd_bf16_cpu_verifier.py](fmha_context_bwd_bf16_cpu_verifier.py).

## Cases

24 logical cases: every row below with both head-dim choices and both masks. `H*D = 2048`,
BF16 MHA, uniform lengths, scale `1/sqrt(D)`.

| B | S | H,D choices | Masks |
|---:|---:|---|---|
| 32 | 512 | 32,64 or 16,128 | full and causal |
| 16 | 1024 | 32,64 or 16,128 | full and causal |
| 8 | 2048 | 32,64 or 16,128 | full and causal |
| 4 | 4096 | 32,64 or 16,128 | full and causal |
| 2 | 8192 | 32,64 or 16,128 | full and causal |
| 1 | 16384 | 32,64 or 16,128 | full and causal |

D64 runs 1CTA; D128 runs 1CTA (`--bench`) and 2CTA (`--bench-2cta`), so 36 measurements in total.
Causal keeps key `j <= query i`, including the diagonal.

The built-in `--verify` covers D64 S=31/129/512/1024 and D128 S=33/130/256/512 with both masks,
batch/head checks, preprocess/postprocess and the 2CTA protocol. It uses its own generator and
CPU reference.

## Inputs and outputs

- Q/K/V/dO/O: uint16 BF16 bits, C-order `[B,H,S,D]`.
- LSE: FP32 natural-log `[B,H,S]`.
- File names: `<q|k|v|do|o|lse>_B<B>_H<H>_S<S>_D<D>_c<0|1>.npy`.
- GPU gradient dumps: raw BF16 `[B,H,S,D]` in `<prefix>_{dq,dk,dv}.bf16`.

`generate` creates Q/K/V/dO from a fixed NumPy seed, computes the real forward O/LSE on CPU in
chunks, and writes a new directory (it never overwrites one). Generate once and let every
implementation read the same files. `--bench` requires `LOAD_NPY` with these files.

## CPU verification

`verify` reads the saved state and checks every dQ, dK and dV element: FP32 products and sums,
BF16 P/dS operands, `delta = sum(dO * O)`, attention scale applied once to dQ/dK, P from the saved
LSE. NaN/Inf fails. Default tolerances: absolute 0.05, relative 0.02. `cases` lists the 24 logical
cases; `self-test` checks the CPU math without a GPU. Large cases are slow (quadratic compute;
memory is bounded by chunking).

## Benchmark

[fmha_context_bwd_bf16_benchmark.cuh](fmha_context_bwd_bf16_benchmark.cuh) owns input loading,
timing and dumps. Defaults: 5 warmups and 20 timed iterations; `BENCH_WARMUP` / `BENCH_ITERS`
or the trailing CLI `ITERS` override them. The timed interval covers preprocess (including the dQ
accumulator clear), main and dQ postprocess; loading, CPU work, copies and dumps are excluded.
The result is a CUDA-event batch mean over reused (L2-warm) buffers, printed as
`BENCH ... ms=... tflops=...`. FLOPs are `10*B*H*D*S*S`, halved for causal. `NO_BENCHMARK=1`
runs one untimed backward for verification.

## Commands

From the repository root:

```bash
cmake -S . -B build && cmake --build build -j --target kernel_sm100a_fmha_context_bwd_bf16
python3 src/kernels/fmha/sm100a/backward/fmha_context_bwd_bf16_cpu_verifier.py generate \
  --batch 1 --seqlen 512 --head-dim 128 --output bwd-inputs
LOAD_NPY=bwd-inputs NO_BENCHMARK=1 DUMP_BWD_GPU=bwd-out \
  build/bin/kernel_sm100a_fmha_context_bwd_bf16 --bench 1 512 128 0
python3 src/kernels/fmha/sm100a/backward/fmha_context_bwd_bf16_cpu_verifier.py verify \
  --batch 1 --seqlen 512 --head-dim 128 --inputs bwd-inputs --actual-prefix bwd-out
LOAD_NPY=bwd-inputs build/bin/kernel_sm100a_fmha_context_bwd_bf16 --bench-2cta 1 512 0
```

For causal, add `--causal` to both Python commands and pass mask `1` to the binary.

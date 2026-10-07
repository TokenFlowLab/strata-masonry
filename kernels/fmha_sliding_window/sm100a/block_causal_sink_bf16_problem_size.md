# Block-causal BF16 FMHA: shared cases, correctness, and timing

This is the current runnable contract for `block_causal_sink_bf16{,_2sm,_fv}.cu`
and `block_causal_sink_bf16_tf{,_2sm}.cu`. The existing `problem_sizes.md` retains
historical measurements. Files under `fv/` and `reference/` are not refactored here.

Kernel filenames, Makefile targets, and companion files use `block_causal_sink_bf16*`.
The benchmark runners resolve the new names; existing `bcs`/`tf` mode names are unchanged.

## Performance grid

Reuse `bench/gen_inputs.py` and the existing cross-party grid:

| Field | Value |
|---|---|
| Batch, heads, head dimension | B=1, HQ=HK=12, D=128 |
| Input/output | BF16 Q/K/V/O; softmax scale `1/sqrt(128)` |
| Tokens per frame | 1456 |
| Frames per block | 3 (4368 tokens) |
| Sink | 1 frame (1456 tokens) |
| Local attention size | 6 frames including the sink; rolling window is 5 frames |
| Frames | 6, 12, 18, 36, 72 |
| Blockwise sequence lengths | 8736, 17472, 26208, 52416, 104832 |
| Teacher-forcing lengths | 17472, 34944, 52416, 104832, 209664 |

Run blockwise and teacher-forcing, each with and without `ROPE_DELTA=1`. Teacher
forcing uses `[clean | noisy]` halves; `NUM_FRAMES` describes one half. Only the
`_tf` programs implement that mode. `_fv` implements blockwise plus optional log2 LSE.

For a query block, blockwise attention sees the union of the sink and rolling window
before that block's end. This is block-causal, not token-triangular. Noisy queries see
previous clean blocks within the same sink/window mask plus their own noisy block.
RoPE rotates Q for sink columns only, using the existing interleaved pair convention.

## Correctness cases

Retain all four cases from `bench/verify_modes.py`, in all four modes:

| HQ=HK | Tokens/frame | Frames/block | Frames | Sink frames | Local frames |
|---:|---:|---:|---:|---:|---:|
| 2 | 48 | 3 | 9 | 2 | 5 |
| 2 | 128 | 3 | 6 | 1 | 4 |
| 4 | 64 | 3 | 12 | 2 | 5 |
| 2 | 48 | 3 | 15 | 4 | 8 |

Also exercise B=2, GQA HQ=8/HK=4 (`MHA=0`), no sink, no window
(`LOCAL_ATTN_SIZE=-1`), and FILL=1..4. Full blocks are required by these drivers.
Performance remains the B=1/MHA grid above; small tests do not replace it.

## Inputs and full CPU verification

Shared files are `q_L<L>.npy`, `k_L<L>.npy`, and `v_L<L>.npy`, with uint16 BF16
bits in C-order `[B*L,H,D]`. Keep blockwise and teacher-forcing input directories
separate: their overlapping L values otherwise collide with different plans/seeds.
The generator's `plan_L<L>.json` records its parameters; pass matching parameters
to the kernel and verifier. Do not regenerate files between implementations.

Without `LOAD_NPY`, both the driver and CPU verifier reproduce FILL=1..4 from the
same unsigned 32-bit index hash and tensor seeds 11/22/33. Default FILL=2.

`block_causal_sink_bf16_cpu_verifier.py` independently computes FP32 QK, softmax,
and PV on CPU, including masks, GQA, and BF16-rounded sink Q rotation. Default
`--mode full` checks every O element, in bounded-memory chunks. `--mode sampled`
is explicitly a smoke check of token/head rows. NaN/Inf outputs fail. Optional
`--lse` verifies the FV dump's log2 LSE, not natural-log LSE.

From `books/code`, for the first retained correctness case:

```bash
make kernel_sm100a_block_causal_sink_bf16_build
CUDA_VISIBLE_DEVICES=0 BCS=1 BATCH=1 HEADS=2 MHA=1 \
  TOKENS_PER_FRAME=48 NUM_FRAME_PER_BLOCK=3 NUM_FRAMES=9 \
  SINK_SIZE=2 LOCAL_ATTN_SIZE=5 FILL=2 NO_BENCHMARK=1 NOVERIFY=1 \
  DUMP_O=/tmp/bcs ./build/kernel_sm100a_block_causal_sink_bf16
python3 kernels/fmha/sm100a/sliding_window/block_causal_sink_bf16_cpu_verifier.py verify \
  --actual /tmp/bcs.out --frames 9 --tokens-per-frame 48 --heads 2 --sink 2 --window 5
```

Add `TF=1` and select a `_tf` binary, with `--teacher-forcing` in the verifier.
Add `ROPE_DELTA=1` and `--rope` together. Use `LOAD_NPY=<dir>` and `--inputs <dir>`
together for saved inputs. `DUMP_O` writes raw FP32-expanded BF16 `[B*L,HQ,128]`.
FV's `LSE=1 DUMP_LSE=<file>` writes raw FP32 `[B,HQ,L]`.

## Shared benchmark

All five drivers include `block_causal_sink_bf16_benchmark.cuh`. Defaults remain
3 warmups and 20 timed launches; `WARMUP` and `ITERS` override them. CUDA events
report a batch mean on reused buffers (L2-warm), with no input generation, copies,
CPU reference, or dump inside the timed interval. `NO_BENCHMARK=1` launches once
for verification without reporting performance. The existing result line is preserved.
Useful TFLOPS count QK+PV over visible pairs; sink/window overlap is counted once.

Use one exclusive GPU per measurement, alternate A/B order, and repeat small gains.
No clock locking or administrator privileges are assumed. Record GPU, source revision,
input set, mode, and timing options. Do not compare warm means to cold-buffer medians.

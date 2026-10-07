# Block-causal sink + sliding-window BF16 FMHA (SM100a)

| Implementation | Mode |
|---|---|
| [block_causal_sink_bf16.cu](block_causal_sink_bf16.cu) | blockwise |
| [block_causal_sink_bf16_tf.cu](block_causal_sink_bf16_tf.cu) | teacher forcing |

CPU reference: [block_causal_sink_bf16_cpu_verifier.py](block_causal_sink_bf16_cpu_verifier.py).
Inputs: [block_causal_sink_bf16_gen_inputs.py](block_causal_sink_bf16_gen_inputs.py).

## Computation

A query block attends the union of the sink and the rolling window that end at that block's end:
block-causal, not token-triangular. Teacher forcing uses `[clean | noisy]` halves (`NUM_FRAMES`
describes one half); noisy queries see previous clean blocks under the same sink/window mask plus
their own noisy block. With `ROPE_DELTA=1`, Q is rotated for sink columns only (interleaved pairs).
Softmax scale `1/sqrt(128)`; BF16 Q/K/V/O. Frame counts must be whole blocks.

## Cases

Performance grid, run blockwise and teacher forcing, each with and without `ROPE_DELTA=1`:

| Field | Value |
|---|---|
| Batch, heads, head dim | B=1, HQ=HK=12, D=128 |
| Tokens per frame | 1456 |
| Frames per block | 3 (4368 tokens) |
| Sink | 1 frame |
| Local attention size | 6 frames including the sink (rolling window of 5) |
| Frames | 6, 12, 18, 36, 72 |
| Blockwise L | 8736, 17472, 26208, 52416, 104832 |
| Teacher-forcing L | 17472, 34944, 52416, 104832, 209664 |

Correctness cases (both drivers, with and without RoPE):

| HQ=HK | Tokens/frame | Frames/block | Frames | Sink frames | Local frames |
|---:|---:|---:|---:|---:|---:|
| 2 | 48 | 3 | 9 | 2 | 5 |
| 2 | 128 | 3 | 6 | 1 | 4 |
| 4 | 64 | 3 | 12 | 2 | 5 |
| 2 | 48 | 3 | 15 | 4 | 8 |

Also check B=2, GQA HQ=8/HK=4 (`MHA=0`), no sink, no window (`LOCAL_ATTN_SIZE=-1`) and FILL=1..4.

## Inputs

- Seeded: without `LOAD_NPY`, the driver and verifier reproduce `FILL=1..4` (default 2) from the
  same uint32 index hash and tensor seeds 11/22/33.
- Saved files: `block_causal_sink_bf16_gen_inputs.py` writes `q_L<L>.npy`, `k_L<L>.npy`,
  `v_L<L>.npy` (uint16 BF16 bits, `[B*L, H, D]`) and `plan_L<L>.json` with the plan parameters.
  Keep blockwise and teacher-forcing inputs in separate directories (their L values overlap),
  and pass the plan parameters to the driver and verifier. Use `LOAD_NPY=<dir>` with `--inputs <dir>`.

## CPU verification

The verifier computes FP32 QK, softmax and PV with the masks, GQA and BF16-rounded sink-Q
rotation. `--mode full` (default) checks every output element in bounded memory;
`--mode sampled` checks token/head rows and is reported as sampled. NaN/Inf fails.
`--actual` is the driver's `DUMP_O` file: FP32-expanded BF16 `[B*L, HQ, 128]`. Use
`--teacher-forcing` for the `_tf` drivers (`TF=1`) and `--rope` together with `ROPE_DELTA=1`.

## Benchmark

[block_causal_sink_bf16_benchmark.cuh](block_causal_sink_bf16_benchmark.cuh): `WARMUP` (default 3)
and `ITERS` (default 20). CUDA events report the batch mean over reused (L2-warm) buffers;
input generation, copies, CPU reference and dumps are outside the timed interval.
`NO_BENCHMARK=1` launches once without timing; `NOVERIFY=1` skips the in-driver CPU check.
TFLOPS count QK + PV over visible pairs, with sink/window overlap counted once.

## Commands

From the repository root:

```bash
cmake -S . -B build && cmake --build build -j --target kernel_sm100a_block_causal_sink_bf16
BCS=1 BATCH=1 HEADS=2 MHA=1 TOKENS_PER_FRAME=48 NUM_FRAME_PER_BLOCK=3 NUM_FRAMES=9 \
  SINK_SIZE=2 LOCAL_ATTN_SIZE=5 FILL=2 NO_BENCHMARK=1 NOVERIFY=1 DUMP_O=bcs \
  build/bin/kernel_sm100a_block_causal_sink_bf16
python3 src/kernels/fmha_sliding_window/sm100a/block_causal_sink_bf16_cpu_verifier.py verify \
  --actual bcs.out --frames 9 --tokens-per-frame 48 --heads 2 --sink 2 --window 5
python3 src/kernels/fmha_sliding_window/sm100a/block_causal_sink_bf16_gen_inputs.py --outdir inputs
BCS=1 BATCH=1 HEADS=12 TOKENS_PER_FRAME=1456 NUM_FRAME_PER_BLOCK=3 NUM_FRAMES=18 \
  SINK_SIZE=1 LOCAL_ATTN_SIZE=6 LOAD_NPY=inputs NOVERIFY=1 \
  build/bin/kernel_sm100a_block_causal_sink_bf16
```

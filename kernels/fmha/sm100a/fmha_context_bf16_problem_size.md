# Dense BF16 context FMHA forward (SM100a)

| Alias | Implementation |
|---|---|
| UI | [fmha_context_bf16_uniform_inline.cu](fmha_context_bf16_uniform_inline.cu) |
| U2I | [fmha_context_bf16_uniform_2sm_inline.cu](fmha_context_bf16_uniform_2sm_inline.cu) |
| V | [fmha_context_bf16_varlen.cu](fmha_context_bf16_varlen.cu) |

CPU reference: [fmha_context_bf16_cpu_verifier.py](fmha_context_bf16_cpu_verifier.py) (NumPy, no GPU).

## Computation

`B` samples; `Sq[i]`, `Sk[i]` are sample `i`'s Q and KV lengths. Attention never crosses samples.

```text
Q:   [sum(Sq), HQ, D] BF16
K,V: [sum(Sk), HK, D] BF16
O:   [sum(Sq), HQ, D] BF16
D = 128; scale = 1 / sqrt(D)
HQ % HK = 0; query head h uses KV head h / (HQ / HK)
O = softmax(Q @ K^T * scale) @ V per sample and query head
```

Full attention uses all keys. Causal is top-left aligned: key `j <= i` for query `i`, also when
`Sq != Sk`. No dropout or attention bias. UI and U2I require `Sq = Sk` and one length for every
sample; V takes per-sample lengths and supports `Sq != Sk`.

## Cases

| Case | B | Sq = Sk | HQ | HK | Mask | Drivers |
|---|---:|---|---:|---:|---|---|
| smoke | 2 | 4 | 8 | 4 | full | UI, U2I |
| und-full | 128 | UND list | 32 | 4 | full | V |
| und-causal | 128 | UND list | 32 | 4 | causal | V |
| gen-full | 128 | 240 | 32 | 4 | full | UI, U2I |
| gen-causal | 128 | 240 | 32 | 4 | causal | UI, U2I |

The UND list has 128 lengths from 29 to 291 (14,046 tokens). Print every case and its lengths with
`python3 kernels/fmha/sm100a/fmha_context_bf16_cpu_verifier.py cases`.

Performance grid (all `D=128`):

| Workload | `--case` | B | Sq | Sk | HQ | HK | Mask | Drivers |
|---|---|---:|---|---|---:|---:|---|---|
| gen GQA short | `gen-full` | 128 | 240 | 240 | 32 | 4 | full | UI, U2I |
| gen GQA mid | `gqa-mid-full` | 8 | 4608 | 4608 | 32 | 4 | full | UI, U2I |
| gen GQA long | `gqa-long-full` | 1 | 75600 | 75600 | 32 | 4 | full | UI, U2I |
| mha mid | `mha-mid-full` | 8 | 4608 | 4608 | 32 | 32 | full | UI, U2I, V |
| mha long | `mha-long-full` | 1 | 75600 | 75600 | 32 | 32 | full | UI, U2I, V |
| mha causal | `mha-mid-causal` | 8 | 4608 | 4608 | 32 | 32 | causal | UI, U2I, V |
| und | `und-causal` | 128 | UND list | UND list | 32 | 4 | causal | V |
| cross varlen | `cross-varlen-{mha,gqa}-{full,causal}` | 4 | [100,240,64,175] | [300,240,512,90] | 32 | 32 or 4 | both | V |

Correctness matrix (UI and U2I, both masks, `Sq=Sk=S`; verifier name = label + `-full`/`-causal`):

| Label | B | S | HQ | HK | Fills |
|---|---:|---:|---:|---:|---|
| gqa-short (`gen-*`) | 128 | 240 | 32 | 4 | 1-4 |
| mha-1024 | 2 | 1024 | 16 | 16 | 1-4 |
| mha-968rag | 2 | 968 | 16 | 16 | 1-4 |
| mha-256min | 1 | 256 | 2 | 2 | 1-4 |
| mha-384odd | 1 | 384 | 2 | 2 | 1-4 |
| gqa-512 | 1 | 512 | 8 | 4 | 1-4 |
| gqa-600rag | 1 | 600 | 8 | 4 | 1-4 |
| mha-832 | 2 | 832 | 8 | 8 | 1-4 |
| mha-4608big | 1 | 4608 | 2 | 2 | 2, 4 |
| gqa-4608big | 1 | 4608 | 8 | 4 | 2, 4 |
| mha-4416rag | 1 | 4416 | 2 | 2 | 2, 4 |

UI and U2I always use HK=4 in GQA mode; the verifier's `-hk2` cases are not reachable from them.

Known issues: V's `und-causal` run hangs before reporting timing and V times out on
`mha-mid-full`.

## Driver interfaces

- UI, U2I: `BATCH`, `SEQLEN`, `HEADS`, `MHA`, `CAUSAL` (default full). `MHA=1` sets `HK=HQ`,
  otherwise `HK=4`.
- V: defaults to UND GQA causal. `MHA=1` selects uniform MHA from `BATCH`/`SEQLEN`/`HEADS`;
  `CROSS=1` selects the cross-varlen lists, with `MHA` choosing HK=32 or 4; `CAUSAL=0` for full.
- `NOVERIFY=1` skips the in-driver CPU check in UI, U2I and in V's uniform MHA branch. UI and U2I
  run it by default only for `SEQLEN <= 1024`; `VERIFY=1` forces it.
- UI, U2I: `DUMP_O=PATH` writes `PATH.out` (kernel output) and `PATH.ref` (in-driver reference),
  both raw FP32 in `[sum(Sq), HQ, D]`, whenever the in-driver check runs.

## Inputs and CPU verification

Q, K, V use seeds 11, 22, 33. For flat logical element index `i`:

```text
x = (i * 2654435761 + seed) mod 2^32
fill 1: (x mod 256) / 256 - 0.5
fill 2: (x mod 2048) / 1024 - 1      (default)
fill 3: 1
fill 4: (x mod 7) - 3
```

Inputs are rounded to BF16 before computation. Drivers select fills with `FILL`.
[fmha_cpu_ref.cuh](fmha_cpu_ref.cuh) is the in-driver FP32 reference.

The Python verifier regenerates inputs from the fill and seeds:

- `verify --mode full` checks every output element in bounded-memory blocks;
  `--mode sampled --samples 64` (default) is a smoke check and is reported as such.
- `--actual` takes `[sum(Sq), HQ, D]` little-endian BF16 (`2 * sum(Sq) * HQ * D` bytes) or FP32
  (`4 * sum(Sq) * HQ * D` bytes, the `DUMP_O` format); the size selects the format.
- `reference --case NAME --output PATH` writes the full CPU reference; `self-test --fill N`
  checks the verifier against a dense FP64 calculation.
- An element passes if finite and `abs(actual - reference) <= max(0.05, 0.10 * abs(reference))`.
  Exit codes: 0 pass, 1 mismatch, 2 invalid invocation.

## Benchmark

```text
Full:   pairs[i] = Sq[i] * Sk[i]
Causal: m = min(Sq[i], Sk[i]); pairs[i] = m * (m + 1) / 2 + max(Sq[i] - Sk[i], 0) * Sk[i]
FLOPs = 4 * D * HQ * sum(pairs[i]);  TFLOP/s = FLOPs / (seconds * 1e12)
```

All three drivers use [fmha_context_bf16_benchmark.cuh](fmha_context_bf16_benchmark.cuh):

| Environment | Meaning | Default |
|---|---|---|
| `GRAPH` | 0: direct launches; 1: capture one launch and replay it | 0 |
| `WARMUP` | Untimed launches (>= 0) | 3 direct, 5 graph |
| `ITERS` | Timed launches (> 0) | 20 |

Timing is the CUDA-event batch mean over reused buffers (L2-warm), excluding allocation, graph
capture and verification. Clocks are not locked: use matched inputs, an exclusive GPU, settled
warmup and interleaved A/B runs.

## Commands

From the repository root:

```bash
cmake -S . -B build && cmake --build build -j --target kernel_sm100a_fmha_context_bf16_uniform_inline
FILL=2 DUMP_O=/tmp/o BATCH=2 SEQLEN=4 HEADS=8 MHA=0 CAUSAL=0 \
  build/bin/kernel_sm100a_fmha_context_bf16_uniform_inline
python3 kernels/fmha/sm100a/fmha_context_bf16_cpu_verifier.py verify \
  --case smoke --fill 2 --mode full --actual /tmp/o.out
FILL=2 BATCH=128 SEQLEN=240 HEADS=32 MHA=0 CAUSAL=0 NOVERIFY=1 GRAPH=1 WARMUP=100 ITERS=1000 \
  build/bin/kernel_sm100a_fmha_context_bf16_uniform_inline
```

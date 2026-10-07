# Dense BF16 context FMHA problem sizes (SM100a)

Shared workload inventory for these six forward implementations:

| Alias | Implementation |
|---|---|
| NP | [fmha_context_bf16_gqa_nonpersistent.cu][np] |
| U | [fmha_context_bf16_uniform.cu][u] |
| U2 | [fmha_context_bf16_uniform_2sm.cu][u2] |
| UI | [fmha_context_bf16_uniform_inline.cu][ui] |
| U2I | [fmha_context_bf16_uniform_2sm_inline.cu][u2i] |
| V | [fmha_context_bf16_varlen.cu][v] |

[np]: fmha_context_bf16_gqa_nonpersistent.cu
[u]: fmha_context_bf16_uniform.cu
[u2]: fmha_context_bf16_uniform_2sm.cu
[ui]: fmha_context_bf16_uniform_inline.cu
[u2i]: fmha_context_bf16_uniform_2sm_inline.cu
[v]: fmha_context_bf16_varlen.cu

This document preserves the existing cases from `problem_sizes.md`,
[the scaling study](docs/problem_size_scaling.md), and [verify_matrix.sh](verify_matrix.sh).
It records current host-driver support, not new correctness or performance results.
Sparse VSA, sliding-window, backward, and `LL/` kernels are outside this scope.

## Inputs and output

`B` is the sample count; `Sq[i]` and `Sk[i]` are sample `i`'s Q and KV lengths.
Uniform cases repeat one length for every sample. Varlen cases use explicit length lists.
Q and KV have separate cumulative-length arrays; attention never crosses sample boundaries.

```text
Q:   [sum(Sq), HQ, D] BF16
K,V: [sum(Sk), HK, D] BF16
O:   [sum(Sq), HQ, D] BF16
D = 128; scale = 1 / sqrt(D)
HQ % HK = 0; query head h uses KV head h / (HQ / HK), with integer division
MHA: HQ = HK; GQA: HQ > HK
```

These are packed, row-major logical layouts. A kernel's internal transposed V buffer
does not change the input contract. The output is `softmax(Q @ K^T * scale) @ V` per
sample and query head. Full attention uses all keys. Top-left causal attention keeps
key position `j <= i` for query position `i`, including when `Sq != Sk`.
No bottom-right causal alignment, dropout, or attention bias is implied.

An experiment must name the complete case, mask, and fill. Do not silently change them
or pad the logical problem to remove a ragged tail.

## Existing verifier cases

These five original names remain accepted by the
[shared CPU verifier](fmha_context_bf16_cpu_verifier.py).
UND and GEN come from Cosmos3 rank 0, iteration 5000.

| Case | B | Sq = Sk per sample | HQ | HK | Mask | Current drivers |
|---|---:|---|---:|---:|---|---|
| smoke | 2 | 4 | 8 | 4 | full | U, U2, UI, U2I |
| und-full | 128 | UND list | 32 | 4 | full | V |
| und-causal | 128 | UND list | 32 | 4 | causal | NP, V |
| gen-full | 128 | 240 | 32 | 4 | full | U, U2, UI, U2I |
| gen-causal | 128 | 240 | 32 | 4 | causal | U, U2, UI, U2I |

The UND list has minimum 29, maximum 291, total 14,046 tokens, and
`sum(Sq[i]^2) = 1,945,032`. GEN has 30,720 tokens and `sum(Sq[i]^2) = 7,372,800`.
Outstanding GPU issue (2026-10-06): V's `und-causal` run hangs before reporting timing.
It reproduced in pre-refactor binaries, including one built before the short-sequence
correctness fixes. NP's `und-causal` passed in direct and graph modes. Keep the workload;
do not count V as verified on it until the hang is fixed.
V's pre-refactor baseline also timed out on `mha-mid-full`; its smaller uniform MHA
and cross-varlen regression cases passed. See the [log](fmha_context_bf16_log.md).

Print the exact ordered lists from the `books/` root:

```bash
python3 code/kernels/fmha/sm100a/fmha_context_bf16_cpu_verifier.py cases
```

## Existing performance cases

The scaling-study labels and sizes below are unchanged. Their verifier names are
listed after the table. All use `D=128`.

| Existing label | B | Sq | Sk | HQ | HK | Mask | Current drivers |
|---|---:|---|---|---:|---:|---|---|
| gen GQA short | 128 | 240 | 240 | 32 | 4 | full | U, U2, UI, U2I |
| gen GQA mid | 8 | 4608 | 4608 | 32 | 4 | full | U, U2, UI, U2I |
| gen GQA long | 1 | 75600 | 75600 | 32 | 4 | full | U, U2, UI, U2I |
| mha mid | 8 | 4608 | 4608 | 32 | 32 | full | U, U2, UI, U2I, V |
| mha long | 1 | 75600 | 75600 | 32 | 32 | full | U, U2, UI, U2I, V |
| mha causal | 8 | 4608 | 4608 | 32 | 32 | causal | U, U2, UI, U2I, V |
| und (varlen) | 128 | UND list | UND list | 32 | 4 | causal | NP, V |
| cross MHA Sq<Sk | 8 | 2048 | 4608 | 32 | 32 | causal | U, U2 |
| cross MHA Sq>Sk | 8 | 4608 | 2048 | 32 | 32 | full | U, U2 |
| cross varlen | 4 | Q list below | KV list below | 32 | 32 | causal | V |

`gen GQA short` is `gen-full`; `und (varlen)` is `und-causal`, not a new workload.
The exact cross-varlen lengths from V's host driver are:

```text
Sq = [100, 240,  64, 175]
Sk = [300, 240, 512,  90]
```

V also exposes these cross-varlen lengths with `HK=4` (GQA), and either mask.
The full and causal versions must remain distinct cases.

### Short cross-attention regression cases

These cases complement the existing production sizes. They exercise the two-tile
causal mask and fully masked rows without a production-sized CPU reference run.
All have `B=2, HQ=8, D=128`; use fills 1-4 and both full and top-left causal masks.
U exposes these via `BATCH`, `SEQLEN`, `SEQLEN_KV`, `HEADS`, `MHA`, and `CAUSAL`.

| Case prefix (append `-full` or `-causal`) | Sq | Sk | HK |
|---|---:|---:|---:|
| cross-smoke-gqa-q-short | 256 | 384 | 4 |
| cross-smoke-mha-q-short | 256 | 384 | 8 |
| cross-smoke-gqa-q-long | 384 | 256 | 4 |
| cross-smoke-mha-q-long | 384 | 256 | 8 |

`smoke-causal` uses the original `smoke` shape with the causal mask. Together with
`smoke`, it also checks the `S=4` V-load alignment case without changing its logical size.

### Performance case names

Verifier case names:

| Workload | `--case` |
|---|---|
| gen GQA short | `gen-full` |
| gen GQA mid / long | `gqa-mid-full` / `gqa-long-full` |
| mha mid / long | `mha-mid-full` / `mha-long-full` |
| mha causal | `mha-mid-causal` |
| und (varlen) | `und-causal` |
| cross MHA Sq<Sk | `cross-mha-q-short-causal` |
| cross MHA Sq>Sk | `cross-mha-q-long-full` |
| cross varlen MHA | `cross-varlen-mha-full` / `cross-varlen-mha-causal` |
| cross varlen GQA | `cross-varlen-gqa-full` / `cross-varlen-gqa-causal` |

## Existing correctness matrix

[verify_matrix.sh](verify_matrix.sh) runs UI and U2I with both full and causal masks.
Preserve all these shapes; they cover short loops, ragged tails, and odd tile counts.
All have `Sq=Sk=S`, `D=128`. The HK column records the current executed shape.

| Existing label | B | S | HQ | HK | Fills |
|---|---:|---:|---:|---:|---|
| gqa-short | 128 | 240 | 32 | 4 | 1, 2, 3, 4 |
| mha-1024 | 2 | 1024 | 16 | 16 | 1, 2, 3, 4 |
| mha-968rag | 2 | 968 | 16 | 16 | 1, 2, 3, 4 |
| mha-256min | 1 | 256 | 2 | 2 | 1, 2, 3, 4 |
| mha-384odd | 1 | 384 | 2 | 2 | 1, 2, 3, 4 |
| gqa-512 | 1 | 512 | 8 | 4 | 1, 2, 3, 4 |
| gqa-600rag | 1 | 600 | 8 | 4 | 1, 2, 3, 4 |
| mha-832 | 2 | 832 | 8 | 8 | 1, 2, 3, 4 |
| mha-4608big | 1 | 4608 | 2 | 2 | 2, 4 |
| gqa-4608big | 1 | 4608 | 8 | 4 | 2, 4 |
| mha-4416rag | 1 | 4416 | 2 | 2 | 2, 4 |

Known mismatch: the script requests `HEADS_KV=2` for `gqa-512`, `gqa-600rag`, and
`gqa-4608big`, but the uniform drivers ignore `HEADS_KV` and use `HK=4` in GQA mode.
Those runs do not establish HK=2 coverage. Both the intended HK=2 and actual HK=4
cases must be retained when the interface is unified; the script is unchanged here.

For verifier names, append `-full` or `-causal` to each matrix label except
`gqa-short`, which uses `gen-full` / `gen-causal`. The three intended HK=2 cases
insert `-hk2` before the mask, for example `gqa-512-hk2-causal`.
`tiny-causal` is also registered. Registering a reference case does not add that
case to a GPU driver's interface or establish that the GPU kernel passes it.

The same script has a repeated-run consistency check, `row46-stress`, with
`B=8, S=4608, HQ=HK=32`, both masks, fill 4, and `STRESS_N=10`.
This is not a replacement for comparison against the CPU reference.

NP additionally runs `tiny-causal`: `B=3, Sq=Sk=[29,101,240], HQ=8, HK=1, D=128`,
fill 2. Its other hardcoded case is `und-causal`; preserve both.

## Current executable interfaces

The six host drivers do not yet expose one common CLI:

- U, U2, UI, U2I select uniform shapes with `BATCH`, `SEQLEN`, `HEADS`, `MHA`, and
  `CAUSAL`. `MHA=1` sets `HK=HQ`; otherwise `HK=4`. Default mask is full.
- Only U and U2 accept `SEQLEN_KV` for uniform cross-attention. UI and U2I are
  self-attention only; setting `SEQLEN_KV` there does not create a cross case.
- V defaults to UND GQA. `MHA=1` selects uniform MHA using `BATCH`, `SEQLEN`, and
  `HEADS`. `CROSS=1` takes precedence and selects the fixed cross-varlen lists above,
  with `MHA` choosing HK=32 or HK=4. V's default mask is causal; use `CAUSAL=0` for full.
- NP runs its two hardcoded causal cases; it has no shape or mask selection interface.

All six drivers now use [fmha_context_bf16_benchmark.cuh](fmha_context_bf16_benchmark.cuh).
`GRAPH=1` enables graph replay in every driver, including NP, UI, and U2I. Before this
extraction, those three ignored `GRAPH`, so old inline sweep results were direct-launch
timings even when [sweep_all_fills.sh](sweep_all_fills.sh) set `GRAPH=1`.

`NOVERIFY=1` skips the CPU reference in the four uniform drivers. V honors it only
in its uniform MHA branch; V's UND/cross branches and NP still run CPU verification.
Do not assume the same environment variable is supported by every implementation.

## Fill and CPU verification

Q, K, and V use seeds 11, 22, and 33 respectively. For flat logical element index `i`:

```text
x = (i * 2654435761 + seed) mod 2^32
fill 1: (x mod 256) / 256 - 0.5
fill 2: (x mod 2048) / 1024 - 1
fill 3: 1
fill 4: (x mod 7) - 3
```

Round inputs to BF16 before reference computation. All drivers except NP select
these modes with `FILL` (default 2); NP always uses fill 2. Compared implementations
must use identical logical inputs, even if their internal layouts differ.

[fmha_cpu_ref.cuh](fmha_cpu_ref.cuh) computes full FP32 reference outputs for
self/cross-attention, MHA/GQA, and both masks. All six drivers include this shared
reference; their existing verification gates and tolerances remain unchanged.

[fmha_context_bf16_cpu_verifier.py](fmha_context_bf16_cpu_verifier.py) provides the
standalone reference for all the registered cases above. It requires NumPy, not a GPU.

- `verify --mode full` checks every output element. Queries and keys are processed
  in blocks, without allocating the full attention matrix. Long production cases can
  still take substantial CPU time: full verification performs the complete attention math.
- `verify --mode sampled --samples 64` is the default, explicitly a smoke check.
  The report records `checked_elements`, `total_elements`, and `full_coverage`.
- `reference --case NAME --output PATH` writes the complete CPU reference as raw BF16.
  This generated output tests the verifier, not a GPU kernel.
- `self-test --fill N` checks the blocked FP32 reference against an independent dense
  FP64 calculation on small MHA/GQA cross cases, including both length directions and
  masks. It also checks full/sample behavior, corruption detection, and NaN/Inf rejection.

The input values are reconstructed from the documented fill, fixed Q/K/V seeds, and
case shape/layout. This is valid only when the tested kernel uses that exact generator.
Arbitrary external Q/K/V inputs are not accepted by this CLI yet.
`--actual` expects exactly `2 * sum(Sq) * HQ * D` bytes: little-endian BF16 in
`[sum(Sq), HQ, D]` row-major order. No header, padding, or transposed output is allowed.

The default tolerances are `atol=0.05`, `rtol=0.10`. An element passes when
`abs(actual-reference) <= max(atol, rtol*abs(reference))`; nonfinite output always fails.
The verifier exits 0 for pass, 1 for mismatch, and 2 for an invalid invocation or artifact.
`--query-block` (default 64) and `--key-block` (default 2048) control CPU working memory,
not which outputs full mode checks.

## Performance accounting

The shared benchmark header implements the logical attended-pair convention:

```text
Full:   pairs[i] = Sq[i] * Sk[i]
Causal: m = min(Sq[i], Sk[i])
        pairs[i] = m * (m + 1) / 2 + max(Sq[i] - Sk[i], 0) * Sk[i]
FLOPs = 4 * D * HQ * sum(pairs[i])
TFLOP/s = FLOPs / (latency_seconds * 10^12)
```

This counts QK and PV multiply-adds, not softmax operations or padded tiles.
For causal self-attention, use `S*(S+1)/2`, not full `S*S`.

### Shared timing interface

| Environment | Meaning | Default |
|---|---|---|
| `GRAPH` | `0`: direct launches; `1`: capture one launch and replay it | `0` |
| `WARMUP` | Untimed launches, independently configurable; must be >= 0 | 3 direct, 5 graph |
| `ITERS` | Timed launches; must be > 0 | 20 |

Malformed values fail instead of silently becoming zero. `WARMUP=0 ITERS=1 GRAPH=0`
runs one measured launch per case. Graph capture/instantiation is outside timing;
the callback must launch on the provided stream. Kernel launch errors are checked.

The header handles warmup, CUDA events, optional graph replay, pair accounting, and
TFLOPS reporting. Each `.cu` retains its tensors, launch geometry, fill, verification,
and optional profiling/stress calls; verification is outside the timed loop.
Neither clocks nor power limits are modified. No tensor allocation/copy occurs between
timed launches. Existing result lines remain compatible with benchmark scripts.

Each run also prints the actual mode, warmup/iteration counts, `buffers=1`,
`cache=L2-warm`, and `statistic=batch-mean`. The latency is total CUDA-event time divided
by timed launches, not a median; either mode can include host submission gaps.
All six drivers reuse buffers. `L2-warm` labels that reuse, not a measured cache hit rate.

Example for the existing GEN case, from `books/` after building U:

```bash
FILL=2 BATCH=128 SEQLEN=240 HEADS=32 MHA=0 CAUSAL=0 NOVERIFY=1 \
  GRAPH=1 WARMUP=100 ITERS=1000 code/build/kernel_sm100a_fmha_context_bf16_uniform
```

These counts are examples, not a guarantee that clocks have settled.

Follow [perf_bench_plan.md](perf_bench_plan.md) and the shared
[benchmarking methodology](../../../../knowledge/benchmarking_methodology.md).
We cannot lock GPU clocks on this machine: use matched inputs, exclusive GPU execution,
settled warmup, clock/power reporting, and interleaved A/B comparisons.
Keep CPU verification outside the timing window and report buffer reuse/rotation.

This shared timing core is not the completed benchmark plan. Automatic settled warmup,
clock/power sampling, buffer rotation, and per-launch A/B interleaving remain outside
this extraction. Do not claim settled clocks or cold-buffer performance from it alone.
KernelBridge's alternating whole-binary A/B runs are also distinct from the plan's
per-launch A/B interleaving.

## Existing smoke command

From the `books/` root, build U for GEN/smoke and V for UND:

```bash
make -C code NATIVE_ARCH=sm_100a kernel_sm100a_fmha_context_bf16_uniform_build
make -C code NATIVE_ARCH=sm_100a kernel_sm100a_fmha_context_bf16_varlen_build

FILL=2 DUMP_IO=1 BATCH=2 SEQLEN=4 HEADS=8 MHA=0 CAUSAL=0 NOVERIFY=1 \
  code/build/kernel_sm100a_fmha_context_bf16_uniform

python3 code/kernels/fmha/sm100a/fmha_context_bf16_cpu_verifier.py self-test
python3 code/kernels/fmha/sm100a/fmha_context_bf16_cpu_verifier.py verify \
  --case smoke --fill 2 --mode full --actual /tmp/io_o.bin \
  --result /tmp/fmha-smoke.result.json
```

U's existing `DUMP_IO=1` hook writes `/tmp/io_o.bin` as raw BF16, as well as Q/K/V
dumps. This is not yet a shared output-dump interface across the six implementations.

# Strata-Masonry

CUDA kernels for NVIDIA Blackwell (sm_100a, sm_103a) and Hopper (sm_90a), built in layers:
PTX primitives, composites, warp-role blocks and end-to-end kernels. Rubin support is planned.

## Kernels

All kernels are BF16 in / BF16 out with FP32 accumulation unless noted; sm_100a unless noted.

| Family | Kernel | Description |
|---|---|---|
| GEMM | [`gemm/sm100a/dense_gemm_bf16.cu`](src/kernels/gemm/sm100a/dense_gemm_bf16.cu) | Dense GEMM, 2-CTA cluster MMA, persistent CLC scheduling |
| | [`gemm/sm100a/nvfp4/dense_gemm_nvfp4.cu`](src/kernels/gemm/sm100a/nvfp4/dense_gemm_nvfp4.cu) | FP4 (E2M1) GEMM with UE8M0 block scales (one per 32 K) |
| | [`gemm/sm103a/dense_gemm_bf16.cu`](src/kernels/gemm/sm103a/dense_gemm_bf16.cu) | Dense GEMM for sm_103a |
| | [`gemm/sm90a/dense_gemm_bf16.cu`](src/kernels/gemm/sm90a/dense_gemm_bf16.cu) | Dense GEMM for sm_90a (WGMMA skeleton) |
| Attention forward | [`fmha/sm100a/fmha_context_bf16_uniform_inline.cu`](src/kernels/fmha/sm100a/fmha_context_bf16_uniform_inline.cu) | Equal sequence lengths, MHA/GQA, full or causal, D=128 |
| | [`fmha/sm100a/fmha_context_bf16_uniform_2sm_inline.cu`](src/kernels/fmha/sm100a/fmha_context_bf16_uniform_2sm_inline.cu) | Same, 2-CTA cluster MMA |
| | [`fmha/sm100a/fmha_context_bf16_varlen.cu`](src/kernels/fmha/sm100a/fmha_context_bf16_varlen.cu) | Variable sequence lengths per sample, including Sq != Sk |
| Attention backward | [`fmha/sm100a/backward/fmha_context_bwd_bf16.cu`](src/kernels/fmha/sm100a/backward/fmha_context_bwd_bf16.cu) | MHA, full or causal, D=64 or 128 |
| Block-sparse attention forward | [`fmha_sparse/sm100a/block_sparse_bf16_uniform.cu`](src/kernels/fmha_sparse/sm100a/block_sparse_bf16_uniform.cu) | Top-k KV blocks per query block, block size 64 (128 with `-DVSA_BLK128=true`) |
| | [`fmha_sparse/sm100a/block_sparse_bf16_uniform_blk256.cu`](src/kernels/fmha_sparse/sm100a/block_sparse_bf16_uniform_blk256.cu) | Block size 256 |
| | [`fmha_sparse/sm100a/block_sparse_bf16_uniform_2sm_blk512.cu`](src/kernels/fmha_sparse/sm100a/block_sparse_bf16_uniform_2sm_blk512.cu) | Block size 512, 2-CTA cluster MMA |
| | [`fmha_sparse/sm100a/block_sparse_bf16_varlen.cu`](src/kernels/fmha_sparse/sm100a/block_sparse_bf16_varlen.cu) | Variable number of valid tokens per KV block |
| Block-sparse attention backward | [`fmha_sparse/sm100a/backward/`](src/kernels/fmha_sparse/sm100a/backward/) | Block size 64 (one-pass and two-pass), 128 and 256 |
| Sliding-window attention | [`fmha_sliding_window/sm100a/block_causal_sink_bf16.cu`](src/kernels/fmha_sliding_window/sm100a/block_causal_sink_bf16.cu) | Block-causal attention with an attention sink and a sliding window |
| | [`fmha_sliding_window/sm100a/block_causal_sink_bf16_tf.cu`](src/kernels/fmha_sliding_window/sm100a/block_causal_sink_bf16_tf.cu) | Teacher-forcing variant (`[clean \| noisy]` halves) |

Each kernel directory has a `*_problem_size.md` that defines the computation, the case grid, the
driver's environment variables, CPU verification and benchmark commands.

## Layout

```text
src/
  primitives/   numbered PTX wrappers (tcgen05, TMA, mbarrier, CLC, ...), one per header
  composites/   reusable device routines built from primitives
  blocks/       warp-role bodies: load, MMA, softmax, correction, epilogue, scheduler
  kernels/<family>/<arch>/   end-to-end kernels: kernel, driver, verification, benchmark
tests/          one test per primitive, composite and block (<N>_<name>_test.cu)
```

## Requirements

- Linux with an NVIDIA driver for the target GPU
- CUDA Toolkit (tested with 13.0)
- CMake 3.24 or newer, or GNU Make
- A GPU of the target architecture to run tests and kernels: sm_100a (B200/GB200),
  sm_103a (B300/GB300) or sm_90a (H100/H200)
- Python 3 with NumPy for the CPU verifiers; PyTorch for the block-sparse and sliding-window
  input generators

## Build

With CMake (binaries go to `build/bin/`):

```bash
cmake -S . -B build -DGPU_ARCH=sm_100a    # or sm_103a, sm_90a
cmake --build build -j
```

`-DBUILD_TESTS=OFF` or `-DBUILD_KERNELS=OFF` skips a group. Kernel targets are named
`kernel_<arch>_<file name>`, for example `kernel_sm100a_dense_gemm_bf16`.

With Make (binaries go to `build/`; set `NATIVE_ARCH=sm_103a` or `sm_90a` for other GPUs):

```bash
make all                                     # build every test
make 0_tcgen05_alloc_test                    # build and run one test
make kernel_sm100a_dense_gemm_bf16_build     # build one kernel
```

## Test

```bash
ctest --test-dir build      # CMake
make tests_sm100a           # Make: build and run all sm_100a tests
```

## Run a kernel

Build, run a small GEMM and check it against the CPU reference:

```bash
cmake --build build -j --target kernel_sm100a_dense_gemm_bf16
build/bin/kernel_sm100a_dense_gemm_bf16 \
  --shape=256,64,128 --fill=1 --no-benchmark --no-gpu-verify --dump-output=gemm-smoke.bf16
python3 src/kernels/gemm/sm100a/dense_gemm_bf16_cpu_verifier.py verify \
  --case smoke --fill 1 --actual gemm-smoke.bf16
```

The other kernels follow the same pattern; see their `*_problem_size.md`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md). Report
security issues privately as described in [SECURITY.md](SECURITY.md).

## License

BSD 3-Clause. See [LICENSE](LICENSE). Third-party code and its licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

# Contributing

Thanks for your interest in Strata-Masonry. Bug reports, fixes, new primitives and new kernels
are welcome.

## Before you start

- For anything larger than a small fix, open an issue first to discuss the design.
- Building and running tests requires a GPU of the target architecture (sm_100a, sm_103a or
  sm_90a). See the [README](README.md) for requirements and build commands.

## Layout and naming

- Primitives (`src/primitives/`), composites (`src/composites/`) and blocks (`src/blocks/`) share
  one number space: each header is `<N>_<name>.cuh` with a unique `N`. Take the next free number.
- Every numbered header has a test `tests/<N>_<name>_test.cu`. The first lines of a test carry
  its architecture guard (`#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)`) and an
  `// ARCH: sm_XXXa` comment within the first 10 lines; the build uses both to select tests.
- A kernel lives in `src/kernels/<family>/<arch>/<name>.cu` and builds to
  `kernel_<arch>_<name>`. A new kernel comes with a `<name>_problem_size.md` that defines the
  computation, cases, driver options and benchmark method, and with a CPU reference (an
  existing `*_cpu_verifier.py` or a new one).

## Correctness and performance

- Run the tests for the layers you touched and the CPU verifier of every affected kernel in
  full mode (`verify --mode full`), on the target GPU.
- Changes that should not alter generated code (comments, renames, refactors) should leave the
  SASS unchanged: compare `cuobjdump -sass` before and after.
- Performance claims need an interleaved A/B (alternate old and new runs) on the same inputs and
  an otherwise idle GPU. Report the shapes, the GPU and the numbers for both sides.

## Code style

- Format new and changed lines with the repository's `.clang-format`
  (`git clang-format` formats only your changes). Do not reformat untouched code.
- Spell identifiers out; avoid unexplained abbreviations.
- Keep comments short (one or two lines) and about the code as it is. Do not add development
  history, dates, benchmark logs or references to private material.

## Pull requests

- One logical change per pull request, with a description of what changed and why.
- Fill in the pull request template: tests and verifiers run, GPU used, and performance numbers
  for performance changes.

## Licensing

By contributing, you agree that your contributions are licensed under the project's
[BSD 3-Clause License](LICENSE). Code adapted from other projects must keep its original
license terms and be listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

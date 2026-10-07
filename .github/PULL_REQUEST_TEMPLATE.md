## What and why
<!-- What this change does and why it is needed. Link the issue if there is one. -->

## Testing
<!-- Tests and CPU verifiers run (with --mode full), and their results. -->
- GPU:
- CUDA Toolkit:

## Performance
<!-- For performance changes: interleaved A/B, same inputs, idle GPU. Shapes and numbers for
     both sides. Otherwise write "no expected change" (and, where it applies, "SASS unchanged"). -->

## Checklist
- [ ] Tests for the touched primitives, composites or blocks pass
- [ ] Affected kernels pass their CPU verifier
- [ ] New or changed lines formatted with `git clang-format`
- [ ] Third-party code, if any, is listed in `THIRD_PARTY_NOTICES.md`

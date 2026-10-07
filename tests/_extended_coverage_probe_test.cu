#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// _extended_coverage_probe_test.cu -- compile-only ptxas-acceptance probe
// for cta_group::2 wrappers whose runtime validation surface is the
// block level (#88 / #100), not a per-wrapper smoke test.
//
// ARCH: sm_100a (Blackwell-only; filtered out of sm_90a builds)
//
// Why this file still exists after Phase 4:
//   The Phase 4 probe-graduation rounds drained every cta_group::1
//   wrapper from this file -- their runtime tests now live in their
//   owning <N>_<name>_test.cu (see COVERAGE_AUDIT.md "Phase 4 round
//   status" table). The cta_group::2 wrappers below were NOT graduated
//   to per-wrapper smoke tests because cta_group::2 dispatch is not
//   testable in isolation: it requires a 2-CTA cluster + paired
//   tcgen05_alloc<2> + cluster.barrier sync + valid swizzle-aligned
//   SMEM matrix descriptors (see COVERAGE_AUDIT.md "Validation surface
//   for cta_group::2 wrappers" section). Those prerequisites only
//   exist at the block level -- runtime validation lives in
//     - tests/88_load_warp_blackwell_test.cu  (cluster + 2SM TMA + alloc/dealloc)
//     - tests/100_pipeline_blackwell_test.cu  (full pipeline + 2SM MMA + commit_multicast)
//   This file's role is now narrowly: keep ptxas-acceptance coverage
//   for the 2SM wrappers' inline-asm strings so syntax errors show up
//   at build time, not at first block-level test run.
//
// The probe kernel never actually runs at non-trivial inputs: it bails
// on `sink == nullptr` which is always true at our launch site (we
// never launch). That keeps the wrappers live for ptxas while requiring
// no real cluster, no descriptors, no scale-factor matrices. Output
// values are nonsense; the test's success is purely "did the binary
// build cleanly".
//
// Issuer: not applicable (probe is build-only).

#include "test_utils.cuh"
#include "../primitives/3_tcgen05_mma_f16.cuh"
#include "../primitives/4_tcgen05_mma_fp8.cuh"
#include "../primitives/5_tcgen05_mma_fp4.cuh"
#include "../primitives/7_tcgen05_mma_i8.cuh"

// =============================================================================
// 2SM probe -- cta_group::2 wrappers (block-level-validated by #88 / #100).
// =============================================================================
__global__ void probe_kernel_2sm(volatile uint32_t* sink) {
  if (sink == nullptr) return;

  const uint32_t tmem_c = 0;
  const uint32_t tmem_a = 0;
  const uint64_t da = 0, db = 0;

  // -- file 3 (f16, 2SM) --
  tcgen05_mma_f16_ts_2sm(tmem_c, tmem_a, db, 0, true,
                         0,0,0,0, 0,0,0,0);
  tcgen05_mma_f16_ss_2sm_scaled<3>(tmem_c, da, db, 0, true,
                                   0,0,0,0, 0,0,0,0);

  // -- file 4 (fp8 + mxf8f6f4, 2SM) --
  tcgen05_mma_fp8_ts_2sm(tmem_c, tmem_a, db, 0, true,
                         0,0,0,0, 0,0,0,0);
  tcgen05_mma_mxf8f6f4_ss_2sm_block32(tmem_c, da, db, 0, 0, 0, true);

  // -- file 5 (mxf4nvf4, 2SM) --
  tcgen05_mma_mxf4nvf4_ss_2sm_block32(tmem_c, da, db, 0, 0, 0, true);

  // -- file 7 (i8 TS, sm_100a-only, 2SM) --
#if defined(PL_AGENTIC_SM100A)
  tcgen05_mma_i8_ts_2sm(tmem_c, tmem_a, db, 0, true,
                        0,0,0,0, 0,0,0,0);
#endif

  *sink = 0;
}

int main() {
  // No launch -- we only care that ptxas accepted the inline asm in the
  // 2SM probe kernel. To make this a runnable PASS, just print and return 0.
  printf("extended coverage probe: compile-only, 2SM ptxas-acceptance\n");
  PASS();
  return 0;
}
#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

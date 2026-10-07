// 78_rcp_approx.cuh -- rcp.approx.f32 / rcp.approx.ftz.f32 (fast HW reciprocal).
//
// ARCH: sm_50+  (MUFU.RCP)
//
// Raw inline-PTX wrappers around `rcp.approx.f32` -- one MUFU.RCP SASS issue,
// ~23-bit accurate (NOT the ~0.5 ULP refined `1.0f/x` / __frcp_rn, which adds a
// Newton step). Used where the reciprocal feeds an already-approximate path
// (e.g. FMHA epilogue O *= 1/l into bf16), so the refinement isn't worth it.
//
// Corner cases (per PTX ISA 9.7.3.18):
//   rcp.approx of +/-0   -> +/-Inf
//   rcp.approx of +/-Inf -> +/-0
//   the .ftz form flushes denormal inputs/results to sign-preserving zero.
//
// PTX:    rcp.approx.f32      [PTX 1.4+, sm_50+]
//         rcp.approx.ftz.f32  [PTX 1.4+, sm_50+]

#pragma once

#include <cstdint>

// d = 1/x, single MUFU.RCP (~23-bit). Faster + less accurate than 1.0f/x.
__device__ __forceinline__ float rcp_approx_f32(float x) {
  float d;
  asm("rcp.approx.f32 %0, %1;\n" : "=f"(d) : "f"(x));
  return d;
}

// .ftz variant: flush-to-zero on denormal input/result.
__device__ __forceinline__ float rcp_approx_ftz_f32(float x) {
  float d;
  asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(d) : "f"(x));
  return d;
}

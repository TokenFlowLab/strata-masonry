#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 86_acc_correction.cuh -- FMHA accumulator rescale when softmax max changes
//
// ARCH: sm_100a
//
// Blackwell FMHA correction warp: after a new softmax max is found, the
// PV accumulator in TMEM must be scaled by correction = exp2(old_m - new_m).
// We do this by loading the accumulator, multiplying, and storing back.
// PTX:    9.7.18.8 (TMEM ld/st), 9.7.18.10.10.1 (mma scale-input-d for rescale)
//
#include <cstdint>
#include "../primitives/9_tcgen05_ld.cuh"
#include "../primitives/10_tcgen05_st.cuh"
#include "../primitives/12_tcgen05_wait.cuh"

// Rescale one 8-reg chunk of a TMEM-resident FP32 accumulator by `correction`.
__device__ __forceinline__ void acc_correction_fp32_x8(
    uint32_t tmem_addr, float correction) {
  uint32_t r[8];
  tcgen05_ld_32x32b_x8(tmem_addr, r);
  tcgen05_wait_ld();
  float* f = reinterpret_cast<float*>(r);
  #pragma unroll
  for (int i = 0; i < 8; ++i) f[i] *= correction;
  tcgen05_st_32x32b_x8(tmem_addr, r);
  tcgen05_wait_st();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

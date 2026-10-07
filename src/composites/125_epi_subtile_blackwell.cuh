#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 125_epi_subtile_blackwell.cuh -- tcgen05.ld + wait + convert + stmatrix
//
// ARCH: sm_100a
//
// One sub-tile of the Blackwell epilogue: read FP32 regs from TMEM via
// tcgen05.ld.16x256b.x1 (4 regs/lane), convert pairs to packed FP16/BF16,
// stmatrix.x1 each packed reg into SMEM.
// PTX:    9.7.18.8.3 (tcgen05.ld), 9.7.10.24 (cvt), 9.7.16.5.16 (stmatrix)
//
#include <cstdint>
#include "../primitives/9_tcgen05_ld.cuh"
#include "../primitives/12_tcgen05_wait.cuh"
#include "../primitives/40_stmatrix.cuh"
#include "../primitives/63_cvt_f32_to_f16_bf16.cuh"

__device__ __forceinline__ void epi_subtile_blackwell_fp16(
    uint32_t tmem_addr, uint32_t smem_out_base) {
  uint32_t r[4];
  tcgen05_ld_16x256b_x1(tmem_addr, r);
  tcgen05_wait_ld();
  float* f = reinterpret_cast<float*>(r);
  uint32_t p0 = cvt_pack_f32_to_f16x2(f[0], f[1]);
  // stmatrix.x1 needs 16-byte-aligned row address.
  int row = (threadIdx.x & 31) / 4;
  uint32_t row_addr = smem_out_base + row * 16;
  stmatrix_x1(row_addr, p0);
}

__device__ __forceinline__ void epi_subtile_blackwell_bf16(
    uint32_t tmem_addr, uint32_t smem_out_base) {
  uint32_t r[4];
  tcgen05_ld_16x256b_x1(tmem_addr, r);
  tcgen05_wait_ld();
  float* f = reinterpret_cast<float*>(r);
  uint32_t p0 = cvt_pack_f32_to_bf16x2(f[0], f[1]);
  // stmatrix.x1 needs 16-byte-aligned row address.
  int row = (threadIdx.x & 31) / 4;
  uint32_t row_addr = smem_out_base + row * 16;
  stmatrix_x1(row_addr, p0);
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

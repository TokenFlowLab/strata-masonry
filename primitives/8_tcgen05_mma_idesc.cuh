#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 8_tcgen05_mma_idesc.cuh -- tcgen05.mma instruction-descriptor builders.
//
// ARCH: sm_100a
//
// Packs the 32-bit `idesc` register consumed by every tcgen05.mma flavor.
// The bit layout depends on the .kind; three ISA tables apply:
//   Table 53 -- .kind::{f16, tf32, f8f6f4, i8}    (M >> 4 in bits 24-28)
//   Table 54 -- .kind::mxf8f6f4                   (M >> 7 in bits 27-28,
//                                                  scale type bit 23,
//                                                  scale-factor data IDs)
//   Table 55 -- .kind::{mxf4, mxf4nvf4}           (M >> 7 in bits 27-28,
//                                                  K=64/96 selector bit 31)
//
// Builders provided here:
//   make_idesc_table44(...)            generic Table 53 packer
//   make_idesc_f16_f32 / _f16_f16      .kind::f16 specializations
//   make_idesc_bf16_f32                .kind::f16 (BF16 inputs)
//   make_idesc_tf32_f32                .kind::tf32
//   make_idesc_e4m3_f32 / _e5m2_f32 / _fp8_mixed_f32 / _fp4_f32
//                                      .kind::f8f6f4 specializations
//   make_idesc_s8_s32 / _u8_s32        .kind::i8 (with saturate flag)
//   idesc_set_sparsity(idesc, sel)     set bit 2 + sparsity selector
//   idesc_set_ws_mode(idesc, mode)     set bits 30-31 (B-reuse shift)
//   make_idesc_mxf8f6f4(...)           Table 54 packer
//   make_idesc_mxf4(...)               Table 55 mxf4 packer
//   make_idesc_mxf4nvf4(...)           Table 55 mxf4nvf4 (UE8M0 / UE4M3)
//
// All builders take M in [16, 256] and N in [8, 256] in steps of {16, 8}
// for Table 53, or M in {128, 256} and N in [8, 512] for Tables 54/55.
// `transpose_a` / `transpose_b` default to false.
//
//
// Issuer: host code or any device thread; the builders are pure bit math
// (constexpr-friendly).
// PTX:    9.7.18.4.2 Tables 53-55 (idesc per kind)
//
#include <cstdint>

// ----------------------- Table 53 common skeleton ---------------------------
// Bits:
//   4-5   dtype  (F32=1, F16=0, S32=2)
//   7-9   atype  (depends on .kind)
//  10-12  btype  (depends on .kind)
//   13    negate A
//   14    negate B
//   15    transpose A
//   16    transpose B
//  17-22  N >> 3
//  24-28  M >> 4
__device__ __forceinline__ uint32_t make_idesc_table44(
    int M, int N,
    uint32_t dtype, uint32_t atype, uint32_t btype,
    bool transpose_a = false, bool transpose_b = false,
    bool negate_a = false, bool negate_b = false) {
  uint32_t idesc = 0;
  idesc |= (dtype & 0x3) << 4;
  idesc |= (atype & 0x7) << 7;
  idesc |= (btype & 0x7) << 10;
  idesc |= (negate_a ? 1u : 0u) << 13;
  idesc |= (negate_b ? 1u : 0u) << 14;
  idesc |= (transpose_a ? 1u : 0u) << 15;
  idesc |= (transpose_b ? 1u : 0u) << 16;
  idesc |= ((static_cast<uint32_t>(N) >> 3) & 0x3F) << 17;
  idesc |= ((static_cast<uint32_t>(M) >> 4) & 0x1F) << 24;
  return idesc;
}

// .kind::f16 -- FP16 A, FP16 B, FP32 out.
__device__ __forceinline__ uint32_t make_idesc_f16_f32(
    int M, int N, bool ta = false, bool tb = false) {
  return make_idesc_table44(M, N, /*dtype=F32*/ 1,
                             /*atype=F16*/ 0, /*btype=F16*/ 0, ta, tb);
}

// .kind::f16 -- BF16 A, BF16 B, FP32 out.
__device__ __forceinline__ uint32_t make_idesc_bf16_f32(
    int M, int N, bool ta = false, bool tb = false) {
  return make_idesc_table44(M, N, /*dtype=F32*/ 1,
                             /*atype=BF16*/ 1, /*btype=BF16*/ 1, ta, tb);
}

// .kind::f16 -- FP16 A, FP16 B, FP16 out.
__device__ __forceinline__ uint32_t make_idesc_f16_f16(
    int M, int N, bool ta = false, bool tb = false) {
  return make_idesc_table44(M, N, /*dtype=F16*/ 0,
                             /*atype=F16*/ 0, /*btype=F16*/ 0, ta, tb);
}

// .kind::tf32 -- TF32 A/B, FP32 out.
__device__ __forceinline__ uint32_t make_idesc_tf32_f32(
    int M, int N, bool ta = false, bool tb = false) {
  return make_idesc_table44(M, N, /*dtype=F32*/ 1,
                             /*atype=TF32*/ 2, /*btype=TF32*/ 2, ta, tb);
}

// .kind::f8f6f4 -- E4M3 x E4M3 -> FP32.
__device__ __forceinline__ uint32_t make_idesc_e4m3_f32(
    int M, int N, bool ta = false, bool tb = false) {
  return make_idesc_table44(M, N, 1, 0, 0, ta, tb);
}

// .kind::f8f6f4 -- E5M2 x E5M2 -> FP32.
__device__ __forceinline__ uint32_t make_idesc_e5m2_f32(
    int M, int N, bool ta = false, bool tb = false) {
  return make_idesc_table44(M, N, 1, 1, 1, ta, tb);
}

// .kind::f8f6f4 -- mixed A/B FP8 types.
__device__ __forceinline__ uint32_t make_idesc_fp8_mixed_f32(
    int M, int N, uint32_t atype, uint32_t btype,
    bool ta = false, bool tb = false) {
  return make_idesc_table44(M, N, 1, atype, btype, ta, tb);
}

// .kind::f8f6f4 -- FP4 E2M1 x E2M1 -> FP32 (non-scaled).
__device__ __forceinline__ uint32_t make_idesc_fp4_f32(
    int M, int N, bool ta = false, bool tb = false) {
  return make_idesc_table44(M, N, 1, 5, 5, ta, tb);
}

// .kind::i8 -- INT8 signed, with saturation control.
__device__ __forceinline__ uint32_t make_idesc_s8_s32(
    int M, int N, bool saturate = false,
    bool ta = false, bool tb = false) {
  uint32_t idesc = make_idesc_table44(M, N, 2, 1, 1, ta, tb);
  idesc |= (saturate ? 1u : 0u) << 3;
  return idesc;
}

// .kind::i8 -- INT8 unsigned.
__device__ __forceinline__ uint32_t make_idesc_u8_s32(
    int M, int N, bool saturate = false,
    bool ta = false, bool tb = false) {
  uint32_t idesc = make_idesc_table44(M, N, 2, 0, 0, ta, tb);
  idesc |= (saturate ? 1u : 0u) << 3;
  return idesc;
}

// Set the sparsity bit (bit 2) in a Table 53 idesc. Bits 0-1 carry the
// 2-bit sparsity-selector index when sparsity is enabled.
__device__ __forceinline__ uint32_t idesc_set_sparsity(
    uint32_t idesc, uint32_t sparsity_selector) {
  idesc |= 1u << 2;
  idesc |= (sparsity_selector & 0x3) << 0;
  return idesc;
}

// Set the .ws (B-matrix reuse shift) mode (bits 30-31) in a Table 53 idesc.
//   ws_mode = 0 -> max shift 0 (default)
//   ws_mode = 1 -> max shift 8
//   ws_mode = 2 -> max shift 16
//   ws_mode = 3 -> max shift 32
__device__ __forceinline__ uint32_t idesc_set_ws_mode(
    uint32_t idesc, uint32_t ws_mode) {
  idesc &= ~(0x3u << 30);
  idesc |= (ws_mode & 0x3) << 30;
  return idesc;
}

// ----------------------- Table 54 (.kind::mxf8f6f4) -------------------------
// Block-scaled FP8/FP6/FP4 in the f8f6f4 family.
// Bits:
//   0-1   reserved (0)
//    2    sparsity (0=dense, 1=sparse)
//    3    reserved
//   4-5   B scale-factor data ID (0..3)
//    6    reserved
//   7-9   atype  (FP8 E4M3=0, E5M2=1, FP6 E2M3=3, E3M2=4, FP4 E2M1=5)
//  10-12  btype  (same encoding as atype)
//   13    negate A
//   14    negate B
//   15    transpose A
//   16    transpose B
//  17-22  N >> 3
//   23    scale type for both scales: UE8M0 = 1
//  24-26  reserved
//  27-28  M >> 7  (M must be 128 or 256)
//  29-30  A scale-factor data ID (0..3)
//   31    reserved
__device__ __forceinline__ uint32_t make_idesc_mxf8f6f4(
    int M, int N,
    uint32_t atype, uint32_t btype,
    bool ta = false, bool tb = false,
    bool negate_a = false, bool negate_b = false,
    uint32_t sf_a_data_id = 0, uint32_t sf_b_data_id = 0,
    bool ue8m0 = true, bool sparse = false) {
  uint32_t idesc = 0;
  idesc |= (sparse ? 1u : 0u) << 2;
  idesc |= (sf_b_data_id & 0x3) << 4;
  idesc |= (atype & 0x7) << 7;
  idesc |= (btype & 0x7) << 10;
  idesc |= (negate_a ? 1u : 0u) << 13;
  idesc |= (negate_b ? 1u : 0u) << 14;
  idesc |= (ta ? 1u : 0u) << 15;
  idesc |= (tb ? 1u : 0u) << 16;
  idesc |= ((static_cast<uint32_t>(N) >> 3) & 0x3F) << 17;
  idesc |= (ue8m0 ? 1u : 0u) << 23;
  idesc |= ((static_cast<uint32_t>(M) >> 7) & 0x3) << 27;
  idesc |= (sf_a_data_id & 0x3) << 29;
  return idesc;
}

// ----------------------- Table 55 (mxf4 / mxf4nvf4) -------------------------
// Bits:
//   4-5   dtype  (F32=1)
//   7-9   atype  (FP4=1 in mxf4 table)
//  10-12  btype  (FP4=1)
//   13    negate A
//   14    negate B
//   15    transpose A
//   16    transpose B
//  17-22  N >> 3
//  24-27  M >> 7  (NB: only 4 bits; M must be 128 or 256)
// Scale-factor data IDs occupy bits 0-6 and 28-31 and are typically 0 for
// block-scaled MMA configured via explicit scale_A / scale_B TMEM addresses.
__device__ __forceinline__ uint32_t make_idesc_mxf4(
    int M, int N, bool ta = false, bool tb = false,
    bool negate_a = false, bool negate_b = false) {
  uint32_t idesc = 0;
  idesc |= 1u << 4;             // dtype = F32
  idesc |= 1u << 7;             // atype = FP4 (Table 55)
  idesc |= 1u << 10;            // btype = FP4
  idesc |= (negate_a ? 1u : 0u) << 13;
  idesc |= (negate_b ? 1u : 0u) << 14;
  idesc |= (ta ? 1u : 0u) << 15;
  idesc |= (tb ? 1u : 0u) << 16;
  idesc |= ((static_cast<uint32_t>(N) >> 3) & 0x3F) << 17;
  idesc |= ((static_cast<uint32_t>(M) >> 7) & 0xF) << 24;
  return idesc;
}

// .kind::mxf4nvf4 with explicit scale-type selection. Per Table 55, bit 23
// encodes the scale matrix type for both scale_A and scale_B:
//   ue8m0 = true  -> UE8M0 (8-bit unsigned exponent)
//   ue8m0 = false -> UE4M3 (4-bit unsigned float; only legal with mxf4nvf4)
// Bit 31 = 0 selects the dense K=64 / sparse K=128 atom (the K=96 path lives
// in primitives/6_tcgen05_mma_fp4_k96.cuh).
__device__ __forceinline__ uint32_t make_idesc_mxf4nvf4(
    int M, int N,
    bool ta = false, bool tb = false,
    bool negate_a = false, bool negate_b = false,
    uint32_t sf_a_data_id = 0, uint32_t sf_b_data_id = 0,
    bool ue8m0 = false, bool sparse = false) {
  uint32_t idesc = 0;
  idesc |= (sparse ? 1u : 0u) << 2;
  idesc |= (sf_b_data_id & 0x3) << 4;
  idesc |= 1u << 7;             // atype = E2M1
  idesc |= 1u << 10;            // btype = E2M1
  idesc |= (negate_a ? 1u : 0u) << 13;
  idesc |= (negate_b ? 1u : 0u) << 14;
  idesc |= (ta ? 1u : 0u) << 15;
  idesc |= (tb ? 1u : 0u) << 16;
  idesc |= ((static_cast<uint32_t>(N) >> 3) & 0x3F) << 17;
  idesc |= (ue8m0 ? 1u : 0u) << 23;
  idesc |= ((static_cast<uint32_t>(M) >> 7) & 0x3) << 27;
  idesc |= (sf_a_data_id & 0x3) << 29;
  // bit 31 = 0 (K=64 dense / K=128 sparse).
  return idesc;
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

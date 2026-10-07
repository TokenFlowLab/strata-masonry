#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 62_clc_query_cancel.cuh -- clusterlaunchcontrol.query_cancel.*
//
// ARCH: sm_100a
//
// Decodes a try_cancel (#61) response. The response is a 128-bit opaque
// handle; callers pass it as 4 uint32_t register operands (loaded from
// the 16-byte SMEM slot with ld.shared.v4.b32).
// PTX:    9.7.15.19 (clusterlaunchcontrol.query_cancel)
//
#include <cstdint>

// Load the 128-bit response from an SMEM slot (4 x b32).
__device__ __forceinline__ void clc_load_response(
    uint32_t smem_slot, uint32_t& r0, uint32_t& r1,
    uint32_t& r2, uint32_t& r3) {
  asm volatile("ld.shared::cta.v4.b32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(smem_slot));
}

// Returns 1 if try_cancel succeeded.
__device__ __forceinline__ uint32_t clc_query_is_canceled(
    uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  uint32_t pred;
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    ".reg .b128 h;\n\t"
    "mov.b128 h, {%1, %2, %3, %4};\n\t"
    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, h;\n\t"
    "selp.b32 %0, 1, 0, p;\n\t"
    "}\n"
    : "=r"(pred)
    : "r"(r0), "r"(r1), "r"(r2), "r"(r3));
  return pred;
}

__device__ __forceinline__ uint32_t clc_query_first_ctaid_x(
    uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  uint32_t out;
  asm volatile(
    "{\n\t"
    ".reg .b128 h;\n\t"
    "mov.b128 h, {%1, %2, %3, %4};\n\t"
    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, h;\n\t"
    "}\n"
    : "=r"(out)
    : "r"(r0), "r"(r1), "r"(r2), "r"(r3));
  return out;
}

__device__ __forceinline__ uint32_t clc_query_first_ctaid_y(
    uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  uint32_t out;
  asm volatile(
    "{\n\t"
    ".reg .b128 h;\n\t"
    "mov.b128 h, {%1, %2, %3, %4};\n\t"
    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, h;\n\t"
    "}\n"
    : "=r"(out)
    : "r"(r0), "r"(r1), "r"(r2), "r"(r3));
  return out;
}

__device__ __forceinline__ uint32_t clc_query_first_ctaid_z(
    uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
  uint32_t out;
  asm volatile(
    "{\n\t"
    ".reg .b128 h;\n\t"
    "mov.b128 h, {%1, %2, %3, %4};\n\t"
    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, h;\n\t"
    "}\n"
    : "=r"(out)
    : "r"(r0), "r"(r1), "r"(r2), "r"(r3));
  return out;
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

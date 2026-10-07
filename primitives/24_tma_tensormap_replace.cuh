// 24_tma_tensormap_replace.cuh -- tensormap.replace.* (runtime tensormap mutation)
//
// ARCH: sm_90a
//
// Mutates fields of a tensormap without host round-trip. Used in
// persistent / grouped GEMM kernels to change the base pointer, dimensions,
// strides, etc. between tiles.
//
// The mutated tensormap resides in .global or .shared::cta memory; after
// each tensormap.replace, a fence.proxy.tensormap::generic.release (#36) is
// required before the next TMA read sees the update.
//
// Note: the "ordinal" operand in the PTX is an immediate integer, so the
// generic templated forms parameterize on it. The hardcoded `_dim0` /
// `_dim1` / `_stride0` flavors are convenience wrappers.

#pragma once

// PTX:    9.7.10.29 (tensormap.replace)
//
#include <cuda.h>
#include <cstdint>

// -- global_address (two overloads: uint64_t vs const void*) -----------------

__device__ __forceinline__
void tensormap_replace_global_address(void* tensormap_addr, uint64_t new_base) {
  asm volatile(
    "tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;\n"
    :: "l"(tensormap_addr), "l"(new_base)
    : "memory");
}

__device__ __forceinline__
void tensormap_replace_global_address(void* tensormap_addr,
                                      const void* new_gmem_ptr) {
  asm volatile(
    "tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;\n"
    :: "l"(tensormap_addr), "l"(new_gmem_ptr)
    : "memory");
}

// -- global_dim --------------------------------------------------------------

template <int ORDINAL>
__device__ __forceinline__
void tensormap_replace_global_dim(void* tensormap_addr, uint32_t new_dim) {
  asm volatile(
    "tensormap.replace.tile.global_dim.global.b1024.b32 [%0], %1, %2;\n"
    :: "l"(tensormap_addr), "n"(ORDINAL), "r"(new_dim)
    : "memory");
}

// Hardcoded dim-0 (columns) wrapper.
__device__ __forceinline__
void tensormap_replace_global_dim0(void* tensormap_ptr, uint32_t new_dim0) {
  asm volatile(
    "tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 0, %1;\n"
    :: "l"(tensormap_ptr), "r"(new_dim0)
    : "memory");
}

// Hardcoded dim-1 (rows) wrapper.
__device__ __forceinline__
void tensormap_replace_global_dim1(void* tensormap_ptr, uint32_t new_dim1) {
  asm volatile(
    "tensormap.replace.tile.global_dim.global.b1024.b32 [%0], 1, %1;\n"
    :: "l"(tensormap_ptr), "r"(new_dim1)
    : "memory");
}

// -- global_stride -----------------------------------------------------------

template <int ORDINAL>
__device__ __forceinline__
void tensormap_replace_global_stride(void* tensormap_addr,
                                     uint64_t new_stride) {
  asm volatile(
    "tensormap.replace.tile.global_stride.global.b1024.b64 [%0], %1, %2;\n"
    :: "l"(tensormap_addr), "n"(ORDINAL), "l"(new_stride)
    : "memory");
}

// Hardcoded stride-0 wrapper.
__device__ __forceinline__
void tensormap_replace_global_stride0(void* tensormap_ptr,
                                      uint64_t new_stride) {
  asm volatile(
    "tensormap.replace.tile.global_stride.global.b1024.b64 [%0], 0, %1;\n"
    :: "l"(tensormap_ptr), "l"(new_stride)
    : "memory");
}

// -- box_dim -----------------------------------------------------------------

template <int ORDINAL>
__device__ __forceinline__
void tensormap_replace_box_dim(void* tensormap_addr, uint32_t new_box) {
  asm volatile(
    "tensormap.replace.tile.box_dim.global.b1024.b32 [%0], %1, %2;\n"
    :: "l"(tensormap_addr), "n"(ORDINAL), "r"(new_box)
    : "memory");
}

// -- rank / elemtype (rarely used) -------------------------------------------

__device__ __forceinline__
void tensormap_replace_rank(void* tensormap_ptr, uint32_t new_rank) {
  asm volatile(
    "tensormap.replace.tile.rank.global.b1024.b32 [%0], %1;\n"
    :: "l"(tensormap_ptr), "r"(new_rank)
    : "memory");
}

__device__ __forceinline__
void tensormap_replace_elemtype(void* tensormap_ptr, uint32_t new_elemtype) {
  asm volatile(
    "tensormap.replace.tile.elemtype.global.b1024.b32 [%0], %1;\n"
    :: "l"(tensormap_ptr), "r"(new_elemtype)
    : "memory");
}

// ---------------------------------------------------------------------------
// `.shared::cta` dst-space variants (tensormap resident in SMEM)
// ---------------------------------------------------------------------------
// Used when the kernel allocates a tensormap in shared memory, mutates it
// in-place, then issues a TMA load/store through it. Caller passes a
// 32-bit SMEM address (e.g. `cvta_to_shared_u32(&smem_tmap)`).
// After mutating, callers must issue `fence.proxy.tensormap::generic.release.<scope>`
// (file #36) before the next TMA read of the modified tensormap.

__device__ __forceinline__
void tensormap_replace_global_address_smem(uint32_t tmap_smem,
                                           uint64_t new_base) {
  asm volatile(
    "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;\n"
    :: "r"(tmap_smem), "l"(new_base)
    : "memory");
}

__device__ __forceinline__
void tensormap_replace_global_address_smem(uint32_t tmap_smem,
                                           const void* new_gmem_ptr) {
  asm volatile(
    "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;\n"
    :: "r"(tmap_smem), "l"(new_gmem_ptr)
    : "memory");
}

template <int ORDINAL>
__device__ __forceinline__
void tensormap_replace_global_dim_smem(uint32_t tmap_smem, uint32_t new_dim) {
  asm volatile(
    "tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [%0], %1, %2;\n"
    :: "r"(tmap_smem), "n"(ORDINAL), "r"(new_dim)
    : "memory");
}

template <int ORDINAL>
__device__ __forceinline__
void tensormap_replace_global_stride_smem(uint32_t tmap_smem,
                                          uint64_t new_stride) {
  asm volatile(
    "tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [%0], %1, %2;\n"
    :: "r"(tmap_smem), "n"(ORDINAL), "l"(new_stride)
    : "memory");
}

template <int ORDINAL>
__device__ __forceinline__
void tensormap_replace_box_dim_smem(uint32_t tmap_smem, uint32_t new_box) {
  asm volatile(
    "tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [%0], %1, %2;\n"
    :: "r"(tmap_smem), "n"(ORDINAL), "r"(new_box)
    : "memory");
}

__device__ __forceinline__
void tensormap_replace_rank_smem(uint32_t tmap_smem, uint32_t new_rank) {
  asm volatile(
    "tensormap.replace.tile.rank.shared::cta.b1024.b32 [%0], %1;\n"
    :: "r"(tmap_smem), "r"(new_rank)
    : "memory");
}

__device__ __forceinline__
void tensormap_replace_elemtype_smem(uint32_t tmap_smem, uint32_t new_elemtype) {
  asm volatile(
    "tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [%0], %1;\n"
    :: "r"(tmap_smem), "r"(new_elemtype)
    : "memory");
}

// ---------------------------------------------------------------------------
// swizzle_mode replace -- mutates the tensormap's swizzle field at runtime.
// Field values match `CUtensorMapSwizzle` (0=NONE, 1=32B, 2=64B, 3=128B;
// 4=128B_FLIP_8B is sm_103a-only and intentionally not exposed here).
// PTX requires the swizzle operand as a 32-bit value (immediate or register);
// template-immediate is used for type-safe compile-time selection consistent
// with the ordinal-immediate convention in this file.
// ---------------------------------------------------------------------------

template <int SWIZZLE>
__device__ __forceinline__
void tensormap_replace_swizzle_mode(void* tensormap_addr) {
  static_assert(SWIZZLE >= 0 && SWIZZLE <= 3,
                "SWIZZLE must be 0(NONE)/1(32B)/2(64B)/3(128B) on sm_90a/sm_100a");
  asm volatile(
    "tensormap.replace.tile.swizzle_mode.global.b1024.b32 [%0], %1;\n"
    :: "l"(tensormap_addr), "n"(SWIZZLE)
    : "memory");
}

template <int SWIZZLE>
__device__ __forceinline__
void tensormap_replace_swizzle_mode_smem(uint32_t tmap_smem) {
  static_assert(SWIZZLE >= 0 && SWIZZLE <= 3,
                "SWIZZLE must be 0(NONE)/1(32B)/2(64B)/3(128B) on sm_90a/sm_100a");
  asm volatile(
    "tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [%0], %1;\n"
    :: "r"(tmap_smem), "n"(SWIZZLE)
    : "memory");
}

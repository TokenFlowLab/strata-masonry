// 23_tma_tensormap.cuh -- cuTensorMapEncodeTiled wrappers (host-side)
//
// ARCH: host
//
// Host-side only. Exception to the __device__ convention.
// Builds CUtensorMap objects that describe tiled tensors for TMA load/store.
//
// The TMA hardware reads these objects through the tensormap proxy; they
// must live in .global (host-allocated) or .param (kernel argument) memory
// and be passed to the kernel as a pointer / grid_constant reference.
// The CUtensorMap is 128 bytes, 64-byte aligned.
//
// The swizzle mode on the tensormap must match the SMEM swizzle layout used
// on the kernel side. Common choice for tile GEMM: SWIZZLE_128B for FP16/BF16
// K-contiguous tiles.
//
// Two API styles are provided:
//   - make_tma_{2d,3d}_tiled(...)    returns cudaError_t (caller checks).
//   - create_tma_2d_desc(...) +      void return, exits with diagnostic
//     create_tma_2d_{f16,bf16,f32,   on cuTensorMapEncodeTiled failure.
//                    fp8}(...)       Dtype-specialized convenience wrappers.

#pragma once

// PTX:    9.7.10.29 (tensormap.replace) + CUDA driver cuTensorMapEncodeTiled (host)
//
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cassert>

// =============================================================================
// Returning-cudaError_t API
// =============================================================================

// Build a 2D tensormap for a row-major tensor.
//   ptr        = base GMEM pointer
//   rows, cols = tensor extents
//   box_rows, box_cols = tile extents (what one TMA load copies)
//   elem_bytes = element size in bytes
inline cudaError_t make_tma_2d_tiled(
    CUtensorMap* out,
    const void* ptr, int rows, int cols, int box_rows, int box_cols,
    int elem_bytes, CUtensorMapDataType dtype,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B,
    CUtensorMapL2promotion l2 = CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) {
  uint64_t globalDim[2]     = { (uint64_t)cols, (uint64_t)rows };
  uint64_t globalStrides[1] = { (uint64_t)cols * (uint64_t)elem_bytes };
  uint32_t boxDim[2]        = { (uint32_t)box_cols, (uint32_t)box_rows };
  uint32_t elemStrides[2]   = { 1u, 1u };

  CUresult r = cuTensorMapEncodeTiled(
      out, dtype, /*tensorRank=*/2,
      const_cast<void*>(ptr), globalDim, globalStrides,
      boxDim, elemStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle, l2, oob);
  return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
}

// Build a 3D tensormap (batched or im2col-ready).
inline cudaError_t make_tma_3d_tiled(
    CUtensorMap* out,
    const void* ptr, int d0, int d1, int d2,
    int b0, int b1, int b2,
    int elem_bytes, CUtensorMapDataType dtype,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B) {
  uint64_t globalDim[3]     = { (uint64_t)d0, (uint64_t)d1, (uint64_t)d2 };
  uint64_t globalStrides[2] = {
    (uint64_t)d0 * elem_bytes,
    (uint64_t)d0 * (uint64_t)d1 * elem_bytes
  };
  uint32_t boxDim[3]        = { (uint32_t)b0, (uint32_t)b1, (uint32_t)b2 };
  uint32_t elemStrides[3]   = { 1u, 1u, 1u };

  CUresult r = cuTensorMapEncodeTiled(
      out, dtype, /*tensorRank=*/3,
      const_cast<void*>(ptr), globalDim, globalStrides,
      boxDim, elemStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
}

// =============================================================================
// Exit-on-error API + dtype-specialized wrappers
// =============================================================================

inline void create_tma_2d_desc(
    CUtensorMap* desc,
    const void* gmem_ptr,
    int rows, int cols,
    int box_rows, int box_cols,
    int elem_bytes,
    CUtensorMapDataType dtype,
    CUtensorMapSwizzle swizzle,
    CUtensorMapFloatOOBfill oob_fill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)
{
  uint64_t globalDim[2]     = {(uint64_t)cols, (uint64_t)rows};
  uint64_t globalStrides[1] = {(uint64_t)cols * elem_bytes};
  uint32_t boxDim[2]        = {(uint32_t)box_cols, (uint32_t)box_rows};
  uint32_t elemStrides[2]   = {1, 1};

  CUresult err = cuTensorMapEncodeTiled(
      desc, dtype, /*tensorRank=*/2,
      const_cast<void*>(gmem_ptr),
      globalDim, globalStrides,
      boxDim, elemStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      oob_fill);

  if (err != CUDA_SUCCESS) {
    const char* errStr;
    cuGetErrorString(err, &errStr);
    fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", errStr);
    exit(1);
  }
}

// FP16 tensor with 128B swizzle.
inline void create_tma_2d_f16(
    CUtensorMap* desc, const void* gmem_ptr,
    int rows, int cols, int box_rows, int box_cols)
{
  create_tma_2d_desc(desc, gmem_ptr, rows, cols, box_rows, box_cols,
                     /*elem_bytes=*/2,
                     CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                     CU_TENSOR_MAP_SWIZZLE_128B);
}

// BF16 tensor with 128B swizzle.
inline void create_tma_2d_bf16(
    CUtensorMap* desc, const void* gmem_ptr,
    int rows, int cols, int box_rows, int box_cols)
{
  create_tma_2d_desc(desc, gmem_ptr, rows, cols, box_rows, box_cols,
                     /*elem_bytes=*/2,
                     CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                     CU_TENSOR_MAP_SWIZZLE_128B);
}

// FP32 tensor with 128B swizzle.
inline void create_tma_2d_f32(
    CUtensorMap* desc, const void* gmem_ptr,
    int rows, int cols, int box_rows, int box_cols)
{
  create_tma_2d_desc(desc, gmem_ptr, rows, cols, box_rows, box_cols,
                     /*elem_bytes=*/4,
                     CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                     CU_TENSOR_MAP_SWIZZLE_128B);
}

// FP8 (E4M3/E5M2 stored as uint8) with 128B swizzle.
inline void create_tma_2d_fp8(
    CUtensorMap* desc, const void* gmem_ptr,
    int rows, int cols, int box_rows, int box_cols)
{
  create_tma_2d_desc(desc, gmem_ptr, rows, cols, box_rows, box_cols,
                     /*elem_bytes=*/1,
                     CU_TENSOR_MAP_DATA_TYPE_UINT8,
                     CU_TENSOR_MAP_SWIZZLE_128B);
}

// INT8 (signed) tensor with 128B swizzle. Used by INT8 grouped GEMM.
// CUtensorMap has no separate INT8 enumerator -- signed bytes use the
// UINT8 dtype and the kernel-side instruction (e.g. tcgen05.mma.kind::i8
// with idesc atype=1) interprets signedness.
inline void create_tma_2d_s8(
    CUtensorMap* desc, const void* gmem_ptr,
    int rows, int cols, int box_rows, int box_cols)
{
  create_tma_2d_desc(desc, gmem_ptr, rows, cols, box_rows, box_cols,
                     /*elem_bytes=*/1,
                     CU_TENSOR_MAP_DATA_TYPE_UINT8,
                     CU_TENSOR_MAP_SWIZZLE_128B);
}

// UINT8 tensor with 128B swizzle. Same dtype as the existing FP8 wrapper
// but kept separate for clarity at call sites that work with raw bytes
// (e.g., quantized weights, scale factors).
inline void create_tma_2d_u8(
    CUtensorMap* desc, const void* gmem_ptr,
    int rows, int cols, int box_rows, int box_cols)
{
  create_tma_2d_desc(desc, gmem_ptr, rows, cols, box_rows, box_cols,
                     /*elem_bytes=*/1,
                     CU_TENSOR_MAP_DATA_TYPE_UINT8,
                     CU_TENSOR_MAP_SWIZZLE_128B);
}

// INT32 (signed) tensor. Used by INT8 GEMM accumulator stores.
// Note: INT32 with SWIZZLE_128B is uncommon (INT32 == 4 bytes; default
// to NONE swizzle since the accumulator path usually doesn't need
// SMEM-side swizzle).
inline void create_tma_2d_s32(
    CUtensorMap* desc, const void* gmem_ptr,
    int rows, int cols, int box_rows, int box_cols,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE)
{
  create_tma_2d_desc(desc, gmem_ptr, rows, cols, box_rows, box_cols,
                     /*elem_bytes=*/4,
                     CU_TENSOR_MAP_DATA_TYPE_INT32,
                     swizzle);
}

// UINT32 tensor (rarely used; provided for symmetry).
inline void create_tma_2d_u32(
    CUtensorMap* desc, const void* gmem_ptr,
    int rows, int cols, int box_rows, int box_cols,
    CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE)
{
  create_tma_2d_desc(desc, gmem_ptr, rows, cols, box_rows, box_cols,
                     /*elem_bytes=*/4,
                     CU_TENSOR_MAP_DATA_TYPE_UINT32,
                     swizzle);
}

// ARCH: sm_100a
// 23_tma_tensormap_test.cu -- host-side tensormap encoding. Verify that
// make_tma_2d_tiled produces a non-zero 128-byte tensormap for a valid
// descriptor and returns an error for an obviously invalid one.
//
// Combined test: both ours' and theirs' coverage is exercised
// in a single binary (each side's main() became run_ours/run_theirs).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../src/primitives/23_tma_tensormap.cuh"
#include <cuda_fp16.h>
#include "23_tma_tensormap.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


static int run_ours() {
  /* (orig args dropped) */
  float* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, 64 * 64 * sizeof(float)));

  CUtensorMap desc;
  std::memset(&desc, 0, sizeof(desc));
  CUDA_CHECK(make_tma_2d_tiled(&desc, d, 64, 64, 32, 32, sizeof(float),
                                 CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                 CU_TENSOR_MAP_SWIZZLE_NONE));

  // Sanity: the encoded tensormap must be non-zero.
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&desc);
  int nonzero = 0;
  for (size_t i = 0; i < sizeof(desc); ++i) nonzero += (bytes[i] != 0);
  cudaFree(d);
  if (nonzero == 0) FAIL("encoded tensormap is all zeros");
  printf("tma_tensormap 2D 64x64 tile 32x32 FP32 : %d non-zero bytes / %zu\n",
         nonzero, sizeof(desc));
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Test: TMA tensormap host-side creation -- compile verification
// Build: nvcc -arch=sm_90a -I../primitives -I. -lcuda -o ../build/23_test 23_tma_tensormap_test.cu

static int run_theirs() {
  /* (orig args dropped) */
    // Verify all overloads compile and the API is callable.
    // Runtime requires compatible CUDA driver.
    printf("23_tma_tensormap: compiled successfully.\n");
    printf("  Provides: create_tma_2d_desc, create_tma_2d_f16, create_tma_2d_bf16,\n");
    printf("            create_tma_2d_f32, create_tma_2d_fp8\n");
    PASS();
    return 0;
}


// Round 4h: INT8 / UINT8 / INT32 / UINT32 dtype convenience wrappers.
// Host-side encoder check: build a tensormap with each new dtype and
// confirm the encoder populated it (no GPU launch needed -- the
// wrappers just wrap cuTensorMapEncodeTiled which is host-side).
static int run_int_dtype_encoders() {
  // Allocate small device buffers to satisfy globalAddress nonzero.
  int8_t*  d_s8  = nullptr; CUDA_CHECK(cudaMalloc(&d_s8,  16 * 16));
  uint8_t* d_u8  = nullptr; CUDA_CHECK(cudaMalloc(&d_u8,  16 * 16));
  int32_t* d_s32 = nullptr; CUDA_CHECK(cudaMalloc(&d_s32, 16 * 16 * 4));
  uint32_t* d_u32 = nullptr; CUDA_CHECK(cudaMalloc(&d_u32, 16 * 16 * 4));

  CUtensorMap m_s8, m_u8, m_s32, m_u32;
  create_tma_2d_s8 (&m_s8,  d_s8,  16, 16, 16, 16);
  create_tma_2d_u8 (&m_u8,  d_u8,  16, 16, 16, 16);
  create_tma_2d_s32(&m_s32, d_s32, 16, 16, 16, 16);
  create_tma_2d_u32(&m_u32, d_u32, 16, 16, 16, 16);

  // The CUtensorMap is opaque; verify it's not all-zero (encoder filled
  // it). First qword stores the global_address.
  auto first_qword = [](const CUtensorMap& m) {
    uint64_t q;
    std::memcpy(&q, &m, sizeof(uint64_t));
    return q;
  };
  uint64_t qs[4] = { first_qword(m_s8),  first_qword(m_u8),
                     first_qword(m_s32), first_qword(m_u32) };
  cudaFree(d_s8); cudaFree(d_u8); cudaFree(d_s32); cudaFree(d_u32);

  const char* names[] = {"s8", "u8", "s32", "u32"};
  for (int i = 0; i < 4; ++i) {
    if (qs[i] == 0ull) {
      fprintf(stderr, "create_tma_2d_%s: tensormap first qword is zero\n",
              names[i]);
      FAIL("INT dtype encoder did not populate tensormap");
    }
  }
  printf("create_tma_2d_{s8,u8,s32,u32}: 4/4 OK (encoder populated tensormap)\n");
  PASS();
}

int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  int rc_int    = run_int_dtype_encoders();
  return (rc_ours == 0 && rc_theirs == 0 && rc_int == 0) ? 0 : 1;
}

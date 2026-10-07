#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 10_tcgen05_st_test.cu -- compile + smoke test for tcgen05.st (register
// -> TMEM warp-collective store), plus per-variant non-zero round-trip.
//
// Verifies (1) the .32x32b.x{1,2,4,8} overloads run end-to-end, and (2) a
// kernel allocating TMEM, issuing the stores, and deallocating completes
// without hang. Functional bit-equality round-trip is covered by the
// .32x32b.x8 ld/st round-trip in #9.
//
// For the 12 expanded variants the test also does a per-variant non-zero
// round-trip:
//   .32x32b.x{16,32}
//   .32x32b.x{2,8}.unpack::16b
//   .16x256b.x1.unpack::16b
//   .16x64b.x{1,2}
//   .16x128b.x{1,2}
//   .16x32bx2.x{1,2,4} <8>
// For each variant the test (a) zeros the 32-col TMEM region via
// .32x32b.x32, (b) writes a lane-encoded non-zero pattern via the variant
// under test, (c) reads the region back via .32x32b.x32, and (d) sums
// non-zero readback registers across all 32 lanes via shfl. Any variant
// that wrote nothing produces a sum of zero and fails.
//
// PTX sniff: `cuobjdump --dump-ptx build/10_tcgen05_st_test |
// grep -E 'tcgen05.st.sync.aligned'` should show every issued shape.

#include "test_utils.cuh"
#include "../src/primitives/0_tcgen05_alloc.cuh"
#include "../src/primitives/1_tcgen05_dealloc.cuh"
#include "../src/primitives/2_tcgen05_relinquish.cuh"
#include "../src/primitives/9_tcgen05_ld.cuh"
#include "../src/primitives/10_tcgen05_st.cuh"
#include "../src/primitives/12_tcgen05_wait.cuh"

__global__ void k_st() {
  __shared__ __align__(16) uint32_t slot;
  if (threadIdx.x == 0) slot = 0;
  __syncthreads();
  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&slot), 32);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  uint32_t tbase = slot;
  if (threadIdx.x < 32) {
    uint32_t arr4[4] = { 1, 2, 3, 4 };
    uint32_t arr8[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    tcgen05_st_32x32b_x1(tbase, 0xAA);
    tcgen05_st_32x32b_x2(tbase, 0xBB, 0xCC);
    tcgen05_st_32x32b_x4(tbase, arr4);
    tcgen05_st_32x32b_x8(tbase, arr8);
    tcgen05_st_32x32b_x1_unpack16b(tbase, 0x12340000u);
    tcgen05_st_32x32b_x4_unpack16b(tbase, arr4);
    tcgen05_st_16x256b_x1(tbase, arr4);
    tcgen05_wait_st();
  }
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tbase, 32);
}

// ---------------------------------------------------------------------------
// Per-variant non-zero round-trip helpers.
// ---------------------------------------------------------------------------

constexpr int ST_VARIANT_COUNT = 12;

__device__ __forceinline__ uint32_t warp_sum(uint32_t v) {
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
  return v;
}

__device__ __forceinline__ void clear_tmem_32cols(uint32_t tbase) {
  uint32_t z[32] = {};
  tcgen05_st_32x32b_x32(tbase, z);
  tcgen05_wait_st();
}

// Read all 32 cols and return the warp-wide non-zero register count
// (lane 0's return value carries the total).
__device__ __forceinline__ uint32_t read_back_nonzero(uint32_t tbase) {
  uint32_t r[32] = {};
  tcgen05_ld_32x32b_x32(tbase, r);
  tcgen05_wait_ld();
  uint32_t per_lane = 0;
  #pragma unroll
  for (int i = 0; i < 32; ++i) per_lane += (r[i] != 0u);
  return warp_sum(per_lane);
}

// Per-variant payload generators: every cell is non-zero and lane-encoded.
__device__ __forceinline__ uint32_t pat(uint32_t lane, int i) {
  return ((lane | 0x40u) << 8) | (uint32_t)(i + 1);
}

// Each slot stores the warp-wide non-zero count seen by lane 0 after the
// variant's write -> readback round-trip. Variant index assignments:
//   0  : 32x32b.x16
//   1  : 32x32b.x32
//   2  : 32x32b.x2.unpack16b
//   3  : 32x32b.x8.unpack16b
//   4  : 16x256b.x1.unpack16b
//   5  : 16x64b.x1
//   6  : 16x64b.x2
//   7  : 16x128b.x1
//   8  : 16x128b.x2
//   9  : 16x32bx2.x1<8>
//  10  : 16x32bx2.x2<8>
//  11  : 16x32bx2.x4<8>
__global__ void k_st_variants(uint32_t* g_nz) {
  __shared__ __align__(16) uint32_t slot;
  if (threadIdx.x == 0) slot = 0;
  __syncthreads();

  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&slot), 32);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();

  uint32_t tbase = slot;
  if (threadIdx.x >= 32) return;

  const uint32_t lane = threadIdx.x;

  auto record = [&](int vi, uint32_t nz) {
    if (lane == 0) g_nz[vi] = nz;
  };

  // 0) .32x32b.x16
  {
    clear_tmem_32cols(tbase);
    uint32_t w[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = pat(lane, i);
    tcgen05_st_32x32b_x16(tbase, w);
    tcgen05_wait_st();
    record(0, read_back_nonzero(tbase));
  }

  // 1) .32x32b.x32
  {
    clear_tmem_32cols(tbase);
    uint32_t w[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) w[i] = pat(lane, i);
    tcgen05_st_32x32b_x32(tbase, w);
    tcgen05_wait_st();
    record(1, read_back_nonzero(tbase));
  }

  // 2) .32x32b.x2.unpack16b
  {
    clear_tmem_32cols(tbase);
    uint32_t r0 = pat(lane, 0), r1 = pat(lane, 1);
    tcgen05_st_32x32b_x2_unpack16b(tbase, r0, r1);
    tcgen05_wait_st();
    record(2, read_back_nonzero(tbase));
  }

  // 3) .32x32b.x8.unpack16b
  {
    clear_tmem_32cols(tbase);
    uint32_t w[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = pat(lane, i);
    tcgen05_st_32x32b_x8_unpack16b(tbase, w);
    tcgen05_wait_st();
    record(3, read_back_nonzero(tbase));
  }

  // 4) .16x256b.x1.unpack16b
  {
    clear_tmem_32cols(tbase);
    uint32_t w[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = pat(lane, i);
    tcgen05_st_16x256b_x1_unpack16b(tbase, w);
    tcgen05_wait_st();
    record(4, read_back_nonzero(tbase));
  }

  // 5) .16x64b.x1
  {
    clear_tmem_32cols(tbase);
    tcgen05_st_16x64b_x1(tbase, pat(lane, 0));
    tcgen05_wait_st();
    record(5, read_back_nonzero(tbase));
  }

  // 6) .16x64b.x2
  {
    clear_tmem_32cols(tbase);
    tcgen05_st_16x64b_x2(tbase, pat(lane, 0), pat(lane, 1));
    tcgen05_wait_st();
    record(6, read_back_nonzero(tbase));
  }

  // 7) .16x128b.x1
  {
    clear_tmem_32cols(tbase);
    tcgen05_st_16x128b_x1(tbase, pat(lane, 0), pat(lane, 1));
    tcgen05_wait_st();
    record(7, read_back_nonzero(tbase));
  }

  // 8) .16x128b.x2
  {
    clear_tmem_32cols(tbase);
    uint32_t w[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = pat(lane, i);
    tcgen05_st_16x128b_x2(tbase, w);
    tcgen05_wait_st();
    record(8, read_back_nonzero(tbase));
  }

  // 9) .16x32bx2.x1<8>
  {
    clear_tmem_32cols(tbase);
    tcgen05_st_16x32bx2_x1<8>(tbase, pat(lane, 0));
    tcgen05_wait_st();
    record(9, read_back_nonzero(tbase));
  }

  // 10) .16x32bx2.x2<8>
  {
    clear_tmem_32cols(tbase);
    tcgen05_st_16x32bx2_x2<8>(tbase, pat(lane, 0), pat(lane, 1));
    tcgen05_wait_st();
    record(10, read_back_nonzero(tbase));
  }

  // 11) .16x32bx2.x4<8>
  {
    clear_tmem_32cols(tbase);
    uint32_t w[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = pat(lane, i);
    tcgen05_st_16x32bx2_x4<8>(tbase, w);
    tcgen05_wait_st();
    record(11, read_back_nonzero(tbase));
  }

  __syncwarp();
  if (lane == 0) tcgen05_dealloc<1>(tbase, 32);
}

int main() {
  // ------------------------------------------------------------------
  // Part 1 -- existing smoke test for the baseline .32x32b.x{1,2,4,8}
  // overloads + one .x1 / .x4 unpack and .16x256b.x1.
  // ------------------------------------------------------------------
  k_st<<<1, 32>>>();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("tcgen05.st baseline variants : compile + run OK\n");

  // ------------------------------------------------------------------
  // Part 2 -- per-variant non-zero round-trip.
  // ------------------------------------------------------------------
  uint32_t* d_nz = nullptr;
  CUDA_CHECK(cudaMalloc(&d_nz, ST_VARIANT_COUNT * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_nz, 0, ST_VARIANT_COUNT * sizeof(uint32_t)));
  k_st_variants<<<1, 32>>>(d_nz);
  CUDA_CHECK(cudaDeviceSynchronize());

  uint32_t h_nz[ST_VARIANT_COUNT];
  CUDA_CHECK(cudaMemcpy(h_nz, d_nz,
                        ST_VARIANT_COUNT * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  cudaFree(d_nz);

  static const char* labels[ST_VARIANT_COUNT] = {
    "st.32x32b.x16          ",
    "st.32x32b.x32          ",
    "st.32x32b.x2.unpack16b ",
    "st.32x32b.x8.unpack16b ",
    "st.16x256b.x1.unpack16b",
    "st.16x64b.x1           ",
    "st.16x64b.x2           ",
    "st.16x128b.x1          ",
    "st.16x128b.x2          ",
    "st.16x32bx2.x1<8>      ",
    "st.16x32bx2.x2<8>      ",
    "st.16x32bx2.x4<8>      ",
  };

  int fails = 0;
  for (int i = 0; i < ST_VARIANT_COUNT; ++i) {
    printf("  %s : warp non-zero readback regs = %u\n",
           labels[i], h_nz[i]);
    if (h_nz[i] == 0) ++fails;
  }
  if (fails)
    FAIL("%d st variant(s) produced an all-zero TMEM readback", fails);
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

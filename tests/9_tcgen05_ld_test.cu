#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 9_tcgen05_ld_test.cu -- end-to-end TMEM round-trip via tcgen05.{st,ld}.
//
// Verifies (1) tcgen05.st writes a known register pattern into TMEM, (2)
// tcgen05.wait::st observes store completion, (3) tcgen05.ld reads the
// pattern back (collectively across all 32 lanes of the warp), and (4) the
// loaded register values exactly match what was written. Exercises the
// .32x32b shape across .x{1,2,4,8} num levels.
//
// Also: per-variant non-zero round-trip for
// the expanded shape x num matrix:
//   .16x64b.x{2,4,8}, .16x128b.x{2,4}, .16x256b.x2,
//   .32x32b.x{1,4}.pack::16b, .16x256b.x1.pack::16b,
//   .16x32bx2.x{1,2,4} <8>
// A second kernel pre-fills 32 TMEM cols via tcgen05_st_32x32b_x32 with a
// lane-encoded non-zero pattern, then issues each ld variant and reports
// how many of lane 0's destination regs come back non-zero. Bit-equality
// is verified by the existing .32x32b round-trip; per-variant correctness
// here is "the asm executes and populates the destination regs".
//
// Also: 5 sm_103a-only ld.red runtime
// tests (tcgen05.ld.red.32x32b.x2 with min/max/max.abs/max.u32/min.s32).
// Gated by `#if defined(PL_AGENTIC_SM103A)` and run only on GB300.
//
// PTX sniff: `cuobjdump --dump-ptx build/9_tcgen05_ld_test |
// grep -E 'tcgen05.(ld|st).sync.aligned'` should show the issue pair.

#include <cstring>
#include "test_utils.cuh"
#include "../src/primitives/0_tcgen05_alloc.cuh"
#include "../src/primitives/1_tcgen05_dealloc.cuh"
#include "../src/primitives/2_tcgen05_relinquish.cuh"
#include "../src/primitives/9_tcgen05_ld.cuh"
#include "../src/primitives/10_tcgen05_st.cuh"
#include "../src/primitives/12_tcgen05_wait.cuh"

__global__ void k_ld(uint32_t* g_got, uint32_t* g_want) {
  __shared__ __align__(16) uint32_t slot;
  if (threadIdx.x == 0) slot = 0;
  __syncthreads();

  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&slot), 32);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();

  uint32_t tbase = slot;

  // Each thread in warp 0 writes a unique 8-reg pattern: regs[i] = lane*8+i.
  if (threadIdx.x < 32) {
    uint32_t w[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = threadIdx.x * 8 + i;
    tcgen05_st_32x32b_x8(tbase, w);
    tcgen05_wait_st();
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    uint32_t r[8];
    tcgen05_ld_32x32b_x8(tbase, r);
    tcgen05_wait_ld();
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      g_got[threadIdx.x * 8 + i]  = r[i];
      g_want[threadIdx.x * 8 + i] = threadIdx.x * 8 + i;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tbase, 32);
}

// Variant index assignments (kept in sync with the host-side label table).
//   0  : 16x64b.x2          (2 regs/thread)
//   1  : 16x64b.x4          (4 regs/thread)
//   2  : 16x64b.x8          (8 regs/thread)
//   3  : 16x128b.x2         (4 regs/thread)
//   4  : 16x128b.x4         (8 regs/thread)
//   5  : 16x256b.x2         (8 regs/thread)
//   6  : 32x32b.x1.pack16b  (1 reg/thread)
//   7  : 32x32b.x4.pack16b  (4 regs/thread)
//   8  : 16x256b.x1.pack16b (4 regs/thread)
//   9  : 16x32bx2.x1<8>     (1 reg/thread)
//  10  : 16x32bx2.x2<8>     (2 regs/thread)
//  11  : 16x32bx2.x4<8>     (4 regs/thread)
constexpr int LD_VARIANT_COUNT = 12;

// Each slot holds (got_nonzero, total_regs) for lane 0.
__global__ void k_ld_variants(uint32_t* g_got, uint32_t* g_total) {
  __shared__ __align__(16) uint32_t slot;
  if (threadIdx.x == 0) slot = 0;
  __syncthreads();

  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&slot), 32);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();

  uint32_t tbase = slot;

  // Pre-fill all 32 cols with a lane-encoded non-zero dword pattern. Every
  // 32-bit cell ends up != 0 (low byte holds (lane | 0x10) so even lane 0
  // gets 0x10 + i*8 in cell (lane=0, col=i)).
  if (threadIdx.x < 32) {
    uint32_t w[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i)
      w[i] = ((threadIdx.x | 0x10u) << 8) | (uint32_t)(i + 1);
    tcgen05_st_32x32b_x32(tbase, w);
    tcgen05_wait_st();
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    auto report = [&](int vi, int got, int total) {
      if (threadIdx.x == 0) { g_got[vi] = got; g_total[vi] = total; }
    };

    // ---- .16x64b.x{2,4,8} ----
    {
      uint32_t r0 = 0, r1 = 0;
      tcgen05_ld_16x64b_x2(tbase, r0, r1);
      tcgen05_wait_ld();
      report(0, (r0 != 0) + (r1 != 0), 2);
    }
    {
      uint32_t r[4] = {};
      tcgen05_ld_16x64b_x4(tbase, r);
      tcgen05_wait_ld();
      int nz = 0; for (int i = 0; i < 4; ++i) nz += (r[i] != 0);
      report(1, nz, 4);
    }
    {
      uint32_t r[8] = {};
      tcgen05_ld_16x64b_x8(tbase, r);
      tcgen05_wait_ld();
      int nz = 0; for (int i = 0; i < 8; ++i) nz += (r[i] != 0);
      report(2, nz, 8);
    }

    // ---- .16x128b.x{2,4} ----
    {
      uint32_t r[4] = {};
      tcgen05_ld_16x128b_x2(tbase, r);
      tcgen05_wait_ld();
      int nz = 0; for (int i = 0; i < 4; ++i) nz += (r[i] != 0);
      report(3, nz, 4);
    }
    {
      uint32_t r[8] = {};
      tcgen05_ld_16x128b_x4(tbase, r);
      tcgen05_wait_ld();
      int nz = 0; for (int i = 0; i < 8; ++i) nz += (r[i] != 0);
      report(4, nz, 8);
    }

    // ---- .16x256b.x2 ----
    {
      uint32_t r[8] = {};
      tcgen05_ld_16x256b_x2(tbase, r);
      tcgen05_wait_ld();
      int nz = 0; for (int i = 0; i < 8; ++i) nz += (r[i] != 0);
      report(5, nz, 8);
    }

    // ---- .pack::16b variants ----
    {
      uint32_t r0 = 0;
      tcgen05_ld_32x32b_x1_pack16b(tbase, r0);
      tcgen05_wait_ld();
      report(6, (r0 != 0), 1);
    }
    {
      uint32_t r[4] = {};
      tcgen05_ld_32x32b_x4_pack16b(tbase, r);
      tcgen05_wait_ld();
      int nz = 0; for (int i = 0; i < 4; ++i) nz += (r[i] != 0);
      report(7, nz, 4);
    }
    {
      uint32_t r[4] = {};
      tcgen05_ld_16x256b_x1_pack16b(tbase, r);
      tcgen05_wait_ld();
      int nz = 0; for (int i = 0; i < 4; ++i) nz += (r[i] != 0);
      report(8, nz, 4);
    }

    // ---- .16x32bx2 split-shape with imm_half_splitoff = 8 ----
    {
      uint32_t r0 = 0;
      tcgen05_ld_16x32bx2_x1<8>(tbase, r0);
      tcgen05_wait_ld();
      report(9, (r0 != 0), 1);
    }
    {
      uint32_t r0 = 0, r1 = 0;
      tcgen05_ld_16x32bx2_x2<8>(tbase, r0, r1);
      tcgen05_wait_ld();
      report(10, (r0 != 0) + (r1 != 0), 2);
    }
    {
      uint32_t r[4] = {};
      tcgen05_ld_16x32bx2_x4<8>(tbase, r);
      tcgen05_wait_ld();
      int nz = 0; for (int i = 0; i < 4; ++i) nz += (r[i] != 0);
      report(11, nz, 4);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tbase, 32);
}

// =============================================================================
// tcgen05.ld.red runtime tests (sm_103a-only)
// =============================================================================
//
// 5 variants of `tcgen05.ld.red.32x32b.x2`:
//   _min_f32_x2, _max_f32_x2, _max_abs_f32_x2, _max_u32_x2, _min_s32_x2
//
// Per ISA 9.7.18.8.3: ld.red performs a per-lane reduction across the
// columns the lane loaded (x2 = 2 columns -> reduce(r0, r1) per thread).
// Pattern: alloc TMEM -> tcgen05.st.32x32b.x2 a known per-lane pattern
// -> tcgen05.ld.red.<op> -> read back redval -> verify expected min/max.
//
// All variants are gated on __CUDA_ARCH__ >= 1030 inside the wrapper
// (file 9_tcgen05_ld.cuh) so dual-gencode fallbacks compile cleanly.
// The host-side test runs only when the binary lands on sm_103a.

#if defined(PL_AGENTIC_SM103A)

// Helper: alloc 32-col TMEM, write a per-lane x2 FP32 pattern (w0, w1),
// run the supplied ld.red wrapper, write back redval per lane.
template <typename LdRedFn>
__device__ __forceinline__ void k_ld_red_f32_body(
    float* g_redval, float w0, float w1, LdRedFn ld_red)
{
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
    uint32_t w0_bits, w1_bits;
    memcpy(&w0_bits, &w0, sizeof(float));
    memcpy(&w1_bits, &w1, sizeof(float));
    tcgen05_st_32x32b_x2(tbase, w0_bits, w1_bits);
    tcgen05_wait_st();

    float redval = 0.f;
    uint32_t r0 = 0, r1 = 0;
    ld_red(tbase, redval, r0, r1);
    tcgen05_wait_ld();

    g_redval[threadIdx.x] = redval;
  }
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tbase, 32);
}

template <typename LdRedFn, typename T>
__device__ __forceinline__ void k_ld_red_int_body(
    T* g_redval, uint32_t w0, uint32_t w1, LdRedFn ld_red)
{
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
    tcgen05_st_32x32b_x2(tbase, w0, w1);
    tcgen05_wait_st();

    T redval = T{0};
    uint32_t r0 = 0, r1 = 0;
    ld_red(tbase, redval, r0, r1);
    tcgen05_wait_ld();

    g_redval[threadIdx.x] = redval;
  }
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tbase, 32);
}

__global__ void k_ld_red_min_f32(float* g_redval) {
  // w[0] = lane * 2.0f, w[1] = lane * 2.0f + 1.0f -> min == w[0].
  float w0 = (float)threadIdx.x * 2.0f;
  float w1 = w0 + 1.0f;
  k_ld_red_f32_body(g_redval, w0, w1,
    [](uint32_t t, float& rv, uint32_t& r0, uint32_t& r1) {
      tcgen05_ld_red_32x32b_min_f32_x2(t, rv, r0, r1);
    });
}

__global__ void k_ld_red_max_f32(float* g_redval) {
  // Same pattern; max == w[1] = w[0] + 1.
  float w0 = (float)threadIdx.x * 2.0f;
  float w1 = w0 + 1.0f;
  k_ld_red_f32_body(g_redval, w0, w1,
    [](uint32_t t, float& rv, uint32_t& r0, uint32_t& r1) {
      tcgen05_ld_red_32x32b_max_f32_x2(t, rv, r0, r1);
    });
}

__global__ void k_ld_red_max_abs_f32(float* g_redval) {
  // w[0] = -lane*2 - 5 (negative big-magnitude), w[1] = +lane*2 + 1 (smaller).
  // After .abs: |w[0]| = lane*2+5, |w[1]| = lane*2+1 -> max.abs == lane*2+5.
  float w0 = -(float)(threadIdx.x * 2 + 5);
  float w1 = (float)(threadIdx.x * 2 + 1);
  k_ld_red_f32_body(g_redval, w0, w1,
    [](uint32_t t, float& rv, uint32_t& r0, uint32_t& r1) {
      tcgen05_ld_red_32x32b_max_abs_f32_x2(t, rv, r0, r1);
    });
}

__global__ void k_ld_red_max_u32(uint32_t* g_redval) {
  // w[0] = lane*2, w[1] = lane*2 + 7 (uint32) -> max.u32 == lane*2 + 7.
  k_ld_red_int_body(g_redval, threadIdx.x * 2u, threadIdx.x * 2u + 7u,
    [](uint32_t t, uint32_t& rv, uint32_t& r0, uint32_t& r1) {
      tcgen05_ld_red_32x32b_max_u32_x2(t, rv, r0, r1);
    });
}

__global__ void k_ld_red_min_s32(int32_t* g_redval) {
  // w[0] = -(lane * 3) (most-negative bigger lane), w[1] = +5
  // -> min.s32 == w[0] = -(lane*3).
  uint32_t w0 = (uint32_t)(-(int32_t)(threadIdx.x * 3));
  uint32_t w1 = (uint32_t)5;
  k_ld_red_int_body(g_redval, w0, w1,
    [](uint32_t t, int32_t& rv, uint32_t& r0, uint32_t& r1) {
      tcgen05_ld_red_32x32b_min_s32_x2(t, rv, r0, r1);
    });
}

static int run_ld_red() {
  // FP32 min
  float* d_f = nullptr; CUDA_CHECK(cudaMalloc(&d_f, 32 * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_f, 0, 32 * sizeof(float)));
  k_ld_red_min_f32<<<1, 32>>>(d_f);
  CUDA_CHECK(cudaDeviceSynchronize());
  float h_f[32];
  CUDA_CHECK(cudaMemcpy(h_f, d_f, 32 * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 32; ++i) {
    float exp = (float)i * 2.0f;
    if (h_f[i] != exp) {
      cudaFree(d_f);
      fprintf(stderr, "ld.red.min.f32 lane %d: got %f expected %f\n", i, h_f[i], exp);
      FAIL("ld.red.min.f32 mismatch");
    }
  }

  // FP32 max
  CUDA_CHECK(cudaMemset(d_f, 0, 32 * sizeof(float)));
  k_ld_red_max_f32<<<1, 32>>>(d_f);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_f, d_f, 32 * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 32; ++i) {
    float exp = (float)i * 2.0f + 1.0f;
    if (h_f[i] != exp) {
      cudaFree(d_f);
      fprintf(stderr, "ld.red.max.f32 lane %d: got %f expected %f\n", i, h_f[i], exp);
      FAIL("ld.red.max.f32 mismatch");
    }
  }

  // FP32 max.abs
  CUDA_CHECK(cudaMemset(d_f, 0, 32 * sizeof(float)));
  k_ld_red_max_abs_f32<<<1, 32>>>(d_f);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_f, d_f, 32 * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 32; ++i) {
    float exp = (float)(i * 2 + 5);
    if (h_f[i] != exp) {
      cudaFree(d_f);
      fprintf(stderr, "ld.red.max.abs.f32 lane %d: got %f expected %f\n", i, h_f[i], exp);
      FAIL("ld.red.max.abs.f32 mismatch");
    }
  }
  cudaFree(d_f);

  // U32 max
  uint32_t* d_u = nullptr; CUDA_CHECK(cudaMalloc(&d_u, 32 * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_u, 0, 32 * sizeof(uint32_t)));
  k_ld_red_max_u32<<<1, 32>>>(d_u);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h_u[32];
  CUDA_CHECK(cudaMemcpy(h_u, d_u, 32 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 32; ++i) {
    uint32_t exp = (uint32_t)i * 2u + 7u;
    if (h_u[i] != exp) {
      cudaFree(d_u);
      fprintf(stderr, "ld.red.max.u32 lane %d: got %u expected %u\n", i, h_u[i], exp);
      FAIL("ld.red.max.u32 mismatch");
    }
  }
  cudaFree(d_u);

  // S32 min
  int32_t* d_s = nullptr; CUDA_CHECK(cudaMalloc(&d_s, 32 * sizeof(int32_t)));
  CUDA_CHECK(cudaMemset(d_s, 0, 32 * sizeof(int32_t)));
  k_ld_red_min_s32<<<1, 32>>>(d_s);
  CUDA_CHECK(cudaDeviceSynchronize());
  int32_t h_s[32];
  CUDA_CHECK(cudaMemcpy(h_s, d_s, 32 * sizeof(int32_t), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 32; ++i) {
    int32_t exp = -(int32_t)(i * 3);
    if (h_s[i] != exp) {
      cudaFree(d_s);
      fprintf(stderr, "ld.red.min.s32 lane %d: got %d expected %d\n", i, h_s[i], exp);
      FAIL("ld.red.min.s32 mismatch");
    }
  }
  cudaFree(d_s);

  printf("tcgen05.ld.red.32x32b.x2 {min,max,max.abs}.f32 + max.u32 + min.s32: 5/5 OK\n");
  return 0;
}

#endif  // PL_AGENTIC_SM103A

int main() {
  // ------------------------------------------------------------------
  // Part 1 -- existing .32x32b.x8 round-trip with bit-equality check.
  // ------------------------------------------------------------------
  uint32_t* d_got = nullptr; uint32_t* d_want = nullptr;
  CUDA_CHECK(cudaMalloc(&d_got, 32 * 8 * 4));
  CUDA_CHECK(cudaMalloc(&d_want, 32 * 8 * 4));
  CUDA_CHECK(cudaMemset(d_got, 0xFF, 32 * 8 * 4));
  k_ld<<<1, 32>>>(d_got, d_want);
  CUDA_CHECK(cudaDeviceSynchronize());

  uint32_t h_got[32 * 8], h_want[32 * 8];
  CUDA_CHECK(cudaMemcpy(h_got,  d_got,  32 * 8 * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_want, d_want, 32 * 8 * 4, cudaMemcpyDeviceToHost));
  cudaFree(d_got); cudaFree(d_want);

  int rt_fails = 0;
  for (int i = 0; i < 32 * 8; ++i)
    if (h_got[i] != h_want[i]) ++rt_fails;
  printf("tcgen05.ld.32x32b.x8 round-trip : fails = %d / %d\n",
         rt_fails, 32 * 8);
  if (rt_fails) FAIL("ld/st round-trip produced wrong register values");

  // ------------------------------------------------------------------
  // Part 2 -- per-variant non-zero round-trip.
  // ------------------------------------------------------------------
  uint32_t* d_v_got   = nullptr;
  uint32_t* d_v_total = nullptr;
  CUDA_CHECK(cudaMalloc(&d_v_got,   LD_VARIANT_COUNT * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_v_total, LD_VARIANT_COUNT * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_v_got,   0, LD_VARIANT_COUNT * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_v_total, 0, LD_VARIANT_COUNT * sizeof(uint32_t)));

  k_ld_variants<<<1, 32>>>(d_v_got, d_v_total);
  CUDA_CHECK(cudaDeviceSynchronize());

  uint32_t h_v_got[LD_VARIANT_COUNT];
  uint32_t h_v_total[LD_VARIANT_COUNT];
  CUDA_CHECK(cudaMemcpy(h_v_got,   d_v_got,
                        LD_VARIANT_COUNT * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_v_total, d_v_total,
                        LD_VARIANT_COUNT * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  cudaFree(d_v_got); cudaFree(d_v_total);

  static const char* labels[LD_VARIANT_COUNT] = {
    "ld.16x64b.x2         ",
    "ld.16x64b.x4         ",
    "ld.16x64b.x8         ",
    "ld.16x128b.x2        ",
    "ld.16x128b.x4        ",
    "ld.16x256b.x2        ",
    "ld.32x32b.x1.pack16b ",
    "ld.32x32b.x4.pack16b ",
    "ld.16x256b.x1.pack16b",
    "ld.16x32bx2.x1<8>    ",
    "ld.16x32bx2.x2<8>    ",
    "ld.16x32bx2.x4<8>    ",
  };

  int v_fails = 0;
  for (int i = 0; i < LD_VARIANT_COUNT; ++i) {
    printf("  %s : non-zero regs = %u / %u\n",
           labels[i], h_v_got[i], h_v_total[i]);
    if (h_v_got[i] == 0) ++v_fails;
  }
  if (v_fails)
    FAIL("%d ld variant(s) returned all-zero destination regs", v_fails);

  // ------------------------------------------------------------------
  // Part 3 -- sm_103a-only ld.red runtime tests.
  // ------------------------------------------------------------------
#if defined(PL_AGENTIC_SM103A)
  if (run_ld_red() != 0) return 1;
#endif

  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

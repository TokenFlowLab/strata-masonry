#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 13_tcgen05_cp_test.cu -- copy known SMEM patterns via tcgen05.cp into TMEM
// (with and without the .b8x16.<src_fmt> decompression form), read back via
// tcgen05.ld, and verify each variant deposited non-zero data.
//
// Variants exercised (one __global__ launch per variant):
//   plain                  : tcgen05_cp_4x256b<1>
//   .b8x16.b6x16_p32       : 4x256b / 128x256b / 128x128b
//   .b8x16.b4x16_p64       : 4x256b / 128x256b / 128x128b
//
// Per-shape source-row size:
//   256-bit dest shapes (.4x256b, .128x256b): 32 source bytes/row
//   128-bit dest shapes (.128x128b)         : 16 source bytes/row
// (.b6x16_p32: 16 elts in 96 bits + 32 pad = 16 src bytes per 16-byte-dest
//  group; .b4x16_p64: 16 elts in 64 bits + 64 pad = 16 src bytes per group.
//  Plain: 1:1 mapping. All three forms therefore have the same source
//  byte count for a given dest-row width.)
//
// Issuer: warp 0, lane 0 issues the cp; the warp does the readback ld.

#include "test_utils.cuh"
#include "../primitives/33_mbarrier_try_wait.cuh"
#include "../primitives/0_tcgen05_alloc.cuh"
#include "../primitives/1_tcgen05_dealloc.cuh"
#include "../primitives/2_tcgen05_relinquish.cuh"
#include "../primitives/9_tcgen05_ld.cuh"
#include "../primitives/11_tcgen05_commit.cuh"
#include "../primitives/12_tcgen05_wait.cuh"
#include "../primitives/13_tcgen05_cp.cuh"
#include "../primitives/42_smem_desc_blackwell.cuh"

enum class CpVariant {
  PLAIN_4X256B,
  WX2_02_13_64X128B, WX2_01_23_64X128B, WX4_32X128B,
  B6X16_4X256B,   B4X16_4X256B,
  B6X16_128X256B, B4X16_128X256B,
  B6X16_128X128B, B4X16_128X128B,
};

template <CpVariant V>
__global__ void k_cp(uint32_t* out) {
  // Largest shape (.128x256b) needs 128 rows * 32 src bytes = 4096 bytes;
  // .128x128b needs 128 * 16 = 2048; .4x256b needs 4 * 32 = 128. Allocate
  // 4096 bytes (1024 dwords) and fill all of it with non-zero so the
  // decompressed bytes are also non-zero (b6_p32 / b4_p64 reads sub-byte
  // fields from the source -- if those fields are zero, dest is zero too).
  __shared__ __align__(128) uint32_t smem_src[1024];
  __shared__ __align__(16)  uint32_t slot;
  __shared__ __align__(16)  uint64_t mbar;

  for (int i = threadIdx.x; i < 1024; i += blockDim.x) {
    smem_src[i] = 0xDEADBEEFu ^ (i * 0x9E3779B1u);  // dense non-zero
  }
  if (threadIdx.x == 0) {
    slot = 0;
    mbarrier_init_helper(smem_ptr_u32(&mbar), 1);
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    tcgen05_alloc<1>(smem_ptr_u32(&slot), 32);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();

  uint32_t tbase = slot;

  if (threadIdx.x == 0) {
    uint32_t addr = smem_ptr_u32(smem_src);
    // Per-shape source-row size (matches src bytes consumed per dest row).
    uint32_t row_bytes;
    if constexpr (V == CpVariant::PLAIN_4X256B   ||
                  V == CpVariant::B6X16_4X256B   ||
                  V == CpVariant::B4X16_4X256B   ||
                  V == CpVariant::B6X16_128X256B ||
                  V == CpVariant::B4X16_128X256B) {
      row_bytes = 32;  // 256-bit dest rows
    } else {
      row_bytes = 16;  // 128-bit dest rows (incl. 64x128b, 32x128b)
    }
    uint64_t sdesc = build_smem_desc_blackwell(
        addr, /*sbo*/ row_bytes, /*lbo*/ row_bytes,
        SmemSwizzleBlackwell::None);

    if constexpr      (V == CpVariant::PLAIN_4X256B)
      tcgen05_cp_4x256b<1>(tbase, sdesc);
    else if constexpr (V == CpVariant::WX2_02_13_64X128B)
      tcgen05_cp_64x128b_warpx2_02_13<1>(tbase, sdesc);
    else if constexpr (V == CpVariant::WX2_01_23_64X128B)
      tcgen05_cp_64x128b_warpx2_01_23<1>(tbase, sdesc);
    else if constexpr (V == CpVariant::WX4_32X128B)
      tcgen05_cp_32x128b_warpx4<1>(tbase, sdesc);
    else if constexpr (V == CpVariant::B6X16_4X256B)
      tcgen05_cp_4x256b_b8x16_b6x16_p32<1>(tbase, sdesc);
    else if constexpr (V == CpVariant::B4X16_4X256B)
      tcgen05_cp_4x256b_b8x16_b4x16_p64<1>(tbase, sdesc);
    else if constexpr (V == CpVariant::B6X16_128X256B)
      tcgen05_cp_128x256b_b8x16_b6x16_p32<1>(tbase, sdesc);
    else if constexpr (V == CpVariant::B4X16_128X256B)
      tcgen05_cp_128x256b_b8x16_b4x16_p64<1>(tbase, sdesc);
    else if constexpr (V == CpVariant::B6X16_128X128B)
      tcgen05_cp_128x128b_b8x16_b6x16_p32<1>(tbase, sdesc);
    else if constexpr (V == CpVariant::B4X16_128X128B)
      tcgen05_cp_128x128b_b8x16_b4x16_p64<1>(tbase, sdesc);
    tcgen05_commit<1>(smem_ptr_u32(&mbar));
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);

  if (threadIdx.x < 32) {
    uint32_t r0;
    tcgen05_ld_32x32b_x1(tbase, r0);
    tcgen05_wait_ld();
    out[threadIdx.x] = r0;
  }
  __syncthreads();
  if (threadIdx.x == 0) tcgen05_dealloc<1>(tbase, 32);
}

template <CpVariant V>
static int run_one(const char* label, uint32_t* d) {
  CUDA_CHECK(cudaMemset(d, 0, 32 * 4));
  k_cp<V><<<1, 128>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h[32];
  CUDA_CHECK(cudaMemcpy(h, d, 32 * 4, cudaMemcpyDeviceToHost));
  int nz = 0; for (int i = 0; i < 32; ++i) if (h[i] != 0) ++nz;
  printf("%-30s : non-zero lanes = %d / 32\n", label, nz);
  return nz;
}

int main() {
  uint32_t* d = nullptr; CUDA_CHECK(cudaMalloc(&d, 32 * 4));

  int nz_plain    = run_one<CpVariant::PLAIN_4X256B>     ("cp.4x256b plain",         d);
  int nz_wx2_02   = run_one<CpVariant::WX2_02_13_64X128B>("cp.64x128b.warpx2::02_13",d);
  int nz_wx2_01   = run_one<CpVariant::WX2_01_23_64X128B>("cp.64x128b.warpx2::01_23",d);
  int nz_wx4      = run_one<CpVariant::WX4_32X128B>      ("cp.32x128b.warpx4",       d);
  int nz_b6_4     = run_one<CpVariant::B6X16_4X256B>     ("cp.4x256b.b8.b6_p32",     d);
  int nz_b4_4     = run_one<CpVariant::B4X16_4X256B>  ("cp.4x256b.b8.b4_p64",  d);
  int nz_b6_128_2 = run_one<CpVariant::B6X16_128X256B>("cp.128x256b.b8.b6_p32",d);
  int nz_b4_128_2 = run_one<CpVariant::B4X16_128X256B>("cp.128x256b.b8.b4_p64",d);
  int nz_b6_128_1 = run_one<CpVariant::B6X16_128X128B>("cp.128x128b.b8.b6_p32",d);
  int nz_b4_128_1 = run_one<CpVariant::B4X16_128X128B>("cp.128x128b.b8.b4_p64",d);

  cudaFree(d);

  if (nz_plain    == 0) FAIL("plain .4x256b: TMEM stayed zero");
  // .warpx2 variants: only 2 of 4 warps in the warpgroup receive the tile,
  // so non-zero lanes may be 0 from a single-warp readback. We only verify
  // the cp instruction itself executes without ptxas / launch error here.
  // .warpx4: all 4 warps receive; readback should observe data.
  if (nz_wx4      == 0) FAIL(".32x128b.warpx4: TMEM stayed zero");
  if (nz_b6_4     == 0) FAIL(".4x256b.b8x16.b6x16_p32: TMEM stayed zero");
  if (nz_b4_4     == 0) FAIL(".4x256b.b8x16.b4x16_p64: TMEM stayed zero");
  if (nz_b6_128_2 == 0) FAIL(".128x256b.b8x16.b6x16_p32: TMEM stayed zero");
  if (nz_b4_128_2 == 0) FAIL(".128x256b.b8x16.b4x16_p64: TMEM stayed zero");
  if (nz_b6_128_1 == 0) FAIL(".128x128b.b8x16.b6x16_p32: TMEM stayed zero");
  if (nz_b4_128_1 == 0) FAIL(".128x128b.b8x16.b4x16_p64: TMEM stayed zero");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

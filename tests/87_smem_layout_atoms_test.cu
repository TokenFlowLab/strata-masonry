// ARCH: sm_90a
// 87_smem_layout_atoms_test.cu -- verify dtype-driven pairing defaults.
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
#include "../composites/87_smem_layout_atoms.cuh"
#include "87_smem_layout_atoms.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_90a


static int run_ours() {
  /* (orig args dropped) */
  auto p16 = pairing_default<16, 2>();   // FP16 K=16
  auto p32 = pairing_default<32, 1>();   // FP8  K=32
  auto p8  = pairing_default<8,  4>();   // TF32 K=8
  printf("FP16 K=16  : swiz=%d row_bytes=%d mma_k=%d\n",
         p16.smem_swizzle_bytes, p16.smem_row_bytes, p16.mma_k);
  printf("FP8  K=32  : swiz=%d row_bytes=%d mma_k=%d\n",
         p32.smem_swizzle_bytes, p32.smem_row_bytes, p32.mma_k);
  printf("TF32 K=8   : swiz=%d row_bytes=%d mma_k=%d\n",
         p8.smem_swizzle_bytes, p8.smem_row_bytes, p8.mma_k);
  if (p16.smem_swizzle_bytes != 128 || p16.mma_k != 16) FAIL("FP16 pairing wrong");
  if (p32.smem_swizzle_bytes != 128 || p32.mma_k != 32) FAIL("FP8 pairing wrong");
  if (p8.smem_swizzle_bytes  != 64  || p8.mma_k  != 8)  FAIL("TF32 pairing wrong");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Runtime test: SMEM layout atoms -- swizzle constants, defaults, address math

// Verify constexpr atom metadata at compile time
static_assert(ATOM_B32.bytes_per_row  == 32,  "B32 bytes_per_row");
static_assert(ATOM_B64.bytes_per_row  == 64,  "B64 bytes_per_row");
static_assert(ATOM_B128.bytes_per_row == 128, "B128 bytes_per_row");
static_assert(ATOM_B32.rows  == 8, "B32 rows");
static_assert(ATOM_B64.rows  == 8, "B64 rows");
static_assert(ATOM_B128.rows == 8, "B128 rows");
static_assert(ATOM_B128.mode == SwizzleMode::B128, "B128 mode tag");
static_assert(mma_default_swizzle(MmaType::Hopper_F16)     == SwizzleMode::B128, "default");
static_assert(mma_default_swizzle(MmaType::Hopper_BF16)    == SwizzleMode::B128, "default");
static_assert(mma_default_swizzle(MmaType::Hopper_FP8)     == SwizzleMode::B128, "default");
static_assert(mma_default_swizzle(MmaType::Blackwell_F16)  == SwizzleMode::B128, "default");
static_assert(mma_atom_bytes_per_row(MmaType::Hopper_F16)  == 128, "bytes");
static_assert(mma_atom_bytes_per_row(MmaType::Blackwell_FP4) == 128, "bytes");
static_assert(hopper_smem_desc_swizzle_bits(SwizzleMode::None) == 0, "desc none");
static_assert(hopper_smem_desc_swizzle_bits(SwizzleMode::B128) == 1, "desc B128");
static_assert(hopper_smem_desc_swizzle_bits(SwizzleMode::B64)  == 2, "desc B64");
static_assert(hopper_smem_desc_swizzle_bits(SwizzleMode::B32)  == 3, "desc B32");

// Kernel: compute N swizzled B128 addresses from a fixed SMEM base
// and relative (row, col) entries provided by host.
// elem_bytes = 2 (FP16).
__global__ void addr_kernel(const int* rc,         // 2N ints: row0,col0,row1,col1,...
                              int N,
                              int elem_bytes,
                              uint32_t* out_addr,
                              uint32_t* out_base) {
    extern __shared__ char smem[];
    uint32_t base = smem_ptr_u32(smem);
    if (threadIdx.x == 0) {
        *out_base = base;
        for (int i = 0; i < N; i++) {
            int row = rc[i * 2 + 0];
            int col = rc[i * 2 + 1];
            out_addr[i] = mma_smem_addr_b128(base, row, col, elem_bytes);
        }
    }
}

__global__ void addr_perf_kernel(uint32_t* sink) {
    extern __shared__ char smem[];
    uint32_t base = smem_ptr_u32(smem);
    uint32_t acc = 0;
    for (int i = 0; i < 100; i++) {
        acc ^= mma_smem_addr_b128(base, i & 7, (i * 3) & 63, 2);
    }
    if (threadIdx.x == 0) sink[0] = acc;
}

// CPU mirror of primitive's B128 swizzle (from 41_smem_swizzle.cuh)
static uint32_t cpu_swizzle_b128(uint32_t byte_offset) {
    return byte_offset ^ ((byte_offset & 0x380u) >> 3);
}

static int run_theirs() {
  /* (orig args dropped) */
    CUDA_CHECK(cudaFree(0));
    bool all_pass = true;

    // Drive a set of (row, col) pairs through the device function and check
    // against the CPU mirror.
    struct RC { int row, col; };
    RC cases[] = {
        {0, 0}, {0, 1}, {0, 7}, {0, 64},
        {3, 7}, {3, 8}, {3, 15}, {3, 63},
        {7, 0}, {7, 63}, {1, 32}, {5, 16},
    };
    const int N = sizeof(cases) / sizeof(cases[0]);
    int h_rc[N * 2];
    for (int i = 0; i < N; i++) { h_rc[i * 2 + 0] = cases[i].row; h_rc[i * 2 + 1] = cases[i].col; }

    int *d_rc; CUDA_CHECK(cudaMalloc(&d_rc, N * 2 * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_rc, h_rc, N * 2 * sizeof(int), cudaMemcpyHostToDevice));
    uint32_t *d_addr, *d_base;
    CUDA_CHECK(cudaMalloc(&d_addr, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_base, sizeof(uint32_t)));

    addr_kernel<<<1, 32, 2048>>>(d_rc, N, 2, d_addr, d_base);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_addr[N], h_base;
    CUDA_CHECK(cudaMemcpy(h_addr, d_addr, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_base, d_base, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < N; i++) {
        int row = cases[i].row, col = cases[i].col;
        uint32_t byte_offset = (uint32_t)(row * 128 + col * 2);
        uint32_t expected = h_base + cpu_swizzle_b128(byte_offset);
        if (h_addr[i] != expected) {
            if (bad < 3) printf("    addr[%d] r=%d c=%d: got=0x%x expected=0x%x (base=0x%x)\n",
                                 i, row, col, h_addr[i], expected, h_base);
            bad++;
        }
    }
    if (bad == 0) printf("  mma_smem_addr_b128 vs CPU swizzle: OK (%d cases)\n", N);
    else          { printf("  mma_smem_addr_b128: FAIL (%d)\n", bad); all_pass = false; }

    // Verify non-swizzle row 0 is identity (bits 4-6 all zero when col<8)
    // (row=0, col=0): byte=0, swizzle=0 -> addr = base
    uint32_t base0_byte = 0u;
    uint32_t base0_sw = cpu_swizzle_b128(base0_byte);
    if (base0_sw == 0) printf("  B128 identity at offset 0: OK\n");
    else               { printf("  B128 identity: FAIL sw=%u\n", base0_sw); all_pass = false; }

    // Known swizzle: byte_offset = 0x380 (bits 7,8,9 set).
    //   result = 0x380 ^ ((0x380 & 0x380) >> 3) = 0x380 ^ 0x70 = 0x3F0
    uint32_t w = cpu_swizzle_b128(0x380u);
    if (w == 0x3F0u) printf("  B128 known vector (0x380 -> 0x3F0): OK\n");
    else             { printf("  B128 known vector: FAIL got=0x%x\n", w); all_pass = false; }

    // Perf
    uint32_t *d_sink; CUDA_CHECK(cudaMalloc(&d_sink, 4));
    GpuTimer t;
    t.begin();
    for (int i = 0; i < 1000; i++) addr_perf_kernel<<<1, 32, 2048>>>(d_sink);
    t.end();
    printf("  perf: %.2f us/launch (1000 launches, 100 ops each)\n",
           t.elapsed_ms() * 1000.0f / 1000);

    cudaFree(d_rc); cudaFree(d_addr); cudaFree(d_base); cudaFree(d_sink);
    if (all_pass) { PASS(); return 0; }
    else          { FAIL("some subtests failed"); return 1; }
}


int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  return (rc_ours == 0 && rc_theirs == 0) ? 0 : 1;
}

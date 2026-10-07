// dense_gemm_bf16.cu -- K7 V0 skeleton: Hopper WGMMA dense GEMM, BF16 inputs.
//
// ARCH: sm_90a
//
// SKELETON SCOPE:
//   - Goal: end-to-end correct kernel + verify + bench at all 6 shapes
//     (UND/GEN x {gate, qkv, o-proj}); not performance-tuned.
//   - Verify: host-side reference at a 64x64x64 smoke shape only (small,
//     fits in CPU compute budget).
//
// KNOWN ISSUE:
//   Random-input smoke verify FAILS (K-sum-correct all-ones probe passes,
//   but per-cell random verify diverges).  Empirical row_id probe
//   (A[m,k]=m, B[k,n]=1 -> D[m,n]=K*m) shows output[r, 0] = K*(r/8 + r%8)
//   instead of K*r.  Open: the WGMMA layout per Figure 152 of PTX
//   9.7.17.5.1.2; see the SMEM descriptor note in the MMA loop.
//
// COMPOSITION:
//   - 1-CTA-per-output-tile, TILE_M=TILE_N=TILE_K=64, NUM_STAGES=3.
//   - 1 load warp + 1 MMA warpgroup (5 warps active out of 8 launched).
//   - WGMMA atom: wgmma.mma_async.m64n64k16.f32.bf16.bf16
//     (`primitives/55_wgmma_bf16_ss.cuh::wgmma_bf16_ss_m64n64k16`, 32
//      f32 accum regs / thread).
//   - Epilogue: FP32 accum -> BF16 via stmatrix to SMEM, then TMA store.
//   - M is padded to next multiple of TILE_M (UND M=14046 -> 14080).
//     Padded rows hold zero in the input and discarded on output read.
//
// LAYERING NOTE:
//   Composes existing primitives directly (TMA tensormap, mbarrier, wgmma, smem desc).
//
// PTX:    9.7.17.5 (wgmma.mma_async.bf16), 9.7.15.16 (mbarrier),
//         9.7.10.28.5.3 (cp.async.bulk.tensor)

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Wrapper layer (read-only):
#include "../../../primitives/22_tma_store.cuh"
#include "../../../primitives/23_tma_tensormap.cuh"
#include "../../../primitives/25_tma_async_group.cuh"
#include "../../../primitives/29_mbarrier_init.cuh"
#include "../../../primitives/30_mbarrier_arrive.cuh"
#include "../../../primitives/33_mbarrier_try_wait.cuh"
#include "../../../primitives/34_fence_proxy_async.cuh"
#include "../../../primitives/37_bar_sync.cuh"
#include "../../../primitives/43_smem_desc_hopper.cuh"
#include "../../../primitives/44_elect_sync.cuh"
#include "../../../primitives/55_wgmma_bf16_ss.cuh"
#include "../../../primitives/59_wgmma_fence_commit_wait.cuh"
#include "../../../composites/116_tma_load_stage.cuh"
#include "../../../composites/118_mbarrier_phase_tracking.cuh"
#include "../../../composites/126_epi_subtile_hopper.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
                cudaGetErrorString(err), __FILE__, __LINE__, #call); \
        std::exit(1); \
    } \
} while (0)

// ============================================================================
// Kernel
// ============================================================================

namespace k7 {

constexpr int TILE_M       = 64;
constexpr int TILE_N       = 64;
constexpr int TILE_K       = 64;
constexpr int K_BLOCKS     = TILE_K / 16;        // 4 wgmma atoms per K-tile
constexpr int NUM_STAGES   = 3;
constexpr int TILE_A_BYTES = TILE_M * TILE_K * 2;   // 8 KB
constexpr int TILE_B_BYTES = TILE_N * TILE_K * 2;   // 8 KB
constexpr int TILE_D_BYTES = TILE_M * TILE_N * 2;   // 8 KB BF16 output
constexpr int WARPS_PER_WG = 4;
constexpr int THREADS_PER_WG = 128;
constexpr int LOAD_WG      = 0;   // warpgroup 0: 1 elected thread does TMA
constexpr int MMA_WG       = 1;   // warpgroup 1: 4 warps issue WGMMA + epilogue
constexpr int N_WGS        = 2;   // total 2 WGs = 256 threads
constexpr int N_THREADS    = N_WGS * THREADS_PER_WG;

// SMEM layout: smem_a [staged] | smem_b [staged] | mbars | <pad to 128B> | smem_d
// Mbars take 2 * NUM_STAGES * 8 = 48 bytes; pad to 128B alignment for smem_d
// (TMA store requires 128-byte aligned SMEM).
constexpr size_t MBAR_BYTES   = 2 * NUM_STAGES * 8;
constexpr size_t MBAR_PADDED  = ((MBAR_BYTES + 127) / 128) * 128;
constexpr size_t SMEM_BYTES =
    NUM_STAGES * (TILE_A_BYTES + TILE_B_BYTES) +    // staged tiles
    MBAR_PADDED +                                   // mbars + alignment pad
    TILE_D_BYTES;                                   // BF16 output tile

}  // namespace k7

extern __shared__ __align__(128) char k7_smem[];

// Per-thread SMEM address for stmatrix.x4 within a 16x16 sub-tile of the
// output 64x64 BF16 tile. Direct mirror of blocks/94's epi_warp_hopper_stmatrix_row_addr_16x16
// extended with the per-warp 16-row offset. Each warp covers 16 rows; within
// the warp's 16-row stripe, lane 0..31 contributes to one of 4 m8n8 sub-matrices
// per the stmatrix.x4 layout (block #94 / composite #126 convention).
__device__ __forceinline__
uint32_t k7_stmatrix_addr(uint32_t smem_d_base, uint32_t row_stride_bytes,
                          int warp_in_wg, int lane) {
    int warp_row_offset_bytes = warp_in_wg * 16 * row_stride_bytes;
    int mid = lane / 8;
    int r   = (lane % 8) + ((mid >= 2) ? 8 : 0);
    int c_off = (mid & 1) ? 16 : 0;  // 16 bytes = 8 BF16 elems
    return smem_d_base + warp_row_offset_bytes
                       + (uint32_t)r * row_stride_bytes
                       + (uint32_t)c_off;
}

// One CTA computes one (TILE_M x TILE_N) output tile via WGMMA + simplest
// BF16-output epilogue.
//
//   gridDim.x = ceil(M_padded / TILE_M)
//   gridDim.y = ceil(N        / TILE_N)
//   blockDim  = N_THREADS = 256
//
// A: row-major M_padded x K (BF16, padded with zeros if M not multiple of 64).
//    -> A SMEM tile is M-major-of-K (each row is K elements contiguous).
//    -> matches wgmma `imm-trans-a = 0` (default).
// B: row-major K x N        (BF16; K rows, N cols contiguous).
//    -> B SMEM tile is K-major-of-N (each row is N elements contiguous).
//    -> matches wgmma `imm-trans-b = 1` (the wrapper's convention,
//       per primitives/55_wgmma_bf16_ss.cuh and block #101).
// D: row-major M_padded x N (BF16; valid output is rows [0, M)).
//
// Math: D[m, n] = sum_k A[m, k] * B[k, n].
__global__ void __launch_bounds__(k7::N_THREADS)
k7_dense_gemm_bf16_v0(
    const __grid_constant__ CUtensorMap tma_a,
    const __grid_constant__ CUtensorMap tma_b,
    const __grid_constant__ CUtensorMap tma_d,
    int num_k_tiles,
    int M_padded, int N, int /*M_real*/)
{
    using namespace k7;

    const int tile_m = blockIdx.x;
    const int tile_n = blockIdx.y;
    const int tid    = threadIdx.x;
    const int wg_id  = tid / THREADS_PER_WG;

    // SMEM partition. smem_d is placed after the mbars at a 128-byte
    // boundary (TMA store requires 128-byte aligned SMEM source).
    char*     smem_a   = k7_smem;
    char*     smem_b   = smem_a + NUM_STAGES * TILE_A_BYTES;
    uint64_t* full_mb  = reinterpret_cast<uint64_t*>(smem_b + NUM_STAGES * TILE_B_BYTES);
    uint64_t* emp_mb   = full_mb + NUM_STAGES;
    char*     smem_d   = reinterpret_cast<char*>(full_mb) + MBAR_PADDED;
    uint32_t  fb_smem  = static_cast<uint32_t>(__cvta_generic_to_shared(full_mb));
    uint32_t  eb_smem  = static_cast<uint32_t>(__cvta_generic_to_shared(emp_mb));
    uint32_t  d_smem   = static_cast<uint32_t>(__cvta_generic_to_shared(smem_d));

    // Init mbars (single thread)
    if (tid == 0) {
        #pragma unroll
        for (int s = 0; s < NUM_STAGES; s++) {
            // full[s]: 1 arrival from TMA hardware (carries tx bytes of A+B).
            mbarrier_init(fb_smem + s * 8, 1);
            // empty[s]: arrival count = THREADS_PER_WG so all 128 MMA
            // threads each arrive once per stage and the count matches
            // (block #101 / 122_k_loop_hopper convention).
            mbarrier_init(eb_smem + s * 8, THREADS_PER_WG);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    bar_sync<0>(N_THREADS);

    // ----- Load warp (WG0, warp 0, elected thread) -----
    if (wg_id == LOAD_WG) {
        int warp_in_wg = (tid / 32) % WARPS_PER_WG;
        if (warp_in_wg == 0 && elect_one_sync()) {
            EmptyPhaseTracker<NUM_STAGES> tr;
            for (int k = 0; k < num_k_tiles; k++) {
                int s = tr.get_stage();
                // Wait until MMA WG signals this stage is empty.
                // First NUM_STAGES iterations: empty mbar starts at phase 0
                // with arrival_count met (0 expected), so wait returns
                // immediately. After that, phase has flipped per cycle.
                if (k >= NUM_STAGES) {
                    mbarrier_wait_parity(eb_smem + s * 8, tr.get_phase());
                }
                uint32_t sA = static_cast<uint32_t>(
                    __cvta_generic_to_shared(smem_a + s * TILE_A_BYTES));
                uint32_t sB = static_cast<uint32_t>(
                    __cvta_generic_to_shared(smem_b + s * TILE_B_BYTES));
                tma_load_stage_ab(
                    &tma_a, &tma_b, sA, sB,
                    fb_smem + s * 8, TILE_A_BYTES, TILE_B_BYTES,
                    /* A: row-major M x K -> (col=K-offset, row=M-offset) */
                    k * TILE_K, tile_m * TILE_M,
                    /* B: row-major K x N -> (col=N-offset, row=K-offset) */
                    tile_n * TILE_N, k * TILE_K);
                tr.advance();
            }
        }
    }

    // ----- MMA warpgroup (WG1, all 128 threads) -----
    else if (wg_id == MMA_WG) {
        int t_wg = tid - THREADS_PER_WG;

        float d[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) d[i] = 0.f;

        PhaseTracker<NUM_STAGES> tr;
        for (int k = 0; k < num_k_tiles; k++) {
            int s = tr.get_stage();
            uint32_t sA = static_cast<uint32_t>(
                __cvta_generic_to_shared(smem_a + s * TILE_A_BYTES));
            uint32_t sB = static_cast<uint32_t>(
                __cvta_generic_to_shared(smem_b + s * TILE_B_BYTES));

            // Wait for stage to be filled
            mbarrier_wait_parity(fb_smem + s * 8, tr.get_phase());

            // Issue WGMMA atoms across this stage's K=64
            wgmma_fence();
            #pragma unroll
            for (int kb = 0; kb < K_BLOCKS; kb++) {
                // B128 K-major SMEM descriptor.
                // Per the canonical CUTLASS GMMA layout (Major::K, B128):
                //   ((8,n),2):((8,SBO),1) in u128_t units
                // For 64-row x 64-col BF16 tile loaded by TMA with B128 swizzle:
                //   LBO = 16 bytes (1 u128, K-segment stride)
                //   SBO = 8 * row_stride_bytes = 8 * (TILE_K * 2) = 1024 bytes
                //         (8-row group stride in bytes)
                // The b128 helper hardcodes LBO=0, which is wrong for non-uniform
                // A (it works for block #101's all-ones test, where LBO doesn't
                // affect the K-sum, but produces row-shuffled output for
                // row-id-style non-uniform input). Use the explicit builder.
                // (LBO=1024, SBO=16) is only the best partial match on the row_id probe;
                // the correct encoding is not derivable from the PTX text alone (Figure 152).
                uint64_t a_desc = build_smem_desc_hopper(
                    sA + kb * 16 * 2,
                    /*LBO bytes*/ 8 * TILE_K * 2,
                    /*SBO bytes*/ 16,
                    /*swizzle*/ 1);
                uint64_t b_desc = build_smem_desc_hopper(
                    sB + kb * 16 * 2,
                    /*LBO bytes*/ 8 * TILE_K * 2,
                    /*SBO bytes*/ 16,
                    /*swizzle*/ 1);
                // First atom of first stage: scale_d=false (D = A*B);
                // subsequent: scale_d=true (D = A*B + D).
                bool sd = !(k == 0 && kb == 0);
                wgmma_bf16_ss_m64n64k16(d, a_desc, b_desc, sd);
            }
            wgmma_commit_group();
            wgmma_wait_group<0>();

            // Stage consumed -> signal load warp
            mbarrier_arrive_nostate(eb_smem + s * 8);

            tr.advance();
        }

        // ----- Epilogue: convert FP32 -> BF16, stmatrix to SMEM, TMA store -----
        // Compose from composites/126::epi_f32_accum_to_bf16_subtiles.
        // Each warp covers 16 rows of the 64x64 output. Per warp, 4 stmatrix.x4
        // batches tile 16x64 along N (4 x 16 cols each). 4 warps tile M.
        const int warp_in_wg = t_wg / 32;
        const int lane       = t_wg % 32;
        constexpr uint32_t row_stride_d  = TILE_N * 2;     // 128 bytes per row
        constexpr uint32_t stride_per_g  = 16 * 2;         // 16 BF16 cols per group
        uint32_t thread_smem = k7_stmatrix_addr(d_smem, row_stride_d,
                                                warp_in_wg, lane);
        epi_f32_accum_to_bf16_subtiles(thread_smem, stride_per_g, d);

        // Sync MMA WG (128 threads) so all stmatrix writes are visible.
        // bar.sync uses barrier id 1 to avoid colliding with the CTA-wide
        // sync (id 0) used at kernel init.
        asm volatile("bar.sync 1, 128;\n" ::: "memory");

        // Generic-proxy -> async-proxy fence required between stmatrix
        // (generic) and TMA store (async).
        fence_proxy_async_shared_cta();

        // One elected thread issues the TMA store of the 64x64 BF16 tile.
        if (warp_in_wg == 0 && elect_one_sync()) {
            tma_store_2d(&tma_d, tile_n * TILE_N, tile_m * TILE_M, d_smem);
            tma_store_commit_group();
            tma_store_wait_group<0>();
        }
    }
}

// ============================================================================
// Host driver
// ============================================================================

namespace k7_host {

struct Shape {
    const char* name;
    int M, K, N;
};

const Shape SHAPES[] = {
    // smoke shape -- single output tile, host-reference verifies correctness
    {"smoke",       64,   64,    64},
    // production shapes
    {"und-gate", 14046, 2048,   128},
    {"und-qkv",  14046, 2048,  5120},
    {"und-o",    14046, 4096,  2048},
    {"gen-gate", 30720, 2048,   128},
    {"gen-qkv",  30720, 2048,  5120},
    {"gen-o",    30720, 4096,  2048},
};
constexpr int N_SHAPES = sizeof(SHAPES) / sizeof(SHAPES[0]);

inline int round_up(int x, int m) { return ((x + m - 1) / m) * m; }

// Fill BF16 buffer with deterministic small values for verification.
void fill_bf16(__nv_bfloat16* h, size_t n, unsigned seed) {
    if (seed == 0xFFFF) {
        // debug: all-ones probe
        for (size_t i = 0; i < n; i++) h[i] = __float2bfloat16(1.0f);
        return;
    }
    // Use small magnitudes so accumulation in FP32 doesn't overflow.
    for (size_t i = 0; i < n; i++) {
        // values in [-1, 1] approximately
        unsigned x = (unsigned)i ^ seed;
        x = (x * 1103515245u + 12345u);
        float f = (float)((int)(x & 0xFFFF) - 0x8000) / 32768.0f;
        h[i] = __float2bfloat16(f * 0.25f);
    }
}

float compare_fp32(const float* h_got, const float* h_ref, size_t n,
                   double rel_tol, int& bad, int max_print = 8) {
    bad = 0;
    double max_rel = 0.0;
    int printed = 0;
    for (size_t i = 0; i < n; i++) {
        float g = h_got[i], r = h_ref[i];
        float diff = std::fabs(g - r);
        float denom = std::fmax(std::fabs(r), 1e-6f);
        double rel = diff / denom;
        if (rel > max_rel) max_rel = rel;
        if (rel > rel_tol) {
            bad++;
            if (printed < max_print) {
                printed++;
                fprintf(stderr, "  mismatch[%zu]: got %g exp %g rel %g\n",
                        i, g, r, rel);
            }
        }
    }
    return (float)max_rel;
}

// Host reference. A is M x K row-major. The kernel's WGMMA atom is
// configured with imm-trans-a=0, imm-trans-b=1 (per primitives/55), which
// in CUTLASS Hopper convention means BOTH A and B are read with their K
// dimension contiguous in SMEM. With A stored M x K row-major (K contig)
// loaded into SMEM unchanged, that's K-contig ✓. With B stored K x N
// row-major (N contig) loaded into SMEM unchanged, the contig dim is N,
// not K. Block #101's working test stores B as K x N row-major; that
// layout for the WGMMA atom yields D[m,n] = sum_k A[m,k] * B[n,k]
// (effectively reading B with N as the row index, K as the column index
// because of imm-trans-b=1's transposing semantic).
//
// Net: with our GMEM layout B as K x N row-major, the MMA effectively
// computes D[m,n] = sum_k A[m,k] * B_storage[n*K + k], i.e. it indexes
// B as if it were N x K row-major. The host reference must match that
// indexing.
//
// O(M*N*K); only safe for the smoke shape (64^3 = 262K muladds = ~ms on host).
void host_gemm_ref(const __nv_bfloat16* hA, const __nv_bfloat16* hB,
                   float* hD, int M, int N, int K) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++) {
                float a = __bfloat162float(hA[m * K + k]);
                float b = __bfloat162float(hB[k * N + n]);
                acc += a * b;
            }
            hD[m * N + n] = acc;
        }
    }
}

void run_shape(const Shape& sh,
               int warmup_iters, int bench_iters,
               bool verify, FILE* out) {
    const int M = sh.M, K = sh.K, N = sh.N;
    const int M_padded = round_up(M, k7::TILE_M);
    const int N_padded = round_up(N, k7::TILE_N);
    const int K_padded = round_up(K, k7::TILE_K);
    if (K != K_padded) {
        fprintf(stderr, "shape %s: K=%d not multiple of TILE_K=%d, skipping\n",
                sh.name, K, k7::TILE_K);
        return;
    }
    if (N != N_padded) {
        fprintf(stderr, "shape %s: N=%d not multiple of TILE_N=%d, skipping\n",
                sh.name, N, k7::TILE_N);
        return;
    }

    // Allocate.
    // A: M_padded x K row-major (BF16).
    // B: K x N row-major (BF16) -- K rows, N contiguous.
    // D: M_padded x N row-major (BF16) -- output of stmatrix + TMA store.
    __nv_bfloat16 *dA = nullptr, *dB = nullptr, *dD_ours = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(__nv_bfloat16) * (size_t)M_padded * K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(__nv_bfloat16) * (size_t)K * N));
    CUDA_CHECK(cudaMalloc(&dD_ours, sizeof(__nv_bfloat16) * (size_t)M_padded * N));

    // Fill host inputs (kept around for verify); init device.
    std::vector<__nv_bfloat16> hA((size_t)M * K);
    std::vector<__nv_bfloat16> hB((size_t)K * N);
    bool diag = (sh.name && std::strcmp(sh.name, "smoke") == 0 &&
                 std::getenv("K7_DIAG"));
    if (diag) {
        const char* mode = std::getenv("K7_DIAG");
        if (std::string(mode) == "row_id") {
            // A[m,k] = m, B[k,n] = 1 -> D[m,n] = K * m. Reveals row mapping.
            for (int m = 0; m < M; m++)
                for (int k = 0; k < K; k++)
                    hA[m * K + k] = __float2bfloat16((float)m);
            for (size_t i = 0; i < (size_t)K * N; i++) hB[i] = __float2bfloat16(1.f);
        } else if (std::string(mode) == "col_id_kn") {
            // A[m,k] = 1, B[k,n] = n (B as K×N storage) -> D[m,n] = K * n.
            for (size_t i = 0; i < (size_t)M * K; i++) hA[i] = __float2bfloat16(1.f);
            for (int k = 0; k < K; k++)
                for (int n = 0; n < N; n++)
                    hB[k * N + n] = __float2bfloat16((float)n);
        } else if (std::string(mode) == "col_id_nk") {
            // A[m,k] = 1, B as N×K row-major: B[storage_idx = n*K + k] = n -> if kernel
            // reads B as N×K, D[m,n] = K * n. Distinguishes from col_id_kn.
            for (size_t i = 0; i < (size_t)M * K; i++) hA[i] = __float2bfloat16(1.f);
            for (int n = 0; n < N; n++)
                for (int k = 0; k < K; k++)
                    hB[n * K + k] = __float2bfloat16((float)n);
        }
    } else {
        fill_bf16(hA.data(), (size_t)M * K, 0xA5A5);
        fill_bf16(hB.data(), (size_t)K * N, 0x5A5A);
    }
    CUDA_CHECK(cudaMemset(dA, 0, sizeof(__nv_bfloat16) * (size_t)M_padded * K));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(),
                          sizeof(__nv_bfloat16) * (size_t)M * K,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(),
                          sizeof(__nv_bfloat16) * (size_t)K * N,
                          cudaMemcpyHostToDevice));

    // Build tensormaps for our kernel.
    // A: M_padded x K row-major; tile box (TILE_M rows, TILE_K cols).
    CUtensorMap tma_a;
    CUDA_CHECK(make_tma_2d_tiled(&tma_a, dA, M_padded, K, k7::TILE_M, k7::TILE_K,
                                 sizeof(__nv_bfloat16),
                                 CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                 CU_TENSOR_MAP_SWIZZLE_128B));
    // B: K x N row-major; tile box (TILE_K rows, TILE_N cols).
    CUtensorMap tma_b;
    CUDA_CHECK(make_tma_2d_tiled(&tma_b, dB, K, N, k7::TILE_K, k7::TILE_N,
                                 sizeof(__nv_bfloat16),
                                 CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                 CU_TENSOR_MAP_SWIZZLE_128B));
    // D: M_padded x N row-major; tile box (TILE_M rows, TILE_N cols).
    CUtensorMap tma_d;
    CUDA_CHECK(make_tma_2d_tiled(&tma_d, dD_ours, M_padded, N, k7::TILE_M, k7::TILE_N,
                                 sizeof(__nv_bfloat16),
                                 CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                 CU_TENSOR_MAP_SWIZZLE_128B));

    int num_k_tiles = K / k7::TILE_K;
    dim3 grid(M_padded / k7::TILE_M, N / k7::TILE_N, 1);
    dim3 block(k7::N_THREADS, 1, 1);

    // Set max dynamic SMEM
    CUDA_CHECK(cudaFuncSetAttribute(
        k7_dense_gemm_bf16_v0,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)k7::SMEM_BYTES));

    // ---- Warmup ----
    for (int i = 0; i < warmup_iters; i++) {
        k7_dense_gemm_bf16_v0<<<grid, block, k7::SMEM_BYTES>>>(
            tma_a, tma_b, tma_d, num_k_tiles, M_padded, N, M);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Bench ----
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int i = 0; i < bench_iters; i++) {
        k7_dense_gemm_bf16_v0<<<grid, block, k7::SMEM_BYTES>>>(
            tma_a, tma_b, tma_d, num_k_tiles, M_padded, N, M);
    }
    cudaEventRecord(e1);
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    double ms_per = (double)ms / bench_iters;
    double tflops = (2.0 * (double)M * (double)N * (double)K) /
                    (ms_per * 1e-3) / 1e12;

    // ---- Verify (host reference, only at the smoke shape) ----
    int bad = 0;
    float max_rel = 0;
    bool pass = true;
    bool ran_verify = false;
    if (verify && (size_t)M * N * K <= (size_t)1024 * 1024) {
        // Smoke: compare BF16 kernel output against FP32 host reference.
        std::vector<__nv_bfloat16> h_ours_bf16((size_t)M_padded * N);
        CUDA_CHECK(cudaMemcpy(h_ours_bf16.data(), dD_ours,
                              sizeof(__nv_bfloat16) * (size_t)M_padded * N,
                              cudaMemcpyDeviceToHost));
        std::vector<float> h_ours_f32((size_t)M * N);
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++)
                h_ours_f32[m * N + n] = __bfloat162float(h_ours_bf16[m * N + n]);
        std::vector<float> h_ref((size_t)M * N);
        host_gemm_ref(hA.data(), hB.data(), h_ref.data(), M, N, K);
        // BF16 output -> larger tolerance (~1e-2 relative is realistic).
        max_rel = compare_fp32(h_ours_f32.data(), h_ref.data(),
                               (size_t)M * N, /*rel_tol*/ 1e-1, bad, 4);
        pass = (bad == 0);
        ran_verify = true;
    }

    fprintf(out,
        "shape=%-10s M=%-6d K=%-5d N=%-5d  ms/iter=%8.4f  TFLOPS=%8.2f  %s\n",
        sh.name, M, K, N, ms_per, tflops,
        ran_verify ? (pass ? "verify=OK" : "verify=FAIL")
                   : "verify=skipped (production shape)");
    if (ran_verify && !pass) {
        fprintf(out, "  bad=%d max_rel=%.3g\n", bad, max_rel);
    }
    if (diag) {
        std::vector<__nv_bfloat16> h_ours_bf16((size_t)M_padded * N);
        CUDA_CHECK(cudaMemcpy(h_ours_bf16.data(), dD_ours,
                              sizeof(__nv_bfloat16) * (size_t)M_padded * N,
                              cudaMemcpyDeviceToHost));
        fprintf(out, "  diag dump (col 0, all 64 rows): val = ?\n");
        for (int r = 0; r < 64; r++) {
            fprintf(out, "    row %2d: %7.1f", r,
                    __bfloat162float(h_ours_bf16[r * N + 0]));
            if ((r & 7) == 7) fprintf(out, "\n"); else fprintf(out, "  ");
        }
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dD_ours);
}

}  // namespace k7_host

int main(int argc, char** argv) {
    int warmup = 3;
    int iters  = 20;
    bool verify = true;
    if (argc >= 2) iters = std::atoi(argv[1]);
    if (argc >= 3 && std::string(argv[2]) == "noverify") verify = false;

    CUDA_CHECK(cudaFree(nullptr));  // init context

    int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s (cc %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("K7 V0 SKELETON  TILE=%dx%dx%d  STAGES=%d  THREADS=%d\n",
           k7::TILE_M, k7::TILE_N, k7::TILE_K, k7::NUM_STAGES, k7::N_THREADS);
    printf("warmup=%d  iters=%d  verify=%s\n\n",
           warmup, iters, verify ? "true" : "false");

    int rc = 0;
    for (int i = 0; i < k7_host::N_SHAPES; i++) {
        k7_host::run_shape(k7_host::SHAPES[i], warmup, iters, verify, stdout);
    }

    return rc;
}

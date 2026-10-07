// ARCH: sm_90a
// 20_tma_load_multicast_test.cu -- multicast TMA load to 2 CTAs.
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
#include "../src/primitives/19_tma_load_2sm.cuh"
#include "../src/primitives/20_tma_load_multicast.cuh"
#include "../src/primitives/23_tma_tensormap.cuh"
#include "../src/primitives/29_mbarrier_init.cuh"
#include "../src/primitives/31_mbarrier_arrive_tx.cuh"
#include "../src/primitives/33_mbarrier_try_wait.cuh"
#include <cuda_fp16.h>
#include "20_tma_load_multicast.cuh"

// =============================================================================
// ours (originally guarded by PL_AGENTIC_SM100A || PL_AGENTIC_SM103A)
// =============================================================================

// ARCH: sm_100a


__global__ void __cluster_dims__(2, 1, 1)
k_mcast(const __grid_constant__ CUtensorMap desc) {
  __shared__ __align__(128) float smem[16 * 16];
  __shared__ __align__(16)  uint64_t mbar;
  if (threadIdx.x == 0) {
    mbarrier_init(smem_ptr_u32(&mbar), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  // Every destination CTA must register expected_tx; the load is issued
  // once (from CTA 0) and the complete_tx signal multicasts to all CTAs.
  if (threadIdx.x == 0) {
    mbarrier_arrive_expect_tx(smem_ptr_u32(&mbar), 16 * 16 * sizeof(float));
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    tma_load_2d_multicast(smem_ptr_u32(smem), &desc, smem_ptr_u32(&mbar),
                             0, 0, /*ctamask=*/0x3);
  }
  mbarrier_wait_parity(smem_ptr_u32(&mbar), 0);
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n"
                 :: "r"(smem_ptr_u32(&mbar)));
}

static int run_ours() {
  /* (orig args dropped) */
  std::vector<float> hIn(16 * 16, 1.f);
  float* dIn = nullptr; CUDA_CHECK(cudaMalloc(&dIn, 16 * 16 * 4));
  CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), 16 * 16 * 4, cudaMemcpyHostToDevice));
  CUtensorMap desc;
  CUDA_CHECK(make_tma_2d_tiled(&desc, dIn, 16, 16, 16, 16, sizeof(float),
                                 CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                 CU_TENSOR_MAP_SWIZZLE_NONE));
  k_mcast<<<2, 128>>>(desc);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaFree(dIn);
  printf("tma_load_multicast : compile + run OK\n");
  PASS();
}

// =============================================================================
// theirs (originally guarded by PL_AGENTIC_SM90A)
// =============================================================================

// Test: TMA 2D load with multicast -- compile + PTX verification
// Build: nvcc -arch=sm_90a -I../primitives -I. -o ../build/20_test 20_tma_load_multicast_test.cu

extern __shared__ char smem_buf[];

__global__ void tma_multicast_kernel(const __grid_constant__ CUtensorMap tensormap,
                                      half* output) {
    half* smem_tile = reinterpret_cast<half*>(smem_buf);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem_buf + 8192);
    if (threadIdx.x == 0) {
        uint32_t mbar_addr = smem_ptr_u32(mbar);
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                     :: "r"(mbar_addr), "r"(1) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                     :: "r"(mbar_addr), "r"(8192u) : "memory");
        tma_load_2d_multicast(&tensormap,
                              smem_ptr_u32(smem_tile),
                              mbar_addr,
                              0, 0,
                              0x3u);  // broadcast to CTAs 0+1
    }
}

__global__ void tma_multicast_l2_kernel(const __grid_constant__ CUtensorMap tensormap,
                                         half* output) {
    half* smem_tile = reinterpret_cast<half*>(smem_buf);
    uint64_t* mbar = reinterpret_cast<uint64_t*>(smem_buf + 8192);
    if (threadIdx.x == 0) {
        uint32_t mbar_addr = smem_ptr_u32(mbar);
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n"
                     :: "r"(mbar_addr), "r"(1) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
                     :: "r"(mbar_addr), "r"(8192u) : "memory");
        tma_load_2d_multicast_l2hint(&tensormap,
                                     smem_ptr_u32(smem_tile),
                                     mbar_addr,
                                     0, 0,
                                     0x3u, 0ull);
    }
}

static int run_theirs() {
  /* (orig args dropped) */
    printf("20_tma_load_multicast: compiled successfully.\n");
    PASS();
    return 0;
}


// =============================================================================
// 2SM + multicast composition (Blackwell only) -- Phase 3 HIGH
// =============================================================================

#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)

// Cluster of 2 = one 2SM CTA-pair. The 2sm load fans out to both peer CTAs'
// SMEM at the same offset; the multicast::cluster ctamask (= 0x3, both
// CTAs in the cluster) signals complete_tx to each peer's mbar. The
// peer-bit mask on the mbar address routes the issuing-side complete_tx
// to "CTA 0 of the pair" -- canonical 2SM idiom.
// __CUDA_ARCH__ gating: cta_group::2 is rejected by ptxas on sm_90a, but
// test 20's ARCH comment is sm_90a so the build pass cross-compiles to
// sm_90a too. The kernel body is empty on the sm_90a pass; the SASS for
// sm_103a (the running GPU) has the real implementation.
// 2SM+multicast pattern.
// Cluster of 4 = 2 CTA-pairs. cta_group::2 delivers each pair internally;
// multicast::cluster fans out to the additional pair via ctamask. Each
// CTA arrives_expect_tx on its own mbar; only the issuer (CTA 0) calls
// the load. ctamask = 0x5 selects "CTA 0 of each pair" (bits 0 and 2);
// with cta_group::2, the hardware also delivers to peer CTA 1 / 3 of
// each pair.
__global__ void __cluster_dims__(4, 1, 1)
k_2sm_mcast(const __grid_constant__ CUtensorMap desc, float* out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  __shared__ __align__(128) float smem[16 * 16];
  __shared__ __align__(16)  uint64_t mbar;
  uint32_t mbar_addr = smem_ptr_u32(&mbar);
  if (threadIdx.x == 0) {
    mbarrier_init(mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  // Pair issuer = even-ranked CTA (0, 2). Only the issuer of each pair
  // arrives_expect_tx + waits; peer CTA (1, 3) gets data via cta_group::2
  // mechanism and never touches the mbar.
  bool is_pair_issuer = ((blockIdx.x & 1) == 0);
  if (threadIdx.x == 0 && is_pair_issuer) {
    mbarrier_arrive_expect_tx(mbar_addr, 16 * 16 * sizeof(float));
  }
  // Only CTA 0 issues the multicast load. ctamask=0x5 selects CTA 0 and
  // CTA 2 as multicast destinations (= "CTA 0 of each pair").
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint32_t mbar_masked = tma_peer_bit_mask(mbar_addr);
    tma_load_2d_2sm_multicast(&desc, smem_ptr_u32(smem), mbar_masked,
                              0, 0, /*ctamask=*/0x5);
  }
  if (is_pair_issuer)
    mbarrier_wait_parity(mbar_addr, 0);
  __syncthreads();
  if (threadIdx.x == 0) {
    out[blockIdx.x] = smem[0];
  }
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(mbar_addr));
#else
  (void)desc; (void)out;
#endif
}

// L2-hint variant: same launch shape, just exercises the cache-policy operand.
__global__ void __cluster_dims__(4, 1, 1)
k_2sm_mcast_l2(const __grid_constant__ CUtensorMap desc, float* out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  __shared__ __align__(128) float smem[16 * 16];
  __shared__ __align__(16)  uint64_t mbar;
  uint32_t mbar_addr = smem_ptr_u32(&mbar);
  if (threadIdx.x == 0) {
    mbarrier_init(mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  bool is_pair_issuer = ((blockIdx.x & 1) == 0);
  if (threadIdx.x == 0 && is_pair_issuer) {
    mbarrier_arrive_expect_tx(mbar_addr, 16 * 16 * sizeof(float));
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint32_t mbar_masked = tma_peer_bit_mask(mbar_addr);
    tma_load_2d_2sm_multicast_l2hint(&desc, smem_ptr_u32(smem), mbar_masked,
                                     0, 0, /*ctamask=*/0x5, /*policy=*/0ull);
  }
  if (is_pair_issuer)
    mbarrier_wait_parity(mbar_addr, 0);
  __syncthreads();
  if (threadIdx.x == 0) {
    out[blockIdx.x] = smem[0];
  }
  __syncthreads();
  if (threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];\n" :: "r"(mbar_addr));
#else
  (void)desc; (void)out;
#endif
}

static int run_2sm_multicast() {
  std::vector<float> hIn(16 * 16, 7.f);  // sentinel value
  float* dIn = nullptr; CUDA_CHECK(cudaMalloc(&dIn, 16 * 16 * 4));
  CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), 16 * 16 * 4, cudaMemcpyHostToDevice));
  CUtensorMap desc;
  CUDA_CHECK(make_tma_2d_tiled(&desc, dIn, 16, 16, 16, 16, sizeof(float),
                               CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                               CU_TENSOR_MAP_SWIZZLE_NONE));
  float* dOut = nullptr; CUDA_CHECK(cudaMalloc(&dOut, 4 * sizeof(float)));

  // Verify the multicast fanout: CTAs 0 and 2 are the pair issuers
  // (waited on the mbar) -- their SMEM was populated and they wrote 7.0.
  // CTAs 1 and 3 are peer CTAs in their pairs; they get data via
  // cta_group::2 but don't wait, so their out value is racy.
  CUDA_CHECK(cudaMemset(dOut, 0, 4 * sizeof(float)));
  k_2sm_mcast<<<4, 128>>>(desc, dOut);
  CUDA_CHECK(cudaDeviceSynchronize());
  float h[4] = {0.f};
  CUDA_CHECK(cudaMemcpy(h, dOut, 4 * sizeof(float), cudaMemcpyDeviceToHost));
  if (h[0] != 7.f || h[2] != 7.f) {
    fprintf(stderr, "2sm+multicast: issuer outs (%.1f, _, %.1f, _) expected 7,_,7,_\n",
            h[0], h[2]);
    cudaFree(dIn); cudaFree(dOut);
    FAIL("2sm+multicast issuer CTAs (0, 2) did not receive the tile");
  }

  CUDA_CHECK(cudaMemset(dOut, 0, 4 * sizeof(float)));
  k_2sm_mcast_l2<<<4, 128>>>(desc, dOut);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h, dOut, 4 * sizeof(float), cudaMemcpyDeviceToHost));
  if (h[0] != 7.f || h[2] != 7.f) {
    fprintf(stderr, "2sm+multicast+l2: issuer outs (%.1f, _, %.1f, _) expected 7,_,7,_\n",
            h[0], h[2]);
    cudaFree(dIn); cudaFree(dOut);
    FAIL("2sm+multicast+l2 issuer CTAs (0, 2) did not receive the tile");
  }

  cudaFree(dIn); cudaFree(dOut);
  printf("tma_load_2d_2sm_multicast{,_l2hint} : both pair-issuer CTAs (0, 2) received the tile\n");
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

int main() {
  int rc_ours   = run_ours();
  int rc_theirs = run_theirs();
  int rc_2sm    = 0;
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
  rc_2sm = run_2sm_multicast();
#endif
  return (rc_ours == 0 && rc_theirs == 0 && rc_2sm == 0) ? 0 : 1;
}

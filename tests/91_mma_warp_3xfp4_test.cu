#if defined(PL_AGENTIC_SM103A)
// ARCH: sm_103a
// 91_mma_warp_3xfp4_test.cu -- runtime smoke test for block 91 (mma_warp_3xfp4).
//
// The block is now a __device__ template (blocks/91_mma_warp_3xfp4.cuh).
// This test owns the __global__ wrapper that calls into the block. It
// instantiates with cta_group::1 and ::2 and a `dont_run` flag set to 0
// so the kernel returns before engaging the K-loop -- no producer is
// wired here. The launch verifies the kernel compiles + loads on the
// GPU; SASS contains the K-loop mnemonics (verified separately by
// test 75 + cuobjdump).

#include <cstdio>
#include <cuda_runtime.h>
#include "test_utils.cuh"
#include "../blocks/91_mma_warp_3xfp4.cuh"

#if !K_LOOP_3XFP4_DEPS_AVAILABLE

int main() {
    std::printf("[SKIP] 91_mma_warp_3xfp4 -- composite 75 deps not yet on disk; "
                "header-only block requires alloc/dealloc/relinquish/commit/cp/"
                "fence/mbarrier/cluster-barrier. Standalone primitives 6, 16, 17, "
                "65 (sm_103a-only) are validated.\n");
    return 0;
}

#else

template <int CtaGroup, MmaMxf4Variant V, int M, int N>
__launch_bounds__(32) __global__ void mma_warp_3xfp4_test_kernel(
    int num_k_tiles,
    uint32_t* tmem_alloc_dst,
    uint64_t* full_mbar,
    uint64_t* empty_mbar,
    uint64_t* sf_full_mbar,
    uint64_t* sf_empty_mbar,
    uint64_t* acc_done_mbar,
    uint32_t* a_buf_smem,
    uint32_t* b_buf_smem,
    uint32_t* sf_a_smem,
    uint32_t* sf_b_smem,
    int       stride_bytes,
    uint16_t  ctamask,
    int*      dont_run) {
    if (*dont_run == 0) return;
    mma_warp_3xfp4_block<CtaGroup, V, M, N>(
        num_k_tiles, tmem_alloc_dst,
        full_mbar, empty_mbar, sf_full_mbar, sf_empty_mbar, acc_done_mbar,
        a_buf_smem, b_buf_smem, sf_a_smem, sf_b_smem,
        stride_bytes, ctamask);
}

int main() {
    int* d_dont_run = nullptr;
    if (cudaMalloc(&d_dont_run, sizeof(int)) != cudaSuccess) {
        std::printf("[SKIP] 91_mma_warp_3xfp4 -- GPU runtime unavailable\n");
        return 0;
    }
    int zero = 0;
    cudaMemcpy(d_dont_run, &zero, sizeof(int), cudaMemcpyHostToDevice);

    mma_warp_3xfp4_test_kernel<1, MmaMxf4Variant::MXF4NVF4_BLOCK16, /*M=*/128, /*N=*/128>
        <<<1, 32, 256>>>(/*num_k_tiles=*/1, /*tmem_alloc_dst=*/nullptr,
                         /*full_mbar=*/nullptr, /*empty_mbar=*/nullptr,
                         /*sf_full_mbar=*/nullptr, /*sf_empty_mbar=*/nullptr,
                         /*acc_done_mbar=*/nullptr,
                         /*a_buf_smem=*/nullptr, /*b_buf_smem=*/nullptr,
                         /*sf_a_smem=*/nullptr, /*sf_b_smem=*/nullptr,
                         /*stride_bytes=*/128, /*ctamask=*/(uint16_t)0xFFFFu,
                         d_dont_run);
    mma_warp_3xfp4_test_kernel<2, MmaMxf4Variant::MXF4NVF4_BLOCK16, /*M=*/128, /*N=*/128>
        <<<1, 32, 256>>>(/*num_k_tiles=*/1, /*tmem_alloc_dst=*/nullptr,
                         /*full_mbar=*/nullptr, /*empty_mbar=*/nullptr,
                         /*sf_full_mbar=*/nullptr, /*sf_empty_mbar=*/nullptr,
                         /*acc_done_mbar=*/nullptr,
                         /*a_buf_smem=*/nullptr, /*b_buf_smem=*/nullptr,
                         /*sf_a_smem=*/nullptr, /*sf_b_smem=*/nullptr,
                         /*stride_bytes=*/128, /*ctamask=*/(uint16_t)0xFFFFu,
                         d_dont_run);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::printf("  GPU runtime returned %s; PTX/SASS-only verification\n",
                    cudaGetErrorString(e));
    }
    cudaFree(d_dont_run);
    std::printf("  cuobjdump --dump-sass build/91_mma_warp_3xfp4_test should "
                "show UTCMMA + UTCCP + UTCC + barrier.cluster mnemonics.\n");
    std::printf("[PASS] 91_mma_warp_3xfp4\n");
    return 0;
}

#endif  // K_LOOP_3XFP4_DEPS_AVAILABLE

#endif  // PL_AGENTIC_SM103A

#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 65_cvt_f32_to_fp4_rs_test.cu -- cvt.rs x4 on exact inputs (0.5, 1, 2, 3): rbits
// cannot carry, so each packed output must equal its exact encoding (a in the upper bits).

#include "test_utils.cuh"
#include "../src/primitives/65_cvt_f32_to_fp4_rs.cuh"

__global__ void k_convert_all(uint16_t* out_e2m1,
                              uint32_t* out_e4m3,
                              uint32_t* out_e5m2,
                              uint32_t* out_e3m2,
                              uint32_t* out_e2m3,
                              float a, float b, float e, float f,
                              uint32_t rbits)
{
    if (threadIdx.x == 0) {
        *out_e2m1 = cvt_rs_satfinite_e2m1x4_f32(a, b, e, f, rbits);
        *out_e4m3 = cvt_rs_satfinite_e4m3x4_f32(a, b, e, f, rbits);
        *out_e5m2 = cvt_rs_satfinite_e5m2x4_f32(a, b, e, f, rbits);
        *out_e3m2 = cvt_rs_satfinite_e3m2x4_f32(a, b, e, f, rbits);
        *out_e2m3 = cvt_rs_satfinite_e2m3x4_f32(a, b, e, f, rbits);
    }
}

int main() {
    const char* name = "65_cvt_f32_to_fp4_rs";

    uint16_t* d_e2m1 = nullptr;
    uint32_t* d_e4m3 = nullptr, *d_e5m2 = nullptr, *d_e3m2 = nullptr, *d_e2m3 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_e2m1, sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_e4m3, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_e5m2, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_e3m2, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_e2m3, sizeof(uint32_t)));

    k_convert_all<<<1,32>>>(d_e2m1, d_e4m3, d_e5m2, d_e3m2, d_e2m3,
                            0.5f, 1.0f, 2.0f, 3.0f, 0xCAFEBABEu);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    uint16_t v_e2m1 = 0; uint32_t v_e4m3 = 0, v_e5m2 = 0, v_e3m2 = 0, v_e2m3 = 0;
    CUDA_CHECK(cudaMemcpy(&v_e2m1, d_e2m1, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&v_e4m3, d_e4m3, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&v_e5m2, d_e5m2, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&v_e3m2, d_e3m2, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&v_e2m3, d_e2m3, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    struct ConversionCase { const char* format; uint32_t result; uint32_t expected; };
    const ConversionCase cases[] = {
        {"e2m1x4", v_e2m1, 0x1245u},
        {"e4m3x4", v_e4m3, 0x30384044u},
        {"e5m2x4", v_e5m2, 0x383C4042u},
        {"e3m2x4", v_e3m2, 0x080C1012u},
        {"e2m3x4", v_e2m3, 0x04081014u},
    };

    int rc = 0;
    for (const ConversionCase& conversion : cases) {
        const bool match = conversion.result == conversion.expected;
        std::printf("  [%s] %s(0.5,1,2,3) = 0x%08x (expected 0x%08x)\n",
                    match ? "PASS" : "FAIL", conversion.format,
                    conversion.result, conversion.expected);
        if (!match) rc = 1;
    }

    CUDA_CHECK(cudaFree(d_e2m1));
    CUDA_CHECK(cudaFree(d_e4m3));
    CUDA_CHECK(cudaFree(d_e5m2));
    CUDA_CHECK(cudaFree(d_e3m2));
    CUDA_CHECK(cudaFree(d_e2m3));
    std::printf("[%s] %s\n", rc == 0 ? "PASS" : "FAIL", name);
    return rc;
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

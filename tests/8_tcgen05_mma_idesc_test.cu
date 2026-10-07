#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 8_tcgen05_mma_idesc_test.cu -- host-side bit-pattern verification for
// the tcgen05.mma instruction-descriptor builders.
//
// Verifies (1) `make_idesc_table44` matches a hand-rolled reference per
// PTX 9.7.18.4.2 Table 53 (dtype/atype/btype/transpose/negate/M/N), and
// (2) the kind-specific specializations (f16, bf16, tf32, e4m3, e5m2,
// fp8 mixed, fp4, s8, u8 with saturate) all emit the same bits as the
// generic Table 53 packer plus their specialization deltas. Pure host
// code (no kernel launch).
//
// Also: direct host-side bit-pattern
// verification for the Table 54 (mxf8f6f4), Table 55 (mxf4nvf4), and the
// `idesc_set_sparsity` / `idesc_set_ws_mode` helpers.
// Each is checked against a hand-rolled reference derived from PTX
// 9.7.18.4.2 Tables 54 and 55.

#include "test_utils.cuh"
#include "../src/primitives/8_tcgen05_mma_idesc.cuh"

// Compute expected idesc by hand per PTX 9.7.18.4.2 Table 53.
static uint32_t expected_table44(
    int M, int N,
    uint32_t dtype, uint32_t atype, uint32_t btype,
    bool ta = false, bool tb = false,
    uint32_t saturate_bit = 0) {
  uint32_t idesc = 0;
  idesc |= saturate_bit << 3;
  idesc |= dtype << 4;
  idesc |= atype << 7;
  idesc |= btype << 10;
  idesc |= (ta ? 1u : 0u) << 15;
  idesc |= (tb ? 1u : 0u) << 16;
  idesc |= ((uint32_t)(N >> 3) & 0x3F) << 17;
  idesc |= ((uint32_t)(M >> 4) & 0x1F) << 24;
  return idesc;
}

// Per PTX 9.7.18.4.2 Table 54 (.kind::mxf8f6f4):
//   bit  2     : sparsity
//   bits 4-5   : sf_b_data_id
//   bits 7-9   : atype  (E4M3=0, E5M2=1, E2M3=3, E3M2=4, E2M1=5)
//   bits 10-12 : btype
//   bits 13-14 : negate A / B
//   bits 15-16 : transpose A / B
//   bits 17-22 : N >> 3
//   bit  23    : scale type for both scales (UE8M0 = 1)
//   bits 27-28 : M >> 7  (M must be 128 or 256)
//   bits 29-30 : sf_a_data_id
static uint32_t expected_table45(
    int M, int N,
    uint32_t atype, uint32_t btype,
    bool ta, bool tb,
    bool neg_a, bool neg_b,
    uint32_t sf_a_data_id, uint32_t sf_b_data_id,
    bool ue8m0, bool sparse) {
  uint32_t idesc = 0;
  idesc |= (sparse ? 1u : 0u) << 2;
  idesc |= (sf_b_data_id & 0x3u) << 4;
  idesc |= (atype & 0x7u) << 7;
  idesc |= (btype & 0x7u) << 10;
  idesc |= (neg_a ? 1u : 0u) << 13;
  idesc |= (neg_b ? 1u : 0u) << 14;
  idesc |= (ta ? 1u : 0u) << 15;
  idesc |= (tb ? 1u : 0u) << 16;
  idesc |= ((uint32_t)(N >> 3) & 0x3Fu) << 17;
  idesc |= (ue8m0 ? 1u : 0u) << 23;
  idesc |= ((uint32_t)(M >> 7) & 0x3u) << 27;
  idesc |= (sf_a_data_id & 0x3u) << 29;
  return idesc;
}

// Per PTX 9.7.18.4.2 Table 55 (.kind::mxf4nvf4): same layout as Table 54
// except atype and btype are fixed to E2M1 (= 1) at bits 7 and 10, and
// bit 31 selects the K-atom (0 = K=64 dense / K=128 sparse). The wrapper
// always emits bit 31 = 0; the K=96 atom lives in primitive #6.
static uint32_t expected_table46_mxf4nvf4(
    int M, int N,
    bool ta, bool tb,
    bool neg_a, bool neg_b,
    uint32_t sf_a_data_id, uint32_t sf_b_data_id,
    bool ue8m0, bool sparse) {
  uint32_t idesc = 0;
  idesc |= (sparse ? 1u : 0u) << 2;
  idesc |= (sf_b_data_id & 0x3u) << 4;
  idesc |= 1u << 7;             // atype = E2M1
  idesc |= 1u << 10;            // btype = E2M1
  idesc |= (neg_a ? 1u : 0u) << 13;
  idesc |= (neg_b ? 1u : 0u) << 14;
  idesc |= (ta ? 1u : 0u) << 15;
  idesc |= (tb ? 1u : 0u) << 16;
  idesc |= ((uint32_t)(N >> 3) & 0x3Fu) << 17;
  idesc |= (ue8m0 ? 1u : 0u) << 23;
  idesc |= ((uint32_t)(M >> 7) & 0x3u) << 27;
  idesc |= (sf_a_data_id & 0x3u) << 29;
  return idesc;
}

// Number of outputs the device-side kernel populates: 9 baseline +
// 2 mxf8f6f4 + 2 mxf4nvf4 + 2 sparsity + 2 ws_mode = 17.
constexpr int IDESC_OUT_COUNT = 17;

__global__ void k_idesc(uint32_t* out) {
  out[0] = make_idesc_f16_f32(128, 256);
  out[1] = make_idesc_bf16_f32(128, 64);
  out[2] = make_idesc_f16_f16(128, 128);
  out[3] = make_idesc_tf32_f32(128, 128);
  out[4] = make_idesc_e4m3_f32(128, 256);
  out[5] = make_idesc_e5m2_f32(128, 8);
  out[6] = make_idesc_fp4_f32(128, 128);
  out[7] = make_idesc_s8_s32(128, 8, /*saturate=*/true);
  out[8] = make_idesc_u8_s32(128, 8, /*saturate=*/false);

  // ----- Table 54 mxf8f6f4 -----
  // Case A: minimal -- M=128, N=128, E4M3 x E4M3, UE8M0 scales, dense.
  out[9] = make_idesc_mxf8f6f4(
      128, 128,
      /*atype*/ 0, /*btype*/ 0,
      /*ta*/ false, /*tb*/ false,
      /*neg_a*/ false, /*neg_b*/ false,
      /*sf_a_data_id*/ 0, /*sf_b_data_id*/ 0,
      /*ue8m0*/ true, /*sparse*/ false);

  // Case B: maxed -- M=256, N=256, E2M1 x E2M1, transposed, negated, both
  // scale-factor IDs nonzero, UE4M3 selected (ue8m0=false), sparse.
  out[10] = make_idesc_mxf8f6f4(
      256, 256,
      /*atype*/ 5, /*btype*/ 5,
      /*ta*/ true, /*tb*/ true,
      /*neg_a*/ true, /*neg_b*/ true,
      /*sf_a_data_id*/ 2, /*sf_b_data_id*/ 3,
      /*ue8m0*/ false, /*sparse*/ true);

  // ----- Table 55 mxf4nvf4 -----
  // Case A: defaults -- M=128, N=128 (UE4M3, dense, all flags clear).
  out[11] = make_idesc_mxf4nvf4(128, 128);

  // Case B: M=256, N=64, sparse + UE8M0 + transposed B + sf IDs nonzero.
  out[12] = make_idesc_mxf4nvf4(
      256, 64,
      /*ta*/ false, /*tb*/ true,
      /*neg_a*/ false, /*neg_b*/ false,
      /*sf_a_data_id*/ 1, /*sf_b_data_id*/ 2,
      /*ue8m0*/ true, /*sparse*/ true);

  // ----- idesc_set_sparsity -----
  // Apply to a Table 53 base idesc (f16_f32, M=128, N=128) with selector=2.
  uint32_t base_t44 = make_idesc_f16_f32(128, 128);
  out[13] = base_t44;                              // base, for cross-check
  out[14] = idesc_set_sparsity(base_t44, /*sel*/ 2);

  // ----- idesc_set_ws_mode -----
  // Set ws_mode=2 over the same Table 53 base (bits 30-31 are otherwise 0).
  out[15] = idesc_set_ws_mode(base_t44, /*mode*/ 2);
  // Set ws_mode=3 then back to 0 -- exercises the clear-mask branch.
  out[16] = idesc_set_ws_mode(idesc_set_ws_mode(base_t44, 3), 0);
}

int main() {
  uint32_t* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, IDESC_OUT_COUNT * sizeof(uint32_t)));
  k_idesc<<<1, 1>>>(d);
  CUDA_CHECK(cudaDeviceSynchronize());
  uint32_t h[IDESC_OUT_COUNT];
  CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  cudaFree(d);

  // Expected helpers for the new builders.
  const uint32_t want_t44_f16_f32_128_128 =
      expected_table44(128, 128, 1, 0, 0);
  const uint32_t want_mxf8f6f4_a =
      expected_table45(128, 128, 0, 0, false, false, false, false,
                       0, 0, true, false);
  const uint32_t want_mxf8f6f4_b =
      expected_table45(256, 256, 5, 5, true, true, true, true,
                       2, 3, false, true);
  const uint32_t want_mxf4nvf4_a =
      expected_table46_mxf4nvf4(128, 128, false, false, false, false,
                                0, 0, false, false);
  const uint32_t want_mxf4nvf4_b =
      expected_table46_mxf4nvf4(256, 64, false, true, false, false,
                                1, 2, true, true);
  const uint32_t want_sparsity =
      want_t44_f16_f32_128_128 | (1u << 2) | (2u << 0);
  const uint32_t want_ws_mode_2 =
      want_t44_f16_f32_128_128 | (2u << 30);
  const uint32_t want_ws_mode_0 = want_t44_f16_f32_128_128;

  struct Case { uint32_t got, want; const char* name; } cases[] = {
    // -- Existing Table 53 specializations --
    { h[0], expected_table44(128, 256, 1, 0, 0), "f16_f32(128,256)"  },
    { h[1], expected_table44(128, 64,  1, 1, 1), "bf16_f32(128,64)"  },
    { h[2], expected_table44(128, 128, 0, 0, 0), "f16_f16(128,128)"  },
    { h[3], expected_table44(128, 128, 1, 2, 2), "tf32_f32(128,128)" },
    { h[4], expected_table44(128, 256, 1, 0, 0), "e4m3_f32(128,256)" },
    { h[5], expected_table44(128, 8,   1, 1, 1), "e5m2_f32(128,8)"   },
    { h[6], expected_table44(128, 128, 1, 5, 5), "fp4_f32(128,128)"  },
    { h[7], expected_table44(128, 8,   2, 1, 1, false, false, 1),
                                                "s8_s32_sat(128,8)" },
    { h[8], expected_table44(128, 8,   2, 0, 0, false, false, 0),
                                                "u8_s32(128,8)"      },
    // -- Tables 54 / 55 + mutators --
    { h[9],  want_mxf8f6f4_a,  "mxf8f6f4 minimal(128,128 E4M3xE4M3)" },
    { h[10], want_mxf8f6f4_b,  "mxf8f6f4 maxed(256,256 E2M1xE2M1)"   },
    { h[11], want_mxf4nvf4_a,  "mxf4nvf4 default(128,128)"           },
    { h[12], want_mxf4nvf4_b,  "mxf4nvf4 sparse+UE8M0(256,64)"       },
    { h[13], want_t44_f16_f32_128_128,
                              "f16_f32(128,128) base for mutators"   },
    { h[14], want_sparsity,    "set_sparsity(sel=2) over T44 base"   },
    { h[15], want_ws_mode_2,   "set_ws_mode(2) over T44 base"        },
    { h[16], want_ws_mode_0,   "set_ws_mode(3 then 0) clears bits"   },
  };

  int fails = 0;
  for (auto& c : cases) {
    if (c.got != c.want) {
      fprintf(stderr, "  %s: got 0x%08x want 0x%08x\n",
              c.name, c.got, c.want);
      ++fails;
    }
  }
  if (fails) FAIL("idesc mismatch (%d / %d)",
                  fails, (int)(sizeof(cases) / sizeof(cases[0])));
  printf("tcgen05 idesc builders : all %d patterns match\n",
         (int)(sizeof(cases) / sizeof(cases[0])));
  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

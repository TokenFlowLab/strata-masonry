// fmha_cpu_ref.cuh -- host fp32 reference for the sm100a BF16 FMHA context kernels.
// Shared by the dense context drivers (uniform_inline, uniform_2sm_inline, varlen).
#pragma once

#include <cmath>
#include <vector>
#include <cuda_bf16.h>

// CPU reference: per-(sample, q-head) flash-attention in fp32. Q/K/V are the natural
// [token, head, hd] layouts (NOT the transposed V_T the kernel uses). cq/ck are cumulative-seqlen
// prefix sums (cq[s]..cq[s+1] = sample s's tokens). (sample, q-head) pairs write disjoint output
// rows, so the outer pair is parallelized -- a serial reference is ~60 GFLOP and takes ~40 s.
static void cpu_fmha_ref(const __nv_bfloat16* hQ, const __nv_bfloat16* hK,
                         const __nv_bfloat16* hV, float* hO,
                         const std::vector<int>& cq, const std::vector<int>& ck,
                         int nqh, int nkh, int hd, bool causal) {
  const long  Nq    = cq.back();
  const int   g     = nqh / nkh;              // q-heads per kv-head (GQA group)
  const float scale = 1.0f / sqrtf((float)hd);
  const int   ns    = (int)cq.size() - 1;

  for (long i = 0; i < Nq * nqh * hd; ++i) hO[i] = 0.f;

  #pragma omp parallel for schedule(dynamic) collapse(2)
  for (int s = 0; s < ns; ++s) {
    for (int h = 0; h < nqh; ++h) {
      const int q_lo = cq[s], k_lo = ck[s];
      const int sq   = cq[s + 1] - q_lo, sk = ck[s + 1] - k_lo;
      const int hk   = h / g;                 // kv-head this q-head reads

      for (int i = 0; i < sq; ++i) {
        const int qp = q_lo + i;
        std::vector<float> z(sk);
        float row_max = -INFINITY;

        // scores z[j] = scale * <Q[qp], K[k_lo+j]>, with causal/padding mask
        for (int j = 0; j < sk; ++j) {
          if (causal && j > i) { z[j] = -INFINITY; continue; }
          float dot = 0.f;
          for (int e = 0; e < hd; ++e)
            dot += __bfloat162float(hQ[(qp * nqh + h) * hd + e])
                 * __bfloat162float(hK[((k_lo + j) * nkh + hk) * hd + e]);
          z[j] = dot * scale;
          row_max = fmaxf(row_max, z[j]);
        }

        // softmax over the row
        float sum = 0.f;
        for (int j = 0; j < sk; ++j) {
          if (z[j] == -INFINITY) { z[j] = 0.f; continue; }
          z[j] = expf(z[j] - row_max);
          sum += z[j];
        }
        if (sum == 0.f) continue;
        const float inv_sum = 1.f / sum;

        // O[qp] = (z @ V) / sum
        for (int e = 0; e < hd; ++e) {
          float acc = 0.f;
          for (int j = 0; j < sk; ++j)
            acc += z[j] * __bfloat162float(hV[((k_lo + j) * nkh + hk) * hd + e]);
          hO[(qp * nqh + h) * hd + e] = acc * inv_sum;
        }
      }
    }
  }
}

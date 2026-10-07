// 68_l2cache_policy.cuh -- createpolicy: build an L2 cache-hint policy token.
//
// ARCH: sm_80+
//
// Produces a 64-bit opaque policy register that downstream loads/stores
// consume via the .L2::cache_hint qualifier (see primitive 19's
// tma_load_2d_2sm_l2hint, primitive 26's cp_async_*_l2hint, etc.). The
// token alone has no effect; it must be paired with a hint-capable
// instruction.
//
// Common GEMM use case: `make_l2cache_policy_evict_last_full()` -- bias every
// hint-qualified access toward eviction-last so the working set stays in
// L2 across persistent-kernel iterations.
//
// Per-thread instruction. Cheap (one PTX op); standard pattern is to
// build once in the kernel prologue and pass through to the load warp.
// Source: knowledge/instructions/copy/createpolicy.md
// PTX:    9.7.10.21 (createpolicy)

#pragma once

#include <cstdint>

// =====================================================================
// Fractional policies (most common form for kernels)
//
// `fraction` in (0, 1] controls the probability that an access gets the
// primary priority; the remainder gets the secondary priority. If the
// secondary priority is omitted on the PTX side, it defaults to
// .L2::evict_unchanged.
// =====================================================================

// fractional.L2::evict_last.b64 policy, 1.0
//   Every hint-qualified access is biased toward eviction-last. Pair
//   with TMA loads in a persistent kernel that re-touches the same M/N
//   slabs across iterations.
__device__ __forceinline__
uint64_t make_l2cache_policy_evict_last_full() {
  uint64_t policy;
  asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n"
      : "=l"(policy));
  return policy;
}

// fractional.L2::evict_normal.b64 policy, 1.0
//   Default LRU. Equivalent to no hint at all; provided for completeness
//   and for callers that want to keep the .L2::cache_hint qualifier on
//   the consumer instruction without changing eviction semantics.
__device__ __forceinline__
uint64_t make_l2cache_policy_evict_normal_full() {
  uint64_t policy;
  asm("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;\n"
      : "=l"(policy));
  return policy;
}

// fractional.L2::evict_first.b64 policy, 1.0
//   Streaming reads -- evict before normal lines. Use for one-pass
//   scans where reuse is impossible.
__device__ __forceinline__
uint64_t make_l2cache_policy_evict_first_full() {
  uint64_t policy;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n"
      : "=l"(policy));
  return policy;
}

// fractional.L2::evict_unchanged.b64 policy, 1.0
//   Don't change eviction priority. Mostly useful as the *secondary*
//   priority; offered as primary for completeness.
__device__ __forceinline__
uint64_t make_l2cache_policy_evict_unchanged_full() {
  uint64_t policy;
  asm("createpolicy.fractional.L2::evict_unchanged.b64 %0, 1.0;\n"
      : "=l"(policy));
  return policy;
}

// Generic fractional builder. `fraction_f32` must be in (0.0, 1.0].
// Primary priority is the caller's choice (encoded in the asm template);
// see the named entry points above for the common cases.
__device__ __forceinline__
uint64_t make_l2cache_policy_fractional_evict_last_unchanged(float fraction_f32) {
  uint64_t policy;
  asm("createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64"
      " %0, %1;\n"
      : "=l"(policy)
      : "f"(fraction_f32));
  return policy;
}

__device__ __forceinline__
uint64_t make_l2cache_policy_fractional_evict_first_unchanged(float fraction_f32) {
  uint64_t policy;
  asm("createpolicy.fractional.L2::evict_first.L2::evict_unchanged.b64"
      " %0, %1;\n"
      : "=l"(policy)
      : "f"(fraction_f32));
  return policy;
}

// =====================================================================
// CUDA access-property -> policy
//
// When the host side built the policy via cudaStreamSetAttribute /
// cudaAccessProperty, the resulting 64-bit access-property handle is
// converted into a cache-policy token by createpolicy.cvt.L2.b64.
// =====================================================================

__device__ __forceinline__
uint64_t make_l2cache_policy_from_access_property(uint64_t access_property) {
  uint64_t policy;
  asm("createpolicy.cvt.L2.b64 %0, %1;\n"
      : "=l"(policy)
      : "l"(access_property));
  return policy;
}

// =====================================================================
// Compile-time menu dispatch -- index -> policy.
//
// Standard L2 cache hint menu used by kernel template params (e.g.
// blocks/88_load_warp_blackwell.cuh's L2CACHE_POLICY_A / _B):
//   0 = evict_last (default; working set stays in L2)
//   1 = evict_normal
//   2 = fractional evict_last/unchanged @ 0.5
//   3 = fractional evict_last/unchanged @ 0.25
//   4 = evict_unchanged (pure stream)
// `if constexpr` resolves at compile time -- no runtime dispatch.
// =====================================================================

template <int IDX>
__device__ __forceinline__
uint64_t make_l2cache_policy() {
  if constexpr (IDX == 0) return make_l2cache_policy_evict_last_full();
  else if constexpr (IDX == 1) return make_l2cache_policy_evict_normal_full();
  else if constexpr (IDX == 2) return make_l2cache_policy_fractional_evict_last_unchanged(0.5f);
  else if constexpr (IDX == 3) return make_l2cache_policy_fractional_evict_last_unchanged(0.25f);
  else return make_l2cache_policy_evict_unchanged_full();
}

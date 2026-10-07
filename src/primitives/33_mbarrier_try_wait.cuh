// 33_mbarrier_try_wait.cuh -- mbarrier.try_wait.parity + test_wait
//
// ARCH: sm_90a
//
// Consumer side of the mbarrier. try_wait.parity passes the expected phase
// parity (0 or 1); the variants below provide:
//   - single-attempt non-blocking checks (return predicate as uint32_t/bool)
//   - blocking spin loops (void return)
//   - cluster-scope and acquire-semantics flavors
//
// In pipelines, try_wait.parity is preferred because the consumer never
// calls arrive -- it tracks expected phase locally and flips its parity bit
// every cycle.
//
// test_wait uses the opaque state token returned by a prior mbarrier.arrive.

#pragma once

// PTX:    9.7.15.16.19 (mbarrier.test_wait / try_wait)
//
#include <cstdint>

// -- single-attempt try_wait --------------------------------------------------

// Single-attempt: returns 1 (done) / 0 (not yet) as uint32_t.
//
// The `"memory"` clobber gates consumer reads against producer writes
// (and vice versa) -- without it, ptxas can hoist memory ops across
// the wait and break the synchronization the wait is supposed to enforce.
__device__ __forceinline__
uint32_t mbarrier_try_wait_parity_once(uint32_t mbar_smem,
                                       uint32_t phase_parity) {
  uint32_t done;
  asm volatile(
    "{\n"
    ".reg .pred P1;\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
    "selp.b32 %0, 1, 0, P1;\n"
    "}\n"
    : "=r"(done) : "r"(mbar_smem), "r"(phase_parity) : "memory");
  return done;
}

// Single-attempt: returns true (done) / false (not yet) as bool.
__device__ __forceinline__
bool mbarrier_try_wait_parity(uint32_t mbar_smem, uint32_t phase_parity) {
  return mbarrier_try_wait_parity_once(mbar_smem, phase_parity) != 0;
}

// Single-attempt with .acquire semantics (consumer sees producer's writes).
__device__ __forceinline__
bool mbarrier_try_wait_parity_acquire(uint32_t mbar_smem,
                                      uint32_t phase_parity) {
  uint32_t done;
  asm volatile(
    "{\n"
    ".reg .pred P1;\n"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%1], %2;\n"
    "selp.b32 %0, 1, 0, P1;\n"
    "}\n"
    : "=r"(done) : "r"(mbar_smem), "r"(phase_parity) : "memory");
  return done != 0;
}

// -- blocking wait: loop until the mbarrier's phase flips ----------------------
// try_wait polls once (non-blocking); we loop it. Written as inline asm, NOT a C++
// `while (!try_wait()) {}`, because that C++ loop compiles with a YIELD each iteration
// (ptxas deschedules the warp mid-loop, so it reacts slowly to the flip). The asm below
// is a tight try_wait + backward branch with no YIELD. (pattern: CUTLASS arch/barrier.h)

// SUSPEND form: passes a timeout hint (10^7 ns; FA4/quack use 0x989680). On a poll miss
// the HW parks the warp in NANOSLEEP until the flip (or the hint elapses), freeing the
// SM's issue slots instead of busy-looping. Default for off-critical-path warps.
__device__ __forceinline__
void mbarrier_wait_parity_suspend(uint32_t mbar_smem, uint32_t phase_parity) {
  asm volatile(
    "{\n"
    ".reg .pred P1;\n"
    "LAB_WAIT:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, 10000000;\n"
    "@!P1 bra.uni LAB_WAIT;\n"
    "}\n"
    :: "r"(mbar_smem), "r"(phase_parity) : "memory");
}

// HOT form: same wait WITHOUT the suspend hint -- busy-spins, reacts at spin speed.
// Use only on a critical-path warp (FMHA mma); off-path warps suspend to yield issue
// slots. 
__device__ __forceinline__
void mbarrier_wait_parity(uint32_t mbar_smem, uint32_t phase_parity) {
  asm volatile(
    "{\n"
    ".reg .pred P1;\n"
    "LAB_WAIT_HOT:\n"
    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
    "@!P1 bra.uni LAB_WAIT_HOT;\n"
    "}\n"
    :: "r"(mbar_smem), "r"(phase_parity) : "memory");
}

// Cluster-scope blocking spin (mbarrier visible across cluster).
//
// TODO: unify with the cta-scope pattern -- rename to
// `mbarrier_wait_parity_cluster` and reduce the body to
// `while (!mbarrier_try_wait_parity_acquire_cluster(...)) { }`. Held
// off because the rename has zero callers today, so introducing a new
// name now is bloat without demand. Revisit when a real caller appears.
__device__ __forceinline__
void mbarrier_try_wait_parity_spin_cluster(uint32_t mbar_smem,
                                           uint32_t phase_parity) {
  uint32_t done;
  do {
    asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.b32 %0, 1, 0, p;\n\t"
      "}\n"
      : "=r"(done) : "r"(mbar_smem), "r"(phase_parity) : "memory");
  } while (!done);
}

// -- cluster-scope single-attempt variants -----------------------------------
// Non-blocking checks against a peer-CTA-visible mbarrier. Caller integrates
// them into custom polling logic (e.g. progress reporting, conditional
// fallback, multi-mbar arbitration).

// Single-attempt cluster scope (relaxed): returns 1/0 as uint32_t.
__device__ __forceinline__
uint32_t mbarrier_try_wait_parity_cluster_once(uint32_t mbar_smem,
                                               uint32_t phase_parity) {
  uint32_t done;
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "mbarrier.try_wait.parity.relaxed.cluster.shared::cta.b64 p, [%1], %2;\n\t"
    "selp.b32 %0, 1, 0, p;\n\t"
    "}\n"
    : "=r"(done) : "r"(mbar_smem), "r"(phase_parity) : "memory");
  return done;
}

// Single-attempt cluster scope (relaxed): bool flavor.
__device__ __forceinline__
bool mbarrier_try_wait_parity_cluster(uint32_t mbar_smem,
                                      uint32_t phase_parity) {
  return mbarrier_try_wait_parity_cluster_once(mbar_smem, phase_parity) != 0;
}

// Single-attempt cluster scope with .acquire semantics (consumer observes
// producer's writes published via mbarrier.arrive.release).
__device__ __forceinline__
bool mbarrier_try_wait_parity_acquire_cluster(uint32_t mbar_smem,
                                              uint32_t phase_parity) {
  uint32_t done;
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n\t"
    "selp.b32 %0, 1, 0, p;\n\t"
    "}\n"
    : "=r"(done) : "r"(mbar_smem), "r"(phase_parity) : "memory");
  return done != 0;
}

// -- try_wait.parity with suspendNanos hint ----------------------------------
// The optional `suspendNanos` operand on mbarrier.try_wait.parity tells the
// hardware scheduler that, on a "not yet" result, the warp may be suspended
// for up to N nanoseconds before retrying. This reduces SM occupancy and
// power consumption compared to a busy-spin in C++. Hint only; the hardware
// may suspend for less than the requested budget (or not at all). Useful for
// long expected waits where a small latency penalty is acceptable.
//
// Returns: 1 (done) / 0 (still waiting after suspend) as uint32_t. The wait
// is still single-attempt -- callers loop on the wrapper if they need
// blocking semantics with suspend-aware retry.

__device__ __forceinline__
uint32_t mbarrier_try_wait_parity_suspend_once(uint32_t mbar_smem,
                                                uint32_t phase_parity,
                                                uint32_t suspend_ns) {
  uint32_t done;
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2, %3;\n\t"
    "selp.b32 %0, 1, 0, p;\n\t"
    "}\n"
    : "=r"(done) : "r"(mbar_smem), "r"(phase_parity), "r"(suspend_ns) : "memory");
  return done;
}

// Bool flavor.
__device__ __forceinline__
bool mbarrier_try_wait_parity_suspend(uint32_t mbar_smem,
                                       uint32_t phase_parity,
                                       uint32_t suspend_ns) {
  return mbarrier_try_wait_parity_suspend_once(mbar_smem, phase_parity, suspend_ns) != 0;
}

// Blocking spin variant -- retries until done, suspending up to suspend_ns
// each attempt. Lower SM occupancy than mbarrier_wait_parity_suspend for long waits.
__device__ __forceinline__
void mbarrier_try_wait_parity_suspend_spin(uint32_t mbar_smem,
                                            uint32_t phase_parity,
                                            uint32_t suspend_ns) {
  while (!mbarrier_try_wait_parity_suspend_once(mbar_smem, phase_parity, suspend_ns)) { }
}

// -- test_wait via opaque state token ----------------------------------------

__device__ __forceinline__
bool mbarrier_test_wait(uint32_t mbar_smem, uint64_t state) {
  uint32_t done;
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "mbarrier.test_wait.shared::cta.b64 p, [%1], %2;\n\t"
    "selp.b32 %0, 1, 0, p;\n\t"
    "}\n"
    : "=r"(done) : "r"(mbar_smem), "l"(state) : "memory");
  return done != 0;
}

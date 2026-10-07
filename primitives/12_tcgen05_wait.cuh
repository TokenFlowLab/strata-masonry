#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 12_tcgen05_wait.cuh -- tcgen05.wait::{ld,st}.sync.aligned
//
// ARCH: sm_100a
//
// Thread-local wait: blocks until all prior tcgen05.ld (or .st) from this
// thread have retired. Does NOT wait for tcgen05.mma / .cp / .shift -- those
// go through tcgen05.commit + mbarrier (see #11).
//
// Typical use: issue tcgen05.ld, then tcgen05.wait::ld before consuming the
// register outputs.
//
// Source: knowledge/instructions/tmem/tcgen05_tmem.md
// PTX:    9.7.18.8.5 (tcgen05.wait)
//
__device__ __forceinline__ void tcgen05_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

// 50_atom_global.cuh -- atom.global.{add,min,max,cas,exch}.{u32,s32,f32,u64}
//
// ARCH: sm_90a
//
// Thin wrappers around atomic ops on GMEM. Returns the OLD value.
// Used for persistent scheduler tile counters, split-K accumulation.
// Bypasses the CUDA atomicAdd family which sometimes inserts extra sync
// that isn't needed for non-blocking counters.

#pragma once

// Source: knowledge/instructions/atomic/atom.md
// PTX:    9.7.15.5 (atom.global)
//
#include <cstdint>

// -- add ---------------------------------------------------------------------

__device__ __forceinline__
uint32_t atom_global_add_u32(uint32_t* addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.global.add.u32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__
int32_t atom_global_add_s32(int32_t* addr, int32_t v) {
  int32_t old;
  asm volatile("atom.global.add.s32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__
float atom_global_add_f32(float* addr, float v) {
  float old;
  asm volatile("atom.global.add.f32 %0, [%1], %2;\n"
               : "=f"(old) : "l"(addr), "f"(v) : "memory");
  return old;
}

__device__ __forceinline__
uint64_t atom_global_add_u64(uint64_t* addr, uint64_t v) {
  uint64_t old;
  asm volatile("atom.global.add.u64 %0, [%1], %2;\n"
               : "=l"(old) : "l"(addr), "l"(v) : "memory");
  return old;
}

// -- min / max ---------------------------------------------------------------

__device__ __forceinline__
uint32_t atom_global_min_u32(uint32_t* addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.global.min.u32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__
uint32_t atom_global_max_u32(uint32_t* addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.global.max.u32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

// -- exch / cas --------------------------------------------------------------

__device__ __forceinline__
uint32_t atom_global_exch_u32(uint32_t* addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.global.exch.b32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__
uint32_t atom_global_cas_u32(uint32_t* addr, uint32_t cmp, uint32_t v) {
  uint32_t old;
  asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;\n"
               : "=r"(old) : "l"(addr), "r"(cmp), "r"(v) : "memory");
  return old;
}

// -- sem / scope plumbing for u32 add ---------------------------------------
//
// PTX `atom` accepts {.relaxed,.acquire,.release,.acq_rel} x
// {.cta,.cluster,.gpu,.sys}. Default is .relaxed.gpu (what the wrappers
// above emit implicitly). Split-K accumulation typically uses
// .release.gpu (writer) + .acquire.gpu (reader). Cluster-scope variants
// are needed for cross-CTA coordination within a thread block cluster.
//
// Naming: atom_global_<op>_<sem>_<scope>_<type>(...). The four most
// common patterns are exposed below for u32 add; follow the same pattern
// for other ops (min/max/exch/cas) by demand.

__device__ __forceinline__
uint32_t atom_global_add_u32_relaxed_gpu(uint32_t* addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__
uint32_t atom_global_add_u32_acquire_gpu(uint32_t* addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__
uint32_t atom_global_add_u32_release_gpu(uint32_t* addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__
uint32_t atom_global_add_u32_acq_rel_gpu(uint32_t* addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

// CTA-scope: only synchronizes within one CTA. Useful for scratch
// counters that don't escape the block.
__device__ __forceinline__
uint32_t atom_global_add_u32_relaxed_cta(uint32_t* addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.relaxed.cta.global.add.u32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

// Cluster-scope: synchronizes across CTAs in the same thread block cluster.
// Required when multiple CTAs in a cluster share a counter.
__device__ __forceinline__
uint32_t atom_global_add_u32_relaxed_cluster(uint32_t* addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.relaxed.cluster.global.add.u32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

// SYS-scope: synchronizes across the whole system (multi-GPU + CPU).
// Heaviest; needed only for cross-device atomics.
__device__ __forceinline__
uint32_t atom_global_add_u32_relaxed_sys(uint32_t* addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.relaxed.sys.global.add.u32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

// -- sem variants for u32 cas (split-K reader-writer) -----------------------

__device__ __forceinline__
uint32_t atom_global_cas_u32_release_gpu(uint32_t* addr, uint32_t cmp, uint32_t v) {
  uint32_t old;
  asm volatile("atom.release.gpu.global.cas.b32 %0, [%1], %2, %3;\n"
               : "=r"(old) : "l"(addr), "r"(cmp), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__
uint32_t atom_global_cas_u32_acquire_gpu(uint32_t* addr, uint32_t cmp, uint32_t v) {
  uint32_t old;
  asm volatile("atom.acquire.gpu.global.cas.b32 %0, [%1], %2, %3;\n"
               : "=r"(old) : "l"(addr), "r"(cmp), "r"(v) : "memory");
  return old;
}

// -- signed integer min / max -----------------------------------------------

__device__ __forceinline__
int32_t atom_global_min_s32(int32_t* addr, int32_t v) {
  int32_t old;
  asm volatile("atom.global.min.s32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__
int32_t atom_global_max_s32(int32_t* addr, int32_t v) {
  int32_t old;
  asm volatile("atom.global.max.s32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

// -- bitwise (and / or / xor) -----------------------------------------------

__device__ __forceinline__
uint32_t atom_global_and_b32(uint32_t* addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.global.and.b32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__
uint32_t atom_global_or_b32(uint32_t* addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.global.or.b32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

__device__ __forceinline__
uint32_t atom_global_xor_b32(uint32_t* addr, uint32_t v) {
  uint32_t old;
  asm volatile("atom.global.xor.b32 %0, [%1], %2;\n"
               : "=r"(old) : "l"(addr), "r"(v) : "memory");
  return old;
}

// -- 64-bit cas --------------------------------------------------------------

__device__ __forceinline__
uint64_t atom_global_cas_u64(uint64_t* addr, uint64_t cmp, uint64_t v) {
  uint64_t old;
  asm volatile("atom.global.cas.b64 %0, [%1], %2, %3;\n"
               : "=l"(old) : "l"(addr), "l"(cmp), "l"(v) : "memory");
  return old;
}

// 36_fence_proxy_tensormap.cuh -- fence.proxy.tensormap::generic.{release,
//                                  acquire}.{cta,cluster,gpu,sys}
//
// ARCH: sm_90a
//
// Cross-proxy fence between generic-proxy writes to a tensormap (e.g.
// tensormap.replace) and subsequent TMA reads through the tensormap proxy.
// Required after tensormap.replace and before the next TMA load/store using
// the modified tensormap.
//
// Per PTX ISA 9.7.15.4:
//   .release form takes NO arguments  -- "fence.proxy.tensormap::generic.release.<scope>;"
//   .acquire form takes [addr], size  -- "fence.proxy.tensormap::generic.acquire.<scope> [tmap], 128;"

#pragma once

// PTX:    9.7.15.4 (fence.proxy.tensormap)
//
#include <cstdint>

// -- producer side: .release variants by scope --------------------------------

__device__ __forceinline__
void fence_proxy_tensormap_release_cta() {
  asm volatile("fence.proxy.tensormap::generic.release.cta;\n" ::: "memory");
}

__device__ __forceinline__
void fence_proxy_tensormap_release_cluster() {
  asm volatile("fence.proxy.tensormap::generic.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__
void fence_proxy_tensormap_release_gpu() {
  asm volatile("fence.proxy.tensormap::generic.release.gpu;\n" ::: "memory");
}

__device__ __forceinline__
void fence_proxy_tensormap_release_sys() {
  asm volatile("fence.proxy.tensormap::generic.release.sys;\n" ::: "memory");
}

// -- consumer side: .acquire (takes [addr], size = 128) -----------------------

__device__ __forceinline__
void fence_proxy_tensormap_acquire_gpu(const void* addr) {
  asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;\n"
               :: "l"(addr) : "memory");
}

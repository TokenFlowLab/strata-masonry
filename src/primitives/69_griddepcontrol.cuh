// 69_griddepcontrol.cuh -- griddepcontrol: grid-to-grid dependency control.
//
// ARCH: sm_90+
//
// Two actions, each a single-instruction PTX op:
//
//   griddepcontrol_wait()              -- dependent grid blocks until all
//                                          prerequisite grids in flight have
//                                          completed and their writes are
//                                          visible. Standard placement is at
//                                          sched-warp entry of a persistent
//                                          GEMM kernel.
//   griddepcontrol_launch_dependents() -- prerequisite grid signals that
//                                          dependent grids configured by the
//                                          launch attribute may begin. Take
//                                          effect grid-wide once all CTAs
//                                          have issued the instruction.
//
// Pairs with the host-side launch attribute
// `cudaLaunchAttributeProgrammaticStreamSerialization`. Without that
// attribute, both actions are functionally no-ops. With it, the dependent
// grid's setup (mbar init, tcgen05.alloc, etc.) overlaps with the
// prerequisite's tail (final TMA stores, dealloc).
//
// Per-thread instruction; no .sync.aligned. Idempotent within a CTA.
// PTX:    9.7.15.14 (griddepcontrol)

#pragma once

// Dependent-side wait. Blocks until all prerequisite grids in flight have
// completed and their memory writes are visible to this grid.
__device__ __forceinline__
void griddepcontrol_wait() {
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
}

// Prerequisite-side release. Allows dependent grids configured via the
// host-side programmatic-stream-serialization attribute to start. Issue
// from every CTA (or rely on early-exit threads to do it before exit) for
// the grid-wide effect. Pair with `fence.release.gpu` (or `.sys`) BEFORE
// this op to ensure the prerequisite's writes are visible to the dependent.
__device__ __forceinline__
void griddepcontrol_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
}

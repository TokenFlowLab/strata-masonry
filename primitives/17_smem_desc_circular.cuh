#pragma once
#if defined(PL_AGENTIC_SM103A)
// 17_smem_desc_circular.cuh -- 3xFP4 circular SMEM descriptor builder (composes #16)
//
// ARCH: sm_103a
//
// 3-block circular layout for FP4 K=96 operands. K=96 with FP4 = 48 bytes
// per K-slice, which does not fit within a single aligned 128B SMEM
// region. The hardware's 3xFP4 data-movement pattern stores the operand
// across THREE 128B-aligned blocks and uses the absolute-address SMEM
// descriptor (bit 52 = 1; see file 16) to point at TWO blocks per
// descriptor; consecutive MMAs walk the 3-block ring (start, next) =
// (b0, b1) -> (b1, b2) -> (b2, b0) -> (b0, b1) ...
//
// Issues no PTX. Composes file 16's `build_smem_desc_byte_addr` over the
// 3-buffer rotation. Reference: knowledge/building_blocks/smem_layout.md
// section 9.
//
// Issuer: any thread; output is fed into a later tcgen05.{mma,cp} via the
// k_loop_3xfp4 composite (file 75).
// Source: knowledge/instructions/mma/tcgen05_mma.md
// PTX:    9.7.18.4.1 (descriptor format), 9.7.18.3.1.2 (absolute address mode for K=48B)
//
#include "16_tmem_desc_byte_addr.cuh"

// Three 128B-aligned SMEM buffer base addresses for one operand in the
// 3xFP4 circular pattern. Each block holds one of the 3 K-tile slices.
struct CircularSmemBuffers {
    uint32_t buf[3];   // SMEM addresses; each must be 128-byte aligned.
    int stride_bytes;  // Stride-dim byte offset (same for all 3 blocks).
};

// Build the SMEM descriptor for one MMA issue at a given K-phase.
//
// The hardware reads from `buf[(phase) % 3]` with `next_start = buf[(phase+1) % 3]`
// for the first half, then wraps. Pass the K-phase index so we select the
// correct (start, next) pair. K-phases run 0..7 in an 8-phase k_loop_3xfp4
// composite (file 75); phases that select the same (start, next) pair share
// the same descriptor.
__device__ __forceinline__ uint64_t build_circular_smem_desc(
    CircularSmemBuffers const& bufs, int phase)
{
    int idx0 = phase % 3;
    int idx1 = (phase + 1) % 3;
    return build_smem_desc_byte_addr(bufs.buf[idx0], bufs.buf[idx1], bufs.stride_bytes);
}

// Convenience: build all 3 unique (start, next) descriptor variants up-front
// (the 8-phase loop alternates among only 3 distinct descriptors).
//
//   out[0] = (buf[0], buf[1])
//   out[1] = (buf[1], buf[2])
//   out[2] = (buf[2], buf[0])
__device__ __forceinline__ void build_circular_smem_desc_set(
    CircularSmemBuffers const& bufs, uint64_t out[3])
{
    out[0] = build_smem_desc_byte_addr(bufs.buf[0], bufs.buf[1], bufs.stride_bytes);
    out[1] = build_smem_desc_byte_addr(bufs.buf[1], bufs.buf[2], bufs.stride_bytes);
    out[2] = build_smem_desc_byte_addr(bufs.buf[2], bufs.buf[0], bufs.stride_bytes);
}

// Map an MMA phase index in [0, 8) to its descriptor index in [0, 3).
// 8 phases x 96 K-elements / 32 K-per-block = 3 blocks rotating.
__device__ __forceinline__ int circular_phase_to_desc_index(int phase) {
    // Each consecutive phase advances by 1 in the 3-block ring.
    return phase % 3;
}

#endif  // PL_AGENTIC_SM103A

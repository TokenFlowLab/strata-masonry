// 128_epi_smem_pingpong.cuh -- Double-buffered epilogue SMEM management
//
// ARCH: sm_90a
//
// Manages two alternating SMEM buffers for the epilogue. Producer writes
// into one buffer while the TMA store drains the other; swap roles each
// sub-tile.
//
// Two flavors:
//   EpiSmemPingpong            two pre-known SMEM bases, simple swap.
//   EpiPingPong<EPI_BYTES>     one base + size, derives second base, also
//                              tracks outstanding TMA stores via the bulk
//                              async-group so callers can drain on exit.

#pragma once

// Source: knowledge/building_blocks/epi_warp.md
// PTX:    9.7.10.28.5.3 (TMA store), 9.7.15.1 (bar.sync), 9.7.15.4 (fence.proxy.async)
//
#include <cstdint>
#include "../primitives/25_tma_async_group.cuh"

// =============================================================================
// Simple two-base ping-pong
// =============================================================================

struct EpiSmemPingpong {
  uint32_t smem_base[2];   // SMEM addresses of the two buffers
  int slot;                 // current active slot (0 or 1)

  __device__ __forceinline__
  void init(uint32_t base0, uint32_t base1) {
    smem_base[0] = base0;
    smem_base[1] = base1;
    slot = 0;
  }

  __device__ __forceinline__
  uint32_t current() const { return smem_base[slot]; }

  __device__ __forceinline__
  uint32_t other() const { return smem_base[1 - slot]; }

  __device__ __forceinline__
  void swap() { slot = 1 - slot; }
};

// =============================================================================
// Templated ping-pong with TMA-store tracking
// =============================================================================

template <int EPI_SMEM_BYTES>
struct EpiPingPong {
  uint32_t smem_buf[2];       // two SMEM base addresses
  int current_stage;
  int outstanding_stores;

  __device__ __forceinline__
  EpiPingPong(uint32_t smem_base) : current_stage(0), outstanding_stores(0) {
    smem_buf[0] = smem_base;
    smem_buf[1] = smem_base + EPI_SMEM_BYTES;
  }

  // SMEM address for the current (write) stage.
  __device__ __forceinline__
  uint32_t write_addr() const { return smem_buf[current_stage]; }

  // Wait for the write stage to be free (no outstanding TMA store on it).
  __device__ __forceinline__
  void wait_write_ready() {
    if (outstanding_stores >= 2) {
      tma_store_wait_group<1>();  // leave 1 in-flight
      outstanding_stores = 1;
    }
  }

  // Commit the current stage's store and swap.
  __device__ __forceinline__
  void commit_and_swap() {
    tma_store_commit_group();
    outstanding_stores++;
    current_stage ^= 1;
  }

  // Drain all outstanding stores.
  __device__ __forceinline__
  void drain() {
    tma_store_wait_group<0>();
    outstanding_stores = 0;
  }
};

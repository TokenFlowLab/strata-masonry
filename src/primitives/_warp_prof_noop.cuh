#pragma once

#include <cstdint>
#include <cuda_runtime.h>

enum WpRegion : uint32_t {
  WP_LOAD_WAIT = 0, WP_LOAD_WAIT_THROTTLE = 1, WP_LOAD_ISSUE = 2,
  WP_MMA_TMEM_2CTA_ALLOC = 3, WP_MMA_WAIT_FULL = 4, WP_MMA_WAIT_ACC = 5, WP_MMA_ISSUE = 6,
  WP_MMA_TMEM_2CTA_FREE = 7,
  WP_EPI_WAIT_TMEM = 8, WP_EPI_WAIT_ACC = 9, WP_EPI_TMEM_LD = 10, WP_EPI_STORE = 11,
  WP_EPI_WAIT_STORE = 12,
  WP_SCHED_WAIT_THROTTLE = 13, WP_SCHED_WAIT_CLC = 14, WP_SCHED_ISSUE = 15, WP_CLC_FETCH = 16,
  WP_EPI_SWIGLU = 17, WP_LOAD_GATHER = 18, WP_PREFETCH_IDX = 19, WP_PREFETCH_L2 = 20,
  WP_USER0 = 21, WP_USER1 = 22,
  WP_SM_WAIT_S = 23, WP_SM_SOFTMAX = 24, WP_CORR_WAIT = 25, WP_CORR_EPI = 27,
  WP_SM_WAIT_SCALE = 28, WP_MMA_COMMIT = 29, WP_SM_STORE_P = 30, WP_MMA_WAIT_P = 31,
  WP_MMA_WAIT_FULL_K = 32, WP_MMA_WAIT_FULL_V = 33, WP_MMA_WAIT_FULL_Q = 34,
  WP_LOAD_ISSUE_K = 35, WP_LOAD_ISSUE_V = 36, WP_LOAD_ISSUE_Q = 37, WP_SM_SCALE = 38,
  WP_CORR_READ_ALPHA = 39, WP_CORR_O_SCALE = 40, WP_SM_READ_L = 41, WP_ITEM = 42, WP_ITER = 43,
  WP_SM_ROWSUM = 44, WP_MMA_WAIT_FULL_RING = 45,
  WP_SM_WAIT_FULL_LSE = 46, WP_SM_WAIT_FULL_DELTA = 47, WP_SM_WAIT_FULL_DPT = 48,
  WP_SM_STORE_DST = 49, WP_SM_WAIT_FULL_DV = 50, WP_SM_WAIT_FULL_DK = 51, WP_SM_STORE_DV = 52,
  WP_SM_STORE_DK = 53, WP_MMA_WAIT_EMPTY_DQ = 54, WP_MMA_WAIT_FULL_PT = 55,
  WP_MMA_WAIT_EMPTY_EPI = 56, WP_LOAD_WAIT_EMPTY_KV = 57, WP_LOAD_WAIT_EMPTY_RING = 58,
  WP_LOAD_WAIT_EMPTY_LSE = 59, WP_LOAD_WAIT_EMPTY_DELTA = 60, WP_LOAD_WAIT_EMPTY_EPI = 61,
  WP_LOAD_ISSUE_RING = 62, WP_LOAD_ISSUE_LSE = 63, WP_LOAD_ISSUE_DELTA = 64,
  WP_MMA_WAIT_FULL_DST = 65, WP_SM_WAIT_FULL_ST = 66, WP_SM_STORE_PT = 67,
  WP_EPI_WAIT_FULL_DQ = 68, WP_EPI_LOAD_DQ = 69, WP_EPI_STORE_DQ = 70,
  WP_LOAD_WAIT_EMPTY_Q = 71, WP_LOAD_WAIT_EMPTY_DO = 72, WP_MMA_WAIT_FULL_DO = 73,
  WP_EPI_STAGE_DQ = 74, WP_EPI_SYNC_DQ_READY = 75, WP_EPI_REDUCE_DQ_ISSUE = 76,
  WP_EPI_REDUCE_DQ_WAIT = 77, WP_EPI_SYNC_DQ_REUSE = 78, WP_N = 79
};

struct WpCtx {
  unsigned long long* slot;
  unsigned cnt;
};

__device__ __forceinline__ void wp_init() {}
__device__ __forceinline__ void wp_begin(WpRegion) {}
__device__ __forceinline__ void wp_end(WpRegion) {}
__device__ __forceinline__ void wp_flush() {}
__device__ __forceinline__ WpCtx wp_ctx_init() { WpCtx c; c.slot = nullptr; c.cnt = 0; return c; }
__device__ __forceinline__ void wp_marker(WpCtx&, WpRegion, int) {}
__device__ __forceinline__ void wp_begin(WpCtx&, WpRegion) {}
__device__ __forceinline__ void wp_end(WpCtx&, WpRegion) {}
__device__ __forceinline__ void wp_flush(WpCtx&) {}

struct WpBuffer {};
inline WpBuffer wp_alloc(dim3, int = 0) { return {}; }
inline int wp_env_sample() { return 0; }
inline unsigned wp_view_block(unsigned blk) { return blk; }
inline void wp_reset(WpBuffer&) {}
inline void wp_readback(WpBuffer&) {}
inline void wp_free(WpBuffer&) {}
inline void wp_print_busy(const WpBuffer&, const char**, int, unsigned, double = 0.0) {}
inline void wp_dump_raw(const WpBuffer&, const char*, unsigned, unsigned = 0, double = 0.0) {}

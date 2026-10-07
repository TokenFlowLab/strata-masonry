#if defined(PL_AGENTIC_SM103A)
// ARCH: sm_103a
// 66_tcgen05_ld_red_test.cu -- all 24 tcgen05.ld.red wrappers x2..x128: st -> ld.red -> bit-check regs + redval vs CPU

#include <algorithm>
#include <vector>
#include "test_utils.cuh"
#include "../src/primitives/0_tcgen05_alloc.cuh"
#include "../src/primitives/1_tcgen05_dealloc.cuh"
#include "../src/primitives/2_tcgen05_relinquish.cuh"
#include "../src/primitives/10_tcgen05_st.cuh"
#include "../src/primitives/12_tcgen05_wait.cuh"
#include "../src/primitives/66_tcgen05_ld_red.cuh"

constexpr int THREADS = 32;
constexpr int TMEM_COLUMNS = 256;
constexpr int WIDTH_COUNT = 7;

enum class ValueKind { F32, U32, S32 };

#define LOAD_REDUCE_OP(SUFFIX, VALUE, KIND, IS_MAX, IS_ABS, PROPAGATES_NAN)              \
  struct load_reduce_##SUFFIX {                                                          \
    using Value = VALUE;                                                                 \
    static constexpr const char* name = #SUFFIX;                                         \
    static constexpr ValueKind kind = KIND;                                              \
    static constexpr bool is_max = IS_MAX;                                               \
    static constexpr bool is_abs = IS_ABS;                                               \
    static constexpr bool propagates_nan = PROPAGATES_NAN;                               \
    template <int N, bool SPLIT>                                                         \
    static __device__ __forceinline__ void run(uint32_t taddr, Value& redval,            \
                                               uint32_t (&r)[N]) {                       \
      if constexpr (SPLIT) tcgen05_ld_red_16x32bx2_##SUFFIX<N, N>(taddr, redval, r);     \
      else tcgen05_ld_red_32x32b_##SUFFIX<N>(taddr, redval, r);                          \
    }                                                                                    \
  };

LOAD_REDUCE_OP(min_f32,         float,    ValueKind::F32, false, false, false)
LOAD_REDUCE_OP(max_f32,         float,    ValueKind::F32, true,  false, false)
LOAD_REDUCE_OP(min_abs_f32,     float,    ValueKind::F32, false, true,  false)
LOAD_REDUCE_OP(max_abs_f32,     float,    ValueKind::F32, true,  true,  false)
LOAD_REDUCE_OP(min_NaN_f32,     float,    ValueKind::F32, false, false, true)
LOAD_REDUCE_OP(max_NaN_f32,     float,    ValueKind::F32, true,  false, true)
LOAD_REDUCE_OP(min_abs_NaN_f32, float,    ValueKind::F32, false, true,  true)
LOAD_REDUCE_OP(max_abs_NaN_f32, float,    ValueKind::F32, true,  true,  true)
LOAD_REDUCE_OP(min_u32,         uint32_t, ValueKind::U32, false, false, false)
LOAD_REDUCE_OP(max_u32,         uint32_t, ValueKind::U32, true,  false, false)
LOAD_REDUCE_OP(min_s32,         int32_t,  ValueKind::S32, false, false, false)
LOAD_REDUCE_OP(max_s32,         int32_t,  ValueKind::S32, true,  false, false)

#undef LOAD_REDUCE_OP

template <int N, bool SPLIT>
__device__ __forceinline__ void store_columns(uint32_t taddr, const uint32_t (&w)[N]) {
  if constexpr (SPLIT) {
    if constexpr (N == 2)       tcgen05_st_16x32bx2_x2<N>(taddr, w[0], w[1]);
    else if constexpr (N == 4)  tcgen05_st_16x32bx2_x4<N>(taddr, w);
    else if constexpr (N == 8)  tcgen05_st_16x32bx2_x8<N>(taddr, w);
    else if constexpr (N == 16) tcgen05_st_16x32bx2_x16<N>(taddr, w);
    else if constexpr (N == 32) tcgen05_st_16x32bx2_x32<N>(taddr, w);
    else if constexpr (N == 64) tcgen05_st_16x32bx2_x64<N>(taddr, w);
    else                        tcgen05_st_16x32bx2_x128<N>(taddr, w);
  } else {
    if constexpr (N == 2)       tcgen05_st_32x32b_x2(taddr, w[0], w[1]);
    else if constexpr (N == 4)  tcgen05_st_32x32b_x4(taddr, w);
    else if constexpr (N == 8)  tcgen05_st_32x32b_x8(taddr, w);
    else if constexpr (N == 16) tcgen05_st_32x32b_x16(taddr, w);
    else if constexpr (N == 32) tcgen05_st_32x32b_x32(taddr, w);
    else if constexpr (N == 64) tcgen05_st_32x32b_x64(taddr, w);
    else                        tcgen05_st_32x32b_x128(taddr, w);
  }
}

__device__ __forceinline__ uint32_t value_bits(float v)    { return __float_as_uint(v); }
__device__ __forceinline__ uint32_t value_bits(uint32_t v) { return v; }
__device__ __forceinline__ uint32_t value_bits(int32_t v)  { return static_cast<uint32_t>(v); }

template <typename Op, int N, bool SPLIT>
__global__ void k_ld_red(const uint32_t* g_input, uint32_t* g_loaded, uint32_t* g_redval,
                         typename Op::Value redval_init) {
  __shared__ __align__(16) uint32_t slot;
  if (threadIdx.x == 0) slot = 0;
  __syncthreads();
  tcgen05_alloc<1>(smem_ptr_u32(&slot), TMEM_COLUMNS);
  tcgen05_relinquish_alloc_permit<1>();
  __syncthreads();
  uint32_t tbase = slot;

  uint32_t values[N];
  #pragma unroll
  for (int j = 0; j < N; ++j) values[j] = g_input[threadIdx.x * N + j];
  store_columns<N, SPLIT>(tbase, values);
  tcgen05_wait_st();
  __syncthreads();

  uint32_t r[N];
  typename Op::Value redval = redval_init;
  Op::template run<N, SPLIT>(tbase, redval, r);
  tcgen05_wait_ld();
  #pragma unroll
  for (int j = 0; j < N; ++j) g_loaded[threadIdx.x * N + j] = r[j];
  g_redval[threadIdx.x] = value_bits(redval);

  __syncthreads();
  tcgen05_dealloc<1>(tbase, TMEM_COLUMNS);
}

static uint32_t mix_bits(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

static float float_from_bits(uint32_t bits) { float x; std::memcpy(&x, &bits, sizeof x); return x; }
static uint32_t bits_from_float(float x) { uint32_t b; std::memcpy(&b, &x, sizeof b); return b; }

template <typename Op>
static uint32_t input_word(uint32_t seed, int thread, int column, int width) {
  uint32_t hash = mix_bits(seed * 0x9e3779b9u + thread * 4096u + column);
  if constexpr (Op::kind != ValueKind::F32) {
    return hash;
  } else {
    if (thread % 8 == 3 && column == (int)((thread * 5u + seed) % (uint32_t)width))
      return 0x7fc00000u | (uint32_t)thread;
    return bits_from_float((float)((int)(hash % 2001u) - 1000) + 0.5f);
  }
}

template <typename Op>
static typename Op::Value dominating_redval() {
  if constexpr (Op::kind == ValueKind::F32) {
    if (Op::is_max) return INFINITY;
    return Op::is_abs ? 0.0f : -INFINITY;
  } else if constexpr (Op::kind == ValueKind::U32) {
    return Op::is_max ? 0xffffffffu : 0u;
  } else {
    return Op::is_max ? INT32_MAX : INT32_MIN;
  }
}

template <typename Op>
static uint32_t reference_reduce(const uint32_t* words, int width) {
  if constexpr (Op::kind == ValueKind::F32) {
    bool saw_nan = false, have_value = false;
    float best = 0.0f;
    for (int j = 0; j < width; ++j) {
      float x = float_from_bits(words[j]);
      if (std::isnan(x)) { saw_nan = true; continue; }
      if (Op::is_abs) x = std::fabs(x);
      if (!have_value || (Op::is_max ? x > best : x < best)) { best = x; have_value = true; }
    }
    if (Op::propagates_nan && saw_nan) return 0x7fffffffu;
    return bits_from_float(best);
  } else if constexpr (Op::kind == ValueKind::U32) {
    uint32_t best = words[0];
    for (int j = 1; j < width; ++j)
      best = Op::is_max ? std::max(best, words[j]) : std::min(best, words[j]);
    return best;
  } else {
    int32_t best = static_cast<int32_t>(words[0]);
    for (int j = 1; j < width; ++j) {
      int32_t x = static_cast<int32_t>(words[j]);
      best = Op::is_max ? std::max(best, x) : std::min(best, x);
    }
    return static_cast<uint32_t>(best);
  }
}

template <typename Op, int N, bool SPLIT>
static bool run_case(uint32_t seed) {
  const char* shape = SPLIT ? "16x32bx2" : "32x32b";
  const uint32_t case_seed = seed * 1000u + N;
  std::vector<uint32_t> input(THREADS * N), loaded(THREADS * N), redval(THREADS);
  for (int t = 0; t < THREADS; ++t)
    for (int j = 0; j < N; ++j) input[t * N + j] = input_word<Op>(case_seed, t, j, N);

  uint32_t *d_input = nullptr, *d_loaded = nullptr, *d_redval = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_loaded, loaded.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_redval, redval.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  k_ld_red<Op, N, SPLIT><<<1, THREADS>>>(d_input, d_loaded, d_redval, dominating_redval<Op>());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(loaded.data(), d_loaded, loaded.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(redval.data(), d_redval, redval.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  cudaFree(d_input); cudaFree(d_loaded); cudaFree(d_redval);

  int load_mismatches = 0, reduce_mismatches = 0;
  for (int t = 0; t < THREADS; ++t) {
    for (int j = 0; j < N; ++j) {
      if (loaded[t * N + j] == input[t * N + j]) continue;
      if (load_mismatches++ < 4)
        printf("    %s.x%d %s: thread %d r[%d] = 0x%08x, want 0x%08x\n",
               shape, N, Op::name, t, j, loaded[t * N + j], input[t * N + j]);
    }
    const uint32_t want = reference_reduce<Op>(&input[t * N], N);
    if (redval[t] == want) continue;
    if (reduce_mismatches++ < 4)
      printf("    %s.x%d %s: thread %d redval = 0x%08x, want 0x%08x\n",
             shape, N, Op::name, t, redval[t], want);
  }
  if (load_mismatches || reduce_mismatches)
    printf("    %s.x%d %s: FAIL (%d register, %d redval mismatches)\n",
           shape, N, Op::name, load_mismatches, reduce_mismatches);
  return load_mismatches == 0 && reduce_mismatches == 0;
}

template <typename Op, bool SPLIT>
static int run_wrapper(uint32_t seed) {
  int failed = 0;
  failed += !run_case<Op, 2, SPLIT>(seed);
  failed += !run_case<Op, 4, SPLIT>(seed);
  failed += !run_case<Op, 8, SPLIT>(seed);
  failed += !run_case<Op, 16, SPLIT>(seed);
  failed += !run_case<Op, 32, SPLIT>(seed);
  failed += !run_case<Op, 64, SPLIT>(seed);
  failed += !run_case<Op, 128, SPLIT>(seed);
  printf("  [%s] tcgen05_ld_red_%s_%s x2..x128: %d/%d widths\n", failed ? "FAIL" : "PASS",
         SPLIT ? "16x32bx2" : "32x32b", Op::name, WIDTH_COUNT - failed, WIDTH_COUNT);
  return failed;
}

template <typename Op>
static int run_both_shapes(uint32_t seed) {
  return run_wrapper<Op, false>(2 * seed) + run_wrapper<Op, true>(2 * seed + 1);
}

int main() {
  int failed = 0;
  failed += run_both_shapes<load_reduce_min_f32>(1);
  failed += run_both_shapes<load_reduce_max_f32>(2);
  failed += run_both_shapes<load_reduce_min_abs_f32>(3);
  failed += run_both_shapes<load_reduce_max_abs_f32>(4);
  failed += run_both_shapes<load_reduce_min_NaN_f32>(5);
  failed += run_both_shapes<load_reduce_max_NaN_f32>(6);
  failed += run_both_shapes<load_reduce_min_abs_NaN_f32>(7);
  failed += run_both_shapes<load_reduce_max_abs_NaN_f32>(8);
  failed += run_both_shapes<load_reduce_min_u32>(9);
  failed += run_both_shapes<load_reduce_max_u32>(10);
  failed += run_both_shapes<load_reduce_min_s32>(11);
  failed += run_both_shapes<load_reduce_max_s32>(12);

  const int total = 12 * 2 * WIDTH_COUNT;
  printf("tcgen05.ld.red: %d/%d cases passed\n", total - failed, total);
  if (failed) FAIL("%d tcgen05.ld.red case(s) mismatched", failed);
  PASS();
}

#endif  // PL_AGENTIC_SM103A

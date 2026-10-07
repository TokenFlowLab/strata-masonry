#pragma once
#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// 66_tcgen05_ld_red.cuh -- tcgen05.ld.red.sync.aligned.<shape>.<num>
//                          .redOp{.abs}{.NaN}.f32           (form 1, f32)
//                          .redOp.{u32,s32}                 (form 2, integer)
//
// ARCH: sm_103a+ (renamed from sm_101a; PTX ISA 8.8+).
//       On sm_100a builds the wrappers compile to a stub that zeros r[]
//       and ignores taddr/redval. UNTESTED on sm_100a hardware (no .red
//       support); the fallback is provided so kernels that conditionally
//       use these wrappers behind their own arch guards still compile.
//
// TODO: runtime-test on sm_103a hardware once available. Both the sm_103a
//       (real asm) and sm_100a (fallback) paths have been verified to
//       compile cleanly, and the sm_103a PTX output is correct, but no
//       runtime smoke test has been executed on real sm_103a silicon yet.
//       When sm_103a hardware becomes accessible, add a test under
//       tests/66_tcgen05_ld_red_test.cu following the pattern of
//       tests/9_tcgen05_ld_test.cu (write known values to TMEM, run each
//       wrapper, verify the loaded regs and the reduced redval against
//       hand-computed expected values).
//
// PTX 9.7.18.8.3 (.ld.red form).
//
// Coverage matrix (Y = wrapper provided; - = combination not defined by PTX).
// .x1 is NOT a legal .num for the reduction form (ptxas rejects it); the
// regular tcgen05.ld in primitive 9 has the .x1 wrappers.
//
//   shape    | x1 | x2 | x4 | x8 | x16 | x32 | x64 | x128
//   ---------|----|----|----|----|-----|-----|-----|-----
//   32x32b   |  - |  Y |  Y |  Y |  Y  |  Y  |  Y  |  Y
//   16x32bx2 |  - |  Y |  Y |  Y |  Y  |  Y  |  Y  |  Y    (+ immHalfSplitoff)
//
// Per-shape op/modifier/type combos (Y = wrapper provided):
//
//   redOp / modifier / type  | provided
//   -------------------------|---------
//   min .f32                 |  Y
//   max .f32                 |  Y
//   min .abs .f32            |  Y
//   max .abs .f32            |  Y
//   min .NaN .f32            |  Y
//   max .NaN .f32            |  Y
//   min .abs .NaN .f32       |  Y
//   max .abs .NaN .f32       |  Y
//   min .u32                 |  Y
//   max .u32                 |  Y
//   min .s32                 |  Y
//   max .s32                 |  Y
//
// 12 op/modifier/type combos x 2 shapes = 24 templated functions.
// Each templated function handles all 7 legal .num levels (x2..x128) via
// if-constexpr.  Total asm bodies in source = 24 * 7 = 168.
//
// Caller usage:
//   uint32_t regs[32]; float redv = -INFINITY;
//   tcgen05_ld_red_32x32b_max_f32<32>(taddr, redv, regs);
//   tcgen05_ld_red_32x32b_max_abs_NaN_f32<32>(taddr, redv, regs);
//
//   tcgen05_ld_red_16x32bx2_max_f32<32, /*IMM=*/16>(taddr, redv, regs);
//
//   uint32_t iregs[16]; uint32_t imax = 0;
//   tcgen05_ld_red_32x32b_max_u32<16>(taddr, imax, iregs);
//
// Issuer: every thread in the warp executes the same instruction
// collectively (.sync.aligned). All lanes must pass the same taddr.
//

#include <cstdint>

// =========================================================================
// Generator macros: each invocation emits ONE templated function with
// 8 if-constexpr branches (one per .num level). Macro args supply the
// bits that vary per (op, modifier, type, shape):
//
//   NAME       : token pasted into function name, e.g. `max_abs_NaN`
//   OP_PTX     : asm string for .redOp,  e.g. ".max"
//   ABS_PTX    : asm string for .abs,    e.g. ".abs" or ""
//   NAN_PTX    : asm string for .NaN,    e.g. ".NaN" or ""
//   TYPE_TAG   : token + stringified, used in name and asm, e.g. f32
//   TYPE_C     : C type for redval parameter, e.g. float
//   TYPE_CONSTR: asm constraint for redval (read-modify-write), e.g. "+f"
//
// Per-N branch: the asm regs list `{%0, ..., %(N-1)}` and operand indices
// for redval, taddr (and IMM for 16x32bx2) are spelled out inline -- no
// helper macros for them. The C++ preprocessor concatenates the OP_PTX,
// ABS_PTX, NAN_PTX, etc. string literals into a single asm string at
// compile time.
// =========================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1030

// -------------------------------------------------------------------------
// Fallback (sm_100a or earlier device pass): tcgen05.ld.red is unsupported.
// Stub body zeros r[] and ignores taddr/redval. The function NAMES match
// the real macros so any caller that includes this header compiles cleanly;
// at runtime on sm_100a the wrapper returns zeroed regs.
// -------------------------------------------------------------------------

#define DEFN_LDRED_32X32B(NAME, OP_PTX, ABS_PTX, NAN_PTX,                                  \
                          TYPE_TAG, TYPE_C, TYPE_CONSTR)                                    \
template <int N>                                                                             \
__device__ __forceinline__ void                                                              \
tcgen05_ld_red_32x32b_##NAME##_##TYPE_TAG(                                                   \
    uint32_t taddr, TYPE_C& redval, uint32_t (&r)[N]) {                                      \
  static_assert(N==2||N==4||N==8||N==16||N==32||N==64||N==128,                               \
                "tcgen05.ld.red .num must be x2/x4/x8/x16/x32/x64/x128 "                     \
                "(.x1 not legal for the reduction form per PTX 9.7.18.8.3)");                \
  (void)taddr; (void)redval;                                                                  \
  for (int _i = 0; _i < N; ++_i) r[_i] = 0;                                                   \
}

#define DEFN_LDRED_16X32BX2(NAME, OP_PTX, ABS_PTX, NAN_PTX,                                \
                            TYPE_TAG, TYPE_C, TYPE_CONSTR)                                  \
template <int N, int IMM>                                                                    \
__device__ __forceinline__ void                                                              \
tcgen05_ld_red_16x32bx2_##NAME##_##TYPE_TAG(                                                 \
    uint32_t taddr, TYPE_C& redval, uint32_t (&r)[N]) {                                      \
  static_assert(N==2||N==4||N==8||N==16||N==32||N==64||N==128,                               \
                "tcgen05.ld.red .num must be x2/x4/x8/x16/x32/x64/x128 "                     \
                "(.x1 not legal for the reduction form per PTX 9.7.18.8.3)");                \
  (void)taddr; (void)redval; (void)IMM;                                                       \
  for (int _i = 0; _i < N; ++_i) r[_i] = 0;                                                   \
}

#else

// -------------------------------------------------------------------------
// Real (sm_103a+ or host pass): emit the 8-branch if-constexpr chain.
// Each branch contains an asm volatile with the literal reg list and
// operand indices for that .num level. The shape/op/modifier/type bits
// come in via the macro args and are inserted as adjacent string literals
// (preprocessor-concatenated into a single asm string at compile time).
// -------------------------------------------------------------------------

#define DEFN_LDRED_32X32B(NAME, OP_PTX, ABS_PTX, NAN_PTX,                                   \
                          TYPE_TAG, TYPE_C, TYPE_CONSTR)                                    \
template <int N>                                                                             \
__device__ __forceinline__ void                                                              \
tcgen05_ld_red_32x32b_##NAME##_##TYPE_TAG(                                                   \
    uint32_t taddr, TYPE_C& redval, uint32_t (&r)[N]) {                                      \
  static_assert(N==2||N==4||N==8||N==16||N==32||N==64||N==128,                               \
                "tcgen05.ld.red .num must be x2/x4/x8/x16/x32/x64/x128 "                     \
                "(.x1 not legal for the reduction form per PTX 9.7.18.8.3)");                \
  if constexpr (N == 2) {                                                                     \
    asm volatile(                                                                              \
      "tcgen05.ld.red.sync.aligned.32x32b.x2" OP_PTX ABS_PTX NAN_PTX "." #TYPE_TAG " "      \
      "{%0, %1}, %2, [%3];\n"                                                                  \
      : "=r"(r[0]), "=r"(r[1]),                                                                \
        TYPE_CONSTR(redval) : "r"(taddr));                                                     \
  } else if constexpr (N == 4) {                                                               \
    asm volatile(                                                                              \
      "tcgen05.ld.red.sync.aligned.32x32b.x4" OP_PTX ABS_PTX NAN_PTX "." #TYPE_TAG " "      \
      "{%0, %1, %2, %3}, %4, [%5];\n"                                                          \
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),                                        \
        TYPE_CONSTR(redval) : "r"(taddr));                                                     \
  } else if constexpr (N == 8) {                                                               \
    asm volatile(                                                                              \
      "tcgen05.ld.red.sync.aligned.32x32b.x8" OP_PTX ABS_PTX NAN_PTX "." #TYPE_TAG " "      \
      "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9];\n"                                          \
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),                                        \
        "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),                                        \
        TYPE_CONSTR(redval) : "r"(taddr));                                                     \
  } else if constexpr (N == 16) {                                                              \
    asm volatile(                                                                              \
      "tcgen05.ld.red.sync.aligned.32x32b.x16" OP_PTX ABS_PTX NAN_PTX "." #TYPE_TAG " "     \
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "                                              \
       "%8,  %9,  %10, %11, %12, %13, %14, %15}, "                                             \
      "%16, [%17];\n"                                                                          \
      : "=r"(r[ 0]), "=r"(r[ 1]), "=r"(r[ 2]), "=r"(r[ 3]),                                    \
        "=r"(r[ 4]), "=r"(r[ 5]), "=r"(r[ 6]), "=r"(r[ 7]),                                    \
        "=r"(r[ 8]), "=r"(r[ 9]), "=r"(r[10]), "=r"(r[11]),                                    \
        "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),                                    \
        TYPE_CONSTR(redval) : "r"(taddr));                                                     \
  } else if constexpr (N == 32) {                                                              \
    asm volatile(                                                                              \
      "tcgen05.ld.red.sync.aligned.32x32b.x32" OP_PTX ABS_PTX NAN_PTX "." #TYPE_TAG " "     \
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "                                              \
       "%8,  %9,  %10, %11, %12, %13, %14, %15, "                                              \
       "%16, %17, %18, %19, %20, %21, %22, %23, "                                              \
       "%24, %25, %26, %27, %28, %29, %30, %31}, "                                             \
      "%32, [%33];\n"                                                                          \
      : "=r"(r[ 0]), "=r"(r[ 1]), "=r"(r[ 2]), "=r"(r[ 3]),                                    \
        "=r"(r[ 4]), "=r"(r[ 5]), "=r"(r[ 6]), "=r"(r[ 7]),                                    \
        "=r"(r[ 8]), "=r"(r[ 9]), "=r"(r[10]), "=r"(r[11]),                                    \
        "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),                                    \
        "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),                                    \
        "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),                                    \
        "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),                                    \
        "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]),                                    \
        TYPE_CONSTR(redval) : "r"(taddr));                                                     \
  } else if constexpr (N == 64) {                                                              \
    asm volatile(                                                                              \
      "tcgen05.ld.red.sync.aligned.32x32b.x64" OP_PTX ABS_PTX NAN_PTX "." #TYPE_TAG " "     \
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "                                              \
       "%8,  %9,  %10, %11, %12, %13, %14, %15, "                                              \
       "%16, %17, %18, %19, %20, %21, %22, %23, "                                              \
       "%24, %25, %26, %27, %28, %29, %30, %31, "                                              \
       "%32, %33, %34, %35, %36, %37, %38, %39, "                                              \
       "%40, %41, %42, %43, %44, %45, %46, %47, "                                              \
       "%48, %49, %50, %51, %52, %53, %54, %55, "                                              \
       "%56, %57, %58, %59, %60, %61, %62, %63}, "                                             \
      "%64, [%65];\n"                                                                          \
      : "=r"(r[ 0]), "=r"(r[ 1]), "=r"(r[ 2]), "=r"(r[ 3]),                                    \
        "=r"(r[ 4]), "=r"(r[ 5]), "=r"(r[ 6]), "=r"(r[ 7]),                                    \
        "=r"(r[ 8]), "=r"(r[ 9]), "=r"(r[10]), "=r"(r[11]),                                    \
        "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),                                    \
        "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),                                    \
        "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),                                    \
        "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),                                    \
        "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]),                                    \
        "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]),                                    \
        "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]),                                    \
        "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]),                                    \
        "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]),                                    \
        "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]),                                    \
        "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]),                                    \
        "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]),                                    \
        "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63]),                                    \
        TYPE_CONSTR(redval) : "r"(taddr));                                                     \
  } else if constexpr (N == 128) {                                                             \
    asm volatile(                                                                              \
      "tcgen05.ld.red.sync.aligned.32x32b.x128" OP_PTX ABS_PTX NAN_PTX "." #TYPE_TAG " "    \
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "                                     \
       "%8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "                                      \
       "%16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "                                      \
       "%24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "                                      \
       "%32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "                                      \
       "%40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "                                      \
       "%48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "                                      \
       "%56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "                                      \
       "%64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "                                      \
       "%72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "                                      \
       "%80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "                                      \
       "%88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "                                      \
       "%96,  %97,  %98,  %99,  %100, %101, %102, %103, "                                      \
       "%104, %105, %106, %107, %108, %109, %110, %111, "                                      \
       "%112, %113, %114, %115, %116, %117, %118, %119, "                                      \
       "%120, %121, %122, %123, %124, %125, %126, %127}, "                                     \
      "%128, [%129];\n"                                                                        \
      : "=r"(r[  0]), "=r"(r[  1]), "=r"(r[  2]), "=r"(r[  3]),                                \
        "=r"(r[  4]), "=r"(r[  5]), "=r"(r[  6]), "=r"(r[  7]),                                \
        "=r"(r[  8]), "=r"(r[  9]), "=r"(r[ 10]), "=r"(r[ 11]),                                \
        "=r"(r[ 12]), "=r"(r[ 13]), "=r"(r[ 14]), "=r"(r[ 15]),                                \
        "=r"(r[ 16]), "=r"(r[ 17]), "=r"(r[ 18]), "=r"(r[ 19]),                                \
        "=r"(r[ 20]), "=r"(r[ 21]), "=r"(r[ 22]), "=r"(r[ 23]),                                \
        "=r"(r[ 24]), "=r"(r[ 25]), "=r"(r[ 26]), "=r"(r[ 27]),                                \
        "=r"(r[ 28]), "=r"(r[ 29]), "=r"(r[ 30]), "=r"(r[ 31]),                                \
        "=r"(r[ 32]), "=r"(r[ 33]), "=r"(r[ 34]), "=r"(r[ 35]),                                \
        "=r"(r[ 36]), "=r"(r[ 37]), "=r"(r[ 38]), "=r"(r[ 39]),                                \
        "=r"(r[ 40]), "=r"(r[ 41]), "=r"(r[ 42]), "=r"(r[ 43]),                                \
        "=r"(r[ 44]), "=r"(r[ 45]), "=r"(r[ 46]), "=r"(r[ 47]),                                \
        "=r"(r[ 48]), "=r"(r[ 49]), "=r"(r[ 50]), "=r"(r[ 51]),                                \
        "=r"(r[ 52]), "=r"(r[ 53]), "=r"(r[ 54]), "=r"(r[ 55]),                                \
        "=r"(r[ 56]), "=r"(r[ 57]), "=r"(r[ 58]), "=r"(r[ 59]),                                \
        "=r"(r[ 60]), "=r"(r[ 61]), "=r"(r[ 62]), "=r"(r[ 63]),                                \
        "=r"(r[ 64]), "=r"(r[ 65]), "=r"(r[ 66]), "=r"(r[ 67]),                                \
        "=r"(r[ 68]), "=r"(r[ 69]), "=r"(r[ 70]), "=r"(r[ 71]),                                \
        "=r"(r[ 72]), "=r"(r[ 73]), "=r"(r[ 74]), "=r"(r[ 75]),                                \
        "=r"(r[ 76]), "=r"(r[ 77]), "=r"(r[ 78]), "=r"(r[ 79]),                                \
        "=r"(r[ 80]), "=r"(r[ 81]), "=r"(r[ 82]), "=r"(r[ 83]),                                \
        "=r"(r[ 84]), "=r"(r[ 85]), "=r"(r[ 86]), "=r"(r[ 87]),                                \
        "=r"(r[ 88]), "=r"(r[ 89]), "=r"(r[ 90]), "=r"(r[ 91]),                                \
        "=r"(r[ 92]), "=r"(r[ 93]), "=r"(r[ 94]), "=r"(r[ 95]),                                \
        "=r"(r[ 96]), "=r"(r[ 97]), "=r"(r[ 98]), "=r"(r[ 99]),                                \
        "=r"(r[100]), "=r"(r[101]), "=r"(r[102]), "=r"(r[103]),                                \
        "=r"(r[104]), "=r"(r[105]), "=r"(r[106]), "=r"(r[107]),                                \
        "=r"(r[108]), "=r"(r[109]), "=r"(r[110]), "=r"(r[111]),                                \
        "=r"(r[112]), "=r"(r[113]), "=r"(r[114]), "=r"(r[115]),                                \
        "=r"(r[116]), "=r"(r[117]), "=r"(r[118]), "=r"(r[119]),                                \
        "=r"(r[120]), "=r"(r[121]), "=r"(r[122]), "=r"(r[123]),                                \
        "=r"(r[124]), "=r"(r[125]), "=r"(r[126]), "=r"(r[127]),                                \
        TYPE_CONSTR(redval) : "r"(taddr));                                                     \
  }                                                                                             \
}

// -------------------------------------------------------------------------
// 16x32bx2 shape: same 8-branch chain, plus an extra `, %imm` operand and
// `template <int IMM>` parameter (immHalfSplitoff is encoded as an asm
// immediate via "n"(IMM)).
// -------------------------------------------------------------------------

#define DEFN_LDRED_16X32BX2(NAME, OP_PTX, ABS_PTX, NAN_PTX,                                 \
                            TYPE_TAG, TYPE_C, TYPE_CONSTR)                                  \
template <int N, int IMM>                                                                    \
__device__ __forceinline__ void                                                              \
tcgen05_ld_red_16x32bx2_##NAME##_##TYPE_TAG(                                                 \
    uint32_t taddr, TYPE_C& redval, uint32_t (&r)[N]) {                                      \
  static_assert(N==2||N==4||N==8||N==16||N==32||N==64||N==128,                               \
                "tcgen05.ld.red .num must be x2/x4/x8/x16/x32/x64/x128 "                     \
                "(.x1 not legal for the reduction form per PTX 9.7.18.8.3)");                \
  if constexpr (N == 2) {                                                                     \
    asm volatile(                                                                              \
      "tcgen05.ld.red.sync.aligned.16x32bx2.x2" OP_PTX ABS_PTX NAN_PTX "." #TYPE_TAG " "    \
      "{%0, %1}, %2, [%3], %4;\n"                                                              \
      : "=r"(r[0]), "=r"(r[1]),                                                                \
        TYPE_CONSTR(redval) : "r"(taddr), "n"(IMM));                                           \
  } else if constexpr (N == 4) {                                                               \
    asm volatile(                                                                              \
      "tcgen05.ld.red.sync.aligned.16x32bx2.x4" OP_PTX ABS_PTX NAN_PTX "." #TYPE_TAG " "    \
      "{%0, %1, %2, %3}, %4, [%5], %6;\n"                                                      \
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),                                        \
        TYPE_CONSTR(redval) : "r"(taddr), "n"(IMM));                                           \
  } else if constexpr (N == 8) {                                                               \
    asm volatile(                                                                              \
      "tcgen05.ld.red.sync.aligned.16x32bx2.x8" OP_PTX ABS_PTX NAN_PTX "." #TYPE_TAG " "    \
      "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, [%9], %10;\n"                                     \
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),                                        \
        "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),                                        \
        TYPE_CONSTR(redval) : "r"(taddr), "n"(IMM));                                           \
  } else if constexpr (N == 16) {                                                              \
    asm volatile(                                                                              \
      "tcgen05.ld.red.sync.aligned.16x32bx2.x16" OP_PTX ABS_PTX NAN_PTX "." #TYPE_TAG " "   \
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "                                              \
       "%8,  %9,  %10, %11, %12, %13, %14, %15}, "                                             \
      "%16, [%17], %18;\n"                                                                     \
      : "=r"(r[ 0]), "=r"(r[ 1]), "=r"(r[ 2]), "=r"(r[ 3]),                                    \
        "=r"(r[ 4]), "=r"(r[ 5]), "=r"(r[ 6]), "=r"(r[ 7]),                                    \
        "=r"(r[ 8]), "=r"(r[ 9]), "=r"(r[10]), "=r"(r[11]),                                    \
        "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),                                    \
        TYPE_CONSTR(redval) : "r"(taddr), "n"(IMM));                                           \
  } else if constexpr (N == 32) {                                                              \
    asm volatile(                                                                              \
      "tcgen05.ld.red.sync.aligned.16x32bx2.x32" OP_PTX ABS_PTX NAN_PTX "." #TYPE_TAG " "   \
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "                                              \
       "%8,  %9,  %10, %11, %12, %13, %14, %15, "                                              \
       "%16, %17, %18, %19, %20, %21, %22, %23, "                                              \
       "%24, %25, %26, %27, %28, %29, %30, %31}, "                                             \
      "%32, [%33], %34;\n"                                                                     \
      : "=r"(r[ 0]), "=r"(r[ 1]), "=r"(r[ 2]), "=r"(r[ 3]),                                    \
        "=r"(r[ 4]), "=r"(r[ 5]), "=r"(r[ 6]), "=r"(r[ 7]),                                    \
        "=r"(r[ 8]), "=r"(r[ 9]), "=r"(r[10]), "=r"(r[11]),                                    \
        "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),                                    \
        "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),                                    \
        "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),                                    \
        "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),                                    \
        "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]),                                    \
        TYPE_CONSTR(redval) : "r"(taddr), "n"(IMM));                                           \
  } else if constexpr (N == 64) {                                                              \
    asm volatile(                                                                              \
      "tcgen05.ld.red.sync.aligned.16x32bx2.x64" OP_PTX ABS_PTX NAN_PTX "." #TYPE_TAG " "   \
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "                                              \
       "%8,  %9,  %10, %11, %12, %13, %14, %15, "                                              \
       "%16, %17, %18, %19, %20, %21, %22, %23, "                                              \
       "%24, %25, %26, %27, %28, %29, %30, %31, "                                              \
       "%32, %33, %34, %35, %36, %37, %38, %39, "                                              \
       "%40, %41, %42, %43, %44, %45, %46, %47, "                                              \
       "%48, %49, %50, %51, %52, %53, %54, %55, "                                              \
       "%56, %57, %58, %59, %60, %61, %62, %63}, "                                             \
      "%64, [%65], %66;\n"                                                                     \
      : "=r"(r[ 0]), "=r"(r[ 1]), "=r"(r[ 2]), "=r"(r[ 3]),                                    \
        "=r"(r[ 4]), "=r"(r[ 5]), "=r"(r[ 6]), "=r"(r[ 7]),                                    \
        "=r"(r[ 8]), "=r"(r[ 9]), "=r"(r[10]), "=r"(r[11]),                                    \
        "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),                                    \
        "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),                                    \
        "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),                                    \
        "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),                                    \
        "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]),                                    \
        "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]),                                    \
        "=r"(r[36]), "=r"(r[37]), "=r"(r[38]), "=r"(r[39]),                                    \
        "=r"(r[40]), "=r"(r[41]), "=r"(r[42]), "=r"(r[43]),                                    \
        "=r"(r[44]), "=r"(r[45]), "=r"(r[46]), "=r"(r[47]),                                    \
        "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]),                                    \
        "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]),                                    \
        "=r"(r[56]), "=r"(r[57]), "=r"(r[58]), "=r"(r[59]),                                    \
        "=r"(r[60]), "=r"(r[61]), "=r"(r[62]), "=r"(r[63]),                                    \
        TYPE_CONSTR(redval) : "r"(taddr), "n"(IMM));                                           \
  } else if constexpr (N == 128) {                                                             \
    asm volatile(                                                                              \
      "tcgen05.ld.red.sync.aligned.16x32bx2.x128" OP_PTX ABS_PTX NAN_PTX "." #TYPE_TAG " "  \
      "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "                                     \
       "%8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "                                      \
       "%16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "                                      \
       "%24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "                                      \
       "%32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "                                      \
       "%40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "                                      \
       "%48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "                                      \
       "%56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "                                      \
       "%64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "                                      \
       "%72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "                                      \
       "%80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "                                      \
       "%88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "                                      \
       "%96,  %97,  %98,  %99,  %100, %101, %102, %103, "                                      \
       "%104, %105, %106, %107, %108, %109, %110, %111, "                                      \
       "%112, %113, %114, %115, %116, %117, %118, %119, "                                      \
       "%120, %121, %122, %123, %124, %125, %126, %127}, "                                     \
      "%128, [%129], %130;\n"                                                                  \
      : "=r"(r[  0]), "=r"(r[  1]), "=r"(r[  2]), "=r"(r[  3]),                                \
        "=r"(r[  4]), "=r"(r[  5]), "=r"(r[  6]), "=r"(r[  7]),                                \
        "=r"(r[  8]), "=r"(r[  9]), "=r"(r[ 10]), "=r"(r[ 11]),                                \
        "=r"(r[ 12]), "=r"(r[ 13]), "=r"(r[ 14]), "=r"(r[ 15]),                                \
        "=r"(r[ 16]), "=r"(r[ 17]), "=r"(r[ 18]), "=r"(r[ 19]),                                \
        "=r"(r[ 20]), "=r"(r[ 21]), "=r"(r[ 22]), "=r"(r[ 23]),                                \
        "=r"(r[ 24]), "=r"(r[ 25]), "=r"(r[ 26]), "=r"(r[ 27]),                                \
        "=r"(r[ 28]), "=r"(r[ 29]), "=r"(r[ 30]), "=r"(r[ 31]),                                \
        "=r"(r[ 32]), "=r"(r[ 33]), "=r"(r[ 34]), "=r"(r[ 35]),                                \
        "=r"(r[ 36]), "=r"(r[ 37]), "=r"(r[ 38]), "=r"(r[ 39]),                                \
        "=r"(r[ 40]), "=r"(r[ 41]), "=r"(r[ 42]), "=r"(r[ 43]),                                \
        "=r"(r[ 44]), "=r"(r[ 45]), "=r"(r[ 46]), "=r"(r[ 47]),                                \
        "=r"(r[ 48]), "=r"(r[ 49]), "=r"(r[ 50]), "=r"(r[ 51]),                                \
        "=r"(r[ 52]), "=r"(r[ 53]), "=r"(r[ 54]), "=r"(r[ 55]),                                \
        "=r"(r[ 56]), "=r"(r[ 57]), "=r"(r[ 58]), "=r"(r[ 59]),                                \
        "=r"(r[ 60]), "=r"(r[ 61]), "=r"(r[ 62]), "=r"(r[ 63]),                                \
        "=r"(r[ 64]), "=r"(r[ 65]), "=r"(r[ 66]), "=r"(r[ 67]),                                \
        "=r"(r[ 68]), "=r"(r[ 69]), "=r"(r[ 70]), "=r"(r[ 71]),                                \
        "=r"(r[ 72]), "=r"(r[ 73]), "=r"(r[ 74]), "=r"(r[ 75]),                                \
        "=r"(r[ 76]), "=r"(r[ 77]), "=r"(r[ 78]), "=r"(r[ 79]),                                \
        "=r"(r[ 80]), "=r"(r[ 81]), "=r"(r[ 82]), "=r"(r[ 83]),                                \
        "=r"(r[ 84]), "=r"(r[ 85]), "=r"(r[ 86]), "=r"(r[ 87]),                                \
        "=r"(r[ 88]), "=r"(r[ 89]), "=r"(r[ 90]), "=r"(r[ 91]),                                \
        "=r"(r[ 92]), "=r"(r[ 93]), "=r"(r[ 94]), "=r"(r[ 95]),                                \
        "=r"(r[ 96]), "=r"(r[ 97]), "=r"(r[ 98]), "=r"(r[ 99]),                                \
        "=r"(r[100]), "=r"(r[101]), "=r"(r[102]), "=r"(r[103]),                                \
        "=r"(r[104]), "=r"(r[105]), "=r"(r[106]), "=r"(r[107]),                                \
        "=r"(r[108]), "=r"(r[109]), "=r"(r[110]), "=r"(r[111]),                                \
        "=r"(r[112]), "=r"(r[113]), "=r"(r[114]), "=r"(r[115]),                                \
        "=r"(r[116]), "=r"(r[117]), "=r"(r[118]), "=r"(r[119]),                                \
        "=r"(r[120]), "=r"(r[121]), "=r"(r[122]), "=r"(r[123]),                                \
        "=r"(r[124]), "=r"(r[125]), "=r"(r[126]), "=r"(r[127]),                                \
        TYPE_CONSTR(redval) : "r"(taddr), "n"(IMM));                                           \
  }                                                                                             \
}

#endif  // __CUDA_ARCH__ guard

// =========================================================================
// Wrapper invocations: 24 templated functions (12 per shape).
// Each invocation expands to one templated function with all 8 .num
// branches inside.
// =========================================================================

// --- 32x32b f32: 8 op x modifier combos ---
DEFN_LDRED_32X32B(min,         ".min", "",     "",     f32, float,    "+f")
DEFN_LDRED_32X32B(max,         ".max", "",     "",     f32, float,    "+f")
DEFN_LDRED_32X32B(min_abs,     ".min", ".abs", "",     f32, float,    "+f")
DEFN_LDRED_32X32B(max_abs,     ".max", ".abs", "",     f32, float,    "+f")
DEFN_LDRED_32X32B(min_NaN,     ".min", "",     ".NaN", f32, float,    "+f")
DEFN_LDRED_32X32B(max_NaN,     ".max", "",     ".NaN", f32, float,    "+f")
DEFN_LDRED_32X32B(min_abs_NaN, ".min", ".abs", ".NaN", f32, float,    "+f")
DEFN_LDRED_32X32B(max_abs_NaN, ".max", ".abs", ".NaN", f32, float,    "+f")

// --- 32x32b u32 / s32: no abs / no NaN ---
DEFN_LDRED_32X32B(min, ".min", "", "", u32, uint32_t, "+r")
DEFN_LDRED_32X32B(max, ".max", "", "", u32, uint32_t, "+r")
DEFN_LDRED_32X32B(min, ".min", "", "", s32, int32_t,  "+r")
DEFN_LDRED_32X32B(max, ".max", "", "", s32, int32_t,  "+r")

// --- 16x32bx2 f32: 8 op x modifier combos ---
DEFN_LDRED_16X32BX2(min,         ".min", "",     "",     f32, float,    "+f")
DEFN_LDRED_16X32BX2(max,         ".max", "",     "",     f32, float,    "+f")
DEFN_LDRED_16X32BX2(min_abs,     ".min", ".abs", "",     f32, float,    "+f")
DEFN_LDRED_16X32BX2(max_abs,     ".max", ".abs", "",     f32, float,    "+f")
DEFN_LDRED_16X32BX2(min_NaN,     ".min", "",     ".NaN", f32, float,    "+f")
DEFN_LDRED_16X32BX2(max_NaN,     ".max", "",     ".NaN", f32, float,    "+f")
DEFN_LDRED_16X32BX2(min_abs_NaN, ".min", ".abs", ".NaN", f32, float,    "+f")
DEFN_LDRED_16X32BX2(max_abs_NaN, ".max", ".abs", ".NaN", f32, float,    "+f")

// --- 16x32bx2 u32 / s32: no abs / no NaN ---
DEFN_LDRED_16X32BX2(min, ".min", "", "", u32, uint32_t, "+r")
DEFN_LDRED_16X32BX2(max, ".max", "", "", u32, uint32_t, "+r")
DEFN_LDRED_16X32BX2(min, ".min", "", "", s32, int32_t,  "+r")
DEFN_LDRED_16X32BX2(max, ".max", "", "", s32, int32_t,  "+r")

// Don't pollute the namespace with the generator macro names.
#undef DEFN_LDRED_32X32B
#undef DEFN_LDRED_16X32BX2

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

#if defined(PL_AGENTIC_SM100A) || defined(PL_AGENTIC_SM103A)
// ARCH: sm_100a
// 42_smem_desc_blackwell_test.cu -- host-side bit-pattern verification for
// the Blackwell SMEM descriptor builder.

#include "test_utils.cuh"
#include "../primitives/42_smem_desc_blackwell.cuh"

// Verify all 5 swizzle modes encode into bits [61:64) of the descriptor
// per PTX 9.7.18.4.1 Table 51:
//   None=0, B128_32atom=1, B128=2, B64=4, B32=6.
static int check_swizzle(const char* label,
                         SmemSwizzleBlackwell mode, uint32_t expected) {
  uint32_t addr = 0x4000;
  uint32_t sbo  = 256;
  uint32_t lbo  = 128;
  uint64_t d = build_smem_desc_blackwell(addr, sbo, lbo, mode);
  uint32_t got = static_cast<uint32_t>((d >> 61) & 0x7);
  printf("  %s: encoded swizzle bits = %u (want %u)\n", label, got, expected);
  if (got != expected) {
    FAIL("swizzle mode encoding mismatch");
    return 1;
  }
  // Also verify the lower-bit fields are unchanged when the swizzle changes.
  uint64_t want_low = 0;
  want_low |= static_cast<uint64_t>((addr >> 4) & 0x3FFF);
  want_low |= static_cast<uint64_t>((lbo >> 4)  & 0x3FFF) << 16;
  want_low |= static_cast<uint64_t>((sbo >> 4)  & 0x3FFF) << 32;
  want_low |= static_cast<uint64_t>(1) << 46;
  if ((d & ((1ULL << 53) - 1)) != want_low) {
    FAIL("non-swizzle bits drifted across modes");
    return 1;
  }
  return 0;
}

int main() {
  uint32_t addr = 0x4000;
  uint32_t sbo  = 256;
  uint32_t lbo  = 128;
  uint64_t d = build_smem_desc_blackwell(
      addr, sbo, lbo, SmemSwizzleBlackwell::B128);

  uint64_t want = 0;
  want |= static_cast<uint64_t>((addr >> 4) & 0x3FFF);
  want |= static_cast<uint64_t>((lbo >> 4)  & 0x3FFF) << 16;
  want |= static_cast<uint64_t>((sbo >> 4)  & 0x3FFF) << 32;
  want |= static_cast<uint64_t>(1) << 46;        // version
  want |= static_cast<uint64_t>(2) << 61;        // B128

  printf("smem_desc_blackwell(addr=0x%x,sbo=%u,lbo=%u,B128):\n", addr, sbo, lbo);
  printf("  got  = 0x%016llx\n", (unsigned long long)d);
  printf("  want = 0x%016llx\n", (unsigned long long)want);
  if (d != want) FAIL("blackwell descriptor bit-pattern mismatch");

  uint64_t da = build_smem_desc_blackwell_abs(addr, sbo, 0x8000,
                                                SmemSwizzleBlackwell::B128);
  if ((da & (1ULL << 52)) == 0) FAIL("absolute-mode bit not set");

  // All 5 swizzle modes round-trip through the [61:64) field.
  printf("swizzle-mode coverage:\n");
  check_swizzle("None        ", SmemSwizzleBlackwell::None,        0);
  check_swizzle("B128_32atom ", SmemSwizzleBlackwell::B128_32atom, 1);
  check_swizzle("B128        ", SmemSwizzleBlackwell::B128,        2);
  check_swizzle("B64         ", SmemSwizzleBlackwell::B64,         4);
  check_swizzle("B32         ", SmemSwizzleBlackwell::B32,         6);

  PASS();
}

#endif  // PL_AGENTIC_SM100A || PL_AGENTIC_SM103A

#pragma once
#if defined(PL_AGENTIC_SM103A)
// 6_tcgen05_mma_fp4_k96.cuh -- tcgen05.mma.kind::{mxf4,mxf4nvf4}.block_scale.block{16,32} (K=96 atom)
//
// ARCH: sm_103a
//
// Block-scaled FP4 MMA with K=96 instruction descriptor encoding. Same PTX
// mnemonic as the K=64/128 mxf4 path (file 5); the K=96 atom is selected
// purely by setting bit 31 of the 32-bit instruction descriptor (PTX ISA
// Table 55). Atom shape is 128x128x96 -- one MMA consumes 96 K elements
// per pass. Caller passes pre-built idesc + 64-bit SMEM descriptors.
//
// Constraints (PTX ISA 9.7.18):
//   M in {128, 256}; N a multiple of 8 in [8, 512]
//   transpose A/B forbidden (absolute-address SMEM desc requires K-major)
//   scale-factor matrices live in TMEM (caller passes addresses)
//
// Issuer: one thread per CTA (cta_group::1) or per CTA-pair (cta_group::2).
// Source: knowledge/instructions/mma/tcgen05_mma.md
// PTX:    9.7.18.10.10.1 (mma syntax + .block_scale.scale_vec K=96), 9.7.18.10.7.2.4 (block32 K=96 SF A), 9.7.18.10.7.3.4 (block32 K=96 SF B)
//
#include <cstdint>

// ---------------------------------------------------------------------------
// K=96 instruction-descriptor builder for kind::mxf4 / mxf4nvf4 (Table 55).
// ---------------------------------------------------------------------------

// Build the 32-bit instruction descriptor for K=96 MMA atom.
//
//   M, N         : MMA shape; M must be 128 or 256 (encoded as M >> 7).
//   sf_a_data_id : Matrix A scale factor data ID (0 or 2). Bits [29:30].
//   sf_b_data_id : Matrix B scale factor data ID (0 or 2). Bits [4:5].
//   ue8m0        : true -> UE8M0 scale type (default for mxf4); false -> UE4M3
//                  (only legal with mxf4nvf4). Bit 23.
//   negate_a/_b  : negate A/B (bits 13/14).
//   sparse       : sparsity bit (bit 2). Default dense.
template <int M, int N>
__device__ __forceinline__ uint32_t build_idesc_mxf4_k96(
    int sf_a_data_id = 0,
    int sf_b_data_id = 0,
    bool ue8m0 = true,
    bool negate_a = false,
    bool negate_b = false,
    bool sparse = false)
{
    static_assert(M == 128 || M == 256, "K=96 MMA atom requires M in {128, 256}");
    static_assert((N % 8) == 0 && N >= 8 && N <= 512, "N must be multiple of 8 in [8, 512]");
    uint32_t d = 0;
    d |= (sparse ? 1u : 0u) << 2;                                 // bit  2: sparsity
    d |= static_cast<uint32_t>(sf_b_data_id & 0x3) << 4;          // bits 4-5: B SF data ID
    d |= 1u << 7;                                                 // bits 7-9: atype = E2M1 (1)
    d |= 1u << 10;                                                // bits 10-11: btype = 1 (E2M1)
    d |= (negate_a ? 1u : 0u) << 13;                              // bit 13: negate A
    d |= (negate_b ? 1u : 0u) << 14;                              // bit 14: negate B
    // bits 15-16 (transpose A/B) must be 0 -- absolute address mode (3xFP4)
    // requires K-major and forbids transpose. Caller should not flip A/B layout
    // for the K=96 path.
    d |= (static_cast<uint32_t>(N) >> 3) << 17;                   // bits 17-22: N >> 3
    d |= (ue8m0 ? 1u : 0u) << 23;                                 // bit 23: scale type (1=UE8M0)
    d |= (static_cast<uint32_t>(M) >> 7) << 27;                   // bits 27-28: M >> 7
    d |= static_cast<uint32_t>(sf_a_data_id & 0x3) << 29;         // bits 29-30: A SF data ID
    d |= 1u << 31;                                                // bit 31: K=96
    return d;
}

// ---------------------------------------------------------------------------
// MMA wrapper -- block-scaled FP4 with K=96 atom.
// ---------------------------------------------------------------------------

enum class MmaMxf4Variant : int {
    MXF4_BLOCK32     = 0,  // .kind::mxf4    .block_scale.block32  (scale_vec::2X)
    MXF4NVF4_BLOCK16 = 1,  // .kind::mxf4nvf4 .block_scale.block16  (scale_vec::4X)
    MXF4NVF4_BLOCK32 = 2,  // .kind::mxf4nvf4 .block_scale.block32  (scale_vec::2X)
};

// Issue one block-scaled FP4 MMA (K=96 atom). Asynchronous; completion is
// observed via tcgen05.commit + mbarrier.try_wait.parity (file 11/33).
//
//   tmem_c           : 32-bit TMEM addr for accumulator D
//   desc_a, desc_b   : 64-bit SMEM descriptors (file 17 for circular variant)
//   idesc            : 32-bit instruction descriptor (build_idesc_mxf4_k96)
//   scale_a, scale_b : 32-bit TMEM addrs for scale_A / scale_B matrices
//   enable_input_d   : if false, D = A*B (drops accumulator); if true, D = A*B + D
template <int CtaGroup, MmaMxf4Variant V>
__device__ __forceinline__ void tcgen05_mma_fp4_k96(
    uint32_t tmem_c,
    uint64_t desc_a,
    uint64_t desc_b,
    uint32_t idesc,
    uint32_t scale_a,
    uint32_t scale_b,
    bool enable_input_d)
{
    static_assert(CtaGroup == 1 || CtaGroup == 2, "cta_group must be 1 or 2");
    uint32_t pred_imm = enable_input_d ? 1u : 0u;
    if constexpr (CtaGroup == 1 && V == MmaMxf4Variant::MXF4_BLOCK32) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 "
            "[%0], %1, %2, %3, [%5], [%6], p;\n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
               "r"(pred_imm), "r"(scale_a), "r"(scale_b));
    } else if constexpr (CtaGroup == 1 && V == MmaMxf4Variant::MXF4NVF4_BLOCK16) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
            "[%0], %1, %2, %3, [%5], [%6], p;\n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
               "r"(pred_imm), "r"(scale_a), "r"(scale_b));
    } else if constexpr (CtaGroup == 1 && V == MmaMxf4Variant::MXF4NVF4_BLOCK32) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block32 "
            "[%0], %1, %2, %3, [%5], [%6], p;\n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
               "r"(pred_imm), "r"(scale_a), "r"(scale_b));
    } else if constexpr (CtaGroup == 2 && V == MmaMxf4Variant::MXF4_BLOCK32) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.block32 "
            "[%0], %1, %2, %3, [%5], [%6], p;\n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
               "r"(pred_imm), "r"(scale_a), "r"(scale_b));
    } else if constexpr (CtaGroup == 2 && V == MmaMxf4Variant::MXF4NVF4_BLOCK16) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 "
            "[%0], %1, %2, %3, [%5], [%6], p;\n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
               "r"(pred_imm), "r"(scale_a), "r"(scale_b));
    } else if constexpr (CtaGroup == 2 && V == MmaMxf4Variant::MXF4NVF4_BLOCK32) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block32 "
            "[%0], %1, %2, %3, [%5], [%6], p;\n\t"
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc),
               "r"(pred_imm), "r"(scale_a), "r"(scale_b));
    }
}

#endif  // PL_AGENTIC_SM103A

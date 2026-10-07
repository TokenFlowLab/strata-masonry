#pragma once
// test_utils.cuh -- host-side test helpers PLUS the smem pointer primitive.
//
// Split so a translation unit can take the helpers WITHOUT the primitives: the FastVideo
// drop (kernels/fmha/sm100a/sliding_window/fv/) carries its own flattened copy of
// primitives/70_smem_ptr.cuh, and including both here would define smem_ptr_u32 twice.
// Such a unit includes test_helpers.cuh directly.
//
// Everything that included test_utils.cuh before still gets exactly what it got before --
// ~90 files rely on reaching smem_ptr_u32 / sts_f32 through this header.
#include "test_helpers.cuh"
#include "../primitives/70_smem_ptr.cuh"

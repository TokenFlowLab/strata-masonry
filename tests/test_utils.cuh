#pragma once
// test_utils.cuh -- host-side test helpers PLUS the smem pointer primitive.
//
// Split so a translation unit that carries its own copy of primitives/70_smem_ptr.cuh
// can take the helpers WITHOUT redefining smem_ptr_u32: it includes test_helpers.cuh directly.
//
// Everything that included test_utils.cuh before still gets exactly what it got before --
// ~90 files rely on reaching smem_ptr_u32 / sts_f32 through this header.
#include "test_helpers.cuh"
#include "../src/primitives/70_smem_ptr.cuh"

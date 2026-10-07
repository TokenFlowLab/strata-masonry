#pragma once
// test_utils.cuh -- host-side test helpers PLUS the smem pointer primitive.
//
// Split so a translation unit that carries its own copy of primitives/70_smem_ptr.cuh
// can take the helpers WITHOUT redefining smem_ptr_u32: it includes test_helpers.cuh directly.
//
// Most tests reach smem_ptr_u32 / sts_f32 through this header.
#include "test_helpers.cuh"
#include "../src/primitives/70_smem_ptr.cuh"

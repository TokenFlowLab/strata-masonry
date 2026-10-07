// dense_gemm_nvfp4.cu -- K1 dense NVFP4 GEMM, sm_100a (GB200).
//
// Forked from dense_gemm_bf16.cu. Same 8-warp warp-specialized
// 2SM skeleton; the operands are packed E2M1 (FP4) with block-scaled
// tcgen05 MMA instead of BF16, and D is still BF16.
//
//   warp 0    : MMA   (tcgen05.mma.cta_group::2.kind::mxf4nvf4)
//   warp 1    : sched (CLC try_cancel)          -- CLC kernel only
//   warp 2    : load  (TMA A + B)
//   warp 3    : idle  (CLC arrive-count filler) -- CLC kernel only
//   warps 4-7 : epilogue (TMEM -> regs -> SMEM -> TMA store, BF16 out)
//
// One kernel, dense_gemm_nvfp4_k1_impl. Its STATIC_SCHED template flag picks
// the tile scheduler (details on the kernel banner):
//   false (default) -- CLC try_cancel, warps 1/3 run sched/idle.
//   true            -- fixed-stride walk, warps 1/3 retire. Unlocks the
//                      BLOCK_M x BLOCK_N raster that large square shapes need.
// Host entry is run_k1<ORDER, STATIC_SCHED, BLOCK_M, BLOCK_N>(M, N, K, verify).
//
// Tile geometry (both kernels):
//   M_TILE_CLUSTER = 256, M_TILE_PER_CTA = 128  (2SM splits M)
//   N_TILE_CLUSTER = NTC = 128                  (see TMEM note below)
//   K_TILE         = 256 -> 4 FP4 MMA atoms per stage (MMA K=64 each)
//   NUM_STAGES     = NS, 3..8, chosen per K by the dispatch tables so the
//                    A+B+D SMEM ring stays under the 232 KB budget
//   EPI_SUB_COLS   = 64, EPI_NUM_BUFS = 2, DRAIN_PER_TILE = false
//
// K_TILE=256 is forced by the B128 SMEM swizzle: FP4 packs 2 elements per
// byte, so a K_TILE=256 row is 128 B wide, which is what B128 wants. The
// K_TILE=128 / B64 variant was measured much slower.
//
// Scale factors: mxf4nvf4 block32 (.scale_vec::2X), UE8M0, REAL per-block
// values delivered every stage: GMEM -TMA-> SMEM -tcgen05.cp-> TMEM, all
// covered by the mainloop full_bar. Layout verified against PTX ISA 9.4
// Fig 239/258 (sec 9.7.18.10.7).
//
// Operand bytes, per CTA ("loose" = the per-atom-cell ALTERNATIVE below):
//
// +-----------------+------------------------+------------------+---------------------+----------------------+
// | Tensor          | Split across 2 CTAs    | Per CTA (full K) | Per stage, compact  | Per stage, loose     |
// +-----------------+------------------------+------------------+---------------------+----------------------+
// | A (fp4, 2/byte) | M-split: 128 rows each | 128 x K/2        | 128 x 256/2 = 16 KB | same = 16 KB         |
// +-----------------+------------------------+------------------+---------------------+----------------------+
// | B (fp4)         | N-split: 64 cols each  | 64 x K/2         | 64 x 256/2 = 8 KB   | same = 8 KB          |
// +-----------------+------------------------+------------------+---------------------+----------------------+
// | SFA (UE8M0)     | M-split: own 128 rows  | 128 x K/32       | 128 x 256/32 = 1 KB | 128 x 4 x 4 B = 2 KB |
// +-----------------+------------------------+------------------+---------------------+----------------------+
// | SFB (UE8M0)     | REPLICATED: all 128 N  | 128 x K/32       | 128 x 256/32 = 1 KB | 128 x 4 x 4 B = 2 KB |
// +-----------------+------------------------+------------------+---------------------+----------------------+
// | total           |                        |                  | 26 KB (NS=7)        | 28 KB (NS=6)         |
// +-----------------+------------------------+------------------+---------------------+----------------------+
//
// (loose SFA/SFB: 128 rows x 4 atoms x one 4-B cell each)
//
// Which format uses which: MXFP4 block32 (this kernel) uses COMPACT --
// 2 scale bytes per row per atom, so two atoms pack into one cell via
// SF_ID (ID-flip). LOOSE gives every atom its own cell and fits both
// formats, differing only in waste: MXFP4 fills 2 of the 4 bytes (bytes
// [2,3] padding, ~6-8pp slower), NVFP4 block16 fills all
// 4 (4 scale bytes per row per atom, no waste -- loose is its only option).
//
// Why SFB is replicated while B is N-split: at MMA time B is read from
// SMEM, and the 2SM hardware reads BOTH CTAs' halves (that is what lets B
// split). Scales are read from each CTA's OWN TMEM -- there is no
// cross-CTA read for SFB -- and each CTA computes all 128 N columns of its
// M-half, so both CTAs must hold all 128 N scales. SFA needs no such
// treatment: each CTA only computes its own 128 rows.
//
// TMEM (512 cols allocated, cta_group::2):
//   cols   0..127         : accumulator bank 0
//   cols 128..255         : accumulator bank 1
//   cols 256..256+16*NS-1 : scale ring, one 16-col slot per pipeline stage
//                           (SFA two 4-col 2atoms, then SFB same)
//   rest                  : unused
//
// One SFA 2atom = 4 TMEM cols x 128 lanes (shown: 2atom 0, K 0-127):
//
//            col 0        col 1        col 2        col 3
//          +------------+------------+------------+------------+
//   lanes  | rows 0-31  | rows 32-63 | rows 64-95 | rows 96-127|  copy 0
//   0-31   | (lane i =  |            |            |            |  (sub-part 0)
//          |  row i)    | row 32+i   | row 64+i   | row 96+i   |
//          +------------+------------+------------+------------+
//   lanes  |            identical copy of the above            |  copy 1
//   32-63  |                                                   |  (sub-part 1)
//          +---------------------------------------------------+
//   lanes  |            identical copy                         |  copy 2
//   64-95  |                                                   |  (sub-part 2)
//          +---------------------------------------------------+
//   lanes  |            identical copy                         |  copy 3
//   96-127 |                                                   |  (sub-part 3)
//          +---------------------------------------------------+
//
// ONE tcgen05.cp.32x128b.warpx4 writes the whole 2atom (4 cols x 128 lanes
// = 512 cells) from 512 B of SMEM: the 128-row x 4-B payload is broadcast
// to all 4 lane quads, because each TMEM sub-partition's MMA datapath
// reads only its own 32 lanes and so needs its own copy.
//
// Each 32-bit cell = one row's 4 scale bytes for this 2atom's K range:
//     byte 0     byte 1     byte 2     byte 3
//     [K 0-31]   [K 32-63]  [K 64-95]  [K 96-127]
//     \__ atom 0 (SF_ID=0) _/ \__ atom 1 (SF_ID=2) _/
//
// Full stage ring slot (16 cols):
//   cols 0-3: SFA 2atom 0 (K 0-127)     cols  8-11: SFB 2atom 0
//   cols 4-7: SFA 2atom 1 (K 128-255)   cols 12-15: SFB 2atom 1
//   (SFB: same shape with N columns 0-127 in place of M rows)
//
// ALTERNATIVE (not used): per-atom-cell layout -- each K=64 atom owns its
// own 4-col group of cells, SF_ID=0 always. This is what
// canonical NVFP4 block16 (scale_vec::4X, 4 scale bytes per row per atom)
// would fill fully:
//
//   SFA per stage = 4 cps = 16 cols:
//     cols 0-3      cols 4-7      cols 8-11     cols 12-15
//     atom 0        atom 1        atom 2        atom 3
//     K 0-63        K 64-127      K 128-191     K 192-255
//
//   Cell for atom a (MXFP4 block32 -- half wasted):
//     byte 0        byte 1        byte 2     byte 3
//     [K 64a+0-31]  [K 64a+32-63] [ unused ] [ unused ]
//     \__ atom a (SF_ID=0) __/    \__ 2 B wasted/row __/
//
//   Ring slot = 32 cols (SFA 0-15, SFB 16-31). Costs per stage per operand:
//   4 cps vs 2, 2 KB SMEM vs 1 (padding also rides the TMA and the packed
//   GMEM buffer), NS 7 -> 6 at 28 KB/stage; TMEM still fits (256 + 6x32 = 448).
//   Measured ~6-8pp slower for MXFP4 -- kept only as the shape
//   a block16 NVFP4 variant would use, with bytes 2-3 valid and SF_ID
//   sequence 0,0,0,0 instead of 0,2,0,2.
//
// Gotcha: the acc bank stride is N_TILE_CLUSTER, so NTC=128 is what leaves
// the scale region intact. With NTC=256 bank 1 starts at col 256 and the
// first MMA overwrites the scales -> garbage output. tcgen05.alloc n_cols is
// capped at 512 per SM, so there is no room to move the scales up; do NOT
// raise NTC without re-solving that. A 1-bank NTC=256 variant avoids the
// overlap but serializes MMA against the epilogue and measured worse
// (60.6% vs 70.4% SOL).
//
//
// Worked example -- the three SFA layouts (natural -> GMEM/SMEM -> TMEM).
// M=512, N=256, K=1024; CTA at (m_tile=1, n_tile=0, peer=1), rows 384-511.
//
// SFA layout 1 -- natural: 512 rows x 32 bytes
//         bytes ->       0..7   8..15  16..23 24..31
// rows   0-127         |  .   |  .   |  .   |  .   |
//      128-255         |  .   |  .   |  .   |  .   |
//      256-383         |  .   |  .   |  .   |  .   |
//      384-511         | [k0] | [k1] | [k2] | [k3] |
//
//   k0   = 1 byte for one 32-K-element block
//   [k0] = 128 x 8 bytes
//
// SFA layout 2 -- GMEM/SMEM (after sf_pack): 4 bands x 512 u64s
// (GMEM holds all bands; this CTA's SMEM holds only band 3, one slice per
// ring buffer)
//         u64s ->        0..127    128..255  256..383  384..511
// band 0               |    .    |    .    |    .    |    .    |
// band 1               |    .    |    .    |    .    |    .    |
// band 2               |    .    |    .    |    .    |    .    |
// band 3 (rows 384-511)|[cp0|cp1]|[cp0|cp1]|[cp0|cp1]|[cp0|cp1]|
//
//   cp0|cp1   = 1 u64 (cp0 = 4 bytes, cp1 = 4 bytes)
//   [cp0|cp1] = 1 x 128 u64s
//
//   Band 3, first [cp0|cp1]:
//     u64 0  : row 384 bytes 0-3 | row 416 bytes 0-3
//     u64 1  : row 448 bytes 0-3 | row 480 bytes 0-3
//     u64 2  : row 385 bytes 0-3 | row 417 bytes 0-3
//     u64 3  : row 449 bytes 0-3 | row 481 bytes 0-3
//        ...
//     u64 62 : row 415 bytes 0-3 | row 447 bytes 0-3
//     u64 63 : row 479 bytes 0-3 | row 511 bytes 0-3
//     u64 64 : row 384 bytes 4-7 | row 416 bytes 4-7
//     u64 65 : row 448 bytes 4-7 | row 480 bytes 4-7
//        ...
//     u64 126: row 415 bytes 4-7 | row 447 bytes 4-7
//     u64 127: row 479 bytes 4-7 | row 511 bytes 4-7
//
//   Pattern: for u64 x, l = (x%64)/2;
//     x even -> rows 384+l | 384+l+32,  x odd -> rows 384+l+64 | 384+l+96;
//     u64 0..63 carry bytes 0-3, u64 64..127 carry bytes 4-7.
//
// SFA layout 3 -- TMEM: one slice [cp0|cp1] -> 8 TMEM cols, cell = 4 bytes
//     u64 0  : cp0 = row 384 bytes 0-3 -> (TMEM row 0, col 0)
//              cp1 = row 416 bytes 0-3 -> (TMEM row 0, col 1)
//     u64 1  : cp0 = row 448 bytes 0-3 -> (TMEM row 0, col 2)
//              cp1 = row 480 bytes 0-3 -> (TMEM row 0, col 3)
//     u64 2  : cp0 = row 385 bytes 0-3 -> (TMEM row 1, col 0)
//              cp1 = row 417 bytes 0-3 -> (TMEM row 1, col 1)
//     u64 3  : cp0 = row 449 bytes 0-3 -> (TMEM row 1, col 2)
//              cp1 = row 481 bytes 0-3 -> (TMEM row 1, col 3)
//        ...
//     u64 62 : cp0 = row 415 bytes 0-3 -> (TMEM row 31, col 0)
//              cp1 = row 447 bytes 0-3 -> (TMEM row 31, col 1)
//     u64 63 : cp0 = row 479 bytes 0-3 -> (TMEM row 31, col 2)
//              cp1 = row 511 bytes 0-3 -> (TMEM row 31, col 3)
//     u64 64 : cp0 = row 384 bytes 4-7 -> (TMEM row 0, col 4)
//              cp1 = row 416 bytes 4-7 -> (TMEM row 0, col 5)
//        ...
//     u64 127: cp0 = row 479 bytes 4-7 -> (TMEM row 31, col 6)
//              cp1 = row 511 bytes 4-7 -> (TMEM row 31, col 7)
//
//   plus the hardware writes copies into TMEM rows 32-63, 64-95, 96-127.
//   u64s 0-63 go via the first tcgen05.cp (cols 0-3), u64s 64-127 via the
//   second (cols 4-7).
//
//   So the 128 rows fold into 4 cols in units of 32 rows (col = (r-384)/32,
//   TMEM row = (r-384)%32). In a cell, bytes 0-1 = the even atom's scales,
//   bytes 2-3 = the odd atom's (SF_ID picks the pair). SFB: same shapes
//   with N rows 0-127, band 0, loaded by BOTH peers.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <climits>
#include <vector>

#include "dense_gemm_nvfp4_benchmark.cuh"
#include "../../../../../tests/test_utils.cuh"
#include "../../../../primitives/_warp_prof_noop.cuh"

#include "../../../../primitives/0_tcgen05_alloc.cuh"
#include "../../../../primitives/1_tcgen05_dealloc.cuh"
#include "../../../../primitives/2_tcgen05_relinquish.cuh"
#include "../../../../primitives/5_tcgen05_mma_fp4.cuh"
#include "../../../../primitives/8_tcgen05_mma_idesc.cuh"
#include "../../../../primitives/9_tcgen05_ld.cuh"
#include "../../../../primitives/11_tcgen05_commit.cuh"
#include "../../../../primitives/12_tcgen05_wait.cuh"
#include "../../../../primitives/13_tcgen05_cp.cuh"
#include "../../../../primitives/15_tcgen05_fence.cuh"
#include "../../../../primitives/18_tma_load.cuh"
#include "../../../../primitives/19_tma_load_2sm.cuh"
#include "../../../../primitives/20_tma_load_multicast.cuh"
#include "../../../../primitives/21_tma_load_prefetch.cuh"
#include "../../../../primitives/22_tma_store.cuh"
#include "../../../../primitives/23_tma_tensormap.cuh"
#include "../../../../primitives/25_tma_async_group.cuh"
#include "../../../../primitives/29_mbarrier_init.cuh"
#include "../../../../primitives/30_mbarrier_arrive.cuh"
#include "../../../../primitives/31_mbarrier_arrive_tx.cuh"
#include "../../../../primitives/33_mbarrier_try_wait.cuh"
#include "../../../../primitives/34_fence_proxy_async.cuh"
#include "../../../../primitives/35_fence_mbarrier_init.cuh"
#include "../../../../primitives/38_barrier_cluster.cuh"
#include "../../../../primitives/40_stmatrix.cuh"
#include "../../../../primitives/42_smem_desc_blackwell.cuh"
#include "../../../../primitives/44_elect_sync.cuh"
#include "../../../../primitives/61_clc_try_cancel.cuh"
#include "../../../../primitives/62_clc_query_cancel.cuh"
#include "../../../../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../../../../primitives/68_l2cache_policy.cuh"
#include "../../../../primitives/69_griddepcontrol.cuh"
#include "../../../../primitives/70_smem_ptr.cuh"

#include "../../../../composites/119_pipeline_init_blackwell.cuh"
#include "../../../../composites/82_tile_rasterize.cuh"
#include "../../../../composites/104_acc_pipeline_2bank_blackwell.cuh"
#include "../../../../composites/106_clc_fetch_next_tile.cuh"
#include "../../../../composites/118_mbarrier_phase_tracking.cuh"
#include "../../../../blocks/88_load_warp_blackwell.cuh"
#include "../../../../blocks/90_mma_warp_blackwell.cuh"
#include "../../../../blocks/93_epi_warp_blackwell.cuh"
#include "../../../../blocks/97_sched_warp_clc.cuh"
#include "../../../../blocks/107_idle_warp_blackwell.cuh"

constexpr int K1_M_TILE_CLUSTER = 256;
constexpr int K1_M_TILE_PER_CTA = 128;
// K_TILE=256 is not tunable in practice: FP4 packs 2 elements/byte, so a
// K_TILE=256 row is exactly the 128 B that the B128 SMEM swizzle wants.
// K_TILE=128 would need B64, which measured far worse.
constexpr int K1_K_TILE         = 256;   // 4 FP4 MMA atoms/stage (MMA K=64 each)
constexpr int K1_NUM_STAGES     = 5;     // default only; dispatch picks NS per K
#if defined(K1_TUNE_EPI)
constexpr int K1_EPI_SUB_COLS   = K1_TUNE_EPI;
#else
constexpr int K1_EPI_SUB_COLS   = 16;  // 16: finer store pipelining, +1-2pp on every
                                       // NTC=256 row and neutral-to-positive at NTC=128;
                                       // 32 freed the D ring for NS=8
#endif
#if defined(K1_TUNE_EPI_BUFS)
constexpr int K1_EPI_NUM_BUFS   = K1_TUNE_EPI_BUFS;
#else
constexpr int K1_EPI_NUM_BUFS   = 2;
#endif

// FP4: 2 elements per byte, hence the /2 on every K extent.
constexpr int K1_EPI_BUF_BYTES = K1_M_TILE_PER_CTA * K1_EPI_SUB_COLS * 2;
constexpr int K1_D_TILE_BYTES  = K1_EPI_NUM_BUFS * K1_EPI_BUF_BYTES;
// A 4-deep D ring at NTC=128 was tested and LOSES 2-4pp
// on the qkv rows -- more concurrent TMA stores contend with loads,
// same mechanism as the NTC=256 per-warp regression. Keep 2.
constexpr int k1_epi_num_bufs(int ntc) {
  (void)ntc;
  return K1_EPI_NUM_BUFS;
}

// TMEM layout, cta_group::2, 512 cols allocated:
//   cols   0..127         : acc bank 0    (bank stride is N_TILE_CLUSTER)
//   cols 128..255         : acc bank 1
//   cols 256..256+16*NS-1 : per-stage scale ring (see K1_SF_RING_OFFSET)
// This is why NTC is pinned at 128. At NTC=256 bank 1 would start at col 256
// and the first MMA would clobber the scales. There is nowhere to relocate
// them either: tcgen05.alloc takes a power-of-two n_cols and each SM only
// has 512 columns, so cols 512+ do not exist (verified -- it faults with an
// illegal instruction). See the header for the variants that were tried.
constexpr int K1_TMEM_NCOLS = 512;

constexpr int K1_ATOM_K = 64;         // K dim of one FP4 MMA atom (256x128x64)
constexpr int K1_SF_BLOCK_SIZE = 32;  // K elems covered by one 1-B scale factor
constexpr int K1_SF_BYTES_PER_KTILE = K1_K_TILE / K1_SF_BLOCK_SIZE;  // 8
constexpr uint32_t K1_SF_RING_OFFSET = 256;  // TMEM col of stage-0 slot
// 128 rows fold into 4 groups of 32 rows (one TMEM col per group, on lanes 0-31).
constexpr uint32_t K1_SF_NUM_ROW_GROUPS = K1_M_TILE_PER_CTA / 32;

template <int SF_ATOMS_PER_CELL, int NTC = 128>
struct K1SfLayout {
  static_assert(SF_ATOMS_PER_CELL == 1 || SF_ATOMS_PER_CELL == 2);
  static_assert(NTC == 128 || NTC == 256);
  static constexpr int SFA_PER_ATOM_BYTES =
      K1_M_TILE_PER_CTA * (4 / SF_ATOMS_PER_CELL);  // 128 rows x (4-B cell / atoms sharing it)
  static constexpr int SFA_STAGE_BYTES =
      SFA_PER_ATOM_BYTES * (K1_K_TILE / K1_ATOM_K);  // x 4 atoms
  // SFB: all NTC N-cols per CTA (replicated); at NTC=256 that is 2x SFA.
  static constexpr int SFB_STAGE_BYTES = (NTC / 128) * SFA_STAGE_BYTES;
  static constexpr uint32_t SFA_STAGE_COLS =
      (K1_K_TILE / K1_ATOM_K / SF_ATOMS_PER_CELL) * K1_SF_NUM_ROW_GROUPS;  // 8
  static constexpr uint32_t SFB_STAGE_COLS = (NTC / 128) * SFA_STAGE_COLS;  // 8 or 16
  static constexpr uint32_t STAGE_COLS = SFA_STAGE_COLS + SFB_STAGE_COLS;
  // TMEM map. NTC=128: 2 banks [0,256) + NS-deep ring at 256.
  // NTC=256: banks [0,256) and [192,448) (64-col overlap, freed early by
  // the epi via the ovl_free bar) + depth-2 scale ring at 448.
  // (A 48-col window -- bank1 at 208, ring at 464 -- was built and
  // REFUTED: exact but -10%, TMEM alignment penalty.)
  static constexpr uint32_t ACC_BANK_STRIDE = (NTC == 128) ? NTC : 192;
  static constexpr uint32_t RING_OFFSET     = (NTC == 128) ? 256 : 448;
  static constexpr int      TMEM_RING_DEPTH = (NTC == 128) ? 0 : 2;  // 0 = NS-deep
};

// A/B SMEM descriptors: B128 swizzle, K-major, row = K_TILE/2 = 128 B.
constexpr uint32_t K1_A_LBO = 16;
constexpr uint32_t K1_A_SBO = 1024;
constexpr uint32_t K1_B_LBO = 16;
constexpr uint32_t K1_B_SBO = 1024;

extern __shared__ __align__(1024) uint8_t k1_smem[];

// Bit 24 of a cluster SMEM address selects the peer CTA. XOR-ing it turns a
// local mbarrier address into the same mbarrier in the other CTA of the pair.
// Matches cute::Sm100MmaPeerBitMask (which masks it off rather than flipping).
constexpr uint32_t K1_SM100_PEER_BIT = 0x01000000u;

// Map a cluster's flat work index to (m_tile, n_tile).
//
// BLOCK_M/BLOCK_N == 0 -> plain raster: AlongN sweeps N fastest (so the same
// B column is revisited only after n_dim tiles), AlongM sweeps M fastest.
//
// BLOCK_M/BLOCK_N > 0 -> walk BLOCK_M x BLOCK_N tile blocks, M-major within a
// block. A block touches BLOCK_M A-tiles and BLOCK_N B-tiles, and revisits
// each of them BLOCK_N / BLOCK_M times before moving on, so both operands can
// stay resident in L2 for the whole block. That is what large square shapes
// need: under plain AlongN a 16384^3 B column is evicted long before it is
// reused and every stage falls back to HBM. Measured on 16384^3:
// AlongN 65.0% -> BLOCK_M = BLOCK_N = 16 77.1% SOL. Sizing rule is
// BLOCK_M * A_tile_bytes + BLOCK_N * B_tile_bytes <~ L2 per cluster; it
// degrades once the block working set no longer fits (BM=BN=32 -> 74.2%).
//
// Each block dimension is clamped to the grid so an over-large BLOCK_M/N
// cannot index past m_dim / n_dim.
// M-run remap: the strided walk (flat = c + t*NC) is re-mapped to balanced
// CONTIGUOUS per-cluster ranges walked n-fastest, so one cluster sees long
// same-m runs. With K_BLOCKS %% NUM_STAGES == 0 the A ring slots of tile t+1
// then hold byte-identical A already, and the loader skips A+SFA loads.
__device__ __forceinline__
int k1_mrun_remap(int flat, int num_clusters, int total_tiles) {
  const int c = flat % num_clusters;
  const int t = flat / num_clusters;
  const int T = total_tiles / num_clusters;
  const int rem = total_tiles % num_clusters;
  return (c < rem) ? c * (T + 1) + t
                   : rem * (T + 1) + (c - rem) * T + t;
}

// RUN-STRIDED (wide N): runs of run_len same-m tiles in column-major run
// order (n-window outer, m inner), dealt round-robin to clusters -- all
// clusters then share one small B n-window in L2 (instead of each cluster
// re-streaming all of B per m-run) while keeping the A/SFA skip inside runs.
__device__ __forceinline__
int k1_rstride_my_tiles(int cluster_id, int num_clusters,
                        int runs_total, int run_len) {
  const int r = (runs_total > cluster_id)
                    ? (runs_total - 1 - cluster_id) / num_clusters + 1 : 0;
  return r * run_len;
}
__device__ __forceinline__
void k1_rstride_decode(int cluster_id, int num_clusters, int t, int run_len,
                       int m_dim, int& m_tile, int& n_tile) {
  const int i   = t / run_len;
  const int pos = t % run_len;
  const int r   = cluster_id + i * num_clusters;
  m_tile = r % m_dim;
  n_tile = (r / m_dim) * run_len + pos;
}

template <ClcRasterOrder ORDER, int BLOCK_M, int BLOCK_N, bool N_FAST = false>
__device__ __forceinline__
void k1_decode_tile(int flat, int m_dim, int n_dim, int& m_tile, int& n_tile) {
  if constexpr (BLOCK_M > 0 && BLOCK_N > 0) {
    const int bm_eff     = (BLOCK_M < m_dim) ? BLOCK_M : m_dim;
    const int bn_eff     = (BLOCK_N < n_dim) ? BLOCK_N : n_dim;
    const int block_size = bm_eff * bn_eff;
    const int block_id   = flat / block_size;
    const int pos        = flat % block_size;
    const int blocks_m   = (m_dim + bm_eff - 1) / bm_eff;
    if constexpr (N_FAST) {  // n varies fastest inside a block: same-m runs
      m_tile = (block_id % blocks_m) * bm_eff + pos / bn_eff;
      n_tile = (block_id / blocks_m) * bn_eff + pos % bn_eff;
    } else {
      m_tile = (block_id % blocks_m) * bm_eff + pos % bm_eff;
      n_tile = (block_id / blocks_m) * bn_eff + pos / bm_eff;
    }
  } else if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = flat / n_dim;
    n_tile = flat % n_dim;
  } else {
    m_tile = flat % m_dim;
    n_tile = flat / m_dim;
  }
}

// ============================================================================
// Load warp: 1 tile body for FP4.
// Key differences from load_warp_blackwell_1tile_2sm_bf16:
//   - kStageBytes = M_TILE * K_TILE / 2  (FP4 = 0.5 bytes/elem)
//   - k_off = k * K_TILE / 2             (byte-space offset in UINT8 tensor)
//   - expect_tx uses FP4 byte counts
//   - per stage also TMAs the scale slices: SFA for THIS CTA's 128 M-rows
//     (per-peer y), SFB for the full N=128 tile (same y on both peers --
//     tcgen05.cp.cta_group::2 reads each CTA's own SMEM, so SFB must be
//     replicated in both). Plain 1-CTA TMA, completion routed to the leader
//     full bar like everything else.
// ============================================================================
template <int NUM_STAGES, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int SF_ATOMS_PER_CELL>
__device__ __forceinline__
void load_warp_k1_1tile_2sm_fp4(WpCtx& wpc,
    const CUtensorMap* tma_a, const CUtensorMap* tma_b,
    const CUtensorMap* tma_sfa, const CUtensorMap* tma_sfb,
    uint8_t* smem_a, uint8_t* smem_b,
    uint8_t* smem_sfa, uint8_t* smem_sfb,
    uint64_t* full_bar, uint64_t* empty_bar,
    int K, int m_offset, int n_offset_b, int peer,
    EmptyPhaseTracker<NUM_STAGES>& empty_ph, bool skip_a = false,
    bool pingpong_reuse = false, bool pingpong_backward = false) {
  // Ping-pong K (legal partial A-reuse, requires K_BLOCKS % NS == 0):
  // consecutive same-m tiles alternate the ROUND order of the K walk
  // (order within each NS-round stays forward, so slots, barriers and
  // the whole MMA warp are untouched). A forward tile ends with the
  // ring holding the LAST NS k-blocks; the next (backward) tile's
  // round 0 is exactly those blocks -> its A/SFA loads are skipped.
  // A backward tile ends holding blocks 0..NS-1 = the next forward
  // tile's round 0. Saves NS/K_BLOCKS of the A+SFA bytes per tile;
  // degenerates to the full legal skip when K_BLOCKS == NS.
  constexpr int kStageABytes = M_TILE_PER_CTA * K_TILE / 2;
  constexpr int kStageBBytes = N_TILE_PER_CTA * K_TILE / 2;
  // Both CTAs' A+B halves plus both CTAs' scale slices land on the leader bar.
  using SF = K1SfLayout<SF_ATOMS_PER_CELL, 2 * N_TILE_PER_CTA>;
  constexpr int kStageSfaBytes = SF::SFA_STAGE_BYTES;
  constexpr int kStageSfbBytes = SF::SFB_STAGE_BYTES;
#if defined(K1_DISABLE_SF_TMA_CP)
  const uint32_t expect_b_only = 2 * kStageBBytes;
  const uint32_t expect_all    = 2 * (kStageABytes + kStageBBytes);
#else
  const uint32_t expect_b_only = 2 * (kStageBBytes + kStageSfbBytes);
  const uint32_t expect_all    =
      2 * (kStageABytes + kStageBBytes + kStageSfaBytes + kStageSfbBytes);
#endif
  const int K_BLOCKS = K / K_TILE;
  // SFA is M-split like A (sfa_y differs per peer). SFB is NOT N-split:
  // each CTA computes all 128 N cols and reads scales from its own TMEM
  // (no 2SM union as for B), so sfb_y is the same on both peers -- the
  // peer*64 term floors away and the load is replicated.
  const int sfa_y = m_offset / 128;
  const int sfb_y = (n_offset_b - peer * N_TILE_PER_CTA) / 128;
  // A policy depends on the tile ORDER:
  //   AlongN-style orders: A rows are shared across concurrent clusters
  //   -> 25% evict_last + 75% unchanged.
  //   Same-m-run orders (ping-pong reuse): each cluster streams its own
  //   PRIVATE A row (SMEM-resident via ping-pong, never L2-reused) --
  //   with evict_last it thrashes B out of L2 (measured 4.5 TB/s DRAM
  //   vs 0.6 for AlongN) -> evict_first.
  // B: full evict_last -- B is the shared stream in every order.
  const uint64_t cache_policy_a =
      make_l2cache_policy_fractional_evict_last_unchanged(0.25f);
  const uint64_t cache_policy_b = make_l2cache_policy_evict_last_full();

  for (int k = 0; k < K_BLOCKS; ++k) {
    const int      s          = empty_ph.get_stage();
    const uint32_t full       = smem_ptr_u32(&full_bar[s]);
    const uint32_t full_route = tma_peer_bit_mask(full);
    // GMEM k-block this stage fetches: identity forward, round-reversed
    // when ping-pong walks this tile backward.
    const int kg = pingpong_backward
        ? (K_BLOCKS - (k / NUM_STAGES + 1) * NUM_STAGES + (k % NUM_STAGES))
        : k;
    // A/SFA of round 0 are already resident on a ping-pong continuation.
    const bool stage_skip_a =
        skip_a || (pingpong_reuse && k < NUM_STAGES);
    // k_off is in UINT8-element units (the packed tensor's column index).
    const int k_off = kg * (K_TILE / 2);
    // Scale tensors are uint64 rows; one stage = kStageSfaBytes/8 uint64s.
    const int sf_x = kg * (kStageSfaBytes / 8);  // same stride for SFB

    wp_begin(wpc, WP_LOAD_WAIT);
    mbarrier_wait_parity(smem_ptr_u32(&empty_bar[s]), empty_ph.get_phase());
    wp_end(wpc, WP_LOAD_WAIT);
    wp_begin(wpc, WP_LOAD_ISSUE);
    if (elect_one_sync()) {
      if (peer == 0)
        mbarrier_arrive_expect_tx(full, stage_skip_a ? expect_b_only
                                                     : expect_all);
      if (!stage_skip_a) {
        tma_load_2d_2sm_l2hint(smem_ptr_u32(smem_a + s * kStageABytes),
                               tma_a, full_route, k_off, m_offset, cache_policy_a);
      }
      tma_load_2d_2sm_l2hint(smem_ptr_u32(smem_b + s * kStageBBytes),
                             tma_b, full_route, k_off, n_offset_b, cache_policy_b);
      // NTC=256 only: L2 prefetch K1_PF_DIST k-blocks ahead. The NS=4 ring
      // buffers ~1 us, so an L2 miss stalls the MMA; warming L2 turns the
      // in-window load into a hit. (At NTC=128 it loses 2-6pp to the extra
      // L2 engine requests, so gated by tile width.)
#if defined(K1_PF_DIST)
      constexpr int kPfDist = K1_PF_DIST;
#elif 1
      // Swept: 12 beats NUM_STAGES on the
      // K=4096 A-streaming rows (+0.4pp 2M+2N), neutral on big rows.
      constexpr int kPfDist = 12;
#else
      constexpr int kPfDist = NUM_STAGES;
#endif
      if constexpr (2 * N_TILE_PER_CTA == 256) {
        if (k + kPfDist < K_BLOCKS) {
          const int k_pf = (k + kPfDist) * (K_TILE / 2);
          tma_prefetch_2d_l2hint(tma_b, k_pf, n_offset_b, cache_policy_b);
          if (!stage_skip_a)
            tma_prefetch_2d_l2hint(tma_a, k_pf, m_offset, cache_policy_a);
        }
      }
#if !defined(K1_DISABLE_SF_TMA_CP)
      if (!stage_skip_a) {
        tma_load_2d_2sm(smem_ptr_u32(smem_sfa + s * kStageSfaBytes),
                        tma_sfa, full_route, sf_x, sfa_y);
      }
      // SFB is identical on both peers: one multicast read feeds both SMEMs.
      // At NTC=256 the tensormap box spans both 128-col scale bands
      // (box_rows = NTC/128), so one load delivers the full stage slice.
      if (peer == 0) {
        tma_load_2d_2sm_multicast(tma_sfb,
                                  smem_ptr_u32(smem_sfb + s * kStageSfbBytes),
                                  full_route, sf_x, sfb_y, /*cta_mask=*/0x3);
      }
#endif
    }
    wp_end(wpc, WP_LOAD_ISSUE);
    empty_ph.advance();
  }
  __syncwarp();
}

// Two-load-warp splitb halves (static, NS_B > 0): warp 2 drives the A
// ring, warp 3 the B ring, so neither stream's TMA issue or wait gates
// the other. Contracts match the combined splitb loader exactly.
template <int NS_A, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int SF_ATOMS_PER_CELL>
__device__ __forceinline__
void load_warp_k1_1tile_a_side(WpCtx& wpc,
    const CUtensorMap* tma_a, const CUtensorMap* tma_sfa,
    uint8_t* smem_a, uint8_t* smem_sfa,
    uint64_t* full_a_bar, uint64_t* empty_a_bar,
    int K, int m_offset, int peer,
    EmptyPhaseTracker<NS_A>& empty_ph_a) {
  constexpr int kStageABytes = M_TILE_PER_CTA * K_TILE / 2;
  using SF = K1SfLayout<SF_ATOMS_PER_CELL, 2 * N_TILE_PER_CTA>;
  constexpr int kStageSfaBytes = SF::SFA_STAGE_BYTES;
  const int K_BLOCKS = K / K_TILE;
#if defined(K1_DISABLE_SF_TMA_CP)
  const uint32_t expect_a = 2 * kStageABytes;
#else
  const uint32_t expect_a = 2 * (kStageABytes + kStageSfaBytes);
#endif
  const int sfa_y = m_offset / 128;
  const uint64_t cache_policy_a =
      make_l2cache_policy_fractional_evict_last_unchanged(0.25f);

  for (int k = 0; k < K_BLOCKS; ++k) {
    const int sa = empty_ph_a.get_stage();
    const int k_off = k * (K_TILE / 2);
    wp_begin(wpc, WP_LOAD_WAIT);
    mbarrier_wait_parity(smem_ptr_u32(&empty_a_bar[sa]), empty_ph_a.get_phase());
    wp_end(wpc, WP_LOAD_WAIT);
    wp_begin(wpc, WP_LOAD_ISSUE);
    if (elect_one_sync()) {
      const uint32_t full_a = smem_ptr_u32(&full_a_bar[sa]);
      if (peer == 0) mbarrier_arrive_expect_tx(full_a, expect_a);
      tma_load_2d_2sm_l2hint(smem_ptr_u32(smem_a + sa * kStageABytes),
                             tma_a, tma_peer_bit_mask(full_a),
                             k_off, m_offset, cache_policy_a);
#if !defined(K1_DISABLE_SF_TMA_CP)
      const int sf_x = k * (kStageSfaBytes / 8);
      tma_load_2d_2sm(smem_ptr_u32(smem_sfa + sa * kStageSfaBytes),
                      tma_sfa, tma_peer_bit_mask(full_a), sf_x, sfa_y);
#endif
    }
    wp_end(wpc, WP_LOAD_ISSUE);
    empty_ph_a.advance();
  }
  __syncwarp();
}

// CGA_PAIRS=2 B loader: rank 0's warp 3 is the SOLE B/SFB issuer for all
// four CTAs. Per stage: wait the b_gate (both pairs' MMAs done with the
// slot), arm rank 0's AND rank 2's full_b bars (remote arm via mapa --
// single-thread arm-before-issue ordering), then issue B half0 to ranks
// {0,2} (ctamask 0x5), B half1 to ranks {1,3} (0xA) and SFB to all (0xF).
// Each pair leader's bar collects its pair's completions via the bit-24
// peer mask (rank 1 -> 0, rank 3 -> 2), so the expect stays 2*(B+SFB).
template <int NS_B, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int SF_ATOMS_PER_CELL>
__device__ __forceinline__
void load_warp_k1_1tile_b_side_cga(WpCtx& wpc,
    const CUtensorMap* tma_b, const CUtensorMap* tma_sfb,
    uint8_t* smem_b, uint8_t* smem_sfb,
    uint64_t* full_b_bar, uint64_t* b_gate_bar,
    int K, int n_offset_base,
    EmptyPhaseTracker<NS_B>& gate_ph) {
  constexpr int kStageBBytes = N_TILE_PER_CTA * K_TILE / 2;
  using SF = K1SfLayout<SF_ATOMS_PER_CELL, 2 * N_TILE_PER_CTA>;
  constexpr int kStageSfaBytes = SF::SFA_STAGE_BYTES;
  constexpr int kStageSfbBytes = SF::SFB_STAGE_BYTES;
  const int K_BLOCKS = K / K_TILE;
#if defined(K1_DISABLE_SF_TMA_CP)
  const uint32_t expect_b = 2 * kStageBBytes;
#else
  const uint32_t expect_b = 2 * (kStageBBytes + kStageSfbBytes);
#endif
  const int sfb_y = n_offset_base / 128;
  const uint64_t cache_policy_b = make_l2cache_policy_evict_last_full();

  for (int k = 0; k < K_BLOCKS; ++k) {
    const int sb = gate_ph.get_stage();
    const int k_off = k * (K_TILE / 2);
    wp_begin(wpc, WP_LOAD_WAIT);
    mbarrier_wait_parity(smem_ptr_u32(&b_gate_bar[sb]), gate_ph.get_phase());
    wp_end(wpc, WP_LOAD_WAIT);
    wp_begin(wpc, WP_LOAD_ISSUE);
    if (elect_one_sync()) {
      const uint32_t full_local  = smem_ptr_u32(&full_b_bar[sb]);
      const uint32_t full_remote = mapa_shared_cluster_u32(full_local, 2);
      mbarrier_arrive_expect_tx(full_local, expect_b);
      mbarrier_arrive_expect_tx_cluster(full_remote, expect_b);
      tma_load_2d_2sm_multicast_l2hint(tma_b,
          smem_ptr_u32(smem_b + sb * kStageBBytes),
          tma_peer_bit_mask(full_local), k_off, n_offset_base,
          /*cta_mask=*/0x5, cache_policy_b);
      tma_load_2d_2sm_multicast_l2hint(tma_b,
          smem_ptr_u32(smem_b + sb * kStageBBytes),
          tma_peer_bit_mask(full_local), k_off,
          n_offset_base + N_TILE_PER_CTA,
          /*cta_mask=*/0xA, cache_policy_b);
#if !defined(K1_DISABLE_SF_TMA_CP)
      const int sf_x = k * (kStageSfaBytes / 8);
      tma_load_2d_2sm_multicast(tma_sfb,
          smem_ptr_u32(smem_sfb + sb * kStageSfbBytes),
          tma_peer_bit_mask(full_local), sf_x, sfb_y,
          /*cta_mask=*/0xF);
#endif
    }
    wp_end(wpc, WP_LOAD_ISSUE);
    gate_ph.advance();
  }
  __syncwarp();
}

template <int NS_B, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int SF_ATOMS_PER_CELL>
__device__ __forceinline__
void load_warp_k1_1tile_b_side(WpCtx& wpc,
    const CUtensorMap* tma_b, const CUtensorMap* tma_sfb,
    uint8_t* smem_b, uint8_t* smem_sfb,
    uint64_t* full_b_bar, uint64_t* empty_b_bar,
    int K, int n_offset_b, int peer,
    EmptyPhaseTracker<NS_B>& empty_ph_b, int cga_pair = 0) {
  constexpr int kStageBBytes = N_TILE_PER_CTA * K_TILE / 2;
  using SF = K1SfLayout<SF_ATOMS_PER_CELL, 2 * N_TILE_PER_CTA>;
  constexpr int kStageSfaBytes = SF::SFA_STAGE_BYTES;
  constexpr int kStageSfbBytes = SF::SFB_STAGE_BYTES;
  const int K_BLOCKS = K / K_TILE;
#if defined(K1_DISABLE_SF_TMA_CP)
  const uint32_t expect_b = 2 * kStageBBytes;
#else
  const uint32_t expect_b = 2 * (kStageBBytes + kStageSfbBytes);
#endif
  const int sfb_y = (n_offset_b - peer * N_TILE_PER_CTA) / 128;
  const uint64_t cache_policy_b = make_l2cache_policy_evict_last_full();

  for (int k = 0; k < K_BLOCKS; ++k) {
    const int sb = empty_ph_b.get_stage();
    const int k_off = k * (K_TILE / 2);
    wp_begin(wpc, WP_LOAD_WAIT);
    mbarrier_wait_parity(smem_ptr_u32(&empty_b_bar[sb]), empty_ph_b.get_phase());
    wp_end(wpc, WP_LOAD_WAIT);
    wp_begin(wpc, WP_LOAD_ISSUE);
    if (elect_one_sync()) {
      const uint32_t full_b = smem_ptr_u32(&full_b_bar[sb]);
      if (peer == 0) mbarrier_arrive_expect_tx(full_b, expect_b);
      tma_load_2d_2sm_l2hint(smem_ptr_u32(smem_b + sb * kStageBBytes),
                             tma_b, tma_peer_bit_mask(full_b),
                             k_off, n_offset_b, cache_policy_b);
#if !defined(K1_DISABLE_SF_TMA_CP)
      const int sf_x = k * (kStageSfaBytes / 8);
      if (peer == 0) {
        tma_load_2d_2sm_multicast(tma_sfb,
                                  smem_ptr_u32(smem_sfb + sb * kStageSfbBytes),
                                  tma_peer_bit_mask(full_b), sf_x, sfb_y,
                                  (uint16_t)(0x3 << (2 * cga_pair)));
      }
#endif
#if defined(K1_PF_DIST)
      constexpr int kPfDistB = K1_PF_DIST;
#else
      constexpr int kPfDistB = 12;
#endif
      if (k + kPfDistB < K_BLOCKS) {
        tma_prefetch_2d_l2hint(tma_b, (k + kPfDistB) * (K_TILE / 2),
                               n_offset_b, cache_policy_b);
      }
    }
    wp_end(wpc, WP_LOAD_ISSUE);
    empty_ph_b.advance();
  }
  __syncwarp();
}

// Split-ring loader (static NTC=256 only): A+SFA on an NS_A-deep ring
// (M-RUN skip applies), B+SFB on a deeper NS_B ring. Deeper B absorbs
// L2-miss latency on the non-skippable stream without paying A bytes.
// Every stage waits BOTH empties (the MMA commits both every stage, skip
// or not, so the parity ledgers stay 1:1); full_a is armed/loaded only on
// non-skip tiles and the MMA only waits it there.
template <int NS_A, int NS_B, int M_TILE_PER_CTA, int N_TILE_PER_CTA,
          int K_TILE, int SF_ATOMS_PER_CELL>
__device__ __forceinline__
void load_warp_k1_1tile_2sm_fp4_splitb(WpCtx& wpc,
    const CUtensorMap* tma_a, const CUtensorMap* tma_b,
    const CUtensorMap* tma_sfa, const CUtensorMap* tma_sfb,
    uint8_t* smem_a, uint8_t* smem_b,
    uint8_t* smem_sfa, uint8_t* smem_sfb,
    uint64_t* full_a_bar, uint64_t* empty_a_bar,
    uint64_t* full_b_bar, uint64_t* empty_b_bar,
    uint64_t* run_free_bar, int tile_idx,
    int K, int m_offset, int n_offset_b, int peer,
    EmptyPhaseTracker<NS_A>& empty_ph_a,
    EmptyPhaseTracker<NS_B>& empty_ph_b, bool skip_a) {
  constexpr int kStageABytes = M_TILE_PER_CTA * K_TILE / 2;
  constexpr int kStageBBytes = N_TILE_PER_CTA * K_TILE / 2;
  using SF = K1SfLayout<SF_ATOMS_PER_CELL, 2 * N_TILE_PER_CTA>;
  constexpr int kStageSfaBytes = SF::SFA_STAGE_BYTES;
  constexpr int kStageSfbBytes = SF::SFB_STAGE_BYTES;
  const int K_BLOCKS = K / K_TILE;
#if defined(K1_DISABLE_SF_TMA_CP)
  const uint32_t expect_a = 2 * kStageABytes;
  const uint32_t expect_b = 2 * kStageBBytes;
#else
  const uint32_t expect_a = 2 * (kStageABytes + kStageSfaBytes);
  const uint32_t expect_b = 2 * (kStageBBytes + kStageSfbBytes);
#endif
  const int sfa_y = m_offset / 128;
  const int sfb_y = (n_offset_b - peer * N_TILE_PER_CTA) / 128;
  const uint64_t cache_policy_a =
      make_l2cache_policy_fractional_evict_last_unchanged(0.25f);
  const uint64_t cache_policy_b = make_l2cache_policy_evict_last_full();

  // Run boundary: the A ring is about to be refilled while earlier (skip)
  // tiles may still be reading it. run_free_bar[t&1] is committed by the
  // MMA at the end of every tile; waiting for tile_idx-1's commit proves
  // every prior reader of the A ring has retired. (Skip tiles do not
  // touch the A handshake at all -- that is what frees the loader to run
  // NS_B ahead on the B ring.)
  if (!skip_a && tile_idx > 0) {
    wp_begin(wpc, WP_LOAD_WAIT);
    const int tp = tile_idx - 1;
    mbarrier_wait_parity(smem_ptr_u32(&run_free_bar[tp & 1]),
                         (uint32_t)((tp >> 1) & 1));
    wp_end(wpc, WP_LOAD_WAIT);
  }

  for (int k = 0; k < K_BLOCKS; ++k) {
    const int sa = empty_ph_a.get_stage();
    const int sb = empty_ph_b.get_stage();
    const int k_off = k * (K_TILE / 2);
    const int sf_x = k * (kStageSfaBytes / 8);

    wp_begin(wpc, WP_LOAD_WAIT);
    if (!skip_a)
      mbarrier_wait_parity(smem_ptr_u32(&empty_a_bar[sa]), empty_ph_a.get_phase());
    mbarrier_wait_parity(smem_ptr_u32(&empty_b_bar[sb]), empty_ph_b.get_phase());
    wp_end(wpc, WP_LOAD_WAIT);
    wp_begin(wpc, WP_LOAD_ISSUE);
    if (elect_one_sync()) {
      const uint32_t full_a = smem_ptr_u32(&full_a_bar[sa]);
      const uint32_t full_b = smem_ptr_u32(&full_b_bar[sb]);
      if (!skip_a) {
        if (peer == 0) mbarrier_arrive_expect_tx(full_a, expect_a);
        tma_load_2d_2sm_l2hint(smem_ptr_u32(smem_a + sa * kStageABytes),
                               tma_a, tma_peer_bit_mask(full_a),
                               k_off, m_offset, cache_policy_a);
#if !defined(K1_DISABLE_SF_TMA_CP)
        tma_load_2d_2sm(smem_ptr_u32(smem_sfa + sa * kStageSfaBytes),
                        tma_sfa, tma_peer_bit_mask(full_a), sf_x, sfa_y);
#endif
      }
      if (peer == 0) mbarrier_arrive_expect_tx(full_b, expect_b);
      tma_load_2d_2sm_l2hint(smem_ptr_u32(smem_b + sb * kStageBBytes),
                             tma_b, tma_peer_bit_mask(full_b),
                             k_off, n_offset_b, cache_policy_b);
#if !defined(K1_DISABLE_SF_TMA_CP)
      if (peer == 0) {
        tma_load_2d_2sm_multicast(tma_sfb,
                                  smem_ptr_u32(smem_sfb + sb * kStageSfbBytes),
                                  tma_peer_bit_mask(full_b), sf_x, sfb_y,
                                  /*cta_mask=*/0x3);
      }
#endif
      constexpr int PF_DIST = 8;
      if (k + PF_DIST < K_BLOCKS) {
        const int k_pf = (k + PF_DIST) * (K_TILE / 2);
        tma_prefetch_2d_l2hint(tma_b, k_pf, n_offset_b, cache_policy_b);
        if (!skip_a)
          tma_prefetch_2d_l2hint(tma_a, k_pf, m_offset, cache_policy_a);
      }
    }
    wp_end(wpc, WP_LOAD_ISSUE);
    if (!skip_a) empty_ph_a.advance();
    empty_ph_b.advance();
  }
  __syncwarp();
}

// FP4 load warp persistent loop (wraps 1-tile body).
// Mirrors load_warp_blackwell_ntiles_2sm_bf16:
//   - ALL 32 lanes call 1tile (elect_one_sync is inside)
//   - empty_ph NOT reset between tiles
//   - throttle handshake on peer 0
//   - NUM_STAGES tail drain after loop
template <int NUM_STAGES, int M_TILE_PER_CTA, int N_TILE_PER_CTA, int K_TILE,
          int SF_ATOMS_PER_CELL,
          int M_TILE_CLUSTER, int N_TILE_CLUSTER,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          bool RUN_CLC = false, int K_BLOCKS_T = 0, int NS_T = 0>
__device__ inline
void load_warp_k1_ntiles_2sm_fp4(WpCtx& wpc,
    const CUtensorMap* tma_a, const CUtensorMap* tma_b,
    const CUtensorMap* tma_sfa, const CUtensorMap* tma_sfb,
    uint8_t* smem_a, uint8_t* smem_b,
    uint8_t* smem_sfa, uint8_t* smem_sfb,
    uint64_t* full_bar, uint64_t* empty_bar,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint64_t* throttle_full, uint64_t* throttle_empty,
    int K, int peer, int run_len = 1) {
  setmaxnreg_dec<40>();

  EmptyPhaseTracker<NUM_STAGES> empty_ph;
  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int      thr_prod_stage = 0;
  uint32_t thr_prod_phase = 1;

  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = (int)blockIdx.y;
    n_tile = (int)(blockIdx.x >> 1);
  } else {
    m_tile = (int)(blockIdx.x >> 1);
    n_tile = (int)blockIdx.y;
  }

  int prev_m_tile = -1;
  bool prev_pp_backward = false;
  while (true) {
    const int m_off = m_tile * M_TILE_CLUSTER + peer * M_TILE_PER_CTA;

    if (peer == 0) {
      wp_begin(wpc, WP_LOAD_WAIT_THROTTLE);
      mbarrier_wait_parity(smem_ptr_u32(&throttle_empty[thr_prod_stage]),
                           thr_prod_phase);
      wp_end(wpc, WP_LOAD_WAIT_THROTTLE);
      mbarrier_arrive_nostate(smem_ptr_u32(&throttle_full[thr_prod_stage]));
      advance_stage_phase<2>(thr_prod_stage, thr_prod_phase);
    }

    // RUN_CLC: one CLC grab = a run of run_len same-m tiles (n_tile is the
    // n-window index). The A/SFA skip applies from the run's 2nd tile on.
    const int rl = RUN_CLC ? run_len : 1;
    for (int pos = 0; pos < rl; ++pos) {
      const int n_t = RUN_CLC ? (n_tile * run_len + pos) : n_tile;
      const int n_off = n_t * N_TILE_CLUSTER + peer * N_TILE_PER_CTA;
      // Ping-pong K reuse within a same-m run (legal for any
      // K_BLOCKS % NS == 0; the KB == NS full skip is the degenerate
      // case).
      const bool pp_reuse = RUN_CLC && (K_BLOCKS_T % NS_T == 0)
                            && (m_tile == prev_m_tile);
      const bool pp_backward = pp_reuse && !prev_pp_backward;
      prev_pp_backward = pp_backward;
      prev_m_tile = m_tile;
      // ALL 32 lanes call 1tile (elect is inside).
      load_warp_k1_1tile_2sm_fp4<NUM_STAGES, M_TILE_PER_CTA, N_TILE_PER_CTA,
                                 K_TILE, SF_ATOMS_PER_CELL>(
          wpc, tma_a, tma_b, tma_sfa, tma_sfb,
          smem_a, smem_b, smem_sfa, smem_sfb,
          full_bar, empty_bar,
          K, m_off, n_off, peer, empty_ph, /*skip_a=*/false,
          pp_reuse, pp_backward);
    }

    wp_begin(wpc, WP_CLC_FETCH);
    ClcTileInfo next = clc_fetch_next_tile<CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, elect_one_sync());
    wp_end(wpc, WP_CLC_FETCH);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = next.m_tile;
    n_tile = next.n_tile;
  }

  // Tail drain.
  wp_begin(wpc, WP_LOAD_WAIT);
  #pragma unroll
  for (int i = 0; i < NUM_STAGES; ++i) {
    mbarrier_wait_parity(smem_ptr_u32(&empty_bar[empty_ph.get_stage()]),
                         empty_ph.get_phase());
    empty_ph.advance();
  }
  wp_end(wpc, WP_LOAD_WAIT);
}

// ============================================================================
// MMA warp: 1 tile body
// Differences from the BF16 _1tile_ body:
//   - tcgen05_mma_mxf4nvf4_ss_2sm_block32 instead of tcgen05_mma_f16_ss<2>
//   - MMA K is 64 (not 16), so K_ATOMS_PER_TILE = K_TILE/64 = 4 at K_TILE=256
//   - K_ATOM_DELTA = 2: one atom spans 64 FP4 elements = 32 B = 2 descriptor
//     units of 16 B, versus 8 units for a BF16 atom
//   - per stage, before the atoms, the elected lane stages the scale factors
//     SMEM->TMEM with tcgen05.cp.32x128b.warpx4 (512 B of SMEM -> one 4-col
//     2atom). cp -> mma from the same thread is a guaranteed pipeline pair
//     (PTX 9.7.18.6.2), so no barrier is needed between them. The TMEM slot
//     is a per-stage ring; its reuse is protected by the existing empty_bar,
//     because the same tcgen05.commit tracks both the stage's cps and MMAs.
//   - ID-flip packing: each cell serves two K=64 atoms (even atom
//     SF_ID=0 -> cell bytes [0,1], odd atom SF_ID=2 -> bytes [2,3]), so
//     2 2atoms + 2 cps per operand per stage, zero SMEM padding.
// ============================================================================
template <int NUM_STAGES, int M_TILE_CLUSTER, int N_TILE_CLUSTER, int K_TILE,
          uint64_t A_STAGE_DELTA, uint64_t B_STAGE_DELTA, int SF_ATOMS_PER_CELL>
__device__ __forceinline__
void mma_warp_k1_1tile_2sm_nvfp4(WpCtx& wpc,
    uint64_t desc_a0, uint64_t desc_b0,
    uint64_t desc_sfa0, uint64_t desc_sfb0,
    uint64_t* full_bar, uint64_t* empty_bar,
    AccPipeline2BankBars acc_bars,
    AccPipeline2BankState& prod_state,
    uint32_t tmem_base, int K,
    PhaseTracker<NUM_STAGES>& full_ph,
    AccPipeline2BankBars ovl_bars = {nullptr, nullptr}) {
  static_assert(N_TILE_CLUSTER <= 256);
  constexpr int K_ATOMS_PER_TILE = K_TILE / K1_ATOM_K;  // = 4 for K_TILE=256
  constexpr uint64_t K_ATOM_DELTA = 2;            // 64 K-elem x 0.5B = 32B = 2 x 16B units
  using SF = K1SfLayout<SF_ATOMS_PER_CELL, N_TILE_CLUSTER>;
  constexpr uint64_t SFA_STAGE_DELTA = SF::SFA_STAGE_BYTES / 16;  // desc units per stage
  constexpr uint64_t SFB_STAGE_DELTA = SF::SFB_STAGE_BYTES / 16;
  constexpr uint64_t SF_CP_DELTA = 512 / 16;  // desc step between one cp's source and the next
  const int K_BLOCKS = K / K_TILE;
  const uint16_t ctamask = 0x3;
  const uint32_t idesc_id0 = make_idesc_mxf4nvf4(
      M_TILE_CLUSTER, N_TILE_CLUSTER,
      /*ta=*/false, /*tb=*/false,
      /*negate_a=*/false, /*negate_b=*/false,
      /*sf_a_data_id=*/0, /*sf_b_data_id=*/0,
      /*ue8m0=*/true);
  const uint32_t idesc_id2 = make_idesc_mxf4nvf4(
      M_TILE_CLUSTER, N_TILE_CLUSTER,
      /*ta=*/false, /*tb=*/false,
      /*negate_a=*/false, /*negate_b=*/false,
      /*sf_a_data_id=*/2, /*sf_b_data_id=*/2,
      /*ue8m0=*/true);

  wp_begin(wpc, WP_MMA_WAIT_ACC);
  acc_pipeline_2bank_producer_acquire(acc_bars, prod_state);
  if constexpr (N_TILE_CLUSTER == 256) {
    // Overlapped banks: abs cols [192,256) are in EVERY tile's MMA
    // footprint, so this is a depth-1 handshake -- tile t waits for
    // epi(t-1)'s early window release. One bar, one completion per
    // tile: parity = (t + 1) & 1 (fresh-bar parity 1 passes for t=0).
    uint32_t ovl_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&ovl_bars.acc_empty[0]));
    mbarrier_wait_parity(ovl_addr, (prod_state.count + 1) & 1);
  }
  wp_end(wpc, WP_MMA_WAIT_ACC);
  const int acc_stage = acc_pipeline_2bank_state_index(prod_state);
  const uint32_t tmem_c =
      tmem_base + (uint32_t)(acc_stage * (int)SF::ACC_BANK_STRIDE);

  for (int k = 0; k < K_BLOCKS; ++k) {
    const int s = full_ph.get_stage();
    wp_begin(wpc, WP_MMA_WAIT_FULL);
    mbarrier_wait_parity(smem_ptr_u32(&full_bar[s]), full_ph.get_phase());
    wp_end(wpc, WP_MMA_WAIT_FULL);

    const uint64_t da_s   = desc_a0   + s * A_STAGE_DELTA;
    const uint64_t db_s   = desc_b0   + s * B_STAGE_DELTA;
    const uint64_t dsfa_s = desc_sfa0 + s * SFA_STAGE_DELTA;
    const uint64_t dsfb_s = desc_sfb0 + s * SFB_STAGE_DELTA;
    // This stage's TMEM scale slot: SFA cells first, then SFB cells.
    // Depth-2 ring relies on the tensor pipe's FIFO cp/mma order (validated
    // by the K1_SF_RING2 probe; the exact verify is the guard).
#if defined(K1_SF_RING2)
    const uint32_t ring_slot = (uint32_t)(s & 1);
#else
    const uint32_t ring_slot =
        SF::TMEM_RING_DEPTH ? (uint32_t)(s & 1) : (uint32_t)s;
#endif
    const uint32_t sfa_t = tmem_base + SF::RING_OFFSET + ring_slot * SF::STAGE_COLS;
    const uint32_t sfb_t = sfa_t + SF::SFA_STAGE_COLS;

    wp_begin(wpc, WP_MMA_ISSUE);
    if (elect_one_sync()) {
      // cps interleaved with the atoms: each atom-pair's cps are issued just
      // before that pair, so the later cps overlap the earlier atoms.
      #pragma unroll
      for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
#if !defined(K1_DISABLE_SF_TMA_CP) && !defined(K1_DISABLE_SF_CP)
        if (ki % SF_ATOMS_PER_CELL == 0) {
          const uint32_t w = (uint32_t)(ki / SF_ATOMS_PER_CELL);
          tcgen05_cp_32x128b_warpx4<2>(sfa_t + K1_SF_NUM_ROW_GROUPS * w,
                                       dsfa_s + w * SF_CP_DELTA);
          if constexpr (N_TILE_CLUSTER == 256) {
            // SFB spans 256 N-rows: 8 cols per K-chunk. SMEM slice order is
            // [band0 c0 c1][band1 c0 c1]; two cps per chunk (row halves).
            tcgen05_cp_32x128b_warpx4<2>(sfb_t + 8u * w,
                                         dsfb_s + w * SF_CP_DELTA);
            tcgen05_cp_32x128b_warpx4<2>(sfb_t + 8u * w + 4u,
                                         dsfb_s + 2 * SF_CP_DELTA + w * SF_CP_DELTA);
          } else {
            tcgen05_cp_32x128b_warpx4<2>(sfb_t + K1_SF_NUM_ROW_GROUPS * w,
                                         dsfb_s + w * SF_CP_DELTA);
          }
        }
#endif
        const bool enable_d = (k != 0) || (ki != 0);
        tcgen05_mma_mxf4nvf4_ss_2sm_block32(
            tmem_c,
            da_s + K_ATOM_DELTA * ki,
            db_s + K_ATOM_DELTA * ki,
            (SF_ATOMS_PER_CELL == 2 && (ki & 1)) ? idesc_id2 : idesc_id0,
            sfa_t + K1_SF_NUM_ROW_GROUPS * (uint32_t)(ki / SF_ATOMS_PER_CELL),
            sfb_t + (SF::SFB_STAGE_COLS / 2) * (uint32_t)(ki / SF_ATOMS_PER_CELL),
            enable_d);
      }
      tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_bar[s]), ctamask);
    }
    wp_end(wpc, WP_MMA_ISSUE);
    full_ph.advance();
  }
  if (elect_one_sync()) {
    acc_pipeline_2bank_producer_commit_cluster<2>(acc_bars, prod_state, ctamask);
  }
}

// Split-ring MMA 1tile (static NTC=256 only). A operands from the NS_A
// ring (slot = k % NS_A, valid because K_BLOCKS % NS_A == 0 keeps tiles
// ring-aligned), B from the NS_B ring (persistent tracker). On skip
// tiles full_a is neither armed nor waited; empties are committed on
// BOTH rings every stage to keep the loader's parity ledger 1:1.
template <int NS_A, int NS_B, int M_TILE_CLUSTER, int N_TILE_CLUSTER,
          int K_TILE, uint64_t A_STAGE_DELTA, uint64_t B_STAGE_DELTA,
          int SF_ATOMS_PER_CELL, int CGA_PAIRS = 1, bool ALIGNED = true>
__device__ __forceinline__
void mma_warp_k1_1tile_2sm_nvfp4_splitb(WpCtx& wpc,
    uint64_t desc_a0, uint64_t desc_b0,
    uint64_t desc_sfa0, uint64_t desc_sfb0,
    uint64_t* full_a_bar, uint64_t* empty_a_bar,
    uint64_t* full_b_bar, uint64_t* empty_b_bar,
    AccPipeline2BankBars acc_bars,
    AccPipeline2BankState& prod_state,
    uint32_t tmem_base, int K,
    PhaseTracker<NS_A>& full_ph_a, PhaseTracker<NS_B>& full_ph_b,
    bool skip_a, uint64_t* run_free_bar, int tile_idx,
    AccPipeline2BankBars ovl_bars,
    uint64_t* b_gate_bar = nullptr, int cga_pair = 0) {
  static_assert(N_TILE_CLUSTER == 128 || N_TILE_CLUSTER == 256);
  constexpr int K_ATOMS_PER_TILE = K_TILE / K1_ATOM_K;
  constexpr uint64_t K_ATOM_DELTA = 2;
  using SF = K1SfLayout<SF_ATOMS_PER_CELL, N_TILE_CLUSTER>;
  constexpr uint64_t SFA_STAGE_DELTA = SF::SFA_STAGE_BYTES / 16;
  constexpr uint64_t SFB_STAGE_DELTA = SF::SFB_STAGE_BYTES / 16;
  constexpr uint64_t SF_CP_DELTA = 512 / 16;
  const int K_BLOCKS = K / K_TILE;
  const uint16_t ctamask = (uint16_t)(0x3 << (2 * cga_pair));
  const uint32_t idesc_id0 = make_idesc_mxf4nvf4(
      M_TILE_CLUSTER, N_TILE_CLUSTER, false, false, false, false,
      /*sf_a_data_id=*/0, /*sf_b_data_id=*/0, /*ue8m0=*/true);
  const uint32_t idesc_id2 = make_idesc_mxf4nvf4(
      M_TILE_CLUSTER, N_TILE_CLUSTER, false, false, false, false,
      /*sf_a_data_id=*/2, /*sf_b_data_id=*/2, /*ue8m0=*/true);

  wp_begin(wpc, WP_MMA_WAIT_ACC);
  acc_pipeline_2bank_producer_acquire(acc_bars, prod_state);
  // NTC=256: the overlap-window wait is taken INSIDE the k=0 iteration,
  // after the k=0 full-bar waits and scale-cp issues, so part of the
  // window-drain chain (~360 ns) hides behind work the warp must do
  // anyway.
  wp_end(wpc, WP_MMA_WAIT_ACC);
  const int acc_stage = acc_pipeline_2bank_state_index(prod_state);
  const uint32_t tmem_c =
      tmem_base + (uint32_t)(acc_stage * (int)SF::ACC_BANK_STRIDE);

  for (int k = 0; k < K_BLOCKS; ++k) {
    // ALIGNED (skip-capable, K_BLOCKS % NS_A == 0): per-tile k % NS_A,
    // valid because tiles start ring-aligned. Otherwise the persistent
    // tracker (advances every stage since skip never fires).
    const int sa = ALIGNED ? (k % NS_A) : full_ph_a.get_stage();
    const int sb = full_ph_b.get_stage();
    wp_begin(wpc, WP_MMA_WAIT_FULL);
    if (!skip_a)
      mbarrier_wait_parity(smem_ptr_u32(&full_a_bar[sa]), full_ph_a.get_phase());
    mbarrier_wait_parity(smem_ptr_u32(&full_b_bar[sb]), full_ph_b.get_phase());
    wp_end(wpc, WP_MMA_WAIT_FULL);

    const uint64_t da_s   = desc_a0   + sa * A_STAGE_DELTA;
    const uint64_t db_s   = desc_b0   + sb * B_STAGE_DELTA;
    const uint64_t dsfa_s = desc_sfa0 + sa * SFA_STAGE_DELTA;
    const uint64_t dsfb_s = desc_sfb0 + sb * SFB_STAGE_DELTA;
    // Depth-2 TMEM scale ring. Depth-1 (every stage the same slot) was
    // MEASURED EXACT even with per-stage-varying scales: atoms latch
    // their scale bytes at dequeue and the
    // next stage's cps sit behind them in the tensor pipe's FIFO. We
    // keep depth 2 anyway -- depth-1 rests on two behaviors PTX does
    // not promise (FIFO dequeue, operand latch at dequeue) and the 24
    // columns it would save are not needed.
    const uint32_t ring_slot = (uint32_t)(k & 1);
    const uint32_t sfa_t = tmem_base + SF::RING_OFFSET + ring_slot * SF::STAGE_COLS;
    const uint32_t sfb_t = sfa_t + SF::SFA_STAGE_COLS;

    wp_begin(wpc, WP_MMA_ISSUE);
    const bool hoist = (N_TILE_CLUSTER == 256) && (k == 0);
    // (Hoisting stage-1's cps too was tested and LOSES ~0.3pp: the
    // stage-1 full-bar peek waits stall the warp earlier than the
    // window wait would.)
    if (hoist) {
#if !defined(K1_DISABLE_SF_TMA_CP) && !defined(K1_DISABLE_SF_CP)
      if (elect_one_sync()) {
        #pragma unroll
        for (int w2 = 0; w2 < K_ATOMS_PER_TILE / SF_ATOMS_PER_CELL; ++w2) {
          const uint32_t w = (uint32_t)w2;
          tcgen05_cp_32x128b_warpx4<2>(sfa_t + K1_SF_NUM_ROW_GROUPS * w,
                                       dsfa_s + w * SF_CP_DELTA);
          tcgen05_cp_32x128b_warpx4<2>(sfb_t + 8u * w,
                                       dsfb_s + w * SF_CP_DELTA);
          tcgen05_cp_32x128b_warpx4<2>(sfb_t + 8u * w + 4u,
                                       dsfb_s + 2 * SF_CP_DELTA + w * SF_CP_DELTA);
        }
      }
#endif
      uint32_t ovl_addr = static_cast<uint32_t>(
          __cvta_generic_to_shared(&ovl_bars.acc_empty[0]));
      mbarrier_wait_parity(ovl_addr, (prod_state.count + 1) & 1);
    }
    if (elect_one_sync()) {
      #pragma unroll
      for (int ki = 0; ki < K_ATOMS_PER_TILE; ++ki) {
#if !defined(K1_DISABLE_SF_TMA_CP) && !defined(K1_DISABLE_SF_CP)
        if (!hoist && ki % SF_ATOMS_PER_CELL == 0) {
          const uint32_t w = (uint32_t)(ki / SF_ATOMS_PER_CELL);
          tcgen05_cp_32x128b_warpx4<2>(sfa_t + K1_SF_NUM_ROW_GROUPS * w,
                                       dsfa_s + w * SF_CP_DELTA);
          if constexpr (N_TILE_CLUSTER == 256) {
            tcgen05_cp_32x128b_warpx4<2>(sfb_t + 8u * w,
                                         dsfb_s + w * SF_CP_DELTA);
            tcgen05_cp_32x128b_warpx4<2>(sfb_t + 8u * w + 4u,
                                         dsfb_s + 2 * SF_CP_DELTA + w * SF_CP_DELTA);
          } else {
            tcgen05_cp_32x128b_warpx4<2>(sfb_t + K1_SF_NUM_ROW_GROUPS * w,
                                         dsfb_s + w * SF_CP_DELTA);
          }
        }
#endif
        const bool enable_d = (k != 0) || (ki != 0);
        tcgen05_mma_mxf4nvf4_ss_2sm_block32(
            tmem_c,
            da_s + K_ATOM_DELTA * ki,
            db_s + K_ATOM_DELTA * ki,
            (SF_ATOMS_PER_CELL == 2 && (ki & 1)) ? idesc_id2 : idesc_id0,
            sfa_t + K1_SF_NUM_ROW_GROUPS * (uint32_t)(ki / SF_ATOMS_PER_CELL),
            sfb_t + (SF::SFB_STAGE_COLS / 2) * (uint32_t)(ki / SF_ATOMS_PER_CELL),
            enable_d);
      }
      if (!skip_a)
        tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_a_bar[sa]), ctamask);
      if constexpr (CGA_PAIRS == 2) {
        // B slot free for THIS pair: post at rank 0's gate (count 2 --
        // the other pair's MMA posts the second arrival).
        tcgen05_commit_multicast<2>(smem_ptr_u32(&b_gate_bar[sb]),
                                    /*ctamask=*/0x1);
      } else {
        tcgen05_commit_multicast<2>(smem_ptr_u32(&empty_b_bar[sb]), ctamask);
      }
    }
    wp_end(wpc, WP_MMA_ISSUE);
    if (!skip_a) full_ph_a.advance();
    full_ph_b.advance();
  }
  if (elect_one_sync()) {
    acc_pipeline_2bank_producer_commit_cluster<2>(acc_bars, prod_state, ctamask);
    tcgen05_commit_multicast<2>(
        smem_ptr_u32(&run_free_bar[tile_idx & 7]), ctamask);
  }
}

// ============================================================================
// MMA warp: persistent CLC loop
// ============================================================================
template <int NUM_STAGES, int M_TILE_CLUSTER, int N_TILE_CLUSTER, int K_TILE,
          uint64_t A_STAGE_DELTA, uint64_t B_STAGE_DELTA, int SF_ATOMS_PER_CELL,
          int CLUSTER_SHAPE_M = 1, int CLUSTER_SHAPE_N = 2,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          bool RUN_CLC = false>
__device__ inline
void mma_warp_k1_ntiles_2sm_nvfp4(WpCtx& wpc,
    uint64_t desc_a0, uint64_t desc_b0,
    uint64_t desc_sfa0, uint64_t desc_sfb0,
    uint64_t* full_bar, uint64_t* empty_bar,
    AccPipeline2BankBars acc_bars,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    uint64_t* tmem_dealloc_bar,
    int K, int peer, int lane,
    uint32_t* tmem_slot, int run_len = 1,
    AccPipeline2BankBars ovl_bars = {nullptr, nullptr}) {
  // TMEM: two acc banks (overlapping at NTC=256), then the scale ring.
  using SFN = K1SfLayout<SF_ATOMS_PER_CELL, N_TILE_CLUSTER>;
  static_assert((int)SFN::ACC_BANK_STRIDE + N_TILE_CLUSTER
                    + (SFN::TMEM_RING_DEPTH ? SFN::TMEM_RING_DEPTH : NUM_STAGES)
                          * (int)SFN::STAGE_COLS
                    <= K1_TMEM_NCOLS,
                "acc banks + scale ring exceed the 512-col TMEM budget");
  AccPipeline2BankState prod_state = acc_pipeline_2bank_state_init();
  PhaseTracker<NUM_STAGES> full_ph;
  int      clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  (void)lane;

  wp_begin(wpc, WP_MMA_TMEM_2CTA_ALLOC);
  tcgen05_alloc<2>(smem_ptr_u32(tmem_slot), K1_TMEM_NCOLS);
  asm volatile("bar.arrive 6, 160;\n" ::: "memory");
  uint32_t tmem_base = *tmem_slot;
  wp_end(wpc, WP_MMA_TMEM_2CTA_ALLOC);

#if defined(K1_DISABLE_SF_TMA_CP) || defined(K1_DISABLE_SF_CP)
  // No scale delivery: fill the whole scale ring with UE8M0 0x7F (= 1.0)
  // once, so the atoms read a defined value and verify expects unscaled
  // results. 32x32b writes this warp's 32 TMEM rows, which is all the MMA
  // scale read uses (the constant-scale kernel verified with exactly this).
  using SFI = K1SfLayout<SF_ATOMS_PER_CELL, N_TILE_CLUSTER>;
  constexpr uint32_t SFI_RING_COLS =
      (SFI::TMEM_RING_DEPTH ? (uint32_t)SFI::TMEM_RING_DEPTH
                            : (uint32_t)NUM_STAGES) * SFI::STAGE_COLS;
  for (uint32_t col = SFI::RING_OFFSET;
       col < SFI::RING_OFFSET + SFI_RING_COLS;
       ++col) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n"
                 :: "r"(tmem_base + col), "r"(0x7F7F7F7Fu));
  }
  tcgen05_fence_before_thread_sync();
#endif

  while (true) {
    const int rl = RUN_CLC ? run_len : 1;
    for (int pos = 0; pos < rl; ++pos) {
      if (peer == 0) {
        mma_warp_k1_1tile_2sm_nvfp4<
            NUM_STAGES, M_TILE_CLUSTER, N_TILE_CLUSTER, K_TILE,
            A_STAGE_DELTA, B_STAGE_DELTA, SF_ATOMS_PER_CELL>(
            wpc, desc_a0, desc_b0, desc_sfa0, desc_sfb0,
            full_bar, empty_bar, acc_bars, prod_state,
            tmem_base, K, full_ph, ovl_bars);
      }
      acc_pipeline_2bank_state_advance(prod_state);
    }
    wp_begin(wpc, WP_CLC_FETCH);
    ClcTileInfo next = clc_fetch_next_tile<CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, elect_one_sync());
    wp_end(wpc, WP_CLC_FETCH);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
  }

  // Tail teardown.
  wp_begin(wpc, WP_MMA_WAIT_ACC);
  if (peer == 0) {
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
      acc_pipeline_2bank_producer_acquire(acc_bars, prod_state);
      acc_pipeline_2bank_state_advance(prod_state);
    }
  }
  wp_end(wpc, WP_MMA_WAIT_ACC);

  const uint32_t bar_local = smem_ptr_u32(tmem_dealloc_bar);
  const uint32_t bar_peer  = bar_local ^ K1_SM100_PEER_BIT;
  mbarrier_arrive_cluster_default(bar_peer);
  wp_begin(wpc, WP_MMA_TMEM_2CTA_FREE);
  mbarrier_wait_parity(bar_local, 0);
  wp_end(wpc, WP_MMA_TMEM_2CTA_FREE);
  tcgen05_relinquish_alloc_permit<2>();
  tcgen05_dealloc<2>(tmem_base, K1_TMEM_NCOLS);
}

// ============================================================================
// K1-local epilogue: early TMEM release. Both 64-col tcgen05.ld's are issued
// up front and the acc bank is released right after their wait::ld, BEFORE
// any cvt/STS/TMA work. The shared-block epi holds the bank through the
// whole 4-sub pipeline (~1.3 us), which paces the MMA at short K
// (acc_wait 271 ns/tile at K=2048). Stores still walk 4 x 32-col sub-buffers
// over the same 2-buffer D ring.
// ============================================================================
template <int M_TILE_PER_CTA, int N_TILE_CLUSTER, int EPI_SUB_COLS,
          int EPI_NUM_BUFS, bool DRAIN_PER_TILE>
__device__ __forceinline__
void k1_epi_1tile_earlyrelease(WpCtx& wpc,
    const CUtensorMap* tma_d, uint8_t* smem_d,
    AccPipeline2BankBars acc_bars, AccPipeline2BankState& cons_state,
    uint32_t tmem_base, int peer, int warp, int lane,
    int m_offset, int n_offset_d) {
  static_assert(N_TILE_CLUSTER == 128 &&
                (EPI_SUB_COLS == 32 || EPI_SUB_COLS == 16) &&
                (EPI_NUM_BUFS == 2 || EPI_NUM_BUFS == 4) && !DRAIN_PER_TILE,
                "early-release epi is specialized to the K1 configuration");
  constexpr int EPI_SUB_COUNT = N_TILE_CLUSTER / EPI_SUB_COLS;  // 4
  constexpr int EPI_BUF_BYTES = M_TILE_PER_CTA * EPI_SUB_COLS *
                                static_cast<int>(sizeof(__nv_bfloat16));
  (void)peer;
  const int epi_warp = (warp >= 4) ? (warp - 4) : 0;
  const uint32_t my_tmem_row_offset = ((uint32_t)(epi_warp * 32) << 16);
  const int row = epi_warp * 32 + lane;

  __nv_bfloat16* d_smem_buf[EPI_NUM_BUFS];
  #pragma unroll
  for (int b = 0; b < EPI_NUM_BUFS; ++b)
    d_smem_buf[b] = reinterpret_cast<__nv_bfloat16*>(smem_d + b * EPI_BUF_BYTES);

  wp_begin(wpc, WP_EPI_WAIT_ACC);
  acc_pipeline_2bank_consumer_wait(acc_bars, cons_state);
  wp_end(wpc, WP_EPI_WAIT_ACC);
  const int acc_stage_cons = acc_pipeline_2bank_state_index(cons_state);
  const uint32_t my_tmem = tmem_base + my_tmem_row_offset
                           + (uint32_t)(acc_stage_cons * N_TILE_CLUSTER);

  uint32_t regs[128];
  wp_begin(wpc, WP_EPI_TMEM_LD);
  tcgen05_ld_32x32b_x128(my_tmem, regs);
  tcgen05_wait_ld();
  wp_end(wpc, WP_EPI_TMEM_LD);
  acc_pipeline_2bank_consumer_release(acc_bars, cons_state);
  acc_pipeline_2bank_state_advance(cons_state);

  // (Direct st.global from lanes-as-rows measured -43% -- 32-way address
  // divergence per store; the SMEM bounce + TMA box store IS the coalescer.)

  // Per-warp store pipeline: each epi warp owns its 32-row slice of
  // every buffer and runs its own TMA chain (wait_group state is
  // per-thread). No CTA-wide bar.sync on the store path -- a
  // 2 x bar.sync(128) per sub plus single-lane issue costs ~1.1 us/tile,
  // pacing the MMA at short K.
  #pragma unroll
  for (int sub = 0; sub < EPI_SUB_COUNT; ++sub) {
    const int buf  = sub & (EPI_NUM_BUFS - 1);
    const int col0 = sub * EPI_SUB_COLS;

    wp_begin(wpc, WP_EPI_WAIT_STORE);
    if (lane == 0) {
      cp_async_bulk_wait_group<EPI_NUM_BUFS - 1>();
    }
    __syncwarp();
    wp_end(wpc, WP_EPI_WAIT_STORE);

    wp_begin(wpc, WP_EPI_STORE);
    uint32_t* d_row_u32 = reinterpret_cast<uint32_t*>(
        d_smem_buf[buf] + row * EPI_SUB_COLS);
    #pragma unroll
    for (int j = 0; j < EPI_SUB_COLS; j += 2) {
      const float a = __int_as_float(regs[col0 + j]);
      const float b = __int_as_float(regs[col0 + j + 1]);
      d_row_u32[j >> 1] = cvt_pack_f32_to_bf16x2(a, b);
    }
    __syncwarp();

    if (lane == 0) {
      fence_proxy_async_shared_cta();
      const uint32_t d_smem_addr = smem_ptr_u32(
          d_smem_buf[buf] + epi_warp * 32 * EPI_SUB_COLS);
      tma_store_2d(tma_d, /*x=*/n_offset_d + col0,
                   /*y=*/m_offset + epi_warp * 32, d_smem_addr);
      cp_async_bulk_commit_group();
    }
    wp_end(wpc, WP_EPI_STORE);
  }
}

// ============================================================================
// K1-local epi ntiles wrapper for RUN_CLC: one CLC grab = run_len tiles,
// and the early-release 1tile drain is used. Mirrors the shared block's
// ntiles wrapper otherwise (clc_empty released by warp 4 lane 0 per grab).
// ============================================================================
template <int M_TILE_PER_CTA, int N_TILE_CLUSTER, int EPI_SUB_COLS,
          int EPI_NUM_BUFS, int CLUSTER_SHAPE_M, int CLUSTER_SHAPE_N,
          ClcRasterOrder ORDER>
__device__ inline
void k1_epi_ntiles_runclc(WpCtx& wpc,
    const CUtensorMap* tma_d, uint8_t* smem_d,
    AccPipeline2BankBars acc_bars,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    int peer, int warp, int lane, uint32_t* tmem_slot, int run_len) {
  constexpr int M_TILE_CLUSTER = 2 * M_TILE_PER_CTA;  // 2SM M-split
  AccPipeline2BankState cons_state = acc_pipeline_2bank_state_init();

  wp_begin(wpc, WP_EPI_WAIT_TMEM);
  asm volatile("bar.sync 6, 160;\n" ::: "memory");
  const uint32_t tmem_base = *tmem_slot;
  wp_end(wpc, WP_EPI_WAIT_TMEM);

  int clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = (int)blockIdx.y;
    n_tile = (int)(blockIdx.x >> 1);
  } else {
    m_tile = (int)(blockIdx.x >> 1);
    n_tile = (int)blockIdx.y;
  }

  while (true) {
    const int m_offset = m_tile * M_TILE_CLUSTER + peer * M_TILE_PER_CTA;
    for (int pos = 0; pos < run_len; ++pos) {
      const int n_offset_d = (n_tile * run_len + pos) * N_TILE_CLUSTER;
      k1_epi_1tile_earlyrelease<
          M_TILE_PER_CTA, N_TILE_CLUSTER, EPI_SUB_COLS, EPI_NUM_BUFS,
          /*DRAIN_PER_TILE=*/false>(
          wpc, tma_d, smem_d, acc_bars, cons_state,
          tmem_base, peer, warp, lane, m_offset, n_offset_d);
    }
    // With the per-warp store pipeline the epi warps are no longer in
    // per-sub lockstep, so warp 4 must not release the CLC slot until
    // ALL four epi warps have parsed the response (a lagging warp
    // would otherwise read a refilled slot -- observed as a hang on
    // multi-grab NTC128 CLC shapes). Fetch without
    // release, bar-sync the epi warps, then release.
    wp_begin(wpc, WP_CLC_FETCH);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/false);
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
    if (warp == 4 && lane == 0) {
      uint32_t empty_local = static_cast<uint32_t>(
          __cvta_generic_to_shared(&clc_empty_bar[clc_cons_stage]));
      clc_consumer_release(empty_local);
    }
    wp_end(wpc, WP_CLC_FETCH);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = next.m_tile;
    n_tile = next.n_tile;
  }
  wp_begin(wpc, WP_EPI_WAIT_STORE);
  if (lane == 0) cp_async_bulk_wait_group<0>();
  asm volatile("bar.sync 1, 128;\n" ::: "memory");
  wp_end(wpc, WP_EPI_WAIT_STORE);
}

template <int M_TILE_PER_CTA, int EPI_SUB_COLS, int EPI_NUM_BUFS>
__device__ __forceinline__
void k1_epi_1tile_ntc256(WpCtx& wpc,
    const CUtensorMap* tma_d, uint8_t* smem_d,
    AccPipeline2BankBars acc_bars, AccPipeline2BankBars ovl_bars,
    AccPipeline2BankState& cons_state,
    uint32_t tmem_base, int peer, int warp, int lane,
    int m_offset, int n_offset_d);

// CLC ntiles epilogue at NTC=256: one tile per CLC grab, each drained by
// k1_epi_1tile_ntc256 (overlap window first + ovl release). Mirrors
// k1_epi_ntiles_runclc's CLC-consume mechanics.
template <int M_TILE_PER_CTA, int N_TILE_CLUSTER, int EPI_SUB_COLS,
          int EPI_NUM_BUFS, int CLUSTER_SHAPE_M, int CLUSTER_SHAPE_N,
          ClcRasterOrder ORDER>
__device__ inline
void k1_epi_ntiles_clc256(WpCtx& wpc,
    const CUtensorMap* tma_d, uint8_t* smem_d,
    AccPipeline2BankBars acc_bars, AccPipeline2BankBars ovl_bars,
    uint64_t* clc_full_bar, uint64_t* clc_empty_bar, uint32_t* clc_response,
    int peer, int warp, int lane, uint32_t* tmem_slot) {
  constexpr int M_TILE_CLUSTER = 2 * M_TILE_PER_CTA;  // 2SM M-split
  AccPipeline2BankState cons_state = acc_pipeline_2bank_state_init();

  wp_begin(wpc, WP_EPI_WAIT_TMEM);
  asm volatile("bar.sync 6, 160;\n" ::: "memory");
  const uint32_t tmem_base = *tmem_slot;
  wp_end(wpc, WP_EPI_WAIT_TMEM);

  int clc_cons_stage = 0;
  uint32_t clc_cons_phase = 0;
  int m_tile, n_tile;
  if constexpr (ORDER == ClcRasterOrder::AlongN) {
    m_tile = (int)blockIdx.y;
    n_tile = (int)(blockIdx.x >> 1);
  } else {
    m_tile = (int)(blockIdx.x >> 1);
    n_tile = (int)blockIdx.y;
  }

  while (true) {
    const int m_offset = m_tile * M_TILE_CLUSTER + peer * M_TILE_PER_CTA;
    const int n_offset_d = n_tile * N_TILE_CLUSTER;
    k1_epi_1tile_ntc256<M_TILE_PER_CTA, EPI_SUB_COLS, EPI_NUM_BUFS>(
        wpc, tma_d, smem_d, acc_bars, ovl_bars, cons_state,
        tmem_base, peer, warp, lane, m_offset, n_offset_d);
    // With the per-warp store pipeline the epi warps are no longer in
    // per-sub lockstep, so warp 4 must not release the CLC slot until
    // ALL four epi warps have parsed the response (a lagging warp
    // would otherwise read a refilled slot -- observed as a hang on
    // multi-grab NTC128 CLC shapes). Fetch without
    // release, bar-sync the epi warps, then release.
    wp_begin(wpc, WP_CLC_FETCH);
    ClcTileInfo next = clc_fetch_next_tile<
        CLUSTER_SHAPE_M, CLUSTER_SHAPE_N, ORDER>(
        clc_full_bar, clc_empty_bar, clc_response,
        clc_cons_stage, clc_cons_phase, /*do_release=*/false);
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
    if (warp == 4 && lane == 0) {
      uint32_t empty_local = static_cast<uint32_t>(
          __cvta_generic_to_shared(&clc_empty_bar[clc_cons_stage]));
      clc_consumer_release(empty_local);
    }
    wp_end(wpc, WP_CLC_FETCH);
    clc_fetch_next_tile_advance(clc_cons_stage, clc_cons_phase);
    if (!next.valid) break;
    m_tile = next.m_tile;
    n_tile = next.n_tile;
  }
  wp_begin(wpc, WP_EPI_WAIT_STORE);
  if (lane == 0) cp_async_bulk_wait_group<0>();
  asm volatile("bar.sync 1, 128;\n" ::: "memory");
  wp_end(wpc, WP_EPI_WAIT_STORE);
}

// ============================================================================
// NTC=256 epilogue: overlapped banks. Bank base = acc_stage * 192; the abs
// col window [192,256) is shared by both banks, so it is read FIRST and
// released via ovl_bars (acc_empty-style, count 2) before the rest of the
// drain; acc_empty itself is released after the last tcgen05.wait::ld.
// Per CTA the epi reads all 256 cols (M-split). Stores walk 8 x 32-col
// sub-tiles over the same 2-buffer D ring.
// ============================================================================
template <int M_TILE_PER_CTA, int EPI_SUB_COLS, int EPI_NUM_BUFS>
__device__ __forceinline__
void k1_epi_1tile_ntc256(WpCtx& wpc,
    const CUtensorMap* tma_d, uint8_t* smem_d,
    AccPipeline2BankBars acc_bars, AccPipeline2BankBars ovl_bars,
    AccPipeline2BankState& cons_state,
    uint32_t tmem_base, int peer, int warp, int lane,
    int m_offset, int n_offset_d) {
  static_assert((EPI_SUB_COLS == 32 || EPI_SUB_COLS == 16)
                && EPI_NUM_BUFS == 2);
  constexpr int EPI_BUF_BYTES = M_TILE_PER_CTA * EPI_SUB_COLS *
                                static_cast<int>(sizeof(__nv_bfloat16));
  (void)peer;
  const int epi_warp = (warp >= 4) ? (warp - 4) : 0;
  const uint32_t my_tmem_row_offset = ((uint32_t)(epi_warp * 32) << 16);
  const int row = epi_warp * 32 + lane;

  __nv_bfloat16* d_smem_buf[EPI_NUM_BUFS];
  #pragma unroll
  for (int b = 0; b < EPI_NUM_BUFS; ++b)
    d_smem_buf[b] = reinterpret_cast<__nv_bfloat16*>(smem_d + b * EPI_BUF_BYTES);

  wp_begin(wpc, WP_EPI_WAIT_ACC);
  acc_pipeline_2bank_consumer_wait(acc_bars, cons_state);
  wp_end(wpc, WP_EPI_WAIT_ACC);
  const int acc_stage_cons = acc_pipeline_2bank_state_index(cons_state);
  constexpr int ACC_BANK_STRIDE_EPI = 192;
  const uint32_t bank = tmem_base + my_tmem_row_offset
                        + (uint32_t)(acc_stage_cons * ACC_BANK_STRIDE_EPI);
  // Relative col range of the shared abs window [192,256).
  const uint32_t ovl0 = (acc_stage_cons == 0) ? 192u : 0u;

  auto store_sub = [&](int sub, const uint32_t* r) {
    const int buf  = sub & (EPI_NUM_BUFS - 1);
    const int col0 = sub * EPI_SUB_COLS;
    wp_begin(wpc, WP_EPI_WAIT_STORE);
    if (warp == 4 && lane == 0) cp_async_bulk_wait_group<EPI_NUM_BUFS - 1>();
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
    wp_end(wpc, WP_EPI_WAIT_STORE);
    wp_begin(wpc, WP_EPI_STORE);
    uint32_t* d_row_u32 = reinterpret_cast<uint32_t*>(
        d_smem_buf[buf] + row * EPI_SUB_COLS);
    #pragma unroll
    for (int j = 0; j < EPI_SUB_COLS; j += 2)
      d_row_u32[j >> 1] = cvt_pack_f32_to_bf16x2(__int_as_float(r[j]),
                                                 __int_as_float(r[j + 1]));
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
    if (warp == 4 && lane == 0) {
      fence_proxy_async_shared_cta();
      tma_store_2d(tma_d, /*x=*/n_offset_d + col0, /*y=*/m_offset,
                   smem_ptr_u32(d_smem_buf[buf]));
      cp_async_bulk_commit_group();
    }
    wp_end(wpc, WP_EPI_STORE);
  };

  // 1) Shared-window cols first: read, release ovl, store its subs.
  uint32_t rov[64];
  wp_begin(wpc, WP_EPI_TMEM_LD);
  tcgen05_ld_32x32b_x64(bank + ovl0, rov);
  tcgen05_wait_ld();
  wp_end(wpc, WP_EPI_TMEM_LD);
  if (lane == 0) {
    uint32_t ovl_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&ovl_bars.acc_empty[0]));
    mbarrier_arrive_cluster_default(ovl_addr & SM100_ACC_PIPE_2BANK_PEER_MASK);
  }
  #pragma unroll
  for (int i = 0; i < 64 / EPI_SUB_COLS; ++i)
    store_sub((int)(ovl0 / EPI_SUB_COLS) + i, rov + EPI_SUB_COLS * i);

  // 2) The remaining 192 cols: x128 + x64, then release the bank.
  const uint32_t rest0 = (acc_stage_cons == 0) ? 0u : 64u;
  uint32_t r0[128];
  wp_begin(wpc, WP_EPI_TMEM_LD);
  tcgen05_ld_32x32b_x128(bank + rest0, r0);
  tcgen05_wait_ld();
  wp_end(wpc, WP_EPI_TMEM_LD);
  #pragma unroll
  for (int i = 0; i < 128 / EPI_SUB_COLS; ++i)
    store_sub((int)(rest0 / EPI_SUB_COLS) + i, r0 + EPI_SUB_COLS * i);

  uint32_t r1[64];
  const uint32_t rest1 = rest0 + 128u;
  wp_begin(wpc, WP_EPI_TMEM_LD);
  tcgen05_ld_32x32b_x64(bank + rest1, r1);
  tcgen05_wait_ld();
  wp_end(wpc, WP_EPI_TMEM_LD);
  acc_pipeline_2bank_consumer_release(acc_bars, cons_state);
  acc_pipeline_2bank_state_advance(cons_state);
  #pragma unroll
  for (int i = 0; i < 64 / EPI_SUB_COLS; ++i)
    store_sub((int)(rest1 / EPI_SUB_COLS) + i, r1 + EPI_SUB_COLS * i);
}

// ============================================================================
// Kernel. One body, two tile schedulers, chosen by the STATIC_SCHED flag:
//
// STATIC_SCHED == false (CLC, default):
//   clusterlaunchcontrol.try_cancel. Warps 1 and 3 run the sched / idle
//   roles; the hardware picks the tile order, so BLOCK_M / BLOCK_N do not
//   apply. total_tiles / num_clusters are ignored; grid is 2D.
//
// STATIC_SCHED == true:
//   Each cluster walks the tile list with a fixed stride (cluster_id,
//   cluster_id + num_clusters, ...). Warps 1 and 3 have nothing to do and
//   retire. Grid is 1D (2 * num_clusters CTAs). The tile order is ours,
//   which is what makes the BLOCK_M x BLOCK_N blocked raster possible
//   (see k1_decode_tile).
//
// On throughput the two are a wash (gen-o 71.1% vs 70.9% SOL); Static exists
// for the tile ordering, not for the dispatch cost.
//
// Gotcha (Static): warps 1 and 3 must NOT arrive at named barrier 6, and must
// NOT fall through to griddepcontrol_launch_dependents(). Bar 6 is sized
// 160 = 32 (MMA, after tcgen05.alloc) + 128 (EPI); two extra warps make it
// complete early and the EPI warps then read an unwritten tmem_slot. An early
// griddepcontrol lets the next launch start before this one drains, which
// silently shortens every timed iteration.
//
// The SMEM bar block is laid out for the CLC superset in both modes. Static
// leaves the CLC and throttle bars allocated but uninitialized and unused;
// that costs ~112 B and keeps one layout to reason about.
// ============================================================================
template <int K_BLOCKS_T, int NTC, int NS = K1_NUM_STAGES,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          bool STATIC_SCHED = false,
          int BLOCK_M = 0, int BLOCK_N = 0,
          int SF_ATOMS_PER_CELL = 2,
          bool M_RUN = false,
          int NS_B = 0,
          int CGA_PAIRS = 1>
__global__ void __launch_bounds__(256, 1)
dense_gemm_nvfp4_k1_impl(
    const __grid_constant__ CUtensorMap tmap_a,
    const __grid_constant__ CUtensorMap tmap_b,
    const __grid_constant__ CUtensorMap tmap_sfa,
    const __grid_constant__ CUtensorMap tmap_sfb,
    const __grid_constant__ CUtensorMap tmap_d,
    int M, int N, int total_tiles, int num_clusters, int run_len,
    int gate_early) {
  static_assert(STATIC_SCHED || (BLOCK_M == 0 && BLOCK_N == 0),
                "BLOCK_M/BLOCK_N need STATIC_SCHED; CLC picks the tile order");
  constexpr int K              = K_BLOCKS_T * K1_K_TILE;
  constexpr int M_TILE_CLUSTER = K1_M_TILE_CLUSTER;
  constexpr int M_TILE_PER_CTA = K1_M_TILE_PER_CTA;
  constexpr int N_TILE_CLUSTER = NTC;
  constexpr int N_TILE_PER_CTA = NTC / 2;
  constexpr int K_TILE         = K1_K_TILE;
  constexpr int NUM_STAGES     = NS;
  constexpr int EPI_SUB_COLS   = K1_EPI_SUB_COLS;
  constexpr int EPI_NUM_BUFS   = k1_epi_num_bufs(NTC);
  constexpr int A_TILE_BYTES   = K1_M_TILE_PER_CTA * K_TILE / 2;
  constexpr int B_TILE_BYTES   = N_TILE_PER_CTA * K_TILE / 2;

  constexpr int CLC_CSM = (ORDER == ClcRasterOrder::AlongN) ? 1 : 2;
  constexpr int CLC_CSN = (ORDER == ClcRasterOrder::AlongN) ? 2 : 1;

  static_assert(NS_B == 0
                || (STATIC_SCHED && (K_BLOCKS_T % NS == 0 || !M_RUN)),
                "split B ring: static; ring-aligned K required for M_RUN");
  static_assert(CGA_PAIRS == 1
                || (CGA_PAIRS == 2 && STATIC_SCHED && NS_B > 0 && NTC == 256),
                "CGA_PAIRS=2 needs the static NTC=256 split-ring path");
  // Split-ring mode (NS_B > 0): full_bar/empty_bar become the A ring
  // (depth NS) and a second B ring of depth NS_B is carved after them.
  constexpr int MAIN_BARS = NUM_STAGES * 2
      + ((NS_B > 0) ? 2 * NS_B + 8 : 0)
      + ((CGA_PAIRS == 2) ? NS_B : 0);
  uint64_t* smem_bar       = reinterpret_cast<uint64_t*>(k1_smem);
  uint64_t* full_bar       = smem_bar + 0;
  uint64_t* empty_bar      = smem_bar + NUM_STAGES;
  uint64_t* full_b_bar     = smem_bar + NUM_STAGES * 2;            // [NS_B]
  uint64_t* empty_b_bar    = smem_bar + NUM_STAGES * 2 + NS_B;     // [NS_B]
  uint64_t* run_free_bar   = smem_bar + NUM_STAGES * 2 + 2 * NS_B; // [8]
  // CGA_PAIRS=2: B-slot gate in EVERY CTA's SMEM at the same offset; only
  // rank 0's copy is waited (sole B issuer). Count 2 = one tcgen05.commit
  // from each pair's MMA per slot round.
  uint64_t* b_gate_bar     = smem_bar + NUM_STAGES * 2 + 2 * NS_B + 8; // [NS_B]
  uint64_t* acc_full       = smem_bar + MAIN_BARS + 0;   // [2]
  uint64_t* acc_empty      = smem_bar + MAIN_BARS + 3;   // [2]
  uint64_t* clc_full_bar   = smem_bar + MAIN_BARS + 6;
  uint64_t* clc_empty_bar  = smem_bar + MAIN_BARS + 8;
  uint64_t* throttle_full  = smem_bar + MAIN_BARS + 10;
  uint64_t* throttle_empty = smem_bar + MAIN_BARS + 12;
  uint64_t* tmem_dealloc_bar = smem_bar + MAIN_BARS + 14;
  uint64_t* ovl_free       = smem_bar + MAIN_BARS + 15;  // [2], NTC=256 only
  // +18, not +17: the CLC try_cancel response is written 16 B at a time
  // and needs 16-B alignment (slot parity must stay even).
  uint32_t* clc_response   = reinterpret_cast<uint32_t*>(smem_bar + MAIN_BARS + 18);
  uint32_t* tmem_slot      = clc_response + 8;

  constexpr int BAR_BYTES = (MAIN_BARS + 18) * 8 + 8 * 4 + 4;
  constexpr int DATA_OFF  = ((BAR_BYTES + 127) / 128) * 128;
  constexpr int NSB_DATA = (NS_B > 0) ? NS_B : NUM_STAGES;
  uint8_t* smem_a   = k1_smem + DATA_OFF;
  uint8_t* smem_b   = smem_a   + NUM_STAGES * A_TILE_BYTES;
  uint8_t* smem_sfa = smem_b   + NSB_DATA * B_TILE_BYTES;
  using SF = K1SfLayout<SF_ATOMS_PER_CELL, NTC>;
#if defined(K1_DISABLE_SF_TMA_CP)
  uint8_t* smem_sfb = smem_sfa;  // no scale ring in SMEM at all
  uint8_t* smem_d   = smem_sfb;
#else
  uint8_t* smem_sfb = smem_sfa + NUM_STAGES * SF::SFA_STAGE_BYTES;
  uint8_t* smem_d   = smem_sfb + NSB_DATA * SF::SFB_STAGE_BYTES;
#endif

  AccPipeline2BankBars acc_bars{ acc_full, acc_empty };
  // Static never touches the CLC / throttle rings, so leave them uninitialized.
  BlackwellPipelineBars pipe_bars =
      (!STATIC_SCHED)
          ? BlackwellPipelineBars{ full_bar, empty_bar, nullptr, nullptr,
                                   clc_full_bar, clc_empty_bar,
                                   throttle_full, throttle_empty }
          : BlackwellPipelineBars{ full_bar, empty_bar, nullptr, nullptr,
                                   nullptr, nullptr, nullptr, nullptr };

  const int peer = blockIdx.x & 1;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  WpCtx wpc = wp_ctx_init();

  // Static only: this cluster's slot in the strided walk over total_tiles.
  // CGA_PAIRS=2: two MMA pairs per launch cluster share one owner unit;
  // the unit decodes over an m-PAIR grid and pair p takes m = 2*mm + p.
  const int cga_pair       = (CGA_PAIRS == 2) ? ((int)(blockIdx.x >> 1) & 1) : 0;
  const int cluster_id     = (int)(blockIdx.x >> 1) / CGA_PAIRS;
  const int m_clusters_dim = M / (M_TILE_CLUSTER * CGA_PAIRS);
  const int n_clusters_dim = N / N_TILE_CLUSTER;

  if (threadIdx.x == 0) {
    if constexpr (!STATIC_SCHED) {
      for (int i = 0; i < 8; ++i) clc_response[i] = 0;
    }
    mbarrier_init(smem_ptr_u32(tmem_dealloc_bar), 32);
    // acc bars are K1-owned; the composite init below gets nullptr.
    // (A 3rd acc bank for static NTC=128 was REFUTED: the deeper MMA/epi
    // overlap slows the epi store path by about what it saves in acc_wait
    // -- net -6..-10% on qkv rows.)
    for (int i = 0; i < 2; ++i) {
      mbarrier_init(smem_ptr_u32(&acc_full[i]), 1);
      mbarrier_init(smem_ptr_u32(&acc_empty[i]), 256);
    }
    if constexpr (NTC == 256) {
      // 8 = lane 0 of each epi warp, both CTAs. Fewer arrivals on the
      // window-release path = less mbarrier traffic on the MMA's
      // critical wake.
      for (int i = 0; i < 2; ++i)
        mbarrier_init(smem_ptr_u32(&ovl_free[i]), 8);
    }
    if constexpr (NS_B > 0) {
      for (int i = 0; i < NS_B; ++i) {
        mbarrier_init(smem_ptr_u32(&full_b_bar[i]),  1);  // expect_tx
        mbarrier_init(smem_ptr_u32(&empty_b_bar[i]), 1);  // tcgen05.commit
      }
      for (int i = 0; i < 8; ++i)
        mbarrier_init(smem_ptr_u32(&run_free_bar[i]), 1);  // tile-end commit
      if constexpr (CGA_PAIRS == 2) {
        for (int i = 0; i < NS_B; ++i)
          mbarrier_init(smem_ptr_u32(&b_gate_bar[i]), 2);  // both pairs' MMAs
      }
    }
  }
  pipeline_init_blackwell<NUM_STAGES, 2>(pipe_bars);

  // SMEM matrix descriptors: B128 swizzle, K-major, row = K_TILE/2 = 128 B.
  const uint64_t desc_a0 = build_smem_desc_blackwell(
      smem_ptr_u32(smem_a), K1_A_SBO, K1_A_LBO, SmemSwizzleBlackwell::B128);
  const uint64_t desc_b0 = build_smem_desc_blackwell(
      smem_ptr_u32(smem_b), K1_B_SBO, K1_B_LBO, SmemSwizzleBlackwell::B128);
  // Scale SMEM descriptors for tcgen05.cp.32x128b: an unswizzled 32x16B
  // core matrix, rows contiguous -> SBO = 8 rows x 16 B = 128, LBO = 16.
  // (Matches CUTLASS make_umma_desc<Major::K>.)
  const uint64_t desc_sfa0 = build_smem_desc_blackwell(
      smem_ptr_u32(smem_sfa), /*SBO=*/128, /*LBO=*/16, SmemSwizzleBlackwell::None);
  const uint64_t desc_sfb0 = build_smem_desc_blackwell(
      smem_ptr_u32(smem_sfb), /*SBO=*/128, /*LBO=*/16, SmemSwizzleBlackwell::None);
  constexpr uint64_t A_STAGE_DELTA = A_TILE_BYTES / 16;
  constexpr uint64_t B_STAGE_DELTA = B_TILE_BYTES / 16;

  if constexpr (!STATIC_SCHED) {
    if (warp == 0) {
      mma_warp_k1_ntiles_2sm_nvfp4<
          NUM_STAGES, M_TILE_CLUSTER, N_TILE_CLUSTER, K_TILE,
          A_STAGE_DELTA, B_STAGE_DELTA, SF_ATOMS_PER_CELL,
          CLC_CSM, CLC_CSN, ORDER, /*RUN_CLC=*/M_RUN>(
          wpc, desc_a0, desc_b0, desc_sfa0, desc_sfb0,
          full_bar, empty_bar, acc_bars,
          clc_full_bar, clc_empty_bar, clc_response,
          tmem_dealloc_bar, K, peer, lane, tmem_slot, run_len,
          AccPipeline2BankBars{ovl_free, ovl_free});
    } else if (warp == 1) {
      sched_warp_clc_blackwell_ntiles_2sm_bf16<
          true, CLC_CSM, CLC_CSN, ORDER>(wpc,
          clc_full_bar, clc_empty_bar, clc_response,
          throttle_full, throttle_empty, peer, lane);
    } else if (warp == 2) {
      load_warp_k1_ntiles_2sm_fp4<
          NUM_STAGES, M_TILE_PER_CTA, N_TILE_PER_CTA, K_TILE,
          SF_ATOMS_PER_CELL,
          M_TILE_CLUSTER, N_TILE_CLUSTER,
          CLC_CSM, CLC_CSN, ORDER,
          /*RUN_CLC=*/M_RUN, K_BLOCKS_T, NS>(wpc,
          &tmap_a, &tmap_b, &tmap_sfa, &tmap_sfb,
          smem_a, smem_b, smem_sfa, smem_sfb,
          full_bar, empty_bar,
          clc_full_bar, clc_empty_bar, clc_response,
          throttle_full, throttle_empty,
          K, peer, run_len);
    } else if (warp == 3) {
      idle_warp_blackwell_ntiles_2sm_bf16<24, CLC_CSM, CLC_CSN, ORDER>(
          wpc, clc_full_bar, clc_empty_bar, clc_response);
    } else {
      if constexpr (N_TILE_CLUSTER == 256 && !M_RUN) {
        k1_epi_ntiles_clc256<
            M_TILE_PER_CTA, N_TILE_CLUSTER, EPI_SUB_COLS, EPI_NUM_BUFS,
            CLC_CSM, CLC_CSN, ORDER>(wpc,
            &tmap_d, smem_d, acc_bars,
            AccPipeline2BankBars{ovl_free, ovl_free},
            clc_full_bar, clc_empty_bar, clc_response,
            peer, warp, lane, tmem_slot);
      } else if constexpr (N_TILE_CLUSTER == 256) {
        // NTC=256 CLC-RUN (M_RUN) not implemented.
        __trap();
      } else if constexpr (M_RUN) {
        k1_epi_ntiles_runclc<
            M_TILE_PER_CTA, N_TILE_CLUSTER, EPI_SUB_COLS, EPI_NUM_BUFS,
            CLC_CSM, CLC_CSN, ORDER>(wpc,
            &tmap_d, smem_d, acc_bars,
            clc_full_bar, clc_empty_bar, clc_response,
            peer, warp, lane, tmem_slot, run_len);
      } else {
        // The shared-block epi stores 128-high from warp 4 only; tmap_d
        // is a 32-high per-warp box, so route plain CLC
        // through the runclc wrapper with run_len=1 (identical consume
        // mechanics, per-warp earlyrelease drain).
        k1_epi_ntiles_runclc<
            M_TILE_PER_CTA, N_TILE_CLUSTER, EPI_SUB_COLS, EPI_NUM_BUFS,
            CLC_CSM, CLC_CSN, ORDER>(wpc,
            &tmap_d, smem_d, acc_bars,
            clc_full_bar, clc_empty_bar, clc_response,
            peer, warp, lane, tmem_slot, /*run_len=*/1);
      }
    }
  } else {  // STATIC_SCHED
    const bool warp3_works =
        (NS_B > 0) && (CGA_PAIRS == 1 || (blockIdx.x & 3) == 0);
    if (warp == 1 || (warp == 3 && !warp3_works)) {
      // No sched / idle role here. Return before griddepcontrol -- see the
      // gotcha in this kernel's banner.
      wp_flush(wpc);
      return;
    }

    if (warp == 0) {
      AccPipeline2BankState prod_state = acc_pipeline_2bank_state_init();
      PhaseTracker<NUM_STAGES> full_ph;

      wp_begin(wpc, WP_MMA_TMEM_2CTA_ALLOC);
      tcgen05_alloc<2>(smem_ptr_u32(tmem_slot), K1_TMEM_NCOLS);
      asm volatile("bar.arrive 6, 160;\n" ::: "memory");
      uint32_t tmem_base = *tmem_slot;
      wp_end(wpc, WP_MMA_TMEM_2CTA_ALLOC);

#if defined(K1_DISABLE_SF_TMA_CP) || defined(K1_DISABLE_SF_CP)
  // No scale delivery: fill the whole scale ring with UE8M0 0x7F (= 1.0)
  // once, so the atoms read a defined value and verify expects unscaled
  // results. 32x32b writes this warp's 32 TMEM rows, which is all the MMA
  // scale read uses (the constant-scale kernel verified with exactly this).
  using SFI = K1SfLayout<SF_ATOMS_PER_CELL, N_TILE_CLUSTER>;
  constexpr uint32_t SFI_RING_COLS =
      (SFI::TMEM_RING_DEPTH ? (uint32_t)SFI::TMEM_RING_DEPTH
                            : (uint32_t)NUM_STAGES) * SFI::STAGE_COLS;
  for (uint32_t col = SFI::RING_OFFSET;
       col < SFI::RING_OFFSET + SFI_RING_COLS;
       ++col) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};\n"
                 :: "r"(tmem_base + col), "r"(0x7F7F7F7Fu));
  }
  tcgen05_fence_before_thread_sync();
#endif

      const bool rstride = M_RUN && (run_len > 0);
      const int runs_total =
          rstride ? m_clusters_dim * (n_clusters_dim / run_len) : 0;
      const int my_tiles = rstride
          ? k1_rstride_my_tiles(cluster_id, num_clusters, runs_total, run_len)
          : ((total_tiles > cluster_id)
                 ? (total_tiles - 1 - cluster_id) / num_clusters + 1 : 0);
      [[maybe_unused]] PhaseTracker<(NS_B > 0) ? NS_B : 1> full_ph_b;
      int prev_m_tile_mma = -1;
      for (int t = 0; t < my_tiles; ++t) {
        // PDL gate point (host-chosen). gate_early: open at
        // the FIRST tile -- freed SM pairs pick up next-launch CTAs
        // continuously, amortizing wave quantization (K=2048 rows
        // +2-7%). Otherwise open at the LAST tile: big blocked rows
        // lose 1-31pp under early gating (two launches' block regions
        // co-resident thrash L2; 30720^3 collapsed to 57%). Consumers
        // using griddepcontrol.wait are unaffected either way.
        if (t == (gate_early ? 0 : my_tiles - 1) && elect_one_sync())
          griddepcontrol_launch_dependents();
        if constexpr (NS_B > 0) {
          int m_tile, n_tile;
          if (rstride) {
            k1_rstride_decode(cluster_id, num_clusters, t, run_len,
                              m_clusters_dim, m_tile, n_tile);
          } else {
            const int flat = cluster_id + t * num_clusters;
            const int g = M_RUN ? k1_mrun_remap(flat, num_clusters, total_tiles)
                                : flat;
            k1_decode_tile<ORDER, BLOCK_M, BLOCK_N, M_RUN>(
                g, m_clusters_dim, n_clusters_dim, m_tile, n_tile);
          }
          const bool skip_a = M_RUN && (K_BLOCKS_T == NS)
                              && (m_tile == prev_m_tile_mma);
          prev_m_tile_mma = m_tile;
          if (peer == 0) {
            mma_warp_k1_1tile_2sm_nvfp4_splitb<
                NUM_STAGES, NS_B, M_TILE_CLUSTER, N_TILE_CLUSTER, K_TILE,
                A_STAGE_DELTA, B_STAGE_DELTA, SF_ATOMS_PER_CELL, CGA_PAIRS,
                /*ALIGNED=*/(K_BLOCKS_T % NS == 0)>(
                wpc, desc_a0, desc_b0, desc_sfa0, desc_sfb0,
                full_bar, empty_bar, full_b_bar, empty_b_bar,
                acc_bars, prod_state, tmem_base, K, full_ph, full_ph_b,
                skip_a, run_free_bar, t,
                AccPipeline2BankBars{ovl_free, ovl_free},
                b_gate_bar, cga_pair);
          }
        } else if (peer == 0) {
          mma_warp_k1_1tile_2sm_nvfp4<
              NUM_STAGES, M_TILE_CLUSTER, N_TILE_CLUSTER, K_TILE,
              A_STAGE_DELTA, B_STAGE_DELTA, SF_ATOMS_PER_CELL>(
              wpc, desc_a0, desc_b0, desc_sfa0, desc_sfb0,
              full_bar, empty_bar, acc_bars, prod_state,
              tmem_base, K, full_ph,
              AccPipeline2BankBars{ovl_free, ovl_free});
        }
        acc_pipeline_2bank_state_advance(prod_state);
      }

      // (The PDL gate was signaled at the top of the last tile above;
      // repeated launch_dependents are no-ops, and the end-of-kernel
      // signal covers zero-tile CTAs and the CLC mode.)
      wp_begin(wpc, WP_MMA_WAIT_ACC);
      if (peer == 0) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
          acc_pipeline_2bank_producer_acquire(acc_bars, prod_state);
          acc_pipeline_2bank_state_advance(prod_state);
        }
      }
      wp_end(wpc, WP_MMA_WAIT_ACC);

      const uint32_t bar_local = smem_ptr_u32(tmem_dealloc_bar);
      mbarrier_arrive_cluster_default(bar_local ^ K1_SM100_PEER_BIT);
      wp_begin(wpc, WP_MMA_TMEM_2CTA_FREE);
      mbarrier_wait_parity(bar_local, 0);
      wp_end(wpc, WP_MMA_TMEM_2CTA_FREE);
      tcgen05_relinquish_alloc_permit<2>();
      tcgen05_dealloc<2>(tmem_base, K1_TMEM_NCOLS);

    } else if (warp == 2) {
      setmaxnreg_dec<40>();
      EmptyPhaseTracker<NUM_STAGES> empty_ph;

      int prev_m_tile = -1;
      bool prev_pp_backward = false;
      const bool rstride = M_RUN && (run_len > 0);
      const int runs_total =
          rstride ? m_clusters_dim * (n_clusters_dim / run_len) : 0;
      const int my_tiles = rstride
          ? k1_rstride_my_tiles(cluster_id, num_clusters, runs_total, run_len)
          : ((total_tiles > cluster_id)
                 ? (total_tiles - 1 - cluster_id) / num_clusters + 1 : 0);
      for (int t = 0; t < my_tiles; ++t) {
        int m_tile, n_tile;
        if (rstride) {
          k1_rstride_decode(cluster_id, num_clusters, t, run_len,
                            m_clusters_dim, m_tile, n_tile);
        } else {
          const int flat = cluster_id + t * num_clusters;
          const int g = M_RUN ? k1_mrun_remap(flat, num_clusters, total_tiles)
                              : flat;
          k1_decode_tile<ORDER, BLOCK_M, BLOCK_N, M_RUN>(
              g, m_clusters_dim, n_clusters_dim, m_tile, n_tile);
        }
        const int m_off = (m_tile * CGA_PAIRS + cga_pair) * M_TILE_CLUSTER
                          + peer * M_TILE_PER_CTA;
        const int n_off = n_tile * N_TILE_CLUSTER + peer * N_TILE_PER_CTA;
        // Same m as the previous tile and ring-aligned K sweep: the A ring
        // already holds byte-identical A (and SFA), skip those loads.
        // Ping-pong K reuse: legal for any K_BLOCKS % NS == 0 (round-0
        // A/SFA are resident from the previous same-m tile; the full
        // KB == NS skip is the degenerate case).
        const bool pp_reuse = M_RUN && (K_BLOCKS_T % NS == 0)
                              && (m_tile == prev_m_tile);
        const bool pp_backward = pp_reuse && !prev_pp_backward;
        prev_pp_backward = pp_backward;
        // Split-ring A skip: whole-tile A/SFA reuse when the ring holds
        // the full K sweep (KB == NS_A) and the previous tile shares m.
        // Must mirror the MMA warp's skip_a exactly (same decode).
        const bool skip_a = M_RUN && (K_BLOCKS_T == NS)
                            && (m_tile == prev_m_tile) && (NS_B > 0);
        prev_m_tile = m_tile;

        if constexpr (NS_B > 0) {
          (void)n_off;
          // run_free[8]: the MMA commits slot t&7 at the end of tile t.
          // Waiting slot t&7 for tile t-8 every tile bounds this warp's
          // lookahead to 8 tiles, which makes the refill wait on tile
          // t-1's commit alias-free (mod-2 parity is safe when the lag
          // is under 2 ring depths). Without the bound, the A warp
          // sprints across skip tiles and the parity aliases -- SFA/A
          // get refilled under a running MMA (clean 2x outputs).
          // run_free protects the A-ring REFILL against still-reading
          // skip tiles -- only meaningful when skip can occur at all.
          // Without M-RUN reuse the per-stage empty_a handshake fully
          // orders reloads, and these tile-granular waits would drain
          // the pipe at EVERY boundary (~1 us/tile).
          const bool any_skip = M_RUN && (K_BLOCKS_T % NS == 0);
          if (any_skip) {
            // Suspend form: this warp is lag-locked to the MMA across
            // long skip runs; a hot spin here steals issue slots from
            // the MMA warp on the same SM.
            wp_begin(wpc, WP_LOAD_WAIT);
            mbarrier_wait_parity_suspend(smem_ptr_u32(&run_free_bar[t & 7]),
                                         (uint32_t)(((t >> 3) + 1) & 1));
            if (!skip_a && t > 0) {
              // Needs commit #(tp>>3)+1 on bar tp&7 (tile tp itself), so
              // parity = (c_needed - 1) & 1 = (tp >> 3) & 1.
              const int tp = t - 1;
              mbarrier_wait_parity_suspend(smem_ptr_u32(&run_free_bar[tp & 7]),
                                           (uint32_t)((tp >> 3) & 1));
            }
            wp_end(wpc, WP_LOAD_WAIT);
          }
          if (!skip_a) {
            load_warp_k1_1tile_a_side<
                NUM_STAGES, M_TILE_PER_CTA, N_TILE_PER_CTA, K_TILE,
                SF_ATOMS_PER_CELL>(
                wpc, &tmap_a, &tmap_sfa, smem_a, smem_sfa,
                full_bar, empty_bar,
                K, m_off, peer, empty_ph);
          }
        } else {
          load_warp_k1_1tile_2sm_fp4<
              NUM_STAGES, M_TILE_PER_CTA, N_TILE_PER_CTA, K_TILE,
              SF_ATOMS_PER_CELL>(
              wpc, &tmap_a, &tmap_b, &tmap_sfa, &tmap_sfb,
              smem_a, smem_b, smem_sfa, smem_sfb,
              full_bar, empty_bar, K, m_off, n_off, peer, empty_ph, skip_a,
              pp_reuse, pp_backward);
        }
      }

      wp_begin(wpc, WP_LOAD_WAIT);
      #pragma unroll
      for (int i = 0; i < NUM_STAGES; ++i) {
        mbarrier_wait_parity(smem_ptr_u32(&empty_bar[empty_ph.get_stage()]),
                             empty_ph.get_phase());
        empty_ph.advance();
      }
      wp_end(wpc, WP_LOAD_WAIT);

    } else if (warp == 3) {
      // NS_B > 0 only (warp 3 retired above otherwise): B-ring loader.
      setmaxnreg_dec<40>();
      [[maybe_unused]] EmptyPhaseTracker<(NS_B > 0) ? NS_B : 1> empty_ph_b;
      if constexpr (NS_B > 0) {
        const bool rstride = M_RUN && (run_len > 0);
        const int runs_total =
            rstride ? m_clusters_dim * (n_clusters_dim / run_len) : 0;
        const int my_tiles = rstride
            ? k1_rstride_my_tiles(cluster_id, num_clusters, runs_total, run_len)
            : ((total_tiles > cluster_id)
                   ? (total_tiles - 1 - cluster_id) / num_clusters + 1 : 0);
        for (int t = 0; t < my_tiles; ++t) {
          int m_tile, n_tile;
          if (rstride) {
            k1_rstride_decode(cluster_id, num_clusters, t, run_len,
                              m_clusters_dim, m_tile, n_tile);
          } else {
            const int flat = cluster_id + t * num_clusters;
            const int g = M_RUN ? k1_mrun_remap(flat, num_clusters, total_tiles)
                                : flat;
            k1_decode_tile<ORDER, BLOCK_M, BLOCK_N, M_RUN>(
                g, m_clusters_dim, n_clusters_dim, m_tile, n_tile);
          }
          if constexpr (CGA_PAIRS == 2) {
            load_warp_k1_1tile_b_side_cga<
                NS_B, M_TILE_PER_CTA, N_TILE_PER_CTA, K_TILE,
                SF_ATOMS_PER_CELL>(
                wpc, &tmap_b, &tmap_sfb, smem_b, smem_sfb,
                full_b_bar, b_gate_bar, K,
                n_tile * N_TILE_CLUSTER, empty_ph_b);
          } else {
            const int n_off = n_tile * N_TILE_CLUSTER + peer * N_TILE_PER_CTA;
            load_warp_k1_1tile_b_side<
                NS_B, M_TILE_PER_CTA, N_TILE_PER_CTA, K_TILE,
                SF_ATOMS_PER_CELL>(
                wpc, &tmap_b, &tmap_sfb, smem_b, smem_sfb,
                full_b_bar, empty_b_bar, K, n_off, peer, empty_ph_b,
                cga_pair);
          }
        }
        wp_begin(wpc, WP_LOAD_WAIT);
        for (int i = 0; i < NS_B; ++i) {
          uint64_t* drain = (CGA_PAIRS == 2) ? b_gate_bar : empty_b_bar;
          mbarrier_wait_parity(smem_ptr_u32(&drain[empty_ph_b.get_stage()]),
                               empty_ph_b.get_phase());
          empty_ph_b.advance();
        }
        wp_end(wpc, WP_LOAD_WAIT);
      }

    } else {
      AccPipeline2BankState cons_state = acc_pipeline_2bank_state_init();

      wp_begin(wpc, WP_EPI_WAIT_TMEM);
      asm volatile("bar.sync 6, 160;\n" ::: "memory");
      const uint32_t tmem_base = *tmem_slot;
      wp_end(wpc, WP_EPI_WAIT_TMEM);

      const bool rstride = M_RUN && (run_len > 0);
      const int runs_total =
          rstride ? m_clusters_dim * (n_clusters_dim / run_len) : 0;
      const int my_tiles = rstride
          ? k1_rstride_my_tiles(cluster_id, num_clusters, runs_total, run_len)
          : ((total_tiles > cluster_id)
                 ? (total_tiles - 1 - cluster_id) / num_clusters + 1 : 0);
      for (int t = 0; t < my_tiles; ++t) {
        int m_tile, n_tile;
        if (rstride) {
          k1_rstride_decode(cluster_id, num_clusters, t, run_len,
                            m_clusters_dim, m_tile, n_tile);
        } else {
          const int flat = cluster_id + t * num_clusters;
          const int g = M_RUN ? k1_mrun_remap(flat, num_clusters, total_tiles)
                              : flat;
          k1_decode_tile<ORDER, BLOCK_M, BLOCK_N, M_RUN>(
              g, m_clusters_dim, n_clusters_dim, m_tile, n_tile);
        }
        const int m_off = (m_tile * CGA_PAIRS + cga_pair) * M_TILE_CLUSTER
                          + peer * M_TILE_PER_CTA;
        const int n_off = n_tile * N_TILE_CLUSTER;

        if constexpr (N_TILE_CLUSTER == 256) {
          k1_epi_1tile_ntc256<M_TILE_PER_CTA, EPI_SUB_COLS, EPI_NUM_BUFS>(
              wpc, &tmap_d, smem_d, acc_bars,
              AccPipeline2BankBars{ovl_free, ovl_free}, cons_state,
              tmem_base, peer, warp, lane, m_off, n_off);
        } else {
          k1_epi_1tile_earlyrelease<
              M_TILE_PER_CTA, N_TILE_CLUSTER, EPI_SUB_COLS, EPI_NUM_BUFS,
              /*DRAIN_PER_TILE=*/false>(
              wpc, &tmap_d, smem_d, acc_bars, cons_state,
              tmem_base, peer, warp, lane, m_off, n_off);
        }
      }

      wp_begin(wpc, WP_EPI_WAIT_STORE);
      if (lane == 0) cp_async_bulk_wait_group<0>();
      asm volatile("bar.sync 1, 128;\n" ::: "memory");
      wp_end(wpc, WP_EPI_WAIT_STORE);
    }
  }

  wp_flush(wpc);
  if (lane == 0) griddepcontrol_launch_dependents();
}

// ============================================================================
// Host: reference + run / bench
// ============================================================================

// E2M1 decode: 4-bit signed float (1 sign, 2 exp, 1 mantissa).
// Bias = 1. Values: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives).
// FILL modes (env FILL, default 1), the fp4 adaptation of the FMHA set:
//   1 = constant 4.0 data + patterned separable scales; closed-form exact
//       full-matrix verify (the fast everyday gate)
//   2 = hash-random over all 16 E2M1 codes + generic random scales
//       (exponent in [-2,2] per row x K-block); sampled-reference verify
//   3 = constant 1.0 data + patterned scales; closed-form verify
//   4 = hash-random integers {0,+-1,+-2,+-3} + generic random scales;
//       sampled-reference verify
static int k1_fill_mode() {
  const char* e = getenv("FILL");
  const int m = e ? atoi(e) : 1;
  return (m >= 1 && m <= 4) ? m : 1;
}
static uint32_t k1_hash(uint32_t x) {
  x *= 2654435761u; x ^= x >> 16; x *= 2246822519u; x ^= x >> 13;
  return x;
}
constexpr uint32_t K1_FILL_SEED_A = 0x9E3779B9u;
constexpr uint32_t K1_FILL_SEED_B = 0x7F4A7C15u;
// E2M1 nibble for global element index i (row * K + k, truncated to u32 --
// collisions are fine, fill and reference just must agree).
static uint8_t k1_e2m1_nibble(int mode, uint32_t i, uint32_t seed) {
  const uint32_t x = k1_hash(i * 2u + seed);
  if (mode == 2) return (uint8_t)(x & 15u);
  static const uint8_t integer_codes[4] = {0, 2, 4, 5};  // 0, 1, 2, 3
  return (uint8_t)(integer_codes[x & 3u] | (((x >> 5) & 1u) << 3));
}
// Generic random scale exponent in [-2, 2] per (row, K-block).
// which: 0 = SFA, 1 = SFB. Shared by the fill and the reference.
static int k1_generic_sf_exp(int r, int j, uint32_t which) {
  return (int)(k1_hash((uint32_t)r * 131071u + (uint32_t)j * 8191u
                       + which * 977u) % 5u) - 2;
}

static float e2m1_decode(uint8_t nibble) {
  nibble &= 0xF;
  if (nibble == 0 || nibble == 8) return 0.0f;
  const int sign = (nibble >> 3) & 1;
  const int exp  = (nibble >> 1) & 0x3;
  const int mant = nibble & 1;
  // Normals are (1 + mant/2) * 2^(exp-1) -- the single mantissa bit is worth
  // 0.5, not 1. Writing it as (1 + mant) would give 2/4/8 for codes 3/5/7
  // instead of 1.5/3/6. Full table: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
  const float val = (exp == 0)
      ? (mant ? 0.5f : 0.0f)
      : (1.0f + 0.5f * (float)mant) * (float)(1 << (exp - 1));
  return sign ? -val : val;
}

// UE8M0 decode: unsigned 8-bit exponent only. bias = 127.
static float ue8m0_decode(uint8_t s) {
  return (s == 0) ? 0.0f : powf(2.0f, (float)s - 127.0f);
}

// CPU reference for block32 NVFP4 GEMM (A: MxK, B: NxK transposed).
// A and B are packed E2M1 (2 per byte). Scales are UE8M0 per (16-row, 32-K) block.
// Unused: the runners verify in closed form (FILL 1/3) or by sampled reference (FILL 2/4).
static void ref_nvfp4_gemm(
    const uint8_t* A, const uint8_t* B,
    const uint8_t* sca, const uint8_t* scb,
    float* D,
    int M, int N, int K) {
  // sca shape: (M/16, K/32), scb shape: (N/16, K/32)
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        // Unpack A[m][k] from packed byte A[m*K/2 + k/2]
        const uint8_t ab = A[(size_t)m * (K/2) + k/2];
        float a_val = e2m1_decode((k & 1) ? (ab >> 4) : (ab & 0xF));
        // Unpack B[n][k]
        const uint8_t bb = B[(size_t)n * (K/2) + k/2];
        float b_val = e2m1_decode((k & 1) ? (bb >> 4) : (bb & 0xF));
        // Scale factors for block32
        float sa = ue8m0_decode(sca[(m/16) * (K/32) + k/32]);
        float sb = ue8m0_decode(scb[(n/16) * (K/32) + k/32]);
        acc += a_val * sa * b_val * sb;
      }
      D[(size_t)m * N + n] = acc;
    }
  }
}

// GB200 has 152 SMs, so 76 2SM clusters. Static caps its 1D grid there and
// strides; CLC launches one cluster per tile and lets the hardware persist.
constexpr int K1_MAX_CLUSTERS = 76;

// Host controls only; defaults preserve the standalone sweep and probe callers.
struct K1RunOptions {
  bool benchmark = true;
  bool standard_output = false;
  int fill = 0;  // Zero retains the legacy FILL environment variable.
  const char* dump_output = nullptr;
};

template <int K_BLOCKS_T, int NTC, int NS = K1_NUM_STAGES,
          ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          bool STATIC_SCHED = false,
          int BLOCK_M = 0, int BLOCK_N = 0,
          int SF_ATOMS_PER_CELL = 2,
          bool M_RUN = false,
          int NS_B = 0,
          int CGA_PAIRS = 1>
static double run_one_k1(int M, int N, bool verify = false, int run_len = 0,
                         const K1RunOptions& options = {}) {
  constexpr int K            = K_BLOCKS_T * K1_K_TILE;
  constexpr int N_TILE_PER_CTA = NTC / 2;
  constexpr int A_TILE_BYTES = K1_M_TILE_PER_CTA * K1_K_TILE / 2;
  constexpr int B_TILE_BYTES = N_TILE_PER_CTA * K1_K_TILE / 2;
  constexpr int NUM_STAGES   = NS;
  // K-blocks packed per 4-B cell: 4 shared-cell, 2 per-atom-cell
  // (bytes [2,3] of each cell then stay zero -- padding).
  using SF = K1SfLayout<SF_ATOMS_PER_CELL, NTC>;
  constexpr int SF_BLOCKS_PER_CELL = 2 * SF_ATOMS_PER_CELL;
  constexpr size_t kSfaStageBytes = SF::SFA_STAGE_BYTES;
  constexpr size_t kSfbStageBytes = SF::SFB_STAGE_BYTES;

  if (M % (K1_M_TILE_CLUSTER * CGA_PAIRS) || N % NTC) {
    printf("  shape (%d, %d, %d): NOT MULTIPLE OF TILE -- skipping\n", M, N, K);
    if (options.standard_output) std::exit(EXIT_FAILURE);
    return 0.0;
  }

  // Packed FP4: 2 elements per byte.
  uint8_t *dA = nullptr, *dB = nullptr, *dSFA = nullptr, *dSFB = nullptr;
  __nv_bfloat16 *dD = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K / 2));
  CUDA_CHECK(cudaMalloc(&dB, (size_t)N * K / 2));
  CUDA_CHECK(cudaMalloc(&dD, (size_t)M * N * 2));
  auto sf_total_bytes = [&](int rows) {
    return (size_t)(rows / 128) * ((size_t)(K / 32 / SF_BLOCKS_PER_CELL) * 512);
  };
  CUDA_CHECK(cudaMalloc(&dSFA, sf_total_bytes(M)));
  CUDA_CHECK(cudaMalloc(&dSFB, sf_total_bytes(N)));

  const int fill_mode = options.fill ? options.fill : k1_fill_mode();
  if (fill_mode == 1 || fill_mode == 3) {
    // E2M1 4.0 packed = 0x6 (mode 1) or 1.0 packed = 0x2 (mode 3).
    const int byte = (fill_mode == 1) ? 0x66 : 0x22;
    CUDA_CHECK(cudaMemset(dA, byte, (size_t)M * K / 2));
    CUDA_CHECK(cudaMemset(dB, byte, (size_t)N * K / 2));
  } else {
    auto fill_fp4 = [&](uint8_t* dev, int rows, uint32_t seed) {
      std::vector<uint8_t> h((size_t)rows * K / 2);
      for (size_t bi = 0; bi < h.size(); ++bi) {
        const uint8_t lo = k1_e2m1_nibble(fill_mode, (uint32_t)(bi * 2), seed);
        const uint8_t hi = k1_e2m1_nibble(fill_mode, (uint32_t)(bi * 2 + 1), seed);
        h[bi] = (uint8_t)(lo | (hi << 4));
      }
      CUDA_CHECK(cudaMemcpy(dev, h.data(), h.size(), cudaMemcpyHostToDevice));
    };
    fill_fp4(dA, M, K1_FILL_SEED_A);
    fill_fp4(dB, N, K1_FILL_SEED_B);
  }
  CUDA_CHECK(cudaMemset(dD, 0, (size_t)M * N * 2));

  // Real scales, non-uniform so a scrambled row/col mapping FAILS verify:
  //   sfa(m) = 2^((m%5)-2)  in {1/4..4},  sfb(n) = 2^((n%3)-1)  in {1/2..2}
  // (separable, j-independent -> the analytic check stays O(1) per element).
  //
  // The buffer is laid out in the exact order the cps consume (TMA and cp
  // never reorder bytes). One cp's 512-B source packs SF_BLOCKS_PER_CELL
  // consecutive K-blocks per row; with r = row, j = K-block index:
  //   byte(r, j) = (j/4)*512 + (r%32)*16 + ((r/32)%4)*4 + (j%4)   [shared]
  // i.e. cp source -> lane (r%32, 16 B) -> cell (r/32, 4 B) -> K-block byte.
  auto sf_pack = [&](uint8_t* h, int rows, int Kdim, auto exp_of) {
    const int kb = Kdim / 32;             // scale bytes per row (payload)
    const size_t band_bytes = (size_t)(kb / SF_BLOCKS_PER_CELL) * 512;
    for (int band = 0; band < rows / 128; ++band)
      for (int r = 0; r < 128; ++r) {
        uint8_t* base = h + (size_t)band * band_bytes;
        for (int j = 0; j < kb; ++j) {
          const uint8_t e = (uint8_t)(127 + exp_of(band * 128 + r, j));
          base[(j / SF_BLOCKS_PER_CELL) * 512 + (r % 32) * 16 + (r / 32) * 4
               + (j % SF_BLOCKS_PER_CELL)] = e;
        }
      }
  };
  auto sa_exp = [](int m) { return (m % 5) - 2; };
  auto sb_exp = [](int n) { return (n % 3) - 1; };
  const bool generic_sf = (fill_mode == 2 || fill_mode == 4);
  {
    std::vector<uint8_t> h(sf_total_bytes(M), 0);
    if (generic_sf)
      sf_pack(h.data(), M, K, [](int r, int j) { return k1_generic_sf_exp(r, j, 0); });
    else
      sf_pack(h.data(), M, K, [&](int r, int j) { (void)j; return sa_exp(r); });
    CUDA_CHECK(cudaMemcpy(dSFA, h.data(), h.size(), cudaMemcpyHostToDevice));
    h.assign(sf_total_bytes(N), 0);
    if (generic_sf)
      sf_pack(h.data(), N, K, [](int r, int j) { return k1_generic_sf_exp(r, j, 1); });
    else
      sf_pack(h.data(), N, K, [&](int r, int j) { (void)j; return sb_exp(r); });
    CUDA_CHECK(cudaMemcpy(dSFB, h.data(), h.size(), cudaMemcpyHostToDevice));
  }

  CUtensorMap tmap_a{}, tmap_b{}, tmap_d{}, tmap_sfa{}, tmap_sfb{};
  // Scales as uint64 tensors: one row per 128-row band, band_bytes/8 uint64s
  // per row. Per-stage box = kSfaStageBytes/8 (128 shared, 256 per-atom).
  const int sf_row_u64s = (K / 32 / SF_BLOCKS_PER_CELL) * (512 / 8);
  CUDA_CHECK(make_tma_2d_tiled(&tmap_sfa, dSFA, M / 128, sf_row_u64s,
                               /*box_rows=*/1, /*box_cols=*/(int)(kSfaStageBytes / 8),
                               sizeof(uint64_t),
                               CU_TENSOR_MAP_DATA_TYPE_UINT64,
                               CU_TENSOR_MAP_SWIZZLE_NONE));
  CUDA_CHECK(make_tma_2d_tiled(&tmap_sfb, dSFB, N / 128, sf_row_u64s,
                               /*box_rows=*/NTC / 128,
                               /*box_cols=*/(int)(kSfaStageBytes / 8),
                               sizeof(uint64_t),
                               CU_TENSOR_MAP_DATA_TYPE_UINT64,
                               CU_TENSOR_MAP_SWIZZLE_NONE));
  // A: (M, K/2) UINT8, tile=(M_PER_CTA, K_TILE/2), B128 swizzle.
  CUDA_CHECK(make_tma_2d_tiled(&tmap_a, dA, M, K / 2,
                               K1_M_TILE_PER_CTA, K1_K_TILE / 2,
                               sizeof(uint8_t),
                               CU_TENSOR_MAP_DATA_TYPE_UINT8,
                               CU_TENSOR_MAP_SWIZZLE_128B));
  CUDA_CHECK(make_tma_2d_tiled(&tmap_b, dB, N, K / 2,
                               N_TILE_PER_CTA, K1_K_TILE / 2,
                               sizeof(uint8_t),
                               CU_TENSOR_MAP_DATA_TYPE_UINT8,
                               CU_TENSOR_MAP_SWIZZLE_128B));
  // NTC=128: box height 32 -- each epi warp stores its own 32-row slice
  // with its own TMA chain (per-warp store pipeline).
  // NTC=256: the classic 128-high warp-4-issued store (per-warp stores
  // cost the deep-K rows ~0.5pp in extra TMA-store requests).
  CUDA_CHECK(make_tma_2d_tiled(&tmap_d, dD, M, N,
                               (NTC == 128) ? 32 : K1_M_TILE_PER_CTA,
                               K1_EPI_SUB_COLS,
                               sizeof(__nv_bfloat16),
                               CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                               CU_TENSOR_MAP_SWIZZLE_NONE));

  const int m_clusters  = M / (K1_M_TILE_CLUSTER * CGA_PAIRS);
  const int n_clusters  = N / NTC;
  const int total_tiles = m_clusters * n_clusters;
  // Static: one cluster per SM pair, each striding the tile list. CLC: one
  // cluster per tile, 2D so the hardware's grid.x-major sweep matches ORDER.
  const int max_units = K1_MAX_CLUSTERS / CGA_PAIRS;  // 76 pairs or 38 CGAs
  const int num_clusters =
      (total_tiles < max_units) ? total_tiles : max_units;
  dim3 grid = STATIC_SCHED
      ? dim3(2 * CGA_PAIRS * num_clusters, 1, 1)
      : ((ORDER == ClcRasterOrder::AlongN)
             ? dim3(2 * n_clusters, m_clusters, 1)
             : dim3(2 * m_clusters, n_clusters, 1));
  if (!STATIC_SCHED && M_RUN) {
    // CLC-RUN: one CLC work item = a run of run_len same-m tiles. Grid is
    // (2*m, n_windows) with AlongM decode so the x-major launch order hands
    // out column-major runs (cross-cluster lockstep in one B n-window).
    if (run_len <= 0 || (n_clusters % run_len) || ORDER != ClcRasterOrder::AlongM) {
      printf("  CLC-RUN: bad run_len %d for n_clusters %d -- skipping\n",
             run_len, n_clusters);
      cudaFree(dA); cudaFree(dB); cudaFree(dSFA); cudaFree(dSFB); cudaFree(dD);
      return 0.0;
    }
    grid = dim3(2 * m_clusters, n_clusters / run_len, 1);
  }
  dim3 block(256, 1, 1);

  constexpr int NSB_DATA = (NS_B > 0) ? NS_B : NUM_STAGES;
  constexpr size_t smem_bytes =
      1024
      + (size_t)NUM_STAGES * A_TILE_BYTES
      + (size_t)NSB_DATA * B_TILE_BYTES
#if !defined(K1_DISABLE_SF_TMA_CP)
      + (size_t)NUM_STAGES * kSfaStageBytes
      + (size_t)NSB_DATA * kSfbStageBytes
#endif
      + (size_t)k1_epi_num_bufs(NTC) * K1_EPI_BUF_BYTES + 256;

  if (smem_bytes > 232448) {
    printf("  (%d, %d, %d): smem %zu B over cap -- skipping config\n",
           M, N, K, smem_bytes);
    cudaFree(dA); cudaFree(dB); cudaFree(dSFA); cudaFree(dSFB); cudaFree(dD);
    if (options.standard_output) std::exit(EXIT_FAILURE);
    return 0.0;
  }
  auto* kfn = dense_gemm_nvfp4_k1_impl<K_BLOCKS_T, NTC, NS, ORDER,
                                       STATIC_SCHED, BLOCK_M, BLOCK_N,
                                       SF_ATOMS_PER_CELL, M_RUN, NS_B,
                                       CGA_PAIRS>;
  CUDA_CHECK(cudaFuncSetAttribute((const void*)kfn,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));
  CUDA_CHECK(cudaFuncSetAttribute((const void*)kfn,
      cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaLaunchConfig_t config{};
  config.gridDim  = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem_bytes;
  config.stream   = stream;
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim = {2 * CGA_PAIRS, 1, 1};
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  config.attrs    = attrs;
  config.numAttrs = 2;

  // Early PDL gate for short-K rows (measured): the win is
  // cross-launch wave-quantization amortization; the loss cases are big
  // blocked rows whose two co-resident block regions thrash L2.
  const int gate_early =
      (K <= 2048) || (K <= 4096 && N <= 8192);

  WpBuffer wpbuf = wp_alloc(grid);

  auto launch = [&](size_t) {
    return cudaLaunchKernelEx(&config, kfn, tmap_a, tmap_b, tmap_sfa, tmap_sfb, tmap_d,
                       M, N, total_tiles, num_clusters, run_len, gate_early);
  };
  double ms = options.benchmark ? dense_gemm_nvfp4_benchmark::measure(stream, launch) : 0.0;
  wp_reset(wpbuf);
  // Also supplies the single untimed launch used for an output dump.
  CUDA_CHECK(launch(0));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  cudaStreamDestroy(stream);

  double tflops = dense_gemm_nvfp4_benchmark::report(
      M, N, K, ms, STATIC_SCHED, BLOCK_M, BLOCK_N, num_clusters, total_tiles,
      options.benchmark, options.standard_output);
  {
    static const char* roles_clc[8] =
        {"mma","sched","load","idle","epi","epi","epi","epi"};
    static const char* roles_static[8] =
        {"mma","-","load","-","epi","epi","epi","epi"};
    const char** roles = STATIC_SCHED ? roles_static : roles_clc;
    wp_readback(wpbuf);
    wp_print_busy(wpbuf, roles, 8, 0);
    wp_free(wpbuf);
  }

  if (options.dump_output) {
    std::vector<__nv_bfloat16> output((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(output.data(), dD, output.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));
    FILE* file = fopen(options.dump_output, "wb");
    if (!file) {
      perror(options.dump_output);
      std::exit(EXIT_FAILURE);
    }
    const size_t written = fwrite(output.data(), sizeof(__nv_bfloat16), output.size(), file);
    const int close_status = fclose(file);
    if (written != output.size() || close_status != 0) {
      fprintf(stderr, "failed to write complete output: %s\n", options.dump_output);
      std::exit(EXIT_FAILURE);
    }
    printf("  wrote %zu BF16 values to %s\n", output.size(), options.dump_output);
  }

  if (verify && (fill_mode == 2 || fill_mode == 4)) {
    // Random-data modes: sampled reference. 2048 hashed (m, n) pairs,
    // fp32 accumulation then bf16 rounding, mirroring the kernel's store.
    // Tolerance: |got - ref| <= max(0.8% * |ref|, 16.0). The 0.8% is two
    // bf16 ulps (store rounding of got and ref can differ by one each);
    // the absolute floor covers fp32 accumulation-order noise when random
    // signs cancel the sum toward zero. Layout bugs move sampled points
    // by factors of 2+, far above both.
    std::vector<__nv_bfloat16> hD((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(hD.data(), dD, hD.size() * 2, cudaMemcpyDeviceToHost));
    float max_err = 0.0f, max_ref = 0.0f, max_got = 0.0f;
    int nan_count = 0, fail = 0, bad_m = -1, bad_n = -1;
    for (int smp = 0; smp < 2048; ++smp) {
      const int m = (int)(k1_hash((uint32_t)(2 * smp + 1)) % (uint32_t)M);
      const int nn = (int)(k1_hash((uint32_t)(2 * smp + 2)) % (uint32_t)N);
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        const uint32_t ia = (uint32_t)((size_t)m * K + k);
        const uint32_t ib = (uint32_t)((size_t)nn * K + k);
        const float a = e2m1_decode(k1_e2m1_nibble(fill_mode, ia, K1_FILL_SEED_A));
        const float b = e2m1_decode(k1_e2m1_nibble(fill_mode, ib, K1_FILL_SEED_B));
        if (a == 0.0f || b == 0.0f) continue;
        const float sa = exp2f((float)k1_generic_sf_exp(m, k / 32, 0));
        const float sb = exp2f((float)k1_generic_sf_exp(nn, k / 32, 1));
        acc += a * b * sa * sb;
      }
      const float ref = __bfloat162float(__float2bfloat16(acc));
      const float got = __bfloat162float(hD[(size_t)m * N + nn]);
      if (std::isnan(got) || std::isinf(got)) { ++nan_count; continue; }
      const float err = fabsf(got - ref);
      if (err > fmaxf(0.008f * fabsf(ref), 16.0f)) {
        ++fail;
        if (err > max_err) { max_err = err; bad_m = m; bad_n = nn;
                             max_ref = ref; max_got = got; }
      }
    }
    const bool ok = (nan_count == 0) && (fail == 0);
    printf("  verify M=%d,N=%d,K=%d (FILL=%d sampled ref x2048): fail=%d nan=%d %s",
           M, N, K, fill_mode, fail, nan_count, ok ? "OK" : "FAIL");
    if (!ok && bad_m >= 0)
      printf(" [worst at m=%d n=%d: got %.3f ref %.3f]",
             bad_m, bad_n, max_got, max_ref);
    printf("\n");
    if (!ok && options.standard_output) std::exit(EXIT_FAILURE);
  } else if (verify) {
    // Analytic check with REAL scales: A=B=all E2M1 4.0 (FILL=1; 16 per
    // product) or all 1.0 (FILL=3), so
    //   D[m][n] = K * 16 * sfa(m) * sfb(n),  sfa(m)=2^((m%5)-2), sfb(n)=2^((n%3)-1).
    // Every term is a power of two times 16, so both the fp32 accumulation
    // and the bf16 store are exact -- demand max relative error 0.
    // Non-uniform scales make this a real layout test: any row/col scramble
    // in the SF path (TMA packing, tcgen05.cp, TMEM addressing, SF_ID) shows
    // up as a wrong power of two somewhere in D.
    std::vector<__nv_bfloat16> hD((size_t)M * N);
    CUDA_CHECK(cudaMemcpy(hD.data(), dD, hD.size() * 2, cudaMemcpyDeviceToHost));
    float max_rel = 0.0f;
    int nan_count = 0, bad_m = -1, bad_n = -1;
    for (int m = 0; m < M; ++m) {
      const float sa = exp2f((float)sa_exp(m));
      for (int n = 0; n < N; ++n) {
        float v = __bfloat162float(hD[(size_t)m * N + n]);
        if (std::isnan(v) || std::isinf(v)) { ++nan_count; continue; }
#if defined(K1_DISABLE_SF_TMA_CP) || defined(K1_DISABLE_SF_CP)
        const float expected = 16.0f * K;  // ring inited to 1.0, sfa/sfb unused
#else
        const float per_product = (fill_mode == 3) ? 1.0f : 16.0f;
        const float expected = per_product * K * sa * exp2f((float)sb_exp(n));
#endif
        float rel = fabsf(v - expected) / expected;
        if (rel > max_rel) { max_rel = rel; bad_m = m; bad_n = n; }
      }
    }
    bool ok = (nan_count == 0) && (max_rel < 1e-3f);
    printf("  verify M=%d,N=%d,K=%d (real scales): max_rel=%.2e nan=%d %s",
           M, N, K, max_rel, nan_count, ok ? "OK" : "FAIL");
    if (!ok && bad_m >= 0) {
      const float exp_bad = 16.0f * K * exp2f((float)sa_exp(bad_m))
                                       * exp2f((float)sb_exp(bad_n));
      printf(" [worst at m=%d n=%d: got %.1f want %.1f]",
             bad_m, bad_n,
             __bfloat162float(hD[(size_t)bad_m * N + bad_n]), exp_bad);
    }
    printf("\n");
    if (!ok && options.standard_output) std::exit(EXIT_FAILURE);
  }

  cudaFree(dA); cudaFree(dB); cudaFree(dSFA); cudaFree(dSFB); cudaFree(dD);
  return tflops;
}

// K-dispatch. NS rises with K until the SMEM ring hits its budget: at NTC=128
// a stage costs A 16 KB + B 8 KB + SFA 1 KB + SFB 1 KB = 26 KB, so NS=7 is
// 182 KB, plus 32 KB of D staging and the bars -> ~216 KB against the 232 KB
// limit (NS=8 = 241 KB does not fit).
template <ClcRasterOrder ORDER, bool STATIC_SCHED = false,
          int BLOCK_M = 0, int BLOCK_N = 0,
          int SF_ATOMS_PER_CELL = 2, bool M_RUN = false>
static double run_k1_kdispatch_ntc128(int M, int N, int K, bool verify = false,
                                      int run_len = 0, const K1RunOptions& options = {}) {
  // Per-atom-cell stages are 2 KB fatter (28 vs 26 KB), so the deep-K NS
  // drops 7 -> 6 to stay under the 227-KB SMEM cap.
#if defined(K1_DISABLE_SF_TMA_CP)
  constexpr int NSD = 8;  // no scale SMEM -> the constant-scale NS
#else
  constexpr int NSD = (SF_ATOMS_PER_CELL == 2) ? 8 : 6;
#endif
  switch (K) {
    case   256: return run_one_k1<  1, 128,   3, ORDER, STATIC_SCHED, BLOCK_M, BLOCK_N, SF_ATOMS_PER_CELL, M_RUN>(M, N, verify, run_len, options);
    case   512: return run_one_k1<  2, 128,   4, ORDER, STATIC_SCHED, BLOCK_M, BLOCK_N, SF_ATOMS_PER_CELL, M_RUN>(M, N, verify, run_len, options);
    case  1024: return run_one_k1<  4, 128,   5, ORDER, STATIC_SCHED, BLOCK_M, BLOCK_N, SF_ATOMS_PER_CELL, M_RUN>(M, N, verify, run_len, options);
#if defined(K1_TUNE_NS2048)
    case  2048: return run_one_k1<  8, 128, K1_TUNE_NS2048, ORDER, STATIC_SCHED, BLOCK_M, BLOCK_N, SF_ATOMS_PER_CELL, M_RUN>(M, N, verify, run_len, options);
#else
    case  2048: return run_one_k1<  8, 128, NSD, ORDER, STATIC_SCHED, BLOCK_M, BLOCK_N, SF_ATOMS_PER_CELL, M_RUN>(M, N, verify, run_len, options);
#endif
    case  4096: return run_one_k1< 16, 128, NSD, ORDER, STATIC_SCHED, BLOCK_M, BLOCK_N, SF_ATOMS_PER_CELL, M_RUN>(M, N, verify, run_len, options);
    case  8192: return run_one_k1< 32, 128, NSD, ORDER, STATIC_SCHED, BLOCK_M, BLOCK_N, SF_ATOMS_PER_CELL, M_RUN>(M, N, verify, run_len, options);
    case 16384: return run_one_k1< 64, 128, NSD, ORDER, STATIC_SCHED, BLOCK_M, BLOCK_N, SF_ATOMS_PER_CELL, M_RUN>(M, N, verify, run_len, options);
    case 30720: return run_one_k1<120, 128, NSD, ORDER, STATIC_SCHED, BLOCK_M, BLOCK_N, SF_ATOMS_PER_CELL, M_RUN>(M, N, verify, run_len, options);
  }
  printf("  K=%d not in dispatch table (NTC=128)\n", K); return 0.0;
}

// Entry point. NTC is pinned at 128 for every shape -- the TMEM scale region
// leaves no room for 256; see the TMEM note above K1_TMEM_NCOLS.
template <ClcRasterOrder ORDER = ClcRasterOrder::AlongN,
          bool STATIC_SCHED = false, int BLOCK_M = 0, int BLOCK_N = 0,
          int SF_ATOMS_PER_CELL = 2, bool M_RUN = false>
static double run_k1(int M, int N, int K, bool verify = false, int run_len = 0,
                     const K1RunOptions& options = {}) {
  if (N >= 128 && N % 128 == 0)
    return run_k1_kdispatch_ntc128<ORDER, STATIC_SCHED, BLOCK_M, BLOCK_N,
                                   SF_ATOMS_PER_CELL, M_RUN>(M, N, K, verify,
                                                             run_len, options);
  printf("  N=%d not multiple of 128\n", N); return 0.0;
}

// One point of the blocked-raster ladder. BLOCK_M/BLOCK_N have to be template
// arguments (they shape the kernel's index math), so each probe is its own
// instantiation.
template <int BLOCK_M, int BLOCK_N>
static double k1_blocked_probe(int M, int N, int K, bool verify = false) {
  return run_k1<ClcRasterOrder::AlongN, /*STATIC_SCHED=*/true, BLOCK_M, BLOCK_N>(
      M, N, K, verify);
}

// NTC=256 probe: static M-RUN, NS=4 (skip needs K_BLOCKS %% 4 == 0).
static double k1_ntc256_probe(int M, int N, int K, bool verify = false) {
  switch (K) {
    case  2048: return run_one_k1<  8, 256, 4, ClcRasterOrder::AlongN, true, 0, 0, 2, true>(M, N, verify);
    case  4096: return run_one_k1< 16, 256, 4, ClcRasterOrder::AlongN, true, 0, 0, 2, true>(M, N, verify);
    case  8192: return run_one_k1< 32, 256, 4, ClcRasterOrder::AlongN, true, 0, 0, 2, true>(M, N, verify);
    case 16384: return run_one_k1< 64, 256, 4, ClcRasterOrder::AlongN, true, 0, 0, 2, true>(M, N, verify);
  }
  printf("  K=%d not in NTC=256 table\n", K); return 0.0;
}

template <int NS_T>
static double k1_ntc256_ns_probe(int M, int N, int K, bool verify = false) {
  switch (K) {
    case  2048: return run_one_k1<  8, 256, NS_T, ClcRasterOrder::AlongN, true, 0, 0, 2, true>(M, N, verify);
    case  4096: return run_one_k1< 16, 256, NS_T, ClcRasterOrder::AlongN, true, 0, 0, 2, true>(M, N, verify);
    case  8192: return run_one_k1< 32, 256, NS_T, ClcRasterOrder::AlongN, true, 0, 0, 2, true>(M, N, verify);
    case 16384: return run_one_k1< 64, 256, NS_T, ClcRasterOrder::AlongN, true, 0, 0, 2, true>(M, N, verify);
  }
  printf("  K=%d not in NTC=256 table\n", K); return 0.0;
}

static double k1_ntc256_splitb_probe(int M, int N, int K, bool verify = false,
                                     int run_len = 0) {
  switch (K) {
    case  2048: return run_one_k1<  8, 256, 4, ClcRasterOrder::AlongN, true, 0, 0, 2, true, 7>(M, N, verify, run_len);
    case  4096: return run_one_k1< 16, 256, 4, ClcRasterOrder::AlongN, true, 0, 0, 2, true, 7>(M, N, verify, run_len);
    case  8192: return run_one_k1< 32, 256, 4, ClcRasterOrder::AlongN, true, 0, 0, 2, true, 7>(M, N, verify, run_len);
    case 16384: return run_one_k1< 64, 256, 4, ClcRasterOrder::AlongN, true, 0, 0, 2, true, 7>(M, N, verify, run_len);
    case 30720: return run_one_k1<120, 256, 4, ClcRasterOrder::AlongN, true, 0, 0, 2, true, 7>(M, N, verify, run_len);
  }
  printf("  K=%d not in splitb table\n", K); return 0.0;
}

// Best blocked-raster candidate: bm x bn ladder winners at NTC=128 and
// (where the shape fits) NTC=256 splitb. Big-square/cube L2-locality
// lever; only worth probing when the B working set exceeds L2
// (N * K / 2 > ~100 MB) or the square is large.
template <int BM, int BN>
static double k1_blocked128(int M, int N, int K) {
  return run_k1<ClcRasterOrder::AlongN, /*STATIC_SCHED=*/true, BM, BN>(M, N, K);
}
template <int BM, int BN>
static double k1_blocked256(int M, int N, int K) {
  // NS_A=6 / NS_B=6 (12 stages, SMEM max): +4.5pp on big rows vs the
  // (5,6) split. (7,6)/(6,7) do not fit.
  switch (K) {
    case  4096: return run_one_k1<16, 256, 6, ClcRasterOrder::AlongN, true, BM, BN, 2, false, 6>(M, N);
    case  8192: return run_one_k1<32, 256, 6, ClcRasterOrder::AlongN, true, BM, BN, 2, false, 6>(M, N);
    case 16384: return run_one_k1<64, 256, 6, ClcRasterOrder::AlongN, true, BM, BN, 2, false, 6>(M, N);
    case 30720: return run_one_k1<120, 256, 6, ClcRasterOrder::AlongN, true, BM, BN, 2, false, 6>(M, N);
  }
  return 0.0;
}
template <int NSA, int NSB>
static double k1_ntc256_ns(int M, int N, int K) {
  switch (K) {
    case  2048: return run_one_k1<  8, 256, NSA, ClcRasterOrder::AlongN, true, 0, 0, 2, false, NSB>(M, N);
    case  4096: return run_one_k1< 16, 256, NSA, ClcRasterOrder::AlongN, true, 0, 0, 2, false, NSB>(M, N);
    case  8192: return run_one_k1< 32, 256, NSA, ClcRasterOrder::AlongN, true, 0, 0, 2, false, NSB>(M, N);
    case 16384: return run_one_k1< 64, 256, NSA, ClcRasterOrder::AlongN, true, 0, 0, 2, false, NSB>(M, N);
  }
  return 0.0;
}
template <int BM, int BN, int NSA, int NSB>
static double k1_blocked256_ns(int M, int N, int K) {
  switch (K) {
    case  4096: return run_one_k1<16, 256, NSA, ClcRasterOrder::AlongN, true, BM, BN, 2, false, NSB>(M, N);
    case  8192: return run_one_k1<32, 256, NSA, ClcRasterOrder::AlongN, true, BM, BN, 2, false, NSB>(M, N);
    case 16384: return run_one_k1<64, 256, NSA, ClcRasterOrder::AlongN, true, BM, BN, 2, false, NSB>(M, N);
    case 30720: return run_one_k1<120, 256, NSA, ClcRasterOrder::AlongN, true, BM, BN, 2, false, NSB>(M, N);
  }
  return 0.0;
}
static double k1_blocked_best(int M, int N, int K, const char** label) {
  static char lbl[40];
  // Blocked raster pays when EITHER operand spills L2 (126 MB): the
  // gen-o family is A-bound (A = M*K/2 up to 252 MB), not B-bound.
  const size_t a_bytes = (size_t)M * K / 2, b_bytes = (size_t)N * K / 2;
  if (a_bytes < 50u * 1024 * 1024 && b_bytes < 50u * 1024 * 1024)
    return 0.0;
  double best = 0.0;
  auto upd = [&](double v, const char* fmt, int bm, int bn) {
    if (v > best) { best = v; snprintf(lbl, sizeof lbl, fmt, bm, bn); }
  };
  upd(k1_blocked128< 8,  8>(M, N, K), "BLK128 bm%d bn%d",  8,  8);
  upd(k1_blocked128<16,  8>(M, N, K), "BLK128 bm%d bn%d", 16,  8);
  upd(k1_blocked128<16, 16>(M, N, K), "BLK128 bm%d bn%d", 16, 16);
  if (M % 256 == 0 && N % 256 == 0) {
    upd(k1_blocked256< 8,  8>(M, N, K), "BLK256 A66 bm%d bn%d",  8,  8);
    upd(k1_blocked256<16,  8>(M, N, K), "BLK256 A66 bm%d bn%d", 16,  8);
    upd(k1_blocked256<16, 16>(M, N, K), "BLK256 A66 bm%d bn%d", 16, 16);
    // A-heavy narrow-N rows (B is L2-resident): a deeper A ring beats
    // the balanced split (gen-o-4M 86.4 -> 87.4). With the
    // hoisted window wait the maximal (8,4) split wins outright
    // (gen-o 85.1, gen-o-4M 90.3).
    if (N <= 2048 && (K == 4096 || K == 8192)) {
      upd(k1_blocked256_ns< 8, 8, 7, 5>(M, N, K), "BLK256 A75 bm%d bn%d", 8, 8);
      upd(k1_blocked256_ns< 8, 8, 8, 4>(M, N, K), "BLK256 A84 bm%d bn%d", 8, 8);
    }
  }
  *label = lbl;
  return best;
}

// Best NTC=256 candidate for a bench shape: plain splitb M-RUN, plus
// RUN-STRIDED run lengths for wide N. Returns 0 if the shape or K does
// not fit the NTC=256 dispatch.
static double k1_ntc256_best(int M, int N, int K, const char** label) {
  static char lbl[40];
  if (M % 256 || N % 256) return 0.0;
  if (K != 2048 && K != 4096 && K != 8192 && K != 16384 && K != 30720)
    return 0.0;
  double best = k1_ntc256_splitb_probe(M, N, K);
  snprintf(lbl, sizeof lbl, "NTC256 M-RUN");
  {
    // Plain strided AlongN at NTC=256 (no M-RUN order): the honest
    // ingress-halving candidate for per-SM-walled flagship shapes.
    // NS_A=6 / NS_B=6 (12 stages, SMEM max; legal without skip) --
    // ring depth covers the TMA round trip.
    double pl = 0.0;
    switch (K) {
      case  2048: pl = run_one_k1<  8, 256, 6, ClcRasterOrder::AlongN, true, 0, 0, 2, false, 6>(M, N); break;
      case  4096: pl = run_one_k1< 16, 256, 6, ClcRasterOrder::AlongN, true, 0, 0, 2, false, 6>(M, N); break;
      case  8192: pl = run_one_k1< 32, 256, 6, ClcRasterOrder::AlongN, true, 0, 0, 2, false, 6>(M, N); break;
      case 16384: pl = run_one_k1< 64, 256, 6, ClcRasterOrder::AlongN, true, 0, 0, 2, false, 6>(M, N); break;
      case 30720: pl = run_one_k1<120, 256, 6, ClcRasterOrder::AlongN, true, 0, 0, 2, false, 6>(M, N); break;
    }
    if (pl > best) { best = pl; snprintf(lbl, sizeof lbl, "NTC256 A66"); }
  }
  if (K <= 4096) {
    // Short-K rows with L2-resident B: the A-heavy (7,5) split beats
    // the balanced (6,6) (und-o 72.4 -> 74.8, gen-qkv-2M 72.7 -> 73.2).
    const double a75 = k1_ntc256_ns<7, 5>(M, N, K);
    if (a75 > best) { best = a75; snprintf(lbl, sizeof lbl, "NTC256 A75"); }
  }
  if (N >= 8192) {
    for (int rl : {8, 16, 24, 32}) {
      if ((N / 256) % rl) continue;
      double rs = k1_ntc256_splitb_probe(M, N, K, false, rl);
      if (rs > best) {
        best = rs;
        snprintf(lbl, sizeof lbl, "NTC256 R-STRIDE rl=%d", rl);
      }
    }
  }
  *label = lbl;
  return best;
}

// CLC-RUN probe: one CLC grab = run_len same-m tiles (AlongM decode).
static double k1_runclc_probe(int M, int N, int K, int run_len,
                              bool verify = false) {
  return run_k1<ClcRasterOrder::AlongM, /*STATIC_SCHED=*/false, 0, 0,
                2, /*M_RUN=*/true>(M, N, K, verify, run_len);
}

// RUN-STRIDED M-RUN probe: run_len must divide N/128.
static double k1_rstride_probe(int M, int N, int K, int run_len,
                               bool verify = false) {
  return run_k1<ClcRasterOrder::AlongN, /*STATIC_SCHED=*/true, 0, 0,
                2, /*M_RUN=*/true>(M, N, K, verify, run_len);
}

// Blocked raster + M-RUN: contiguous per-cluster ranges over the blocked
// order, n-fastest inside blocks, A/SFA skipped on same-m continuation.
template <int BLOCK_M, int BLOCK_N>
static double k1_blocked_mrun_probe(int M, int N, int K, bool verify = false) {
  return run_k1<ClcRasterOrder::AlongN, /*STATIC_SCHED=*/true, BLOCK_M, BLOCK_N,
                2, /*M_RUN=*/true>(M, N, K, verify);
}

#ifndef MOE_DISABLE_MAIN
static bool k1_parse_positive_int(const char*& text, int& value) {
  char* end = nullptr;
  errno = 0;
  const long parsed = strtol(text, &end, 10);
  if (errno || end == text || parsed <= 0 || parsed > INT_MAX) return false;
  value = static_cast<int>(parsed);
  text = end;
  return true;
}

int main(int argc, char** argv) {
  K1RunOptions options;
  int M = 0, K = 0, N = 0;
  bool verify = true;
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--shape=", 8) == 0) {
      const char* text = argv[i] + 8;
      if (!k1_parse_positive_int(text, M) || *text++ != ',' ||
          !k1_parse_positive_int(text, K) || *text++ != ',' ||
          !k1_parse_positive_int(text, N) || *text != '\0') {
        fprintf(stderr, "--shape must be positive integers M,K,N\n");
        return 1;
      }
    } else if (strncmp(argv[i], "--fill=", 7) == 0) {
      const char* text = argv[i] + 7;
      if (!k1_parse_positive_int(text, options.fill) || *text || options.fill > 4) {
        fprintf(stderr, "--fill must be 1, 2, 3, or 4\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--no-benchmark") == 0) {
      options.benchmark = false;
    } else if (strcmp(argv[i], "--no-gpu-verify") == 0) {
      verify = false;  // Common CLI spelling; K1's built-in verifier runs on CPU.
    } else if (strncmp(argv[i], "--dump-output=", 14) == 0 && argv[i][14]) {
      options.dump_output = argv[i] + 14;
    } else if (strcmp(argv[i], "--help") == 0) {
      printf("Usage: %s [--shape=M,K,N [--fill=1..4] [--no-benchmark]\n"
             "       [--no-gpu-verify] [--dump-output=PATH]]\n"
             "No arguments: legacy sweep (FILL environment variable supported).\n", argv[0]);
      return 0;
    } else {
      fprintf(stderr, "unknown or empty argument: %s\n", argv[i]);
      return 1;
    }
  }
  if (argc > 1) {
    const bool supported_k = K == 256 || K == 512 || K == 1024 || K == 2048 ||
                             K == 4096 || K == 8192 || K == 16384 || K == 30720;
    if (!M || !N || M % 256 || N % 128 || !supported_k ||
        static_cast<long long>(M / 256) * (N / 128) > INT_MAX) {
      fprintf(stderr, "require --shape=M,K,N: M multiple of 256, N multiple of 128,\n"
                      "K in {256,512,1024,2048,4096,8192,16384,30720}\n");
      return 1;
    }
    options.standard_output = true;
  }
  CUDA_CHECK(cudaFree(0));
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 64 << 20));
  printf("K1 dense_gemm_nvfp4 (sm_100a, GB200) -- block32 UE8M0\n\n");
  if (options.standard_output) {
    run_k1<ClcRasterOrder::AlongN>(M, N, K, verify, 0, options);
    return 0;
  }
#if defined(K1_PROBE_ONLY)
  printf("=== NTC=256 bring-up probes ===\n");
  if (getenv("K1_VERIFY_ALL")) {
    // Correctness-only pass: every bench shape on its PICKED winner
    // config, verify on. Run under each FILL mode.
    struct { const char* name; int M, N, K; int cfg; } v[] = {
      // cfg 0 = NTC256 AlongN (5,6) splitb   1 = BLK256 bm8bn8 (5? uses 4/7)
      // cfg 2 = M-RUN NTC128 (legal skip)    3 = AlongN CLC NTC128
      {"und-gate",   14080,   128,  2048, 3}, {"und-qkv", 14080, 5120, 2048, 2},
      {"und-o",      14080,  2048,  4096, 0}, {"gen-gate", 30720, 128, 2048, 3},
      {"gen-qkv",    30720,  5120,  2048, 2}, {"gen-o",   30720, 2048, 4096, 0},
      {"gen-o-2K",   30720,  2048,  8192, 0}, {"gen-o-4K", 30720, 2048, 16384, 0},
      {"2K+2N",      30720,  4096,  8192, 0}, {"4K+2N",   30720, 4096, 16384, 0},
      {"2M+4K+2N",   61440,  4096, 16384, 0}, {"gen-o-2M", 61440, 2048, 4096, 0},
      {"gen-o-4M",  122880,  2048,  4096, 0}, {"gen-o-2N", 30720, 4096, 4096, 0},
      {"gen-qkv-2M", 61440,  5120,  2048, 0}, {"gen-qkv-2N", 30720, 10240, 2048, 0},
      {"16384^3",    16384, 16384, 16384, 1}, {"30720^3", 30720, 30720, 30720, 1},
      {"32k-mixed",  32768, 32768, 16384, 1}, {"2M+2N",   61440, 4096, 4096, 0},
      {"2M+4N",      61440,  8192,  4096, 0}, {"2M+8N",   61440, 16384, 4096, 0},
      {"4M+4K+2N",  122880,  4096, 16384, 0}, {"2M+4K+4N", 61440, 8192, 16384, 0},
      {"4M+4K+4N",  122880,  8192, 16384, 0}, {"2M+4K+8N", 61440, 16384, 16384, 1},
    };
    auto ntc256_56 = [](int M, int N, int K) {
      switch (K) {
        case  2048: return run_one_k1<  8, 256, 6, ClcRasterOrder::AlongN, true, 0, 0, 2, false, 6>(M, N, true);
        case  4096: return run_one_k1< 16, 256, 6, ClcRasterOrder::AlongN, true, 0, 0, 2, false, 6>(M, N, true);
        case  8192: return run_one_k1< 32, 256, 6, ClcRasterOrder::AlongN, true, 0, 0, 2, false, 6>(M, N, true);
        case 16384: return run_one_k1< 64, 256, 6, ClcRasterOrder::AlongN, true, 0, 0, 2, false, 6>(M, N, true);
        case 30720: return run_one_k1<120, 256, 6, ClcRasterOrder::AlongN, true, 0, 0, 2, false, 6>(M, N, true);
      }
      return 0.0;
    };
    auto blk256 = [](int M, int N, int K) {
      switch (K) {
        case 16384: return run_one_k1< 64, 256, 6, ClcRasterOrder::AlongN, true, 8, 8, 2, false, 6>(M, N, true);
        case 30720: return run_one_k1<120, 256, 6, ClcRasterOrder::AlongN, true, 8, 8, 2, false, 6>(M, N, true);
      }
      return 0.0;
    };
    for (auto& t : v) {
      printf("[%s]\n", t.name);
      switch (t.cfg) {
        case 0: if (t.M % 256 == 0 && t.N % 256 == 0) { ntc256_56(t.M, t.N, t.K); break; }
                run_k1<ClcRasterOrder::AlongN>(t.M, t.N, t.K, true); break;
        case 1: blk256(t.M, t.N, t.K); break;
        case 2: run_k1<ClcRasterOrder::AlongN, true, 0, 0, 2, true>(t.M, t.N, t.K, true); break;
        default: run_k1<ClcRasterOrder::AlongN>(t.M, t.N, t.K, true); break;
      }
    }
    return 0;
  }
  k1_ntc256_probe(14080, 5120, 2048, /*verify=*/true);
  k1_ntc256_probe(30720, 4096, 8192, /*verify=*/true);   // 2K+2N
  k1_ntc256_probe(30720, 2048, 4096, /*verify=*/true);   // gen-o
  k1_ntc256_probe(30720, 5120, 2048, /*verify=*/true);   // gen-qkv
  k1_ntc256_probe(61440, 4096, 16384, /*verify=*/true);  // 2M+4K+2N
  printf("--- splitb NS_B=7 full grid ---\n");
  struct { const char* name; int M, N, K; } grid[] = {
    {"gen-o",      30720,  2048,  4096},
    {"gen-o-2K",   30720,  2048,  8192},
    {"gen-o-4K",   30720,  2048, 16384},
    {"gen-qkv",    30720,  5120,  2048},
    {"gen-qkv-s",  14080,  5120,  2048},
    {"gen-qkv-2M", 61440,  5120,  2048},
    {"gen-qkv-2N", 30720, 10240,  2048},
    {"2K+2N",      30720,  4096,  8192},
    {"4K+2N",      30720,  4096, 16384},
    {"2M+4K+2N",   61440,  4096, 16384},
    {"gen-o-2M",   61440,  2048,  4096},
    {"gen-o-4M",  122880,  2048,  4096},
    {"gen-o-2N",   30720,  4096,  4096},
    {"16384^3",    16384, 16384, 16384},
    {"30720^3",    30720, 30720, 30720},
    {"32k-mixed",  32768, 32768, 16384},
    {"2M+2N",      61440,  4096,  4096},
    {"2M+4N",      61440,  8192,  4096},
    {"2M+8N",      61440, 16384,  4096},
    {"4M+4K+2N",  122880,  4096, 16384},
    {"2M+4K+4N",   61440,  8192, 16384},
    {"4M+4K+4N",  122880,  8192, 16384},
    {"2M+4K+8N",   61440, 16384, 16384},
  };
  printf("--- 2K+2N squeeze ---\n");
  printf("[NS_A=5 NS_B=6]\n");
  run_one_k1<32, 256, 5, ClcRasterOrder::AlongN, true, 0, 0, 2, false, 6>(30720, 4096, true);
  run_one_k1<32, 256, 5, ClcRasterOrder::AlongN, true, 0, 0, 2, false, 6>(30720, 4096, false);
  printf("[NS_A=5 NS_B=7 (needs EPI16)]\n");
  run_one_k1<32, 256, 5, ClcRasterOrder::AlongN, true, 0, 0, 2, false, 7>(30720, 4096, true);
  return 0;
  printf("=== CLC-RUN quick probes ===\n");
  k1_runclc_probe(30720, 5120, 2048, 8, /*verify=*/true);   // gen-qkv
  k1_runclc_probe(30720, 5120, 2048, 20);
  k1_runclc_probe(30720, 5120, 2048, 40);
  k1_runclc_probe(30720, 2048, 4096, 8);                    // gen-o
  k1_runclc_probe(30720, 2048, 4096, 16);
  k1_runclc_probe(30720, 4096, 8192, 8);                    // 2K+2N
  k1_runclc_probe(30720, 4096, 8192, 16);
  k1_runclc_probe(30720, 4096, 8192, 32);
  k1_runclc_probe(61440, 16384, 4096, 16, /*verify=*/true); // 2M+8N
  k1_runclc_probe(61440, 16384, 4096, 32);
  k1_runclc_probe(61440, 16384, 4096, 64);
  return 0;
#endif

  struct S { const char* lbl; int M, K, N; };
  // V0 shapes (standard)
  // The V0 inventory plus the shapes that clear 85% SOL. These are the ones
  // run with verify=true. Every shape appears in exactly one list below.
  const S v0_shapes[] = {
    { "und-gate",  14080, 2048,  128 },
    { "und-qkv",   14080, 2048, 5120 },
    { "und-o",     14080, 4096, 2048 },
    { "gen-gate",  30720, 2048,  128 },
    { "gen-qkv",   30720, 2048, 5120 },
    { "gen-o",     30720, 4096, 2048 },
    { "gen-o-2K",  30720, 8192, 2048 },   // 83.7%: larger K amortizes per-tile cost
    { "gen-o-4K",  30720,16384, 2048 },   // 83.3%
    { "2K+2N",     30720, 8192, 4096 },   // 87.6% -- best measured
    { "4K+2N",     30720,16384, 4096 },   // 84.0%
    { "2M+4K+2N",  61440,16384, 4096 },   // 83.9%
  };
  // Larger problem sizes: 2x/4x M, larger K, larger N
  // Scaling study: one axis at a time off gen-o / gen-qkv, plus the two large
  // squares where the plain raster collapses (both ~61% -- see the blocked
  // raster ladder at the end of main).
  const S large_shapes[] = {
    { "gen-o-2M",   61440, 4096,  2048 },
    { "gen-o-4M",  122880, 4096,  2048 },
    { "gen-o-2N",   30720, 4096,  4096 },
    { "gen-qkv-2M", 61440, 2048,  5120 },
    { "gen-qkv-2N", 30720, 2048, 10240 },
    { "16384^3",    16384,16384, 16384 },
    { "30720^3",    30720,30720, 30720 },
    { "32k-mixed",  32768,16384, 32768 },
  };

  printf("=== V0 shapes (CLC-persistent scheduler) ===\n");
  for (const auto& s : v0_shapes) {
    printf("[%s]\n", s.lbl);
    double cn = run_k1<ClcRasterOrder::AlongN>(s.M, s.N, s.K, /*verify=*/true);
    double cm = run_k1<ClcRasterOrder::AlongM>(s.M, s.N, s.K);
    double mr = run_k1<ClcRasterOrder::AlongN, /*STATIC_SCHED=*/true, 0, 0,
                       2, /*M_RUN=*/true>(s.M, s.N, s.K, /*verify=*/true);
    double best = (cn > cm) ? cn : cm;
    const char* raster = (cn > cm) ? "AlongN" : "AlongM";
    if (mr > best) { best = mr; raster = "M-RUN"; }
    { const char* l256 = nullptr;
      double t256 = k1_ntc256_best(s.M, s.N, s.K, &l256);
      if (t256 > best) { best = t256; raster = l256; } }
    { const char* lblk = nullptr;
      double tblk = k1_blocked_best(s.M, s.N, s.K, &lblk);
      if (tblk > best) { best = tblk; raster = lblk; } }
    printf("  %.1fT (%.1f%%) [%s]\n", best, 100.0*best/10000.0, raster);
  }

  printf("\n=== Larger problem sizes (CLC-persistent scheduler) ===\n");
  for (const auto& s : large_shapes) {
    printf("[%s] M=%d K=%d N=%d\n", s.lbl, s.M, s.K, s.N);
    double cn = run_k1<ClcRasterOrder::AlongN>(s.M, s.N, s.K, /*verify=*/false);
    double cm = run_k1<ClcRasterOrder::AlongM>(s.M, s.N, s.K);
    double mr = run_k1<ClcRasterOrder::AlongN, /*STATIC_SCHED=*/true, 0, 0,
                       2, /*M_RUN=*/true>(s.M, s.N, s.K, /*verify=*/false);
    double best = (cn > cm) ? cn : cm;
    const char* raster = (cn > cm) ? "AlongN" : "AlongM";
    if (mr > best) { best = mr; raster = "M-RUN"; }
    if (s.N >= 8192) {
      // Wide N: 1-D runs re-stream all of B per m-run (DRAM-bound). Probe
      // RUN-STRIDED run lengths (must divide N/128); cubes like ~48, wide
      // mixed shapes ~16-64.
      static const int kRls[] = { 16, 32, 48, 64 };
      static char rl_lbl[32];
      for (int rl : kRls) {
        if ((s.N / 128) % rl) continue;
        double rs = k1_rstride_probe(s.M, s.N, s.K, rl);
        if (rs > best) {
          best = rs;
          snprintf(rl_lbl, sizeof rl_lbl, "R-STRIDE rl=%d", rl);
          raster = rl_lbl;
        }
      }
    }
    { const char* l256 = nullptr;
      double t256 = k1_ntc256_best(s.M, s.N, s.K, &l256);
      if (t256 > best) { best = t256; raster = l256; } }
    { const char* lblk = nullptr;
      double tblk = k1_blocked_best(s.M, s.N, s.K, &lblk);
      if (tblk > best) { best = tblk; raster = lblk; } }
    printf("  %.1fT (%.1f%%) [%s]\n", best, 100.0*best/10000.0, raster);
  }

  // Push toward 85%: larger K and N sweeps
  printf("\n=== Pushing toward 85%% SOL ===\n");
  // The curve around the v0_shapes winners: N growing past ~4096 gives the
  // gain back to L2.
  struct { const char* lbl; int M, K, N; } push_shapes[] = {
    { "2M+2N",     61440, 4096,  4096 },
    { "2M+4N",     61440, 4096,  8192 },
    { "2M+8N",     61440, 4096, 16384 },
    { "4M+4K+2N", 122880,16384,  4096 },
    { "2M+4K+4N",  61440,16384,  8192 },
    { "4M+4K+4N", 122880,16384,  8192 },
    { "2M+4K+8N",  61440,16384, 16384 },
  };
  for (const auto& s : push_shapes) {
    printf("[%s] M=%d K=%d N=%d\n", s.lbl, s.M, s.K, s.N);
    double cn = run_k1<ClcRasterOrder::AlongN>(s.M, s.N, s.K, /*verify=*/false);
    double cm = run_k1<ClcRasterOrder::AlongM>(s.M, s.N, s.K);
    double mr = run_k1<ClcRasterOrder::AlongN, /*STATIC_SCHED=*/true, 0, 0,
                       2, /*M_RUN=*/true>(s.M, s.N, s.K, /*verify=*/false);
    double best = (cn > cm) ? cn : cm;
    const char* raster = (cn > cm) ? "AlongN" : "AlongM";
    if (mr > best) { best = mr; raster = "M-RUN"; }
    if (s.N >= 8192) {
      // Wide N: 1-D runs re-stream all of B per m-run (DRAM-bound). Probe
      // RUN-STRIDED run lengths (must divide N/128); cubes like ~48, wide
      // mixed shapes ~16-64.
      static const int kRls[] = { 16, 32, 48, 64 };
      static char rl_lbl[32];
      for (int rl : kRls) {
        if ((s.N / 128) % rl) continue;
        double rs = k1_rstride_probe(s.M, s.N, s.K, rl);
        if (rs > best) {
          best = rs;
          snprintf(rl_lbl, sizeof rl_lbl, "R-STRIDE rl=%d", rl);
          raster = rl_lbl;
        }
      }
    }
    { const char* l256 = nullptr;
      double t256 = k1_ntc256_best(s.M, s.N, s.K, &l256);
      if (t256 > best) { best = t256; raster = l256; } }
    { const char* lblk = nullptr;
      double tblk = k1_blocked_best(s.M, s.N, s.K, &lblk);
      if (tblk > best) { best = tblk; raster = lblk; } }
    printf("  %.1fT (%.1f%%) [%s]\n", best, 100.0*best/10000.0, raster);
  }

  // Static scheduler + 2D blocked raster, on the shape that needs it.
  // 16384^3 is the case where AlongN falls apart: 128 N-tiles means a B column
  // is evicted long before it is reused, so every stage comes from HBM. The
  // ladder below walks the L2 working set (BLOCK_M A-tiles at 1 MB each plus
  // BLOCK_N B-tiles at 512 KB each) across the optimum and out the far side.
  printf("\n=== Static + 2D blocked raster (16384^3, K=16384) ===\n");
  k1_blocked_probe<  1,   1>(16384, 16384, 16384);   // == plain AlongN
  k1_blocked_probe<  4,   8>(16384, 16384, 16384);
  k1_blocked_probe<  8,   8>(16384, 16384, 16384);
  k1_blocked_probe<  8,  16>(16384, 16384, 16384);
  k1_blocked_probe< 16,   8>(16384, 16384, 16384);
  k1_blocked_probe< 16,  16>(16384, 16384, 16384, /*verify=*/true);  // best
  k1_blocked_probe< 16,  32>(16384, 16384, 16384, /*verify=*/true);
  k1_blocked_probe< 32,  32>(16384, 16384, 16384);   // block > L2, falls off
  k1_blocked_probe< 64,  64>(16384, 16384, 16384);
  // Blocked + M-RUN (A/SFA skip inside same-m runs, n-fast in block):
  k1_blocked_mrun_probe<  4,  16>(16384, 16384, 16384);
  k1_blocked_mrun_probe<  8,  16>(16384, 16384, 16384, /*verify=*/true);
  k1_blocked_mrun_probe<  8,  32>(16384, 16384, 16384);
  k1_blocked_mrun_probe< 16,  16>(16384, 16384, 16384);
  k1_blocked_mrun_probe< 16,  32>(16384, 16384, 16384);
  k1_blocked_mrun_probe<  8,  16>(30720, 30720, 30720);
  k1_blocked_mrun_probe< 16,  32>(30720, 30720, 30720);
  k1_blocked_mrun_probe<  8,  16>(32768, 32768, 16384);
  k1_blocked_mrun_probe< 16,  32>(32768, 32768, 16384);

  return 0;
}
#endif

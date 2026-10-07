// block_sparse_bwd_bf16_blk64.cu -- VSA block-sparse BACKWARD, bf16, sm_100a,
// 64-token blocks, ONE-PASS KV-stationary.
//
// One CTA owns one kv64 block and walks that block's q-list. Per visited q-quad it recomputes P^T,
// forms dS^T, accumulates dK/dV in TMEM, and pushes dQ partials to a global accumulator with
// cp.reduce.async.bulk (fp32 by default, f16 with DQ_F16=1; DQ_DTYPE). A postprocess kernel
// unscrambles the accumulator, applies sm_scale, and stores bf16 dQ. The dQ reduce traffic scales
// as S^2 * density, like the FLOPs, so throughput is flat in S (fp32 ~750 TF, f16 ~900 TF at 25%).
// With fp32 the two-pass sibling (_2pv) is faster wherever its dS buffer fits (<= 131k at 25%);
// this kernel is the blk64 option at S >= 262k.
//
// Warps: 16 -- w0-3 epilogue (dQ drain), w4-11 softmax (P^T + dS^T), w12 mma, w13 load,
// w14 CLC scheduler, w15 register donor (see Budgets). GEMMs use the tcgen05.mma.ws Layout-E
// dual-pack (plain m128 for dQ) so 64-row blocks still run full 128-lane atoms.
//
// Structure:
//   1. Quads. The q-list is walked in QUADS of 4 q64 blocks = 256 gathered q tokens per step. When
//      the list length is not a multiple of 4, the final quad is padded by repeating the list's
//      last block (a 6-entry list [8,11,20,3,7,15] runs as quads [8,11,20,3] + [7,15,15,15]); the
//      softmax warps set P^T and dS^T to zero at the repeated (pad) q positions, so the duplicates
//      add nothing to dK/dV/dQ.
//   2. Ring. Per quad the kernel gathers the quad's 256 q tokens from the Q^T and dO^T tensors
//      ([H*hd, tokens], written by the preprocess) into ONE SMEM ring of four 32KB slots; a slot
//      holds two q64 blocks (128 tokens x 128 hd bf16) of one operand. Call the quad's four q64
//      blocks q0..q3 (by position). Each quad fills all four slots: s0 = dO^T q0,q1; s1 = dO^T
//      q2,q3; s2 = Q^T q0,q1; s3 = Q^T q2,q3. The assignment never rotates. K and V are loaded
//      once per CTA, as extra TMA bytes attached to the first quad's slot loads.
//   3. Slot release. A slot is freed by the tcgen05 commit of its LAST-reading GEMM (dV for the
//      dO^T slots, dK for the Q^T slots); freeing at any earlier reader races the tensor core.
//   4. Single-copy Q/dO. The ^T in the formulas is math notation. Q is read by two GEMMs that
//      contract over different dims: S^T = K @ Q^T over hd, dK += dS^T @ Q over q tokens (likewise
//      dO for dP^T and dV). The MMA calls a B tile K-major when its contraction dim is the
//      contiguous one, MN-major otherwise -- a label per GEMM, not per tensor. Natural Q ([tokens,
//      hd]) is K-major for S^T but MN-major for dK; Q^T ([hd, tokens]) is the reverse. Gathering
//      both layouts every quad makes the load path the pacer. This kernel keeps
//      ONE SMEM copy -- Q^T and dO^T (tokens contiguous), the form the TS GEMMs dK/dV need (tb=0,
//      K-major for them) -- and lets S^T/dP^T read the same copy via the instruction descriptor's
//      transpose-B bit tb=1 (B is MN-major: for S^T the contiguous tokens are N, and the MMA
//      transposes on read). Natural Q is never read by the main kernel. Compared to loading both Q
//      and Q^T: +4..31% measured. The price: one SMEM layout serves both GEMMs, which forces
//      S^T/dP^T to issue as 2x n128 (GEMM 1 below) -- a single m64n256 would need LBO = 8*SBO, and
//      then the token groups of adjacent N-atoms overlap.
//
// Data layout (D=128, BLOCK=64, one quad = 4 gathered q blocks = 256 q tokens):
//   name       where dtype shape                                    written by      -> read by
//   Q,K,V,dO,O gmem  bf16  [tokens, H, 128]                         caller          -> pre / TMA
//   LSE, Delta gmem  fp32  [B*H, S] (log2 form)                     fwd / pre       -> softmax
//   Q^T, dO^T  gmem  bf16  [H*128, tokens]                          preprocess      -> TMA gather
//   K, V       SMEM  bf16  2 hd-halves x [64 kv x 64 hd]            TMA (once/CTA)  -> GEMM 1,2,4
//   ring slot  SMEM  bf16  2 q64 blocks x [128 hd x 64 tok]         TMA gather      -> GEMM 1,2,3,5
//   S^T        TMEM  fp32  128 lanes x 128 q (dual, cols 0-127)     GEMM 1          -> softmax
//   P^T        TMEM  bf16  128 lanes x 128 q (bf16x2, cols 0-31, 64-95)     softmax -> GEMM 3
//   dP^T       TMEM  fp32  128 lanes x 128 q (dual, cols 256-383)   GEMM 2          -> softmax
//   dS^T       TMEM  bf16  128 lanes x 128 q (bf16x2, cols 256-287, 320-351) softmax -> GEMM 5
//   dS (sDST)  SMEM  bf16  [256 q x 64 kv], 4 tiles [64 kv x 64 q]   softmax         -> GEMM 4
//   dV, dK     TMEM  fp32  128 lanes x 128 hd (partials, *)         GEMM 3, 5       -> epilogue
//   dQ         TMEM  fp32  128 lanes x 128 hd (cols 256-383)        GEMM 4          -> epilogue
//   dqaccum    gmem  fp32  per (b*h, q64 block) 8192 elems       epilogue        -> postprocess
//   dK,dV,dQ   gmem  bf16  [tokens, H, 128]                         epilogue / post
//   q-half = the 128 q of a quad that share one TMEM lane-half: lanes 0-63 hold q0,q2, lanes
//            64-127 hold q1,q3 (each slot's issue splits its two blocks across the lane-halves).
//   (dual) = the two q-halves stacked in lanes: lane l = kv row (l & 63) of q-half (l >> 6).
//   (bf16x2) = two bf16 values packed in one 32-bit word, low half first. TMEM is addressed in
//            32-bit columns, so a softmax warp rewrites the 64 fp32 columns of its q block as 32
//            columns of bf16 pairs, in place at the start of those same 64 columns (P^T over
//            S^T, dS^T over dP^T): q block 0/1 at cols 0-31, q block 2/3 at cols 64-95. Staying
//            inside its own fp32 range means a warp never overwrites columns the other warp of
//            its lane group still has to load. The MMA reads the two 32-column runs as its bf16
//            A operand.
//   (*) dV at cols 128-255, dK at 384-511; lanes 0-63 / 64-127 hold the partial sum over q-half
//       0 / 1 of the same [64 kv x 128 hd] tile; the epilogue adds the two before storing bf16.
//
// The 5 GEMMs per quad, in issue order:
//   1. S^T = K @ Q^T      ws SS m64n256k16, issued as 2x m64n128 (tb=1): one issue per Q^T slot.
//        Each issue multiplies K [64 kv x 128 hd] by the slot's two q64 blocks and writes a
//        128-lane x 64-col TMEM tile (lane = kv row within the lane-half, col = q within block):
//          s2 issue: cols 0-63    lanes 0-63 = K @ Q^T(q0)    lanes 64-127 = K @ Q^T(q1)
//          s3 issue: cols 64-127  lanes 0-63 = K @ Q^T(q2)    lanes 64-127 = K @ Q^T(q3)
//        Read together: lanes 0-63 hold q0 (cols 0-63) and q2 (cols 64-127), lanes 64-127 hold
//        q1 and q3 -- the q-half layout defined above.
//   2. dP^T = V @ dO^T    ws SS m64n256k16, issued as 2x m64n128 (tb=1): one issue per dO^T slot.
//        Each issue multiplies V [64 kv x 128 hd] by the slot's two q64 blocks and writes a
//        128-lane x 64-col TMEM tile (lane = kv row within the lane-half, col = q within block):
//          s0 issue: cols 256-319  lanes 0-63 = V @ dO^T(q0)  lanes 64-127 = V @ dO^T(q1)
//          s1 issue: cols 320-383  lanes 0-63 = V @ dO^T(q2)  lanes 64-127 = V @ dO^T(q3)
//        The softmax warps then read S^T and dP^T and write P^T = exp2(S^T * scale - LSE)
//        and dS^T = P^T * (dP^T - Delta) in bf16: as TMEM overlays (P^T at cols 0-31 and 64-95,
//        dS^T at cols 256-287 and 320-351, each warp inside the fp32 columns it just read) plus
//        the SMEM dS buffer sDST for GEMM 4. At the pad q positions of a
//        tail quad (Structure item 1) the softmax warps write P^T = 0, which yields dS^T = 0 as
//        well.
//   3. dV += P^T @ dO     ws TS m64n256 (tb=0), accumulating over slot s0 (dO q0,q1) and then
//        slot s1 (dO q2,q3). The two lane-halves of the atom are independent products, so the
//        accumulator at TMEM cols 128-255 (lane = kv row within the lane-half, col = hd) gets:
//          over s0: lanes 0-63 += P^T(q0) @ dO(q0)    lanes 64-127 += P^T(q1) @ dO(q1)
//          over s1: lanes 0-63 += P^T(q2) @ dO(q2)    lanes 64-127 += P^T(q3) @ dO(q3)
//        P^T(qi) [64 kv x 64 q] is q block i's part of the P^T tile in TMEM (q0,q2 in lanes
//        0-63, q1,q3 in lanes 64-127 -- see q-half); dO(qi) [64 q x 128 hd] is q block i in the
//        slot, read tb=0. After the quad, lanes 0-63 hold dV summed over q0,q2 and lanes 64-127
//        dV summed over q1,q3; the epilogue adds the two lane-halves into dV [64 kv x 128 hd].
//        dO^T slots s0, s1: LAST reader; the commit after the s0 part frees s0, the commit after
//        the s1 part frees s1, both mid-quad.
//   4. dQ = dS @ K        [256 q x 128 hd] per quad, computed as plain SS m128n128k64 (ta=1,
//        tb=1) issued twice, once per q-half h; the two issues share one [128 q x 128 hd] fp32
//        tile at TMEM cols 256-383 (col = hd), the h=0 tile being drained before h=1 is issued:
//          h=0: lanes 0-63 = dS(q0) @ K    lanes 64-127 = dS(q2) @ K
//          h=1: lanes 0-63 = dS(q1) @ K    lanes 64-127 = dS(q3) @ K
//        dS is a 32 KB SMEM area (sDST) holding four 8 KB tiles, each [64 kv x 64 q] (kv rows, q
//        contiguous, 128B-swizzled), back to back in the order q0, q2, q1, q3. Together they are
//        the quad's dS^T [64 kv x 256 q]; the softmax warps write them straight from registers
//        (lane = kv row), the same data as the TMEM dS^T overlay. GEMM 4 reads two tiles per
//        issue, transposed, as dS(qi) [64 q x 64 kv]: the h=0 issue reads the first two (q0 ->
//        lanes 0-63, q2 -> lanes 64-127), the h=1 issue the last two (q1, q3). q is the MMA's M
//        and the contiguous dim in the tile, so MN-major, ta=1. K [64 kv x 128 hd] is the CTA's
//        kv64 block, resident in SMEM for the whole CTA as two 8 KB tiles, hd 0-63 then hd 64-127,
//        each [64 kv x 64 hd] (kv rows, hd contiguous, 128B-swizzled): hd is N and contiguous, so
//        MN-major, tb=1. The epilogue warps add each dQ tile into the global fp32 accumulator with
//        cp.reduce.async.bulk (one q64 block per lane-half) and free the columns before the h=1
//        issue reuses them. No Q^T / dO^T slot is read here.
//   5. dK += dS^T @ Q     ws TS m64n256 (tb=0), accumulating over slot s2 (Q q0,q1) and then
//        slot s3 (Q q2,q3), into TMEM cols 384-511 (lane = kv row within the lane-half, col = hd):
//          over s2: lanes 0-63 += dS^T(q0) @ Q(q0)    lanes 64-127 += dS^T(q1) @ Q(q1)
//          over s3: lanes 0-63 += dS^T(q2) @ Q(q2)    lanes 64-127 += dS^T(q3) @ Q(q3)
//        dS^T(qi) [64 kv x 64 q] is q block i's part of the dS^T tile in TMEM (cols 256-287 for
//        q0/q1, 320-351 for q2/q3, same lane-halves as P^T); Q(qi) [64 q x 128 hd] is q block i in
//        the slot, read tb=0. After the
//        quad, lanes 0-63 hold dK summed over q0,q2 and lanes 64-127 dK summed over q1,q3; the
//        epilogue adds the two lane-halves (times sm_scale) into dK [64 kv x 128 hd]. Q^T slots
//        s2, s3: LAST reader; the commit after the s2 part frees s2, the commit after the s3 part
//        frees s3, both at quad end.
//
// Overlap:
//   1. Prefetch order. The dO^T slots s0, s1 free after GEMM 3, the Q^T slots s2, s3 only after
//      GEMM 5. The loader therefore fills the NEXT quad's dO^T into s0, s1 first, overlapping the
//      current quad's GEMMs 4 and 5, and its Q^T into s2, s3 second.
//
// Warp roles:
//   load (w13)       Per item: K and V tiles (with quad 0's slots). Per quad: TMA-gathers the
//                    4 q64 blocks of dO^T, then of Q^T, into the ring; bulk-copies their LSE and
//                    Delta rows.
//   mma (w12)        Issues GEMMs 1-5 in the order above and the commits that publish S^T, dP^T,
//                    dV, dK, dQ and free ring slots / K, V.
//   softmax (w4-11)  8 warps split every dual tile (128 TMEM lanes x 128 fp32 columns) into four
//                    32-lane groups x two 64-column halves. w4-7 take columns 0-63, w8-11 columns
//                    64-127; within each four, the first warp takes lanes 0-31, the second 32-63,

//                    the third 64-95, the fourth 96-127. Each piece is 32 kv rows x the 64 q of
//                    one quad block. Per quad: P^T = exp2(S^T*scale_log2 - LSE), packed to bf16
//                    pairs and stored over S^T's own TMEM columns, where GEMM 3 (dV) reads it as
//                    A; dS^T = P^T*(dP^T - Delta), packed the same way over dP^T's columns for
//                    GEMM 5 (dK) and also written to its sDST tile for GEMM 4 (dQ). Padded blocks
//                    get P = dS = 0. Per item: merge the two lane-half partial sums of dV, then
//                    dK, and TMA-store them.
//   epilogue (w0-3)  Per quad GEMM 4 is two plain SS m128n128 calls, one per q-half, each leaving
//                    a dQ tile of 128 q x 128 hd in TMEM. A q-half is the pair of quad blocks that
//                    share a lane half of the dual S^T / dP^T tiles: lanes 0-63 hold q0 and q2,
//                    lanes 64-127 hold q1 and q3; dS^T keeps that split (sDST tiles q0,q2,q1,q3):
//                    tile 0 = q0 in lanes 0-63, q2 in lanes 64-127; tile 1 = q1 in lanes 0-63, q3
//                    in lanes 64-127. Both calls write the same TMEM columns, so tile 0 is drained
//                    first, then tile 1. For each tile: w0 reads lanes 0-31, w1 lanes 32-63,
//                    w2 lanes 64-95, w3 lanes 96-127, each with all 128 hd columns (a thread holds
//                    one q row); the 4 warps release the tile (empty_bar_dq) and reduce-add it
//                    into dqaccum.
//   sched (w14)      CLC scheduler: fetches the next work item for all warps.
//   idle (w15)       No work; donates its registers.
//
// Barrier contract. arv = arrive count: 1 is one elected thread, a TMA completion or a tcgen05
// commit; 8 is one elected lane per softmax warp; 4 one per epilogue warp; 256 every softmax
// thread. A tcgen05-commit arrive fires when the tensor core has finished the GEMMs issued before
// it (read the SMEM slot, written the TMEM tile), not when the MMA warp issued them.
//   barrier            ring  arv  producer -> consumer   meaning
//   -----------------  ----  ---  ---------------------  --------------------------------------
//   full_bar_ring       [4]   1   load     -> mma        slot's two q64 blocks resident; expect_tx
//                                                        32 KB, +16 KB when the slot also carries
//                                                        K (s2) or V (s3) on an item's first quad.
//   empty_bar_ring      [4]   1   mma      -> load       slot read: commit of its LAST GEMM (dV for
//                                                        s0, s1; dK for s2, s3).
//   full_bar_lse        [1]   1   load     -> softmax    quad's 256 LSE values; expect_tx 1 KB.
//   empty_bar_lse       [1]   8   softmax  -> load       LSE consumed (after the P^T loop).
//   full_bar_delta      [1]   1   load     -> softmax    quad's 256 Delta values; expect_tx 1 KB.
//   empty_bar_delta     [1]   8   softmax  -> load       Delta consumed (after the dS^T loop).
//   full_bar_st         [1]   1   mma      -> softmax    S^T in TMEM: commit after GEMM 1.
//   full_bar_dpt        [1]   1   mma      -> softmax    dP^T in TMEM: commit after GEMM 2.
//   full_bar_pt         [1]   8   softmax  -> mma        P^T overlay stored; GEMM 3 may read it.
//   full_bar_dst        [1]   8   softmax  -> mma        dS^T overlay and sDST written; GEMMs 5
//                                                        and 4 may read them.
//   full_bar_dq         [1]   1   mma      -> epilogue   one dQ tile in TMEM: commit after each
//                                                        GEMM 4 issue, twice per quad.
//   empty_bar_dq        [1]   4   epilogue -> mma        dQ tile read out of TMEM (before it is
//                                                        staged); primed once at kernel start.
//   full_bar_dv         [1]   1   mma      -> softmax    item's dV complete: commit after the last
//                                                        quad's GEMM 3.
//   full_bar_dk         [1]   1   mma      -> softmax    item's dK complete: commit after the last
//                                                        quad's GEMM 5.
//   empty_bar_kv        [1]   1   mma      -> load       K, V read: commit after the item's last
//                                                        GEMM 4; the next item may load K, V.
//   empty_bar_epi       [1] 256   softmax  -> load, mma  dV and dK TMA-stored: s0 (their staging
//                                                        tile) is free for the loader, the dV/dK
//                                                        TMEM tiles for the MMA's zero-init.
//
// Phase trackers: every empty_* wait on the load and MMA warps is an EmptyPhaseTracker (seeded to
// parity 1), so its first wait passes with nobody arrived -- for the ring that is the loader's
// first four waits, one per slot. The exception is the MMA's empty_bar_dq, a PhaseTracker (parity
// 0) that the epilogue warps prime with one real arrive each at kernel start. All full_* trackers
// are PhaseTrackers.
//
// Swimlane: 4 warp columns, time flows DOWN. The arrow tail (+) is the arrive, the head (>/<) the
// warp whose wait it unblocks. Rows follow the MMA warp's own wait/commit order, since it is the
// pacer. n = num_quads. The sched (w14) and donor (w15) warps touch none of these barriers.
//
//     LOAD                    MMA                     SOFTMAX                 EPI
//   item prologue -- quad 0. The loader fills s0, s1 (dO^T) before s2, s3 (Q^T, carrying K and
//   V). On the CTA's first item the empty_* waits here pass on their seeded phase (empty_bar_dq
//   on its prime); on later items they consume the previous item's tail arrives, drawn there:
//     <---------------- empty_bar_epi ----------------+                       |  s0 free
//     +-- full_bar_ring[s0] -->                       |                       |  dO^T q0,q1
//     +-- full_bar_ring[s1] -->                       |                       |  dO^T q2,q3
//     <---- empty_bar_kv -----+                       |                       |  K, V free
//     +-- full_bar_ring[s2] -->                       |                       |  Q^T q0,q1 + K
//     +-- full_bar_ring[s3] -->                       |                       |  Q^T q2,q3 + V
//     +---------------- full_bar_lse ----------------->                       |  LSE(0)
//     |                       +----- full_bar_st ----->                       |  S^T(0)
//     |                       <---------------- empty_bar_dq -----------------+  primed; prev h1
//     |                       +---- full_bar_dpt ----->                       |  dP^T(0)
//     |                       <----- full_bar_pt -----+                       |  P^T(0) stored
//     <---------------- empty_bar_lse ----------------+                       |  LSE(0) read
//     |                       <---- empty_bar_epi ----+                       |  dV, dK TMEM free
//     <- empty_bar_ring[s0] --+                       |                       |  dV(0) over s0
//     <- empty_bar_ring[s1] --+                       |                       |  dV(0) over s1
//     +--------------- full_bar_delta ---------------->                       |  Delta(0)
//   quad j -- dK(j) and dQ(j), then quad j+1's S^T, dP^T and dV; this block runs once per j = 0
//   .. n-2. Loader rows sit where the MMA or softmax consumes them; the loader itself runs up to
//   one quad ahead:
//     |                       <---- full_bar_dst -----+                       |  dS^T(j) stored
//     <--------------- empty_bar_delta ---------------+                       |  Delta(j) read
//     <- empty_bar_ring[s2] --+                       |                       |  dK(j) over s2
//     <- empty_bar_ring[s3] --+                       |                       |  dK(j) over s3
//     |                       +----------------- full_bar_dq ----------------->  dQ(j) h0
//     |                       <---------------- empty_bar_dq -----------------+  h0 read out
//     |                       +----------------- full_bar_dq ----------------->  dQ(j) h1
//     +-- full_bar_ring[s2] -->                       |                       |  Q^T(j+1) q0,q1
//     +-- full_bar_ring[s3] -->                       |                       |  Q^T(j+1) q2,q3
//     +---------------- full_bar_lse ----------------->                       |  LSE(j+1)
//     |                       +----- full_bar_st ----->                       |  S^T(j+1)
//     |                       <---------------- empty_bar_dq -----------------+  h1 read out
//     +-- full_bar_ring[s0] -->                       |                       |  dO^T(j+1) q0,q1
//     +-- full_bar_ring[s1] -->                       |                       |  dO^T(j+1) q2,q3
//     |                       +---- full_bar_dpt ----->                       |  dP^T(j+1)
//     |                       <----- full_bar_pt -----+                       |  P^T(j+1) stored
//     <---------------- empty_bar_lse ----------------+                       |  LSE(j+1) read
//     <- empty_bar_ring[s0] --+                       |                       |  dV(j+1) over s0
//     <- empty_bar_ring[s1] --+                       |                       |  dV(j+1) over s1
//     +--------------- full_bar_delta ---------------->                       |  Delta(j+1)
//     ...                     ...                     ...                     ...  quad j+1, j+2 ..
//   item tail -- quad n-1's dK and dQ (its dV was issued in the last body row, or in the
//   prologue when n = 1). The empty_* arrives at the bottom are what the next item's prologue
//   waits:
//     |                       +----- full_bar_dv ----->                       |  dV complete
//     |                       <---- full_bar_dst -----+                       |  dS^T(n-1) stored
//     <--------------- empty_bar_delta ---------------+                       |  Delta(n-1) read
//     <- empty_bar_ring[s2] --+                       |                       |  dK(n-1) over s2
//     <- empty_bar_ring[s3] --+                       |                       |  dK(n-1) over s3
//     |                       +----- full_bar_dk ----->                       |  dK complete
//     |                       +----------------- full_bar_dq ----------------->  dQ(n-1) h0
//     |                       <---------------- empty_bar_dq -----------------+  h0 read out
//     |                       +----------------- full_bar_dq ----------------->  dQ(n-1) h1
//     <---- empty_bar_kv -----+                       |                       |  next item: K, V
//     |                       <---------------- empty_bar_dq -----------------+  next item: h1 read
//     softmax: merge the dV lane-halves, TMA-store from s0's first 16 KB; then dK (x sm_scale)
//     from its second 16 KB; wait until both stores have read SMEM:
//     <---------------- empty_bar_epi ----------------+                       |  next item: s0
//     |                       <---- empty_bar_epi ----+                       |  next item: dV, dK
//     ...                     ...                     ...                     ...
//   next item: back to the prologue.
//
// Per-warp order within one quad (the rows above interleave these chains):
//   load      wait empty_ring[s0] -> TMA s0 -> wait empty_ring[s1] -> TMA s1 -> wait empty_lse ->
//             bulk LSE -> wait empty_delta -> bulk Delta -> wait empty_ring[s2] -> TMA s2 -> wait
//             empty_ring[s3] -> TMA s3. Quad 0 of an item: wait empty_epi, s0, s1, wait empty_kv,
//             s2 (+K), s3 (+V), then LSE and Delta.
//   mma       wait full_dst -> GEMM 5 (commit empty_ring[s2], [s3]) -> GEMM 4 h0 (commit full_dq)
//             -> wait empty_dq -> GEMM 4 h1 (commit full_dq) -> wait full_ring[s2], [s3] -> GEMM 1
//             (commit full_st) -> wait empty_dq -> wait full_ring[s0], [s1] -> GEMM 2 (commit
//             full_dpt) -> wait full_pt -> GEMM 3 (commit empty_ring[s0], [s1]).
//   softmax   wait full_lse -> wait full_st -> ld S^T -> P^T -> st overlay -> arrive full_pt,
//             empty_lse -> wait full_delta -> wait full_dpt -> ld dP^T -> dS^T -> st overlay +
//             sDST -> arrive full_dst, empty_delta.
//   epilogue  once at kernel start: arrive empty_dq (the prime). Per q-half: wait full_dq -> ld
//             the tile -> arrive empty_dq -> stage and reduce-add.
//
// * empty_bar_dq is waited twice per quad: between the two GEMM 4 halves (same TMEM columns) and
//   before the next GEMM 2 (dP^T shares those columns with dQ: tmem_dq = tmem_dpt).
// * empty_bar_epi has two consumers, so both waits are drawn: the loader (s0 is the dV/dK
//   staging tile) and the MMA (dV(0) zero-inits the tile the softmax warps are still reading).
// * full_bar_pt and full_bar_dst are arrived only after tcgen05.wait::st + fence: the STTM is
//   asynchronous, and an arrive without the wait publishes a tile before it has landed.
// * An item with an empty q-list (local_k2q_num == 0) touches no barrier: every warp skips the body
//   and fetches the next item, so the phase trackers stay aligned.
// * On the CTA's last item the tail's "next item" arrives, the last quad's ring frees
//   (empty_bar_ring[s0..s3]) and its empty_bar_lse / empty_bar_delta have no waiter: the loader
//   has already left its loop. Harmless -- nothing waits on them and the CTA exits.
//
// Teardown (after the last item): the epilogue leader drains its bulk reduce-adds
// (cp.async.bulk.wait_group.read 0) and bar_sync<11>(128) releases the other three warps; then
// epilogue, softmax and MMA (13 warps) meet at bar_sync<10>(416) and the MMA deallocates TMEM, so
// every TMEM read is done before the dealloc. The loader exits on its own.
//
// Named barriers and async-proxy waits (within one role, so not in the swimlane):
//   bar_sync<14>(256)  softmax   three per merge_lane_halves_and_store: after lanes 64-127 stage,
//                                after lanes 0-63 add + fence.proxy.async, after the TMA store
//                                issue; once more before the empty_bar_epi arrive.
//   bar_sync<11>(128)  epilogue  two per hd slice: after staging + fence.proxy.async (the leader
//                                then pushes), after the leader's wait_group_read<1>; once at exit.
//   bar_sync<10>(416)  w0-12     teardown, above.
//   wait_group_read    softmax   <0> by w4 lane 0 before the hand-off: the dV and dK TMA stores
//                                have read s0, so the loader may refill it.
//                      epilogue  <1> by the leader per slice: at most one slice's pushes still
//                                read SMEM, so the other stage buffer may be overwritten; <0> at
//                                exit.
//
// Budgets:
//   TMEM (128 lanes x 512 cols, fp32):
//     cols     holds   written by  also holds
//     0-127    S^T     GEMM 1      P^T bf16x2 overlay at 0-63
//     128-255  dV      GEMM 3      -
//     256-383  dP^T    GEMM 2      dS^T bf16x2 overlay at 256-319; dQ tile (GEMM 4) reuses 256-383
//     384-511  dK      GEMM 5      -
//   Registers (setmaxnreg from the 128/thread base; 4*152 + 8*136 + 3*88 + 24 = 1984 <= 2048):
//     warps    role        regs  does
//     w0-3     epilogue    152   drain dQ tiles from TMEM into dqaccum (cp.reduce.async.bulk)
//     w4-11    softmax     136   form P^T and dS^T from S^T, dP^T (TMEM overlays + sDST)
//     w12      mma          88   issue every tcgen05.mma and commit
//     w13      load         88   TMA gathers into the ring; K, V once per CTA
//     w14      scheduler    88   CLC tile scheduler
//     w15      donor        24   no work; exists only to give its registers to the others
//   dqaccum drain (one dQ tile = 128 q x 128 hd, through one 16 KB SMEM stage):
//     DQ_DTYPE   hd slices per tile   cp.reduce.async.bulk pushes per slice
//     fp32  4 x 32 hd cols    2 x 8 KB (one per q64 block)
//     f16   2 x 64 hd cols    2 x 8 KB (one per q64 block)
//
// Index contract (the padded k2q form FastVideo's invert_indices produces, and the transpose of
// the forward kernel's q2k_idx/q2k_num): k2q_idx int32 [B*H*nb, max_q_blocks] holds, per
// (batch*head, kv64) row, the LOCAL q64 block ids that select this kv block in entries
// [0, k2q_num[row]); entries past the count are never read. The harness inverts its q2k index
// into this form on the host. Timed path = preprocess + main + postprocess;
// TFLOPS = 2.5 * 4 * D * (B*H*nb*topk*64^2) / t (2.5x-selected convention).
// Scheduling knobs: K2Q_WAVES (default on at S>=131K), K2Q_WAVE_SNAKE, K2Q_COUNT_BIN,
// K2Q_TRAVERSAL_SNAKE, DEFENSIVE_WAVES, COOPERATIVE_GRID_SYNC.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <cmath>
#include <chrono>
#include <climits>
#include <vector>
#include <algorithm>
#include <string>
#include "../npy_io.cuh"
#include "block_sparse_bwd_bf16_benchmark.cuh"
#include "../../../../../tests/test_utils.cuh"
#include "../../../../primitives/0_tcgen05_alloc.cuh"
#include "../../../../primitives/1_tcgen05_dealloc.cuh"
#include "../../../../primitives/2_tcgen05_relinquish.cuh"
#include "../../../../primitives/3_tcgen05_mma_f16.cuh"
#include "../../../../primitives/79_tcgen05_mma_ws_f16.cuh"
#include "../../../../primitives/8_tcgen05_mma_idesc.cuh"
#include "../../../../primitives/9_tcgen05_ld.cuh"
#include "../../../../primitives/10_tcgen05_st.cuh"
#include "../../../../primitives/11_tcgen05_commit.cuh"
#include "../../../../primitives/12_tcgen05_wait.cuh"
#include "../../../../primitives/15_tcgen05_fence.cuh"
#include "../../../../primitives/18_tma_load.cuh"
#include "../../../../primitives/22_tma_store.cuh"
#include "../../../../primitives/23_tma_tensormap.cuh"
#include "../../../../primitives/25_tma_async_group.cuh"
#include "../../../../primitives/29_mbarrier_init.cuh"
#include "../../../../primitives/30_mbarrier_arrive.cuh"
#include "../../../../primitives/31_mbarrier_arrive_tx.cuh"
#include "../../../../primitives/33_mbarrier_try_wait.cuh"
#include "../../../../primitives/34_fence_proxy_async.cuh"
#include "../../../../primitives/35_fence_mbarrier_init.cuh"
#include "../../../../primitives/37_bar_sync.cuh"
#include "../../../../primitives/42_smem_desc_blackwell.cuh"
#include "../../../../primitives/44_elect_sync.cuh"
#include "../../../../primitives/46_setmaxnreg.cuh"
#include "../../../../primitives/50_atom_global.cuh"
#include "../../../../primitives/63_cvt_f32_to_f16_bf16.cuh"
#include "../../../../primitives/68_l2cache_policy.cuh"
#include "../../../../primitives/69_griddepcontrol.cuh"
#include "../../../../primitives/76_packed_f32x2.cuh"
#include "../../../../primitives/77_ex2_approx.cuh"
#include "../../../../composites/118_mbarrier_phase_tracking.cuh"
#include "../../../../composites/106_clc_fetch_next_tile.cuh"
#include "../../../../primitives/_warp_prof_noop.cuh"

#ifndef VSA_BHSD
#define VSA_BHSD false  // false: [B*S, H, 128] (repo native); true: FastVideo's [B, H, S, 128]
#endif
// Work scheduling (kernel template parameter SCHED, fixed per build by KERNEL_SCHED):
//   NON_PERSISTENT    1D grid over all items, one work item per CTA, no cross-item hand-off.
//   STATIC_PERSISTENT 1D grid of min(items, #SMs) CTAs, each strides the item list by gridDim.x.
//   CLC               1D grid over all items; w14 runs the clusterlaunchcontrol.try_cancel ring
//                     and each CTA keeps taking the next uncancelled item (HW work stealing).
//   DEFENSIVE         One cooperative <= #SM grid per head. CTAs stride ordered items together
//                     and meet at a grid-wide barrier after every round.
enum class Sched { NON_PERSISTENT, STATIC_PERSISTENT, CLC, DEFENSIVE };
#ifndef KERNEL_SCHED
#define KERNEL_SCHED Sched::CLC
#endif
// Programmatic dependent launch (KERNEL_PDL): preprocess -> main -> postprocess each start while
// the predecessor drains; griddepcontrol.wait sits in front of every read of the predecessor's data.
#ifndef KERNEL_PDL
#define KERNEL_PDL true
#endif

constexpr int BLOCK                   = 64;         // q and kv block size (tokens)
constexpr int KV_TILE                 = BLOCK;      // kv rows owned by one CTA
constexpr int QBLOCKS_PER_QUAD        = 4;          // q64 blocks gathered per step
constexpr int Q_QUAD                  = QBLOCKS_PER_QUAD * BLOCK;  // gathered q tokens per step
constexpr int M_TILE                  = KV_TILE;    // MMA M of S^T, dP^T, dV, dK
[[maybe_unused]] constexpr int K_TILE = Q_QUAD;     // per-step contraction extent of dV, dK
constexpr int HEAD_DIM                = 128;
constexpr int SUB_COLS_BF16           = 64;  // one 128B-swizzle unit
constexpr int SUB_COLS_BYTES          = SUB_COLS_BF16 * (int)sizeof(__nv_bfloat16);  // 128 B
constexpr int KV_SUBTILES             = HEAD_DIM / SUB_COLS_BF16;  // 2 hd-halves of K or V
constexpr int KV_SUB_COLS_BYTES = KV_TILE * SUB_COLS_BYTES;  // 8 KB: K or V, one 64-hd subtile
constexpr int KV_TILE_BYTES     = KV_SUBTILES * KV_SUB_COLS_BYTES;  // 16 KB: K or V, 64 kv x 128 hd
constexpr int Q_BLK_BYTES       = HEAD_DIM * SUB_COLS_BYTES;  // 16 KB: one q64 block of Q^T or dO^T
constexpr int QBLOCKS_PER_SLOT  = 2;                                    // q64 blocks per ring slot
constexpr int Q_RING_SLOT_BYTES = QBLOCKS_PER_SLOT * Q_BLK_BYTES;       // 32 KB
constexpr int SLOTS_PER_QUAD    = QBLOCKS_PER_QUAD / QBLOCKS_PER_SLOT;  // 2, per Q^T and per dO^T
constexpr int PRE_QBLOCKS       = 2;                          // q64 blocks per preprocess CTA
constexpr int PRE_TOKENS        = PRE_QBLOCKS * BLOCK;        // 128 tokens
constexpr int NUM_Q_STAGES      = 2 * SLOTS_PER_QUAD;  // ring: one quad of dO^T and of Q^T
constexpr int DOT_SLOT0         = 0;                   // dO^T slots s0,s1
constexpr int QT_SLOT0          = SLOTS_PER_QUAD;      // Q^T slots s2,s3
constexpr int DST_TILE_BYTES    = KV_TILE * BLOCK * (int)sizeof(__nv_bfloat16);  // [64 kv x 64 q]
constexpr int DST_TILES         = 4;                           // one dS^T tile per q64 block
constexpr int DST_BYTES         = DST_TILES * DST_TILE_BYTES;  // 32 KB sDST, order q0,q2,q1,q3
constexpr int MMA_K               = 16;                     // bf16 tcgen05.mma K per issue
constexpr int K_ATOMS_PER_SUBTILE = SUB_COLS_BF16 / MMA_K;  // 4: GEMM 1/2, hd within a K/V subtile
constexpr int K_ATOMS_PER_QBLOCK  = BLOCK / MMA_K;          // 4: GEMM 3/5, the 64 q of one block
constexpr int K_ATOMS_PER_KV_TILE = KV_TILE / MMA_K;        // 4: GEMM 4, the 64 kv rows
constexpr int BF16X2_COLS_PER_K16 = MMA_K / 2;              // TMEM columns per k16, bf16x2 tile

// dQ drain: the dQ tile leaves TMEM in slices of COLS hd columns, each staged in SMEM and pushed
// as two cp.reduce.async.bulk (one per q64 block); fp32 halves COLS so the push size is fixed.
template <typename DQ_DTYPE = float>
struct DQConfig {
  static constexpr int COLS = sizeof(DQ_DTYPE) == 2 ? 64 : 32;
  // One push = a q block's 64 rows of one slice.
  static constexpr int DQ_ONE_PUSH_BYTES    = BLOCK * COLS * (int)sizeof(DQ_DTYPE);  // 8 KB
  static constexpr int WARP_ELEMS           = 32 * COLS;  // a warp's 32 rows of one slice
  static constexpr int DQ_QBLOCKS_PER_STAGE = 2;          // the tile's two lane-halves
  static constexpr int DQ_STAGE_BYTES       = DQ_QBLOCKS_PER_STAGE * DQ_ONE_PUSH_BYTES;
  static constexpr int DQ_STAGE_BUFFERS     = 2;                 // double-buffered stage
  static constexpr int DQ_BLOCK_ELEMS       = BLOCK * HEAD_DIM;  // accumulator elems per q64 block
};
static_assert(DQConfig<uint16_t>::DQ_STAGE_BYTES == DQConfig<float>::DQ_STAGE_BYTES,
              "f16 and fp32 pushes must match");

constexpr int N_WARPS              = 16;
constexpr int W_EPI0 = 0, W_SOFTMAX0 = 4, W_MMA = 12, W_LOAD = 13, W_SCHED = 14;
constexpr int CLC_STAGES   = 2;
constexpr int CLC_ARRIVALS = 15;  // worker warps 0-13 + the sched fetch

// SMEM (~232 KB): K 16K | V 16K | ring 4x32K | sDST 2x16K (four 8 KB dS^T tiles) |
// dQ stage 2x16K | LSE 1K + Delta 1K (256 f32, 1-stage) | barriers.
// The dK/dV epilogue stages its bf16 tiles in ring slot s0 (dead at item end).
constexpr int NUM_BARS   = 2 * NUM_Q_STAGES + 14 + 2 * CLC_STAGES;
constexpr int SMEM_TOTAL = 2 * KV_TILE_BYTES + NUM_Q_STAGES * Q_RING_SLOT_BYTES + DST_BYTES +
                           DQConfig<>::DQ_STAGE_BUFFERS * DQConfig<>::DQ_STAGE_BYTES +
                           2 * Q_QUAD * (int)sizeof(float) + NUM_BARS * 8 + CLC_STAGES * 16 + 48;

constexpr int ST_COLS        = Q_QUAD / 2;  // S^T / dP^T dual tile: 128 q per lane-half
constexpr int ST_QBLOCK_COLS = BLOCK;       // fp32 columns of one q64 block in the S^T / dP^T tile
constexpr int DV_COLS        = HEAD_DIM;    // dV partial sums
constexpr int DK_COLS        = HEAD_DIM;    // dK partial sums
constexpr int TMEM_TOTAL     = ST_COLS + DV_COLS + ST_COLS + DK_COLS;  // S^T | dV | dP^T | dK
static_assert(ST_COLS == 2 * ST_QBLOCK_COLS, "two q blocks per lane-half");
static_assert(TMEM_TOTAL == 512, "TMEM map must fill exactly 512 columns");

extern __shared__ __align__(1024) uint8_t bwd_smem[];

// Work item = one (batch, head, kv64 block) triple = one row of the padded k2q lists.
struct WorkItem {
  int batch;
  int head;
  int kv_block_id_in_seq;    // kv64 block within the sequence (token row kv_block_id_in_seq * 64)
  const int* local_k2q_idx;  // this row's q-list: k2q_idx + real_item_id * max_q_blocks (size_t
                             // offset: with max_q_blocks == nb, B*H*nb*nb can exceed 2^31)
  int local_k2q_num;         // q-list length in q64 blocks (= k2q_num[real_item_id])
  int num_quads;             // ceil(local_k2q_num / QBLOCKS_PER_QUAD)
};

template <bool PADDED_DEFENSIVE_WAVE = false>
__device__ __forceinline__ WorkItem decode_workitem(int workitem_id,
                                                    const int* __restrict__ workitem_remap,
                                                    const int* __restrict__ k2q_idx,
                                                    const int* __restrict__ k2q_num,
                                                    int max_q_blocks, int num_heads,
                                                    int num_kv_blocks_per_seq) {
  // Every CTA in a defensive grid must execute the same number of barriers. Inactive lanes in
  // the padded final round use a private sentinel and touch no tensor or metadata.
  if constexpr (PADDED_DEFENSIVE_WAVE) {
    if (workitem_id >= num_kv_blocks_per_seq) return {0, 0, 0, nullptr, -1, 0};
  }
  WorkItem it;
  // nullptr: identity order, work id = item id (one launch of every item).
  const int real_item_id = workitem_remap ? workitem_remap[workitem_id] : workitem_id;
  const int batch_head   = real_item_id / num_kv_blocks_per_seq;
  it.batch               = batch_head / num_heads;
  it.head                = batch_head % num_heads;
  it.kv_block_id_in_seq  = real_item_id % num_kv_blocks_per_seq;
  it.local_k2q_idx       = k2q_idx + (size_t)real_item_id * (size_t)max_q_blocks;
  it.local_k2q_num       = k2q_num[real_item_id];
  static_assert(QBLOCKS_PER_QUAD == 4, "num_quads uses >> 2");
  it.num_quads           = (it.local_k2q_num + 3) >> 2;  // ceil(local_k2q_num / QBLOCKS_PER_QUAD)
  return it;
}

// Reverse the actual list before applying the usual tail clamp. This keeps a partial quad valid:
// count=6 descends as [5,4,3,2], [1,0,0,0], so padded positions still repeat the last real entry.
template <bool FIRST_DESCENDING, bool TRAVERSAL_SNAKE>
__device__ __forceinline__ int k2q_block_at(const WorkItem& it, int quad_idx,
                                            int qblock_id_in_quad, int item_ordinal) {
  const bool descending =
      FIRST_DESCENDING ^ (TRAVERSAL_SNAKE && ((item_ordinal & 1) != 0));
  const int position =
      min(QBLOCKS_PER_QUAD * quad_idx + qblock_id_in_quad, it.local_k2q_num - 1);
  return it.local_k2q_idx[descending ? it.local_k2q_num - 1 - position : position];
}

// Element offset of token t of (batch, head) in the activation layout: BHSD = [B, H, S, 128],
// else [B*S, H, 128] (batches concatenated along tokens).
template <bool BHSD>
__device__ __forceinline__ size_t token_offset(int batch, int head, int num_heads, int seqlen,
                                               int t) {
  if constexpr (BHSD)
    return ((size_t)(batch * num_heads + head) * seqlen + t) * HEAD_DIM;
  else
    return ((size_t)(batch * seqlen + t) * num_heads + head) * HEAD_DIM;
}

// Compile-time kernel config (template args, chosen by launch_vsa_bwd_sm100a):
//   DQ_L2_KEEP : true = the dQ reduce-adds carry an L2 evict_last hint (fractional, 0.25) so the
//                repeatedly reduced dqaccum lines stay resident. Picked at runtime when one
//                head's accumulator reaches L2_RESIDENT_DQ_ACCUM_BYTES_PER_HEAD (fp32: S >=
//                262144); both instantiations are compiled because the policy register costs below.
//   SCHED      : work scheduling, see enum Sched. CLC = try_cancel work stealing (w14 sched warp,
//                grid = all items, launched in #SM-sized chunks in the DQ_L2_KEEP regime);
//                STATIC_PERSISTENT = grid-stride loop over the items (w14 idle, grid = #SMs);
//                NON_PERSISTENT = one item per CTA, the loop runs once and the cross-item
//                hand-offs (empty_bar_kv, empty_bar_epi, the primed empty_bar_dq prologue wait)
//                are compiled out. Same pipe.
//   BHSD       : activation layout of q/k/v/o/dO and dq/dk/dv: false = [B*S tokens, H, 128]
//                (repo native), true = FastVideo's [B, H, S, 128] (K/V/dK/dV as 4D tensor maps,
//                token offsets through token_offset<BHSD>). Fixed per build by VSA_BHSD.
//   DQ_DTYPE   : dqaccum element type: float (default) or uint16_t = f16 (DQ_F16 env), halving
//                the accumulator footprint and reduce traffic; DQConfig derives the drain
//                geometry (COLS 32 / 64) from it.
// Kernel arguments (activations bf16 in the BHSD-selected layout unless stated):
//   tmap_k, tmap_v     : K, V as 3D maps [64 hd, B*S tokens, H*2 hd units], box (64, 64, 1)
//                        (BHSD: 4D [64 hd, S, 2 hd units, B*H], box (64, 64, 1, 1)); one CTA
//                        loads its kv64 block once per item (2 TMAs per tile).
//   tmap_qt, tmap_dot  : Q^T, dO^T ([H*128, B*S] bf16, written by the preprocess) as 2D maps,
//                        box [128 hd rows x 64 tokens] = one q64 block per TMA gather.
//   tmap_dk, tmap_dv   : dK, dV output maps, same geometry as K, V (TMA stores, bf16).
//   dqaccum            : DQ_DTYPE [B*H, nb, 64*128] drain-native dQ accumulator; zeroed by the
//                        preprocess, reduce-added here, unscrambled by the postprocess.
//   lse_rows           : fp32 [B*H, S], the forward's LSE in log2 form (M = max + log2(l)).
//   delta_rows         : fp32 [B*H, S], Delta = rowsum(bf16(O) * dO) from the preprocess.
//   k2q_idx, k2q_num   : padded k2q lists (see the index contract above): row real_item_id =
//                        (b*H + h)*nb + kv holds k2q_num[real_item_id] q64 block ids at
//                        k2q_idx[real_item_id * max_q_blocks + i].
//   max_q_blocks       : k2q_idx row stride.
//   workitem_remap     : int32 [B*H*nb], work id -> real item id (the length-binned order the
//                        harness sorts); with CLC chunking the launch passes the chunk's sub-array.
//   num_samples, num_heads, seqlen : B, H, S; nb = num_kv_blocks_per_seq = S/64 is derived.
//   scale_log2         : sm_scale * log2(e), applied to S^T before exp2.
//   sm_scale           : applied to dK in the epilogue (dQ gets it in the postprocess).
template <bool DQ_L2_KEEP = false, Sched SCHED = Sched::CLC, bool BHSD = false,
          typename DQ_DTYPE = float, bool K2Q_FIRST_DESCENDING = false,
          bool K2Q_TRAVERSAL_SNAKE = false, bool USE_COOPERATIVE_GRID_SYNC = false>
__global__ void __cluster_dims__(1, 1, 1) __launch_bounds__(N_WARPS * 32, 1)
    vsa_bwd_main_kernel(const __grid_constant__ CUtensorMap tmap_k,  // box (64,64)
                        const __grid_constant__ CUtensorMap tmap_v,
                        const __grid_constant__ CUtensorMap tmap_qt,  // [H*hd, tok], box (64,128)
                        const __grid_constant__ CUtensorMap tmap_dot,
                        const __grid_constant__ CUtensorMap tmap_dk,
                        const __grid_constant__ CUtensorMap tmap_dv, DQ_DTYPE* __restrict__ dqaccum,
                        const float* __restrict__ lse_rows, const float* __restrict__ delta_rows,
                        const int* __restrict__ k2q_idx, const int* __restrict__ k2q_num,
                        int max_q_blocks, const int* __restrict__ workitem_remap,
                        uint32_t* __restrict__ defensive_counter, int num_samples, int num_heads,
                        int seqlen, float scale_log2, float sm_scale) {
  constexpr bool PERSISTENT = SCHED != Sched::NON_PERSISTENT;
  constexpr bool CLC        = SCHED == Sched::CLC;
  constexpr bool DEFENSIVE  = SCHED == Sched::DEFENSIVE;
  const int num_kv_blocks_per_seq = seqlen / BLOCK;
  [[maybe_unused]] const int defensive_padded_workitems =
      ((num_kv_blocks_per_seq + (int)gridDim.x - 1) / (int)gridDim.x) * (int)gridDim.x;
  using DQ                 = DQConfig<DQ_DTYPE>;
  uint8_t* sK              = bwd_smem;
  uint8_t* sV              = sK + KV_TILE_BYTES;
  uint8_t* sRING           = sV + KV_TILE_BYTES;
  uint8_t* sDST            = sRING + NUM_Q_STAGES * Q_RING_SLOT_BYTES;
  uint8_t* sDQ_STAGE_bytes = sDST + DST_BYTES;
  DQ_DTYPE* sDQ_STAGE[2]   = {reinterpret_cast<DQ_DTYPE*>(sDQ_STAGE_bytes),
                              reinterpret_cast<DQ_DTYPE*>(sDQ_STAGE_bytes + DQ::DQ_STAGE_BYTES)};
  // dV / dK epilogue staging tiles ([64 kv x 128 hd] bf16 each): ring slot s0, which is dead once
  // the item's last dV has been committed.
  __nv_bfloat16* sDV_STAGE = reinterpret_cast<__nv_bfloat16*>(sRING);
  __nv_bfloat16* sDK_STAGE = reinterpret_cast<__nv_bfloat16*>(sRING + KV_TILE_BYTES);
  float* sLSE =
      reinterpret_cast<float*>(sDQ_STAGE_bytes + DQ::DQ_STAGE_BUFFERS * DQ::DQ_STAGE_BYTES);
  float* sDelta = sLSE + Q_QUAD;

  // mbarriers (arrival counts at the init below)
  uint64_t* full_bar_ring   = reinterpret_cast<uint64_t*>(sDelta + Q_QUAD);  // [slot] TMA tx
  uint64_t* empty_bar_ring  = full_bar_ring + NUM_Q_STAGES;   // [slot] the slot's last-GEMM commit
  uint64_t* full_bar_lse    = empty_bar_ring + NUM_Q_STAGES;  // load -> softmax (256 f32)
  uint64_t* empty_bar_lse   = full_bar_lse + 1;               // lane 0 of each softmax warp
  uint64_t* full_bar_delta  = empty_bar_lse + 1;              // load -> softmax (256 f32)
  uint64_t* empty_bar_delta = full_bar_delta + 1;             // lane 0 of each softmax warp
  uint64_t* full_bar_st     = empty_bar_delta + 1;            // commit after the S^T atoms
  uint64_t* full_bar_dpt    = full_bar_st + 1;                // commit after the dP^T atoms
  uint64_t* full_bar_pt     = full_bar_dpt + 1;               // P^T overlay stored (gates dV)
  uint64_t* full_bar_dst    = full_bar_pt + 1;                // dS^T overlay + sDST written
  uint64_t* full_bar_dq     = full_bar_dst + 1;               // commit after a dQ(h) GEMM
  uint64_t* empty_bar_dq    = full_bar_dq + 1;                // epilogue warps drained the dQ tile
  uint64_t* full_bar_dv     = empty_bar_dq + 1;               // all dV accumulation issued
  uint64_t* full_bar_dk     = full_bar_dv + 1;                // last dK issued
  uint64_t* empty_bar_kv    = full_bar_dk + 1;                // item's last dQ committed; K/V free
  uint64_t* empty_bar_epi   = empty_bar_kv + 1;               // epilogue done: s0, dV, dK free
  uint64_t* clc_full        = empty_bar_epi + 1;              // [stage]
  uint64_t* clc_empty       = clc_full + CLC_STAGES;          // [stage]
  uint32_t* clc_response    = reinterpret_cast<uint32_t*>(
      (reinterpret_cast<uintptr_t>(clc_empty + CLC_STAGES) + 15u) & ~uintptr_t(15u));
  uint32_t* tmem_slot = clc_response + CLC_STAGES * 4;

  const int tid = threadIdx.x, warp_id = tid >> 5, lane = tid & 31;
  WpCtx wpc = wp_ctx_init();

  if (warp_id == 0) {
    tcgen05_alloc<1>(smem_ptr_u32(tmem_slot), TMEM_TOTAL);
    tcgen05_relinquish_alloc_permit<1>();
  }
  __syncthreads();
  const uint32_t tmem_base    = *tmem_slot;
  const uint32_t tmem_st      = tmem_base;           // S^T (dual); P^T bf16 overlay on cols 0-63
  const uint32_t tmem_dv      = tmem_st + ST_COLS;   // dV partial sums
  const uint32_t tmem_dpt     = tmem_dv + DV_COLS;   // dP^T (dual); dS^T overlay; dQ tile reuses it
  const uint32_t tmem_dk      = tmem_dpt + ST_COLS;  // dK partial sums
  const uint32_t tmem_pt_bf16 = tmem_st, tmem_dst_bf16 = tmem_dpt, tmem_dq = tmem_dpt;

  if (tid == 0) {
    #pragma unroll
    for (int s = 0; s < NUM_Q_STAGES; ++s) {
      mbarrier_init(smem_ptr_u32(&full_bar_ring[s]), 1);
      mbarrier_init(smem_ptr_u32(&empty_bar_ring[s]), 1);
    }
    mbarrier_init(smem_ptr_u32(full_bar_lse), 1);
    mbarrier_init(smem_ptr_u32(empty_bar_lse), 8);
    mbarrier_init(smem_ptr_u32(full_bar_delta), 1);
    mbarrier_init(smem_ptr_u32(empty_bar_delta), 8);
    mbarrier_init(smem_ptr_u32(full_bar_st), 1);
    mbarrier_init(smem_ptr_u32(full_bar_dpt), 1);
    mbarrier_init(smem_ptr_u32(full_bar_pt), 8);
    mbarrier_init(smem_ptr_u32(full_bar_dst), 8);
    mbarrier_init(smem_ptr_u32(full_bar_dq), 1);
    mbarrier_init(smem_ptr_u32(empty_bar_dq), 4);
    mbarrier_init(smem_ptr_u32(full_bar_dv), 1);
    mbarrier_init(smem_ptr_u32(full_bar_dk), 1);
    if constexpr (PERSISTENT) {
      mbarrier_init(smem_ptr_u32(empty_bar_kv), 1);
      mbarrier_init(smem_ptr_u32(empty_bar_epi), 256);
    }
    if constexpr (CLC) {
      #pragma unroll
      for (int st = 0; st < CLC_STAGES; ++st) {
        mbarrier_init(smem_ptr_u32(&clc_full[st]), 1);
        mbarrier_init(smem_ptr_u32(&clc_empty[st]), CLC_ARRIVALS);
      }
      #pragma unroll
      for (int i = 0; i < CLC_STAGES * 4; ++i) clc_response[i] = 0;
    }
  }
  fence_mbarrier_init_release_cluster();
  __syncthreads();
  // Q^T / dO^T / Delta, the zeroed dqaccum and the k2q metadata come from earlier grids: every
  // warp waits.
  if constexpr (KERNEL_PDL) griddepcontrol_wait();
  [[maybe_unused]] const int total_workitems = num_samples * num_heads * num_kv_blocks_per_seq;

  // Cooperative residency makes the CTA-leader atomic barrier deadlock-free. The optional
  // this_grid().sync() specialization is retained for direct comparison with the cheaper atomic
  // form. CTA barriers keep all 16 warps together before and after the leader's arrival.
  auto defensive_wave_barrier = [&](int workitem_id) {
    if constexpr (DEFENSIVE) {
      if constexpr (USE_COOPERATIVE_GRID_SYNC) {
        cooperative_groups::this_grid().sync();
      } else {
        __syncthreads();
        if (tid == 0) {
          atom_global_add_u32(defensive_counter, 1u);
          const uint32_t target =
              (uint32_t)(workitem_id / (int)gridDim.x + 1) * (uint32_t)gridDim.x;
          while (atom_global_add_u32(defensive_counter, 0u) < target) __nanosleep(64);
        }
        __syncthreads();
      }
    }
  };

  if (warp_id == W_LOAD) {
    setmaxnreg_dec<88>();

    EmptyPhaseTracker<NUM_Q_STAGES> ring_empty_ph;
    EmptyPhaseTracker<1> lse_empty_ph, delta_empty_ph, kv_empty_ph, epi_empty_ph;
    [[maybe_unused]] int clc_stage      = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;

    int workitem_id = (int)blockIdx.x;
    int item_ordinal = 0;
    do {
      wp_marker(wpc, WP_ITEM, workitem_id);
      const WorkItem it = decode_workitem<DEFENSIVE>(workitem_id, workitem_remap, k2q_idx,
                                                     k2q_num, max_q_blocks, num_heads,
                                                     num_kv_blocks_per_seq);

      if (it.local_k2q_num > 0) {
        // Gathered-quad block ids, tail-clamped to the last valid entry.
        auto get_global_qblock_id = [&](int quad_idx, int qblock_id_in_quad) {
          return k2q_block_at<K2Q_FIRST_DESCENDING, K2Q_TRAVERSAL_SNAKE>(
              it, quad_idx, qblock_id_in_quad, item_ordinal);
        };

        // One K or V tile ([64 kv x 128 hd] as two 64-hd subtiles) onto a caller-armed barrier.
        auto load_kv_tile = [&](uint8_t* dst, const CUtensorMap* map, uint64_t* full_bar) {
          #pragma unroll
          for (int s = 0; s < KV_SUBTILES; ++s) {
            if constexpr (BHSD)
              tma_load_4d(smem_ptr_u32(dst + s * KV_SUB_COLS_BYTES), map, smem_ptr_u32(full_bar),
                          0, it.kv_block_id_in_seq * KV_TILE, s, it.batch * num_heads + it.head);
            else
              tma_load_3d(smem_ptr_u32(dst + s * KV_SUB_COLS_BYTES), map, smem_ptr_u32(full_bar),
                          0, it.batch * seqlen + it.kv_block_id_in_seq * KV_TILE,
                          it.head * KV_SUBTILES + s);
          }
        };

        // One quad of Q^T or dO^T: pair 0 = (q0,q1) into the next free slot, pair 1 = (q2,q3)
        // into the one after. Block ids are arbitrary list entries, not contiguous in gmem, so
        // each slot takes two 16 KB TMAs on one barrier. with_kv: pair 0 also carries K, pair 1 V.
        auto load_quad = [&](const CUtensorMap* map, int quad_idx, bool with_kv) {
          if (PERSISTENT && with_kv) {   // K/V SMEM: freed by the MMA commit after the previous
                                         // item's last dQ
            wp_begin(wpc, WP_LOAD_WAIT_EMPTY_KV);
            mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_kv), kv_empty_ph.get_phase());
            kv_empty_ph.advance();
            wp_end(wpc, WP_LOAD_WAIT_EMPTY_KV);
          }
          for (int pair_idx = 0; pair_idx < SLOTS_PER_QUAD; ++pair_idx) {
            const int slot = ring_empty_ph.get_stage();
            wp_begin(wpc, WP_LOAD_WAIT_EMPTY_RING);
            mbarrier_wait_parity_suspend(smem_ptr_u32(&empty_bar_ring[slot]),
                                         ring_empty_ph.get_phase());
            ring_empty_ph.advance();
            wp_end(wpc, WP_LOAD_WAIT_EMPTY_RING);

            const int qblock_in_quad    = QBLOCKS_PER_SLOT * pair_idx;
            const int global_qblock_id0 = get_global_qblock_id(quad_idx, qblock_in_quad);
            const int global_qblock_id1 = get_global_qblock_id(quad_idx, qblock_in_quad + 1);
            wp_begin(wpc, WP_LOAD_ISSUE_RING);
            if (elect_one_sync()) {
              mbarrier_arrive_expect_tx(smem_ptr_u32(&full_bar_ring[slot]),
                                        Q_RING_SLOT_BYTES + (with_kv ? KV_TILE_BYTES : 0));
              tma_load_2d(smem_ptr_u32(sRING + slot * Q_RING_SLOT_BYTES), map,
                          smem_ptr_u32(&full_bar_ring[slot]),
                          it.batch * seqlen + global_qblock_id0 * BLOCK, it.head * HEAD_DIM);
              tma_load_2d(smem_ptr_u32(sRING + slot * Q_RING_SLOT_BYTES + Q_BLK_BYTES), map,
                          smem_ptr_u32(&full_bar_ring[slot]),
                          it.batch * seqlen + global_qblock_id1 * BLOCK, it.head * HEAD_DIM);
              if (with_kv) {
                if (pair_idx == 0) load_kv_tile(sK, &tmap_k, &full_bar_ring[slot]);
                else load_kv_tile(sV, &tmap_v, &full_bar_ring[slot]);
              }
            }
            wp_end(wpc, WP_LOAD_ISSUE_RING);
          }
        };

        auto load_lse_and_delta = [&](int quad_idx) {
          wp_begin(wpc, WP_LOAD_WAIT_EMPTY_LSE);
          mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_lse), lse_empty_ph.get_phase());
          lse_empty_ph.advance();
          wp_end(wpc, WP_LOAD_WAIT_EMPTY_LSE);

          wp_begin(wpc, WP_LOAD_ISSUE_LSE);
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(full_bar_lse), Q_QUAD * 4);
            #pragma unroll
            for (int i = 0; i < QBLOCKS_PER_QUAD; ++i)
              cpasync_bulk_load_mbarrier(
                  smem_ptr_u32(sLSE + i * BLOCK),
                  lse_rows + (size_t)(it.batch * num_heads + it.head) * seqlen +
                      (size_t)get_global_qblock_id(quad_idx, i) * BLOCK,
                  BLOCK * 4,
                  smem_ptr_u32(full_bar_lse));
          }
          wp_end(wpc, WP_LOAD_ISSUE_LSE);

          wp_begin(wpc, WP_LOAD_WAIT_EMPTY_DELTA);
          mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_delta), delta_empty_ph.get_phase());
          delta_empty_ph.advance();
          wp_end(wpc, WP_LOAD_WAIT_EMPTY_DELTA);

          wp_begin(wpc, WP_LOAD_ISSUE_DELTA);
          if (elect_one_sync()) {
            mbarrier_arrive_expect_tx(smem_ptr_u32(full_bar_delta), Q_QUAD * 4);
            #pragma unroll
            for (int i = 0; i < QBLOCKS_PER_QUAD; ++i)
              cpasync_bulk_load_mbarrier(
                  smem_ptr_u32(sDelta + i * BLOCK),
                  delta_rows + (size_t)(it.batch * num_heads + it.head) * seqlen +
                      (size_t)get_global_qblock_id(quad_idx, i) * BLOCK,
                  BLOCK * 4, smem_ptr_u32(full_bar_delta));
          }
          wp_end(wpc, WP_LOAD_ISSUE_DELTA);
        };

        // Item prologue: dO^T first (s0 is also the epilogue staging tile -> wait empty_bar_epi),
        // then Q^T with K/V (wait empty_bar_kv), then quad 0's LSE/Delta.
        if constexpr (PERSISTENT) {
          wp_begin(wpc, WP_LOAD_WAIT_EMPTY_EPI);
          mbarrier_wait_parity_suspend(smem_ptr_u32(empty_bar_epi), epi_empty_ph.get_phase());
          epi_empty_ph.advance();
          wp_end(wpc, WP_LOAD_WAIT_EMPTY_EPI);
        }

        load_quad(&tmap_dot, 0, false);
        load_quad(&tmap_qt, 0, true);
        load_lse_and_delta(0);
        for (int j = 1; j < it.num_quads; ++j) {
          wp_marker(wpc, WP_ITER, j);

          load_quad(&tmap_dot, j, false);
          load_lse_and_delta(j);
          load_quad(&tmap_qt, j, false);
        }
      }

      ++item_ordinal;
      if constexpr (CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        workitem_id = next.valid ? (int)next.n_tile : -1;  // single loop exit: keeps the
      } else if constexpr (DEFENSIVE) {
        defensive_wave_barrier(workitem_id);
        workitem_id += (int)gridDim.x;
        if (workitem_id >= defensive_padded_workitems) workitem_id = -1;
      } else if constexpr (PERSISTENT) {                   // phase trackers in registers
        workitem_id += (int)gridDim.x;
        if (workitem_id >= total_workitems) workitem_id = -1;
      }
    } while (PERSISTENT && workitem_id >= 0);
    wp_flush(wpc);
    return;
  } else if (warp_id == W_MMA) {
    setmaxnreg_dec<88>();

    const uint32_t lead = elect_one_sync() ? 1u : 0u;

    const uint32_t idesc_st_dpt = make_idesc_bf16_f32(M_TILE, 2 * BLOCK, false, true);
    const uint32_t idesc_dv_dk  = make_idesc_bf16_f32(M_TILE, 2 * HEAD_DIM, false, false);
    const uint32_t idesc_dq     = make_idesc_bf16_f32(Q_QUAD / 2, HEAD_DIM, true, true);

    constexpr uint32_t DESC_SBO = 1024, DESC_LBO = 16;
    auto make_smem_desc = [](const uint8_t* smem, uint32_t leading_byte_offset) {
      return build_smem_desc_blackwell(smem_ptr_u32(smem), DESC_SBO, leading_byte_offset,
                                       SmemSwizzleBlackwell::B128);
    };

    const uint64_t desc_k       = make_smem_desc(sK, DESC_LBO);
    const uint64_t desc_v       = make_smem_desc(sV, DESC_LBO);
    const uint64_t desc_ring    = make_smem_desc(sRING, DESC_LBO);
    const uint64_t desc_ring_mn = make_smem_desc(sRING, Q_BLK_BYTES);
    const uint64_t desc_k_mn    = make_smem_desc(sK, KV_SUB_COLS_BYTES);
    const uint64_t desc_dst0    = make_smem_desc(sDST, DST_TILE_BYTES);
    const uint64_t desc_dst1    = make_smem_desc(sDST + 2 * DST_TILE_BYTES, DST_TILE_BYTES);

    // >> 4: the SMEM descriptor address field is in 16-byte units. One k16 step is
    // K16_ROWS_DELTA when K runs along the tile rows (16 rows x 128 B), K16_COLS_DELTA when it
    // runs along a 128 B row (32 B).
    constexpr uint32_t K16_ROWS_DELTA    = (MMA_K * SUB_COLS_BYTES) >> 4;
    constexpr uint64_t K16_COLS_DELTA    = (MMA_K * (int)sizeof(__nv_bfloat16)) >> 4;
    constexpr uint64_t KV_SUB_COLS_DELTA = KV_SUB_COLS_BYTES >> 4;
    constexpr uint64_t RING_DELTA        = Q_RING_SLOT_BYTES >> 4;

    PhaseTracker<1> pt_ph, dst_ph;
    PhaseTracker<1> dq_empty_ph;
    EmptyPhaseTracker<1> epi_empty_ph;
    PhaseTracker<1> ring_full_ph;  // all four slots fill once per quad: one parity per quad

    // GEMM 1 (S^T) and GEMM 2 (dP^T).
    auto gemm12_st_dpt = [&](auto is_st_const) {
      constexpr bool is_st    = decltype(is_st_const)::value;
      const uint32_t tmem_acc = is_st ? tmem_st : tmem_dpt;
      const uint64_t da_base  = is_st ? desc_k : desc_v;
      uint64_t* commit_bar    = is_st ? full_bar_st : full_bar_dpt;
      #pragma unroll
      for (int u = 0; u < SLOTS_PER_QUAD; ++u) {
        const int slot = (is_st ? QT_SLOT0 : DOT_SLOT0) + u;

        wp_begin(wpc, WP_MMA_WAIT_FULL_RING);
        mbarrier_wait_parity(smem_ptr_u32(&full_bar_ring[slot]), ring_full_ph.get_phase());
        wp_end(wpc, WP_MMA_WAIT_FULL_RING);

        wp_begin(wpc, WP_MMA_ISSUE);
        // straight-line issue: address math unguarded (warp-uniform -> URs),
        // the elect predicate rides ON the instruction (no BSSY/BSYNC).
        #pragma unroll
        for (int s = 0; s < KV_SUBTILES; ++s) {
          const uint64_t da = da_base + (uint64_t)s * KV_SUB_COLS_DELTA;
          const uint64_t db = desc_ring_mn + (uint64_t)slot * RING_DELTA
                            + (uint64_t)s * K_ATOMS_PER_SUBTILE * K16_ROWS_DELTA;
          #pragma unroll
          for (int ki = 0; ki < K_ATOMS_PER_SUBTILE; ++ki) {
            const bool enable_d = (s != 0) || (ki != 0);
            tcgen05_mma_ws_f16_ss_1sm_predicated(
                lead, tmem_acc + (uint32_t)(BLOCK * u), da + ki * K16_COLS_DELTA,
                db + (uint64_t)ki * K16_ROWS_DELTA, idesc_st_dpt, enable_d);
          }
        }
        wp_end(wpc, WP_MMA_ISSUE);
      }

      wp_begin(wpc, WP_MMA_COMMIT);
      tcgen05_commit1_lead(lead, smem_ptr_u32(commit_bar));
      wp_end(wpc, WP_MMA_COMMIT);
    };

    // GEMM 3 (dV) and GEMM 5 (dK). Last reader: commits the ring empties.
    auto gemm35_dv_dk = [&](auto is_dv_const, bool first) {
      constexpr bool is_dv       = decltype(is_dv_const)::value;
      const uint32_t tmem_acc    = is_dv ? tmem_dv : tmem_dk;
      const uint32_t tmem_a_base = is_dv ? tmem_pt_bf16 : tmem_dst_bf16;
      #pragma unroll
      for (int p = 0; p < SLOTS_PER_QUAD; ++p) {
        const int slot = (is_dv ? DOT_SLOT0 : QT_SLOT0) + p;
        wp_begin(wpc, WP_MMA_ISSUE);
        const uint64_t db = desc_ring + (uint64_t)slot * RING_DELTA;
        #pragma unroll
        for (int ki = 0; ki < K_ATOMS_PER_QBLOCK; ++ki) {
          const int a = p * K_ATOMS_PER_QBLOCK + ki;  // flat P^T / dS^T atom 0..7
          // Slot p's bf16x2 atoms sit in the first 32 columns of q block p's fp32 columns.
          const uint32_t tmem_a =
              tmem_a_base + (uint32_t)(p * ST_QBLOCK_COLS + ki * BF16X2_COLS_PER_K16);
          const bool accumulate = (!first) || (a != 0);
          tcgen05_mma_ws_f16_ts_1sm_predicated(lead, tmem_acc, tmem_a, db + ki * K16_COLS_DELTA,
                                               idesc_dv_dk, accumulate);
        }
        wp_end(wpc, WP_MMA_ISSUE);

        wp_begin(wpc, WP_MMA_COMMIT);
        tcgen05_commit1_lead(lead, smem_ptr_u32(&empty_bar_ring[slot]));
        wp_end(wpc, WP_MMA_COMMIT);
      }
    };

    // GEMM 4 (dQ), one q-half: the dS tiles of (q0,q2) or (q1,q3) -> [128 q x 128 hd].
    auto gemm4_dq_half = [&](uint64_t desc_dst_h) {
      uint64_t adst = desc_dst_h;
      uint64_t bk   = desc_k_mn;
      wp_begin(wpc, WP_MMA_ISSUE);
      #pragma unroll
      for (int ki = 0; ki < K_ATOMS_PER_KV_TILE; ++ki) {
        tcgen05_mma_f16_ss_lead(lead, tmem_dq, adst, bk, idesc_dq, ki != 0);
        smem_desc_add_lo(adst, K16_ROWS_DELTA);
        smem_desc_add_lo(bk, K16_ROWS_DELTA);
      }
      wp_end(wpc, WP_MMA_ISSUE);

      wp_begin(wpc, WP_MMA_COMMIT);
      tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dq));
      wp_end(wpc, WP_MMA_COMMIT);
    };

    // GEMM 4 (dQ): both q-halves use the same TMEM region, so h1 waits for the h0 drain.
    auto gemm4_dq = [&]() {
      gemm4_dq_half(desc_dst0);

      wp_begin(wpc, WP_MMA_WAIT_EMPTY_DQ);
      mbarrier_wait_parity(smem_ptr_u32(empty_bar_dq), dq_empty_ph.get_phase());
      dq_empty_ph.advance();
      wp_end(wpc, WP_MMA_WAIT_EMPTY_DQ);

      gemm4_dq_half(desc_dst1);
    };

    [[maybe_unused]] int clc_stage      = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;

    int workitem_id = (int)blockIdx.x;
    do {
      wp_marker(wpc, WP_ITEM, workitem_id);
      const WorkItem it = decode_workitem<DEFENSIVE>(workitem_id, workitem_remap, k2q_idx,
                                                     k2q_num, max_q_blocks, num_heads,
                                                     num_kv_blocks_per_seq);

      if (it.local_k2q_num > 0) {
        // Item prologue: quad 0's S^T, dP^T and dV.
        gemm12_st_dpt(/*is_st=*/std::true_type{});

        // dP^T shares TMEM with dQ (tmem_dq = tmem_dpt): wait the previous item's dQ h1 drain.
        if constexpr (PERSISTENT) {
          wp_begin(wpc, WP_MMA_WAIT_EMPTY_DQ);
          mbarrier_wait_parity(smem_ptr_u32(empty_bar_dq), dq_empty_ph.get_phase());
          dq_empty_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_EMPTY_DQ);
        }

        gemm12_st_dpt(/*is_st=*/std::false_type{});
        ring_full_ph.advance();

        // dV's A operand is P^T in TMEM: wait for the softmax warps to publish it.
        wp_begin(wpc, WP_MMA_WAIT_FULL_PT);
        mbarrier_wait_parity(smem_ptr_u32(full_bar_pt), pt_ph.get_phase());
        pt_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_FULL_PT);

        // dV(0) zero-inits tmem_dv: wait until the epilogue has read the previous item's dV/dK.
        if constexpr (PERSISTENT) {
          wp_begin(wpc, WP_MMA_WAIT_EMPTY_EPI);
          mbarrier_wait_parity(smem_ptr_u32(empty_bar_epi), epi_empty_ph.get_phase());
          epi_empty_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_EMPTY_EPI);
        }

        gemm35_dv_dk(/*is_dv=*/std::true_type{}, /*first=*/true);

        // Steady state: quad j's dK and dQ, then quad j+1's S^T, dP^T and dV.
        for (int j = 0; j < it.num_quads - 1; ++j) {
          wp_marker(wpc, WP_ITER, j);

          // dK(j): gated on the dS^T TMEM overlay + sDST tiles.
          wp_begin(wpc, WP_MMA_WAIT_FULL_DST);
          mbarrier_wait_parity(smem_ptr_u32(full_bar_dst), dst_ph.get_phase());
          dst_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_FULL_DST);

          gemm35_dv_dk(/*is_dv=*/std::false_type{}, /*first=*/j == 0);

          // dQ(j): the in-order pipe puts it after dK's dS^T reads.
          gemm4_dq();

          gemm12_st_dpt(/*is_st=*/std::true_type{});

          // dP^T(j+1) shares TMEM with dQ (tmem_dq = tmem_dpt): wait the dQ(j) h1 drain.
          wp_begin(wpc, WP_MMA_WAIT_EMPTY_DQ);
          mbarrier_wait_parity(smem_ptr_u32(empty_bar_dq), dq_empty_ph.get_phase());
          dq_empty_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_EMPTY_DQ);

          gemm12_st_dpt(/*is_st=*/std::false_type{});
          ring_full_ph.advance();

          // dV(j+1)'s A operand is P^T(j+1) in TMEM: wait for the softmax warps to publish it.
          wp_begin(wpc, WP_MMA_WAIT_FULL_PT);
          mbarrier_wait_parity(smem_ptr_u32(full_bar_pt), pt_ph.get_phase());
          pt_ph.advance();
          wp_end(wpc, WP_MMA_WAIT_FULL_PT);

          gemm35_dv_dk(/*is_dv=*/std::true_type{}, /*first=*/false);
        }

        // Last quad: dV is complete -> full_bar_dv; its dK completes dK -> full_bar_dk; then dQ.
        wp_marker(wpc, WP_ITER, it.num_quads - 1);

        wp_begin(wpc, WP_MMA_COMMIT);
        tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dv));
        wp_end(wpc, WP_MMA_COMMIT);

        wp_begin(wpc, WP_MMA_WAIT_FULL_DST);
        mbarrier_wait_parity(smem_ptr_u32(full_bar_dst), dst_ph.get_phase());
        dst_ph.advance();
        wp_end(wpc, WP_MMA_WAIT_FULL_DST);
        gemm35_dv_dk(/*is_dv=*/std::false_type{}, /*first=*/it.num_quads == 1);

        wp_begin(wpc, WP_MMA_COMMIT);
        tcgen05_commit1_lead(lead, smem_ptr_u32(full_bar_dk));
        wp_end(wpc, WP_MMA_COMMIT);

        gemm4_dq();

        if constexpr (PERSISTENT) {
          wp_begin(wpc, WP_MMA_COMMIT);
          tcgen05_commit1_lead(lead, smem_ptr_u32(empty_bar_kv));
          wp_end(wpc, WP_MMA_COMMIT);
        }
      }

      if constexpr (CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        workitem_id = next.valid ? (int)next.n_tile : -1;  // single loop exit: keeps the
      } else if constexpr (DEFENSIVE) {
        defensive_wave_barrier(workitem_id);
        workitem_id += (int)gridDim.x;
        if (workitem_id >= defensive_padded_workitems) workitem_id = -1;
      } else if constexpr (PERSISTENT) {                   // phase trackers in registers
        workitem_id += (int)gridDim.x;
        if (workitem_id >= total_workitems) workitem_id = -1;
      }
    } while (PERSISTENT && workitem_id >= 0);
    wp_flush(wpc);
    bar_sync<10>(416);
    tcgen05_dealloc<1>(tmem_base, TMEM_TOTAL);
    return;
  } else if (warp_id == W_SCHED) {
    setmaxnreg_dec<88>();

    // Scheduler warp: produces try_cancel results into the clc ring; also
    // runs its own consumer fetch to know when to stop.
    if constexpr (CLC) {
      int prod_stage      = 0;
      uint32_t prod_phase = 1;
      int cons_stage      = 0;
      uint32_t cons_phase = 0;
      while (true) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        clc_arrive_expect_tx_cta(smem_ptr_u32(&clc_full[prod_stage]), 16);
        if (lane == 0)
          clc_try_cancel_async(smem_ptr_u32(&clc_response[prod_stage * 4]),
                               smem_ptr_u32(&clc_full[prod_stage]));
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
        ClcTileInfo n = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, cons_stage, cons_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(cons_stage, cons_phase);
        if (!n.valid) break;
      }

      #pragma unroll
      for (int st = 0; st < CLC_STAGES; ++st) {
        if (lane == 0)
          mbarrier_wait_parity_suspend(smem_ptr_u32(&clc_empty[prod_stage]), prod_phase);
        __syncwarp();
        advance_stage_phase<CLC_STAGES>(prod_stage, prod_phase);
      }
    } else if constexpr (DEFENSIVE) {
      for (int workitem_id = (int)blockIdx.x; workitem_id < defensive_padded_workitems;
           workitem_id += (int)gridDim.x)
        defensive_wave_barrier(workitem_id);
    }
    return;
  } else if (warp_id >= W_SOFTMAX0 && warp_id < W_MMA) {
    setmaxnreg_inc<136>();

    PhaseTracker<1> st_ph, dpt_ph, lse_ph, delta_ph, dv_ph, dk_ph;

    constexpr int HALF_COLS = SUB_COLS_BF16;  // fp32 columns per thread: one 128 B swizzle row
    static_assert(ST_COLS == 2 * HALF_COLS && DV_COLS == 2 * HALF_COLS, "two column halves");
    const int softmax_warp_id = warp_id - W_SOFTMAX0;
    const int lane_group      = softmax_warp_id & 3;   // TMEM lanes [32 * lane_group, +32)
    const int col_half        = softmax_warp_id >> 2;  // fp32 columns [64 * col_half, +64)
    const int q_half          = lane_group >> 1;       // lane groups 0,1 -> q-half 0; 2,3 -> 1
    const int kv_row          = (lane_group & 1) * 32 + lane;
    const int qblock_in_quad  = 2 * col_half + q_half;
    // TMEM address offsets (lane << 16 | column) added to a tile base. A warp's bf16x2 overlay
    // (32 columns) starts at the same column as the 64 fp32 columns it loads, so it only ever
    // overwrites columns it has itself consumed: the other warp of the lane group reads the other
    // q block's columns, and no cross-warp ordering is needed between its ld and this st.
    const uint32_t tmem_lane_base     = (uint32_t)(lane_group * 32) << 16;
    const uint32_t tmem_f32_offset    = tmem_lane_base + (uint32_t)(col_half * ST_QBLOCK_COLS);
    const uint32_t tmem_bf16x2_offset = tmem_f32_offset;
    // This block's LSE / Delta columns and its dS tile in sDST (tiles ordered q0,q2,q1,q3).
    const float2* lse2   = reinterpret_cast<const float2*>(sLSE + qblock_in_quad * BLOCK);
    const float2* delta2 = reinterpret_cast<const float2*>(sDelta + qblock_in_quad * BLOCK);
    // 128B-swizzled bf16 SMEM tiles (sDST, sDV_STAGE, sDK_STAGE): each 128 B row is 8 chunks of
    // 16 B; chunk v of row r is stored at chunk slot v ^ (r & 7).
    constexpr int CHUNK_BF16     = 16 / (int)sizeof(__nv_bfloat16);  // 8 bf16 per 16-byte chunk
    constexpr int CHUNKS_PER_ROW = SUB_COLS_BF16 / CHUNK_BF16;        // 8 chunks per 128 B row
    // This thread's row of its dS tile in sDST (tiles ordered q0,q2,q1,q3): 64 bf16 = 128 B,
    // written as 8 chunks of 16 B at swizzled slots v ^ (kv_row & 7).
    __nv_bfloat16* sdst_row =
        reinterpret_cast<__nv_bfloat16*>(sDST + (size_t)(2 * q_half + col_half) * DST_TILE_BYTES) +
        kv_row * SUB_COLS_BF16;

    // Item epilogue for dV or dK: the tile holds two lane-half partial sums (q-half 0 in lanes
    // 0-63, q-half 1 in lanes 64-127). Lanes 64-127 write their half as bf16 into the staging
    // tile, lanes 0-63 add theirs and overwrite, then one thread TMA-stores the tile. dK is
    // multiplied by sm_scale on the way out (dS was formed on the scaled scores); dV is not.
    auto merge_lane_halves_and_store = [&](uint32_t tmem_acc, auto apply_sm_scale_const,
                                           const CUtensorMap* map, __nv_bfloat16* stage_tile,
                                           const WorkItem& it) {
      constexpr bool apply_sm_scale = decltype(apply_sm_scale_const)::value;
      // A thread owns one 128 B row of the staging tile (its kv row, its 64 hd columns) and
      // writes its 64 values as 8 chunks of 16 B at swizzled slots v ^ (kv_row & 7).
      __nv_bfloat16* stage_row =
          stage_tile + (size_t)col_half * (KV_TILE * SUB_COLS_BF16) + kv_row * SUB_COLS_BF16;
      uint32_t acc_regs[HALF_COLS];
      tcgen05_ld_32x32b_x64(tmem_acc + tmem_f32_offset, acc_regs);
      // No tcgen05.wait::ld: the scoreboard makes sure the dependency on the loaded registers is
      // correct.
      // tcgen05_wait_ld();
      tcgen05_fence_before_thread_sync();
      const float2* acc2  = reinterpret_cast<const float2*>(acc_regs);  // column pairs
      const float2 scale2 = f32x2_splat(sm_scale);

      if (q_half == 1) {
        #pragma unroll
        for (int v = 0; v < CHUNKS_PER_ROW; ++v) {
          const float2* a2 = acc2 + v * (CHUNK_BF16 / 2);
          const float2 r0  = apply_sm_scale ? fmul2(a2[0], scale2) : a2[0];
          const float2 r1  = apply_sm_scale ? fmul2(a2[1], scale2) : a2[1];
          const float2 r2  = apply_sm_scale ? fmul2(a2[2], scale2) : a2[2];
          const float2 r3  = apply_sm_scale ? fmul2(a2[3], scale2) : a2[3];
          uint4 packed;
          packed.x = cvt_f32x2_to_bf16x2(r0.x, r0.y);
          packed.y = cvt_f32x2_to_bf16x2(r1.x, r1.y);
          packed.z = cvt_f32x2_to_bf16x2(r2.x, r2.y);
          packed.w = cvt_f32x2_to_bf16x2(r3.x, r3.y);
          *reinterpret_cast<uint4*>(stage_row + (v ^ (kv_row & 7)) * CHUNK_BF16) = packed;
        }
      }
      bar_sync<14>(256);
      if (q_half == 0) {
        #pragma unroll
        for (int v = 0; v < CHUNKS_PER_ROW; ++v) {
          const float2* a2 = acc2 + v * (CHUNK_BF16 / 2);
          uint4* chunk_ptr = reinterpret_cast<uint4*>(stage_row + (v ^ (kv_row & 7)) * CHUNK_BF16);
          const uint4 staged                 = *chunk_ptr;
          const __nv_bfloat162* staged_pairs = reinterpret_cast<const __nv_bfloat162*>(&staged);
          const float2 s0 = __bfloat1622float2(staged_pairs[0]);
          const float2 s1 = __bfloat1622float2(staged_pairs[1]);
          const float2 s2 = __bfloat1622float2(staged_pairs[2]);
          const float2 s3 = __bfloat1622float2(staged_pairs[3]);
          const float2 r0 = apply_sm_scale ? ffma2(a2[0], scale2, s0) : fadd2(a2[0], s0);
          const float2 r1 = apply_sm_scale ? ffma2(a2[1], scale2, s1) : fadd2(a2[1], s1);
          const float2 r2 = apply_sm_scale ? ffma2(a2[2], scale2, s2) : fadd2(a2[2], s2);
          const float2 r3 = apply_sm_scale ? ffma2(a2[3], scale2, s3) : fadd2(a2[3], s3);
          uint4 packed;
          packed.x = cvt_f32x2_to_bf16x2(r0.x, r0.y);
          packed.y = cvt_f32x2_to_bf16x2(r1.x, r1.y);
          packed.z = cvt_f32x2_to_bf16x2(r2.x, r2.y);
          packed.w = cvt_f32x2_to_bf16x2(r3.x, r3.y);
          *chunk_ptr = packed;
        }
      }
      fence_proxy_async_shared();
      bar_sync<14>(256);

      if (softmax_warp_id == 0 && elect_one_sync()) {
        #pragma unroll
        for (int s = 0; s < KV_SUBTILES; ++s) {
          const uint32_t src = smem_ptr_u32(stage_tile + (size_t)s * KV_TILE * SUB_COLS_BF16);
          if constexpr (BHSD)
            tma_store_4d(map, 0, it.kv_block_id_in_seq * KV_TILE, s,
                         it.batch * num_heads + it.head, src);
          else
            tma_store_3d(map, 0, it.batch * seqlen + it.kv_block_id_in_seq * KV_TILE,
                         it.head * KV_SUBTILES + s, src);
        }
        cp_async_bulk_commit_group();
      }
      bar_sync<14>(256);
    };

    [[maybe_unused]] int clc_stage      = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;

    int workitem_id = (int)blockIdx.x;
    do {
      wp_marker(wpc, WP_ITEM, workitem_id);
      const WorkItem it = decode_workitem<DEFENSIVE>(workitem_id, workitem_remap, k2q_idx,
                                                     k2q_num, max_q_blocks, num_heads,
                                                     num_kv_blocks_per_seq);

      if (it.local_k2q_num > 0) {
        for (int j = 0; j < it.num_quads; ++j) {
          wp_marker(wpc, WP_ITER, j);

          // Quad j has local_k2q_num - 4j real q blocks (at most 4); the loader repeats the last
          // one in the padded positions, whose P and dS must be zero.
          const bool qblock_valid = qblock_in_quad < it.local_k2q_num - QBLOCKS_PER_QUAD * j;

          wp_begin(wpc, WP_SM_WAIT_FULL_LSE);
          mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_lse), lse_ph.get_phase());
          lse_ph.advance();
          wp_end(wpc, WP_SM_WAIT_FULL_LSE);

          wp_begin(wpc, WP_SM_WAIT_FULL_ST);
          mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_st), st_ph.get_phase());
          st_ph.advance();
          wp_end(wpc, WP_SM_WAIT_FULL_ST);

          // 1. P^T = exp2(S^T * scale_log2 - LSE): fp32 kept for dS^T, bf16x2 over S^T for dV.
          wp_begin(wpc, WP_SM_STORE_PT);
          uint32_t st_regs[HALF_COLS];
          tcgen05_ld_32x32b_x64(tmem_st + tmem_f32_offset, st_regs);
          // No tcgen05.wait::ld: the scoreboard makes sure the dependency on the loaded registers
          // is correct.
          // tcgen05_wait_ld();
          tcgen05_fence_before_thread_sync();
          // pt_fp32[c] = columns 2c, 2c+1 of this thread's row: S^T before the loop, P^T after it
          // (exp2 written in place; kept in fp32 for the dS^T step).
          float2* pt_fp32 = reinterpret_cast<float2*>(st_regs);
          uint32_t pt_bf16x2[HALF_COLS / 2];
          const float2 scale2 = f32x2_splat(scale_log2);
          // One column pair per step: f32x2 is the widest packed float math on sm_100a, and the
          // fully unrolled float2 LSE / Delta loads are re-merged into LDS.128 by the compiler.
          // Equivalent SASS to a manual unroll by 2 with one float4 load per step (same
          // instruction count and mix; only register numbering differs):
          //   for (int c = 0; c < HALF_COLS / 2; c += 2) {
          //     const float4 lse4 = *reinterpret_cast<const float4*>(lse_cols + 2 * c);
          //     const float2 z0   = ffma2(pt_fp32[c], scale2, make_float2(-lse4.x, -lse4.y));
          //     const float2 z1   = ffma2(pt_fp32[c + 1], scale2, make_float2(-lse4.z, -lse4.w));
          //     ... exp2, pad zero, pt_fp32[c], pt_fp32[c + 1], pt_bf16x2[c], pt_bf16x2[c + 1]
          //   }
          #pragma unroll
          for (int c = 0; c < HALF_COLS / 2; ++c) {
            const float2 z = ffma2(pt_fp32[c], scale2, make_float2(-lse2[c].x, -lse2[c].y));
            float2 p       = make_float2(ex2_approx_f32(z.x), ex2_approx_f32(z.y));
            // Padded blocks zero P with this warp-uniform select, not by branching around the
            // loop: `if (qblock_valid) { loop } else { zero-fill }` miscompiled here twice
            // (STACK 8 -> 120 B, 312 spill ops, dq/dk wrong while dV stayed right; nvcc 13.0,
            // sm_100a). Padding is rare (last quad of an item) so the wasted
            // exp2 work is negligible.
            // TODO: root-cause that miscompile (the SASS pre-zeroes the else path, so the
            // wrong values are unexplained) and then skip the math for padded blocks.
            if (!qblock_valid) p = make_float2(0.f, 0.f);
            pt_fp32[c]   = p;
            pt_bf16x2[c] = cvt_f32x2_to_bf16x2(p.x, p.y);
          }
          tcgen05_st_32x32b_x32(tmem_pt_bf16 + tmem_bf16x2_offset, pt_bf16x2);
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          if (elect_one_sync()) {
            mbarrier_arrive(smem_ptr_u32(full_bar_pt));    // P^T published -> MMA dV
            mbarrier_arrive(smem_ptr_u32(empty_bar_lse));  // LSE consumed -> loader
          }
          wp_end(wpc, WP_SM_STORE_PT);

          wp_begin(wpc, WP_SM_WAIT_FULL_DELTA);
          mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_delta), delta_ph.get_phase());
          delta_ph.advance();
          wp_end(wpc, WP_SM_WAIT_FULL_DELTA);

          wp_begin(wpc, WP_SM_WAIT_FULL_DPT);
          mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dpt), dpt_ph.get_phase());
          dpt_ph.advance();
          wp_end(wpc, WP_SM_WAIT_FULL_DPT);

          // 2. dS^T = P^T * (dP^T - Delta): bf16x2 over dP^T for dK, and into its sDST tile for dQ.
          wp_begin(wpc, WP_SM_STORE_DST);
          uint32_t dpt_regs[HALF_COLS];
          tcgen05_ld_32x32b_x64(tmem_dpt + tmem_f32_offset, dpt_regs);
          // No tcgen05.wait::ld: the scoreboard makes sure the dependency on the loaded registers
          // is correct.
          // tcgen05_wait_ld();
          tcgen05_fence_before_thread_sync();
          const float2* dpt2 = reinterpret_cast<const float2*>(dpt_regs);
          // The bf16x2 dS^T overwrites the first half of the S^T/P^T registers: pt_fp32[c] is
          // dead once packed, and the softmax warp has no room for a third 64-word array.
          uint32_t(&dst_bf16x2)[HALF_COLS / 2] =
              reinterpret_cast<uint32_t(&)[HALF_COLS / 2]>(st_regs);
          #pragma unroll
          for (int c = 0; c < HALF_COLS / 2; ++c) {
            const float2 ds =
                fmul2(pt_fp32[c], fadd2(dpt2[c], make_float2(-delta2[c].x, -delta2[c].y)));
            dst_bf16x2[c] = cvt_f32x2_to_bf16x2(ds.x, ds.y);
          }
          tcgen05_st_32x32b_x32(tmem_dst_bf16 + tmem_bf16x2_offset, dst_bf16x2);
          const uint4* dst_chunks = reinterpret_cast<const uint4*>(dst_bf16x2);
          #pragma unroll
          for (int v = 0; v < CHUNKS_PER_ROW; ++v)
            *reinterpret_cast<uint4*>(sdst_row + (v ^ (kv_row & 7)) * CHUNK_BF16) = dst_chunks[v];
          tcgen05_wait_st();
          tcgen05_fence_before_thread_sync();
          fence_proxy_async_shared();
          if (elect_one_sync()) {
            mbarrier_arrive(smem_ptr_u32(full_bar_dst));     // dS^T published -> MMA dK, dQ
            mbarrier_arrive(smem_ptr_u32(empty_bar_delta));  // Delta consumed -> loader
          }
          wp_end(wpc, WP_SM_STORE_DST);
        }

        // Item epilogue. Ring slot s0 is free now: its dO^T was last read by this item's final dV
        // GEMM, and the loader refills it only after empty_bar_epi (arrived at the end). dV is
        // merged and TMA-stored from its first 16 KB (sDV_STAGE), dK from its second 16 KB
        // (sDK_STAGE): the dV TMA store reads its SMEM asynchronously and is only waited for at
        // the hand-off, so the dK merge must not overwrite those bytes in the meantime.
        wp_begin(wpc, WP_SM_WAIT_FULL_DV);
        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dv), dv_ph.get_phase());
        dv_ph.advance();
        wp_end(wpc, WP_SM_WAIT_FULL_DV);

        wp_begin(wpc, WP_SM_STORE_DV);
        merge_lane_halves_and_store(tmem_dv, /*apply_sm_scale=*/std::false_type{}, &tmap_dv,
                                    sDV_STAGE, it);
        wp_end(wpc, WP_SM_STORE_DV);

        wp_begin(wpc, WP_SM_WAIT_FULL_DK);
        mbarrier_wait_parity_suspend(smem_ptr_u32(full_bar_dk), dk_ph.get_phase());
        dk_ph.advance();
        wp_end(wpc, WP_SM_WAIT_FULL_DK);

        wp_begin(wpc, WP_SM_STORE_DK);
        merge_lane_halves_and_store(tmem_dk, /*apply_sm_scale=*/std::true_type{}, &tmap_dk,
                                    sDK_STAGE, it);

        // Hand-off: both TMA stores done reading SMEM -> s0 free for the loader, dV/dK for the MMA.
        if (softmax_warp_id == 0 && elect_one_sync()) {
          cp_async_bulk_wait_group_read<0>();
        }
        bar_sync<14>(256);
        if constexpr (PERSISTENT) mbarrier_arrive(smem_ptr_u32(empty_bar_epi));
        wp_end(wpc, WP_SM_STORE_DK);
      }

      if constexpr (CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        workitem_id = next.valid ? (int)next.n_tile : -1;  // single loop exit: keeps the
      } else if constexpr (DEFENSIVE) {
        defensive_wave_barrier(workitem_id);
        workitem_id += (int)gridDim.x;
        if (workitem_id >= defensive_padded_workitems) workitem_id = -1;
      } else if constexpr (PERSISTENT) {                   // phase trackers in registers
        workitem_id += (int)gridDim.x;
        if (workitem_id >= total_workitems) workitem_id = -1;
      }
    } while (PERSISTENT && workitem_id >= 0);
    wp_flush(wpc);
    bar_sync<10>(416);
    return;
  } else if (warp_id < W_SOFTMAX0) {
    setmaxnreg_inc<152>();

    // Epilogue warp w owns TMEM lanes [32w, +32) of the dQ tile: 32 q rows x 128 hd columns, one
    // row per thread. Warps 0,1 hold the q block in lanes 0-63, warps 2,3 the one in lanes 64-127.
    // One thread (warp 0, lane 0) issues the bulk reduce-adds.
    const int epi_warp_id      = warp_id - W_EPI0;
    const uint32_t tmem_lane_base = (uint32_t)(epi_warp_id * 32) << 16;
    const bool is_leader          = epi_warp_id == 0 && lane == 0;
    uint64_t dqaccum_l2_policy    = 0;
    if constexpr (DQ_L2_KEEP) {
      dqaccum_l2_policy = make_l2cache_policy_fractional_evict_last_unchanged(0.25f);
    }

    // One 8 KB push: a q block's 64 rows of one hd slice, from the stage into its dqaccum rows.
    auto reduce_add_push = [&](DQ_DTYPE* dst, uint32_t src_smem) {
      if constexpr (sizeof(DQ_DTYPE) == 4) {
        if constexpr (DQ_L2_KEEP) {
          cpasync_reduce_bulk_add_f32_l2hint(dst, src_smem, DQ::DQ_ONE_PUSH_BYTES,
                                             dqaccum_l2_policy);
        } else {
          cpasync_reduce_bulk_add_f32(dst, src_smem, DQ::DQ_ONE_PUSH_BYTES);
        }
      } else {
        if constexpr (DQ_L2_KEEP) {
          cpasync_reduce_bulk_add_f16_l2hint(dst, src_smem, DQ::DQ_ONE_PUSH_BYTES,
                                             dqaccum_l2_policy);
        } else {
          cpasync_reduce_bulk_add_f16(dst, src_smem, DQ::DQ_ONE_PUSH_BYTES);
        }
      }
    };

    PhaseTracker<1> dq_full_ph;
    // Persistent: prime empty_bar_dq once, the MMA's item prologue waits on it before the first
    // dP^T.
    if constexpr (PERSISTENT) {
      if (elect_one_sync()) mbarrier_arrive(smem_ptr_u32(empty_bar_dq));
    }
    [[maybe_unused]] int clc_stage      = 0;
    [[maybe_unused]] uint32_t clc_phase = 0;

    int workitem_id = (int)blockIdx.x;
    int item_ordinal = 0;
    do {
      wp_marker(wpc, WP_ITEM, workitem_id);
      const WorkItem it = decode_workitem<DEFENSIVE>(workitem_id, workitem_remap, k2q_idx,
                                                     k2q_num, max_q_blocks, num_heads,
                                                     num_kv_blocks_per_seq);

      if (it.local_k2q_num > 0) {
        DQ_DTYPE* dqaccum_head = dqaccum + (size_t)(it.batch * num_heads + it.head) *
                                               num_kv_blocks_per_seq * DQ::DQ_BLOCK_ELEMS;

        for (int j = 0; j < it.num_quads; ++j) {
          wp_marker(wpc, WP_ITER, j);

          #pragma unroll 1
          for (int h = 0; h < 2; ++h) {
            // dqaccum rows of the tile's two q blocks: list positions 4j+h (lanes 0-63) and
            // 4j+h+2 (lanes 64-127). Padded positions repeat the last block and add zeros.
            const int qblock_lo =
                k2q_block_at<K2Q_FIRST_DESCENDING, K2Q_TRAVERSAL_SNAKE>(it, j, h,
                                                                        item_ordinal);
            const int qblock_hi =
                k2q_block_at<K2Q_FIRST_DESCENDING, K2Q_TRAVERSAL_SNAKE>(it, j, h + 2,
                                                                        item_ordinal);
            DQ_DTYPE* dqaccum_lo = dqaccum_head + (size_t)qblock_lo * DQ::DQ_BLOCK_ELEMS;
            DQ_DTYPE* dqaccum_hi = dqaccum_head + (size_t)qblock_hi * DQ::DQ_BLOCK_ELEMS;

            wp_begin(wpc, WP_EPI_WAIT_FULL_DQ);
            mbarrier_wait_parity(smem_ptr_u32(full_bar_dq), dq_full_ph.get_phase());
            dq_full_ph.advance();
            wp_end(wpc, WP_EPI_WAIT_FULL_DQ);

            // This thread's q row: 128 hd fp32 columns, loaded 64 at a time (a single x128 load
            // needs 146 registers at that instruction and does not fit the kernel's 128-register
            // compile cap; setmaxnreg only raises the budget at run time). The whole tile is read
            // out before any staging so empty_bar_dq fires early and the MMA's next dQ GEMM
            // overlaps the drain below (loading slice by slice would hold the tile longer).
            wp_begin(wpc, WP_EPI_LOAD_DQ);
            uint32_t dq_regs[HEAD_DIM];
            #pragma unroll
            for (int c = 0; c < HEAD_DIM / 64; ++c) {
              tcgen05_ld_32x32b_x64(tmem_dq + tmem_lane_base + (uint32_t)(c * 64),
                                    reinterpret_cast<uint32_t(&)[64]>(dq_regs[c * 64]));
            }
            // The arrive hands tmem_dq back to the MMA; only the loads' register consumers are
            // scoreboarded, so wait::ld keeps it behind them.
            tcgen05_wait_ld();
            tcgen05_fence_before_thread_sync();
            if (elect_one_sync()) {
              mbarrier_arrive(smem_ptr_u32(empty_bar_dq));  // tile read -> MMA may reuse it
            }
            wp_end(wpc, WP_EPI_LOAD_DQ);

            // Input: the 128 q x 128 hd dQ tile, this thread holding one q row in dq_regs. It is
            // drained in slices of DQ::COLS hd columns through the two 16 KB stages. A stage holds
            // the 4 warps back to back; within a warp's 32 rows x COLS slice the layout is
            // transposed, [4-column group][lane]: group v4 of all 32 lanes is one contiguous
            // 32 x float4 block at + v4 * 32 * 4, so each lane owns column slot lane * 4 of every
            // block. Warps 0,1 (lanes 0-63) fill the first 8 KB and warps 2,3 the second, one bulk
            // reduce-add each; dqaccum keeps this order and the postprocess kernel undoes it.
            wp_begin(wpc, WP_EPI_STORE_DQ);
            #pragma unroll
            for (int hd_slice = 0; hd_slice < HEAD_DIM / DQ::COLS; ++hd_slice) {
              const int stage_buf = hd_slice & 1;  // slice s+1 stages while s's pushes still read
              const float4* dq_row4 =
                  reinterpret_cast<const float4*>(dq_regs + hd_slice * DQ::COLS);
              DQ_DTYPE* stage_row =
                  sDQ_STAGE[stage_buf] + epi_warp_id * DQ::WARP_ELEMS + lane * 4;
              #pragma unroll
              for (int v4 = 0; v4 < DQ::COLS / 4; ++v4) {
                const float4 v = dq_row4[v4];
                if constexpr (sizeof(DQ_DTYPE) == 2) {
                  uint2 packed;
                  packed.x = cvt_f32x2_to_f16x2(v.x, v.y);
                  packed.y = cvt_f32x2_to_f16x2(v.z, v.w);
                  *reinterpret_cast<uint2*>(stage_row + v4 * 32 * 4) = packed;
                } else {
                  *reinterpret_cast<float4*>(stage_row + v4 * 32 * 4) = v;
                }
              }
              fence_proxy_async_shared();
              bar_sync<11>(128);
              if (is_leader) {
                const size_t slice_offset = (size_t)hd_slice * BLOCK * DQ::COLS;
                const uint32_t stage_lo   = smem_ptr_u32(sDQ_STAGE[stage_buf]);
                // Two pushes: the tile's two q blocks have non-adjacent dqaccum regions.
                reduce_add_push(dqaccum_lo + slice_offset, stage_lo);
                reduce_add_push(dqaccum_hi + slice_offset, stage_lo + DQ::DQ_ONE_PUSH_BYTES);
                cp_async_bulk_commit_group();
                cp_async_bulk_wait_group_read<1>();  // the other stage buffer is free again
              }
              bar_sync<11>(128);
            }
            wp_end(wpc, WP_EPI_STORE_DQ);
          }
        }
      }

      ++item_ordinal;
      if constexpr (CLC) {
        ClcTileInfo next = clc_fetch_next_tile<1, 1, ClcRasterOrder::AlongN, 1, true>(
            clc_full, clc_empty, clc_response, clc_stage, clc_phase, elect_one_sync());
        clc_fetch_next_tile_advance<CLC_STAGES>(clc_stage, clc_phase);
        workitem_id = next.valid ? (int)next.n_tile : -1;  // single loop exit: keeps the
      } else if constexpr (DEFENSIVE) {
        defensive_wave_barrier(workitem_id);
        workitem_id += (int)gridDim.x;
        if (workitem_id >= defensive_padded_workitems) workitem_id = -1;
      } else if constexpr (PERSISTENT) {                   // phase trackers in registers
        workitem_id += (int)gridDim.x;
        if (workitem_id >= total_workitems) workitem_id = -1;
      }
    } while (PERSISTENT && workitem_id >= 0);
    if (is_leader) {
      cp_async_bulk_wait_group_read<0>();
    }
    bar_sync<11>(128);
    wp_flush(wpc);
    bar_sync<10>(416);
    // CTA done: the postprocess may launch (its wait still covers completion).
    if constexpr (KERNEL_PDL) {
      if (is_leader) griddepcontrol_launch_dependents();
    }
    return;
  } else {  // idle warp: no work, donates its registers
    setmaxnreg_dec<24>();
    if constexpr (DEFENSIVE) {
      for (int workitem_id = (int)blockIdx.x; workitem_id < defensive_padded_workitems;
           workitem_id += (int)gridDim.x)
        defensive_wave_barrier(workitem_id);
    }
    return;
  }
}

// Preprocess: one CTA per (128-token block, head), 256 threads. Zeroes the
// block's dqaccum slice, computes Delta = rowsum(bf16(O) * dO), and writes
// the TRANSPOSED Q^T / dO^T tensors ([H*hd, tokens]) the ws TS gathers need.
template <bool BHSD = false, typename DQ_DTYPE = float>
__global__ void __launch_bounds__(256, 1)
    vsa_bwd_preprocess_kernel(const __nv_bfloat16* __restrict__ q,
                              const __nv_bfloat16* __restrict__ o,
                              const __nv_bfloat16* __restrict__ dout,
                              float* __restrict__ delta_rows, DQ_DTYPE* __restrict__ dqaccum,
                              __nv_bfloat16* __restrict__ qt, __nv_bfloat16* __restrict__ dot,
                              int num_samples, int num_heads, int seqlen) {
  // The previous postprocess still reads dqaccum: wait before zeroing it.
  if constexpr (KERNEL_PDL) griddepcontrol_wait();
  __shared__ __align__(128) __nv_bfloat16 tile[PRE_TOKENS][SUB_COLS_BF16 + 4];
  const int token_block_id  = (int)blockIdx.x;
  const int batch_head      = (int)blockIdx.y;
  const int batch = batch_head / num_heads, head = batch_head % num_heads;
  const size_t total_tokens = (size_t)num_samples * seqlen;  // Q^T / dO^T row pitch
  const int token_begin     = token_block_id * PRE_TOKENS;
  uint4* dqaccum_zero_destination =
      reinterpret_cast<uint4*>(dqaccum + (size_t)batch_head * seqlen * HEAD_DIM +
                               (size_t)token_block_id * PRE_TOKENS * HEAD_DIM);
  const uint4 zero_uint4       = make_uint4(0u, 0u, 0u, 0u);
  constexpr int DQ_ZERO_CHUNKS = (PRE_TOKENS * HEAD_DIM * (int)sizeof(DQ_DTYPE) / 16) / 256;
  #pragma unroll
  for (int chunk = 0; chunk < DQ_ZERO_CHUNKS; ++chunk)
    dqaccum_zero_destination[chunk * 256 + threadIdx.x] = zero_uint4;
  // Q^T: two 128x64 tiles. Each lane vector-loads four contiguous dimensions from a token row;
  // after the shared transpose it vector-stores four contiguous tokens. The 68-half pitch
  // preserves 8B alignment while breaking column bank aliasing.
  const int row_base        = (int)threadIdx.x >> 4;        // this lane's first tile row
  const int dimension_begin = ((int)threadIdx.x & 15) * 4;  // first of the 4 dims it loads
  #pragma unroll
  for (int dimension_block = 0; dimension_block < HEAD_DIM; dimension_block += 64) {
    #pragma unroll
    for (int row_pass = 0; row_pass < 8; ++row_pass) {
      const int row = row_base + row_pass * 16;
      *reinterpret_cast<uint2*>(&tile[row][dimension_begin]) = *reinterpret_cast<const uint2*>(
          q + token_offset<BHSD>(batch, head, num_heads, seqlen, token_begin + row) +
          dimension_block + dimension_begin);
    }
    __syncthreads();
    const int dimension         = (int)threadIdx.x >> 2;       // tile column this lane stores
    const int token_group_begin = ((int)threadIdx.x & 3) * 4;  // first of its 4 tokens
    #pragma unroll
    for (int row_pass = 0; row_pass < 8; ++row_pass) {
      const int row = token_group_begin + row_pass * 16;
      uint2 packed_tokens;
      uint16_t* packed_halves = reinterpret_cast<uint16_t*>(&packed_tokens);
      #pragma unroll
      for (int token_in_group = 0; token_in_group < 4; ++token_in_group)
        packed_halves[token_in_group] =
            *reinterpret_cast<const uint16_t*>(&tile[row + token_in_group][dimension]);
      __nv_bfloat16* transposed_row =
          qt + ((size_t)head * HEAD_DIM + dimension_block + dimension) * total_tokens;
      *reinterpret_cast<uint2*>(transposed_row + (size_t)batch * seqlen + token_begin + row) =
          packed_tokens;
    }
    __syncthreads();
  }

  // dqaccum and Q^T are out: the main grid may set up while dO^T / Delta form.
  if constexpr (KERNEL_PDL) griddepcontrol_launch_dependents();

  // dO^T, the same choreography with Delta fused in so every dO vector is read only once.
  // A 16-lane subgroup owns one token row; each lane accumulates four dims from each 64-dim
  // tile, then the subgroup reduces the full 128-dim dot.
  float delta_accumulator[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  #pragma unroll
  for (int dimension_block = 0; dimension_block < HEAD_DIM; dimension_block += 64) {
    #pragma unroll
    for (int row_pass = 0; row_pass < 8; ++row_pass) {
      const int row                     = row_base + row_pass * 16;
      const size_t token_element_offset =
          token_offset<BHSD>(batch, head, num_heads, seqlen, token_begin + row) +
          dimension_block + dimension_begin;
      const uint2 dout_vector = *reinterpret_cast<const uint2*>(dout + token_element_offset);
      *reinterpret_cast<uint2*>(&tile[row][dimension_begin]) = dout_vector;
      const uint2 o_vector             = *reinterpret_cast<const uint2*>(o + token_element_offset);
      const __nv_bfloat162* dout_pairs = reinterpret_cast<const __nv_bfloat162*>(&dout_vector);
      const __nv_bfloat162* o_pairs    = reinterpret_cast<const __nv_bfloat162*>(&o_vector);
      #pragma unroll
      for (int pair = 0; pair < 2; ++pair) {
        const float2 dout_pair_as_float = __bfloat1622float2(dout_pairs[pair]);
        const float2 o_pair_as_float    = __bfloat1622float2(o_pairs[pair]);
        delta_accumulator[row_pass] += o_pair_as_float.x * dout_pair_as_float.x +
                                       o_pair_as_float.y * dout_pair_as_float.y;
      }
    }
    __syncthreads();
    const int dimension         = (int)threadIdx.x >> 2;
    const int token_group_begin = ((int)threadIdx.x & 3) * 4;
    #pragma unroll
    for (int row_pass = 0; row_pass < 8; ++row_pass) {
      const int row = token_group_begin + row_pass * 16;
      uint2 packed_tokens;
      uint16_t* packed_halves = reinterpret_cast<uint16_t*>(&packed_tokens);
      #pragma unroll
      for (int token_in_group = 0; token_in_group < 4; ++token_in_group)
        packed_halves[token_in_group] =
            *reinterpret_cast<const uint16_t*>(&tile[row + token_in_group][dimension]);
      __nv_bfloat16* transposed_row =
          dot + ((size_t)head * HEAD_DIM + dimension_block + dimension) * total_tokens;
      *reinterpret_cast<uint2*>(transposed_row + (size_t)batch * seqlen + token_begin + row) =
          packed_tokens;
    }
    __syncthreads();
  }
  #pragma unroll
  for (int row_pass = 0; row_pass < 8; ++row_pass) {
    float delta_sum = delta_accumulator[row_pass];
    #pragma unroll
    for (int shuffle_offset = 8; shuffle_offset > 0; shuffle_offset >>= 1)
      delta_sum += __shfl_down_sync(0xffffffffu, delta_sum, shuffle_offset, 16);
    if (((int)threadIdx.x & 15) == 0)
      delta_rows[(size_t)batch_head * seqlen + token_begin + row_base + row_pass * 16] = delta_sum;
  }
}

// Postprocess: one CTA per (q64 block, head), 128 threads. Unscrambles the
// drain-native dqaccum region (2 hd slices x [16 half4-groups x 64 rows x 4]),
// applies sm_scale, stores bf16 dQ.
template <bool BHSD = false, typename DQ_DTYPE = float>
__global__ void __launch_bounds__(128, 1)
    vsa_bwd_postprocess_kernel(const DQ_DTYPE* __restrict__ dqaccum, __nv_bfloat16* __restrict__ dq,
                               int num_heads, int seqlen, float sm_scale) {
  const int q_block_id = (int)blockIdx.x;
  const int batch_head = (int)blockIdx.y;
  const int batch = batch_head / num_heads, head = batch_head % num_heads;
  const DQ_DTYPE* dqaccum_block = dqaccum + ((size_t)batch_head * (seqlen / BLOCK) + q_block_id) *
                                                DQConfig<DQ_DTYPE>::DQ_BLOCK_ELEMS;
  const int row            = (int)threadIdx.x & 63;
  const int dimension_half = (int)threadIdx.x >> 6;  // hd half 0/1
  uint32_t dq_packed_bf16_pairs[32];                 // this thread's 64 dims as bf16 pairs
  if constexpr (KERNEL_PDL) griddepcontrol_wait();   // the main grid's pushes are complete
  #pragma unroll
  for (int dimension_in_half = 0; dimension_in_half < 64; dimension_in_half += 4) {
    const int dimension = dimension_half * 64 + dimension_in_half;
    // drain-native layout: hd slice (dimension / 64) | row group (row / 32) |
    // half4 group ((dimension % 64) / 4)
    const DQ_DTYPE* dqaccum_element =
        dqaccum_block +
        (dimension / DQConfig<DQ_DTYPE>::COLS) * (BLOCK * DQConfig<DQ_DTYPE>::COLS) +
        ((row & 32) >> 5) * DQConfig<DQ_DTYPE>::WARP_ELEMS +
        ((dimension % DQConfig<DQ_DTYPE>::COLS) >> 2) * 128 + (row & 31) * 4;
    float2 dq_pair_low, dq_pair_high;
    if constexpr (sizeof(DQ_DTYPE) == 2) {
      const uint2 dq_four_halves = *reinterpret_cast<const uint2*>(dqaccum_element);
      dq_pair_low  = __half22float2(*reinterpret_cast<const __half2*>(&dq_four_halves.x));
      dq_pair_high = __half22float2(*reinterpret_cast<const __half2*>(&dq_four_halves.y));
    } else {
      const float4 dq_four_floats = *reinterpret_cast<const float4*>(dqaccum_element);
      dq_pair_low  = make_float2(dq_four_floats.x, dq_four_floats.y);
      dq_pair_high = make_float2(dq_four_floats.z, dq_four_floats.w);
    }
    dq_packed_bf16_pairs[dimension_in_half / 2 + 0] =
        cvt_f32x2_to_bf16x2(dq_pair_low.x * sm_scale, dq_pair_low.y * sm_scale);
    dq_packed_bf16_pairs[dimension_in_half / 2 + 1] =
        cvt_f32x2_to_bf16x2(dq_pair_high.x * sm_scale, dq_pair_high.y * sm_scale);
  }
  // dqaccum read out: the next preprocess may launch (it waits before re-zeroing).
  if constexpr (KERNEL_PDL) griddepcontrol_launch_dependents();
  const int token       = q_block_id * BLOCK + row;
  uint4* dq_destination = reinterpret_cast<uint4*>(
      dq + token_offset<BHSD>(batch, head, num_heads, seqlen, token) + dimension_half * 64);
  const uint4* dq_packed_uint4 = reinterpret_cast<const uint4*>(dq_packed_bf16_pairs);
  #pragma unroll
  for (int vector_index = 0; vector_index < 8; ++vector_index)
    dq_destination[vector_index] = dq_packed_uint4[vector_index];
}

// ---------------------------------------------------------------------------
// Host launchers (stream-chained: pre -> main -> post).
// ---------------------------------------------------------------------------

struct VsaBwdArgs {
  const __nv_bfloat16 *q, *k, *v, *dout;  // bf16, VSA_BHSD ? [B, H, S, 128] : [B*S, H, 128]
  void* dqaccum;                       // drain-native DQ_DTYPE scratch (pre zeroes it)
  __nv_bfloat16 *dk, *dv, *dq;         // bf16 outputs, same layout as the inputs
  __nv_bfloat16 *qt, *dot;             // [H*128, B*S] bf16 scratch (pre writes them)
  const float *lse_rows, *delta_rows;  // [B*H, S]
  const int* k2q_idx;                  // [B*H*nb64, max_q_blocks] plain q64 ids, padded rows
  const int* k2q_num;                  // [B*H*nb64] entries valid per row
  const int* workitem_remap;           // work id -> real item id ((batch*H + head)*nb64 + kv64)
  uint32_t* defensive_counter;         // [B*H], one monotonic grid-barrier counter per head
  bool cta_waves;                      // ordered non-persistent grids of <= one CTA per SM
  bool alternate_wave_direction;      // alternate ascending/descending k2q traversal by wave
  bool k2q_traversal_snake;            // alternate traversal direction by persistent CTA round
  bool cooperative_grid_sync;          // defensive: this_grid().sync() instead of atomic barrier
  int num_samples, num_heads, seqlen, num_kv_blocks_per_seq;  // kv64 blocks per sequence (S/64)
  int max_q_blocks;                    // k2q_idx row stride
  float sm_scale;
};

// K, V, dK, dV tensor maps, one 64-token x 64-hd box per TMA (two per tile, for the K/V loads
// and the dK/dV stores):
//   BSHD: 3D [SUB_COLS_BF16 hd, B*S tokens, H*2 hd units], strides {H*128*2, 128} bytes.
//   BHSD: 4D [SUB_COLS_BF16 hd, S tokens, 2 hd units, B*H], strides {128*2, 128, S*128*2} bytes.
inline cudaError_t make_tma_kv_units(CUtensorMap* map, const __nv_bfloat16* ptr, int B, int H,
                                     int S) {
  CUresult r;
  if (VSA_BHSD) {
    uint64_t gd[4] = {(uint64_t)SUB_COLS_BF16, (uint64_t)S, (uint64_t)KV_SUBTILES,
                      (uint64_t)B * H};
    uint64_t gs[3] = {(uint64_t)HEAD_DIM * 2, (uint64_t)SUB_COLS_BYTES,
                      (uint64_t)S * HEAD_DIM * 2};
    uint32_t bd[4] = {(uint32_t)SUB_COLS_BF16, (uint32_t)BLOCK, 1u, 1u};
    uint32_t es[4] = {1u, 1u, 1u, 1u};
    r = cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
                               const_cast<__nv_bfloat16*>(ptr), gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  } else {
    uint64_t gd[3] = {(uint64_t)SUB_COLS_BF16, (uint64_t)B * S, (uint64_t)H * KV_SUBTILES};
    uint64_t gs[2] = {(uint64_t)H * HEAD_DIM * 2, (uint64_t)SUB_COLS_BYTES};
    uint32_t bd[3] = {(uint32_t)SUB_COLS_BF16, (uint32_t)BLOCK, 1u};
    uint32_t es[3] = {1u, 1u, 1u};
    r = cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
                               const_cast<__nv_bfloat16*>(ptr), gd, gs, bd, es,
                               CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  }
  return (r == CUDA_SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
}

// One head's dQ accumulator (S x HEAD_DIM x sizeof) beyond which chunked DQ_L2_KEEP launches pay.
constexpr size_t L2_RESIDENT_DQ_ACCUM_BYTES_PER_HEAD = size_t{128} << 20;

template <Sched SCHED = Sched::CLC, typename DQ_DTYPE = float>
inline cudaError_t launch_vsa_bwd_sm100a(const VsaBwdArgs& a, cudaStream_t stream) {
  const int B = a.num_samples, H = a.num_heads, S = a.seqlen;
  const long n_tokens = (long)B * S;  // samples concatenated along the token axis

  static CUtensorMap tk_, tv_, tqt_, tdot_, tdk_, tdv_;
  static const void* cached_q = nullptr;
  static int cached_B = 0, cached_S = 0, cached_H = 0;
  if (cached_q != (const void*)a.q || cached_B != B || cached_S != S || cached_H != H) {
    if (make_tma_kv_units(&tk_, a.k, B, H, S) != cudaSuccess) return cudaErrorInvalidValue;
    if (make_tma_kv_units(&tv_, a.v, B, H, S) != cudaSuccess) return cudaErrorInvalidValue;
    // Q^T, dO^T ([H*hd rows, tokens cols], token contiguous): box [hd rows, BLOCK cols] = one q64
    // block (one TMA per block).
    if (make_tma_2d_tiled(&tqt_, a.qt, H * HEAD_DIM, (int)n_tokens, HEAD_DIM, BLOCK, 2,
                          CU_TENSOR_MAP_DATA_TYPE_BFLOAT16) != cudaSuccess)
      return cudaErrorInvalidValue;
    if (make_tma_2d_tiled(&tdot_, a.dot, H * HEAD_DIM, (int)n_tokens, HEAD_DIM, BLOCK, 2,
                          CU_TENSOR_MAP_DATA_TYPE_BFLOAT16) != cudaSuccess)
      return cudaErrorInvalidValue;
    if (make_tma_kv_units(&tdk_, a.dk, B, H, S) != cudaSuccess) return cudaErrorInvalidValue;
    if (make_tma_kv_units(&tdv_, a.dv, B, H, S) != cudaSuccess) return cudaErrorInvalidValue;
    cached_q = (const void*)a.q;
    cached_B = B;
    cached_S = S;
    cached_H = H;
  }
  // At 1M the main kernel is DRAM-bound (78.6% of peak, 9.5% L2 hit rate).
  // Keeping a fractional subset of the repeatedly reduced dQ lines saves
  // ~0.6% at 524K/1M, but the policy register/spills lose at 262K. Compile
  // both paths so the medium regime pays none of that cost.
  const bool keep_dq_l2 =
      (size_t)S * HEAD_DIM * sizeof(DQ_DTYPE) >= L2_RESIDENT_DQ_ACCUM_BYTES_PER_HEAD;
  auto kernel_ascending =
      keep_dq_l2
          ? vsa_bwd_main_kernel<true, SCHED, VSA_BHSD, DQ_DTYPE, false, false, false>
          : vsa_bwd_main_kernel<false, SCHED, VSA_BHSD, DQ_DTYPE, false, false, false>;
  auto kernel_snake =
      keep_dq_l2
          ? vsa_bwd_main_kernel<true, SCHED, VSA_BHSD, DQ_DTYPE, false, true, false>
          : vsa_bwd_main_kernel<false, SCHED, VSA_BHSD, DQ_DTYPE, false, true, false>;
  auto kernel_wave_ascending =
      keep_dq_l2
          ? vsa_bwd_main_kernel<true, Sched::NON_PERSISTENT, VSA_BHSD, DQ_DTYPE, false, false,
                                false>
          : vsa_bwd_main_kernel<false, Sched::NON_PERSISTENT, VSA_BHSD, DQ_DTYPE, false, false,
                                false>;
  auto kernel_wave_descending =
      keep_dq_l2
          ? vsa_bwd_main_kernel<true, Sched::NON_PERSISTENT, VSA_BHSD, DQ_DTYPE, true, false,
                                false>
          : vsa_bwd_main_kernel<false, Sched::NON_PERSISTENT, VSA_BHSD, DQ_DTYPE, true, false,
                                false>;
  auto kernel = a.k2q_traversal_snake ? kernel_snake : kernel_ascending;
  if constexpr (SCHED == Sched::DEFENSIVE) {
    if (a.cooperative_grid_sync) {
      auto kernel_grid_ascending =
          keep_dq_l2
              ? vsa_bwd_main_kernel<true, SCHED, VSA_BHSD, DQ_DTYPE, false, false, true>
              : vsa_bwd_main_kernel<false, SCHED, VSA_BHSD, DQ_DTYPE, false, false, true>;
      auto kernel_grid_snake =
          keep_dq_l2
              ? vsa_bwd_main_kernel<true, SCHED, VSA_BHSD, DQ_DTYPE, false, true, true>
              : vsa_bwd_main_kernel<false, SCHED, VSA_BHSD, DQ_DTYPE, false, true, true>;
      kernel = a.k2q_traversal_snake ? kernel_grid_snake : kernel_grid_ascending;
    }
  }

  static bool smem_set[2] = {false, false};
  if (!smem_set[keep_dq_l2]) {
    auto set_smem = [](auto selected_kernel) {
      return cudaFuncSetAttribute(selected_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  SMEM_TOTAL);
    };
    cudaError_t e = set_smem(kernel_ascending);
    if (e != cudaSuccess) return e;
    e = set_smem(kernel_snake);
    if (e != cudaSuccess) return e;
    e = set_smem(kernel_wave_ascending);
    if (e != cudaSuccess) return e;
    e = set_smem(kernel_wave_descending);
    if (e != cudaSuccess) return e;
    if constexpr (SCHED == Sched::DEFENSIVE) {
      e = keep_dq_l2
              ? set_smem(vsa_bwd_main_kernel<true, SCHED, VSA_BHSD, DQ_DTYPE, false, false, true>)
              : set_smem(vsa_bwd_main_kernel<false, SCHED, VSA_BHSD, DQ_DTYPE, false, false,
                                                 true>);
      if (e != cudaSuccess) return e;
      e = keep_dq_l2
              ? set_smem(vsa_bwd_main_kernel<true, SCHED, VSA_BHSD, DQ_DTYPE, false, true, true>)
              : set_smem(vsa_bwd_main_kernel<false, SCHED, VSA_BHSD, DQ_DTYPE, false, true,
                                                 true>);
      if (e != cudaSuccess) return e;
    }
    smem_set[keep_dq_l2] = true;
  }
  const float scale_log2 = a.sm_scale * 1.4426950408889634f;
  int sms                = 0;
  cudaError_t attr_error = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  if (attr_error != cudaSuccess) return attr_error;
  const int total = B * H * a.num_kv_blocks_per_seq;

  cudaLaunchConfig_t cfg = {};
  cfg.blockDim           = dim3(N_WARPS * 32, 1, 1);
  cfg.dynamicSmemBytes   = SMEM_TOTAL;
  cfg.stream             = stream;
  cudaLaunchAttribute at[3] = {};
  int num_attrs             = 0;
  at[num_attrs].id               = cudaLaunchAttributeClusterDimension;
  at[num_attrs].val.clusterDim.x = 1;
  at[num_attrs].val.clusterDim.y = 1;
  at[num_attrs].val.clusterDim.z = 1;
  ++num_attrs;
  if constexpr (KERNEL_PDL) {
    at[num_attrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[num_attrs].val.programmaticStreamSerializationAllowed = 1;
    ++num_attrs;
  }
  if constexpr (SCHED == Sched::DEFENSIVE) {
    at[num_attrs].id              = cudaLaunchAttributeCooperative;
    at[num_attrs].val.cooperative = 1;
    ++num_attrs;
  }
  cfg.attrs    = at;
  cfg.numAttrs = num_attrs;

  auto launch = [&](auto selected_kernel, const int* workitem_remap,
                    uint32_t* defensive_counter) {
    return cudaLaunchKernelEx(&cfg, selected_kernel, tk_, tv_, tqt_, tdot_, tdk_, tdv_,
                              static_cast<DQ_DTYPE*>(a.dqaccum), a.lse_rows, a.delta_rows,
                              a.k2q_idx, a.k2q_num, a.max_q_blocks, workitem_remap,
                              defensive_counter, B, H, S, scale_log2, a.sm_scale);
  };

  if (a.cta_waves) {
    if constexpr (SCHED != Sched::CLC) return cudaErrorInvalidValue;
    if (a.k2q_traversal_snake) return cudaErrorInvalidValue;
    // Do not let a wave straddle heads: each launch consumes one contiguous neighborhood from
    // exactly one head's permutation, and each CTA executes exactly one item.
    for (int batch_head = 0; batch_head < B * H; ++batch_head) {
      const int head_begin = batch_head * a.num_kv_blocks_per_seq;
      for (int local_begin = 0, wave = 0; local_begin < a.num_kv_blocks_per_seq;
           local_begin += sms, ++wave) {
        cfg.gridDim = dim3((unsigned)std::min(sms, a.num_kv_blocks_per_seq - local_begin), 1, 1);
        auto wave_kernel = a.alternate_wave_direction && (wave & 1)
                               ? kernel_wave_descending
                               : kernel_wave_ascending;
        cudaError_t e = launch(wave_kernel, a.workitem_remap + head_begin + local_begin,
                               a.defensive_counter + batch_head);
        if (e != cudaSuccess) return e;
      }
    }
    return cudaSuccess;
  }

  if constexpr (SCHED == Sched::DEFENSIVE) {
    if (!a.defensive_counter) return cudaErrorInvalidValue;
    cfg.gridDim = dim3((unsigned)std::min(sms, a.num_kv_blocks_per_seq), 1, 1);
    for (int batch_head = 0; batch_head < B * H; ++batch_head) {
      cudaError_t e = launch(kernel,
                             a.workitem_remap + (size_t)batch_head * a.num_kv_blocks_per_seq,
                             a.defensive_counter + batch_head);
      if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
  }

  if constexpr (SCHED == Sched::CLC) {
    // Only the L2-policy regime is chunked. K2Q_WAVES selects the
    // true non-persistent, per-head path above.
    const int chunk = keep_dq_l2 ? std::min(sms, total) : total;
    for (int base = 0; base < total; base += chunk) {
      cfg.gridDim = dim3((unsigned)std::min(chunk, total - base), 1, 1);
      cudaError_t e = launch(kernel, a.workitem_remap + base, a.defensive_counter);
      if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
  }

  const int grid = SCHED == Sched::STATIC_PERSISTENT ? std::min(total, sms) : total;
  cfg.gridDim = dim3((unsigned)grid, 1, 1);
  return launch(kernel, a.workitem_remap, a.defensive_counter);
}

// Preprocess / postprocess launches carry the PDL attribute so each may start while the previous
// kernel in the stream drains (their griddepcontrol.wait guards the data).
inline cudaLaunchConfig_t pdl_launch_config(dim3 grid, dim3 block, cudaStream_t stream,
                                            cudaLaunchAttribute* at) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim          = grid;
  cfg.blockDim         = block;
  cfg.stream           = stream;
  at->id               = cudaLaunchAttributeProgrammaticStreamSerialization;
  at->val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs            = at;
  cfg.numAttrs         = KERNEL_PDL ? 1 : 0;
  return cfg;
}

template <typename DQ_DTYPE = float>
inline cudaError_t launch_vsa_bwd_preprocess(const __nv_bfloat16* q, const __nv_bfloat16* o,
                                             const __nv_bfloat16* dout, float* delta_rows,
                                             DQ_DTYPE* dqaccum, __nv_bfloat16* qt,
                                             __nv_bfloat16* dot, int num_samples, int num_heads,
                                             int seqlen, cudaStream_t stream) {
  cudaLaunchAttribute at[1];
  cudaLaunchConfig_t cfg = pdl_launch_config(
      dim3((unsigned)(seqlen / PRE_TOKENS), (unsigned)(num_samples * num_heads), 1),
      dim3(256, 1, 1), stream, at);
  return cudaLaunchKernelEx(&cfg, vsa_bwd_preprocess_kernel<VSA_BHSD, DQ_DTYPE>, q, o, dout,
                            delta_rows, dqaccum, qt, dot, num_samples, num_heads, seqlen);
}

template <typename DQ_DTYPE = float>
inline cudaError_t launch_vsa_bwd_postprocess(const DQ_DTYPE* dqaccum, __nv_bfloat16* dq,
                                              int num_samples, int num_heads, int seqlen,
                                              float sm_scale, cudaStream_t stream) {
  cudaLaunchAttribute at[1];
  cudaLaunchConfig_t cfg = pdl_launch_config(
      dim3((unsigned)(seqlen / BLOCK), (unsigned)(num_samples * num_heads), 1), dim3(128, 1, 1),
      stream, at);
  return cudaLaunchKernelEx(&cfg, vsa_bwd_postprocess_kernel<VSA_BHSD, DQ_DTYPE>, dqaccum, dq,
                            num_heads, seqlen, sm_scale);
}

// Harness: the host reference tensors are [B*S, H, hd]; under VSA_BHSD the device copies are
// permuted to [B, H, S, hd] on upload and back on download.
static std::vector<__nv_bfloat16> to_device_layout(const std::vector<__nv_bfloat16>& src, int B,
                                                   int H, int S, int head_dim) {
  if (!VSA_BHSD) return src;
  std::vector<__nv_bfloat16> dst(src.size());
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int t = 0; t < S; ++t)
        memcpy(&dst[(((long)b * H + h) * S + t) * head_dim],
               &src[(((long)b * S + t) * H + h) * head_dim], (size_t)head_dim * 2);
  return dst;
}

static std::vector<__nv_bfloat16> from_device_layout(const std::vector<__nv_bfloat16>& src,
                                                     int B, int H, int S, int head_dim) {
  if (!VSA_BHSD) return src;
  std::vector<__nv_bfloat16> dst(src.size());
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int t = 0; t < S; ++t)
        memcpy(&dst[(((long)b * S + t) * H + h) * head_dim],
               &src[(((long)b * H + h) * S + t) * head_dim], (size_t)head_dim * 2);
  return dst;
}

// ---------------------------------------------------------------------------
// Bench harness (CPU reference + verify + timing).
// ---------------------------------------------------------------------------

static const float LOG2E = 1.4426950408889634f;

// Minimal fp32 .npy v1.0 writer (npy_io.cuh only reads). C order, 64-byte-aligned header.
static void npy_save_f32(const std::string& path, const float* data,
                         std::initializer_list<long> shape) {
  std::string dims;
  long n   = 1;
  size_t i = 0;
  for (long d : shape) {
    dims += std::to_string(d);
    n *= d;
    if (++i < shape.size()) dims += ", ";
  }
  if (shape.size() == 1) dims += ",";
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + dims + "), }";
  const size_t pad   = (64 - (10 + header.size() + 1) % 64) % 64;
  header.append(pad, ' ');
  header += '\n';
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    fprintf(stderr, "npy_save: cannot open %s\n", path.c_str());
    exit(1);
  }
  const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  fwrite(magic, 1, 8, f);
  const uint16_t header_len = (uint16_t)header.size();
  fwrite(&header_len, 2, 1, f);
  fwrite(header.data(), 1, header.size(), f);
  fwrite(data, sizeof(float), (size_t)n, f);
  fclose(f);
}

// Deterministic sorted k2q inversion: count per kv block + concatenated q-block lists,
// sorted ascending (built by ascending q-block walk); feeds the CPU reference and
// build_k2q_padded -> the kernel's k2q_idx/k2q_num.
struct KvToQ {
  std::vector<int> count;     // per global kv block, size B*H*num_blocks
  std::vector<int> offset;    // prefix sum, size B*H*num_blocks + 1
  std::vector<int> q_blocks;  // concatenated sorted local q-block ids
};

static KvToQ invert_q2k(const int* q2k_idx, const int* q2k_num, int B, int H, int num_blocks,
                        int max_kv) {
  const int total = B * H * num_blocks;
  KvToQ inv;
  inv.count.assign(total, 0);
  for (int global_mtile = 0; global_mtile < total; ++global_mtile) {
    const int batch_head = global_mtile / num_blocks;
    for (int i = 0; i < q2k_num[global_mtile]; ++i)
      inv.count[batch_head * num_blocks + q2k_idx[(size_t)global_mtile * max_kv + i]]++;
  }
  inv.offset.assign(total + 1, 0);
  for (int i = 0; i < total; ++i) inv.offset[i + 1] = inv.offset[i] + inv.count[i];
  inv.q_blocks.assign(inv.offset[total], 0);
  std::vector<int> cursor(inv.offset.begin(), inv.offset.end() - 1);
  for (int global_mtile = 0; global_mtile < total; ++global_mtile) {
    const int batch_head = global_mtile / num_blocks;
    const int mtile      = global_mtile % num_blocks;
    for (int i = 0; i < q2k_num[global_mtile]; ++i) {
      const int gkb = batch_head * num_blocks + q2k_idx[(size_t)global_mtile * max_kv + i];
      inv.q_blocks[cursor[gkb]++] = mtile;
    }
  }
  return inv;
}

// The kernel's index form (FastVideo's invert_indices layout): one padded row of max_q_blocks
// plain q-block ids per (batch*head, kv block) plus a count; entries past the count are never
// read (poisoned with -1 so a stray read shows). Rows are the widest list here; FastVideo pads
// to num_blocks.
struct K2qPadded {
  int max_q_blocks;
  std::vector<int> idx;  // [B*H*nb, max_q_blocks]
  std::vector<int> num;  // [B*H*nb]
};

static K2qPadded build_k2q_padded(const KvToQ& k2q, int B, int H, int nb) {
  const int items = B * H * nb;
  K2qPadded out;
  int widest = 0;
  for (int c : k2q.count) widest = std::max(widest, c);
  out.max_q_blocks = std::max(widest, 1);
  out.idx.assign((size_t)items * out.max_q_blocks, -1);
  out.num = k2q.count;
  for (int it = 0; it < items; ++it)
    for (int i = k2q.offset[it]; i < k2q.offset[it + 1]; ++i)
      out.idx[(size_t)it * out.max_q_blocks + (i - k2q.offset[it])] = k2q.q_blocks[i];
  return out;
}

// CPU reference. Pass 1 (parallel over q-blocks): forward O + M, Delta, dQ.
// Pass 2 (parallel over kv blocks, sorted k2q walk): dK, dV -- race-free and
// deterministic because each kv block owns its dK/dV rows.
static void cpu_vsa_bwd_ref(const __nv_bfloat16* hQ, const __nv_bfloat16* hK,
                            const __nv_bfloat16* hV, const __nv_bfloat16* hdO, float* hO, float* hM,
                            float* hDelta, float* hdQ, float* hdK, float* hdV, int B, int H, int S,
                            int hd, int num_blocks, int max_kv, const int* q2k_idx,
                            const int* q2k_num, const KvToQ& k2q, const float* file_O,
                            const float* file_M, float* file_err) {
  const float sm_scale   = 1.0f / sqrtf((float)hd);
  const long total_elems = (long)B * S * H * hd;
  for (long i = 0; i < total_elems; ++i) {
    hO[i]  = 0.f;
    hdQ[i] = 0.f;
    hdK[i] = 0.f;
    hdV[i] = 0.f;
  }

  float lse_err = 0.f, o_err = 0.f;
  #pragma omp parallel for schedule(dynamic) reduction(max : lse_err, o_err)
  for (int bhq = 0; bhq < B * H * num_blocks; ++bhq) {
    const int mtile         = bhq % num_blocks;
    const int batch_head    = bhq / num_blocks;
    const int h             = batch_head % H;
    const int b             = batch_head / H;
    const int num_kv_blocks = q2k_num[bhq];

    for (int qi = 0; qi < BLOCK; ++qi) {
      const long qp  = (long)b * S + (long)mtile * BLOCK + qi;
      const long row = (long)batch_head * S + (long)mtile * BLOCK + qi;
      std::vector<float> z((size_t)num_kv_blocks * BLOCK);
      float m = -INFINITY;
      int idx = 0;
      for (int kk = 0; kk < num_kv_blocks; ++kk) {
        const int blk = q2k_idx[(size_t)bhq * max_kv + kk];
        for (int kj = 0; kj < BLOCK; ++kj) {
          const long kp = (long)b * S + (long)blk * BLOCK + kj;
          float dot     = 0.f;
          for (int e = 0; e < hd; ++e)
            dot += __bfloat162float(hQ[(qp * H + h) * hd + e]) *
                   __bfloat162float(hK[(kp * H + h) * hd + e]);
          z[idx] = dot * sm_scale * LOG2E;
          m      = fmaxf(m, z[idx]);
          ++idx;
        }
      }
      float l = 0.f;
      for (int j = 0; j < num_kv_blocks * BLOCK; ++j) l += exp2f(z[j] - m);
      const float inv_l = 1.f / l;
      idx               = 0;
      for (int kk = 0; kk < num_kv_blocks; ++kk) {
        const int blk = q2k_idx[(size_t)bhq * max_kv + kk];
        for (int kj = 0; kj < BLOCK; ++kj) {
          const long kp = (long)b * S + (long)blk * BLOCK + kj;
          const float p = exp2f(z[idx] - m) * inv_l;
          for (int e = 0; e < hd; ++e)
            hO[(qp * H + h) * hd + e] += p * __bfloat162float(hV[(kp * H + h) * hd + e]);
          ++idx;
        }
      }
      float M_row = m + log2f(l);
      if (file_M) {
        lse_err = fmaxf(lse_err, fabsf(file_M[row] - M_row));
        for (int e = 0; e < hd; ++e) {
          const long o_index = (qp * H + h) * hd + e;
          o_err = fmaxf(o_err, fabsf(file_O[o_index] - hO[o_index]) /
                                   (fabsf(hO[o_index]) / 128.f + 1e-4f));
          hO[o_index] = file_O[o_index];
        }
        M_row = file_M[row];
      }
      hM[row] = M_row;

      // Delta from bf16-ROUNDED O: production backward consumes the forward's
      // saved bf16 O, so the reference must quantize O before the rowsum or
      // the GPU comparison inherits a spurious ~4e-3 quantization skew.
      float delta = 0.f;
      for (int e = 0; e < hd; ++e)
        delta += __bfloat162float(hdO[(qp * H + h) * hd + e]) *
                 __bfloat162float(__float2bfloat16(hO[(qp * H + h) * hd + e]));
      hDelta[row] = delta;

      idx = 0;
      for (int kk = 0; kk < num_kv_blocks; ++kk) {
        const int blk = q2k_idx[(size_t)bhq * max_kv + kk];
        for (int kj = 0; kj < BLOCK; ++kj) {
          const long kp = (long)b * S + (long)blk * BLOCK + kj;
          const float P = exp2f(z[idx] - M_row);
          float dP      = 0.f;
          for (int e = 0; e < hd; ++e)
            dP += __bfloat162float(hdO[(qp * H + h) * hd + e]) *
                  __bfloat162float(hV[(kp * H + h) * hd + e]);
          // dS is cast to bf16 before the dQ/dK dots on both the Triton and
          // the sm100a GPU paths; the reference matches that quantization.
          const float coef = __bfloat162float(__float2bfloat16(P * (dP - delta)));
          for (int e = 0; e < hd; ++e)
            hdQ[(qp * H + h) * hd + e] += coef * __bfloat162float(hK[(kp * H + h) * hd + e]);
          ++idx;
        }
      }
      for (int e = 0; e < hd; ++e) hdQ[(qp * H + h) * hd + e] *= sm_scale;
    }
  }

  if (file_err) {
    file_err[0] = lse_err;
    file_err[1] = o_err;
  }

  #pragma omp parallel for schedule(dynamic)
  for (int gkb = 0; gkb < B * H * num_blocks; ++gkb) {
    const int kb         = gkb % num_blocks;
    const int batch_head = gkb / num_blocks;
    const int h          = batch_head % H;
    const int b          = batch_head / H;
    for (int qi_list = k2q.offset[gkb]; qi_list < k2q.offset[gkb + 1]; ++qi_list) {
      const int mtile = k2q.q_blocks[qi_list];
      for (int qi = 0; qi < BLOCK; ++qi) {
        const long qp     = (long)b * S + (long)mtile * BLOCK + qi;
        const long row    = (long)batch_head * S + (long)mtile * BLOCK + qi;
        const float M_row = hM[row];
        const float delta = hDelta[row];
        for (int kj = 0; kj < BLOCK; ++kj) {
          const long kp = (long)b * S + (long)kb * BLOCK + kj;
          float dot = 0.f, dP = 0.f;
          for (int e = 0; e < hd; ++e) {
            dot += __bfloat162float(hQ[(qp * H + h) * hd + e]) *
                   __bfloat162float(hK[(kp * H + h) * hd + e]);
            dP += __bfloat162float(hdO[(qp * H + h) * hd + e]) *
                  __bfloat162float(hV[(kp * H + h) * hd + e]);
          }
          const float P = exp2f(dot * sm_scale * LOG2E - M_row);
          // P feeds dV as bf16 (MMA operand) and dS feeds dK as bf16, matching
          // the Triton and sm100a GPU quantization points.
          const float Pq   = __bfloat162float(__float2bfloat16(P));
          const float coef = __bfloat162float(__float2bfloat16(P * (dP - delta)));
          for (int e = 0; e < hd; ++e) {
            hdV[(kp * H + h) * hd + e] += Pq * __bfloat162float(hdO[(qp * H + h) * hd + e]);
            hdK[(kp * H + h) * hd + e] += coef * __bfloat162float(hQ[(qp * H + h) * hd + e]);
          }
        }
      }
    }
    const float sm_scale2 = 1.0f / sqrtf((float)hd);
    for (int kj = 0; kj < BLOCK; ++kj) {
      const long kp = (long)b * S + (long)kb * BLOCK + kj;
      for (int e = 0; e < hd; ++e) hdK[(kp * H + h) * hd + e] *= sm_scale2;
    }
  }
}

// Deterministic fill in [-1, 1) (same hash as the forward bench).
static void fillr(__nv_bfloat16* h, long n, unsigned seed) {
  for (long i = 0; i < n; ++i) {
    uint32_t x = (uint32_t)i * 2654435761u + seed * 40503u + 0x9e3779b9u;
    x ^= x >> 15;
    x *= 2246822519u;
    x ^= x >> 13;
    x *= 3266489917u;
    x ^= x >> 16;
    h[i] = __float2bfloat16((float)(x % 2039u) / 1019.5f - 1.0f);
  }
}

struct Sh {
  int B, H, num_blocks, topk, hd;
  const char* lab;
};

static void run(const Sh& sh) {
  const int B = sh.B, H = sh.H, num_blocks = sh.num_blocks, topk = sh.topk, hd = sh.hd;
  const int S                   = num_blocks * BLOCK;
  const int max_kv              = topk;
  const long tq                 = (long)B * S;
  const int num_global_q_blocks = B * H * num_blocks;
  const size_t q2k_entries      = (size_t)num_global_q_blocks * (size_t)max_kv;
  if (q2k_entries > (size_t)INT_MAX) {
    fprintf(stderr, "metadata overflow: %zu q2k entries exceed int32 offsets\n", q2k_entries);
    exit(EXIT_FAILURE);
  }

  printf("  [%-9s H%-2d num_blocks%-3d topk%-3d S%d blk%d] N_q=%ld\n", sh.lab, H, num_blocks, topk,
         S, BLOCK, tq);

  std::vector<__nv_bfloat16> hQ(tq * H * hd), hK(tq * H * hd), hV(tq * H * hd), hdO(tq * H * hd);
  const char* load_npy = getenv("LOAD_NPY");
  if (load_npy) {
    const std::string d(load_npy);
    auto ld = [&](const char* nm, std::vector<__nv_bfloat16>& h) {
      char p[64];
      snprintf(p, sizeof p, "/%s_S%d.npy", nm, S);
      auto bits = npy_load_vec<uint16_t>(d + p);
      if (bits.size() != h.size()) {
        fprintf(stderr, "LOAD_NPY: %s size %zu != %zu\n", nm, bits.size(), h.size());
        exit(1);
      }
      memcpy(h.data(), bits.data(), h.size() * 2);
    };

    ld("q", hQ);
    ld("k", hK);
    ld("v", hV);
    ld("do", hdO);
  } else {
    auto FILL = fillr;
    FILL(hQ.data(), hQ.size(), 11);
    FILL(hK.data(), hK.size(), 22);
    FILL(hV.data(), hV.size(), 33);
    FILL(hdO.data(), hdO.size(), 44);
  }

  // q2k index: LOAD_NPY head-independent [num_blocks, topk] broadcast, or topk DISTINCT
  // block ids per (b,h,mtile) via partial Fisher-Yates (same knobs as the forward bench).
  std::vector<int> hq2k_idx(q2k_entries, 0);
  std::vector<int> hq2k_num(num_global_q_blocks, topk);

  if (load_npy) {
    char p[64];
    snprintf(p, sizeof p, "/idx_S%d_blk%d.npy", S, BLOCK);
    auto idx = npy_load_vec<int32_t>(std::string(load_npy) + p);
    if (idx.size() != (size_t)num_blocks * topk) {
      fprintf(stderr, "LOAD_NPY: idx size %zu != %d\n", idx.size(), num_blocks * topk);
      exit(1);
    }
    for (int global_mtile = 0; global_mtile < num_global_q_blocks; ++global_mtile) {
      const int mtile = global_mtile % num_blocks;
      for (int i = 0; i < topk; ++i)
        hq2k_idx[(size_t)global_mtile * max_kv + i] = idx[(size_t)mtile * topk + i];
    }
  } else {
    std::vector<int> perm(num_blocks);
    for (int global_mtile = 0; global_mtile < num_global_q_blocks; ++global_mtile) {
      for (int i = 0; i < num_blocks; ++i) perm[i] = i;
      uint32_t st = (uint32_t)global_mtile * 2654435761u + 12345u;
      for (int i = 0; i < topk; ++i) {
        st ^= st << 13;
        st ^= st >> 17;
        st ^= st << 5;
        const int j                                 = i + (int)(st % (uint32_t)(num_blocks - i));
        const int t                                 = perm[i];
        perm[i]                                     = perm[j];
        perm[j]                                     = t;
        hq2k_idx[(size_t)global_mtile * max_kv + i] = perm[i];
      }
    }
  }

  const KvToQ k2q    = invert_q2k(hq2k_idx.data(), hq2k_num.data(), B, H, num_blocks, max_kv);
  const K2qPadded k2q_padded = build_k2q_padded(k2q, B, H, num_blocks);
  {
    const long usum = k2q.offset[B * H * num_blocks];
    printf("  q-lists: items=%d entries=%ld quads/item=%.1f max_q_blocks=%d\n", B * H * num_blocks,
           usum, (double)usum / (B * H * num_blocks), k2q_padded.max_q_blocks);
  }
  {
    int cmin = INT_MAX, cmax = 0, zeros = 0;
    long csum = 0;
    for (int c : k2q.count) {
      cmin = std::min(cmin, c);
      cmax = std::max(cmax, c);
      csum += c;
      if (c == 0) ++zeros;
    }
    // kv-block count == q-block count here (square S x S block grid).
    printf("  k2q: kv_blocks=%d count min=%d max=%d mean=%.2f zero-count=%d\n", num_global_q_blocks,
           cmin, cmax, (double)csum / num_global_q_blocks, zeros);
  }

  const char* dump_prefix     = getenv("DUMP_BWD");
  const char* cpu_env         = getenv("CPU_REF");
  const double selected_pairs = (double)B * H * num_blocks * topk * (double)BLOCK * BLOCK;
  bool run_cpu = cpu_env ? (atoi(cpu_env) != 0) : (dump_prefix != nullptr || selected_pairs <= 4e6);
  if (dump_prefix && !run_cpu) printf("  DUMP_BWD set but CPU_REF=0: no reference to dump\n");

  std::vector<float> hO(tq * H * hd), hdQ(tq * H * hd), hdK(tq * H * hd), hdV(tq * H * hd);
  std::vector<float> hM((size_t)B * H * S), hDelta((size_t)B * H * S);
  std::vector<float> file_O, file_M;
  if (load_npy) {
    char p[64];
    snprintf(p, sizeof p, "/o_S%d_blk%d.npy", S, BLOCK);
    const auto o_bits = npy_load_vec<uint16_t>(std::string(load_npy) + p);
    snprintf(p, sizeof p, "/lse_S%d_blk%d.npy", S, BLOCK);
    file_M = npy_load_vec<float>(std::string(load_npy) + p);
    if (o_bits.size() != hO.size() || file_M.size() != hM.size()) {
      fprintf(stderr, "LOAD_NPY: forward state size mismatch; rerun block_sparse_bf16_gen_inputs.py\n");
      exit(1);
    }
    file_O.resize(o_bits.size());
    for (size_t i = 0; i < o_bits.size(); ++i) {
      __nv_bfloat16 o;
      memcpy(&o, &o_bits[i], 2);
      file_O[i] = __bfloat162float(o);
    }
    printf("  fwd state: o/lse_S%d_blk%d.npy\n", S, BLOCK);
  } else if (!run_cpu) {
    if (cpu_env) {
      fprintf(stderr, "CPU_REF=0 needs LOAD_NPY: perf runs use the real forward O/LSE\n");
      exit(1);
    }
    printf("  skipped: above the CPU_REF size limit and no LOAD_NPY forward state\n");
    return;
  }
  if (!run_cpu) {
    hO = std::move(file_O);
    hM = std::move(file_M);
  }
  if (run_cpu) {
    const auto t0 = std::chrono::steady_clock::now();
    float file_err[2] = {0.f, 0.f};
    cpu_vsa_bwd_ref(hQ.data(), hK.data(), hV.data(), hdO.data(), hO.data(), hM.data(),
                    hDelta.data(), hdQ.data(), hdK.data(), hdV.data(), B, H, S, hd, num_blocks,
                    max_kv, hq2k_idx.data(), hq2k_num.data(), k2q,
                    load_npy ? file_O.data() : nullptr, load_npy ? file_M.data() : nullptr,
                    file_err);
    const auto t1       = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  cpu ref: %.1f ms (fwd O/M + Delta + dQ + k2q dK/dV, fp32)\n", cpu_ms);
    if (load_npy) {
      const bool fwd_ok = file_err[0] < 1e-4f && file_err[1] <= 1.f;
      printf("  fwd state vs cpu: lse max|diff|=%.2e  o max|diff|/(|o|/128+1e-4)=%.2f  %s\n",
             file_err[0], file_err[1], fwd_ok ? "OK" : "FAIL");
      if (!fwd_ok) exit(1);
    }

    if (dump_prefix) {
      const std::string prefix(dump_prefix);
      npy_save_f32(prefix + "_dq.npy", hdQ.data(), {tq, (long)H, (long)hd});
      npy_save_f32(prefix + "_dk.npy", hdK.data(), {tq, (long)H, (long)hd});
      npy_save_f32(prefix + "_dv.npy", hdV.data(), {tq, (long)H, (long)hd});
      npy_save_f32(prefix + "_M.npy", hM.data(), {(long)B * H, (long)S});
      npy_save_f32(prefix + "_delta.npy", hDelta.data(), {(long)B * H, (long)S});
      printf(
          "  dump: %s_{dq,dk,dv}.npy [%ld,%d,%d] f32; %s_{M,delta}.npy [%d,%d] f32"
          " ([H,S] at B=1; M log2-domain)\n",
          dump_prefix, tq, H, hd, dump_prefix, B * H, S);
    }
  } else {
    printf("  cpu ref: skipped (%.0f selected pairs; CPU_REF=1 to force)\n", selected_pairs);
  }

  // GPU backward (FA4-aligned): stream-chained preprocess (Delta + dQaccum
  // zeroing) -> non-persistent main kernel -> postprocess (bf16 dQ).
  {
    const long elems = tq * H * hd;
    __nv_bfloat16 *dQg, *dKg, *dVg, *dDOg, *dOg, *dDKout, *dDVout, *dDQout;
    float *dMg, *dDeltag;
    void* dDQA;
    int *dK2qIdx, *dK2qNum, *dOrder;
    uint32_t* dDefensiveCounter;
    // Reverse lists are sorted q ids. Grouping similarly sized lists keeps
    // persistent CTAs at nearly the same q quantile, so their Q/dO loads and
    // dQ reductions reuse L2. At >=524K, sort each wider count bin in a
    // midpoint snake; the launcher consumes exactly one SM-wide bin slice at
    // a time, avoiding the end-to-start midpoint jump between adjacent bins.
    const bool dq_f16      = getenv("DQ_F16") != nullptr;  // default = fp32 accum
    const bool cache_waves =
        (size_t)S * HEAD_DIM * (dq_f16 ? 2 : 4) >= L2_RESIDENT_DQ_ACCUM_BYTES_PER_HEAD;
    const char* cta_waves_env = getenv("K2Q_WAVES");
    const bool defensive_waves =
        getenv("DEFENSIVE_WAVES") && atoi(getenv("DEFENSIVE_WAVES")) != 0;
    if (defensive_waves && cta_waves_env && atoi(cta_waves_env) != 0) {
      fprintf(stderr, "K2Q_WAVES and DEFENSIVE_WAVES are mutually exclusive\n");
      exit(1);
    }
    // Measured crossover: ordinary CLC wins through 65K; true one-item waves win from 131K.
    const bool cta_waves =
        !defensive_waves && (cta_waves_env ? atoi(cta_waves_env) != 0 : S >= 131072);
    const char* wave_snake_env = getenv("K2Q_WAVE_SNAKE");
    const bool alternate_wave_direction =
        cta_waves && (!wave_snake_env || atoi(wave_snake_env) != 0);
    const char* traversal_snake_env = getenv("K2Q_TRAVERSAL_SNAKE");
    const bool traversal_snake =
        !cta_waves && traversal_snake_env && atoi(traversal_snake_env) != 0;
    const bool cooperative_grid_sync =
        defensive_waves && getenv("COOPERATIVE_GRID_SYNC") &&
        atoi(getenv("COOPERATIVE_GRID_SYNC")) != 0;
    const int default_order_bin = cta_waves ? 4 : (cache_waves ? 12 : 8);
    const int order_bin =
        getenv("K2Q_COUNT_BIN") ? std::max(1, atoi(getenv("K2Q_COUNT_BIN"))) : default_order_bin;
    const bool kv_snake = cache_waves || cta_waves || defensive_waves;
    std::vector<int> workitem_remap((size_t)B * H * num_blocks);
    for (int batch_head = 0; batch_head < B * H; ++batch_head) {
      int* first = workitem_remap.data() + (size_t)batch_head * num_blocks;
      for (int n = 0; n < num_blocks; ++n) first[n] = batch_head * num_blocks + n;
      std::stable_sort(first, first + num_blocks, [&](int a, int b) {
        const int ca = k2q.count[a];
        const int cb = k2q.count[b];
        const int ka = ca / order_bin, kb = cb / order_bin;
        if (ka != kb) return ka < kb;
        if (kv_snake) {
          const int ma = ca ? k2q_padded.idx[(size_t)a * k2q_padded.max_q_blocks + ca / 2] : -1;
          const int mb = cb ? k2q_padded.idx[(size_t)b * k2q_padded.max_q_blocks + cb / 2] : -1;
          if (ma != mb) return ((ka & 1) != 0) ? ma > mb : ma < mb;
        }
        return false;
      });
    }
    printf("  k2q order: count-bin%d%s\n", order_bin,
           kv_snake ? " + midpoint snake" : "");
    if (cta_waves)
      printf("  k2q waves: per-head <= one CTA/SM, %s traversal\n",
             alternate_wave_direction ? "alternating" : "ascending");
    if (defensive_waves)
      printf("  defensive waves: per-head persistent, %s barrier%s\n",
             cooperative_grid_sync ? "cooperative-grid" : "CTA-leader atomic",
             traversal_snake ? ", traversal snake" : "");
    CUDA_CHECK(cudaMalloc(&dQg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dKg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dVg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDOg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dOg, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDKout, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDVout, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDQout, elems * 2));
    CUDA_CHECK(cudaMalloc(&dMg, (size_t)B * H * S * 4));
    CUDA_CHECK(cudaMalloc(&dDeltag, (size_t)B * H * S * 4));
    CUDA_CHECK(cudaMalloc(&dDQA, (size_t)B * H * S * hd * (dq_f16 ? 2 : 4)));
    __nv_bfloat16 *dQT, *dDOT;
    CUDA_CHECK(cudaMalloc(&dQT, elems * 2));
    CUDA_CHECK(cudaMalloc(&dDOT, elems * 2));
    CUDA_CHECK(cudaMalloc(&dK2qIdx, k2q_padded.idx.size() * 4));
    CUDA_CHECK(cudaMalloc(&dK2qNum, k2q_padded.num.size() * 4));
    CUDA_CHECK(cudaMalloc(&dOrder, workitem_remap.size() * 4));
    CUDA_CHECK(cudaMalloc(&dDefensiveCounter, (size_t)B * H * sizeof(uint32_t)));
    auto upload_act = [&](__nv_bfloat16* dst, const std::vector<__nv_bfloat16>& host) {
      const std::vector<__nv_bfloat16> permuted = to_device_layout(host, B, H, S, hd);
      CUDA_CHECK(cudaMemcpy(dst, permuted.data(), elems * 2, cudaMemcpyHostToDevice));
    };
    upload_act(dQg, hQ);
    upload_act(dKg, hK);
    upload_act(dVg, hV);
    upload_act(dDOg, hdO);
    {
      std::vector<__nv_bfloat16> hObf(elems);
      for (long i = 0; i < elems; ++i) hObf[i] = __float2bfloat16(hO[i]);
      upload_act(dOg, hObf);
    }
    CUDA_CHECK(cudaMemcpy(dMg, hM.data(), (size_t)B * H * S * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK2qIdx, k2q_padded.idx.data(), k2q_padded.idx.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK2qNum, k2q_padded.num.data(), k2q_padded.num.size() * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dOrder, workitem_remap.data(), workitem_remap.size() * 4,
                          cudaMemcpyHostToDevice));

    VsaBwdArgs args;
    args.q                     = dQg;
    args.k                     = dKg;
    args.v                     = dVg;
    args.dout                  = dDOg;
    args.dqaccum               = dDQA;
    args.dk                    = dDKout;
    args.dv                    = dDVout;
    args.dq                    = dDQout;
    args.qt                    = dQT;
    args.dot                   = dDOT;
    args.lse_rows              = dMg;
    args.delta_rows            = dDeltag;
    args.k2q_idx               = dK2qIdx;
    args.k2q_num               = dK2qNum;
    args.workitem_remap        = dOrder;
    args.defensive_counter     = dDefensiveCounter;
    args.cta_waves             = cta_waves;
    args.alternate_wave_direction = alternate_wave_direction;
    args.k2q_traversal_snake      = traversal_snake;
    args.cooperative_grid_sync    = cooperative_grid_sync;
    args.num_samples           = B;
    args.num_heads             = H;
    args.seqlen                = S;
    args.num_kv_blocks_per_seq = num_blocks;
    args.max_q_blocks          = k2q_padded.max_q_blocks;
    args.sm_scale              = 1.0f / sqrtf((float)hd);

    bool any_zero_count = false;
    for (int c : k2q.count)
      if (c == 0) {
        any_zero_count = true;
        break;
    }
    constexpr Sched SCHED = KERNEL_SCHED;
    auto run_once          = [&]() {
      if (defensive_waves)
        CUDA_CHECK(cudaMemsetAsync(dDefensiveCounter, 0, (size_t)B * H * sizeof(uint32_t), 0));
      if (any_zero_count) {
        CUDA_CHECK(cudaMemsetAsync(dDKout, 0, elems * 2, 0));
        CUDA_CHECK(cudaMemsetAsync(dDVout, 0, elems * 2, 0));
      }
      if (dq_f16) {
        CUDA_CHECK(launch_vsa_bwd_preprocess<uint16_t>(
            dQg, dOg, dDOg, dDeltag, static_cast<uint16_t*>(dDQA), dQT, dDOT, B, H, S, 0));
        if (defensive_waves)
          CUDA_CHECK((launch_vsa_bwd_sm100a<Sched::DEFENSIVE, uint16_t>(args, 0)));
        else
          CUDA_CHECK((launch_vsa_bwd_sm100a<SCHED, uint16_t>(args, 0)));
        CUDA_CHECK(launch_vsa_bwd_postprocess<uint16_t>(static_cast<const uint16_t*>(dDQA), dDQout,
                                                        B, H, S, args.sm_scale, 0));
      } else {
        CUDA_CHECK(launch_vsa_bwd_preprocess<float>(
            dQg, dOg, dDOg, dDeltag, static_cast<float*>(dDQA), dQT, dDOT, B, H, S, 0));
        if (defensive_waves)
          CUDA_CHECK((launch_vsa_bwd_sm100a<Sched::DEFENSIVE, float>(args, 0)));
        else
          CUDA_CHECK((launch_vsa_bwd_sm100a<SCHED, float>(args, 0)));
        CUDA_CHECK(launch_vsa_bwd_postprocess<float>(static_cast<const float*>(dDQA), dDQout, B, H,
                                                     S, args.sm_scale, 0));
      }
    };

    run_once();
    CUDA_CHECK(cudaDeviceSynchronize());

    if (run_cpu || getenv("DUMP_BWD_GPU")) {
      std::vector<float> gDelta((size_t)B * H * S);
      std::vector<__nv_bfloat16> gDK(elems), gDV(elems), gDQ(elems);
      CUDA_CHECK(cudaMemcpy(gDelta.data(), dDeltag, gDelta.size() * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(gDQ.data(), dDQout, elems * 2, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(gDK.data(), dDKout, elems * 2, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(gDV.data(), dDVout, elems * 2, cudaMemcpyDeviceToHost));
      gDQ = from_device_layout(gDQ, B, H, S, hd);
      gDK = from_device_layout(gDK, B, H, S, hd);
      gDV = from_device_layout(gDV, B, H, S, hd);

      auto rel_norm = [](const float* ref, const float* got, long n) {
        double mref = 0, mdiff = 0;
        for (long i = 0; i < n; ++i) {
          const double r = fabs((double)ref[i]);
          const double d = fabs((double)ref[i] - got[i]);
          if (r > mref) mref = r;
          if (d > mdiff) mdiff = d;
        }
        return mref > 0 ? mdiff / mref : mdiff;
      };

      double dmax = 0;
      for (long i = 0; i < (long)gDelta.size(); ++i)
        dmax = std::max(dmax, fabs((double)gDelta[i] - hDelta[i]));
      // dq: the postprocess already unscrambled + scaled + rounded to bf16.
      std::vector<float> gdq(elems);
      for (long i = 0; i < elems; ++i) gdq[i] = __bfloat162float(gDQ[i]);
      std::vector<float> gdk(elems), gdv(elems);
      for (long i = 0; i < elems; ++i) {
        gdk[i] = __bfloat162float(gDK[i]);
        gdv[i] = __bfloat162float(gDV[i]);
      }
      const double rq = rel_norm(hdQ.data(), gdq.data(), elems);
      const double rk = rel_norm(hdK.data(), gdk.data(), elems);
      const double rv = rel_norm(hdV.data(), gdv.data(), elems);
      if (run_cpu && getenv("VERIFY_ARGMAX")) {
        long am     = 0;
        double best = -1;
        for (long i = 0; i < elems; ++i) {
          const double d = fabs((double)hdQ[i] - gdq[i]);
          if (d > best) {
            best = d;
            am   = i;
          }
        }
        const int d_  = (int)(am % hd);
        const int h_  = (int)((am / hd) % H);
        const long t_ = am / hd / H;
        printf("  dq argmax: t=%ld (blk %ld, row %ld) h=%d d=%d ref=%.6f got=%.6f\n", t_, t_ / 64,
               t_ % 64, h_, d_, hdQ[am], gdq[am]);
        long over1e3 = 0, over3e4 = 0;
        double sum = 0;
        for (long i = 0; i < elems; ++i) {
          const double d = fabs((double)hdQ[i] - gdq[i]);
          sum += d;
          if (d > 1e-3) ++over1e3;
          if (d > 3e-4) ++over3e4;
        }
        printf("  dq diff: mean=%.2e  >3e-4: %ld/%ld  >1e-3: %ld\n", sum / elems, over3e4, elems,
               over1e3);
        int shown = 0;
        for (long i = 0; i < elems && shown < 12; ++i) {
          const double d = fabs((double)hdQ[i] - gdq[i]);
          if (d > 3e-4) {
            printf("    t=%ld h=%ld d=%ld ref=%.6f got=%.6f\n", i / hd / H, (i / hd) % H, i % hd,
                   hdQ[i], gdq[i]);
            ++shown;
          }
        }
      }
      if (const char* gp = getenv("DUMP_BWD_GPU")) {
        const std::string prefix(gp);
        npy_save_f32(prefix + "_dq.npy", gdq.data(), {tq, (long)H, (long)hd});
        npy_save_f32(prefix + "_dk.npy", gdk.data(), {tq, (long)H, (long)hd});
        npy_save_f32(prefix + "_dv.npy", gdv.data(), {tq, (long)H, (long)hd});
      }
      // dq gate 8e-3: the CPU-ref-vs-torch-fp32 noise floor from the
      // production bf16 quantization points is ~1.9-2.7e-3, GPU-vs-CPU can
      // legitimately reach ~2x that, and the
      // bf16-rounded dq output adds its own rounding on top.
      if (run_cpu) {
        const double rq_gate = 8e-3;
        const bool pass      = dmax < 1e-4 && rq < rq_gate && rk < 8e-3 && rv < 8e-3;
        printf("  gpu verify: delta max|diff|=%.2e  dq rel=%.2e  dk rel=%.2e  dv rel=%.2e  %s\n",
               dmax, rq, rk, rv, pass ? "OK" : "FAIL");
        if (!pass) exit(1);
      }

      if (const char* sn = getenv("STRESS_N")) {
        const int n = atoi(sn);
        std::vector<__nv_bfloat16> rDK(elems), rDV(elems), rDQ(elems);
        bool ok = true;
        for (int it = 0; it < n && ok; ++it) {
          run_once();
          CUDA_CHECK(cudaDeviceSynchronize());
          CUDA_CHECK(cudaMemcpy(rDK.data(), dDKout, elems * 2, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(rDV.data(), dDVout, elems * 2, cudaMemcpyDeviceToHost));
          CUDA_CHECK(cudaMemcpy(rDQ.data(), dDQout, elems * 2, cudaMemcpyDeviceToHost));
          rDK = from_device_layout(rDK, B, H, S, hd);
          rDV = from_device_layout(rDV, B, H, S, hd);
          rDQ = from_device_layout(rDQ, B, H, S, hd);
          ok        = memcmp(rDK.data(), gDK.data(), elems * 2) == 0 &&
                      memcmp(rDV.data(), gDV.data(), elems * 2) == 0;
          double mq = 0;
          for (long i = 0; i < elems; ++i)
            mq = std::max(mq, fabs((double)__bfloat162float(rDQ[i]) - __bfloat162float(gDQ[i])));
          if (mq > 1e-2) ok = false;  // reduce-add order x bf16 rounding
        }
        printf("  stress x%d: %s\n", n, ok ? "OK (dk/dv bitwise, dq stable)" : "FAIL");
        if (!ok) exit(1);
      }
    }

    if (block_sparse_bwd_bf16_benchmark::enabled()) {
      const auto options = block_sparse_bwd_bf16_benchmark::options_from_env();
      const double ms = block_sparse_bwd_bf16_benchmark::measure(run_once, options);
      const double tflops = block_sparse_bwd_bf16_benchmark::tflops(hd, selected_pairs, ms);
      printf("  gpu bwd: %.4f ms  %.1f TFLOPS (bwd 2.5x sel; pre+main+post)\n", ms, tflops);
    }

#ifdef WARP_PROF
    {
      WpBuffer wp = wp_alloc(dim3((unsigned)(num_blocks * H), 1, 1));
      run_once();
      CUDA_CHECK(cudaDeviceSynchronize());
      wp_readback(wp);
      const char* roles[16] = {"red", "red", "red", "red", "cmp", "cmp",  "cmp", "cmp",
                               "cmp", "cmp", "cmp", "cmp", "mma", "load", "rly", "emp"};
      printf("  WARP_PROF block %u:\n", wp.view_block);
      wp_print_busy(wp, roles, 16, wp.view_block);
      wp_dump_raw(wp, "warp_raw_vsa_bwd_blk64.bin.gz", wp.view_block, 2);
      wp_free(wp);
    }
#endif

    cudaFree(dQg);
    cudaFree(dKg);
    cudaFree(dVg);
    cudaFree(dDOg);
    cudaFree(dOg);
    cudaFree(dDKout);
    cudaFree(dDVout);
    cudaFree(dDQout);
    cudaFree(dMg);
    cudaFree(dDeltag);
    cudaFree(dDQA);
    cudaFree(dQT);
    cudaFree(dDOT);
    cudaFree(dK2qIdx);
    cudaFree(dK2qNum);
    cudaFree(dOrder);
    cudaFree(dDefensiveCounter);
  }
}

int main() {
  CUDA_CHECK(cudaFree(0));
  printf(
      "VSA block-sparse BACKWARD bench bf16 (blk64-native, ws dual-pack) sm_100a\n"
      "%s 16-warp kernel, mma.ws quads; pre+main+post (block=%d)\n"
      "=====================================\n",
      KERNEL_SCHED == Sched::CLC                 ? "CLC persistent"
      : KERNEL_SCHED == Sched::STATIC_PERSISTENT ? "static persistent"
                                                 : "non-persistent",
      BLOCK);

  // shapes: {B, H, num_blocks, topk, hd, label} (as the forward bench).
  Sh shapes[] = {
      {1, 4, 8, 4, 128, "small"},
      {1, 16, 32, 8, 128, "fastvideo"},
      {1, 8, 64, 16, 128, "25pct"},
      {1, 8, 4096, 1024, 128, "25pct-262k"},
      {1, 8, 8192, 2048, 128, "25pct-524k"},
  };

  constexpr int default_shapes = 3;
  constexpr int num_shapes     = sizeof(shapes) / sizeof(shapes[0]);

  if (const char* s = getenv("SHAPE")) {
    const int shape_index = atoi(s);
    if (shape_index < 0 || shape_index >= num_shapes) {
      fprintf(stderr, "SHAPE=%d out of range [0, %d)\n", shape_index, num_shapes);
      return 1;
    }
    Sh sh = shapes[shape_index];
    if (getenv("BATCH")) sh.B = atoi(getenv("BATCH"));
    if (getenv("HEADS")) sh.H = atoi(getenv("HEADS"));
    if (getenv("NB")) sh.num_blocks = atoi(getenv("NB"));
    if (getenv("TOPK")) sh.topk = atoi(getenv("TOPK"));
    sh.lab = "custom";
    run(sh);
    return 0;
  }
  const int B = getenv("BATCH") ? atoi(getenv("BATCH")) : 1;
  for (int i = 0; i < default_shapes; ++i) {
    Sh sh = shapes[i];
    sh.B  = B;
    run(sh);
  }
  return 0;
}

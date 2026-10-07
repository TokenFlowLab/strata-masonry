#!/usr/bin/env python3
"""Generate the shared block-sparse (VSA) benchmark inputs as .npy files.

Every implementation, and any external comparison, loads these files, so all
consume identical Q/K/V values, top-k block selections and forward state.

Canonical spec:
  H = 8, D = 128, B = 1, bf16, density = 25% (topk = nb/4), MHA (HQ==HK).
  S sweep = 4096 8192 16384 32768 65536 131072 262144 524288.
  Block sizes 64, 128, 256 and 512 (Q/K/V are per-token so shared).

Files written under <outdir>/:
  q_S{S}.npy k_S{S}.npy v_S{S}.npy    dtype uint16 (raw bf16 bits), shape [S, H, D]
  idx_S{S}_blk{BLOCK}.npy             dtype int32, shape [nb, topk]  (head-independent)
  do_S{S}.npy                         dtype uint16 (raw bf16 bits), shape [S, H, D]
  o_S{S}_blk{BLOCK}.npy               dtype uint16 (raw bf16 bits), shape [S, H, D]
  lse_S{S}_blk{BLOCK}.npy             dtype float32, shape [H, S], log2 domain
                                      (max(s * scale * log2e) + log2(sum)), scale = D**-0.5
O and LSE are the real VSA forward state for that block size's selection (fp32
math, no TF32, O rounded to bf16), computed for --fwd-blocks only.

Selection = partial Fisher-Yates over xorshift32 seeded by the q-block index,
sorted ascending -- the same algorithm the kernel drivers use with
VSA_SEED_QBLK + VSA_SORT_SEL.

Requires numpy and torch (the forward state uses the GPU when available).
Run: python3 block_sparse_bf16_gen_inputs.py --outdir inputs
"""
import argparse
import os
import tempfile

import numpy as np
import torch

H = 8
D = 128
DENSITY = 0.25
DEFAULT_SEQLENS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
BLOCKS = [64, 128, 256, 512]
QKV_SEED_BASE = 1000  # per-S seed = QKV_SEED_BASE + S


def fy_select(qblk, nblk, topk):
    """Canonical block selection: partial Fisher-Yates over xorshift32, seeded by
    the q-block index, sorted ascending. Bit-identical to the kernel drivers
    (VSA_SEED_QBLK + VSA_SORT_SEL)."""
    st = (qblk * 2654435761 + 12345) & 0xFFFFFFFF
    perm = list(range(nblk))
    out = []
    for i in range(topk):
        st = (st ^ (st << 13)) & 0xFFFFFFFF
        st = (st ^ (st >> 17)) & 0xFFFFFFFF
        st = (st ^ (st << 5)) & 0xFFFFFFFF
        j = i + st % (nblk - i)
        perm[i], perm[j] = perm[j], perm[i]
        out.append(perm[i])
    return sorted(out)


def existing(path, shape, dtype):
    if not os.path.exists(path):
        return False
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (ValueError, EOFError):
        print(f"  replacing incomplete file {path}", flush=True)
        return False
    if array.shape != shape or array.dtype != np.dtype(dtype) or not array.flags.c_contiguous:
        raise ValueError(f"Existing input has wrong shape, dtype or order: {path}")
    return True


def save_npy(path, array):
    stream = tempfile.NamedTemporaryFile(dir=os.path.dirname(path) or ".",
                                         prefix=os.path.basename(path) + ".", suffix=".tmp",
                                         delete=False)
    try:
        with stream:
            np.save(stream, array)
        os.replace(stream.name, path)
    except BaseException:
        os.unlink(stream.name)
        raise


def dump_qkv(outdir, S):
    paths = [os.path.join(outdir, f"{name}_S{S}.npy") for name in ("q", "k", "v")]
    present = [existing(path, (S, H, D), "uint16") for path in paths]
    if all(present):
        return
    torch.manual_seed(QKV_SEED_BASE + S)
    for path, keep in zip(paths, present):
        # Advance the same random stream even when only one tensor is missing.
        t = torch.randn(S, H, D, dtype=torch.bfloat16)          # [S, H, D], N(0,1)
        bits = t.view(torch.uint16).numpy()                      # raw bf16 bits
        if not keep:
            save_npy(path, bits)


DO_SEED_BASE = 9000  # per-S seed = DO_SEED_BASE + S; separate stream so q/k/v draws stay bit-identical


def dump_do(outdir, S):
    path = os.path.join(outdir, f"do_S{S}.npy")
    if existing(path, (S, H, D), "uint16"):
        return
    torch.manual_seed(DO_SEED_BASE + S)
    t = torch.randn(S, H, D, dtype=torch.bfloat16)               # dO for the backward bench
    save_npy(path, t.view(torch.uint16).numpy())


def dump_idx(outdir, S, block):
    if S <= 0 or block <= 0 or S % (4 * block):
        raise ValueError("S must be a positive multiple of 4 * block for 25% density")
    nb = S // block
    topk = nb // 4
    path = os.path.join(outdir, f"idx_S{S}_blk{block}.npy")
    if existing(path, (nb, topk), "int32"):
        return nb, topk
    idx = np.empty((nb, topk), dtype=np.int32)
    for qb in range(nb):
        idx[qb] = fy_select(qb, nb, topk)
    save_npy(path, idx)
    return nb, topk


def load_heads_major(outdir, name, S, device):
    bits = np.load(os.path.join(outdir, f"{name}_S{S}.npy"), allow_pickle=False)
    t = torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)
    return t.to(device=device, dtype=torch.float32).permute(1, 0, 2).contiguous()


def dump_fwd_state(outdir, S, block, device):
    o_path = os.path.join(outdir, f"o_S{S}_blk{block}.npy")
    lse_path = os.path.join(outdir, f"lse_S{S}_blk{block}.npy")
    have_o = existing(o_path, (S, H, D), "uint16")
    have_lse = existing(lse_path, (H, S), "float32")
    if have_o and have_lse:
        return
    q, k, v = (load_heads_major(outdir, name, S, device) for name in ("q", "k", "v"))
    idx = torch.from_numpy(np.load(os.path.join(outdir, f"idx_S{S}_blk{block}.npy")))
    idx = idx.to(device=device, dtype=torch.long)
    nb, topk = idx.shape
    keys = topk * block
    scale_log2 = torch.tensor((1.0 / np.sqrt(D)) * np.log2(np.e), dtype=torch.float32)
    offsets = torch.arange(block, device=device)
    o = torch.empty(H, S, D, dtype=torch.float32, device=device)
    lse = torch.empty(H, S, dtype=torch.float32, device=device)
    chunk = max(1, (1 << 30) // (H * keys * (2 * D + block)))
    for first in range(0, nb, chunk):
        last = min(nb, first + chunk)
        n = last - first
        tokens = (idx[first:last, :, None] * block + offsets).reshape(n, keys)
        k_sel = k[:, tokens]
        v_sel = v[:, tokens]
        q_blk = q[:, first * block:last * block].reshape(H, n, block, D)
        s = torch.matmul(q_blk, k_sel.transpose(-1, -2)) * scale_log2.to(device)
        m = s.amax(dim=-1, keepdim=True)
        p = torch.exp2(s - m)
        l = p.sum(dim=-1, keepdim=True)
        o[:, first * block:last * block] = torch.matmul(p / l, v_sel).reshape(H, n * block, D)
        lse[:, first * block:last * block] = (m + torch.log2(l)).reshape(H, n * block)
    o_bits = o.permute(1, 0, 2).contiguous().to(torch.bfloat16).view(torch.int16).cpu().numpy()
    save_npy(o_path, o_bits.view(np.uint16))
    save_npy(lse_path, lse.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="inputs")
    ap.add_argument("--seqlens", type=int, nargs="+", default=DEFAULT_SEQLENS)
    ap.add_argument("--blocks", type=int, nargs="+", default=BLOCKS)
    ap.add_argument("--fwd-blocks", type=int, nargs="*", default=[64, 128, 256])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if any(s <= 0 or b <= 0 or s % (4 * b) for s in args.seqlens for b in args.blocks):
        ap.error("every sequence length must be a positive multiple of 4 * every block size")
    if not set(args.fwd_blocks) <= set(args.blocks):
        ap.error("--fwd-blocks must be a subset of --blocks")
    torch.backends.cuda.matmul.allow_tf32 = False
    os.makedirs(args.outdir, exist_ok=True)
    print(f"block_sparse_bf16_gen_inputs: H={H} D={D} B=1 bf16 density={DENSITY} -> {os.path.abspath(args.outdir)}")
    for S in args.seqlens:
        dump_qkv(args.outdir, S)
        dump_do(args.outdir, S)
        line = f"  S={S:<7}"
        for blk in args.blocks:
            nb, topk = dump_idx(args.outdir, S, blk)
            line += f"  blk{blk}: nb={nb} topk={topk}"
        print(line, flush=True)
        for blk in args.fwd_blocks:
            dump_fwd_state(args.outdir, S, blk, args.device)
            print(f"    fwd state blk{blk}: o + lse", flush=True)
    print("done.")


if __name__ == "__main__":
    main()

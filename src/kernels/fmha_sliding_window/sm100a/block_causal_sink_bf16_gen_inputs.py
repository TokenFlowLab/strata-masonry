#!/usr/bin/env python3
"""Generate the shared block-causal + sink + sliding-window benchmark inputs as .npy
files. Every implementation, and any external comparison, loads these files.

Canonical config (Wan 2.1 T2V 1.3B causal):
  H = 12, D = 128, B = 1, bf16.
  tokens_per_frame = 1456, num_frame_per_block = 3, sink_size = 1,
  local_attn_size = 6 (frames; window incl. sink), kind = blockwise.
  num_frames sweep -> L = num_frames * tokens_per_frame (blockwise);
  teacher_forcing doubles L.

Files written under <outdir>/:
  q_L{L}.npy k_L{L}.npy v_L{L}.npy   dtype uint16 (raw bf16 bits), shape [L, H, D]
  plan_L{L}.json                     the scalar attention-plan params

Requires numpy and torch.
Run: python3 block_causal_sink_bf16_gen_inputs.py --outdir inputs            (blockwise)
     python3 block_causal_sink_bf16_gen_inputs.py --outdir inputs_tf --kind teacher_forcing
"""
import argparse
import json
import os

import numpy as np
import torch

H = 12
D = 128
TOKENS_PER_FRAME = 1456
NUM_FRAME_PER_BLOCK = 3
SINK_SIZE = 1
LOCAL_ATTN_SIZE = 6
KIND = "blockwise"
QKV_SEED_BASE = 2000  # per-L seed = QKV_SEED_BASE + num_frames
DEFAULT_NUM_FRAMES = [6, 12, 18, 36, 72]  # L = 8736 .. 104832 (blockwise)


def dump_qkv(outdir, L, num_frames):
    torch.manual_seed(QKV_SEED_BASE + num_frames)
    for name in ("q", "k", "v"):
        t = torch.randn(L, H, D, dtype=torch.bfloat16)   # [L, H, D], N(0,1)
        bits = t.view(torch.uint16).numpy()
        np.save(os.path.join(outdir, f"{name}_L{L}.npy"), bits)


def dump_plan(outdir, L, num_frames, kind):
    plan = {
        "kind": kind,
        "num_frames": num_frames,          # per half for teacher_forcing
        "frame_seqlen": TOKENS_PER_FRAME,
        "num_frame_per_block": NUM_FRAME_PER_BLOCK,
        "local_attn_size": LOCAL_ATTN_SIZE,
        "sink_size": SINK_SIZE,
        "H": H,
        "D": D,
        "sm_scale": 1.0 / (D ** 0.5),
        "L": L,
    }
    with open(os.path.join(outdir, f"plan_L{L}.json"), "w") as f:
        json.dump(plan, f, indent=2)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="inputs")
    ap.add_argument("--num-frames", type=int, nargs="+", default=DEFAULT_NUM_FRAMES)
    ap.add_argument("--kind", choices=["blockwise", "teacher_forcing"], default=KIND)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    halves = 2 if args.kind == "teacher_forcing" else 1
    print(f"gen_inputs: H={H} D={D} B=1 bf16 tokens_per_frame={TOKENS_PER_FRAME} "
          f"nfpb={NUM_FRAME_PER_BLOCK} sink={SINK_SIZE} window={LOCAL_ATTN_SIZE} "
          f"kind={args.kind} -> {os.path.abspath(args.outdir)}")
    for nf in args.num_frames:
        L = nf * TOKENS_PER_FRAME * halves
        dump_qkv(args.outdir, L, nf)
        dump_plan(args.outdir, L, nf, args.kind)
        print(f"  num_frames={nf:<4} L={L}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()

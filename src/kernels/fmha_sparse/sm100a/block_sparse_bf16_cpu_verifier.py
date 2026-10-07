#!/usr/bin/env python3
"""Independent CPU reference for the shared-input block-sparse BF16 FMHA grid.

Reproduces seeded inputs or reads the actual Q/K/V and per-block selection files.
Full verification is the default. Chunked online softmax avoids an S-by-S allocation.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

SEQLENS = (4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288)
BLOCKS = (64, 128, 256, 512)
HEADS, DIM = 8, 128


def bf16_to_f32(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def generated_bits(indices, tensor_seed, input_seed):
    x = (np.asarray(indices, dtype=np.uint32) * np.uint32(2654435761)
         + np.uint32(((tensor_seed + input_seed) * 40503 + 0x9e3779b9) & 0xffffffff))
    x ^= x >> 15
    x *= np.uint32(2246822519)
    x ^= x >> 13
    x *= np.uint32(3266489917)
    x ^= x >> 16
    values = (x % 2039).astype(np.float32) / np.float32(1019.5) - np.float32(1)
    bits = values.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


class SeededTensor:
    """Generate only requested CPU rows, using the kernel's flat [S,H,D] indices."""
    def __init__(self, seqlen, tensor_seed, seed):
        self.shape = (seqlen, HEADS, DIM)
        self.tensor_seed, self.seed = tensor_seed, seed

    def __getitem__(self, key):
        tokens, head = key
        if isinstance(tokens, slice):
            tokens = np.arange(*tokens.indices(self.shape[0]), dtype=np.uint32)
        indices = ((np.asarray(tokens, dtype=np.uint32)[:, None] * HEADS + head) * DIM
                   + np.arange(DIM, dtype=np.uint32))
        return generated_bits(indices, self.tensor_seed, self.seed)


class SeededIndices:
    def __init__(self, nb, topk, seed):
        self.nb, self.topk, self.seed = nb, topk, seed
        self.cache = {}

    def __getitem__(self, qb):
        qb = int(qb)
        if qb not in self.cache:
            permutation = list(range(self.nb))
            state = (qb * 2654435761 + 12345 + self.seed) & 0xffffffff
            selected = []
            for i in range(self.topk):
                state = (state ^ (state << 13)) & 0xffffffff
                state = (state ^ (state >> 17)) & 0xffffffff
                state = (state ^ (state << 5)) & 0xffffffff
                j = i + state % (self.nb - i)
                permutation[i], permutation[j] = permutation[j], permutation[i]
                selected.append(permutation[i])
            self.cache[qb] = np.array(sorted(selected), dtype=np.int32)
        return self.cache[qb]


def load_inputs(directory, seqlen, block, seed=0):
    if directory is None:
        return (*(SeededTensor(seqlen, tensor_seed, seed) for tensor_seed in (11, 22, 33)),
                SeededIndices(seqlen // block, seqlen // (4 * block), seed))
    directory = Path(directory)
    tensors = []
    for name in ("q", "k", "v"):
        array = np.load(directory / f"{name}_S{seqlen}.npy", mmap_mode="r", allow_pickle=False)
        if array.shape != (seqlen, HEADS, DIM) or array.dtype != np.dtype("<u2"):
            raise ValueError(f"{name}: expected uint16 BF16 bits [{seqlen},{HEADS},{DIM}]")
        if not array.flags.c_contiguous:
            raise ValueError(f"{name}: expected C-order input")
        tensors.append(array)
    nb, topk = seqlen // block, seqlen // (4 * block)
    idx = np.load(directory / f"idx_S{seqlen}_blk{block}.npy", mmap_mode="r",
                  allow_pickle=False)
    if idx.shape != (nb, topk) or idx.dtype != np.dtype("<i4") or not idx.flags.c_contiguous:
        raise ValueError(f"indices: expected C-order int32 [{nb},{topk}]")
    for row in idx:
        if np.any(row < 0) or np.any(row >= nb) or np.any(row[1:] <= row[:-1]):
            raise ValueError("indices must be in range, distinct and sorted in each query block")
    return (*tensors, idx)


def attention_rows(q, k, v, indices, block, key_chunk):
    """FP32 QK, stable online softmax and PV on CPU; no BF16 intermediate rounding."""
    selected = (np.asarray(indices, dtype=np.int64)[:, None] * block
                + np.arange(block, dtype=np.int64)).ravel()
    maximum = np.full((len(q), 1), -np.inf, dtype=np.float32)
    denominator = np.zeros((len(q), 1), dtype=np.float32)
    numerator = np.zeros_like(q, dtype=np.float32)
    scale = np.float32(1 / math.sqrt(q.shape[1]))
    for start in range(0, len(selected), key_chunk):
        tokens = selected[start:start + key_chunk]
        scores = (q @ k[tokens].T) * scale
        next_maximum = np.maximum(maximum, scores.max(axis=1, keepdims=True))
        correction = np.exp(maximum - next_maximum)
        probabilities = np.exp(scores - next_maximum)
        numerator = numerator * correction + probabilities @ v[tokens]
        denominator = denominator * correction + probabilities.sum(axis=1, keepdims=True)
        maximum = next_maximum
    return numerator / denominator


def reference_chunks(tensors, block, query_chunk=128, key_chunk=2048, coordinates=None):
    """Yield (query token IDs, head, reference rows), covering all D output values."""
    q_bits, k_bits, v_bits, idx = tensors
    seqlen, heads, _ = q_bits.shape
    for head in range(heads):
        rows = None if coordinates is None else coordinates[coordinates[:, 1] == head, 0]
        if rows is not None and not len(rows):
            continue
        k, v = bf16_to_f32(k_bits[:, head]), bf16_to_f32(v_bits[:, head])
        if not np.isfinite(k).all() or not np.isfinite(v).all():
            raise ValueError("K/V contain non-finite values")
        blocks = range(seqlen // block) if rows is None else np.unique(rows // block)
        for qb in blocks:
            tokens = (np.arange(qb * block, (qb + 1) * block) if rows is None
                      else rows[rows // block == qb])
            for start in range(0, len(tokens), query_chunk):
                positions = tokens[start:start + query_chunk]
                q = bf16_to_f32(q_bits[positions, head])
                if not np.isfinite(q).all():
                    raise ValueError("Q contains non-finite values")
                yield positions, head, attention_rows(q, k, v, idx[qb], block, key_chunk)


def verify(args):
    tensors = load_inputs(args.inputs, args.seqlen, args.block_size, args.seed)
    shape = (args.seqlen, HEADS, DIM)
    actual_path = Path(args.actual)
    if actual_path.stat().st_size != math.prod(shape) * 4:
        raise ValueError(f"actual: expected {math.prod(shape)} raw FP32 elements in [S,H,D] order")
    actual = np.memmap(actual_path, mode="r", dtype="<f4", shape=shape)
    coordinates = None
    if args.mode == "sampled":
        count = min(args.samples, args.seqlen * HEADS)
        chosen = np.random.default_rng(args.sample_seed).choice(args.seqlen * HEADS, count, replace=False)
        coordinates = np.column_stack((chosen // HEADS, chosen % HEADS))
    checked = mismatches = nonfinite = 0
    max_error = 0.0
    for tokens, head, expected in reference_chunks(tensors, args.block_size,
                                                  args.query_chunk, args.key_chunk, coordinates):
        observed = actual[tokens, head]
        finite = np.isfinite(observed) & np.isfinite(expected)
        error = np.abs(observed - expected)
        # Standard absolute-plus-relative tolerance; every non-finite value fails.
        mismatches += int(np.count_nonzero(~finite | (error > args.atol + args.rtol * np.abs(expected))))
        nonfinite += int(np.count_nonzero(~finite))
        if finite.any():
            max_error = max(max_error, float(error[finite].max()))
        checked += observed.size
    return {
        "passed": mismatches == 0, "mode": args.mode,
        "seqlen": args.seqlen, "block_size": args.block_size,
        "batch": 1, "heads": HEADS, "head_dim": DIM,
        "topk": args.seqlen // (4 * args.block_size),
        "inputs": ({"mode": "files", "directory": str(Path(args.inputs).resolve())} if args.inputs
                   else {"mode": "seeded", "seed": args.seed, "generator": "hash32-fy-v1"}),
        "actual": str(actual_path.resolve()),
        "actual_dtype": "float32", "checked_elements": checked,
        "total_elements": math.prod(shape), "mismatches": mismatches,
        "nonfinite": nonfinite, "max_abs_error": max_error,
        "atol": args.atol, "rtol": args.rtol,
    }


def self_test():
    # Small arithmetic fixtures are internal reference tests, not new workload cases.
    rng = np.random.default_rng(71)
    q, k, v = [rng.standard_normal((16, 3, 8)).astype(np.float32) for _ in range(3)]
    bits = [(x.view(np.uint32) >> 16).astype(np.uint16) for x in (q, k, v)]
    q, k, v = [bf16_to_f32(x) for x in bits]
    idx = np.array([[1, 3], [0, 2], [0, 3], [1, 2]], dtype=np.int32)
    reference = np.empty_like(q)
    for head in range(3):
        scores = q[:, head].astype(np.float64) @ k[:, head].astype(np.float64).T / math.sqrt(8)
        mask = np.zeros((16, 16), dtype=bool)
        for qb, row in enumerate(idx):
            for kb in row:
                mask[qb * 4:(qb + 1) * 4, kb * 4:(kb + 1) * 4] = True
        scores[~mask] = -np.inf
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        reference[:, head] = (weights / weights.sum(axis=1, keepdims=True)) @ v[:, head]
    for key_chunk in (1, 3, 32):
        result = np.empty_like(q)
        for tokens, head, values in reference_chunks((*bits, idx), 4, 3, key_chunk):
            result[tokens, head] = values
        np.testing.assert_allclose(result, reference, atol=1e-6, rtol=1e-5)
    coords = np.array([[0, 0], [9, 2], [15, 1]])
    for tokens, head, values in reference_chunks((*bits, idx), 4, 2, 3, coords):
        np.testing.assert_allclose(values, reference[tokens, head], atol=1e-6, rtol=1e-5)
    print("CPU reference self-test: PASS (full, sampled, chunked vs dense FP64 oracle)")


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def tolerance(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return result


def seed_value(value):
    result = int(value)
    if not 0 <= result <= 2147483647:
        raise argparse.ArgumentTypeError("must be between 0 and 2147483647")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("cases")
    commands.add_parser("self-test")
    for name in ("verify", "reference"):
        command = commands.add_parser(name)
        command.add_argument("--inputs", type=Path, help="optional .npy directory; otherwise use seeded inputs")
        command.add_argument("--seed", type=seed_value, default=0, help="kernel INPUT_SEED (default 0)")
        command.add_argument("--seqlen", type=int, choices=SEQLENS, required=True)
        command.add_argument("--block-size", type=int, choices=BLOCKS, required=True)
        command.add_argument("--query-chunk", type=positive_int, default=128)
        command.add_argument("--key-chunk", type=positive_int, default=2048)
        if name == "reference":
            command.add_argument("--output", type=Path, required=True, help="raw FP32 [S,H,D]")
        else:
            command.add_argument("--actual", type=Path, required=True, help="DUMP_O's .out file")
            command.add_argument("--mode", choices=("full", "sampled"), default="full")
            command.add_argument("--samples", type=positive_int, default=64, help="rows in sampled mode")
            command.add_argument("--sample-seed", type=seed_value, default=0)
            command.add_argument("--atol", type=tolerance, default=0.002)
            command.add_argument("--rtol", type=tolerance, default=0.02)
            command.add_argument("--result", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "cases":
            print("B=1 Hq=Hkv=8 D=128 BF16 non-causal; Sq=Skv=S; density=25%")
            print("S       block  num_blocks  topk")
            for s in SEQLENS:
                for block in BLOCKS:
                    print(f"{s:<7} {block:<6} {s // block:<11} {s // (4 * block)}")
            return 0
        if args.command == "self-test":
            self_test()
            return 0
        if args.command == "reference":
            tensors = load_inputs(args.inputs, args.seqlen, args.block_size, args.seed)
            # Never overwrite an existing reference or input file.
            with args.output.open("xb") as stream:
                stream.truncate(args.seqlen * HEADS * DIM * 4)
            result = np.memmap(args.output, mode="r+", dtype="<f4", shape=(args.seqlen, HEADS, DIM))
            for tokens, head, values in reference_chunks(tensors, args.block_size,
                                                        args.query_chunk, args.key_chunk):
                result[tokens, head] = values
            result.flush()
            print(f"Wrote full FP32 CPU reference: {args.output}")
            return 0
        result = verify(args)
        output = json.dumps(result, indent=2, allow_nan=False)
        if args.result:
            args.result.write_text(output + "\n")
        print(output)
        return 0 if result["passed"] else 1
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

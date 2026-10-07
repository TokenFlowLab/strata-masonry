#!/usr/bin/env python3
"""CPU-only verification of block-sparse BF16 attention backward (dQ, dK, dV).

Read the actual saved Q/K/V/dO, indices, BF16 O and FP32 log2 LSE. Never
regenerate forward state or use GPU outputs as reference inputs. Full checking
is the default; --mode sampled checks selected rows of each gradient.

NumPy FP32 products mirror the documented BF16 P/dS quantization points.
Chunking bounds score memory; no dense S-by-S attention matrix is allocated.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

SEQLENS = (4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288)
BLOCKS = (64, 128, 256)
HEADS, DIM = 8, 128
GRADIENTS = ("dq", "dk", "dv")
LOG2E = np.float32(math.log2(math.e))


def bf16_to_f32(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)


def round_bf16(values):
    """Round finite FP32 values to nearest-even BF16, returning FP32 values."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def load_array(path, shape, dtype):
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (array.shape != shape or array.dtype != np.dtype(dtype)
            or not array.flags.c_contiguous):
        raise ValueError(f"{path}: expected C-order {dtype} {shape}")
    return array


def load_inputs(directory, seqlen, block):
    shape = (seqlen, HEADS, DIM)
    arrays = {name: load_array(directory / f"{name}_S{seqlen}.npy", shape, "<u2")
              for name in ("q", "k", "v", "do")}
    arrays["o"] = load_array(directory / f"o_S{seqlen}_blk{block}.npy", shape, "<u2")
    arrays["lse"] = load_array(directory / f"lse_S{seqlen}_blk{block}.npy",
                               (HEADS, seqlen), "<f4")
    nb, topk = seqlen // block, seqlen // (4 * block)
    indices = load_array(directory / f"idx_S{seqlen}_blk{block}.npy", (nb, topk), "<i4")
    for row in indices:
        if np.any(row < 0) or np.any(row >= nb) or np.any(row[1:] <= row[:-1]):
            raise ValueError("indices must be distinct, sorted and in range in every row")
    return arrays, indices


def token_chunks(block_ids, block, chunk_size):
    tokens = (np.asarray(block_ids, dtype=np.int64)[:, None] * block
              + np.arange(block, dtype=np.int64)).ravel()
    for start in range(0, len(tokens), chunk_size):
        yield tokens[start:start + chunk_size]


def coefficients(q, k, v, dout, lse, delta):
    scale = np.float32(1.0 / math.sqrt(q.shape[1]))
    # Scale after QK, never pre-scale BF16 K. P is FP32 when forming dS.
    p = np.exp2((q @ k.T) * scale * LOG2E - lse[:, None])
    ds = round_bf16(p * ((dout @ v.T) - delta[:, None]))
    return round_bf16(p), ds


def full_head(q, k, v, dout, o, lse, indices, block, query_chunk, key_chunk):
    """Accumulate every gradient for one head using the actual saved state."""
    dq, dk, dv = (np.zeros_like(q) for _ in GRADIENTS)
    delta = np.sum(o * dout, axis=1, dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(q.shape[1]))
    for qb, selected in enumerate(indices):
        for first in range(qb * block, (qb + 1) * block, query_chunk):
            rows = slice(first, min((qb + 1) * block, first + query_chunk))
            for tokens in token_chunks(selected, block, key_chunk):
                p, ds = coefficients(q[rows], k[tokens], v[tokens], dout[rows],
                                     lse[rows], delta[rows])
                dq[rows] += ds @ k[tokens]
                dk[tokens] += ds.T @ q[rows]
                dv[tokens] += p.T @ dout[rows]
    dq *= scale
    dk *= scale
    return dq, dk, dv


def sampled_head(q, k, v, dout, o, lse, indices, block, rows, query_chunk, key_chunk):
    """Check selected dQ rows and selected dK/dV rows, including all contributors."""
    dq, dk, dv = (np.zeros((len(rows), q.shape[1]), dtype=np.float32) for _ in GRADIENTS)
    delta = np.sum(o * dout, axis=1, dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(q.shape[1]))
    for qb in np.unique(rows // block):
        positions = np.flatnonzero(rows // block == qb)
        for start in range(0, len(positions), query_chunk):
            out_rows = positions[start:start + query_chunk]
            q_rows = rows[out_rows]
            for tokens in token_chunks(indices[qb], block, key_chunk):
                _, ds = coefficients(q[q_rows], k[tokens], v[tokens], dout[q_rows],
                                      lse[q_rows], delta[q_rows])
                dq[out_rows] += ds @ k[tokens]
    for kb in np.unique(rows // block):
        positions = np.flatnonzero(rows // block == kb)
        # A sampled KV row needs every query block selecting it, not sampled Q rows.
        query_blocks = np.flatnonzero(np.any(indices == kb, axis=1))
        for start in range(0, len(positions), key_chunk):
            out_rows = positions[start:start + key_chunk]
            kv_rows = rows[out_rows]
            for tokens in token_chunks(query_blocks, block, query_chunk):
                p, ds = coefficients(q[tokens], k[kv_rows], v[kv_rows], dout[tokens],
                                     lse[tokens], delta[tokens])
                dk[out_rows] += ds.T @ q[tokens]
                dv[out_rows] += p.T @ dout[tokens]
    dq *= scale
    dk *= scale
    return dq, dk, dv


def reference_chunks(arrays, indices, block, query_chunk=128, key_chunk=2048,
                     coordinates=None):
    seqlen, heads, _ = arrays["q"].shape
    for head in range(heads):
        rows = (np.arange(seqlen) if coordinates is None
                else coordinates[coordinates[:, 1] == head, 0])
        if not len(rows):
            continue
        q, k, v, dout, o = (bf16_to_f32(arrays[name][:, head])
                            for name in ("q", "k", "v", "do", "o"))
        lse = np.asarray(arrays["lse"][head])
        if not all(np.isfinite(x).all() for x in (q, k, v, dout, o, lse)):
            raise ValueError(f"non-finite input or saved forward state in head {head}")
        if coordinates is None:
            gradients = full_head(q, k, v, dout, o, lse, indices, block,
                                  query_chunk, key_chunk)
        else:
            gradients = sampled_head(q, k, v, dout, o, lse, indices, block, rows,
                                     query_chunk, key_chunk)
        for start in range(0, len(rows), query_chunk):
            end = start + query_chunk
            yield rows[start:end], head, tuple(g[start:end] for g in gradients)


def new_metrics():
    return {"checked_elements": 0, "mismatches": 0, "nonfinite": 0,
            "max_abs_error": 0.0, "max_abs_reference": 0.0}


def accumulate_metrics(metrics, observed, expected, atol, rtol):
    finite = np.isfinite(observed) & np.isfinite(expected)
    error = np.zeros_like(expected)
    np.subtract(observed, expected, out=error, where=finite)
    np.abs(error, out=error)
    metrics["checked_elements"] += expected.size
    metrics["nonfinite"] += int(np.count_nonzero(~finite))
    metrics["mismatches"] += int(np.count_nonzero(
        ~finite | (error > atol + rtol * np.abs(expected))))
    if finite.any():
        metrics["max_abs_error"] = max(metrics["max_abs_error"], float(error[finite].max()))
    finite_reference = np.isfinite(expected)
    if finite_reference.any():
        metrics["max_abs_reference"] = max(metrics["max_abs_reference"],
                                           float(np.abs(expected[finite_reference]).max()))


def verify(args):
    arrays, indices = load_inputs(args.inputs, args.seqlen, args.block_size)
    shape = (args.seqlen, HEADS, DIM)
    paths = {name: Path(f"{args.actual_prefix}_{name}.npy") for name in GRADIENTS}
    actual = {name: load_array(path, shape, "<f4") for name, path in paths.items()}
    coordinates = None
    if args.mode == "sampled":
        count = min(args.samples, args.seqlen * HEADS)
        chosen = np.random.default_rng(args.sample_seed).choice(args.seqlen * HEADS,
                                                               count, replace=False)
        coordinates = np.column_stack((chosen // HEADS, chosen % HEADS))
    metrics = {name: new_metrics() for name in GRADIENTS}
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for rows, head, expected in reference_chunks(arrays, indices, args.block_size,
                                                     args.query_chunk, args.key_chunk,
                                                     coordinates):
            for name, values in zip(GRADIENTS, expected):
                accumulate_metrics(metrics[name], actual[name][rows, head], values,
                                   args.atol, args.rtol)
    for item in metrics.values():
        item["passed"] = item["mismatches"] == 0
        item["total_elements"] = math.prod(shape)
    return {
        "passed": all(item["passed"] for item in metrics.values()),
        "mode": args.mode, "seqlen": args.seqlen, "block_size": args.block_size,
        "batch": 1, "heads": HEADS, "head_dim": DIM, "topk": indices.shape[1],
        "inputs": str(args.inputs.resolve()), "lse_domain": "log2",
        "saved_output": str((args.inputs / f"o_S{args.seqlen}_blk{args.block_size}.npy").resolve()),
        "saved_lse": str((args.inputs / f"lse_S{args.seqlen}_blk{args.block_size}.npy").resolve()),
        "actual": {name: str(path.resolve()) for name, path in paths.items()},
        "reference": "CPU FP32; Delta from saved BF16 O; BF16 P and dS operands",
        "query_chunk": args.query_chunk, "key_chunk": args.key_chunk,
        "sample_seed": args.sample_seed if coordinates is not None else None,
        "sampled_rows_per_gradient": len(coordinates) if coordinates is not None else None,
        "atol": args.atol, "rtol": args.rtol, "gradients": metrics,
        "checked_elements": sum(item["checked_elements"] for item in metrics.values()),
        "total_elements": 3 * math.prod(shape),
    }


def self_test():
    # Internal arithmetic fixture, not an additional benchmark workload.
    rng = np.random.default_rng(73)
    seqlen, heads, dim, block = 16, 2, 8, 4
    values = {name: round_bf16(rng.standard_normal((seqlen, heads, dim)).astype(np.float32))
              for name in ("q", "k", "v", "do")}
    indices = np.array([[0, 1], [0, 2], [0, 1], [1, 2]], dtype=np.int32)
    # KV block 3 is deliberately never selected: its dK/dV must be zero.
    mask = np.zeros((seqlen, seqlen), dtype=bool)
    for qb, selected in enumerate(indices):
        for kb in selected:
            mask[qb * block:(qb + 1) * block, kb * block:(kb + 1) * block] = True
    values["o"] = np.empty_like(values["q"])
    lse = np.empty((heads, seqlen), dtype=np.float32)
    expected = [np.empty_like(values["q"]) for _ in GRADIENTS]
    for head in range(heads):
        q, k, v, dout = (values[name][:, head].astype(np.float64)
                         for name in ("q", "k", "v", "do"))
        scores = q @ k.T / math.sqrt(dim)
        scores[~mask] = -np.inf
        maximum = scores.max(axis=1, keepdims=True)
        weights = np.exp(scores - maximum)
        sums = weights.sum(axis=1, keepdims=True)
        values["o"][:, head] = round_bf16((weights / sums) @ v)
        lse[head] = ((maximum + np.log(sums)) * math.log2(math.e)).ravel()
        # Independent dense FP64 formula, consuming the rounded saved O/LSE.
        p = np.exp2(scores * math.log2(math.e) - lse[head, :, None])
        delta = np.sum(values["o"][:, head].astype(np.float64) * dout, axis=1)
        ds = round_bf16(p * (dout @ v.T - delta[:, None])).astype(np.float64)
        expected[0][:, head] = ds @ k / math.sqrt(dim)
        expected[1][:, head] = ds.T @ q / math.sqrt(dim)
        expected[2][:, head] = round_bf16(p).astype(np.float64).T @ dout
    arrays = {name: (value.view(np.uint32) >> 16).astype(np.uint16)
              for name, value in values.items()}
    arrays["lse"] = lse
    coordinates = np.array([[0, 0], [9, 1], [15, 0], [12, 1]])
    for query_chunk, key_chunk in ((1, 1), (3, 5), (8, 32)):
        full = [np.empty_like(values["q"]) for _ in GRADIENTS]
        for rows, head, gradients in reference_chunks(arrays, indices, block,
                                                       query_chunk, key_chunk):
            for name, actual, reference, gradient in zip(GRADIENTS, full, expected, gradients):
                actual[rows, head] = gradient
                np.testing.assert_allclose(gradient, reference[rows, head], atol=2e-6,
                                           rtol=2e-5, err_msg=name)
        for rows, head, gradients in reference_chunks(arrays, indices, block,
                                                       query_chunk, key_chunk, coordinates):
            for actual, gradient in zip(full, gradients):
                np.testing.assert_allclose(gradient, actual[rows, head], atol=2e-6, rtol=2e-5)
        np.testing.assert_array_equal(full[1][12:], 0)
        np.testing.assert_array_equal(full[2][12:], 0)
    for bad in (np.float32(100), np.float32(np.nan), np.float32(np.inf)):
        metrics = new_metrics()
        accumulate_metrics(metrics, np.array([bad]), np.zeros(1, dtype=np.float32), .002, .02)
        assert metrics["mismatches"] == 1
    print("PASS: CPU backward full/sampled/chunked vs dense FP64; zero-count KV; bad outputs")


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def tolerance(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    commands.add_parser("cases")
    command = commands.add_parser("verify")
    command.add_argument("--inputs", type=Path, required=True, help="shared .npy input directory")
    command.add_argument("--seqlen", type=positive_int, required=True)
    command.add_argument("--block-size", type=int, choices=BLOCKS, required=True)
    command.add_argument("--actual-prefix", type=Path, required=True,
                         help="DUMP_BWD_GPU prefix: reads PREFIX_{dq,dk,dv}.npy")
    command.add_argument("--mode", choices=("full", "sampled"), default="full")
    command.add_argument("--samples", type=positive_int, default=64,
                         help="number of (token,head) rows per gradient in sampled mode")
    command.add_argument("--sample-seed", type=nonnegative_int, default=0)
    command.add_argument("--query-chunk", type=positive_int, default=128)
    command.add_argument("--key-chunk", type=positive_int, default=2048)
    command.add_argument("--atol", type=tolerance, default=0.002)
    command.add_argument("--rtol", type=tolerance, default=0.02)
    command.add_argument("--result", type=Path, help="optional JSON report; will not overwrite")
    args = parser.parse_args()
    try:
        if args.command == "self-test":
            self_test()
            return 0
        if args.command == "cases":
            print("B=1 H=8 D=128 BF16 non-causal backward, density=25%")
            print("S       block  num_blocks  topk")
            for seqlen in SEQLENS:
                for block in BLOCKS:
                    print(f"{seqlen:<7} {block:<6} {seqlen // block:<11} {seqlen // (4 * block)}")
            print("Small checks: S=1024 blk64/blk256, S=2048 blk128 (not performance cases)")
            return 0
        if args.seqlen % (4 * args.block_size):
            raise ValueError("sequence length must be a multiple of 4 * block size (25% density)")
        report = verify(args)
        text = json.dumps(report, indent=2, allow_nan=False)
        if args.result:
            with args.result.open("x") as stream:
                stream.write(text + "\n")
        print(text)
        return 0 if report["passed"] else 1
    except (OSError, ValueError, FloatingPointError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())

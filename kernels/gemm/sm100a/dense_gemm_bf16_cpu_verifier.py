#!/usr/bin/env python3

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


CASES = {
    "smoke": {"m": 256, "k": 64, "n": 128, "production_m": 256},
    "und-gate": {"m": 14080, "k": 2048, "n": 128, "production_m": 14046},
    "und-qkv": {"m": 14080, "k": 2048, "n": 5120, "production_m": 14046},
    "und-o": {"m": 14080, "k": 4096, "n": 2048, "production_m": 14046},
    "gen-gate": {"m": 30720, "k": 2048, "n": 128, "production_m": 30720},
    "gen-qkv": {"m": 30720, "k": 2048, "n": 5120, "production_m": 30720},
    "gen-o": {"m": 30720, "k": 4096, "n": 2048, "production_m": 30720},
}


def bf16_to_f32(values):
    return (values.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2")


def fill(indices, mode, seed):
    x = (np.asarray(indices, dtype=np.uint64) * 2654435761 + seed) & 0xFFFFFFFF
    if mode == 1:
        values = (x % 256).astype(np.float32) / 256.0 - 0.5
    elif mode == 2:
        values = (x % 2048).astype(np.float32) / 1024.0 - 1.0
    elif mode == 3:
        values = np.ones(x.shape, dtype=np.float32)
    elif mode == 4:
        values = (x % 7).astype(np.float32) - 3.0
    else:
        raise ValueError("fill mode must be 1, 2, 3, or 4")
    return bf16_to_f32(f32_to_bf16(values))


def sample_coordinates(m, n, count, seed):
    total = m * n
    fixed = np.array([0, n - 1, (m // 2) * n + n // 2, total - n, total - 1])
    if count >= total:
        flat = np.arange(total)
    else:
        rng = np.random.default_rng(seed)
        random = rng.integers(0, total, size=max(0, count - fixed.size))
        flat = np.unique(np.concatenate((fixed, random)))
    return flat // n, flat % n


def references(case, rows, cols, mode):
    m, k, n = case["m"], case["k"], case["n"]
    del m
    k_indices = np.arange(k, dtype=np.uint64)
    result = np.empty(rows.size, dtype=np.float32)
    for i, (row, col) in enumerate(zip(rows, cols, strict=True)):
        a = fill(np.uint64(row) * k + k_indices, mode, 1)
        b = fill(k_indices * n + np.uint64(col), mode, 2)
        result[i] = np.dot(a, b)
    return result


def verify(case_name, actual_path, samples, mode, atol, rtol, seed):
    case = CASES[case_name]
    expected_elements = case["m"] * case["n"]
    actual = np.memmap(actual_path, dtype="<u2", mode="r")
    if actual.size != expected_elements:
        raise ValueError(
            f"expected {expected_elements} BF16 values, found {actual.size} in {actual_path}"
        )
    rows, cols = sample_coordinates(case["m"], case["n"], samples, seed)
    observed = bf16_to_f32(actual[rows * case["n"] + cols])
    reference = references(case, rows, cols, mode)
    error = np.abs(observed - reference)
    mismatch = ~np.isfinite(observed) | ((error > atol) & (error > rtol * np.abs(reference)))
    return {
        "schema": "kernelbridge.cpu-verifier-result/v0",
        "family": "gemm/sm100a",
        "case": case_name,
        "passed": not bool(np.any(mismatch)),
        "samples": int(rows.size),
        "mismatches": int(np.count_nonzero(mismatch)),
        "max_abs_error": float(np.max(error)),
        "atol": atol,
        "rtol": rtol,
        "fill": mode,
        "actual": str(Path(actual_path).resolve()),
    }


def write_smoke_reference(path, mode):
    case = CASES["smoke"]
    a = fill(np.arange(case["m"] * case["k"]), mode, 1).reshape(case["m"], case["k"])
    b = fill(np.arange(case["k"] * case["n"]), mode, 2).reshape(case["k"], case["n"])
    f32_to_bf16(a @ b).tofile(path)


def emit(result, result_path=None):
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if result_path:
        Path(result_path).write_text(text + "\n")


def main():
    parser = argparse.ArgumentParser(description="CPU verifier for sm100a dense GEMM outputs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("cases")

    reference_parser = subparsers.add_parser("reference")
    reference_parser.add_argument("--case", choices=["smoke"], default="smoke")
    reference_parser.add_argument("--output", required=True)
    reference_parser.add_argument("--fill", type=int, default=1)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--case", choices=CASES, required=True)
    verify_parser.add_argument("--actual", required=True)
    verify_parser.add_argument("--samples", type=int, default=64)
    verify_parser.add_argument("--fill", type=int, default=1)
    verify_parser.add_argument("--atol", type=float, default=0.05)
    verify_parser.add_argument("--rtol", type=float, default=0.05)
    verify_parser.add_argument("--seed", type=int, default=20260827)
    verify_parser.add_argument("--result")

    self_test_parser = subparsers.add_parser("self-test")
    self_test_parser.add_argument("--fill", type=int, default=1)
    args = parser.parse_args()

    if args.command == "cases":
        emit(CASES)
        return 0
    if args.command == "reference":
        write_smoke_reference(args.output, args.fill)
        return 0
    if args.command == "self-test":
        with tempfile.TemporaryDirectory() as directory:
            actual = Path(directory) / "smoke.bf16"
            write_smoke_reference(actual, args.fill)
            result = verify("smoke", actual, 64, args.fill, 0.05, 0.05, 20260827)
            emit(result)
            return 0 if result["passed"] else 1

    result = verify(
        args.case, args.actual, args.samples, args.fill,
        args.atol, args.rtol, args.seed,
    )
    emit(result, args.result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}))
        raise SystemExit(2)

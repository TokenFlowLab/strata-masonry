#!/usr/bin/env python3

"""Independent CPU verifier for the KernelBridge dense NVFP4 contract."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


CASES = {
    "smoke": {"m": 256, "k": 256, "n": 128, "production_m": 256},
    "flagship": {"m": 30720, "k": 8192, "n": 4096, "production_m": 30720},
}
MASK32 = np.uint64(0xFFFFFFFF)
SEED_A = np.uint64(0x9E3779B9)
SEED_B = np.uint64(0x7F4A7C15)
E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def hash32(values):
    values = np.asarray(values, dtype=np.uint64) & MASK32
    values = (values * np.uint64(2654435761)) & MASK32
    values ^= values >> np.uint64(16)
    values = (values * np.uint64(2246822519)) & MASK32
    values ^= values >> np.uint64(13)
    return values & MASK32


def f32_to_bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2")


def bf16_to_f32(values):
    return (np.asarray(values, dtype=np.uint32) << 16).view(np.float32)


def data_values(indices, mode, seed):
    indices = np.asarray(indices, dtype=np.uint64)
    if mode == 1:
        return np.full(indices.shape, 4.0, dtype=np.float32)
    if mode == 3:
        return np.ones(indices.shape, dtype=np.float32)
    hashed = hash32((indices * np.uint64(2) + seed) & MASK32)
    if mode == 2:
        codes = hashed & np.uint64(15)
    elif mode == 4:
        magnitudes = np.array([0, 2, 4, 5], dtype=np.uint8)
        codes = magnitudes[(hashed & np.uint64(3)).astype(np.intp)]
        codes |= (((hashed >> np.uint64(5)) & np.uint64(1)) << np.uint64(3)).astype(
            np.uint8
        )
    else:
        raise ValueError("fill mode must be 1, 2, 3, or 4")
    return E2M1[np.asarray(codes, dtype=np.intp)]


def scale_values(rows, k_blocks, mode, which):
    rows = np.asarray(rows, dtype=np.uint64)
    k_blocks = np.asarray(k_blocks, dtype=np.uint64)
    if mode in {1, 3}:
        modulus = 5 if which == 0 else 3
        bias = 2 if which == 0 else 1
        exponents = (rows % np.uint64(modulus)).astype(np.int32) - bias
    else:
        raw = rows * np.uint64(131071) + k_blocks * np.uint64(8191)
        raw += np.uint64(which * 977)
        exponents = (hash32(raw) % np.uint64(5)).astype(np.int32) - 2
    return np.exp2(exponents.astype(np.float32))


def reference_value(m, n, k_size, mode):
    k = np.arange(k_size, dtype=np.uint64)
    a_indices = np.uint64(m) * np.uint64(k_size) + k
    b_indices = np.uint64(n) * np.uint64(k_size) + k
    a = data_values(a_indices, mode, SEED_A)
    b = data_values(b_indices, mode, SEED_B)
    blocks = k // np.uint64(32)
    sa = scale_values(np.full(k.shape, m), blocks, mode, 0)
    sb = scale_values(np.full(k.shape, n), blocks, mode, 1)
    value = np.sum(a * sa * b * sb, dtype=np.float32)
    return bf16_to_f32(f32_to_bf16(np.array([value])))[0]


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


def verify(case_name, actual_path, samples, mode, atol, rtol, seed):
    case = CASES[case_name]
    actual = np.memmap(actual_path, dtype="<u2", mode="r")
    expected_elements = case["m"] * case["n"]
    if actual.size != expected_elements:
        raise ValueError(
            f"expected {expected_elements} BF16 values, found {actual.size}"
        )
    rows, cols = sample_coordinates(case["m"], case["n"], samples, seed)
    observed = bf16_to_f32(actual[rows * case["n"] + cols])
    reference = np.array(
        [reference_value(int(m), int(n), case["k"], mode) for m, n in zip(rows, cols)],
        dtype=np.float32,
    )
    error = np.abs(observed - reference)
    mismatch = ~np.isfinite(observed) | (
        (error > atol) & (error > rtol * np.abs(reference))
    )
    return {
        "schema": "kernelbridge.cpu-verifier-result/v0",
        "family": "gemm/sm100a/nvfp4",
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
    output = np.empty((case["m"], case["n"]), dtype=np.float32)
    for m in range(case["m"]):
        for n in range(case["n"]):
            output[m, n] = reference_value(m, n, case["k"], mode)
    f32_to_bf16(output).tofile(path)


def emit(result, result_path=None):
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if result_path:
        Path(result_path).write_text(text + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="CPU verifier for dense NVFP4 GEMM")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("cases")

    reference = subparsers.add_parser("reference")
    reference.add_argument("--case", choices=["smoke"], default="smoke")
    reference.add_argument("--output", required=True)
    reference.add_argument("--fill", type=int, default=2)

    check = subparsers.add_parser("verify")
    check.add_argument("--case", choices=CASES, required=True)
    check.add_argument("--actual", required=True)
    check.add_argument("--samples", type=int, default=2048)
    check.add_argument("--fill", type=int, default=2)
    check.add_argument("--atol", type=float, default=16.0)
    check.add_argument("--rtol", type=float, default=0.008)
    check.add_argument("--seed", type=int, default=20260827)
    check.add_argument("--result")

    self_test = subparsers.add_parser("self-test")
    self_test.add_argument("--fill", type=int, default=2)
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
            result = verify("smoke", actual, 64, args.fill, 16.0, 0.008, 20260827)
            emit(result)
            return 0 if result["passed"] else 1

    result = verify(
        args.case,
        args.actual,
        args.samples,
        args.fill,
        args.atol,
        args.rtol,
        args.seed,
    )
    emit(result, args.result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}))
        raise SystemExit(2)

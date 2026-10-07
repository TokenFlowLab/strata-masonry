#!/usr/bin/env python3
"""Independent CPU verification of packed dense BF16 context FMHA outputs.

See fmha_context_bf16_problem_size.md for shapes, fills, and top-left causal semantics.
Only NumPy is required. Full verification streams query/key blocks and checks every output.
"""

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np


UND_SEQLENS = [
    29, 101, 29, 53, 135, 85, 106, 94, 29, 95, 104, 69, 156, 214, 95, 164,
    159, 29, 170, 60, 118, 58, 93, 151, 50, 163, 133, 58, 61, 134, 203, 182,
    62, 56, 29, 29, 67, 29, 92, 118, 153, 108, 202, 62, 108, 87, 29, 29,
    52, 68, 175, 99, 64, 160, 29, 47, 179, 59, 55, 220, 49, 131, 95, 29,
    88, 153, 171, 89, 195, 96, 108, 53, 193, 60, 105, 110, 113, 67, 92, 113,
    60, 53, 175, 177, 91, 175, 128, 133, 207, 141, 186, 68, 192, 56, 94, 76,
    105, 159, 64, 199, 192, 128, 104, 102, 110, 54, 95, 195, 55, 29, 98, 58,
    291, 61, 89, 177, 221, 179, 187, 61, 102, 107, 100, 206, 102, 104, 189, 154,
]

def make_case(q_lengths, q_heads, kv_heads, causal=False, kv_lengths=None):
    return {
        "seqlens": list(q_lengths),
        "kv_seqlens": list(q_lengths if kv_lengths is None else kv_lengths),
        "q_heads": q_heads, "kv_heads": kv_heads, "head_dim": 128, "causal": causal,
    }


CASES = {
    "smoke": make_case([4, 4], 8, 4),
    "smoke-causal": make_case([4, 4], 8, 4, True),
    "und-full": make_case(UND_SEQLENS, 32, 4),
    "und-causal": make_case(UND_SEQLENS, 32, 4, True),
    "gen-full": make_case([240] * 128, 32, 4),
    "gen-causal": make_case([240] * 128, 32, 4, True),
    "gqa-mid-full": make_case([4608] * 8, 32, 4),
    "gqa-long-full": make_case([75600], 32, 4),
    "mha-mid-full": make_case([4608] * 8, 32, 32),
    "mha-mid-causal": make_case([4608] * 8, 32, 32, True),
    "mha-long-full": make_case([75600], 32, 32),
}
for _kind, _hk in (("mha", 32), ("gqa", 4)):
    for _mask in ("full", "causal"):
        CASES[f"cross-varlen-{_kind}-{_mask}"] = make_case(
            [100, 240, 64, 175], 32, _hk, _mask == "causal", [300, 240, 512, 90],
        )

# Correctness-matrix cases. Executed HK=4 and intended HK=2 are kept separately.
for _label, _b, _s, _hq, _hk in (
    ("mha-1024", 2, 1024, 16, 16), ("mha-968rag", 2, 968, 16, 16),
    ("mha-256min", 1, 256, 2, 2), ("mha-384odd", 1, 384, 2, 2),
    ("gqa-512", 1, 512, 8, 4), ("gqa-600rag", 1, 600, 8, 4),
    ("mha-832", 2, 832, 8, 8), ("mha-4608big", 1, 4608, 2, 2),
    ("gqa-4608big", 1, 4608, 8, 4), ("mha-4416rag", 1, 4416, 2, 2),
    ("gqa-512-hk2", 1, 512, 8, 2), ("gqa-600rag-hk2", 1, 600, 8, 2),
    ("gqa-4608big-hk2", 1, 4608, 8, 2),
):
    for _mask in ("full", "causal"):
        CASES[f"{_label}-{_mask}"] = make_case([_s] * _b, _hq, _hk, _mask == "causal")


def validate_case(case):
    q_lengths, kv_lengths = case["seqlens"], case["kv_seqlens"]
    if not q_lengths or len(q_lengths) != len(kv_lengths):
        raise ValueError("Q and KV lengths must have the same nonzero sample count")
    values = [*q_lengths, *kv_lengths, case["q_heads"], case["kv_heads"], case["head_dim"]]
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("lengths, head counts, and head dimension must be positive integers")
    if case["head_dim"] != 128 or case["q_heads"] % case["kv_heads"]:
        raise ValueError("expected D=128 and HQ divisible by HK")
    if type(case["causal"]) is not bool:
        raise ValueError("causal must be a boolean")


def output_elements(case):
    return sum(case["seqlens"]) * case["q_heads"] * case["head_dim"]


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


def prefixes(seqlens):
    result = np.zeros(len(seqlens) + 1, dtype=np.int64)
    result[1:] = np.cumsum(seqlens)
    return result


def sample_coordinates(case, count, seed):
    if count <= 0:
        raise ValueError("samples must be positive")
    row_width = case["q_heads"] * case["head_dim"]
    total = output_elements(case)
    # Include sample boundaries when the requested sample count permits it.
    cu = prefixes(case["seqlens"])
    fixed = np.unique(np.concatenate((cu[:-1] * row_width, cu[1:] * row_width - 1)))
    if count >= total:
        flat = np.arange(total)
    else:
        rng = np.random.default_rng(seed)
        boundary_count = min(len(fixed), max(1, count // 2))
        boundary_indices = np.linspace(0, len(fixed) - 1, boundary_count, dtype=np.int64)
        chosen = set(int(x) for x in fixed[boundary_indices])
        while len(chosen) < count:
            chosen.update(int(x) for x in rng.integers(0, total, size=count - len(chosen)))
        flat = np.array(sorted(chosen), dtype=np.int64)
    token = flat // row_width
    remainder = flat % row_width
    return token, remainder // case["head_dim"], remainder % case["head_dim"]


def reference_rows(case, sample, head, positions, mode, key_block):
    """FP32 online softmax; temporary score storage is rows x key_block, not Sq x Sk."""
    cq, ck = prefixes(case["seqlens"]), prefixes(case["kv_seqlens"])
    hq, hk, d = case["q_heads"], case["kv_heads"], case["head_dim"]
    dimensions = np.arange(d, dtype=np.uint64)
    q_indices = ((cq[sample] + positions[:, None]).astype(np.uint64) * hq + head) * d
    q = fill(q_indices + dimensions, mode, 11)
    limit = case["kv_seqlens"][sample]
    if case["causal"]:
        limit = min(limit, int(positions.max()) + 1)
    maximum = np.full(len(positions), -np.inf, dtype=np.float32)
    normalizer = np.zeros(len(positions), dtype=np.float32)
    numerator = np.zeros((len(positions), d), dtype=np.float32)
    for start in range(0, limit, key_block):
        keys = np.arange(start, min(start + key_block, limit), dtype=np.int64)
        kv_indices = ((ck[sample] + keys[:, None]).astype(np.uint64) * hk + head // (hq // hk)) * d
        k = fill(kv_indices + dimensions, mode, 22)
        v = fill(kv_indices + dimensions, mode, 33)
        scores = (q @ k.T) * np.float32(1.0 / math.sqrt(d))
        if case["causal"]:
            scores[keys[None, :] > positions[:, None]] = -np.inf
        updated_maximum = np.maximum(maximum, scores.max(axis=1))
        rescale = np.exp(maximum - updated_maximum)
        probabilities = np.exp(scores - updated_maximum[:, None])
        numerator *= rescale[:, None]
        numerator += probabilities @ v
        normalizer = normalizer * rescale + probabilities.sum(axis=1, dtype=np.float32)
        maximum = updated_maximum
    return numerator / normalizer[:, None]


def references(case, tokens, heads, dims, mode, query_block=64, key_block=2048):
    """Sampled reference; compute a query row once even if several dimensions were sampled."""
    cu = prefixes(case["seqlens"])
    samples = np.searchsorted(cu, tokens, side="right") - 1
    result = np.empty(tokens.size, dtype=np.float32)
    for sample, head in sorted(set(zip(samples.tolist(), heads.tolist()))):
        indices = np.flatnonzero((samples == sample) & (heads == head))
        positions, inverse = np.unique(tokens[indices] - cu[sample], return_inverse=True)
        for start in range(0, len(positions), query_block):
            stop = start + query_block
            reference = reference_rows(case, sample, head, positions[start:stop], mode, key_block)
            selected = (inverse >= start) & (inverse < stop)
            selected_indices = indices[selected]
            result[selected_indices] = reference[inverse[selected] - start, dims[selected_indices]]
    return result


def reference_chunks(case, mode, query_block=64, key_block=2048):
    cq = prefixes(case["seqlens"])
    hq, d = case["q_heads"], case["head_dim"]
    for sample, length in enumerate(case["seqlens"]):
        for head in range(hq):
            for start in range(0, length, query_block):
                positions = np.arange(start, min(start + query_block, length))
                reference = reference_rows(case, sample, head, positions, mode, key_block)
                flat = ((cq[sample] + positions[:, None]) * hq + head) * d + np.arange(d)
                yield flat, reference


def verify(case_name, actual_path, samples=64, mode=2, atol=0.05, rtol=0.10,
           seed=20260827, verification_mode="sampled", query_block=64, key_block=2048):
    case = CASES[case_name]
    validate_case(case)
    if verification_mode not in ("full", "sampled") or mode not in (1, 2, 3, 4):
        raise ValueError("invalid verification mode or fill")
    if min(samples, query_block, key_block) <= 0:
        raise ValueError("samples and block sizes must be positive")
    if any(not math.isfinite(x) or x < 0 for x in (atol, rtol)):
        raise ValueError("tolerances must be finite and nonnegative")
    expected_elements = output_elements(case)
    actual_bytes = Path(actual_path).stat().st_size
    if actual_bytes == expected_elements * 2:
        actual = np.memmap(actual_path, dtype="<u2", mode="r")
    elif actual_bytes == expected_elements * 4:
        actual = np.memmap(actual_path, dtype="<f4", mode="r")
    else:
        raise ValueError(f"expected {expected_elements * 2} bytes of BF16 or "
                         f"{expected_elements * 4} bytes of FP32 output")
    if verification_mode == "full" or samples >= expected_elements:
        chunks = reference_chunks(case, mode, query_block, key_block)
    else:
        tokens, heads, dims = sample_coordinates(case, samples, seed)
        flat = (tokens * case["q_heads"] + heads) * case["head_dim"] + dims
        chunks = [(flat, references(case, tokens, heads, dims, mode, query_block, key_block))]
    checked = mismatches = nonfinite = 0
    max_error = 0.0
    for flat, reference in chunks:
        if not np.isfinite(reference).all():
            raise ValueError("CPU reference produced a nonfinite value")
        observed = (bf16_to_f32(actual[flat]) if actual.dtype == np.uint16
                    else np.asarray(actual[flat], dtype=np.float32))
        finite = np.isfinite(observed)
        error = np.abs(observed - reference)
        mismatch = ~finite | ((error > atol) & (error > rtol * np.abs(reference)))
        checked += observed.size
        mismatches += int(np.count_nonzero(mismatch))
        nonfinite += int(np.count_nonzero(~finite))
        max_error = max(max_error, float(np.max(error[finite], initial=0.0)))
    return {
        "schema": "cpu-verifier-result/v0",
        "family": "fmha/sm100a", "case": case_name,
        "passed": mismatches == 0, "mode": verification_mode,
        "samples": checked, "checked_elements": checked, "total_elements": expected_elements,
        "full_coverage": checked == expected_elements,
        "mismatches": mismatches, "nonfinite_elements": nonfinite,
        "max_abs_error": None if nonfinite else max_error,
        "atol": atol, "rtol": rtol, "fill": mode,
        "actual": str(Path(actual_path).resolve()),
    }


def write_reference(case, path, mode, query_block=64, key_block=2048):
    validate_case(case)
    if min(query_block, key_block) <= 0 or mode not in (1, 2, 3, 4):
        raise ValueError("invalid block size or fill")
    output = np.memmap(path, dtype="<u2", mode="w+", shape=(output_elements(case),))
    for flat, reference in reference_chunks(case, mode, query_block, key_block):
        output[flat] = f32_to_bf16(reference)
    output.flush()


def self_test(mode):
    """Compare the blocked FP32 algorithm to a separate dense FP64 oracle on tiny cases."""
    checks = 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "output.bf16"
        for causal in (False, True):
            for hk in (2, 4):
                case = make_case([5, 3], 4, hk, causal, [2, 7])
                cq, ck = prefixes(case["seqlens"]), prefixes(case["kv_seqlens"])
                q = fill(np.arange(cq[-1] * 4 * 128), mode, 11).reshape(-1, 4, 128)
                k = fill(np.arange(ck[-1] * hk * 128), mode, 22).reshape(-1, hk, 128)
                v = fill(np.arange(ck[-1] * hk * 128), mode, 33).reshape(-1, hk, 128)
                oracle = np.empty(q.shape, dtype=np.float64)
                for sample in range(2):
                    for head in range(4):
                        qh = q[cq[sample]:cq[sample + 1], head].astype(np.float64)
                        kh = k[ck[sample]:ck[sample + 1], head // (4 // hk)].astype(np.float64)
                        vh = v[ck[sample]:ck[sample + 1], head // (4 // hk)].astype(np.float64)
                        scores = qh @ kh.T / math.sqrt(128)
                        if causal:
                            masked = np.arange(len(kh))[None, :] > np.arange(len(qh))[:, None]
                            scores[masked] = -np.inf
                        p = np.exp(scores - scores.max(axis=1, keepdims=True))
                        oracle[cq[sample]:cq[sample + 1], head] = (p / p.sum(axis=1)[:, None]) @ vh
                full = np.empty(output_elements(case), dtype=np.float32)
                for flat, reference in reference_chunks(case, mode, 3, 2):
                    full[flat] = reference
                np.testing.assert_allclose(full, oracle.ravel(), atol=1e-5, rtol=1e-5)
                tokens, heads, dims = sample_coordinates(case, 37, 17)
                sampled = references(case, tokens, heads, dims, mode, 2, 3)
                np.testing.assert_allclose(
                    sampled, oracle[tokens, heads, dims], atol=1e-5, rtol=1e-5,
                )
                checks += 1
        write_reference(CASES["smoke"], path, mode, 3, 2)
        result = verify("smoke", path, mode=mode, verification_mode="full", key_block=3)
        assert result["passed"] and result["checked_elements"] == output_elements(CASES["smoke"])
        # A corruption outside the sampled set must still fail full verification.
        tokens, heads, dims = sample_coordinates(CASES["smoke"], 64, 20260827)
        sampled_flat = set(((tokens * 8 + heads) * 128 + dims).tolist())
        corrupt = next(i for i in range(output_elements(CASES["smoke"])) if i not in sampled_flat)
        output = np.memmap(path, dtype="<u2", mode="r+")
        output[corrupt] = f32_to_bf16(np.array([100.0]))[0]
        output.flush()
        assert verify("smoke", path, mode=mode)["passed"]
        assert verify("smoke", path, mode=mode, verification_mode="full")["mismatches"] == 1
        for bits in (0x7FC0, 0x7F80, 0xFF80):  # NaN, +inf, -inf must never pass.
            output[corrupt] = bits
            output.flush()
            bad = verify("smoke", path, mode=mode, verification_mode="full")
            assert not bad["passed"] and bad["nonfinite_elements"] == 1
            json.dumps(bad, allow_nan=False)
    return {"passed": True, "fill": mode, "oracle_cases": checks, "full_check": result}


def case_report():
    return {
        name: {
            **case,
            "batch": len(case["seqlens"]),
            "total_tokens": sum(case["seqlens"]),
            "total_kv_tokens": sum(case["kv_seqlens"]),
            "output_elements": output_elements(case),
        }
        for name, case in CASES.items()
    }


def emit(result, result_path=None):
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if result_path:
        Path(result_path).write_text(text + "\n")


def main():
    parser = argparse.ArgumentParser(description="CPU verifier for sm100a FMHA outputs")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("cases")

    reference_parser = subparsers.add_parser("reference")
    reference_parser.add_argument("--case", choices=CASES, default="smoke")
    reference_parser.add_argument("--output", required=True)
    reference_parser.add_argument("--fill", type=int, choices=(1, 2, 3, 4), default=2)
    reference_parser.add_argument("--query-block", type=int, default=64)
    reference_parser.add_argument("--key-block", type=int, default=2048)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--case", choices=CASES, required=True)
    verify_parser.add_argument("--actual", required=True)
    verify_parser.add_argument("--mode", choices=("sampled", "full"), default="sampled")
    verify_parser.add_argument("--samples", type=int, default=64)
    verify_parser.add_argument("--fill", type=int, choices=(1, 2, 3, 4), default=2)
    verify_parser.add_argument("--query-block", type=int, default=64)
    verify_parser.add_argument("--key-block", type=int, default=2048)
    verify_parser.add_argument("--atol", type=float, default=0.05)
    verify_parser.add_argument("--rtol", type=float, default=0.10)
    verify_parser.add_argument("--seed", type=int, default=20260827)
    verify_parser.add_argument("--result")

    self_test_parser = subparsers.add_parser("self-test")
    self_test_parser.add_argument("--fill", type=int, choices=(1, 2, 3, 4), default=2)
    args = parser.parse_args()

    if args.command == "cases":
        emit(case_report())
        return 0
    if args.command == "reference":
        write_reference(CASES[args.case], args.output, args.fill, args.query_block, args.key_block)
        return 0
    if args.command == "self-test":
        emit(self_test(args.fill))
        return 0

    result = verify(
        args.case, args.actual, args.samples, args.fill,
        args.atol, args.rtol, args.seed, args.mode, args.query_block, args.key_block,
    )
    emit(result, args.result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}))
        raise SystemExit(2)

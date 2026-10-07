#!/usr/bin/env python3
"""CPU-only block-causal/sink/window and teacher-forcing reference (NumPy).

Full output verification is the default. Uses the driver's deterministic fills or
actual shared BF16 input files; never substitutes a GPU implementation as reference.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fmha_context_bf16_cpu_verifier import bf16_to_f32, f32_to_bf16, fill


def make_case(frames=6, tokens_per_frame=1456, frames_per_block=3, sink=1, window=6,
              heads=12, kv_heads=None, batch=1, teacher_forcing=False, rope=False):
    case = dict(frames=frames, tokens_per_frame=tokens_per_frame,
                frames_per_block=frames_per_block, sink=sink, window=window,
                heads=heads, kv_heads=heads if kv_heads is None else kv_heads,
                batch=batch, teacher_forcing=teacher_forcing, rope=rope)
    if min(frames, tokens_per_frame, frames_per_block, heads, case['kv_heads'], batch) <= 0:
        raise ValueError('dimensions must be positive')
    if frames % frames_per_block or heads % case['kv_heads']:
        raise ValueError('require complete frame blocks and HQ divisible by HK')
    if sink < 0 or window < -1 or (window >= 0 and window < sink):
        raise ValueError('require sink >= 0 and window == -1 or window >= sink')
    case['length'] = frames * tokens_per_frame * (2 if teacher_forcing else 1)
    return case


def cases():
    return {f'{mode}-{nf}f': make_case(frames=nf, teacher_forcing='tf' in mode,
                                      rope='rope' in mode)
            for mode in ('bcs', 'rope', 'tf', 'tf-rope') for nf in (6, 12, 18, 36, 72)}


class Inputs:
    def __init__(self, case, directory=None, mode=2):
        self.case, self.mode, self.arrays = case, mode, {}
        if mode not in (1, 2, 3, 4):
            raise ValueError('fill must be 1..4')
        if directory:
            for name in ('q', 'k', 'v'):
                heads = case['heads'] if name == 'q' else case['kv_heads']
                shape = (case['batch'] * case['length'], heads, 128)
                path = Path(directory) / f"{name}_L{case['length']}.npy"
                a = np.load(path, mmap_mode='r', allow_pickle=False)
                if a.shape != shape or a.dtype != np.dtype('<u2') or not a.flags.c_contiguous:
                    raise ValueError(f'{path}: expected C-order uint16 BF16 bits {shape}')
                self.arrays[name] = a

    def rows(self, name, batch, head, tokens):
        tokens = np.asarray(tokens, dtype=np.int64) + batch * self.case['length']
        if name in self.arrays:
            result = bf16_to_f32(self.arrays[name][tokens, head])
        else:
            heads = self.case['heads'] if name == 'q' else self.case['kv_heads']
            indices = (tokens[:, None] * heads + head) * 128 + np.arange(128)
            result = fill(indices, self.mode, {'q': 11, 'k': 22, 'v': 33}[name])
        if not np.isfinite(result).all():
            raise ValueError(f'{name}: nonfinite input')
        return result


def visible_keys(case, query):
    """Ordered union for one query block; no dense mask allocation."""
    block = case['tokens_per_frame'] * case['frames_per_block']
    half = case['frames'] * case['tokens_per_frame']
    local = query % half
    start, end = local // block * block, (local // block + 1) * block
    noisy = case['teacher_forcing'] and query >= half
    context_end = start if noisy else end
    sink = min(case['sink'] * case['tokens_per_frame'], context_end)
    rolling = ((case['window'] - case['sink']) * case['tokens_per_frame']
               if case['window'] >= 0 else case['length'] + block)
    window_start = max(0, end - rolling)
    parts = [np.arange(sink), np.arange(max(sink, window_start), context_end)]
    if noisy:
        parts.append(np.arange(half + start, half + end))
    return np.concatenate(parts).astype(np.int64)


def sink_query(case, query, q):
    if not case['rope'] or case['window'] < 0 or case['sink'] == 0:
        return q
    block = case['tokens_per_frame'] * case['frames_per_block']
    half = case['frames'] * case['tokens_per_frame']
    delta = max(0, (query % half // block + 1) * case['frames_per_block'] - case['window'])
    angle = delta * np.power(10000.0, -2 * np.arange(64, dtype=np.float64) / 128)
    c, s = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
    rotated = np.empty_like(q)
    rotated[:, 0::2] = q[:, 0::2] * c + q[:, 1::2] * s
    rotated[:, 1::2] = q[:, 1::2] * c - q[:, 0::2] * s
    return bf16_to_f32(f32_to_bf16(rotated))


def reference_rows(case, inputs, batch, head, queries, key_chunk):
    """Queries must belong to one block. Return FP32 O and natural-log LSE."""
    q = inputs.rows('q', batch, head, queries)
    qs = sink_query(case, int(queries[0]), q)
    keys = visible_keys(case, int(queries[0]))
    if not len(keys):
        raise ValueError('mask gives an empty attention row')
    kv_head = head // (case['heads'] // case['kv_heads'])
    maximum = np.full((len(queries), 1), -np.inf, dtype=np.float32)
    denominator = np.zeros_like(maximum)
    numerator = np.zeros_like(q)
    for start in range(0, len(keys), key_chunk):
        ids = keys[start:start + key_chunk]
        k, v = (inputs.rows(name, batch, kv_head, ids) for name in ('k', 'v'))
        scores = q @ k.T
        sink = ids < case['sink'] * case['tokens_per_frame']
        if case['rope'] and sink.any():
            scores[:, sink] = qs @ k[sink].T
        scores *= np.float32(1 / math.sqrt(128))
        updated = np.maximum(maximum, scores.max(axis=1, keepdims=True))
        correction = np.exp(maximum - updated)
        probabilities = np.exp(scores - updated)
        numerator = numerator * correction + probabilities @ v
        denominator = denominator * correction + probabilities.sum(axis=1, keepdims=True)
        maximum = updated
    return numerator / denominator, (maximum + np.log(denominator)).ravel()


def reference_chunks(case, inputs, query_chunk=64, key_chunk=2048, coordinates=None):
    length, heads = case['length'], case['heads']
    block = case['tokens_per_frame'] * case['frames_per_block']
    for batch in range(case['batch']):
        for head in range(heads):
            queries = (np.arange(length) if coordinates is None else
                       coordinates[(coordinates // length) % heads == head])
            if coordinates is not None:
                queries = queries[queries // (length * heads) == batch] % length
            for qb in np.unique(queries // block):
                rows = queries[queries // block == qb]
                for start in range(0, len(rows), query_chunk):
                    ids = rows[start:start + query_chunk]
                    output, lse = reference_rows(case, inputs, batch, head, ids, key_chunk)
                    yield batch, head, ids, output, lse


def verify(case, actual, directory=None, fill_mode=2, mode='full', samples=64,
           seed=1, query_chunk=64, key_chunk=2048, atol=.03, rtol=.03, lse=None):
    if min(samples, query_chunk, key_chunk) <= 0:
        raise ValueError('sample count and chunk sizes must be positive')
    if any(not math.isfinite(x) or x < 0 for x in (atol, rtol)):
        raise ValueError('tolerances must be finite and nonnegative')
    shape = (case['batch'], case['length'], case['heads'], 128)
    if Path(actual).stat().st_size != math.prod(shape) * 4:
        raise ValueError(f'expected raw FP32 output {shape}')
    observed = np.memmap(actual, mode='r', dtype='<f4', shape=shape)
    lse_observed = None
    if lse:
        lse_shape = (case['batch'], case['heads'], case['length'])
        if Path(lse).stat().st_size != math.prod(lse_shape) * 4:
            raise ValueError(f'expected raw log2 FP32 LSE {lse_shape}')
        lse_observed = np.memmap(lse, mode='r', dtype='<f4', shape=lse_shape)
    coordinates = None
    if mode == 'sampled':
        total = math.prod(shape[:-1])
        coordinates = np.sort(np.random.default_rng(seed).choice(
            total, min(samples, total), replace=False))
    inputs = Inputs(case, directory, fill_mode)
    checked = mismatches = nonfinite = lse_mismatches = 0
    max_error = 0.0
    for b, h, ids, reference, expected_lse in reference_chunks(
            case, inputs, query_chunk, key_chunk, coordinates):
        values = observed[b, ids, h]
        if not np.isfinite(reference).all():
            raise ValueError('nonfinite CPU reference')
        error = np.abs(values - reference)
        finite = np.isfinite(values)
        mismatches += int(np.count_nonzero(~finite | (error > atol + rtol * np.abs(reference))))
        nonfinite += int(np.count_nonzero(~finite))
        checked += values.size
        max_error = max(max_error, float(error[finite].max(initial=0)))
        if lse_observed is not None:
            values = lse_observed[b, h, ids]
            lse_mismatches += int(np.count_nonzero(
                ~np.isfinite(values) | (np.abs(values - expected_lse / np.log(2.0)) > atol)))
    return dict(passed=mismatches == 0 and lse_mismatches == 0, mode=mode,
                checked_elements=checked, total_elements=math.prod(shape),
                full_coverage=checked == math.prod(shape), mismatches=mismatches,
                nonfinite_elements=nonfinite, max_abs_error=None if nonfinite else max_error,
                lse_mismatches=lse_mismatches if lse else None, atol=atol, rtol=rtol,
                case=case, actual=str(Path(actual).resolve()),
                inputs=str(Path(directory).resolve()) if directory else f'FILL={fill_mode}')


def self_test():
    checks = 0
    for tf in (False, True):
        for rope in (False, True):
            case = make_case(9, 4, 3, 2, 5, heads=2, batch=2,
                             teacher_forcing=tf, rope=rope)
            inputs = Inputs(case)
            # Separate dense FP64 oracle, explicit scalar visibility predicate.
            for b, h, ids, actual, actual_lse in reference_chunks(case, inputs, 7, 5):
                all_keys = np.arange(case['length'])
                k = inputs.rows('k', b, h, all_keys).astype(np.float64)
                v = inputs.rows('v', b, h, all_keys).astype(np.float64)
                for row, i in enumerate(ids):
                    pos = int(i) % 36
                    end = (pos // 12 + 1) * 12
                    noisy = tf and i >= 36
                    keep = np.array([(j < (end - 12 if noisy else end) and
                                      (j < 8 or j >= max(0, end - 12))) or
                                     (noisy and 36 + end - 12 <= j < 36 + end)
                                     for j in all_keys])
                    q = inputs.rows('q', b, h, [i])
                    qs = sink_query(case, int(i), q)
                    score = (k @ q[0].astype(np.float64)) / math.sqrt(128)
                    score[:8] = (k[:8] @ qs[0].astype(np.float64)) / math.sqrt(128)
                    score = score[keep]
                    p = np.exp(score - score.max())
                    expected = p @ v[keep] / p.sum()
                    np.testing.assert_allclose(actual[row], expected, atol=2e-6, rtol=2e-5)
                    np.testing.assert_allclose(actual_lse[row], score.max() + np.log(p.sum()),
                                               atol=2e-6, rtol=2e-5)
                    checks += 1
    print(json.dumps(dict(passed=True, rows_checked=checks)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('cases')
    sub.add_parser('self-test')
    p = sub.add_parser('verify')
    p.add_argument('--actual', required=True)
    p.add_argument('--inputs')
    p.add_argument('--lse', help='optional FV raw log2 FP32 LSE dump')
    p.add_argument('--case', choices=cases())
    for name, default in [('frames', 6), ('tokens-per-frame', 1456), ('frames-per-block', 3),
                          ('sink', 1), ('window', 6), ('heads', 12), ('batch', 1)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--kv-heads', type=int)
    p.add_argument('--teacher-forcing', action='store_true')
    p.add_argument('--rope', action='store_true')
    p.add_argument('--fill', type=int, choices=(1, 2, 3, 4), default=2)
    p.add_argument('--mode', choices=('full', 'sampled'), default='full')
    p.add_argument('--samples', type=int, default=64, help='sampled token/head rows')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--query-chunk', type=int, default=64)
    p.add_argument('--key-chunk', type=int, default=2048)
    p.add_argument('--atol', type=float, default=.03)
    p.add_argument('--rtol', type=float, default=.03)
    p.add_argument('--result')
    args = parser.parse_args()
    try:
        if args.command == 'cases':
            print(json.dumps(cases(), indent=2))
            return 0
        if args.command == 'self-test':
            self_test()
            return 0
        case = cases()[args.case] if args.case else make_case(
            args.frames, args.tokens_per_frame, args.frames_per_block, args.sink, args.window,
            args.heads, args.kv_heads, args.batch, args.teacher_forcing, args.rope)
        report = verify(case, args.actual, args.inputs, args.fill, args.mode, args.samples,
                        args.seed, args.query_chunk, args.key_chunk, args.atol, args.rtol, args.lse)
        text = json.dumps(report, indent=2, allow_nan=False)
        if args.result:
            with open(args.result, 'x') as f:
                f.write(text + '\n')
        print(text)
        return 0 if report['passed'] else 1
    except (ValueError, OSError) as error:
        print(f'error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

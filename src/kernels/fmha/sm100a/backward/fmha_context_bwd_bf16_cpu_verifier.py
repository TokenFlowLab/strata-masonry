#!/usr/bin/env python3
"""CPU-only dense BF16 FMHA backward input generation and full gradient verification.

NumPy reference; real BF16 forward O and natural-log FP32 LSE are shared inputs,
not placeholders. All dQ/dK/dV output elements are checked by default and always.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fmha_context_bf16_cpu_verifier import bf16_to_f32, f32_to_bf16


def rounded(x):
    return bf16_to_f32(f32_to_bf16(x))


def suffix(shape, causal):
    b, h, s, d = shape
    return f'_B{b}_H{h}_S{s}_D{d}_c{int(causal)}.npy'


def validate_shape(shape):
    if len(shape) != 4 or min(shape) <= 0 or shape[-1] not in (64, 128):
        raise ValueError('require positive [B,H,S,D] and D=64 or 128')


def forward(q, k, v, causal, query_chunk=64, key_chunk=1024):
    """Streaming FP32 softmax; returns BF16 O and FP32 natural-log LSE."""
    length, dim = q.shape
    output = np.empty_like(q, dtype=np.uint16)
    lse = np.empty(length, dtype=np.float32)
    scale = np.float32(1 / math.sqrt(dim))
    for i in range(0, length, query_chunk):
        ids = np.arange(i, min(i + query_chunk, length))
        maximum = np.full((len(ids), 1), -np.inf, dtype=np.float32)
        denominator = np.zeros_like(maximum)
        numerator = np.zeros((len(ids), dim), dtype=np.float32)
        limit = int(ids[-1]) + 1 if causal else length
        for j in range(0, limit, key_chunk):
            keys = np.arange(j, min(j + key_chunk, limit))
            score = (q[ids] @ k[keys].T) * scale
            if causal:
                score[keys[None, :] > ids[:, None]] = -np.inf
            updated = np.maximum(maximum, score.max(axis=1, keepdims=True))
            correction = np.exp(maximum - updated)
            p = np.exp(score - updated)
            numerator = numerator * correction + p @ v[keys]
            denominator = denominator * correction + p.sum(axis=1, keepdims=True)
            maximum = updated
        output[ids] = f32_to_bf16(numerator / denominator)
        lse[ids] = (maximum + np.log(denominator)).ravel()
    return output, lse


def gradients(q, k, v, dout, output, lse, causal, query_chunk=64, key_chunk=1024):
    """The documented BF16 P/dS operands, FP32 sums, and scale-once dQ/dK."""
    length, dim = q.shape
    dq, dk, dv = (np.zeros_like(q) for _ in range(3))
    scale = np.float32(1 / math.sqrt(dim))
    delta = (output * dout).sum(axis=1, dtype=np.float32)
    log2e = np.float32(math.log2(math.e))
    for i in range(0, length, query_chunk):
        ids = np.arange(i, min(i + query_chunk, length))
        limit = int(ids[-1]) + 1 if causal else length
        for j in range(0, limit, key_chunk):
            keys = np.arange(j, min(j + key_chunk, limit))
            p = np.exp2(((q[ids] @ k[keys].T) * scale - lse[ids, None]) * log2e)
            if causal:
                p[keys[None, :] > ids[:, None]] = 0
            ds = rounded(p * ((dout[ids] @ v[keys].T) - delta[ids, None]))
            dq[ids] += ds @ k[keys]
            dk[keys] += ds.T @ q[ids]
            dv[keys] += rounded(p).T @ dout[ids]
    return dq * scale, dk * scale, dv


def load_inputs(directory, shape, causal):
    tail = suffix(shape, causal)
    result = {}
    for name in ('q', 'k', 'v', 'do', 'o', 'lse'):
        expected_shape = shape[:-1] if name == 'lse' else shape
        dtype = np.dtype('<f4' if name == 'lse' else '<u2')
        path = Path(directory) / (name + tail)
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        if (array.shape != tuple(expected_shape) or array.dtype != dtype or
                not array.flags.c_contiguous):
            raise ValueError(f'{path}: expected C-order {dtype} {expected_shape}')
        result[name] = array
    return result


def head_inputs(arrays, b, h):
    result = [arrays[name][b, h] if name == 'lse' else bf16_to_f32(arrays[name][b, h])
              for name in ('q', 'k', 'v', 'do', 'o', 'lse')]
    if any(not np.isfinite(a).all() for a in result):
        raise ValueError('nonfinite saved input or forward state')
    return result


def generate(directory, shape, causal, seed=20260825, query_chunk=64, key_chunk=1024):
    # Exclusive directory creation prevents mixing old forward state with new inputs.
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tail = suffix(shape, causal)
    rng = np.random.default_rng(seed)
    arrays = {}
    for name in ('q', 'k', 'v', 'do', 'o', 'lse'):
        path = directory / (name + tail)
        arrays[name] = np.lib.format.open_memmap(
            str(path) + '.tmp', mode='w+', dtype='<f4' if name == 'lse' else '<u2',
            shape=shape[:-1] if name == 'lse' else shape)
    for b in range(shape[0]):
        for h in range(shape[1]):
            for name in ('q', 'k', 'v', 'do'):
                bound = .5 if name == 'do' else .75
                arrays[name][b, h] = f32_to_bf16(
                    rng.uniform(-bound, bound, size=shape[2:]).astype(np.float32))
            q, k, v = (bf16_to_f32(arrays[name][b, h]) for name in ('q', 'k', 'v'))
            arrays['o'][b, h], arrays['lse'][b, h] = forward(
                q, k, v, causal, query_chunk, key_chunk)
    for name, array in arrays.items():
        array.flush()
        path = directory / (name + tail)
        Path(str(path) + '.tmp').rename(path)
    print(json.dumps(dict(generated=str(directory.resolve()), shape=shape,
                          causal=causal, seed=seed, lse_base='natural-log')))


def verify(directory, prefix, shape, causal, atol=.05, rtol=.02,
           query_chunk=64, key_chunk=1024):
    arrays = load_inputs(directory, shape, causal)
    observed, stats = {}, {}
    for name in ('dq', 'dk', 'dv'):
        path = Path(str(prefix) + '_' + name + '.bf16')
        if path.stat().st_size != math.prod(shape) * 2:
            raise ValueError(f'{path}: expected raw BF16 {shape}')
        observed[name] = np.memmap(path, mode='r', dtype='<u2', shape=shape)
        stats[name] = dict(checked_elements=0, mismatches=0, nonfinite_elements=0,
                           max_abs_error=0.0)
    for b in range(shape[0]):
        for h in range(shape[1]):
            references = gradients(*head_inputs(arrays, b, h), causal, query_chunk, key_chunk)
            for name, expected in zip(('dq', 'dk', 'dv'), references):
                if not np.isfinite(expected).all():
                    raise ValueError('nonfinite CPU gradient')
                actual = bf16_to_f32(observed[name][b, h])
                error = np.abs(actual - expected)
                finite = np.isfinite(actual)
                entry = stats[name]
                entry['checked_elements'] += actual.size
                entry['mismatches'] += int(np.count_nonzero(
                    ~finite | ((error > atol) & (error > rtol * np.abs(expected)))))
                entry['nonfinite_elements'] += int(np.count_nonzero(~finite))
                entry['max_abs_error'] = max(entry['max_abs_error'],
                                             float(error[finite].max(initial=0)))
    for entry in stats.values():
        if entry['nonfinite_elements']:
            entry['max_abs_error'] = None
    return dict(passed=all(v['mismatches'] == 0 for v in stats.values()), mode='full',
                shape=shape, causal=causal, atol=atol, rtol=rtol,
                checked_elements=3 * math.prod(shape), full_coverage=True,
                inputs=str(Path(directory).resolve()), actual_prefix=str(Path(prefix).resolve()),
                gradients=stats)


def self_test():
    rng = np.random.default_rng(4)
    for dim in (64, 128):
        for causal in (False, True):
            q, k, v, do = [rounded(rng.normal(0, .3, (19, dim))) for _ in range(4)]
            o, lse = forward(q, k, v, causal, 7, 5)
            score = q.astype('f8') @ k.astype('f8').T / math.sqrt(dim)
            if causal:
                score[np.triu_indices(19, 1)] = -np.inf
            maximum = score.max(axis=1, keepdims=True)
            e = np.exp(score - maximum)
            p = e / e.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(bf16_to_f32(o), rounded(p @ v), atol=1e-6)
            np.testing.assert_allclose(lse, (maximum + np.log(e.sum(axis=1, keepdims=True))).ravel(),
                                       atol=2e-6)
            # Dense FP64 formula, with the same explicit BF16 quantization points.
            p_saved = np.exp(score - lse[:, None])
            delta = (do.astype('f8') * bf16_to_f32(o)).sum(axis=1)
            ds = rounded(p_saved * (do.astype('f8') @ v.T - delta[:, None]))
            expected = (ds.astype('f8') @ k / math.sqrt(dim),
                        ds.astype('f8').T @ q / math.sqrt(dim), rounded(p_saved).astype('f8').T @ do)
            actual = gradients(q, k, v, do, bf16_to_f32(o), lse, causal, 7, 5)
            for a, e in zip(actual, expected):
                np.testing.assert_allclose(a, e, atol=2e-5, rtol=2e-3)
    print(json.dumps(dict(passed=True, modes=4)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('self-test')
    sub.add_parser('cases')
    for command in ('generate', 'verify'):
        p = sub.add_parser(command)
        p.add_argument('--batch', type=int, default=1)
        p.add_argument('--heads', type=int, help='default: 2048 / head dimension')
        p.add_argument('--seqlen', type=int, required=True)
        p.add_argument('--head-dim', type=int, choices=(64, 128), required=True)
        p.add_argument('--causal', action='store_true')
        p.add_argument('--query-chunk', type=int, default=64)
        p.add_argument('--key-chunk', type=int, default=1024)
        if command == 'generate':
            p.add_argument('--output', required=True, help='new shared-input directory')
            p.add_argument('--seed', type=int, default=20260825)
        else:
            p.add_argument('--inputs', required=True)
            p.add_argument('--actual-prefix', required=True)
            p.add_argument('--atol', type=float, default=.05)
            p.add_argument('--rtol', type=float, default=.02)
            p.add_argument('--result')
    args = parser.parse_args()
    try:
        if args.command == 'self-test':
            self_test()
            return 0
        if args.command == 'cases':
            print(json.dumps([dict(batch=b, seqlen=s, heads=2048 // d, head_dim=d, causal=c)
                              for b, s in ((32,512),(16,1024),(8,2048),(4,4096),(2,8192),(1,16384))
                              for d in (64,128) for c in (False,True)], indent=2))
            return 0
        shape = (args.batch, 2048 // args.head_dim if args.heads is None else args.heads,
                 args.seqlen, args.head_dim)
        validate_shape(shape)
        if min(args.query_chunk, args.key_chunk) <= 0:
            raise ValueError('chunk sizes must be positive')
        if args.command == 'generate':
            generate(args.output, shape, args.causal, args.seed, args.query_chunk, args.key_chunk)
            return 0
        if any(not math.isfinite(x) or x < 0 for x in (args.atol, args.rtol)):
            raise ValueError('tolerances must be finite and nonnegative')
        report = verify(args.inputs, args.actual_prefix, shape, args.causal, args.atol, args.rtol,
                        args.query_chunk, args.key_chunk)
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

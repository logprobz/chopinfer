#!/usr/bin/env python3
"""Compare ordered F32 and BF16-partial folds on authenticated real operands.

The original-input f64 error and bound usage are reported without moving the
bound. The BF16-partial kernel has its own exact rounded-input contract; this
CPU report is not observed GPU output or a model accuracy qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    if path.is_symlink():
        raise ValueError('capture metadata must not be a symlink')
    return json.loads(path.read_bytes())


def bf16_bits(value):
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(value), np.uint16(0x7fff), rounded).astype(np.uint16)


def widen(bits):
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def folded(partials, rounded):
    value = np.zeros_like(partials[0], dtype=np.float32)
    for rank in partials:
        np.add(value, widen(bf16_bits(rank)) if rounded else rank, out=value)
    return bf16_bits(value)


def errors(partials):
    if len(partials) != 4 or not all(np.isfinite(x).all() for x in partials):
        raise ValueError('expected four finite original F32 rank partials')
    reference = np.zeros_like(partials[0], dtype=np.float64)
    magnitude = reference.copy()
    for rank in partials:
        reference += rank.astype(np.float64)
        magnitude += np.abs(rank.astype(np.float64))
    old_bits, new_bits = folded(partials, False), folded(partials, True)
    old, new = widen(old_bits).astype(np.float64), widen(new_bits).astype(np.float64)
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('nonfinite reduction output must be investigated')
    # kernel_core::oracle::numeric_bound for BF16 output, K=4. frexp's
    # exponent is floor(log2(abs(reference)))+1 for a nonzero value.
    exponent = np.frexp(np.abs(reference))[1]
    half_ulp = np.where(np.abs(reference) < 2.0**-126, 2.0**-134,
                        np.ldexp(np.ones_like(reference), exponent - 9))
    bound = half_ulp + 2.0**-22 * 4 * magnitude
    old_error, new_error = np.abs(old-reference), np.abs(new-reference)
    scale = np.where(magnitude == 0, 1.0, magnitude)
    arrays = dict(current_abs_f64=old_error, candidate_abs_f64=new_error,
                  candidate_abs_current=np.abs(new-old),
                  current_bound_use=old_error/bound, candidate_bound_use=new_error/bound,
                  current_error_over_sum_abs=old_error/scale,
                  candidate_error_over_sum_abs=new_error/scale)
    counts = dict(elements=int(reference.size), changed_bits=int(np.count_nonzero(old_bits != new_bits)),
                  worse_f64=int(np.count_nonzero(new_error > old_error)),
                  better_f64=int(np.count_nonzero(new_error < old_error)),
                  current_outside_original_bound=int(np.count_nonzero(old_error > bound)),
                  candidate_outside_original_bound=int(np.count_nonzero(new_error > bound)),
                  zero_reference=int(np.count_nonzero(reference == 0)))
    return arrays, counts


def distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('empty or nonfinite error distribution')
    percentiles = np.quantile(values, [.5, .9, .99, .999])
    return dict(mean=float(values.mean()), rms=float(np.sqrt(np.mean(values*values))),
                p50=float(percentiles[0]), p90=float(percentiles[1]),
                p99=float(percentiles[2]), p999=float(percentiles[3]), max=float(values.max()))


def safe_file(directory, name):
    if Path(name).name != name:
        raise ValueError('record paths must be local filenames')
    path = directory/name
    if path.is_symlink():
        raise ValueError('record must not be a symlink')
    return path


def capture(path):
    data = path.read_bytes()
    document = read_json(path)
    if document['schema'] != 'chopinfer.deepseek41.decode-operands.v1' or document['ranks'] != 4:
        raise ValueError('unsupported capture schema or rank count')
    transaction = read_json(safe_file(path.parent.parent, document['transaction']))
    members = [m for m in transaction['members'] if m['directory'] == path.parent.name]
    if not transaction['committed'] or len(members) != 1 or members[0]['manifest_sha256'] != digest(data):
        raise ValueError('capture transaction is not committed or its manifest changed')
    dimensions = document['dimensions']
    width, live, launch = dimensions['hidden'], document['live_rows'], document['launch_rows']
    if width != 5120 or not 0 < live <= launch:
        raise ValueError('unexpected hidden width or live rows')
    found = {}
    for record in document['records']:
        match = re.fullmatch(r'layer-(\d+)-sum', record['site'])
        if not match:
            continue
        layer, rank = int(match[1]), record['rank']
        if rank not in range(4) or (layer, rank) in found:
            raise ValueError('duplicate or invalid rank partial')
        payload = safe_file(path.parent, record['file']).read_bytes()
        if len(payload) != record['bytes'] or digest(payload) != record['sha256']:
            raise ValueError('partial payload hash or size differs')
        outputs = [t for t in record['tensors'] if t['name'] == 'output']
        if len(outputs) != 1:
            raise ValueError('partial output is absent or duplicated')
        tensor = outputs[0]
        if (tensor['dtype'] != 'f32' or tensor['shape'] != [launch, width]
                or tensor['bytes'] != launch*width*4 or tensor['offset'] % 4
                or tensor['offset']+tensor['bytes'] > len(payload)):
            raise ValueError('partial tensor view differs from its declared shape')
        found[layer, rank] = np.frombuffer(payload, dtype='<f4', count=launch*width,
            offset=tensor['offset']).reshape(launch, width)[:live].copy().reshape(-1)
    layers = range(dimensions['first_layer'], dimensions['first_layer']+dimensions['layers'])
    if set(found) != {(layer, rank) for layer in layers for rank in range(4)}:
        raise ValueError('missing or extra layer/rank coverage')
    return document, [(layer, [found[layer, rank] for rank in range(4)]) for layer in layers]


def analyze(run, output, export=None, expected_captures=9):
    paths = sorted(run.glob('*/operands/*/capture.json'))
    if len(paths) != expected_captures or expected_captures <= 0:
        raise ValueError('wrong capture count')
    records, inputs, pooled, totals = [], [], {}, {}
    if export:
        export.mkdir(parents=True, exist_ok=False)
    for path in paths:
        document, cases = capture(path)
        for layer, partials in cases:
            arrays, counts = errors(partials)
            record = dict(capture=path.parent.name, phase=document['phase'], bank=document['bank'],
                          layer=layer, rows=document['live_rows'], **counts,
                          distributions={name:distribution(value) for name,value in arrays.items()})
            records.append(record)
            for name, value in arrays.items():
                pooled.setdefault(name, []).append(value)
            for name, value in counts.items():
                totals[name] = totals.get(name, 0)+value
            if export:
                name = f'{path.parent.name}-layer-{layer}.bin'
                data = b''.join(x.astype('<f4', copy=False).tobytes() for x in partials)
                (export/name).write_bytes(data)
                inputs.append(dict(file=name, sha256=digest(data), bytes=len(data),
                    rows=document['live_rows'], width=5120, ranks=4, layer=layer,
                    capture=path.parent.name, phase=document['phase'], bank=document['bank']))
    result = dict(schema='chopinfer.deepseek41.bf16-partial-errors.v1', captures=len(paths),
        cases=len(records), counts=totals,
        distributions={name:distribution(np.concatenate(values)) for name,values in pooled.items()},
        records=records, source=read_json(run/'source.json'),
        source_manifests={str(path):digest(path.read_bytes()) for path in paths},
        limitation='Earlier diagnostic MoE inputs only. CPU contract simulation, not observed post-reduction GPU output, attention/prefill coverage, or current-build accuracy acceptance.',
        original_bound='0.5 BF16 ulp(original f64 sum) + 2^-22 * 4 * sum(abs(original F32 partials)); unchanged, report-only for the changed-input contract.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    if export:
        (export/'manifest.json').write_text(json.dumps(dict(cases=inputs, capture_count=len(paths),
            provenance_report=str(output.resolve()), provenance_sha256=digest(output.read_bytes())),
            indent=2, sort_keys=True)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    if any(p and p.resolve().is_relative_to(root) for p in [args.output, args.export]):
        parser.error('reports and operands belong outside the repository')
    result = analyze(args.run, args.output, args.export)
    print(json.dumps({key:result[key] for key in ['captures','cases','counts','distributions']}, indent=2))

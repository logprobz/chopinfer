#!/usr/bin/env python3
"""Prepare explicitly concatenated MiMo route replays from model-audit dumps."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def prepare(audit, output, layer):
    manifest = json.loads((audit / 'manifest.json').read_text())
    routes = []
    sources = []
    for stream in manifest['streams']:
        path = audit / stream['name'] / f'layer{layer:02}.ids'
        data = path.read_bytes()
        if len(data) != len(stream['ids']) * 8 * 4:
            raise ValueError(f'route shape differs from manifest: {path}')
        ids = [v[0] for v in struct.iter_unpack('<i', data)]
        for start in range(0, len(ids), 8):
            row = ids[start:start + 8]
            if len(set(row)) != 8 or any(e < 0 or e >= 256 for e in row):
                raise ValueError(f'invalid route row: {path}, {start // 8}')
            routes.append(row)
        sources.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                            tokens=len(ids) // 8))
    if len(routes) < 2048:
        raise ValueError('need at least 2048 captured rows; no repetition is allowed')
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    for tokens in (1024, 2048):
        logits = []
        counts = [0] * 256
        for row in routes[:tokens]:
            values = [-8.0] * 256
            for slot, expert in enumerate(row):
                values[expert] = 1.75 - slot * 0.25
                counts[expert] += 1
            logits.extend(values)
        data = struct.pack(f'<{len(logits)}f', *logits)
        (output / f't{tokens}.f32').write_bytes(data)
        ids = [expert for row in routes[:tokens] for expert in row]
        (output / f't{tokens}.ids').write_bytes(struct.pack(f'<{len(ids)}i', *ids))
        cases.append(dict(tokens=tokens, expert_rows=counts,
                          logits_sha256=hashlib.sha256(data).hexdigest()))
    result = dict(schema='chopinfer.moe-prefill-replay.v1', layer=layer,
                  note='Concatenated captured streams, not a contiguous serving chunk. '
                       'Logits reproduce expert IDs and order. Weights and activations are synthetic.',
                  sources=sources, cases=cases)
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--layer', type=int, required=True)
    args = parser.parse_args()
    prepare(args.audit, args.output, args.layer)


if __name__ == '__main__':
    main()

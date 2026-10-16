"""Preserve captured rows and repeat them explicitly for shape controls."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('audit', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
ROOT, AUDIT = args.output, args.audit
manifest = json.loads((AUDIT / 'manifest.json').read_text())
streams = [s for s in manifest['streams'] if len(s['chunks']) == 1]
unique_rows = sum(len(s['ids']) for s in streams)
if unique_rows < 1:
    parser.error('capture has no single-chunk rows')
ROOT.mkdir(exist_ok=False)
for layer in (1, 5, 21, 30, 47):
    directory = ROOT / f'layer{layer:02}'
    directory.mkdir()
    sources = []
    values = {}
    for role, suffix, width in [('input', f'layer{layer-1:02}.bf16', 4096*2),
                                ('residual', f'layer{layer:02}.resid.bf16', 4096*2),
                                ('ids', f'layer{layer:02}.ids', 8*4)]:
        rows = []
        for stream in streams:
            path = AUDIT / stream['name'] / suffix
            data = path.read_bytes()
            assert len(data) == len(stream['ids']) * width, path
            sources.append({'role': role, 'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()})
            rows.extend(data[i:i+width] for i in range(0,len(data),width))
        values[role] = rows
    cases = []
    for tokens in (1024, 2048):
        files = {}
        counts = [0]*256
        for role, rows in values.items():
            data = b''.join(rows[i % len(rows)] for i in range(tokens))
            name = f't{tokens}.{role}'
            (directory/name).write_bytes(data)
            files[role] = {'file':name, 'sha256':hashlib.sha256(data).hexdigest(), 'bytes':len(data)}
            if role == 'ids':
                ids = [x[0] for x in struct.iter_unpack('<i',data)]
                for i in range(0,len(ids),8):
                    assert len(set(ids[i:i+8])) == 8 and all(0<=e<256 for e in ids[i:i+8])
                for expert in ids: counts[expert] += 1
        cases.append({'tokens':tokens, 'files':files, 'expert_rows':counts})
    (directory/'manifest.json').write_text(json.dumps({'schema':'chopinfer.moe-prefill-captured-layer.v1', 'layer':layer, 'unique_rows':unique_rows, 'repeated_rows':True, 'audit_manifest':{'path':str(AUDIT/'manifest.json'),'sha256':hashlib.sha256((AUDIT/'manifest.json').read_bytes()).hexdigest()}, 'sources':sources, 'cases':cases, 'limitations':'Real hidden states, attention residuals and route IDs from three short TP4 model-audit streams, repeated to target size. Not a contiguous serving chunk. Route weights were not captured; use fixed 1/8 weights only for the diagnostic down/combine.'},indent=2)+'\n')
print(ROOT)

"""Write a small synthetic packed checkpoint for SDK tests, without calibration."""
import argparse
import json
from pathlib import Path
import struct


def write_checkpoint(output, schema, seed, expert_scales):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps({'architecture': 'example.precision.v1', 'seed': seed}) + '\n')
    metadata = (Path(__file__).resolve().parents[1] / 'fixtures' /
                ('precision-' + schema + '.json')).read_bytes()
    (output / 'hf_quant_config.json').write_bytes(metadata)
    tensors = []
    for stem, shape, table in [('adapter.projection', [256, 256], False),
                               ('switch.pair', [4, 512, 256], False),
                               ('dictionary.table', [37, 160], True)]:
        elements = 1
        for n in shape:
            elements *= n
        codes = bytes(((i * 17 + seed) % 16) | (((i * 7 + seed * 3) % 16) << 4)
                      for i in range(elements // 2))
        tensors.append((stem + '.weight', 'U8', [*shape[:-1], shape[-1] // 2], codes))
        local = bytes(0x30 + (i + seed) % 8 for i in range(elements // 16))
        tensors.append((stem + '.weight_scale', 'F8_E4M3', [*shape[:-1], shape[-1] // 16], local))
        count = 4 if expert_scales and stem == 'switch.pair' else 1
        globals_ = b''.join(struct.pack('<f', (e + 1) / 64.0) for e in range(count))
        tensors.append((stem + '.weight_scale_2', 'F32', [count], globals_))
        if not table:
            calibration = b''.join(struct.pack('<f', (e + 1) / 1024.0) for e in range(count))
            tensors.append((stem + '.input_scale', 'F32', [] if count == 1 else [count], calibration))
    index = {}
    # Companion payloads intentionally live in a different shard from codes.
    for filename, selected in [('weights.safetensors', [t for t in tensors if t[0].endswith('.weight')]),
                               ('scales.safetensors', [t for t in tensors if not t[0].endswith('.weight')])]:
        header, payload = {}, bytearray()
        for name, dtype, shape, data in selected:
            header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(payload), len(payload) + len(data)]}
            payload.extend(data)
            index[name] = filename
        encoded = json.dumps(header, separators=(',', ':')).encode()
        encoded += b' ' * (-len(encoded) % 8)
        (output / filename).write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
    (output / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--schema', choices=['nested', 'flat'], default='nested')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--expert-scales', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.seed <= 255:
        parser.error('fixture seed must be between 0 and 255')
    write_checkpoint(args.output, args.schema, args.seed, args.expert_scales)

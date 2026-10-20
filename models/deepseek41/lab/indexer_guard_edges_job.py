#!/usr/bin/env python3
"""Require the complete guard-edge supplement when a later pair includes it."""
import argparse
import json
from pathlib import Path
import re

GPU_TEST = 'indexer_mma::guard::edges::gpu_domain_edges_and_native_accumulation'


def cases():
    result = []
    for layer in (2, 20, 24):
        for spread in (7, 8, 9, 10):
            for lo in (129, 130, 131, 254, 368-spread, 369-spread, 370-spread):
                for factor in range(3):
                    for pattern in range(3):
                        result.append(dict(layer=layer, lo=lo, spread=spread, factor=factor, pattern=pattern))
        for lo in (130, 131, 254, 368, 369):
            for factor in range(3):
                result.append(dict(layer=layer, lo=lo, spread=0, factor=factor, pattern=3))
    assert len(result) == 801
    for i, c in enumerate(result):
        exact = c['lo'] >= 130 and c['lo'] + c['spread'] <= 369
        c.update(id=i, accepted=exact and c['spread'] <= 9, raw_comparisons=2,
                 selection_comparisons=2, native_oracle_checks=(1024 if c['pattern'] == 3 else 384) if exact else 0,
                 mismatches=0)
    return result


def analyze(logs):
    assert len(logs) == 2
    expected = cases()
    for shard, text in enumerate(logs):
        records = [json.loads(line.split('GUARD_EDGE_CASE ', 1)[1]) for line in text.splitlines() if 'GUARD_EDGE_CASE ' in line]
        here = [c for c in expected if c['id'] % 2 == shard]
        assert records == here, f'guard edge coverage or comparison mismatch on shard {shard}'
        checked = sum(c['native_oracle_checks'] for c in here)
        complete = re.findall(r'GUARD_EDGES_COMPLETE shard=(\d+) cases=(\d+) total=(\d+) native_oracle_checks=(\d+)', text)
        assert complete == [(str(shard), str(len(here)), '801', str(checked))]
        results = re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored', text)
        assert results and all(r == ('ok', '1', '0', '0') for r in results)
    checked = sum(c['native_oracle_checks'] for c in expected)
    assert checked == 253440
    return dict(cases=801, raw_comparisons=1602, selection_comparisons=1602,
                native_oracle_checks=checked, mismatch_count=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--logs', type=Path, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze([p.read_text() for p in args.logs])
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('GUARD_EDGES_PAIR_COMPLETE ' + json.dumps(report))


if __name__ == '__main__':
    main()

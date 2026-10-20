"""Extract recorded cost feedback for the runtime's CPU replay test."""
import argparse
import hashlib
import json
from pathlib import Path


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def extract(root, output):
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['status'] == 'passed' and summary['case_count'] == 14 and summary['bitwise']
    adaptive = root / '5-adaptive/C16/steps.jsonl'
    measured = root / '3-uniform3/C16/steps.jsonl'
    stuck = [row['speculation'] for row in records(adaptive)
             if not row['prefill_items'] and row['decode_rows'] == 16
             and row['speculation']['reason'] == 'estimated_slower'][-1]
    cost = next(c for c in stuck['measured_verify_costs'] if c['rows'] == 32)
    assert cost['micros'] > 180_000 and stuck['realized_draft_steps'] < 8
    observations = []
    previous = set()
    for row in records(measured):
        current = {b['request_id'] for b in row['blocks']}
        if not row['prefill_items'] and row['verify_items'] == 16 and row['live_rows'] == 32:
            assert row['row_capacity'] == 32 and row['shape'] == [1] * 16
            observations.append(dict(drafts=row['shape'], fresh=len(current - previous),
                                     micros=row['execution_us'], accepted=row['accepted_lower'], batch=row['batch']))
        previous = current
    assert len(observations) >= 16 and observations[0]['fresh'] > 0
    assert observations[0]['micros'] < cost['micros'] / 2
    output.mkdir(parents=True, exist_ok=True)
    result = dict(schema='chopinfer.speculation-cost-replay.v1',
                  initial=dict(ar_us=stuck['ar_estimate_us'], verify_us=cost['micros'], drafting_items=int(cost['drafting_items'])),
                  observations=observations,
                  sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (adaptive, measured)],
                  scope='Replay a recorded stuck adaptive cost followed by actual full-C16 K1 feedback from a separate matrix arm. This tests cost learning and the planning response; it is not a served timing counterfactual or proof of a speedup. The first observation includes fresh drafters and bounds its steady cost from above.')
    (output / 'replay.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(samples=len(observations), initial=result['initial'], first=observations[0])))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    extract(args.run, args.output)

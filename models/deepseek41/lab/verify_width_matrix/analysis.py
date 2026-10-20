"""Audit completed engine steps against independent speculation counters."""
import collections
import json
import statistics

def parse(text, require_phase=True):
    steps, phase, blocks = [], None, []
    seen_steps, seen_blocks = set(), set()
    for line in text.splitlines():
        if 'diagnostic writer: dropped ' in line:
            assert int(line.split('diagnostic writer: dropped ')[1].split()[0]) == 0
        at = line.find('{')
        if at < 0:
            continue
        try:
            row = json.loads(line[at:])
        except ValueError:
            assert not any(s in line for s in ['chopinfer.step.v1', 'drafter-confidence.v1', 'step-shape.v1'])
            continue
        schema = row.get('schema')
        if schema == 'chopinfer.deepseek41.step-shape.v1' and 'micros' in row:
            assert phase is None
            phase = row
        elif schema == 'chopinfer.deepseek41.drafter-confidence.v1':
            blocks.append(row)
        elif schema == 'chopinfer.step.v1':
            if phase is None:
                assert not require_phase and row['verify_items'] == 0
                phase = dict(instance_id='ar', step_id=str(row['batch']),
                             items=row['prefill_items']+row['decode_rows'], bulk_passes=0)
            key=(phase['instance_id'],phase['step_id'])
            assert key not in seen_steps
            seen_steps.add(key)
            assert phase['items'] == row['prefill_items'] + row['decode_rows'] + row['verify_items']
            assert len(blocks) == row['verify_items']
            assert all(b['step_id'] == phase['step_id'] and b['instance_id'] == phase['instance_id'] and b['completion'] == 'committed' for b in blocks)
            for b in blocks:
                key=(b['instance_id'],b['step_id'],b['request_id'])
                assert key not in seen_blocks
                seen_blocks.add(key)
            assert sum(b['verify_rows'] for b in blocks) == row['verify_rows']
            row = dict(row, phase=phase, blocks=blocks)
            row['accepted_lower'] = sum(b['outcomes'].count('accepted') for b in blocks)
            row['shape'] = sorted([b['emitted_positions'] for b in blocks] + [0] * row['decode_rows'], reverse=True)
            row['tokens_lower'] = row['decode_rows'] + row['verify_items'] + row['accepted_lower']
            steps.append(row)
            phase, blocks = None, []
    assert phase is None and not blocks and steps
    return steps

def dist(values):
    assert values
    ordered = sorted(values)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=ordered[round(.95*(len(ordered)-1))], minimum=ordered[0], maximum=ordered[-1])

def coverage(steps, a):
    for field in ('accepted_positions','drafted_positions'):
        a[field] = {str(k):v for k,v in a[field].items()}
    pure = [s for s in steps if s['prefill_items'] == 0 and s['decode_rows'] + s['verify_rows']]
    expected = collections.Counter((s['decode_rows'], s['verify_rows'], s['row_capacity']) for s in pure)
    hist = a['step_rows']
    actual = {(b['decode_rows'], b['verify_rows'], b['row_capacity']):b['steps'] for b in hist['bins']}
    assert hist['dropped'] == 0 and expected == actual, (expected, actual)
    blocks = [b for s in steps for b in s['blocks']]
    assert len(blocks) == a['verify_items']
    assert sum(b['emitted_positions'] for b in blocks) == a['draft_tokens']
    assert sum(b['verify_rows'] for b in blocks) == a['rows']
    assert sum(a['accepted_positions'].values()) == a['accepted_tokens']
    accepted = [sum(b['outcomes'][:j].count('accepted') == j for b in blocks) for j in range(1,6)]
    drafted = [sum(b['emitted_positions'] >= j for b in blocks) for j in range(1,6)]
    assert drafted == [a['drafted_positions'][str(j)] for j in range(1,6)]
    missing = [a['accepted_positions'][str(j)] - accepted[j-1] for j in range(1,6)]
    assert all(x >= 0 for x in missing)
    if not any(missing):
        assert a['emitted_tokens'] == len(blocks) + sum(accepted)
    # A complete set of pointwise lower bounds whose sum equals the exact
    # aggregate counters is exact item by item. Otherwise keep the bounds.
    return dict(steps=len(steps), pure_steps=len(pure), verify_items=len(blocks),
                accepted_lower=accepted, accepted_unlocalized=missing, per_item_exact=not any(missing))

def group(steps):
    out=[]
    groups=collections.defaultdict(list)
    for s in steps:
        groups[(s['decode_rows']+s['verify_items'], s['decode_rows']+s['verify_rows'])].append(s)
    for (items, rows), ss in sorted(groups.items()):
        shapes=collections.Counter(tuple(s['shape']) for s in ss)
        us=[s['execution_us'] for s in ss]
        boundary=[s['host']['boundary_us'] for s in ss if s.get('host',{}).get('boundary_us') is not None]
        tokens=sum(s['tokens_lower'] for s in ss)
        out.append(dict(items=items, rows=rows, steps=len(ss), execution_ms=dist([u/1000 for u in us]),
                        boundary_ms=dist([u/1000 for u in boundary]) if boundary else None,
                        tokens_per_step_lower=tokens/len(ss), tokens_per_execution_second_lower=tokens*1e6/sum(us),
                        shapes=[dict(drafts=list(k),steps=v) for k,v in shapes.most_common()],
                        bulk_steps=sum(s['phase']['bulk_passes']>0 for s in ss)))
    return out

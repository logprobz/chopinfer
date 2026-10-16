"""Print paired eviction statistics from kernel-lab receipts."""
import json
from pathlib import Path
import statistics
import sys


def summarize(rounds):
    groups = {}
    for sample in rounds:
        if 'measurement' not in sample or 'arm' not in sample:
            continue
        group = groups.setdefault(sample['measurement'], [{}, {}, 0, sample.get('operation', 'unlabelled')])
        if group[3] != sample.get('operation', 'unlabelled'):
            raise ValueError('inconsistent operation labels')
        group[sample['arm']][sample['round']] = sample['elapsed_ms']
        group[2] += bool(sample.get('gpu_state', {}).get('flags'))
    rows = []
    for measurement, (a, b, flagged, operation) in sorted(groups.items()):
        if len(a) < 3 or a.keys() != b.keys():
            raise ValueError('incomplete A/B measurement')
        ratios = [b[r] / a[r] for r in a]
        rows.append((measurement, operation, statistics.median(a.values()) * 1000,
                     statistics.median(b.values()) * 1000,
                     statistics.median(ratios), flagged))
    return rows


def main(root):
    print('receipt\toperation\tmeasurement\twrite_us\tread_us\tpaired_read/write\tflagged_arms\tmaterial_gt_2pct')
    count = 0
    for path in sorted(Path(root).rglob('*.json')):
        data = json.loads(path.read_text())
        for measurement, label, a, b, ratio, flags in summarize(data.get('timing_rounds', [])):
            print(f'{path.relative_to(root)}\t{label}\t{measurement}\t{a:.3f}\t{b:.3f}\t{ratio:.5f}\t{flags}\t{abs(ratio - 1) > .02}')
            count += 1
    if count == 0:
        raise ValueError('no complete A/B measurements')


if __name__ == '__main__':
    main(sys.argv[1])

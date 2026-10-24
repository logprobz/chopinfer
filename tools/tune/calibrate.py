#!/usr/bin/env python3
"""Fit a drafter confidence calibration from served drafter records, on CPU.

Usage: python3 -B tools/tune/calibrate.py --sampled RUN_OR_LOG... \
           [--greedy RUN_OR_LOG...] [--floor N] [--positions N] \
           [--min-samples N] --out FILE

Each RUN_OR_LOG is a server log or a tuner run directory (every
*/server/server.log under it; its config.json must measure only the kind it
is passed as). Servers log one `*.drafter-confidence.v1` record per drafted
block: the package's confidence contract, the request and its committed
rows, each position's raw confidence `s` as f32 bits, and outcomes. Leading
`accepted` outcomes are certain. The first `unknown` after them is where
verification stopped, but on its own it is not a rejection: an accepted stop
token or a token limit also stops it and ends the request. So it counts as a
rejection only when the same request drafts again from more committed rows
(a later record of the same server, instance and request), which proves the
request went on; otherwise it is left out. Position j's samples are thus the
committed records whose drafts before j were accepted and whose draft j was
emitted, labelled 1 when it was accepted and 0 when it was proven rejected:
the chance a draft is accepted when every earlier one was, which the
calibration's `sigmoid(scale * s + offset)` models. Cancelled blocks,
un-emitted positions and non-finite scores are left out, and a repeated
(instance, step, request) is refused. Dropped log lines lose whole records
without biasing labels; the report counts them and marks the capture
incomplete. Fit on runs served without a calibration: an allocated step
verifies fewer drafts than it emitted.

Each position gets a logistic fit (Newton's method with a small ridge). The
greedy pairs repeat the sampled ones unless greedy runs are given. FILE
receives the calibration as `speculation.confidence` and
`--speculation-confidence` take it; FILE with `.report.json` in place of its
suffix receives the sources' SHA256 digests and run settings, per-position
counts, accepted/drafted (the share of emitted drafts accepted,
unconditional), the conditional rate, the fit, log loss in sample, held out
(two folds by request) and for a constant rate, AUC and the expected
calibration error over ten bins.
"""
import argparse
import hashlib
import json
import math
import re
import struct
import sys
import zlib
from pathlib import Path

SCHEMA_SUFFIX = '.drafter-confidence.v1'
MAX_POSITIONS = 16
RIDGE = 1e-3
DROPPED = 'diagnostic writer: dropped '


def f32(value):
    """`value` rounded to the nearest f32, as a float that prints exactly."""
    return struct.unpack('<f', struct.pack('<f', value))[0]


def server_logs(source, kind, settings):
    """The server logs of one source: a log file, or a tuner run measuring
    only `kind`, whose settings are added to `settings`."""
    path = Path(source)
    if path.is_file():
        return [path]
    config = json.loads((path / 'config.json').read_text())
    settings.append({'run': str(path), **{key: config.get(key) for key in (
        'job', 'model', 'kinds', 'sampling', 'serve_arguments', 'speculation_mode', 'arms')}})
    kinds = sorted(config.get('kinds', {'greedy': 1, 'sampled': 1}))
    if kinds != [kind]:
        raise SystemExit(f'{path} measures {kinds}, not only {kind}')
    logs = sorted(path.glob('*/server/server.log'))
    if not logs:
        raise SystemExit(f'{path} holds no */server/server.log')
    return logs


def records(log, index=0):
    """The drafter confidence records of one server log, each tagged with
    `index`, and the lines its writer dropped."""
    rows, dropped = [], 0
    for line in log.read_text(errors='replace').splitlines():
        if re.search(r'\bstartup speculation_confidence: on(?:,|$)', line):
            raise SystemExit(f'{log} served with confidence allocation; '
                             'fit only logs served without confidence')
        if line.startswith(DROPPED):
            dropped += int(line[len(DROPPED):].split()[0])
            continue
        if SCHEMA_SUFFIX not in line or '{' not in line:
            continue
        try:
            row = json.loads(line[line.index('{'):])
        except ValueError:
            continue
        if isinstance(row, dict) and str(row.get('schema', '')).endswith(SCHEMA_SUFFIX):
            row['log'] = index
            rows.append(row)
    return rows, dropped


def samples(rows):
    """Per position, `(score, label, fold)` samples, the emitted and
    accepted drafts per position, and the stops left unlabelled."""
    seen, later = set(), {}
    for row in rows:
        step = (row['log'], row['instance_id'], row['step_id'], row['request_id'])
        if step in seen:
            raise SystemExit(f'record {step} repeats')
        seen.add(step)
        request = (row['log'], row['instance_id'], row['request_id'])
        later[request] = max(later.get(request, -1), int(row['main_rows']))
    by_position, emitted, accepted, unproven = {}, {}, {}, 0
    for row in rows:
        if row.get('completion') != 'committed':
            continue
        request = (row['log'], row['instance_id'], row['request_id'])
        continued = later[request] > int(row['main_rows'])
        scores = [struct.unpack('<f', struct.pack('<I', bits))[0]
                  for bits in row['confidence_f32_bits']]
        fold = zlib.crc32(f"{row['log']}/{row['instance_id']}/{row['request_id']}".encode()) & 1
        for position, outcome in enumerate(row['outcomes']):
            if outcome not in ('accepted', 'unknown'):
                continue
            emitted[position] = emitted.get(position, 0) + 1
            accepted[position] = accepted.get(position, 0) + (outcome == 'accepted')
        for position, outcome in enumerate(row['outcomes']):
            if outcome not in ('accepted', 'unknown'):
                break
            if outcome == 'unknown' and not continued:
                unproven += 1
                break
            score = scores[position]
            if math.isfinite(score):
                by_position.setdefault(position, []).append(
                    (score, 1 if outcome == 'accepted' else 0, fold))
            if outcome == 'unknown':
                break
    return by_position, emitted, accepted, unproven


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit(points):
    """`(scale, offset)` minimizing log loss plus a small ridge."""
    scale, offset = 0.0, 0.0
    n = len(points)
    for _ in range(100):
        g_s = RIDGE * n * scale
        g_o = RIDGE * n * offset
        h_ss = h_so = h_oo = RIDGE * n
        for s, y, _ in points:
            p = sigmoid(scale * s + offset)
            w = p * (1.0 - p)
            g_s += (p - y) * s
            g_o += p - y
            h_ss += w * s * s
            h_so += w * s
            h_oo += w
        det = h_ss * h_oo - h_so * h_so
        if det <= 0:
            break
        d_s = (h_oo * g_s - h_so * g_o) / det
        d_o = (h_ss * g_o - h_so * g_s) / det
        scale, offset = scale - d_s, offset - d_o
        if abs(d_s) < 1e-10 and abs(d_o) < 1e-10:
            break
    return scale, offset


def log_loss(points, scale, offset):
    total = 0.0
    for s, y, _ in points:
        p = min(max(sigmoid(scale * s + offset), 1e-12), 1.0 - 1e-12)
        total -= math.log(p) if y else math.log(1.0 - p)
    return total / len(points)


def constant(points):
    rate = sum(y for _, y, _ in points) / len(points)
    rate = min(max(rate, 1e-12), 1.0 - 1e-12)
    return -(rate * math.log(rate) + (1 - rate) * math.log(1 - rate))


def auc(points):
    """The chance a random accepted sample scores above a random rejected
    one (ties count half), or None without both."""
    ranked = sorted(points, key=lambda point: point[0])
    positives = sum(y for _, y, _ in ranked)
    negatives = len(ranked) - positives
    if not positives or not negatives:
        return None
    rank_sum, index = 0.0, 0
    while index < len(ranked):
        end = index
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        mean_rank = (index + 1 + end) / 2
        rank_sum += mean_rank * sum(y for _, y, _ in ranked[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def ece(points, scale, offset, bins=10):
    """Expected calibration error over equal-width probability bins."""
    totals = [[0, 0.0, 0] for _ in range(bins)]
    for s, y, _ in points:
        p = sigmoid(scale * s + offset)
        cell = totals[min(int(p * bins), bins - 1)]
        cell[0] += 1
        cell[1] += p
        cell[2] += y
    return sum(abs(predicted - accepted) for _, predicted, accepted in totals) / len(points)


def calibrate(sources, positions, min_samples):
    """One kind's pairs and report, from `sources` of that kind."""
    rows, dropped, digests = [], 0, []
    for index, log in enumerate(sources):
        found, lost = records(log, index)
        rows.extend(found)
        dropped += lost
        digests.append({'log': str(log), 'sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
                        'records': len(found), 'dropped_lines': lost})
    contracts = sorted({row.get('confidence_contract') for row in rows})
    if len(contracts) != 1 or not contracts[0]:
        raise SystemExit(f'records name confidence contracts {contracts}; fit one at a time')
    by_position, emitted, accepted, unproven = samples(rows)
    deepest = max(by_position, default=-1) + 1
    positions = positions or deepest
    if not 1 <= positions <= MAX_POSITIONS:
        raise SystemExit(f'{positions} positions; records cover {deepest}, at most {MAX_POSITIONS}')
    pairs, table = [], []
    for position in range(positions):
        points = by_position.get(position, [])
        if len(points) < min_samples:
            raise SystemExit(
                f'position {position + 1} has {len(points)} samples, fewer than {min_samples}; '
                'record a run that drafts this deep')
        scale, offset = fit(points)
        folds = [[p for p in points if p[2] == fold] for fold in (0, 1)]
        held = None
        if all(folds):
            held = sum(log_loss(folds[1 - k], *fit(folds[k])) * len(folds[1 - k])
                       for k in (0, 1)) / len(points)
        pairs.append([f32(scale), f32(offset)])
        table.append({
            'position': position + 1,
            'samples': len(points),
            'accepted_given_earlier': sum(y for _, y, _ in points) / len(points),
            'drafted': emitted.get(position, 0),
            'accepted_over_drafted': accepted.get(position, 0) / emitted[position]
            if emitted.get(position) else None,
            'scale': f32(scale),
            'offset': f32(offset),
            'mean_predicted': sum(sigmoid(scale * s + offset) for s, _, _ in points) / len(points),
            'log_loss': log_loss(points, scale, offset),
            'log_loss_held_out': held,
            'log_loss_constant': constant(points),
            'auc': auc(points),
            'ece': ece(points, scale, offset),
        })
    return contracts[0], pairs, {
        'logs': digests, 'records': len(rows), 'dropped_log_lines': dropped,
        'complete': dropped == 0, 'unproven_stops': unproven, 'positions': table}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--sampled', nargs='+', required=True)
    parser.add_argument('--greedy', nargs='+')
    parser.add_argument('--floor', type=int, default=1)
    parser.add_argument('--positions', type=int, default=0)
    parser.add_argument('--min-samples', type=int, default=200)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} exists')
    settings = []
    sampled = [log for source in args.sampled for log in server_logs(source, 'sampled', settings)]
    contract, sampled_pairs, sampled_report = calibrate(sampled, args.positions, args.min_samples)
    positions = len(sampled_pairs)
    report = {'runs': settings, 'sampled': sampled_report}
    greedy_pairs = sampled_pairs
    if args.greedy:
        greedy = [log for source in args.greedy for log in server_logs(source, 'greedy', settings)]
        greedy_contract, greedy_pairs, report['greedy'] = calibrate(
            greedy, positions, args.min_samples)
        if greedy_contract != contract:
            raise SystemExit(f'greedy records follow {greedy_contract}, sampled {contract}')
    if not 0 <= args.floor <= positions:
        raise SystemExit(f'floor {args.floor} is outside 0 to {positions}')
    calibration = {'contract': contract, 'floor': args.floor,
                   'greedy': greedy_pairs, 'sampled': sampled_pairs}
    report['greedy_from'] = 'greedy runs' if args.greedy else 'sampled fit'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calibration) + '\n')
    out.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(calibration))
    print(f"{sampled_report['records']} records, {sampled_report['unproven_stops']} stops left "
          f"unlabelled, {sampled_report['dropped_log_lines']} log lines dropped"
          + ('' if sampled_report['complete'] else ' (INCOMPLETE capture)'))
    for row in sampled_report['positions']:
        print(f"position {row['position']}: {row['samples']} samples, accepted given earlier "
              f"{row['accepted_given_earlier']:.3f}, accepted/drafted "
              f"{row['accepted_over_drafted']:.3f}, log loss {row['log_loss']:.4f} "
              f"(held out {row['log_loss_held_out']}, constant {row['log_loss_constant']:.4f}), "
              f"AUC {row['auc']}, ECE {row['ece']:.4f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())

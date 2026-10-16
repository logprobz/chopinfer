#!/usr/bin/env python3
"""Validate bounded diagnostic evidence and report its first observed difference.

Fixed-schedule logits comparisons are performed by the Rust recorder. This
reader verifies its retained artifacts and reference binding, not GPU numerics.
Journal comparison is structural and uses labels supplied by the model fixture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FORMAT = 'chopinfer.diagnostic.v1'
CODES = {'match': 0, 'diverged': 1, 'incompatible': 2, 'insufficient': 3}
METADATA_LIMIT = 16 << 20
MAX_METADATA_LIMIT = 256 << 20
JOURNAL_LIMIT = 64 << 20


def external(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('diagnostic paths must be absolute and external')
    for part in (path, *path.parents):
        if part.is_symlink() or (part / '.git').exists():
            raise ValueError('diagnostic paths cannot enter a repository or traverse symlinks')
    return path


def digest(path):
    with external(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path, limit=MAX_METADATA_LIMIT):
    with external(path).open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('diagnostic metadata budget exceeded')
    return json.loads(data, parse_constant=invalid_constant)


def invalid_constant(value):
    raise ValueError('non-finite JSON constant ' + value)


def equal(a, b):
    # JSON booleans and numbers are distinct in the Rust evidence contract.
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def changes(prefix, a, b):
    if equal(a, b):
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(a.keys() | b.keys()):
            field = prefix + '/' + key
            if key not in a or key not in b:
                result.append(dict(field=field, expected=dict(present=key in a, value=a.get(key)),
                                   observed=dict(present=key in b, value=b.get(key))))
            else:
                result.extend(changes(field, a[key], b[key]))
        return result
    return [dict(field=prefix, expected=a, observed=b)]


def manifest_identity(manifest):
    result = dict(manifest)
    if isinstance(result.get('components'), list):
        result['components'] = {c['id']: c for c in result['components']}
    return result


def compatibility(a, b):
    for key in ('host', 'trace', 'vocabulary', 'topology', 'ranks', 'fold', 'capacity', 'sampling'):
        if a['invariants'].get(key) is None or b['invariants'].get(key) is None:
            raise ValueError('missing invariant ' + key)
    differences = changes('invariants', a['invariants'], b['invariants'])
    if differences:
        return 'incompatible ' + differences[0]['field']
    if (a.get('source_checkpoint') is not None and b.get('source_checkpoint') is not None
            and not equal(a['source_checkpoint'], b['source_checkpoint'])):
        return 'incompatible source_checkpoint'
    for key in ('expected_steps', 'expected_rows'):
        if a[key] != b[key]:
            return 'incompatible ' + key
    actual = (changes('manifest', a['manifest'], b['manifest'])
              + changes('environment', a['environment'], b['environment']))
    declared = b['declared_differences']
    if len(actual) != len(declared) or any(not any(equal(d, allowed) for allowed in declared) for d in actual):
        return 'undeclared or stale manifest/environment differences'
    return None


def load(root, mode):
    value = read_json(root / 'diagnostic.json')
    if value['format'] != FORMAT or value['mode'] != mode:
        raise ValueError('missing or unsupported diagnostic format/mode')
    artifacts = value['artifacts']
    if not isinstance(artifacts, dict):
        raise ValueError('invalid artifact table')
    if mode == 'fixed-schedule':
        desc = value['description']
        limits = desc['limits']
        if (value['contract'] != 'bitwise'
                or any(type(limits[key]) is not int or limits[key] <= 0
                       for key in ('metadata_bytes', 'logits_bytes', 'step_bytes'))
                or limits['metadata_bytes'] > MAX_METADATA_LIMIT):
            raise ValueError('unsupported comparison contract or capture budget')
        if (any(type(desc[key]) is not int or desc[key] <= 0
                for key in ('expected_steps', 'expected_rows'))
                or not isinstance(desc['manifest'], dict) or not isinstance(desc['environment'], dict)
                or not desc['reproduction'] or not all(isinstance(arg, str) for arg in desc['reproduction'])):
            raise ValueError('incomplete diagnostic description')
        allowed = {'frames.jsonl', 'rows.bin', 'receipt.json', 'mismatch.json', 'mismatch.bin'}
        if value['status'] in ('baseline', 'match'):
            if (value['completion'] != 'full' or value['completed_steps'] != desc['expected_steps']
                    or value['rows'] != desc['expected_rows'] or not value['rows']):
                raise ValueError('incomplete step/row coverage')
            required = {'frames.jsonl', 'receipt.json'}
            if value['status'] == 'baseline':
                required.add('rows.bin')
            if not required <= artifacts.keys():
                raise ValueError('missing required capture artifacts')
        elif value['status'] == 'diverged':
            if (value['completion'] != 'first-divergence' or not value['first']
                    or not {'frames.jsonl', 'mismatch.json', 'mismatch.bin'} <= artifacts.keys()):
                raise ValueError('incomplete first-divergence record')
    else:
        allowed = {'journal.txt'}
        limits = {'metadata_bytes': JOURNAL_LIMIT, 'logits_bytes': 0}
        if (value['completion'] != 'full' or not value['events']
                or value['events'] != value['expected_events'] or set(artifacts) != allowed):
            raise ValueError('incomplete journal')
    metadata_bytes = (root / 'diagnostic.json').stat().st_size
    for name, artifact in artifacts.items():
        if name not in allowed:
            raise ValueError('unknown artifact path')
        path = external(root / name)
        bound = limits['logits_bytes' if name.endswith('.bin') else 'metadata_bytes']
        if not name.endswith('.bin'):
            metadata_bytes += artifact['bytes']
        if path.stat().st_size > bound or path.stat().st_size != artifact['bytes'] or digest(path) != artifact['sha256']:
            raise ValueError('stale, truncated or over-budget artifact ' + name)
    if mode == 'fixed-schedule':
        if metadata_bytes > limits['metadata_bytes']:
            raise ValueError('metadata budget exceeded')
        if value['status'] == 'diverged':
            frame = read_json(root / 'mismatch.json')
            size = sum(row['byte_count'] for row in frame['rows'])
            if (size != artifacts['mismatch.bin']['bytes'] or size > limits['step_bytes']
                    or value['first']['step'] != value['completed_steps']
                    or value['completed_steps'] >= desc['expected_steps']):
                raise ValueError('incomplete first-divergence payload or coverage')
        if value['status'] in ('baseline', 'match') and read_json(root / 'receipt.json')['passed'] is not True:
            raise ValueError('execution receipt did not pass')
        if 'frames.jsonl' in artifacts:
            # Bound each line, count contiguous successful steps and rows, and
            # check that the retained byte stream covers exactly those frames.
            steps = rows = byte_count = 0
            with (root / 'frames.jsonl').open('rb') as stream:
                json.loads(stream.readline(limits['metadata_bytes'] + 1), parse_constant=invalid_constant)
                while line := stream.readline(limits['metadata_bytes'] + 1):
                    if len(line) > limits['metadata_bytes']:
                        raise ValueError('frame metadata budget exceeded')
                    frame = json.loads(line, parse_constant=invalid_constant)
                    steps += 1
                    rows += len(frame['rows'])
                    for row in frame['rows']:
                        width = {1: 4, 2: 2}[row['element_type']]
                        if row['width'] <= 0 or row['byte_count'] != row['width'] * width:
                            raise ValueError('invalid logits row shape')
                        byte_count += row['byte_count']
            if (steps, rows, byte_count) != (value['completed_steps'], value['rows'], value['bytes_checked']):
                raise ValueError('incomplete or trailing frames')
            if 'rows.bin' in artifacts and artifacts['rows.bin']['bytes'] != byte_count:
                raise ValueError('incomplete or trailing logits')
    return value


def bound_to(binding, root):
    return binding is not None and binding['sha256'] == digest(root / 'diagnostic.json')


def compare(reference, candidate, mode, scope=None, floor_reference=None):
    if mode == 'oracle':
        from tools.qual.oracle_diagnostics import compare as compare_oracle
        return compare_oracle(reference, candidate, scope, floor_reference)
    report = dict(format=FORMAT, mode=mode, status='insufficient', evidence='unavailable')
    try:
        if mode == 'journal':
            import lifecycle_diagnostics as lifecycle
            headers = [lifecycle.read_json(root / 'diagnostic.json') for root in (reference, candidate)]
            if any('extension' in header for header in headers):
                return lifecycle.compare(reference, candidate)
        a, b = load(reference, mode), load(candidate, mode)
        if mode == 'journal':
            if not equal(a['identity'], b['identity']):
                report.update(status='incompatible', reason='journal scenario/host identity differs')
                return report
            report.update(evidence='structural', reproduction=b['reproduction'],
                          producer_differences=changes('producer', a.get('producer'), b.get('producer')))
            step = None
            with (reference / 'journal.txt').open() as left, (candidate / 'journal.txt').open() as right:
                from itertools import zip_longest
                count = 0
                first = None
                for index, (expected, observed) in enumerate(zip_longest(left, right)):
                    count += 1
                    if observed and observed.startswith('step '):
                        step = observed.strip().removeprefix('step ')
                    if expected != observed and first is None:
                        first = dict(stage='journal', event=index, step=step,
                                     operation=(observed or expected).split(' ', 1)[0],
                                     expected=expected.rstrip() if expected else None,
                                     observed=observed.rstrip() if observed else None)
                if count != max(a['events'], b['events']):
                    raise ValueError('journal event count differs')
                # Verify each side, including an unequal-length prefix.
            for root, value in ((reference, a), (candidate, b)):
                with (root / 'journal.txt').open() as stream:
                    if sum(1 for _ in stream) != value['events']:
                        raise ValueError('journal event count differs')
            report.update(status='diverged' if first else 'match', first=first, events=count)
        else:
            desc = b['description']
            report.update(declared_differences=desc['declared_differences'], reproduction=desc['reproduction'])
            problem = compatibility(a['description'], desc)
            if problem:
                report.update(status='incompatible', reason=problem)
                return report
            if a['status'] != 'baseline' or not bound_to(b['reference'], reference):
                raise ValueError('comparison is not bound to this complete baseline')
            if b.get('control'):
                root = external(Path(b['control']['path']))
                control = load(root, mode)
                if (not bound_to(b['control'], root) or control['status'] != 'match'
                        or not bound_to(control['reference'], reference)
                        or control['description']['declared_differences']
                        or compatibility(a['description'], control['description']) is not None):
                    raise ValueError('baseline repeat is missing, stale or changes the baseline')
            if b['status'] not in ('match', 'diverged'):
                raise ValueError(b.get('reason', 'incomplete comparison'))
            report.update(status=b['status'], evidence='recorded-online-bitwise', first=b['first'],
                          completed_steps=b['completed_steps'], expected_steps=desc['expected_steps'],
                          rows=b['rows'], completion=b['completion'])
        if report.get('reproduction'):
            report['command'] = shlex.join(report['reproduction'])
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        report.update(status='insufficient', reason=str(error), evidence='unavailable')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['compare', 'seal', 'reuse'])
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--mode', choices=['fixed-schedule', 'journal', 'oracle'], required=True)
    parser.add_argument('--floor-reference', type=Path)
    parser.add_argument('--scope', type=Path, help='explicit narrower oracle scope, outside the repository')
    args = parser.parse_args()
    if args.action == 'reuse':
        from tools.qual.oracle_diagnostics import reuse
        if args.mode!='oracle' or not args.scope: parser.error('reuse requires oracle mode and an explicit scope')
        report=reuse(args.candidate,read_json(args.scope))
        print(json.dumps(report,indent=2,allow_nan=False))
        return 0 if report['status']=='reusable' else CODES[report['status']]
    if args.action == 'seal':
        from tools.model_sdk.diagnostic import seal
        if args.mode != 'oracle':
            parser.error('offline sealing applies to oracle producers')
        try:
            value = seal(args.candidate)
            print(json.dumps({'status':'sealed', 'completion':value['completion']}, allow_nan=False))
            return 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(json.dumps({'status':'insufficient','reason':str(error)}))
            return CODES['insufficient']
    if args.reference is None:
        parser.error('compare requires --reference')
    report = compare(args.reference, args.candidate, args.mode, read_json(args.scope) if args.scope else None, args.floor_reference)
    if args.scope and report.get('reproduction'):
        report['reproduction'] += ['--scope', str(args.scope)]
        report['command'] = shlex.join(report['reproduction'])
    print(json.dumps(report, indent=2, allow_nan=False))
    return CODES[report['status']]


if __name__ == '__main__':
    sys.exit(main())

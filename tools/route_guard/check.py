#!/usr/bin/env python3
"""Check route metadata against journals, Nsight SQLite exports and step records.

Exit 0: covered. Exit 1: unallowed fallback. Exit 2: invalid or ambiguous evidence.
"""
import argparse
import collections
import json
import pathlib
import sqlite3
import sys
import tomllib
from fnmatch import fnmatchcase


class EvidenceError(ValueError):
    pass


def load_classes(path):
    document = json.loads(path.read_text())
    if document.get('schema') != 'chopinfer.route-tags.v1':
        raise EvidenceError('unsupported classes schema')
    entries = {}
    symbols = collections.defaultdict(list)
    for tag in document['tags']:
        if tag['class'] not in ('Fast', 'Baseline', 'Reference'):
            raise EvidenceError('invalid route class')
        identity = tag['entry']
        if identity in entries:
            raise EvidenceError(f'duplicate route entry {identity}')
        entries[identity] = tag
        symbols[tag['symbol']].append(tag)
    return document, entries, symbols


def journal(path):
    counts = collections.Counter()
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == 'launch':
            if len(fields) < 3:
                raise EvidenceError('malformed journal launch')
            counts[fields[2]] += 1
    if not counts:
        raise EvidenceError('journal has no launches')
    return {symbol: (count, 0) for symbol, count in counts.items()}


def nsys(path):
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        columns = {row[1] for row in db.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
        if not {'start', 'end', 'shortName'} <= columns:
            raise EvidenceError('Nsight export needs kernel start/end/shortName')
        rows = db.execute('SELECT s.value, COUNT(*), SUM(k.end-k.start) '
                          'FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName '
                          'GROUP BY s.value').fetchall()
    if not rows:
        raise EvidenceError('Nsight export has no kernel launches')
    return {symbol: (count, ns) for symbol, count, ns in rows}


def steps(paths):
    counts = collections.Counter()
    records = 0
    for path in paths:
        for line in path.read_text().splitlines():
            if '"chopinfer.step.v1"' not in line:
                continue
            record = json.loads(line[line.index('{'):])
            route = record.get('route')
            if not isinstance(route, dict) or not isinstance(route.get('entries'), dict):
                raise EvidenceError('step is missing selected-entry route evidence')
            total = 0
            families = collections.Counter()
            for entry, value in route['entries'].items():
                count = value['launches']
                if type(count) is not int or count < 0:
                    raise EvidenceError('invalid selected-entry launch count')
                if value['class'] != 'Fast':
                    total += count
                    families[value['family']] += count
                counts[entry] += count
            if total != route['fallback_launches'] or dict(families) != route['families']:
                raise EvidenceError('inconsistent selected-entry totals')
            records += 1
    if paths and not records:
        raise EvidenceError('no step records')
    return counts


def allowances(path):
    if path is None:
        return []
    if path.suffix == '.toml':
        value = tomllib.loads(path.read_text())
    else:
        value = json.loads(path.read_text())
    if 'allow_fallbacks' in value:
        result = []
        for family, reason in value['allow_fallbacks'].items():
            if not isinstance(reason, str) or not reason.strip():
                raise EvidenceError('allow_fallbacks needs a reason for every family')
            result.append({'family': family, 'symbol': '*', 'entry': '*'})
        return result
    result = value.get('allow', [])
    for allow in result:
        if not allow.get('reason', '').strip() or not allow.get('task', '').strip():
            raise EvidenceError('allowlist entry needs reason and task')
    return result


def allowed(tag, allow, profile, phase, rows):
    for item in allow:
        if item['family'] != tag['family'] or not fnmatchcase(tag['symbol'], item.get('symbol', '*')):
            continue
        if not fnmatchcase(tag['entry'], item.get('entry', '*')):
            continue
        if 'profiles' in item and (profile is None or not any(fnmatchcase(profile, p) for p in item['profiles'])):
            continue
        if 'phase' in item and item['phase'] != phase:
            continue
        if 'rows' in item and (rows is None or not any(lo <= rows <= hi for lo, hi in item['rows'])):
            continue
        return True
    return False


def check(args):
    document, entries, classes = load_classes(args.classes)
    allow = allowances(args.allow)
    selected = steps(args.steps)
    profile = args.profile or document.get('profile')
    observed = collections.defaultdict(lambda: [0, 0])
    for paths, reader in ((args.journal, journal), (args.nsys, nsys)):
        for path in paths:
            for symbol, (count, ns) in reader(path).items():
                observed[symbol][0] += count
                observed[symbol][1] += ns
    failures, ambiguous = [], []
    for symbol, (count, ns) in sorted(observed.items()):
        tags = classes.get(symbol, [])
        kinds = {tag['class'] for tag in tags}
        if len(kinds) > 1:
            ambiguous.append(symbol)
            print(f'ambiguous symbol={symbol} launches={count} device_ns={ns} (time unattributed)')
        elif kinds and kinds != {'Fast'}:
            tag = tags[0]
            print(f"{tag['class']} family={tag['family']} symbol={symbol} launches={count} device_ns={ns}")
            if not args.steps and not all(allowed(t, allow, profile, args.phase, args.rows) for t in tags):
                failures.append(symbol)
    for entry, count in sorted(selected.items()):
        if entry not in entries:
            raise EvidenceError(f'step entry missing from classes: {entry}')
        tag = entries[entry]
        if count and tag['class'] != 'Fast':
            print(f"{tag['class']} family={tag['family']} entry={entry} symbol={tag['symbol']} launches={count}")
            if not allowed(tag, allow, profile, args.phase, args.rows):
                failures.append(entry)
    if failures:
        print('unallowed fallbacks: ' + ', '.join(failures), file=sys.stderr)
        return 1
    if ambiguous and not args.steps:
        raise EvidenceError('ambiguous symbols require selected-entry --steps evidence')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classes', type=pathlib.Path, required=True)
    parser.add_argument('--allow', type=pathlib.Path)
    for name in ('journal', 'nsys', 'steps'):
        parser.add_argument('--' + name, type=pathlib.Path, action='append', default=[])
    parser.add_argument('--profile')
    parser.add_argument('--phase', choices=('prefill', 'decode', 'verify'))
    parser.add_argument('--rows', type=int)
    args = parser.parse_args(argv)
    if not (args.journal or args.nsys or args.steps):
        parser.error('provide --journal, --nsys or --steps')
    try:
        return check(args)
    except (EvidenceError, OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        print(f'route evidence: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

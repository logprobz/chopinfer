"""Run one prepared case per stage symbol, or every case with --all."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

DENSE = ('ops', 'gemm::generated_dense::generated_dense_offers_have_oracles_and_replay_invariance')
CATALOG = ('ops', 'generated_projections::generated_catalog_offers_have_oracles_and_replay_invariance')
GROUPED = ('generated_projection_rows', 'generated_grouped_offers_have_oracles_and_invariance')
LOOKUP = ('generated_projection_rows', 'generated_lookup_offers_have_oracles_and_invariance')
COPY = ('generated_projection_rows', 'generated_layout_copies_are_exact_and_replay_invariant')
SELECTORS = (DENSE, CATALOG, GROUPED, LOOKUP, COPY)


def selected(cases, symbols=None, all_cases=False, case_indices=None):
    entries = {}
    for i, case in enumerate(cases['operations']):
        kind = case['capability']['kind']
        selector = CATALOG if i >= cases['typed_count'] else (
            DENSE if kind == 'Dense' else LOOKUP if kind == 'Lookup' else GROUPED)
        entries[i] = (selector, case)
    entries.update({i: (COPY, case) for i, case in enumerate(cases['copies'], len(cases['operations']))})
    if case_indices is not None:
        if symbols is not None or all_cases or not case_indices:
            raise ValueError('explicit cases require an exclusive nonempty selection')
        if any(type(i) is not int or i not in entries for i in case_indices) or len(set(case_indices)) != len(case_indices):
            raise ValueError('invalid or duplicate prepared case index')
        return {i: entries[i] for i in case_indices}
    candidates = entries if all_cases else {i: entries[i] for i in cases['representatives']}
    def stages(entry):
        selector, case = entry
        return {case['symbol']} if selector == COPY else set(case['capability']['stages'])
    if symbols is not None:
        symbols = set(symbols)
        available = set().union(*(stages(entry) for entry in candidates.values()))
        if not symbols or symbols - available:
            raise ValueError(f'unknown or empty symbol selection: {sorted(symbols - available)}')
        candidates = {i: entry for i, entry in candidates.items() if stages(entry) & symbols}
    if not candidates:
        raise ValueError('selection has no cases')
    return candidates


def coverage(expected, receipts):
    observed = set()
    wrong_output_refused = False
    for record in receipts:
        if 'case_index' not in record:
            continue
        i = record['case_index']
        if type(i) is not int or i not in expected or i in observed:
            raise ValueError(f'unexpected or duplicate case index: {i}')
        selector, case = expected[i]
        if selector == COPY:
            valid = record.get('exact') is True and all(
                record.get(key) == case[key] for key in ('symbol', 'capacity'))
        else:
            capability = record.get('capability', record.get('case', {}).get('capability'))
            valid = record.get('oracle') is True and record.get('replay') is True and capability == case['capability']
        if selector == DENSE:
            valid = valid and record.get('eager_vs_graph') is True
            wrong_output_refused |= record.get('wrong_output_refused') is True
        if not valid:
            raise ValueError(f'invalid evidence for case {i}')
        observed.add(i)
    if observed != set(expected):
        raise ValueError(f'missing case indices: {sorted(set(expected) - observed)}')
    if any(selector == DENSE and case['capability']['usage']['capacity'] >= 64 for selector, case in expected.values()) and not wrong_output_refused:
        raise ValueError('missing finite wrong-output rejection on row 47')
    return {'case_indices': sorted(observed), 'complete': True}


def run(prepared, output, symbols=None, all_cases=False, case_indices=None, dry_run=False):
    """Run the selected cases. A dry run makes every check up to the first
    GPU launch (manifest, cases, inventory, binary provenance and selectors)
    and records the commands it would run in `dry-run.json`."""
    manifest = json.loads((prepared / 'prepared.json').read_text())
    case_bytes = (prepared / 'artifacts/cases.json').read_bytes()
    if hashlib.sha256(case_bytes).hexdigest() != manifest['cases_sha256']:
        raise ValueError('case manifest hash differs')
    expected = selected(json.loads(case_bytes), symbols, all_cases, case_indices)
    needed = {selector for selector, _ in expected.values()}
    configured = {(name, test) for name, tests in manifest['tests'].items() for test in tests}
    if configured != set(SELECTORS):
        raise ValueError('prepared test inventory differs')
    results = []
    planned = []
    env = dict(os.environ, CHOPINFER_PROJECTION_CASES=",".join(map(str, expected)))
    for name, selector in SELECTORS:
        if (name, selector) not in needed:
            continue
        record = manifest['binaries'][name]
        binary = Path(record['path']).resolve()
        if not binary.is_relative_to(prepared.resolve() / 'target') or hashlib.sha256(binary.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'binary provenance differs: {name}')
        available = subprocess.check_output([str(binary), '--list'], text=True)
        if f'{selector}: test' not in available.splitlines():
            raise ValueError(f'missing selector: {selector}')
        command = [str(binary), '--exact', selector, '--test-threads=1', '--nocapture']
        if dry_run:
            planned.append({'test': selector, 'command': command})
            continue
        result = subprocess.run(command, env=env, check=False)
        results.append({'test': selector, 'exit': result.returncode})
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'{selector} exited {result.returncode}')
    if dry_run:
        if not planned:
            raise ValueError('dry run selected no test')
        (output / 'dry-run.json').write_text(json.dumps({
            'revision': manifest['revision'], 'cases': sorted(expected), 'tests': planned}, indent=2) + '\n')
        return planned
    evidence = coverage(expected, (json.loads(p.read_text()) for p in (output / 'receipts').rglob('*.json')))
    evidence['mode'] = 'explicit-cases' if case_indices is not None else 'all' if all_cases else 'per-symbol'
    evidence['executed_tests'] = len(results)
    (output / 'coverage.json').write_text(json.dumps(evidence, indent=2) + '\n')
    (output / 'revision.txt').write_text(manifest['revision'] + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path)
    parser.add_argument('--cases', type=Path, help='JSON array of prepared case indices, for family qualification')
    parser.add_argument('--symbols', type=Path, help='JSON array of changed stage symbols')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--all', action='store_true', help='run every use instead of symbol representatives')
    parser.add_argument('--dry-run', action='store_true', help='check everything up to the first GPU launch')
    args = parser.parse_args()
    symbols = json.loads(args.symbols.read_text()) if args.symbols else None
    indices = json.loads(args.cases.read_text()) if args.cases else None
    run(args.prepared, args.output, symbols, args.all, indices, args.dry_run)


if __name__ == '__main__':
    main()

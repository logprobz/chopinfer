#!/usr/bin/env python3
"""Declare the captured-graph node-count changes a package spec implies.

A merge, removal or expansion of a pinned launch inside a graph capture
changes that graph's recorded node count. For every fixture this reads the
base device history and the spec's own journal and history policies, counts the base
graphs whose capture interval gains or loses launches under those policies,
and writes that number as `history_policy.graph_nodes`. With
`--piece-policy A B` (or a spec `graph_piece_policy`), a program captured as
several pieces is re-split by that source-authenticated policy over its
expected head launch total, and the number of re-split programs is written
as `history_policy.graph_pieces`. The head's node counts are never read;
the checker separately requires every node count on both sides to equal
the launches recorded inside its capture.

Usage: graph_nodes.py --repo REPO --model ID --base REV --head REV
       --base-index INDEX --head-index INDEX --spec SPEC --out SPEC [--piece-policy A B]
"""
import argparse
import collections
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
TOOL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOL))
from common import need  # noqa: E402
from pins import snapshot  # noqa: E402
from proof import read_spec  # noqa: E402
from auxproof import load, derive  # noqa: E402
from allocation_args import bindings  # noqa: E402
from engine import transform  # noqa: E402
import history  # noqa: E402


def expected_launches(notes, start, end, multiplicity):
    return sum(multiplicity.get(t, 1) for t in range(start + 1, end) if notes[t].startswith('launch '))


def changed_graphs(notes, multiplicity, divisors=None):
    """Base graphs whose expected head node count differs, and programs whose
    pieces re-split. With a piece policy, a program's expected head pieces
    are the policy applied to its expected head launch total; otherwise a
    graph keeps its own launches."""
    in_program = {}
    resplit = 0
    if divisors:
        for program in history.programs(notes, divisors):
            sizes = [n for _, _, n in program]
            total = expected_launches(notes, program[0][0], program[-1][1], multiplicity)
            new = history.piece_sizes(total, divisors)
            need(len(new) == len(sizes), 'a re-split program would change its piece count')
            resplit += new != sizes
            for (_, end, _), size in zip(program, new):
                in_program[end] = size
    open_ = {}
    changed = 0
    for i, line in enumerate(notes):
        if line.startswith('capture '):
            open_[line.split()[1]] = i
        elif (m := history.CAPTURED.fullmatch(line)) and m[1] in open_:
            start = open_.pop(m[1])
            size = in_program.get(i, expected_launches(notes, start, i, multiplicity))
            changed += size != int(m[3])
    return changed, resplit


def multiplicities(lines, policy, context, positions, out):
    """Head launches per base line under one policy: a merge keeps one launch
    for its first source and none for the rest, an expansion one per member,
    a removal none."""
    trace = []
    transform(lines, policy, collections.Counter(), context, trace)
    seen = collections.Counter(j for source in trace for j in source)
    merged = {j for source in trace if len(source) > 1 for j in source[1:]}
    for j, position in enumerate(positions):
        out[position] = 0 if j in merged else seen[j]


def generate(a):
    spec = read_spec(a.spec)
    base, b, bo = snapshot(a.repo, a.base, a.model, a.base_index)
    head, h, ho = snapshot(a.repo, a.head, a.model, a.head_index)
    ba = load(base, a.base_index, b, bo)
    ha = load(head, a.head_index, h, ho)
    from replacements import authenticate
    replacements = authenticate(a.repo, base['commit'], head['commit'], spec.get('module_replacements', []), base['root'])
    divisors = None
    if a.piece_policy:
        spec['graph_piece_policy'] = dict(source=history.piece_source(a.repo, base['commit']), divisors=a.piece_policy)
    if 'graph_piece_policy' in spec:
        divisors = history.authenticate_piece_policy(a.repo, base['commit'], head['commit'],
                                                     spec['graph_piece_policy'])['divisors']
    totals = collections.Counter()
    for fixture in spec['fixtures']:
        # Device journals only: text journals and declared additions or
        # removals have no device history to count.
        names = [n for n in fixture['journals'] if n in b and n in h and base['pins'][n]['kind'] == 'device']
        if not names:
            continue
        rule = spec['auxiliary_rules'][fixture['name']]
        ab = [ba['manifest']['bindings'][n] for n in names]
        hb = [ha['manifest']['bindings'][n] for n in names]
        ae = ba['hosts'][ab[0]['host']]['events'][:max(v['records'] for v in ab)]
        be = ha['hosts'][hb[0]['host']]['events'][:max(v['records'] for v in hb)]
        buffers, _, _, _, ai, bi = derive(ae, be, rule, replacements)
        context = bindings(rule, ai, bi, buffers)
        notes = [e[1] for e in ae if e[0] == 'note']
        owner = history.owners(notes, dict(zip(names, ab)), ab[0]['create'])
        multiplicity = {}
        for name, binding in zip(names, ab):
            multiplicities(b[name], spec.get('journals', {}).get(name, {}), context, binding['positions'], multiplicity)
        hist = [i for i, o in enumerate(owner) if o == 'history']
        multiplicities([notes[i] for i in hist], rule.get('history_policy', {}), context, hist, multiplicity)
        count, resplit = changed_graphs(notes, multiplicity, divisors)
        policy = rule.setdefault('history_policy', {})
        need('graph_nodes' not in policy and 'graph_pieces' not in policy,
             f'{fixture["name"]}: graph declarations already present')
        if count:
            policy['graph_nodes'] = [dict(count=count)]
            totals['fixtures_with_graph_node_changes'] += 1
            totals['graph_node_changes'] += count
        if resplit:
            policy['graph_pieces'] = [dict(count=resplit)]
            totals['resplit_programs'] += resplit
    Path(a.out).write_text(json.dumps(spec, indent=1) + '\n')
    print(json.dumps(dict(totals), sort_keys=True))


if __name__ == '__main__':
    os.umask(0o022)
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key in ('repo', 'model', 'base', 'head', 'base-index', 'head-index', 'spec', 'out'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--piece-policy', type=int, nargs=2, metavar=('A', 'B'),
                   help='declare the graph piece split [0, L/A, L/B, L]; checked against both revisions')
    generate(p.parse_args())

"""Fixed capture coverage, exact comparisons, and validated speculation counters."""
from decimal import Decimal, InvalidOperation
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

from common import Refusal, need, read, write, digest, finite

HERE = Path(__file__).resolve().parent
# The client sizes each prompt through /tokenize to the largest word count
# whose prompt is at most the phase's target, so a prompt is at most one word
# (a few tokens) short of it.
SLACK = 16
PREFIX_TOKENS = 3072
# A decode request is the shared prefix plus a tail of TAIL_WORDS words and
# its label, together at most TAIL_TOKENS tokens in any tokenizer.
TAIL_WORDS = 48
TAIL_TOKENS = 256
SPECULATIVE_KINDS = ('decode', 'gsm8k')


def catalog():
    out = {}
    def add(name, kind, tokens):
        spec = dict(kind=kind, gated=True, requests=1, max_tokens=1, top=5,
                    prompt_tokens=dict(min=tokens - SLACK, max=tokens))
        if kind == 'teacher':
            spec['teacher'] = dict(offset=1, prompt_ids='score_map_text')
        out[name] = dict(name=name, kind=kind, args=['--tokens', str(tokens)], spec=spec, timeout=180)
    for name in ['prefill-32k', 'prefill-32k-b']:
        add(name, 'prefill', 32700)
    add('prefill-9k', 'prefill', 9000)
    for name in ['prefill-3k', 'prefill-3k-b']:
        add(name, 'prefill', 3000)
    for name, tokens in [('teacher64', 64), ('teacher1040', 1040), ('teacher2060', 2060)]:
        add(name, 'teacher', tokens)
    prefix = ['--prefix-tokens', str(PREFIX_TOKENS)]
    out['warm'] = dict(name='warm', kind='warm', args=prefix, timeout=120)
    for count in [1, 4, 8, 16, 32]:
        name = f'decode-c{count}'
        out[name] = dict(name=name, kind='decode', args=[*prefix, '--concurrency', str(count), '--max-tokens', '128'],
            timeout=180, spec=dict(kind='decode', gated=True, requests=count, max_tokens=128, top=1,
                                  prompt_tokens=dict(min=count * (PREFIX_TOKENS - SLACK + TAIL_WORDS),
                                                     max=count * (PREFIX_TOKENS + TAIL_TOKENS))))
    for count in [1, 8]:
        name = f'gsm8k-c{count}'
        out[name] = dict(name=name, kind='gsm8k', args=['--count', '8', '--concurrency', str(count),
            '--max-tokens', '256'], timeout=240, requests=8, concurrency=count, max_tokens=256)
    # tools/qual/determinism.py's request set, compared by its own rules.
    out['determinism'] = dict(name='determinism', kind='determinism', args=[], timeout=600)
    return out


# Phases a run file must name: decode at 32 streams, the sequence limit
# many serve profiles declare. The standard and short sets leave them out.
OPT_IN = ('decode-c32',)
STANDARD = [name for name, phase in catalog().items()
            if phase['kind'] not in ('gsm8k', 'determinism') and name not in OPT_IN]
# A smoke run (run file "smoke": true) covers at least these: the
# determinism.py request set and single-stream decode. A model need not serve
# concurrent streams bit for bit alike across runs (batch composition follows
# arrival timing), so batched phases are additions.
SMOKE = ['determinism', 'warm', 'decode-c1']
SHORT = ['prefill-3k', 'prefill-3k-b', 'teacher64', 'teacher1040', 'warm',
       'decode-c1', 'decode-c4', 'decode-c8', 'decode-c16']


def phases_from(value):
    available = catalog()
    if value == 'standard':
        value = STANDARD
    elif value == 'short':
        value = SHORT
    need(type(value) is list and value and all(type(x) is str and x in available for x in value),
         'phases must be standard, short, or a nonempty list of catalog names')
    need(len(value) == len(set(value)), 'duplicate phase names')
    need(any('spec' in available[x] for x in value), 'at least one numerical phase is required')
    if any(available[x]['kind'] == 'decode' for x in value):
        need('warm' in value and value.index('warm') < min(i for i, x in enumerate(value)
             if available[x]['kind'] == 'decode'), 'decode phases require an earlier warm phase')
    return [available[x] for x in value]


def comparison_spec(phases):
    return dict(version=2, phases={p['name']: p['spec'] for p in phases if 'spec' in p},
                ignored_phases=[p['name'] for p in phases if 'spec' not in p])


def warm_complete(phase):
    """A decode prefix warmup that completed: one request of the prefix
    length, one choice, and its terminal status. max_tokens is 1, so a length
    finish carries one generated token and a stop (end of sequence) at most
    one; an empty choice list or an absent finish reason is not a completion."""
    need(phase.get('requests') == 1 and type(phase.get('prompt_tokens')) is int
         and PREFIX_TOKENS - SLACK <= phase['prompt_tokens'] <= PREFIX_TOKENS, 'invalid decode warmup')
    outputs = phase.get('outputs')
    need(phase.get('choices') == 1 and type(outputs) is list and len(outputs) == 1 and type(outputs[0]) is dict,
         'decode warmup returned no completion choice')
    output, tokens = outputs[0], phase.get('completion_tokens')
    ids = output.get('token_ids')
    need(type(output.get('text')) is str and type(ids) is list and type(tokens) is int and len(ids) == tokens
         and all(type(x) is int and 0 <= x <= 0xffffffff for x in ids), 'decode warmup without its generated tokens')
    need((output.get('finish_reason') == 'length' and tokens == 1)
         or (output.get('finish_reason') == 'stop' and tokens <= 1), 'decode warmup did not finish')


def finite_tree(value):
    if type(value) is float:
        need(math.isfinite(value), 'nonfinite capture value')
    elif type(value) is dict:
        for child in value.values():
            finite_tree(child)
    elif type(value) is list:
        for child in value:
            finite_tree(child)


class Comparator:
    """compare.py, loaded from a given copy (a run's tool snapshot)."""

    def __init__(self, path=HERE / 'compare.py', qual=None):
        self.path = Path(path)
        self.qual = Path(qual) if qual else None
        self.sha256 = digest(path)
        sys.path.insert(0, str(self.path.parent))
        try:
            module = importlib.util.spec_from_file_location('ab_compare_' + self.sha256[:12], self.path)
            self.module = importlib.util.module_from_spec(module)
            module.loader.exec_module(self.module)
        finally:
            sys.path.remove(str(self.path.parent))

    def validate_phase(self, phase, declaration):
        finite_tree(phase)
        need(phase.get('phase') == declaration['name'] and phase.get('kind') == declaration['kind'],
             'phase name/kind differs from plan')
        finite(phase.get('wall_ms'), 'phase wall_ms', 0)
        need(type(phase.get('start_unix_ns')) is int and type(phase.get('end_unix_ns')) is int
             and phase['end_unix_ns'] >= phase['start_unix_ns'], 'missing/invalid phase times')
        kind = declaration['kind']
        if kind == 'determinism':
            receipt = phase.get('receipt')
            need(type(receipt) is str and Path(receipt).is_file() and digest(receipt) == phase.get('receipt_sha256'),
                 'determinism receipt missing or changed')
            record = read(receipt)
            need(record.get('passed') is True and type(record.get('requests')) is list and record['requests']
                 and phase.get('requests') == len(record['requests']), 'determinism record failed or is empty')
            return
        if kind == 'warm':
            warm_complete(phase)
            return
        if kind == 'gsm8k':
            need(type(phase.get('requests')) is int and phase['requests'] == declaration['requests']
                 and phase.get('concurrency') == declaration['concurrency'], 'GSM8K request coverage differs')
            outputs = phase.get('outputs')
            need(type(outputs) is list and len(outputs) == declaration['requests'], 'missing GSM8K answers')
            need(type(phase.get('prompt_tokens')) is int and phase['prompt_tokens'] > 0,
                 'missing GSM8K prompts')
            total = 0
            for output in outputs:
                need(type(output) is dict and type(output.get('completion_tokens')) is int
                     and 0 < output['completion_tokens'] <= declaration['max_tokens'], 'invalid GSM8K token count')
                need(all(output.get(k) is None or type(output[k]) is str for k in ('text', 'reasoning'))
                     and bool(output.get('text') or output.get('reasoning')), 'empty GSM8K answer')
                need(output.get('finish_reason') in ('stop', 'length'), 'incomplete GSM8K response')
                need(type(output.get('token_ids')) is list and len(output['token_ids']) == output['completion_tokens']
                     and all(type(x) is int and 0 <= x <= 0xffffffff for x in output['token_ids']),
                     'GSM8K answer without one generated token ID per completion token')
                total += output['completion_tokens']
            need(phase.get('completion_tokens') == total, 'GSM8K total count mismatch')
            return
        try:
            self.module.validate(phase, declaration['spec'], declaration['name'])
        except (ValueError, KeyError, TypeError, IndexError, self.module.Invalid) as error:
            raise Refusal(f'invalid capture: {error}') from error

    def capture(self, directory, phases):
        record = read(Path(directory) / 'capture.json')
        need(type(record) is dict and type(record.get('model')) is str and record['model'], 'missing captured model')
        rows = record.get('phases')
        need(type(rows) is list and len(rows) == len(phases), 'missing or extra capture phases')
        need([p.get('phase') for p in rows] == [p['name'] for p in phases], 'phase order/inventory mismatch')
        for row, phase in zip(rows, phases):
            self.validate_phase(row, phase)
        return record

    def compare(self, before, after, phases, output, commands, timeout, required=True, scorer=None):
        a, b = self.capture(before, phases), self.capture(after, phases)
        need(a['model'] == b['model'], 'captured models differ')
        output = Path(output)
        need(not output.exists(), f'refusing a stale comparison report: {output}')
        spec_path = output.with_suffix('.spec.json')
        write(spec_path, comparison_spec(phases))
        result = commands.run([sys.executable, '-B', str(self.path), str(before), str(after),
                               str(output), '--spec', str(spec_path)], timeout,
                              stdout_path=output.with_suffix('.log'))
        need(output.is_file() and result.returncode in (0, 1), 'strict comparator failed or emitted no fresh report')
        report = read(output)
        need(report.get('valid') is True, 'strict comparator rejected evidence')
        need(not required or result.returncode == 0, 'strict comparator found different gated outputs')
        gsm = []
        for left, right, phase in zip(a['phases'], b['phases'], phases):
            if phase['kind'] == 'gsm8k':
                identical = (self.module.same(left['outputs'], right['outputs'])
                     and left['prompt_tokens'] == right['prompt_tokens']
                     and left['completion_tokens'] == right['completion_tokens'])
                gsm.append(dict(phase=phase['name'], identical=identical))
        write(output.with_suffix('.gsm.json'), dict(required=required, phases=gsm))
        need(not required or all(p['identical'] for p in gsm), 'GSM8K answers or counts differ')
        records = []
        for left, right, phase in zip(a['phases'], b['phases'], phases):
            if phase['kind'] == 'determinism':
                # determinism.py compare: identical answers across the two
                # records, and within each its cold repeats and prefix hits.
                need(self.qual is not None and scorer is not None, 'determinism phase without its tools')
                report = output.with_name(f'{output.stem}.{phase["name"]}.json')
                run = commands.run([sys.executable, '-B', str(self.qual / 'determinism.py'), 'compare',
                                    left['receipt'], right['receipt']], timeout,
                                   env=dict(os.environ, CHOPINFER_EVAL=str(scorer)), stdout_path=report)
                need(run.returncode in (0, 1), f'determinism compare failed with {run.returncode}')
                records.append(dict(phase=phase['name'], identical=run.returncode == 0, report=str(report)))
        need(not required or all(p['identical'] for p in records), 'determinism records differ')
        return dict(comparator_sha256=self.sha256, gsm=gsm, determinism=records, report=str(output),
                    report_sha256=digest(output),
                    identical=result.returncode == 0 and all(p['identical'] for p in gsm + records))


SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)\s*$')
LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"(?:,|$)')
TOTALS = ['chop_spec_verify_items_total', 'chop_spec_draft_tokens_total',
          'chop_spec_accepted_tokens_total', 'chop_spec_emitted_tokens_total', 'chop_spec_rows_total']
POSITIONS = ['chop_spec_drafted_position_total', 'chop_spec_accepted_position_total']
K = 'chop_speculation_k_total'


def parse_metrics(text):
    need(isinstance(text, str) and text.strip(), 'empty metrics response')
    values = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        match = SAMPLE.fullmatch(line)
        need(match is not None, 'malformed metrics sample')
        name, raw_labels, raw_value = match.groups()
        labels = {}
        end = 0
        for label in LABEL.finditer(raw_labels or ''):
            need(label.start() == end and label[1] not in labels, 'malformed/duplicate metric labels')
            labels[label[1]] = json.loads('"' + label[2] + '"')
            end = label.end()
        need(end == len(raw_labels or ''), 'malformed metric label tail')
        key = (name, tuple(sorted(labels.items())))
        need(key not in values, 'duplicate metric series')
        try:
            value = Decimal(raw_value)
        except InvalidOperation as error:
            raise Refusal('unparsable metric value') from error
        need(value.is_finite() and value >= 0, 'nonfinite or negative metric')
        values[key] = value
    need(('chop_requests_completed_total', ()) in values and ('chop_requests_waiting', ()) in values,
         'response is not a complete serving metrics snapshot')
    selected = {}
    for (name, labels), value in values.items():
        if name not in TOTALS + POSITIONS + [K]:
            continue
        need(value == value.to_integral_value() and value <= 2**64 - 1, 'speculation counter is not u64')
        selected[(name, labels)] = int(value)
    present = {name for name, _ in selected}
    # Initial zeros are deliberately omitted by MetricsSnapshot::prometheus.
    if present & set(TOTALS + POSITIONS):
        need(set(TOTALS + POSITIONS) <= present, 'partial speculation counter family')
        for name in TOTALS:
            need((name, ()) in selected and sum(n == name for n, _ in selected) == 1,
                 'invalid speculation total labels')
        positions = []
        for name in POSITIONS:
            records = {dict(labels).get('position'): value for (n, labels), value in selected.items() if n == name}
            need(all(len(labels) == 1 and labels[0][0] == 'position' for n, labels in selected if n == name),
                 'unexpected position labels')
            need(records and all(x and re.fullmatch(r'[1-9][0-9]*', x) for x in records), 'invalid position IDs')
            need(set(map(int, records)) == set(range(1, 17)), 'missing position counter')
            positions.append(records)
        need(positions[0].keys() == positions[1].keys(), 'position counter sets differ')
    if K in present:
        groups = {}
        for (name, labels), value in selected.items():
            if name != K:
                continue
            labels = dict(labels)
            need(set(labels) == {'model_name', 'engine', 'k'}, 'invalid effective-K labels')
            need(re.fullmatch(r'0|[1-9][0-9]*', labels['k']), 'invalid K index')
            groups.setdefault((labels['model_name'], labels['engine']), set()).add(int(labels['k']))
        need(len(groups) == 1 and next(iter(groups.values())) == set(range(17)), 'effective-K inventory is incomplete or mixed-model')
    if present & set(TOTALS):
        need(K in present, 'speculation counters lack effective-K histogram')
    return selected


def fixed_k_exposure(record, fixed_k, where='phase'):
    """Verification items at exactly `fixed_k` drafts, not accepted tokens or steps.

    The server counts appended_tokens - 1 per non-prefill item in the K
    histogram. acceptance() ties its positive bins to completed verify items,
    drafted tokens and every drafted-position counter, so with no bins above
    fixed_k, histogram[fixed_k] equals drafted_position[fixed_k].
    """
    need(type(fixed_k) is int and 1 <= fixed_k <= 16, 'invalid required fixed K')
    hist = record.get('k_histogram')
    need(type(hist) is dict and set(hist) == {str(k) for k in range(17)}
         and all(type(n) is int and n >= 0 for n in hist.values()),
         f'{where}: missing or invalid effective-K exposure histogram')
    count = hist[str(fixed_k)]
    need(count > 0, f'{where}: no verifications at configured fixed K={fixed_k}')
    return count


def acceptance(before, after, *, speculative, maximum, require_drafts, fixed_k=None):
    """Speculation counter deltas over one phase. A drafting phase of a fixed-K
    arm must verify at least one item at that K; prefill, teacher and warm
    phases need not verify, and adaptive mode only needs positive-K coverage."""
    if fixed_k is not None:
        need(type(fixed_k) is int and 1 <= fixed_k <= 16 and speculative and fixed_k == maximum,
             'fixed K must equal the speculative depth')
    a, b = parse_metrics(before), parse_metrics(after)
    need(a.keys() <= b.keys(), 'speculation counter family disappeared')
    delta = {key: value - a.get(key, 0) for key, value in b.items()}
    need(all(value >= 0 for value in delta.values()), 'speculation counters decreased/reset')
    drafted = delta.get(('chop_spec_draft_tokens_total', ()), 0)
    accepted = delta.get(('chop_spec_accepted_tokens_total', ()), 0)
    need(accepted <= drafted, 'accepted exceeds drafted')
    if not speculative:
        need(drafted == accepted == 0, 'AR arm drafted tokens')
    if require_drafts:
        need(drafted > 0, 'speculative decode phase has no drafted-token coverage')
    for name, total in zip(POSITIONS, [drafted, accepted]):
        need(sum(v for (n, _), v in delta.items() if n == name) == total, 'position counts do not sum to total')
    for (name, labels), value in delta.items():
        if name == POSITIONS[1]:
            need(value <= delta.get((POSITIONS[0], labels), 0), 'accepted position exceeds drafted position')
        if name in POSITIONS and value:
            need(int(dict(labels)['position']) <= maximum, 'draft position exceeds configured K')
    hist = {}
    for (name, labels), value in delta.items():
        if name != K:
            continue
        k = dict(labels).get('k')
        need(k is not None and re.fullmatch(r'0|[1-9][0-9]*', k), 'invalid effective-K label')
        need(int(k) <= maximum or value == 0, 'effective K exceeds configured maximum')
        hist[k] = hist.get(k, 0) + value
    if drafted:
        need(sum(v for k, v in hist.items() if int(k) > 0) > 0, 'drafted tokens lack effective-K evidence')
    verified = delta.get(('chop_spec_verify_items_total', ()), 0)
    emitted = delta.get(('chop_spec_emitted_tokens_total', ()), 0)
    rows = delta.get(('chop_spec_rows_total', ()), 0)
    need(rows == drafted + verified, 'verify row count is inconsistent')
    need(sum(int(k) * n for k, n in hist.items()) == drafted and
         sum(n for k, n in hist.items() if int(k) > 0) == verified, 'effective-K counts disagree with verification')
    need(verified <= emitted <= verified + accepted, 'invalid verified emitted-token count')
    previous_d, previous_a = verified, verified
    for position in range(1, 17):
        labels = (('position', str(position)),)
        d = delta.get((POSITIONS[0], labels), 0); a = delta.get((POSITIONS[1], labels), 0)
        need(d == sum(n for k, n in hist.items() if int(k) >= position), 'draft position disagrees with effective K')
        need(a <= d <= previous_d and a <= previous_a, 'position counts are not prefix-consistent')
        previous_d, previous_a = d, a
    record = dict(drafted=drafted, accepted=accepted, verified=verified, emitted=emitted,
                  mean_accepted_length=1 + accepted / verified if verified else None,
                  emitted_per_verify=emitted / verified if verified else None,
                  rate=accepted / drafted if drafted else None,
                  k_histogram=hist, counters=[dict(name=n, labels=dict(labels), value=v)
                      for (n, labels), v in sorted(delta.items())])
    if require_drafts and fixed_k is not None:
        record['fixed_k'] = fixed_k
        record['fixed_k_verifies'] = fixed_k_exposure(record, fixed_k)
    return record


def timing_summary(arms, phases):
    rows = []
    for variant in sorted({a['variant'] for a in arms}):
        ordered = [a for a in arms if a['variant'] == variant]
        for phase in phases:
            name = phase['name']
            values = []
            for arm in ordered:
                entry = next(p for p in arm['capture']['phases'] if p['phase'] == name)
                values.append(dict(arm=arm['name'], install=arm['install'], ordinal=arm['ordinal'],
                    wall_ms=entry['wall_ms'], ms_per_token=statistics.median(entry['ms_per_token'])
                    if entry.get('ms_per_token') else None,
                    tokens_per_s=entry.get('tokens_per_s', entry.get('prompt_tokens_per_s'))))
            means = {}
            for install in ['base', 'new']:
                means[install] = {key: statistics.mean(v[key] for v in values if v['install'] == install and v[key] is not None)
                    for key in ['wall_ms', 'ms_per_token', 'tokens_per_s']
                    if any(v['install'] == install and v[key] is not None for v in values)}
            orders = []
            for first in range(0, len(values), 2):
                pair = values[first:first + 2]
                by_install = {v['install']: v for v in pair}
                ratios = {key: by_install['new'][key] / by_install['base'][key]
                          for key in ['wall_ms', 'ms_per_token', 'tokens_per_s']
                          if by_install['base'][key] and by_install['new'][key] is not None}
                orders.append(dict(order=[v['install'] for v in pair], arms=[v['arm'] for v in pair],
                                   new_over_base=ratios))
            rows.append(dict(variant=variant, phase=name, ordered=values, means=means, by_order=orders,
                mean_new_over_base={key: statistics.mean(o['new_over_base'][key] for o in orders)
                    for key in orders[0]['new_over_base'] if all(key in o['new_over_base'] for o in orders)}))
    return rows

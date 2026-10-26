"""Build a depth-tuning workload's prompts from a source tree.

Usage: python3 -I prompts.py WORKLOAD SOURCE_TREE OUT.json

The workload (workloads/*.json) names a mix of prompt kinds and the source
files to excerpt. Prompts come out in the mix's proportions for every prefix,
so the window of C concurrent streams, which serves the first C prompts, keeps
the mix. Each prompt is either raw text (completion) or chat messages (edit,
chat), which the job tokenizes with the server's own template. The same
workload and tree always give the same prompts; OUT records each source
file's SHA-256.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FUNCTION = {
    '.rs': re.compile(r'^\s*(pub(\([^)]*\))?\s+)?(const\s+)?(async\s+)?(unsafe\s+)?fn\s+\w+.*\{\s*$'),
    '.py': re.compile(r'^\s*(async\s+)?def\s+\w+.*:\s*$'),
}
LANGUAGE = {'.rs': 'rust', '.py': 'python'}
COMMENT = {'.rs': '//', '.py': '#'}


def exhausted():
    raise RuntimeError('unseen cold prompt budget exhausted')


class VisitPool:
    def __init__(self, tokens, assignment=None):
        self.remaining = tokens
        self.assignment = assignment

    def reserve(self, count):
        if count < 1 or count > len(self.remaining):
            exhausted()
        values, self.remaining = self.remaining[:count], self.remaining[count:]
        return values


def arm_visits(visits):
    arms = {}
    for visit in visits:
        arms.setdefault(visit.get('arm', visit['label']), []).append(visit['label'])
    return arms


def visit_pool_sizes(count, visits):
    """Nearest-length groups lose only the final incomplete arm group."""
    arms = arm_visits(visits)
    per_arm = count // len(arms)
    return {label: (per_arm + len(members) - 1 - index) // len(members)
            for members in arms.values() for index, label in enumerate(members)}


def visit_pools(tokens, visits):
    """Pair nearest lengths across arms, preserving source order per arm."""
    if len({tuple(p) for p in tokens}) != len(tokens):
        raise ValueError('cold prompt pool contains duplicates')
    arms = arm_visits(visits)
    ordered = sorted(range(len(tokens)), key=lambda i: (len(tokens[i]), i))
    by_arm = {arm: [] for arm in arms}
    for start in range(0, len(ordered) - len(arms) + 1, len(arms)):
        group = ordered[start:start + len(arms)]
        span = len(tokens[group[-1]]) - len(tokens[group[0]])
        for arm, index in zip(arms, group):
            by_arm[arm].append((index, span))
    result = {}
    for arm, members in arms.items():
        for index, label in enumerate(members):
            assigned = sorted(by_arm[arm])[index::len(members)]
            indices = [i for i, _ in assigned]
            result[label] = VisitPool([tokens[i] for i in indices], {
                'indices': indices, 'lengths': [len(tokens[i]) for i in indices],
                'neighbor_span_tokens': [span for _, span in assigned]})
    return result


def matched_visit_pools(tokens, visits, lengths, tolerance, per_visit):
    """One disjoint input set per length and ABBA pair, shared across arms."""
    if len({tuple(p) for p in tokens}) != len(tokens):
        raise ValueError('matched cold prompt pool contains duplicates')
    result = {}
    for length in lengths:
        indices = [i for i, prompt in enumerate(tokens) if abs(len(prompt) - length) <= tolerance]
        if len(indices) < 2 * per_visit:
            raise ValueError(f'matched cold length {length} needs {2 * per_visit} prompts within tolerance {tolerance}')
        for visit in visits:
            repeat = visit['passes'][0][0]
            assigned = indices[repeat * per_visit:(repeat + 1) * per_visit]
            result[f"{visit['label']}/l{length}"] = VisitPool([tokens[i] for i in assigned], {
                'indices': assigned, 'lengths': [len(tokens[i]) for i in assigned],
                'target_length': length, 'tolerance': tolerance, 'repeat': repeat})
    return result


class Unseen:
    """A job-wide tokenized prompt pool, exhausted rather than reused."""

    def __init__(self, tokens):
        import threading
        self.tokens, self.next, self.lock = tokens, 0, threading.Lock()
        if len({tuple(p) for p in tokens}) != len(tokens):
            raise ValueError('cold prompt pool contains duplicates')

    def take(self, unused=None):
        with self.lock:
            if self.next == len(self.tokens):
                raise RuntimeError('unseen cold prompt pool exhausted')
            prompt = self.tokens[self.next]
            self.next += 1
            return prompt


def order(mix, count):
    """Kinds in proportion for every prefix: smooth weighted round robin."""
    weights = {entry['kind']: entry['weight'] for entry in mix}
    total, current, out = sum(weights.values()), dict.fromkeys(weights, 0), []
    for _ in range(count):
        for kind in weights:
            current[kind] += weights[kind]
        kind = max(weights, key=lambda k: (current[k], -list(weights).index(k)))
        current[kind] -= total
        out.append(kind)
    return out


def sources(workload, tree):
    spec = workload['sources']
    paths = sorted({p for pattern in spec['globs'] for p in tree.glob(pattern) if p.is_file()})
    chosen = []
    for path in paths:
        size = path.stat().st_size
        if spec['min_bytes'] <= size <= spec['max_bytes'] and path.suffix in FUNCTION:
            text = path.read_text(errors='replace')
            starts = [i for i, line in enumerate(text.splitlines()) if FUNCTION[path.suffix].match(line) and i >= 20]
            if starts:
                chosen.append((path, text, starts))
    key = lambda item: hashlib.sha256(f"{workload['name']}/{item[0].relative_to(tree)}".encode()).digest()
    return sorted(chosen, key=key)


def excerpt(text, start, bounds, before):
    """Lines around function `start`: up to the signature when `before`
    (a completion prefix), or from the signature on (an edit target), within
    the byte bounds."""
    lines = text.splitlines()
    low, high = bounds
    if before:
        out, size, i = [], 0, start
        while i >= 0 and size < high:
            out.insert(0, lines[i])
            size += len(lines[i]) + 1
            i -= 1
        while size > high and len(out) > 1:
            size -= len(out.pop(0)) + 1
        return '\n'.join(out) + '\n'
    out, size, i = [], 0, start
    while i < len(lines) and size < low:
        out.append(lines[i])
        size += len(lines[i]) + 1
        i += 1
    while i < len(lines) and size < high and lines[i].strip():
        out.append(lines[i])
        size += len(lines[i]) + 1
        i += 1
    return '\n'.join(out) + '\n'


def build(workload, tree, diverse):
    """The workload's prompts; `diverse(i)` gives the chat share's ith prompt."""
    tree = Path(tree).resolve()
    files = sources(workload, tree)
    kinds = order(workload['mix'], workload['prompts'])
    needed = sum(kind != 'chat' for kind in kinds)
    if len(files) < needed:
        raise ValueError(f'{len(files)} source files qualify, {needed} needed')
    prompts, used, chats, edits = [], {}, 0, 0
    for index, kind in enumerate(kinds):
        if kind == 'chat':
            body = {'messages': [{'role': 'user', 'content': diverse(chats)}]}
            source = {'diverse': chats}
            chats += 1
        else:
            path, text, starts = files[len(used)]
            relative = str(path.relative_to(tree))
            used[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            start = starts[int.from_bytes(hashlib.sha256(relative.encode()).digest()[:4], 'little') % len(starts)]
            source = {'path': relative, 'line': start + 1}
            if kind == 'completion':
                body = {'prompt': f'{COMMENT[path.suffix]} {relative}\n'
                                  + excerpt(text, start, workload['sources']['excerpt_bytes'], True)}
            else:
                instruction = workload['edit_instructions'][edits % len(workload['edit_instructions'])]
                edits += 1
                code = excerpt(text, start, workload['sources']['excerpt_bytes'], False)
                body = {'messages': [{'role': 'system', 'content': workload['system']},
                                     {'role': 'user', 'content': f'In `{relative}`:\n\n```{LANGUAGE[path.suffix]}\n'
                                                                 f'{code}```\n\n{instruction}'}]}
        record = {'index': index, 'kind': kind, 'source': source, **body}
        record['sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        prompts.append(record)
    return {'schema': 'chopinfer-depth-tune-prompts.v1', 'workload': workload['name'], 'prompts': prompts,
            'sources': used}


def main(workload, tree, out):
    sys.path.insert(0, str(Path(tree) / 'tools/qual'))
    import bench
    value = build(json.loads(Path(workload).read_text()), tree, lambda i: bench.stream_prompt('diverse', i).strip())
    Path(out).write_text(json.dumps(value, indent=1) + '\n')
    counts = {}
    for p in value['prompts']:
        counts[p['kind']] = counts.get(p['kind'], 0) + 1
    print(json.dumps({'prompts': len(value['prompts']), 'kinds': counts, 'sources': len(value['sources'])}))


if __name__ == '__main__':
    main(*sys.argv[1:4])

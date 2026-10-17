#!/usr/bin/env python3
"""Fail when tracked files carry machine identities, personal data, internal
process labels or bulky data.

Local paths, GPU identities and evaluation output belong under $CHOPINFER_DATA,
never in the repository. Each rule lists the only places it tolerates a match.

Patterns that would name one machine (its hostname, local tool directories,
container names) live outside the tree, in an untracked deny-list read when
present: the file $CHOPINFER_LEAK_DENYLIST names, else
~/.config/chopinfer/leak-denylist, and also leak-denylist.local in the
checkout root (gitignored). Each non-empty line not starting with `#` is a
Python regular expression, matched case-insensitively against every line of
every tracked file.

With no arguments it checks the working tree's tracked files. The Git modes
check what a push would publish instead (tools/dev/pre-push.sh runs them):

- `--rev REV` checks REV's tree. With `--base BASE` (repeatable) it checks
  every file version each commit in `REV --not BASE...` adds or changes,
  under each path it appears at, so a version a later commit removes, a
  rename and a copy are all checked. `--history` does the same for every
  commit of REV.
- `--messages REV` checks those commits' messages and author and committer
  lines (every commit of REV without `--base`).
- `--tag TAG` checks an annotated tag's name, tagger and annotation, and
  every tag object it points through, and fails unless it peels to a commit.

Messages and annotations may name noreply addresses, and get no file's
allowances. In the Git modes, a message, annotation or added line that cites
a local commit outside the checked history (a private commit, for example)
also fails.
"""
import argparse
import os
import posixpath
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = 'tests/leak_guard.py'
MAX_BYTES = 10 * 1024 * 1024

# Synthetic paths in unit tests and paths inside the release container.
DATA_ALLOWED = re.compile(r'(^|/)test_[^/]*\.py$|^deploy/docker/|^docs/deploy-[a-z0-9-]+\.md$|^docs/developer-preview\.md$|'
                          r'^crates/runtime-server/src/aliases\.rs$')
RULES = [
    ('internal task label', re.compile(r'\bW[0-9]+\.[0-9]+[a-z]?(?:-[A-Z0-9-]+)?\b', re.I), None),
    ('internal plan label', re.compile(r'\bC[0-9]+-(?:SMALL|MTP)\b|\bplan +[A-UW-Z][0-9]+(?:[.-][0-9]+)*\b', re.I), None),
    ('internal GPU job', re.compile(r'\bw[0-9]+[a-z]*-[a-z]+-(?:parity|probe|capture)[a-z0-9-]*\b', re.I), None),
    # Qualification, review and work-batch labels: a numbered tag such as
    # PERF-B12, REVIEW-A4 or FIX-SC1, and the named families used so far.
    # State the result with its date and configuration instead.
    ('internal review label', re.compile(
        r'(?<![\w-])(?!HMAC-SHA|SUM-BF16\b)[A-Z]{2,}[0-9]*-[A-Z]{1,3}[0-9]+(?:-[0-9]+)?\b|'
        r'\bGATES[0-9]+\b|\bTOPO-[A-Z]+\b|\bCONC-(?:QWEN|MIMO|COMM)\b|\bSPEC-(?:HICONC|VAL)\b|'
        r'\bMIMO-(?:REQUAL|B64-NUMERICS)\b|\bPREFIX-EXACT\b|\bDFLASH-(?:ACCURACY|VERIFY-[A-Z]+)\b|'
        r'\bQWEN-AGENTIC-[A-Z]+\b|\bKVO-(?:LAB|VALIDATE)\b|\bKOPT-[A-Z]+|\bPREFILL-COST\b|'
        r'\bENCODER-FRAMEWORK\b|\bAUDIO-CHECK\b|\bQDCP[0-9]+-[0-9]+|\bJPROOF[0-9]+-[0-9]+|'
        r'\bMM-[0-9]+\b|\bGLM-F\b|\bAR-EXACT\b|\bREFERENCE-RUNNER\.md\b'), None),
    # A review note that cites a private commit or a numbered review round.
    ('internal review citation', re.compile(
        r'\b[Rr]eview of (?:[0-9a-f]{7,40}|[a-z]+-[0-9]+)\b|\bR[0-9]+ finding [0-9]+\b|\breview, finding [0-9]+'), None),
    ('internal process wording', re.compile(
        r"\bGPU locks?\b|operator(?:'s)? decisions?|"
        r"\boperator(?:'s)?[- ](?:adopted|approved|accepted|chose|held|waived|clarified|decided|confirmed)\b|"
        r"\b(?:adopted|approved|accepted|chosen|held|waived|clarified|decided|confirmed) by the operator\b|"
        r"\boperator's (?:approval|rule)\b|\(operator,|\bplan step [0-9]+", re.I), None),
    ('home directory', re.compile(r'/(home|Users)/[A-Za-z0-9_.-]+/'), None),
    ('local /data path', re.compile(r'(?<![A-Za-z0-9_.$}-])/data/'), DATA_ALLOWED),
    ('local /root, /mnt or /srv path', re.compile(r'(?<![A-Za-z0-9_.$}-])/(?:root|mnt|srv)/'), None),
    ('local /tmp path in prose', re.compile(r'(?<![A-Za-z0-9_.$}-])/tmp/(?!nvidia-mps)'),
     re.compile(r'^(?!.*\.md$)')),
    ('GPU UUID', re.compile(r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-'), None),
    ('PCI bus ID in prose', re.compile(r'\b[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]\b'),
     re.compile(r'^(?!.*\.(md|json|txt|log|csv)$)')),
    # The Qwen Community License, quoted verbatim in THIRD_PARTY_NOTICES,
    # names a contact address; it is the only address allowed.
    ('email address', re.compile(r'(?<![A-Za-z0-9._%+-])'
                                 r'(?!model-business@notice\.qwencloud\.com(?![A-Za-z0-9-]|\.[A-Za-z0-9]))'
                                 r'[A-Za-z0-9._%+-]+@(?!example\.[a-z]+\b|[A-Za-z0-9.-]+\.(invalid|test|example|localhost)\b)[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}'), None),
]
# Files whose remaining matches change bytes a model's host, bundle or
# package builds from, or sit on a model's smoke-staleness paths
# (tools/smoke/smoke.py paths), so they are rewritten in their own batch.
# Remove each entry when its file is clean.
PENDING = {
    'internal review label': {
        # CUDA source: editing it moves its lock family's source provenance.
        'crates/kernel-lab/cuda/probes/strip_race.cuh',
        'crates/kernel-core/src/rank_shape.rs',
        'crates/kernels-sm120/cuda/gemm/decoded_balanced.cuh',
        'crates/kernels-sm120/cuda/gemm/decoded_tiles.cuh',
        'crates/kernels-sm120/cuda/gemm/encoder_gemm.cuh',
        'crates/kernels-sm120/cuda/gemm/README.md',
        'crates/kernels-sm120/cuda/gemm/staged_tiles.cuh',
        'crates/kernels-sm120/cuda/gemm/strip_chains_ordered.cuh',
        'crates/kernels-sm120/cuda/gemm/strip_chains_stacked_ordered.cuh',
        'crates/kernels-sm120/cuda/moe/chained_decoded.cuh',
        'crates/kernels-sm120/src/attention/sparse.rs',
        'crates/kernels-sm120/src/backend/moe_experts.rs',
        'crates/kernels-sm120/src/gemm/mod.rs',
        'crates/kernels-sm120/src/gemm/stacked.rs',
        'crates/kernels-sm120/src/linear/mod.rs',
        'crates/kernels-sm120/src/moe/chained.rs',
        'crates/runtime-bundle-sdk/src/batch.rs',
        'crates/runtime-engine/src/speculation.rs',
        'crates/runtime-media/tests/jpeg.rs',
        'crates/runtime-scheduler/src/mixed_budget.rs',
        'models/glm5next/README.md',
        'models/glm5next/bundle/src/fake_host_tests/invariance.rs',
        'models/glm5next/bundle/src/instance/prefill_graphs.rs',
        'models/glm5next/bundle/src/ops/exact.rs',
        'models/glm5next/bundle/src/shared_overlap.rs',
        'models/glm5next/contract/src/sm120/engines.rs',
        'models/glm5next/contract/src/sm120/precision.rs',
        'models/glm5next/profiles.json',
        'models/glm5next/qual.json',
        'models/mimo26/bundle/src/ops.rs',
        'models/mimo26/bundle/src/served_tests.rs',
        'models/mimo26/bundle/src/sm120/dense.rs',
        'models/mimo26/contract/src/serving.rs',
        'models/mimo26/contract/src/sm120/encoders/rows.rs',
        'models/mimo26/contract/tests/sm120/kernel_engine.rs',
        'models/mimo26/profiles.json',
        'models/qwen38/bundle/src/fake_host_tests.rs',
        'models/qwen38/bundle/src/fake_host_tests/served.rs',
        'models/qwen38/bundle/src/fake_host_tests/speculation.rs',
        'models/qwen38/bundle/src/lower.rs',
        'models/qwen38/contract/src/sm120/linear.rs',
        'models/qwen38/contract/tests/sm120/kernel_attention.rs',
        'models/qwen38/contract/tests/sm120/kernel_linear_build.rs',
        'models/qwen38/profiles.json',
        'models/qwen38/reference.py',
        # Asserts GLM's profiles.json qualification text; changes with it.
        'tests/test_serve_profiles.py',
    },
    'internal process wording': {
        'models/glm5next/README.md',
        'models/glm5next/contract/src/sm120/engines.rs',
        'models/glm5next/encoder_parity_run.py',
        'models/glm5next/profiles.json',
        'models/glm5next/qual.json',
        'models/mimo26/qual.json',
        'models/qwen38/bundle/src/context_parallel.rs',
        'models/qwen38/bundle/src/fake_host_tests.rs',
        'models/qwen38/profiles.json',
        'models/qwen38/qual.json',
    },
    # Test modules of a shared kernel crate every model's host links.
    'internal review citation': {
        'crates/kernels-sm120/src/backend/attention.rs',
        'crates/kernels-sm120/src/backend/projection.rs',
        'crates/kernels-sm120/src/backend/recurrent.rs',
        'crates/kernels-sm120/src/backend/rows.rs',
    },
    'local /root, /mnt or /srv path': {
        'crates/runtime-bundle/src/lib.rs',
    },
}
DENYLIST_FILES = [Path(os.environ.get('CHOPINFER_LEAK_DENYLIST') or
                       Path.home() / '.config/chopinfer/leak-denylist'),
                  ROOT / 'leak-denylist.local']
LINK = re.compile(r'\]\(([^)\s]+)\)')
# Addresses a commit message or tag may carry: GitHub's per-user noreply
# form and the one used in co-author trailers.
NOREPLY = re.compile(r'[A-Za-z0-9._%+-]+@users\.noreply\.github\.com\b|\bnoreply@anthropic\.com\b')
# An abbreviated or full commit name, as messages and docs cite them.
CITED = re.compile(r'\b[0-9a-f]{9,40}\b')


def tracked():
    out = subprocess.run(['git', 'ls-files', '-z'], cwd=ROOT, check=True, capture_output=True).stdout
    return [name for name in out.decode().split('\0') if name and (ROOT / name).is_file()]


def local_rules():
    """The untracked deny-list's patterns, as rules with no tolerated place."""
    rules = []
    for path in DENYLIST_FILES:
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.strip() and not line.lstrip().startswith('#'):
                rules.append((f'local deny-list pattern ({path.name}:{number})',
                              re.compile(line.strip(), re.I), None))
    return rules


def check_file(name, size, text, rules, failures, where=None):
    """A tracked file's size, rules and the places each rule tolerates."""
    where = where or name
    if size > MAX_BYTES:
        failures.append(f'{where}: {size} bytes exceeds {MAX_BYTES}; keep data under $CHOPINFER_DATA')
    lines = text.splitlines()
    for label, pattern, allowed in rules:
        if allowed is not None and allowed.search(name) or name in PENDING.get(label, ()):
            continue
        for number, line in enumerate(lines, 1):
            if pattern.search(line):
                failures.append(f'{where}:{number}: {label}: {line.strip()[:160]}')


def check_links(name, text, exists, failures):
    for number, line in enumerate(text.splitlines(), 1):
        for target in LINK.findall(line):
            target = target.split('#', 1)[0].strip('<>')
            if not target or re.match(r'[a-z]+:', target):
                continue
            if not exists(target):
                failures.append(f'{name}:{number}: broken link: {target}')


def check_prose(where, lines, rules, failures):
    """A message's lines: every rule, no file's allowances, noreply addresses allowed."""
    for number, line in enumerate(lines, 1):
        bare = NOREPLY.sub('noreply', line)
        for label, pattern, _ in rules:
            if pattern.search(bare):
                failures.append(f'{where}:{number}: {label}: {line.strip()[:160]}')


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True).stdout.decode('utf-8', 'replace')


class Objects:
    """Reads objects through one `git cat-file --batch`."""

    def __init__(self):
        self.process = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=ROOT,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def find(self, name):
        """(full name, type, bytes) of an object; type is `missing` or
        `ambiguous` (an abbreviation several objects share) otherwise."""
        self.process.stdin.write(name.encode() + b'\n')
        self.process.stdin.flush()
        header = self.process.stdout.readline().decode().split()
        if len(header) != 3:
            return name, header[-1] if header else 'missing', b''
        data = self.process.stdout.read(int(header[2]))
        self.process.stdout.read(1)
        return header[0], header[1], data

    def read(self, name):
        """(type, bytes) of an object."""
        return self.find(name)[1:]

    def close(self):
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.wait()


class Citations:
    """Commit names that resolve locally to a commit outside REV's history."""

    def __init__(self, objects, rev):
        self.objects = objects
        self.history = set(git('rev-list', rev).split())
        self.known = {}

    def commits(self, token):
        """Every local commit the name or abbreviation can mean."""
        name, kind, _ = self.objects.find(token)
        if kind == 'commit':
            return [name]
        if kind != 'ambiguous':
            return []
        names = git('rev-parse', f'--disambiguate={token}').split()
        return [name for name in names if self.objects.read(name)[0] == 'commit']

    def outside(self, line):
        found = []
        for token in CITED.findall(line):
            if token not in self.known:
                self.known[token] = any(name not in self.history for name in self.commits(token))
            if self.known[token]:
                found.append(token)
        return found

    def check(self, where, lines, failures):
        for number, line in enumerate(lines, 1):
            for token in self.outside(line):
                failures.append(f'{where}:{number}: cites commit {token} outside this history: {line.strip()[:160]}')


def commits(rev, bases):
    return git('rev-list', rev, '--not', *bases).split()


def introduced(commit):
    """(path, blob) for each file a commit adds or changes against each of its
    parents, with renames and copies as additions under the new path."""
    fields = git('diff-tree', '-r', '-z', '--root', '-m', '--no-renames', '--no-commit-id', commit).split('\0')
    pairs = []
    for meta, path in zip(fields[::2], fields[1::2]):
        mode, blob, status = meta.split()[1], meta.split()[3], meta.split()[4]
        if status[0] in 'AMT' and mode != '160000':
            pairs.append((path, blob))
    return pairs


def added_lines(commit):
    """{path: added lines} of a commit's text changes against each parent."""
    patch = git('diff-tree', '-p', '-U0', '-m', '--root', '--no-renames', '--no-commit-id',
                '--no-color', '--no-ext-diff', commit)
    added, path, hunk = {}, None, False
    for line in patch.splitlines():
        if line.startswith('diff --git '):
            path, hunk = None, False
        elif line.startswith('@@'):
            hunk = True
        elif not hunk and line.startswith('+++ '):
            path = line[6:] if line.startswith('+++ b/') else None
        elif hunk and line.startswith('+') and path is not None:
            added.setdefault(path, []).append(line[1:])
    return added


def tree(rev):
    """{path: blob} of REV's tree."""
    entries = {}
    for entry in git('ls-tree', '-r', '-z', '--full-tree', rev).split('\0'):
        if entry:
            meta, path = entry.split('\t', 1)
            mode, kind, blob = meta.split()
            if kind == 'blob':
                entries[path] = blob
    return entries


def in_tree(files):
    directories = {posixpath.dirname(name) for name in files}
    for name in list(directories):
        while name:
            name = posixpath.dirname(name)
            directories.add(name)

    def exists(name, target):
        path = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
        return path in files or path in directories or path == '.'
    return exists


def check_rev(rev, bases, history, rules, objects, failures):
    """REV's tree, or each file version the commits after BASE add."""
    files = tree(rev)
    exists = in_tree(files)
    if not bases and not history:
        pairs = {(name, blob): name for name, blob in files.items()}
        markdown = [name for name in files if name.endswith('.md')]
    else:
        pairs, markdown = {}, set()
        citations = Citations(objects, rev)
        for commit in commits(rev, bases):
            for name, blob in introduced(commit):
                pairs.setdefault((name, blob), f'{commit[:12]}:{name}')
                if name.endswith('.md') and name in files:
                    markdown.add(name)
            for name, lines in added_lines(commit).items():
                if name != SELF:
                    citations.check(f'{commit[:12]}:{name} added line', lines, failures)
    by_blob = {}
    for (name, blob), where in pairs.items():
        if name != SELF:
            by_blob.setdefault(blob, []).append((name, where))
    for blob, places in by_blob.items():
        kind, data = objects.read(blob)
        text = data.decode('utf-8', 'replace')
        for name, where in places:
            check_file(name, len(data), text, rules, failures, where)
    for name in sorted(markdown):
        text = objects.read(files[name])[1].decode('utf-8', 'replace')
        check_links(name, text, lambda target, name=name: exists(name, target), failures)


def check_messages(rev, bases, rules, objects, failures):
    citations = Citations(objects, rev)
    for commit in commits(rev, bases):
        text = objects.read(commit)[1].decode('utf-8', 'replace')
        header, _, message = text.partition('\n\n')
        lines = [line for line in header.splitlines() if line.startswith(('author ', 'committer '))]
        lines += message.splitlines()
        check_prose(f'commit {commit[:12]} message', lines, rules, failures)
        citations.check(f'commit {commit[:12]} message', lines, failures)


def check_tag(tag, rules, objects, failures):
    """An annotated tag chain's names, taggers and annotations."""
    name = git('rev-parse', '--verify', tag).strip()
    lines, seen = [], set()
    while True:
        kind, data = objects.read(name)
        if kind == 'commit':
            break
        if kind != 'tag' or name in seen:
            failures.append(f'tag {tag}: does not peel to a commit')
            return
        seen.add(name)
        header, _, message = data.decode('utf-8', 'replace').partition('\n\n')
        lines += [line for line in header.splitlines() if line.startswith(('tag ', 'tagger '))]
        lines += message.splitlines()
        name = next(line.split()[1] for line in header.splitlines() if line.startswith('object '))
    check_prose(f'tag {tag}', lines, rules, failures)
    Citations(objects, name).check(f'tag {tag}', lines, failures)


def arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--rev', help="check this revision's files instead of the working tree")
    parser.add_argument('--messages', metavar='REV', help="check this revision's commit messages")
    parser.add_argument('--tag', action='append', default=[], help='check an annotated tag (repeatable)')
    parser.add_argument('--base', action='append', default=[],
                        help='check only commits not reachable from BASE (repeatable)')
    parser.add_argument('--history', action='store_true', help="with --rev, check every commit of REV")
    args = parser.parse_args(argv)
    if (args.base or args.history) and not (args.rev or args.messages):
        parser.error('--base and --history need --rev or --messages')
    if args.history and args.base:
        parser.error('--history checks every commit; drop --base')
    return args


def main(argv=()):
    args = arguments(list(argv))
    failures = []
    local = local_rules()
    if not local:
        # A pass then covers the public-tree rules only; say so, so a receipt
        # shows the machine checks were skipped.
        print('leak guard: no local deny-list loaded (' +
              ', '.join(str(path) for path in DENYLIST_FILES) +
              '); machine-specific patterns were not checked', file=sys.stderr)
    rules = RULES + local
    if args.rev or args.messages or args.tag:
        objects = Objects()
        try:
            if args.rev:
                check_rev(args.rev, args.base, args.history, rules, objects, failures)
            if args.messages:
                check_messages(args.messages, args.base, rules, objects, failures)
            for tag in args.tag:
                check_tag(tag, rules, objects, failures)
        finally:
            objects.close()
    else:
        for name in tracked():
            if name == SELF:
                continue
            path = ROOT / name
            data = path.read_bytes()
            text = data.decode('utf-8', 'replace')
            check_file(name, path.stat().st_size, text, rules, failures)
            if name.endswith('.md'):
                check_links(name, text, lambda target, path=path: (path.parent / target).exists(), failures)
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        print(f'leak guard: {len(failures)} finding(s)', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

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
"""
import os
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


def main():
    failures = []
    local = local_rules()
    if not local:
        # A pass then covers the public-tree rules only; say so, so a receipt
        # shows the machine checks were skipped.
        print('leak guard: no local deny-list loaded (' +
              ', '.join(str(path) for path in DENYLIST_FILES) +
              '); machine-specific patterns were not checked', file=sys.stderr)
    rules = RULES + local
    for name in tracked():
        if name == SELF:
            continue
        path = ROOT / name
        size = path.stat().st_size
        if size > MAX_BYTES:
            failures.append(f'{name}: {size} bytes exceeds {MAX_BYTES}; keep data under $CHOPINFER_DATA')
        text = path.read_bytes().decode('utf-8', 'replace')
        for label, pattern, allowed in rules:
            if allowed is not None and allowed.search(name) or name in PENDING.get(label, ()):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    failures.append(f'{name}:{number}: {label}: {line.strip()[:160]}')
        if name.endswith('.md'):
            for number, line in enumerate(text.splitlines(), 1):
                for target in LINK.findall(line):
                    target = target.split('#', 1)[0].strip('<>')
                    if not target or re.match(r'[a-z]+:', target):
                        continue
                    if not (path.parent / target).exists():
                        failures.append(f'{name}:{number}: broken link: {target}')
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        print(f'leak guard: {len(failures)} finding(s)', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

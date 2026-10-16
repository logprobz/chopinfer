"""Scan shared sources, with exact, counted migration and integration exceptions."""
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

SHARED = {'bundle-runtime', 'model-compiler-sdk', 'kernel-core', 'kernel-build',
          'encoder-spec', 'frontend-format', 'projection-spec', 'kernels-sm120', 'kernel-harness',
          'kernel-lab', 'xtask'}
ALLOWLIST = 'tests/model-neutral-allowlist.json'


def vocabulary(root):
    names = [p.parent.name for p in (root / 'models').glob('*/model.toml')]
    # A model directory name matches only as a whole identifier word: its
    # neighbors are non-alphanumeric or a camel-case hump (`ProbeM3Tests`),
    # so a short name such as `m3` does not hit `e4m3`.
    start = r'(?:(?<![a-z0-9])|(?-i:(?<=[a-z])(?=[A-Z])))'
    end = r'(?:(?![a-z0-9])|(?-i:(?<=[a-z0-9])(?=[A-Z])))'
    return re.compile('|'.join([*(start + re.escape(n) + end for n in names),
                      r'glm|qwen|deepseek|mimo|prefill32|decode1|physical[_-]?mla|\bmla\b']), re.I)


def findings(root):
    pattern = vocabulary(root)
    paths = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root)
    found = {}
    for name in sorted(set(paths.decode().split('\0')) - {''}):
        parts = Path(name).parts
        shared = parts[0] in {'tools', 'ci', 'deploy'} or (
            len(parts) > 2 and parts[0] == 'crates' and
            (parts[1].startswith('runtime-') or parts[1] in SHARED))
        if not shared or not (root / name).is_file():
            continue
        try:
            source = (root / name).read_text(encoding='utf-8')
        except UnicodeDecodeError:
            if Path(name).suffix not in {'.rs', '.toml', '.md', '.sh', '.json', '.cu', '.cuh', '.py', '.c', '.h', '.inc'}:
                continue
            source = (root / name).read_text(encoding='utf-8', errors='replace')
        source = name + "\n" + source
        hits = Counter(match.group().lower() for match in pattern.finditer(source))
        if hits:
            found[name] = dict(sorted(hits.items()))
    return found


def check_sources(root):
    allowed = json.loads((root / ALLOWLIST).read_text())
    observed = findings(root)
    errors = []
    for name in sorted(observed.keys() | allowed.keys()):
        entry = allowed.get(name, {})
        if name not in observed or entry.get('matches') != observed[name] or not entry.get('reason'):
            errors.append(f'{name}: model vocabulary {observed.get(name, {})}; '
                          f'allowance {entry.get("matches", {})}. '
                          'Remove stale allowances; do not expand migration exceptions.')
    if errors:
        raise SystemExit('\n'.join(errors))

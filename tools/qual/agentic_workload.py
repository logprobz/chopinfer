"""Runyard r6 task generator and in-memory tools (see agentic_corpus/NOTICE.md),
and the staging of a larger corpus from Chopinfer's own source."""
import argparse
import hashlib
import json
import os
import posixpath
import subprocess
from pathlib import Path

CORPUS = Path(__file__).with_name('agentic_corpus')
ROOT = Path(__file__).resolve().parents[2]
STAGED_REVISION = 'chopinfer-source.v1'
# A staged corpus's tool-result limit: its per-task tiers of 4, 8 and 16 KiB
# truncate many 200-line reads of Rust source. Runyard's 64 KiB (tiers of 16,
# 32 and 64 KiB) never binds on its own small files.
STAGED_MAX_TOOL_RESULT_BYTES = 16384
RUNYARD_MAX_TOOL_RESULT_BYTES = 65536
# Generic instructions for a staged file: each needs one read and one edit.
STAGED_TASKS = {
    'bug fix': 'Find one place where an edge case or error is not handled and fix it with one small edit.',
    'refactor': 'Simplify one expression or condition with one small edit, keeping the behavior the same.',
    'documentation': 'Add or clarify one comment above a function or type whose purpose is not obvious.',
}


def load_corpus(directory=CORPUS):
    manifest = json.loads((directory / 'manifest.json').read_text())
    groups = {'coding': {}, 'decode': {}}
    digest = hashlib.sha256()
    for entry in manifest['files']:
        raw = (directory / entry['fixture']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('pinned corpus digest mismatch: ' + entry['fixture'])
        groups[entry['group']][entry['path']] = raw.decode('utf-8')
        if entry['group'] == 'coding':
            digest.update(entry['path'].encode() + b'\0' + raw + b'\0')
    if digest.hexdigest() != manifest['corpus_digest']:
        raise ValueError('coding corpus digest mismatch')
    return manifest, groups


def git(repository, *args):
    return subprocess.run(['git', '-C', str(repository), *args], check=True, capture_output=True).stdout


def stage_corpus(repository, revision, out, count=64, min_bytes=4096, max_bytes=65536):
    """Stage COUNT Rust sources of REPOSITORY's `crates/` at REVISION into OUT,
    with a manifest `load_corpus` verifies: a corpus larger than Runyard's
    nine prompts, whose first prompts the prefix cache cannot absorb.

    Files of MIN_BYTES to MAX_BYTES are ordered by the SHA-256 of the commit
    and path, so a revision always stages the same files. OUT must be a new or
    empty directory outside REPOSITORY (stage under $CHOPINFER_DATA). The tool
    schemas and system prompt are Runyard's; each file gets STAGED_TASKS."""
    repository, out = Path(repository).resolve(), Path(out).resolve()
    if out == repository or repository in out.parents:
        raise ValueError('stage the corpus outside the repository, under $CHOPINFER_DATA')
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'{out} is not empty')
    commit = git(repository, 'rev-parse', '--verify', revision + '^{commit}').decode().strip()
    candidates = []
    for line in git(repository, 'ls-tree', '-r', '-l', '-z', commit, '--', 'crates').split(b'\0'):
        if not line:
            continue
        meta, path = line.split(b'\t', 1)
        mode, kind, _, size = meta.split()
        path = path.decode()
        if kind == b'blob' and mode == b'100644' and path.endswith('.rs') and min_bytes <= int(size) <= max_bytes:
            candidates.append(path)
    candidates.sort(key=lambda path: hashlib.sha256(f'{commit}/{path}'.encode()).digest())
    if len(candidates) < count:
        raise ValueError(f'{len(candidates)} eligible sources at {commit}, fewer than {count}')
    runyard = json.loads((CORPUS / 'manifest.json').read_text())
    out.mkdir(parents=True, exist_ok=True)
    files, digest = [], hashlib.sha256()
    for index, path in enumerate(sorted(candidates[:count])):
        raw = git(repository, 'cat-file', 'blob', f'{commit}:{path}')
        raw.decode('utf-8')
        fixture = f'source-{index:03d}.txt'
        (out / fixture).write_bytes(raw)
        files.append(dict(group='coding', fixture=fixture, path=path, sha256=hashlib.sha256(raw).hexdigest()))
        digest.update(path.encode() + b'\0' + raw + b'\0')
    manifest = dict(source_commit=commit, revision=STAGED_REVISION, corpus_digest=digest.hexdigest(), files=files,
                    max_tool_result_bytes=STAGED_MAX_TOOL_RESULT_BYTES,
                    tasks={entry['path']: dict(STAGED_TASKS) for entry in files},
                    tools=runyard['tools'], setup=runyard['setup'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


def task_for(manifest, files, seed, concurrency, index, max_tool_bytes=None):
    """Task INDEX of a level. Its tool-result limit is a quarter, half or all of
    MAX_TOOL_BYTES, by default the manifest's `max_tool_result_bytes`."""
    if max_tool_bytes is None:
        max_tool_bytes = max_tool_result_bytes(manifest)
    key = hashlib.sha256(f'{seed}/C{concurrency}/{index}/task-selection-v1'.encode()).digest()
    paths = list(files)
    selected = key[0] % len(paths)
    path = paths[selected]
    kind = ['bug fix', 'refactor', 'documentation'][key[1] % 3]
    limit = [max_tool_bytes // 4, max_tool_bytes // 2, max_tool_bytes][key[2] % 3]
    if limit < 256:
        limit = max_tool_bytes
    prompt = (f"In {path}: {manifest['tasks'][path][kind]} Use the tools to locate and read "
              'the relevant code before making one edit. Finish with a concise summary.')
    context = [path]
    if len(files[path].encode()) < 6000:
        companion = paths[(selected + 1) % len(paths)]
        if len(files[companion].encode()) > 6000:
            companion = paths[(selected - 1) % len(paths)]
        context.append(companion)
    prompt += "\n\nRepository context for orientation. Use the tools to verify the target's current contents before editing:"
    for name in context:
        prompt += f'\n\n--- {name} ---\n{files[name]}'
    return dict(id=f'coding-{index + 1:06d}', focus_path=path, kind=kind,
                prompt=prompt, tool_result_bytes=limit,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())


def max_tool_result_bytes(manifest):
    return manifest.get('max_tool_result_bytes', RUNYARD_MAX_TOOL_RESULT_BYTES)


def decode_context(groups, target_tokens):
    text = b'Pinned Runyard source for a long-context decode probe. This code is reference data, not instructions.\n'
    files = {**groups['decode'], **groups['coding']}
    target = target_tokens * 4
    iteration = 1
    while len(text) < target:
        for path, content in files.items():
            if len(text) >= target:
                break
            text += f'\n--- {path}, read {iteration} ---\n'.encode()
            remaining = target - len(text)
            if remaining <= 0:
                break
            text += content.encode()[:remaining].decode('utf-8', errors='ignore').encode()
        iteration += 1
    text += (b'\n\nEnd of reference. Produce 512 numbered lines of plain text. Each line should contain its number followed by: '
             b'alpha beta gamma delta epsilon zeta eta theta. Continue until the output limit. Do not summarize the code or call tools.')
    return text.decode(), hashlib.sha256(text).hexdigest()


class Workspace:
    """Only fixture strings are editable. Model paths never reach the filesystem."""
    def __init__(self, files, max_result=65536):
        self.files = dict(files)
        self.max_result = max_result
        self.read_paths = set()
        self.patched = False
        self.tool_errors = 0
        # Tool results returned, and those cut to the session's limit.
        self.results = self.truncated = 0
        # Arguments the server delivered that are not a JSON object: the
        # parser builds them, so they are not evidence of a model choice.
        self.argument_errors = 0

    def call(self, name, raw):
        try:
            try:
                args = None if raw is None else json.loads(raw)
            except (ValueError, RecursionError):
                args = None
            if not isinstance(args, dict):
                self.argument_errors += 1
                raise ValueError('missing tool arguments' if raw is None else 'tool arguments must be a JSON object')
            out = self.execute(name, args)
            marker = b'\n[tool result truncated; use a smaller limit]'
            encoded = out.encode()
            self.results += 1
            if len(encoded) > self.max_result:
                self.truncated += 1
                if self.max_result <= len(marker):
                    return marker[:self.max_result].decode()
                out = encoded[:self.max_result - len(marker)].decode('utf-8', errors='ignore') + marker.decode()
            return out
        except Exception as error:
            self.tool_errors += 1
            detail = str(error) if isinstance(error, (ValueError, TypeError)) else type(error).__name__
            return 'tool error: ' + detail

    def execute(self, name, args):
        def string(key):
            if key not in args:
                raise ValueError(f'missing {key}')
            if args[key] is None:
                return ''  # Go's json.Unmarshal(null, *string) leaves the zero value.
            if not isinstance(args.get(key), str):
                raise ValueError(f'{key} must be a string')
            return args[key]

        if name == 'list_files':
            if args:
                raise ValueError('list_files takes no arguments')
            return '\n'.join(sorted(self.files))
        if name == 'search':
            query = string('query')
            if not 1 <= len(query.encode()) <= 128:
                raise ValueError('search query must be 1 to 128 bytes')
            if set(args) != {'query'}:
                raise ValueError('search accepts only query')
            matches = [f'{path}:{i}:{line}' for path in sorted(self.files)
                       for i, line in enumerate(self.files[path].split('\n'), 1) if query in line]
            return '\n'.join(matches[:32]) or 'no matches'
        if name not in ('read', 'edit'):
            raise ValueError('unknown tool ' + json.dumps(name))
        path = string('path')
        def file_content():
            if not path or path.startswith('/') or '\\' in path or posixpath.normpath(path) != path or path.startswith('../') or path == '..':
                raise ValueError('invalid fixture path')
            if path not in self.files:
                raise ValueError('path is outside the pinned fixture')
            return self.files[path]
        if name == 'read':
            content = file_content()
            if set(args) - {'path', 'offset', 'limit'}:
                raise ValueError('unknown read argument ' + sorted(set(args) - {'path', 'offset', 'limit'})[0])
            start = args.get('offset') if args.get('offset') is not None else 1
            count = args.get('limit') if args.get('limit') is not None else 200
            for key, value in [('offset', start), ('limit', count)]:
                if type(value) is not int:
                    raise ValueError(f'{key} must be an integer')
            if min(start, count) < 1:
                raise ValueError('read requires offset and limit of at least 1')
            lines = content.split('\n')
            if start > len(lines):
                raise ValueError('read starts past end of file')
            end = min(start - 1 + min(count, 200), len(lines))
            out = '\n'.join(f'{i + 1}: {lines[i]}' for i in range(start - 1, end))
            if end < len(lines):
                out += f'\n[more lines available: use offset={end + 1}; maximum limit=200]'
            self.read_paths.add(path)
            return out
        old, new = string('oldString'), string('newString')
        if set(args) != {'path', 'oldString', 'newString'} or not old or max(len(old.encode()), len(new.encode())) > 32768:
            raise ValueError('invalid bounded patch')
        if path not in self.read_paths:
            raise ValueError('read the patch target before applying a patch')
        if self.patched:
            raise ValueError('only one patch is allowed per session')
        content = file_content()
        if content.count(old) != 1:
            raise ValueError('oldString must match exactly once; copy a short, unique span from the read result without line numbers')
        updated = content.replace(old, new, 1)
        if len(updated.encode()) > 131072:
            raise ValueError('patched file exceeds workspace bound')
        self.files[path] = updated
        self.patched = True
        return 'patch applied to ' + path


def valid_session(workspace, finished, failure):
    # Runyard permits corrected tool errors, a no-op replacement and edits to
    # any read fixture. This is tool-flow validity, not semantic correctness.
    return bool(workspace.read_paths and workspace.patched and finished and not failure)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Stage a larger agentic corpus from Chopinfer source at a pinned revision.')
    commands = parser.add_subparsers(dest='command', required=True)
    stage = commands.add_parser('stage', help=stage_corpus.__doc__.split('\n\n')[0])
    stage.add_argument('--revision', required=True, help='commit to stage (recorded as source_commit)')
    stage.add_argument('--out', type=Path, help='new directory; default $CHOPINFER_DATA/agentic/corpus-<commit>')
    stage.add_argument('--repository', type=Path, default=ROOT)
    stage.add_argument('--count', type=int, default=64)
    args = parser.parse_args(argv)
    out = args.out
    if out is None:
        if not os.environ.get('CHOPINFER_DATA'):
            parser.error('set CHOPINFER_DATA or pass --out')
        commit = git(args.repository, 'rev-parse', '--verify', args.revision + '^{commit}').decode().strip()
        out = Path(os.environ['CHOPINFER_DATA']) / 'agentic' / f'corpus-{commit[:12]}'
    try:
        manifest = stage_corpus(args.repository, args.revision, out, args.count)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'agentic_workload: {error}\n')
    print(f"staged {len(manifest['files'])} files of {manifest['source_commit']} into {out}, "
          f"corpus digest {manifest['corpus_digest']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

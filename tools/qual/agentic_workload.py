"""Runyard r6 task generator and in-memory tools. See agentic_corpus/NOTICE.md."""
import hashlib
import json
import posixpath
from pathlib import Path

CORPUS = Path(__file__).with_name('agentic_corpus')


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


def task_for(manifest, files, seed, concurrency, index, max_tool_bytes=65536):
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
            if len(encoded) > self.max_result:
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

"""Helpers shared by the journal proof modules: input refusals, git reads at
resolved commits, Rust delimiter scanning and the pin fingerprint."""
import os
import subprocess

# Every read sees the repository's own objects: a replace ref must not let
# one commit's sources stand in for another's.
GIT = ['git', '--no-replace-objects']
GIT_ENV = {**os.environ, 'GIT_NO_REPLACE_OBJECTS': '1'}


class Invalid(ValueError):
    pass


def need(condition, reason):
    if not condition:
        raise Invalid(reason)


def git(repo, *args):
    result = subprocess.run([*GIT, '-C', str(repo), *args], capture_output=True, timeout=60, env=GIT_ENV)
    need(result.returncode == 0, f'git {args[0]} failed: {result.stderr.decode(errors="replace")[:300]}')
    return result.stdout


def resolve(repo, revision):
    return git(repo, 'rev-parse', '--verify', '--end-of-options', revision + '^{commit}').decode().strip()


def balanced(text, start):
    pairs = {'(': ')', '[': ']', '{': '}'}
    need(text[start] in pairs, 'expected opening delimiter')
    stack, quoted, escaped = [], False, False
    for i in range(start, len(text)):
        c = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == '"':
                quoted = False
            continue
        if c == '"':
            quoted = True
        elif c in pairs:
            stack.append(pairs[c])
        elif c in ')]}':
            need(stack and stack.pop() == c, 'unbalanced delimiters')
            if not stack:
                return text[start:i + 1], i + 1
    raise Invalid('unterminated delimited expression')


def arguments(text):
    out, start, depth, quoted, escaped = [], 0, 0, False, False
    for i, c in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
            need(depth >= 0, 'unbalanced argument')
        elif c == ',' and depth == 0:
            out.append(text[start:i].strip())
            start = i + 1
    need(depth == 0 and not quoted, 'unterminated argument')
    if text[start:].strip():
        out.append(text[start:].strip())
    return out


def uncomment(text):
    # Preserve string contents and line numbers; nested Rust block comments
    # are handled explicitly. A commented-out pin is not a manifest entry.
    out, i, block, quoted, escaped = [], 0, 0, False, False
    while i < len(text):
        pair, c = text[i:i + 2], text[i]
        if block:
            if pair == '/*':
                block += 1
                out.extend('  ')
                i += 2
                continue
            if pair == '*/':
                block -= 1
                out.extend('  ')
                i += 2
                continue
            out.append('\n' if c == '\n' else ' ')
        elif quoted:
            out.append(c)
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == '"':
                quoted = False
        elif pair == '//':
            end = text.find('\n', i)
            if end < 0:
                end = len(text)
            out.extend(' ' * (end - i))
            i = end
            continue
        elif pair == '/*':
            block = 1
            out.extend('  ')
            i += 2
            continue
        else:
            out.append(c)
            if c == '"':
                quoted = True
        i += 1
    need(block == 0, 'unterminated source comment')
    return ''.join(out)


def integer(value):
    return int(value.replace('_', ''), 0 if value.startswith('0x') else 10)


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, f'duplicate JSON key {key}')
        result[key] = value
    return result


def body_lines(data):
    """A journal body's lines: none for an empty body, which is the journal of
    zero lines (pinned (0, 0xcbf29ce484222325), the FNV-1a offset basis)."""
    return data.decode('utf-8').split('\n') if data else []


def fnv(data):
    result = 0xcbf29ce484222325
    for byte in data:
        result = ((result ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    return f'{result:016x}'

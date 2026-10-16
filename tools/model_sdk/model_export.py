"""Lay out an in-tree model directory as a standalone model workspace.

`export --model models/<m>` (source.py) extracts the pinned SDK first, then
builds the workspace from the SDK's own copy of the model, so the model and
the SDK come from the same revision:

- models/<m> becomes the workspace root and fixtures/<m> its fixtures/.
- Cargo.toml is generated from the SDK's: the model's roles are the members,
  the shared dependencies point into sdk/, and the package defaults, lints
  and profiles are the SDK's, so the roles build as they do in-tree.
- References that the layout moves are relocated in the model's text files:
  a checkout-relative models/<m>/X becomes X (models/<m> itself becomes
  `.`), fixtures/<m>/X becomes fixtures/X, and a shared crate, another
  model or a shared fixture resolves in sdk/; a path with a `..` component
  is resolved in the checkout first. A `../` path that climbs out of the
  model directory in a Markdown file is relocated the same way, read from
  the file.
- Rust files are rewritten in exactly these forms (`RustPaths`), each
  literal read with its escapes decoded and its path normalized, and raw
  identifiers such as `r#include_str` read as their plain names: the
  argument of an include macro (any delimiter and trivia) and a `#[path]`
  value (also in `cfg_attr`), read from the file; a `concat!` of string
  literals, joined into one path and, when it moves, replaced by one
  literal; the literals `concat!` joins to `env!("CARGO_MANIFEST_DIR")`,
  read from the crate (Cargo's manifest directory, also the working
  directory of build scripts and tests); a `.join` literal on a receiver
  whose directory the text establishes (`Bases`), read from it; and
  checkout-rooted literals read from the checkout root where the text
  establishes that root (see docs/external-models.md). A
  climbing or rooted `.join` or `.starts_with` literal on any other
  receiver fails the export, except a build script's or test's climb that
  stays inside the model, which is kept. Any other string literal that
  climbs to a checkout file outside the model directory, or names one from
  the checkout root, fails the export with its file and line, as does a
  `concat!` mixing such a literal with another macro and a `#[path]` inside
  an inline module. A checkout-rooted literal naming in-tree tooling outside
  `SURFACE` is kept and reported. Paths in comments are relocated as prose.
- Fixture data (any path with a `fixtures` directory) and kernel locks are
  copied byte for byte. Other forms, such as Python that derives the
  checkout root from its own path, are not rewritten.

model-export.json records each relocated file's digest before and after.
Each file is rewritten in one pass over its original text, so a reference
is replaced once and an emitted path is never matched again. Before the
receipt is written, the export checks every reference it matched, rewritten
or kept, against the text emitted at its place in the final file. Read from
the exported base, that text must name what the original named from the
checkout base: the same bytes, or the recorded relocation of those bytes
(for a directory, each file in it); nothing, where the original named
nothing; or the workspace root, for the checkout root. A kept reference
that names nothing in the checkout makes no claim: model-export.json lists
it under `unresolved` and the export prints it as a warning with its file
and line. Anything else fails the export.

Only static path literals are rewritten. A path a program builds at run
time, such as a variable joined with a `../` path in a build script, cannot
be resolved from the text; the exported workspace's own check, build and
tests are the final verification.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import sys
import tomllib

from .graph import PolicyError

SCHEMA = 'chopinfer-model-export-v1'
RECEIPT = 'model-export.json'
ROLES = ('bundle', 'compiler', 'contract')
# Files the export writes at the workspace root; the model may not hold them.
GENERATED = ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'sdk', 'sdk-pin.json', 'target',
             '.gitignore', RECEIPT)
# Cargo.lock and sdk-pin.json are pinned inputs the model repository commits.
IGNORED = '/sdk/\n/target/\n/.chopinfer-sdk-lock\n'
SEGMENT = r'[A-Za-z0-9_.-]'
# A checkout-relative reference starts a path: no path character before it.
ROOTED = re.compile(rf'(?<![A-Za-z0-9_./-])((?:models|fixtures|crates)/{SEGMENT}+(?:/{SEGMENT}+)*)(/?)(?!{SEGMENT})')
# A `../` path, at least one climb then its segments.
CLIMB = re.compile(rf'(?<![A-Za-z0-9_./-])(?:\.\./)+(?:{SEGMENT}+(?:/{SEGMENT}+)*/?)?(?!{SEGMENT})')
# Rust tokens for `rust_paths`; comments and whitespace are skipped.
RUST_TOKEN = re.compile(r"""
    (?P<space>\s+|//[^\n]*)
  | (?P<raw>[bc]?r(?P<hashes>\#*)")
  | (?P<string>[bc]?")
  | (?P<char>b?'(?:\\(?:u\{[0-9A-Fa-f]{1,6}\}|x[0-9A-Fa-f]{2}|.)|[^\\'\n])')
  | (?P<ident>(?:r\#)?[^\W\d]\w*)
  | (?P<lifetime>'[^\W\d]\w*)
  | (?P<other>\d\w*|.)
""", re.VERBOSE | re.DOTALL)
INCLUDES = ('include', 'include_str', 'include_bytes')
# The checkout directories a model builds against; a rooted Rust literal
# naming anything else names in-tree tooling.
SURFACE = ('crates', 'fixtures', 'include', 'models')
TOOLING = 'names checkout tooling the export does not include'


def rust_tokens(text):
    """The tokens of Rust TEXT as (kind, value, start, end). A string
    literal's value is the (start, end) span of its contents; an identifier's
    is its name, a raw identifier's without `r#`."""
    position = 0
    while position < len(text):
        if text.startswith('/*', position):
            # Block comments nest.
            depth, position = 1, position + 2
            while depth and position < len(text):
                if text.startswith('/*', position):
                    depth, position = depth + 1, position + 2
                elif text.startswith('*/', position):
                    depth, position = depth - 1, position + 2
                else:
                    position += 1
            continue
        match = RUST_TOKEN.match(text, position)
        kind = match.lastgroup
        start, position = position, match.end()
        if kind == 'raw':
            end = text.find('"' + match.group('hashes'), position)
            end = len(text) if end < 0 else end
            closed = min(end + 1 + len(match.group('hashes')), len(text))
            yield 'string', (position, end), start, closed
            position = closed
        elif kind == 'string':
            end = position
            while end < len(text) and text[end] != '"':
                end += 2 if text[end] == '\\' else 1
            end = min(end, len(text))
            yield 'string', (position, end), start, min(end + 1, len(text))
            position = end + 1
        elif kind == 'ident':
            yield kind, match.group().removeprefix('r#'), start, position
        elif kind != 'space':
            yield kind, match.group(), start, position


# The single-character escapes of a Rust string literal.
ESCAPES = {'n': '\n', 'r': '\r', 't': '\t', '\\': '\\', '0': '\0', "'": "'", '"': '"'}


def literal_text(text, token_start, start, end):
    """The text of the Rust string literal whose token starts at TOKEN_START
    and whose contents span START to END: a raw literal as written, any other
    with its escapes decoded."""
    contents = text[start:end]
    if text[token_start:start].lstrip('bc').startswith('r'):
        return contents
    decoded, position = [], 0
    while position < len(contents):
        character = contents[position]
        escape = contents[position + 1:position + 2]
        if character != '\\':
            decoded.append(character)
            position += 1
        elif escape == 'x':
            decoded.append(chr(int(contents[position + 2:position + 4], 16)))
            position += 4
        elif escape == 'u':
            close = contents.index('}', position)
            decoded.append(chr(int(contents[position + 3:close].replace('_', ''), 16)))
            position = close + 1
        elif escape in '\r\n':
            # A line continuation skips the line break and the next line's
            # leading whitespace.
            position += 2
            while position < len(contents) and contents[position] in ' \t\r\n':
                position += 1
        else:
            decoded.append(ESCAPES.get(escape, '\\' + escape))
            position += 2
    return ''.join(decoded)


class Bases:
    """The directories that Rust expressions of one file name, where the text
    establishes them: the crate directory (`Path::new` or `PathBuf::from` of
    `env!("CARGO_MANIFEST_DIR")`, or of `env::var_os` or `env::var` of it,
    unwrapped), a `.join` of a literal to a known directory, a `&`, clone or
    conversion of one, an identifier `let`-bound to one in the same function,
    and a `&Path` or `PathBuf` parameter that every call of its function in
    the file passes the same known directory. Directories are checkout
    paths; anything else is unknown (None)."""

    PASSING = ('clone', 'to_path_buf', 'as_path', 'unwrap', 'expect', 'to_owned')

    def __init__(self, tokens, text, crate):
        self.tokens, self.text, self.crate = tokens, text, crate
        self.match, stack = {}, []
        for k, (kind, value, _, _) in enumerate(tokens):
            if kind == 'other' and value in '([{':
                stack.append(k)
            elif kind == 'other' and value in ')]}' and stack:
                self.match[k] = stack.pop()
                self.match[self.match[k]] = k
        # (name, parameters [(name, index, path typed)], body start, body end)
        self.functions, self.bindings = [], []
        for k in range(len(tokens)):
            if self.ident(k, 'fn') and self.ident(k + 1):
                self.function(k)
        self.values = {}
        # Bindings and parameters depend on each other; settle them.
        for _ in range(16):
            before = dict(self.values)
            for function, at, name, expression in self.bindings:
                self.values[('let', at)] = self.value(expression, at)
            for index, (name, parameters, start, end) in enumerate(self.functions):
                for parameter, position, typed in parameters:
                    if typed:
                        self.values[('parameter', index, parameter)] = self.passed(name, position)
            if self.values == before:
                break

    def token(self, k):
        return self.tokens[k] if 0 <= k < len(self.tokens) else (None, None, None, None)

    def punct(self, k, chars):
        kind, value, _, _ = self.token(k)
        return kind == 'other' and value in chars

    def ident(self, k, *names):
        kind, value, _, _ = self.token(k)
        return kind == 'ident' and (not names or value in names)

    def string(self, k):
        kind, value, start, _ = self.token(k)
        return literal_text(self.text, start, *value) if kind == 'string' else None

    def groups(self, open_index):
        """The top-level comma-separated token ranges of the group opening
        at OPEN_INDEX."""
        close, start, groups, k = self.match.get(open_index, open_index), open_index + 1, [], open_index + 1
        while k < close:
            if k in self.match and self.match[k] > k:
                k = self.match[k]
            elif self.punct(k, ','):
                groups.append((start, k))
                start = k + 1
            k += 1
        if start < close:
            groups.append((start, close))
        return groups

    def function(self, k):
        name, k = self.token(k + 1)[1], k + 2
        while k < len(self.tokens) and not self.punct(k, '('):
            k += 1
        if k not in self.match:
            return
        parameters, groups = [], self.groups(k)
        # A method's receiver is not among its call's arguments.
        method = bool(groups) and any(self.ident(index, 'self') for index in range(*groups[0]))
        for position, (first, last) in enumerate(groups, start=-1 if method else 0):
            names = [index for index in range(first, last) if self.ident(index)]
            colon = next((index for index in range(first, last) if self.punct(index, ':')), None)
            if colon is None:
                continue
            types = [self.token(index)[1] for index in range(colon + 1, last) if self.ident(index)]
            parameter = next((self.token(index)[1] for index in names if index < colon
                              and self.token(index)[1] != 'mut'), None)
            parameters.append((parameter, position, bool(types) and types[-1] in ('Path', 'PathBuf')))
        k = self.match[k] + 1
        while k < len(self.tokens) and not self.punct(k, '{;'):
            k += 1
        if not self.punct(k, '{') or k not in self.match:
            return
        index, end = len(self.functions), self.match[k]
        self.functions.append((name, parameters, k, end))
        for at in range(k + 1, end):
            if self.ident(at, 'let'):
                first = at + 1 + self.ident(at + 1, 'mut')
                if not self.ident(first):
                    continue
                equals = first + 1
                while equals < end and not self.punct(equals, '=;'):
                    equals = self.match.get(equals, equals) + 1 if equals in self.match else equals + 1
                if not self.punct(equals, '='):
                    continue
                last = equals + 1
                while last < end and not self.punct(last, ';'):
                    last = self.match[last] + 1 if last in self.match and self.match[last] > last else last + 1
                self.bindings.append((index, at, self.token(first)[1], (equals + 1, last)))

    def enclosing(self, k):
        inside = [index for index, (_, _, start, end) in enumerate(self.functions) if start < k < end]
        return max(inside, key=lambda index: self.functions[index][2]) if inside else None

    def lookup(self, name, k):
        function = self.enclosing(k)
        if function is None:
            return None
        # A binding is in scope after its own expression.
        bound = [at for index, at, bound_name, (_, end) in self.bindings
                 if index == function and bound_name == name and end <= k]
        if bound:
            return self.values.get(('let', max(bound)))
        return self.values.get(('parameter', function, name))

    def passed(self, name, position):
        """The directory every call of function NAME in the file passes at
        POSITION, or None."""
        found = set()
        for k in range(len(self.tokens)):
            # A method call is not a call of the function; `..f(x)` is.
            method = self.punct(k - 1, '.') and not self.punct(k - 2, '.')
            if self.ident(k, name) and self.punct(k + 1, '(') and not self.ident(k - 1, 'fn') and not method:
                groups = self.groups(k + 1)
                found.add(self.value(groups[position], k) if position < len(groups) else None)
        return found.pop() if len(found) == 1 else None

    def receiver(self, dot):
        """The token range of the expression a method call at DOT is on."""
        k = dot - 1
        while True:
            if k in self.match and self.match[k] < k:
                k = self.match[k] - 1
                if self.punct(k, '!'):
                    k -= 1
            if not self.ident(k):
                return (k + 1, dot)
            k -= 1
            if self.punct(k, '.'):
                k -= 1
            elif self.punct(k, ':') and self.punct(k - 1, ':'):
                k -= 2
            else:
                return (k + 1, dot)

    def joined_to_root(self, receiver, at):
        """Whether the identifier RECEIVER (a token range) is joined, in the
        function holding token AT, to a receiver that names the checkout
        root: `root.join(x)` or `root.join(&x)`."""
        first, last = receiver
        function = self.enclosing(at)
        if last - first != 1 or not self.ident(first) or function is None:
            return False
        name, (_, _, start, end) = self.token(first)[1], self.functions[function]
        for k in range(start, end):
            if self.ident(k, 'join') and self.punct(k - 1, '.') and self.punct(k + 1, '('):
                argument = k + 2 + self.punct(k + 2, '&')
                if self.ident(argument, name) and self.punct(argument + 1, ')') \
                        and self.value(self.receiver(k - 1)) == '.':
                    return True
        return False

    def value(self, expression, at=None):
        """The checkout directory the token range EXPRESSION names, or None."""
        first, last = expression
        while first < last and (self.punct(first, '&') or self.ident(first, 'mut')):
            first += 1
        if last - first == 1 and self.ident(first):
            return self.lookup(self.token(first)[1], first)
        if last - first < 3 or not self.punct(last - 1, ')') or (last - 1) not in self.match:
            return None
        opening = self.match[last - 1]
        arguments = self.groups(opening)
        literal = self.string(arguments[0][0]) if len(arguments) == 1 and \
            arguments[0][1] - arguments[0][0] == 1 else None
        method = opening - 1
        if self.ident(method) and self.punct(method - 1, '.'):
            name, receiver = self.token(method)[1], (first, method - 1)
            if name in self.PASSING:
                return self.value(receiver)
            if name == 'join' and literal is not None:
                base = self.value(receiver)
                return None if base is None else posixpath.normpath(posixpath.join(base, literal))
            return None
        if self.ident(method, 'new', 'from') and self.punct(method - 1, ':') \
                and self.ident(method - 3, 'Path', 'PathBuf') and len(arguments) == 1:
            return self.value(arguments[0])
        if (self.ident(method, 'var_os', 'var') and self.punct(method - 1, ':') and self.ident(method - 3, 'env')
                or self.punct(method, '!') and self.ident(method - 1, 'env')) \
                and literal == 'CARGO_MANIFEST_DIR':
            return self.crate
        return None


class RustPaths:
    """The path forms of one Rust file the export recognizes.

    - `forms`: an include macro's argument and a `#[path]` value, read from
      the file (`file`), and the literals `concat!` joins to
      `env!("CARGO_MANIFEST_DIR")`, read from the crate (`manifest`). Each is
      (kind, pieces, span): the content spans of its string literals, which
      join into one path, and the span to replace with one literal when
      there are several (the whole `concat!` of an include, or the literals
      after the manifest directory).
    - `joins`: each `.join(LITERAL)` and `.starts_with(LITERAL)` call as
      (method, content span, receiver token range, method token index); its
      base is `bases.value(receiver)`.
    - `literals`: every string literal's content span and whether it is a
      piece of a `concat!`; `tokens` and `text` map a content span to the
      literal's token span and its text, escapes decoded.
    - `inline`: the offsets of `#[path]` attributes inside an inline module,
      which read from the module's directory."""

    def __init__(self, text, crate):
        tokens = list(rust_tokens(text))
        self.forms, self.joins, self.inline, self.literals = [], [], [], []
        self.bases = Bases(tokens, text, crate)
        pieces = set()
        def token(k):
            return tokens[k] if 0 <= k < len(tokens) else (None, None, None, None)
        def punct(k, chars):
            kind, value, _, _ = token(k)
            return kind == 'other' and value in chars
        def ident(k, *names):
            kind, value, _, _ = token(k)
            return kind == 'ident' and (not names or value in names)
        def string(k):
            return token(k)[0] == 'string'
        def arguments(k):
            """The arguments of the delimited group opening at K, as token
            index ranges split at top-level commas, and the index after it."""
            depth, start, groups, k = 1, k + 1, [], k + 1
            while depth and k < len(tokens):
                if punct(k, '([{'):
                    depth += 1
                elif punct(k, ')]}'):
                    depth -= 1
                    if not depth:
                        break
                elif depth == 1 and punct(k, ','):
                    groups.append((start, k))
                    start = k + 1
                k += 1
            if start < k:
                groups.append((start, k))
            return groups, k + 1
        def concat(k):
            """The `concat!` at K: (kind, literal indices, end index), kind
            `literals` when every argument is a string literal, `manifest`
            when the first is env!("CARGO_MANIFEST_DIR") and the rest are,
            else `mixed`."""
            groups, end = arguments(k + 2)
            single = [first for first, last in groups if last - first == 1 and string(first)]
            if len(single) == len(groups) and groups:
                return 'literals', single, end
            first, last = groups[0] if groups else (0, 0)
            manifest = (last - first == 5 and ident(first, 'env') and punct(first + 1, '!')
                        and punct(first + 2, '([{') and string(first + 3)
                        and text[slice(*token(first + 3)[1])] == 'CARGO_MANIFEST_DIR')
            if manifest and len(groups) > 1 and len(single) == len(groups) - 1:
                return 'manifest', single, end
            return 'mixed', [index for first, last in groups for index in range(first, last) if string(index)], end
        def span(indices):
            return [token(index)[1] for index in indices]
        modules = []
        def attribute(k, start):
            if ident(k, 'path') and punct(k + 1, '=') and string(k + 2):
                if any(modules):
                    self.inline.append(start)
                else:
                    self.forms.append(('file', span([k + 2]), None))
            elif ident(k, 'cfg_attr') and punct(k + 1, '('):
                # cfg_attr(predicate, attribute, ...): each attribute after
                # the predicate.
                for first, _ in arguments(k + 1)[0][1:]:
                    attribute(first, start)
        for k, (kind, value, start, _) in enumerate(tokens):
            if punct(k, '{'):
                modules.append(ident(k - 2, 'mod') and ident(k - 1))
            elif punct(k, '}'):
                if modules:
                    modules.pop()
            elif ident(k, *INCLUDES) and punct(k + 1, '!') and punct(k + 2, '([{'):
                argument = k + 3
                if string(argument):
                    self.forms.append(('file', span([argument]), None))
                elif ident(argument, 'concat') and punct(argument + 1, '!') and punct(argument + 2, '([{'):
                    form, literals, end = concat(argument)
                    if form == 'literals':
                        self.forms.append(('file', span(literals), (token(argument)[2], token(end - 1)[3])))
                        pieces.update(literals)
            elif ident(k, 'concat') and punct(k + 1, '!') and punct(k + 2, '([{'):
                form, literals, _ = concat(k)
                if form == 'manifest':
                    self.forms.append(('manifest', span(literals),
                                       (token(literals[0])[2], token(literals[-1])[3])))
                pieces.update(literals)
            elif ident(k, 'join', 'starts_with') and punct(k - 1, '.') and punct(k + 1, '(') \
                    and string(k + 2) and punct(k + 3, ')'):
                self.joins.append((value, token(k + 2)[1], self.bases.receiver(k - 1), k))
            elif punct(k, '#') and punct(k + 1, '['):
                attribute(k + 2, start)
        self.literals = [(token(k)[1], k in pieces) for k in range(len(tokens)) if string(k)]
        # Each literal's token span and text, by its content span.
        self.tokens = {token(k)[1]: token(k)[2:] for k in range(len(tokens)) if string(k)}
        self.text = {span: literal_text(text, start, *span) for span, (start, _) in self.tokens.items()}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def toml_value(value, *, inline=False):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, str)):
        return json.dumps(value)
    if isinstance(value, list):
        return '[' + ', '.join(toml_value(item) for item in value) + ']'
    if isinstance(value, dict) and inline:
        return '{ ' + ', '.join(f'{key} = {toml_value(item, inline=True)}' for key, item in value.items()) + ' }'
    raise PolicyError(f'unsupported SDK manifest value {value!r}')


def toml_tables(name, table, inline_children=False):
    """TOML text of TABLE under header NAME: its values, then its subtables.
    The entries of an INLINE_CHILDREN table are inline tables."""
    values = [f'{key} = {toml_value(item, inline=inline_children)}' for key, item in table.items()
              if inline_children or not isinstance(item, dict)]
    text = f'[{name}]\n' + ''.join(line + '\n' for line in values) if values else ''
    for key, item in table.items():
        if isinstance(item, dict) and not inline_children:
            text += ('\n' if text else '') + toml_tables(f'{name}.{key}', item, key == 'dependencies')
    return text


def manifest(sdk, model, roles):
    """The workspace Cargo.toml for model MODEL with ROLES, from the SDK's."""
    data = tomllib.loads((sdk / 'Cargo.toml').read_text())
    if not isinstance(data.get('workspace'), dict):
        raise PolicyError('the SDK Cargo.toml declares no [workspace]')
    unknown = (set(data) - {'workspace', 'profile'}) | (set(data['workspace']) - {
        'members', 'resolver', 'package', 'dependencies', 'lints'})
    if unknown:
        raise PolicyError(f'unsupported SDK manifest keys {sorted(unknown)}')
    dependencies = {}
    for name, spec in data['workspace'].get('dependencies', {}).items():
        if isinstance(spec, dict) and 'path' in spec:
            parts = PurePosixPath(spec['path']).parts
            if parts[:2] == ('models', model):
                spec = {**spec, 'path': PurePosixPath(*parts[2:]).as_posix()}
            elif parts[0] == 'models':
                continue
            else:
                spec = {**spec, 'path': 'sdk/' + spec['path']}
        dependencies[name] = spec
    # The SDK's optional tables carry over when it declares them.
    workspace = {'members': list(roles), 'exclude': ['sdk'],
                 **{key: data['workspace'][key] for key in ('resolver', 'package') if key in data['workspace']},
                 'dependencies': dependencies, 'lints': data['workspace'].get('lints', {})}
    text = (f'# Generated by tools/model_sdk export from models/{model} and the pinned SDK\'s\n'
            '# Cargo.toml: the shared crates resolve in sdk/.\n')
    text += toml_tables('workspace', workspace)
    if 'profile' in data:
        text += '\n' + toml_tables('profile', data['profile'])
    return text


class Relocation:
    """Rewrites the references of one model's files for the workspace layout
    (module docstring). SDK is the extracted SDK checkout."""

    def __init__(self, sdk, model):
        self.sdk, self.model = sdk, model
        self.directory = f'models/{model}'
        # (file, offset in the emitted file, checkout base, workspace base,
        # original text, emitted text) of each reference matched, rewritten
        # or not, for `verify`.
        self.references = []
        # (file, offset in the emitted file, reference, reason) of each kept
        # reference reported for a reason of its own.
        self.reported = []

    def path(self, target):
        """The workspace-relative path of checkout-relative TARGET, or None
        when the layout does not move it."""
        parts = PurePosixPath(target).parts
        if target == '.':
            return '.'
        if parts[:2] == ('fixtures', self.model):
            return PurePosixPath('fixtures', *parts[2:]).as_posix()
        if parts[:2] == ('models', self.model):
            return PurePosixPath(*parts[2:]).as_posix() if len(parts) > 2 else '.'
        if (self.sdk / target).exists():
            return 'sdk/' + target
        return None

    def rooted(self, match):
        """The relocated text of a checkout-relative path. One with a `..`
        component is resolved in the checkout first and relocated as a whole
        (`path`), so it keeps naming the same file, a top-level one too."""
        value, slash = match.groups()
        if '..' in value.split('/'):
            target = posixpath.normpath(value)
            if target.startswith('..'):
                return match.group()
            relocated = self.path(target)
            parts = target.split('/')
            if relocated is None and len(parts) > 1 and (self.sdk / parts[0] / parts[1]).exists():
                relocated = 'sdk/' + target
            return relocated + slash if relocated is not None else match.group()
        model, fixtures = f'models/{self.model}', f'fixtures/{self.model}'
        if value == model:
            # The model directory is the workspace root; models/<m>/ names
            # its contents.
            return '' if slash else '.'
        if value.startswith(model + '/'):
            return value[len(model) + 1:] + slash
        if value == fixtures or value.startswith(fixtures + '/'):
            return 'fixtures' + value[len(fixtures):] + slash
        parts = value.split('/')
        if (self.sdk / parts[0] / parts[1]).exists():
            return f'sdk/{value}{slash}'
        return match.group()

    def climb(self, value, base):
        """The relocated text of the relative path VALUE (a `../` path, or
        any path an include macro names), read from BASE (a directory
        relative to the model root)."""
        target = posixpath.normpath(posixpath.join(self.directory, base, value))
        if target.startswith('..') or target == self.directory or target.startswith(self.directory + '/'):
            return value
        relocated = self.path(target)
        if relocated is None:
            return value
        result = posixpath.relpath(relocated, base or '.')
        return result + '/' if value.endswith('/') and not result.endswith('/') else result

    def text(self, relative, text):
        """TEXT of the model file at model-root path RELATIVE, relocated in one
        pass over the original: each reference is matched once in the
        original text, so an emitted path is never matched again."""
        suffix = PurePosixPath(relative).suffix
        # (start, end, base, replacement, old path, new path, offset of the
        # new path in the replacement or None when the replacement is the
        # original text, and the reason a kept reference is reported, or
        # None); base is None for a checkout-relative path, else the
        # directory a relative path reads from.
        found = []
        directory = posixpath.dirname(relative)
        crate = self.crate(relative)
        def replace(start, end, base, new):
            found.append((start, end, base, new, text[start:end], new, 0, None))
        # The literals of the Rust forms `rust` relocates; the regexes skip them.
        forms = self.rust(relative, text, found) if suffix == '.rs' else []
        def handled(match):
            return any(start <= match.start() < end for start, end in forms)
        if suffix in ('.rs', '.md'):
            # Climbs in Markdown read from its file; other climbs in Rust
            # (`.join` arguments, comments) from the crate.
            base = crate if suffix == '.rs' else directory
            for match in CLIMB.finditer(text):
                if not handled(match):
                    replace(match.start(), match.end(), base, self.climb(match.group(), base))
        for match in ROOTED.finditer(text):
            if not handled(match):
                replace(match.start(), match.end(), None, self.rooted(match))
        pieces, position, written = [], 0, 0
        for start, end, base, replacement, old, new, inner, reason in sorted(found):
            if start < position:
                raise PolicyError(f'{relative}: overlapping path references at offset {start}')
            pieces.append(text[position:start])
            written += start - position
            if reason is not None:
                self.reported.append((relative, written, old, reason))
                pieces.append(replacement)
                written += len(replacement)
                position = end
                continue
            # A checkout-relative path reads from the checkout root, its
            # relocation from the workspace root.
            # A base is a directory relative to the model root, or a pair of
            # the checkout directory and its exported counterpart.
            checkout, workspace = (base if isinstance(base, tuple) else
                                   (posixpath.join(self.directory, base), base) if base is not None else ('', ''))
            self.references.append((relative, written + (inner or 0), checkout, workspace, old, new,
                                    inner is not None))
            pieces.append(replacement)
            written += len(replacement)
            position = end
        pieces.append(text[position:])
        return ''.join(pieces)

    def rust(self, relative, text, found):
        """Add the recognized path forms of Rust TEXT to FOUND (`text`) and
        refuse any other string literal that names a checkout path outside
        the model directory. Returns the token spans of the literals it
        handled, which the climb and rooted regexes skip."""
        directory, crate = posixpath.dirname(relative), self.crate(relative)
        paths = RustPaths(text, posixpath.normpath(posixpath.join(self.directory, crate)))
        def line(offset):
            return text.count('\n', 0, offset) + 1
        if paths.inline:
            raise PolicyError(f'{relative}:{line(paths.inline[0])}: a #[path] attribute inside an inline module '
                              "reads from the module's directory, which the export does not resolve; declare the "
                              'module in a file of its own')
        def resolve(base, value):
            return posixpath.normpath(posixpath.join(self.directory, base, value))
        def outside(target):
            """Whether TARGET, a checkout path, names a checkout file outside
            the model directory."""
            return (target not in ('.', '..') and not target.startswith(('../', '/'))
                    and target != self.directory and not target.startswith(self.directory + '/')
                    and (self.sdk / target).exists())
        def spelled(span):
            """Whether the literal at content span SPAN spells its text as
            written, so its contents can be rewritten in place."""
            return text[span[0]:span[1]] == paths.text[span]
        def literal(span, base, value, new, lead=''):
            """Record the relocation of the one-literal path VALUE at content
            span SPAN to NEW: in place when the literal spells its text, else
            the whole literal becomes a plain one."""
            start, end = span
            if spelled(span):
                found.append((start + len(lead), end, base, new, value, new, 0, None))
            elif new == value:
                found.append((*paths.tokens[span], base, text[slice(*paths.tokens[span])],
                              value, new, None, None))
            else:
                replacement = '"' + lead + new + '"'
                found.append((*paths.tokens[span], base, replacement, value, new, 1 + len(lead), None))
        def climbs(value):
            return '..' in value.split('/')
        def inside(path):
            return path == self.directory or path.startswith(self.directory + '/')
        def relocate(base, value):
            """VALUE read from checkout directory BASE: the exported base and
            the text that names the same file from it."""
            exported = posixpath.relpath(base, self.directory) if inside(base) else self.path(base)
            exported = exported if exported is not None else posixpath.relpath(base, self.directory)
            target = posixpath.normpath(posixpath.join(base, value))
            relocated = self.path(target)
            if target.startswith('..') or inside(base) and inside(target) or relocated is None:
                return exported, value
            result = posixpath.relpath(relocated, exported)
            return exported, result + '/' if value.endswith('/') and not result.endswith('/') else result
        def refuse(start, source, why):
            raise PolicyError(f'{relative}:{line(start)}: {source}: {why}')
        # Each `.join` and `.starts_with` literal with the directory its
        # receiver names, when the text establishes it.
        sites = [(method, span, at, paths.bases.value(receiver), receiver)
                 for method, span, receiver, at in paths.joins]
        calls = {paths.tokens[span] for _, span, _, _, _ in sites}
        # The file joins its crate directory up to the checkout root, so its
        # checkout-rooted literals name paths from that root.
        root = any(method == 'join' and base is not None
                   and posixpath.normpath(posixpath.join(base, paths.text[span])) == '.'
                   for method, span, _, base, _ in sites)
        forms = []
        for kind, pieces, whole in paths.forms:
            value = ''.join(paths.text[span] for span in pieces)
            base = directory if kind == 'file' else crate
            lead = ''
            if kind == 'manifest':
                if not value.startswith('/'):
                    continue
                value, lead = value[1:], '/'
                root = root or resolve(base, value) == '.'
            forms.extend(paths.tokens[span] for span in pieces)
            parts = posixpath.normpath(value).split('/')
            moves = climbs(value) or (kind == 'file' and len(parts) > 1
                                      and parts[0] in ('models', 'fixtures', 'crates'))
            if not moves:
                continue
            new = self.climb(value, base)
            if len(pieces) == 1:
                literal(pieces[0], base, value, new, lead)
            elif new == value:
                found.append((*whole, base, text[whole[0]:whole[1]], value, new, None, None))
            else:
                # A path joined from several literals becomes one literal.
                replacement = '"' + lead + new + '"'
                found.append((*whole, base, replacement, value, new, 1 + len(lead), None))
        script = self.script_or_test(relative)
        def from_root(span, value, target):
            """Relocate the checkout-rooted literal VALUE, read from the root."""
            if target.split('/')[0] not in SURFACE:
                # In-tree tooling is not part of the SDK's library surface.
                forms.append(paths.tokens[span])
                found.append((*span, None, text[span[0]:span[1]], value, value, 0, TOOLING))
            elif not (spelled(span) and value.rstrip('/') == target
                      and target.split('/')[0] in ('models', 'fixtures', 'crates')):
                # Any other spelling becomes the plain relocated path; the
                # rooted regex relocates a plain one in place.
                forms.append(paths.tokens[span])
                new = self.path(target) + '/' * value.endswith('/')
                found.append((*paths.tokens[span], None, '"' + new + '"', value, new, 1, None))
        unknown = 'the base of this call could not be established from the text'
        for method, span, at, base, receiver in sites:
            value, source = paths.text[span], text[slice(*paths.tokens[span])]
            target = posixpath.normpath(value)
            rooted = '/' in target and outside(target)
            if method == 'starts_with':
                if not rooted:
                    continue
                if base == '.' or paths.bases.joined_to_root(receiver, at):
                    from_root(span, value, target)
                elif target.split('/')[0] not in SURFACE:
                    from_root(span, value, target)
                else:
                    refuse(span[0], source, f'names {target} from the checkout root, but {unknown}')
            elif climbs(value):
                if base is not None:
                    forms.append(paths.tokens[span])
                    exported, new = relocate(base, value)
                    literal(span, (base, exported), value, new)
                elif script and inside(resolve(crate, value)):
                    # A build script or test of the crate may climb inside
                    # the model directory from any receiver; it is kept.
                    forms.append(paths.tokens[span])
                    literal(span, crate, value, value)
                else:
                    refuse(span[0], source, f'climbs, but {unknown}')
            elif base is not None and base != '.':
                # Read from that base, which the export relocates.
                forms.append(paths.tokens[span])
                if rooted and not (self.sdk / posixpath.normpath(posixpath.join(base, value))).exists():
                    refuse(span[0], source, f'names {target} from the checkout root, but the receiver is not '
                           'the checkout root')
            elif rooted:
                if base == '.' or target.split('/')[0] not in SURFACE:
                    from_root(span, value, target)
                else:
                    refuse(span[0], source, f'names {target} from the checkout root, but {unknown}')
        for span, piece in paths.literals:
            if paths.tokens[span] in forms or paths.tokens[span] in calls:
                continue
            start, end = span
            value, source = paths.text[span], text[slice(*paths.tokens[span])]
            candidate = value.lstrip('/') if piece else value
            parts = candidate.split('/')
            if len(parts) < 2 or any(character.isspace() for character in candidate):
                continue
            # A climb out of the model directory outside a recognized form.
            if climbs(candidate) and outside(resolve(crate, candidate)):
                raise PolicyError(f'{relative}:{line(start)}: {source} climbs to '
                                  f'{resolve(crate, candidate)} in the checkout, outside the model directory, '
                                  'but is not an include, #[path], manifest-directory or `.join` path the export '
                                  'rewrites')
            # A checkout-rooted path is read from the root the file joins to.
            # In-tree tooling is not part of the SDK's library surface, so a
            # reference to it is kept and reported.
            target = posixpath.normpath(candidate)
            if not outside(target):
                continue
            if not piece and target.split('/')[0] not in SURFACE:
                from_root(span, value, target)
                continue
            if piece or not root:
                raise PolicyError(f'{relative}:{line(start)}: {source} names {target} from the checkout root, '
                                  'but this file joins no path to the checkout root, so the export cannot tell '
                                  'what it is read from')
            from_root(span, value, target)
        return forms

    def verify(self, output, relocated):
        """Refuse a rewrite whose final emitted reference does not name, from
        the same base in OUTPUT, what the original reference named in the
        checkout (module docstring). RELOCATED maps each rewritten file to
        its digests. Returns the kept references it could not resolve."""
        def same_file(before, after, name):
            if not after.is_file() or after.is_symlink():
                return False
            old, new = before.read_bytes(), after.read_bytes()
            entry = relocated.get(name)
            return old == new or (entry is not None and entry['source_sha256'] == digest(old)
                                  and entry['sha256'] == digest(new))
        def same(before, after):
            if after == '..' or after.startswith('../'):
                return False
            if before == '.':
                return after == '.'
            old, new = self.sdk / before, output / after
            if old.resolve() == new.resolve():
                return True
            if not old.exists():
                return not new.exists() and not new.is_symlink()
            if old.is_file():
                return same_file(old, new, after)
            return new.is_dir() and all(
                same_file(path, new / path.relative_to(old),
                          posixpath.normpath(posixpath.join(after, path.relative_to(old).as_posix())))
                for path in sorted(old.rglob('*')) if path.is_file())
        checked, emitted, unresolved = {}, {}, []
        for relative, offset, checkout_base, workspace_base, old, new, emitted_text in self.references:
            if relative not in emitted:
                emitted[relative] = (output / relative).read_text()
            if emitted_text and emitted[relative][offset:offset + len(new)] != new:
                raise PolicyError(f'{relative}: the exported file does not hold the rewrite of {old!r} '
                                  f'as {new!r} at offset {offset}')
            key = (posixpath.normpath(posixpath.join(checkout_base, old)),
                   posixpath.normpath(posixpath.join(workspace_base, new)))
            # A reference left as written that names nothing in the checkout
            # (a crate-relative or prose path, or one a program completes at
            # run time) makes no claim to check; the export reports it.
            if old == new and (key[0] == '..' or key[0].startswith('../') or not (self.sdk / key[0]).exists()):
                unresolved.append({'file': relative, 'line': emitted[relative].count('\n', 0, offset) + 1,
                                   'reference': old, 'read_from': 'checkout root' if checkout_base == ''
                                   else workspace_base or '.'})
                continue
            if key not in checked:
                checked[key] = same(*key)
            if not checked[key]:
                kept = 'kept' if old == new else f'rewrote {old!r} as'
                raise PolicyError(f'{relative}: the export {kept} {new!r}, which does not name '
                                  f'what {old!r} names in the checkout')
        for relative, offset, reference, reason in self.reported:
            if relative not in emitted:
                emitted[relative] = (output / relative).read_text()
            unresolved.append({'file': relative, 'line': emitted[relative].count('\n', 0, offset) + 1,
                               'reference': reference, 'read_from': 'checkout root', 'reason': reason})
        return unresolved

    def script_or_test(self, relative):
        """Whether model file RELATIVE is a crate's build script or under its
        tests directory."""
        parts = PurePosixPath(relative).parts
        if len(parts) < 2 or parts[0] not in ROLES:
            return False
        if parts[1] == 'tests':
            return True
        manifest = self.sdk / self.directory / parts[0] / 'Cargo.toml'
        build = tomllib.loads(manifest.read_text()).get('package', {}).get('build', 'build.rs') \
            if manifest.is_file() else 'build.rs'
        return isinstance(build, str) and posixpath.normpath(posixpath.join(parts[0], build)) == relative

    def crate(self, relative):
        parts = PurePosixPath(relative).parts
        return parts[0] if len(parts) > 1 and parts[0] in ROLES else ''


def copied_verbatim(relative):
    parts = PurePosixPath(relative).parts
    return 'fixtures' in parts[:-1] or 'kernels.lock.d' in parts[:-1] or parts[-1].endswith('.lock')


def export_model(sdk, output, model_directory):
    """Lay out models/<m> of the extracted SDK checkout SDK as the workspace
    OUTPUT (an empty directory); returns the receipt it writes."""
    parts = PurePosixPath(model_directory).parts
    if len(parts) != 2 or parts[0] != 'models' or not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', parts[1]):
        raise PolicyError('--model must name an in-tree model directory, models/<m>')
    model = parts[1]
    source = sdk / 'models' / model
    if not (source / 'model.toml').is_file():
        raise PolicyError(f'{model_directory} has no model.toml at the exported revision')
    fixtures = sdk / 'fixtures' / model
    clashes = [name for name in (*GENERATED, *(('fixtures',) if fixtures.is_dir() else ()))
               if (source / name).exists() or (source / name).is_symlink()]
    if clashes:
        raise PolicyError(f'{model_directory} holds {", ".join(clashes)}, which the workspace layout writes')
    relocation = Relocation(sdk, model)
    relocated = {}
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source).as_posix()
        if path.is_symlink():
            raise PolicyError(f'model source symlink is unsupported: {relative}')
        destination = output / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        shutil.copy2(path, destination)
        if copied_verbatim(relative):
            continue
        data = path.read_bytes()
        try:
            text = data.decode()
        except UnicodeDecodeError:
            continue
        rewritten = relocation.text(relative, text).encode()
        if rewritten != data:
            destination.write_bytes(rewritten)
            relocated[relative] = {'source_sha256': digest(data), 'sha256': digest(rewritten)}
    if fixtures.is_dir():
        shutil.copytree(fixtures, output / 'fixtures', symlinks=True)
        if any(path.is_symlink() for path in (output / 'fixtures').rglob('*')):
            raise PolicyError(f'fixtures/{model} holds a symlink')
    roles = [role for role in ROLES if (output / role / 'Cargo.toml').is_file()]
    (output / 'Cargo.toml').write_text(manifest(sdk, model, roles))
    shutil.copyfile(sdk / 'rust-toolchain.toml', output / 'rust-toolchain.toml')
    (output / '.gitignore').write_text(IGNORED)
    unresolved = relocation.verify(output, relocated)
    receipt = {'schema': SCHEMA, 'model': model, 'source': model_directory,
               'fixtures': f'fixtures/{model}' if fixtures.is_dir() else None, 'relocated': relocated,
               'unresolved': unresolved}
    (output / RECEIPT).write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    for entry in unresolved:
        reason = entry.get('reason', 'names nothing in the checkout')
        print(f"warning: {entry['file']}:{entry['line']}: kept {entry['reference']!r} (read from "
              f"{entry['read_from']}), which {reason}; the export cannot resolve it, so "
              "check it with the exported workspace's check and tests", file=sys.stderr)
    return receipt

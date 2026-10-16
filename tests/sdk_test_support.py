"""model-compiler-sdk's test support sources (crates/model-compiler-sdk/
test-support/) sit outside its module tree: models' tests include them by
#[path]. No build may read them, so they can never enter the SDK's library,
nor the host, compilers, bundles and packages. Default deny:

- Every file of the checkout (tracked or untracked and not ignored, any
  suffix, any directory) that names the test support directory by a literal
  path must be listed in ALLOWED. So must every file that names a listed
  Rust file by path, or declares it as a module from a file that would load it
  by default.
- Each listed file is verified test-only: it sits in a crate's tests/ and
  every loader it lists is itself verified, or each module declaration that
  loads it in its listed declarers carries #[cfg(test)]. No listed file may be
  a Cargo target other than a test (lib, bin, example, bench, build script)
  or sit under examples/ or benches/.
- The SDK's own sources, manifest and build scripts outside its tests/ may
  not mention test-support or test_support at all.

Literal paths are also matched after normalizing their components (./,
repeated slashes, a/../ pairs). Attributes are read from the code alone, with
comments removed and string contents set aside, and only an attribute
attached to the declaration counts.

Purpose: this guard catches accidental reach, such as a new include of the
test support from a build script or a served module. Deliberate obfuscation
is out of scope: a file that reaches the test support without spelling its
path (concat! fragments, a path computed at build time) is not seen. The
release identity builds of each landing are the proof that no served byte
changed."""
import posixpath
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
SDK = 'crates/model-compiler-sdk'
SUPPORT = f'{SDK}/test-support'
NAMES = ('test-support', 'test_support')
SEPARATOR = r'\s*[/\\]+\s*'

# Each file allowed to name the test support or a listed file: (kind, reason,
# related). kind `tests`: in a crate's tests/, and `related` are the files
# that load it, each verified the same way unless it is itself in a tests/.
# kind `cfg-test`: `related` are the files that declare it as a module, each
# declaration under #[cfg(test)]. kind `guard`: this guard and its tests,
# outside every crate.
ALLOWED = {
    f'{SDK}/tests/test_support_checkpoint.rs':
        ('tests', "the SDK's integration test of checkpoint.rs", ()),
    'tests/sdk_test_support.py': ('guard', 'this guard', ()),
    'tests/test_core_guards.py': ('guard', "this guard's tests", ()),
}
for _model in ('mimo26', 'qwen38'):
    ALLOWED |= {
        f'models/{_model}/contract/tests/projections_support/mod.rs':
            ('tests', 'the projection test support of the contract, which includes checkpoint.rs',
             (f'models/{_model}/contract/tests/sm120/projections.rs',
              f'models/{_model}/contract/src/projections/variants/tests.rs',
              f'models/{_model}/compiler/src/projection_comparison.rs')),
        f'models/{_model}/contract/src/projections/variants/tests.rs':
            ('cfg-test', "the variants' unit tests, which load the projection test support",
             (f'models/{_model}/contract/src/projections/variants.rs',)),
        f'models/{_model}/compiler/src/projection_comparison.rs':
            ('cfg-test', "the compiler's projection comparison tests, which include synthetic.rs",
             (f'models/{_model}/compiler/src/lib.rs',)),
    }
ALLOWED |= {
    'models/glm5next/compiler/src/test_native.rs':
        ('cfg-test', "the preparation tests' native fixture, which includes native.rs",
         ('models/glm5next/compiler/src/preparation.rs',)),
    'models/mimo26/compiler/src/test_native.rs':
        ('cfg-test', "the binaries' unit tests' native fixture, which includes native.rs",
         ('models/mimo26/compiler/src/bin/chopinfer-model-mimo26.rs',
          'models/mimo26/compiler/src/bin/mimo26-package.rs')),
    'models/qwen38/compiler/src/test_native.rs':
        ('cfg-test', "the compiler's unit tests' native fixture, which includes native.rs",
         ('models/qwen38/compiler/src/lib.rs',)),
}

# Over code from code_of(): string literals are "\0N" and comments are gone.
ATTRIBUTE = r'#\s*\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
DECLARATION = re.compile(rf'((?:{ATTRIBUTE}\s*)*)(?:pub(?:\s*\([^)]*\))?\s+)?\bmod\s+(?:r#)?(\w+)\s*;')
PATH_ATTRIBUTE = re.compile(r'#\s*\[\s*path\s*=\s*"\0(\d+)"\s*\]')
CFG_TEST = re.compile(r'#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]')
PATH_TOKEN = re.compile(r'[\w./\\-]+')


def checkout(root):
    """Every tracked file, and every untracked file Git does not ignore."""
    output = subprocess.run(['git', '-C', str(root), 'ls-files', '-z', '--cached', '--others',
                             '--exclude-standard'], check=True, capture_output=True).stdout
    return sorted({name for name in output.decode().split('\0') if name and (root / name).is_file()})


def support_reference(root):
    """A reference to the SDK's test support directory, in any spelling: the
    SDK path, or test-support followed by one of its file names, with or
    without spaces and string breaks (`"a", "/", "b"`), in normal or raw
    strings."""
    names = {Path(name).stem for name in os.listdir(root / SUPPORT)} if (root / SUPPORT).is_dir() else set()
    names |= {'checkpoint', 'synthetic', 'native'}
    files = '|'.join(sorted(map(re.escape, names)))
    gap = r'[\s"\'#,+]*[/\\]+[\s"\'#,+]*'
    return re.compile(rf'model-compiler-sdk{gap}test-support|test-support{gap}(?:{files})\b')


def file_reference(name):
    """A path ending in the listed file `name`: its file name, or its directory
    and file name when the file name alone is generic, with the path before it."""
    parts = Path(name).parts
    tail = parts[-2:] if parts[-1] in ('mod.rs', 'lib.rs', 'main.rs', 'tests.rs') else parts[-1:]
    return re.compile(r'(?<![\w.-])' + SEPARATOR.join(map(re.escape, tail)) + r'(?![\w-])')


def names_file(root, source, line, listed, reference):
    """Whether `line` of `source` names the listed file: a path that resolves
    to it from the source's directory, or that resolves to no file at all (so
    it may reach it some other way). A path to another file of the same name
    is not a reference."""
    for match in reference.finditer(line):
        start = match.start()
        while start and re.match(r'[\w./\\-]', line[start - 1]):
            start -= 1
        text = re.sub(r'\s+', '', line[start:match.end()]).replace('\\', '/')
        target = Path(os.path.normpath(Path(source).parent / text))
        if target == Path(listed) or not (root / target).is_file():
            return True
    return False


def default_loaders(name):
    """The files that would load `name` by a plain `mod <stem>;`, and that stem."""
    path = Path(name)
    if path.name == 'mod.rs':
        stem, home = path.parent.name, path.parent.parent
    else:
        stem, home = path.stem, path.parent
    loaders = [home / 'lib.rs', home / 'main.rs', home / 'mod.rs']
    loaders.append(home.parent / f'{home.name}.rs')
    return stem, [loader.as_posix() for loader in loaders]


def normalized(line):
    """`line` with each literal path's components normalized: separators as
    one slash, no ./ and no a/../ pairs."""
    def path(match):
        token = match.group(0)
        if '/' not in token and '\\' not in token:
            return token
        return posixpath.normpath(re.sub(r'[/\\]+', '/', token))
    return PATH_TOKEN.sub(path, line)


def code_of(text):
    """Rust `text` without comments, each string literal replaced by "\0N"
    where N indexes the returned literal values, line breaks kept."""
    code, literals, i, size = [], [], 0, len(text)
    while i < size:
        char, after_word = text[i], i > 0 and (text[i - 1].isalnum() or text[i - 1] == '_')
        if text.startswith('//', i):
            end = text.find('\n', i)
            i = size if end < 0 else end
            continue
        if text.startswith('/*', i):
            depth, end = 1, i + 2
            while end < size and depth:
                step = 1
                if text.startswith('/*', end):
                    depth, step = depth + 1, 2
                elif text.startswith('*/', end):
                    depth, step = depth - 1, 2
                end += step
            code.append(' ' + '\n' * text.count('\n', i, end))
            i = end
            continue
        raw = None if after_word else re.match(r'b?r(#*)"', text[i:i + 260])
        if raw:
            start = i + raw.end()
            end = text.find('"' + raw.group(1), start)
            end = size if end < 0 else end
            value, i = text[start:end], end + 1 + len(raw.group(1))
        elif char == '"' or (char == 'b' and not after_word and text.startswith('b"', i)):
            start = end = i + (2 if char == 'b' else 1)
            while end < size and text[end] != '"':
                end += 2 if text[end] == '\\' else 1
            value, i = text[start:end], end + 1
        elif char == "'":
            literal = re.match(r"'(?:\\[^']{1,10}|[^'\\])'", text[i:i + 13])
            code.append("' '" if literal else char)
            i += literal.end() if literal else 1
            continue
        else:
            code.append(char)
            i += 1
            continue
        code.append(f'"\0{len(literals)}"' + '\n' * value.count('\n'))
        literals.append(value)
    return ''.join(code), literals


def lines_of(path):
    return path.read_bytes().decode('utf-8', errors='replace').splitlines()


class Crates:
    """The Cargo packages of the checkout and the non-test targets of each."""

    def __init__(self, root, files):
        self.root = root
        self.packages = {}
        for name in files:
            if Path(name).name != 'Cargo.toml':
                continue
            try:
                table = tomllib.loads((root / name).read_text(encoding='utf-8', errors='replace'))
            except tomllib.TOMLDecodeError:
                table = {}
            if isinstance(table.get('package'), dict):
                self.packages[Path(name).parent] = table

    def crate(self, name):
        for parent in Path(name).parents:
            if parent in self.packages:
                return parent
        return None

    def targets(self, crate):
        """Every lib, bin, example, bench and build script path of a crate."""
        table, base = self.packages[crate], self.root / crate
        found = {crate / 'src/lib.rs', crate / 'src/main.rs', crate / 'build.rs'}
        found |= {path.relative_to(self.root) for path in (base / 'src/bin').glob('*.rs')}
        found |= {path.relative_to(self.root) for path in (base / 'src/bin').glob('*/main.rs')}
        build = table['package'].get('build')
        if isinstance(build, str):
            found.add(crate / build)
        for key in ('lib', 'bin', 'example', 'bench'):
            entries = table.get(key, [])
            for entry in entries if isinstance(entries, list) else [entries]:
                if isinstance(entry, dict) and isinstance(entry.get('path'), str):
                    found.add(crate / entry['path'])
        return {Path(os.path.normpath(path)) for path in found}


def cfg_test_declarations(root, declarer, target):
    """Whether `declarer` declares `target` as a module, and whether every such
    declaration carries #[cfg(test)] among the attributes attached to it.
    Comments and string contents never count as attributes. Declarations
    inside inline modules are resolved as if at the top of the file."""
    source = Path(declarer)
    mod_rs = source.name in ('lib.rs', 'main.rs', 'mod.rs') or source.parent.name == 'bin'
    home = source.parent if mod_rs else source.parent / source.stem
    code, literals = code_of((root / declarer).read_bytes().decode('utf-8', errors='replace'))
    found, all_test = False, True
    for declaration in DECLARATION.finditer(code):
        attributes = re.findall(ATTRIBUTE, declaration.group(1))
        paths = [literals[int(match.group(1))] for match in map(PATH_ATTRIBUTE.fullmatch, attributes) if match]
        if paths:
            candidates = [source.parent / paths[-1]]
        else:
            name = declaration.group(2)
            candidates = [home / f'{name}.rs', home / name / 'mod.rs']
        if any(Path(os.path.normpath(candidate)) == Path(target) for candidate in candidates):
            found = True
            all_test &= any(CFG_TEST.fullmatch(attribute) for attribute in attributes)
    return found, all_test


def verify(root, crates, allowed, name, seen=()):
    """Why the listed file `name` is not verifiably test-only, or None."""
    if name not in allowed:
        return f'{name} is not in the allowlist'
    kind, _, related = allowed[name]
    if not (root / name).is_file():
        return f'{name} is listed but missing'
    if kind == 'guard':
        if name.endswith('.rs') or crates.crate(name) is not None:
            return f'{name} is listed as the guard but is Rust source or inside a crate'
        return None
    crate = crates.crate(name)
    if crate is None:
        return f'{name} is in no crate'
    inside = Path(name).relative_to(crate).parts
    if Path(name) in crates.targets(crate) or inside[0] in ('examples', 'benches'):
        return f'{name} is a Cargo lib, bin, example, bench or build target of {crate}'
    if kind == 'tests':
        if inside[0] != 'tests':
            return f"{name} is listed as a test file but is not in {crate}/tests/"
        for loader in related:
            loader_crate = crates.crate(loader)
            if loader_crate is not None and Path(loader).relative_to(loader_crate).parts[0] == 'tests' \
                    and Path(loader) not in crates.targets(loader_crate):
                continue
            if loader in seen:
                return f'{name} is loaded through a cycle at {loader}'
            why = verify(root, crates, allowed, loader, seen + (name,))
            if why:
                return f'{name} is loaded by {loader}, which is not test-only: {why}'
        return None
    if kind == 'cfg-test':
        if not related:
            return f'{name} lists no declaring module'
        for declarer in related:
            if not (root / declarer).is_file():
                return f'{name} lists the missing declarer {declarer}'
            found, test = cfg_test_declarations(root, declarer, name)
            if not found:
                return f'{declarer} does not declare {name} as a module'
            if not test:
                return f'{declarer} declares {name} without #[cfg(test)]'
        return None
    return f'{name} has the unknown kind {kind}'


def violations(root):
    """Each refused file or line, as text."""
    root = Path(root)
    allowed = ALLOWED
    files = checkout(root)
    crates = Crates(root, files)
    support = support_reference(root)
    references = {name: file_reference(name) for name, entry in allowed.items() if entry[0] != 'guard'}
    defaults = {name: default_loaders(name) for name in references}
    # Words every spelling of a reference contains, to skip most files fast.
    word = {name: Path(name).parent.name if Path(name).name in ('mod.rs', 'lib.rs', 'main.rs', 'tests.rs')
            else Path(name).stem for name in references}
    words = {'test-support'} | set(word.values())
    found = []
    for name in allowed:
        why = verify(root, crates, allowed, name)
        if why:
            found.append(f'{name}: allowlisted, but cannot be verified test-only: {why}')
    for name in files:
        if name.startswith(f'{SUPPORT}/'):
            continue
        in_sdk = name.startswith(f'{SDK}/') and not name.startswith(f'{SDK}/tests/')
        text = (root / name).read_bytes().decode('utf-8', errors='replace')
        loader = any(name in loaders for _, loaders in defaults.values())
        if not (in_sdk or loader or any(term in text for term in words)):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if not (in_sdk or (loader and 'mod' in line) or any(term in line for term in words)):
                continue
            where = f'{name}:{number}'
            variants = {line, normalized(line)}
            if in_sdk:
                if any(term in line for term in NAMES):
                    found.append(f'{where}: the SDK names its own test support sources, which only '
                                 'models\' tests may include')
            elif any(support.search(variant) for variant in variants) and name not in allowed:
                found.append(f'{where}: names the SDK\'s test support sources but is not in the allowlist '
                             'of test-only files in tests/sdk_test_support.py')
            if allowed.get(name, ('',))[0] == 'guard':
                continue
            for listed, reference in references.items():
                if name == listed or name in allowed[listed][2]:
                    continue
                stem, loaders = defaults[listed]
                named = word[listed] in line and any(names_file(root, name, variant, listed, reference)
                                                     for variant in variants)
                declared = name in loaders and re.search(rf'\bmod\s+(?:r#)?{re.escape(stem)}\s*;', line)
                if named or declared:
                    found.append(f'{where}: loads {listed}, a test-only file of the allowlist, but is not '
                                 'one of its listed loaders')
    return found


if __name__ == '__main__':
    found = violations(ROOT)
    if found:
        sys.exit('\n'.join(found))

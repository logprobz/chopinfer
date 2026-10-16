"""Core's exact supported inventory and source policy. External data cannot alter it."""
from .graph import Graph, selected_graph

# Each model's files that imported kernels_sm120 before models reached kernels
# through kernel_core::backend, relative to the model's directory. A list only
# shrinks: new SM120 code goes in an sm120 module. The baseline holds the
# SHA-256 of every path a list may name ("models/<m>/<path>"), so a list
# cannot grow without a reviewed baseline change.
KERNEL_IMPORTS = 'kernels_sm120_imports.txt'
KERNEL_IMPORT_BASELINE = 'tools/model_sdk/kernels_sm120_imports.sha256'

# Each model's re-exports of SM120 items outside its sm120 modules, one "<file> <item>"
# per line, the file relative to the model's directory. Generic model code names no
# SM120 item, also not through a re-export; a list only shrinks, and its baseline holds
# exactly the SHA-256 of every listed "models/<m>/<file> <item>" line (sm120_names).
SM120_REEXPORTS = 'sm120_reexports.txt'
SM120_REEXPORT_BASELINE = 'tools/model_sdk/sm120_reexports.sha256'

# Each model's other paths through its sm120 modules outside them (private `use` leaves
# and paths in code), in the same "<file> <item>" form, list and baseline rules.
# kernels_sm120 itself stays in KERNEL_IMPORTS.
SM120_USES = 'sm120_uses.txt'
SM120_USE_BASELINE = 'tools/model_sdk/sm120_uses.sha256'


def rust_tokens(text, strings=False):
    """Rust tokens without strings, character literals and comments, which cannot import items.
    Keeps path punctuation. A string may continue over lines with a trailing backslash. With
    strings, each string literal stays one token, its source text."""
    import re
    literals = []

    def blank(match):
        if strings and match.group(0)[0] in 'br"':
            literals.append(match.group(0))
            return f' \0{len(literals) - 1}\0 '
        return ' '

    text = re.sub(r'(?<![A-Za-z0-9_])b?r(\#*)"[\s\S]*?"\1|"(?:\\[\s\S]|[^"\\])*"'
                  r"|'(?:\\(?:u\{[0-9A-Fa-f]+\}|x[0-9A-Fa-f]{2}|[\s\S])|[^'\\\n])'"
                  r'|//[^\n]*|/\*[\s\S]*?\*/', blank, text)
    text = re.sub(r'\br#([A-Za-z_][A-Za-z_0-9]*)', r'\1', text)
    return [literals[int(token[1:-1])] if token[0] == '\0' else token
            for token in re.findall(r'\0[0-9]+\0|::|[A-Za-z_][A-Za-z_0-9]*|[^\s]', text)]


def kernel_crate_names(manifests):
    """kernels_sm120 and the names the manifests give a dependency on kernels-sm120, also
    through `name.workspace = true` and its workspace's `[workspace.dependencies]`."""
    import tomllib

    def workspace_dependencies(manifest):
        for directory in (manifest.parent, *manifest.parent.parents):
            candidate = directory / 'Cargo.toml'
            if candidate.exists():
                data = tomllib.loads(candidate.read_text())
                if 'workspace' in data:
                    return data['workspace'].get('dependencies', {})
        return {}

    names = {'kernels_sm120'}
    for manifest in manifests:
        data = tomllib.loads(manifest.read_text())
        inherited = None
        for table in [data, *data.get('target', {}).values()]:
            for kind in ('dependencies', 'dev-dependencies', 'build-dependencies'):
                for alias, dependency in table.get(kind, {}).items():
                    if isinstance(dependency, dict) and dependency.get('workspace') is True:
                        if inherited is None:
                            inherited = workspace_dependencies(manifest)
                        dependency = inherited.get(alias)
                    if alias == 'kernels-sm120' or isinstance(dependency, dict) and dependency.get('package') == 'kernels-sm120':
                        names.add(alias.replace('-', '_'))
    return names


def string_value(literal):
    """The text of a string literal token (rust_tokens with strings): a raw string as it stands,
    another with its escapes and line continuations resolved."""
    import re

    body = literal[literal.find('"') + 1:literal.rfind('"')]
    if literal.lstrip('b')[:1] == 'r':
        return body
    simple = {'n': '\n', 'r': '\r', 't': '\t', '0': '\0'}

    def escape(match):
        unicode, byte, other = match.groups()
        if unicode or byte:
            return chr(int(unicode or byte, 16))
        return '' if other[0] == '\n' else simple.get(other, other)

    return re.sub(r'\\(?:u\{([0-9A-Fa-f]+)\}|x([0-9A-Fa-f]{2})|(\n\s*|[\s\S]))', escape, body)


def is_string(token):
    return token[:1] == '"' or token[:2] in ('r"', 'r#', 'b"', 'br')


def module_declarations(path, values):
    """A source's out-of-line module declarations (`mod name;`) as (index, candidate files,
    test, path literal). A declaration is test code when it, or an inline module around it,
    carries `#[cfg(test)]`, in any attribute order and with any visibility. Its file is its
    `#[path]` (also `#[cfg_attr(.., path = ..)]`), else name.rs or name/mod.rs, after the
    module directory: the file's directory for lib.rs, main.rs, mod.rs and build.rs, else the
    file's own name, then the inline modules around it (a `#[path]` outside inline modules is
    relative to the file's directory)."""
    import os
    from pathlib import Path

    base = path.parent if path.name in ('lib.rs', 'main.rs', 'mod.rs', 'build.rs') else path.with_suffix('')
    inline = []  # (brace depth inside, name, test) of the inline modules around a token
    depth = 0
    for i, value in enumerate(values):
        if value in ('{', '}'):
            depth += 1 if value == '{' else -1
            while inline and inline[-1][0] > depth:
                inline.pop()
            continue
        if value != 'mod' or values[i + 2:i + 3] not in (['{'], [';']):
            continue
        start = i
        if values[start - 1:start] == [')']:
            while start > 0 and values[start] != '(':
                start -= 1
        if values[start - 1:start] == ['pub']:
            start -= 1
        attributes = []
        while values[start - 1:start] == [']']:
            close = start - 1
            nesting = 0
            while start > 0:
                start -= 1
                nesting += (values[start] == ']') - (values[start] == '[')
                if not nesting:
                    break
            if values[start - 1:start] != ['#']:
                break
            attributes.append(values[start + 1:close])
            start -= 1
        test = ['cfg', '(', 'test', ')'] in attributes or any(enclosing for _, _, enclosing in inline)
        literal = next((attribute[j + 2] for attribute in attributes if attribute[:1] in (['path'], ['cfg_attr'])
                        for j in range(len(attribute) - 2)
                        if attribute[j:j + 2] == ['path', '='] and is_string(attribute[j + 2])), None)
        if values[i + 2] == '{':
            inline.append((depth + 1, values[i + 1], test))
            continue
        directory = base.joinpath(*(name for _, name, _ in inline))
        if literal is not None:
            relative = directory if inline else path.parent
            files = (Path(os.path.normpath(relative / string_value(literal))),)
        else:
            files = (directory / (values[i + 1] + '.rs'), directory / values[i + 1] / 'mod.rs')
        yield i, files, test, literal


def test_files(sources):
    """The sources only test code reaches: a file is test code when something declares it as a
    module and every declaration is test code or sits in a test file. Files nothing declares
    (crate roots, generated or included files) are not."""
    declared = {}
    for path, values in sources.items():
        for _, files, test, _ in module_declarations(path, values):
            for file in files:
                if file in sources:
                    declared.setdefault(file, []).append((path, test))
    tests = set()
    while True:
        reached = {file for file, sites in declared.items() if file not in tests
                   and all(test or source in tests for source, test in sites)}
        if not reached:
            return tests
        tests |= reached


def production_tokens(sources):
    """Each source's tokens without its `#[cfg(test)]` items, and test_files(sources). Sources
    tokenized with strings resolve `#[path]` declarations."""
    production = {}
    for path, values in sources.items():
        kept = []
        i = 0
        while i < len(values):
            if values[i:i + 7] == ['#', '[', 'cfg', '(', 'test', ')', ']']:
                end = i + 7
                while end < len(values) and values[end] not in (';', '{'):
                    end += 1
                if end < len(values) and values[end] == '{':
                    depth = 1
                    end += 1
                    while end < len(values) and depth:
                        depth += (values[end] == '{') - (values[end] == '}')
                        end += 1
                    i = end
                else:
                    i = end + 1
                continue
            kept.append(values[i])
            i += 1
        production[path] = kept
    return production, test_files(sources)


def use_paths(values, start, end):
    """The paths a `use` tree at values[start:end] names, one per leaf, without renames."""
    cursor = start
    paths = []

    def tree(prefix):
        nonlocal cursor
        parts = list(prefix)
        while cursor < end:
            token = values[cursor]
            cursor += 1
            if token == '::':
                continue
            if token == '{':
                while cursor < end and values[cursor] != '}':
                    tree(parts)
                    if cursor < end and values[cursor] == ',':
                        cursor += 1
                cursor += 1
                return
            parts.append(token)
            if cursor < end and values[cursor] == 'as':
                cursor += 2
                break
            if cursor >= end or values[cursor] in (',', '}'):
                break
        paths.append(parts)

    tree([])
    return paths


def check_model_kernel_imports(root):
    """A model's or example's crates (models/<m>, examples/<e>) name kernels_sm120, or a renamed
    dependency on kernels-sm120, only in an sm120 module (a directory or file named sm120) or in a
    file its KERNEL_IMPORTS lists. A listed file that no longer names it, or that moved into an
    sm120 module, must leave the list, and a list names only paths of KERNEL_IMPORT_BASELINE."""
    import hashlib
    from pathlib import Path

    root = Path(root)
    baseline_path = root / KERNEL_IMPORT_BASELINE
    baseline = {line.strip() for line in baseline_path.read_text().splitlines()
                if line.strip() and not line.startswith('#')} if baseline_path.exists() else set()
    models = [directory for top in ('models', 'examples') if (root / top).is_dir()
              for directory in sorted((root / top).iterdir()) if directory.is_dir()]
    errors = []
    for model in models:
        names = kernel_crate_names(model.rglob('Cargo.toml'))
        list_path = model / KERNEL_IMPORTS
        shown = list_path.relative_to(root)
        listed = set()
        if list_path.exists():
            listed = {line.strip() for line in list_path.read_text().splitlines()
                      if line.strip() and not line.startswith('#')}
        importing = set()
        for path in model.rglob('*.rs'):
            relative = path.relative_to(model)
            if 'target' not in relative.parts and names & set(rust_tokens(path.read_text(errors='replace'))):
                importing.add(relative)
        for path in sorted(importing):
            in_module = 'sm120' in path.parts[:-1] or path.name == 'sm120.rs'
            if in_module and path.as_posix() in listed:
                errors.append(f'{model.relative_to(root) / path}: is in an sm120 module; remove it from {shown}')
            elif not in_module and path.as_posix() not in listed:
                errors.append(f'{model.relative_to(root) / path}: model code names kernels_sm120 outside an '
                              'sm120 module; put SM120 selection in the model\'s sm120 module and reach '
                              'kernels through kernel_core::backend')
        for name in sorted(listed - {path.as_posix() for path in importing}):
            errors.append(f'{model.relative_to(root) / name}: no longer names kernels_sm120; remove it from {shown}')
        for name in sorted(listed):
            path = (model.relative_to(root) / name).as_posix()
            if hashlib.sha256(path.encode()).hexdigest() not in baseline:
                errors.append(f'{path}: is not in {KERNEL_IMPORT_BASELINE}; the lists only shrink')
    if errors:
        raise SystemExit('\n'.join(errors))


def sm120_names(model):
    """A model's or example's directory's "<file> <item>" lines that name SM120 items outside
    sm120 modules and test code, as (re-exports, uses). The check is lexical: it catches
    accidental coupling, not code written to hide it.

    Re-exports: each `pub use` leaf (any visibility but `pub(self)`) through an `sm120` module or
    kernels_sm120 (or a dependency renamed from kernels-sm120, also through the workspace); each
    glob re-export of `crate`, `super` or `self` (it can carry an sm120 module along unnamed);
    each module whose `#[path]` or `#[cfg_attr(.., path = ..)]` points into an sm120 directory or
    file; and each include!, include_str! or include_bytes! of such a path. Uses: the other paths
    through an `sm120` module, private and `pub(self)` `use` leaves and paths in code (an
    `ident::ident` chain around `sm120`); KERNEL_IMPORTS covers kernels_sm120 itself. Declaring
    `mod sm120`, or an item named sm120, names nothing.

    Test code: `#[cfg(test)]` items, test_files, and files under a crate's tests or benches
    directory. Not seen: paths built by concat! of pieces that do not themselves name sm120,
    macros that expand to such paths, aliases of an sm120 module (the alias itself is a line),
    globs of other modules that re-export one, `#[cfg(any(test, ..))]` (production), and
    `#![cfg(test)]` files (production unless test code declares them). Known gaps:
    - a module with several cfg_attr paths keeps one, and a production file a test module also
      declares by path can count as test code;
    - children of a `#[path]` module outside mod.rs are looked for at parent/child.rs;
    - helpers that tests/ or benches/ files declare by `#[path]` are not exempt;
    - a `#[cfg(test)]` item ends at its first semicolon, also one inside an array type;
    - path attributes on inline modules are ignored;
    - a production include! of a test-declared file keeps it test code;
    - `include! { .. }` with braces is not seen;
    - nested block comments are not handled."""
    import re

    names = {'sm120'} | kernel_crate_names(path for path in model.rglob('Cargo.toml')
                                            if 'target' not in path.relative_to(model).parts)
    sources = {path: rust_tokens(path.read_text(errors='replace'), strings=True)
               for path in model.rglob('*.rs') if 'target' not in path.relative_to(model).parts}
    production, tests = production_tokens(sources)

    def sm120_path(literal):
        return bool({'sm120', 'sm120.rs'} & set(re.split(r'[/\\]', string_value(literal))))

    identifier = re.compile(r'[A-Za-z_][A-Za-z_0-9]*').fullmatch
    reexports = set()
    uses = set()
    for path, values in sorted(production.items()):
        relative = path.relative_to(model)
        if ('sm120' in relative.parts[:-1] or relative.name == 'sm120.rs' or path in tests
                or any(part in ('tests', 'benches') and (model.joinpath(*relative.parts[:i]) / 'Cargo.toml').exists()
                       for i, part in enumerate(relative.parts[:-1]))):
            continue
        for _, _, test, literal in module_declarations(path, sources[path]):
            if not test and literal is not None and sm120_path(literal):
                reexports.add(f'{relative.as_posix()} #[path = "{string_value(literal)}"]')
        consumed = set()
        for start, value in enumerate(values):
            if value in ('include', 'include_str', 'include_bytes') and values[start + 1:start + 3] == ['!', '(']:
                end = start + 3
                depth = 1
                while end < len(values) and depth:
                    depth += (values[end] == '(') - (values[end] == ')')
                    if is_string(values[end]) and sm120_path(values[end]):
                        reexports.add(f'{relative.as_posix()} {value}!("{string_value(values[end])}")')
                    end += 1
            if value != 'use':
                continue
            visibility = start - 1
            if values[visibility:start] == [')']:
                depth = 0
                while visibility >= 0:
                    depth += (values[visibility] == ')') - (values[visibility] == '(')
                    if not depth:
                        break
                    visibility -= 1
                visibility -= 1
            public = (visibility >= 0 and values[visibility] == 'pub'
                      and values[visibility + 1:visibility + 4] != ['(', 'self', ')'])
            end = start + 1
            while end < len(values) and values[end] != ';':
                end += 1
            consumed.update(range(start, end))
            for parts in use_paths(values, start + 1, end):
                line = f'{relative.as_posix()} {"::".join(parts)}'
                if public and (names & set(parts) or parts[-1:] == ['*']
                               and set(parts[:-1]) <= {'crate', 'super', 'self'}):
                    reexports.add(line)
                elif not public and 'sm120' in parts:
                    uses.add(line)
        for index, value in enumerate(values):
            if value != 'sm120' or index in consumed or values[index - 1:index] == ['mod']:
                continue
            first = last = index
            while first >= 2 and values[first - 1] == '::' and identifier(values[first - 2]):
                first -= 2
            while last + 2 < len(values) and values[last + 1] == '::' and identifier(values[last + 2]):
                last += 2
            if first == last:
                continue
            consumed.update(range(first, last + 1))
            uses.add(f'{relative.as_posix()} {"".join(values[first:last + 1])}')
    return reexports, uses


def check_frozen_lines(root, list_name, baseline_name, observe, found_as):
    """Each model's or example's (models/<m>, examples/<e>) observed lines are the lines its
    list_name lists: a new one fails, and a listed one that went must leave the list. The
    baseline holds exactly the SHA-256 of every listed "<model directory>/<line>": a line
    leaves the list and its hash the baseline together, so neither can return unreviewed."""
    import hashlib
    from pathlib import Path

    root = Path(root)
    baseline_path = root / baseline_name
    baseline = {line.strip() for line in baseline_path.read_text().splitlines()
                if line.strip() and not line.startswith('#')} if baseline_path.exists() else set()
    models = [directory for top in ('models', 'examples') if (root / top).is_dir()
              for directory in sorted((root / top).iterdir()) if directory.is_dir()]
    errors = []
    hashes = set()
    for model in models:
        list_path = model / list_name
        shown = list_path.relative_to(root)
        prefix = model.relative_to(root).as_posix()
        listed = set()
        if list_path.exists():
            listed = {line.strip() for line in list_path.read_text().splitlines()
                      if line.strip() and not line.startswith('#')}
        found = observe(model)
        for line in sorted(found - listed):
            path, item = line.split(' ', 1)
            errors.append(f'{prefix}/{path}: {found_as(item)} outside an sm120 module; keep SM120 items in the '
                          'model\'s sm120 modules and reach kernels through kernel_core::backend')
        for line in sorted(listed - found):
            path, _, item = line.partition(' ')
            errors.append(f'{prefix}/{path}: no longer names {item} there; remove it from {shown} '
                          f'and its hash from {baseline_name}')
        for line in sorted(listed):
            digest = hashlib.sha256(f'{prefix}/{line}'.encode()).hexdigest()
            hashes.add(digest)
            if digest not in baseline:
                errors.append(f'{prefix}/{line}: is not in {baseline_name}; the lists only shrink')
    for digest in sorted(baseline - hashes):
        errors.append(f'{baseline_name}: {digest} is no listed line\'s hash; remove it')
    if errors:
        raise SystemExit('\n'.join(errors))


def check_model_sm120_reexports(root):
    """No model code outside sm120 modules re-exports SM120 items beyond its SM120_REEXPORTS."""
    def found_as(item):
        if item.startswith('#[path'):
            return f'declares a module from {item}'
        if item.startswith('include'):
            return f'uses {item}'
        if item.endswith('*') and set(item.split('::')[:-1]) <= {'crate', 'super', 'self'}:
            return f're-exports {item}, which can carry an sm120 module along,'
        return f're-exports the SM120 item {item}'

    check_frozen_lines(root, SM120_REEXPORTS, SM120_REEXPORT_BASELINE, lambda model: sm120_names(model)[0], found_as)


def check_model_sm120_uses(root):
    """No model code outside sm120 modules names a path through an sm120 module beyond its SM120_USES."""
    check_frozen_lines(root, SM120_USES, SM120_USE_BASELINE, lambda model: sm120_names(model)[1],
                       lambda item: f'names the SM120 item {item}')


def check_projection_sources(root, crate_names=None):
    """The shared bundle runtime is target-neutral: it links kernels-sm120 only as a
    dev-dependency, and only its test code names kernels_sm120 (or a dependency renamed from
    kernels-sm120), to check the backend path against SM120 catalogs."""
    import tomllib
    from pathlib import Path

    src = Path(root) / 'crates/bundle-runtime/src'
    manifest = src.parent / 'Cargo.toml'
    if crate_names is None:
        crate_names = kernel_crate_names([manifest] if manifest.exists() else [])
    else:
        crate_names = set(crate_names) | {'kernels_sm120'}
    if manifest.exists():
        data = tomllib.loads(manifest.read_text())
        for table in [data, *data.get('target', {}).values()]:
            for kind in ('dependencies', 'build-dependencies'):
                for alias in table.get(kind, {}):
                    if alias.replace('-', '_') in crate_names:
                        raise SystemExit(f'{manifest}: shared runtime links kernels-sm120 in {kind}; '
                                         'only its tests may (dev-dependencies)')

    production, test_only = production_tokens({path: rust_tokens(path.read_text(), strings=True)
                                                for path in src.rglob('*.rs')})
    for path, values in production.items():
        if any(path == target or target in path.parents for target in test_only):
            continue
        for value in values:
            if value in crate_names:
                raise SystemExit(f'{path}: shared runtime production source names {value}; '
                                 'only test code may use kernels-sm120')


def check_bundle():
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    from . import inventory as model_inventory

    # Shared members. Model crates come from models/<m>/model.toml.
    SHARED_MEMBERS = {
        'bundle-recurrent-fixture', 'bundle-tp4-fixture', 'bundle-runtime', 'encoder-spec', 'frontend-format', 'projection-spec',
        'kernel-build', 'kernel-core', 'kernel-harness', 'kernel-lab', 'kernel-lab-families', 'kernels-sm120',
        'model-compiler-sdk', 'runtime-bundle', 'runtime-bundle-abi', 'runtime-bundle-sdk',
        'runtime-conformance', 'runtime-cuda', 'runtime-engine', 'runtime-eval',
        'runtime-frontend-contract', 'runtime-frontend', 'runtime-grammar', 'runtime-host',
        'runtime-kernel-contract', 'runtime-kernels', 'runtime-kv', 'runtime-media', 'runtime-model-api',
        'runtime-sampling', 'runtime-scheduler',
        'runtime-server', 'runtime-tokenizer',
        'runtime-weights', 'xtask', 'chopinfer-top',
    }
    FIXTURES = {'bundle-recurrent-fixture', 'bundle-tp4-fixture'}
    # Direct-loading CPU checks and benchmarks (tools/load-proto).
    DEVELOPMENT_TOOLS = {'qwen38-load-proto'}
    # Shared crates whose build script compiles cubins through kernel-build. Every
    # model bundle is one too; no other model crate may be.
    SHARED_CUDA_BUILD_OWNERS = {'kernel-lab', 'runtime-kernels'}
    CUDA_RUNTIME_OWNERS = {'kernel-harness', 'kernel-lab', 'runtime-host'}
    SHARED_TARGETS = {
        'bundle-recurrent-fixture': {('build-recurrent-fixture', 'bin'), ('bundle_recurrent_fixture', 'cdylib')},
        'bundle-tp4-fixture': {('build-tp4-fixture', 'bin'), ('bundle_tp4_fixture', 'cdylib')},
        'runtime-host': {('chopinfer', 'bin'), ('chopinfer-host', 'bin')},
        'chopinfer-top': {('chopinfer-top', 'bin')},
    }

    metadata = json.loads(subprocess.check_output([
        os.environ.get('CARGO', 'cargo'), 'metadata', '--locked', '--offline',
        '--format-version', '1', '--all-features',
    ]))
    try:
        models = model_inventory.load('.')
        model_inventory.check_metadata(models, metadata, '.')
    except model_inventory.InventoryError as error:
        raise SystemExit(f'model inventory: {error}')
    MODEL_PACKAGES = model_inventory.packages(models)
    BUNDLES = {name for name, package in MODEL_PACKAGES.items() if package.role == 'bundle'}
    EXPECTED_MEMBERS = SHARED_MEMBERS | set(MODEL_PACKAGES) | DEVELOPMENT_TOOLS
    CUDA_BUILD_OWNERS = SHARED_CUDA_BUILD_OWNERS | BUNDLES
    TARGETS = SHARED_TARGETS | {name: set(package.targets) for name, package in MODEL_PACKAGES.items()}

    packages = {package['id']: package for package in metadata['packages']}
    nodes = {node['id']: node for node in metadata['resolve']['nodes']}
    workspace = {packages[id]['name']: id for id in metadata['workspace_members']}
    if set(workspace) != EXPECTED_MEMBERS:
        raise SystemExit(f'workspace members differ: missing {sorted(EXPECTED_MEMBERS - set(workspace))}, extra {sorted(set(workspace) - EXPECTED_MEMBERS)}')

    graph = Graph(metadata)
    def closure(name):
        return {graph.packages[key]['name'] for key in graph.closure(workspace[name])}

    for name, expected in TARGETS.items():
        observed = {(target['name'], kind) for target in packages[workspace[name]]['targets']
                    for kind in target['kind'] if kind in ('bin', 'cdylib')}
        if observed != expected:
            raise SystemExit(f'{name} target inventory differs: {sorted(observed)}')
    owner = {}
    for id in metadata['workspace_members']:
        for target in packages[id]['targets']:
            if 'bin' in target['kind']:
                previous = owner.setdefault(target['name'], packages[id]['name'])
                if previous != packages[id]['name']:
                    raise SystemExit(f'binary {target["name"]} has two owners')

    for name in workspace:
        reached = closure(name)
        forbidden = {edge for edge in reached if edge.startswith(('bundle-', 'model-'))}
        if name.startswith('runtime-') and name != 'runtime-conformance' and forbidden:
            raise SystemExit(f'{name} links bundle or model crates: {sorted(forbidden)}')
        if name == 'runtime-conformance' and forbidden - FIXTURES:
            raise SystemExit(f'conformance links nonfixture bundles: {sorted(forbidden - FIXTURES)}')
        if name in FIXTURES:
            missing = {'runtime-bundle', 'runtime-bundle-abi'} - reached
            forbidden_runtime = reached & {'runtime-cuda', 'runtime-engine', 'runtime-host', 'runtime-server'}
            if missing or forbidden_runtime:
                raise SystemExit(f'{name} boundary differs: missing {sorted(missing)}, forbidden {sorted(forbidden_runtime)}')
        if name in BUNDLES | {'bundle-runtime'}:
            forbidden_runtime = reached & {'runtime-cuda', 'runtime-engine', 'runtime-host', 'runtime-server'}
            if forbidden_runtime:
                raise SystemExit(f'{name} links forbidden runtime crates: {sorted(forbidden_runtime)}')
        if name == 'chopinfer-top':
            # The dashboard is an HTTP client of serve: no workspace crate, so
            # its terminal dependencies never reach a serving binary.
            linked = (reached - {name}) & set(workspace)
            if linked:
                raise SystemExit(f'{name} links workspace crates: {sorted(linked)}')
        if name == 'bundle-runtime':
            # The shared bundle runtime is target- and model-neutral: no kernel set, model or bundle.
            model = {edge for edge in reached if edge.startswith(('bundle-', 'model-'))}
            if model:
                raise SystemExit(f'{name} links model or bundle crates: {sorted(model)}')
            if 'kernels-sm120' in reached:
                raise SystemExit(f'{name} links kernels-sm120 outside its tests')

    # Cargo metadata resolves workspace inheritance and package renames.
    kernel_names = {dependency['name'] for dependency in nodes[workspace['bundle-runtime']]['deps']
                    if packages[dependency['pkg']]['name'] == 'kernels-sm120'}
    check_projection_sources(Path.cwd(), kernel_names)
    check_model_kernel_imports(Path.cwd())
    check_model_sm120_reexports(Path.cwd())
    check_model_sm120_uses(Path.cwd())

    observed_cuda = {name for name in workspace if name != 'runtime-cuda' and 'runtime-cuda' in closure(name)}
    if observed_cuda != CUDA_RUNTIME_OWNERS | DEVELOPMENT_TOOLS:
        raise SystemExit(f'runtime-cuda dependents differ: expected {sorted(CUDA_RUNTIME_OWNERS | DEVELOPMENT_TOOLS)}, observed {sorted(observed_cuda)}')
    def builds_with_kernel_build(name):
        return any(dependency['pkg'] == workspace['kernel-build'] and
                   any(kind['kind'] == 'build' for kind in dependency['dep_kinds'])
                   for dependency in nodes[workspace[name]]['deps'])

    observed_build = {name for name in workspace if builds_with_kernel_build(name)}
    if observed_build != CUDA_BUILD_OWNERS:
        raise SystemExit(f'kernel-build users differ: expected {sorted(CUDA_BUILD_OWNERS)}, observed {sorted(observed_build)}')

    # Test roots include direct dev dependencies. Resolve only these default
    # builds; the architecture checks above intentionally inspect all features.
    contributors = ('runtime-kernel-contract', 'runtime-bundle', 'model-compiler-sdk', 'runtime-host',
                    *(name for name, package in MODEL_PACKAGES.items()
                      if package.role in {'compiler', 'contract'}))
    selected, _ = selected_graph(metadata, Path('Cargo.toml'), packages=contributors)
    selected.require_cuda_free(contributors)
    check_model_contributors(selected, models)

    # A host serves its package's own frontend export and links no chat
    # format: the declared format library reaches the host crates only
    # through their tests. Without dev edges, Cargo resolves the features a
    # production build selects.
    for name in ('runtime-frontend', 'runtime-host'):
        tree = subprocess.check_output([
            os.environ.get('CARGO', 'cargo'), 'tree', '--locked', '--offline', '--package', name,
            '--edges', 'normal,build', '--prefix', 'none', '--format', '{p}', '--color', 'never',
        ], text=True)
        if any(line.split(' ')[0] == 'frontend-format' for line in tree.splitlines()):
            raise SystemExit(f'{name} links frontend-format outside its tests')


def check_model_contributors(graph, models):
    bundles = {model.packages['bundle'].name for model in models}
    for model in models:
        for role in ('compiler', 'contract'):
            if role not in model.packages:
                continue
            name = model.packages[role].name
            package = graph.package_ids((name,))[0]
            linked = {graph.packages[key]['name'] for key in graph.closure(package)}
            forbidden = {edge for edge in linked if edge.startswith('bundle-') and edge != 'bundle-runtime'} | (
                linked & (bundles | {'runtime-host', 'runtime-server', 'runtime-cuda'}))
            if forbidden:
                raise SystemExit(f'{name} {role} links execution dependencies: {sorted(forbidden)}')


def check_neutral():
    import json
    import os
    import pathlib
    import subprocess

    root = pathlib.Path.cwd()
    metadata = json.loads(subprocess.check_output([
        os.environ.get('CARGO', 'cargo'), 'metadata', '--locked', '--offline',
        '--format-version', '1', '--all-features',
    ]))
    packages = {package['id']: package['name'] for package in metadata['packages']}
    nodes = {node['id']: node for node in metadata['resolve']['nodes']}
    workspace = {packages[id]: id for id in metadata['workspace_members']}
    fixtures = {'bundle-recurrent-fixture', 'bundle-tp4-fixture'}

    graph = Graph(metadata)
    def closure(name):
        return {graph.packages[key]['name'] for key in graph.closure(workspace[name])}

    for name in sorted(workspace):
        if not name.startswith('runtime-'):
            continue
        linked = {item for item in closure(name) if item.startswith(('model-', 'bundle-'))}
        allowed = fixtures if name == 'runtime-conformance' else set()
        if linked - allowed:
            raise SystemExit(f'{name} links model or bundle crates: {sorted(linked - allowed)}')

    from .neutral_sources import check_sources
    check_sources(root)

if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["bundle"]:
        check_bundle()
        print("bundle boundary passed")
    elif sys.argv[1:] == ["neutral"]:
        check_neutral()
        print("model-neutral shared crates passed")
    else:
        raise SystemExit("expected bundle or neutral")

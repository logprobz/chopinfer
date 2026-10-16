"""The pinned journals of one revision, read from a sealed capture and its
model's sources at that commit, without knowing how a model writes its pins.

A capture index names every journal the revision's tests checked, with its
body's SHA-256 under the helper seal. Each body's (line count, FNV-1a) is
its pin. The revision's own sources must hold exactly those pins: the
literal `(lines, 0x<fnv>)` tuples of every Rust file under the model's
bundle directory that calls `check_journal` must equal, as a multiset, the
pins of the captured journals. A literal pin the tests never reached, a
computed pin that is not a literal, or an extra captured body therefore
fails closed.

Every `check_journal` call site is found too, and its expected pin must be
a literal tuple or a plain place (a variable, field, index or deref, or a
tuple of those) that reads a pin from a table. An expression that computes a
pin (`(1, 0x... + 1)`, a call) is refused, as is a pin-like tuple
`(N, 0x...` that is not a whole literal, so a computed pin cannot escape the
literal multiset by never being reached. The checker may be imported but not
aliased or used as a function value. A `.pin` sidecar written by the
capture must agree with its body; it is never the authority.

A journal's fixture is its test and the device lifetime that recorded it
(the capture's auxiliary bindings), or its test alone for a journal checked
without a fake device (a text journal).
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tomllib
from common import arguments, balanced, body_lines, fnv, git, need, resolve, uncomment, unique
from provenance import digest, verify

# The pin of the empty journal: zero lines, FNV-1a of no bytes.
EMPTY_FNV = int(fnv(b''), 16)
PIN = r'\(\s*(\d[\d_]*)\s*,\s*0x([0-9a-fA-F_]+)\s*\)'
CAPTURE_SCHEMAS = ('chopinfer.journal-captures.v2', 'chopinfer.journal-captures.v3')
SIDECAR = re.compile(r'\((\d+), 0x([0-9a-f]{16})\) expected \((\d+), 0x([0-9a-f]{16})\)\n?')
JOURNAL = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.txt')
CALL = re.compile(r'(?<![.\w])check_journal\b')
PLACE = r'[&*]*\s*(?:self|[A-Za-z_]\w*)(?:\s*(?:\.\s*\w+|::\s*\w+|\[[^\[\]()]*\]))*'
PLACE_PIN = re.compile(rf'{PLACE}|\(\s*{PLACE}\s*,\s*{PLACE}\s*\)')
PIN_START = re.compile(r'\(\s*\d[\d_]*\s*,\s*0x[0-9a-fA-F_]+')
IMPORT = re.compile(r'\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?use\b')


def bundle_root(repo, commit, model):
    """The bundle directory of `model` at `commit`, from its model.toml."""
    need(re.fullmatch(r'[a-z0-9][a-z0-9_-]*', model or '') is not None, f'invalid model id {model!r}')
    inventory = tomllib.loads(git(repo, 'show', f'{commit}:models/{model}/model.toml').decode())
    package = inventory.get('bundle', {}).get('package')
    need(isinstance(package, str), f'models/{model}/model.toml declares no [bundle] package')
    root = f'models/{model}/bundle/'
    cargo = tomllib.loads(git(repo, 'show', f'{commit}:{root}Cargo.toml').decode())
    need(cargo.get('package', {}).get('name') == package, f'{root}Cargo.toml is not the bundle package {package}')
    return root


def source_pins(repo, commit, root):
    """The pin tuples of every source under `root` that calls check_journal,
    those sources' SHA-256, and every call site: (path, line) to its literal
    pin, or None for a pin read from a table. Every call site's expected pin
    must be a literal or a plain place (call_sites)."""
    paths = git(repo, 'ls-tree', '-r', '--name-only', commit, root).decode().splitlines()
    pins, sources, sites = collections.Counter(), {}, {}
    for path in paths:
        if not path.endswith('.rs'):
            continue
        text = git(repo, 'show', f'{commit}:{path}').decode()
        clean = uncomment(text)
        if re.search(r'\bcheck_journal\b', clean) is None:
            continue
        sources[path] = hashlib.sha256(text.encode()).hexdigest()
        literal_spans = []
        for match in re.finditer(PIN, clean):
            lines, value = int(match[1].replace('_', '')), int(match[2].replace('_', ''), 16)
            need(value < 2**64 and (lines > 0 or value == EMPTY_FNV),
                 f'{path}: invalid pin tuple ({lines}, {value:#x}); a journal of 0 lines is pinned ({EMPTY_FNV:#x})')
            pins[(lines, f'{value:016x}')] += 1
            literal_spans.append(match.start())
        for match in PIN_START.finditer(clean):
            need(match.start() in literal_spans,
                 f'{path}:{clean.count(chr(10), 0, match.start()) + 1}: a pin tuple is computed, not a literal')
        for line, expected in call_sites(path, clean):
            literal = re.fullmatch(PIN, expected)
            need(literal is not None or PLACE_PIN.fullmatch(expected) is not None,
                 f'{path}:{line}: check_journal computes its expected pin ({expected[:80]}); '
                 'pass a literal or a pin read from a table')
            if literal is not None:
                literal = [int(literal[1].replace('_', '')), f'{int(literal[2].replace("_", ""), 16):016x}']
            # A recorded call site is a line: two calls on one line could not
            # be told apart, so one could hide the other.
            need((path, line) not in sites, f'{path}:{line}: two check_journal calls on one line')
            sites[(path, line)] = literal
    need(pins, f'no pinned journal sources under {root}')
    return pins, sources, sites


def check_sites(sites, checked, pins, repeated=None):
    """Every captured journal was checked at a found call site, with that
    site's literal pin if it has one, and every call site was reached. The
    pin multiset alone cannot show this: two journals checked through one
    table read could stand for an unreached literal call with the same pin,
    and an unreached table read could hide a pin that no literal records.
    `checked` maps each journal to its (path, line); `pins` to [lines, fnv];
    `repeated` to the [test, path, line] of its other checks, by other tests
    with the same body (capture.merge_checks)."""
    repeated = repeated or {}
    need(isinstance(repeated, dict) and set(repeated) <= set(checked)
         and all(isinstance(made, list) and made and all(isinstance(c, list) and len(c) == 3 and isinstance(c[0], str)
                                                          and isinstance(c[1], str) and type(c[2]) is int for c in made)
                 for made in repeated.values()), 'invalid repeated checks')
    calls = [(name, site) for name, site in checked.items()]
    calls += [(name, (path, line)) for name, made in repeated.items() for _, path, line in made]
    unknown = sorted({site for _, site in calls if site not in sites})
    need(not unknown, f'journals were checked at no check_journal call site: {[f"{p}:{n}" for p, n in unknown[:5]]}')
    for name, site in calls:
        need(sites[site] is None or list(pins[name]) == sites[site], f'{name}: checked at {site[0]}:{site[1]} with another pin')
    reached = {site for _, site in calls}
    missing = sorted(site for site in sites if site not in reached)
    need(not missing, 'check_journal calls were never reached in the capture run, so what they check is not '
         f'established: {[f"{p}:{n}" for p, n in missing[:5]]}')


def call_sites(path, clean):
    """Each direct check_journal call in `clean` (uncommented source): its
    line and expected-pin argument. Its own definition and plain imports are
    skipped; an alias or any other use is refused."""
    for match in CALL.finditer(clean):
        line = clean.count('\n', 0, match.start()) + 1
        if re.search(r'\bfn\s+$', clean[max(0, match.start() - 40):match.start()]):
            continue
        statement = clean[max(clean.rfind(';', 0, match.start()), clean.rfind('}', 0, match.start())) + 1:match.start()]
        if IMPORT.match(statement):
            need(re.match(r'\s+as\b', clean[match.end():]) is None, f'{path}:{line}: check_journal is imported under an alias')
            continue
        need(re.match(r'\s*\(', clean[match.end():]) is not None,
             f'{path}:{line}: check_journal is used other than as a direct call')
        call, _ = balanced(clean, clean.index('(', match.end()))
        args = arguments(call[1:-1])
        need(len(args) == 3, f'{path}:{line}: check_journal call does not take (lines, expected, file)')
        yield line, ' '.join(args[1].split())


def read_regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as handle:
        need(stat.S_ISREG(os.fstat(handle.fileno()).st_mode), f'not a regular capture: {path}')
        return handle.read()


def read_index(path):
    value = json.loads(Path(path).read_text(), object_pairs_hook=unique)
    need(isinstance(value, dict) and value.get('schema') in CAPTURE_SCHEMAS,
         f'{path}: not a sealed capture index ({" or ".join(CAPTURE_SCHEMAS)})')
    return verify(value)


def journal_path(index, payload, name):
    """A journal's capture path: absolute in a v2 index, a file beside the
    index in v3."""
    value = payload['journals'][name]
    need(isinstance(value, str), 'capture index paths must be strings')
    if payload['schema'] == 'chopinfer.journal-captures.v2':
        need(Path(value).is_absolute(), 'v2 capture index paths must be absolute')
        return Path(value)
    need(value == name, 'v3 capture index journals name their own files')
    return Path(index).resolve().parent / name


def body_pin(path, name):
    """A captured body, its line count and FNV-1a. A `.pin` sidecar beside it
    must record that the run checked exactly this pin."""
    data = read_regular(path)
    lines = len(body_lines(data))
    value = fnv(data)
    sidecar = path.with_name(name + '.pin')
    if sidecar.exists():
        m = SIDECAR.fullmatch(read_regular(sidecar).decode())
        need(m is not None, f'{sidecar}: unreadable pin sidecar')
        need((int(m[1]), m[2]) == (int(m[3]), m[4]) == (lines, value),
             f'{name}: the run checked ({m[3]}, {m[4]}) but the body is ({lines}, {value})')
    return data, lines, value


def check_literals(captured, literals):
    need(captured == literals,
         'captured journals do not match the source pins exactly: '
         f'unreached {sorted((literals - captured).elements())[:5]}, unpinned {sorted((captured - literals).elements())[:5]}')


def read_site(path):
    """A `.site` sidecar: the source path and line of the check_journal call
    that checked the journal."""
    text = read_regular(path).decode()
    m = re.fullmatch(r'(\S+):(\d+)\n?', text)
    need(m is not None and not Path(m[1]).is_absolute() and '..' not in Path(m[1]).parts, f'{path}: unreadable call site')
    return [m[1], int(m[2])]


def dumped(out, repo, commit, root, repeated=None):
    """The journals a capture run dumped into `out`, each with the `.pin`
    sidecar of its check and the `.site` sidecar of its call:
    {name: [lines, fnv]}, the bodies and {name: [path, line]}. They must be
    exactly the revision's source pins, checked at its call sites, with
    every call site reached (`repeated`: capture.merge_checks)."""
    out = Path(out)
    names = sorted(p.name for p in out.glob('*.txt'))
    sidecars = sorted(p.name.removesuffix('.pin') for p in out.glob('*.txt.pin'))
    need(names == sidecars, f'dumped journals and pin sidecars differ: {sorted(set(names) ^ set(sidecars))[:5]}')
    pins, bodies, checked = {}, {}, {}
    for name in names:
        need(JOURNAL.fullmatch(name) is not None, f'unsafe journal name {name}')
        data, lines, value = body_pin(out / name, name)
        pins[name] = [lines, value]
        bodies[name] = body_lines(data)
        need((out / f'{name}.site').is_file(), f'{name}: no call site was recorded')
        checked[name] = read_site(out / f'{name}.site')
    literals, _, sites = source_pins(repo, commit, root)
    check_literals(collections.Counter(tuple(p) for p in pins.values()), literals)
    check_sites(sites, {name: tuple(site) for name, site in checked.items()}, pins, repeated)
    return pins, bodies, checked


def bindings(payload, root):
    """Each journal's (kind, test, host): kind is device, text or unbound."""
    from auxiliary import normalize
    path = Path(payload['auxiliary']['path'])
    if payload['schema'] != 'chopinfer.journal-captures.v2':
        need(str(path) == 'aux/manifest.json', 'v3 auxiliary manifest must be aux/manifest.json')
        path = root / path
    need(path.resolve() == root / 'aux/manifest.json' and digest(path) == payload['auxiliary']['sha256'],
         'auxiliary manifest mismatch')
    manifest = normalize(json.loads(path.read_text()))
    out = {}
    for name, binding in manifest['bindings'].items():
        out[name] = ('device', binding['test'], binding['host'])
    for name, record in manifest['text'].items():
        out[name] = ('text', record['test'], None)
    for name, record in manifest['unrecoverable'].items():
        out[name] = ('unbound', record.get('test'), None)
    return out


def snapshot(repo, revision, model, index):
    """The revision's pins, bodies and body origins from its sealed capture
    `index`. Invalid or incomplete inputs raise Invalid."""
    commit = resolve(repo, revision)
    payload = read_index(index)
    need(payload['revision'] == commit, 'capture index revision differs from requested commit')
    need(payload.get('model', model) == model, f'capture index is of model {payload.get("model")}, not {model}')
    root = bundle_root(repo, commit, model)
    literals, sources, sites = source_pins(repo, commit, root)
    names = payload['journals']
    need(isinstance(names, dict) and set(names) == set(payload['journal_sha256']),
         'capture index journals and digests differ')
    capture_root = Path(index).resolve().parent
    kinds = bindings(payload, capture_root)
    need(set(kinds) == set(names), 'auxiliary bindings do not cover exactly the captured journals')
    pins, bodies, origins = {}, {}, {}
    for name in sorted(names):
        need(JOURNAL.fullmatch(name) is not None, f'unsafe journal name {name}')
        path = journal_path(index, payload, name)
        data, lines, value = body_pin(path, name)
        sha = hashlib.sha256(data).hexdigest()
        need(sha == payload['journal_sha256'][name], f'{name}: body differs from helper output')
        if 'pins' in payload:
            need(payload['pins'].get(name) == [lines, value], f'{name}: body ({lines}, {value}) is not its sealed pin')
        kind, test, host = kinds[name]
        fixture = f'{test}#{host}' if kind == 'device' else f'{test}#{kind}' if kind == 'text' else f'{test}#unbound:{name}'
        pins[name] = dict(lines=lines, fnv1a=value, fixture=fixture, kind=kind, test=test)
        bodies[name] = body_lines(data)
        origins[name] = dict(origin=str(path), sha256=sha, authenticated_origins=[str(path)])
    if 'pins' in payload:
        need(set(payload['pins']) == set(pins), 'sealed pins and captured journals differ')
    check_literals(collections.Counter((p['lines'], p['fnv1a']) for p in pins.values()), literals)
    if payload['schema'] == 'chopinfer.journal-captures.v2':
        coverage = ('not recorded: a v2 capture names no call sites; its helper authenticated the bodies '
                    'against a source manifest that covered every check_journal call')
    else:
        recorded = payload.get('sites')
        need(isinstance(recorded, dict) and set(recorded) == set(pins), 'v3 capture index does not name every call site')
        check_sites(sites, {name: tuple(site) for name, site in recorded.items()},
                    {name: [pin['lines'], pin['fnv1a']] for name, pin in pins.items()}, payload.get('repeated_checks'))
        coverage = 'every call site (literal or table read) checked a captured journal'
        if payload.get('repeated_checks'):
            coverage += (f'; {len(payload["repeated_checks"])} journals were also checked by other tests, '
                         'each with the same body and pin')
    return dict(commit=commit, model=model, root=root, pins=pins, sources=sources, call_sites=coverage), bodies, origins

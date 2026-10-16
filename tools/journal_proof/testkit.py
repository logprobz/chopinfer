"""Synthetic repositories and sealed captures for the journal proof tests.

A GitFixture commits a toy model (model.toml, bundle Cargo.toml and a
source file whose check_journal calls pin the given journal bodies). A
capture directory holds the bodies, aux/ device traces and checkpoints, and
an index sealed under the test's private authority, as the capture helper
writes them. Sealing never creates that authority's key: each test patches
provenance.AUTHORITY to a temporary directory and calls provenance.init().
"""
import json
import os
from pathlib import Path
import subprocess

from auxiliary import collect
from common import fnv
from provenance import digest, helper_identity, seal

TOOL = Path(__file__).resolve().parent
MODEL = 'toy'
ROOT = f'models/{MODEL}/bundle/'
GEN = 1 << 32
CREATE = 'toy-create.txt'
STEP = 'toy-step.txt'
SCHEMA = 'chopinfer.journal-proof.spec.v1'
CONFIG = ('LaunchConfigV1 { flags: 1, grid_x: {grid}, grid_y: 1, grid_z: 1, block_x: 128, block_y: 1, block_z: 1, '
          'shared_memory_bytes: 0, cluster: LaunchClusterV1 { x: 1, y: 1, z: 1 } }')


def span(raw, offset=0, length=64):
    return f'DeviceBuffer(DeviceBufferSpanV1 {{ buffer: BundleDeviceBufferHandleV1 {{ raw: {raw} }}, offset: {offset}, length: {length} }})'


def launch(symbol, raw=GEN + 4, live=128, grid=32):
    args = [span(raw), span(GEN + 1_000_000), span(GEN + 1_000_001), span(GEN + 5000, live, 4), span(GEN + 6003), 'ScalarU32(1)']
    return f'launch 1 {symbol} {CONFIG.replace("{grid}", str(grid))} [' + ', '.join(args) + ']'


def fixtures():
    """A create and a step journal of one device, and the spec of a module
    and resolve added in the head with one launch retargeted."""
    base = {CREATE: ['stream 1', 'module 2 cubin_old', 'resolve 3 chop_old', f'allocate 64 256 -> raw: {GEN + 4}', 'state 32'],
            STEP: [launch('chop_old'), launch('chop_keep_beta'), launch('chop_keep_copy'), launch('chop_consumer')]}
    head = {CREATE: base[CREATE][:3] + ['module 4 cubin_new', 'resolve 5 chop_new', f'allocate 64 256 -> raw: {GEN + 6}', 'state 32'],
            STEP: [launch('chop_new', GEN + 6, grid=8), launch('chop_keep_beta', GEN + 6), launch('chop_keep_copy', GEN + 6),
                   launch('chop_consumer', GEN + 6)]}
    spec = dict(schema=SCHEMA, fixtures=[dict(name='host', create=CREATE, journals=[CREATE, STEP])], journals={
        CREATE: dict(added_modules=[dict(symbol='cubin_new', ids=[4], count=1, positions=[4])],
                     added_resolves=[dict(symbol='chop_new', ids=[5], count=1, positions=[5])]),
        STEP: dict(launch_rules=[dict(from_symbol='chop_old', to_symbol='chop_new', count=1,
                                      config={'grid_x': {'ceil_div': [{'config': 'grid_x'}, 4]}})])})
    return base, head, spec


class GitFixture:
    def __init__(self, path):
        self.repo = path
        self.run('init', '--bare', str(path))

    def run(self, *args, input=None):
        env = {**os.environ, 'GIT_AUTHOR_NAME': 'Journal fixture', 'GIT_COMMITTER_NAME': 'Journal fixture',
               'GIT_AUTHOR_EMAIL': 'fixture@example.invalid', 'GIT_COMMITTER_EMAIL': 'fixture@example.invalid',
               'GIT_AUTHOR_DATE': '2000-01-01T00:00:00+00:00', 'GIT_COMMITTER_DATE': '2000-01-01T00:00:00+00:00'}
        result = subprocess.run(['git', *(['-C', str(self.repo)] if self.repo.exists() else []), *args], input=input,
                                capture_output=True, env=env, timeout=15, check=True)
        return result.stdout.decode().strip()

    def commit(self, journals, source_extra='', extra_files=None, model_toml=None):
        """A commit pinning every journal body of `journals` in one source
        file, plus `extra_files` (path to bytes)."""
        calls = []
        for name, lines in sorted(journals.items()):
            data = '\n'.join(lines).encode()
            calls.append(f'check_journal(&lines, ({len(lines)}, 0x{fnv(data)}), "{name}").unwrap();')
        paths = {
            f'models/{MODEL}/model.toml': (model_toml or f'[bundle]\npackage = "bundle-{MODEL}"\n').encode(),
            ROOT + 'Cargo.toml': f'[package]\nname = "bundle-{MODEL}"\n'.encode(),
            ROOT + 'src/fixture_tests.rs': ('fn fixture() {\n' + '\n'.join(calls) + '\n}\n' + source_extra).encode(),
        }
        paths.update(extra_files or {})
        tree = {}
        for path, data in paths.items():
            cursor = tree
            parts = path.split('/')
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = self.run('hash-object', '-w', '--stdin', input=data)

        def make(items):
            records = []
            for name, value in sorted(items.items()):
                kind = 'tree' if isinstance(value, dict) else 'blob'
                oid = make(value) if kind == 'tree' else value
                records.append(f'{"040000" if kind == "tree" else "100644"} {kind} {oid}\t{name}\n')
            return self.run('mktree', input=''.join(records).encode())
        return self.run('commit-tree', make(tree), input=b'Add synthetic journal fixtures\n')


def write_device(root, ident, events):
    """aux/device-IDENT.jsonl holding `events`."""
    (root / 'aux').mkdir(parents=True, exist_ok=True)
    (root / f'aux/device-{ident}.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))


def checkpoint(root, name, devices, test='fixture'):
    """The checkpoint of journal `name`: `devices` lists (id, events,
    records) of every device its test had created by then."""
    (root / 'aux').mkdir(parents=True, exist_ok=True)
    value = dict(schema='chopinfer.journal-trace.checkpoint.v1', journal=name, test=test,
                 devices=[dict(id=i, path=f'device-{i}.jsonl', records=records,
                               notes=sum(e[0] == 'note' for e in events[:records])) for i, events, records in devices])
    (root / 'aux' / f'{name}.checkpoint.json').write_text(json.dumps(value))


def seal_capture(root, revision, bodies):
    """Write the bodies, bind them (auxiliary.collect) and seal a v2 index
    over everything under `root` with the key the test created through
    provenance.init(). Returns the index path."""
    for name, lines in bodies.items():
        (root / name).write_text('\n'.join(lines))
    auxiliary = collect(root, dict(commit=revision, pins=set(bodies)), bodies)
    helper_digest, helper_files = helper_identity(TOOL)
    payload = dict(schema='chopinfer.journal-captures.v2', revision=revision, archive_commit=revision,
                   producer='capture_revision.py', real_build=True, helper_digest=helper_digest, helper_files=helper_files,
                   journals={n: str(root / n) for n in bodies}, journal_sha256={n: digest(root / n) for n in bodies},
                   auxiliary=auxiliary,
                   auxiliary_files={str(p.relative_to(root)): digest(p) for p in (root / 'aux').iterdir() if p.is_file()})
    (root / 'index.json').write_text(json.dumps(seal(payload)))
    return root / 'index.json'


def create_events(create, payloads=None):
    """Device events for a create journal: each `stream`, `module`,
    `resolve`, `allocate ... -> raw: R` and `event` line issues its ID before
    its note, as FakeDevice does, followed by the instance-create marker.
    `payloads` maps a module name to (fnv, bytes, payload file name)."""
    events, module = [], None
    for entry in create:
        parts = entry.split()
        if parts[0] == 'stream':
            events.extend([['id', int(parts[1])], ['stream', int(parts[1]), 0]])
        elif parts[0] == 'module':
            module = int(parts[1])
            name = parts[2]
            fingerprint, size, path = (payloads or {}).get(name, (f'{name}_hash', 100, None))
            events.extend([['id', module], ['module', module, name, fingerprint, size] + ([path] if path else [])])
        elif parts[0] == 'resolve':
            events.extend([['id', int(parts[1])], ['resolve', int(parts[1]), module, parts[2]]])
        elif parts[0] == 'allocate' and '->' in parts:
            raw = int(parts[-1])
            events.extend([['id', raw & 0xffffffff], ['allocation', raw, 0, int(parts[1]), int(parts[2]), GEN + 1]])
        elif parts[0] == 'event':
            events.append(['id', int(parts[1])])
        events.append(['note', entry])
    events.append(['created', 0, 0, len(events), len(create)])
    return events


def fixture_sites(journals):
    """Where GitFixture.commit checks each journal: one call per line of
    src/fixture_tests.rs, in name order, after its `fn fixture() {` line."""
    return {name: [ROOT + 'src/fixture_tests.rs', 2 + i] for i, name in enumerate(sorted(journals))}


def write_dump(out, bodies, sites=None, test='fixture'):
    """What an instrumented capture run leaves in `out` for `bodies` checked
    on one device of `test`: each body with its `.pin` and `.site` sidecars
    under checks/TEST/, and the device trace and a checkpoint per journal
    under aux/tests/TEST/ (TEST with `::` as `.`). The create journal is
    recorded first, then the others in order. Returns checks/TEST."""
    out = Path(out)
    sites = sites or fixture_sites(bodies)
    events = create_events(bodies[CREATE])
    ends = {CREATE: len(events)}
    for name, lines in bodies.items():
        if name != CREATE:
            events.extend(['note', line] for line in lines)
            ends[name] = len(events)
    directory = 'tests/' + test.replace('::', '.')
    (out / 'aux' / directory).mkdir(parents=True, exist_ok=True)
    (out / 'aux' / directory / 'device-1.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
    checks = out / 'checks' / test.replace('::', '.')
    checks.mkdir(parents=True, exist_ok=True)
    for name, lines in bodies.items():
        data = '\n'.join(lines).encode()
        (checks / name).write_bytes(data)
        pin = f'({len(lines)}, 0x{fnv(data)})'
        (checks / f'{name}.pin').write_text(f'{pin} expected {pin}\n')
        (checks / f'{name}.site').write_text(f'{sites[name][0]}:{sites[name][1]}\n')
        records = ends[name]
        value = dict(schema='chopinfer.journal-trace.checkpoint.v1', journal=name, test=test,
                     devices=[dict(id=1, path=f'{directory}/device-1.jsonl', records=records,
                                   notes=sum(e[0] == 'note' for e in events[:records]))])
        (out / 'aux' / directory / f'{name}.checkpoint.json').write_text(json.dumps(value))
    return checks

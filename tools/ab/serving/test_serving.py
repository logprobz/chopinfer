#!/usr/bin/env python3
"""The serving runner on CPU fixtures: stub GPU queries, a loopback stub
server and client, and dummy processes. Never invokes a GPU tool or a real
server."""
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import Refusal, digest, read, write  # noqa: E402
from evidence import (Comparator, acceptance, catalog, phases_from, parse_metrics,  # noqa: E402
                      TOTALS, POSITIONS)
from processes import Commands, Owned, Proc, Process, parse_stat  # noqa: E402
import client  # noqa: E402
import model  # noqa: E402
import run  # noqa: E402
import stubs  # noqa: E402

ROOT = None

# Commands.run ends a command at its deadline in one of two ways: it sees the
# deadline pass while the command still runs ("command timed out"), or the
# command exits on its own but only after the deadline ("command completed
# after its deadline"). Both refuse the run.
DEADLINE_REFUSALS = ('command timed out', 'command completed after its deadline')

# Runs run.py (argv[2:]) so that Commands.run notices an HTTP helper's exit
# only once the helper has exited by itself. The helper's own timeout equals
# the command's but starts after the command's deadline is fixed, so it ends
# just after that deadline: the second refusal above, deterministically.
LATE_HTTP_DRIVER = '''
import os, runpy, sys, time
sys.path.insert(0, sys.argv[1])
import processes
class LateHttp(processes.Owned):
    def spawn(self, argv, **kwargs):
        self.late = any(str(word).endswith('http_request.py') for word in argv)
        return super().spawn(argv, **kwargs)
    def exited(self):
        if getattr(self, 'late', False) and self.process is not None and not self.reaped:
            limit = time.monotonic() + 60
            while time.monotonic() < limit and os.waitid(
                    os.P_PID, self.process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                time.sleep(.01)
        return super().exited()
processes.Owned = LateHttp
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''


def setUpModule():
    global ROOT, _directory
    # Installs refuse group- and other-writable files, as the runner's own
    # main() ensures.
    os.umask(0o022)
    _directory = tempfile.TemporaryDirectory(prefix='ab-serving-test-')
    ROOT = Path(_directory.name)


def tearDownModule():
    _directory.cleanup()


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def models(root):
    """A stub model: one compiler bin, one bundle cdylib, one drafter."""
    directory = root / 'models' / 'stub'
    directory.mkdir(parents=True)
    (directory / 'model.toml').write_text('[bundle]\ncdylib = ["bundle_stub"]\n\n[compiler]\n'
                                          'bin = ["chopinfer-model-stub", "stub-tool"]\n')
    profiles = {name: dict(bundle_id='stub.v1', topology='stub-tp1', tensor_parallel=1, max_sequences=32,
                           kv_cache_dtype='fp8', prefix_cache=True, max_batch_tokens=8192, prefill_chunk_tokens=2048,
                           gpu_memory_utilization=0.9, model_options={'checkpoint.quantization_layout': 'stub-fp8'},
                           speculation=dict(drafter=drafter, tokens=tokens, mode='adaptive', draft_cost='per-token'))
                for name, drafter, tokens in [('tp1-ar', 'none', 0), ('tp1-spec', 'mtp', 3)]}
    write(directory / 'profiles.json', dict(schema='chopinfer-serve-profiles-v2', model='stub', profiles=profiles))
    return root / 'models'


def fixture(root, phases=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / 'checkpoint').mkdir()
    installs = {}
    for name, commit in [('base', 'a'), ('new', 'b')]:
        path = root / name
        (path / 'bin').mkdir(parents=True); (path / 'lib').mkdir()
        files = {'bin/chopinfer': stubs.__file__, 'bin/chopinfer-model-stub': stubs.__file__}
        for relative, source in files.items():
            (path / relative).write_bytes(Path(source).read_bytes()); (path / relative).chmod(0o755)
        (path / 'lib/libbundle_stub.so').write_text('CPU dummy module ' + name)
        hashes = {f: digest(path / f) for f in [*files, 'lib/libbundle_stub.so']}
        write(path / 'INSTALL.json', dict(commit=commit * 40, files=hashes))
        cache = root / (name + '-cache'); (cache / 'records').mkdir(parents=True)
        module = (path / 'lib/libbundle_stub.so').read_bytes()
        manifest = dict(id='stub.v1', native_module=dict(path='native/module.bin',
            sha256=hashes['lib/libbundle_stub.so'], byte_length=len(module)),
            model_options=dict(options=[dict(name='checkpoint.quantization_layout',
                definition=dict(allowed_values=['stub-fp8', 'stub-fp4']))]),
            target=dict(topologies=[dict(id='stub-tp1', rank_count=1)]))
        temp = cache / 'manifest.json'; write(temp, manifest); identity = digest(temp)
        package = cache / ('p-' + identity); (package / 'native').mkdir(parents=True)
        temp.rename(package / 'manifest.json'); (package / 'native/module.bin').write_bytes(module)
        entry = dict(kind='package', state='complete', dependencies=[], files=[], last_selected=0, payload=dict(checkpoint='stub-checkpoint'))
        checksum = hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        write(cache / 'records' / (package.name + '.json'), dict(version=1, checksum=checksum, value=entry))
        installs[name] = dict(path=str(path), cache=str(cache))
    c = dict(schema=run.SCHEMA, model='stub', models=str(models(root)), checkpoint=str(root / 'checkpoint'),
        checkpoint_identity='stub-checkpoint', installs=installs, port=free_port(), gpus=[0],
        variants=dict(ar=dict(mode='off', tokens=0), spec=dict(mode='fixed', tokens=3)),
        phases=phases or ['prefill-3k', 'teacher64', 'warm', 'decode-c1'],
        query_command=[sys.executable, '-B', str(HERE / 'stubs.py'), 'query'], client=str(HERE / 'stubs.py'),
        env=dict(STUB_STARTS=str(root / 'starts'), STUB_QUERIES=str(root / 'queries'), STUB_CHILDREN=str(root / 'children')),
        budget_seconds=1740, telemetry_interval=0.2, phase_timeout=120,
        # Deadlines far above any scheduling delay on a loaded shared host;
        # a test of a deadline sets its own short one.
        timeouts=dict(query=30, admission=120, startup=120, http=30, term=5, kill=30, release=120, cleanup_reserve=300))
    if any(p.startswith('gsm8k') for p in c['phases']):
        examples = root / 'gsm8k-examples.json'
        examples.write_text(json.dumps([dict(question=f'What is {i} + {i}?') for i in range(8)]))
        write(root / 'gsm8k-manifest.json', dict(examples_sha256=digest(examples), instruction='Solve.'))
        c['client_assets'] = {name: str(root / name) for name in ['gsm8k-examples.json', 'gsm8k-manifest.json']}
    write(root / 'run.json', c)
    run.config(root / 'run.json')
    return c


class Base(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / self.id().split('.')[-1]
        self.root.mkdir()
    def config(self, phases=None):
        return fixture(self.root / 'fixture', phases)
    def load(self, c):
        path = self.root / 'run.json'; write(path, c)
        return run.config(path)
    def execute(self, c, dry=False, driver=None):
        path = self.root / 'run.json'; write(path, c)
        argv = [sys.executable, '-B', *(['-c', driver, str(HERE)] if driver else []), str(HERE / 'run.py'), str(path)]
        argv += ['--dry-run'] if dry else ['--out', str(self.root / 'output')]
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        (self.root / 'driver.log').write_bytes(result.stdout)
        return result
    def assert_dead(self, pid):
        current = Proc().get(pid)
        self.assertTrue(current is None or not current.live, f'CPU fixture survived: {pid}')
    def assert_no_children(self, c):
        for file in ['STUB_STARTS', 'STUB_CHILDREN']:
            path = Path(c['env'][file])
            if path.exists():
                for pid in path.read_text().splitlines():
                    self.assert_dead(int(pid))


class ConfigurationTests(Base):
    def test_abba_has_ar_control_and_two_repeats(self):
        c, _, _ = self.load(self.config()); arms = run.plan(c)
        self.assertEqual([x['install'] for x in arms], ['base', 'new', 'new', 'base'] * 2)
        self.assertEqual(len(arms), 8)
    def test_bad_order_refused(self):
        c = self.config(); c['order'] = ['base', 'new']
        with self.assertRaises(Refusal): self.load(c)
    def test_cache_bookkeeping_mutates_but_identity_does_not(self):
        c = self.config(); path = next(Path(c['installs']['base']['cache']).glob('records/p-*.json'))
        pinned = run.cache_record(path); record = read(path)
        record['value']['last_selected'] = 100
        record['checksum'] = hashlib.sha256(json.dumps(record['value'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        write(path, record); self.assertEqual(run.cache_record(path), pinned)
        record['value']['payload']['checkpoint'] = 'different'; write(path, record)
        with self.assertRaisesRegex(Refusal, 'checksum'): run.cache_record(path)
    def test_install_snapshot_survives_source_replacement(self):
        c, phases, preflight = self.load(self.config())
        runner = run.Runner(c, phases, preflight, self.root/'output')
        original = Path(c['installs']['base']['path'])/'bin/chopinfer'; original.write_text('replaced')
        command = runner.serve_command(run.plan(c)[0])
        self.assertNotEqual(Path(command[0]), original)
        self.assertEqual(digest(command[0]), preflight['installs']['base']['files']['bin/chopinfer'])
    def test_speculation_without_ar_refused(self):
        c = self.config(); c['variants'] = dict(spec=dict(mode='fixed', tokens=3))
        with self.assertRaises(Refusal): self.load(c)
    def test_a_smoke_run_is_one_ab_pair_of_one_variant(self):
        c = self.config(['determinism', 'warm', 'decode-c1']); c['smoke'] = True
        with self.assertRaisesRegex(Refusal, 'one variant'): self.load(c)
        c['variants'] = dict(spec=dict(mode='fixed', tokens=3))
        loaded, _, _ = self.load(c)
        self.assertEqual([a['name'] for a in run.plan(loaded)], ['spec-1-base-1', 'spec-2-new-1'])
        c['order'] = ['base', 'new', 'new', 'base']
        with self.assertRaisesRegex(Refusal, 'smoke order'): self.load(c)
        c.update(smoke=False, order=['base', 'new'], variants=dict(ar=dict(mode='off', tokens=0)))
        with self.assertRaisesRegex(Refusal, 'ABBA'): self.load(c)
    def test_a_smoke_run_is_bitwise(self):
        c = self.config(['determinism', 'warm', 'decode-c1'])
        c.update(smoke=True, variants=dict(ar=dict(mode='off', tokens=0)), order=['base', 'new'], base_new='report-only')
        with self.assertRaisesRegex(Refusal, 'a smoke run gates base against new bitwise'): self.load(c)
        c.update(smoke=False, order=['base', 'new', 'new', 'base']); self.load(c)
    def test_a_smoke_run_needs_the_smoke_phases(self):
        c = self.config(['warm', 'decode-c1', 'decode-c8'])
        c.update(smoke=True, variants=dict(ar=dict(mode='off', tokens=0)), order=['base', 'new'])
        with self.assertRaisesRegex(Refusal, 'a smoke run needs the phases'): self.load(c)
        c['phases'] = ['determinism', 'warm', 'decode-c1', 'decode-c8']
        self.assertEqual(self.load(c)[0]['phases'], c['phases'])
    def test_a_determinism_phase_needs_its_tools(self):
        c = self.config(['determinism', 'warm', 'decode-c1']); c['qual_tools'] = str(self.root / 'missing')
        with self.assertRaises((Refusal, OSError)): self.load(c)
    def test_drafter_must_be_declared(self):
        c = self.config(); c['variants']['spec']['drafter'] = 'dflash'
        with self.assertRaisesRegex(Refusal, 'no drafter'): self.load(c)
        c['variants']['spec']['drafter'] = 'mtp'; self.load(c)
        c['variants']['ar']['drafter'] = 'mtp'
        with self.assertRaisesRegex(Refusal, 'variant name'): self.load(c)
    def test_serve_profile_options_are_passed_explicitly_to_both_installs(self):
        c = self.config(); c['serve_profile'] = 'tp9-missing'
        with self.assertRaisesRegex(Refusal, 'no serve profile'): self.load(c)
        c['serve_profile'] = 'tp1-spec'; c.pop('serve_args', None)
        loaded, phases, preflight = self.load(copy.deepcopy(c))
        self.assertEqual(loaded['serve_args'], [
            '--tensor-parallel-size', '1', '--max-sequences', '32', '--max-batch-tokens', '8192',
            '--prefill-chunk-tokens', '2048', '--gpu-memory-utilization', '0.9', '--kv-cache-dtype', 'fp8',
            '--prefix-cache', 'on', '--draft-cost', 'per-token',
            '--model-option', 'checkpoint.quantization_layout=stub-fp8'])
        self.assertEqual(loaded['expect'], dict(
            profile=dict(tensor_parallel_size=1, max_sequences=32, prefix_cache=True, kv_cache_dtype='fp8'),
            model_options={'checkpoint.quantization_layout': 'stub-fp8'}))
        runner = run.Runner(loaded, phases, preflight, self.root / 'output')
        commands = [runner.serve_command(arm) for arm in run.plan(loaded)]
        # serve --model refuses --profile.
        self.assertTrue(all('--profile' not in command and command[-1] == 'fixed'
                            for command in commands if '--speculation' in command))
        c['serve_args'] = ['--kv-cache-dtype', 'bf16']
        with self.assertRaisesRegex(Refusal, 'repeat options'): self.load(copy.deepcopy(c))
        c['serve_args'] = ['--queue-depth', '4']
        self.assertEqual(self.load(copy.deepcopy(c))[0]['serve_args'][-2:], ['--queue-depth', '4'])
    def test_a_profile_no_prepared_package_serves_is_refused(self):
        c = self.config(); path = Path(c['models']) / 'stub/profiles.json'; original = read(path)
        c['serve_profile'] = 'tp1-ar'; c['variants'] = dict(ar=dict(mode='off', tokens=0))
        for change in [dict(topology='stub-tp2'), dict(context_parallel_size=2),
                       dict(model_options={'checkpoint.quantization_layout': 'stub-bf16'}),
                       dict(model_options={'runtime.undeclared': True})]:
            with self.subTest(change=change):
                profiles = copy.deepcopy(original); profiles['profiles']['tp1-ar'].update(change); write(path, profiles)
                with self.assertRaisesRegex(Refusal, 'serves profile tp1-ar'): self.load(copy.deepcopy(c))
        write(path, original); c['gpus'] = [0, 1]
        with self.assertRaisesRegex(Refusal, 'serves 1 ranks'): self.load(c)
    def test_an_install_is_bound_to_its_requested_commit(self):
        c = self.config(); c['installs']['new']['commit'] = 'b' * 40; self.load(c)
        c['installs']['new']['commit'] = 'c' * 40
        with self.assertRaisesRegex(Refusal, 'install new is built from b+, not the requested c+'): self.load(c)
    def test_install_mutation_refused(self):
        c = self.config(); (Path(c['installs']['new']['path']) / 'lib/libbundle_stub.so').write_text('changed')
        with self.assertRaises(Refusal): self.load(c)
    def test_install_extra_file_refused(self):
        c = self.config(); (Path(c['installs']['new']['path']) / 'bin/extra').write_text('extra')
        with self.assertRaises(Refusal): self.load(c)
    def test_install_symlink_refused(self):
        c = self.config(); p = Path(c['installs']['new']['path']) / 'bin/chopinfer'
        p.unlink(); p.symlink_to(HERE / 'stubs.py')
        with self.assertRaises(Refusal): self.load(c)
    def test_package_native_mismatch_refused(self):
        c = self.config(); p = next(Path(c['installs']['new']['cache']).glob('p-*/native/module.bin'))
        p.write_text('wrong served bundle')
        with self.assertRaises(Refusal): self.load(c)
    def test_reserved_flags_refused(self):
        for value in ['--listen=0.0.0.0:1', '--speculative-tokens=0', '--bundle=foreign', '--profile=x#y']:
            with self.subTest(value=value), self.assertRaises(Refusal): run.serve_args([value])
    def test_dry_run_never_queries_or_spawns(self):
        c = self.config(); result = self.execute(c, True)
        self.assertEqual(result.returncode, 0, result.stdout.decode())
        data = json.loads(result.stdout); self.assertEqual(data['gpu_queries'], 0)
        self.assertFalse(Path(c['env']['STUB_STARTS']).exists()); self.assertFalse(Path(c['env']['STUB_QUERIES']).exists())
    def test_port_check_refuses_listener(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); sock.listen()
            with self.assertRaises(OSError): run.port_free(sock.getsockname()[1])


class EvidenceTests(Base):
    def setUp(self):
        super().setUp(); self.cmp = Comparator()
    def test_all_catalog_phases_validate(self):
        for name, declaration in catalog().items():
            if declaration['kind'] == 'determinism':
                continue  # its receipt: DriverTests and ClientTests
            with self.subTest(name=name): self.cmp.validate_phase(stubs.phase(name), declaration)
    def test_c32_decode_runs_only_when_named(self):
        for value in ['standard', 'short']:
            with self.subTest(value=value):
                self.assertNotIn('decode-c32', [p['name'] for p in phases_from(value)])
        self.assertEqual([p['name'] for p in phases_from(['warm', 'decode-c32'])], ['warm', 'decode-c32'])
        self.assertEqual(catalog()['decode-c32']['spec']['requests'], 32)
    def test_missing_phase_refused(self):
        write(self.root / 'capture.json', dict(model='stub', phases=[]))
        with self.assertRaises(Refusal): self.cmp.capture(self.root, phases_from(['prefill-3k']))
    def test_short_prompts_refused(self):
        for name in ['prefill-32k', 'prefill-9k', 'prefill-3k', 'teacher64', 'teacher1040', 'teacher2060']:
            with self.subTest(name=name):
                p = stubs.phase(name); p['prompt_tokens'] = 1
                with self.assertRaises(Refusal): self.cmp.validate_phase(p, catalog()[name])
    def test_nonfinite_refused(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            p = stubs.phase('prefill-3k'); p['outputs'][0]['logprobs']['token_logprobs'][0] = value
            with self.subTest(value=value), self.assertRaises(Refusal): self.cmp.validate_phase(p, catalog()['prefill-3k'])
    def test_teacher_generated_ids_are_not_prompt_ids(self):
        p = stubs.phase('teacher64'); p['outputs'][0]['prompt_token_ids'] = [1]
        with self.assertRaises(Refusal): self.cmp.validate_phase(p, catalog()['teacher64'])
    def test_teacher_missing_score_map_refused(self):
        p = stubs.phase('teacher64'); p['outputs'][0]['prompt_logprobs'].pop()
        with self.assertRaises(Refusal): self.cmp.validate_phase(p, catalog()['teacher64'])
    def test_gsm_empty_answers_refused(self):
        p = stubs.phase('gsm8k-c1'); p['outputs'] = []
        with self.assertRaises(Refusal): self.cmp.validate_phase(p, catalog()['gsm8k-c1'])
    def compare(self, a, b, required=True):
        phases = phases_from(['prefill-3k'])
        for name, record in [('a', a), ('b', b)]:
            p = self.root / name; p.mkdir(); write(p / 'capture.json', dict(model='stub', phases=[record]))
        return self.cmp.compare(self.root / 'a', self.root / 'b', phases, self.root / 'cmp.json', Commands(), 120, required)
    def test_signed_zero_difference_fails(self):
        a = stubs.phase('prefill-3k', value=0.0); b = stubs.phase('prefill-3k', value=-0.0)
        with self.assertRaises(Refusal): self.compare(a, b)
    def test_report_only_still_reports_difference(self):
        result = self.compare(stubs.phase('prefill-3k'), stubs.phase('prefill-3k', value=-.6), False)
        self.assertFalse(result['identical'])
    def test_every_batched_decode_comparison_is_gated(self):
        for name in ['decode-c4', 'decode-c8', 'decode-c16', 'decode-c32']:
            with self.subTest(phase=name):
                root = self.root/name; root.mkdir()
                phases = phases_from(['warm', name])
                for side in ['a', 'b']:
                    directory = root/side; directory.mkdir()
                    rows = [stubs.phase('warm'), stubs.phase(name)]
                    if side == 'b': rows[1]['outputs'][0]['logprobs'][0]['token_logprobs'][0] = -.6
                    write(directory/'capture.json', dict(model='stub', phases=rows))
                with self.assertRaisesRegex(Refusal, 'different gated'):
                    self.cmp.compare(root/'a', root/'b', phases, root/'cmp.json', Commands(), 120)
    def test_stale_report_refused(self):
        write(self.root / 'cmp.json', dict(valid=True, identical=True))
        with self.assertRaises(Refusal): self.compare(stubs.phase('prefill-3k'), stubs.phase('prefill-3k'))


class CounterTests(Base):
    def check(self, before, after, speculative=True, required=True):
        return acceptance(before, after, speculative=speculative, maximum=3 if speculative else 0, require_drafts=required)
    def test_initial_omitted_zeros_valid(self):
        self.assertEqual(self.check(stubs.metrics(0), stubs.metrics(2))['rate'], 1)
    def test_ar_with_k_zero_valid(self):
        self.assertEqual(self.check(stubs.metrics(0, False), stubs.metrics(2, False), False, False)['drafted'], 0)
    def test_empty_metrics_refused(self):
        with self.assertRaises(Refusal): self.check('', '')
    def test_zero_mtp_coverage_refused(self):
        with self.assertRaises(Refusal): self.check(stubs.metrics(0), stubs.metrics(0))
    def test_reset_refused(self):
        with self.assertRaises(Refusal): self.check(stubs.metrics(2), stubs.metrics(1))
    def test_partial_family_refused(self):
        text = stubs.metrics(2).replace('chop_spec_rows_total 8\n', '')
        with self.assertRaises(Refusal): self.check(stubs.metrics(0), text)
    def test_counter_nonfinite_fractional_duplicate_refused(self):
        for value in ['NaN', '+Inf', '0.5', '-1', str(2**64)]:
            with self.subTest(value=value), self.assertRaises(Refusal):
                self.check(stubs.metrics(0), stubs.metrics(2).replace('chop_spec_rows_total 8', 'chop_spec_rows_total ' + value))
        with self.assertRaises(Refusal): parse_metrics(stubs.metrics(2) + 'chop_spec_rows_total 8\n')
    def test_position_and_histogram_invariants(self):
        mutations = [
            ('chop_spec_drafted_position_total{position="1"} 2', 'chop_spec_drafted_position_total{position="1"} 1'),
            ('chop_spec_accepted_position_total{position="3"} 2', 'chop_spec_accepted_position_total{position="3"} 3'),
            ('k="3"} 2', 'k="3"} 1'), ('chop_spec_emitted_tokens_total 8', 'chop_spec_emitted_tokens_total 9'),
            ('chop_spec_rows_total 8', 'chop_spec_rows_total 7'),
            ('chop_spec_drafted_position_total{position="16"} 0\n', '')]
        for old, new in mutations:
            with self.subTest(old=old), self.assertRaises(Refusal): self.check(stubs.metrics(0), stubs.metrics(2).replace(old, new))


class OwnershipTests(Base):
    def owned(self, code=None, proc=None):
        owner = Owned(self.root / 'owner.json', proc=proc)
        self.addCleanup(self.cleanup, owner)
        with (self.root / 'process.log').open('wb') as output:
            owner.spawn([sys.executable, '-c', code or 'import time; time.sleep(30)'], stdout=output)
        return owner
    def cleanup(self, owner):
        owner.proc = Proc()
        if owner.process is not None and not owner.reaped:
            owner.terminate(.05, 30); owner.reap()
    def test_owned_before_readiness(self):
        owner = self.owned(); pid = owner.process.pid
        self.assertEqual(read(owner.record)['sid'], pid)
        owner.terminate(.05, 30); owner.reap(); self.assert_dead(pid)
    def test_term_resistant_gets_kill(self):
        owner = self.owned('import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print("ready",flush=True); time.sleep(30)')
        deadline = time.monotonic() + 120
        while not (self.root / 'process.log').read_text():
            self.assertLess(time.monotonic(), deadline); time.sleep(.01)
        # Killed after the 0.1 s TERM grace, long before its 30 s sleep ends.
        start = time.monotonic(); owner.terminate(.1, 30); owner.reap()
        self.assertLess(time.monotonic() - start, 15); self.assertEqual(owner.process.returncode, -signal.SIGKILL)
    def test_initial_identity_failure_retains_owner(self):
        class Broken(Proc):
            def get(self, pid): raise Refusal('injected discovery failure')
        owner = Owned(self.root / 'owner.json', proc=Broken()); self.addCleanup(self.cleanup, owner)
        with (self.root / 'process.log').open('wb') as output, self.assertRaises(Refusal):
            owner.spawn([sys.executable, '-c', 'import time; time.sleep(30)'], stdout=output)
        self.assertIsNotNone(owner.process); self.assertIsNotNone(owner.pidfd)
        owner.proc = Proc(); owner.terminate(.05, 30); owner.reap(); self.assert_dead(owner.process.pid)
    def test_scan_failure_never_proves_cleanup(self):
        owner = self.owned()
        with patch.object(owner.proc, 'scan', side_effect=Refusal('injected proc failure')):
            with self.assertRaises(Refusal): owner.terminate(.02, .02)
        self.assertFalse(owner.reaped); self.assertEqual(read(owner.record)['state'], 'cleanup-failed')
    def test_unmarked_reparented_child_in_target_session_is_owned(self):
        owner = self.owned(); parent = owner.leader
        anchor = Process(90000001, parent.pid, 90000001, 90000001, 20, os.getuid(), 'S')
        child = Process(90000002, 1, 90000002, anchor.session, 21, os.getuid(), 'S')
        owner.anchors[anchor.identity] = anchor
        with patch.object(owner.proc, 'scan', return_value=[parent, anchor, child]), patch.object(owner.proc, 'tagged', return_value=False):
            self.assertIn(child, owner.members())
        owner.anchors.clear(); owner.known.pop(child.identity, None); owner.known.pop(anchor.identity, None)
    def test_vanished_anchor_cannot_prove_cleanup(self):
        owner = self.owned(); anchor = Process(90000001, 1, 90000001, 90000001, 20, os.getuid(), 'S')
        owner.anchors[anchor.identity] = anchor
        with self.assertRaisesRegex(Refusal, 'anchor vanished'): owner.members()
        owner.anchors.clear()
    def test_foreign_nsys_agent_name_survives(self):
        sentinel = subprocess.Popen([sys.executable, str(HERE/'stubs.py'), 'child', '--start-agent', '--session-name', 'foreign'], start_new_session=True)
        self.addCleanup(lambda: (sentinel.kill(), sentinel.wait()))
        owner = self.owned(); owner.nsys_session = 'our-unique-session'
        owner.terminate(.05, 30); owner.reap(); self.assertIsNone(sentinel.poll())
    def test_pid_reuse_never_signalled(self):
        original = Process(123, 1, 123, 123, 100, os.getuid(), 'S')
        with patch.object(Proc, 'get', return_value=replace(original, start=101)), patch('os.pidfd_open') as opened:
            self.assertFalse(Proc().send(original, signal.SIGKILL)); opened.assert_not_called()
    def test_foreign_same_executable_survives(self):
        sentinel = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        self.addCleanup(lambda: (sentinel.kill(), sentinel.wait()))
        owner = self.owned(); owner.terminate(.05, 30); owner.reap()
        self.assertIsNone(sentinel.poll())
    def test_escaped_session_child_and_pipe_are_cleaned(self):
        children = self.root / 'children'
        result = Commands().run([sys.executable, str(HERE / 'stubs.py'), 'query', '--query-compute-apps=pid'], 60,
                  env=dict(os.environ, STUB_QUERY='child-pipe', STUB_CHILDREN=str(children)))
        self.assertEqual(result.returncode, 0)
        for pid in children.read_text().splitlines(): self.assert_dead(int(pid))
    def test_cleanup_retry_retains_pending(self):
        commands = Commands(.02, 30); owner = self.owned(); commands.pending.append(owner)
        with patch.object(owner, 'terminate', side_effect=Refusal('unproved')):
            with self.assertRaises(Refusal): commands.cleanup()
        self.assertEqual(commands.pending, [owner]); commands.cleanup(); self.assertEqual(commands.pending, [])
    def test_late_successful_utility_is_refused(self):
        clock = [0.0]
        class Late:
            process = SimpleNamespace(returncode=0)
            def __init__(self, *args): pass
            def spawn(self, *args, **kwargs): clock[0] = 2.0
            def exited(self): return True
            def terminate(self, *args, **kwargs): pass
            def reap(self, *args, **kwargs): pass
        with patch('processes.Owned', Late), patch('processes.time.monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(Refusal, 'after its deadline'): Commands().run(['not-executed'], .1, deadline=1)
    def test_expired_handshake_cannot_release_target(self):
        c, phases, preflight = self.load(self.config())
        runner = run.Runner(c, phases, preflight, self.root/'output')
        runner.owner = SimpleNamespace(exited=lambda: False)
        with self.assertRaisesRegex(Refusal, 'handshake deadline'): runner.handshake(self.root, time.monotonic()-1)
        self.assertFalse((self.root/'target.go').exists())
    def test_late_health_cannot_succeed(self):
        c, phases, preflight = self.load(self.config())
        runner = run.Runner(c, phases, preflight, self.root/'output')
        deadline = time.monotonic() - 1
        with patch.object(runner, 'request') as request:
            with self.assertRaisesRegex(Refusal, 'readiness deadline'): runner.health(run.plan(c)[0], self.root, deadline, True)
            request.assert_not_called()
    def test_proc_comm_with_parentheses(self):
        raw = Path('/proc/self/stat').read_text(); index = raw.rfind(')')
        changed = raw[:raw.index('(')] + '(name ) nested)' + raw[index+1:]
        self.assertEqual(parse_stat(os.getpid(), os.getuid(), raw), parse_stat(os.getpid(), os.getuid(), changed))


class QueryTests(Base):
    def gpu(self, mode):
        return run.GPU(dict(query_command=[sys.executable, str(HERE / 'stubs.py'), 'query'], gpus=['0'],
                            timeouts=dict(query=.1 if mode in ('hang', 'late') else 30)), Commands(.02, 30), dict(os.environ, STUB_QUERY=mode))
    def test_successful_empty_query_is_idle(self):
        self.gpu('').idle(time.monotonic()+120)
    def test_failed_empty_query_refused(self):
        with self.assertRaisesRegex(Refusal, 'GPU query failed'): self.gpu('fail').idle(time.monotonic()+120)
    def test_hung_query_bounded(self):
        # The 0.1 s query timeout ends it, not the 30 s hang or the 120 s deadline.
        start = time.monotonic()
        with self.assertRaises(Refusal): self.gpu('hang').idle(time.monotonic()+120)
        self.assertLess(time.monotonic()-start, 15)
    def test_busy_deadline_refused(self):
        with self.assertRaises(Refusal): self.gpu('busy').idle(time.monotonic()+.4)
    def test_late_success_refused(self):
        with self.assertRaises(Refusal): self.gpu('late').idle(time.monotonic()+.1)
    def test_malformed_and_diagnostic_queries_refused(self):
        for mode, message in [('malformed', 'malformed'), ('noise', 'wrote diagnostics')]:
            with self.subTest(mode=mode), self.assertRaisesRegex(Refusal, message): self.gpu(mode).idle(time.monotonic()+120)
    def test_telemetry_is_also_bounded(self):
        with self.assertRaises(Refusal): self.gpu('hang').telemetry(time.monotonic()+2)


class DriverTests(Base):
    def test_full_standard_abba_and_gsm(self):
        c = self.config([name for name in catalog() if name != 'determinism']); result = self.execute(c)
        self.assertEqual(result.returncode, 0, result.stdout.decode())
        report = read(self.root / 'output/result.json')
        self.assertTrue(report['passed']); self.assertEqual(len(report['comparisons']), 10)
        # One acceptance row per speculative phase: decode at 1, 4, 8, 16 and 32, GSM8K at 1 and 8.
        self.assertEqual(len(report['acceptance']), 7); self.assertEqual(len(report['gpu_means']), 8)
        self.assertTrue(all(row['identical'] for row in report['comparisons']))
        self.assertEqual(len(Path(c['env']['STUB_STARTS']).read_text().splitlines()), 8)
        self.assert_no_children(c)
    def test_failed_release_stops_loop_and_retains_ownership(self):
        # The GPU never shows idle, so the release window, kept short here,
        # always ends the run.
        c = self.config(); c['env']['STUB_QUERY'] = 'release-fail'; c['timeouts']['release'] = 3
        result = self.execute(c)
        self.assertEqual(result.returncode, 1, result.stdout.decode()); self.assert_no_children(c)
        report = read(self.root / 'output/result.json'); self.assertTrue(report['ownership_retained'])
        self.assertEqual(len(Path(c['env']['STUB_STARTS']).read_text().splitlines()), 1)
        self.assertEqual(report['completed_arms'], [])
        self.assertIn('decode-c1 validated', result.stdout.decode())
    def test_failed_release_query_never_counts_as_idle(self):
        c = self.config(); c['env']['STUB_QUERY'] = 'release-query-fail'; result = self.execute(c)
        self.assertEqual(result.returncode, 1, result.stdout.decode()); self.assert_no_children(c)
        self.assertTrue(read(self.root / 'output/result.json')['ownership_retained'])
        self.assertIn('decode-c1 validated', result.stdout.decode())
        self.assertEqual(len(Path(c['env']['STUB_STARTS']).read_text().splitlines()), 1)
    def test_interrupt_before_readiness_cleans_owned_launch(self):
        c = self.config(); c['env']['STUB_SERVER'] = 'hang'
        path = self.root / 'run.json'; write(path, c)
        with (self.root / 'driver.log').open('wb') as log:
            process = subprocess.Popen([sys.executable, '-B', str(HERE/'run.py'), str(path), '--out', str(self.root/'output')],
                                       stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 120
                while not Path(c['env']['STUB_STARTS']).exists():
                    self.assertLess(time.monotonic(), deadline); self.assertIsNone(process.poll()); time.sleep(.02)
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=600), 1)
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()
        self.assert_no_children(c)
        self.assertEqual(len(Path(c['env']['STUB_STARTS']).read_text().splitlines()), 1)
    def test_slow_drip_health_is_bounded(self):
        c = self.config(); c['env']['STUB_SERVER'] = 'slow-health'; c['timeouts']['http'] = .3
        # The stub drips one byte a second (hundreds of seconds for /health):
        # the 0.3 s http deadline, not the drip, ends the run.
        start = time.monotonic(); result = self.execute(c)
        self.assertEqual(result.returncode, 1, result.stdout.decode()); self.assertLess(time.monotonic()-start, 120)
        self.assert_no_children(c); self.assert_deadline_refusal(result.stdout.decode())
    def test_slow_drip_health_ending_just_after_its_deadline_is_refused(self):
        # The order a loaded host can produce: the HTTP helper's own timeout
        # ends it just after the deadline, before the driver's next poll.
        c = self.config(); c['env']['STUB_SERVER'] = 'slow-health'; c['timeouts']['http'] = .3
        start = time.monotonic(); result = self.execute(c, driver=LATE_HTTP_DRIVER); output = result.stdout.decode()
        self.assertEqual(result.returncode, 1, output); self.assertLess(time.monotonic()-start, 120)
        self.assert_no_children(c)
        self.assertIn('FAIL: Refusal: command completed after its deadline', output)
        self.assertNotIn('command timed out', output)
        self.assert_deadline_refusal(output)
    def assert_deadline_refusal(self, output):
        self.assertTrue(any(message in output for message in DEADLINE_REFUSALS), output)
        # The deadline ended readiness: no health was ever accepted.
        self.assertEqual(list((self.root / 'output').rglob('health.json')), [], output)
    def test_never_ready_server_stopped_before_next_arm(self):
        c = self.config(); c['env']['STUB_SERVER'] = 'hang'; c['timeouts']['startup'] = .3
        result = self.execute(c); self.assertEqual(result.returncode, 1, result.stdout.decode())
        self.assert_no_children(c)
        # Under host load the launch may not reach the stub's start record
        # within the 0.3 s deadline; no second arm may start either way.
        starts = Path(c['env']['STUB_STARTS'])
        self.assertLessEqual(len(starts.read_text().splitlines()) if starts.exists() else 0, 1)
        self.assertIn('FAIL: Refusal: ', result.stdout.decode())
    def test_early_exit_stops_loop(self):
        c = self.config(); c['env']['STUB_SERVER'] = 'exit'; result = self.execute(c)
        self.assertEqual(result.returncode, 1, result.stdout.decode()); self.assert_no_children(c)
        self.assertEqual(len(Path(c['env']['STUB_STARTS']).read_text().splitlines()), 1)
    def test_phase_failure_stops_loop(self):
        c = self.config(); c['env']['STUB_CLIENT'] = 'fail'; result = self.execute(c)
        self.assertEqual(result.returncode, 1, result.stdout.decode()); self.assert_no_children(c)
        self.assertIn('capture phase prefill-3k failed with 8', result.stdout.decode())
        self.assertEqual(len(Path(c['env']['STUB_STARTS']).read_text().splitlines()), 1)
    def test_profile_escaped_targets_and_failed_shutdown(self):
        c = self.config(); c['env']['STUB_PROFILE_FAIL'] = '1'
        c['profile'] = dict(command=[sys.executable, str(HERE / 'stubs.py'), 'profiler'])
        result = self.execute(c); self.assertEqual(result.returncode, 1, result.stdout.decode())
        self.assert_no_children(c); self.assertIn('nsys shutdown exit 9', result.stdout.decode())
        self.assertIn('decode-c1 validated', result.stdout.decode())
        self.assertEqual(len(Path(c['env']['STUB_STARTS']).read_text().splitlines()), 1)
        self.assertEqual(len(Path(c['env']['STUB_CHILDREN']).read_text().splitlines()), 2)
        self.assertFalse(read(self.root / 'output/result.json')['ownership_retained'])
    def smoke(self, **env):
        c = self.config(['determinism', 'warm', 'decode-c1'])
        qual = self.root / 'qual'; qual.mkdir()
        (qual / 'determinism.py').write_text(stubs.DETERMINISM)
        c.update(smoke=True, qual_tools=str(qual), variants=dict(spec=dict(mode='fixed', tokens=3)))
        c['installs']['new']['env'] = env
        return c, self.execute(c)
    def test_a_smoke_run_compares_one_ab_pair_with_its_determinism_records(self):
        c, result = self.smoke()
        self.assertEqual(result.returncode, 0, result.stdout.decode())
        report = read(self.root / 'output/result.json')
        self.assertEqual([r['name'] for r in report['comparisons']], ['base-new-spec-1'])
        self.assertEqual([d['identical'] for d in report['comparisons'][0]['determinism']], [True])
        self.assertEqual(report['acceptance'][0]['new']['fixed_k_verifies_by_repeat'], [1])
        self.assertEqual(len(Path(c['env']['STUB_STARTS']).read_text().splitlines()), 2)
        plan = read(self.root / 'output/plan.json')
        self.assertIn(str(self.root / 'qual/determinism.py'), plan['qual_tools_sha256'])
        self.assert_no_children(c)
    def test_both_captures_use_the_new_scorer_when_base_lacks_declaration(self):
        c, result = self.smoke()
        base = Path(c['installs']['base']['path']) / 'bin/chopinfer'
        refused = subprocess.run([str(base), 'eval', 'declaration'], capture_output=True, text=True)
        self.assertEqual(refused.returncode, 2)
        self.assertIn("unsupported eval command 'declaration'", refused.stderr)
        self.assert_no_children(c)
        logs = ''.join(path.read_text() for path in (self.root / 'output').glob('*/determinism.log'))
        self.assertEqual(result.returncode, 0, result.stdout.decode() + logs)
        plan = read(self.root / 'output/plan.json')
        scorer = Path(plan['install_snapshots']['new']['path']) / 'bin/chopinfer'
        self.assertEqual(len(plan['arms']), 2)
        for arm in plan['arms']:
            directory = self.root / 'output' / arm['name']
            receipt = read(read(directory / 'capture.json')['phases'][0]['receipt'])
            self.assertEqual(Path(receipt['scorer']), scorer)
            command = read(directory / 'command.json')['command']
            served = Path(plan['install_snapshots'][arm['install']]['path']) / 'bin/chopinfer'
            self.assertEqual(Path(command[command.index('--') + 1]), served)
    def test_a_report_only_smoke_of_differing_outputs_is_refused(self):
        c = self.config(['determinism', 'warm', 'decode-c1'])
        qual = self.root / 'qual'; qual.mkdir(); (qual / 'determinism.py').write_text(stubs.DETERMINISM)
        c.update(smoke=True, qual_tools=str(qual), variants=dict(ar=dict(mode='off', tokens=0)), base_new='report-only')
        c['installs']['new']['env'] = dict(STUB_DIFFERENT='1', STUB_DETERMINISM_DIFFERENT='1')
        result = self.execute(c)
        self.assertEqual(result.returncode, 2, result.stdout.decode())
        self.assertIn('a smoke run gates base against new bitwise', result.stdout.decode())
        self.assertFalse((self.root / 'output/result.json').exists())
    def test_an_empty_warmup_fails_a_bitwise_smoke(self):
        c, result = self.smoke(STUB_CLIENT='empty-warm')
        self.assertEqual(result.returncode, 1, result.stdout.decode())
        self.assertIn('decode warmup returned no completion choice', result.stdout.decode())
        self.assertFalse(read(self.root / 'output/result.json')['passed'])
        self.assert_no_children(c)
    def test_a_smoke_run_fails_on_different_determinism_records(self):
        c, result = self.smoke(STUB_DETERMINISM_DIFFERENT='1')
        self.assertEqual(result.returncode, 1, result.stdout.decode())
        self.assertIn('determinism records differ', result.stdout.decode())
        self.assert_no_children(c)
    def test_foreign_health_identity_fails(self):
        c = self.config(); c['env']['STUB_SERVER'] = 'bad-identity'; result = self.execute(c)
        self.assertEqual(result.returncode, 1, result.stdout.decode()); self.assert_no_children(c)
    def test_a_served_layout_other_than_the_profiles_fails(self):
        c = self.config(['warm', 'decode-c1']); c.pop('serve_args', None)
        c.update(serve_profile='tp1-ar', variants=dict(ar=dict(mode='off', tokens=0)))
        result = self.execute(c)
        self.assertEqual(result.returncode, 0, result.stdout.decode())
        # stub-fp4 is a layout the package allows, not the profile's.
        (self.root / 'output').rename(self.root / 'output-1'); c['env']['STUB_LAYOUT'] = 'stub-fp4'
        result = self.execute(c)
        self.assertEqual(result.returncode, 1, result.stdout.decode())
        self.assertIn("health.model_options.checkpoint.quantization_layout: got 'stub-fp4'", result.stdout.decode())
        self.assert_no_children(c)


class GateTests(Base):
    def runner(self):
        c, phases, preflight = self.load(self.config(['prefill-3k', 'warm', 'decode-c1']))
        runner = run.Runner(c, phases, preflight, self.root / 'output')
        for arm in run.plan(c):
            directory = runner.output / arm['name']; directory.mkdir()
            capture = dict(model='stub', phases=[stubs.phase(p['name']) for p in phases]); write(directory / 'capture.json', capture)
            counter = dict(phase='decode-c1', **acceptance(stubs.metrics(0), stubs.metrics(100),
                speculative=True, maximum=3, require_drafts=True, fixed_k=3))
            runner.arms.append(dict(**arm, directory=str(directory), capture=capture, acceptance=[counter]))
        return runner
    def mutate(self, runner, predicate):
        for arm in runner.arms:
            if predicate(arm):
                p = Path(arm['directory']) / 'capture.json'; capture = read(p)
                capture['phases'][0]['outputs'][0]['logprobs']['token_logprobs'][0] = -.6
                write(p, capture)
    def test_acceptance_regression_is_failure(self):
        runner = self.runner()
        for arm in runner.arms:
            if arm['install'] == 'new': arm['acceptance'][0]['accepted'] = 10
        with self.assertRaisesRegex(Refusal, 'acceptance fell'): runner.gates()
        self.assertFalse(read(runner.output / 'acceptance-gate.json')['phases'][0]['passed'])
    def test_new_ar_mtp_difference_is_failure_even_report_only(self):
        runner = self.runner(); runner.c['base_new'] = 'report-only'
        self.mutate(runner, lambda a: a['install'] == 'new' and a['variant'] == 'spec')
        with self.assertRaisesRegex(Refusal, 'different gated'): runner.gates()
    def test_same_install_repeat_difference_is_failure(self):
        runner = self.runner(); runner.c['base_new'] = 'report-only'
        self.mutate(runner, lambda a: a['install'] == 'new' and a['repeat'] == 2)
        with self.assertRaisesRegex(Refusal, 'different gated'): runner.gates()
    def test_base_new_difference_is_failure(self):
        runner = self.runner(); self.mutate(runner, lambda a: a['install'] == 'new')
        with self.assertRaisesRegex(Refusal, 'different gated'): runner.gates()
    def test_gsm_changed_answers_fail_and_report_only_exposes_them(self):
        phases = phases_from(['prefill-3k', 'gsm8k-c1'])
        for name in ['a', 'b']:
            p = self.root / name; p.mkdir()
            rows = [stubs.phase(d['name']) for d in phases]
            if name == 'b': rows[1]['outputs'][0]['text'] = 'different'
            write(p/'capture.json', dict(model='stub', phases=rows))
        comparator = Comparator()
        report = comparator.compare(self.root/'a', self.root/'b', phases, self.root/'report.json', Commands(), 120, False)
        self.assertFalse(report['identical']); self.assertFalse(report['gsm'][0]['identical'])
        with self.assertRaisesRegex(Refusal, 'GSM8K'):
            comparator.compare(self.root/'a', self.root/'b', phases, self.root/'gate.json', Commands(), 120)
    def test_gsm_both_missing_is_failure(self):
        c = self.config(['prefill-3k', 'gsm8k-c1']); c, phases, preflight = self.load(c)
        for name in ['a', 'b']:
            p = self.root / name; p.mkdir(); write(p / 'capture.json', dict(model='stub', phases=[stubs.phase('prefill-3k')]))
        with self.assertRaises(Refusal): Comparator().compare(self.root/'a', self.root/'b', phases,
                                                                   self.root/'cmp.json', Commands(), 120)


# The review probe used fully consistent K=1-only snapshots for these six
# phases. Keep that false pass, including the final rate gate, as a control.
EXPOSURE_PHASES = ['decode-c1', 'decode-c4', 'decode-c8', 'decode-c16', 'gsm8k-c1', 'gsm8k-c8']


def histogram_metrics(hist):
    hist = {k: hist.get(k, 0) for k in range(17)}
    verified = sum(hist[k] for k in range(1, 17))
    drafted = sum(k * n for k, n in hist.items())
    lines = [f'chop_requests_completed_total {sum(hist.values())}', 'chop_requests_waiting 0']
    if verified:
        values = [verified, drafted, drafted, drafted + verified, drafted + verified]
        lines += [f'{name} {value}' for name, value in zip(TOTALS, values)]
        for name in POSITIONS:
            lines += [f'{name}{{position="{pos}"}} {sum(n for k, n in hist.items() if k >= pos)}'
                      for pos in range(1, 17)]
    lines += [f'chop_speculation_k_total{{model_name="stub",engine="mtp",k="{k}"}} {n}'
              for k, n in hist.items()]
    return '\n'.join(lines) + '\n'


class FixedKTests(Base):
    def record(self, k=3, start=0, **kwargs):
        return acceptance(histogram_metrics({k: start}), histogram_metrics({k: start + 100}),
                          speculative=True, maximum=3, require_drafts=True, **kwargs)

    def runner(self, k=3, mode='fixed'):
        # Only numerical comparison is stubbed to isolate the real rate gate.
        # DriverTests still exercise all numerical comparisons over real files.
        phases = [dict(name=n, kind='gsm8k' if n.startswith('gsm') else 'decode') for n in EXPOSURE_PHASES]
        runner = SimpleNamespace(c=dict(variants=dict(ar=dict(mode='off', tokens=0),
            spec=dict(mode=mode, tokens=3)), base_new='bitwise', acceptance=dict(max_rate_drop=0.0)),
            phases=phases, arms=[], output=self.root,
            comparator=SimpleNamespace(compare=lambda *a, **kw: dict(valid=True, identical=True)),
            commands=None, remaining=lambda *a: 60)
        for install in ['base', 'new']:
            for variant in ['ar', 'spec']:
                for repeat in [1, 2]:
                    records = []
                    for i, phase in enumerate(phases):
                        record = self.record(k, 100*i) if variant == 'spec' else acceptance(
                            stubs.metrics(i*100, False), stubs.metrics((i+1)*100, False),
                            speculative=False, maximum=0, require_drafts=False)
                        records.append(dict(phase=phase['name'], **record))
                    runner.arms.append(dict(install=install, variant=variant, repeat=repeat,
                                           directory='unused', acceptance=records))
        return runner

    def test_review_k1_only_probe_fails_each_phase(self):
        for i, phase in enumerate(EXPOSURE_PHASES):
            with self.subTest(phase=phase):
                # Legacy admission accepts the same complete metrics.
                self.assertEqual(self.record(1, i*100)['k_histogram']['3'], 0)
                with self.assertRaisesRegex(Refusal, 'no verifications at configured fixed K=3'):
                    self.record(1, i*100, fixed_k=3)

    def test_k1_only_probe_passes_the_rate_gate_but_fails_exposure(self):
        # The same counters pass every rate row when K exposure is not asked
        # (adaptive mode); fixed K=3 refuses them.
        _, rates = run.Runner.gates(self.runner(1, mode='adaptive'))
        self.assertEqual(len(rates), 6)
        self.assertTrue(all(r['passed'] for r in rates))
        runner = self.runner(1)
        with self.assertRaisesRegex(Refusal, 'no verifications at configured fixed K=3'):
            run.Runner.gates(runner)

    def test_each_arm_and_phase_requires_its_own_exposure(self):
        for install in ['base', 'new']:
            for repeat in [1, 2]:
                for phase in EXPOSURE_PHASES:
                    with self.subTest(install=install, repeat=repeat, phase=phase):
                        runner = self.runner()
                        arm = next(a for a in runner.arms if a['install'] == install and
                                   a['variant'] == 'spec' and a['repeat'] == repeat)
                        arm['acceptance'] = [dict(phase=phase, **self.record(1)) if r['phase'] == phase else r
                                             for r in arm['acceptance']]
                        with self.assertRaisesRegex(Refusal, 'no verifications at configured fixed K=3'):
                            run.Runner.gates(runner)

    def test_fixed_control_passes_and_reports_per_repeat(self):
        _, rates = run.Runner.gates(self.runner())
        for row in rates:
            for side in ['base', 'new']:
                self.assertEqual(row[side]['fixed_k_verifies_by_repeat'], [100, 100])
                self.assertEqual(row[side]['fixed_k'], 3)
                self.assertTrue(row['passed'])

    def test_mixed_boundary_allocations_are_allowed(self):
        r = acceptance(histogram_metrics({}), histogram_metrics({0: 50, 1: 20, 2: 10, 3: 1}),
                       speculative=True, maximum=3, require_drafts=True, fixed_k=3)
        self.assertEqual(r['fixed_k_verifies'], 1)
        self.assertEqual(r['k_histogram']['0'], 50)
        self.assertEqual(r['verified'], 31)

    def test_old_k3_counts_cannot_cover_a_later_k1_phase(self):
        with self.assertRaisesRegex(Refusal, 'no verifications at configured fixed K=3'):
            acceptance(histogram_metrics({3: 500}), histogram_metrics({3: 500, 1: 100}),
                       speculative=True, maximum=3, require_drafts=True, fixed_k=3)

    def test_acceptance_is_not_exposure(self):
        r = acceptance(stubs.metrics(0), stubs.metrics(20, accepted_per_step=0),
                       speculative=True, maximum=3, require_drafts=True, fixed_k=3)
        self.assertEqual(r['accepted'], 0)
        self.assertEqual(r['fixed_k_verifies'], 20)

    def test_adaptive_keeps_positive_k_rule(self):
        self.assertEqual(self.record(1)['drafted'], 100)
        _, rates = run.Runner.gates(self.runner(1, mode='adaptive'))
        self.assertTrue(all(r['passed'] for r in rates))
        self.assertNotIn('fixed_k_verifies_by_repeat', rates[0]['new'])

    def test_all_configurable_fixed_depths(self):
        for k in range(1, 17):
            with self.subTest(k=k):
                r = acceptance(histogram_metrics({}), histogram_metrics({k: 1}),
                               speculative=True, maximum=k, require_drafts=True, fixed_k=k)
                self.assertEqual(r['fixed_k_verifies'], 1)
                if k > 1:
                    with self.assertRaisesRegex(Refusal, 'no verifications at configured fixed K'):
                        acceptance(histogram_metrics({}), histogram_metrics({k-1: 2}),
                                   speculative=True, maximum=k, require_drafts=True, fixed_k=k)

    def test_invalid_fixed_requirement_refused(self):
        for k in [0, 17, True, 3.0, '3', 2]:
            with self.subTest(k=k), self.assertRaisesRegex(Refusal, 'fixed K must equal'):
                self.record(fixed_k=k)

    def test_non_verifying_phases_do_not_require_full_k(self):
        for phase in ['prefill', 'teacher', 'warm']:
            with self.subTest(phase=phase):
                r = acceptance(histogram_metrics({}), histogram_metrics({0: 2}),
                               speculative=True, maximum=3, require_drafts=False, fixed_k=3)
                self.assertEqual(r['verified'], 0)
                self.assertNotIn('fixed_k_verifies', r)

    def test_missing_or_duplicate_phase_and_exposure_refused(self):
        for mutant in ['missing', 'duplicate', 'no-hist', 'forged-summary', 'boolean', 'fractional']:
            runner = self.runner()
            arm = next(a for a in runner.arms if a['variant'] == 'spec')
            record = arm['acceptance'][0]
            if mutant == 'missing': arm['acceptance'].pop(0)
            if mutant == 'duplicate': arm['acceptance'].append(record)
            if mutant == 'no-hist': record.pop('k_histogram')
            if mutant == 'forged-summary':
                record['k_histogram']['3'] = 0
                record['fixed_k_verifies'] = 999
            if mutant == 'boolean': record['k_histogram']['3'] = True
            if mutant == 'fractional': record['k_histogram']['3'] = .1
            with self.subTest(mutant=mutant), self.assertRaises(Refusal):
                run.Runner.gates(runner)

    def test_cli_k1_only_fails_after_owned_cleanup(self):
        c = self.config(['warm', 'decode-c1'])
        # Alter only the metrics generator in disposable CPU fixture installs.
        # The health identity still reports fixed K=3, as in the review probe.
        c['env']['STUB_ACCEPT'] = '1'
        for installation in c['installs'].values():
            root = Path(installation['path'])
            binary = root/'bin/chopinfer'
            source = binary.read_text()
            start, end = source.index('def metrics('), source.index('def choice(')
            body = source[start:end].replace('3', '1').replace('4 * steps', '2 * steps')
            binary.write_text(source[:start] + body + source[end:])
            receipt = read(root/'INSTALL.json')
            receipt['files']['bin/chopinfer'] = digest(binary)
            write(root/'INSTALL.json', receipt)
        result = self.execute(c)
        self.assertEqual(result.returncode, 1, result.stdout.decode())
        report = read(self.root/'output/result.json')
        self.assertFalse(report['passed'])
        self.assertIn('no verifications at configured fixed K=3', result.stdout.decode())
        self.assertEqual(len(report['completed_arms']), 4)  # AR controls, then first MTP refuses.
        self.assertEqual(len(Path(c['env']['STUB_STARTS']).read_text().splitlines()), 5)
        self.assertFalse(report['ownership_retained'])
        self.assert_no_children(c)



class ModelTests(unittest.TestCase):
    def test_install_files_bundle_and_drafters_come_from_the_model(self):
        with tempfile.TemporaryDirectory() as name:
            facts = model.facts('stub', models(Path(name)))
        self.assertEqual((facts['server'], facts['compiler'], facts['bundle'], facts['bundle_id'], facts['drafters']),
                         ('bin/chopinfer', 'bin/chopinfer-model-stub', 'lib/libbundle_stub.so', 'stub.v1', ['mtp']))

    def test_every_shipped_model_has_serving_facts(self):
        for directory in sorted(p.parent for p in model.MODELS.glob('*/profiles.json')):
            with self.subTest(model=directory.name):
                facts = model.facts(directory.name)
                self.assertTrue(facts['bundle_id'] and facts['drafters'] is not None)

    def test_typed_health_options_match_their_profile_text(self):
        model.check_options({'prefill_chunk_tokens': '8192', 'prefill_upper_skip': 'true'},
                            {'prefill_chunk_tokens': 8192, 'prefill_upper_skip': True, 'other': 1})
        for live in [{'prefill_chunk_tokens': 4096}, {'prefill_chunk_tokens': '4096'}, {}, None]:
            with self.subTest(live=live), self.assertRaisesRegex(Refusal, r'health\.model_options\.prefill_chunk_tokens'):
                model.check_options({'prefill_chunk_tokens': '8192'}, live)

    def test_two_bundle_ids_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root = models(Path(name)); path = root / 'stub/profiles.json'
            value = read(path); value['profiles']['tp1-spec']['bundle_id'] = 'other.v1'; write(path, value)
            with self.assertRaisesRegex(Refusal, 'one bundle ID'): model.facts('stub', root)


class Completions:
    """A loopback /v1/completions server: every generated token is ID
    `self.token`, with the same text and logprob bits whatever the ID.
    `self.mode`: '' (well formed), 'late-error' (full counts and usage, then
    an error event and [DONE], as the server reports a failure), 'eof' (full
    counts, finish reason and usage, no [DONE]), 'error-body' (a 200
    non-stream body holding an error), 'empty' (a 200 non-stream body with no
    choice) or 'eos' (a non-stream stop at the first position, no token)."""

    def __init__(self, test):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        self.token, self.mode, self.requests = 1, '', []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, body, kind='application/json'):
                self.send_response(200); self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

            def do_GET(self):
                self.send(json.dumps(dict(data=[dict(id='stub-model')])).encode())

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/tokenize':
                    count = len(body['prompt'].split())
                    return self.send(json.dumps(dict(tokens=[1] * count, count=count)).encode())
                owner.requests.append(body)
                n, top = body['max_tokens'], body.get('logprobs', 5)
                ids = [owner.token] * n if body.get('return_token_ids') else None
                usage = dict(prompt_tokens=len(body['prompt'].split()), completion_tokens=n)
                if not body.get('stream'):
                    if owner.mode == 'error-body':
                        return self.send(json.dumps(dict(error=dict(message='injected', type='server_error'))).encode())
                    if owner.mode == 'empty':
                        return self.send(json.dumps(dict(choices=[], usage=dict(usage, completion_tokens=0))).encode())
                    if owner.mode == 'eos':
                        value = dict(index=0, text='', logprobs=None, finish_reason='stop', token_ids=[])
                        return self.send(json.dumps(dict(choices=[value], usage=dict(usage, completion_tokens=0))).encode())
                    value = dict(text='a' * n, logprobs=stubs.choice(n, top), finish_reason='length')
                    if ids is not None:
                        value['token_ids'] = ids
                    return self.send(json.dumps(dict(choices=[dict(value, index=0)], usage=usage)).encode())
                events = []
                for i in range(n):
                    event = dict(index=0, text='a', logprobs=stubs.choice(1, top), finish_reason=None)
                    if ids is not None:
                        event['token_ids'] = [owner.token]
                    events.append(dict(choices=[event]))
                if owner.mode != 'late-error':
                    events.append(dict(choices=[dict(index=0, text='', finish_reason='length')]))
                events.append(dict(choices=[], usage=usage))
                if owner.mode == 'late-error':
                    events.append(dict(error=dict(message='injected failure', type='server_error')))
                text = ''.join(f'data: {json.dumps(e)}\n\n' for e in events)
                if owner.mode != 'eof':
                    text += 'data: [DONE]\n\n'
                self.send(text.encode(), 'text/event-stream')

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        test.addCleanup(self.server.server_close)
        test.addCleanup(self.server.shutdown)
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def capture(self, out, kind, name, *extra):
        return subprocess.run([sys.executable, '-B', str(HERE / 'client.py'), self.url, str(out), kind, name, *extra],
                              capture_output=True, text=True, timeout=120)


class WireTests(unittest.TestCase):
    """client.py against a loopback server, then compare.py on its captures."""

    def compare(self, root, name, before, after):
        spec = root / f'{name}.spec.json'
        write(spec, dict(version=2, phases={name: catalog()[name]['spec']}))
        return subprocess.run([sys.executable, '-B', str(HERE / 'compare.py'), str(before.parent),
                               str(after.parent), str(root / f'{name}.report.json'), '--spec', str(spec)],
                              capture_output=True, text=True, timeout=120)

    def test_generated_ids_are_captured_and_an_alias_id_differs(self):
        server = Completions(self)
        for kind, name, extra in [('prefill', 'prefill-3k', ['--tokens', '3000']), ('decode', 'decode-c1', [])]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                captures = []
                for side, token in [('before', 1), ('after', 2)]:
                    server.token = token
                    (root / side).mkdir()
                    result = server.capture(root / side / 'capture.json', kind, name, *extra)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    captures.append(root / side / 'capture.json')
                self.assertTrue(server.requests[-1]['return_token_ids'])
                output = read(captures[0])['phases'][0]['outputs'][0]
                count = catalog()[name]['spec']['max_tokens']
                self.assertEqual((output['token_ids'], output['finish_reason']), ([1] * count, 'length'))
                self.assertEqual(output.get('done'), True if kind == 'decode' else None)
                self.assertEqual(self.compare(root, name, captures[0], captures[0]).returncode, 0)
                # Same text and logprob bits, different served IDs.
                result = self.compare(root, name, *captures)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                report = read(root / f'{name}.report.json')
                self.assertTrue(report['valid'])
                self.assertEqual(report['phases'][0]['differing_requests'], [0])

    def test_an_errored_or_unterminated_stream_is_recorded_and_refused(self):
        server = Completions(self)
        for mode, problem in [('late-error', 'the stream reported errors'), ('eof', 'without \\[DONE\\]')]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                server.mode = mode
                out = Path(directory) / 'arm/capture.json'
                out.parent.mkdir()
                result = server.capture(out, 'decode', 'decode-c1')
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertRegex(result.stderr, 'invalid capture: .*' + problem)
                output = read(out)['phases'][0]['outputs'][0]
                self.assertEqual(len(output['token_ids']), 128)
                self.assertEqual(self.compare(Path(directory), 'decode-c1', out, out).returncode, 1)
        server.mode = 'error-body'
        with tempfile.TemporaryDirectory() as directory:
            result = server.capture(Path(directory) / 'capture.json', 'prefill', 'prefill-3k', '--tokens', '3000')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('error response', result.stderr)
            self.assertFalse((Path(directory) / 'capture.json').exists())


    def test_a_warmup_must_complete(self):
        server = Completions(self)
        for mode, code in [('', 0), ('eos', 0), ('empty', 1)]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                server.mode = mode
                (Path(directory) / 'arm').mkdir()
                out = Path(directory) / 'arm/capture.json'
                result = server.capture(out, 'warm', 'warm')
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertTrue(server.requests[-1]['return_token_ids'])
                phases = [catalog()['warm']]
                if code:
                    # Recorded, then refused by the client and the phase validator.
                    self.assertIn('invalid capture: decode warmup returned no completion choice', result.stderr)
                    with self.assertRaisesRegex(Refusal, 'no completion choice'):
                        Comparator().capture(out.parent, phases)
                else:
                    Comparator().capture(out.parent, phases)


class ClientTests(unittest.TestCase):
    def test_the_determinism_phase_records_through_tools_qual(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                body = json.dumps(dict(data=[dict(id='stub')]) if self.path == '/v1/models'
                                  else dict(status='ready', modalities=['text', 'image'])).encode()
                self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers()
                self.wfile.write(body)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); (root / 'qual').mkdir(); (root / 'arm').mkdir()
            (root / 'qual/determinism.py').write_text(stubs.DETERMINISM)
            out = root / 'arm/capture.json'
            subprocess.run([sys.executable, '-B', str(HERE / 'client.py'), f'http://127.0.0.1:{server.server_port}',
                            str(out), 'determinism', 'determinism', '--seed', 'run-1', '--qual', str(root / 'qual'),
                            '--scorer', '/bin/scorer'], check=True, capture_output=True)
            entry = read(out)['phases'][0]
            self.assertEqual((entry['kind'], entry['requests'], entry['image'], entry['passed']), ('determinism', 1, True, True))
            record = read(entry['receipt'])
            self.assertEqual(record['scorer'], '/bin/scorer')
            self.assertEqual(record['argv'][record['argv'].index('--arm') + 1], 'arm')
            self.assertIn('--image', record['argv'])
            self.assertEqual(digest(entry['receipt']), entry['receipt_sha256'])

    def test_prompts_are_the_largest_within_the_target(self):
        # Whitespace tokens plus one: any count the server reports works.
        with patch.object(client, 'count', side_effect=lambda url, model, prompt: len(prompt.split()) + 1):
            for target in [3, 4, 64, 1000, 3072]:
                with self.subTest(target=target):
                    prompt = client.sized('url', 'm', lambda words: '[tag] ' + client.text(words), target)
                    self.assertEqual(len(prompt.split()) + 1, target)
            with self.assertRaises(SystemExit):
                client.sized('url', 'm', lambda words: '[tag] ' + client.text(words), 2)

    def test_catalog_bounds_hold_a_sized_prompt(self):
        for name, declaration in catalog().items():
            spec = declaration.get('spec')
            if spec and spec['kind'] != 'decode':
                with self.subTest(name=name):
                    target = int(declaration['args'][1])
                    self.assertEqual(spec['prompt_tokens']['max'], target)



if __name__ == '__main__':
    unittest.main()

"""The package-carried deployment policies resolve every image input exactly as
the image-owned policies they replaced.

`deploy-policy-resolution.json` was recorded from the image-owned validation
policy (`deploy/docker/defaults/*.sh`, `*.json` and the model rules of
`deploy/docker/deployment.py`) at the commit before it moved into
`models/<m>/deploy/policy.json`. For every environment and argument case
below, at every in-tree bundle and drafter, it holds the refusal (or none)
before preparation and after it, and the serve command the entrypoint ran.
These tests resolve the same cases through today's deployment.py, with the
policies the model helpers report and the prepared package carries.
"""
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch
import unittest

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path(__file__).with_name('deploy-policy-resolution.json')
BUNDLES = {'mimo.v26.flash.v2': 'mimo26', 'qwen38.flash.v2': 'qwen38', 'glm5next.v1': 'glm5next'}
DRAFTERS = ('', 'dflash', 'mtp', 'bogus')
ARGUMENTS = ((), ('--speculative-tokens', '0'), ('--speculative-tokens', '3'))
POLICY_ROLE = 'chop.deployment-policy.v1'


def single(*names_values):
    return [list(pair) for pair in names_values]


# Every shared variable with an accepted and a refused value, every model
# knob of the old image policies, names under each model's prefix and none,
# taps, retired names, and the rules that join two variables.
ENVIRONMENTS = [
    [],
    *[[[name, value]] for name, value in [
        ('MAX_SEQUENCES', '16'), ('MAX_SEQUENCES', '0'), ('MAX_SEQUENCES', ''),
        ('MAX_BATCH_TOKENS', '4096'), ('PREFILL_CHUNK_TOKENS', '1024'), ('QUEUE_DEPTH', '0'),
        ('MAX_CONTEXT_TOKENS', '1048576'), ('DRAIN_TIMEOUT_MS', '10000'),
        ('STEP_PROFILE_EVERY', '1'), ('SPECULATIVE_TOKENS', '0'), ('SPECULATIVE_TOKENS', '3'),
        ('SPECULATIVE_TOKENS', 'banana'), ('SPECULATION', 'fixed'), ('SPECULATION', 'bogus'),
        ('DRAFT_COST', 'per-token'), ('KV_CACHE_DTYPE', 'fp8'), ('KV_CACHE_DTYPE', 'fp16'),
        ('TENSOR_PARALLEL_SIZE', '2'), ('TENSOR_PARALLEL_SIZE', '3'), ('KV_MEMORY_FRACTION', '0.4'),
        ('KV_MEMORY_FRACTION', 'nan'), ('LISTEN', '127.0.0.1:9000'), ('LISTEN', 'localhost:8000'),
        ('MODALITIES', 'text,image'), ('MODALITIES', 'image'), ('MODEL_OPTIONS', 'a=1,b=2'),
        ('MODEL_OPTIONS', 'a=1,a=2'), ('API_KEY', 'secret'), ('VERIFY_MODEL', '1'),
        ('VERIFY_MODEL', 'yes'), ('SAMPLING_TIMING', '1'), ('SAMPLING_TIMING', '0'),
        ('REQUEST_TRACE', '1'), ('BATCH_REFUSAL_WINDOW_SECONDS', '0'), ('SERVE_ARGS', ''),
        ('SERVE_ARGS', '--prefix-cache off'), ('MAX_SEQUENCE', '1'), ('ENCODER_CAPTURE_DIR', 'dump'),
        ('ENCODER_CAPTURE_ALL_LAYERS', '1'), ('VIDEO_FRAME_CHECKSUMS', '1'),
        ('MIMO_F32_LOGITS', '1'), ('MIMO_F32_LOGITS', 'yes'), ('MIMO_VERIFY_EAGER', '0'),
        ('MIMO_MTP_EAGER', '1'), ('MIMO_ATTENTION_FOLD', '1'), ('MIMO_PREFILL_OVERLAP', '0'),
        ('MIMO_TRACE_LAUNCHES', '1'), ('MIMO_TRACE_LAUNCHES', '0'),
        ('MIMO_OVERLAP_RESERVED_SMS', '187'), ('MIMO_OVERLAP_RESERVED_SMS', '188'),
        ('MIMO_OVERLAP_COMM_CTAS', '24'), ('MIMO_OVERLAP_COMM_CTAS', '1128'),
        ('MIMO_OVERLAP_COMM_CTAS', '1136'), ('MIMO_OVERLAP_COMM_CTAS', '20'),
        ('MIMO_OVERLAP_MIN_TOKENS', '4294967295'), ('MIMO_OVERLAP_MIN_TOKENS', '4294967296'),
        ('MIMO_UNKNOWN', '1'), ('QWEN38_UNKNOWN', '1'), ('GLM5NEXT_UNKNOWN', '1'),
        ('ENCODER_UNKNOWN', '1'), ('OTHER_UNKNOWN', '1'),
        ('TAPS_DIR', 'taps'), ('TAPS_LAYERS', '1'), ('TAPS_SCOPE_JSON', '{}'),
        ('QWEN38_DUMP_DIR', 'dump'), ('QWEN38_DUMP_OPS', '1'), ('QWEN38_DUMP_TAPS', '1'),
        ('QWEN38_DUMP_FROM', '1'), ('QWEN38_DUMP_BYTES', '1'), ('QWEN38_DUMP_SCOPE_JSON', '{}'),
        ('PACKAGE', '1'), ('VIDEO_INPUT', '1'),
        ('EXPECTED_BUNDLE_ID', 'mimo.v26.flash.v2'), ('EXPECTED_BUNDLE_ID', 'qwen38.flash.v2'),
        ('EXPECTED_BUNDLE_ID', 'glm5next.v1'), ('EXPECTED_BUNDLE_ID', '../bad'),
    ]],
    single(('MIMO_PREFILL_OVERLAP', '0'), ('MIMO_OVERLAP_COMM_CTAS', '24')),
    single(('MIMO_PREFILL_OVERLAP', '1'), ('MIMO_OVERLAP_COMM_CTAS', '24')),
    single(('MIMO_OVERLAP_MIN_TOKENS', '8'), ('MIMO_PREFILL_OVERLAP', '0')),
    single(('SPECULATIVE_TOKENS', '0'), ('SPECULATION', 'fixed')),
    single(('SPECULATIVE_TOKENS', '0'), ('DRAFT_COST', 'per-token')),
    single(('SPECULATIVE_TOKENS', '3'), ('SPECULATION', 'adaptive')),
    single(('SAMPLING_CAPTURE_ROWS', '4')),
    single(('SAMPLING_CAPTURE', 'capture'), ('SAMPLING_CAPTURE_ROWS', '4')),
    single(('ENCODER_CAPTURE_DIR', 'dump'), ('ENCODER_CAPTURE_ALL_LAYERS', '1')),
    single(('VERIFY_MANIFEST', 'manifest')),
    single(('VERIFY_MODEL', '1'), ('VERIFY_MANIFEST', 'manifest')),
    single(('MIMO_MTP_EAGER', '1'), ('SPECULATIVE_TOKENS', '3')),
    single(('QWEN38_UNKNOWN', '1'), ('MIMO_F32_LOGITS', '1')),
    single(('MIMO_F32_LOGITS', '1'), ('GLM5NEXT_UNKNOWN', '1')),
]


def load(path):
    spec = importlib.util.spec_from_file_location('deployment_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def outcome(call):
    try:
        call()
    except ValueError as error:
        return str(error)
    return None


def cases():
    """Every case in recording order: (environment index, bundle, drafter,
    arguments)."""
    for index in range(len(ENVIRONMENTS)):
        for bundle in BUNDLES:
            for drafter in DRAFTERS:
                for arguments in ARGUMENTS:
                    yield index, bundle, drafter, arguments


@contextlib.contextmanager
def environment(pairs):
    with patch.dict(os.environ, {'CHOPINFER_' + name: value for name, value in pairs}, clear=True):
        yield


def policy_bytes(model):
    return (ROOT / 'models' / model / 'deploy/policy.json').read_bytes()


def write_package(directory, bundle, drafter, policy):
    """A prepared package directory for `bundle` that declares `drafter` as
    its bundle drafter and carries `policy` as its deployment policy."""
    (directory / 'components').mkdir(parents=True, exist_ok=True)
    manifest = {'id': bundle, 'components': []}
    if drafter:
        manifest['speculation'] = {'bundle_drafter': True, 'drafter_kind': drafter}
    if policy is not None:
        (directory / 'components/deployment-policy.json').write_bytes(policy)
        manifest['components'].append({
            'id': 'deployment-policy', 'role': POLICY_ROLE, 'format': 'chop.deployment-policy.json.v1',
            'path': 'components/deployment-policy.json', 'byte_length': len(policy),
            'sha256': hashlib.sha256(policy).hexdigest()})
    (directory / 'manifest.json').write_text(json.dumps(manifest))


class DeploymentPolicyEquality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = json.loads(FROZEN.read_text())
        cls.deployment = load(ROOT / 'deploy/docker/deployment.py')
        cls.installed = [cls.deployment.policy(policy_bytes(model), model) for model in BUNDLES.values()]

    def test_the_frozen_cases_are_these_cases(self):
        self.assertEqual(self.frozen['environments'], ENVIRONMENTS)
        self.assertEqual(len(self.frozen['early']), len(ENVIRONMENTS) * len(ARGUMENTS))
        self.assertEqual(len(self.frozen['late']), len(list(cases())))

    def test_every_case_refuses_or_accepts_as_the_image_policies_did(self):
        deployment, messages = self.deployment, self.frozen['messages']
        packages = {bundle: deployment.policy(policy_bytes(model), 'package')
                    for bundle, model in BUNDLES.items()}
        early = iter(self.frozen['early'])
        for index, pairs in enumerate(ENVIRONMENTS):
            for arguments in ARGUMENTS:
                with environment(pairs):
                    got = outcome(lambda: deployment.validate(arguments=arguments, policies=self.installed))
                expected = next(early)
                self.assertEqual(got, None if expected is None else messages[expected], (pairs, arguments))
        for (index, bundle, drafter, arguments), expected in zip(cases(), self.frozen['late']):
            with environment(ENVIRONMENTS[index]):
                got = outcome(lambda: deployment.validate(
                    bundle, drafter, arguments, policies=self.installed, package=packages[bundle]))
            self.assertEqual(got, None if expected is None else messages[expected],
                             (ENVIRONMENTS[index], bundle, drafter, arguments))

    def test_the_prepared_package_policy_is_the_one_read(self):
        # The package's policy, read from its verified component, resolves
        # each bundle's cases as the model's file does.
        deployment = self.deployment
        with tempfile.TemporaryDirectory() as directory:
            for bundle, model in BUNDLES.items():
                package = Path(directory) / model
                write_package(package, bundle, 'mtp', policy_bytes(model))
                reply = Path(directory) / f'{model}.json'
                reply.write_text(json.dumps({'result': 'prepared-direct', 'bundle': str(package),
                                             'bundle_id': bundle}))
                self.assertEqual(deployment.prepared(reply),
                                 (bundle, 'mtp', deployment.policy(policy_bytes(model), 'package')))


class EntrypointCommandEquality(unittest.TestCase):
    """The serve command the entrypoint runs, against the one recorded with
    the image policies, for every bundle and a set of accepted and refused
    environments."""

    def test_entrypoint_commands_match_the_image_policies(self):
        recorded = json.loads(FROZEN.read_text())['entrypoint']
        self.assertTrue(recorded)
        with tempfile.TemporaryDirectory(prefix='chopinfer-policy-equality-') as directory:
            harness = Harness(Path(directory), ROOT / 'deploy/docker/entrypoint.sh', policies=True)
            for case in recorded:
                with self.subTest(case=case['name']):
                    self.assertEqual(harness.run(case['bundle'], case['drafter'], case['env'], case['args']),
                                     case['result'])


ENTRYPOINT_CASES = [
    ('plain', [], []),
    ('mimo-knobs', [['MIMO_F32_LOGITS', '1'], ['MIMO_OVERLAP_COMM_CTAS', '24']], []),
    ('overrides', [['MAX_SEQUENCES', '16'], ['KV_MEMORY_FRACTION', '0.4'], ['SPECULATIVE_TOKENS', '0']],
     ['--prefix-cache', 'off']),
    ('speculation', [['SPECULATION', 'fixed']], ['--speculative-tokens', '5']),
    ('mtp-eager', [['MIMO_MTP_EAGER', '1']], []),
    ('foreign-knob', [['QWEN38_UNKNOWN', '1']], []),
    ('taps', [['QWEN38_DUMP_DIR', 'dump']], []),
    ('typo', [['MAX_SEQUENCE', '1']], []),
    ('expected', [['EXPECTED_BUNDLE_ID', 'qwen38.flash.v2']], []),
]

# A stand-in for an installed family helper: it prepares a package for the
# bundle and drafter the checkpoint's fake.json names, and, with POLICIES
# set, reports and carries its model's deployment policy.
HELPER = '''#!/usr/bin/env python3
import hashlib, json, os, pathlib, sys
family = pathlib.Path(sys.argv[0]).name.removeprefix("chopinfer-model-")
policies = os.environ.get("POLICIES")
policy = pathlib.Path(policies or "/nonexistent") / family / "deploy/policy.json"
if sys.argv[1:] == ["deployment-policy"]:
    if policy.is_file():
        sys.stdout.write(policy.read_text())
    sys.exit(0)
with open(os.environ["CALLS"], "a") as log:
    log.write(json.dumps([family, *sys.argv[1:]]) + "\\n")
option = lambda name: sys.argv[sys.argv.index(name) + 1]
model, cache = pathlib.Path(option("--model")), pathlib.Path(option("--cache-dir"))
fake = json.loads((model / "fake.json").read_text())
if fake["family"] != family:
    sys.exit("Error: not a " + family + " checkpoint")
bundle = cache / "p-fixture"
(bundle / "components").mkdir(parents=True, exist_ok=True)
manifest = {"id": fake["bundle_id"], "components": []}
if fake["drafter"]:
    manifest["speculation"] = {"bundle_drafter": True, "drafter_kind": fake["drafter"]}
if policies:
    data = policy.read_bytes()
    (bundle / "components/deployment-policy.json").write_bytes(data)
    manifest["components"].append({"id": "deployment-policy", "role": "chop.deployment-policy.v1",
        "format": "chop.deployment-policy.json.v1", "path": "components/deployment-policy.json",
        "byte_length": len(data), "sha256": hashlib.sha256(data).hexdigest()})
(bundle / "manifest.json").write_text(json.dumps(manifest))
print(json.dumps({"result": "prepared-direct", "bundle": str(bundle), "cache_root": str(cache),
                  "bundle_id": fake["bundle_id"]}))
'''


class Harness:
    """Runs `entrypoint` against fake helpers for every in-tree family and a
    fake host binary, and reports the serve command with the run's paths
    replaced by names."""

    def __init__(self, root, entrypoint, policies):
        self.root, self.entrypoint = root, entrypoint
        self.model, self.cache, self.bin = root / 'checkpoint', root / 'cache', root / 'bin'
        self.calls = root / 'calls'
        self.model.mkdir()
        self.bin.mkdir()
        binary = self.bin / 'chopinfer'
        binary.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                          'open(os.environ["CALLS"], "a").write(json.dumps(sys.argv[1:]) + "\\n")\n')
        binary.chmod(0o755)
        for family in BUNDLES.values():
            helper = self.bin / f'chopinfer-model-{family}'
            helper.write_text(HELPER)
            helper.chmod(0o755)
        self.env = dict({k: v for k, v in os.environ.items() if not k.startswith('CHOPINFER_')},
                        CHOPINFER_BINARY=str(binary), CHOPINFER_MODEL=str(self.model),
                        CHOPINFER_MODEL_CACHE=str(self.cache), CALLS=str(self.calls))
        if policies:
            self.env['POLICIES'] = str(ROOT / 'models')

    def run(self, bundle, drafter, env, args):
        (self.model / 'fake.json').write_text(json.dumps(
            {'family': BUNDLES[bundle], 'bundle_id': bundle, 'drafter': drafter}))
        self.calls.unlink(missing_ok=True)
        result = subprocess.run([str(self.entrypoint), *args], env={**self.env, **{
            'CHOPINFER_' + name: value for name, value in env}}, capture_output=True, text=True, timeout=30)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []
        names = {str(self.model): 'MODEL', str(self.cache): 'CACHE'}
        served = [[names.get(word, word) for word in call] for call in calls if call[:1] == ['serve']]
        return {'status': result.returncode,
                'prepared': any(call[0] in BUNDLES.values() for call in calls),
                'serve': served[0] if served else None}


if __name__ == '__main__':
    unittest.main()

"""Check deployment selection and argument forwarding without Docker or GPUs."""
import importlib.util
from unittest.mock import patch
import json
import re
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / os.environ.get("CARGO_TARGET_DIR", "target")
BUNDLE = "mimo.v26.flash.v2"
QWEN = "qwen38.flash.v2"

# A stand-in for an installed family helper. The checkpoint's fake.json says
# which family accepts it and what the prepared package declares. Like the
# real helpers, it reports its model's deployment policy and the package
# carries it, unless fake.json replaces it ("policy", null for none).
HELPER = '''#!/usr/bin/env python3
import hashlib, json, os, pathlib, sys, time
family = pathlib.Path(sys.argv[0]).name.removeprefix("chopinfer-model-")
policy = pathlib.Path(os.environ["POLICIES"]) / family / "deploy/policy.json"
if sys.argv[1:] == ["deployment-policy"]:
    sys.stdout.write(policy.read_text())
    sys.exit(0)
with open(os.environ["CALLS"], "a") as log:
    log.write(json.dumps([pathlib.Path(sys.argv[0]).name, *sys.argv[1:]]) + "\\n")
option = lambda name: sys.argv[sys.argv.index(name) + 1]
model, cache = pathlib.Path(option("--model")), pathlib.Path(option("--cache-dir"))
fake = json.loads((model / "fake.json").read_text())
if fake["family"] != family:
    print(json.dumps({"result": "error", "message": "not a " + family + " checkpoint"}))
    sys.exit("Error: not a " + family + " checkpoint")
if "sleep" in fake:
    pathlib.Path(fake["started"]).touch()
    time.sleep(fake["sleep"])
bundle = cache / "p-fixture"
bundle.mkdir(parents=True, exist_ok=True)
manifest = {"id": fake["bundle_id"], "components": [], **fake.get("manifest", {})}
data = fake["policy"].encode() if "policy" in fake and fake["policy"] is not None else policy.read_bytes()
if fake.get("policy", "") is not None:
    (bundle / "components").mkdir(exist_ok=True)
    target = bundle / "components/deployment-policy.json"
    target.unlink(missing_ok=True)
    if fake.get("policy_symlink"):
        (cache / "outside-policy.json").write_bytes(data)
        target.symlink_to(cache / "outside-policy.json")
    else:
        target.write_bytes(data)
    manifest["components"].append({"id": "deployment-policy", "role": "chop.deployment-policy.v1",
        "format": "chop.deployment-policy.json.v1", "path": "components/deployment-policy.json",
        "byte_length": len(data), "sha256": fake.get("policy_sha256", hashlib.sha256(data).hexdigest())})
(bundle / "manifest.json").unlink(missing_ok=True)
if fake.get("manifest_symlink"):
    (cache / "outside.json").write_text(json.dumps(manifest))
    (bundle / "manifest.json").symlink_to(cache / "outside.json")
else:
    (bundle / "manifest.json").write_text(json.dumps(manifest))
reply = {"result": "prepared-direct", "bundle": str(bundle), "cache_root": str(cache),
         "bundle_id": fake["bundle_id"],
         "tensor_parallel_size": int(option("--tensor-parallel-size")),
         "profile": {"name": "fixture", "profile": {}}, **fake.get("reply", {})}
print(fake.get("raw_reply", json.dumps(reply)))
print("prepared " + fake["bundle_id"], file=sys.stderr)
'''


def family_of(bundle):
    return "mimo26" if bundle.startswith("mimo.") else "qwen38" if bundle.startswith("qwen38.") else \
        "glm5next" if bundle.startswith("glm5next.") else "other"


class Entrypoint(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="chopinfer-entrypoint-")
        self.root = Path(self.temp.name)
        self.model = self.root / "checkpoint"
        self.model.mkdir()
        self.cache = self.root / "model-cache"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.binary = self.bin / "chopinfer"
        self.binary.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ["CALLS"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
sys.exit(int(os.environ.get("VERIFY_STATUS", "0")) if sys.argv[1:2] == ["verify"] else 0)
''')
        self.binary.chmod(0o755)
        for family in ("mimo26", "qwen38"):
            helper = self.bin / f"chopinfer-model-{family}"
            helper.write_text(HELPER)
            helper.chmod(0o755)
        self.checkpoint(BUNDLE)
        self.calls = self.root / "calls"
        self.env = dict({k: v for k, v in os.environ.items() if not k.startswith("CHOPINFER_")},
                        CHOPINFER_BINARY=str(self.binary), CHOPINFER_MODEL=str(self.model),
                        CHOPINFER_MODEL_CACHE=str(self.cache), CALLS=str(self.calls),
                        POLICIES=str(ROOT / "models"))

    def tearDown(self):
        self.temp.cleanup()

    def checkpoint(self, bundle, drafter=None, bundle_drafter=True, **fields):
        fake = {"family": family_of(bundle), "bundle_id": bundle, **fields}
        if drafter:
            fake["manifest"] = {"speculation": {"bundle_drafter": bundle_drafter, "drafter_kind": drafter}}
        (self.model / "fake.json").write_text(json.dumps(fake))

    def run_entrypoint(self, *arguments, cwd=None):
        return subprocess.run([str(ROOT / "deploy/docker/entrypoint.sh"), *arguments],
                              env=self.env, capture_output=True, text=True, timeout=10, cwd=cwd)

    def test_a_relative_binary_finds_its_helpers(self):
        # CHOPINFER_BINARY=./chopinfer from the install's bin directory runs
        # the helpers beside it, not commands of the same name on PATH.
        self.env["CHOPINFER_BINARY"] = "./chopinfer"
        self.env["CHOPINFER_MIMO_F32_LOGITS"] = "1"
        result = self.run_entrypoint(cwd=self.bin)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands()[-1][0], "serve")

    def commands(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def served(self):
        return any(command[0] == "serve" for command in self.commands())

    def test_unspecified_serving_policy_is_left_to_native_profiles(self):
        for bundle, drafter in [(BUNDLE, 'dflash'), (QWEN, 'mtp')]:
            with self.subTest(bundle=bundle):
                self.checkpoint(bundle, drafter)
                result = self.run_entrypoint()
                self.assertEqual(result.returncode, 0, result.stderr)
                serve = self.commands()[-1]
                for flag in ('--max-sequences', '--max-batch-tokens', '--prefill-chunk-tokens',
                             '--kv-cache-dtype', '--kv-memory-fraction', '--prefix-cache',
                             '--speculative-tokens', '--speculation', '--draft-cost',
                             '--drain-timeout-ms', '--modalities'):
                    self.assertNotIn(flag, serve)

    def test_image_version_uses_build_identity_without_loading_cuda(self):
        (self.root / 'BUILD_INFO').write_text('version=0.1.0-alpha.1\nrevision=' + 'a' * 40 + '\n')
        self.binary.unlink()
        result = self.run_entrypoint('--version')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'chopinfer 0.1.0-alpha.1 (' + 'a' * 40 + ')')
        self.assertFalse(self.calls.exists())

    def test_cli_commands_do_not_prepare_or_require_a_checkpoint(self):
        self.env['CHOPINFER_MODEL'] = str(self.root / 'absent')
        for args in [('--version',), ('--help',), ('verify', '--model', str(self.model))]:
            with self.subTest(args=args):
                result = self.run_entrypoint(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.commands()[-1], list(args))
        self.assertFalse(self.cache.exists())

    def test_synthetic_bundle_reads_validation_policy(self):
        spec = importlib.util.spec_from_file_location("deployment", ROOT / "deploy/docker/deployment.py")
        deployment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deployment)
        synthetic = deployment.policy(json.dumps({
            "schema": "chopinfer-deployment-policy-v1", "bundle_id": "synthetic.v1",
            "knob_prefixes": ["SYNTHETIC_"], "drafters": ["fixture"], "booleans": ["SYNTHETIC_FLAG"]}).encode(),
            "synthetic")
        another = deployment.policy(json.dumps({
            "schema": "chopinfer-deployment-policy-v1", "bundle_id": "another.v1",
            "knob_prefixes": ["ANOTHER_"], "drafters": [], "booleans": ["ANOTHER_FIXTURE"]}).encode(),
            "another")
        policies = [synthetic, another]
        with patch.dict(os.environ, {"CHOPINFER_SYNTHETIC_FLAG": "1"}, clear=True):
            deployment.validate(policies=policies)
            deployment.validate("synthetic.v1", "fixture", policies=policies, package=synthetic)
            with self.assertRaisesRegex(ValueError, "unsupported drafter mtp"):
                deployment.validate("synthetic.v1", "mtp", policies=policies, package=synthetic)
            with patch.dict(os.environ, {"CHOPINFER_SPECULATION": "adaptive", "CHOPINFER_SPECULATIVE_TOKENS": "0"}), \
                 self.assertRaisesRegex(ValueError, "speculation is disabled"):
                deployment.validate("synthetic.v1", "fixture", policies=policies, package=synthetic)
            with patch.dict(os.environ, {"CHOPINFER_ANOTHER_FIXTURE": "1"}), \
                 self.assertRaisesRegex(ValueError, "does not apply"):
                deployment.validate("synthetic.v1", "fixture", policies=policies, package=synthetic)
            # A variable no installed model declares is unknown.
            with self.assertRaisesRegex(ValueError, "CHOPINFER_SYNTHETIC_FLAG: unknown deployment variable"):
                deployment.validate(policies=[another])
        for field, value in [("schema", "v0"), ("knob_prefixes", []), ("booleans", ["MAX_SEQUENCES"]),
                             ("drafter_only", {"SYNTHETIC_FLAG": ["mtp"]}), ("unknown", 1)]:
            with self.subTest(field=field):
                document = {"schema": "chopinfer-deployment-policy-v1", "bundle_id": "synthetic.v1",
                            "knob_prefixes": ["SYNTHETIC_"], "drafters": ["fixture"],
                            "booleans": ["SYNTHETIC_FLAG"], field: value}
                with self.assertRaisesRegex(ValueError, "invalid deployment policy"):
                    deployment.policy(json.dumps(document).encode(), "synthetic")

    def test_every_model_policy_passes_both_readers(self):
        # The image's reader accepts each in-tree model's policy, which the
        # compiler's reader (deployment_policy.rs) checks at packaging.
        spec = importlib.util.spec_from_file_location("deployment", ROOT / "deploy/docker/deployment.py")
        deployment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deployment)
        paths = sorted((ROOT / "models").glob("*/deploy/policy.json"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(model=path.parents[1].name):
                policy = deployment.policy(path.read_bytes(), path.name)
                profiles = json.loads((path.parents[1] / "profiles.json").read_text())["profiles"]
                self.assertEqual({profile["bundle_id"] for profile in profiles.values()}, {policy["bundle_id"]})

    def test_the_shared_policy_cases_resolve_as_the_compiler_reader_does(self):
        # The same documents model-compiler-sdk's deployment_policy tests
        # read: both readers accept or refuse each alike.
        spec = importlib.util.spec_from_file_location("deployment", ROOT / "deploy/docker/deployment.py")
        deployment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deployment)
        cases = json.loads((ROOT / "tests/deployment-policy-cases.json").read_text())
        self.assertGreater(len(cases), 10)
        for case in cases:
            with self.subTest(case=case["name"]):
                document = case["document"]
                if "padding" in case:
                    end = document.rindex("}")
                    document = document[:end] + " " * case["padding"] + document[end:]
                data = (b"\xef\xbb\xbf" if case.get("prefix") == "bom" else b"") + document.encode()
                try:
                    deployment.policy(data, case["name"])
                    accepted = True
                except ValueError:
                    accepted = False
                self.assertEqual(accepted, case["valid"])

    def test_no_policy_changes_a_shared_variable(self):
        spec = importlib.util.spec_from_file_location("deployment", ROOT / "deploy/docker/deployment.py")
        deployment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(deployment)
        # The compiler's reader refuses the same shared names.
        source = (ROOT / "crates/model-compiler-sdk/src/deployment_policy.rs").read_text()
        listed = source.split("pub const SHARED_VARIABLES: &[&str] = &[", 1)[1].split("];", 1)[0]
        self.assertEqual(set(re.findall(r'"([A-Z0-9_]+)"', listed)), deployment.SHARED)
        # A policy reaching MAX_SEQUENCES is refused, and one that skipped the
        # reader still cannot narrow it: MiMo serves 16 sequences.
        document = {"schema": "chopinfer-deployment-policy-v1", "bundle_id": "rogue.v1",
                    "knob_prefixes": ["MAX_"], "drafters": [], "numbers": {"MAX_SEQUENCES": [1, 8]}}
        with self.assertRaisesRegex(ValueError, "covers a shared variable"):
            deployment.policy(json.dumps(document).encode(), "rogue")
        rogue = {"paths": [], "booleans": ["MAX_SEQUENCES"], "presence": [], "multiples": {"MAX_SEQUENCES": 3},
                 "taps": ["MAX_BATCH_TOKENS"], "drafter_only": {"MAX_SEQUENCES": ["mtp"]},
                 "disabled_by": [{"variable": "MAX_BATCH_TOKENS", "value": "1", "prefix": "MAX_",
                                  "reason": "rogue"}], **document}
        mimo = deployment.policy((ROOT / "models/mimo26/deploy/policy.json").read_bytes(), "mimo26")
        with patch.dict(os.environ, {"CHOPINFER_MAX_SEQUENCES": "16", "CHOPINFER_MAX_BATCH_TOKENS": "1"},
                        clear=True):
            deployment.validate(policies=[rogue, mimo])
            deployment.validate(BUNDLE, "dflash", policies=[rogue, mimo], package=mimo)
            deployment.validate("rogue.v1", "", policies=[rogue, mimo], package=rogue)

    def test_a_package_without_a_policy_serves_with_the_shared_variables(self):
        # The image keeps no per-bundle file: a package of a bundle no
        # installed helper declares serves when it carries no policy, and
        # its knob-free environment passes.
        self.checkpoint("third.tp4.v1", "mtp", family="mimo26", policy=None)
        result = self.run_entrypoint()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.commands()[-1][0], "serve")
        self.calls.write_text("")
        self.env["CHOPINFER_MIMO_F32_LOGITS"] = "1"
        result = self.run_entrypoint()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CHOPINFER_MIMO_F32_LOGITS: does not apply to bundle third.tp4.v1", result.stderr)
        self.assertFalse(self.served())

    def test_checkpoint_serves_directly_for_each_family_and_rank_count(self):
        for family in (BUNDLE, QWEN):
            for ranks in (2, 4):
                with self.subTest(family=family, ranks=ranks):
                    self.checkpoint(family)
                    self.calls.write_text('')
                    mimo = family.startswith('mimo')
                    if mimo:
                        self.env['CHOPINFER_MIMO_F32_LOGITS'] = '1'
                    self.env['CHOPINFER_TENSOR_PARALLEL_SIZE'] = str(ranks)
                    result = self.run_entrypoint()
                    self.env.pop('CHOPINFER_MIMO_F32_LOGITS', None)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    prepare = [c for c in self.commands() if c[0].startswith('chopinfer-model-')][-1]
                    self.assertEqual(prepare[1:], ['prepare', '--direct', '--model', str(self.model),
                                                   '--tensor-parallel-size', str(ranks),
                                                   '--cache-dir', str(self.cache),
                                                   '--modalities', 'auto'])
                    serve = self.commands()[-1]
                    self.assertEqual(serve[:9], ['serve', '--model', str(self.model),
                                                 '--tensor-parallel-size', str(ranks),
                                                 '--model-cache-dir', str(self.cache),
                                                 '--architecture', 'sm_120f'])
                    for removed in ('--bundle', '--install-root', '--topology'):
                        self.assertNotIn(removed, serve)
                    self.assertNotIn('--max-sequences', serve)

    def test_the_first_helper_that_prepares_the_checkpoint_selects_the_model(self):
        self.checkpoint(QWEN)
        self.assertEqual(self.run_entrypoint().returncode, 0)
        helpers = [c[0] for c in self.commands() if c[0].startswith('chopinfer-model-')]
        self.assertEqual(helpers, ['chopinfer-model-mimo26', 'chopinfer-model-qwen38'])
        self.calls.write_text('')
        self.checkpoint(BUNDLE)
        result = self.run_entrypoint()
        self.assertEqual(result.returncode, 0, result.stderr)
        helpers = [c[0] for c in self.commands() if c[0].startswith('chopinfer-model-')]
        self.assertEqual(helpers, ['chopinfer-model-mimo26'])
        # The accepting helper's diagnostics pass through; refusals do not.
        self.assertIn('prepared ' + BUNDLE, result.stderr)

    def test_a_checkpoint_no_helper_prepares_never_serves(self):
        self.checkpoint('other.tp4.v1')
        result = self.run_entrypoint()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no installed model helper prepares the checkpoint', result.stderr)
        for family in ('mimo26', 'qwen38'):
            self.assertIn(f'chopinfer-model-{family}: Error: not a {family} checkpoint', result.stderr)
        self.assertFalse(self.served())
        for helper in self.bin.glob('chopinfer-model-*'):
            helper.unlink()
        result = self.run_entrypoint()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no installed model helper', result.stderr)

    def test_missing_checkpoint_fails_before_preparation(self):
        self.env['CHOPINFER_MODEL'] = str(self.root / 'missing')
        result = self.run_entrypoint()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('CHOPINFER_MODEL', result.stderr)
        self.assertFalse(self.calls.exists())

    def test_api_key_environment_is_accepted_without_argv_disclosure(self):
        self.env["CHOPINFER_API_KEY"] = "test-secret"
        result = self.run_entrypoint()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("test-secret", self.calls.read_text() + result.stdout + result.stderr)

    def test_explicit_context_capacity_is_forwarded_and_cli_takes_precedence(self):
        self.env["CHOPINFER_MAX_CONTEXT_TOKENS"] = "1048576"
        result = self.run_entrypoint()
        self.assertEqual(result.returncode, 0, result.stderr)
        serve = self.commands()[-1]
        self.assertEqual(serve[serve.index("--max-context-tokens") + 1], "1048576")
        result = self.run_entrypoint("--max-context-tokens", "4096")
        self.assertEqual(result.returncode, 0, result.stderr)
        serve = self.commands()[-1]
        self.assertEqual(serve.count("--max-context-tokens"), 1)
        self.assertEqual(serve[serve.index("--max-context-tokens") + 1], "4096")
        self.env["CHOPINFER_MAX_CONTEXT_TOKENS"] = "0"
        self.assertNotEqual(self.run_entrypoint().returncode, 0)

    def test_prepare_and_forward(self):
        result = self.run_entrypoint("--speculative-tokens", "3", "--speculation", "fixed",
                                     "--devices", "1,0,2,3")
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.commands()
        self.assertEqual([command[:2] for command in commands[:-1]],
                         [["chopinfer-model-mimo26", "prepare"]])
        serve = commands[-1]
        self.assertEqual(serve[0], "serve")
        for key, value in [("--listen", "0.0.0.0:8000"), ("--speculative-tokens", "3"),
                           ("--devices", "1,0,2,3"), ("--tensor-parallel-size", "4")]:
            self.assertEqual(serve[serve.index(key) + 1], value)
        self.assertNotIn("--queue-depth", serve)
        self.assertTrue(self.cache.is_dir())

    def test_optional_checkpoint_verification(self):
        manifest = self.root / "checkpoint-files.json"
        self.env["CHOPINFER_VERIFY_MODEL"] = "1"
        self.assertEqual(self.run_entrypoint().returncode, 0)
        self.assertEqual(self.commands()[0], ["verify", "--model", str(self.model)])
        self.calls.write_text("")
        self.env["CHOPINFER_VERIFY_MANIFEST"] = f"{manifest}#mopd"
        self.assertEqual(self.run_entrypoint().returncode, 0)
        self.assertEqual(self.commands()[0], ["verify", "--model", str(self.model),
                                              "--manifest", f"{manifest}#mopd"])
        self.calls.write_text("")
        self.env["VERIFY_STATUS"] = "1"
        self.assertNotEqual(self.run_entrypoint().returncode, 0)
        self.assertEqual([command[0] for command in self.commands()], ["verify"])
        self.env["CHOPINFER_VERIFY_MODEL"] = "0"
        self.calls.write_text("")
        result = self.run_entrypoint()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CHOPINFER_VERIFY_MANIFEST", result.stderr)
        self.assertEqual(self.commands(), [])
        del self.env["CHOPINFER_VERIFY_MANIFEST"]
        self.assertEqual(self.run_entrypoint().returncode, 0)
        self.assertNotIn("verify", [command[0] for command in self.commands()])

    def test_startup_signal_reaches_the_preparing_helper(self):
        started = self.root / "started"
        self.checkpoint(BUNDLE, sleep=30, started=str(started))
        process = subprocess.Popen([str(ROOT / "deploy/docker/entrypoint.sh")], env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 10
            while not started.exists():
                self.assertLess(time.monotonic(), deadline, "helper never started")
                time.sleep(0.05)
            begun = time.monotonic()
            process.send_signal(signal.SIGTERM)
            self.assertEqual(process.wait(timeout=10), 0)
            self.assertLess(time.monotonic() - begun, 10)
        finally:
            process.kill()
            process.communicate()
        self.assertFalse(self.served())

    def serve_arguments(self, *arguments, drafter=None, **env):
        if drafter:
            self.checkpoint(json.loads((self.model / "fake.json").read_text())["bundle_id"], drafter)
        self.env.update(env)
        self.calls.write_text("")
        result = self.run_entrypoint(*arguments)
        self.assertEqual(result.returncode, 0, result.stderr)
        serve = self.commands()[-1]
        self.assertEqual(serve[0], "serve")
        return [value for i, flag in enumerate(serve)
                if flag in ('--speculative-tokens', '--speculation', '--draft-cost')
                for value in (flag, serve[i + 1])]

    def test_package_drafter_keeps_native_speculation_defaults(self):
        self.assertEqual(self.serve_arguments(), [])
        self.assertEqual(self.serve_arguments(drafter="dflash"),
                         [])

    def test_mtp_package_keeps_native_speculation_defaults(self):
        self.assertEqual(self.serve_arguments(drafter="mtp"),
                         [])

    def test_speculation_overrides(self):
        # Zero is explicit so that it also replaces the serving profile's drafter.
        self.assertEqual(self.serve_arguments(drafter="dflash", CHOPINFER_SPECULATIVE_TOKENS="0"),
                         ["--speculative-tokens", "0"])
        self.assertEqual(self.serve_arguments(CHOPINFER_SPECULATIVE_TOKENS="3", CHOPINFER_SPECULATION="fixed"),
                         ["--speculative-tokens", "3", "--speculation", "fixed"])
        self.env.pop("CHOPINFER_SPECULATIVE_TOKENS")
        self.env.pop("CHOPINFER_SPECULATION")
        self.assertEqual(self.serve_arguments("--speculative-tokens", "5"),
                         ["--speculative-tokens", "5"])
        self.assertEqual(self.serve_arguments("--draft-cost", "per-token"),
                         ["--draft-cost", "per-token"])

    def test_policy_environment_with_cli_token_count(self):
        for tokens in ('0', '3'):
            self.assertEqual(self.serve_arguments('--speculative-tokens', '5', drafter='dflash',
                                                 CHOPINFER_SPECULATIVE_TOKENS=tokens,
                                                 CHOPINFER_SPECULATION='fixed'),
                             ['--speculation', 'fixed', '--speculative-tokens', '5'])
        self.env.pop('CHOPINFER_SPECULATION')
        self.assertEqual(self.serve_arguments('--speculative-tokens', '0'),
                         ['--speculative-tokens', '0'])

    def test_systemd_serve_args_must_be_expanded(self):
        self.env['CHOPINFER_SERVE_ARGS'] = '--prefix-cache off'
        result = self.run_entrypoint('--prefix-cache', 'off')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_standalone_speculation_reaches_real_parser(self):
        self.serve_arguments('--speculation', 'fixed', drafter='dflash')
        serve = self.commands()[-1]
        self.assertEqual(serve.count('--speculation'), 1)
        # Stop at the checkpoint check after parsing, before any helper or
        # native loading.
        not_checkpoint = self.root / 'not-a-checkpoint'
        not_checkpoint.touch()
        serve[serve.index('--model') + 1] = str(not_checkpoint)
        result = subprocess.run([str(TARGET / 'debug/chopinfer'), *serve],
                                env=self.env, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--model must name a local checkpoint directory', result.stderr)
        self.assertNotIn('more than once', result.stderr)

    def test_kv_cache_dtype_defers_to_native_with_overrides(self):
        def dtype(*arguments, **env):
            self.env.update(env)
            self.calls.write_text("")
            self.assertEqual(self.run_entrypoint(*arguments).returncode, 0)
            serve = self.commands()[-1]
            return [serve[i + 1] for i, word in enumerate(serve) if word == "--kv-cache-dtype"]
        self.assertEqual(dtype(), [])
        self.assertEqual(dtype("--kv-cache-dtype", "bf16"), ["bf16"])
        self.assertEqual(dtype(CHOPINFER_KV_CACHE_DTYPE="fp8"), ["fp8"])
        self.assertEqual(dtype("--kv-cache-dtype", "auto"), ["auto"])
        self.env["CHOPINFER_KV_CACHE_DTYPE"] = "fp16"
        result = self.run_entrypoint()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CHOPINFER_KV_CACHE_DTYPE", result.stderr)

    def test_modalities_select_the_prepared_and_served_package(self):
        def selected(*arguments, **env):
            self.env.update(env)
            self.calls.write_text("")
            result = self.run_entrypoint(*arguments)
            self.assertEqual(result.returncode, 0, result.stderr)
            prepare = [c for c in self.commands() if c[0].startswith("chopinfer-model-")][-1]
            serve = self.commands()[-1]
            values = lambda words, flag: [words[i + 1] for i, w in enumerate(words) if w == flag]
            return values(prepare, "--modalities"), values(serve, "--modalities")
        self.assertEqual(selected(), (["auto"], []))
        # The helper prepares the package serve selects.
        self.assertEqual(selected(CHOPINFER_MODALITIES="text"), (["text"], ["text"]))
        self.assertEqual(selected(CHOPINFER_MODALITIES="all"), (["all"], ["all"]))
        self.assertEqual(selected(CHOPINFER_MODALITIES="text,image,video"),
                         (["text,image,video"], ["text,image,video"]))
        self.assertEqual(selected("--modalities", "text,audio"), (["text,audio"], ["text,audio"]))
        for value in ("image", "text,text", "text,speech", "", "Text", "text,all"):
            self.env["CHOPINFER_MODALITIES"] = value
            result = self.run_entrypoint()
            self.assertNotEqual(result.returncode, 0, value)
            self.assertIn("CHOPINFER_MODALITIES", result.stderr)

    def test_common_environment_overrides(self):
        overrides = {
            "LISTEN": ("--listen", "127.0.0.1:9000"),
            "MAX_SEQUENCES": ("--max-sequences", "16"),
            "MAX_BATCH_TOKENS": ("--max-batch-tokens", "4096"),
            "PREFILL_CHUNK_TOKENS": ("--prefill-chunk-tokens", "1024"),
            "QUEUE_DEPTH": ("--queue-depth", "512"),
            "KV_MEMORY_FRACTION": ("--kv-memory-fraction", "0.4"),
            "DRAIN_TIMEOUT_MS": ("--drain-timeout-ms", "10000"),
        }
        self.env.update({f"CHOPINFER_{key}": value for key, (_, value) in overrides.items()})
        self.assertEqual(self.run_entrypoint().returncode, 0)
        serve = self.commands()[-1]
        for flag, value in overrides.values():
            self.assertEqual(serve[serve.index(flag) + 1], value)

    def test_mimo_comm_grid_stops_at_what_create_accepts(self):
        # MiMo's create refuses a comm grid past the 1128 CTAs an SM120 GPU
        # holds at once; the image refuses it before preparing.
        for value, accepted in [("24", True), ("1128", True), ("1136", False), ("536870920", False)]:
            with self.subTest(value=value):
                self.env["CHOPINFER_MIMO_OVERLAP_COMM_CTAS"] = value
                result = self.run_entrypoint()
                self.assertEqual(result.returncode == 0, accepted, result.stderr)
                if not accepted:
                    self.assertIn("CHOPINFER_MIMO_OVERLAP_COMM_CTAS", result.stderr)
                del self.env["CHOPINFER_MIMO_OVERLAP_COMM_CTAS"]

    def test_zero_queue_depth_is_an_explicit_bound(self):
        self.env['CHOPINFER_QUEUE_DEPTH'] = '0'
        self.assertEqual(self.run_entrypoint().returncode, 0)
        serve = self.commands()[-1]
        self.assertEqual(serve[serve.index('--queue-depth') + 1], '0')

    def test_invalid_environment_fails_before_preparation(self):
        for name, value in [("MAX_SEQUENCE", "1"), ("LISTEN", "0.0.0.0:0"), ("LISTEN", "localhost:8000"), ("MAX_SEQUENCES", ""),
                            ("SPECULATIVE_TOKENS", "banana"), ("SPECULATION", "bogus"),
                            ("MIMO_F32_LOGITS", "yes"), ("SAMPLING_TIMING", "0"),
                            ("SERVE_ARGS", "--prefix-cache off"), ("KV_MEMORY_FRACTION", "nan"),
                            ("TENSOR_PARALLEL_SIZE", "3"), ("TENSOR_PARALLEL_SIZE", "04"),
                            ("VERIFY_MODEL", "yes"), ("MODEL", ""), ("MODEL_CACHE", "")]:
            with self.subTest(name=name, value=value):
                saved = self.env.get("CHOPINFER_" + name)
                self.env["CHOPINFER_" + name] = value
                result = self.run_entrypoint()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("CHOPINFER_" + name, result.stderr)
                self.assertFalse(self.calls.exists())
                if saved is None:
                    del self.env["CHOPINFER_" + name]
                else:
                    self.env["CHOPINFER_" + name] = saved

    def test_package_deployment_variables_name_their_replacement(self):
        for name, replacement in [("PACKAGE", "set CHOPINFER_MODEL"),
                                  ("INSTALL_ROOT", "CHOPINFER_MODEL_CACHE"),
                                  ("BUNDLE_VERIFY_CACHE_DIR", "CHOPINFER_VERIFY_MODEL=1"),
                                  ("BUNDLE_FORCE_FULL_VERIFY", "CHOPINFER_VERIFY_MODEL=1"),
                                  ("MODULE", "helper"),
                                  ("VIDEO_INPUT", "CHOPINFER_MODALITIES=all")]:
            with self.subTest(name=name):
                self.env["CHOPINFER_" + name] = "1"
                result = self.run_entrypoint()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"CHOPINFER_{name} is gone: ", result.stderr)
                self.assertIn(replacement, result.stderr)
                self.assertFalse(self.calls.exists())
                del self.env["CHOPINFER_" + name]

    def test_unused_model_and_drafter_overrides_are_refused(self):
        for bundle, name, value in [(BUNDLE, "SPECULATIVE_TOKENS", "3"),
                                    (BUNDLE, "SPECULATION", "fixed"),
                                    (BUNDLE, "QWEN38_DUMP_DIR", "dump"),
                                    (QWEN, "MIMO_F32_LOGITS", "1"),
                                    (QWEN, "SPECULATIVE_TOKENS", "0")]:
            with self.subTest(bundle=bundle, name=name):
                self.checkpoint(bundle)
                self.env["CHOPINFER_" + name] = value
                self.calls.write_text("")
                result = self.run_entrypoint()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("CHOPINFER_" + name, result.stderr)
                self.assertFalse(self.served())
                del self.env["CHOPINFER_" + name]

    def test_taps_are_refused_before_preparation(self):
        # The image serves; its bundles refuse to start with taps set.
        names = ["TAPS_DIR", "TAPS_LAYERS", "TAPS_SITES", "TAPS_FROM", "TAPS_BYTES", "TAPS_SCOPE_JSON",
                 "QWEN38_DUMP_DIR", "QWEN38_DUMP_OPS", "QWEN38_DUMP_TAPS", "QWEN38_DUMP_FROM",
                 "QWEN38_DUMP_BYTES", "QWEN38_DUMP_SCOPE_JSON"]
        for bundle in (QWEN, BUNDLE):
            for name in names:
                with self.subTest(bundle=bundle, name=name):
                    self.checkpoint(bundle)
                    self.env["CHOPINFER_" + name] = "1"
                    result = self.run_entrypoint()
                    del self.env["CHOPINFER_" + name]
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(f"CHOPINFER_{name}: per-layer taps need a bundle built with its "
                                  "diagnostic-taps feature", result.stderr)
                    self.assertFalse(self.calls.exists())

    def test_disabled_speculation_rejects_unused_policy(self):
        self.checkpoint(BUNDLE, "dflash")
        self.env.update(CHOPINFER_SPECULATIVE_TOKENS="0", CHOPINFER_SPECULATION="fixed")
        result = self.run_entrypoint()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CHOPINFER_SPECULATION", result.stderr)

    def test_package_manifest_symlink_is_not_followed(self):
        self.checkpoint(BUNDLE, manifest_symlink=True)
        self.assertNotEqual(self.run_entrypoint().returncode, 0)
        self.assertFalse(self.served())

    def test_qwen_defaults(self):
        self.checkpoint(QWEN, "mtp")
        self.serve_arguments()
        serve = self.commands()[-1]
        self.assertEqual(serve[serve.index("--model") + 1], str(self.model))
        for flag in ('--prefix-cache', '--kv-memory-fraction', '--max-sequences',
                     '--max-batch-tokens', '--prefill-chunk-tokens',
                     '--speculative-tokens', '--speculation', '--draft-cost'):
            self.assertNotIn(flag, serve)
        self.assertNotIn("--queue-depth", serve)
        # Preparing again reuses the model cache.
        self.calls.write_text("")
        self.assertEqual(self.run_entrypoint().returncode, 0)
        self.assertEqual(self.commands()[-1], serve)

    def test_qwen_manifest_drafter_and_ar_overrides(self):
        # A drafter kind without a bundle drafter does not enable MTP.
        self.checkpoint(QWEN, "mtp", bundle_drafter=False)
        self.assertEqual(self.serve_arguments(), [])
        self.checkpoint(QWEN, "mtp")
        self.assertEqual(self.serve_arguments(CHOPINFER_SPECULATIVE_TOKENS="0"),
                         ["--speculative-tokens", "0"])
        self.env.pop("CHOPINFER_SPECULATIVE_TOKENS")
        self.assertEqual(self.serve_arguments("--speculative-tokens", "0"),
                         ["--speculative-tokens", "0"])
        self.assertEqual(self.serve_arguments(CHOPINFER_SPECULATION="fixed"),
                         ["--speculation", "fixed"])

    def test_qwen_overrides(self):
        self.checkpoint(QWEN)
        self.serve_arguments("--prefix-cache", "off", "--modalities", "text",
                             CHOPINFER_MAX_SEQUENCES="16", CHOPINFER_KV_MEMORY_FRACTION="0.4")
        serve = self.commands()[-1]
        self.assertEqual(serve.count("--prefix-cache"), 1)
        self.assertEqual(serve.count("--modalities"), 1)
        self.assertEqual(serve[serve.index("--max-sequences") + 1], "16")
        self.assertEqual(serve[serve.index("--kv-memory-fraction") + 1], "0.4")

    def test_expected_bundle_identity(self):
        self.env['CHOPINFER_EXPECTED_BUNDLE_ID'] = BUNDLE
        self.checkpoint(QWEN)
        result = self.run_entrypoint()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('expected ' + BUNDLE, result.stderr)
        self.assertIn(QWEN, result.stderr)
        self.assertFalse(self.served())
        self.checkpoint(BUNDLE)
        self.assertEqual(self.run_entrypoint().returncode, 0)

    def test_unknown_or_invalid_prepared_package_never_serves(self):
        for bundle, fields in [("unknown.tp4.v1", {"family": "mimo26"}),
                               ("../../bad", {"family": "mimo26"}),
                               (BUNDLE, {"raw_reply": "invalid"}),
                               (BUNDLE, {"reply": {"result": "error"}}),
                               (BUNDLE, {"reply": {"bundle_id": QWEN}}),
                               (BUNDLE, {"manifest": {"speculation": {"bundle_drafter": True,
                                                                      "drafter_kind": "../x"}}}),
                               (BUNDLE, {"manifest": {"speculation": {"bundle_drafter": True,
                                                                      "drafter_kind": "unknown"}}}),
                               # The package's policy is read only as its
                               # manifest declares it.
                               (BUNDLE, {"policy_sha256": "0" * 64}),
                               (BUNDLE, {"policy_symlink": True}),
                               (BUNDLE, {"policy": "{}"}),
                               (QWEN, {"policy": (ROOT / "models/mimo26/deploy/policy.json").read_text()})]:
            with self.subTest(bundle=bundle, fields=fields):
                self.checkpoint(bundle)
                fake = json.loads((self.model / "fake.json").read_text())
                (self.model / "fake.json").write_text(json.dumps({**fake, **fields}))
                self.calls.write_text("")
                self.assertNotEqual(self.run_entrypoint().returncode, 0)
                self.assertFalse(self.served())


if __name__ == "__main__":
    unittest.main()

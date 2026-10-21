#!/usr/bin/env python3
"""CPU process coverage of the SDK helper and the generic `serve --model` route,
which loads the checkpoint directly through a package without rank images."""
import hashlib
import json
import re
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "debug"
# How long any one helper or host run, HTTP request or launch may take before
# the test fails as hung. It bounds a hang; no test checks a duration with it.
# The 20 tests take 60 to 80 s on a quiet host. Full gates on a shared 48-core
# host (2026-10-15) ran them in up to 1,545 s, and stopped them at the earlier
# 60 s and 120 s limits in rel-r1 (740 s) and rcfA-r1 (1,196 s): a step that
# takes seconds alone took minutes. Launches took 0.8 to 4.4 s at load 120,
# and one stalled past 60 s in a busy gate (2026-10-11). 20 minutes, the
# limit the gate's load-bound Rust tests have, is 10 times the longest
# earlier limit.
HANG_SECONDS = 20 * 60
# A 1x1 RGB PNG, the size the fixture's image recipe admits.
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"


class PreparationProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="model-preparation-process-")
        cls.root = Path(cls.temporary.name)
        cls.install = cls.root / "install"
        cls.install.mkdir(mode=0o755)
        for source, name in [(TARGET / "chopinfer", "chopinfer"),
                             (TARGET / "examples/chopinfer-model-fixture", "chopinfer-model-fixture")]:
            if not source.is_file():
                raise AssertionError(f"build required CPU fixture executable first: {source.name}")
            shutil.copy2(source, cls.install / name)
            (cls.install / name).chmod(0o755)
        cls.helper = cls.install / "chopinfer-model-fixture"
        cls.host = cls.install / "chopinfer"
        cls.checkpoint = cls.root / "checkpoint with spaces"
        subprocess.run([cls.helper, "fixture-source", cls.checkpoint], check=True, timeout=HANG_SECONDS)
        native = cls.root / "native"
        (native / "src").mkdir(parents=True)
        note = subprocess.check_output([cls.helper, "fixture-note"], timeout=HANG_SECONDS)
        (native / "note").write_bytes(note)
        dependency = json.dumps(str(ROOT / "crates/bundle-tp4-fixture"))
        (native / "Cargo.toml").write_text(
            '[package]\nname="fixture-preparation"\nversion="0.0.0"\nedition="2024"\n'
            '[workspace]\n[lib]\ncrate-type=["cdylib"]\n[dependencies]\n'
            f'bundle-tp4-fixture={{path={dependency},features=["fake-host-services"]}}\n')
        (native / "src/lib.rs").write_text(
            f'#[repr(C,align(4))] struct Note([u8;{len(note)}]);\n'
            '#[used]\n#[unsafe(link_section=".note.chopinfer.model")]\n'
            'static NOTE:Note=Note(*include_bytes!("../note"));\n'
            '#[unsafe(no_mangle)] pub extern "C" fn fixture_anchor() -> *const u8 {\n'
            '&bundle_tp4_fixture::BUNDLE_TP4_FLASH_API_V1 as *const _ as *const u8\n}\n')
        env = dict(os.environ, CARGO_TARGET_DIR=str(native / "target"))
        for args in [("generate-lockfile", "--offline"), ("build", "--locked", "--offline")]:
            result = subprocess.run(["cargo", *args], cwd=native, env=env, capture_output=True, text=True)
            if result.returncode:
                raise AssertionError(result.stderr)
        shutil.copy2(native / "target/debug/libfixture_preparation.so", cls.install)
        # No compiler or qualification tool may be called during preparation or serving.
        cls.environment = dict(os.environ, PATH=str(cls.install), CHOPINFER_DEV_SKIP_WARMUP="1")
        cls.environment.pop("CHOPINFER_BUNDLE_VERIFY_CACHE_DIR", None)
        cls.environment.pop("CHOPINFER_REFERENCE_TOKENS_FILE", None)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def helper_result(self, *args):
        result = subprocess.run([self.helper, *args], env=self.environment, text=True, capture_output=True, timeout=HANG_SECONDS)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def launch(self, checkpoint, cache, alias=None, cwd=None, extra=(), then=None):
        log = self.root / f"serve-{time.monotonic_ns()}.log"
        args = [self.host, "serve", "--model", checkpoint, "--model-cache-dir", cache,
                "--tensor-parallel-size", "4", "--listen", "127.0.0.1:0", "--fake-host-services",
                "--max-context-tokens", "128", "--max-sequences", "1", "--max-batch-tokens", "128",
                "--prefill-chunk-tokens", "128", "--prefix-cache", "off", "--verify-warmup", "off", *extra]
        if alias:
            args += ["--served-model-name", alias]
        with log.open("w") as output:
            process = subprocess.Popen(args, cwd=cwd, env=self.environment, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + HANG_SECONDS
            while time.monotonic() < deadline:
                text = log.read_text()
                events = [json.loads(line) for line in text.splitlines() if line.startswith('{"status":"listening"')]
                if events:
                    address = events[-1]["address"]
                    with urllib.request.urlopen(f"http://{address}/v1/models", timeout=HANG_SECONDS) as response:
                        models = json.load(response)
                    body = json.dumps({"model": models["data"][0]["id"], "prompt": "hello", "max_tokens": 1, "temperature": 0}).encode()
                    request = urllib.request.Request(f"http://{address}/v1/completions", body, {"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=HANG_SECONDS) as response:
                        completion = json.load(response)
                    self.assertEqual(completion["usage"]["completion_tokens"], 1)
                    if then:
                        then(address, models["data"][0]["id"])
                    package = next(json.loads(line.split(": ", 1)[1]) for line in text.splitlines() if line.startswith("startup direct_package: "))
                    # Weights come from the checkpoint through the bundle's weight plan.
                    self.assertIn("startup direct_checkpoint:", text)
                    self.assertIn("startup weight_plan", text)
                    self.assertIn(package["checkpoint_identity"], text)
                    self.assertNotIn("startup preparation:", text)
                    self.assertIn("startup verify:", text)
                    self.assertIn("startup generation warmup: skipped (development)", text)
                    self.assertIn("startup readiness:", text)
                    self.assertEqual(package["qualification"], "compatible-not-evaluated")
                    # No rank image is materialized, packaged or cached.
                    bundle = Path(package["bundle"])
                    self.assertEqual(json.loads((bundle / "manifest.json").read_text()).get("compiled_images", []), [])
                    self.assertEqual([p for p in Path(package["cache_root"]).rglob("*.img")], [])
                    usage = self.helper_result("cache", "prune", "--cache-dir", package["cache_root"])
                    self.assertTrue(next(e for e in usage["entries"] if e["identity"] == "p-" + package["package_identity"])["protected"])
                    self.assertEqual([e["identity"] for e in usage["entries"] if e["identity"][:2] in ("w-", "s-")], [])
                    return package, models, text
                if process.poll() is not None:
                    self.fail(f"host exited before readiness:\n{text}")
                time.sleep(0.05)
            self.fail(f"host did not become ready:\n{log.read_text()}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=HANG_SECONDS)

    def refusal(self, checkpoint, *extra, cache=None):
        cache = cache or self.root / f"refused-cache-{time.monotonic_ns()}"
        result = subprocess.run([self.host, "serve", "--model", checkpoint, "--model-cache-dir", cache,
                                 "--fake-host-services", *extra],
                                env=self.environment, text=True, capture_output=True, timeout=HANG_SECONDS)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('"status":"listening"', result.stdout)
        self.assertNotIn("startup readiness:", result.stderr)
        return result.stderr

    def copy_checkpoint(self, name):
        copy = self.root / name
        shutil.copytree(self.checkpoint, copy)
        return copy

    def test_model_options_list_defaults_and_refuse_before_weights(self):
        checkpoint = self.copy_checkpoint("options-checkpoint")
        config = json.loads((checkpoint / "config.json").read_text())
        config["scratch_multiplier"] = 1
        (checkpoint / "config.json").write_text(json.dumps(config))
        listed = subprocess.run([self.host, "serve", "--model", checkpoint,
                                 "--model-cache-dir", self.root / "options-cache",
                                 "--list-model-options"], env=self.environment,
                                text=True, capture_output=True, timeout=HANG_SECONDS)
        self.assertEqual(listed.returncode, 0, listed.stderr)
        options = json.loads(listed.stdout)
        self.assertEqual(options["schema"], "chop.model-options.v1")
        self.assertEqual(options["options"][0]["definition"],
                         {"type": "integer", "default": 1, "minimum": 1, "maximum": 4})
        self.assertNotIn("startup direct_load:", listed.stderr)
        for override, reason in (("unknown=1", "unknown model option"),
                                 ("scratch_multiplier=true", "wrong type"),
                                 ("scratch_multiplier=5", "allowed range")):
            refusal = self.refusal(checkpoint, "--model-option", override)
            self.assertIn(reason, refusal)
            self.assertNotIn("startup direct_load:", refusal)
        def metadata(address, _model):
            with urllib.request.urlopen(f"http://{address}/metrics", timeout=HANG_SECONDS) as response:
                metrics = response.read().decode()
            self.assertIn('chop_model_option_info{name="scratch_multiplier",value="1"} 1', metrics)
        _, models, log = self.launch(checkpoint, self.root / "options-default-cache", then=metadata)
        self.assertEqual(models["data"][0]["model_options"], {"scratch_multiplier": 1})
        self.assertFalse(models["data"][0]["model_options_qualified"])
        self.assertIn('"model_options":{"scratch_multiplier":1}', log)

    def test_direct_serving_selects_one_package_and_reuses_it_warm(self):
        cache = self.root / "handoff-cache"
        cold, models, _ = self.launch(self.checkpoint, cache, extra=("--load-threads", "2"))
        self.assertEqual(models["data"][0]["id"], "local-" + cold["checkpoint_identity"])
        self.assertEqual(cold["measurements"]["decisions"], ["emitted direct package"])
        warm, models, _ = self.launch(self.checkpoint, cache, "experiment-A")
        self.assertEqual(models["data"][0]["id"], "experiment-A")
        self.assertEqual(cold["package_identity"], warm["package_identity"])
        self.assertEqual(warm["measurements"]["decisions"], ["reused completed direct package"])
        # The same emission inputs reuse the package without emitting it again.
        self.assertIn("package_assembly", cold["measurements"]["stages"])
        self.assertEqual(set(warm["measurements"]["stages"]), {"package_reuse"})
        self.assertEqual(warm["profile"], cold["profile"])
        direct = self.helper_result("prepare", "--direct", "--model", str(self.checkpoint), "--cache-dir", str(cache))
        self.assertEqual(direct["package_identity"], warm["package_identity"])
        # A rebuilt host binary does not enter the package identity.
        with self.host.open("ab") as host:
            host.write(b"host-only-fixture-change")
        host_only, _, _ = self.launch(self.checkpoint, cache)
        self.assertEqual(host_only["package_identity"], warm["package_identity"])
        # The package serves only by loading a checkpoint directly.
        refused = subprocess.run([self.host, "serve", "--bundle", host_only["bundle"], "--fake-host-services"],
                                 env=self.environment, capture_output=True, text=True, timeout=HANG_SECONDS)
        self.assertNotEqual(refused.returncode, 0)
        self.assertNotIn('"status":"listening"', refused.stdout)
        self.assertIn("--checkpoint DIR", refused.stderr)

    def test_a_changed_native_module_emits_its_package_again(self):
        cache = self.root / "module-cache"
        prepare = ("prepare", "--direct", "--model", str(self.checkpoint), "--cache-dir", str(cache))
        first = self.helper_result(*prepare)
        self.assertEqual(first["measurements"]["decisions"], ["emitted direct package"])
        module = self.install / "libfixture_preparation.so"
        original = module.read_bytes()
        try:
            module.write_bytes(original + b"fixture-module-change")
            changed = self.helper_result(*prepare)
        finally:
            module.write_bytes(original)
        # The module is one of the emission's inputs: a package is emitted
        # for it, and the original module's package is still reused as is.
        self.assertEqual(changed["measurements"]["decisions"], ["emitted direct package"])
        self.assertNotEqual(changed["package_identity"], first["package_identity"])
        again = self.helper_result(*prepare)
        self.assertEqual(again["measurements"]["decisions"], ["reused completed direct package"])
        self.assertEqual(set(again["measurements"]["stages"]), {"package_reuse"})
        self.assertEqual(again["package_identity"], first["package_identity"])

    def test_each_emission_input_emits_its_own_package(self):
        checkpoint = self.copy_checkpoint("emission-inputs")
        cache = self.root / "emission-inputs-cache"

        def prepare(*extra, helper=None, env=None):
            result = subprocess.run([helper or self.helper, "prepare", "--direct", "--model", str(checkpoint),
                                     "--cache-dir", str(cache), *extra],
                                    env=env or self.environment, text=True, capture_output=True, timeout=HANG_SECONDS)
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

        def emitted(package):
            # Emission ran, whether or not its manifest was already published.
            self.assertIn("package_assembly", package["measurements"]["stages"], package["measurements"])
            return package

        def reused(package):
            self.assertEqual(package["measurements"]["decisions"], ["reused completed direct package"])
            self.assertEqual(set(package["measurements"]["stages"]), {"package_reuse"})
            return package

        first = emitted(prepare())
        self.assertEqual(reused(prepare())["package_identity"], first["package_identity"])
        # An environment variable the emission reads.
        owner = emitted(prepare(env=dict(self.environment, CHOPINFER_FIXTURE_OWNER="someone")))
        self.assertNotEqual(owner["package_identity"], first["package_identity"])
        # The activation policy the request names.
        emitted(prepare("--activation-policy", "bf16"))
        # Another helper executable, installed beside the same module.
        rebuilt = self.install / "chopinfer-model-fixture-rebuilt"
        shutil.copy2(self.helper, rebuilt)
        try:
            with rebuilt.open("ab") as helper:
                helper.write(b"fixture-helper-change")
            emitted(prepare(helper=rebuilt))
        finally:
            rebuilt.unlink()
        # A metadata file the emission may read. The fixture's package does
        # not depend on it: the same package is published, and its record
        # names both inputs, so each reuses it.
        generation = checkpoint / "generation_config.json"
        generation.write_text('{"temperature": 0.5}')
        self.assertEqual(emitted(prepare())["package_identity"], first["package_identity"])
        self.assertEqual(reused(prepare())["package_identity"], first["package_identity"])
        generation.unlink()
        self.assertEqual(reused(prepare())["package_identity"], first["package_identity"])
        # A safetensors header, with the payload unchanged.
        shard = checkpoint / "rank2.safetensors"
        original = shard.read_bytes()
        length = int.from_bytes(original[:8], "little")
        header = json.loads(original[8:8 + length])
        header["__metadata__"] = {"note": "changed"}
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        shard.write_bytes(len(encoded).to_bytes(8, "little") + encoded + original[8 + length:])
        headers = emitted(prepare())
        self.assertNotEqual(headers["checkpoint_identity"], first["checkpoint_identity"])
        shard.write_bytes(original)
        # The original inputs still reuse the first package.
        self.assertEqual(reused(prepare())["package_identity"], first["package_identity"])

    def test_a_divergent_profile_is_refused_cold_and_warm(self):
        # This checkpoint makes the fixture package another profile than it
        # names. Emission refuses it, so no package or record exists to
        # answer a later start: every start is refused alike.
        drifted = self.copy_checkpoint("recorded-profile")
        config = json.loads((drifted / "config.json").read_text())
        config["fixture_profile_drift"] = True
        (drifted / "config.json").write_text(json.dumps(config))
        cache = self.root / "recorded-profile-cache"
        prepared = subprocess.run([self.helper, "prepare", "--direct", "--model", str(drifted), "--cache-dir", str(cache)],
                                  env=self.environment, text=True, capture_output=True, timeout=HANG_SECONDS)
        self.assertNotEqual(prepared.returncode, 0, prepared.stdout)
        refusals = [prepared.stderr] + [self.refusal(drifted, cache=cache) for _ in range(2)]
        for stderr in refusals:
            self.assertIn("named serving profile tp4-mtp before packaging but packaged a different tp4-mtp",
                          stderr)
        self.assertEqual(list(cache.glob("p-*")), [])
        self.assertEqual(list((cache / "records").iterdir()), [])

    def test_concurrent_inputs_of_one_package_all_stay_recorded(self):
        # Two checkpoints that differ only in a metadata file the fixture's
        # package does not depend on: different inputs, one package. Their
        # concurrent preparations publish, learn and select the same record.
        first = self.copy_checkpoint("concurrent-first")
        second = self.copy_checkpoint("concurrent-second")
        (second / "generation_config.json").write_text('{"temperature": 0.5}')
        for attempt in range(10):
            cache = self.root / f"concurrent-cache-{attempt}"

            def start(checkpoint):
                return subprocess.Popen([self.helper, "prepare", "--direct", "--model", str(checkpoint),
                                         "--cache-dir", str(cache)], env=self.environment, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Each child joins the list as it starts, so a failed later start
            # still leaves every started one to the cleanup below.
            processes = []
            try:
                for checkpoint in [first, second] * 12:
                    processes.append(start(checkpoint))
                deadline = time.monotonic() + HANG_SECONDS
                results = [(process.communicate(timeout=max(0, deadline - time.monotonic())), process.returncode)
                           for process in processes]
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            for (_, stderr), returncode in results:
                self.assertEqual(returncode, 0, stderr)
            self.assertEqual(len({json.loads(stdout)["package_identity"] for (stdout, _), _ in results}), 1)
            # Every input set is still recorded: each reuses the package.
            for checkpoint in (first, second):
                reused = self.helper_result("prepare", "--direct", "--model", str(checkpoint),
                                            "--cache-dir", str(cache))
                self.assertEqual(set(reused["measurements"]["stages"]), {"package_reuse"},
                                 (attempt, checkpoint.name, reused["measurements"]))

    def test_a_module_replaced_during_emission_is_never_published_under_its_inputs(self):
        checkpoint = self.copy_checkpoint("module-race")
        marker = self.root / "module-race-replaced"
        config = json.loads((checkpoint / "config.json").read_text())
        config["fixture_replace_module_once"] = str(marker)
        (checkpoint / "config.json").write_text(json.dumps(config))
        cache = self.root / "module-race-cache"
        prepare = ["prepare", "--direct", "--model", str(checkpoint), "--cache-dir", str(cache)]
        module = self.install / "libfixture_preparation.so"
        original = module.read_bytes()
        try:
            # The emission's inputs name the original module; the installed
            # one is replaced before it is staged, so nothing is published.
            raced = subprocess.run([self.helper, *prepare], env=self.environment, text=True,
                                   capture_output=True, timeout=HANG_SECONDS)
            self.assertTrue(marker.exists())
            self.assertNotEqual(module.read_bytes(), original)
            self.assertNotEqual(raced.returncode, 0, raced.stdout)
            self.assertEqual(list(cache.glob("p-*")), [])
        finally:
            module.write_bytes(original)
        # The original module again: its own package, holding that module.
        package = self.helper_result(*prepare)
        self.assertEqual(package["measurements"]["decisions"], ["emitted direct package"])
        manifest = json.loads((Path(package["bundle"]) / "manifest.json").read_text())
        self.assertEqual(manifest["native_module"]["sha256"], hashlib.sha256(original).hexdigest())
        again = self.helper_result(*prepare)
        self.assertEqual(again["measurements"]["decisions"], ["reused completed direct package"])
        self.assertEqual(again["package_identity"], package["package_identity"])

    def test_a_rewritten_payload_passes_the_header_identity_and_fails_verify(self):
        checkpoint = self.copy_checkpoint("rewritten")
        shard = checkpoint / "rank2.safetensors"
        manifest = self.root / "rewritten-files.json"
        manifest.write_text(json.dumps({p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                        for p in checkpoint.iterdir()}))
        verified = subprocess.run([self.host, "verify", "--model", checkpoint, "--manifest", manifest],
                                  env=self.environment, capture_output=True, text=True, timeout=HANG_SECONDS)
        self.assertEqual(verified.returncode, 0, verified.stderr)
        data = bytearray(shard.read_bytes())
        data[-1] ^= 1
        shard.write_bytes(data)
        verified = subprocess.run([self.host, "verify", "--model", checkpoint, "--manifest", manifest],
                                  env=self.environment, capture_output=True, text=True, timeout=HANG_SECONDS)
        self.assertNotEqual(verified.returncode, 0)
        self.assertIn("MISMATCH rank2.safetensors", verified.stdout + verified.stderr)
        # Direct loading accepts the unchanged headers and serves the new bytes;
        # this fixture's module then refuses an image it cannot regenerate.
        stderr = self.refusal(checkpoint)
        self.assertIn("startup weight_plan", stderr)
        self.assertIn("startup/serving failure:", stderr)

    def test_unknown_architectures_missing_files_and_topologies_are_refused(self):
        unknown = self.root / "unknown-architecture"
        unknown.mkdir()
        (unknown / "config.json").write_text(json.dumps({"architectures": ["UnknownForCausalLM"]}))
        stderr = self.refusal(unknown)
        self.assertIn("no installed family implementation accepts", stderr)
        self.assertIn("UnknownForCausalLM", stderr)
        self.assertIn("chopinfer-model-fixture:", stderr)
        no_config = self.copy_checkpoint("no-config")
        (no_config / "config.json").unlink()
        self.assertIn("has no readable config.json", self.refusal(no_config))
        missing = self.copy_checkpoint("missing-shard")
        (missing / "rank3.safetensors").unlink()
        stderr = self.refusal(missing)
        self.assertIn("no installed family implementation accepts", stderr)
        self.assertIn("chopinfer-model-fixture: No such file or directory", stderr)
        # Metadata changed during emission must be refused before the
        # package can be published or selected for serving.
        racing = self.copy_checkpoint("racing-metadata")
        config = json.loads((racing / "config.json").read_text())
        config["fixture_change_after_emission"] = "generation_config.json"
        (racing / "config.json").write_text(json.dumps(config))
        racing_cache = self.root / "racing-metadata-cache"
        stderr = self.refusal(racing, cache=racing_cache)
        self.assertIn("checkpoint input directory changed during preparation", stderr)
        self.assertEqual(list(racing_cache.glob("p-*")), [])
        # A package whose profile differs from the one named before
        # packaging, under the same name, is refused.
        drifted = self.copy_checkpoint("drifted-profile")
        config = json.loads((drifted / "config.json").read_text())
        config["fixture_profile_drift"] = True
        (drifted / "config.json").write_text(json.dumps(config))
        stderr = self.refusal(drifted)
        self.assertIn("named serving profile tp4-mtp before packaging but packaged a different tp4-mtp",
                      stderr)
        stderr = self.refusal(self.checkpoint, "--tensor-parallel-size", "2")
        self.assertIn("--tensor-parallel-size 2", stderr)
        self.assertIn("CPU fixture needs TP4", stderr)
        self.assertIn("--refresh-model is gone", self.refusal(self.checkpoint, "--refresh-model"))

    @staticmethod
    def resolved_serve(text):
        """The serve configuration the host ran: the last resolved line."""
        return [line for line in text.splitlines() if line.startswith("resolved serve: ")][-1]

    def post(self, address, path, body):
        request = urllib.request.Request(f"http://{address}{path}", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=HANG_SECONDS) as response:
            return json.load(response)

    def test_model_serves_its_default_speculative_profile_and_media(self):
        # The family's default for this checkpoint and rank count is its MTP
        # profile, as the package path serves it; the package declares images.
        replies = {}

        def speculate_and_see(address, model):
            replies["text"] = self.post(address, "/v1/completions",
                                        {"model": model, "prompt": "hello", "max_tokens": 6, "temperature": 0})
            replies["image"] = self.post(address, "/v1/chat/completions", {
                "model": model, "max_tokens": 2, "temperature": 0,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + PNG}},
                    {"type": "text", "text": "what is this"}]}]})

        package, _, text = self.launch(self.checkpoint, self.root / "profile-cache", then=speculate_and_see)
        self.assertIn("startup direct_profile: tp4-mtp", text)
        resolved = self.resolved_serve(text)
        self.assertIn("speculative_tokens=3 speculation=Adaptive", resolved)
        self.assertIn("draft_cost=PerToken", resolved)
        self.assertEqual(package["profile"]["name"], "tp4-mtp")
        manifest = json.loads((Path(package["bundle"]) / "manifest.json").read_text())
        self.assertEqual(manifest["speculation"]["drafter_kind"], "mtp")
        self.assertGreater(replies["text"]["usage"]["completion_tokens"], 1)
        self.assertGreater(replies["image"]["usage"]["prompt_tokens"], 0)
        # Explicit options override the profile, as they do a --profile.
        _, _, text = self.launch(self.checkpoint, self.root / "profile-cache",
                                 extra=("--speculative-tokens", "0"))
        self.assertIn("speculative_tokens=0 speculation=off", self.resolved_serve(text))

    def test_the_helper_names_the_profile_serve_model_would_use(self):
        # `profile` answers the host's profile request from the command line,
        # for tools that must know the profile before serving; it prepares
        # nothing.
        checkpoints = {"tp4-mtp": self.checkpoint}
        for drafters, name in [("dflash,none", "tp4-dflash"), ("none", "tp4-ar")]:
            checkpoints[name] = self.root / f"profile-{name}"
            subprocess.run([self.helper, "fixture-source", checkpoints[name], drafters], check=True, timeout=HANG_SECONDS)
        for name, checkpoint in checkpoints.items():
            before = sorted(self.root.iterdir())
            result = subprocess.run([self.helper, "profile", "--model", checkpoint, "--tensor-parallel-size", "4"],
                                    env=self.environment, text=True, capture_output=True, timeout=HANG_SECONDS, check=True)
            reply = json.loads(result.stdout)
            self.assertEqual((reply["result"], reply["name"]), ("direct-profile-selected", name))
            self.assertEqual(reply["profile"]["tensor_parallel"], 4)
            self.assertEqual(sorted(self.root.iterdir()), before)
        result = subprocess.run([self.helper, "profile", "--model", self.checkpoint, "--cache-dir", self.root],
                                env=self.environment, text=True, capture_output=True, timeout=HANG_SECONDS)
        self.assertNotEqual(result.returncode, 0)

    def test_model_serves_a_dflash_profile_and_autoregressive_checkpoints(self):
        dflash = self.root / "dflash-checkpoint"
        subprocess.run([self.helper, "fixture-source", dflash, "dflash,none"], check=True, timeout=HANG_SECONDS)
        package, _, text = self.launch(dflash, self.root / "dflash-cache")
        self.assertIn("startup direct_profile: tp4-dflash", text)
        resolved = self.resolved_serve(text)
        self.assertIn("speculative_tokens=3 speculation=Adaptive", resolved)
        self.assertIn("draft_cost=PerBlock", resolved)
        self.assertIn("Tbt", resolved)
        manifest = json.loads((Path(package["bundle"]) / "manifest.json").read_text())
        self.assertEqual(manifest["speculation"]["drafter_kind"], "dflash")
        plain = self.root / "ar-checkpoint"
        subprocess.run([self.helper, "fixture-source", plain, "none"], check=True, timeout=HANG_SECONDS)
        package, _, text = self.launch(plain, self.root / "ar-cache")
        self.assertIn("startup direct_profile: tp4-ar", text)
        self.assertIn("speculative_tokens=0 speculation=off", self.resolved_serve(text))

    def test_a_damaged_or_half_published_direct_package_is_replaced(self):
        cache = self.root / "damaged-cache"
        first, _, _ = self.launch(self.checkpoint, cache)
        bundle = Path(first["bundle"])
        # A component changed after publication: reuse would fail to serve.
        tokenizer = next(p for p in (bundle / "components").iterdir() if "tokenizer" in p.name)
        tokenizer.write_bytes(tokenizer.read_bytes() + b" ")
        replaced, _, _ = self.launch(self.checkpoint, cache)
        self.assertEqual(replaced["package_identity"], first["package_identity"])
        self.assertTrue(replaced["measurements"]["decisions"][0].startswith("replaced a damaged direct package"),
                        replaced["measurements"]["decisions"])
        self.assertTrue(any(p.name.startswith("q-") for p in cache.iterdir()))
        # A package directory whose record was never written (a publication
        # cut short) is replaced as well.
        (cache / "records" / f"p-{first['package_identity']}.json").unlink()
        recovered, _, _ = self.launch(self.checkpoint, cache)
        self.assertTrue(recovered["measurements"]["decisions"][0].startswith("replaced a damaged direct package"),
                        recovered["measurements"]["decisions"])
        # So is a package missing one of its files under an intact record.
        next((bundle / "components").iterdir()).unlink()
        partial, _, _ = self.launch(self.checkpoint, cache)
        self.assertTrue(partial["measurements"]["decisions"][0].startswith("replaced a damaged direct package"),
                        partial["measurements"]["decisions"])
        warm, _, _ = self.launch(self.checkpoint, cache)
        self.assertEqual(warm["measurements"]["decisions"], ["reused completed direct package"])

    def test_a_helper_without_direct_loading_is_named_at_discovery(self):
        install = self.root / "old-helper"
        install.mkdir(mode=0o755)
        shutil.copy2(self.host, install / "chopinfer")
        # A helper of the protocol before direct loading: it accepts the
        # checkpoint but cannot decode the direct request.
        old = install / "chopinfer-model-old"
        old.write_text(f"""#!{sys.executable}
import json, sys
envelope = json.loads(sys.stdin.readline())
request = envelope["request"]
if envelope["version"] != 1:
    reply = {{"result": "error", "message": "unsupported preparation protocol version", "measurements": {{"total_seconds": 0, "stages": {{}}, "decisions": []}}}}
elif request["command"] == "describe":
    reply = {{"result": "description", "version": 1, "provider": "chopinfer-model-old", "profiles": ["old"]}}
elif request["command"] == "probe":
    reply = {{"result": "probe", "compatible": True, "reason": "compatible metadata"}}
else:
    reply = {{"result": "error", "message": "unknown variant", "measurements": {{"total_seconds": 0, "stages": {{}}, "decisions": []}}}}
print(json.dumps(reply), flush=True)
""")
        old.chmod(0o755)
        result = subprocess.run([install / "chopinfer", "serve", "--model", self.checkpoint, "--model-cache-dir",
                                 self.root / "old-helper-cache", "--fake-host-services"],
                                env=self.environment, text=True, capture_output=True, timeout=HANG_SECONDS)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("chopinfer-model-old", result.stderr)
        self.assertIn("predates direct loading", result.stderr)
        self.assertFalse((self.root / "old-helper-cache").exists())

    def test_a_host_of_protocol_version_1_still_reads_this_helper(self):
        # What a host from before direct loading sends and strictly parses.
        def exchange(version, request):
            envelope = json.dumps({"version": version, "request": request}) + "\n"
            result = subprocess.run([self.helper, "--protocol"], input=envelope, env=self.environment,
                                    text=True, capture_output=True, timeout=HANG_SECONDS)
            return json.loads(result.stdout.splitlines()[0])
        described = exchange(1, {"command": "describe"})
        self.assertEqual(set(described), {"result", "version", "provider", "profiles"})
        self.assertEqual((described["result"], described["version"]), ("description", 1))
        probed = exchange(1, {"command": "probe", "path": str(self.checkpoint), "tensor_parallel_size": 4})
        self.assertEqual((probed["result"], probed["compatible"]), ("probe", True))
        # A current host asks with version 2 and learns the capability.
        described = exchange(2, {"command": "describe"})
        self.assertEqual(described["version"], 2)
        self.assertIn("prepare-direct", described["capabilities"])
        self.assertEqual(exchange(3, {"command": "describe"})["result"], "error")

    def test_options_the_profile_refuses_are_refused_before_package_work(self):
        # The default MTP profile has a fixed prefill budget, so a TBT target
        # is refused; the helper names the profile without emitting anything.
        cache = self.root / "profile-refused-cache"
        result = subprocess.run([self.host, "serve", "--model", self.checkpoint, "--model-cache-dir", cache,
                                 "--fake-host-services", "--decode-tbt-target-ms", "100"],
                                env=self.environment, text=True, capture_output=True, timeout=HANG_SECONDS)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --prefill-budget tbt", result.stderr)
        self.assertNotIn("startup direct_package", result.stderr)
        self.assertFalse(cache.exists())
        # An autoregressive profile has no draft length for --speculation.
        plain = self.root / "profile-refused-ar"
        subprocess.run([self.helper, "fixture-source", plain, "none"], check=True, timeout=HANG_SECONDS)
        result = subprocess.run([self.host, "serve", "--model", plain, "--model-cache-dir", cache,
                                 "--fake-host-services", "--speculation", "fixed"],
                                env=self.environment, text=True, capture_output=True, timeout=HANG_SECONDS)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires positive --speculative-tokens", result.stderr)
        self.assertFalse(cache.exists())

    def test_a_helper_without_the_profile_request_still_serves(self):
        # A version 2 helper from before the profile request: the host skips
        # the early check, serves, and still refuses after packaging.
        environment = dict(self.environment, CHOPINFER_FIXTURE_NO_DIRECT_PROFILE="1")
        saved, self.environment = self.environment, environment
        try:
            package, _, _ = self.launch(self.checkpoint, self.root / "no-profile-cache")
        finally:
            self.environment = saved
        self.assertEqual(package["profile"]["name"], "tp4-mtp")
        cache = self.root / "no-profile-refused-cache"
        result = subprocess.run([self.host, "serve", "--model", self.checkpoint, "--model-cache-dir", cache,
                                 "--fake-host-services", "--decode-tbt-target-ms", "100"],
                                env=environment, text=True, capture_output=True, timeout=HANG_SECONDS)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --prefill-budget tbt", result.stderr)
        self.assertIn("startup direct_package", result.stderr)

    def test_readiness_counts_package_selection(self):
        slow = self.copy_checkpoint("slow-emission")
        config = json.loads((slow / "config.json").read_text())
        config["fixture_emit_delay_ms"] = 3000
        (slow / "config.json").write_text(json.dumps(config))
        _, _, text = self.launch(slow, self.root / "slow-emission-cache")
        readiness = float(re.search(r"^startup readiness: ([0-9.]+)s", text, re.M)[1])
        self.assertGreaterEqual(readiness, 3.0, text)

    def test_relative_model_and_cache_paths_resolve_from_calling_directory(self):
        cache = self.root / "relative cache"
        relative, _, _ = self.launch(Path(self.checkpoint.name), Path(cache.name), cwd=self.root)
        self.assertEqual(relative["cache_root"], str(cache))
        self.assertFalse((self.install / cache.name).exists())
        absolute, _, _ = self.launch(self.checkpoint, cache)
        self.assertEqual(relative["package_identity"], absolute["package_identity"])
        self.assertEqual(absolute["measurements"]["decisions"], ["reused completed direct package"])

    def test_discovery_refuses_unsafe_missing_and_ambiguous_providers(self):
        for scenario in ["missing", "ambiguous", "writable", "symlink"]:
            install = self.root / scenario
            install.mkdir(mode=0o755)
            host = install / "chopinfer"
            shutil.copy2(self.host, host)
            if scenario == "symlink":
                (install / "chopinfer-model-fixture").symlink_to(self.helper)
            elif scenario != "missing":
                shutil.copy2(self.helper, install / "chopinfer-model-fixture")
                (install / "chopinfer-model-fixture").chmod(0o755)
            if scenario == "ambiguous":
                shutil.copy2(self.helper, install / "chopinfer-model-second")
            if scenario == "writable":
                install.chmod(0o777)
            result = subprocess.run([host, "serve", "--model", self.checkpoint, "--model-cache-dir", self.root / "unused-cache"],
                                    cwd=self.checkpoint, env=self.environment, text=True, capture_output=True, timeout=HANG_SECONDS)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('"status":"listening"', result.stdout)
            expected = {"missing": "no installed family implementation accepts", "ambiguous": "ambiguous", "writable": "group/world writable", "symlink": "unsafe installed"}[scenario]
            self.assertIn(expected, result.stderr)
            self.assertFalse((self.root / "unused-cache").exists())
            install.chmod(0o755)


if __name__ == "__main__":
    unittest.main()

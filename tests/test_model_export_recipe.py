"""Check the export recipe's synthesis, recorded-format and data-only guards."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zlib

import model_export_recipe as recipe
from model_export_recipe import (ADOPTIONS, LOCK, SCHEDULES, assert_export_diff, assert_recorded_formats,
                                 decode_component, read_header, rule_of, synthesize)

ROOT = Path(__file__).resolve().parents[1]
TABLE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


def setUpModule():
    # Fixture repositories and the recipe under test must not inherit the
    # calling Git's repository: `git rebase -x` in a linked worktree exports
    # GIT_DIR, and a fixture's `git init` and commits would write it.
    environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
    isolated = patch.dict(os.environ, environment, clear=True)
    isolated.start()
    unittest.addModuleCleanup(isolated.stop)


def git(tree, *args):
    subprocess.run(["git", "-C", str(tree), *args], check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)


def commit(tree, *paths):
    git(tree, "add", *paths)
    git(tree, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "Fixture")


def ledger(records):
    return json.dumps(records, indent=2) + "\n"


class DataDiffTests(unittest.TestCase):
    LOCK_RECORDS = [{"name": "a", "digest": "1"}, {"name": "c", "digest": "3"}]
    ADOPTION_RECORDS = [{"symbol": "chop_a_v1", "generated": "chop_proj_a_v1", "entry_sha256": "0" * 64}]
    SCHEDULE_ROWS = [{"inputs": {"format": "nvfp4"}, "schedule": {"kind": "Grouped"}}]

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tree = Path(temp.name)
        git(self.tree, "init")
        self.before = {LOCK: ledger(self.LOCK_RECORDS), ADOPTIONS: ledger(self.ADOPTION_RECORDS),
                       SCHEDULES: ledger(self.SCHEDULE_ROWS)}
        for path, text in self.before.items():
            (self.tree / path).parent.mkdir(parents=True, exist_ok=True)
            (self.tree / path).write_text(text)
        source = self.tree / "models/qwen38/contract/src/lib.rs"
        source.parent.mkdir(parents=True)
        source.write_text("// fixture\n")
        # The repository's own ignore rules decide what an untracked file is.
        (self.tree / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
        commit(self.tree, *self.before, "models/qwen38/contract/src/lib.rs", ".gitignore")

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return assert_export_diff(self.tree, self.before)

    def write(self, path, text):
        (self.tree / path).write_text(text)

    def test_an_unchanged_tree_and_appended_records_pass(self):
        self.assertEqual(self.check(), ([], []))
        self.write(LOCK, ledger([self.LOCK_RECORDS[0], {"name": "b"}, self.LOCK_RECORDS[1]]))
        added = {"symbol": "chop_b_v1", "generated": "chop_proj_b_v1", "entry_sha256": "1" * 64}
        self.write(ADOPTIONS, ledger([*self.ADOPTION_RECORDS, added]))
        lock_added, adopted = self.check()
        self.assertEqual(lock_added, ["b"])
        self.assertEqual([symbol for symbol, _ in adopted], ["chop_b_v1"])

    def test_source_edits_fail(self):
        for path in ("models/qwen38/contract/src/lib.rs", "models/qwen38/bundle/src/new.rs", "tests/new.py",
                     "models/qwen38/bundle/kernels/new.cu"):
            with self.subTest(path=path):
                existed = (self.tree / path).exists()
                original = (self.tree / path).read_text() if existed else None
                (self.tree / path).parent.mkdir(parents=True, exist_ok=True)
                (self.tree / path).write_text("// changed\n")
                with self.assertRaisesRegex(AssertionError, "unexpected data-only paths"):
                    self.check()
                git(self.tree, "add", path)
                with self.assertRaisesRegex(AssertionError, "unexpected data-only paths"):
                    self.check()
                git(self.tree, "reset", "-q", "--", path)
                if existed:
                    (self.tree / path).write_text(original)
                else:
                    (self.tree / path).unlink()
                self.check()

    def test_ignored_source_additions_fail(self):
        for path in ("target/generated_override.rs", "models/qwen38/bundle/__pycache__/build.rs",
                     "tools/__pycache__/injected.py", "tests/injected.py.log", "local.env"):
            with self.subTest(path=path):
                git(self.tree, "check-ignore", "-q", path)
                (self.tree / path).parent.mkdir(parents=True, exist_ok=True)
                (self.tree / path).write_text("// changed\n")
                with self.assertRaisesRegex(AssertionError, "unexpected data-only paths"):
                    self.check()
                (self.tree / path).unlink()
                self.check()

    def test_existing_records_must_keep_their_bytes_and_order(self):
        rewritten = [{**self.LOCK_RECORDS[0], "digest": "2"}, self.LOCK_RECORDS[1]]
        cases = [
            (LOCK, ledger(rewritten), "pre-existing lock record"),
            (LOCK, ledger(self.LOCK_RECORDS[::-1]), "pre-existing lock record"),
            (LOCK, ledger(self.LOCK_RECORDS[:1]), "pre-existing lock record"),
            (LOCK, ledger(self.LOCK_RECORDS).replace('"digest": "1"', '"digest":  "1"'),
             "pre-existing lock record"),
            (LOCK, ledger([*self.LOCK_RECORDS, self.LOCK_RECORDS[0]]), "duplicate lock records"),
            (ADOPTIONS, ledger([]), "pre-existing adoption record"),
            (ADOPTIONS, ledger([{**self.ADOPTION_RECORDS[0], "entry_sha256": "f" * 64}]),
             "pre-existing adoption record"),
            (ADOPTIONS, ledger([{"symbol": "chop_z_v1", "generated": "chop_proj_z_v1", "entry_sha256": "2" * 64},
                                *self.ADOPTION_RECORDS]), "pre-existing adoption record"),
            (SCHEDULES, ledger([*self.SCHEDULE_ROWS, *self.SCHEDULE_ROWS]), "unexpected data-only paths"),
        ]
        for path, text, message in cases:
            with self.subTest(path=path, message=message, text=text):
                self.write(path, text)
                with self.assertRaisesRegex(AssertionError, message):
                    self.check()
                self.write(path, self.before[path])
                self.check()


def reference_checkpoint(root):
    """A miniature of the geometry generator's FP8 layout: one linear and
    one full-attention layer, routed experts with block scales, an FP8
    lookup table and the MTP layer."""
    layers = "model.language_model.layers"
    tensors = {
        f"{layers}.0.linear_attn.in_proj_qkv.weight": ("BF16", [64, 32]),
        f"{layers}.0.linear_attn.norm.weight": ("BF16", [16]),
        f"{layers}.1.self_attn.q_proj.weight": ("BF16", [64, 32]),
        f"{layers}.1.self_attn.indexer.index_qk_proj.weight": ("BF16", [32, 32]),
        f"{layers}.1.mlp.shared_expert.down_proj.weight": ("BF16", [32, 64]),
        f"{layers}.1.mlp.experts.0.gate_proj.weight": ("F8_E4M3", [64, 32]),
        f"{layers}.1.mlp.experts.0.gate_proj.weight_scale_inv": ("BF16", [1, 1]),
        f"{TABLE}.shard_0.weight": ("F8_E4M3", [8, 32]),
        f"{TABLE}.weight_scale": ("BF16", [1]),
        "mtp.layers.0.mlp.experts.0.up_proj.weight": ("F8_E4M3", [64, 32]),
        "mtp.layers.0.mlp.experts.0.up_proj.weight_scale_inv": ("BF16", [1, 1]),
        "mtp.layers.0.self_attn.q_proj.weight": ("BF16", [64, 32]),
    }
    root.mkdir()
    recipe.write_sparse(root / "model.safetensors", tensors)
    config = {"architectures": ["Fixture"], "quantization_config": {"quant_method": "fp8"},
              "text_config": {"layer_types": ["linear_attention", "full_attention"]}}
    (root / "config.json").write_text(json.dumps(config))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text("{}")
    return tensors


class SynthesisTests(unittest.TestCase):
    def test_rules_bind_the_nearest_module(self):
        rules = {"a.1.experts": {"quant_algo": "NVFP4"}, "a.1.experts.0.up": {"quant_algo": "MXFP8"}}
        self.assertEqual(rule_of(rules, "a.1.experts.0.gate"), "NVFP4")
        self.assertEqual(rule_of(rules, "a.1.experts.0.up"), "MXFP8")
        self.assertIsNone(rule_of(rules, "a.10.experts.0.gate"))
        self.assertIsNone(rule_of(rules, "a.1.experts_gate"))

    def test_the_export_encodes_each_rule_and_keeps_other_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            reference = Path(temp) / "reference"
            reference_checkpoint(reference)
            with contextlib.redirect_stdout(io.StringIO()):
                rules = synthesize(reference, Path(temp) / "export")
            export = Path(temp) / "export"
            header = read_header(export / "model.safetensors")
            layers = "model.language_model.layers"
            stored = {name: (entry["dtype"], entry["shape"]) for name, entry in header.items()}
            self.assertEqual(stored, {
                f"{layers}.0.linear_attn.in_proj_qkv.weight": ("BF16", [64, 32]),
                f"{layers}.0.linear_attn.norm.weight": ("BF16", [16]),
                f"{layers}.1.self_attn.q_proj.weight": ("F8_E4M3", [64, 32]),
                f"{layers}.1.self_attn.q_proj.weight_scale": ("U8", [64, 1]),
                f"{layers}.1.self_attn.indexer.index_qk_proj.weight": ("F8_E4M3", [32, 32]),
                f"{layers}.1.self_attn.indexer.index_qk_proj.weight_scale": ("U8", [32, 1]),
                f"{layers}.1.mlp.shared_expert.down_proj.weight": ("F8_E4M3", [32, 64]),
                f"{layers}.1.mlp.shared_expert.down_proj.weight_scale": ("U8", [32, 2]),
                f"{layers}.1.mlp.experts.0.gate_proj.weight": ("U8", [64, 16]),
                f"{layers}.1.mlp.experts.0.gate_proj.weight_scale": ("F8_E4M3", [64, 2]),
                f"{layers}.1.mlp.experts.0.gate_proj.weight_scale_2": ("F32", []),
                f"{layers}.1.mlp.experts.0.gate_proj.input_scale": ("F32", []),
                f"{TABLE}.shard_0.weight": ("U8", [8, 16]),
                f"{TABLE}.shard_0.weight_scale": ("F8_E4M3", [8, 2]),
                f"{TABLE}.weight_scale_2": ("F32", [1]),
                "mtp.layers.0.mlp.experts.0.up_proj.weight": ("U8", [64, 16]),
                "mtp.layers.0.mlp.experts.0.up_proj.weight_scale": ("F8_E4M3", [64, 2]),
                "mtp.layers.0.mlp.experts.0.up_proj.weight_scale_2": ("F32", []),
                "mtp.layers.0.self_attn.q_proj.weight": ("BF16", [64, 32]),
            })
            ends = sorted(entry["data_offsets"] for entry in header.values())
            self.assertEqual([start for start, _ in ends][1:], [end for _, end in ends][:-1])
            payload = ends[-1][1]
            index = json.loads((export / "model.safetensors.index.json").read_text())
            self.assertEqual(index["metadata"]["total_size"], payload)
            self.assertEqual(set(index["weight_map"]), set(header))
            config = json.loads((export / "config.json").read_text())
            quantization = config["quantization_config"]
            self.assertEqual(json.loads((export / "hf_quant_config.json").read_text()), quantization)
            self.assertEqual(quantization["quantized_layers"], rules)
            self.assertNotIn(f"{layers}.0.linear_attn.in_proj_qkv", rules)
            self.assertNotIn(f"{layers}.0.self_attn.q_proj", rules)
            self.assertEqual(rules[f"{layers}.1.mlp.experts"]["quant_algo"], "NVFP4")
            self.assertEqual(rules["mtp.layers.0.mlp.experts"]["quant_algo"], "W4A16_NVFP4")
            self.assertIn("mtp.layers.0.self_attn*", quantization["ignore"])

    def test_an_fp8_weight_without_a_rule_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            reference = Path(temp) / "reference"
            tensors = reference_checkpoint(reference)
            tensors["model.language_model.layers.0.linear_attn.in_proj_z.weight"] = ("F8_E4M3", [64, 32])
            recipe.write_sparse(reference / "model.safetensors", tensors)
            with self.assertRaisesRegex(AssertionError, "no rule for an FP8 reference weight"):
                with contextlib.redirect_stdout(io.StringIO()):
                    synthesize(reference, Path(temp) / "export")


def component(document):
    encoded = json.dumps(document).encode()
    compressed = zlib.compress(encoded)
    return recipe.COMPONENT_MAGIC + struct.pack("<II", len(encoded), len(compressed)) + compressed


class RecordedFormatTests(unittest.TestCase):
    RULES = {"m.0.q": {"quant_algo": "MXFP8"}, "m.0.experts": {"quant_algo": "NVFP4"},
             "mtp.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4"}}

    def operation(self, name, *formats):
        return {"source": {"id": name, "weights": [
            {"codes": {"tensor": f"{name}{suffix}.weight"}, "format": format}
            for suffix, format in zip(("", "_b"), formats)]}}

    def document(self):
        return {"version": 1, "operations": [
            self.operation("m.0.q", "Mxfp8"), self.operation("m.0.linear", "Bf16"),
            self.operation("m.0.experts.gate_up_proj", "Nvfp4", "Nvfp4"),
            self.operation("mtp.layers.0.mlp.experts.down_proj", "Nvfp4"),
            self.operation("mtp.layers.0.self_attn.q_proj", "Bf16")]}

    def check(self, document):
        with contextlib.redirect_stdout(io.StringIO()):
            assert_recorded_formats(document, self.RULES, "fixture-tp2")

    def test_components_decode_and_formats_follow_the_rules(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "precision.fixture-tp2.bin"
            path.write_bytes(component(self.document()))
            self.check(decode_component(path))
            data = path.read_bytes()
            for broken in (b"CHOPPRC0" + data[8:], data[:-1], data[:8] + struct.pack("<I", 1) + data[12:]):
                path.write_bytes(broken)
                with self.assertRaises(Exception):
                    decode_component(path)

    def test_a_format_the_rules_do_not_assign_is_refused(self):
        for index, format in ((0, "Bf16"), (1, "Mxfp8"), (3, "Bf16")):
            with self.subTest(index=index, format=format):
                document = self.document()
                document["operations"][index]["source"]["weights"][0]["format"] = format
                with self.assertRaisesRegex(AssertionError, "the export makes it"):
                    self.check(document)

    def test_the_mix_and_the_mtp_layer_must_be_recorded(self):
        document = self.document()
        del document["operations"][0]
        with self.assertRaises(AssertionError):
            self.check(document)
        document = self.document()
        del document["operations"][3]
        with self.assertRaises(AssertionError):
            self.check(document)


class CreateArgumentTests(unittest.TestCase):
    """create runs a profile as serve does, with its drafter when it has one."""

    def manifest(self, narrow=True, other=False):
        capabilities = ["chop.speculative-narrow-rows.v1"] if narrow else []
        requirements = {"required_capabilities": capabilities}
        return {"speculation": {"maximum_draft_tokens": 7, "maximum_verify_tokens": 256},
                "resource_requirements": {"required_capabilities": [] if other else capabilities},
                "topology_resources": [{"topology": "fixture-tp4", "resource_requirements": requirements}]}

    def profile(self, drafter="mtp", sequences=32, topology="fixture-tp2"):
        return {"topology": topology, "max_sequences": sequences,
                "speculation": {"mode": "adaptive", "tokens": 3, "drafter": drafter}}

    def test_a_drafter_profile_creates_with_its_tokens_and_serves_rows(self):
        self.assertEqual(recipe.create_arguments(self.profile(), self.manifest()),
                         ["--max-sequences", "32", "--speculative-tokens", "3", "--verify-tokens", "32"])
        self.assertEqual(recipe.create_arguments(self.profile(sequences=64), self.manifest()),
                         ["--max-sequences", "64", "--speculative-tokens", "3", "--verify-tokens", "256"])
        self.assertEqual(recipe.create_arguments(self.profile(), self.manifest(narrow=False)),
                         ["--max-sequences", "32", "--speculative-tokens", "3", "--verify-tokens", "256"])
        # Another topology's resources decide its rows.
        self.assertEqual(recipe.create_arguments(self.profile(topology="fixture-tp4"), self.manifest(other=True)),
                         ["--max-sequences", "32", "--speculative-tokens", "3", "--verify-tokens", "32"])

    def test_an_autoregressive_profile_creates_without_speculation(self):
        self.assertEqual(recipe.create_arguments(self.profile(drafter="none"), self.manifest()),
                         ["--max-sequences", "32"])


class SignalCleanupTests(unittest.TestCase):
    def test_sigterm_removes_the_worktree_and_target(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "repo"
            root.mkdir()
            git(root, "init")
            for path in (LOCK, ADOPTIONS, SCHEDULES):
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text("[]\n")
            commit(root, LOCK, ADOPTIONS, SCHEDULES)
            tokenizer = base / "tokenizer"
            tokenizer.mkdir()
            for name in ("tokenizer.json", "tokenizer_config.json"):
                (tokenizer / name).write_text("{}")
            scratch = base / "scratch"
            scratch.mkdir()
            marker = base / "ready"
            script = '''import json, os, signal, sys, tempfile
from pathlib import Path
import data_only_recipe
import model_export_recipe as recipe
recipe.ROOT = Path(sys.argv[1])
data_only_recipe.shutil.which = lambda _: "fixture-tool"
cuda = Path(os.environ["CHOPINFER_CUDA_HOME"], "bin")
cuda.mkdir(parents=True)
(cuda / "nvcc").write_text("")
run = recipe.run
def pause_build(*args, cwd=None, env=None):
    if args[0] == "cargo":
        Path(env["CARGO_TARGET_DIR"]).mkdir(exist_ok=True)
        Path(sys.argv[2]).write_text(json.dumps({"scratch": str(Path(cwd).parent), "target": env["CARGO_TARGET_DIR"]}))
        signal.pause()
    return run(*args, cwd=cwd, env=env)
recipe.run = pause_build
with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as target:
    recipe.recipe(Path(target))
'''
            env = dict(os.environ, TMPDIR=str(scratch), CHOPINFER_QWEN38_TOKENIZER_DIR=str(tokenizer),
                       CHOPINFER_CUDA_HOME=str(base / "cuda"), PYTHONPATH=str(ROOT / "tests"))
            env.pop("CHOPINFER_QWEN38_CHECKPOINT", None)
            child = subprocess.Popen([sys.executable, "-B", "-c", script, str(root), str(marker)], env=env,
                                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 10
                while not marker.exists() and child.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.exists(), child.stdout.read() if child.poll() is not None else "")
                child.send_signal(signal.SIGTERM)
                output, _ = child.communicate(timeout=10)
                self.assertEqual(child.returncode, 143, output)
                for path in json.loads(marker.read_text()).values():
                    self.assertFalse(Path(path).exists())
                self.assertEqual(list(scratch.iterdir()), [])
                listed = subprocess.check_output(["git", "-C", str(root), "worktree", "list", "--porcelain"],
                                                 text=True)
                self.assertEqual(listed.count("worktree "), 1)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate()


if __name__ == "__main__":
    unittest.main()

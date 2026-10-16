"""Golden and sensitivity checks for reference cache keys, workloads and mode
policies. Runs without torch."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from tools.refrun import keys, modes, workload

NUMERICS = {path: f"{index:064x}" for index, path in enumerate(keys.NUMERICS, start=1)}
ADAPTER = {"models/demo/reference.py": "a" * 64, **NUMERICS}
ENVIRONMENT = {"torch": "2.14.0+cu130", "cuda_runtime": "13.0", "cublas": "13.1.1.3",
               "device": "sm_120", "env_freeze_sha256": "e" * 64}
LOCK = {"config.json": {"byte_length": 10, "sha256": "c" * 64},
        "model.safetensors": {"byte_length": 20, "sha256": "d" * 64}}
# Pinned values: a change here is a new key schema or mode policy, which
# invalidates every cached reference. Update them only on purpose.
GOLDEN_KEY = "02ed3edd56f0aa9ce2c451f98466ae0943d616d5f0db350a7f85ef5891515d82"
GOLDEN_POLICIES = {
    "f32": "e58f0932ed77dca45edf2d288d0fba3f69829bc8d6a9f489c69df4fb49dc8b7f",
    "f64": "1f82e993b2678e506abb3cf1edd4dac146d9c1afee360328fcaa340d6a13580e",
    "bf16": "fe052bd30dc42991608053074a76d2b14248ff427a9c83e660217e0943f85b4d",
}


def write_workload(root, sequences):
    path = Path(root) / "workload.json"
    path.write_text(json.dumps({"schema": workload.SCHEMA, "model": "demo", "sequences": sequences}))
    return workload.load(path)


def media_sequence(root):
    (Path(root) / "pixels.bin").write_bytes(bytes(range(24)))
    (Path(root) / "grid.bin").write_bytes((1).to_bytes(8, "little") * 3)
    return {"name": "image", "category": "media", "token_ids": [5, 6, 7, 8], "score_from": 2,
            "answer": {"policy": "forced", "ids": [9, 10]},
            "media": [{"kind": "image", "tensor": {"path": "pixels.bin", "dtype": "f32", "shape": [2, 3]},
                       "grid": {"path": "grid.bin", "dtype": "i64", "shape": [1, 3]},
                       "processor": {"code_sha256": "1" * 64, "config_sha256": "2" * 64},
                       "raw_sha256": "3" * 64}]}


def key(sequence, policy=None, environment=ENVIRONMENT, outputs=None, lock=LOCK, adapter=ADAPTER,
        transformers_sha256="e" * 64):
    return keys.key_sha256(keys.document(
        model="demo", checkpoint=keys.checkpoint(lock, ["model.safetensors", "config.json"]),
        sequence=keys.sequence(sequence),
        reference_code=keys.reference_code({"modeling_demo.py": "f" * 64}, "5.16.1", transformers_sha256),
        adapter=keys.adapter(adapter), mode=keys.mode(policy or modes.policy("f32")),
        environment=environment, outputs=outputs or keys.outputs(20)))


class KeyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        _, (self.sequence,) = write_workload(self.directory.name, [media_sequence(self.directory.name)])

    def tearDown(self):
        self.directory.cleanup()

    def test_golden_key(self):
        self.assertEqual(key(self.sequence), GOLDEN_KEY)

    def test_key_document(self):
        """The document behind GOLDEN_KEY, field by field."""
        document = keys.document(
            model="demo", checkpoint=keys.checkpoint(LOCK, ["model.safetensors", "config.json"]),
            sequence=keys.sequence(self.sequence),
            reference_code=keys.reference_code({"modeling_demo.py": "f" * 64}, "5.16.1", "e" * 64),
            adapter=keys.adapter(ADAPTER), mode=keys.mode(modes.policy("f32")),
            environment=ENVIRONMENT, outputs=keys.outputs(20))
        lock = b'{"config.json":{"byte_length":10,"sha256":"' + b"c" * 64 + \
            b'"},"model.safetensors":{"byte_length":20,"sha256":"' + b"d" * 64 + b'"}}'
        self.assertEqual(document, {
            "schema": "chopinfer.refcache.key.v1",
            "model": "demo",
            "checkpoint": {"source_lock_sha256": keys.sha256(lock),
                           "files_used": ["config.json", "model.safetensors"]},
            "sequence": {
                "token_ids_sha256": keys.sha256(b"[5,6,7,8]"),
                "score_from": 2,
                "answer": {"policy": "forced", "ids_sha256": keys.sha256(b"[9,10]"),
                           "max_new_tokens": None, "stop_ids": None},
                "media": [{"kind": "image",
                           "tensor": {"sha256": keys.sha256(bytes(range(24))), "dtype": "f32", "shape": [2, 3]},
                           "grid": {"sha256": keys.sha256((1).to_bytes(8, "little") * 3), "dtype": "i64",
                                    "shape": [1, 3]},
                           "processor": {"code_sha256": "1" * 64, "config_sha256": "2" * 64}}]},
            "reference_code": {"files": {"modeling_demo.py": "f" * 64}, "transformers": "5.16.1",
                               "transformers_sha256": "e" * 64},
            "adapter": {"files": ADAPTER},
            "mode": {"name": "f32", "policy_sha256": GOLDEN_POLICIES["f32"]},
            "environment": ENVIRONMENT,
            "outputs": {"topk": 20, "taps": None},
        })
        self.assertEqual(keys.key_sha256(document), GOLDEN_KEY)

    def test_golden_mode_policies(self):
        for name, digest in GOLDEN_POLICIES.items():
            self.assertEqual(modes.policy_sha256(modes.policy(name)), digest, name)

    def test_every_semantic_input_changes_the_key(self):
        sequence = self.sequence
        image = sequence.media[0]
        variants = {
            "token": replace(sequence, token_ids=(5, 6, 7, 9)),
            "score_from": replace(sequence, score_from=3),
            "answer": sequence.forced((9, 11)),
            "greedy": replace(sequence, answer=workload.Answer("greedy", (), 4, (2,))),
            "pixels": replace(sequence, media=(replace(image, tensor=replace(image.tensor, data=bytes(24))),)),
            "shape": replace(sequence, media=(replace(image, tensor=replace(image.tensor, shape=(3, 2))),)),
            "grid": replace(sequence, media=(replace(image, grid=None),)),
            "processor": replace(sequence, media=(replace(image, processor={**image.processor,
                                                                              "code_sha256": "4" * 64}),)),
        }
        seen = {GOLDEN_KEY}
        for name, changed in variants.items():
            digest = key(changed)
            self.assertNotIn(digest, seen, name)
            seen.add(digest)
        for name, digest in {
            "mode": key(sequence, modes.policy("f64")),
            "dequant": key(sequence, modes.policy("f32", "hf-bf16")),
            "environment": key(sequence, environment={**ENVIRONMENT, "torch": "2.15.0"}),
            "outputs": key(sequence, outputs=keys.outputs(5)),
            "lock": key(sequence, lock={**LOCK, "config.json": {"byte_length": 11, "sha256": "c" * 64}}),
            "adapter": key(sequence, adapter={**ADAPTER, "tools/refrun/quant.py": "9" * 64}),
            "loader": key(sequence, adapter={**ADAPTER, "tools/refrun/place.py": "9" * 64}),
            "transformers files": key(sequence, transformers_sha256="d" * 64),
        }.items():
            self.assertNotIn(digest, seen, name)
            seen.add(digest)

    def test_provenance_does_not_change_the_key(self):
        image = self.sequence.media[0]
        other = replace(self.sequence, name="renamed", category="other",
                        media=(replace(image, raw_sha256="5" * 64),))
        self.assertEqual(key(other), GOLDEN_KEY)

    def test_adapter_section_requires_runner_numerics(self):
        with self.assertRaisesRegex(ValueError, "numerics"):
            keys.adapter({"models/demo/reference.py": "a" * 64})
        # Sequence.forced_ids assembles the IDs a forced pass reads.
        for path in ("tools/refrun/place.py", "tools/refrun/run.py", "tools/refrun/workload.py"):
            with self.assertRaisesRegex(ValueError, "numerics", msg=path):
                keys.adapter({file: digest for file, digest in ADAPTER.items() if file != path})

    def test_transformers_content_digest_follows_every_module(self):
        """A patched cache or SDPA helper under one transformers version
        changes the digest the key holds; bytecode caches do not."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in ("__init__.py", "cache_utils.py", "integrations/sdpa_attention.py"):
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text(f"# {path}\n")
            digest = keys.package_sha256(root)
            (root / "__pycache__").mkdir()
            (root / "__pycache__/cache_utils.cpython-312.pyc").write_bytes(b"\0")
            self.assertEqual(keys.package_sha256(root), digest)
            seen = {digest}
            for path in ("cache_utils.py", "integrations/sdpa_attention.py"):
                (root / path).write_text("# patched\n")
                self.assertNotIn(keys.package_sha256(root), seen, path)
                seen.add(keys.package_sha256(root))

    def test_numerics_files_exist(self):
        for path in keys.NUMERICS:
            self.assertTrue((keys.ROOT / path).is_file(), path)


class WorkloadTests(unittest.TestCase):
    def test_rejects_malformed_sequences(self):
        base = {"name": "t", "category": "c", "token_ids": [1, 2, 3], "score_from": 1}
        bad = [
            {**base, "score_from": 0},
            {**base, "score_from": 3},
            {**base, "token_ids": [1, -2]},
            {**base, "answer": {"policy": "greedy", "max_new_tokens": 0, "stop_ids": []}},
            {**base, "answer": {"policy": "sampled"}},
            {**base, "extra": 1},
        ]
        with tempfile.TemporaryDirectory() as root:
            for spec in bad:
                with self.assertRaises(ValueError, msg=spec):
                    write_workload(root, [spec])
            with self.assertRaisesRegex(ValueError, "unique"):
                write_workload(root, [base, base])
            (Path(root) / "short.bin").write_bytes(bytes(7))
            with self.assertRaisesRegex(ValueError, "bytes"):
                write_workload(root, [{**base, "media": [{
                    "kind": "image", "tensor": {"path": "short.bin", "dtype": "f32", "shape": [2]},
                    "processor": {"code_sha256": "", "config_sha256": ""}}]}])

    def test_forced_answer_follows_the_prompt(self):
        with tempfile.TemporaryDirectory() as root:
            _, (sequence,) = write_workload(root, [{"name": "t", "category": "c", "token_ids": [1, 2],
                                                     "score_from": 2,
                                                     "answer": {"policy": "forced", "ids": [3]}}])
        self.assertEqual(sequence.forced_ids, (1, 2, 3))
        greedy = replace(sequence, answer=workload.Answer("greedy", (), 4, (0,)))
        with self.assertRaisesRegex(ValueError, "generated"):
            greedy.forced_ids
        self.assertEqual(greedy.forced([7, 0]).forced_ids, (1, 2, 7, 0))


class ModeTests(unittest.TestCase):
    def test_policies(self):
        f32, f64, bf16 = (modes.policy(name) for name in ("f32", "f64", "bf16"))
        self.assertEqual((f32["dtype"], f32["dequant"], f32["log_softmax"]), ("float32", "exact", "float32"))
        self.assertEqual((f64["dtype"], f64["log_softmax"]), ("float64", "float64"))
        self.assertEqual((bf16["dtype"], bf16["dequant"]), ("bfloat16", "hf-bf16"))
        for policy in (f32, f64, bf16):
            self.assertEqual(policy["flags"]["fp32_precision"], "ieee")
            self.assertFalse(policy["flags"]["allow_bf16_reduced_precision_reduction"])
            self.assertEqual(policy["environment"], {"CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                                                     "NVIDIA_TF32_OVERRIDE": "0"})
            self.assertEqual(policy["sdpa_backend"], "math")
        with self.assertRaises(ValueError):
            modes.policy("bf16", "exact")
        with self.assertRaises(ValueError):
            modes.policy("f16")


if __name__ == "__main__":
    unittest.main()

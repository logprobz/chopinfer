"""router_bias.py on a synthetic checkpoint: extraction verified against the
checkpoint lock, the sidecar format, and the declared input a layer check's
ROUTER names, bound to the declaration's model and checkpoint pin."""

from qual_test_config import setUpModule

import base64
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import router_bias

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "router bias extraction from a synthetic checkpoint and a synthetic declaration"

PATTERN = "model.layers.{layer}.mlp.gate.e_score_correction_bias"


def safetensors(path, tensors):
    """A safetensors file of `tensors` {name: (dtype, raw bytes, shape)}."""
    header, offset, data = {}, 0, b""
    for name, (dtype, raw, shape) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        data += raw
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + data)


def entry(path):
    return {"byte_length": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class ExtractTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repository = self.root / "repository"
        self.checkpoint = self.root / "checkpoint"
        self.repository.mkdir()
        self.checkpoint.mkdir()
        self.values = {3: [0.5, -1.25, 8.0, 9.625], 4: [1.0, 2.0, 3.0, 4.0]}
        safetensors(self.checkpoint / "a.safetensors",
                    {PATTERN.format(layer=3): ("F32", struct.pack("<4f", *self.values[3]), [4]),
                     "model.layers.3.mlp.gate.weight": ("F32", struct.pack("<4f", 1, 2, 3, 4), [4])})
        # Layer 4's bias is stored as BF16: 1, 2, 3 and 4 are exact.
        bf16 = b"".join(struct.pack("<f", value)[2:] for value in self.values[4])
        safetensors(self.checkpoint / "b.safetensors", {PATTERN.format(layer=4): ("BF16", bf16, [4])})
        index = {"weight_map": {PATTERN.format(layer=3): "a.safetensors",
                                "model.layers.3.mlp.gate.weight": "a.safetensors",
                                PATTERN.format(layer=4): "b.safetensors"}}
        (self.checkpoint / "model.safetensors.index.json").write_text(json.dumps(index))
        self.lock = self.repository / "checkpoint-files.json"
        self.write_lock()
        root = patch.object(router_bias, "ROOT", self.repository)
        root.start()
        self.addCleanup(root.stop)

    def write_lock(self):
        self.lock.write_text(json.dumps({name: entry(self.checkpoint / name) for name in
                                         ("a.safetensors", "b.safetensors", "model.safetensors.index.json")}))

    def test_extracts_every_layer_verified_against_the_lock(self):
        sidecar = router_bias.extract(self.checkpoint, self.lock, "m", PATTERN)
        self.assertEqual(sidecar["schema"], router_bias.SCHEMA)
        self.assertEqual(sidecar["checkpoint"]["lock"], "checkpoint-files.json")
        self.assertEqual(set(sidecar["checkpoint"]["files"]), {"a.safetensors", "b.safetensors"})
        self.assertEqual(sidecar["experts"], 4)
        meta, values = router_bias.load(json.dumps(sidecar))
        self.assertNotIn("layers", meta)
        self.assertEqual(values, {layer: tuple(v) for layer, v in self.values.items()})

    def test_a_shard_that_differs_from_the_lock_is_refused(self):
        path = self.checkpoint / "b.safetensors"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(bytes(data))
        with self.assertRaisesRegex(ValueError, "b.safetensors differs"):
            router_bias.extract(self.checkpoint, self.lock, "m", PATTERN)

    def test_an_index_or_shard_the_lock_does_not_pin_is_refused(self):
        self.lock.write_text(json.dumps({"model.safetensors.index.json":
                                         entry(self.checkpoint / "model.safetensors.index.json")}))
        with self.assertRaisesRegex(ValueError, "does not pin a.safetensors"):
            router_bias.extract(self.checkpoint, self.lock, "m", PATTERN)

    def test_a_pattern_that_names_no_tensor_or_no_layer_is_refused(self):
        with self.assertRaisesRegex(ValueError, "names no tensor"):
            router_bias.extract(self.checkpoint, self.lock, "m", "model.layers.{layer}.missing")
        with self.assertRaisesRegex(ValueError, "must name"):
            router_bias.extract(self.checkpoint, self.lock, "m", "model.layers.3.mlp.gate.weight")

    def test_a_lock_outside_the_repository_is_refused(self):
        outside = self.root / "lock.json"
        outside.write_bytes(self.lock.read_bytes())
        with self.assertRaisesRegex(ValueError, "not a checkpoint lock of this repository"):
            router_bias.extract(self.checkpoint, outside, "m", PATTERN)

    def test_a_malformed_sidecar_is_refused(self):
        good = router_bias.extract(self.checkpoint, self.lock, "m", PATTERN)
        for broken in ({**good, "schema": "other"}, {**good, "experts": 5},
                       {**good, "layers": {"x": good["layers"]["3"]}},
                       {**good, "layers": {"3": base64.b64encode(b"\0" * 12).decode()}}, {**good, "layers": {}}):
            with self.assertRaises(ValueError):
                router_bias.load(json.dumps(broken))


class DeclaredInputTests(unittest.TestCase):
    """tapdiff_args reads the input ROUTER names as the reference lock
    authenticates it and binds it to the declaration's model and pin."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.sidecar = {"schema": router_bias.SCHEMA, "model": "m", "experts": 2,
                        "checkpoint": {"source_lock_sha256": "a" * 64},
                        "layers": {"3": base64.b64encode(struct.pack("<2f", 1, 2)).decode()}}
        self.path = self.root / "bias.json"
        self.path.write_text(json.dumps(self.sidecar))
        self.config = {"model": "m", "inputs": {"bias": str(self.path)}}
        for target, value in (("common.checkpoint_pin", {"source_lock_sha256": "a" * 64}),
                              ("refs.verify_input", None)):
            patched = patch(target, return_value=value)
            self.mock = patched.start()
            self.addCleanup(patched.stop)
        self.spec = {"score": "sigmoid", "top_k": 2, "bias": "bias"}

    def test_an_empty_spec_judges_no_routes(self):
        self.assertEqual(router_bias.tapdiff_args({}, self.config, self.root), [])

    def test_a_spec_without_a_bias_passes_only_the_router(self):
        args = router_bias.tapdiff_args({"score": "softmax", "top_k": 10}, self.config, self.root)
        self.assertEqual(args, ["--router", json.dumps({"score": "softmax", "top_k": 10}, sort_keys=True)])

    def test_the_bias_input_is_copied_after_it_is_authenticated(self):
        scratch = self.root / "scratch"
        scratch.mkdir()
        args = router_bias.tapdiff_args(self.spec, self.config, scratch)
        self.assertEqual(args[2:], ["--router-bias", str(scratch / "router-bias.json")])
        self.assertEqual((scratch / "router-bias.json").read_bytes(), self.path.read_bytes())
        self.mock.assert_called_once_with(self.config, "bias", self.path)

    def test_a_bias_of_another_model_checkpoint_or_an_undeclared_input_is_refused(self):
        for change, message in ((lambda s, c: s.update(model="other"), "belongs to model"),
                                (lambda s, c: s["checkpoint"].update(source_lock_sha256="b" * 64),
                                 "another checkpoint"),
                                (lambda s, c: c["inputs"].clear(), "does not declare")):
            sidecar, config = json.loads(json.dumps(self.sidecar)), json.loads(json.dumps(self.config))
            change(sidecar, config)
            self.path.write_text(json.dumps(sidecar))
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                router_bias.tapdiff_args(self.spec, config, self.root)


if __name__ == "__main__":
    unittest.main()

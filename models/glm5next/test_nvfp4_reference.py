"""The GLM 5.3 Flash reference adapter against NVFP4 checkpoints, from their
headers: each checkpoint of CHOPINFER_GLM5NEXT_EXTRA_CHECKPOINTS (`:`
separated) that stores NVFP4 weights, whatever its other classes store.

The HF model is built on the meta device from the checkpoint's config.json,
and every linear and routed expert the adapter loads must find its stored
weight and format: NVFP4 weights (U8 in the header) with one finite F32
global scale each (the only payload bytes read), BF16 weights without
scales. The adapter skips the checkpoint's activation scales (the reference
is A16) and its MTP layer, whose experts the reference never runs. The tiny
NVFP4 and MXFP8 forward is in test_reference.py. The same checks run on a
small synthetic header checkpoint written here; the checkpoint tests skip
without torch, transformers 5.16.1 or a named NVFP4 checkpoint.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B -m unittest models/glm5next/test_nvfp4_reference.py
"""

import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch import nn
    from tools.refrun import place, run
except ModuleNotFoundError:
    torch = None

ADAPTER = ROOT / "models/glm5next/reference.py"
TEXT = "model.language_model"
EXTRA = "CHOPINFER_GLM5NEXT_EXTRA_CHECKPOINTS"


class HeaderStore:
    """The adapter's store interface over headers: `get` reads only scalar
    tensors (the NVFP4 global scales)."""

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def get(self, name):
        entry = self.checkpoint.entries[name]
        if entry.dtype != "F32" or math.prod(entry.shape) != 1:
            raise ValueError(f"{name}: the header check reads only F32 scalars")
        with open(self.checkpoint.root / entry.file, "rb") as file:
            file.seek(entry.offset)
            (value,) = struct.unpack("<f", file.read(4))
        return torch.tensor(value, dtype=torch.float32)


def nvfp4_checkpoints():
    """Each named extra checkpoint that stores an NVFP4 weight, placed."""
    paths = [path for path in os.environ.get(EXTRA, "").split(os.pathsep) if path]
    checkpoints = [place.Checkpoint(path) for path in paths]
    return [checkpoint for checkpoint in checkpoints
            if any(name.endswith(".weight") and entry.dtype == "U8"
                   for name, entry in checkpoint.entries.items())]


def write_checkpoint(root, config, tensors):
    """A single-file checkpoint under `root`: `config` and `tensors`, each
    name to (dtype, shape, payload bytes)."""
    header, data = {}, b""
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(data), len(data) + len(payload)]}
        data += payload
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    (root / "config.json").write_text(json.dumps(config))


class SyntheticNvfpTests(unittest.TestCase):
    """The adapter on a two-layer header checkpoint of each stored format:
    an NVFP4 routed expert with its scales and activation scale, a BF16
    projection and an MTP layer tensor."""

    EXPERT = f"{TEXT}.layers.0.mlp.experts.0.down_proj."

    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest("needs torch (set CHOPINFER_REFRUN_PYTHON)")
        cls.adapter, _ = run.load_adapter(ADAPTER)

    def checkpoint(self, tensors):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        write_checkpoint(Path(directory.name), {"text_config": {"num_hidden_layers": 2}}, tensors)
        return place.Checkpoint(directory.name)

    def tensors(self):
        return {
            self.EXPERT + "weight": ("U8", (8, 16), bytes(8 * 16)),
            self.EXPERT + "weight_scale": ("F8_E4M3", (8, 2), bytes(8 * 2)),
            self.EXPERT + "weight_scale_2": ("F32", (), struct.pack("<f", 0.25)),
            self.EXPERT + "input_scale": ("F32", (), struct.pack("<f", 1.0)),
            f"{TEXT}.layers.0.self_attn.o_proj.weight": ("BF16", (4, 4), bytes(4 * 4 * 2)),
            f"{TEXT}.layers.2.mlp.experts.0.down_proj.weight": ("BF16", (4, 4), bytes(4 * 4 * 2)),
        }

    def test_units_skip_activation_scales_and_the_mtp_layer(self):
        checkpoint = self.checkpoint(self.tensors())
        loaded = {name for unit in self.adapter.units(checkpoint) for name in unit.tensors}
        self.assertEqual(set(checkpoint.entries) - loaded,
                         {self.EXPERT + "input_scale", f"{TEXT}.layers.2.mlp.experts.0.down_proj.weight"})

    def test_each_weight_has_its_stored_format(self):
        checkpoint = self.checkpoint(self.tensors())
        store = HeaderStore(checkpoint)
        fmt, (rows, cols), scales = self.adapter.stored_format(store, self.EXPERT + "weight")
        self.assertEqual((fmt.element, fmt.scale, fmt.block_cols, fmt.global_scale), ("e2m1", "e4m3", 16, 0.25))
        self.assertEqual((rows, cols), (8, 32))
        self.assertEqual(list(checkpoint.entries[scales[0]].shape), fmt.scale_shape(rows, cols))
        self.assertEqual(scales, (self.EXPERT + "weight_scale", self.EXPERT + "weight_scale_2"))
        fmt, shape, scales = self.adapter.stored_format(store, f"{TEXT}.layers.0.self_attn.o_proj.weight")
        self.assertEqual((fmt.element, shape, scales), ("bf16", (4, 4), ()))

    def test_nvfp4_codes_without_their_global_are_refused(self):
        tensors = self.tensors()
        del tensors[self.EXPERT + "weight_scale_2"]
        checkpoint = self.checkpoint(tensors)
        with self.assertRaisesRegex(ValueError, "no supported format for U8 codes"):
            self.adapter.stored_format(HeaderStore(checkpoint), self.EXPERT + "weight")


class NvfpCheckpointTests(unittest.TestCase):
    """Each NVFP4 checkpoint's adapter units and stored formats, with
    expectations read from the checkpoint's own headers."""

    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest("needs torch (set CHOPINFER_REFRUN_PYTHON)")
        cls.checkpoints = nvfp4_checkpoints()
        if not cls.checkpoints:
            raise unittest.SkipTest(f"needs an NVFP4 checkpoint in {EXTRA}")
        cls.adapter, _ = run.load_adapter(ADAPTER)

    def test_units_skip_activation_scales_and_the_mtp_layer(self):
        for checkpoint in self.checkpoints:
            with self.subTest(checkpoint=str(checkpoint.root)):
                layers = self.adapter.text_layers(checkpoint.config)
                self.assertEqual(layers, 45)
                loaded = {name for unit in self.adapter.units(checkpoint) for name in unit.tensors}
                skipped = set(checkpoint.entries) - loaded
                mtp = f"{TEXT}.layers.{layers}."
                self.assertTrue(skipped)
                for name in skipped:
                    self.assertTrue(name.endswith(".input_scale") or name.startswith(mtp), name)
                scales = {name for name in checkpoint.entries
                          if name.endswith(".input_scale") and not name.startswith(mtp)}
                self.assertEqual({name for name in skipped if name.endswith(".input_scale")
                                  and not name.startswith(mtp)}, scales)
                self.assertFalse(any(name.endswith(".input_scale") for name in loaded))

    def test_every_loaded_weight_has_its_stored_format(self):
        configuration, hf = self.adapter.hf_modules()
        for checkpoint in self.checkpoints:
            with self.subTest(checkpoint=str(checkpoint.root)):
                self.check_formats(checkpoint, configuration, hf)

    def check_formats(self, checkpoint, configuration, hf):
        config = configuration.Glm5NextConfig.from_dict(dict(checkpoint.config))
        with torch.device("meta"):
            model = hf.Glm5NextForConditionalGeneration(config)
        store = HeaderStore(checkpoint)
        stored = {name for unit in self.adapter.units(checkpoint) for name in unit.tensors}
        by_hf = {self.adapter.hf_name(name): name for name in stored}
        weights = [by_hf[path + ".weight"] for path, module in model.named_modules()
                   if isinstance(module, nn.Linear) and not path.endswith("indexer.weights_proj")]
        text = config.text_config
        for layer, kind in enumerate(text.mlp_layer_types):
            if kind == "sparse":
                weights += [f"{TEXT}.layers.{layer}.mlp.experts.{expert}.{part}_proj.weight"
                            for expert in range(text.n_routed_experts) for part in ("gate", "up", "down")]
        self.assertEqual(len(weights), len(set(weights)))
        nvfp4 = 0
        for weight in weights:
            fmt, (rows, cols), scales = self.adapter.stored_format(store, weight)
            dtype = checkpoint.entries[weight].dtype
            if dtype == "U8":
                nvfp4 += 1
                self.assertEqual((fmt.element, fmt.scale, fmt.block_cols), ("e2m1", "e4m3", 16), weight)
                self.assertTrue(math.isfinite(fmt.global_scale) and fmt.global_scale > 0, weight)
                self.assertEqual(list(checkpoint.entries[scales[0]].shape), fmt.scale_shape(rows, cols))
                self.assertEqual(checkpoint.entries[weight].shape, (rows, cols // 2))
            elif dtype == "BF16":
                self.assertEqual((fmt.element, scales), ("bf16", ()), weight)
        self.assertTrue(nvfp4)


if __name__ == "__main__":
    unittest.main()

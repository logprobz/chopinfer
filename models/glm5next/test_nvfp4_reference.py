"""The GLM 5.3 Flash reference adapter against the NVFP4 exports, from their
headers: local-inference-lab's (models/glm5next/checkpoint-files-nvfp4.json)
and NVIDIA's (checkpoint-files-nvidia-nvfp4.json, whose dense MLPs of layers
0-2 are NVFP4 too and whose MTP experts are BF16).

The HF model is built on the meta device from the export's config.json, and
every linear and routed expert the adapter loads must find its stored weight
and format: NVFP4 weights with one finite F32 global scale each (the
only payload bytes read), BF16 everywhere else. The adapter skips the export's
activation scales (the reference is A16) and its MTP layer, whose MXFP8
experts the reference never runs. The tiny NVFP4 and MXFP8 forward is in
test_reference.py. Each export's tests skip without torch, transformers
5.16.1 or its variable (CHOPINFER_GLM5NEXT_NVFP4_CHECKPOINT,
CHOPINFER_GLM5NEXT_NVIDIA_NVFP4_CHECKPOINT).

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B -m unittest models/glm5next/test_nvfp4_reference.py
"""

import math
import os
from pathlib import Path
import struct
import sys
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


class ExportChecks:
    """One export's checks; `VARIABLE` names its directory and `NVFP4_DENSE`
    whether its dense MLPs are NVFP4."""
    VARIABLE = None
    NVFP4_DENSE = False

    @classmethod
    def setUpClass(cls):
        root = os.environ.get(cls.VARIABLE)
        if torch is None:
            raise unittest.SkipTest("needs torch (set CHOPINFER_REFRUN_PYTHON)")
        if not root:
            raise unittest.SkipTest(f"needs {cls.VARIABLE}")
        cls.adapter, _ = run.load_adapter(ADAPTER)
        cls.checkpoint = place.Checkpoint(root)
        cls.layers = cls.adapter.text_layers(cls.checkpoint.config)

    def test_units_skip_activation_scales_and_the_mtp_layer(self):
        loaded = {name for unit in self.adapter.units(self.checkpoint) for name in unit.tensors}
        skipped = set(self.checkpoint.entries) - loaded
        mtp = f"{TEXT}.layers.{self.layers}."
        self.assertEqual(self.layers, 45)
        self.assertTrue(skipped)
        for name in skipped:
            self.assertTrue(name.endswith(".input_scale") or name.startswith(mtp), name)
        dense = 3 * 3 if self.NVFP4_DENSE else 0
        self.assertEqual(sum(name.endswith(".input_scale") for name in skipped), 42 * 288 * 3 + dense)
        self.assertFalse(any(name.endswith(".input_scale") for name in loaded))

    def test_every_loaded_weight_has_its_stored_format(self):
        configuration, hf = self.adapter.hf_modules()
        config = configuration.Glm5NextConfig.from_dict(dict(self.checkpoint.config))
        with torch.device("meta"):
            model = hf.Glm5NextForConditionalGeneration(config)
        store = HeaderStore(self.checkpoint)
        stored = {name for unit in self.adapter.units(self.checkpoint) for name in unit.tensors}
        by_hf = {self.adapter.hf_name(name): name for name in stored}
        weights = [by_hf[path + ".weight"] for path, module in model.named_modules()
                   if isinstance(module, nn.Linear) and not path.endswith("indexer.weights_proj")]
        text = config.text_config
        for layer, kind in enumerate(text.mlp_layer_types):
            if kind == "sparse":
                weights += [f"{TEXT}.layers.{layer}.mlp.experts.{expert}.{part}_proj.weight"
                            for expert in range(text.n_routed_experts) for part in ("gate", "up", "down")]
        self.assertEqual(len(weights), len(set(weights)))
        formats = {}
        for weight in weights:
            fmt, (rows, cols), scales = self.adapter.stored_format(store, weight)
            routed = ".mlp.experts." in weight
            formats[routed, fmt.element] = formats.get((routed, fmt.element), 0) + 1
            if fmt.element == "e2m1":
                self.assertEqual((fmt.element, fmt.scale, fmt.block_cols), ("e2m1", "e4m3", 16), weight)
                self.assertTrue(math.isfinite(fmt.global_scale) and fmt.global_scale > 0, weight)
                self.assertEqual(list(self.checkpoint.entries[scales[0]].shape), fmt.scale_shape(rows, cols))
                self.assertEqual(self.checkpoint.entries[weight].shape, (rows, cols // 2))
            else:
                self.assertEqual(scales, (), weight)
        dense = {(False, "e2m1"): 3 * 3} if self.NVFP4_DENSE else {}
        self.assertEqual({key: formats[key] for key in formats if key != (False, "bf16")},
                         {(True, "e2m1"): 42 * 288 * 3, **dense})
        self.assertIn((False, "bf16"), formats)


class LocalInferenceLabTests(ExportChecks, unittest.TestCase):
    VARIABLE = "CHOPINFER_GLM5NEXT_NVFP4_CHECKPOINT"


class NvidiaTests(ExportChecks, unittest.TestCase):
    VARIABLE = "CHOPINFER_GLM5NEXT_NVIDIA_NVFP4_CHECKPOINT"
    NVFP4_DENSE = True


if __name__ == "__main__":
    unittest.main()

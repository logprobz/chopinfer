"""The generic encoder reference against HF's qwen4_exp vision tower, on CPU.

The committed declaration (contract/encoders/qwen38-vision.json), shrunk to
a two-block tower, runs over a tiny random checkpoint in Qwen3.8's tensor
layout. HF's tower over the same weights is the upstream code the
reference adapter runs. `exact` must match HF in f64 block by block to the
F32 rounding HF keeps there (rotary tables and their application),
`policy` (f32-pairs.v1) must match HF in F32 to pair rounding before its
output rounding, and `reference-bf16` HF in BF16 to BF16 rounding. Needs
torch and transformers 5.16.1; skipped without them.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B -m unittest models/qwen38/test_encoder_reference.py
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from transformers.models.qwen4_exp import configuration_qwen4_exp, modeling_qwen4_exp as hf
    from tools.refrun import encoder, modes, place
except ModuleNotFoundError:
    torch = None

DECLARATION = ROOT / "models/qwen38/contract/encoders/qwen38-vision.json"
VISION = {"depth": 2, "hidden_size": 16, "hidden_act": "gelu_pytorch_tanh", "intermediate_size": 36,
          "num_heads": 2, "in_channels": 3, "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
          "out_hidden_size": 24, "num_position_embeddings": 16, "deepstack_visual_indexes": [],
          "model_type": "qwen4_exp_vision"}
GRIDS = [(1, 4, 6), (1, 2, 2)]


def tiny_declaration():
    """The committed declaration at VISION's sizes; every name, transform and
    variant stays as committed."""
    data = json.loads(DECLARATION.read_text())
    pins = data["checkpoint"]["config"][0]["values"]
    data["checkpoint"]["config"][0]["values"] = {key: VISION[key] for key in pins}
    data["input"]["columns"] = 24
    linear, position, tower, _, _, merger_up, merger_down = data["stages"]
    linear["out"] = 16
    position["table"] = [4, 4]
    tower.update(blocks=2, width=16)
    tower["attention"].update(q_heads=2, kv_heads=2, head_dim=8)
    tower["rotary"]["pairs_per_axis"] = 2
    tower["mlp"].update(hidden=36, pad_to=40)
    weights = tower["weights"]
    for role, transform in (("up.weight", "pad_rows"), ("up.bias", "pad_rows"), ("down.weight", "pad_cols")):
        weights[role]["transforms"] = [{transform: 40}]
    merger_up["out"], merger_down["out"] = 64, 24
    data["output"]["hidden"] = 24
    return data


def write_checkpoint(root, generator):
    """config.json and the vision tower's BF16 tensors in one shard."""
    from safetensors.torch import save_file
    root.mkdir(parents=True)
    with torch.device("meta"):
        model = hf.Qwen4ExpVisionModel(configuration_qwen4_exp.Qwen4ExpVisionConfig(**VISION))
    tensors = {}
    for name, value in model.state_dict().items():
        if "rotary" in name:
            continue
        scale = 0.1 if "norm" in name else 0.4
        tensors[f"model.visual.{name}"] = ((1.0 if "norm" in name and name.endswith("weight") else 0.0)
                                           + scale * torch.randn(value.shape, generator=generator)).to(torch.bfloat16)
    save_file(tensors, str(root / "model.safetensors"))
    (root / "config.json").write_text(json.dumps({"vision_config": VISION}))
    return tensors


def hf_tower(tensors, dtype):
    """HF's tower in `dtype` over the stored weights, SDPA attention and the
    F32 rotary table, as the reference adapter builds it."""
    config = configuration_qwen4_exp.Qwen4ExpVisionConfig(**VISION)
    config._attn_implementation = "sdpa"
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = hf.Qwen4ExpVisionModel(config).eval()
    finally:
        torch.set_default_dtype(default)
    model.load_state_dict({n.removeprefix("model.visual."): t.to(dtype) for n, t in tensors.items()}, strict=False)
    model.rotary_pos_emb = hf.Qwen4ExpVisionRotaryEmbedding(VISION["hidden_size"] // VISION["num_heads"] // 2)
    return model


def relative(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


@unittest.skipIf(torch is None, "needs torch and transformers (set CHOPINFER_REFRUN_PYTHON)")
class Qwen38VisionReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modes.prepare_environment(cuda=False)
        cls.directory = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        root = Path(cls.directory.name)
        generator = torch.Generator().manual_seed(7)
        cls.tensors = write_checkpoint(root / "checkpoint", generator)
        cls.checkpoint = place.Checkpoint(root / "checkpoint")
        path = root / "qwen38-vision.json"
        path.write_text(json.dumps(tiny_declaration()))
        cls.declaration = encoder.Declaration(path)
        cls.pixels = [torch.randn((t * h * w, 24), generator=generator) for t, h, w in GRIDS]

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def upstream(self, dtype, policy):
        """HF's merged rows per image and each block's output, one call over
        both images as the adapter makes it."""
        model = hf_tower(self.tensors, dtype)
        blocks = []
        for block in model.blocks:
            block.register_forward_hook(lambda _m, _a, out: blocks.append(out))
        with modes.active(modes.policy(policy)):
            merged = model(torch.cat(self.pixels).to(dtype), grid_thw=torch.tensor(GRIDS)).pooler_output
        rows = [t * h * w for t, h, w in GRIDS]
        return list(merged.split([r // 4 for r in rows])), [b.split(rows) for b in blocks]

    def generic(self, mode):
        names = encoder.Names(self.declaration, self.checkpoint, {})
        precision = self.declaration.precision if mode == "policy" else None
        model = encoder.build(self.declaration, names, self.tensors, mode, precision, ["cpu"])
        taps = encoder.tap_map(model)
        recorded = {}
        for name in ("enc.vision.B00.out", "enc.vision.B01.out", "enc.vision.x.unrounded"):
            hook = taps[name]
            module = model.get_submodule(hook.module)

            def keep(_m, args, out, name=name, hook=hook):
                recorded.setdefault(name, []).append(args[0] if hook.hook == "input" else out)
            module.register_forward_hook(keep)
        with modes.active(encoder.mode_document(mode, precision)):
            out = [model(encoder.Item("image", pixels, grid)) for pixels, grid in zip(self.pixels, GRIDS)]
        return out, recorded

    def compare(self, mode, dtype, policy, bound, unrounded):
        merged, blocks = self.upstream(dtype, policy)
        out, recorded = self.generic(mode)
        for image in range(len(GRIDS)):
            rows = recorded["enc.vision.x.unrounded"][image] if unrounded else out[image]
            self.assertLess(relative(rows, merged[image]), bound, (mode, image))
            for block in range(2):
                ours = recorded[f"enc.vision.B{block:02d}.out"][image]
                self.assertLess(relative(ours, blocks[block][image]), bound, (mode, image, block))
        return out

    def test_exact_matches_hf_in_f64(self):
        out = self.compare("exact", torch.float64, "f64", 1e-6, False)
        self.assertEqual(out[0].dtype, torch.float64)

    def test_policy_matches_hf_in_f32_to_pair_rounding(self):
        out = self.compare("policy", torch.float32, "f32", 2e-4, True)
        self.assertEqual(out[0].dtype, torch.bfloat16)

    def test_reference_bf16_matches_hf_in_bf16(self):
        out = self.compare("reference-bf16", torch.bfloat16, "bf16", 0.03, False)
        self.assertEqual(out[0].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()

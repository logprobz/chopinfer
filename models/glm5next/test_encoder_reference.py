"""The generic encoder reference against transformers' glm5_next vision tower, on CPU.

The committed declaration (contract/encoders/glm5next-vision.json), shrunk
to a tiny synthetic tower (2 blocks of width 16, 2 heads of 8, MLP 12,
2 x 2 x 2 x 3 patches zero-padded from 24 to 32 columns, merge into 8,
projector through 10, clamp limit 1.5 so the clamps bite), runs beside
`Glm5NextVisionModel` built from the same BF16 weights:

- `exact` matches HF in f64 to f64 rounding once HF's F32 casts (its
  RMSNorm and rotary) are taken out and its rotary table is the F32 one
  every mode uses, and stays within F32 rounding of unmodified HF in f64;
- `policy` (f32-pairs.v1) stays within pair rounding of HF in F32;
- `reference-bf16` stays within BF16 rounding of HF in BF16.

HF's attention runs as sdpa (its eager path takes its softmax in F32), and
its rotary inverse frequencies are F32 in every run, as the reference
adapter builds them. Needs torch and transformers with glm5_next (the
refrun venv); skipped otherwise.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B -m unittest models/glm5next/test_encoder_reference.py
"""

import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1].parent

try:
    import torch
    from safetensors.torch import save_file
    from transformers.models.glm5_next import modeling_glm5_next as hf
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from tools.refrun import encoder, encoder_ops, modes, place
except ModuleNotFoundError:
    torch = None

DECLARATION = ROOT / "models/glm5next/contract/encoders/glm5next-vision.json"
GRID = (1, 4, 6)
LIMIT = 1.5


def tiny_config():
    config = Glm5NextVisionConfig(depth=2, hidden_size=16, num_heads=2, intermediate_size=12, patch_size=2,
                                  temporal_patch_size=2, in_channels=3, spatial_merge_size=2, out_hidden_size=8,
                                  projection_intermediate_size=10, swiglu_limit=LIMIT, rms_norm_eps=1e-5,
                                  attention_bias=True)
    config._attn_implementation = "sdpa"
    return config


def tiny_declaration():
    """glm5next-vision.json at the tiny tower's sizes."""
    data = json.loads(DECLARATION.read_text())
    data["checkpoint"].update(config=[], files={"model.visual.": "model.safetensors"})
    data["input"].update(columns=24, capacity_rows=1024)
    pad, patch, tower, post, merge, downsample, proj, projector = data["stages"]
    pad["columns"] = 32
    patch["out"] = 16
    patch["weights"]["weight"]["transforms"] = ["flatten", {"pad_cols": 32}]
    tower.update(blocks=2, width=16)
    tower["attention"].update(q_heads=2, kv_heads=2, head_dim=8)
    tower["rotary"]["pairs_per_axis"] = 2
    tower["mlp"].update(hidden=12, limit=LIMIT)
    downsample["out"], proj["out"] = 8, 8
    projector["mlp"].update(hidden=10, limit=LIMIT)
    data["output"]["hidden"] = 8
    return data


def relative(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def without_f32_casts(f32_table):
    """HF's RMSNorm and vision rotary in their input dtype (both cast to
    F32 first), for the f64 comparison; with `f32_table` the rotary's cos
    and sin are rounded to F32 first, as the declaration's table is.
    Returns the originals to restore."""
    original = (hf.Glm5NextRMSNorm.forward, hf.apply_rotary_pos_emb_vision)

    def rms(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))

    def rotary(q, k, cos, sin):
        if f32_table:
            cos, sin = cos.float().to(q.dtype), sin.float().to(q.dtype)
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
        return q * cos + hf.rotate_half(q) * sin, k * cos + hf.rotate_half(k) * sin
    hf.Glm5NextRMSNorm.forward, hf.apply_rotary_pos_emb_vision = rms, rotary
    return original


@unittest.skipIf(torch is None, "needs torch and transformers with glm5_next")
class Glm5NextEncoderReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modes.prepare_environment(cuda=False)
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        torch.manual_seed(0)
        model = hf.Glm5NextVisionModel(tiny_config())
        cls.weights = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                value = torch.randn_like(parameter) * (0.5 if parameter.dim() > 1 else 0.3)
                if name.endswith("norm.weight") or ".norm" in name or "layernorm" in name:
                    value += 1.0
                cls.weights[name] = value.to(torch.bfloat16)
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        cls.tensors = {f"model.visual.{name}": value for name, value in cls.weights.items()}
        save_file(cls.tensors, str(checkpoint / "model.safetensors"))
        (checkpoint / "config.json").write_text("{}")
        cls.checkpoint = place.Checkpoint(checkpoint)
        path = root / "glm5next-vision.json"
        path.write_text(json.dumps(tiny_declaration()))
        cls.declaration = encoder.Declaration(path)
        generator = torch.Generator().manual_seed(1)
        rows = GRID[0] * GRID[1] * GRID[2]
        cls.pixels = torch.randn((rows, 24), generator=generator) * 2

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def hf_rows(self, dtype, f32_casts=True, f32_table=False):
        """HF's merged rows from the BF16 weights in `dtype`; without
        `f32_casts`, its RMSNorm and rotary run in `dtype`, and with
        `f32_table` its rotary table is the F32 one of the declaration's
        rotary (F32 frequencies and angle products, cos and sin rounded to
        F32)."""
        model = hf.Glm5NextVisionModel(tiny_config())
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(self.weights[name].float())
        model = model.to(dtype).eval()
        head_dim = 16 // 2
        # F32 inverse frequencies in every run, as the reference adapter builds them.
        model.rotary_pos_emb = hf.Glm5NextVisionRotaryEmbedding(head_dim // 2)
        if f32_table:
            # The declaration's F32 frequencies and F32 angle products.
            rotary = encoder_ops.Axial2d(tiny_declaration()["stages"][2]["rotary"], encoder_ops.Numerics("pairs"))

            def angles(positions):
                return (positions.unsqueeze(-1).to(torch.float32) * rotary.frequencies).flatten(1).to(dtype)
            model.rotary_pos_emb.forward = angles
        original = None if f32_casts else without_f32_casts(f32_table)
        try:
            with torch.no_grad():
                out = model(self.pixels.to(dtype), grid_thw=torch.tensor([GRID])).pooler_output
        finally:
            if original is not None:
                hf.Glm5NextRMSNorm.forward, hf.apply_rotary_pos_emb_vision = original
        return out

    def generic(self, mode):
        names = encoder.Names(self.declaration, self.checkpoint, {})
        precision = self.declaration.precision if mode == "policy" else None
        model = encoder.build(self.declaration, names, self.tensors, mode, precision, ["cpu"])
        with modes.active(encoder.mode_document(mode, precision)):
            return model(encoder.Item("image", self.pixels, GRID))

    def test_the_clamps_bite(self):
        """Some gate and up values of the tower and the projector pass the limit."""
        model = hf.Glm5NextVisionModel(tiny_config())
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(self.weights[name].float())
        seen = []
        for module in (model.blocks[0].mlp, model.merger):
            module.gate_proj.register_forward_hook(lambda _m, _a, out: seen.append(out))
            module.up_proj.register_forward_hook(lambda _m, _a, out: seen.append(out))
        with torch.no_grad():
            model(self.pixels, grid_thw=torch.tensor([GRID]))
        for values in seen:
            self.assertTrue(bool((values.abs() > LIMIT).any()))

    def test_exact_matches_hf_in_f64(self):
        ours = self.generic("exact")
        self.assertEqual(ours.dtype, torch.float64)
        # HF without its F32 casts, on the F32 rotary table: the same math.
        self.assertLess(relative(ours, self.hf_rows(torch.float64, f32_casts=False, f32_table=True)), 1e-12)
        # Unmodified HF in f64 rounds its norms and rotary through F32.
        error = relative(ours, self.hf_rows(torch.float64))
        self.assertLess(error, 1e-5)
        self.assertGreater(error, 1e-12)

    def test_policy_stays_within_pair_rounding_of_hf_in_f32(self):
        ours = self.generic("policy")
        self.assertEqual(ours.dtype, torch.bfloat16)
        self.assertLess(relative(ours, self.hf_rows(torch.float32)), 1e-2)
        # Before its output rounding the pair rows are near F32.
        names = encoder.Names(self.declaration, self.checkpoint, {})
        model = encoder.build(self.declaration, names, self.tensors, "policy", "f32-pairs.v1", ["cpu"])
        unrounded = []
        model.boundary.register_forward_hook(lambda _m, args, _out: unrounded.append(args[0]))
        with modes.active(encoder.mode_document("policy", "f32-pairs.v1")):
            model(encoder.Item("image", self.pixels, GRID))
        self.assertEqual(unrounded[0].dtype, torch.float32)
        self.assertLess(relative(unrounded[0], self.hf_rows(torch.float32)), 1e-4)

    def test_reference_bf16_stays_within_bf16_rounding_of_hf_in_bf16(self):
        ours = self.generic("reference-bf16")
        self.assertEqual(ours.dtype, torch.bfloat16)
        theirs = self.hf_rows(torch.bfloat16)
        exact = self.hf_rows(torch.float64)
        # As near HF in BF16 as HF in BF16 is to exact, to a small factor.
        spread = relative(theirs, exact)
        self.assertLess(relative(ours, theirs), 2 * spread + 1e-3)
        self.assertLess(relative(ours, exact), 3 * spread + 1e-3)


if __name__ == "__main__":
    unittest.main()

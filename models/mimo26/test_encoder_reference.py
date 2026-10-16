"""The generic encoder reference against MiMo's vendor encoders, on CPU.

The committed declarations (contract/encoders/mimo-vision.json and
mimo-audio.json), shrunk to the tiny checkpoint of test_reference.py, run
with MiMo's own variants (encoder_reference.py) beside the vendor's vision
tower, audio tokenizer and audio encoder as the reference adapter builds
them (`reference.build_encoders`). `exact` must match the vendor in f64
block by block, with equal audio codes, to the F32 rounding of the vendor's
rotary tables; `policy` (the declarations' f32-pairs.v1) and
`reference-bf16` must stay within BF16 rounding of the vendor in BF16, and
`policy` within pair rounding of the vendor in F32. Needs torch and the MiMo vendor
modeling files ($CHOPINFER_MIMO_CHECKPOINT or the tokenizer directory CI
passes); skipped otherwise.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B -m unittest models/mimo26/test_encoder_reference.py
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("mimo26_test_reference", Path(__file__).with_name("test_reference.py"))
tiny = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tiny)

try:
    import torch
    from tools.refrun import adapter, encoder, encoder_ops, modes, place, run
except ModuleNotFoundError:
    torch = None

CONTRACT = ROOT / "models/mimo26/contract/encoders"
GRID = (1, 4, 4)
# sha256 of the cos then the sin F32 bytes of MiMo's bundle rotary tables
# (kernel-core `ComputedTable` with `AngleRule::MulRecipPowF32`, as the
# bundle's `RotaryRows::push` calls it), computed by a Rust program on
# glibc 2.39: vision over every (row, column) in 0..128 x 0..128 (theta 1e4,
# 32 axial pairs), audio over positions 0..6000 (theta 1e4, 32 pairs), the
# local groups over 0..4 (theta 640,000).
BUNDLE_TABLES = {
    "vision": "2f26bedce8a298c0b2c41916dde7dca6ed5c35bb009117fc999a040e94a49a0a",
    "audio": "e7991ec9e9807a864b40720df5da567ccb00b849f1188f0ede47b5f8b6c4fd53",
    "local": "b037d4db44b791f7106d49ab19090def7f5cd4ab2e413c353cf5182ef3a54220",
}


def tiny_vision():
    """mimo-vision.json at the tiny vision tower's sizes."""
    data = json.loads((CONTRACT / "mimo-vision.json").read_text())
    data["checkpoint"].update(config=[], files={"visual.": "model.safetensors"})
    patch, tower, _, _, up, down = data["stages"]
    patch["out"] = 32
    tower.update(blocks=3, width=32)
    tower["attention"].update(q_heads=2, kv_heads=1, head_dim=64, mask={"window": [2, 2]})
    tower["overrides"][0]["blocks"] = "0"
    tower["reorder"][0]["blocks"] = "2"
    tower["weights"]["sinks"]["blocks"] = "1-2"
    tower["rotary"]["pairs_per_axis"] = 16
    tower["mlp"]["hidden"] = 48
    up["out"], down["out"] = 128, 64
    data["output"]["hidden"] = 64
    return data


def tiny_audio():
    """mimo-audio.json at the tiny audio tokenizer's and encoder's sizes."""
    data = json.loads((CONTRACT / "mimo-audio.json").read_text())
    data["checkpoint"]["config"] = []
    data["checkpoint"]["files"].update({"audio_encoder.": "model.safetensors",
                                        "speech_embeddings.": "model.safetensors"})
    data["input"]["recipes"][0]["mel_bins"] = 8
    data["input"]["columns"] = 8
    (conv1, conv2, tokenizer, _, downsample, _, quantize, _, embed, local, _, _, up, down) = data["stages"]
    conv1["out"] = conv2["out"] = downsample["out"] = 32
    tokenizer.update(blocks=2, width=32, carry=[{"save_after": 0, "add_after": 1}])
    tokenizer["attention"].update(q_heads=2, kv_heads=2, head_dim=16)
    tokenizer["overrides"][0]["attention"]["mask"] = {"causal_window": 4}
    tokenizer["rotary"]["pairs"] = 8
    tokenizer["mlp"]["hidden"] = 64
    tokenizer["weights"]["qkv.bias"]["concat_rows"][1] = {"zeros": 32}
    quantize["codebooks"] = [16, 16, 16]
    quantize["weights"]["codebooks"]["count"] = 3
    embed.update(vocab=16, width=32)
    embed["weights"]["tables"]["count"] = 3
    local.update(blocks=1, width=32)
    local["attention"].update(q_heads=2, kv_heads=2, head_dim=16)
    local["rotary"]["pairs"] = 8
    local["mlp"]["hidden"] = 64
    local["weights"].update({"proj.bias": {"zeros": 32}, "gate_up.bias": {"zeros": 128}, "down.bias": {"zeros": 32}})
    up["out"], down["out"] = 512, 64
    data["output"]["hidden"] = 64
    return data


def relative(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


@unittest.skipIf(torch is None, "needs torch")
class BundleTables(unittest.TestCase):
    @unittest.skipUnless(platform.libc_ver() == ("glibc", "2.39"),
                         "the digests are of glibc 2.39's F32 functions; run the reference checks on a "
                         "glibc 2.39 host (Ubuntu 24.04)")
    def test_rotary_tables_are_the_bundles_bit_for_bit(self):
        spec = importlib.util.spec_from_file_location("mimo26_encoder_reference",
                                                      Path(__file__).with_name("encoder_reference.py"))
        variants = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(variants)
        rows = torch.arange(6000)[:, None]
        for name, positions, denominator, theta in (
                ("vision", torch.cartesian_prod(torch.arange(128), torch.arange(128)), 32, 1e4),
                ("audio", rows, 64, 1e4), ("local", rows[:4], 64, 640000.0)):
            cos, sin = variants.f32_table(positions, 32, denominator, theta)
            digest = hashlib.sha256(cos.numpy().tobytes() + sin.numpy().tobytes()).hexdigest()
            self.assertEqual(digest, BUNDLE_TABLES[name], name)


@unittest.skipIf(torch is None, "needs torch")
@unittest.skipUnless(tiny.VENDOR_DIR,
                     "needs the MiMo vendor modeling files from CHOPINFER_MIMO_CHECKPOINT or CHOPINFER_MIMO_TOKENIZER_DIR")
class MimoEncoderReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modes.prepare_environment(cuda=False)
        cls.directory = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        root = Path(cls.directory.name)
        cls.reference, _ = run.load_adapter(ROOT / "models/mimo26/reference.py")
        digest = hashlib.sha256(json.dumps(tiny.TOKENIZER).encode()).hexdigest()
        cls.reference.VENDOR = {**cls.reference.VENDOR, "audio_tokenizer/config.json": digest}
        generator = torch.Generator().manual_seed(13)
        tiny.write_checkpoint(root / "checkpoint", cls.reference, generator)
        cls.checkpoint = place.Checkpoint(root / "checkpoint", cls.reference.EXTRA_SHARDS)
        units = [u for u in cls.reference.units(cls.checkpoint) if u.module in cls.reference.ENCODERS]
        cls.placement = place.plan([(u.module, cls.checkpoint.stored_bytes(u.tensors)) for u in units], ["cpu"])
        cls.store = place.Store(cls.checkpoint, cls.placement, units)
        cls.pixels = (torch.rand((16, 1536), generator=generator) * 2 - 1).to(torch.bfloat16)
        cls.mel = torch.randn((tiny.MEL_FRAMES, 8), generator=generator)
        cls.declarations = {}
        for name, data in (("vision", tiny_vision()), ("audio", tiny_audio())):
            path = root / f"{name}.json"
            path.write_text(json.dumps(data))
            cls.declarations[name] = encoder.Declaration(path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def vendor(self, dtype, policy):
        """The vendor's encoder rows, audio codes and block outputs, all in
        row order. The tiny tower's last block is a column block, after
        which the vendor merges the rows still in column order (the real
        tower ends with a full-attention block, which restores row order),
        so its merged rows and that block's rows are put back here."""
        config = adapter.BuildConfig(self.checkpoint, modes.policy(policy), dtype, self.store)
        model = self.reference.build_encoders(config, self.placement).model
        blocks = {"vision": [], "audio": []}
        for block in model.visual.blocks:
            block.register_forward_hook(lambda _m, _a, out: blocks["vision"].append(out.clone()))
        # Clones: the tokenizer adds its carry to the last layer's output in place.
        layers = [*model.audio_tokenizer.encoder.layers, *model.audio_encoder.input_local_transformer.layers]
        for layer in layers:
            layer.register_forward_hook(lambda _m, _a, out: blocks["audio"].append(
                (out[0] if isinstance(out, tuple) else out).reshape(-1, 32).clone()))
        with modes.active(modes.policy(policy)):
            vision = model.visual(pixel_values=self.pixels, grid_thw=torch.tensor([GRID]))
            codes = self.reference.audio_codes(model, self.mel, model.audio_encoder.audio_segment_size)
            audio = model.audio_encoder(speech_embeddings=model.speech_embeddings, audio_codes=codes)
        order = encoder_ops.merge_group_columns(GRID)
        blocks["vision"][2] = blocks["vision"][2][torch.argsort(order)]
        vision = vision[torch.argsort(order[::4] // 4)]
        return {"vision": vision, "audio": audio}, codes, blocks

    def generic(self, mode, precision=None):
        out, recorded = {}, {}
        for name, item in (("vision", encoder.Item("image", self.pixels, GRID)),
                           ("audio", encoder.Item("audio", self.mel))):
            declaration = self.declarations[name]
            names = encoder.Names(declaration, self.checkpoint, self.reference.EXTRA_SHARDS)
            if mode == "policy":
                precision = precision or declaration.precision
            model = encoder.build(declaration, names, self.store.tensors, mode, precision, ["cpu"])
            recorded[name] = []
            for tap, hook in encoder.tap_map(model).items():
                if ".B" in tap or tap.endswith("x.codes") or tap.endswith("x.unrounded"):
                    module = model.get_submodule(hook.module)
                    module.register_forward_hook(
                        lambda _m, args, value, tap=tap: recorded[name].append((tap, args[0] if "unrounded" in tap
                                                                                else value)))
            with modes.active(encoder.mode_document(mode, precision)):
                out[name] = model(item)
        return out, recorded

    def compare(self, mode, dtype, policy, bound, precision=None, unrounded=False):
        expected, codes, blocks = self.vendor(dtype, policy)
        out, recorded = self.generic(mode, precision)
        ours = dict(recorded["audio"])
        self.assertTrue(torch.equal(ours["enc.audio.x.codes"], codes), mode)
        errors = {}
        for name in ("vision", "audio"):
            rows = dict(recorded[name])["enc.%s.x.unrounded" % name] if unrounded else out[name]
            errors[name] = relative(rows, expected[name])
            self.assertLess(errors[name], bound, (mode, name))
            ours = [value for tap, value in recorded[name] if ".B" in tap]
            self.assertEqual(len(ours), len(blocks[name]))
            for index, (got, want) in enumerate(zip(ours, blocks[name])):
                self.assertLess(relative(got, want), bound, (mode, name, index))
        return out, errors

    def test_exact_matches_the_vendor_in_f64(self):
        out, _ = self.compare("exact", torch.float64, "f64", 1e-6)
        self.assertEqual(out["audio"].dtype, torch.float64)

    def test_policy_and_reference_bf16_stay_near_the_vendor_in_bf16(self):
        for mode in ("policy", "reference-bf16"):
            out, _ = self.compare(mode, torch.bfloat16, "bf16", 0.05)
            self.assertEqual(out["vision"].dtype, torch.bfloat16)

    def test_the_f32_policy_matches_the_vendor_in_f32(self):
        self.compare("policy", torch.float32, "f32", 2e-4, "f32-pairs.v1", unrounded=True)


if __name__ == "__main__":
    unittest.main()

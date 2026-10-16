"""The MiMo reference adapter on a tiny random checkpoint in MiMo's layout, on CPU.

The checkpoint stores what the real one stores: TP-chunked FP8 QKV with F32
scales on a chunked grid, an FP8 layer-0 MLP, MXFP4 experts, BF16 o_proj,
sinks, the vision tower without merger biases, the audio encoder and the
audio tokenizer in its own file. The runner's F32 and BF16 scores must equal
a separately loaded copy of the vendor model over weights decoded here by
the definitions and de-interleaved by reshaping. That copy shares three
adaptations with the adapter, so they are not tested against the vendor:

- the merger normalization (`reference.MERGERS`, a model-semantic choice
  the checkpoint does not fix): RMSNorm for reference.py, the vendor's
  LayerNorm without bias for reference_merger_layernorm.py; the test
  validates each implementation, not which one the model was trained with;
- the merger linears without biases (none are stored);
- the causal and sliding-window masks of `reference.forward`, which are
  checked against their definition separately.

Needs torch and the vendor modeling files
($CHOPINFER_MIMO_CHECKPOINT, or the tokenizer directory CI passes, which is
the same checkpoint); skipped otherwise.
"""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.model_sdk import taps as tap_names

try:
    import numpy as np
    import torch
    from torch import nn
    from tools.refrun import modes, place, quant, run, stamps
    from tools.refrun.test_quant import fp8_codes, naive
except ModuleNotFoundError:
    torch = None

VENDOR_DIR = os.environ.get("CHOPINFER_MIMO_CHECKPOINT") or os.environ.get("CHOPINFER_MIMO_TOKENIZER_DIR")
ROOT = Path(__file__).resolve().parents[2]
IMAGE, VIDEO, AUDIO = 150, 151, 152
TEXT = {
    "architectures": ["MiMoV2ForCausalLM"], "model_type": "mimo_v2", "vocab_size": 160, "hidden_size": 64,
    "intermediate_size": 128, "moe_intermediate_size": 32, "num_hidden_layers": 3,
    "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 24, "v_head_dim": 16,
    "swa_num_attention_heads": 8, "swa_num_key_value_heads": 8, "swa_head_dim": 24, "swa_v_head_dim": 16,
    "hybrid_layer_pattern": [0, 1, 1], "moe_layer_freq": [0, 1, 1], "sliding_window": 4,
    "add_swa_attention_sink_bias": True, "add_full_attention_sink_bias": False,
    "attention_projection_layout": "fused_qkv", "attention_value_scale": 0.707, "partial_rotary_factor": 0.334,
    "rope_theta": 10000000.0, "swa_rope_theta": 10000.0,
    "rope_parameters": {"partial_rotary_factor": 0.334, "rope_theta": 10000000.0, "rope_type": "default",
                        "type": "default"},
    "n_routed_experts": 4, "num_experts_per_tok": 2, "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
    "scoring_func": "sigmoid", "topk_method": "noaux_tc", "routed_scaling_factor": None,
    "layernorm_epsilon": 1e-6, "max_position_embeddings": 256, "tie_word_embeddings": False,
    "image_token_id": IMAGE, "video_token_id": 151,
    "vision_config": {"depth": 3, "hidden_size": 32, "intermediate_size": 48, "num_heads": 2,
                      "num_key_value_heads": 1, "out_hidden_size": 64, "patch_size": 16, "spatial_merge_size": 2,
                      "temporal_patch_size": 2, "in_chans": 3, "hidden_act": "silu", "fullatt_block_indexes": [0],
                      "use_sink": True, "visual_token_window_size": 2, "vit_window_attn_types": [-1, 0, 1]},
    "audio_config": {"audio_channels": 3, "group_size": 4, "input_local_dim": 32, "input_local_layers": 1,
                     "input_local_attn_heads": 2, "input_local_intermediate_size": 64, "out_hidden_size": 64,
                     "projection_layers": 2, "rope_theta": 640000, "partial_rotary_factor": 1.0,
                     "speech_vocab_size": "16", "speech_zeroemb_idx": "15", "add_post_norm": True,
                     "input_full_attention": True, "audio_segment_size": 6000},
    "processor_config": {"audio_token_id": AUDIO},
}
TOKENIZER = {"d_model": 32, "encoder_layers": 2, "encoder_attention_heads": 2, "encoder_ffn_dim": 64,
             "n_mels": 8, "kernel_size": 3, "stride_size": 2, "avg_pooler": 2, "num_quantizers": 3,
             "codebook_size": [16, 16, 16], "hybrid_attention": True, "swa_per_block": 2,
             "encoder_attn_window_size": [4, 0], "encoder_causal": True, "encoder_skip_layer_id": 1,
             "rope_theta": 10000, "scale_embedding": False, "max_audio_seconds": 10, "ln_type": "LayerNorm"}
MEL_FRAMES, IMAGE_GRID = 40, (1, 4, 4)


class Files:
    """The part of place.Checkpoint the vendor loader reads, before any shard exists."""

    def __init__(self, root):
        self.root = Path(root)
        self.config = json.loads((self.root / "config.json").read_text())

    def pinned(self, file, sha256):
        return place.Checkpoint.pinned(self, file, sha256)


def fp8_scale_rows(rows, chunk):
    per_chunk = -(-chunk // 128)
    return [r // chunk * per_chunk + r % chunk // 128 for r in range(rows)]


def write_checkpoint(root, reference, generator):
    """The tiny checkpoint, its lock, and the f64 dense weights by definition
    in the vendor's own row order."""
    root.mkdir(parents=True)
    (root / "audio_tokenizer").mkdir()
    for file in ("configuration_mimo_v2.py", "modeling_mimo_v2.py"):
        (root / file).write_bytes((Path(VENDOR_DIR) / file).read_bytes())
    (root / "config.json").write_text(json.dumps(TEXT))
    (root / "audio_tokenizer/config.json").write_text(json.dumps(TOKENIZER))
    plain = plain_vendor(root, reference)
    text_config, tokenizer_config = reference.configs(Files(root), plain)
    with torch.device("meta"):
        model = plain.MiMoV2ForCausalLM(text_config)
        tokenizer = plain.MiMoAudioTokenizer(tokenizer_config)
    unstored = {"visual.merger.ln_q.bias", "visual.merger.mlp.0.bias", "visual.merger.mlp.2.bias",
                "audio_encoder.input_local_transformer.embed_tokens.weight"}
    tensors, dense = {}, {}
    for name, value in model.state_dict().items():
        if name in unstored:
            continue
        if name.endswith("e_score_correction_bias"):
            tensors[name] = torch.randn(value.shape, generator=generator, dtype=torch.float32) * 0.05
        elif name.endswith("norm.weight") or "layernorm" in name or name.endswith(("norm1.weight", "norm2.weight")):
            tensors[name] = (1 + 0.1 * torch.randn(value.shape, generator=generator)).to(torch.bfloat16)
        elif "qkv_proj" in name or ".experts." in name or name.startswith("model.layers.0.mlp."):
            continue
        else:
            tensors[name] = (torch.randn(value.shape, generator=generator) * 0.2).to(torch.bfloat16)
    for layer, block in enumerate(model.model.layers):
        stem = f"model.layers.{layer}"
        attention = block.self_attn
        rows = attention.q_size + attention.k_size + attention.v_size
        chunk = rows // 4
        codes = fp8_codes(generator, (rows, 64))
        scales = torch.rand((4 * -(-chunk // 128), 1), generator=generator) * 0.01 + 0.002
        tensors[f"{stem}.self_attn.qkv_proj.weight"] = codes.view(torch.float8_e4m3fn)
        tensors[f"{stem}.self_attn.qkv_proj.weight_scale_inv"] = scales
        scale_rows = fp8_scale_rows(rows, chunk)
        stored = naive(codes, lambda r, c: float(scales[scale_rows[r], c // 128]), rows, 64, "e4m3")
        parts = stored.view(4, chunk, 64).split([attention.q_size // 4, attention.k_size // 4,
                                                 attention.v_size // 4], dim=1)
        dense[f"{stem}.self_attn.qkv_proj.weight"] = torch.cat([p.reshape(-1, 64) for p in parts])
        if layer == 0:
            for projection, shape in (("gate_proj", (128, 64)), ("up_proj", (128, 64)), ("down_proj", (64, 128))):
                codes = fp8_codes(generator, shape)
                scales = torch.rand((1, 1), generator=generator) * 0.01 + 0.002
                tensors[f"{stem}.mlp.{projection}.weight"] = codes.view(torch.float8_e4m3fn)
                tensors[f"{stem}.mlp.{projection}.weight_scale_inv"] = scales
                dense[f"{stem}.mlp.{projection}.weight"] = naive(codes, lambda r, c: float(scales[0, 0]),
                                                                 *shape, "e4m3")
            continue
        for expert in range(4):
            for projection, (rows, cols) in (("gate_proj", (32, 64)), ("up_proj", (32, 64)),
                                             ("down_proj", (64, 32))):
                name = f"{stem}.mlp.experts.{expert}.{projection}.weight"
                codes = torch.randint(0, 256, (rows, cols // 2), generator=generator, dtype=torch.uint8)
                scales = torch.randint(118, 124, (rows, cols // 32), generator=generator, dtype=torch.uint8)
                tensors[name], tensors[name + "_scale"] = codes, scales
                dense[name] = naive(codes, lambda r, c: quant.E8M0[int(scales[r, c // 32])], rows, cols, "e2m1")
    codec = {}
    for name, value in tokenizer.state_dict().items():
        if not name.startswith("encoder."):
            continue
        if ".quantizer." in name:
            codec[name] = torch.randn(value.shape, generator=generator, dtype=torch.float32)
        else:
            codec[name] = (1 + 0.1 * torch.randn(value.shape, generator=generator)).to(torch.bfloat16) \
                if "layer_norm" in name or name.endswith("norm.weight") \
                else (torch.randn(value.shape, generator=generator) * 0.2).to(torch.bfloat16)
    from safetensors.torch import save_file
    save_file(tensors, str(root / "model.safetensors"))
    save_file(codec, str(root / "audio_tokenizer/model.safetensors"))
    lock = {}
    for file in ("config.json", "model.safetensors", "audio_tokenizer/model.safetensors"):
        data = (root / file).read_bytes()
        lock[file] = {"byte_length": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (root / "checkpoint-files.json").write_text(json.dumps(lock))
    stored = {k: v.double() for k, v in tensors.items() if k.startswith("model.") or k.startswith("lm_head")
              or k.startswith(("visual.", "speech_embeddings.", "audio_encoder."))}
    weights = {k: v for k, v in stored.items() if not k.endswith(("_scale", "_scale_inv"))
               and k not in dense and not (k.endswith(".weight") and f"{k}_scale" in tensors)
               and not (k.endswith(".weight") and f"{k}_scale_inv" in tensors)}
    return {**weights, **dense}, set(dense), codec


def plain_vendor(root, reference):
    """An unpatched copy of the vendor modules (a package of its own)."""
    import sys
    name = "mimo26_test_plain_vendor"
    if f"{name}.modeling_mimo_v2" not in sys.modules:
        reference.load_vendor(Files(root), name)
    return sys.modules[f"{name}.modeling_mimo_v2"]


def plain_model(root, reference, weights, dense, codec, dtype, rule, merger):
    """The vendor model over dense weights, with the merger normalization
    `merger` (RMSNorm, or the vendor's LayerNorm without its bias) and no
    merger biases."""
    plain = plain_vendor(root, reference)
    text_config, tokenizer_config = reference.configs(Files(root), plain)
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = plain.MiMoV2ForCausalLM(text_config).eval()
        model.audio_tokenizer = plain.MiMoAudioTokenizer(tokenizer_config).eval()
        if merger == "rmsnorm":
            model.visual.merger.ln_q = nn.RMSNorm(32, eps=1e-6)
        else:
            model.visual.merger.ln_q.bias = None
    finally:
        torch.set_default_dtype(default)
    for index in (0, 2):
        model.visual.merger.mlp[index].bias = None

    def cast(name, value):
        if name in dense and dtype != torch.float64:
            value = value.float()
            if rule == "hf-bf16":
                value = value.to(torch.bfloat16)
        return value.to(dtype)
    result = model.load_state_dict({name: cast(name, value) for name, value in weights.items()}, strict=False)
    missing = [n for n in result.missing_keys if not n.startswith("audio_tokenizer.")
               and n != "audio_encoder.input_local_transformer.embed_tokens.weight"]
    if result.unexpected_keys or missing:
        raise AssertionError(f"plain weights do not fit: {result.unexpected_keys[:3]} {missing[:3]}")
    result = model.audio_tokenizer.load_state_dict({k: v.to(dtype) for k, v in codec.items()}, strict=False)
    if result.unexpected_keys or any(n.startswith("encoder.") for n in result.missing_keys):
        raise AssertionError(f"plain tokenizer weights do not fit: {result.missing_keys[:3]}")
    return model


@unittest.skipIf(torch is None, "needs torch")
class MediaInputsTests(unittest.TestCase):
    def test_multiple_images_keep_their_rows_and_grids_with_other_media(self):
        from types import SimpleNamespace
        reference, _ = run.load_adapter(ROOT / "models/mimo26/reference.py")
        model = SimpleNamespace(audio_encoder=SimpleNamespace(audio_segment_size=6000))
        def spec(value):
            return SimpleNamespace(data=value.numpy().tobytes(), dtype="f32" if value.is_floating_point() else "i64",
                                   shape=tuple(value.shape))
        def item(kind, value):
            return SimpleNamespace(kind=kind, tensor=spec(torch.full((4, 3), float(value))),
                                   grid=spec(torch.tensor([[1, 2, 2]])))
        for count in (2, 32):
            images = [item("image", i) for i in range(count)]
            for extra in ([], [item("video", 99)], [item("audio", 98)]):
                sequence = SimpleNamespace(name="mixed", media=[*extra, *images])
                with mock.patch.object(reference, "audio_codes", return_value="codes") as audio:
                    result = reference.media_inputs(model)(sequence, "cpu", torch.float32)
                self.assertTrue(torch.equal(result["pixel_values"], torch.cat([
                    torch.full((4, 3), float(i)) for i in range(count)])))
                self.assertEqual(result["image_grid_thw"].tolist(), [[1, 2, 2]] * count)
                self.assertEqual("video_pixel_values" in result, bool(extra and extra[0].kind == "video"))
                self.assertEqual(audio.call_count, int(bool(extra and extra[0].kind == "audio")))

    def test_serving_media_limits_still_apply(self):
        from types import SimpleNamespace
        reference, _ = run.load_adapter(ROOT / "models/mimo26/reference.py")
        inputs = reference.media_inputs(SimpleNamespace(audio_encoder=SimpleNamespace(audio_segment_size=6000)))
        for kind, count in (("image", 33), ("video", 2), ("audio", 2), ("unknown", 1)):
            with self.assertRaises(ValueError):
                inputs(SimpleNamespace(name="invalid", media=[SimpleNamespace(kind=kind)] * count),
                       "cpu", torch.float32)


@unittest.skipIf(torch is None, "needs torch")
@unittest.skipUnless(VENDOR_DIR,
                     "needs the MiMo vendor modeling files from CHOPINFER_MIMO_CHECKPOINT or CHOPINFER_MIMO_TOKENIZER_DIR")
class TinyMimoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modes.prepare_environment(cuda=False)
        cls.saved_racy, stamps.RACY_NS = stamps.RACY_NS, 100_000_000
        cls.directory = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        root = Path(cls.directory.name)
        cls.reference, cls.reference_path = run.load_adapter(ROOT / "models/mimo26/reference.py")
        cls.control, cls.control_path = run.load_adapter(ROOT / "models/mimo26/reference_contiguous_qkv.py")
        cls.layernorm, cls.layernorm_path = run.load_adapter(ROOT / "models/mimo26/reference_merger_layernorm.py")
        digest = hashlib.sha256(json.dumps(TOKENIZER).encode()).hexdigest()
        for module in (cls.reference, cls.control.reference, cls.layernorm.reference):
            module.VENDOR = {**module.VENDOR, "audio_tokenizer/config.json": digest}
        generator = torch.Generator().manual_seed(11)
        cls.checkpoint = root / "checkpoint"
        cls.weights, cls.dense, cls.codec = write_checkpoint(cls.checkpoint, cls.reference, generator)
        pixels = (torch.rand((16, 1536), generator=generator) * 2 - 1).to(torch.bfloat16)
        mel = torch.randn((MEL_FRAMES, 8), generator=generator)
        (root / "grid.bin").write_bytes(torch.tensor([IMAGE_GRID], dtype=torch.int64).numpy().tobytes())
        (root / "mel.bin").write_bytes(mel.numpy().tobytes())
        processor = {"code_sha256": "0" * 64, "config_sha256": "0" * 64}
        text = torch.randint(0, 140, (24,), generator=generator).tolist()
        # Drawn last, so the other sequences' inputs stay as they were.
        second, video = ((torch.rand((16, 1536), generator=generator) * 2 - 1).to(torch.bfloat16) for _ in range(2))
        for name, values in (("pixels", pixels), ("second", second), ("video", video)):
            (root / f"{name}.bin").write_bytes(values.view(torch.int16).numpy().tobytes())
        media = [3, 7, IMAGE, IMAGE, IMAGE, IMAGE, 9, AUDIO, AUDIO, AUDIO, 11, 12, 13, 14, 15]
        # A video before an image: the tower runs for the image first.
        interleaved = [3, *[VIDEO] * 4, 9, *[IMAGE] * 4, 10, 11, 12, 13]

        def vision(kind, path):
            return {"kind": kind, "processor": processor,
                    "tensor": {"path": path, "dtype": "bf16", "shape": [16, 1536]},
                    "grid": {"path": "grid.bin", "dtype": "i64", "shape": [1, 3]}}
        cls.sequences = [
            {"name": "text", "category": "prose", "token_ids": text, "score_from": 1},
            {"name": "media", "category": "image", "token_ids": media, "score_from": 10, "media": [
                {"kind": "image", "processor": processor,
                 "tensor": {"path": "pixels.bin", "dtype": "bf16", "shape": [16, 1536]},
                 "grid": {"path": "grid.bin", "dtype": "i64", "shape": [1, 3]}},
                {"kind": "audio", "processor": processor,
                 "tensor": {"path": "mel.bin", "dtype": "f32", "shape": [MEL_FRAMES, 8]}}]},
            {"name": "interleaved", "category": "image", "token_ids": interleaved, "score_from": 11, "media": [
                vision("video", "video.bin"), vision("image", "second.bin")]},
            {"name": "two-images", "category": "image", "token_ids": [3, *[IMAGE] * 4, 9, *[IMAGE] * 4, 10, 11],
             "score_from": 11, "media": [vision("image", "pixels.bin"), vision("image", "second.bin")]},
            {"name": "two-images-video", "category": "image",
             "token_ids": [3, *[IMAGE] * 4, 9, *[VIDEO] * 4, 10, *[IMAGE] * 4, 11, 12],
             "score_from": 16, "media": [vision("image", "pixels.bin"), vision("video", "video.bin"),
                                          vision("image", "second.bin")]}]
        cls.pixels, cls.second, cls.video, cls.mel = pixels, second, video, mel
        workload = root / "workload.json"
        workload.write_text(json.dumps({"schema": "chopinfer.refrun.workload.v1", "model": "mimo26",
                                        "sequences": cls.sequences}))
        cls.scores = {}
        for label, (module, path) in {"reference": (cls.reference, cls.reference_path),
                                      "control": (cls.control, cls.control_path),
                                      "layernorm": (cls.layernorm, cls.layernorm_path)}.items():
            output = root / label
            with mock.patch.object(run, "load_adapter", return_value=(module, path)):
                cls.provenance = run.run(run.parse_args([
                    "--adapter", str(ROOT / path), "--checkpoint", str(cls.checkpoint),
                    "--lock", str(cls.checkpoint / "checkpoint-files.json"), "--workload", str(workload),
                    "--modes", "f32,bf16,f64" if label == "reference" else "f32", "--dequant", "f32=rounded",
                    "--devices", "cpu,cpu", "--output", str(output), "--stamps", str(root / "stamps"),
                    "--no-cache", *(["--taps", "--tap-sites", "all"] if label == "reference" else [])]))
            cls.scores[label] = {mode: {e["name"]: e for e in json.loads((output / f"{mode}.json").read_text())
                                        ["entries"]} for mode in ("f32", "bf16", "f64") if (output / f"{mode}.json").exists()}

    @classmethod
    def tearDownClass(cls):
        stamps.RACY_NS = cls.saved_racy
        cls.directory.cleanup()

    def plain_scores(self, dtype, rule, sequence, merger):
        from tools.refrun import score
        model = plain_model(self.checkpoint, self.reference, self.weights, self.dense, self.codec, dtype, rule,
                            merger)
        plain = plain_vendor(self.checkpoint, self.reference)
        ids = torch.tensor([sequence["token_ids"]])
        media = {}
        if sequence["name"] == "media":
            media = {"pixel_values": self.pixels, "image_grid_thw": torch.tensor([IMAGE_GRID]),
                     "audio_codes": plain.tokenize_audio_batch([self.mel], model.audio_tokenizer.encoder)[0]}
        if sequence["name"] == "interleaved":
            media = {"pixel_values": self.second, "image_grid_thw": torch.tensor([IMAGE_GRID]),
                     "video_pixel_values": self.video, "video_grid_thw": torch.tensor([IMAGE_GRID])}
        if sequence["name"].startswith("two-images"):
            media = {"pixel_values": torch.cat([self.pixels, self.second]),
                     "image_grid_thw": torch.tensor([IMAGE_GRID, IMAGE_GRID])}
            if sequence["name"].endswith("video"):
                media.update(video_pixel_values=self.video, video_grid_thw=torch.tensor([IMAGE_GRID]))
        policy = modes.policy({torch.float32: "f32", torch.bfloat16: "bf16"}[dtype])
        with modes.active(policy):
            keep = len(sequence["token_ids"]) - sequence["score_from"] + 1
            logits = self.reference.forward(model, ids, keep, media)
            return score.rows(logits[:-1], sequence["token_ids"][sequence["score_from"]:],
                              modes.torch_dtype(policy["log_softmax"]))

    def test_scores_equal_the_vendor_model_with_the_shared_merger_and_mask_adaptations(self):
        runs = (("reference", "rmsnorm", torch.float32, "f32", "rounded"),
                ("reference", "rmsnorm", torch.bfloat16, "bf16", "hf-bf16"),
                ("layernorm", "layernorm", torch.float32, "f32", "rounded"))
        for label, merger, dtype, mode, rule in runs:
            for sequence in self.sequences:
                expected = self.plain_scores(dtype, rule, sequence, merger)
                got = self.scores[label][mode][sequence["name"]]["scores"]
                for column in ("logprob", "top1", "topk_ids", "logit"):
                    self.assertEqual(got[column], expected[column], (label, mode, sequence["name"], column))

    def test_the_merger_choice_changes_only_media_and_is_recorded(self):
        rms, layer = self.scores["reference"]["f32"], self.scores["layernorm"]["f32"]
        self.assertEqual(rms["text"]["scores"], layer["text"]["scores"])
        self.assertNotEqual(rms["media"]["scores"]["logprob"], layer["media"]["scores"]["logprob"])
        for label, merger in (("reference", "rmsnorm"), ("layernorm", "layernorm")):
            layout = json.loads((Path(self.directory.name) / label / "run.json").read_text())["layout"]
            self.assertIn(merger, layout["visual.merger.ln_q"])
        self.assertIn("overriding the vendor's LayerNorm", self.reference.MERGERS["rmsnorm"])

    def test_f64_keeps_f64_and_stays_near_f32(self):
        for name in ("text", "media"):
            f64 = self.scores["reference"]["f64"][name]["scores"]["logprob"]
            f32 = self.scores["reference"]["f32"][name]["scores"]["logprob"]
            self.assertNotEqual(f64, f32)
            self.assertLess(max(abs(a - b) for a, b in zip(f64, f32)), 1e-3)

    def test_the_contiguous_qkv_control_scores_differently(self):
        got = self.scores["control"]["f32"]["text"]["scores"]["logprob"]
        expected = self.scores["reference"]["f32"]["text"]["scores"]["logprob"]
        self.assertGreater(sum(abs(a - b) for a, b in zip(got, expected)) / len(got), 1e-2)
        layout = json.loads((Path(self.directory.name) / "control/run.json").read_text())["layout"]
        self.assertIn("negative control", layout["qkv_proj"])

    def test_a_cached_vendor_module_is_refused_for_other_vendor_bytes(self):
        self.reference.vendor(Files(self.checkpoint))
        other = Path(self.directory.name) / "edited"
        other.mkdir()
        (other / "config.json").write_text(json.dumps(TEXT))
        (other / "configuration_mimo_v2.py").write_bytes((self.checkpoint / "configuration_mimo_v2.py").read_bytes())
        (other / "modeling_mimo_v2.py").write_bytes(
            (self.checkpoint / "modeling_mimo_v2.py").read_bytes() + b"# edited\n")
        with self.assertRaisesRegex(ValueError, "not the pinned"):
            self.reference.vendor(Files(other))
        with self.assertRaisesRegex(ValueError, "not the pinned"):
            self.reference.reference_code(Files(other))
        self.reference.vendor(Files(self.checkpoint))

    def test_taps_follow_the_decoder_and_diff_against_themselves(self):
        """The scores above were taken with every tap hooked. Each layer's
        output is its input plus the attention and MLP outputs, router ids
        and weights come from the MoE layers' gate, encoder rows are the rows
        merged into the sequence, and the bf16 taps scored as an engine
        against f32 sit at the floor."""
        from tools.qual import tapdiff
        root = Path(self.directory.name) / "reference" / "taps"
        step = root / "f32" / "step0001"
        index = {e["name"]: e for e in map(json.loads, (step / "index.jsonl").read_text().splitlines())}
        layers = TEXT["num_hidden_layers"]
        self.assertEqual(json.loads((step / "completion.json").read_text())["completion"], "full")
        self.assertEqual(list(index)[:4], ["embed.out", "enc.vision.out", "enc.audio.out", "L00.in"])
        self.assertEqual(list(index)[-1], "final.norm")
        self.assertNotIn("L00.router.ids", index)
        self.assertEqual((index["L01.router.ids"]["dtype"], index["L01.router.weights"]["dtype"]), ("u32", "f32"))
        self.assertEqual(index["L01.in"]["rows"], len(self.sequences[1]["token_ids"]))
        # The engine declares every runner tap (models/mimo26/taps.json).
        declaration = tap_names.model_declaration("mimo26")
        self.assertEqual([name for name in index if declaration.entry(name) is None], [])

        def read(name):
            return tapdiff.read(step / f"{name}.r0.bin", index[name])
        for layer in range(layers):
            scope = f"L{layer:02d}"
            self.assertTrue(np.array_equal((read(f"{scope}.in") + read(f"{scope}.attn.out")) + read(f"{scope}.mlp.out"),
                                           read(f"{scope}.out")), scope)
        # The encoder rows are the rows merged into the sequence at its image and audio positions.
        ids = self.sequences[1]["token_ids"]
        for name, token in (("enc.vision.out", IMAGE), ("enc.audio.out", AUDIO)):
            self.assertEqual(index[name]["axis"], "encoder")
            self.assertTrue(np.array_equal(read(name), read("L00.in")[[p for p, t in enumerate(ids) if t == token]]))
        # The video-then-image request: the image's call, then the video's;
        # each item's rows are recorded under its kind and are the rows
        # merged at its own placeholders.
        step = root / "f32" / "step0002"
        index = {e["name"]: e for e in map(json.loads, (step / "index.jsonl").read_text().splitlines())}
        segments = json.loads((step / "items.json").read_text())["encoder"]["enc.vision.out"]
        self.assertEqual([(s["kind"], s["item"], s["start"], s["rows"]) for s in segments],
                         [("image", 0, 0, 4), ("video", 0, 4, 4)])
        rows = read("enc.vision.out")
        merged = read("L00.in")
        ids = self.sequences[2]["token_ids"]
        image_rows = [p for p, t in enumerate(ids) if t == IMAGE]
        video_rows = [p for p, t in enumerate(ids) if t == VIDEO]
        for (kind, item, start, count), at in zip(
                [(s["kind"], s["item"], s["start"], s["rows"]) for s in segments],
                [image_rows, video_rows]):
            self.assertTrue(np.array_equal(rows[start:start + count], merged[at]), (kind, item))
        text = root / "f32" / "step0000"
        self.assertEqual(json.loads((text / "completion.json").read_text())["completion"], "full")
        self.assertNotIn("enc.vision.out", (text / "index.jsonl").read_text())
        receipt = Path(self.directory.name) / "tapdiff.json"
        with contextlib.redirect_stdout(io.StringIO()):
            code = tapdiff.main(["--engine", str(root / "bf16"), "--f32", str(root / "f32"), "--bf16",
                                 str(root / "bf16"), "--f64", str(root / "f64"), "--json", str(receipt)])
        report = json.loads(receipt.read_text())
        self.assertEqual((code, report["status"], report["requests"]),
                         (0, "match", {sequence["name"]: sequence["name"] for sequence in self.sequences}))
        taps = {row["tap"]: row for row in report["taps"]}
        self.assertEqual((taps["L01.router.weights"]["compare"], taps["L01.router.weights"]["status"]),
                         ("paired", "ok"))
        self.assertEqual((taps["enc.vision.out"]["rows"], taps["enc.audio.out"]["rows"]), (32, 3))

    def test_answer_decode_matches_full_prefix_logits(self):
        answers, _ = run.load_adapter(ROOT / "models/mimo26/reference_answers.py")
        model = plain_model(self.checkpoint, self.reference, self.weights, self.dense, self.codec,
                            torch.float32, "rounded", "rmsnorm")
        ids = torch.tensor([[3, *[IMAGE] * 4, 9, *[VIDEO] * 4, 10]])
        media = {"pixel_values": self.pixels, "image_grid_thw": torch.tensor([IMAGE_GRID]),
                 "video_pixel_values": self.video, "video_grid_thw": torch.tensor([IMAGE_GRID])}
        with modes.active(modes.policy("f32")):
            answer, rows = answers.decode(model, ids, media, 8, set())
            for index, token in enumerate(answer):
                prefix = torch.cat((ids, torch.tensor([answer[:index]], dtype=torch.long)), dim=1)
                expected = self.reference.forward(model, prefix, 1, media)[0]
                self.assertEqual(token, int(expected.argmax()))
                torch.testing.assert_close(rows[index], expected, atol=2e-5, rtol=2e-5)
            stopped, _ = answers.decode(model, ids, media, 8, {answer[0]})
            self.assertEqual(stopped, answer[:1])

    def test_layout_and_replacements_are_recorded(self):
        layout = json.loads((Path(self.directory.name) / "reference/run.json").read_text())["layout"]
        self.assertIn("rows permuted", layout["qkv_proj"])
        self.assertIn("RMSNorm", layout["visual.merger.ln_q"])


@unittest.skipIf(torch is None, "needs torch")
class LayoutTests(unittest.TestCase):
    def test_qkv_rows_take_each_chunk_s_part_in_turn(self):
        from tools.refrun.run import load_adapter
        reference, _ = load_adapter(ROOT / "models/mimo26/reference.py")
        q, k, v = 64 * 192, 4 * 192, 4 * 128
        stored = torch.arange(q + k + v)
        chunks = stored.view(4, -1).split([q // 4, k // 4, v // 4], dim=1)
        expected = torch.cat([part.reshape(-1) for part in chunks])
        self.assertTrue(torch.equal(stored[reference.qkv_rows(q, k, v)], expected))
        self.assertEqual(reference.qkv_rows(8, 4, 4, chunks=2), [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15])
        with self.assertRaisesRegex(ValueError, "do not split"):
            reference.qkv_rows(6, 4, 4)

    def test_hf_eager_masks_are_causal_and_slide_over_the_window(self):
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
        from transformers import PretrainedConfig
        config = PretrainedConfig(sliding_window=4)
        config._attn_implementation = "eager"
        shape = torch.empty((1, 9, 1))
        full = create_causal_mask(config, shape, None, None)[0, 0] == 0
        window = create_sliding_window_causal_mask(config, shape, None, None)[0, 0] == 0
        i, j = torch.arange(9)[:, None], torch.arange(9)[None, :]
        self.assertTrue(torch.equal(full, j <= i))
        self.assertTrue(torch.equal(window, (j <= i) & (j > i - 4)))


@unittest.skipIf(torch is None, "needs torch")
class ReplacementTests(unittest.TestCase):
    """In f64 the replacements keep f64 where the vendor casts to F32."""

    @classmethod
    def setUpClass(cls):
        from tools.refrun.run import load_adapter
        cls.reference, _ = load_adapter(ROOT / "models/mimo26/reference.py")
        cls.generator = torch.Generator().manual_seed(5)

    def test_rms_norm(self):
        norm = self.reference.RMSNorm(16, 1e-6).double()
        x = torch.randn((3, 16), generator=self.generator, dtype=torch.float64)
        expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        self.assertTrue(torch.equal(norm(x), expected))
        x32 = x.float()
        self.assertTrue(torch.equal(norm.float()(x32), x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)))

    def test_eager_attention_blocks_and_sinks(self):
        module = nn.Module()
        module.num_key_value_groups = 2
        module.attention_sink_bias = torch.randn(4, generator=self.generator, dtype=torch.float64)
        q = torch.randn((1, 4, 7, 8), generator=self.generator, dtype=torch.float64)
        k, v = (torch.randn((1, 2, 7, 8), generator=self.generator, dtype=torch.float64) for _ in range(2))
        mask = torch.triu(torch.full((7, 7), float("-inf"), dtype=torch.float64), 1)[None, None]
        out, _ = self.reference.eager_attention(module, q, k, v, mask, 0.3, sinks=True)
        keys = k.repeat_interleave(2, dim=1)
        scores = torch.cat([q @ keys.transpose(2, 3) * 0.3 + mask,
                            module.attention_sink_bias.view(1, 4, 1, 1).expand(1, 4, 7, 1)], dim=-1)
        expected = (torch.softmax(scores, dim=-1)[..., :-1] @ v.repeat_interleave(2, dim=1)).transpose(1, 2)
        torch.testing.assert_close(out, expected, rtol=1e-14, atol=1e-15)
        with mock.patch.object(self.reference.ops, "CHUNK", 3):
            blocked, _ = self.reference.eager_attention(module, q, k, v, mask, 0.3, sinks=True)
        torch.testing.assert_close(blocked, out, rtol=1e-14, atol=1e-15)

    def test_router_and_vision_rotary_keep_f64(self):
        gate = nn.Module().eval()
        gate.__dict__.update(scoring_func="sigmoid", topk_method="noaux_tc", n_group=1, topk_group=1,
                             n_routed_experts=8, top_k=2, norm_topk_prob=True, routed_scaling_factor=1.0)
        gate.weight = nn.Parameter(torch.randn((8, 16), generator=self.generator, dtype=torch.float64))
        gate.e_score_correction_bias = nn.Parameter(torch.zeros(8, dtype=torch.float64))
        x = torch.randn((1, 5, 16), generator=self.generator, dtype=torch.float64)
        ids, weights = self.reference.route(gate, x)
        scores = torch.sigmoid(x[0] @ gate.weight.T)
        expected = scores.topk(2, dim=-1).values
        self.assertEqual(weights.dtype, torch.float64)
        torch.testing.assert_close(weights.sort(-1).values, (expected / expected.sum(-1, keepdim=True)).sort(-1).values,
                                   rtol=1e-15, atol=0)
        q = torch.randn((4, 2, 8), generator=self.generator, dtype=torch.float64)
        cos, sin = torch.rand((4, 8), dtype=torch.float32), torch.rand((4, 8), dtype=torch.float32)
        rotated, _ = self.reference.rotary_vision(q, q, cos, sin)
        expected = q * cos[:, None].double() + torch.cat((-q[..., 4:], q[..., :4]), -1) * sin[:, None].double()
        self.assertTrue(torch.equal(rotated, expected))


if __name__ == "__main__":
    unittest.main()

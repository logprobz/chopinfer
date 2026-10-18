"""The Qwen3.8 reference adapter on a tiny random qwen4_exp checkpoint, on CPU.

The checkpoint has Qwen3.8's layout at toy sizes: two gated delta net layers
(the second with PLE and a sharded FP8 n-gram table) and one QSA attention
layer with FP8 routed experts, and a two-block vision tower. The runner's
scores, through the adapter's chunked vision attention and chunked prefill,
must match a plain HF `model(...)` forward over the same weights decoded by
the definitions. `MixedExportTests` checks reference_mixed.py the same way
on tiny ModelOpt mixed exports of the nvfp4-routed-fp8-mtp and
mxfp8-dense-nvfp4 layouts. Needs torch and transformers 5.16.1; skipped
without them.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B models/qwen38/test_reference.py
"""

import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from transformers.models.qwen4_exp import configuration_qwen4_exp, modeling_qwen4_exp as hf
    from tools.refrun import modes, quant, run, stamps
    from tools.refrun.test_quant import naive
except ModuleNotFoundError:
    torch = None

ADAPTER = ROOT / "models/qwen38/reference.py"
IMAGE, VISION_START, VISION_END, EOS = 60, 58, 59, 63
TEXT = {"vocab_size": 64, "hidden_size": 32, "num_hidden_layers": 3, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 32, "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "layer_types": ["linear_attention", "linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 8, "linear_value_head_dim": 8,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16, "num_experts": 8, "num_experts_per_tok": 2,
        "hc_count": 4, "hc_lowrank": 8, "ple_layer_ids": [2], "ple_embed_dim": 32, "ple_conv_kernel_size": 4,
        "ngram_size": 3, "heads_per_ngram": 8, "ngram_vocab_size_base": 50,
        "make_ngram_vocab_size_divisible_by": 8, "split_ngram_parts": 4, "indexer_n_heads": 2,
        "indexer_kv_heads": 1, "indexer_head_dim": 16, "indexer_budget": 8, "indexer_compress_ratio": 2,
        "output_gate_type": "sigmoid", "eos_token_id": EOS, "bos_token_id": EOS, "max_position_embeddings": 512,
        "tie_word_embeddings": False, "model_type": "qwen4_exp_text",
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [2, 1, 1], "partial_rotary_factor": 0.25,
                            "rope_theta": 10000, "rope_type": "default"}}
VISION = {"depth": 2, "hidden_size": 16, "hidden_act": "gelu_pytorch_tanh", "intermediate_size": 32,
          "num_heads": 2, "in_channels": 3, "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
          "out_hidden_size": 32, "num_position_embeddings": 16, "deepstack_visual_indexes": [],
          "model_type": "qwen4_exp_vision"}
CONFIG = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
          "image_token_id": IMAGE, "video_token_id": 61, "vision_start_token_id": VISION_START,
          "vision_end_token_id": VISION_END, "tie_word_embeddings": False, "text_config": TEXT,
          "vision_config": VISION}
GRID = [1, 4, 6]
PROMPT = [3, 9, 17, 4, 22, 5, 30, 7, 11, 12, 40, 41, 2]
IMAGE_PROMPT = [3, 9, VISION_START, *[IMAGE] * 6, VISION_END, 17, 4, 22, 5]
NGRAM = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."


def setUpModule():
    if torch is None:
        return
    modes.prepare_environment(cuda=False)
    global RACY_NS
    RACY_NS, stamps.RACY_NS = stamps.RACY_NS, 100_000_000


def tearDownModule():
    if torch is not None:
        stamps.RACY_NS = RACY_NS


def codes(generator, shape):
    """Random E4M3 codes without the two NaN codes."""
    out = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
    out[(out & 0x7F) == 0x7F] = 0x38
    return out


def write_checkpoint(root, seed=0):
    """config.json, two safetensors shards with an index, and a source lock."""
    from safetensors.torch import save_file
    root.mkdir(parents=True)
    generator = torch.Generator().manual_seed(seed)
    config = configuration_qwen4_exp.Qwen4ExpConfig.from_dict(json.loads(json.dumps(CONFIG)))
    with torch.device("meta"):
        model = hf.Qwen4ExpForConditionalGeneration(config)
    text = config.text_config
    tensors = {}
    for name, parameter in model.named_parameters():
        if ".experts." in name or "ngram_embedding" in name:
            continue
        scale = 0.05 if "norm" in name else 0.3
        tensors[name] = (torch.randn(parameter.shape, generator=generator) * scale).to(torch.bfloat16)
    real = hf.Qwen4ExpTextNGramEmbedding(text, text.ple_embed_dim, 1, 0)
    for key in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
        tensors[f"model.language_model.layers.1.ple.ple_embedding.{key}"] = getattr(real, key).clone()
    inter, hidden = text.moe_intermediate_size, text.hidden_size
    for layer in range(text.num_hidden_layers):
        for expert in range(text.num_experts):
            prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
            for projection, shape in (("gate_proj", (inter, hidden)), ("up_proj", (inter, hidden)),
                                      ("down_proj", (hidden, inter))):
                tensors[prefix + projection + ".weight"] = codes(generator, shape).view(torch.float8_e4m3fn)
                tensors[prefix + projection + ".weight_scale_inv"] = (
                    torch.rand(quant.fp8_block().scale_shape(*shape), generator=generator) * 0.02 + 0.005
                ).to(torch.bfloat16)
    padded = math.ceil(real.total_vocab_size / 8) * 8
    columns = text.ple_embed_dim // 16
    for shard in range(text.split_ngram_parts):
        tensors[f"{NGRAM}shard_{shard}.weight"] = codes(
            generator, (padded // text.split_ngram_parts, columns)).view(torch.float8_e4m3fn)
    tensors[NGRAM + "weight_scale"] = torch.tensor([0.0371], dtype=torch.bfloat16)
    names = sorted(tensors)
    halves = {"model-00001-of-00002.safetensors": names[::2], "model-00002-of-00002.safetensors": names[1::2]}
    for file, part in halves.items():
        save_file({n: tensors[n].contiguous() for n in part}, str(root / file))
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {n: f for f, part in halves.items() for n in part}}))
    (root / "config.json").write_text(json.dumps(CONFIG))
    lock = {}
    for file in ("config.json", "model.safetensors.index.json", *halves):
        data = (root / file).read_bytes()
        lock[file] = {"byte_length": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (root / "lock.json").write_text(json.dumps(lock))
    return tensors


def decoded(tensors, name, rounded):
    """An FP8 block weight by the definitions; `rounded` rounds to BF16."""
    stored, scale = tensors[name].view(torch.uint8), tensors[name + "_scale_inv"]
    rows, cols = stored.shape
    value = naive(stored, lambda r, c: float(scale[r // 128, c // 128]), rows, cols, "e4m3")
    return value.float().to(torch.bfloat16).double() if rounded else value


def plain_model(tensors, dtype, rounded):
    """HF's model with every weight decoded and in place, vision attention SDPA."""
    config = configuration_qwen4_exp.Qwen4ExpConfig.from_dict(json.loads(json.dumps(CONFIG)))
    config.text_config._attn_implementation = "eager"
    config.text_config._experts_implementation = "eager"
    config.vision_config._attn_implementation = "sdpa"
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = hf.Qwen4ExpForConditionalGeneration(config).eval()
    finally:
        torch.set_default_dtype(default)
    text = config.text_config
    for module in model.modules():
        for owner in ("config",):
            if isinstance(getattr(module, owner, None), configuration_qwen4_exp.Qwen4ExpTextConfig):
                module.config._attn_implementation = module.config._experts_implementation = "eager"
            if isinstance(getattr(module, owner, None), configuration_qwen4_exp.Qwen4ExpVisionConfig):
                module.config._attn_implementation = "sdpa"
    state = {n: (t.to(dtype) if t.is_floating_point() else t) for n, t in tensors.items()
             if ".experts." not in n and "ngram_embedding" not in n}
    for layer in range(text.num_hidden_layers):
        prefix = f"model.language_model.layers.{layer}.mlp.experts."
        weight = {p: torch.stack([decoded(tensors, f"{prefix}{e}.{p}.weight", rounded)
                                  for e in range(text.num_experts)]) for p in ("gate_proj", "up_proj", "down_proj")}
        state[prefix + "gate_up_proj"] = torch.cat([weight["gate_proj"], weight["up_proj"]], dim=1).to(dtype)
        state[prefix + "down_proj"] = weight["down_proj"].to(dtype)
    shards = [tensors[f"{NGRAM}shard_{s}.weight"] for s in range(text.split_ngram_parts)]
    table = torch.cat(shards).view(torch.uint8)
    scale = float(tensors[NGRAM + "weight_scale"][0])
    rows = naive(table, lambda r, c: scale, table.shape[0], table.shape[1], "e4m3")
    state[NGRAM + "weight"] = (rows.float().to(torch.bfloat16).double() if rounded else rows).to(dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or [m for m in missing if "rotary" not in m and "inv_freq" not in m]:
        raise ValueError(f"plain model load: missing {missing}, unexpected {unexpected}")
    return model


def pixels(generator):
    return torch.randn((GRID[1] * GRID[2], 3 * 2 * 2 * 2), generator=generator)


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        root = Path(cls.directory.name)
        cls.tensors = write_checkpoint(root / "checkpoint")
        cls.pixels = pixels(torch.Generator().manual_seed(1))
        (root / "image.bin").write_bytes(cls.pixels.numpy().tobytes())
        (root / "grid.bin").write_bytes(torch.tensor([GRID], dtype=torch.int64).numpy().tobytes())
        media = [{"kind": "image", "tensor": {"path": "image.bin", "dtype": "f32", "shape": list(cls.pixels.shape)},
                  "grid": {"path": "grid.bin", "dtype": "i64", "shape": [1, 3]},
                  "processor": {"code_sha256": "0" * 64, "config_sha256": "0" * 64}}]
        sequences = [{"name": "text", "category": "prose", "token_ids": PROMPT, "score_from": 1},
                     {"name": "image", "category": "probe", "token_ids": IMAGE_PROMPT, "score_from": 9,
                      "answer": {"policy": "forced", "ids": [33, 34, 35]}, "media": media},
                     {"name": "greedy", "category": "probe", "token_ids": IMAGE_PROMPT,
                      "score_from": len(IMAGE_PROMPT),
                      "answer": {"policy": "greedy", "max_new_tokens": 4, "stop_ids": [EOS]}, "media": media}]
        (root / "workload.json").write_text(json.dumps({"schema": "chopinfer.refrun.workload.v1", "model": "qwen38",
                                                        "sequences": sequences}))
        load_adapter = run.load_adapter
        cls.adapters = []

        def small_chunks(path):
            module, relative = load_adapter(path)
            # Several vision query chunks and prefill chunks on toy sizes.
            module.VISION_CHUNK, module.PREFILL_CHUNK = 7, 5
            cls.adapters.append(module)
            return module, relative
        cls.output = root / "out"
        with mock.patch.object(run, "load_adapter", side_effect=small_chunks):
            cls.provenance = run.run(run.parse_args([
                "--adapter", str(ADAPTER), "--checkpoint", str(root / "checkpoint"),
                "--lock", str(root / "checkpoint/lock.json"), "--workload", str(root / "workload.json"),
                "--modes", "f32,f64,f32:hf-bf16,bf16", "--devices", "cpu,cpu", "--output", str(cls.output),
                "--stamps", str(root / "stamps"), "--no-cache"]))
            # The same f32 pass with every tap written, over the same chunks.
            cls.tapped = root / "tapped"
            run.run(run.parse_args([
                "--adapter", str(ADAPTER), "--checkpoint", str(root / "checkpoint"),
                "--lock", str(root / "checkpoint/lock.json"), "--workload", str(root / "workload.json"),
                "--modes", "f32", "--devices", "cpu,cpu", "--output", str(cls.tapped),
                "--stamps", str(root / "stamps"), "--taps", "--tap-sites", "all", "--no-cache"]))
        cls.scores = {label: {(e["name"], e["key"]["sequence"]["answer"]["policy"]): e["scores"]
                              for e in json.loads((cls.output / f"{label}.json").read_text())["entries"]}
                      for label in ("f32", "f64", "f32-hf-bf16", "bf16")}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def plain_logprobs(self, model, ids, score_from, dtype, image=False):
        kwargs = {}
        if image:
            kwargs = {"pixel_values": self.pixels, "image_grid_thw": torch.tensor([GRID]),
                      "mm_token_type_ids": (torch.tensor([ids]) == IMAGE).int()}
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]), use_cache=False, **kwargs).logits[0].to(dtype)
        return torch.log_softmax(logits[score_from - 1:-1], dim=-1)

    def check(self, label, dtype, rounded, tolerance):
        model = plain_model(self.tensors, dtype, rounded)
        for name, ids, score_from, image in (("text", PROMPT, 1, False),
                                             ("image", IMAGE_PROMPT + [33, 34, 35], 9, True)):
            expected = self.plain_logprobs(model, ids, score_from, dtype, image)
            scores = self.scores[label][(name, "forced")]
            self.assertEqual(scores["token"], ids[score_from:])
            got = torch.tensor(scores["logprob"], dtype=torch.float64)
            chosen = expected.gather(1, torch.tensor(ids[score_from:])[:, None])[:, 0].double()
            self.assertLess((got - chosen).abs().max().item(), tolerance, (label, name))
            self.assertEqual(scores["top1"], expected.argmax(dim=-1).tolist(), (label, name))
        return model

    def test_f64_and_f32_match_a_plain_forward(self):
        # HF computes norms, the gated delta rule and softmaxes in F32 even in
        # f64, so chunked prefill differs from one pass at F32 rounding.
        self.check("f64", torch.float64, False, 1e-6)
        self.check("f32", torch.float32, False, 2e-4)

    def test_hf_bf16_rule_rounds_experts_and_ngram_rows(self):
        self.check("f32-hf-bf16", torch.float32, True, 2e-4)
        exact, rounded = (self.scores[label][("text", "forced")]["logprob"] for label in ("f32", "f32-hf-bf16"))
        self.assertGreater(max(abs(a - b) for a, b in zip(exact, rounded)), 1e-5)
        bf16 = self.scores["bf16"][("text", "forced")]["logprob"]
        self.assertLess(max(abs(a - b) for a, b in zip(rounded, bf16)), 0.5)

    def test_greedy_answer_is_the_plain_models_argmax(self):
        model = plain_model(self.tensors, torch.float32, False)
        ids = list(IMAGE_PROMPT)
        for _ in range(4):
            logits = self.plain_logprobs(model, ids + [0], len(ids), torch.float32, image=True)
            ids.append(int(logits[-1].argmax()))
            if ids[-1] == EOS:
                break
        answer = ids[len(IMAGE_PROMPT):]
        self.assertEqual(self.scores["f32"][("greedy", "greedy")]["answer"], answer)
        self.assertEqual(self.scores["f64"][("greedy", "forced")]["token"], answer)

    def test_taps_over_chunked_prefill(self):
        """Prefill chunks of 5 tokens: every text tap holds every position at
        its absolute row, the image's encoder rows are written once, each
        step completes, and the scores equal the untapped run's."""
        from tools.qual import tapdiff
        tapped = json.loads((self.tapped / "f32.json").read_text())["entries"]
        untapped = json.loads((self.output / "f32.json").read_text())["entries"]
        self.assertEqual([e["scores"] for e in tapped], [e["scores"] for e in untapped])
        steps = sorted((self.tapped / "taps/f32").iterdir())
        self.assertEqual(len(steps), 3)
        embed = self.tensors["model.language_model.embed_tokens.weight"].float()
        for step, ids in zip(steps, (PROMPT, IMAGE_PROMPT + [33, 34, 35], None)):
            self.assertEqual(json.loads((step / "completion.json").read_text())["completion"], "full", step.name)
            index = {e["name"]: e for e in map(json.loads, (step / "index.jsonl").read_text().splitlines())}
            items = json.loads((step / "items.json").read_text())
            length = items["items"][0]["seq_len"]
            self.assertEqual(items["positions"], list(range(length)))
            for name, entry in index.items():
                self.assertEqual(entry["rows"], (6 if name == "enc.vision.out" else 24) if name.startswith("enc.")
                                 else length, (step.name, name))
            if ids is None:
                continue
            rows = torch.from_numpy(tapdiff.read(step / "embed.out.r0.bin", index["embed.out"]).copy())
            image = [p for p, token in enumerate(ids) if token == IMAGE]
            text = [p for p, token in enumerate(ids) if token != IMAGE]
            self.assertTrue(torch.equal(rows[text], embed[[ids[p] for p in text]].repeat(1, 4)), step.name)
            if image:
                self.assertEqual(sorted(n for n in index if n.startswith("enc.")),
                                 ["enc.vision.B00.out", "enc.vision.B01.out", "enc.vision.out"])
                merged = torch.from_numpy(tapdiff.read(step / "enc.vision.out.r0.bin", index["enc.vision.out"]).copy())
                self.assertTrue(torch.equal(rows[image], merged.repeat(1, 4)))
                self.assertEqual(items["encoder"]["enc.vision.out"][0]["rows"], 6)
            else:
                self.assertFalse([n for n in index if n.startswith("enc.")])

    def test_layout_names_and_taps(self):
        layout = self.provenance["layout"]
        self.assertEqual(layout["dequant"], "hf-bf16")
        self.assertIn("HostRows", self.provenance["replacements"]["ple.ple_embedding.ngram_embedding"])
        units = dict(self.provenance["placement"]["units"])
        self.assertNotIn(NGRAM.rstrip("."), units)
        adapter = self.adapters[0]
        config = configuration_qwen4_exp.Qwen4ExpConfig.from_dict(json.loads(json.dumps(CONFIG)))
        with torch.device("meta"):
            model = hf.Qwen4ExpForConditionalGeneration(config)
        taps = adapter.taps(model)
        name = re.compile(r"(embed|L\d{2}|final|enc\.vision(\.B\d{2})?)\.(in\.streams|attn\.core|attn\.out"
                          r"|router\.(logits|ids|weights)|moe\.(routed|shared)|mlp\.out|out|x(\.[a-z0-9_]+)+)")
        for tap, spec in taps.items():
            self.assertRegex(tap, name)
            model.get_submodule(spec if isinstance(spec, str) else spec.module)
        self.assertEqual(taps["L01.x.ple.out"].module, "model.language_model.layers.1.attn_hyper_connection")
        self.assertEqual((taps["enc.vision.out"].module, taps["enc.vision.out"].rows),
                         ("model.visual.merger", ("image", "video")))
        self.assertEqual(taps["enc.vision.B01.out"].module, "model.visual.blocks.1")
        self.assertEqual(taps["L00.out"], "model.language_model.layers.0")
        self.assertIn("L02.attn.core", taps)
        self.assertEqual(taps["L00.attn.core"].module, "model.language_model.layers.0.linear_attn.out_proj")
        core = torch.zeros(5 * 4, 8)
        self.assertEqual(tuple(taps["L00.x.gdn.core"].transform(core).shape), (5, 32))


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class ProbePixelTests(unittest.TestCase):
    """The image probe workload's rows rebuilt from tracked inputs
    (reference_workloads.py --probe-workload) equal the pinned processor's."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "models/qwen38"))
        self.addCleanup(sys.path.remove, str(ROOT / "models/qwen38"))

    def test_smart_resize_matches_the_pinned_cases(self):
        import reference_workloads
        recipe = json.loads((ROOT / "fixtures/qwen38/image-processor.json").read_text())
        for height, width, rows, columns in recipe["smart_resize"]:
            with self.subTest(height=height, width=width):
                self.assertEqual(reference_workloads.smart_resize(
                    height, width, recipe["recipe"]["factor"], recipe["recipe"]["minimum_pixels"],
                    recipe["recipe"]["maximum_pixels"]), (rows, columns))

    def test_tracked_images_give_the_pinned_rows(self):
        import reference_workloads  # puts fixtures/qwen38/capture on the path
        import image_rows
        import images
        pinned = json.loads((ROOT / "fixtures/qwen38/image-processor.json").read_text())
        identity = json.loads((ROOT / "fixtures/qwen38/frontend/identity.json").read_text())["images"]
        pngs = images.functional() | {"small-100x60": images.png(*images.SIZES["small-100x60"])}
        self.assertEqual({name: hashlib.sha256(data).hexdigest() for name, data in pngs.items()},
                         {name: identity[name] for name in pngs})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "small.png")
            path.write_bytes(pngs["small-100x60"])
            rows, grid = reference_workloads.processed_rows(image_rows.read_png(path), pinned["recipe"])
        self.assertEqual((hashlib.sha256(rows.tobytes()).hexdigest(), grid),
                         (pinned["images"]["small-100x60"]["pixels_sha256"], pinned["images"]["small-100x60"]["grid"]))


MIXED = ROOT / "models/qwen38/reference_mixed.py"
# K of every MXFP8 projection a multiple of 32, NVFP4 n-gram rows of 32 values.
MIXED_TEXT = {**TEXT, "shared_expert_intermediate_size": 32, "ple_embed_dim": 512}
MXFP8_ROLES = re.compile(r"\.layers\.\d+\.(self_attn\.([qkvo]_proj|indexer\.index_qk_proj)"
                         r"|linear_attn\.(in_proj_(qkv|z|a|b)|out_proj)|mlp\.shared_expert\.(gate|up|down)_proj)$")
F32_NORMS = re.compile(r"(hc_norm|ple\.norm_(key|conv|query))\.weight$")
EXPERTS = "model.language_model.layers.{}.mlp.experts"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def mixed_config(shape):
    """A modelopt MIXED_PRECISION config of the nvfp4-routed-fp8-mtp or
    mxfp8-dense-nvfp4 layout."""
    layers = {EXPERTS.format(layer): {"quant_algo": "NVFP4", "group_size": 16} for layer in range(3)}
    layers["mtp.layers.0.mlp.experts"] = {"quant_algo": "FP8_PB_WO", "group_size": 128}
    if shape == "nvfp4-routed-fp8-mtp":
        layers[NGRAM.rstrip(".")] = {"quant_algo": "FP8"}
    else:
        layers[NGRAM.rstrip(".")] = {"quant_algo": "NVFP4", "group_size": 16}
        config = configuration_qwen4_exp.Qwen4ExpConfig.from_dict({**CONFIG, "text_config": MIXED_TEXT})
        with torch.device("meta"):
            model = hf.Qwen4ExpForConditionalGeneration(config)
        layers |= {name: {"quant_algo": "MXFP8", "group_size": 32} for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and MXFP8_ROLES.search(name)}
    return {**CONFIG, "text_config": MIXED_TEXT,
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                    "kv_cache_quant_algo": None, "quantized_layers": layers}}


def write_mixed_checkpoint(root, shape, seed=0):
    """A tiny export of `shape`: NVFP4 experts with input scales and an FP8
    n-gram table (nvfp4-routed-fp8-mtp), plus MXFP8 projections, an NVFP4
    n-gram table and F32 norms (mxfp8-dense-nvfp4)."""
    from safetensors.torch import save_file
    root.mkdir(parents=True)
    generator = torch.Generator().manual_seed(seed)
    value = mixed_config(shape)
    declared = value["quantization_config"]["quantized_layers"]
    config = configuration_qwen4_exp.Qwen4ExpConfig.from_dict(json.loads(json.dumps(value)))
    with torch.device("meta"):
        model = hf.Qwen4ExpForConditionalGeneration(config)
    text = config.text_config

    def global_scale():
        return torch.tensor(f32(float(torch.rand((), generator=generator)) * 0.02 + 0.003), dtype=torch.float32)

    def nvfp4(prefix, rows, cols, input_scale=True):
        tensors[prefix + "weight"] = torch.randint(0, 256, (rows, cols // 2), generator=generator, dtype=torch.uint8)
        tensors[prefix + "weight_scale"] = torch.randint(
            0x28, 0x48, (rows, cols // 16), generator=generator, dtype=torch.uint8).view(torch.float8_e4m3fn)
        tensors[prefix + "weight_scale_2"] = global_scale()
        if input_scale:
            tensors[prefix + "input_scale"] = global_scale()
    tensors = {}
    for name, parameter in model.named_parameters():
        if ".experts." in name or "ngram_embedding" in name:
            continue
        module = name.removesuffix(".weight")
        if module in declared:
            rows, cols = parameter.shape
            tensors[module + ".weight"] = codes(generator, (rows, cols)).view(torch.float8_e4m3fn)
            tensors[module + ".weight_scale"] = torch.randint(118, 126, (rows, cols // 32), generator=generator,
                                                              dtype=torch.uint8)
            continue
        scale = 0.05 if "norm" in name else 0.3
        weight = torch.randn(parameter.shape, generator=generator) * scale
        keep = shape == "mxfp8-dense-nvfp4" and F32_NORMS.search(name)
        tensors[name] = weight if keep else weight.to(torch.bfloat16)
    real = hf.Qwen4ExpTextNGramEmbedding(text, text.ple_embed_dim, 1, 0)
    for key in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
        tensors[f"model.language_model.layers.1.ple.ple_embedding.{key}"] = getattr(real, key).clone()
    inter, hidden = text.moe_intermediate_size, text.hidden_size
    for layer in range(text.num_hidden_layers):
        for expert in range(text.num_experts):
            prefix = f"{EXPERTS.format(layer)}.{expert}."
            for projection, (rows, cols) in (("gate_proj", (inter, hidden)), ("up_proj", (inter, hidden)),
                                             ("down_proj", (hidden, inter))):
                nvfp4(prefix + projection + ".", rows, cols)
    padded = math.ceil(real.total_vocab_size / 8) * 8
    columns = real.ngram_embedding.embedding_dim
    rows = padded // text.split_ngram_parts
    for shard in range(text.split_ngram_parts):
        if shape == "nvfp4-routed-fp8-mtp":
            tensors[f"{NGRAM}shard_{shard}.weight"] = codes(generator, (rows, columns)).view(torch.float8_e4m3fn)
        else:
            nvfp4(f"{NGRAM}shard_{shard}.", rows, columns, input_scale=False)
            del tensors[f"{NGRAM}shard_{shard}.weight_scale_2"]
    if shape == "nvfp4-routed-fp8-mtp":
        tensors[NGRAM + "weight_scale"] = torch.tensor([0.0371], dtype=torch.bfloat16)
    else:
        tensors[NGRAM + "weight_scale_2"] = global_scale().reshape(1)
    names = sorted(tensors)
    halves = {"model-00001-of-00002.safetensors": names[::2], "model-00002-of-00002.safetensors": names[1::2]}
    for file, part in halves.items():
        save_file({n: tensors[n].contiguous() for n in part}, str(root / file))
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {n: f for f, part in halves.items() for n in part}}))
    (root / "config.json").write_text(json.dumps(value))
    lock = {}
    for file in ("config.json", "model.safetensors.index.json", *halves):
        data = (root / file).read_bytes()
        lock[file] = {"byte_length": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (root / "lock.json").write_text(json.dumps(lock))
    return tensors


def decode_nvfp4(tensors, prefix, global_scale, rule):
    """An NVFP4 weight by the definitions: exact, the exact product rounded
    once to F32 (rounded), or the F32 product of the E4M3 scale times the
    global, then the element, rounded to BF16 (hf-bf16)."""
    stored, scale = tensors[prefix + "weight"], tensors[prefix + "weight_scale"].view(torch.uint8)
    rows, cols = stored.shape[0], stored.shape[1] * 2
    if rule == "hf-bf16":
        value = naive(stored, lambda r, c: f32(quant.E4M3[int(scale[r, c // 16])] * global_scale), rows, cols, "e2m1")
        return value.float().to(torch.bfloat16).double()
    value = naive(stored, lambda r, c: quant.E4M3[int(scale[r, c // 16])] * global_scale, rows, cols, "e2m1")
    return value.float().double() if rule == "rounded" else value


def decode_mxfp8(tensors, module, rule):
    """An MXFP8 weight by the definitions (exact in F32 and BF16 here)."""
    stored, scale = tensors[module + ".weight"].view(torch.uint8), tensors[module + ".weight_scale"]
    value = naive(stored, lambda r, c: quant.E8M0[int(scale[r, c // 32])], *stored.shape, "e4m3")
    return value.float().to(torch.bfloat16).double() if rule == "hf-bf16" else value


def plain_mixed_model(tensors, shape, dtype, rule, keep_f32=False):
    """HF's model with every weight decoded by the definitions and in place;
    with `keep_f32`, stored F32 parameters stay F32 in a narrower model."""
    value = mixed_config(shape)
    declared = value["quantization_config"]["quantized_layers"]
    config = configuration_qwen4_exp.Qwen4ExpConfig.from_dict(json.loads(json.dumps(value)))
    config.vision_config._attn_implementation = "sdpa"
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = hf.Qwen4ExpForConditionalGeneration(config).eval()
    finally:
        torch.set_default_dtype(default)
    for module in model.modules():
        if isinstance(getattr(module, "config", None), configuration_qwen4_exp.Qwen4ExpTextConfig):
            module.config._attn_implementation = module.config._experts_implementation = "eager"
        if isinstance(getattr(module, "config", None), configuration_qwen4_exp.Qwen4ExpVisionConfig):
            module.config._attn_implementation = "sdpa"
    text = config.text_config
    skip = ("weight_scale", "weight_scale_2", "input_scale")
    def cast(t):
        return t if keep_f32 and t.dtype == torch.float32 and dtype == torch.bfloat16 else t.to(dtype)
    state = {n: (cast(t) if t.is_floating_point() else t) for n, t in tensors.items()
             if ".experts." not in n and "ngram_embedding" not in n and not n.endswith(skip)
             and n.removesuffix(".weight") not in declared}
    for module in declared:
        if MXFP8_ROLES.search(module):
            state[module + ".weight"] = decode_mxfp8(tensors, module, rule).to(dtype)
    for layer in range(text.num_hidden_layers):
        prefix = EXPERTS.format(layer) + "."

        def weight(expert, projection):
            name = f"{prefix}{expert}.{projection}."
            return decode_nvfp4(tensors, name, float(tensors[name + "weight_scale_2"]), rule)
        stacked = {p: torch.stack([weight(e, p) for e in range(text.num_experts)])
                   for p in ("gate_proj", "up_proj", "down_proj")}
        state[prefix + "gate_up_proj"] = torch.cat([stacked["gate_proj"], stacked["up_proj"]], dim=1).to(dtype)
        state[prefix + "down_proj"] = stacked["down_proj"].to(dtype)
    if shape == "nvfp4-routed-fp8-mtp":
        table = torch.cat([tensors[f"{NGRAM}shard_{s}.weight"] for s in range(text.split_ngram_parts)]).view(torch.uint8)
        scale = float(tensors[NGRAM + "weight_scale"][0])
        rows = naive(table, lambda r, c: scale, table.shape[0], table.shape[1], "e4m3")
        rows = rows.float().to(torch.bfloat16).double() if rule == "hf-bf16" else rows
    else:
        global_scale = float(tensors[NGRAM + "weight_scale_2"][0])
        rows = torch.cat([decode_nvfp4(tensors, f"{NGRAM}shard_{s}.", global_scale, rule)
                          for s in range(text.split_ngram_parts)])
    state[NGRAM + "weight"] = rows.to(dtype)
    # Assigned, so kept F32 parameters are not copied into BF16 ones.
    missing, unexpected = model.load_state_dict(state, strict=False, assign=keep_f32)
    if unexpected or [m for m in missing if "rotary" not in m and "inv_freq" not in m]:
        raise ValueError(f"plain model load: missing {missing}, unexpected {unexpected}")
    return model


def built(root, mode, rule=None, change=None):
    """reference_mixed.build for one mode on CPU, without the runner; `change`
    edits the parsed checkpoint and the loaded store (negative controls)."""
    from tools.refrun import adapter, place
    module, _ = run.load_adapter(MIXED)
    checkpoint = place.Checkpoint(root)
    units = module.units(checkpoint)
    placement = place.plan([(u.module, checkpoint.stored_bytes(u.tensors)) for u in units if not u.host], ["cpu"])
    policy = modes.policy(mode, rule)
    store = place.Store(checkpoint, placement, units)
    if change is not None:
        change(checkpoint, store)
    return module.build(adapter.BuildConfig(checkpoint, policy, modes.torch_dtype(policy["dtype"]), store),
                        placement)


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class MixedExportTests(unittest.TestCase):
    """reference_mixed.py on tiny exports of the nvfp4-routed-fp8-mtp and
    mxfp8-dense-nvfp4 layouts:
    every mode's scores equal a plain HF forward over weights decoded by the
    definitions, and malformed exports are refused."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        root = Path(cls.directory.name)
        sequences = [{"name": "text", "category": "prose", "token_ids": PROMPT, "score_from": 1}]
        (root / "workload.json").write_text(json.dumps({"schema": "chopinfer.refrun.workload.v1", "model": "qwen38",
                                                        "sequences": sequences}))
        cls.tensors, cls.scores, cls.provenance = {}, {}, {}
        for shape in ("nvfp4-routed-fp8-mtp", "mxfp8-dense-nvfp4"):
            cls.tensors[shape] = write_mixed_checkpoint(root / shape, shape)
            output = root / f"out-{shape}"
            cls.provenance[shape] = run.run(run.parse_args([
                "--adapter", str(MIXED), "--checkpoint", str(root / shape), "--lock", str(root / shape / "lock.json"),
                "--workload", str(root / "workload.json"), "--modes", "f64,f32,f32:hf-bf16,bf16",
                "--dequant", "f32=rounded", "--devices", "cpu,cpu", "--output", str(output),
                "--stamps", str(root / "stamps"), "--no-cache"]))
            cls.scores[shape] = {label: json.loads((output / f"{label}.json").read_text())["entries"][0]["scores"]
                                 for label in ("f64", "f32", "f32-hf-bf16", "bf16")}
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def check(self, shape, label, dtype, rule, tolerance):
        model = plain_mixed_model(self.tensors[shape], shape, dtype, rule)
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([PROMPT]), use_cache=False).logits[0].to(dtype)
        expected = torch.log_softmax(logits[:-1], dim=-1)
        scores = self.scores[shape][label]
        chosen = expected.gather(1, torch.tensor(PROMPT[1:])[:, None])[:, 0].double()
        got = torch.tensor(scores["logprob"], dtype=torch.float64)
        self.assertLess((got - chosen).abs().max().item(), tolerance, (shape, label))
        self.assertEqual(scores["top1"], expected.argmax(dim=-1).tolist(), (shape, label))

    def test_f64_exact_and_f32_rounded_match_a_plain_forward(self):
        for shape in ("nvfp4-routed-fp8-mtp", "mxfp8-dense-nvfp4"):
            self.check(shape, "f64", torch.float64, "exact", 1e-6)
            self.check(shape, "f32", torch.float32, "rounded", 2e-4)

    def test_hf_bf16_rounds_the_f32_product(self):
        for shape in ("nvfp4-routed-fp8-mtp", "mxfp8-dense-nvfp4"):
            self.check(shape, "f32-hf-bf16", torch.float32, "hf-bf16", 2e-4)
            exact, rounded = (self.scores[shape][label]["logprob"] for label in ("f32", "f32-hf-bf16"))
            self.assertNotEqual(exact, rounded, shape)

    def test_f32_needs_the_rounded_rule_for_nvfp4(self):
        reference = built(self.root / "nvfp4-routed-fp8-mtp", "f32", "exact")
        with self.assertRaisesRegex(ValueError, "not exact in torch.float32"):
            quant.configure(reference.model, torch.float32, "exact")

    def test_f32_norms_never_narrow(self):
        tensors = self.tensors["mxfp8-dense-nvfp4"]
        norms = sorted(name for name in tensors if F32_NORMS.search(name))
        self.assertEqual(len(norms), 3 * 2 + 1 + 3)
        self.assertTrue(all(tensors[name].dtype == torch.float32 for name in norms))
        for mode, dtype in (("bf16", torch.float32), ("f64", torch.float64)):
            model = built(self.root / "mxfp8-dense-nvfp4", mode).model
            parameters = dict(model.named_parameters())
            for name in norms:
                self.assertEqual(parameters[name].dtype, dtype, (mode, name))
                self.assertTrue(torch.equal(parameters[name].double(), tensors[name].double()), (mode, name))
            bf16 = parameters["model.language_model.layers.0.linear_attn.norm.weight"].dtype
            self.assertEqual(bf16, torch.bfloat16 if mode == "bf16" else torch.float64)
        self.assertEqual(self.provenance["mxfp8-dense-nvfp4"]["layout"]["f32_parameters"]["count"], len(norms))
        self.assertEqual(self.provenance["nvfp4-routed-fp8-mtp"]["layout"]["f32_parameters"]["count"], 0)

    def test_layout_records_the_declaration(self):
        layout = self.provenance["mxfp8-dense-nvfp4"]["layout"]
        self.assertEqual(layout["dequant"], "hf-bf16")
        self.assertEqual(layout["declared"]["NVFP4"], {"mlp.experts": 3, "ple.ple_embedding.ngram_embedding": 1})
        self.assertEqual(layout["declared"]["MXFP8"]["mlp.shared_expert.down_proj"], 3)
        self.assertEqual(layout["declared"]["MXFP8"]["self_attn.indexer.index_qk_proj"], 1)
        self.assertEqual(layout["not_run"], {"declared": {"mtp.layers.0.mlp.experts": "FP8_PB_WO"},
                                             "mtp_tensors": 0})
        self.assertEqual(self.provenance["nvfp4-routed-fp8-mtp"]["layout"]["ngram_embedding"],
                         {NGRAM.rstrip("."): "FP8"})
        module, _ = run.load_adapter(MIXED)
        code = module.reference_code(None)
        self.assertEqual(code["models/qwen38/reference.py"], ADAPTER)
        self.assertEqual(hashlib.sha256(ADAPTER.read_bytes()).hexdigest(), module.REFERENCE_SHA256)

    def test_nvfp4_ngram_rows_by_hand(self):
        """Row 1 of shard 0: byte 0x71 is 0.5 (low nibble 1) then 6.0 (high
        nibble 7), byte 0xA3 is 1.5 then -1.0; scale E4M3 0x38 is 1.0, 0x40
        is 2.0; times the global 0.5."""
        from tools.refrun import place
        from safetensors.torch import save_file
        module, _ = run.load_adapter(MIXED)
        directory = self.root / "rows"
        directory.mkdir()
        codes = torch.zeros((2, 16), dtype=torch.uint8)
        codes[1, :2] = torch.tensor([0x71, 0xA3], dtype=torch.uint8)
        codes[1, 8] = 0x71
        scales = torch.tensor([[0x38, 0x38], [0x38, 0x40]], dtype=torch.uint8).view(torch.float8_e4m3fn)
        for shard in (0, 1):
            save_file({f"t.shard_{shard}.weight": codes, f"t.shard_{shard}.weight_scale": scales},
                      str(directory / f"shard-{shard}.safetensors"))
        (directory / "config.json").write_text("{}")
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            **{f"t.shard_{s}.weight": f"shard-{s}.safetensors" for s in (0, 1)},
            **{f"t.shard_{s}.weight_scale": f"shard-{s}.safetensors" for s in (0, 1)}}}))
        checkpoint = place.Checkpoint(directory)
        for dtype, rule in ((torch.float64, "exact"), (torch.float32, "rounded"), (torch.bfloat16, "hf-bf16")):
            rows = module.NVFP4Rows(checkpoint, ["t.shard_0", "t.shard_1"], 0.5, 4, 32, dtype, rule, "cpu")
            value = rows(torch.tensor([[1, 3]]))
            self.assertEqual(value.dtype, dtype)
            self.assertEqual(value[0, 0, :4].tolist(), [0.25, 3.0, 0.75, -0.5])
            self.assertEqual(value[0, 0, 16:18].tolist(), [0.5, 6.0])
            self.assertEqual(value[0, 1].tolist(), value[0, 0].tolist())
            self.assertEqual(value[0, 0, 4:16].abs().sum().item(), 0.0)
        # A gathered row whose block scale is negative (0xB8, -1.0) or NaN is refused.
        for code in (0xB8, 0x7F):
            bad = self.root / f"rows-{code:x}"
            bad.mkdir()
            invalid = torch.tensor([[0x38, 0x38], [0x38, code]], dtype=torch.uint8).view(torch.float8_e4m3fn)
            save_file({"t.shard_0.weight": codes, "t.shard_0.weight_scale": invalid}, str(bad / "shard.safetensors"))
            (bad / "config.json").write_text("{}")
            (bad / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                "t.shard_0.weight": "shard.safetensors", "t.shard_0.weight_scale": "shard.safetensors"}}))
            rows = module.NVFP4Rows(place.Checkpoint(bad), ["t.shard_0"], 0.5, 2, 32, torch.float64, "exact", "cpu")
            self.assertEqual(rows(torch.tensor([[0]])).abs().sum().item(), 0.0)
            with self.assertRaisesRegex(ValueError, "E4M3 block scale is negative or NaN"):
                rows(torch.tensor([[1]]))

    def test_malformed_exports_are_refused(self):
        module, _ = run.load_adapter(MIXED)
        declared = mixed_config("mxfp8-dense-nvfp4")
        with self.assertRaisesRegex(ValueError, "not a ModelOpt MIXED_PRECISION export"):
            module.declaration({**declared, "quantization_config": {"quant_method": "fp8"}})
        recipe = declared["quantization_config"]
        for change, message in (({"quant_algo": "NVFP4_AWQ", "group_size": 16}, "unsupported NVFP4_AWQ"),
                                ({"quant_algo": "MXFP8", "group_size": 16}, "unsupported MXFP8 with group size 16")):
            layers = {**recipe["quantized_layers"], EXPERTS.format(0): change}
            with self.assertRaisesRegex(ValueError, message):
                module.declaration({**declared, "quantization_config": {**recipe, "quantized_layers": layers}})
        with self.assertRaisesRegex(ValueError, "KV cache quantization FP8"):
            module.declaration({**declared, "quantization_config": {**recipe, "kv_cache_quant_algo": "FP8"}})
        projection = "model.language_model.layers.2.self_attn.q_proj"

        def undeclare(checkpoint, _store):
            checkpoint.config = json.loads(json.dumps(checkpoint.config))
            del checkpoint.config["quantization_config"]["quantized_layers"][projection]

        def misshape(checkpoint, _store):
            entry = checkpoint.entries[projection + ".weight_scale"]
            checkpoint.entries[projection + ".weight_scale"] = dataclasses.replace(entry, shape=(entry.shape[0], 3))

        def weight_only(checkpoint, _store):
            checkpoint.config = json.loads(json.dumps(checkpoint.config))
            checkpoint.config["quantization_config"]["quantized_layers"][EXPERTS.format(1)]["quant_algo"] = \
                "W4A16_NVFP4"

        def experts_undeclared(checkpoint, _store):
            checkpoint.config = json.loads(json.dumps(checkpoint.config))
            del checkpoint.config["quantization_config"]["quantized_layers"][EXPERTS.format(2)]

        undeclared = re.escape(projection) + r"\.weight(_scale)?: stored (F8_E4M3|U8), but the export declares no"
        for change, message in ((undeclare, undeclared),
                                (misshape, re.escape(f"{projection}.weight_scale: MXFP8 stores U8")),
                                (weight_only, re.escape("input_scale: W4A16_NVFP4 is weight-only")),
                                (experts_undeclared, re.escape(f"{EXPERTS.format(2)}: experts the export does not"))):
            with self.assertRaisesRegex(ValueError, message):
                built(self.root / "mxfp8-dense-nvfp4", "f64", change=change)

    def test_coverage_is_the_consumed_set(self):
        """A stored tensor counts as covered only when a module consumes it
        by name; MTP sources are the only stored tensors left unexecuted."""
        from tools.refrun import place
        expert = EXPERTS.format(0) + ".0.gate_proj."

        def extra(name, host=False):
            def change(checkpoint, store):
                checkpoint.entries[name] = place.Entry("model-00001-of-00002.safetensors", "F32", (), 0, 4)
                if not host:
                    store.tensors[name] = torch.ones(())
            return change
        for change, message in ((extra(expert + "undeclared_scale"), "with no place in the HF model"),
                                (extra(NGRAM + "shard_0.extra", host=True),
                                 f"{NGRAM}shard_0.extra: a stored n-gram table tensor the table does not read"),
                                (extra("foo.weight", host=True), "foo.weight: a stored tensor outside every")):
            with self.assertRaisesRegex(ValueError, re.escape(message)):
                built(self.root / "mxfp8-dense-nvfp4", "f64", change=change)
        reference = built(self.root / "mxfp8-dense-nvfp4", "f64", change=extra("mtp.layers.0.mlp.gate.weight", host=True))
        self.assertEqual(reference.layout["not_run"]["mtp_tensors"], 1)

    def test_ngram_shards_are_exactly_shard_0_to_n_minus_1(self):
        def rename(checkpoint, _store):
            for name in [n for n in checkpoint.entries if n.startswith(NGRAM + "shard_3.")]:
                checkpoint.entries[name.replace("shard_3.", "shard_99.")] = checkpoint.entries.pop(name)
        for shape in ("nvfp4-routed-fp8-mtp", "mxfp8-dense-nvfp4"):
            with self.assertRaisesRegex(ValueError, re.escape("missing ['shard_3'], unexpected ['shard_99']")):
                built(self.root / shape, "f64", change=rename)

    def test_invalid_nvfp4_scales_are_refused(self):
        expert = EXPERTS.format(1) + ".2.down_proj."

        def global_scale(value):
            def change(_checkpoint, store):
                store.tensors[expert + "weight_scale_2"] = torch.tensor(value, dtype=torch.float32)
            return change

        def local_scale(code):
            def change(_checkpoint, store):
                scale = store.tensors[expert + "weight_scale"].clone()
                scale.view(torch.uint8)[3, 0] = code
                store.tensors[expert + "weight_scale"] = scale
            return change
        for value in (-1.0, 0.0, math.inf, math.nan):
            with self.assertRaisesRegex(ValueError, "global scale .* is not positive and finite"):
                built(self.root / "mxfp8-dense-nvfp4", "f64", change=global_scale(value))
        for code in (0xB8, 0x7F, 0xFF):
            with self.assertRaisesRegex(ValueError, re.escape(f"{expert}weight_scale: an E4M3 block scale")):
                built(self.root / "nvfp4-routed-fp8-mtp", "f64", change=local_scale(code))

        def table_global(_checkpoint, store):
            store.tensors[NGRAM + "weight_scale_2"] = torch.tensor([-0.5])
        with self.assertRaisesRegex(ValueError, "weight_scale_2: NVFP4 global scale -0.5"):
            built(self.root / "mxfp8-dense-nvfp4", "f64", change=table_global)

    def test_bf16_scores_match_a_plain_bf16_forward_with_f32_norms(self):
        """The bf16 run against HF in BF16 over hf-bf16 weights, keeping the
        stored F32 norm weights F32; log-softmax in F32 as the runner does."""
        for shape in ("nvfp4-routed-fp8-mtp", "mxfp8-dense-nvfp4"):
            model = plain_mixed_model(self.tensors[shape], shape, torch.bfloat16, "hf-bf16", keep_f32=True)
            norms = [n for n, t in self.tensors[shape].items() if t.dtype == torch.float32 and F32_NORMS.search(n)]
            parameters = dict(model.named_parameters())
            self.assertTrue(all(parameters[n].dtype == torch.float32 for n in norms))
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([PROMPT]), use_cache=False).logits[0].float()
            expected = torch.log_softmax(logits[:-1], dim=-1)
            scores = self.scores[shape]["bf16"]
            chosen = expected.gather(1, torch.tensor(PROMPT[1:])[:, None])[:, 0].double()
            got = torch.tensor(scores["logprob"], dtype=torch.float64)
            self.assertLess((got - chosen).abs().max().item(), 0.05, shape)
            self.assertEqual(scores["top1"], expected.argmax(dim=-1).tolist(), shape)

    def test_changing_inherited_reference_code_stales_a_bound_mixed_reference(self):
        """A fixture emitted from the mixed runs, verified in an isolated
        tree: editing the reference.py it inherits names reference_code."""
        import shutil
        from tools.refrun import cache, emit, keys
        output = self.root / "emitted"
        output.mkdir()
        skeleton = {"schema": "demo.teacher-reference.v1",
                    "texts": [{"name": "text", "category": "prose", "prompt_token_ids": PROMPT, "score_from": 1}]}
        report = emit.emit(skeleton, [self.root / "out-mxfp8-dense-nvfp4"], self.root / "workload.json",
                           output / "reference.json", str(self.root / "mxfp8-dense-nvfp4/lock.json"), None,
                           ["f32=exact", "bf16", "f64"])
        value = json.loads(Path(report["fixture"]).read_text())
        tree = self.root / "tree"
        for path in (*keys.NUMERICS, "models/qwen38/reference_mixed.py", "models/qwen38/reference.py"):
            (tree / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, tree / path)
        workload = Path(report["workload"])
        self.assertEqual(cache.verify(value, tree, None, None, workload)["status"], "bound")
        inherited = tree / "models/qwen38/reference.py"
        inherited.write_text(inherited.read_text().replace("PREFILL_CHUNK = 2048", "PREFILL_CHUNK = 1024"))
        with self.assertRaises(cache.StaleReference) as raised:
            cache.verify(value, tree, None, None, workload)
        self.assertEqual(raised.exception.field, "reference_code")


def variant_tool():
    """tools/qual/variant.py, imported."""
    sys.path.insert(0, str(ROOT / "tools/qual"))
    import variant
    return variant


class VariantToolTests(unittest.TestCase):
    """Another Qwen3.8 checkpoint (such as a ModelOpt mixed export) is
    qualified as an external variant with `python3 -m tools.qual variant`
    (tools/qual/variant.py): its lock, the engine's checkpoint identity, the
    teacher skeleton over the frozen workload, and adoption into a temporary
    variant root, which never touches the tree. No torch."""

    @classmethod
    def setUpClass(cls):
        cls.tool = variant_tool()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_identity_is_the_engines(self):
        """Every .safetensors file of the root and its non-hidden immediate
        subdirectories, by name, length and header; the FP8 checkpoint gives
        the pin its server reports, when it is present."""
        checkpoint = os.environ.get("CHOPINFER_QWEN38_CHECKPOINT")
        if checkpoint and Path(checkpoint).is_dir():
            gate = json.loads((ROOT / "models/qwen38/qual.json").read_text())
            self.assertEqual(self.tool.identity(checkpoint), gate["qualification"]["checkpoint"]["checkpoint_identity"])

        def save(path, name):
            header = json.dumps({name: {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))
        save(self.root / "model.safetensors", "a")
        before = self.tool.identity(self.root)
        save(self.root / ".hidden/x.safetensors", "b")
        (self.root / "config.json").write_text("{}")
        self.assertEqual(self.tool.identity(self.root), before)
        save(self.root / "audio/model.safetensors", "c")
        self.assertNotEqual(self.tool.identity(self.root), before)

    def test_a_lock_covers_every_safetensors_file_a_direct_load_opens(self):
        """Shards the index no longer names (an export may keep superseded
        ones) and subdirectory files are locked too, as `chopinfer verify`
        and the checkpoint identity count them; hidden directories are not."""
        from tools.refrun import stamps
        checkpoint = self.root / "checkpoint"
        for name in ("indexed.safetensors", "superseded.safetensors", "audio/model.safetensors",
                     ".hidden/x.safetensors", "config.json", "tokenizer.json", "merges.txt", "vocab.json"):
            (checkpoint / name).parent.mkdir(parents=True, exist_ok=True)
            (checkpoint / name).write_text(name)
        (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "indexed.safetensors"}}))
        target = self.root / "variants/qwen38" / self.tool.lock_name("requant")
        with mock.patch.object(stamps, "RACY_NS", 1):
            path = self.tool.write_lock(target, checkpoint, self.root / "stamps")
            self.assertEqual(sorted(json.loads(path.read_text())),
                             ["audio/model.safetensors", "config.json", "indexed.safetensors", "merges.txt",
                              "model.safetensors.index.json", "superseded.safetensors", "tokenizer.json",
                              "vocab.json"])
            with self.assertRaises(FileExistsError):
                self.tool.write_lock(target, checkpoint, self.root / "stamps")

    def test_the_teacher_skeleton_is_the_frozen_teacher_workload(self):
        """A skeleton over Qwen3.8's frozen teacher workload, with the pin as
        its provenance and the base reference's variant descriptions."""
        pin = {"repository": "someone/qwen-requant", "revision": "a" * 40, "served_model_name": "someone/qwen-requant",
               "source_lock_sha256": "1" * 64, "checkpoint_identity": "2" * 64}
        document, workload = self.tool.skeleton("qwen38", "teacher", pin, ["exact", "bf16"])
        base = json.loads((ROOT / "fixtures/qwen38/teacher-reference.json").read_text())
        self.assertEqual(workload, ROOT / "fixtures/qwen38/teacher-reference.workload.json")
        self.assertEqual(document["schema"], base["schema"])
        self.assertEqual(document["provenance"]["checkpoint"], pin)
        self.assertEqual(list(document["provenance"]["variants"]), ["exact", "bf16"])
        sequences = json.loads(workload.read_text())["sequences"]
        self.assertEqual([(text["name"], text["prompt_token_ids"], text["score_from"]) for text in document["texts"]],
                         [(s["name"], s["token_ids"], s["score_from"]) for s in sequences])

    def emitted(self):
        """The FP8 teacher reference as a stand-in for an export's (bound, so
        adoptable), re-homed onto a copy of the FP8 lock as the variant's."""
        base = json.loads((ROOT / "models/qwen38/qual.json").read_text())
        directory = self.root / "variants/qwen38"
        directory.mkdir(parents=True)
        shutil.copyfile(ROOT / base["qualification"]["checkpoint"]["lock"], directory / self.tool.lock_name("requant"))
        fixture = json.loads((ROOT / base["references"]["teacher"]["path"]).read_text())
        fixture["binding"]["lock"]["path"] = f"qwen38/{self.tool.lock_name('requant')}"
        emitted = self.root / "emitted"
        emitted.mkdir()
        (emitted / "teacher-reference.json").write_text(json.dumps(fixture, indent=1) + "\n")
        shutil.copyfile(ROOT / "fixtures/qwen38/teacher-reference.workload.json",
                        emitted / "teacher-reference.workload.json")
        return dict(base["qualification"]["checkpoint"], lock=self.tool.lock_name("requant")), emitted

    def adopt(self, pin, emitted):
        arguments = ["adopt", "--model", "qwen38", "--variant", "requant", "--dir", str(self.root / "variants"),
                     "--checkpoint", str(self.root / "checkpoint"), "--repository", "someone/qwen-requant",
                     "--revision", "a" * 40, "--reference", "teacher", "--emitted", str(emitted),
                     "--check", "teacher", "--check", "spec_verify", "--check", "verify_exact"]
        with mock.patch.object(self.tool, "pin", return_value=pin), contextlib.redirect_stdout(io.StringIO()):
            return self.tool.main(arguments)

    def test_an_export_adopts_as_an_external_variant_with_its_teacher_and_verify_checks(self):
        """The variant runs qual.json's teacher, spec_verify and verify_exact
        checks, pinned to its checkpoint and its own teacher reference; its
        lock repeats models/qwen38's record of every shared input and
        authenticates each read of what it owns."""
        sys.path.insert(0, str(ROOT / "tools/qual"))
        import common
        import declaration
        import refs
        pin, emitted = self.emitted()
        self.assertEqual(self.adopt(pin, emitted), 0)
        directory = self.root / "variants/qwen38"
        with mock.patch.dict(os.environ, {declaration.VARIANTS_ENV: str(self.root / "variants")}):
            config = declaration.load(declaration.named("qwen38.requant"))
            self.assertEqual((config["model"], config["variant"]), ("qwen38", "requant"))
            self.assertEqual({check["id"] for check in config["checks"]}, {"teacher", "spec_verify", "verify_exact"})
            self.assertEqual(config["profiles"]["quick"]["checks"], ["teacher"])
            self.assertEqual(set(config["qualification"]["references"]), {"teacher"})
            owned = {"env": declaration.VARIANTS_ENV, "path": "qwen38/requant.teacher-reference.json"}
            self.assertEqual(config["qualification"]["references"]["teacher"]["path"], owned)
            self.assertEqual((config["inputs"]["exact"], config["inputs"]["teacher_reference"]), (owned, owned))
            self.assertEqual(config["qualification"]["checkpoint"]["lock"], "qwen38/requant.checkpoint-files.json")
            lock = refs.load_lock(config)
            base_lock = json.loads((ROOT / "models/qwen38/qual-references.lock.json").read_text())
            shared = [key for key in lock["entries"] if key in base_lock["entries"]
                      and lock["entries"][key]["path"] == base_lock["entries"][key]["path"]]
            self.assertLessEqual({"fixture_root", "frontend", "gsm8k", "spec_teacher"}, set(shared))
            for key in shared:
                self.assertEqual(lock["entries"][key], base_lock["entries"][key], key)
            producer = lock["objects"][lock["entries"]["reference:teacher"]["members"]["."]]["production"]
            self.assertEqual(producer["producer"], "runner-binding")
            self.tool.verify(config)
            reference = directory / "requant.teacher-reference.json"
            self.assertEqual(common.input_bytes("exact", config=config), reference.read_bytes())
            reference.write_bytes(reference.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "lock entry exact: sha256 differs"):
                common.input_bytes("exact", config=config)

    def test_adopt_refuses_existing_evidence_before_writing(self):
        pin, emitted = self.emitted()
        existing = self.root / "variants/qwen38/requant.json"
        existing.write_text("{}")
        with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            self.adopt(pin, emitted)
        self.assertIn("refusing to replace qwen38.requant evidence", error.getvalue())
        self.assertEqual(sorted(path.name for path in existing.parent.iterdir()),
                         ["requant.checkpoint-files.json", "requant.json"])
        self.assertEqual(existing.read_text(), "{}")


def declaration_of(name):
    import declaration
    return declaration.load(str(ROOT / "models/qwen38" / name))


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class LongContextAdapterTests(unittest.TestCase):
    def test_bounded_prefill_keeps_hf_f32_operators_and_has_its_own_identity(self):
        bounded = ROOT / "models/qwen38/reference_long_context.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_checkpoint(root / "checkpoint", seed=19)
            tokens = [16 + i % 40 for i in range(1057)]
            workload = root / "workload.json"
            workload.write_text(json.dumps({"schema": "chopinfer.refrun.workload.v1", "model": "qwen38",
                "sequences": [{"name": "boundary", "category": "prose", "token_ids": tokens,
                               "score_from": 1020}]}))
            results, layouts = {}, {}
            for name, adapter in (("original", ADAPTER), ("bounded", bounded)):
                out = root / name
                provenance = run.run(run.parse_args([
                    "--adapter", str(adapter), "--checkpoint", str(root / "checkpoint"),
                    "--lock", str(root / "checkpoint/lock.json"), "--workload", str(workload),
                    "--modes", "f32", "--devices", "cpu,cpu", "--output", str(out),
                    "--stamps", str(root / "stamps"), "--no-cache"]))
                layouts[name] = provenance["layout"]
                results[name] = json.loads((out / "f32.json").read_text())["entries"][0]
            old, new = results["original"], results["bounded"]
            self.assertEqual(new["scores"]["token"], old["scores"]["token"])
            self.assertEqual(new["scores"]["top1"], old["scores"]["top1"])
            self.assertLess(max(abs(a - b) for a, b in zip(new["scores"]["logprob"],
                                                          old["scores"]["logprob"])), 2e-4)
            self.assertEqual(new["key"]["mode"], old["key"]["mode"])
            self.assertNotEqual(new["key_sha256"], old["key_sha256"])
            self.assertEqual(layouts["original"]["prefill_chunk"], 2048)
            self.assertEqual(layouts["bounded"]["prefill_chunk"], 1024)
            self.assertEqual(layouts["original"]["dequant"], "exact")
            self.assertEqual(layouts["bounded"]["dequant"], "exact")
            self.assertEqual(layouts["original"]["upstream_f32"], layouts["bounded"]["upstream_f32"])
            module, _ = run.load_adapter(bounded)
            self.assertEqual(module.reference_code(None)["models/qwen38/reference.py"], ADAPTER)
            canonical, _ = run.load_adapter(ADAPTER)
            self.assertEqual(canonical.PREFILL_CHUNK, 2048)


class LongContextWorkloadTests(unittest.TestCase):
    def test_128k_workload_selects_the_frozen_tail_and_refuses_changed_tokens(self):
        from models.qwen38 import reference_workloads as workloads
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "prompts").mkdir()
            full = {"long": [{"name": name, "token_ids": list(range(count))}
                             for name, count in (("long_32k", 2048), ("long_128k", 4096))]}
            full_path = root / "prompts/prompts.json"
            full_path.write_text(json.dumps(full))
            frozen = {"full_file": {"sha256": workloads.sha256(full_path.read_bytes())},
                      "long": [{"name": text["name"], "token_count": len(text["token_ids"]),
                                "score_last": 1024,
                                "token_ids_sha256": workloads.sha256(json.dumps(text["token_ids"]).encode())}
                               for text in full["long"]]}
            (root / "prompts.json").write_text(json.dumps(frozen))
            with mock.patch.object(workloads, "FIXTURES", root):
                selected = workloads.long_context(root, workloads.LONG_CONTEXT_128K)
                self.assertEqual([text["name"] for text in selected], ["long_128k"])
                self.assertEqual(selected[0]["token_ids"], full["long"][1]["token_ids"])
                self.assertEqual(selected[0]["score_from"], 3072)
                self.assertEqual([text["name"] for text in workloads.long_context(root)], ["long_32k"])
                frozen["long"][1]["token_ids_sha256"] = "0" * 64
                (root / "prompts.json").write_text(json.dumps(frozen))
                with self.assertRaisesRegex(ValueError, "do not hash"):
                    workloads.long_context(root, workloads.LONG_CONTEXT_128K)


class VariantTests(unittest.TestCase):
    def test_verify_item_widths_match_the_served_draft_limit(self):
        import declaration
        base = json.loads((ROOT / "models/qwen38/qual.json").read_text())
        for name in [None, *base["configurations"]]:
            value = base if name is None else json.loads(declaration.materialized("qwen38", name).read_text())
            if "verify_exact" in value["cohorts"]["full"]["checks"]:
                # package.rs retains selector widths 2/4/8; served K3 can
                # reach only widths 2/4. The wide-row experiment is separate.
                self.assertEqual(value["qualification"]["verify_buckets"], [2, 4], name)

    """The dcp2 configuration of qual.json gates TP4 at context-parallel
    size 2 with qual.json's checks, limits and references, plus the shared
    determinism check."""

    def setUp(self):
        for path in (ROOT / "tools/qual", ROOT / "models/qwen38"):
            sys.path.insert(0, str(path))
            self.addCleanup(sys.path.remove, str(path))

    def test_no_configuration_has_a_declaration_file_beside_qual_json(self):
        """qual.json declares each configuration once, and no qual.VARIANT.json
        beside it: another checkpoint is an external variant, kept outside
        the tree (tools/qual/README.md#external-variants)."""
        import declaration
        self.assertEqual(declaration.variants("qwen38", ROOT), [])

    def test_dcp2_keeps_every_limit_and_requires_its_servers(self):
        """Each of qual.json's checks keeps its rules and constants; the
        configuration only adds the serving profile (and the served model)
        to the live checks and every record."""
        import declaration
        qwen = json.loads((ROOT / "models/qwen38/qual.json").read_text())
        dcp2 = json.loads(declaration.materialized("qwen38", "dcp2").read_text())
        self.assertEqual(dcp2["variant"], "dcp2")
        self.assertEqual(dcp2["cohorts"]["full"]["checks"],
                         ["teacher", "long_context", "long_context_128k", "spec_verify", "verify_exact",
                          "prefix_exact", "preemption_exact", "preemption_exact_c1", "determinism"])
        # qual.json declares determinism for its configurations; its own cohorts do not run it.
        self.assertNotIn("determinism", {check for cohort in qwen["cohorts"].values() for check in cohort["checks"]})
        self.assertEqual(dcp2["references"], {name: qwen["references"][name]
                                              for name in ("teacher", "long_context", "long_context_128k")})
        self.assertEqual(dcp2["qualification"]["checkpoint"], qwen["qualification"]["checkpoint"])
        profile = ["server", "health", "serving", "profile"]
        for name, check in dcp2["checks"].items():
            rules = check["rules"] + [rule for rules in check.get("input_rules", {}).values() for rule in rules]
            sizes = [rule["value"] for rule in rules if rule["path"] == [*profile, "context_parallel_size"]]
            self.assertTrue(sizes and set(sizes) == {2}, name)
            if name == "determinism":
                continue
            base = qwen["checks"][name]
            self.assertEqual(check["rules"][:len(base["rules"])], base["rules"], name)
            self.assertEqual(check.get("constants"), base.get("constants"), name)
            self.assertEqual(check["args"], base["args"], name)
            for key, rules in base.get("input_rules", {}).items():
                self.assertEqual(check["input_rules"][key][:len(rules)], rules, f"{name} {key}")
            added = check["rules"][len(base["rules"]):]
            for key, rules in check.get("input_rules", {}).items():
                added = added + rules[len(base.get("input_rules", {}).get(key, [])):]
            self.assertTrue(all(rule["op"] == "eq" and rule["path"][:1] == ["server"] for rule in added), name)

    def test_dcp2_records_send_the_prompts_qual_json_records_send(self):
        """spec_verify's and preempt_compare's records build their prompts
        from declared inputs (gsm8k among them); a DCP 2 record must send
        what a DCP 1 record sends."""
        import common
        import declaration
        try:
            common.input_path("gsm8k", config=declaration_of("qual.json"))
        except (OSError, ValueError) as error:
            self.skipTest(f"needs the gsm8k fixture: {error}")
        import preempt_compare
        import spec_verify
        sent = {}
        for name, path in (("default", ROOT / "models/qwen38/qual.json"),
                           ("dcp2", declaration.materialized("qwen38", "dcp2"))):
            with mock.patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(path)}):
                sent[name] = (spec_verify.prompt_sets(), preempt_compare.prompts())
        self.assertEqual(sent["dcp2"], sent["default"])


if __name__ == "__main__":
    unittest.main()

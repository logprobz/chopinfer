"""The GLM 5.3 Flash reference adapter on tiny random glm5_next checkpoints, on CPU.

The checkpoints have GLM 5.3 Flash's stored layout at toy sizes: KDA layers
(separate q, k and v convs, the forget gate beside the projections) and
sparse-attention layers with an indexer, a dense layer then MoE layers with
FP8 E4M3 weights and F32 scales per 128 x 128 block, mHC tensors under their
checkpoint names, an MTP layer the reference must not load, and a two-block
vision tower. A second checkpoint stores one layer's routed experts as
NVFP4 (with an activation scale the reference must not read) and another's
shared expert as MXFP8. The runner's scores, through the adapter's chunked
prefill, blocked sparse attention, grouped experts and chunked vision
attention, must match a plain HF `model(...)` forward over the same weights
decoded by the definitions. Needs torch and transformers 5.16.1; skipped
without them.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B -m unittest models/glm5next/test_reference.py
"""

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from transformers.models.glm5_next import configuration_glm5_next, modeling_glm5_next as hf
    from tools.refrun import modes, place, quant, run, stamps
except ModuleNotFoundError:
    torch = None

ADAPTER = ROOT / "models/glm5next/reference.py"
TEXT_PATH = "model.language_model"
IMAGE, IMAGE_START, IMAGE_END, EOS = 60, 58, 59, 63
LAYER_TYPES = ["linear_attention", "deepseek_sparse_attention", "linear_attention", "deepseek_sparse_attention"]
TEXT = {"vocab_size": 64, "hidden_size": 160, "intermediate_size": 288, "moe_intermediate_size": 160,
        "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 2, "n_routed_experts": 8,
        "num_experts_per_tok": 2, "n_shared_experts": 1, "n_group": 1, "topk_group": 1, "routed_scaling_factor": 2.5,
        "kv_lora_rank": 32, "q_lora_rank": 48, "qk_rope_head_dim": 0, "qk_nope_head_dim": 16, "v_head_dim": 16,
        "index_topk": 8, "index_head_dim": 16, "index_n_heads": 8, "index_kpool": 4,
        "layer_types": LAYER_TYPES, "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
        "indexer_types": ["full"] * 4, "num_nextn_predict_layers": 1,
        "linear_attn_config": {"num_heads": 2, "head_dim": 16, "short_conv_kernel_size": 4, "gate_lower_bound": -5.0},
        "hc_mult": 4, "hc_sinkhorn_iters": 20, "rms_norm_eps": 1e-5, "swiglu_limit": 10.0,
        "model_type": "glm5_next_text", "pad_token_id": 0, "eos_token_id": [EOS], "tie_word_embeddings": False}
VISION = {"depth": 2, "hidden_size": 32, "num_heads": 2, "intermediate_size": 64, "out_hidden_size": 160,
          "projection_intermediate_size": 64, "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2,
          "in_channels": 3, "model_type": "glm5_next_vision"}
CONFIG = {"architectures": ["Glm5NextForConditionalGeneration"], "model_type": "glm5_next", "image_token_id": IMAGE,
          "video_token_id": 61, "image_start_token_id": IMAGE_START, "image_end_token_id": IMAGE_END,
          "video_start_token_id": 56, "video_end_token_id": 57, "tie_word_embeddings": False,
          "text_config": TEXT, "vision_config": VISION,
          "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                  "weight_block_size": [128, 128], "modules_to_not_convert": []}}
GRID = [1, 4, 6]
PROMPT = [3, 9, 17, 4, 22, 5, 30, 7, 11, 12, 40, 41, 2, 19, 23, 8, 45, 6, 33, 27, 14]
IMAGE_PROMPT = [3, 9, IMAGE_START, *[IMAGE] * 6, IMAGE_END, 17, 4, 22, 5, 30, 7, 11]
FP8_LINEAR = re.compile(r"self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa)\.weight$"
                        r"|mlp\.(gate_proj|up_proj|down_proj)\.weight$|shared_experts\.\w+\.weight$")
F32_STORED = ("dt_bias", "A_log", "e_score_correction_bias", "hc.base", "hc.scale")
# The mixed checkpoint: layer 1's routed experts NVFP4, layer 2's shared expert MXFP8.
NVFP4_EXPERTS, MXFP8_SHARED = f"{TEXT_PATH}.layers.1.mlp.experts.", f"{TEXT_PATH}.layers.2.mlp.shared_experts."
MODES = ("f32", "f64", "f32-hf-bf16", "bf16")


def setUpModule():
    if torch is None:
        return
    modes.prepare_environment(cuda=False)
    global RACY_NS
    RACY_NS, stamps.RACY_NS = stamps.RACY_NS, 100_000_000


def tearDownModule():
    if torch is not None:
        stamps.RACY_NS = RACY_NS


def e4m3_codes(generator, shape):
    """Random E4M3 codes without the two NaN codes."""
    out = torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)
    out[(out & 0x7F) == 0x7F] = 0x38
    return out


def checkpoint_name(name):
    """The checkpoint name of an HF tensor (transformers' renames, reversed)."""
    for old, new in (("forget_gate.", ""), ("attn_hc.fn", "hc_attn_fn"), ("attn_hc.base", "hc_attn_base"),
                     ("attn_hc.scale", "hc_attn_scale"), ("ffn_hc.fn", "hc_ffn_fn"), ("ffn_hc.base", "hc_ffn_base"),
                     ("ffn_hc.scale", "hc_ffn_scale")):
        name = name.replace(old, new)
    return name


def hf_config():
    return configuration_glm5_next.Glm5NextConfig.from_dict(json.loads(json.dumps(CONFIG)))


def write_checkpoint(root, mixed=False, seed=0):
    """config.json, two safetensors shards with an index, and a source lock;
    returns the stored tensors."""
    from safetensors.torch import save_file
    root.mkdir(parents=True)
    generator = torch.Generator().manual_seed(seed)
    config = hf_config()
    text = config.text_config
    with torch.device("meta"):
        model = hf.Glm5NextForConditionalGeneration(config)
    tensors = {}

    def fp8(name, shape):
        tensors[name] = e4m3_codes(generator, shape).view(torch.float8_e4m3fn)
        grid = quant.fp8_block().scale_shape(*shape)
        tensors[name + "_scale_inv"] = torch.rand(grid, generator=generator) * 0.02 + 0.005

    def nvfp4(name, shape):
        tensors[name] = torch.randint(0, 256, (shape[0], shape[1] // 2), generator=generator, dtype=torch.uint8)
        tensors[name + "_scale"] = e4m3_codes(generator, (shape[0], shape[1] // 16)).clamp(max=0x50).view(
            torch.float8_e4m3fn)
        tensors[name + "_scale_2"] = torch.tensor(0.00390625 * 1.3, dtype=torch.float32)
        tensors[name.replace(".weight", ".input_scale")] = torch.tensor(0.5, dtype=torch.float32)

    def mxfp8(name, shape):
        tensors[name] = e4m3_codes(generator, shape).view(torch.float8_e4m3fn)
        tensors[name + "_scale"] = torch.randint(118, 124, (shape[0], shape[1] // 32), generator=generator,
                                                 dtype=torch.uint8)

    def stored(name, shape):
        if mixed and name.startswith(NVFP4_EXPERTS):
            nvfp4(name, shape)
        elif mixed and name.startswith(MXFP8_SHARED):
            mxfp8(name, shape)
        else:
            fp8(name, shape)
    sparse = {layer for layer, kind in enumerate(LAYER_TYPES) if kind != "linear_attention"}
    for name, value in [*model.named_parameters(), *model.named_buffers()]:
        if "inv_freq" in name:
            continue
        layer = int(name.split(".layers.")[1].split(".")[0]) if name.startswith(TEXT_PATH + ".layers.") else None
        if name.endswith("experts.gate_up_proj"):
            base = name[:-len("gate_up_proj")]
            experts, rows, cols = value.shape
            for expert in range(experts):
                stored(f"{base}{expert}.gate_proj.weight", (rows // 2, cols))
                stored(f"{base}{expert}.up_proj.weight", (rows // 2, cols))
        elif name.endswith("experts.down_proj"):
            base = name[:-len("down_proj")]
            for expert in range(value.shape[0]):
                stored(f"{base}{expert}.down_proj.weight", tuple(value.shape[1:]))
        elif name.endswith("self_attn.conv1d.weight"):
            for part in "qkv":
                tensors[name.replace("conv1d", f"{part}_conv1d")] = (
                    torch.randn((value.shape[0] // 3, 1, value.shape[2]), generator=generator) * 0.3).to(torch.bfloat16)
        elif layer is not None and (FP8_LINEAR.search(name)
                                    or (layer in sparse and name.endswith("self_attn.o_proj.weight"))):
            stored(checkpoint_name(name), tuple(value.shape))
        else:
            scale = 0.05 if "norm" in name else 0.3
            random = torch.randn(value.shape, generator=generator) * scale
            if "hc.scale" in name:
                random = random.abs() + 0.5
            tensors[checkpoint_name(name)] = random if any(k in name for k in F32_STORED) else random.to(torch.bfloat16)
    mtp = f"{TEXT_PATH}.layers.{text.num_hidden_layers}."
    hidden = text.hidden_size
    for name, shape in (("eh_proj.weight", (hidden, 2 * hidden)), ("enorm.weight", (hidden,)),
                        ("hnorm.weight", (hidden,)), ("shared_head.norm.weight", (hidden,))):
        tensors[mtp + name] = torch.randn(shape, generator=generator).to(torch.bfloat16)
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


def table(values, codes):
    return torch.tensor(values, dtype=torch.float64)[codes.long()]


def decoded(tensors, name):
    """A stored weight in f64 by the definitions (BF16 widened)."""
    weight = tensors[name]
    if weight.dtype in (torch.bfloat16, torch.float32):
        return weight.double()
    if name + "_scale_inv" in tensors:
        values, scale = table(quant.E4M3, weight.view(torch.uint8)), tensors[name + "_scale_inv"].double()
        rows, cols = values.shape
        return values * scale.repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)[:, :cols]
    if name + "_scale_2" in tensors:
        codes = weight.view(torch.uint8)
        values = torch.stack([table(quant.E2M1, codes & 0xF), table(quant.E2M1, codes >> 4)], dim=-1).flatten(1)
        scale = table(quant.E4M3, tensors[name + "_scale"].view(torch.uint8)) * float(tensors[name + "_scale_2"])
        return values * scale.repeat_interleave(16, 1)
    values, scale = table(quant.E4M3, weight.view(torch.uint8)), table(quant.E8M0, tensors[name + "_scale"])
    return values * scale.repeat_interleave(32, 1)


def weight(tensors, name, dtype, rounded):
    """`name` in the plain model: the exact product rounded once to `dtype`
    (the rounded rule; exact in f64), or rounded to BF16 first (hf-bf16)."""
    value = decoded(tensors, name)
    if rounded and tensors[name].dtype not in (torch.bfloat16, torch.float32):
        value = value.float().to(torch.bfloat16)
    return value.to(dtype)


def plain_model(tensors, dtype, rounded=False, conv_order="qkv"):
    """HF's model with every weight decoded and in place: eager sparse
    attention, SDPA vision attention, and HF's grouped_mm experts (its eager
    experts in f64, which torch's grouped_mm does not take)."""
    config = hf_config()
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = hf.Glm5NextForConditionalGeneration(config).eval()
    finally:
        torch.set_default_dtype(default)
    for module in model.modules():
        owner = getattr(module, "config", None)
        if isinstance(owner, configuration_glm5_next.Glm5NextTextConfig):
            owner._attn_implementation = "eager"
            owner._experts_implementation = "eager" if dtype == torch.float64 else "grouped_mm"
        if isinstance(owner, configuration_glm5_next.Glm5NextVisionConfig):
            owner._attn_implementation = "sdpa"
    text = config.text_config
    state = {}
    for name, value in model.state_dict().items():
        if "inv_freq" in name:
            continue
        if name.endswith("experts.gate_up_proj"):
            base = name[:-len("gate_up_proj")]
            state[name] = torch.stack([torch.cat([weight(tensors, f"{base}{e}.{p}.weight", dtype, rounded)
                                                  for p in ("gate_proj", "up_proj")])
                                       for e in range(text.n_routed_experts)])
        elif name.endswith("experts.down_proj"):
            base = name[:-len("down_proj")]
            state[name] = torch.stack([weight(tensors, f"{base}{e}.down_proj.weight", dtype, rounded)
                                       for e in range(text.n_routed_experts)])
        elif name.endswith("self_attn.conv1d.weight"):
            state[name] = torch.cat([tensors[name.replace("conv1d", f"{p}_conv1d")] for p in conv_order]).to(dtype)
        else:
            state[name] = weight(tensors, checkpoint_name(name), dtype, rounded)
    model.load_state_dict(state, strict=False)
    return model


def pixels(generator):
    return torch.randn((GRID[1] * GRID[2], 3 * 2 * 2 * 2), generator=generator)


def media(pixel_rows):
    return [{"kind": "image", "tensor": {"path": "image.bin", "dtype": "f32", "shape": list(pixel_rows.shape)},
             "grid": {"path": "grid.bin", "dtype": "i64", "shape": [1, 3]},
             "processor": {"code_sha256": "0" * 64, "config_sha256": "0" * 64}}]


def run_args(root, checkpoint, output, *extra):
    return run.parse_args(["--adapter", str(ADAPTER), "--checkpoint", str(root / checkpoint),
                           "--lock", str(root / checkpoint / "lock.json"), "--workload", str(root / "workload.json"),
                           "--devices", "cpu,cpu", "--output", str(root / output), "--stamps", str(root / "stamps"),
                           *extra])


def configured(name):
    """The declaration of qual.json's configuration `name`, as `chopinfer
    eval declaration` expands it."""
    if str(ROOT / "tools/qual") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools/qual"))
    import declaration
    return json.loads(declaration.materialized("glm5next", name).read_text())


def default():
    """qual.json, the default configuration's declaration."""
    return json.loads((ROOT / "models/glm5next/qual.json").read_text())


def gate_checks(value):
    """A declaration's checks its cohorts run."""
    run = {check for cohort in value["cohorts"].values() for check in cohort["checks"]}
    return {key: check for key, check in value["checks"].items() if key in run}


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        root = cls.root = Path(cls.directory.name)
        # Seed 5: the greedy probe answers without an image token.
        cls.tensors = write_checkpoint(root / "checkpoint", seed=5)
        cls.mixed = write_checkpoint(root / "mixed", mixed=True, seed=1)
        cls.pixels = pixels(torch.Generator().manual_seed(2))
        (root / "image.bin").write_bytes(cls.pixels.numpy().tobytes())
        (root / "grid.bin").write_bytes(torch.tensor([GRID], dtype=torch.int64).numpy().tobytes())
        sequences = [{"name": "text", "category": "prose", "token_ids": PROMPT, "score_from": 1},
                     {"name": "image", "category": "probe", "token_ids": IMAGE_PROMPT, "score_from": 11,
                      "answer": {"policy": "forced", "ids": [33, 34, 35]}, "media": media(cls.pixels)},
                     {"name": "greedy", "category": "probe", "token_ids": IMAGE_PROMPT,
                      "score_from": len(IMAGE_PROMPT),
                      "answer": {"policy": "greedy", "max_new_tokens": 4, "stop_ids": [EOS]},
                      "media": media(cls.pixels)}]
        (root / "workload.json").write_text(json.dumps({"schema": "chopinfer.refrun.workload.v1", "model": "glm5next",
                                                        "sequences": sequences}))
        load_adapter = run.load_adapter
        cls.adapters = []

        def small_chunks(path):
            module, relative = load_adapter(path)
            # Several prefill chunks, attention blocks and vision chunks on toy sizes.
            # Prefill chunks of 6 rows end inside pools of 4 keys.
            module.PREFILL_ROWS, module.ROW_ALIGN, module.VISION_CHUNK = 6, 2, 7
            # Query blocks of 4 rows, halved when a block selects more than
            # 12 keys.
            module.QUERY_ROWS, module.EXPAND_BYTES = 4, 12 * 2 * (16 + 16 + 16) * 8
            cls.adapters.append(module)
            return module, relative
        with mock.patch.object(run, "load_adapter", side_effect=small_chunks):
            cls.provenance = run.run(run_args(root, "checkpoint", "out", "--modes", "f32,f64,f32:hf-bf16,bf16",
                                              "--dequant", "f32=rounded", "--no-cache"))
            # The same f32 pass with every tap written, over the same chunks.
            run.run(run_args(root, "checkpoint", "tapped", "--modes", "f32", "--dequant", "f32=rounded",
                             "--taps", "--tap-sites", "all", "--no-cache"))
            cls.mixed_provenance = run.run(run_args(root, "mixed", "mixed-out", "--modes", "f32,f64",
                                                    "--dequant", "f32=rounded", "--no-cache"))
        cls.scores = cls.read(root / "out", MODES)
        cls.mixed_scores = cls.read(root / "mixed-out", ("f32", "f64"))

    @classmethod
    def read(cls, output, labels):
        return {label: {(e["name"], e["key"]["sequence"]["answer"]["policy"]): e["scores"]
                        for e in json.loads((output / f"{label}.json").read_text())["entries"]} for label in labels}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def plain_logprobs(self, model, ids, score_from, dtype, image=False):
        kwargs = {"pixel_values": self.pixels, "image_grid_thw": torch.tensor([GRID])} if image else {}
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]), use_cache=False, **kwargs).logits[0].to(dtype)
        return torch.log_softmax(logits[score_from - 1:-1], dim=-1)

    def check(self, scores, tensors, label, dtype, rounded, tolerance):
        model = plain_model(tensors, dtype, rounded)
        worst = 0.0
        for name, ids, score_from, image in (("text", PROMPT, 1, False),
                                             ("image", IMAGE_PROMPT + [33, 34, 35], 11, True)):
            expected = self.plain_logprobs(model, ids, score_from, dtype, image)
            got_scores = scores[label][(name, "forced")]
            self.assertEqual(got_scores["token"], ids[score_from:])
            got = torch.tensor(got_scores["logprob"], dtype=torch.float64)
            chosen = expected.gather(1, torch.tensor(ids[score_from:])[:, None])[:, 0].double()
            worst = max(worst, (got - chosen).abs().max().item())
            self.assertLess(worst, tolerance, (label, name))
            self.assertEqual(got_scores["top1"], expected.argmax(dim=-1).tolist(), (label, name))
        return worst

    def test_f64_and_f32_match_a_plain_forward(self):
        # HF computes norms, mHC, KDA, the router and the indexer in F32 even
        # in f64, so chunked prefill differs from one pass at F32 rounding.
        self.check(self.scores, self.tensors, "f64", torch.float64, False, 1e-5)
        self.check(self.scores, self.tensors, "f32", torch.float32, False, 2e-5)

    def test_nvfp4_and_mxfp8_weights_match_a_plain_forward(self):
        self.check(self.mixed_scores, self.mixed, "f64", torch.float64, False, 1e-5)
        self.check(self.mixed_scores, self.mixed, "f32", torch.float32, False, 2e-5)
        # The NVFP4 global scales are loaded, the activation scales are not.
        checkpoint = place.Checkpoint(self.root / "mixed")
        loaded = {n for unit in self.adapters[-1].units(checkpoint) for n in unit.tensors}
        self.assertIn(NVFP4_EXPERTS + "0.gate_proj.weight_scale_2", loaded)
        self.assertIn(MXFP8_SHARED + "down_proj.weight_scale", loaded)
        self.assertTrue([n for n in checkpoint.entries if n.endswith(".input_scale")])
        self.assertFalse([n for n in loaded if n.endswith(".input_scale")])

    def test_hf_bf16_rule_rounds_quantized_weights(self):
        self.check(self.scores, self.tensors, "f32-hf-bf16", torch.float32, True, 2e-5)
        exact, rounded = (self.scores[label][("text", "forced")]["logprob"] for label in ("f32", "f32-hf-bf16"))
        self.assertGreater(max(abs(a - b) for a, b in zip(exact, rounded)), 1e-5)
        bf16 = self.scores["bf16"][("text", "forced")]["logprob"]
        self.assertLess(max(abs(a - b) for a, b in zip(rounded, bf16)), 0.5)

    def test_conv_order_is_a_layout_correction(self):
        """Negative control: concatenating the stored convs in another order
        moves the scores far past the f64 tolerance."""
        model = plain_model(self.tensors, torch.float64, conv_order="kqv")
        expected = self.plain_logprobs(model, PROMPT, 1, torch.float64)
        chosen = expected.gather(1, torch.tensor(PROMPT[1:])[:, None])[:, 0]
        got = torch.tensor(self.scores["f64"][("text", "forced")]["logprob"], dtype=torch.float64)
        self.assertGreater((got - chosen).abs().max().item(), 1e-3)

    def test_greedy_answer_is_the_plain_models_argmax(self):
        model = plain_model(self.tensors, torch.float32)
        ids = list(IMAGE_PROMPT)
        for _ in range(4):
            logits = self.plain_logprobs(model, ids + [0], len(ids), torch.float32, image=True)
            ids.append(int(logits[-1].argmax()))
            if ids[-1] == EOS:
                break
        answer = ids[len(IMAGE_PROMPT):]
        self.assertEqual(self.scores["f32"][("greedy", "greedy")]["answer"], answer)
        self.assertEqual(self.scores["f64"][("greedy", "forced")]["token"], answer)

    def test_exact_f32_refuses_f32_scaled_fp8(self):
        with self.assertRaisesRegex(ValueError, "not exact in torch.float32"):
            run.run(run_args(self.root, "checkpoint", "exact", "--modes", "f32", "--no-cache"))

    def test_cache_replay_does_not_load(self):
        cache = self.root / "cache"
        first = run.run(run_args(self.root, "checkpoint", "cached-1", "--modes", "f32", "--dequant", "f32=rounded",
                                 "--cache", str(cache)))
        second = run.run(run_args(self.root, "checkpoint", "cached-2", "--modes", "f32", "--dequant", "f32=rounded",
                                  "--cache", str(cache)))
        self.assertIsNotNone(first["load"])
        self.assertIsNone(second["load"])
        # Three forced entries and the greedy one.
        self.assertEqual(second["modes"]["f32"]["cache_hits"], 4)
        self.assertEqual(json.loads((self.root / "cached-1/f32.json").read_text())["entries"],
                         json.loads((self.root / "cached-2/f32.json").read_text())["entries"])

    def test_taps_over_chunked_prefill(self):
        """Prefill chunks of 6 tokens: every text tap holds every position at
        its absolute row, the image's encoder rows are written once, each
        step completes, and the scores equal the untapped run's."""
        from tools.qual import tapdiff
        tapped = json.loads((self.root / "tapped/f32.json").read_text())["entries"]
        untapped = json.loads((self.root / "out/f32.json").read_text())["entries"]
        self.assertEqual([e["scores"] for e in tapped], [e["scores"] for e in untapped])
        steps = sorted((self.root / "tapped/taps/f32").iterdir())
        self.assertEqual(len(steps), 3)
        embed = self.tensors[f"{TEXT_PATH}.embed_tokens.weight"].float()
        for step, ids in zip(steps, (PROMPT, IMAGE_PROMPT + [33, 34, 35], None)):
            self.assertEqual(json.loads((step / "completion.json").read_text())["completion"], "full", step.name)
            index = {e["name"]: e for e in map(json.loads, (step / "index.jsonl").read_text().splitlines())}
            items = json.loads((step / "items.json").read_text())
            length = items["items"][0]["seq_len"]
            self.assertEqual(items["positions"], list(range(length)))
            for name, entry in index.items():
                self.assertEqual(entry["rows"], (6 if name == "enc.vision.out" else 24) if name.startswith("enc.")
                                 else length, (step.name, name))
            self.assertEqual(index["L01.x.indexer.ids"]["cols"], 8 + 3)
            self.assertEqual(index["L01.x.latent"]["cols"], 32)
            self.assertEqual(index["L00.x.kda.g"]["cols"], 32)
            self.assertEqual(index["L00.in.streams"]["cols"], 4 * 160)
            selected = torch.from_numpy(tapdiff.read(step / "L03.x.indexer.ids.r0.bin",
                                                     index["L03.x.indexer.ids"]).astype("int64"))
            none = selected == 0xFFFFFFFF
            positions = torch.arange(length)[:, None].expand_as(selected)
            # Causal: every selected key is at or before its query.
            self.assertTrue(bool(((selected <= positions) | none).all()), step.name)
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
            else:
                self.assertFalse([n for n in index if n.startswith("enc.")])

    def test_layout_names_and_taps(self):
        layout = self.provenance["layout"]
        self.assertEqual(layout["dequant"], "hf-bf16")
        self.assertEqual(layout["keep_f32"], ["e_score_correction_bias", "conv1d", "dt_bias", "A_log"])
        units = dict(self.provenance["placement"]["units"])
        self.assertNotIn(f"{TEXT_PATH}.layers.4", units)
        self.assertEqual(list(units)[-2:], [f"{TEXT_PATH}.norm", "lm_head"])
        adapter = self.adapters[0]
        with torch.device("meta"):
            model = hf.Glm5NextForConditionalGeneration(hf_config())
        taps = adapter.taps(model)
        name = re.compile(r"(embed|L\d{2}|final|enc\.vision(\.B\d{2})?)\.(in\.streams|attn\.(in|q|k|v|core|out)"
                          r"|mlp\.(in|out)|router\.(logits|ids|weights)|moe\.(routed|shared)|out|norm"
                          r"|x(\.[a-z0-9_]+)+)")
        for tap, spec in taps.items():
            self.assertRegex(tap, name)
            model.get_submodule(spec if isinstance(spec, str) else spec.module)
        self.assertEqual(taps["L03.x.indexer.ids"].module, f"{TEXT_PATH}.layers.3.self_attn.indexer")
        self.assertEqual(taps["L00.x.kda.core"].module, f"{TEXT_PATH}.layers.0.self_attn.o_norm")
        self.assertNotIn("L00.router.ids", taps)
        self.assertIn("L01.router.ids", taps)
        self.assertEqual(int(taps["L01.x.indexer.ids"].transform(torch.tensor([-1]))[0]), 0xFFFFFFFF)

    def test_built_bf16_model_has_hf_dtypes(self):
        """The bf16 model as built: HF's strict-F32 tensors and the F32-stored
        mHC base and scale in F32, everything else BF16, and the pinned
        experts and attention."""
        from tools.refrun import adapter as interface, place
        module = self.adapters[0]
        checkpoint = place.Checkpoint(self.root / "checkpoint")
        units = module.units(checkpoint)
        placement = place.plan([(u.module, checkpoint.stored_bytes(u.tensors)) for u in units], ["cpu"])
        store = place.Store(checkpoint, placement, units)
        model = module.build(interface.BuildConfig(checkpoint, modes.policy("bf16"), torch.bfloat16, store),
                             placement).model
        text = f"{TEXT_PATH}.layers"
        for name, dtype in ((f"{text}.0.self_attn.conv1d.weight", torch.float32),
                            (f"{text}.0.self_attn.forget_gate.A_log", torch.float32),
                            (f"{text}.0.self_attn.forget_gate.dt_bias", torch.float32),
                            (f"{text}.1.mlp.gate.e_score_correction_bias", torch.float32),
                            (f"{text}.1.attn_hc.base", torch.float32), (f"{text}.1.ffn_hc.scale", torch.float32),
                            (f"{text}.1.attn_hc.fn", torch.bfloat16),
                            (f"{text}.1.input_layernorm.weight", torch.bfloat16),
                            (f"{text}.1.mlp.gate.weight", torch.bfloat16),
                            (f"{text}.1.self_attn.indexer.weights_proj.weight", torch.bfloat16),
                            (f"{text}.0.self_attn.o_norm.weight", torch.bfloat16)):
            tensor = model.get_parameter(name) if name in dict(model.named_parameters()) else model.get_buffer(name)
            self.assertEqual(tensor.dtype, dtype, name)
        block = model.model.language_model.layers[1]
        self.assertIsInstance(block.self_attn.q_a_proj, quant.QuantLinear)
        self.assertEqual(block.mlp.experts.config._experts_implementation, module.EXPERTS)
        self.assertEqual(block.self_attn.config._attn_implementation, module.SPARSE_ATTENTION)

    def test_grouped_experts_match_hf_grouped_mm(self):
        """The adapter's experts forward equals HF's grouped_mm forward with F32
        routing weights: each expert's output weighted in F32 and the top-k
        summed before one cast. Bit for bit in BF16; in F32 torch's grouped
        GEMM and torch.mm block their sums differently."""
        from transformers.integrations.moe import grouped_mm_experts_forward
        config = hf_config().text_config
        generator = torch.Generator().manual_seed(4)
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                experts = hf.Glm5NextTextExperts(config).to(dtype)
                with torch.no_grad():
                    for parameter in experts.parameters():
                        parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)
                hidden = torch.randn(40, config.hidden_size, generator=generator).to(dtype)
                index = torch.stack([torch.randperm(config.n_routed_experts, generator=generator)[:2]
                                     for _ in range(40)])
                weights = torch.rand(40, 2, generator=generator) * 1.7
                with torch.no_grad():
                    got = self.adapters[0].grouped_experts(experts, hidden, index, weights)
                    expected = grouped_mm_experts_forward(experts, hidden, index, weights)
                self.assertEqual(got.dtype, dtype)
                if dtype == torch.bfloat16:
                    self.assertTrue(torch.equal(got, expected))
                else:
                    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-4)

    def test_bf16_keeps_hf_f32_tensors(self):
        adapter = self.adapters[0]
        for name, stored, dtype in (("model.language_model.layers.0.self_attn.conv1d.weight", torch.bfloat16,
                                     torch.float32),
                                    ("model.language_model.layers.0.attn_hc.base", torch.float32, torch.float32),
                                    ("model.language_model.layers.0.attn_hc.fn", torch.bfloat16, torch.bfloat16),
                                    ("model.language_model.layers.1.mlp.gate.weight", torch.bfloat16, torch.bfloat16)):
            self.assertEqual(adapter.parameter_dtype(name, stored, torch.bfloat16), dtype, name)
            self.assertEqual(adapter.parameter_dtype(name, stored, torch.float64), torch.float64, name)


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.adapter, _ = run.load_adapter(ADAPTER)

    def test_prefill_rows_shrink_with_the_indexer_scores(self):
        rows = self.adapter.chunk_rows
        self.assertEqual(rows(4096, 32, 4), 512)
        self.assertEqual(rows(131072, 32, 4), 512)
        self.assertEqual(rows(262144, 32, 4), 256)
        self.assertEqual(rows(1 << 20, 32, 4), 64)
        chunks = self.adapter.chunks(300000, 32, 4)
        self.assertEqual(chunks[0], (0, 512))
        self.assertEqual(chunks[-2][1] - chunks[-2][0], 192)
        self.assertEqual(chunks[-1][1], 300000)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(chunks, chunks[1:])))
        self.assertTrue(all((end - start) % 64 == 0 for start, end in chunks[:-1]))

    def test_context_beyond_the_cache_budget_is_refused(self):
        config = configuration_glm5_next.Glm5NextTextConfig()
        plan = self.adapter.Plan(config, ["cuda:0", "cuda:0", "cuda:0", "cuda:1"])
        # Three sparse layers on cuda:0, each caching 512 latent and 128 + 128 + 1
        # indexer values per token, in F32.
        self.assertEqual(self.adapter.cache_values(config), 769)
        most = self.adapter.KV_BUDGET_BYTES // (3 * 769 * 4)
        self.assertGreater(most, 128 * 1024)
        plan.check_context(most, torch.float32)
        with self.assertRaisesRegex(ValueError, f"holds at most {most} tokens"):
            plan.check_context(most + 1, torch.float32)
        plan.check_context(2 * most, torch.bfloat16)

    def test_query_blocks_halve_until_their_keys_fit(self):
        # Eight query rows over 16 keys: rows 0-3 select keys 0-3, rows 4-7 keys 8-15.
        mask = torch.full((1, 1, 8, 16), torch.finfo(torch.float32).min)
        mask[0, 0, :4, :4] = 0
        mask[0, 0, 4:, 8:] = 0
        with mock.patch.multiple(self.adapter, QUERY_ROWS=8, EXPAND_BYTES=8, ATTENTION_SCORE_BYTES=1 << 20):
            blocks = self.adapter.query_blocks(mask, 1, 1, 4)
        self.assertEqual([(start, stop, columns.tolist()) for start, stop, columns in blocks],
                         [(0, 4, [0, 1, 2, 3]), (4, 8, list(range(8, 16)))])
        mask[0, 0, 5] = torch.finfo(torch.float32).min
        with self.assertRaisesRegex(ValueError, "selects no key"):
            self.adapter.query_blocks(mask, 1, 1, 4)

    def test_video_is_refused(self):
        from tools.refrun import workload
        tensor = workload.Tensor("f32", (1, 24), bytes(96))
        sequence = mock.Mock(media=[workload.Media("video", tensor, None)])
        sequence.name = "clip"
        with self.assertRaisesRegex(ValueError, "does not take video"):
            self.adapter.media_inputs(sequence, "cpu", torch.float32)


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class TieTests(unittest.TestCase):
    """The adapter's indexer picks tied pools by the lower index, so chunked
    and unchunked prefill select the same keys where HF's torch.topk order
    of equal scores may not."""

    def setUp(self):
        self.adapter, _ = run.load_adapter(ADAPTER)

    def indexer(self, tie):
        """A one-layer, two-head indexer whose scores all tie: at zero (a zero
        query) or at one positive value (every key, so every pool key, equal)."""
        config = configuration_glm5_next.Glm5NextTextConfig(
            hidden_size=32, q_lora_rank=16, index_n_heads=2, index_head_dim=8, index_topk=8, index_kpool=4,
            num_hidden_layers=1, layer_types=["deepseek_sparse_attention"], mlp_layer_types=["dense"],
            indexer_types=["full"])
        torch.manual_seed(0)
        indexer = hf.Glm5NextTextIndexer(config, 0).double().eval()
        with torch.no_grad():
            if tie == "zero":
                indexer.wq_b.weight.zero_()
            else:
                indexer.wk.weight.zero_()
                indexer.k_norm.bias.fill_(0.5)
                indexer.wq_b.weight.fill_(0.1)
                indexer.weights_proj.weight.fill_(0.2)
        return config, self.adapter.pin_ties(indexer)

    def select(self, config, indexer, hidden, q_resid, chunks):
        from transformers.cache_utils import DynamicCache
        cache, rows = DynamicCache(config=config), []
        for start, stop in chunks:
            # The attention layer caches its keys before the indexer runs.
            cache.update(torch.zeros(1, 1, stop - start, 1, dtype=torch.float64),
                         torch.zeros(1, 1, stop - start, 0, dtype=torch.float64), 0)
            mask = torch.ones(1, stop - start, dtype=torch.bool)
            with torch.no_grad():
                rows.append(indexer(hidden[:, start:stop], q_resid[:, start:stop], mask, cache)[0])
        return [sorted(i for i in row.tolist() if i >= 0) for row in torch.cat(rows)]

    def test_tied_pools_go_to_the_lower_index_in_every_chunking(self):
        for tie in ("zero", "positive"):
            with self.subTest(tie=tie):
                config, indexer = self.indexer(tie)
                generator = torch.Generator().manual_seed(1)
                hidden = torch.randn(1, 16, 32, generator=generator, dtype=torch.float64)
                q_resid = torch.randn(1, 16, 16, generator=generator, dtype=torch.float64).abs()
                whole = self.select(config, indexer, hidden, q_resid, [(0, 16)])
                for chunks in ([(0, 12), (12, 16)], [(0, 5), (5, 10), (10, 15), (15, 16)], [(0, 3), (3, 16)]):
                    self.assertEqual(self.select(config, indexer, hidden, q_resid, chunks), whole, chunks)
                # Query 11 completes pool 2, which ties and loses to pools 0 and 1.
                self.assertEqual(whole[11], list(range(8)))
                # Query 13 keeps its incomplete tail beside the two lower pools.
                self.assertEqual(whole[13], [*range(8), 12, 13])
                # Below two complete pools every visible key is selected.
                self.assertEqual(whole[6], list(range(7)))

    def test_two_head_model_prefills_alike_in_any_chunking(self):
        """HF's model with two indexer heads, where relu ties pool scores at
        zero: with the pinned ties, chunked prefill selects the same keys and
        gives the same logits as one pass."""
        from transformers.cache_utils import DynamicCache
        config = hf_config()
        config.text_config.index_n_heads = 2
        torch.manual_seed(3)
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            model = hf.Glm5NextForConditionalGeneration(config).eval()
        finally:
            torch.set_default_dtype(default)
        for module in model.modules():
            owner = getattr(module, "config", None)
            if isinstance(owner, configuration_glm5_next.Glm5NextTextConfig):
                owner._attn_implementation, owner._experts_implementation = "eager", "eager"
        indexers = [block.self_attn.indexer for block in model.model.language_model.layers
                    if getattr(block.self_attn, "indexer", None) is not None]
        for indexer in indexers:
            self.adapter.pin_ties(indexer)
        ties, pinned = [], self.adapter.lower_pools

        def spy(scores, select_k):
            ordered = torch.sort(scores, dim=-1, descending=True).values
            if select_k < scores.shape[-1]:
                boundary = ordered[..., select_k - 1] == ordered[..., select_k]
                ties.append(bool((boundary & (ordered[..., select_k] > torch.finfo(scores.dtype).min)).any()))
            return pinned(scores, select_k)
        selections = {}
        for indexer in indexers:
            indexer.register_forward_hook(lambda module, _args, out: selections.setdefault(
                module.layer_idx, []).append(out[0]))
        ids = torch.tensor([PROMPT])

        def prefill(size):
            selections.clear()
            cache, hidden = DynamicCache(config=config.text_config), []
            embeds = model.model.get_input_embeddings()(ids)
            with torch.no_grad(), mock.patch.object(self.adapter, "lower_pools", side_effect=spy):
                for start in range(0, ids.shape[1], size):
                    out = model.model.language_model(inputs_embeds=embeds[:, start:start + size],
                                                     past_key_values=cache, use_cache=True)
                    hidden.append(out.last_hidden_state)
                logits = model.lm_head(torch.cat(hidden, dim=1))[0]
            return logits, {layer: [sorted(i for i in row.tolist() if i >= 0) for row in torch.cat(rows)]
                            for layer, rows in selections.items()}
        whole, chosen = prefill(len(PROMPT))
        self.assertTrue(any(ties), "the two-head case must tie pool scores at the selection boundary")
        for size in (3, 4, 5, 6):
            logits, picked = prefill(size)
            self.assertEqual(picked, chosen, size)
            self.assertLess((logits - whole).abs().max().item(), 1e-5, size)


CHECKPOINT = os.environ.get("CHOPINFER_GLM5NEXT_CHECKPOINT")


def runner_scores(tokens, shift=0.0):
    """Runner score columns for forced `tokens`: each the top-1, one logit
    above the next of five."""
    columns = {"logprob": [], "top1": [], "margin": [], "topk_ids": [], "topk_logits": [], "topk_logprobs": []}
    for token in tokens:
        ids = [token, *(i for i in (1, 2, 3, 4, 5) if i != token)][:5]
        for column, value in (("logprob", -0.1 - shift), ("top1", token), ("margin", 1.0), ("topk_ids", ids),
                              ("topk_logits", [10.0 - i for i in range(5)]),
                              ("topk_logprobs", [-0.1 - shift - i for i in range(5)])):
            columns[column].append(value)
    return columns


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
@unittest.skipUnless(CHECKPOINT and Path(CHECKPOINT, "processor_config.json").is_file(),
                     "needs CHOPINFER_GLM5NEXT_CHECKPOINT (tokenizer, template, processor)")
class ProducerTests(unittest.TestCase):
    """The probe workload and skeleton producers (tokenizer, template and
    processor files only; no weights) and the media gate's reader."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "models/glm5next"))
        import reference_fixtures
        import reference_workloads
        cls.workloads, cls.fixtures = reference_workloads, reference_fixtures
        cls.directory = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        cls.root = Path(cls.directory.name)
        cls.proc = reference_workloads.processor(CHECKPOINT)
        cls.sequences, _ = reference_workloads.probes(cls.proc, CHECKPOINT, cls.root)
        reference_workloads.write(cls.root / "probes-greedy.json", cls.sequences)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()
        sys.path.remove(str(ROOT / "models/glm5next"))

    def answer(self, text, end=True):
        ids = self.proc.tokenizer.encode(text, add_special_tokens=False)
        return ids + ([self.workloads.STOP_IDS[1]] if end else [])

    def greedy_run(self, answers, model="glm5next", lock=None):
        """A runner f32.json answering every probe of probes-greedy.json."""
        from tools.model_sdk.diagnostic import canonical
        from tools.refrun import keys, workload
        _, records = workload.load(self.root / "probes-greedy.json")
        source = hashlib.sha256(canonical(keys.read_lock(lock or self.workloads.LOCK))).hexdigest()
        entries = [{"name": r.name, "key_sha256": "0" * 64,
                    "key": {"model": model, "checkpoint": {"source_lock_sha256": source, "files_used": []},
                            "sequence": keys.sequence(r)},
                    "scores": {"answer": answers[r.name], **runner_scores(answers[r.name])}} for r in records]
        path = self.root / f"run-{len(list(self.root.glob('run-*')))}" / "f32.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"policy": {"name": "f32"}, "entries": entries}))
        return path

    def test_chat_set_extends_the_teacher_chat_texts(self):
        tok = self.workloads.tokenizer(CHECKPOINT)
        chat = self.workloads.chat_texts(tok, self.proc)
        teacher = [s for s in self.workloads.short_texts(tok, self.proc) if s["category"] == "chat"]
        self.assertEqual(chat[:len(teacher)], teacher)
        extended = chat[len(teacher):]
        self.assertEqual([s["name"] for s in extended], list(self.workloads.CHATS_EXTENDED))
        self.assertEqual(len({s["name"] for s in chat}), len(chat))
        for sequence in extended:
            self.assertEqual((sequence["category"], sequence["score_from"]), ("chat", 1))
            self.assertTrue(150 <= len(sequence["token_ids"]) <= 604, sequence["name"])

    def test_served_continuation_keeps_reasoning_and_strips_content(self):
        name, prompt = self.sequences[0]["name"], self.sequences[0]["token_ids"]
        text, forced = self.workloads.continuation(self.proc, name, prompt,
                                                   self.answer("I see a red square.</think> A red square.  "))
        self.assertEqual(text, "I see a red square.</think>A red square.")
        self.assertEqual(self.proc.tokenizer.decode(forced, skip_special_tokens=False), text)
        for answer in (self.answer("I see a red square, and"), [self.workloads.STOP_IDS[1]],
                       self.answer("a</think>b</think>c")):
            with self.assertRaisesRegex(ValueError, "does not close its reasoning"):
                self.workloads.continuation(self.proc, name, prompt, answer)

    def test_greedy_runs_must_be_this_workloads(self):
        answers = {s["name"]: self.answer("Looking.</think>A gradient.") for s in self.sequences}
        self.workloads.greedy_entries(self.root, self.greedy_run(answers))
        with self.assertRaisesRegex(ValueError, "not glm5next"):
            self.workloads.greedy_entries(self.root, self.greedy_run(answers, model="qwen38"))
        with self.assertRaisesRegex(ValueError, "another checkpoint lock"):
            self.workloads.greedy_entries(self.root, self.greedy_run(
                answers, lock=ROOT / "models/qwen38/checkpoint-files.json"))
        run = self.greedy_run(answers)
        rows = self.root / f"{self.sequences[1]['name']}.f32"
        original = rows.read_bytes()
        try:
            # Same shape, other pixels: the run scored another sequence.
            rows.write_bytes(bytes(len(original)))
            with self.assertRaisesRegex(ValueError, "another sequence"):
                self.workloads.greedy_entries(self.root, run)
        finally:
            rows.write_bytes(original)

    def test_skeleton_reads_as_image_probe_references(self):
        """probes.json, the skeleton filled as emit fills it, and the capture
        read through tools/qual/multimodal.py's image-probe reader."""
        sys.path.insert(0, str(ROOT / "tools/qual"))
        import multimodal
        import probe_images
        answers = {s["name"]: self.answer(f"The image {s['name']} shows bands.</think> A gradient with an edge.")
                   for s in self.sequences}
        run = self.greedy_run(answers)
        workload = self.root / "probes.json"
        workload.unlink(missing_ok=True)
        self.workloads.write(workload, self.workloads.answered(self.root, run, CHECKPOINT))
        skeleton = self.fixtures.probes(workload, run, CHECKPOINT, self.proc)
        for item in skeleton["probes"].values():
            self.assertEqual(item["kind"], "free-text")
            for variant, shift in (("exact", 0.0), ("bf16", 0.05), ("f64", 0.001)):
                item[variant] = runner_scores(item["answer_token_ids"], shift)
        path = self.root / "references.json"
        path.write_text(json.dumps(skeleton))
        policy = {"format": "image-probes", "reference": "media", "fixture_images": {}, "expected": {},
                  "items": {s["name"]: "free-text" for s in self.sequences}, "stop_token": "<|user|>",
                  "thinking": True}
        config = {"qualification": {"references": {"media": {"path": str(path), "schema": skeleton["schema"]}},
                                    "media": policy}}
        with mock.patch.object(multimodal, "qualification_config", return_value=config), \
                mock.patch.object(multimodal, "media_config", return_value=config), \
                mock.patch.object(multimodal, "input_path", return_value=Path(probe_images.__file__)):
            probes, hashes, digest = multimodal.load_probes(self.root / "capture/probes.json")
            references, rule, _ = multimodal.probe_references(digest, path)
            images = multimodal.probe_images(self.root / "images", self.root, hashes)
        self.assertEqual(rule, self.fixtures.RULE)
        self.assertEqual(sorted(references), sorted(policy["items"]))
        for probe in probes:
            reference = references[probe["name"]]
            item = skeleton["probes"][probe["name"]]
            self.assertEqual(probe["token_ids"], item["prompt_token_ids"])
            self.assertEqual(reference["scored"], len(item["answer_token_ids"]))
            self.assertIn("limits", reference)
            self.assertTrue(images[probe["name"]].startswith(b"\x89PNG"))
        # A served chat renders the reference text as the forced IDs.
        name = probes[0]["name"]
        text = references[name]["answer"]["text"]
        rendered = self.proc.apply_chat_template(
            self.workloads.probe_messages(name) + [{"role": "assistant", "content": text}],
            continue_final_message=True, tokenize=False)
        served, _ = self.workloads.processed(self.proc, name, rendered)
        self.assertEqual(served, probes[0]["token_ids"] + [row[0] for row in references[name]["forced"]])
        with self.assertRaisesRegex(ValueError, "measured on capture"):
            with mock.patch.object(multimodal, "qualification_config", return_value=config), \
                    mock.patch.object(multimodal, "media_config", return_value=config):
                multimodal.probe_references("0" * 64, path)


class SkeletonTests(unittest.TestCase):
    def test_long_context_skeleton_names_only_its_run_variants(self):
        """A pinned reference's provenance must name exactly its declared
        variants; the long-context reference has no f64 run."""
        sys.path.insert(0, str(ROOT / "models/glm5next"))
        self.addCleanup(sys.path.remove, str(ROOT / "models/glm5next"))
        import reference_fixtures
        with tempfile.TemporaryDirectory() as directory:
            workload = Path(directory) / "long-context.json"
            workload.write_text(json.dumps({"sequences": [
                {"name": "long_4k", "category": "long-context", "token_ids": [1, 2, 3], "score_from": 1}]}))
            long = reference_fixtures.texts(workload, reference_fixtures.LONG_CONTEXT_VARIANTS)
            teacher = reference_fixtures.texts(workload)
        self.assertEqual(list(long["provenance"]["variants"]), ["exact", "bf16"])
        self.assertEqual(list(teacher["provenance"]["variants"]), ["exact", "bf16", "f64"])

    def test_frozen_teacher_texts_are_the_teacher_reference_texts(self):
        """The teacher gate scores exactly the frozen texts (chopinfer eval run
        reads them from fixture_root): the shipped file is the producer's
        output over the pinned workload, naming the reference's texts in order."""
        sys.path.insert(0, str(ROOT / "models/glm5next"))
        self.addCleanup(sys.path.remove, str(ROOT / "models/glm5next"))
        import reference_fixtures
        shipped = json.loads((ROOT / "fixtures/glm5next/gate/teacher-texts.json").read_text())
        produced = reference_fixtures.frozen_texts(ROOT / "fixtures/glm5next/teacher-reference.workload.json")
        self.assertEqual(shipped, produced)
        reference = json.loads((ROOT / "fixtures/glm5next/teacher-reference.json").read_text())
        self.assertEqual([text["name"] for text in shipped["texts"]], [text["name"] for text in reference["texts"]])
        self.assertEqual(
            [text["prompt_token_ids_sha256"] for text in shipped["texts"]],
            [hashlib.sha256(json.dumps(text["prompt_token_ids"]).encode()).hexdigest() for text in reference["texts"]])


@unittest.skipIf(torch is None, "needs torch and transformers with glm5_next")
class TapDeclarationTests(unittest.TestCase):
    """models/glm5next/taps.json against the adapter's taps on the pinned
    config's model (built on the meta device, nothing computed): every tap
    the reference writes is declared, every declared entry is tapped (the
    vision blocks' inputs, which only the engine writes, aside), and each
    entry's width is its module's."""

    def test_the_declaration_covers_the_reference_taps(self):
        from tools.model_sdk.taps import Declaration
        declaration = Declaration(ROOT / "models/glm5next/taps.json")
        self.assertEqual(declaration.model, "glm5next")
        config = configuration_glm5_next.Glm5NextConfig.from_dict(
            json.loads((ROOT / "models/glm5next/contract/fixtures/config.json").read_text()))
        with torch.device("meta"):
            model = hf.Glm5NextForConditionalGeneration(config)
        module, _ = run.load_adapter(ADAPTER)
        taps = module.taps(model)
        engine_only = re.compile(r"enc\.vision\.B\d{2}\.in")
        matched = set()
        for name in taps:
            entry = declaration.entry(name)
            self.assertIsNotNone(entry, name)
            matched.add(id(entry))
        for _, _, entry in declaration.entries:
            if id(entry) not in matched:
                self.assertRegex(entry["name"].replace("{nn}", "00"), engine_only)

        def cols(name):
            return declaration.entry(name)["cols"]
        text = config.text_config
        hidden, streams = text.hidden_size, text.hidden_size * text.hc_mult
        layers = model.model.language_model.layers
        kda = next(i for i, layer in enumerate(layers) if layer.block_type == "linear_attention")
        mla = next(i for i, layer in enumerate(layers) if layer.block_type != "linear_attention")
        sparse = text.mlp_layer_types.index("sparse")
        for name in ("embed.out", f"L{kda:02d}.in.streams", f"L{mla:02d}.out"):
            self.assertEqual(cols(name), streams, name)
        for name in ("final.norm", "final.x.hc.head", f"L{kda:02d}.attn.in", f"L{mla:02d}.attn.out",
                     f"L{mla:02d}.mlp.in", f"L{sparse:02d}.mlp.out", f"L{mla:02d}.x.hc.attn"):
            self.assertEqual(cols(name), hidden, name)
        attn = layers[kda].self_attn
        self.assertEqual(cols(f"L{kda:02d}.attn.q"), attn.q_proj.out_features)
        self.assertEqual(cols(f"L{kda:02d}.attn.core"), attn.o_proj.in_features)
        self.assertEqual(cols(f"L{kda:02d}.x.kda.beta"), attn.b_proj.out_features)
        attn = layers[mla].self_attn
        self.assertEqual(cols(f"L{mla:02d}.attn.q"), attn.q_b_proj.out_features)
        self.assertEqual(cols(f"L{mla:02d}.attn.core"), attn.o_proj.in_features)
        self.assertEqual(cols(f"L{mla:02d}.x.latent"), attn.kv_a_layernorm.weight.shape[0])
        self.assertEqual(cols(f"L{mla:02d}.x.q_latent"), attn.q_a_layernorm.weight.shape[0])
        self.assertEqual(cols(f"L{mla:02d}.x.indexer.k"), attn.indexer.k_norm.weight.shape[0])
        self.assertEqual(cols(f"L{mla:02d}.x.indexer.ids"), text.index_topk + text.index_kpool - 1)
        self.assertEqual(cols(f"L{sparse:02d}.router.logits"), text.n_routed_experts)
        self.assertEqual(cols(f"L{sparse:02d}.router.ids"), text.num_experts_per_tok)
        vision = json.loads((ROOT / "models/glm5next/contract/encoders/glm5next-vision.json").read_text())
        self.assertEqual(cols("enc.vision.out"), vision["output"]["hidden"])
        tower = next(stage for stage in vision["stages"] if stage["op"] == "transformer.pre-norm.v1")
        self.assertEqual(cols("enc.vision.B23.out"), tower["width"])


class GateDeclarationTests(unittest.TestCase):
    def test_verify_item_widths_include_each_reachable_draft_depth(self):
        import declaration
        base = json.loads((ROOT / "models/glm5next/qual.json").read_text())
        for name in [None, *base["configurations"]]:
            value = base if name is None else json.loads(declaration.materialized("glm5next", name).read_text())
            if "verify_exact" not in value["cohorts"]["full"]["checks"]:
                continue
            words = [] if name is None else base["configurations"][name].get("serve", [])
            depth = int(words[words.index("--speculative-tokens") + 1]) if "--speculative-tokens" in words else 3
            # compiler package.rs declares every item width through K+1.
            self.assertEqual(value["qualification"]["verify_buckets"], list(range(2, depth + 2)), name)

    PROFILE = ("server", "health", "serving", "profile")

    def setUp(self):
        sys.path.insert(0, str(ROOT / "tools/qual"))
        self.addCleanup(sys.path.remove, str(ROOT / "tools/qual"))

    def receipt(self, size, kv="fp8"):
        """A receipt's server at TP4, `kv` KV and context-parallel size `size`
        (the field the GLM bundle's topology reports, 1 included)."""
        return {"server": {"model": "GLM-5.3-Flash", "health": {"serving": {"profile": {
            "name": "tp4-mtp", "tensor_parallel_size": 4, "kv_cache_dtype": kv, "context_parallel_size": size}}}}}

    def holds(self, rules, size, kv="fp8"):
        import kinds
        # The profile's own fields; determinism's arms also bind their speculation.
        profile = [rule for rule in rules if tuple(rule["path"][:4]) == self.PROFILE and len(rule["path"]) == 5]
        self.assertTrue(profile)
        return all(kinds.predicate(self.receipt(size, kv), rule) for rule in profile)

    def test_each_check_admits_only_its_own_context_parallel_size(self):
        """qual.json's checks and records admit DCP 1 FP8 KV evidence only,
        and each serving configuration's only its own size and KV element."""
        bound = 0
        for variant, size, kv in ((None, 1, "fp8"), ("dcp2", 2, "fp8"), ("dcp4", 4, "fp8"), ("k5", 1, "fp8"),
                                  ("kv-bf16", 1, "bf16")):
            name = variant or "default"
            # qual.json's own gate: the checks its cohorts run (it also
            # declares the checks only its configurations run).
            checks = gate_checks(default() if variant is None else configured(variant))
            for check_id, check in checks.items():
                for key, rules in [("rules", check["rules"]), *check.get("input_rules", {}).items()]:
                    if not any(tuple(rule["path"][:4]) == self.PROFILE for rule in rules):
                        continue
                    bound += 1
                    # teacher_arms' input is the FP8 arm qual.json's run recorded.
                    expected = (1, "fp8") if key == "fp8_teacher" else (size, kv)
                    for other in (1, 2, 4):
                        for element in ("fp8", "bf16"):
                            with self.subTest(declaration=name, check=check_id, rules=key, size=other, kv=element):
                                self.assertEqual(self.holds(rules, other, element), (other, element) == expected)
            for check_id in ("teacher", "chat", "long_context", "multimodal", "spec_verify", "verify_exact",
                             "prefix_exact", "preemption_exact", "determinism"):
                if check_id in checks:
                    check = checks[check_id]
                    self.assertTrue(any(tuple(rule["path"][:4]) == self.PROFILE for rules in
                                        [check["rules"], *check.get("input_rules", {}).values()] for rule in rules),
                                    check_id)
            self.assertFalse([check_id for check_id in checks if "_dcp" in check_id or "kv_bf16" in check_id])
        # prefix_exact reads one record per shard (two, the kv-bf16
        # configuration's included).
        self.assertEqual(bound, 56)

    def test_live_checks_and_recorded_fp8_arms_admit_only_drafting_tp4_mtp(self):
        """A live check, and an FP8 arm a later run compares against, must
        come from the tp4-mtp profile with drafting on: tp4-ar, or tp4-mtp
        with its drafts off, is refused. The AR comparison records keep
        their own rules."""
        import kinds
        profile = list(self.PROFILE)

        def server(name, drafts, kv="fp8"):
            return {"server": {"model": "GLM-5.3-Flash", "health": {"serving": {"profile": {
                "name": name, "tensor_parallel_size": 4, "kv_cache_dtype": kv, "context_parallel_size": 1,
                "speculation": {"tokens": drafts}}}}}}

        groups = []
        for name, kv in (("default", "fp8"), ("kv-bf16", "bf16")):
            checks = gate_checks(default() if name == "default" else configured(name))
            for check_id, check in checks.items():
                rules = [rule for rule in check["rules"] if rule["path"][:4] == profile]
                if rules:
                    groups.append((f"{name}:{check_id}", rules, kv))
        checks = configured("kv-bf16")["checks"]
        groups.append(("kv-bf16:fp8_teacher", [rule for rule in checks["teacher_arms"]["input_rules"]["fp8_teacher"]
                                               if rule["path"][:4] == profile], "fp8"))
        self.assertEqual(len(groups), 8)
        for label, rules, kv in groups:
            with self.subTest(rules=label):
                self.assertTrue(all(kinds.predicate(server("tp4-mtp", 3, kv), rule) for rule in rules))
                for refused in (server("tp4-ar", 0, kv), server("tp4-mtp", 0, kv)):
                    self.assertFalse(all(kinds.predicate(refused, rule) for rule in rules))

    def test_the_preview_runs_the_quick_cohort_on_the_drafting_arm_alone(self):
        """The quick cohort's live checks admit only a drafting server, so the
        release preview's AR arm runs no quick stage (preview.quick_arms)."""
        value = json.loads((ROOT / "models/glm5next/qual.json").read_text())
        self.assertEqual(value["preview"]["quick_arms"], ["spec"])
        for check_id in value["cohorts"]["quick"]["checks"]:
            drafting = [rule for rule in value["checks"][check_id]["rules"]
                        if rule["path"] == [*self.PROFILE, "speculation", "tokens"]]
            if drafting:
                self.assertEqual((drafting[0]["op"], drafting[0]["value"]), ("ge", 1), check_id)
        self.assertTrue(any(rule["path"] == [*self.PROFILE, "speculation", "tokens"]
                            for rule in value["checks"]["teacher"]["rules"]))

    def test_determinism_scores_the_mtp_record_against_the_ar_record(self):
        """Each size's determinism check is gated: it reads the MTP record and
        scores it against the AR record, and each record must come from a
        server with that speculation (K=5's MTP record exactly five drafts)."""
        tokens = list(self.PROFILE) + ["speculation", "tokens"]
        for name, drafts in (("default", ("ge", 1)), ("dcp2", ("ge", 1)), ("dcp4", ("ge", 1)), ("k5", ("eq", 5))):
            with self.subTest(declaration=name):
                check = (default() if name == "default" else configured(name))["checks"]["determinism"]
                self.assertNotIn("informational_reason", check)
                ar, spec = "determinism_ar", "determinism_spec"
                self.assertEqual((check["json_file"], check["reference_file"]), ("{%s}" % spec, "{%s}" % ar))
                self.assertEqual([(rule["op"], rule["value"]) for rule in check["input_rules"][ar]
                                  if rule["path"] == tokens], [("eq", 0)])
                self.assertEqual([(rule["op"], rule["value"]) for rule in check["input_rules"][spec]
                                  if rule["path"] == tokens], [drafts])

    def test_the_k5_declaration_binds_five_drafts_on_text(self):
        """The k5 configuration gates tp4-mtp-k5: its teacher and determinism's MTP
        record admit only a server drafting five tokens (not tp4-mtp's three),
        the AR record only one drafting none, and no rule asks for the image
        request a text package cannot serve."""
        import kinds
        value = configured("k5")
        self.assertEqual(value["variant"], "k5")
        self.assertEqual(value["cohorts"]["full"]["checks"],
                         ["teacher", "chat", "spec_verify", "verify_exact", "preemption_exact", "determinism"])
        self.assertEqual((value["capabilities"]["speculation"], value["capabilities"]["media"]), ("mtp", []))
        checks = value["checks"]

        def server(drafts):
            return {"server": {"model": "GLM-5.3-Flash", "health": {"serving": {"profile": {
                "name": "tp4-mtp", "tensor_parallel_size": 4, "kv_cache_dtype": "fp8", "context_parallel_size": 1,
                "speculation": {"tokens": drafts}}}}}}

        def admits(rules, drafts):
            return all(kinds.predicate(server(drafts), rule) for rule in rules if rule["path"][:4] == list(self.PROFILE))

        groups = {"teacher": (checks["teacher"]["rules"], 5),
                  "determinism_spec": (checks["determinism"]["input_rules"]["determinism_spec"], 5),
                  "determinism_ar": (checks["determinism"]["input_rules"]["determinism_ar"], 0)}
        for label, (rules, drafts) in groups.items():
            for other in (0, 3, 5):
                with self.subTest(rules=label, drafts=other):
                    self.assertEqual(admits(rules, other), other == drafts)
        self.assertFalse([rule for check in checks.values()
                          for rules in [check["rules"], *check.get("input_rules", {}).values()]
                          for rule in rules if rule["path"] == ["image", "included"]])


    def test_the_bf16_kv_declaration_compares_with_the_recorded_fp8_teacher(self):
        """qual.json's full cohort's teacher writes its capture (the quick
        cohort none, so chopinfer eval captures it); the kv-bf16
        configuration's teacher_arms compares its own arm with that recorded
        FP8 arm."""
        base = json.loads((ROOT / "models/glm5next/qual.json").read_text())
        capture = ["--capture-out", "{raw}/teacher-capture.json"]
        self.assertEqual(base["checks"]["teacher"]["cohort_args"], {"full": capture})
        self.assertNotIn("--capture-out", base["checks"]["teacher"]["args"])
        # Each configuration's full teacher writes its own capture the same way.
        for name in ("dcp2", "dcp4"):
            teacher = configured(name)["checks"]["teacher"]
            self.assertNotIn("--capture-out", teacher["args"])
            self.assertEqual(teacher["cohort_args"], {"full": capture})
        value = configured("kv-bf16")
        self.assertEqual(value["cohorts"]["full"]["checks"], ["teacher", "teacher_arms", "chat", "prefix_exact"])
        self.assertEqual(value["checks"]["teacher"]["cohort_args"], {"full": capture})
        arms = value["checks"]["teacher_arms"]
        # The BF16 arm reads the capture from the teacher's own raw directory.
        self.assertEqual(arms["args"], ["--arm", "bf16={teacher}={check_raw[teacher]}/teacher-capture.json",
                                        "--arm", "fp8={fp8_teacher}={fp8_teacher_capture}"])
        rules = {tuple(rule["path"]): rule["value"] for rule in arms["input_rules"]["fp8_teacher"]}
        self.assertEqual(rules[("reference_sha256",)], base["references"]["teacher"]["sha256"])
        self.assertEqual(rules[("server", "checkpoint", "checkpoint_identity")],
                         base["qualification"]["checkpoint"]["checkpoint_identity"])
        self.assertEqual(rules[(*self.PROFILE, "kv_cache_dtype")], "fp8")
        self.assertEqual(value["capabilities"]["speculation"], "none")
        self.assertEqual(value["capabilities"]["media"], [])


    def test_each_paired_configuration_compares_by_the_default_margins_on_its_own_reference(self):
        """qual.json declares one set of paired margins, tp4-mtp; every
        configuration that declares paired_teacher (default and both DCP
        sizes) selects it by its profile, on its own teacher reference, and compares with its own stored arm. The five-draft, BF16
        KV and taps configurations declare none."""
        import declaration
        import paired_teacher

        base = json.loads((ROOT / "models/glm5next/qual.json").read_text())
        policies = base["checks"]["paired_teacher"]["constants"]["paired_teacher"]["PROFILE_POLICIES"]["value"]
        self.assertEqual(set(policies), {"tp4-mtp"})
        profiles = json.loads((ROOT / "models/glm5next/profiles.json").read_text())["profiles"]
        expected = {"default": "tp4-mtp", "dcp2": "tp4-mtp-dcp2", "dcp4": "tp4-mtp-dcp4"}
        for name in ["default", *sorted(base["configurations"])]:
            value = (base if name == "default" else
                     json.loads(declaration.materialized("glm5next", name).read_text()))
            checks = value["checks"] if isinstance(value["checks"], dict) else {
                check["id"]: check for check in value["checks"]}
            with self.subTest(configuration=name):
                if name not in expected:
                    self.assertNotIn("paired_teacher", checks)
                    continue
                self.assertIn("paired_teacher", value["cohorts"]["full"]["checks"])
                declared = checks["paired_teacher"]["constants"]["paired_teacher"]["PROFILE_POLICIES"]["value"]
                self.assertEqual(set(declared), {"tp4-mtp"})
                references = value.get("references") or value["qualification"]["references"]
                self.assertEqual(declared["tp4-mtp"]["reference_sha256"], references["teacher"]["sha256"])
                profile = profiles[expected[name]]
                live = {"tensor_parallel_size": profile["tensor_parallel"],
                        "context_parallel_size": profile.get("context_parallel_size", 1),
                        "speculation": {"tokens": profile["speculation"]["tokens"],
                                        "drafter": profile["speculation"]["drafter"]}}
                self.assertEqual(paired_teacher.policy_name(live, declared), "tp4-mtp")

    def test_encoder_parity_runs_only_in_the_taps_declaration(self):
        """The default gate runs on the release build: encoder_parity and
        text_layers, whose captures bind to the diagnostic-taps build that
        wrote them, are declared only by the taps configuration (text_layers in its
        full cohort only), and no other declaration can be blocked by them
        or by the BF16 KV report."""
        base = json.loads((ROOT / "models/glm5next/qual.json").read_text())
        # qual.json declares them for its configurations; its own cohorts run none.
        for check in ("encoder_parity", "text_layers", "teacher_arms"):
            self.assertNotIn(check, gate_checks(base))
        for name in ("dcp2", "dcp4", "k5", "kv-bf16"):
            checks = configured(name)["checks"]
            self.assertNotIn("encoder_parity", checks)
            self.assertNotIn("text_layers", checks)
        value = configured("taps")
        self.assertEqual(value["cohorts"]["quick"]["checks"], ["encoder_parity"])
        self.assertEqual(value["cohorts"]["full"]["checks"], ["encoder_parity", "text_layers"])
        self.assertEqual(value["checks"]["encoder_parity"]["args"],
                         ["taps", "--run", "{encoder_parity_run}", "--url", "{url}"])
        self.assertEqual(value["checks"]["text_layers"]["constants"]["text_layers"]["MODES"]["value"],
                         {"f32": "rounded", "bf16": "hf-bf16"})
        self.assertEqual(value["references"], {})
        self.assertFalse(any(value["capabilities"][key] for key in ("text", "api", "prefix_cache", "preemption")))

    def test_the_layer_floor_runs_with_the_served_fp8_latent(self):
        """text_layers measures the served FP8 latent KV against HF's plain
        F32 run and takes its floor from HF's BF16 run with the served FP8
        latent round trip (reference_latent_fp8.py): one runner run per
        mode, `f32/` and `bf16/`. Its limits stay those of every layer
        check."""
        import text_layers
        value = configured("taps")
        constants = value["checks"]["text_layers"]["constants"]["text_layers"]
        self.assertEqual(constants["ADAPTERS"]["value"], {"bf16": "models/glm5next/reference_latent_fp8.py"})
        self.assertEqual(constants["MODES"]["value"], {"f32": "rounded", "bf16": "hf-bf16"})
        self.assertEqual((constants["FLOOR_RATIO"]["value"], constants["RELATIVE_RMS_LIMIT"]["value"]), (3.0, 0.05))
        path = Path(self.enterContext(tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")))) / "taps.json"
        path.write_text(json.dumps(value))
        reference = Path("/reference")
        with mock.patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(path)}):
            self.assertEqual({mode: (text_layers.adapter(mode), text_layers.mode_run(reference, mode))
                              for mode in ("f32", "bf16")},
                             {"f32": ("models/glm5next/reference.py", reference / "f32"),
                              "bf16": ("models/glm5next/reference_latent_fp8.py", reference / "bf16")})
        # The served taps configuration keeps the default FP8 latent KV.
        self.assertNotIn("--kv-cache-dtype", default()["configurations"]["taps"]["serve"])
        self.assertTrue((ROOT / constants["ADAPTERS"]["value"]["bf16"]).is_file())


class DeterminismPairTests(unittest.TestCase):
    """qual.json's determinism check, through the gate's preflight, scorer
    and rescore: the AR and MTP records must be bound to the pinned
    checkpoint and share one run and build."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "tools/qual"))
        self.addCleanup(sys.path.remove, str(ROOT / "tools/qual"))
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def server(self, drafts, byte="0"):
        pin = json.loads((ROOT / "models/glm5next/qual.json").read_text())["qualification"]["checkpoint"][
            "checkpoint_identity"]
        identity = {"checkpoint_identity": pin, "startup_nonce": byte * 32}
        serving = {**identity, "package_identity": "p" * 64, "version": "1.0", "source_commit": "c" * 40,
                   "profile": {"name": "tp4-mtp", "tensor_parallel_size": 4, "kv_cache_dtype": "fp8",
                               "context_parallel_size": 1, "speculation": {"tokens": drafts}}}
        return {"model": "GLM-5.3-Flash", "health": {"serving": serving}, "checkpoint": dict(identity)}

    def judged(self, name, change=None, image=True):
        import checks
        import declaration
        import release_gate as gate
        import test_determinism
        config = declaration.load(ROOT / "models/glm5next/qual.json")
        check = next(check for check in config["checks"] if check["id"] == "determinism")
        ar, spec = (test_determinism.recorded(arm=arm, image=image) for arm in ("ar", "spec"))
        ar["server"], spec["server"] = self.server(0, "a"), self.server(3, "b")
        if change:
            change(ar)
        run = Path(self.directory.name) / name
        (run / "checks").mkdir(parents=True)
        left, right = run / "checks/ar.json", run / "checks/spec.json"
        left.write_text(json.dumps(ar))
        right.write_text(json.dumps(spec))
        policy = run / "policy.json"
        policy.write_text(json.dumps(config))
        context = {"determinism_ar": str(left), "determinism_spec": str(right), "model": "GLM-5.3-Flash",
                   "url": "http://unused", "run_id": "gate", "raw": str(run / "checks"), "run": str(run)}
        try:
            gate.prerequisites(config, ["determinism"], "full", context, {})
        except (ValueError, KeyError) as error:
            # The gate fails the run before any live check.
            return False, str(error)
        rows = []
        with mock.patch.object(gate.subprocess, "run",
                               return_value=subprocess.CompletedProcess([], 0)):
            row = gate.run_check(check, rows, "full", context, run, policy, 10 ** 12, False, None)
        inputs = {key: context[key] for key in ("determinism_ar", "determinism_spec")}
        summary = {"server": {"model": context["model"]}, "inputs": inputs,
                   "input_sha256": {key: gate.digest(Path(path)) for key, path in inputs.items()}}
        rescored = checks.rescore_row(run / "summary.json", row, check, config, policy, False, None,
                                      summary=summary)
        self.assertEqual(rescored["status"] == "pass", row["passed"] is True, (row, rescored))
        return row["passed"] is True, row.get("error") or rescored.get("error")

    def test_a_matched_pair_passes_and_a_mismatched_one_fails(self):
        self.assertEqual(self.judged("matched"), (True, None))
        serving = lambda ar: ar["server"]["health"]["serving"]
        cases = {
            "another run": lambda ar: ar.update(run_id="other"),
            "another checkpoint": lambda ar: (ar["server"]["checkpoint"].update(checkpoint_identity="1" * 64),
                                              serving(ar).update(checkpoint_identity="1" * 64)),
            "unbound": lambda ar: ar["server"].pop("checkpoint"),
            "another build": lambda ar: serving(ar).update(source_commit="d" * 40),
            "another package": lambda ar: serving(ar).update(package_identity="q" * 64),
            "unqualified options": lambda ar: serving(ar).update(model_options={"x": 1},
                                                                 model_options_qualified=False),
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                self.assertFalse(self.judged(name.replace(" ", "-"), change)[0])

    def test_a_stored_run_rescores_against_the_current_input_rules(self):
        """A stored run whose records the current preflight refuses (no
        image request, or another checkpoint than the pin) fails a
        current-policy rescore, though its pair matches and its gate row
        passed."""
        import checks
        import release_gate as gate
        import test_determinism
        pin = json.loads((ROOT / "models/glm5next/qual.json").read_text())["qualification"]["checkpoint"][
            "checkpoint_identity"]
        identities = {"control": (pin, True), "without-image": (pin, False), "foreign-pin": ("1" * 64, True)}
        for name, (identity_value, image) in identities.items():
            with self.subTest(case=name):
                run = Path(self.directory.name) / f"stored-{name}"
                (run / "checks").mkdir(parents=True)
                records = {}
                for arm, drafts in (("ar", 0), ("spec", 3)):
                    value = test_determinism.recorded(arm=arm, image=image)
                    value["server"] = self.server(drafts, arm[0])
                    for identity in (value["server"]["checkpoint"], value["server"]["health"]["serving"]):
                        identity["checkpoint_identity"] = identity_value
                    records[arm] = run / f"checks/{arm}.json"
                    records[arm].write_text(json.dumps(value))
                inputs = {"determinism_ar": str(records["ar"]), "determinism_spec": str(records["spec"])}
                row = {"id": "determinism", "passed": True, "receipt": inputs["determinism_spec"],
                       "sha256": gate.digest(records["spec"]), "reference_file": inputs["determinism_ar"],
                       "reference_sha256": gate.digest(records["ar"]), "log": str(run / "determinism.log")}
                summary = {"model": "glm5next", "profile": "full", "mechanics_only": False, "checks": [row],
                           "server": {"model": "GLM-5.3-Flash"}, "inputs": inputs,
                           "input_sha256": {key: gate.digest(Path(path)) for key, path in inputs.items()}}
                rescored = checks.rescore_gate(run / "summary.json", summary, "glm5next", "current")
                result = next(row for row in rescored if row["id"] == "determinism")
                self.assertEqual(result["status"], "pass" if name == "control" else "fail", result.get("error"))
                if name != "control":
                    self.assertIn("determinism_ar: input predicate failed", result["error"])

    def test_the_image_capable_declaration_requires_the_image_request(self):
        """qual.json claims image input, so both records must include the
        image request; the configurations, which claim no media, do not ask it."""
        self.assertFalse(self.judged("without-image", image=False)[0])
        for name in ("dcp2", "dcp4"):
            check = configured(name)["checks"]["determinism"]
            self.assertFalse([rule for rules in check["input_rules"].values() for rule in rules
                              if rule["path"] == ["image", "included"]], name)


class ExternalVariantTests(unittest.TestCase):
    """Another checkpoint of the model, qualified as an external variant
    (tools/qual/README.md#external-variants): a synthetic one under a
    temporary variant root, its teacher reference the model's re-homed onto
    another checkpoint pin (tools/qual/test_variant.py)."""

    REQUANT = {"repository": "someone/glm-requant", "revision": "a" * 40}

    def setUp(self):
        for path in (ROOT / "tools/qual", ROOT / "models/glm5next"):
            sys.path.insert(0, str(path))
            self.addCleanup(sys.path.remove, str(path))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "variants"

    def variant(self, adopt=True, **overlay):
        from test_variant import selected, synthetic_variant
        synthetic_variant(self.root, "glm5next", overlay=overlay, adopt=adopt, checkpoint=self.REQUANT)
        return selected(self.root)

    def test_each_reference_names_only_its_own_checkpoint(self):
        import common
        import declaration
        with self.variant():
            configs = {variant: declaration.load(declaration.named("glm5next" if variant is None else
                                                                   "glm5next." + variant))
                       for variant in (None, "requant")}
            pins = {variant: common.checkpoint_pin(config) for variant, config in configs.items()}
            self.assertEqual(pins["requant"]["repository"], self.REQUANT["repository"])
            for variant, config in configs.items():
                spec = config["qualification"]["references"]["teacher"]
                path = common.reference_path("teacher", config)
                value = common.pinned_reference("teacher", path, config)
                stated = value["provenance"]["checkpoint"]
                keys = spec.get("checkpoint_keys", common.PINNED_CHECKPOINT_KEYS)
                for other, pin in pins.items():
                    with self.subTest(reference=variant, pin=other):
                        matches = all(stated[key] == pin[key] for key in keys)
                        self.assertEqual(matches, other == variant)
                        if other != variant:
                            # Another declaration's lock and pin refuse it.
                            with self.assertRaises(ValueError):
                                common.pinned_reference("teacher", path, configs[other])

    def test_a_variant_record_names_the_variant_checkpoint_and_the_recorded_fp8_arm_the_fp8_one(self):
        """Rules that name qual.json's checkpoint identity name the
        variant's; the FP8 arm the default configuration's run recorded
        (from_default) keeps the FP8 pin and profile."""
        own = "e" * 64
        fp8 = default()["qualification"]["checkpoint"]["checkpoint_identity"]
        pin = {**default()["qualification"]["checkpoint"], **self.REQUANT, "checkpoint_identity": own,
               "lock": "requant.checkpoint-files.json"}
        with self.variant(adopt=False, checks=["teacher", "evals_candidate", "evals_compare", "determinism"],
                          from_default=["fp8_gsm8k", "fp8_gsm8k_run"], serving={"profile_name": "tp4-mtp-requant"},
                          qualification={"checkpoint": pin}):
            import declaration
            value = json.loads(declaration.named("glm5next.requant").read_text())
        named = {(name, key): rule["value"] for name, check in value["checks"].items()
                 for key, rules in [("rules", check["rules"]), *check.get("input_rules", {}).items()]
                 for rule in rules if rule["path"][-1] == "checkpoint_identity"}
        self.assertEqual({key for key, identity in named.items() if identity == fp8}, {("evals_compare", "fp8_gsm8k")})
        self.assertTrue({("determinism", "determinism_ar"), ("evals_candidate", "rules")}
                        <= {key for key, identity in named.items() if identity == own})
        self.assertEqual(set(named.values()), {own, fp8})
        profile = [*GateDeclarationTests.PROFILE, "name"]
        names = {(name, key): rule["value"] for name, check in value["checks"].items()
                 for key, rules in [("rules", check["rules"]), *check.get("input_rules", {}).items()]
                 for rule in rules if rule["path"] == profile}
        self.assertEqual(names.pop(("evals_compare", "fp8_gsm8k")), "tp4-mtp")
        self.assertEqual(set(names.values()), {"tp4-mtp-requant"})

    def test_the_paired_gsm8k_arms_are_bound_and_compared_bound(self):
        """A variant that selects evals_candidate and evals_compare answers
        the whole GSM8K test split on its own server, bound to its own pin,
        and compares it, bound, with the FP8 arm qual.json's gate recorded
        (evals_gsm8k), bound by its run, workload, profile and pin."""
        import kinds
        fp8 = default()
        recorded = fp8["checks"]["evals_gsm8k"]
        self.assertEqual(recorded["args"][recorded["args"].index("--arm") + 1], "fp8")
        self.assertEqual(recorded["args"][-2:], ["--server-log", "{server_log}"])
        self.assertIn("evals_gsm8k", fp8["cohorts"]["full"]["checks"])
        workload = next(rule["value"] for rule in recorded["rules"] if rule["path"] == ["workload_sha256"])
        with self.variant(adopt=False, checks=["teacher", "evals_candidate", "evals_compare"],
                          from_default=["fp8_gsm8k", "fp8_gsm8k_run"]):
            import declaration
            value = json.loads(declaration.named("glm5next.requant").read_text())
        checks = value["checks"]
        candidate, compare = checks["evals_candidate"], checks["evals_compare"]
        self.assertEqual([candidate["args"][candidate["args"].index(flag) + 1] for flag in ("--url", "--arm", "--run-id")],
                         ["{url}", "candidate", "{run_id}"])
        self.assertEqual(candidate["args"][-2:], ["--server-log", "{server_log}"])
        self.assertEqual(compare["args"][1:], ["--arms", "candidate", "fp8", "--receipt", "candidate={evals_candidate}",
                                               "--receipt", "fp8={fp8_gsm8k}", "--bound"])
        self.assertIn({"path": ["bound"], "op": "eq", "value": True, "source": "tools/qual/README.md#release-gates"},
                      compare["rules"])
        pin = value["qualification"]["checkpoint"]["checkpoint_identity"]
        self.assertEqual([(rule["path"], rule["value"]) for rule in candidate["rules"]
                          if rule["path"][-1] == "checkpoint_identity"],
                         [(["server", "checkpoint", "checkpoint_identity"], pin)])
        rules = {tuple(rule["path"]): rule["value"] for rule in compare["input_rules"]["fp8_gsm8k"]}
        self.assertEqual(rules[("run_id",)], "{fp8_gsm8k_run}")
        self.assertEqual(rules[("workload_sha256",)], workload)
        self.assertEqual(rules[("server", "checkpoint", "checkpoint_identity")],
                         fp8["qualification"]["checkpoint"]["checkpoint_identity"])
        self.assertEqual(rules[(*GateDeclarationTests.PROFILE, "name")], "tp4-mtp")
        # An unbound arm (no server.checkpoint) fails the FP8 arm's rules.
        identity = [rule for rule in compare["input_rules"]["fp8_gsm8k"] if rule["path"][-1] == "checkpoint_identity"]
        with self.assertRaises(KeyError):
            kinds.predicate({"server": {"health": {"serving": {"checkpoint_identity": "0" * 64}}}}, identity[0])

    def test_a_variant_receipt_and_gate_run_rescore_under_the_variant(self):
        """`tools.qual rescore --model glm5next.requant --variant-dir DIR`
        judges a receipt and a gate run by the variant's checks (qual.json's,
        expanded), and refuses a run of the base declaration."""
        import declaration
        with self.variant(checks=["teacher", "determinism"]):
            variant = declaration.load(declaration.named("glm5next.requant"))
        base = declaration.load(declaration.path("glm5next", ROOT))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "determinism.json"
            receipt.write_text(json.dumps({"schema": "chopinfer-qualification/v1", "check": "determinism",
                                           "passed": True}))
            for name, config in (("variant", variant), ("base", base)):
                run = root / name
                run.mkdir()
                (run / "summary.json").write_text(json.dumps({
                    "schema": "chopinfer-release-receipt/v1", "model": "glm5next", "profile": "full",
                    "mechanics_only": True, "config": config, "checks": [], "passed": False}))

            def rescore(*paths):
                report = root / "report.json"
                result = subprocess.run([sys.executable, "-B", "-m", "tools.qual", "rescore", "--model",
                                         "glm5next.requant", "--variant-dir", str(self.root), "--json", str(report),
                                         *map(str, paths)], cwd=ROOT, capture_output=True, text=True, timeout=300)
                return result, json.loads(report.read_text()) if report.is_file() else None

            result, report = rescore(receipt)
            self.assertIsNotNone(report, result.stderr)
            # A lone determinism record has no other arm to be scored against.
            self.assertEqual([(row["id"], row["status"]) for row in report["checks"]],
                             [("determinism", "insufficient")])
            result, report = rescore(root / "variant")
            self.assertIsNotNone(report, result.stderr)
            self.assertEqual([row["id"] for row in report["checks"]], variant["profiles"]["full"]["checks"])
            result, _ = rescore(root / "base")
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn("a gate run of glm5next, not glm5next.requant", result.stderr)

    def test_verify_exact_names_the_variant_gate(self):
        """`verify_exact.py --exact glm5next.requant` reads the variant's exact input."""
        import verify_exact
        with self.variant():
            self.assertEqual(verify_exact.exact_path("glm5next.requant"),
                             self.root.resolve() / "glm5next/requant.teacher-reference.json")
        self.assertEqual(verify_exact.exact_path("glm5next"), ROOT / "fixtures/glm5next/teacher-reference.json")


class Nvfp4SkeletonTests(unittest.TestCase):
    """reference_fixtures.py nvfp4-teacher: an NVFP4 checkpoint's teacher
    skeleton names the pin it is given and describes its variants from the
    checkpoint's layout, whatever the variant is called."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "models/glm5next"))
        self.addCleanup(sys.path.remove, str(ROOT / "models/glm5next"))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def checkpoint(self, name, parts):
        """A header-only checkpoint of three text layers and an MTP layer,
        whose `parts` (prefixes under layers.N.) are NVFP4, the rest BF16."""
        directory = self.root / name
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 3}}))
        header = {}
        for layer in range(4):
            for module in ("mlp.gate_proj", "mlp.experts.0.up_proj", "mlp.shared_experts.down_proj",
                           "self_attn.q_a_proj"):
                base = f"model.language_model.layers.{layer}.{module}."
                if layer == 3 or not any(module.startswith(part) for part in parts):
                    header[base + "weight"] = {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}
                    continue
                header[base + "weight"] = {"dtype": "U8", "shape": [2, 1], "data_offsets": [0, 2]}
                header[base + "weight_scale"] = {"dtype": "F8_E4M3", "shape": [2, 1], "data_offsets": [0, 2]}
                header[base + "weight_scale_2"] = {"dtype": "F32", "shape": [], "data_offsets": [0, 4]}
        data = json.dumps(header).encode()
        (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(data)) + data + bytes(8))
        return directory

    def test_the_variants_are_described_from_the_layout(self):
        import reference_fixtures
        routed = self.checkpoint("routed", ["mlp.experts."])
        dense = self.checkpoint("dense", ["mlp.experts.", "mlp.gate_proj"])
        self.assertEqual(reference_fixtures.nvfp4_parts(routed), ["routed experts"])
        self.assertEqual(reference_fixtures.nvfp4_parts(dense), ["dense MLPs", "routed experts"])
        self.assertIn("routed experts' E2M1 codes", reference_fixtures.nvfp4_variants(routed)["exact"])
        self.assertIn("dense MLPs and routed experts' E2M1 codes", reference_fixtures.nvfp4_variants(dense)["exact"])
        self.assertEqual(list(reference_fixtures.nvfp4_variants(dense)), ["exact", "bf16"])
        # Only the MTP layer (not loaded) is NVFP4: no NVFP4 text layer.
        mtp = self.checkpoint("mtp", [])
        with self.assertRaisesRegex(ValueError, "stores no projection of its text layers as NVFP4"):
            reference_fixtures.nvfp4_variants(mtp)

    def test_the_skeleton_names_the_given_pin(self):
        import reference_fixtures
        checkpoint = self.checkpoint("routed", ["mlp.experts."])
        pin = {"lock": "any.checkpoint-files.json", "repository": "someone/glm-requant", "revision": "a" * 40,
               "served_model_name": "GLM-5.3-Flash", "source_lock_sha256": "1" * 64, "checkpoint_identity": "2" * 64}
        (self.root / "pin.json").write_text(json.dumps(pin))
        output = self.root / "skeleton.json"
        workload = ROOT / "fixtures/glm5next/teacher-reference.workload.json"
        with mock.patch.object(sys, "argv", ["reference_fixtures.py", "nvfp4-teacher", "--checkpoint", str(checkpoint),
                                             "--pin", str(self.root / "pin.json"), "--workload", str(workload),
                                             "--output", str(output)]):
            reference_fixtures.main()
        value = json.loads(output.read_text())
        self.assertEqual(value["provenance"]["checkpoint"], {key: pin[key] for key in pin if key != "lock"})
        self.assertEqual(value["provenance"]["variants"], reference_fixtures.nvfp4_variants(checkpoint))
        self.assertEqual([text["name"] for text in value["texts"]],
                         [sequence["name"] for sequence in json.loads(workload.read_text())["sequences"]])


def parity_run():
    """models/glm5next/encoder_parity_run.py, imported."""
    sys.path.insert(0, str(ROOT / "models/glm5next"))
    import encoder_parity_run
    return encoder_parity_run


class EncoderParityRunTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(ROOT / "tools/qual"))
        self.addCleanup(sys.path.remove, str(ROOT / "tools/qual"))
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_requests_send_each_probe_image_as_captured(self):
        """One body per probe: its captured images (by digest) and question,
        the gate's thinking switch and one output token."""
        _, multimodal = parity_run().tools()
        probes, hashes, _ = multimodal.load_probes()
        names = parity_run().write_requests(self.root / "requests", self.root / "images")
        self.assertEqual(names, sorted(probe["name"] for probe in probes))
        for probe in probes:
            body = json.loads((self.root / "requests" / f"{probe['name']}.json").read_text())
            self.assertEqual((body["max_tokens"], body["temperature"]), (1, 0))
            parts = body["messages"][0]["content"]
            images = [base64.b64decode(part["image_url"]["url"].split(",", 1)[1]) for part in parts[:-1]]
            self.assertEqual([hashlib.sha256(image).hexdigest() for image in images],
                             [hashes[name] for name in probe["images"]])
            self.assertEqual(parts[-1], {"type": "text", "text": probe["text"]})

    DECLARATION = "models/glm5next/contract/encoders/glm5next-vision.json"

    def reference(self, label, workload, modes):
        """A reference run of tools/refrun/encoder.py in `modes`, as it lays
        them out: run.json, <mode>.json and taps/<mode>/stepNNNN."""
        declared = ROOT / self.DECLARATION
        precision = json.loads(declared.read_text())["precision"]
        directory = self.root / label
        directory.mkdir()
        taps = ["enc.vision.out"]
        (directory / "run.json").write_text(json.dumps({
            "argv": ["encoder.py", "--workload", str(workload)],
            "declaration": {"path": self.DECLARATION, "sha256": hashlib.sha256(declared.read_bytes()).hexdigest()},
            "modes": {mode: {"taps": taps} for mode in modes}}))
        for mode in modes:
            (directory / f"{mode}.json").write_text(json.dumps({"mode": {"precision": precision}}))
            (directory / "taps" / mode / "step0000").mkdir(parents=True)
        return directory

    def test_assemble_completes_a_captured_run_the_scorer_reads(self):
        workload = self.root / "w" / "probes.json"
        workload.parent.mkdir()
        for name in ("a.f32", "a-grid.i64"):
            (workload.parent / name).write_bytes(b"\0" * 8)
        workload.write_text(json.dumps({"sequences": [{"name": "a", "media": [
            {"tensor": {"path": "a.f32"}, "grid": {"path": "a-grid.i64"}}]}]}))
        exact = self.reference("exact", workload, ["exact"])
        policy = self.reference("policy", workload, ["policy", "reference-bf16"])
        run = self.root / "run"
        with self.assertRaisesRegex(ValueError, "holds no tap capture"):
            parity_run().assemble(run, exact, policy)
        (run / "engine").mkdir(parents=True)
        (run / "tap-capture.json").write_text(json.dumps({"requests": [{"name": "b"}]}))
        with self.assertRaisesRegex(ValueError, "not the workload's sequences"):
            parity_run().assemble(run, exact, policy)
        (run / "tap-capture.json").write_text(json.dumps({"requests": [{"name": "a"}]}))
        value = parity_run().assemble(run, exact, policy)
        self.assertEqual({key: value[key] for key in ("schema", "model", "engine", "workload", "references")},
                         {"schema": "chopinfer.encoder-parity-run.v1", "model": "glm5next", "engine": "engine",
                          "workload": "workload.json", "references": ["reference/vision"]})
        self.assertEqual(json.loads((run / "parity.json").read_text()), value)
        self.assertTrue((run / "a.f32").is_file() and (run / "a-grid.i64").is_file())
        # The scorer reads the references as the check does: the scale rule
        # from the policy, and each mode's taps.
        import encoder_declarations
        import encoder_parity
        spec = encoder_declarations.parity_rules(encoder_declarations.for_model("glm5next"))
        references = [run / path for path in value["references"]]
        self.assertEqual(encoder_parity.scale_rule(references, spec["direction_scale"]), "f32-policy")
        for mode, source in (("exact", exact), ("policy", policy), ("reference-bf16", policy)):
            merged = encoder_parity.merged_taps(references, mode, self.root / "merged" / mode)
            self.assertEqual((merged / "step0000").resolve(), (source / "taps" / mode / "step0000").resolve())

if __name__ == "__main__":
    unittest.main()

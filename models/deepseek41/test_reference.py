"""The DeepSeek V4.1 Flash reference adapter on a tiny random checkpoint in
the official layout, on CPU.

The checkpoint stores what the official one stores, by the inference code's
own names: FP8 E4M3 projections with E8M0 scales per 32 x 32 block, FP8
`wo_a`, packed E2M1 experts (I8) with E8M0 scales per 32 K, an FP8 Engram
table, F32 sinks, router biases and mHC tensors, BF16 for the rest, a vision
tower and aligner. The vendor path is the vendor's own: convert.py shards it
(mp 1, dequantizing `wo_a`), `safetensors.torch.load_model` fills a
`Transformer` built on the CPU, and `Transformer.forward` runs, with the same
replacement `kernel` module and RoPE tables as the adapter. The adapter
(units, the bounded store, `build`, HostEngram, `Plan`) must give the same
logits bit for bit in bf16 mode, for prefill and decode, with and without an
image; the runner then scores a workload in every mode end to end. Both
sides run reference_kernel.py, so this checks the adapter's loading and
scheduling against the vendor's, not the replacements' fidelity to the
tilelang kernels, which test_reference_kernel.py and the executed vendor
control (reference_vendor_control.py) cover.

Needs torch, tokenizers, safetensors and the vendor files
($CHOPINFER_DEEPSEEK_CHECKPOINT/inference); skipped otherwise.
"""

import contextlib
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock
import weakref

try:
    import torch
    from tools.refrun import adapter, modes, place, run, stamps
    from tools.refrun.run import load_adapter
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = os.environ.get("CHOPINFER_DEEPSEEK_CHECKPOINT")
IMAGE = 95
CONFIG = {
    "vocab_size": 96, "dim": 128, "moe_inter_dim": 64, "n_layers": 5, "n_mtp_layers": 1,
    "dspark_block_size": 0, "n_heads": 4, "n_routed_experts": 4, "n_shared_experts": 1, "n_activated_experts": 2,
    "score_func": "sqrtsoftplus", "route_scale": 1.5, "swiglu_limit": 10.0, "q_lora_rank": 64, "head_dim": 64,
    "rope_head_dim": 16, "norm_eps": 1e-20, "o_groups": 2, "o_lora_rank": 32, "window_size": 8,
    "kv_source_layers": [1, 3], "index_source_layers": [1, 2, 3, 4], "original_seq_len": 64, "rope_theta": 10000,
    "rope_factor": 4, "beta_fast": 32, "beta_slow": 1, "index_n_heads": 4, "index_head_dim": 32, "index_topk": 4,
    "candidate_source_layer": 3, "candidate_topk_blocks": 4, "candidate_block_size": 2, "hc_mult": 4,
    "hc_sinkhorn_iters": 20, "hc_eps": 1e-06, "engram_layer_ids": [1], "engram_vocab_size": 11,
    "engram_num_embeddings": [60], "engram_max_ngram_size": 3, "engram_pad_id": 2, "engram_n_heads": 2,
    "engram_head_dim": 32, "dtype": "fp8", "expert_dtype": "fp4", "compress_rope_theta": 160000,
    "compress_ratios": [0, 2, 2, 1, 1, 0], "vision_n_layers": 1, "vision_dim": 64, "vision_n_heads": 2,
    "vision_inter_dim": 64, "vision_patch_size": 14, "vision_downsample_ratio": 3, "vision_max_n_token": 64,
    "vision_min_pixels": 784, "vision_max_wh_ratio": None, "image_token_id": IMAGE, "vision_rope_theta": 10000,
}
F32_STORED = ("attn_sink", "gate.bias", "gate.bias_vl", "hc_")
MAX_SEQ_LEN = 64
VIT_GRID = (6, 6)  # an image of 6 x 6 patches, 2 x 2 tokens after the aligner: 8 span positions


def tokenizer_json(path):
    """96 WordLevel tokens, some of which normalize alike (case, accents,
    whitespace), so the Engram token map compresses them."""
    import tokenizers
    from tokenizers import models
    words = [*"abcdefghijklmnopqrstuvwxyz", *"ABCDEFGHIJKLMNOPQRSTUVWXYZ", *"0123456789",
             " ", "  ", "\t", "é", "É", "é", " the", " The", "THE", "\n"]
    words += [f"w{i}" for i in range(96 - len(words) - 1)] + ["<image>"]
    tokenizer = tokenizers.Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="a"))
    tokenizer.save(str(path))


def e4m3_codes(generator, shape):
    """Random finite E4M3 codes below 2 in magnitude."""
    exponent = torch.randint(0, 8, shape, generator=generator, dtype=torch.uint8)
    mantissa = torch.randint(0, 8, shape, generator=generator, dtype=torch.uint8)
    sign = torch.randint(0, 2, shape, generator=generator, dtype=torch.uint8) << 7
    return (sign | (exponent << 3) | mantissa).view(torch.float8_e4m3fn)


def e8m0(generator, shape, low=120, high=127):
    return torch.randint(low, high, shape, generator=generator, dtype=torch.uint8).view(torch.float8_e8m0fnu)


def write_checkpoint(root, generator, reference):
    """The tiny checkpoint: inference code, config, tokenizer and weights."""
    (root / "inference").mkdir(parents=True)
    for file in ("model.py", "kernel.py", "engram.py", "vision.py", "image_processor.py", "convert.py"):
        shutil.copy(Path(VENDOR_DIR) / "inference" / file, root / "inference" / file)
    tokenizer_json(root / "tokenizer.json")
    (root / "config.json").write_text(json.dumps({"model_type": "deepseek_v41", "tiny": True}))
    config = dict(CONFIG)
    (root / "inference/config.json").write_text(json.dumps(config))
    checkpoint = place.Checkpoint.__new__(place.Checkpoint)
    checkpoint.root = root
    modeling = reference.vendor(checkpoint)
    tokenizer = reference.Tokenizer(root / "tokenizer.json")
    _, size = sys.modules[f"{modeling.__name__.rpartition('.')[0]}.engram"].build_compressed_token_map(tokenizer)
    config["engram_compressed_vocab_size"] = size
    (root / "inference/config.json").write_text(json.dumps(config))
    args = reference.model_args(checkpoint, modeling, MAX_SEQ_LEN)
    declared = declared_parameters(reference, modeling, args)
    tensors = {}
    for name, (dtype, shape) in declared.items():
        if name.endswith(".scale"):
            continue
        scale = f"{name.rpartition('.')[0]}.scale"
        if dtype == torch.float8_e4m3fn or name.endswith("attn.wo_a.weight"):
            tensors[name] = e4m3_codes(generator, shape)
            scale_shape = declared[scale][1] if scale in declared else (-(-shape[0] // 32), -(-shape[1] // 32))
            tensors[scale] = e8m0(generator, scale_shape)
        elif dtype == torch.float4_e2m1fn_x2:
            tensors[name] = torch.randint(-128, 128, shape, generator=generator, dtype=torch.int8)
            tensors[scale] = e8m0(generator, declared[scale][1], 118, 125)
        elif any(key in name for key in F32_STORED):
            values = torch.randn(shape, generator=generator) * 0.1
            if "hc_" in name and name.endswith("_scale"):
                values = values.abs() + 0.5
            tensors[name] = values
        else:
            norm = "norm" in name or name.endswith(("q_weight", "k_weight"))
            values = 1 + 0.1 * torch.randn(shape, generator=generator) if norm \
                else torch.randn(shape, generator=generator) * (0.5 if name in ("embed.weight",) else 0.08)
            tensors[name] = values.to(torch.bfloat16)
    from safetensors.torch import save_file
    save_file(tensors, str(root / "model.safetensors"))
    lock = {}
    for file in ("config.json", "model.safetensors"):
        data = (root / file).read_bytes()
        lock[file] = {"byte_length": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (root / "checkpoint-files.json").write_text(json.dumps(lock))
    return tensors


def declared_parameters(reference, modeling, args):
    """Every parameter of the vendor Transformer (no MTP) and its declared
    dtype and shape, from a meta build."""
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    state = modeling.NgramHashState
    modeling.NgramHashState = lambda *_args: None
    try:
        with torch.device("meta"):
            model = modeling.Transformer(args)
    finally:
        modeling.NgramHashState = state
        torch.set_default_dtype(default)
    return {name: (p.dtype, tuple(p.shape)) for name, p in model.named_parameters() if not name.startswith("mtp.")}


def vendor_model(root, reference):
    """convert.py (mp 1) then load_model into a CPU Transformer, as the
    vendor's generate.py loads it."""
    checkpoint = place.Checkpoint(root)
    modeling = reference.vendor(checkpoint)
    convert = types.ModuleType("deepseek41_convert")
    path = root / "inference/convert.py"
    exec(compile(path.read_bytes(), str(path), "exec"), convert.__dict__)
    converted = root / "converted"
    with mock.patch.object(convert, "tqdm", lambda items: items), mock.patch.object(convert, "trange", range), \
            contextlib.redirect_stdout(io.StringIO()):
        convert.main(str(root), str(converted), 1, None, None)
    args = reference.model_args(checkpoint, modeling, MAX_SEQ_LEN)
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    # torch cannot fill uninitialized FP4 memory, which deterministic mode asks for.
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = False
    try:
        model = modeling.Transformer(args, reference.Tokenizer(root / "tokenizer.json"))
        from safetensors.torch import load_model
        load_model(model, str(converted / "model0-mp1.safetensors"))
    finally:
        torch.set_default_dtype(default)
        torch.utils.deterministic.fill_uninitialized_memory = fill
    return model.eval(), modeling, args


def vendor_forward(model, *args, **kwargs):
    """Transformer.forward with BF16 as torch's default dtype, as generate.py runs it."""
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return model.forward(*args, **kwargs)
    finally:
        torch.set_default_dtype(default)


def adapter_reference(root, reference, mode, devices=("cpu",)):
    checkpoint = place.Checkpoint(root)
    units = reference.units(checkpoint)
    placement = place.plan([(u.module, checkpoint.stored_bytes(u.tensors)) for u in units if not u.host],
                           list(devices))
    store = place.Store(checkpoint, placement, units)
    policy = modes.policy(mode)
    built = reference.build(adapter.BuildConfig(checkpoint, policy, modes.torch_dtype(policy["dtype"]), store),
                            placement, max_seq_len=MAX_SEQ_LEN)
    adapter.check_tensor_names(built, store.tensors)
    place.attach(built.model, placement)
    return built, policy


def image_patches(generator):
    count = VIT_GRID[0] * VIT_GRID[1]
    return (torch.rand((count, 3, 14, 14), generator=generator) * 2 - 1).to(torch.bfloat16)


@unittest.skipIf(torch is None or not VENDOR_DIR or not Path(VENDOR_DIR, "inference/model.py").exists(),
                 "needs torch and the vendor inference code ($CHOPINFER_DEEPSEEK_CHECKPOINT)")
class TinyCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        modes.prepare_environment(cuda=False)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name) / "checkpoint"
        generator = torch.Generator().manual_seed(20261017)
        cls.reference, cls.reference_path = load_adapter(ROOT / "models/deepseek41/reference.py")
        cls.tensors = write_checkpoint(cls.root, generator, cls.reference)
        cls.ids = torch.randint(0, IMAGE, (1, 40), generator=generator)
        cls.patches = image_patches(generator)
        cls.saved_racy = stamps.RACY_NS
        stamps.RACY_NS = 0

    @classmethod
    def tearDownClass(cls):
        stamps.RACY_NS = cls.saved_racy
        cls.directory.cleanup()

    def image_ids(self):
        ids = self.ids.clone()
        ids[0, 5:13] = IMAGE
        return ids

    def vendor_image(self, modeling, ids):
        image_processor = sys.modules[f"{modeling.__name__.rpartition('.')[0]}.image_processor"]
        types_ = image_processor.image_token_types(2, 2)
        token_types = torch.full(ids.shape, -1, dtype=torch.long)
        token_types[0, 5:13] = types_
        return [[image_processor.ImageInput(5, self.patches, *VIT_GRID, types_)]], token_types

    def test_bf16_prefill_and_decode_equal_the_vendor_load_and_forward(self):
        model, modeling, args = vendor_model(self.root, self.reference)
        built, policy = adapter_reference(self.root, self.reference, "bf16")
        with modes.active(policy):
            _, vendor_logits, _ = vendor_forward(model, self.ids, 0)
            ours = built.forward(built.model, self.ids, 1, {})
            self.assertEqual(ours.dtype, torch.float32)
            self.assertTrue(torch.equal(ours[-1], vendor_logits[0]))
            # More kept rows change only the head GEMM's shape (an F32 GEMM's order).
            rows = built.forward(built.model, self.ids, 3, {})
            torch.testing.assert_close(rows[-1], ours[-1], rtol=1e-6, atol=1e-5)
            _, vendor_logits, _ = vendor_forward(model, self.ids, 0)
            # Decode: the vendor's forward at start_pos > 0 against Plan.step.
            for position, token in enumerate((7, 31, 64), start=self.ids.shape[1]):
                _, vendor_logits, _ = vendor_forward(model, torch.tensor([[token]]), position)
                step = built.forward.__self__.step(built.model, token, position)
                self.assertTrue(torch.equal(step[0, -1], vendor_logits[0]), position)

    def test_bf16_image_prefill_equals_the_vendor(self):
        model, modeling, _ = vendor_model(self.root, self.reference)
        built, policy = adapter_reference(self.root, self.reference, "bf16")
        ids = self.image_ids()
        images, token_types = self.vendor_image(modeling, ids)
        with modes.active(policy):
            _, vendor_logits, _ = vendor_forward(model, ids, 0, images=images, token_types=token_types)
            ours = built.forward(built.model, ids, 1, {"images": [(self.patches, VIT_GRID)]})
        self.assertTrue(torch.equal(ours[-1], vendor_logits[0]))

    def test_vision_tower_makes_its_rotary_tables_on_its_own_device(self):
        # The runner leaves torch's default device on the CPU while the tower
        # sits on a GPU; a meta default device stands in for that here.
        built, policy = adapter_reference(self.root, self.reference, "bf16")
        vision = sys.modules[f"{built.forward.__self__.modeling.__name__.rpartition('.')[0]}.vision"]
        # Its lru_cache keys no device: leave no table of this test behind.
        self.addCleanup(vision.get_vision_cos_sin.cache_clear)
        with modes.active(policy):
            expected = built.model.vision(self.patches, *VIT_GRID)
            vision.get_vision_cos_sin.cache_clear()
            with torch.device("meta"):
                got = built.model.vision(self.patches, *VIT_GRID)
        self.assertEqual(got.device, expected.device)
        self.assertTrue(torch.equal(got, expected))

    def test_wo_a_is_the_vendor_conversion(self):
        built, _ = adapter_reference(self.root, self.reference, "bf16")
        from safetensors.torch import load_file
        if not (self.root / "converted/model0-mp1.safetensors").exists():
            vendor_model(self.root, self.reference)
        converted = load_file(str(self.root / "converted/model0-mp1.safetensors"))
        for layer in range(CONFIG["n_layers"]):
            name = f"layers.{layer}.attn.wo_a.weight"
            self.assertTrue(torch.equal(built.model.get_parameter(name), converted[name]))
        self.assertEqual(built.tensor_names[f"layers.0.attn.wo_a.scale"], "layers.0.attn.wo_a.weight")
        self.assertEqual(built.tensor_names["layers.1.engram.embed.weight"], "host:engram-rows")

    def test_modes_widen_and_agree(self):
        logits = {}
        for mode in ("bf16", "f32", "f64"):
            built, policy = adapter_reference(self.root, self.reference, mode)
            with modes.active(policy):
                logits[mode] = built.forward(built.model, self.ids, 4, {})
        self.assertEqual(logits["f64"].dtype, torch.float64)
        self.assertEqual(logits["f32"].dtype, torch.float32)
        f64 = logits["f64"]
        f32_error = float((logits["f32"].double() - f64).abs().mean())
        bf16_error = float((logits["bf16"].double() - f64).abs().mean())
        self.assertGreater(f32_error, 0.0)
        self.assertLess(f32_error, bf16_error)
        self.assertLess(f32_error, 1e-5 * float(f64.abs().max()))
        self.assertLess(bf16_error, float(f64.abs().mean()))

    def test_widened_source_is_f64_except_the_rope_angles(self):
        checkpoint = place.Checkpoint(self.root)
        wide = self.reference.vendor(checkpoint, wide=True)
        self.assertTrue(wide.WIDE)
        self.assertIsNot(wide, self.reference.vendor(checkpoint))
        source = self.reference.widen("model", (self.root / "inference/model.py").read_bytes()).decode()
        lines = source.split("\n")
        self.assertIn("torch.float32", lines[375])
        widened = "\n".join(line for i, line in enumerate(lines) if i + 1 not in (376, 384, 1294))
        self.assertNotIn(".float()", widened)
        self.assertNotIn("torch.float32", widened)

    def test_rope_table_keeps_the_vendor_angles(self):
        """An unpinned geometry (the tiny one) takes the vendor's own lines on
        the build device; cos and sin are correctly rounded from the angles."""
        checkpoint = place.Checkpoint(self.root)
        modeling = self.reference.vendor(checkpoint)
        original = modeling.Attention.__init__.__globals__  # the executed module's globals
        vendor = self.vendor_freqs(checkpoint)
        for args in ((16, 64, 64, 160000, 4, 32, 1), (16, 64, 0, 10000, 4, 32, 1)):
            angles = self.reference.rope_angles(*args)
            self.assertTrue(torch.equal(torch.polar(torch.ones_like(angles), angles), vendor(*args)))
            table = self.reference.rope_table(*args)
            self.assertTrue(torch.equal(table.real, torch.cos(angles.double()).float()))
            self.assertTrue(torch.equal(table.imag, torch.sin(angles.double()).float()))
        self.assertEqual(original["precompute_freqs_cis"].__qualname__, "rope_table")

    def test_rope_angles_match_the_pin(self):
        """The official geometries take the CUDA frequencies of the program's
        RoPE pin: their angle tables over 2^20 positions have its digests, and
        they differ from the CPU evaluation in exactly the three pinned pairs."""
        pins = {(64, 0, 10000, 16, 32, 1): ("e52b4866731714b90ebf78c31eaa698d8013aff8b3324cfd10ee5e7975e1445d", {11, 25}),
                (64, 65536, 160000, 16, 32, 1): ("99c92572f98f747bbe2fa85fed689a653f4809bedc6e2568343dc13acaf9c00d",
                                                 {24})}
        for (dim, original, base, factor, fast, slow), (digest, pairs) in pins.items():
            angles = self.reference.rope_angles(dim, 1 << 20, original, base, factor, fast, slow)
            self.assertEqual(hashlib.sha256(angles.contiguous().numpy().tobytes()).hexdigest(), digest)
            # Without the pin, the vendor's lines on the CPU (test_rope_table_keeps_the_vendor_angles).
            with mock.patch.dict(self.reference.FREQUENCY_BITS, clear=True):
                cpu = self.reference.rope_frequencies(dim, original, base, factor, fast, slow)
            differing = set((cpu != angles[1]).nonzero().flatten().tolist())
            self.assertEqual(differing, pairs)

    def vendor_freqs(self, checkpoint):
        """The pinned model.py's own precompute_freqs_cis, from a fresh exec."""
        namespace = {}
        source = (self.root / "inference/model.py").read_text()
        start = source.index("@lru_cache(2)\ndef precompute_freqs_cis")
        end = source.index("def apply_rotary_emb")
        exec("import math, torch\nfrom functools import lru_cache\n" + source[start:end], namespace)
        return namespace["precompute_freqs_cis"].__wrapped__

    def test_engram_rows_come_from_the_host(self):
        built, policy = adapter_reference(self.root, self.reference, "bf16")
        embed = built.model.layers[1].engram.embed
        self.assertIsInstance(embed, self.reference.HostEngram)
        indices = torch.tensor([[[0, 59, 7, 7], [3, 3, 3, 12]]])
        values = embed(indices)
        weight = self.tensors["layers.1.engram.embed.weight"].float()
        scale = self.tensors["layers.1.engram.embed.scale"].float()
        expected = (weight.view(60, -1, 32) * scale.unsqueeze(-1)).view(60, -1).bfloat16()
        self.assertTrue(torch.equal(values, expected[indices]))

    def test_shared_state_follows_the_layer_device(self):
        """A layer reusing a source's compressed KV, index keys, top-k or
        candidates on another device reads copies there (cpu to meta here,
        the one second device a CPU test has)."""
        checkpoint = place.Checkpoint(self.root)
        modeling = self.reference.vendor(checkpoint)
        shared = modeling.shared_attn
        saved = {name: getattr(shared, name) for name in self.reference.SHARED}
        try:
            for name in self.reference.SHARED:
                setattr(shared, name, torch.zeros(2, 3))
            shared.topk_idxs = None
            self.reference.share_to(modeling, torch.device("meta"))
            for name in ("compress_kv", "index_k", "candidates"):
                self.assertEqual(getattr(shared, name).device.type, "meta")
            self.assertIsNone(shared.topk_idxs)
        finally:
            for name, value in saved.items():
                setattr(shared, name, value)

    def test_every_layer_shares_before_it_runs(self):
        built, policy = adapter_reference(self.root, self.reference, "bf16")
        calls = []
        original = self.reference.share_to
        with modes.active(policy):
            # reference_memory.prefill moves what each layer reads itself; decode steps share.
            built.forward(built.model, self.ids, 1, {})
            with mock.patch.object(self.reference, "share_to",
                                   side_effect=lambda m, d: (calls.append(d), original(m, d))):
                built.forward.__self__.step(built.model, 5, self.ids.shape[1])
        self.assertEqual(len(calls), CONFIG["n_layers"])

    def test_released_reference_frees_its_tensors_without_the_cycle_collector(self):
        # The runner drops a mode's model and empties the device cache before
        # the next build; tensors held in a cycle would survive until then.
        built, policy = adapter_reference(self.root, self.reference, "f64")
        with modes.active(policy):
            built.forward(built.model, self.ids, 1, {})
        widened = weakref.ref(built.model.layers[0].attn.wo_a.weight)
        cache = weakref.ref(built.model.layers[0].attn.window_kv_cache)
        enabled = gc.isenabled()
        gc.disable()
        try:
            built = None
            self.assertIsNone(widened())
            self.assertIsNone(cache())
        finally:
            if enabled:
                gc.enable()

    def test_reference_code_keys_the_tokenizer(self):
        files = self.reference.reference_code(place.Checkpoint(self.root))
        self.assertEqual(files["deepseek41/tokenizer.json"], self.root / "tokenizer.json")
        self.assertEqual(files["deepseek41/inference/config.json"], self.root / "inference/config.json")

    def test_runner_scores_every_mode_with_taps(self):
        root = Path(self.directory.name)
        tokens = self.image_ids()[0].tolist()
        (root / "patches.bin").write_bytes(self.patches.view(torch.int16).numpy().tobytes())
        (root / "grid.bin").write_bytes(torch.tensor(VIT_GRID, dtype=torch.int64).numpy().tobytes())
        processor = {"code_sha256": "0" * 64, "config_sha256": "0" * 64}
        sequences = [
            {"name": "text", "category": "text", "token_ids": self.ids[0].tolist(), "score_from": 30},
            {"name": "image", "category": "image", "token_ids": tokens, "score_from": 20, "media": [
                {"kind": "image", "processor": processor,
                 "tensor": {"path": "patches.bin", "dtype": "bf16", "shape": [36, 3, 14, 14]},
                 "grid": {"path": "grid.bin", "dtype": "i64", "shape": [2]}}]},
            {"name": "greedy", "category": "text", "token_ids": self.ids[0, :20].tolist(), "score_from": 15,
             "answer": {"policy": "greedy", "max_new_tokens": 3, "stop_ids": [1]}}]
        workload = root / "workload.json"
        workload.write_text(json.dumps({"schema": "chopinfer.refrun.workload.v1", "model": "deepseek41",
                                        "sequences": sequences}))
        output = root / "run"
        small = types.SimpleNamespace(**{k: getattr(self.reference, k) for k in dir(self.reference)
                                         if not k.startswith("__")})
        small.build = lambda config, placement: self.reference.build(config, placement, max_seq_len=MAX_SEQ_LEN)
        with mock.patch.object(run, "load_adapter", return_value=(small, self.reference_path)):
            provenance = run.run(run.parse_args([
                "--adapter", str(ROOT / "models/deepseek41/reference.py"), "--checkpoint", str(self.root),
                "--lock", str(self.root / "checkpoint-files.json"), "--workload", str(workload),
                "--modes", "f32,bf16,f64", "--devices", "cpu,cpu", "--output", str(output),
                "--stamps", str(root / "stamps"), "--no-cache", "--taps", "--tap-sites", "all"]))
        for mode in ("f32", "bf16", "f64"):
            entries = json.loads((output / f"{mode}.json").read_text())["entries"]
            expected = ["greedy", "text", "image", "greedy"] if mode == "f32" else ["text", "image", "greedy"]
            self.assertEqual([e["name"] for e in entries], expected)
            for entry in entries:
                self.assertTrue(all(math.isfinite(v) for v in entry["scores"]["logprob"]))
            self.assertTrue(provenance["modes"][mode]["repeat_bitwise_equal"])
        self.assertEqual(provenance["layout"]["quantizer_inputs"], "bfloat16")
        names = provenance["taps"]
        for name in ("L01.x.engram.out", "L03.x.indexer.ids", "L00.x.hc.attn.comb", "enc.vision.out",
                     "final.norm", "L04.router.ids", "L02.out"):
            self.assertIn(name, names)
        index = [json.loads(line) for line in (output / "taps/f32/step0000/index.jsonl").read_text().splitlines()]
        rows = {entry["name"]: entry for entry in index}
        self.assertEqual(rows["L00.in.streams"]["cols"], 4 * CONFIG["dim"])
        self.assertEqual(rows["L00.x.hc.attn.comb"]["cols"], 16)


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(torch is None or not VENDOR_DIR or not Path(VENDOR_DIR, "model.safetensors.index.json").exists(),
                 "needs torch and the official checkpoint ($CHOPINFER_DEEPSEEK_CHECKPOINT)")
class OfficialLayout(unittest.TestCase):
    """The official checkpoint's headers against the vendor model the adapter
    builds, without reading a weight: every stored tensor but the MTP's has a
    vendor parameter of its shape in a dtype the load accepts, every vendor
    parameter a stored tensor, and the Engram tables stay on the host."""

    def test_headers_bind_to_the_vendor_parameters(self):
        reference, _ = load_adapter(ROOT / "models/deepseek41/reference.py")
        checkpoint = place.Checkpoint(VENDOR_DIR)
        self.assertEqual(hashlib.sha256((checkpoint.root / reference.CONFIG).read_bytes()).hexdigest(),
                         reference.OFFICIAL_CONFIG)
        modeling = reference.vendor(checkpoint)
        args = reference.model_args(checkpoint, modeling)
        declared = declared_parameters(reference, modeling, args)
        units = reference.units(checkpoint)
        loaded = {name for unit in units if not unit.host for name in unit.tensors}
        host = {name for unit in units if unit.host for name in unit.tensors}
        self.assertEqual(host, {f"layers.{layer}.engram.embed.{part}" for layer in (1, 14)
                                for part in ("weight", "scale")})
        stored = set(checkpoint.entries)
        self.assertEqual(stored - loaded - host, {n for n in stored if n.startswith("mtp.")})
        kinds = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "I8": torch.float4_e2m1fn_x2}
        for name in sorted(loaded | host):
            entry = checkpoint.entries[name]
            if name.endswith("attn.wo_a.scale"):
                continue
            dtype, shape = declared[name]
            self.assertEqual(tuple(entry.shape), shape, name)
            if name.endswith("attn.wo_a.weight"):
                self.assertEqual((entry.dtype, dtype), ("F8_E4M3", torch.bfloat16), name)
            elif entry.dtype in kinds:
                self.assertEqual(kinds[entry.dtype], dtype, name)
            else:
                self.assertIn(entry.dtype, ("BF16", "F32"), name)
                self.assertTrue(dtype.is_floating_point and dtype.itemsize >= 2, name)
        missing = {n for n in declared if n not in stored}
        self.assertEqual(missing, set())

    def test_official_placement_puts_reuse_layers_beside_other_devices_sources(self):
        """On the four-GPU placement some layers reuse a compressed KV source
        on another device (layer 10 reads layer 8's), which share_to covers."""
        reference, _ = load_adapter(ROOT / "models/deepseek41/reference.py")
        checkpoint = place.Checkpoint(VENDOR_DIR)
        args = reference.model_args(checkpoint, reference.vendor(checkpoint))
        units = [u for u in reference.units(checkpoint) if not u.host]
        placement = place.plan([(u.module, checkpoint.stored_bytes(u.tensors)) for u in units],
                               ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        device = dict(placement.units)
        crossing = []
        source = None
        for layer in range(args.n_layers):
            if layer in args.kv_source_layers:
                source = layer
            elif args.compress_ratios[layer] and device[f"layers.{layer}"] != device[f"layers.{source}"]:
                crossing.append((source, layer))
        self.assertIn((8, 10), crossing)

    def test_official_config_refuses_another_tokenizer(self):
        reference, _ = load_adapter(ROOT / "models/deepseek41/reference.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(Path(VENDOR_DIR) / "inference", root / "inference",
                            ignore=shutil.ignore_patterns("examples"))
            tokenizer_json(root / "tokenizer.json")
            checkpoint = place.Checkpoint.__new__(place.Checkpoint)
            checkpoint.root = root
            with self.assertRaisesRegex(ValueError, "not the official"):
                reference.reference_code(checkpoint)
            shutil.copy(Path(VENDOR_DIR) / "tokenizer.json", root / "tokenizer.json")
            self.assertIn("deepseek41/tokenizer.json", reference.reference_code(checkpoint))

    def test_engram_token_map_has_the_configured_size(self):
        """NgramHashState's compressed vocabulary from the checkpoint's
        tokenizer through the adapter's Tokenizer equals the config's 99,092."""
        reference, _ = load_adapter(ROOT / "models/deepseek41/reference.py")
        checkpoint = place.Checkpoint(VENDOR_DIR)
        modeling = reference.vendor(checkpoint)
        engram = sys.modules[f"{modeling.__name__.rpartition('.')[0]}.engram"]
        tokenizer = reference.Tokenizer(checkpoint.root / reference.TOKENIZER)
        self.assertEqual(len(tokenizer), 129280)
        _, size = engram.build_compressed_token_map(tokenizer)
        self.assertEqual(size, reference.model_args(checkpoint, modeling).engram_compressed_vocab_size)
        self.assertEqual(size, 99092)


@unittest.skipIf(torch is None or not VENDOR_DIR or not Path(VENDOR_DIR, "inference/model.py").exists(),
                 "needs torch and the vendor inference code ($CHOPINFER_DEEPSEEK_CHECKPOINT)")
class PureF32(unittest.TestCase):
    def test_pins_reference_and_quantizes_unrounded_inputs(self):
        modes.prepare_environment(cuda=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            reference, _ = load_adapter(ROOT / "models/deepseek41/reference.py")
            pure, _ = load_adapter(ROOT / "models/deepseek41/reference_pure_f32.py")
            write_checkpoint(root, torch.Generator().manual_seed(3), reference)
            ids = torch.randint(0, IMAGE, (1, 24), generator=torch.Generator().manual_seed(4))
            logits = {}
            for name, module in (("tq", reference), ("pure", pure)):
                built, policy = adapter_reference(root, module, "f32")
                self.assertEqual(built.layout["quantizer_inputs"], "bfloat16" if name == "tq" else "mode")
                with modes.active(policy):
                    logits[name] = built.forward(built.model, ids, 2, {})
            self.assertFalse(torch.equal(logits["tq"], logits["pure"]))
            with self.assertRaisesRegex(ValueError, "f32 mode only"):
                adapter_reference(root, pure, "bf16")
            reference.kernels().set_policy("bfloat16")

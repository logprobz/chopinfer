"""The latent KV format-study adapter (reference_latent_kv.py) on CPU.

Its served format reproduces reference_latent_fp8.py's round trip bit for
bit; the other formats follow their element and scale rules; and the
adapter's scores on test_reference.py's tiny checkpoint match a plain HF
forward whose sparse layers' `kv_a_layernorm` outputs take the chosen
round trip. Needs torch and transformers 5.16.1; skipped without them. The
GPU comparison runs only when selected explicitly, with
CHOPINFER_LATENT_KV_CUDA=1 and `-k CudaRoundTripTests`, on a reserved GPU.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B -m unittest models/glm5next/test_reference_latent_kv.py
"""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from tools.refrun import modes, run, stamps
except ModuleNotFoundError:
    torch = None

ADAPTER = ROOT / "models/glm5next/reference_latent_kv.py"
CUDA_FLAG = "CHOPINFER_LATENT_KV_CUDA"
ON_CUDA = os.environ.get(CUDA_FLAG) == "1"
SERVED = ROOT / "models/glm5next/reference_latent_fp8.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setUpModule():
    if torch is None:
        return
    modes.prepare_environment(cuda=ON_CUDA)
    global RACY_NS
    RACY_NS, stamps.RACY_NS = stamps.RACY_NS, 100_000_000


def tearDownModule():
    if torch is not None:
        stamps.RACY_NS = RACY_NS


def sample(seed, rows=4, width=512):
    """Rows whose 128-blocks differ in magnitude by 2^16, with a zero block."""
    generator = torch.Generator().manual_seed(seed)
    magnitude = torch.tensor([2.0 ** -9, 1.0, 2.0 ** 7, 0.0]).repeat_interleave(width // 4)
    return torch.randn((rows, width), generator=generator) * magnitude


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class RoundTripTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load(ADAPTER, "kv1_latent_kv")
        cls.served = load(SERVED, "kv1_latent_fp8")

    def test_the_served_format_is_the_fp8_adapters_round_trip(self):
        form = self.adapter.FORMATS[self.adapter.FORMAT]
        for dtype in (torch.float32, torch.bfloat16):
            x = sample(1).to(dtype)
            got = self.adapter.round_trip(x, form)
            self.assertEqual(got.dtype, dtype)
            self.assertTrue(torch.equal(got, self.served.latent_fp8(x)), dtype)

    def test_the_scale_is_the_rounded_f32_division(self):
        # amax 1.203125 / 448 rounds to 0.002685546875 in F32; 0.01275634765625
        # is then 4.75, a tie that rounds to the even code 5.0 (0x4a). A scale
        # from the rounded reciprocal of 448 (CUDA's scalar division) gives
        # code 4.5 (0x49) instead.
        x = torch.zeros(1, 128, dtype=torch.bfloat16)
        x[0, 0], x[0, 1] = 1.203125, 0.01275634765625
        form = self.adapter.FORMATS["e4m3-b128"]
        got = self.adapter.round_trip(x.float(), form)
        self.assertEqual(got[0, 1].item(), 5.0 * 0.002685546875)
        self.assertEqual(got[0, 0].item(), 1.203125)
        self.assertEqual(self.adapter.round_trip(x, form)[0, 1].item(), 0.013427734375)

    def test_no_division_takes_a_python_scalar_divisor(self):
        # CPU divides by a scalar exactly, so the rule is checked structurally.
        divide = torch.Tensor.__truediv__

        def tensor_divisors_only(a, b):
            if not isinstance(b, torch.Tensor):
                raise AssertionError(f"division by the Python scalar {b!r}")
            return divide(a, b)
        with mock.patch.object(torch.Tensor, "__truediv__", tensor_divisors_only):
            for form in self.adapter.FORMATS.values():
                self.adapter.round_trip(sample(5), form)

    def test_int8_codes_are_the_nearest_integers_times_the_exact_scale(self):
        x = sample(2)
        for name, block in (("int8-b128", 128), ("int8-b64", 64), ("int8-b32", 32)):
            got = self.adapter.round_trip(x, self.adapter.FORMATS[name]).reshape(4, -1, block)
            blocks = x.reshape(4, -1, block)
            amax = blocks.abs().amax(-1, keepdim=True)
            scale = torch.where(amax == 0, torch.ones_like(amax), amax / 127.0)
            codes = torch.tensor([float(round(v)) for v in (blocks / scale).flatten().tolist()])
            self.assertTrue(torch.equal(got, codes.reshape(blocks.shape) * scale), name)
            # Every element is within half a step of its value, and the block
            # maximum is exactly 127 steps.
            self.assertTrue(bool(((got - blocks).abs() <= scale / 2 * (1 + 2 ** -20)).all()), name)
            self.assertTrue(torch.equal(got.abs().amax(-1, keepdim=True), torch.where(amax == 0, amax, 127 * scale)))

    def test_pow2_scales_are_the_power_at_or_above_the_exact_one(self):
        form = self.adapter.FORMATS["e4m3-b128-pow2"]
        x = sample(3)
        x[0, :128] = 0.0
        x[0, 0] = 448.0 * 2.0 ** -3  # an exact power: its scale is 2^-3
        got = self.adapter.round_trip(x, form)
        self.assertEqual(got[0, 0].item(), 448.0 * 2.0 ** -3)
        for r in range(4):
            for b in range(3):
                values = x[r, 128 * b:128 * (b + 1)]
                exact = values.abs().max() / 448.0
                power = 2.0 ** torch.ceil(torch.log2(exact))
                self.assertGreaterEqual(power.item(), exact.item())
                self.assertLess((power / 2).item(), exact.item())
                expected = (values / power).to(torch.float8_e4m3fn).float() * power
                self.assertTrue(torch.equal(got[r, 128 * b:128 * (b + 1)], expected), (r, b))

    def test_e5m2_and_other_blocks_follow_their_element(self):
        x = sample(4)
        for name, element, block, top in (("e5m2-b128", torch.float8_e5m2, 128, 57344.0),
                                          ("e4m3-b32", torch.float8_e4m3fn, 32, 448.0),
                                          ("e4m3-b512", torch.float8_e4m3fn, 512, 448.0)):
            got = self.adapter.round_trip(x, self.adapter.FORMATS[name]).reshape(4, -1, block)
            blocks = x.reshape(4, -1, block)
            amax = blocks.abs().amax(-1, keepdim=True)
            scale = torch.where(amax == 0, torch.ones_like(amax), amax / top)
            expected = (blocks / scale).clamp(-top, top).to(element).float() * scale
            self.assertTrue(torch.equal(got, expected), name)

    def test_memory_per_token(self):
        self.assertEqual({name: f.token_bytes(512) for name, f in self.adapter.FORMATS.items()},
                         {"e4m3-b128": 528, "e4m3-b32": 576, "e4m3-b512": 516, "e4m3-b128-pow2": 528,
                          "e5m2-b128": 528, "int8-b128": 528, "int8-b64": 544, "int8-b32": 576})

    def test_a_width_off_the_block_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not a multiple of 128"):
            self.adapter.round_trip(torch.zeros(1, 96), self.adapter.FORMATS["int8-b128"])


@unittest.skipIf(torch is None or not ON_CUDA, f"the CUDA round trip runs with {CUDA_FLAG}=1")
class CudaRoundTripTests(unittest.TestCase):
    """Every format's round trip on the GPU (PyTorch's CUDA division kernels)
    equals the CPU's bit for bit. Opt-in only: `-k CudaRoundTripTests` with
    CUDA_FLAG set, on a GPU the caller reserved."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{CUDA_FLAG}=1 but CUDA is not available")
        cls.adapter = load(ADAPTER, "kv1_latent_kv_cuda")

    def test_cuda_round_trips_equal_the_cpu_ones(self):
        x = torch.cat([sample(6), torch.zeros(1, 512)])
        x[-1, 0], x[-1, 1] = 1.203125, 0.01275634765625
        for name, form in self.adapter.FORMATS.items():
            for dtype in (torch.float32, torch.bfloat16):
                cpu = self.adapter.round_trip(x.to(dtype), form)
                self.assertTrue(torch.equal(self.adapter.round_trip(x.to(dtype).cuda(), form).cpu(), cpu),
                                (name, dtype))


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class AdapterTests(unittest.TestCase):
    # The tiny checkpoint's latent is 32 wide: two INT8 blocks of 16.
    FORMAT = "int8-b16"

    @classmethod
    def setUpClass(cls):
        cls.base = load(ROOT / "models/glm5next/test_reference.py", "kv1_test_reference")
        cls.directory = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        root = cls.root = Path(cls.directory.name)
        cls.tensors = cls.base.write_checkpoint(root / "checkpoint", seed=5)
        (root / "workload.json").write_text(json.dumps({
            "schema": "chopinfer.refrun.workload.v1", "model": "glm5next",
            "sequences": [{"name": "text", "category": "prose", "token_ids": cls.base.PROMPT, "score_from": 1}]}))
        load_adapter = run.load_adapter
        cls.adapters = []

        def tiny_format(path):
            module, relative = load_adapter(path)
            if hasattr(module, "FORMATS"):
                module.FORMATS = {**module.FORMATS, cls.FORMAT: module.LatentFormat("int8", 16)}
                module.FORMAT = cls.FORMAT
                cls.adapters.append(module)
            return module, relative
        with mock.patch.object(run, "load_adapter", side_effect=tiny_format):
            for adapter, output in ((ADAPTER, "int8"), (cls.base.ADAPTER, "plain")):
                run.run(run.parse_args([
                    "--adapter", str(adapter), "--checkpoint", str(root / "checkpoint"),
                    "--lock", str(root / "checkpoint/lock.json"), "--workload", str(root / "workload.json"),
                    "--devices", "cpu,cpu", "--output", str(root / output), "--stamps", str(root / "stamps"),
                    "--modes", "f32", "--dequant", "f32=rounded", "--no-cache"]))
        cls.scores = {output: json.loads((root / output / "f32.json").read_text())["entries"][0]["scores"]
                      for output in ("int8", "plain")}
        cls.run_json = json.loads((root / "int8" / "run.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_scores_match_a_plain_forward_with_the_latent_round_trip(self):
        adapter = self.adapters[0]
        form = adapter.FORMATS[self.FORMAT]
        model = self.base.plain_model(self.tensors, torch.float32)
        hooked = 0
        for block in model.model.language_model.layers:
            if block.block_type != "linear_attention":
                block.self_attn.kv_a_layernorm.register_forward_hook(
                    lambda _m, _i, output: adapter.round_trip(output, form))
                hooked += 1
        self.assertEqual(hooked, 2)
        ids = self.base.PROMPT
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0]
        expected = torch.log_softmax(logits[:-1], dim=-1).gather(1, torch.tensor(ids[1:])[:, None])[:, 0].double()
        got = torch.tensor(self.scores["int8"]["logprob"], dtype=torch.float64)
        self.assertLess((got - expected).abs().max().item(), 2e-5)
        self.assertEqual(self.scores["int8"]["top1"], logits[:-1].argmax(dim=-1).tolist())

    def test_the_round_trip_changes_the_scores(self):
        quantized = torch.tensor(self.scores["int8"]["logprob"], dtype=torch.float64)
        plain = torch.tensor(self.scores["plain"]["logprob"], dtype=torch.float64)
        self.assertGreater((quantized - plain).abs().max().item(), 1e-5)

    def test_the_run_records_the_format(self):
        self.assertIn("int8-b16 latent round trip", self.run_json["layout"]["kv_cache"])
        self.assertIn("INT8 codes, one F32 scale per 16",
                      self.run_json["replacements"]["kv_a_layernorm (sparse layers)"])

    def test_the_adapter_runs_the_pinned_reference(self):
        self.assertEqual(self.adapters[0].reference.__file__, str(self.base.ADAPTER))


if __name__ == "__main__":
    unittest.main()

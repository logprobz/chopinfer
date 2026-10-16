"""The FP8 latent ablation adapter (reference_latent_fp8.py).

The round trip follows the served writer's rule (kv_write.cuh, kernel-core's
`LatentKvFormat::Fp8E4M3Block128`), checked element by element against
scales, codes and values computed in exact rational arithmetic, on CPU and,
with CHOPINFER_LATENT_FP8_CUDA=1 and `-k CudaRoundTripTests`, on the GPU;
and the adapter's scores on test_reference.py's tiny checkpoint match a
plain HF forward whose sparse layers' `kv_a_layernorm` outputs take the
same round trip, and its x.latent tap is the latent before the round trip.
Needs torch and transformers 5.16.1; skipped without them.

    CUDA_VISIBLE_DEVICES= $CHOPINFER_REFRUN_PYTHON -B -m unittest models/glm5next/test_reference_latent_fp8.py
"""

from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
    from tools.refrun import modes, quant, run, stamps
except ModuleNotFoundError:
    torch = None

ADAPTER = ROOT / "models/glm5next/reference_latent_fp8.py"
# Set by the GPU job: run CudaRoundTripTests (alone, -k CudaRoundTripTests).
CUDA_FLAG = "CHOPINFER_LATENT_FP8_CUDA"
ON_CUDA = os.environ.get(CUDA_FLAG) == "1"
# The tiny checkpoint's latent is 32 wide: two scale blocks of 16.
TINY_BLOCK = 16


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


def f32(value):
    """`value` (a Fraction or float) rounded to the nearest F32. float() of a
    Fraction is the correctly rounded f64; rounding that to F32 is the
    correctly rounded F32 for a quotient or product of two F32 values (53 >=
    2 * 24 + 2 bits), so no torch arithmetic enters the expected values."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def nearest_e4m3(value):
    """The finite E4M3 value nearest `value` (ties to the even code), saturated."""
    finite = [(v, code) for code, v in enumerate(quant.E4M3) if v == v and code < 0x80]
    magnitude = min(abs(value), 448.0)
    best = min(finite, key=lambda item: (abs(item[0] - magnitude), item[1] & 1))[0]
    return best if value >= 0 else -best


def expected_block(values):
    """The served writer's (kv_write.cuh) F32 values for one block of F32
    inputs, in exact rational arithmetic: scale = RN(amax / 448), code =
    E4M3(RN(x / scale)), value = RN(code * scale)."""
    amax = max(abs(v) for v in values)
    scale = 1.0 if amax == 0 else f32(Fraction(amax) / 448)
    return scale, [f32(Fraction(nearest_e4m3(f32(Fraction(v) / Fraction(scale)))) * Fraction(scale))
                   for v in values]


class RoundTrip:
    """latent_fp8 against the served writer's rule on `device`."""
    device = None

    @classmethod
    def setUpClass(cls):
        cls.adapter = load(ADAPTER, "acc1_latent_fp8")

    def round_trip(self, x):
        return self.adapter.latent_fp8(x.to(self.device)).cpu()

    def test_contract_example_round_trips_exactly(self):
        # amax 3.5 gives scale 3.5 / 448 = 2^-7, so every E4M3 multiple of it
        # round trips; the zero block keeps zero codes.
        block = torch.tensor([3.5, -3.5, 0.125, 2.0 ** -7, -1.75, 0.0] + [0.5] * 122, dtype=torch.bfloat16)
        row = torch.cat([block, torch.zeros(128, dtype=torch.bfloat16)])[None]
        self.assertTrue(torch.equal(self.round_trip(row), row))

    def test_scale_and_code_of_a_reciprocal_sensitive_block(self):
        # BF16 amax 1.203125: RN(amax / 448) is 0.002685546875, and the
        # element 0.01275634765625 is then 4.75 scales, a tie that rounds to
        # code 74 (5.0). The reciprocal-multiply scale 0.0026855471... (what
        # CUDA's division by a Python scalar computes) gives code 73 (4.5).
        amax, element = 1.203125, 0.01275634765625
        self.assertEqual(f32(Fraction(amax) / 448), 0.002685546875)
        self.assertEqual(f32(Fraction(element) / Fraction(0.002685546875)), 4.75)
        self.assertEqual(nearest_e4m3(4.75), 5.0)
        reciprocal = f32(Fraction(amax) * Fraction(f32(Fraction(1, 448))))
        self.assertNotEqual(reciprocal, 0.002685546875)
        self.assertEqual(nearest_e4m3(f32(Fraction(element) / Fraction(reciprocal))), 4.5)
        x = torch.zeros(1, 128, dtype=torch.bfloat16)
        x[0, 0], x[0, 1] = amax, element
        amax_f32 = torch.tensor([[amax]], dtype=torch.float32, device=self.device)
        self.assertEqual(self.adapter.f32_div(amax_f32, torch.full_like(amax_f32, 448.0)).item(), 0.002685546875)
        got = self.adapter.latent_fp8(x.float().to(self.device)).cpu()
        self.assertEqual(got[0, 1].item(), 5.0 * 0.002685546875)
        self.assertEqual(got[0, 0].item(), f32(Fraction(448) * Fraction(0.002685546875)))

    def test_scales_are_rn_quotients_for_every_bf16_amax_of_an_octave(self):
        # Every BF16 magnitude in [1, 2) as a block maximum, each beside an
        # element near 4.75 scales: the scale and the code must be RN's.
        amaxes = torch.arange(0x3F80, 0x4000, dtype=torch.int16).view(torch.bfloat16).float()
        x = torch.zeros(amaxes.numel(), 128)
        x[:, 0] = amaxes
        x[:, 1] = amaxes / 448 * 4.75
        got = self.round_trip(x)
        for row, values in zip(got.tolist(), x.tolist()):
            self.assertEqual(row[:2], expected_block(values)[1][:2], values[:2])

    def test_each_element_matches_the_writer_rule(self):
        generator = torch.Generator().manual_seed(3)
        row = (torch.randn((2, 256), generator=generator) * torch.tensor([0.02, 3.0]).repeat_interleave(128))
        for dtype in (torch.float32, torch.bfloat16):
            x = row.to(dtype)
            got = self.round_trip(x)
            self.assertEqual(got.dtype, dtype)
            for r in range(2):
                for b in range(2):
                    _, values = expected_block(x[r, 128 * b:128 * (b + 1)].float().tolist())
                    expected = torch.tensor(values, dtype=torch.float32).to(dtype)
                    self.assertTrue(torch.equal(got[r, 128 * b:128 * (b + 1)], expected), (dtype, r, b))

    def test_the_largest_magnitude_saturates_to_448(self):
        x = torch.full((1, 128), 1e-3, dtype=torch.float32)
        x[0, 7] = -0.3
        scale, values = expected_block(x[0].tolist())
        self.assertEqual(self.round_trip(x)[0, 7].item(), values[7])
        self.assertEqual(values[7], -f32(Fraction(448) * Fraction(scale)))

    def test_a_width_off_the_scale_block_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not a multiple of 128"):
            self.adapter.latent_fp8(torch.zeros(1, 96, device=self.device))


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class RoundTripTests(RoundTrip, unittest.TestCase):
    device = "cpu"


@unittest.skipIf(torch is None or not ON_CUDA, f"the CUDA round trip runs with {CUDA_FLAG}=1")
class CudaRoundTripTests(RoundTrip, unittest.TestCase):
    """The same checks on the GPU (PyTorch's CUDA division kernels), run by
    the ACC1 GPU job alone: `-k CudaRoundTripTests` with CUDA_FLAG set, so
    no CPU test of this file shares the process with CUDA."""
    device = "cuda"

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{CUDA_FLAG}=1 but CUDA is not available")
        super().setUpClass()


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = load(ROOT / "models/glm5next/test_reference.py", "acc1_test_reference")
        cls.directory = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        root = cls.root = Path(cls.directory.name)
        cls.tensors = cls.base.write_checkpoint(root / "checkpoint", seed=5)
        (root / "workload.json").write_text(json.dumps({
            "schema": "chopinfer.refrun.workload.v1", "model": "glm5next",
            "sequences": [{"name": "text", "category": "prose", "token_ids": cls.base.PROMPT, "score_from": 1}]}))
        load_adapter = run.load_adapter
        cls.adapters = []

        def tiny_blocks(path):
            module, relative = load_adapter(path)
            module.SCALE_BLOCK = TINY_BLOCK
            cls.adapters.append(module)
            return module, relative
        with mock.patch.object(run, "load_adapter", side_effect=tiny_blocks):
            for adapter, output in ((ADAPTER, "fp8"), (cls.base.ADAPTER, "plain")):
                run.run(run.parse_args([
                    "--adapter", str(adapter), "--checkpoint", str(root / "checkpoint"),
                    "--lock", str(root / "checkpoint/lock.json"), "--workload", str(root / "workload.json"),
                    "--devices", "cpu,cpu", "--output", str(root / output), "--stamps", str(root / "stamps"),
                    "--modes", "f32", "--dequant", "f32=rounded", "--no-cache",
                    "--taps", "--tap-sites", "x.latent,attn.out"]))
        cls.scores = {output: json.loads((root / output / "f32.json").read_text())["entries"][0]["scores"]
                      for output in ("fp8", "plain")}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_scores_match_a_plain_forward_with_the_latent_round_trip(self):
        adapter = self.adapters[0]
        model = self.base.plain_model(self.tensors, torch.float32)
        hooked = 0
        for block in model.model.language_model.layers:
            if block.block_type != "linear_attention":
                block.self_attn.kv_a_layernorm.register_forward_hook(
                    lambda _module, _inputs, output: adapter.latent_fp8(output))
                hooked += 1
        self.assertEqual(hooked, 2)
        ids = self.base.PROMPT
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0]
        expected = torch.log_softmax(logits[:-1], dim=-1).gather(1, torch.tensor(ids[1:])[:, None])[:, 0].double()
        got = torch.tensor(self.scores["fp8"]["logprob"], dtype=torch.float64)
        self.assertLess((got - expected).abs().max().item(), 2e-5)
        self.assertEqual(self.scores["fp8"]["top1"], logits[:-1].argmax(dim=-1).tolist())

    def test_the_round_trip_changes_the_scores(self):
        fp8 = torch.tensor(self.scores["fp8"]["logprob"], dtype=torch.float64)
        plain = torch.tensor(self.scores["plain"]["logprob"], dtype=torch.float64)
        self.assertGreater((fp8 - plain).abs().max().item(), 1e-4)

    def taps(self, output):
        """{tap name: rows} of the run's one tapped step."""
        from tools.qual import tapdiff
        (step,) = sorted((self.root / output / "taps/f32").iterdir())
        index = {e["name"]: e for e in map(json.loads, (step / "index.jsonl").read_text().splitlines())}
        return {name: torch.from_numpy(tapdiff.read(step / f"{name}.r0.bin", entry).copy())
                for name, entry in index.items()}

    def test_the_latent_tap_is_read_before_the_fp8_write(self):
        """The served x.latent is the normalized latent before LatentKvWrite
        encodes it; the adapter's must be too, with the round trip only on
        the rows the attention expands. In the first sparse layer nothing
        upstream differs, so its x.latent equals the plain run's, while its
        attention output and the next sparse layer's latent carry the FP8
        effect."""
        fp8, plain = self.taps("fp8"), self.taps("plain")
        first, second = sorted(name for name in plain if name.endswith(".x.latent"))
        self.assertEqual((first, second), ("L01.x.latent", "L03.x.latent"))
        self.assertTrue(torch.equal(fp8[first], plain[first]))
        # The round trip does change this latent: the tap is not after it.
        self.assertFalse(torch.equal(fp8[first], self.adapters[0].latent_fp8(plain[first])))
        self.assertFalse(torch.equal(fp8["L01.attn.out"], plain["L01.attn.out"]))
        self.assertFalse(torch.equal(fp8[second], plain[second]))

    def test_the_adapter_runs_the_pinned_reference(self):
        self.assertEqual(self.adapters[0].reference.__file__, str(self.base.ADAPTER))


if __name__ == "__main__":
    unittest.main()

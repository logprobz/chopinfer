"""The encoder_parity check (encoder_parity.py taps) on synthetic captures:
an engine capture against encoder reference outputs in the exact, policy
and reference-bf16 modes. Needs numpy (the runner's Python, which also
runs tapdiff); skipped without it."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    import numpy as np
    from tools.qual.test_tapdiff import item, write_step
except ModuleNotFoundError:
    np = None
import common  # noqa: E402
import declaration  # noqa: E402
import encoder_capture  # noqa: E402
import tap_binding  # noqa: E402
import test_tap_binding as bound_taps  # noqa: E402
import encoder_parity  # noqa: E402
import kinds  # noqa: E402

COLS = 16
SEQUENCES = {"pic": list(range(1, 11)), "speech": list(range(20, 30))}


def reference_step(directory, name, tensors, kind):
    """One encoder reference step: encoder rows only, no positions."""
    rows = next(iter(tensors.values()))[0][1].shape[0]
    write_step(directory, [{"request": name, "start": 0, "rows": 0, "prefix": 0, "positions": [], "tokens": []}],
               tensors, encoder={tap: [{"request": name, "start": 0, "kind": kind, "item": 0, "index": 0,
                                        "rows": parts[0][1].shape[0]}] for tap, parts in tensors.items()})
    return rows


def describe(reference, declaration, taps):
    """An encoder reference output's run.json and policy mode record: the
    declaration it ran and its precision policy."""
    path = common.ROOT / declaration
    (reference / "run.json").write_text(json.dumps({
        "declaration": {"path": declaration, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "modes": {mode: {"taps": list(taps)} for mode in ("exact", "policy", "reference-bf16")}}))
    (reference / "policy.json").write_text(json.dumps({"mode": {"precision": json.loads(path.read_text())["precision"]}}))


def captured(run, model, requests):
    """`run` as a tap capture of `requests` ([(sequence name, steps)], where
    steps(engine request ID) gives the (items, tensors, encoder) steps the
    request writes) from a bound fake diagnostic-taps server of `model`."""
    plans = dict(requests)
    taps = run.parent / f"{run.name}-taps"
    taps.mkdir()
    identity = bound_taps.serving(model)
    log = bound_taps.server_log(run.parent / f"{run.name}.log", identity)
    fake = bound_taps.FakeServer(taps, log, {}, [], serving=identity,
                                 steps_of=lambda engine, body: plans[body["prompt"]](engine))
    with bound_taps.serve(fake):
        tap_binding.capture(bound_taps.URL, log, taps, [(name, "/v1/completions", {"prompt": name})
                                                        for name, _ in requests],
                            run, declaration.load(declaration.path(model)))
    return identity


MODEL = "mimo26"
AUDIO_ENCODER = f"models/{MODEL}/contract/encoders/mimo-audio.json"
AUDIO_REASON = "served, not qualified in-tree"


def capabilities_with(model, audio):
    """`model`'s declared capabilities with audio qualified (`audio` true) or
    served but unqualified, whichever the declaration states."""
    capabilities = dict(declaration.load(declaration.named(model))["capabilities"])
    media = [kind for kind in capabilities["media"] if kind != "audio"]
    unqualified = {kind: reason for kind, reason in capabilities.get("unqualified_media", {}).items()
                   if kind != "audio"}
    if audio:
        media.append("audio")
    else:
        unqualified["audio"] = AUDIO_REASON
    capabilities["media"] = media
    capabilities.pop("unqualified_media", None)
    return {**capabilities, **({"unqualified_media": unqualified} if unqualified else {})}


def judged(run, identity=None, model=MODEL, capabilities=None):
    """encoder_parity taps --run `run` against a live candidate with
    `identity`, under the declaration with `capabilities` when given:
    (exit status, receipt)."""
    candidate = bound_taps.FakeServer(run, run / "unused.log", {}, [], serving=identity or bound_taps.serving(model))
    argv = ["encoder_parity.py", "taps", "--run", str(run), "--url", bound_taps.URL]
    declared = common.qualification_config
    config = (declared if capabilities is None else
              lambda: {**declared(), "capabilities": capabilities})
    # These fixtures exercise numerical rules and live tap binding with
    # synthetic references. Production reference admission has separate tests.
    with bound_taps.serve(candidate), mock.patch.object(sys, "argv", argv), \
            mock.patch.object(encoder_capture, "configured", return_value=False), \
            mock.patch.object(encoder_parity, "qualification_config", config), \
            mock.patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(declaration.named(model))}), \
            contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        status = encoder_parity.main()
    receipts = sorted((common.RECEIPTS).glob("*-encoder-parity.json"))
    receipt = json.loads(receipts[-1].read_text())
    for path in receipts:
        path.unlink()
    return status, receipt


@unittest.skipIf(np is None, "needs numpy (set CHOPINFER_REFRUN_PYTHON)")
class ParityGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        patch = mock.patch.object(common, "RECEIPTS", self.root / "receipts")
        patch.start()
        self.addCleanup(patch.stop)
        rng = np.random.default_rng(3)
        self.vision = rng.standard_normal((6, COLS)).astype(np.float32)
        self.audio = rng.standard_normal((3, COLS)).astype(np.float32)
        self.codes = rng.integers(0, 1024, (10, 20)).astype(np.float64)
        self.rng = rng

    def noisy(self, values, scale):
        return (values * (1 + scale * self.rng.standard_normal(values.shape))).astype(np.float32)

    def moved(self, codes, rows):
        out = codes.copy()
        out[:rows, 1:] += 1
        return out

    def write_run(self, vision, audio, codes, with_codes=True, later_chunk=False, with_audio=True):
        run = self.root / "run"
        modes = {"exact": (self.vision, self.audio, self.codes),
                 "reference-bf16": (self.noisy(self.vision, 4e-3), self.noisy(self.audio, 4e-3),
                                    self.moved(self.codes, 3)),
                 "policy": (self.noisy(self.vision, 4e-3), self.noisy(self.audio, 4e-3), self.moved(self.codes, 3))}
        tensors = {"enc.audio.out": (audio, "f32")}
        if with_codes:
            tensors["enc.audio.x.codes"] = (codes, "u32")

        def pic(engine):
            steps = [([item(engine, 0, list(range(10)), SEQUENCES["pic"])], {"enc.vision.out": (vision, "f32")},
                      {"enc.vision.out": [{"request": engine, "start": 0, "kind": "image", "item": 0, "index": 0,
                                           "rows": 6}]})]
            if later_chunk:
                # A later prefill chunk: it reads only rows its encode's step
                # tapped, so it captures nothing.
                steps.append(([item(engine, 0, list(range(10, 14)), [1, 2, 3, 4])], {}, None))
            return steps

        def speech(engine):
            encoder = {"enc.audio.out": [{"request": engine, "start": 0, "kind": "audio", "item": 0, "index": 0,
                                          "rows": 3}]}
            if with_codes:
                encoder["enc.audio.x.codes"] = [{"request": engine, "start": 0, "kind": "audio", "item": 0,
                                                 "index": 0, "rows": 10}]
            return [([item(engine, 0, list(range(10)), SEQUENCES["speech"])], tensors, encoder)]
        captured(run, MODEL, [("pic", pic), ("speech", speech)] if with_audio else [("pic", pic)])
        for label, (v, a, c) in modes.items():
            reference_step(run / "reference/vision/taps" / label / "step0000", "pic",
                           {"enc.vision.out": [(0, v, "f32")]}, "image")
            if with_audio:
                reference_step(run / "reference/audio/taps" / label / "step0000", "speech",
                               {"enc.audio.out": [(0, a, "f32")], "enc.audio.x.codes": [(0, c, "u32")]}, "audio")
        describe(run / "reference/vision", f"models/{MODEL}/contract/encoders/mimo-vision.json", ["enc.vision.out"])
        references = ["reference/vision"]
        if with_audio:
            describe(run / "reference/audio", AUDIO_ENCODER, ["enc.audio.out", "enc.audio.x.codes"])
            references.append("reference/audio")
        (run / "workload.json").write_text(json.dumps({"sequences": [
            {"name": name, "token_ids": tokens} for name, tokens in SEQUENCES.items()
            if with_audio or name == "pic"]}))
        (run / "parity.json").write_text(json.dumps({
            "schema": encoder_parity.PARITY_RUN, "model": MODEL, "engine": "engine", "workload": "workload.json",
            "references": references, "source": "0" * 40}))
        return run

    def check(self, *arguments, audio=True):
        """The gate over a run, under the declaration with audio qualified
        (`audio`) or served but unqualified."""
        return judged(self.write_run(*arguments), capabilities=capabilities_with(MODEL, audio))

    def test_generated_receipt_satisfies_parent_item_coverage(self):
        status, receipt = self.check(self.vision, self.audio, self.codes)
        self.assertEqual(status, 0)
        check = next(check for check in declaration.load(declaration.path(receipt['server']['model']))['checks']
                     if check['id'] == 'encoder_parity')
        kind = kinds.of(check['module'])
        scored = kinds.rescore(kind, receipt, None, check)
        self.assertEqual(scored['status'], 'pass', scored)
        self.assertEqual(receipt['rows'], sum(tap['rows'] for tap in receipt['taps'].values()))
        self.assertEqual(receipt['rows'], 19)
        for rows in (None, 0, [], False):
            bad = dict(receipt)
            if rows is None:
                bad.pop('rows')
            else:
                bad['rows'] = rows
            with self.subTest(rows=rows):
                self.assertNotEqual(kinds.rescore(kind, bad, None, check)['status'], 'pass')
        bad = {**receipt, 'passed': False, 'errors': ['a required tap failed']}
        self.assertNotEqual(kinds.rescore(kind, bad, None, check)['status'], 'pass')

    def test_rows_inside_their_floors_and_codes_inside_the_margin_pass(self):
        status, receipt = self.check(self.noisy(self.vision, 2e-3), self.noisy(self.audio, 2e-3),
                                     self.moved(self.codes, 3))
        self.assertEqual((status, receipt["passed"], receipt["errors"]), (0, True, []))
        self.assertEqual(sorted(receipt["taps"]), ["enc.audio.out", "enc.audio.x.codes", "enc.vision.out"])
        vision = receipt["taps"]["enc.vision.out"]
        self.assertEqual(sorted(vision["direction"]["e_over_floor"]), ["p100", "p50", "p90"])
        scale = vision["scale"]["pic/image/0"]
        self.assertEqual((scale["limit"], scale["passed"], sorted(scale)),
                         (0.003, True, ["bf16", "engine", "engine_median", "limit", "passed", "policy"]))
        self.assertEqual(receipt["taps"]["enc.audio.out"]["code_flipped_rows"], 1)
        codes = receipt["codes"]
        self.assertEqual((codes["engine"]["agreement"], codes["vendor_bf16"]["agreement"],
                          codes["legacy_policy"]["agreement"], codes["margin"]), (0.715, 0.715, 0.715, 0.03))
        self.assertEqual(receipt["rule"], {"ratio": 3.0, "row_floor": 2.0 ** -8, "codes_margin": 0.03,
                                           "scale": "f32-policy", "reference": "exact",
                                           "spread": "reference-bf16"})

    def test_a_later_chunk_without_encoder_rows_is_not_read(self):
        status, receipt = self.check(self.noisy(self.vision, 2e-3), self.noisy(self.audio, 2e-3),
                                     self.moved(self.codes, 3), True, True)
        self.assertEqual((status, receipt["passed"], receipt["tapdiff_status"]), (0, True, "match"))

    def test_one_corrupt_row_fails(self):
        vision = self.noisy(self.vision, 2e-3)
        vision[4] = -vision[4]
        status, receipt = self.check(vision, self.noisy(self.audio, 2e-3), self.codes)
        self.assertEqual((status, receipt["passed"]), (1, False))
        self.assertEqual(receipt["taps"]["enc.vision.out"]["direction"]["every_row"]["exceeding"], 1)

    def test_codes_below_the_vendor_less_the_margin_fail(self):
        status, receipt = self.check(self.noisy(self.vision, 2e-3), self.noisy(self.audio, 2e-3),
                                     self.moved(self.codes, 4))
        self.assertEqual((status, receipt["passed"]), (1, False))
        self.assertEqual(receipt["codes"]["engine"]["agreement"], 0.62)

    def test_a_capture_without_codes_fails(self):
        status, receipt = self.check(self.vision, self.audio, self.codes, False)
        self.assertEqual((status, receipt["passed"]), (1, False))
        self.assertIn("enc.audio.x.codes: no compared rows (reference only)", receipt["errors"])

    def test_qualified_audio_without_rows_fails(self):
        status, receipt = self.check(self.vision, self.audio, self.codes, True, False, False)
        self.assertEqual((status, receipt["passed"], receipt["unqualified"]), (1, False, {}))
        self.assertIn("enc.audio.out: no compared rows (absent)", receipt["errors"])

    def test_unqualified_audio_needs_no_rows_and_is_named(self):
        status, receipt = self.check(self.noisy(self.vision, 2e-3), self.audio, self.codes, True, False, False,
                                     audio=False)
        self.assertEqual((status, receipt["passed"], receipt["errors"]), (0, True, []))
        self.assertEqual(receipt["unqualified"], {AUDIO_ENCODER: AUDIO_REASON})
        self.assertEqual((sorted(receipt["taps"]), receipt["codes"]), (["enc.vision.out"], None))

    def test_vision_rows_stay_required_when_audio_is_unqualified(self):
        vision = self.noisy(self.vision, 2e-3)
        vision[4] = -vision[4]
        status, receipt = self.check(vision, self.audio, self.codes, True, False, False, audio=False)
        self.assertEqual((status, receipt["passed"]), (1, False))
        self.assertEqual(receipt["taps"]["enc.vision.out"]["direction"]["every_row"]["exceeding"], 1)


@unittest.skipIf(np is None, "needs numpy (set CHOPINFER_REFRUN_PYTHON)")
class F32ScaleTests(unittest.TestCase):
    """An F32-policy engine's item scale is held to its policy run's, at least
    0.1%; the vendor BF16 run's larger scale errors set no limit."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        patch = mock.patch.object(common, "RECEIPTS", self.root / "receipts")
        patch.start()
        self.addCleanup(patch.stop)
        environment = mock.patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(declaration.named("qwen38"))})
        environment.start()
        self.addCleanup(environment.stop)
        self.vision = np.random.default_rng(5).standard_normal((6, COLS)).astype(np.float32)

    def check(self, scale, declared="models/qwen38/contract/encoders/qwen38-vision.json", model="qwen38"):
        run = self.root / f"run-{scale}"
        engine = (self.vision * scale).astype(np.float32)
        captured(run, model, [("pic", lambda request: [
            ([item(request, 0, list(range(10)), SEQUENCES["pic"])], {"enc.vision.out": (engine, "f32")},
             {"enc.vision.out": [{"request": request, "start": 0, "kind": "image", "item": 0, "index": 0,
                                  "rows": 6}]})])])
        # The vendor BF16 run is 6.8% large, as on Qwen3.8's blue-square
        # probe; the F32 policy run 0.05%.
        for label, factor in (("exact", 1.0), ("reference-bf16", 1.068), ("policy", 1.0005)):
            reference_step(run / "reference/vision/taps" / label / "step0000", "pic",
                           {"enc.vision.out": [(0, (self.vision * factor).astype(np.float32), "f32")]}, "image")
        describe(run / "reference/vision", declared, ["enc.vision.out"])
        (run / "workload.json").write_text(json.dumps({"sequences": [{"name": "pic", "token_ids": SEQUENCES["pic"]}]}))
        (run / "parity.json").write_text(json.dumps({
            "schema": encoder_parity.PARITY_RUN, "model": model, "engine": "engine", "workload": "workload.json",
            "references": ["reference/vision"], "source": "0" * 40}))
        self.run = run
        return judged(run, model=model)

    def test_a_capture_another_process_wrote_or_a_moved_step_is_insufficient(self):
        status, receipt = self.check(1.0003)
        self.assertEqual(status, 0)
        status, receipt = judged(self.run, bound_taps.serving("qwen38") | {"startup_nonce": "b" * 32}, "qwen38")
        self.assertEqual((status, receipt["passed"]), (1, False))
        self.assertIn("startup_nonce", receipt["errors"][0])
        engine = self.run / "engine"
        (engine / "step0000").rename(engine / "step0007")
        status, receipt = judged(self.run, model="qwen38")
        self.assertIn("other steps than its manifest", receipt["errors"][0])

    def test_a_scale_within_the_policy_run_passes(self):
        status, receipt = self.check(1.0003)
        self.assertEqual((status, receipt["rule"]["scale"]), (0, "f32-policy"))
        scale = receipt["taps"]["enc.vision.out"]["scale"]["pic/image/0"]
        self.assertEqual(scale["limit"], 0.001)
        self.assertAlmostEqual(scale["bf16"], 0.068, places=5)

    def test_one_and_five_percent_whole_row_regressions_fail(self):
        for factor in (1.01, 1.05, 0.95):
            status, receipt = self.check(factor)
            self.assertEqual((status, receipt["passed"]), (1, False), factor)
            self.assertFalse(receipt["taps"]["enc.vision.out"]["scale"]["pic/image/0"]["passed"])

    def test_a_bf16_policy_keeps_the_spread_rule(self):
        # The same 5% engine under a BF16 policy (MiMo's before the 2026-10-02 F32 change): the
        # vendor BF16 run's 6.8% sets the limit.
        legacy = json.loads((common.ROOT / "models/mimo26/contract/encoders/mimo-vision.json").read_text())
        legacy["precision"] = "bf16-legacy.v1"
        declared = self.root / "mimo-vision-bf16-legacy.json"
        declared.write_text(json.dumps(legacy))
        with mock.patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(declaration.named("mimo26"))}):
            status, receipt = self.check(1.05, str(declared), "mimo26")
        self.assertEqual(receipt["rule"]["scale"], "bf16-spread")
        self.assertEqual(receipt["taps"]["enc.vision.out"]["scale"]["pic/image/0"]["limit"], 0.07)

    def test_a_declaration_other_than_the_one_run_is_refused(self):
        reference = self.root / "reference"
        reference.mkdir()
        describe(reference, "models/qwen38/contract/encoders/qwen38-vision.json", ["enc.vision.out"])
        value = json.loads((reference / "run.json").read_text())
        value["declaration"]["sha256"] = "0" * 64
        (reference / "run.json").write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "differs from the declaration it ran"):
            encoder_parity.precision(reference)


if __name__ == "__main__":
    unittest.main()

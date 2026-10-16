import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import qwen38_hc_linear_e5 as e5


class E5Test(unittest.TestCase):
    def test_no_go_refuses_before_any_environment_or_device_access(self):
        result = subprocess.run([sys.executable, "-B", e5.__file__, "run"],
                                env={}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("run requires --go, --device and --end-time", result.stderr)

    def test_assignment_is_one_gpu_and_deadline_is_bounded(self):
        for device in ("0,1", "4", "GPU-unknown"):
            with self.assertRaisesRegex(ValueError, "one explicitly assigned"):
                e5.run(Path("/unread"), {}, device, "unused")
        now = datetime(2026, 9, 26, tzinfo=timezone.utc).timestamp()
        with patch.object(e5.time, "monotonic", return_value=100):
            self.assertEqual(e5.deadline("2026-09-26T00:40:00Z", now), 1000)
            self.assertEqual(e5.deadline("2026-09-26T00:40:00Z", now, mode="ncu"), 1300)
            self.assertEqual(e5.deadline("2026-09-26T00:15:00Z", now), 1000)
        for end in ("2026-09-26T00:09:00Z", "2026-09-26T00:40:00", "2026-09-26T00:40:00+01:00"):
            with self.assertRaises(ValueError):
                e5.deadline(end, now)

    def test_counts_cannot_hide_a_missing_shape_or_shared_device(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {"qwen38-hc-e5": [(rows, mode) for rows in (16, 32) for mode in ("in", "mix", "chain")],
                     "qwen38-linear-e5": [(shape, mode) for shape in ("16x1", "32x1", "16x2")
                                           for mode in ("conv", "delta", "norm") if (shape, mode) != ("16x2", "norm")]}
            for kind, entries in cases.items():
                folder = root / "receipts" / kind
                folder.mkdir(parents=True)
                for index, (shape, mode) in enumerate(entries):
                    record = {"measurement_label": "exclusive-device", "eager_graph_bits_equal": True,
                              "median_us_per_call": 12, "mode": mode,
                              "rows" if kind == "qwen38-hc-e5" else "shape": shape}
                    (folder / f"{index}.json").write_text(json.dumps(record))
            gate_dir = root / "receipts/qwen38-hc-up-mix-g1"
            gate_dir.mkdir()
            gate = gate_dir / "gate.json"
            gate.write_text(json.dumps({"measurement_label": "exclusive-device",
                                        "graph_determinism_pdl": True, "lives": [16, 32]}))
            bench = root / "receipts/bench"
            bench.mkdir()
            for operation in ("lin_in", "out"):
                (bench / f"{operation}.json").write_text(json.dumps({
                    "symbol": f"chop_gemm_qwen38_{operation}_c32_v1", "m": 32,
                    "measurement_label": "exclusive-device", "chain_us_per_call": 12}))
            self.assertEqual(e5.check_receipts(root), {"qwen38-hc-e5": 6, "qwen38-linear-e5": 8,
                                                     "hc_gate": 1, "projections": 2})
            saved_gate = gate.read_text()
            gate.unlink()
            with self.assertRaisesRegex(ValueError, "HC unfused-chain gate"):
                e5.check_receipts(root)
            gate.write_text(saved_gate)
            projection = bench / "out.json"
            saved_projection = projection.read_text()
            projection.unlink()
            with self.assertRaisesRegex(ValueError, "missing 32-row projection"):
                e5.check_receipts(root)
            projection.write_text(saved_projection)
            first = root / "receipts/qwen38-hc-e5/0.json"
            duplicate = first.with_name("1.json")
            saved = duplicate.read_text()
            duplicate.write_text(first.read_text())
            with self.assertRaisesRegex(ValueError, "duplicate"):
                e5.check_receipts(root)
            duplicate.write_text(saved)
            record = json.loads(first.read_text())
            record["measurement_label"] = "shared-device"
            first.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "shared-device"):
                e5.check_receipts(root)

    def test_no_counters_orchestration_never_invokes_ncu(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            binary = base / "fake-binary"
            binary.write_bytes(b"prepared")
            identity = {"path": str(binary), "sha256": e5.digest(binary)}
            prepared = {"mode": "no-counters", "binaries": {name: identity for name in e5.TESTS},
                        "diagnostics": dict(identity, evidence_path=str(base)), "ncu": "/must-not-run-ncu"}
            (base / "e5-prepared.json").write_text(json.dumps(prepared))
            calls = []
            def fake_call(command, env, log, finish, limit=240):
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-assigned-device")
                self.assertEqual(env["CHOPINFER_KERNEL_GPU"], "0")
                self.assertNotIn("CHOPINFER_KERNEL_GPUS", env)
                self.assertEqual(env["CHOPINFER_E5_DIAGNOSTIC_MANIFEST"], str(base / "manifest.json"))
                calls.append(command)
                if "--manifest" in command:
                    log.write_text(json.dumps(command) + "\n{}\n")
            end = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
            with patch.object(e5, "idle", return_value="GPU-assigned-device"), patch.object(e5, "call_owned", side_effect=fake_call), \
                 patch.object(e5, "check_receipts", return_value={}), \
                 patch.object(e5.diagnostics, "decide", return_value={"diagnostic_receipts": 42}):
                e5.run(base, {"CHOPINFER_KERNEL_GPUS": "0,1,2,3"}, "2", end)
            self.assertEqual(len(calls), 18)
            self.assertFalse(any("ncu" in str(arg) for command in calls for arg in command))
            root = next(base.glob("e5-20*"))
            self.assertEqual(json.loads((root / "summary.json").read_text())["mode"], "no-counters")

    def test_timeout_collects_owned_process_and_leaves_unrelated_process(self):
        unrelated = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(30)"])
        try:
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(subprocess.TimeoutExpired):
                    e5.call_owned([sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                                  {}, Path(directory) / "owned.log", time.monotonic() + 45.2)
                self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait()


if __name__ == "__main__":
    unittest.main()

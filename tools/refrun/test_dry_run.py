"""The dry run accepts exactly the command lines the runner accepts, and
refuses on CPU what the runner would refuse before loading weights. Needs
torch and transformers; skipped under a Python without them."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

try:
    import torch
    from tools.refrun import dry_run, keys, run, tiny
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "needs torch (set CHOPINFER_REFRUN_PYTHON)")
class DryRunTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.checkpoint = tiny.write_checkpoint(self.root / "checkpoint")
        workload = self.root / "workload.json"
        workload.write_text(json.dumps({"schema": "chopinfer.refrun.workload.v1", "model": "tiny", "sequences": [
            {"name": "text", "category": "prose", "token_ids": [1, 5, 9, 33], "score_from": 1}]}))
        self.arguments = [
            "--adapter", str(keys.ROOT / "tools/refrun/tiny.py"), "--checkpoint", str(self.checkpoint),
            "--lock", str(self.checkpoint / tiny.LOCK), "--workload", str(workload),
            "--modes", "f32,bf16", "--dequant", "f32=rounded", "--devices", "cuda:0,cuda:1",
            "--output", str(self.root / "runs" / "out"), "--stamps", str(self.root / "stamps")]

    def dry(self, argv, visible="0,1"):
        output, error = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            try:
                dry_run.main(argv)
            except SystemExit as stopped:
                return stopped.code, output.getvalue() + error.getvalue()
        return 0, output.getvalue()

    def test_the_run_command_line_is_checked_without_loading_weights(self):
        with mock.patch("tools.refrun.place.Store", side_effect=AssertionError("weights loaded")):
            code, text = self.dry(["run", *self.arguments])
        self.assertEqual(code, 0, text)
        result = json.loads(text)
        self.assertEqual(result["runner"], str(Path(run.__file__).resolve()))
        self.assertEqual(result["modes"], {"f32": "rounded", "bf16": "hf-bf16"})
        self.assertEqual((result["model"], result["sequences"], result["devices"]), ("tiny", 1, ["cuda:0", "cuda:1"]))
        self.assertEqual(set(result["stored_gib"]), {"cuda:0", "cuda:1"})
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse(torch.cuda.is_initialized())

    def test_a_command_line_without_the_run_subcommand_is_refused(self):
        code, text = self.dry(self.arguments)
        self.assertEqual(code, 2)
        self.assertIn("argument command: invalid choice", text)

    def test_other_subcommands_are_parsed_but_not_run(self):
        with mock.patch.object(run, "verify_binding", side_effect=AssertionError("ran")) as command:
            code, text = self.dry(["verify-binding", str(self.root / "fixture.json")])
            self.assertIs(run.verify_binding, command)
        self.assertIn("checks the run subcommand", str(code))

    def test_what_the_runner_refuses_before_loading_is_refused(self):
        (self.root / "exists").mkdir()
        tampered = self.root / "tampered"
        tampered.mkdir()
        for file in self.checkpoint.iterdir():
            (tampered / file.name).write_bytes(file.read_bytes())
        config = json.loads((tampered / "config.json").read_text())
        config["rms_norm_eps"] = 1e-5
        (tampered / "config.json").write_text(json.dumps(config))

        def changed(option, value):
            arguments = list(self.arguments)
            arguments[arguments.index(option) + 1] = value
            return ["run", *arguments]

        for argv, visible, message in (
                (changed("--modes", "bf16"), "0,1", "--dequant names modes not run"),
                (changed("--modes", "f32,f99"), "0,1", "unknown mode f99"),
                (changed("--devices", "cuda:0,cuda:1,cuda:2,cuda:3"), "0,1", "makes 2 visible"),
                (changed("--devices", "gpu0"), "0,1", "neither cpu nor cuda:N"),
                (changed("--output", str(self.root / "exists")), "0,1", "exists; the runner creates it"),
                (changed("--checkpoint", str(tampered)), "0,1", "does not match the checkpoint lock"),
                (changed("--workload", str(self.root / "missing.json")), "0,1", "missing.json")):
            with self.subTest(message):
                code, text = self.dry(argv, visible)
                self.assertIn("dry run refused", str(code))
                self.assertIn(message, str(code))


if __name__ == "__main__":
    unittest.main()

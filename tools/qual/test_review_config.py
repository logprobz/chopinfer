"""Refuse mismatched scoring policies and invalid startup settings."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import common
import declaration
import prefix_exact
import verify_exact
import soak
import sys


class ReviewConfigTests(unittest.TestCase):
    def test_checkout_local_env_precedes_user_override(self):
        import shutil
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "tools/qual"
            tools.mkdir(parents=True)
            for name in ("common.py", "declaration.py"):
                shutil.copyfile(common.ROOT / "tools/qual" / name, tools / name)
            checkout = root / "local.env"
            checkout.write_text("export CHOPINFER_DATA=checkout-data\n")
            config = root / ".config/chopinfer"
            config.mkdir(parents=True)
            (config / "local.env").write_text("export CHOPINFER_DATA=user-data\n")
            env = {key: value for key, value in os.environ.items() if not key.startswith("CHOPINFER_")}
            env.update(HOME=str(root), XDG_CONFIG_HOME=str(root / ".config"))
            command = [sys.executable, "-B", "-c",
                       "import common; print(common.os.environ.get('CHOPINFER_DATA'))"]
            for present in (True, False):
                if not present:
                    checkout.unlink()
                for selected in (None, root / "missing.env", config / "local.env"):
                    with self.subTest(checkout=present, selected=selected):
                        env.pop("CHOPINFER_LOCAL_ENV", None)
                        if selected is not None:
                            env["CHOPINFER_LOCAL_ENV"] = str(selected)
                        expected = "checkout-data" if present else (
                            "None" if selected == root / "missing.env" else "user-data")
                        result = subprocess.run(command, cwd=tools, env=env,
                                                capture_output=True, text=True, check=True)
                        self.assertEqual(result.stdout.strip(), expected)

    def test_empty_input_environment_is_unset(self):
        config = {"receipts": {"env": "CHOPINFER_DATA", "path": "qual"},
                  "inputs": {"optional": {"env": "CHOPINFER_DATA", "path": "optional"},
                             "default": {"env": "CHOPINFER_DATA", "default": "/fallback", "path": "input"}}}
        with patch.dict(os.environ, {"CHOPINFER_DATA": ""}, clear=True):
            with self.assertRaisesRegex(ValueError, "needs CHOPINFER_DATA"):
                common.input_path("receipts", config=config)
            self.assertIsNone(common.input_path("optional", optional=True, config=config))
            self.assertEqual(common.input_path("default", config=config), Path("/fallback/input"))

    def test_wrong_reference_is_refused_before_scoring(self):
        with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(
                declaration.path("mimo26"))}):
            from test_prefix_exact import aligned_rows, qwen_reference
            ids = list(range(20, 30))
            # The faulty hit passed all three inherited checks under the wrong
            # policy. The reference must be refused before choosing any checks.
            recorded = {"positions": aligned_rows(ids, -1.10, -1.00, -1.05),
                        "detokenized": {}, "family": "qwen38"}
            with self.assertRaisesRegex(ValueError, "reference schema"):
                prefix_exact.evaluate(recorded, qwen_reference(ids, -1.00))

    def test_record_family_is_refused_before_scoring(self):
        with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(
                declaration.path("qwen38"))}):
            with self.assertRaisesRegex(ValueError, "recorded family"):
                prefix_exact.evaluate({"positions": [], "detokenized": {}, "family": "mimo26"},
                    {"schema": "chopinfer.qwen38.teacher-reference.v2", "texts": []})

    def test_model_settings_are_required_at_argument_parsing(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "CHOPINFER_QUAL_CONFIG"):
                common.arguments("model facts", model_facts=True).parse_args([])

    def test_neutral_tool_checks_receipt_writability_at_startup(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {"CHOPINFER_DATA": root}):
                common.arguments("neutral").parse_args([])
                self.assertEqual(common.receipt_root(), Path(root) / "qual")
            path = Path(root) / "file"
            path.write_text("not a directory")
            with patch.dict(os.environ, {"CHOPINFER_QUAL_RECEIPTS": str(path)}):
                with self.assertRaises(OSError):
                    common.arguments("neutral").parse_args([])

    def test_soak_refuses_missing_model_settings_before_constructing_client(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "argv", [
                "soak", "--remote-only", "--structured-mix", "--duration", "1200"]), \
             patch.object(soak, "Client", side_effect=AssertionError("client constructed before config validation")) as client:
            with self.assertRaisesRegex(ValueError, "CHOPINFER_QUAL_CONFIG"):
                soak.main()
            client.assert_not_called()

    def test_transport_client_has_no_receipt_requirement(self):
        # Embedders such as release_gate own their output directory separately.
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(common.Client("http://offline").url, "http://offline")

    def test_audio_offline_analysis_needs_no_serving_config(self):
        from dev import audio_check
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(sys, "argv", ["audio_check", "layers", "first", "second"]), \
             patch.object(audio_check, "layers", return_value=0) as layers:
            self.assertEqual(audio_check.main(), 0)
            layers.assert_called_once()

    def test_named_exact_does_not_change_process_policy(self):
        with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": "immutable-policy.json",
                                     "CHOPINFER_DATA": "/data/example"}):
            self.assertEqual(verify_exact.exact_path("qwen38"),
                             common.ROOT / "fixtures/qwen38/teacher-reference.json")
            self.assertEqual(os.environ["CHOPINFER_QUAL_CONFIG"], "immutable-policy.json")

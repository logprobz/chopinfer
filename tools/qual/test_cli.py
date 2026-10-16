"""CPU checks for the unified entry point and compatibility paths."""

import hashlib
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cli import COMMANDS
import declaration
from kinds import eval_binary
from qual_test_config import clean_checkout

ROOT = Path(__file__).resolve().parents[2]


class CliTests(unittest.TestCase):
    def run_cli(self, *args, cwd=ROOT, env=None):
        return subprocess.run([sys.executable, "-B", *args], cwd=cwd, env=env,
                              capture_output=True, timeout=120)

    def assert_same(self, command, arguments):
        module = COMMANDS[command]
        old_args = [command, *arguments] if command in ("check", "rescore", "list") else arguments
        old = (subprocess.run(["bash", str(ROOT / "tools/qual/release_preview.sh"), *old_args],
                              cwd=ROOT, capture_output=True, timeout=120)
               if command == "preview" else self.run_cli(f"tools/qual/{module}.py", *old_args))
        new = self.run_cli("-m", "tools.qual", command, *arguments)
        self.assertEqual((new.returncode, new.stdout, new.stderr),
                         (old.returncode, old.stdout, old.stderr), command)
        return new

    def test_every_command_help_matches_script(self):
        for command in COMMANDS:
            with self.subTest(command=command):
                self.assertEqual(self.assert_same(command, ["--help"]).returncode, 0)

    def test_vector_help_is_read_only(self):
        def state():
            return {path.name: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
                    for path in (ROOT / "tests/qual-vectors").glob("*.json")}
        before = state()
        result = self.run_cli("-m", "tools.qual", "vectors", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(state(), before, "help rewrote conformance vectors")

    def test_offline_output_matches(self):
        for model in ("mimo26", "qwen38", "mechanics"):
            with self.subTest(model=model):
                self.assertEqual(self.assert_same("gate", ["--model", model, "--plan"]).returncode, 0)
        self.assertEqual(self.assert_same("list", ["--json"]).returncode, 0)

    def test_reference_lock_verification_matches(self):
        import common
        import refs
        for model in ("mimo26", "qwen38"):
            with self.subTest(model=model):
                config = declaration.load(declaration.path(model))
                paths = [common.declared_path(value) for value in refs.sources(config).values()]
                complete = all(path is not None and path.exists() for path in paths)
                result = self.assert_same("refs", ["verify", "--model", model])
                self.assertEqual(result.returncode, 0 if complete else 2)
                if not complete:
                    self.assertIn(b"missing input", result.stderr)
                    self.assertNotIn(b"Traceback", result.stderr)

    def test_stored_receipt_rescore_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "api.json"
            receipt.write_text(json.dumps({
                "schema": "chopinfer-qualification/v1", "check": "api",
                "passed": True, "errors": [], "skipped": [], "mechanics_only": True,
            }))
            self.assertEqual(self.assert_same(
                "rescore", ["--model", "mechanics", "--require-recorded", str(receipt)]).returncode, 0)

    def test_preview_plan_loads_environment_from_outside_root(self):
        # The tools read the checkout's local.env before CHOPINFER_LOCAL_ENV,
        # so run them from a copy without the checkout's.
        with tempfile.TemporaryDirectory() as temporary, clean_checkout() as root:
            base = Path(temporary)
            data = base / "data"
            data.mkdir()
            tools = base / "bin"
            tools.mkdir()
            capture = base / "environment.json"
            observer = tools / "python3"
            observer.write_text("#!/usr/bin/python3\nimport json, os, sys\n"
                                "from pathlib import Path\n"
                                "Path(os.environ['QUAL_ENV_CAPTURE']).write_text(json.dumps({"
                                "'cwd': os.getcwd(), 'CARGO_HOME': os.environ['CARGO_HOME'], "
                                "'CARGO_NET_OFFLINE': os.environ['CARGO_NET_OFFLINE'], "
                                "'TMPDIR': os.environ['TMPDIR']}))\n"
                                "sys.dont_write_bytecode = True\n"
                                "import runpy\n"
                                "arguments = sys.argv[1:]\n"
                                "if arguments[:1] == ['-B']: arguments = arguments[1:]\n"
                                "sys.argv = arguments\n"
                                # As python3 SCRIPT does, import beside the script.
                                "sys.path.insert(0, str(Path(arguments[0]).resolve().parent))\n"
                                "runpy.run_path(arguments[0], run_name='__main__')\n")
            observer.chmod(0o755)
            env_file = base / "local.env"
            cargo_home = base / "cargo-home"
            cargo_home.mkdir()
            # plan neither builds nor reads this cargo home.
            settings = {"CHOPINFER_DATA": str(data), "CARGO_HOME": str(cargo_home),
                        "CARGO_NET_OFFLINE": "true", "TMPDIR": str(base),
                        "PATH": f"{tools}:/usr/bin:/bin", "QUAL_ENV_CAPTURE": str(capture),
                        "CHOPINFER_EVAL": str(eval_binary().resolve())}
            env_file.write_text("".join(f"export {key}={shlex.quote(value)}\n" for key, value in settings.items()))
            env = {"PATH": "/usr/bin:/bin", "CHOPINFER_LOCAL_ENV": str(env_file),
                   "PYTHONPATH": str(root), "CHOPINFER_QUAL_CONFIG": "exported-policy.json"}
            model = next(iter(declaration.previewed()))
            args = ["plan", "--model", model, "--suite", "full"]
            old = subprocess.run(["bash", str(root / "tools/qual/release_preview.sh"), *args],
                                 cwd=base, env=env, capture_output=True, timeout=120)
            self.assertEqual(old.returncode, 0, old.stderr)
            self.assertIn("default", json.loads(old.stdout))
            capture.unlink()
            new = self.run_cli("-m", "tools.qual", "preview", *args, cwd=base, env=env)
            self.assertEqual((new.returncode, new.stdout, new.stderr),
                             (old.returncode, old.stdout, old.stderr))
            self.assertEqual(json.loads(capture.read_text()), {
                "cwd": str(root), "CARGO_HOME": str(cargo_home), "CARGO_NET_OFFLINE": "true", "TMPDIR": str(base)})

    def test_exported_qualification_config_matches(self):
        env = dict(os.environ, CHOPINFER_QUAL_CONFIG=str(ROOT / "models/qwen38/qual.json"))
        arguments = ["--model", "mimo26", "--plan"]
        old = self.run_cli("tools/qual/release_gate.py", *arguments, env=env)
        new = self.run_cli("-m", "tools.qual", "gate", *arguments, env=env)
        self.assertEqual((new.returncode, new.stdout, new.stderr),
                         (old.returncode, old.stdout, old.stderr))
        self.assertEqual(new.returncode, 0)
        self.assertEqual(json.loads(new.stdout)["model"], "mimo26")

    def test_usage_errors_match(self):
        for command in ("gate", "check", "rescore", "api", "multimodal", "preview"):
            with self.subTest(command=command):
                self.assertEqual(self.assert_same(command, ["--invalid-option"]).returncode, 2)

    def test_retired_preview_entry_points_are_usage_errors(self):
        """The previous preview (replay, `release_preview.sh legacy`) and the
        gate's native quick capture are gone; asking for them is a usage error."""
        self.assertEqual(self.run_cli("-m", "tools.qual", "replay", "--help").returncode, 2)
        legacy = subprocess.run(["bash", str(ROOT / "tools/qual/release_preview.sh"), "legacy", "--help"],
                                cwd=ROOT, capture_output=True, timeout=120)
        self.assertEqual(legacy.returncode, 2, legacy.stderr)
        gate = self.run_cli("tools/qual/release_gate.py", "--model", "mechanics", "--plan", "--eval-binary", "x")
        self.assertEqual(gate.returncode, 2, gate.stderr)

    def test_top_level_help_and_unknown_command(self):
        help_result = self.run_cli("-m", "tools.qual", "--help")
        self.assertEqual(help_result.returncode, 0)
        for command in COMMANDS:
            self.assertIn(command.encode(), help_result.stdout)
        self.assertEqual(self.run_cli("-m", "tools.qual", "unknown").returncode, 2)


if __name__ == "__main__":
    unittest.main()

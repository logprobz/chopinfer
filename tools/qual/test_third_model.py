"""A new text/image model needs declarations, not a loader registration."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import declaration
import kinds

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/qual-third-model/qual.json"


class ThirdModelTests(unittest.TestCase):
    def test_template_validates(self):
        config = declaration.load(ROOT / "tools/qual/templates/qual.json")
        self.assertEqual(config["model"], "sample-vlm")

    def test_new_names_load_and_plan_text_and_image(self):
        for name in ("synthetic-vlm", "another-unregistered-vlm"):
            with self.subTest(model=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model = root / "models" / name
                model.mkdir(parents=True)
                value = json.loads(FIXTURE.read_text())
                value["model"] = name
                path = model / "qual.json"
                path.write_text(json.dumps(value))
                self.assertEqual(declaration.path(name, root), path)
                self.assertIn(path, declaration.shipped(root))
                config = declaration.load(path)
                self.assertEqual(config["model"], name)
                self.assertEqual([kinds.of(c["module"]).name for c in config["checks"]],
                                 ["teacher_forced", "media_forced"])
                self.assertEqual(config["qualification"]["media"]["expectations"]["kinds"],
                                 {"image": {"field": "image_expected", "single": True}})
                for cohort in ("quick", "full"):
                    result = subprocess.run(
                        [sys.executable, "-B", str(ROOT / "tools/qual/release_gate.py"),
                         "--config", str(path), "--model-root", str(root),
                         "--profile", cohort, "--plan"],
                        text=True, capture_output=True, check=False)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    plan = json.loads(result.stdout)
                    self.assertEqual(plan["model"], name)
                    self.assertEqual(plan["selected_checks"], ["teacher", "image"])
                    self.assertEqual([row["module"] for row in plan["checks"]],
                                     ["teacher", "multimodal"])
                    self.assertFalse(plan["qualified"])
                    self.assertEqual(plan["execution"], "not_run")
                self.assertEqual(sorted(p.relative_to(root).as_posix()
                                        for p in root.rglob("*") if p.is_file()),  # check-inputs: walks outside the repository
                                 [f"models/{name}/qual.json"])

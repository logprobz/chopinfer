"""Model declarations (qual.json): validation, compilation and refusal."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import common
import declaration
from qual_test_config import inventory, released
import release_gate


def declared(name="qwen38"):
    """A model's declaration without its configurations, which a variant
    built from it cannot declare."""
    value = json.loads(declaration.path(name).read_text())
    value.pop("configurations", None)
    return value


class DeclarationTests(unittest.TestCase):
    def test_materialization_does_not_create_a_missing_input_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "missing-data"
            count = 0
            with patch.dict(os.environ, {"CHOPINFER_DATA": str(data)}), \
                    patch("tempfile.gettempdir", return_value=str(root)):
                for model in inventory.shipped(declaration.ROOT):
                    for name in declaration.configurations(model):
                        path = declaration.materialized(model, name)
                        self.assertTrue(path.is_file())
                        self.assertFalse(data.exists())
                        count += 1
            self.assertGreater(count, 0)

    def test_shipped_declarations_compile_to_valid_gate_policies(self):
        for path in declaration.shipped()[1:]:
            with self.subTest(model=path.parent.name):
                value = json.loads(path.read_text())
                config = declaration.load(path)
                release_gate.validate(config)
                self.assertEqual(config["model"], path.parent.name)
                self.assertEqual(config["profiles"], value["cohorts"])
                self.assertEqual([check["id"] for check in config["checks"]], list(value["checks"]))
                self.assertNotIn("references", value["qualification"])
                if released(path):
                    self.released_facts(value, config)
                classes = (declaration.change_classes(path.parent.name) if path.name == declaration.NAME
                           else value["change_classes"])
                for change, names in classes.items():
                    self.assertTrue(set(names) <= set(value["checks"]), change)

    def released_facts(self, value, config):
        self.assertEqual(config["qualification"]["references"], value["references"])
        self.assertIn("soak", value["checks"])
        soak = next(check for check in config["checks"] if check["id"] == "soak")
        self.assertEqual(soak["profile_args"], value["checks"]["soak"]["cohort_args"])
        self.assertEqual(list(soak)[:2], ["id", "module"])

    def test_a_released_declaration_missing_a_fact_still_fails(self):
        path = next(path for path in declaration.shipped()[1:] if released(path))
        value = json.loads(path.read_text())
        config = declaration.load(path)
        self.released_facts(value, config)
        del value["checks"]["soak"]
        with self.assertRaises(AssertionError):
            self.released_facts(value, config)

    def test_policy_layouts_pass_through_unchanged(self):
        mechanics = json.loads(declaration.MECHANICS.read_text())
        self.assertEqual(declaration.load(declaration.MECHANICS), mechanics)
        self.assertEqual(declaration.named("mechanics"), declaration.MECHANICS)
        self.assertEqual(declaration.named("qwen38"), declaration.path("qwen38"))

    def test_invalid_declarations_name_the_field(self):
        def drop(value, *keys):
            for key in keys[:-1]:
                value = value[key]
            del value[keys[-1]]

        cases = {
            "schema": lambda d: d.update(schema="chopinfer-qual-model/v0"),
            "gates": lambda d: d.update(gates={}),
            "model": lambda d: d.update(model=""),
            "checks.api.timeout_seconds": lambda d: drop(d, "checks", "api", "timeout_seconds"),
            "checks.api.module": lambda d: d["checks"]["api"].update(module="no_such_module"),
            "checks.api.rules[0].op": lambda d: d["checks"]["api"]["rules"][0].update(op="lt"),
            "checks.api.rules[0].source": lambda d: d["checks"]["api"]["rules"][0].update(source=""),
            "checks.api.profile_args": lambda d: d["checks"]["api"].update(profile_args={}),
            "checks.soak.cohort_args.nightly": lambda d: d["checks"]["soak"]["cohort_args"].update(nightly=[]),
            "checks.teacher.constants.teacher.MAX_MEAN.source": (
                lambda d: drop(d, "checks", "teacher", "constants", "teacher", "MAX_MEAN", "source")),
            "cohorts.quick.checks": lambda d: d["cohorts"]["quick"]["checks"].append("absent"),
            # A cohort has no overall deadline; each check has its own timeout.
            "cohorts.full.budget_seconds": lambda d: d["cohorts"]["full"].update(budget_seconds=600),
            "change_classes.quality": lambda d: d["change_classes"]["quality"].append("absent"),
            "references.teacher.sha256": lambda d: d["references"]["teacher"].update(sha256="abc"),
            "qualification.references": lambda d: d["qualification"].update(references={}),
            "inputs.gsm8k.env": lambda d: drop(d, "inputs", "gsm8k", "env"),
            "receipts": lambda d: drop(d, "receipts"),
            # Change classes: CI selects gate checks by class, so none may be missing.
            "change_classes": lambda d: drop(d, "change_classes"),
            "change_classes.frontend": lambda d: drop(d, "change_classes", "frontend"),
            "change_classes.frontned": lambda d: d["change_classes"].update(frontned=["api"]),
            # Nested policy values are typed where they are declared.
            "checks.teacher.constants.teacher.MAX_MEAN.value": (
                lambda d: d["checks"]["teacher"]["constants"]["teacher"]["MAX_MEAN"].update(value="zero")),
            "checks.teacher.constants.teacher.NO_SUCH_LIMIT": (
                lambda d: d["checks"]["teacher"]["constants"]["teacher"].update(
                    NO_SUCH_LIMIT={"value": 1.0, "source": "tools/qual/README.md"})),
            "checks.teacher.constants.no_such_module": (
                lambda d: d["checks"]["teacher"]["constants"].update(no_such_module={})),
            "checks.qwen38_layers.constants.qwen38_layers.FLOOR_RATIO.value": (
                lambda d: d["checks"]["qwen38_layers"]["constants"]["qwen38_layers"]["FLOOR_RATIO"].update(value=True)),
            "checks.teacher.rules[1].value": (
                lambda d: d["checks"]["teacher"]["rules"][1].update(value="zero")),
            "checks.teacher.rules[6].value": (
                lambda d: d["checks"]["teacher"]["rules"][6].update(value=float("nan"))),
            "checks.qwen38_capture.rules[4].value": lambda d: d["checks"]["qwen38_capture"]["rules"][4].update(value=-1),
            "checks.qwen38_layers.rules[1].value": lambda d: d["checks"]["qwen38_layers"]["rules"][1].update(value=2.5),
            "references.teacher.extra": lambda d: d["references"]["teacher"].update(extra=1),
            "references.teacher.variants": lambda d: d["references"]["teacher"].update(variants=[]),
            "references.long_context.checkpoint_keys": (
                lambda d: d["references"]["long_context"].update(checkpoint_keys=["revison"])),
            "references.teacher": lambda d: drop(d, "references"),
            "qualification.needel": lambda d: d["qualification"].update(needel={}),
            "qualification.needle.match": lambda d: d["qualification"]["needle"].update(match="exact"),
            "qualification.needle.frozen": lambda d: drop(d, "qualification", "needle", "frozen"),
            "qualification.prefix.anchor_boundary": lambda d: d["qualification"]["prefix"].update(anchor_boundary="2051"),
            "qualification.prefix.prefill_granule": lambda d: d["qualification"]["prefix"].update(prefill_granule=0),
            "qualification.checkpoint.lock": lambda d: drop(d, "qualification", "checkpoint", "lock"),
            "qualification.checkpoint.checkpoint_identity": (
                lambda d: d["qualification"]["checkpoint"].update(checkpoint_identity="74a79e")),
            "qualification.verify_buckets[1]": lambda d: d["qualification"].update(verify_buckets=[2, 0]),
            "qualification.teacher_reference_kind": lambda d: d["qualification"].update(teacher_reference_kind="f64"),
            "checkpoint.status": lambda d: d.update(checkpoint={"status": 1}),
            # Optional reference fields keep their types.
            "references.teacher.workload": lambda d: d["references"]["teacher"].update(workload=""),
            "references.image_probes.variants": lambda d: d["references"]["image_probes"].update(variants=[]),
            # The teacher module scores the teacher reference's variants, so they stay required there.
            "references.teacher.variants": lambda d: drop(d, "references", "teacher", "variants"),
            # A former check ID names the declared check that replaced it.
            "renamed_checks": lambda d: d.update(renamed_checks=[]),
            "renamed_checks.qwen38-teacher": lambda d: d["renamed_checks"].update({"qwen38-teacher": "absent"}),
            "renamed_checks.api": lambda d: d["renamed_checks"].update(api="teacher"),
            # The release preview's settings.
            "preview.colour": lambda d: d["preview"].update(colour="red"),
            "preview.profile": lambda d: drop(d, "preview", "profile"),
            "preview.speculation": lambda d: drop(d, "preview", "speculation"),
            "preview.speculation.tokens": lambda d: d["preview"]["speculation"].update(tokens=0),
            "preview.speculation.draft_cost": lambda d: d["preview"]["speculation"].update(draft_cost="per-row"),
            "preview.spec_greedy.rule": lambda d: d["preview"]["spec_greedy"].update(rule="close"),
            "preview.spec_greedy.reason": lambda d: d["preview"]["spec_greedy"].update(rule="informational"),
            "preview.soak_prefix_mix": lambda d: d["preview"].update(soak_prefix_mix="no"),
            "preview.package_max_sequences": lambda d: d["preview"].update(package_max_sequences=0),
            # The previous screen's fields are retired and refused as unknown.
            "preview.sweep": lambda d: d["preview"].update(sweep=[1, 16]),
            "preview.greedy_pin": lambda d: d["preview"].update(greedy_pin={"sha256": "0" * 64, "approved": "x"}),
            "preview.budgets": lambda d: d["preview"].update(budgets={"startup": 240}),
            "preview.checkpoint_verify": lambda d: d["preview"].update(checkpoint_verify="prepare"),
        }
        for field, change in cases.items():
            with self.subTest(field=field):
                value = declared()
                change(value)
                with self.assertRaisesRegex(declaration.DeclarationError, "^" + field.replace("[", r"\[") + ": "):
                    declaration.parse(value, declared=True)

    def test_declaration_with_only_required_fields_compiles(self):
        # Optional v1 fields are additive: a declaration without any of them stays valid.
        value = declared()
        api = {key: value["checks"]["api"][key] for key in ("module", "args", "timeout_seconds", "receipt", "rules")}
        qualification = {key: item for key, item in value["qualification"].items() if key not in ("prefix", "needle")}
        minimal = {"schema": declaration.SCHEMA, "model": "minimal",
                   "cohorts": {"quick": {"checks": ["api"]}},
                   "change_classes": dict.fromkeys(declaration.CHANGE_CLASSES, []),
                   "checks": {"api": api}, "inputs": {}, "qualification": qualification,
                   "receipts": value["receipts"]}
        config = declaration.parse(minimal)
        release_gate.validate(config)
        self.assertEqual(list(config), ["schema", "model", "profiles", "checks", "receipts", "inputs", "qualification"])
        self.assertEqual(config["checks"], [{"id": "api", **api}])

    def test_reference_workload_is_an_optional_v1_field(self):
        # A bound reference names the runner workload it scored (common.pinned_binding).
        value = declared()
        self.assertNotIn("workload", value["references"]["teacher"])
        value["references"]["teacher"]["workload"] = "fixtures/qwen38/teacher-workload.json"
        config = declaration.parse(value, declared=True)
        self.assertEqual(config["qualification"]["references"]["teacher"]["workload"],
                         "fixtures/qwen38/teacher-workload.json")
        value["references"]["teacher"]["workload"] = {"env": "CHOPINFER_DATA", "path": "teacher-workload.json"}
        config = declaration.parse(value, declared=True)
        self.assertEqual(config["qualification"]["references"]["teacher"]["workload"]["env"], "CHOPINFER_DATA")
        value["references"]["teacher"]["workload"] = {"path": "teacher-workload.json"}
        with self.assertRaisesRegex(declaration.DeclarationError, "^references.teacher.workload.env: "):
            declaration.parse(value, declared=True)
        del value["references"]["teacher"]["workload"]
        self.assertEqual(declaration.parse(value, declared=True), declaration.load(declaration.path("qwen38")))

    def test_image_probe_thinking_is_an_optional_v1_field(self):
        # A model whose frontend refuses disabled thinking sends its probes with thinking on.
        value = declared()
        self.assertNotIn("thinking", value["qualification"]["media"])
        value["qualification"]["media"]["thinking"] = True
        self.assertTrue(declaration.parse(value, declared=True)["qualification"]["media"]["thinking"])
        value["qualification"]["media"]["thinking"] = "yes"
        with self.assertRaisesRegex(declaration.DeclarationError, "^qualification.media.thinking: "):
            declaration.parse(value, declared=True)

    def test_reference_variants_are_optional_where_no_module_scores_them(self):
        # Qwen3.8's image probe answers are pinned without variants; teacher and media keep them required.
        value = declared()
        self.assertNotIn("variants", value["references"]["image_probes"])
        config = declaration.parse(value, declared=True)
        self.assertEqual(config["qualification"]["references"]["image_probes"], value["references"]["image_probes"])
        value = declared("mimo26")
        del value["references"]["media"]["variants"]
        with self.assertRaisesRegex(declaration.DeclarationError, "^references.media.variants: is required"):
            declaration.parse(value, declared=True)

    def test_previewed_models_are_those_with_a_preview_section(self):
        # Every released model is previewed; a model may preview before it releases.
        previewed = list(declaration.previewed())
        self.assertEqual(previewed, sorted(previewed))
        self.assertLessEqual(set(inventory.released(declaration.ROOT)), set(previewed))
        self.assertLessEqual(set(previewed), set(inventory.shipped(declaration.ROOT)))
        with tempfile.TemporaryDirectory() as root:
            for name in ("qwen38", "unpreviewed"):
                value = declared()
                if name == "unpreviewed":
                    del value["preview"]
                (Path(root) / "models" / name).mkdir(parents=True)
                (Path(root) / "models" / name / "qual.json").write_text(json.dumps(dict(value, model=name)))
            self.assertEqual(list(declaration.previewed(root)), ["qwen38"])
            (Path(root) / "models/qwen38/qual.json").write_text(json.dumps(dict(declared(), model="other")))
            with self.assertRaisesRegex(declaration.DeclarationError, "qual.json: model: must name its directory qwen38"):
                declaration.previewed(root)
            (Path(root) / "models/qwen38/qual.json").write_text("{bad")
            with self.assertRaisesRegex(declaration.DeclarationError, f"^{root}/models/qwen38/qual.json: "):
                declaration.previewed(root)

    def test_media_reference_is_required_by_the_media_check(self):
        value = declared("mimo26")
        del value["references"]["media"]
        with self.assertRaisesRegex(declaration.DeclarationError, "^references.media: is required by checks.multimodal"):
            declaration.parse(value, declared=True)

    def test_no_tool_names_a_deleted_model_gate(self):
        # Paths may be built from parts ('qual' / 'gates'), so match both forms.
        pattern = re.compile(r"""qual["']?\s*/\s*["']?gates|with_name\(["']gates["']\)""")
        # The mechanics fixture's home, trace_inputs' fallback for older trees, and this test.
        exempt = {"tools/qual/declaration.py", "tools/qual/dev/trace_inputs.py", "tools/qual/test_declaration.py"}
        found = []
        for directory, names, files in os.walk(common.ROOT):
            names[:] = [name for name in names if name not in (".git", "target")]
            for name in files:
                path = Path(directory, name)
                relative = path.relative_to(common.ROOT).as_posix()
                if path.suffix not in (".py", ".sh", ".md", ".rs", ".toml", ".yml") or relative in exempt:
                    continue
                for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                    if pattern.search(line) and "mechanics" not in line:
                        found.append(f"{relative}:{number}: {line.strip()}")
        self.assertEqual(found, [])

    def test_invalid_file_is_refused_by_the_runner_and_the_tools(self):
        value = declared()
        del value["checks"]["api"]["timeout_seconds"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qual.json"
            # A qual.json is a declaration whatever its schema says.
            path.write_text(json.dumps({**value, "schema": declaration.GATE_SCHEMA}))
            with self.assertRaisesRegex(ValueError, f"^{path}: schema: must be {declaration.SCHEMA}$"):
                declaration.load(path)
            path.write_text(json.dumps(value))
            result = subprocess.run([sys.executable, "-B", str(Path(release_gate.__file__)), "--config", str(path),
                                     "--model-root", directory, "--plan"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn(f"{path}: checks.api.timeout_seconds: is required", result.stderr)
            with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(path)}), \
                    self.assertRaisesRegex(ValueError, "checks.api.timeout_seconds"):
                common.input_path("exact")

    def test_a_checkpoint_variant_has_its_own_declaration_and_lock(self):
        """MODEL.VARIANT names qual.VARIANT.json beside qual.json, whose
        lock is qual-references.VARIANT.lock.json; each file names exactly
        its variant, and qual.json none."""
        import refs
        self.assertEqual(declaration.named("example.export"),
                         declaration.ROOT / "models/example/qual.export.json")
        self.assertEqual(declaration.named("example"), declaration.ROOT / "models/example/qual.json")
        self.assertEqual(declaration.split("example.export"), ("example", "export"))
        self.assertEqual(declaration.split("example"), ("example", None))
        for name in ("", "a/b", "a.b.c", "a.", ".b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                declaration.split(name)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(refs.lock_path("example", root, "export"),
                             root / "models/example/qual-references.export.lock.json")
            self.assertEqual(refs.lock_path("example", root), root / "models/example/qual-references.lock.json")
            value = {**declared(), "variant": "export"}
            variant = root / "qual.export.json"
            variant.write_text(json.dumps(value))
            self.assertEqual(declaration.load(variant)["variant"], "export")
            for name, stated, message in (("qual.export.json", None, "variant: must be export"),
                                          ("qual.export.json", "other", "variant: must be export"),
                                          ("qual.json", "export", "variant: is not allowed in qual.json")):
                with self.subTest(file=name, variant=stated):
                    path = root / name
                    path.write_text(json.dumps({**value, "variant": stated} if stated else declared()))
                    with self.assertRaisesRegex(ValueError, message):
                        declaration.load(path)

    def test_a_variant_file_is_always_a_declaration_of_its_variant(self):
        """qual.VARIANT.json is read as a declaration whatever it holds: a
        compiled policy, or one without the declaration schema, is refused
        rather than gated against the base checkpoint and lock."""
        base = declared()
        compiled = declaration.compile_gate(base)
        without_schema = {key: item for key, item in base.items() if key != "schema"}
        with tempfile.TemporaryDirectory() as directory:
            for name, value in (("qual.export.json", compiled),
                                ("qual.export.json", {**compiled, "variant": "export"}),
                                ("qual.export.json", {**without_schema, "variant": "export"}),
                                ("qual.export.json", {**base, "schema": declaration.GATE_SCHEMA, "variant": "export"}),
                                ("qual.json", {**compiled, "variant": "export"})):
                with self.subTest(file=name, schema=value.get("schema"), variant=value.get("variant")):
                    path = Path(directory) / name
                    path.write_text(json.dumps(value))
                    with self.assertRaisesRegex(ValueError, f"^{path}: schema: must be {declaration.SCHEMA}$"):
                        declaration.load(path)

    def test_the_gate_config_keeps_the_declared_capabilities(self):
        """Encoder parity reads the media a model serves but does not
        qualify from the gate config."""
        for path in declaration.shipped():
            with self.subTest(path=path):
                self.assertEqual(declaration.load(path).get("capabilities"),
                                 json.loads(path.read_text()).get("capabilities"))

    def test_only_declaration_file_names_are_read_as_declarations(self):
        """qual.json and qual.VARIANT.json are declarations; another name,
        such as a saved copy, loads as the policy it holds."""
        compiled = declaration.compile_gate(declared())
        with tempfile.TemporaryDirectory() as directory:
            for name in ("qual.export.json.bak", "qual.json.bak", "qual.export.policy", "policy.json"):
                with self.subTest(file=name):
                    path = Path(directory) / name
                    path.write_text(json.dumps(compiled))
                    self.assertEqual(declaration.load(path), compiled)
            path = Path(directory) / "qual.export.json"
            path.write_text(json.dumps(compiled))
            with self.assertRaisesRegex(ValueError, "schema: must be"):
                declaration.load(path)

    def test_a_variant_names_its_own_renamed_checks_and_change_classes(self):
        """MODEL.VARIANT reads the variant's renamed checks and change
        classes, as rescoring and change selection need."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "models/example"
            model.mkdir(parents=True)
            base = {**declared(), "model": "example", "renamed_checks": {"former": "api"}}
            variant = {**base, "variant": "export", "renamed_checks": {"older": "api"},
                       "change_classes": {**base["change_classes"], "frontend": ["api"]}}
            (model / "qual.json").write_text(json.dumps(base))
            (model / "qual.export.json").write_text(json.dumps(variant))
            self.assertEqual(declaration.renamed("example", root), {"former": "api"})
            self.assertEqual(declaration.renamed("example.export", root), {"older": "api"})
            self.assertEqual(declaration.change_classes("example.export", root)["frontend"], ["api"])


if __name__ == "__main__":
    unittest.main()

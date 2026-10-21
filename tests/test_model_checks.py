"""Inventory consumers plan every declared CPU check and check the explicit catalog seam."""
import dataclasses
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ci"))
import contributor
import kernel_lab_inventory
import model_checks
from tools.model_sdk import inventory

# Every shared reference-runner module CPU CI syntax-checks.
REFRUN = ["tools/refrun/" + name for name in (
    "__init__.py", "__main__.py", "adapter.py", "cache.py", "calibrate.py", "dry_run.py", "emit.py", "encoder.py",
    "encoder_ops.py", "frontend.py", "frontend_cases.py", "frontend_corpus.py", "image_decode.py", "keys.py",
    "margins.py",
    "mixed_answers.py", "modes.py", "ops.py", "place.py", "preprocessing.py", "preprocessing_ab.py",
    "python_semantics.py", "quant.py", "run.py", "score.py", "stamps.py", "taps.py", "template_coverage.py",
    "test_cache.py", "test_dry_run.py", "test_emit.py", "test_encoder.py", "test_frontend.py",
    "test_frontend_corpus.py", "test_image_decode.py", "test_keys.py", "test_margins.py",
    "test_mixed_answers.py", "test_preprocessing.py", "test_preprocessing_ab.py", "test_quant.py", "test_runner.py",
    "test_stamps.py", "test_taps.py", "tiny.py", "workload.py")]


class Plans(unittest.TestCase):
    def assert_plan(self, planned, models, python="python3", root=ROOT):
        """Every declared check is planned once, in the gate's order."""
        # One unittest step per declared reference test, grouped by kind in
        # first-declaration order, then by model in inventory order.
        expected = []
        kinds = dict.fromkeys(path for model in models for path in model.checks.reference_tests)
        for path in kinds:
            kind = Path(path).stem.removeprefix("test_").removesuffix("reference").rstrip("_")
            for model in models:
                if path in model.checks.reference_tests:
                    expected.append(("refrun-" + (kind + "-" if kind else "") + model.name,
                                     [python, "-B", "-m", "unittest", f"{model.directory}/{path}", "-v"]))
        self.assertEqual(planned["references"], expected)
        self.assertEqual(len({label for label, _ in expected}), len(expected))
        # One build of every profile target: libraries, then the host, then
        # each binary once, package entrypoints before the commands.
        targets = [target for model in models for target in model.checks.profile_targets]
        argv = planned["profile"]
        self.assertEqual(argv[:2], ["cargo", "build"])
        self.assertEqual(argv[-2:], ["--locked", "--offline"])
        flags = {flag: [argv[i + 1] for i, word in enumerate(argv) if word == flag] for flag in ("-p", "--bin")}
        libraries = [target.package for target in targets if target.kind == "cdylib"]
        self.assertEqual(flags["-p"], list(dict.fromkeys([*libraries, "runtime-host",
                                                          *(t.package for t in targets if t.kind == "bin")])))
        self.assertEqual(argv.count("--lib"), int(bool(libraries)))
        if libraries:
            self.assertEqual(argv.index("--lib"), argv.index("runtime-host") - 2)
        bins = flags["--bin"]
        self.assertEqual(bins[:2], ["chopinfer", "chopinfer-host"])
        self.assertEqual(sorted(bins[2:]), sorted(t.name for t in targets if t.kind == "bin"))
        packaged = [name.endswith("-package") for name in bins[2:]]
        self.assertEqual(packaged, sorted(packaged, reverse=True))
        # Every refrun module, then each model's declared compile inputs.
        compile = planned["compile"]
        self.assertEqual(compile[:3], ["python3", "-m", "py_compile"])
        files = compile[3:]
        refrun = sorted(path.relative_to(root).as_posix() for path in (root / "tools/refrun").glob("*.py"))
        self.assertTrue(refrun)
        self.assertEqual(files, list(dict.fromkeys([*refrun, *(path for model in models
                                                                 for path in inventory.compile_files(root, model))])))
        for path in files:
            self.assertTrue((root / path).is_file(), path)

    def test_cpu_plan_runs_every_declared_check(self):
        models = inventory.load(ROOT)
        self.assertTrue(any(model.checks.compile_inputs for model in models))
        for python in (None, "", "/fixture/reference-python", "/fixture/reference python"):
            env = {} if python is None else {"CHOPINFER_REFRUN_PYTHON":python}
            with self.subTest(python=python):
                self.assert_plan(model_checks.plan(env=env), models, python or "python3")

    def test_two_model_plan_keeps_the_gate_order(self):
        """The planner's own order, independent of any model's declarations:
        references by kind in first-declaration order, then by model; one
        `--lib` for every library first; package entrypoints ascending, then
        the other commands descending; the shared refrun modules first."""
        def model(name, references, library=False):
            bins = {(name + "-package", "bin"), ("chopinfer-model-" + name, "bin")}
            packages = {"compiler": inventory.Package(name, "compiler", "model-" + name, frozenset(bins), "models/" + name)}
            targets = [inventory.Target("model-" + name, "chopinfer-model-" + name, "bin"),
                       inventory.Target("model-" + name, name + "-package", "bin")]
            if library:
                packages["bundle"] = inventory.Package(name, "bundle", "bundle-" + name,
                                                       frozenset({("bundle_" + name, "cdylib")}), "models/" + name)
                targets.append(inventory.Target("bundle-" + name, "bundle_" + name, "cdylib"))
            return inventory.Model(name, None, packages, checks=inventory.Checks(references, (), tuple(targets)))
        models = [model("alpha", ("test_reference.py", "test_encoder_reference.py")),
                  model("beta", ("test_reference.py", "test_extra_reference.py", "test_encoder_reference.py"), library=True)]
        with patch.object(inventory, "check_files"):
            planned = model_checks.plan(models=models, env={"CHOPINFER_REFRUN_PYTHON": "/runner/python"})
        unittest_ = ["/runner/python", "-B", "-m", "unittest"]
        self.assertEqual(planned["references"], [
            ("refrun-alpha", [*unittest_, "models/alpha/test_reference.py", "-v"]),
            ("refrun-beta", [*unittest_, "models/beta/test_reference.py", "-v"]),
            ("refrun-encoder-alpha", [*unittest_, "models/alpha/test_encoder_reference.py", "-v"]),
            ("refrun-encoder-beta", [*unittest_, "models/beta/test_encoder_reference.py", "-v"]),
            ("refrun-extra-beta", [*unittest_, "models/beta/test_extra_reference.py", "-v"]),
        ])
        self.assertEqual(planned["profile"], [
            "cargo", "build", "-p", "bundle-beta", "--lib", "-p", "runtime-host", "-p", "model-alpha", "-p", "model-beta",
            "--bin", "chopinfer", "--bin", "chopinfer-host", "--bin", "alpha-package", "--bin", "beta-package",
            "--bin", "chopinfer-model-beta", "--bin", "chopinfer-model-alpha", "--locked", "--offline",
        ])
        self.assertEqual(planned["compile"], ["python3", "-m", "py_compile", *REFRUN])

    def test_compile_plan_covers_the_shared_refrun_modules(self):
        files = model_checks.plan(env={})["compile"][3:]
        self.assertEqual([path for path in files if not path.startswith("models/")], REFRUN)

    def test_selected_extra_model_checks_are_planned(self):
        models = inventory.load(ROOT)
        package = inventory.Package("extra", "compiler", "model-extra", frozenset({("extra-package", "bin"), ("extra-cli", "bin")}),
                                    "models/extra")
        library = inventory.Package("extra", "bundle", "bundle-extra", frozenset({("bundle_extra", "cdylib")}), "models/extra")
        checks = inventory.Checks(("test_reference.py", "test_extra_reference.py"), (), (
            inventory.Target("model-extra", "extra-cli", "bin"), inventory.Target("model-extra", "extra-package", "bin"),
            inventory.Target("bundle-extra", "bundle_extra", "cdylib")))
        extra = inventory.Model("extra", None, {"bundle": library, "compiler": package}, checks=checks)
        with patch.object(inventory, "check_files"):
            planned = model_checks.plan(models=[*models, extra], env={})
        self.assert_plan(planned, [*models, extra])
        self.assertIn(("refrun-extra", ["python3", "-B", "-m", "unittest", "models/extra/test_reference.py", "-v"]),
                      planned["references"])
        self.assertIn(("refrun-extra-extra", ["python3", "-B", "-m", "unittest", "models/extra/test_extra_reference.py", "-v"]),
                      planned["references"])
        self.assertIn("bundle-extra", planned["profile"])
        self.assertIn("extra-cli", planned["profile"])

    def test_unselected_extra_model_does_not_change_plans_or_modes(self):
        models = inventory.load(ROOT)
        extra = dataclasses.replace(models[0], name="extra", contributor=None,
                                    checks=inventory.Checks(), kernel_lab=inventory.KernelLab(), geometry_recipe=None)
        self.assertEqual(model_checks.plan(models=models + [extra], env={}), model_checks.plan(models=models, env={}))
        with patch.object(contributor, "models", return_value=models + [extra]):
            self.assertNotIn("extra", contributor.modes())

    def test_empty_required_plans_are_refused(self):
        models = inventory.load(ROOT)
        for kind, field in (("references", "reference_tests"), ("profile", "profile_targets"), ("compile", "compile_inputs")):
            empty = [dataclasses.replace(model, checks=dataclasses.replace(model.checks, **{field:()})) for model in models]
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                with patch.object(inventory, "check_files"), self.assertRaises(inventory.InventoryError) as error:
                    model_checks.plan(Path(directory), env={}, models=empty)
                self.assertEqual(str(error.exception), f"CPU {kind} plan is empty")

    def assert_recipe_rows(self, mode, recipe):
        rows = [row for selection in recipe.tests for row in selection.expanded()]
        commands = contributor.steps(mode)[3:-1]
        self.assertEqual(len(commands), len(rows))
        self.assertEqual([command["name"] for command in commands], [row.name for row in rows])
        self.assertEqual(len({command["name"] for command in commands}), len(rows))
        for row, command in zip(rows, commands):
            with self.subTest(mode=mode, row=row.name):
                argv = command["argv"]
                self.assertEqual(argv[:4], ["cargo", "test", "-p", row.package])
                self.assertEqual(command["phase"], row.phase)
                self.assertEqual(command["tests"], not row.compile)
                self.assertEqual(argv.count("--lib"), int(row.lib))
                for flag, values in (("--bin", row.bins), ("--test", row.suites)):
                    self.assertEqual([argv[i + 1] for i, arg in enumerate(argv) if arg == flag], list(values))
                self.assertEqual(argv.count("--no-run"), int(row.compile))
                self.assertEqual(argv.count("--exact"), int(row.case is not None))
                self.assertEqual(argv.count("--locked"), 1)
                self.assertEqual(argv.count("--offline"), 1)
                if row.case:
                    self.assertEqual(argv[-3:], [row.case, "--", "--exact"])
                elif row.compile:
                    self.assertEqual(argv[-1], "--no-run")

    def test_every_declared_row_is_planned_once_in_order(self):
        for model in inventory.load(ROOT):
            if model.contributor:
                self.assert_recipe_rows(model.name, model.contributor)

    def test_selected_extra_recipe_is_discovered_and_planned(self):
        recipe = inventory.Contributor(roots=("extra-compiler",), tests=(
            inventory.TestSelection("compile extra", "compile", "extra-compiler", lib=True,
                                    bins=("extra-package",), suites=("extra-suite",), compile=True),
            inventory.TestSelection("", "config", "extra-compiler", lib=True,
                                    cases=(inventory.TestCase("first", "config::first"), inventory.TestCase("second", "config::second"))),
            inventory.TestSelection("suite", "frontend", "extra-compiler", suites=("extra-suite",)),
        ))
        extra = inventory.Model("extra", "EXTRA", {}, contributor=recipe)
        with patch.object(contributor, "models", return_value=[*inventory.load(ROOT), extra]):
            self.assertIn("extra", contributor.modes())
            self.assertEqual(contributor.roots("extra"), ("extra-compiler", "model-compiler-sdk"))
            self.assert_recipe_rows("extra", recipe)
            commands = contributor.steps("extra")[3:-1]
            self.assertEqual([command["name"] for command in commands], ["compile extra", "first", "second", "suite"])
            self.assertEqual([command["argv"][-3] for command in commands[1:3]], ["config::first", "config::second"])
            self.assertEqual([command["phase"] for command in commands], ["compile", "config", "config", "frontend"])

    def test_environment_clears_union_and_checkpoint_index_keys(self):
        models = inventory.load(ROOT)
        extra = inventory.Model("extra", "EXTRA", {}, contributor=inventory.Contributor(clear_env=("CHOPINFER_EXTRA_FIXTURE",)))
        models.append(extra)
        keys = {"CHOPINFER_PACKAGE_UNKNOWN", "CHOPINFER_INDEX_CHECKPOINT", "CHOPINFER_INDEX_TOKENIZER"}
        for model in models:
            if model.contributor:
                keys.update(model.contributor.clear_env)
            if model.checkpoint_env:
                keys.update("CHOPINFER_" + model.checkpoint_env + suffix for suffix in
                            ("_CHECKPOINT", "_TOKENIZER_DIR"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "models/extra/compiler/fixtures/checkpoints/index.json"
            index.parent.mkdir(parents=True)
            index.write_text(json.dumps([{"checkpoint_env":"CHOPINFER_INDEX_CHECKPOINT", "tokenizer_env":"CHOPINFER_INDEX_TOKENIZER"}]))
            env = {key:"production-input" for key in keys}
            env["CHOPINFER_UNRELATED"] = "keep"
            with patch.object(contributor, "ROOT", root), patch.object(contributor, "models", return_value=models), patch.dict(os.environ, env, clear=True):
                cleared = contributor.environment()
            self.assertTrue(keys.isdisjoint(cleared))
            self.assertEqual(cleared["CHOPINFER_UNRELATED"], "keep")
            self.assertEqual(cleared["CARGO_NET_OFFLINE"], "true")
            self.assertEqual(cleared["RUSTUP_AUTO_INSTALL"], "0")


class GateSteps(unittest.TestCase):
    def steps(self):
        import cpu
        with tempfile.TemporaryDirectory() as directory:
            shared, independent = cpu.steps(Path(directory) / "target", Path(directory), 8, 8)
        return {step.name: step for step in shared + independent}

    def test_gate_runs_the_planned_model_checks(self):
        planned = model_checks.plan()
        steps = self.steps()
        self.assertEqual(steps["profile-cli-build"].argv, planned["profile"])
        self.assertEqual(steps["refrun-python-compile"].argv, planned["compile"])
        previous = ("refrun",)
        for label, argv in planned["references"]:
            self.assertEqual(steps[label].argv, argv)
            self.assertEqual(steps[label].after, ("reference-prerequisites", "profile-copy") + previous)
            self.assertEqual(steps[label].env["CUDA_VISIBLE_DEVICES"], "")
            previous = (label,)
        self.assertEqual(steps["tapdiff"].after, ("reference-prerequisites", "profile-copy") + previous)
        self.assertEqual(steps["kernel-lab-inventory"].argv, ["python3", "-B", "ci/kernel_lab_inventory.py"])
        self.assertEqual(steps["model-check-plans"].argv, ["python3", "-B", "tests/test_model_checks.py"])

    def test_an_empty_plan_stops_the_gate_before_any_step(self):
        with patch.object(model_checks, "plan", side_effect=inventory.InventoryError("CPU references plan is empty")):
            with self.assertRaisesRegex(inventory.InventoryError, "^CPU references plan is empty$"):
                self.steps()


class Registration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.lab = self.root / "crates/kernel-lab"
        self.lab.mkdir(parents=True)
        self.models = [inventory.Model("selected", None, {"contract":inventory.Package("selected", "contract", "selected-contract", frozenset(), "models/selected")}, kernel_lab=inventory.KernelLab(True)),
                       inventory.Model("extra", None, {"contract":inventory.Package("extra", "contract", "extra-contract", frozenset(), "models/extra")})]
        (self.lab / "models.rs").write_text('[&selected_contract::kernels::CATALOG, &lab_shapes::CATALOG]')
        self.manifest('[dependencies]\nselected-contract.workspace = true\n[build-dependencies]\nselected-contract.workspace = true\n')

    def manifest(self, text):
        (self.lab / "Cargo.toml").write_text(text)

    def test_missing_or_empty_inventory_is_refused(self):
        (self.lab / "models.rs").write_text('[&lab_shapes::CATALOG]')
        self.manifest('[dependencies]\n[build-dependencies]\n')
        self.assertEqual(kernel_lab_inventory.check(self.root), ['admission: no model inventories found'])
        self.assertEqual(kernel_lab_inventory.check(self.root, []), ['admission: no model inventories found'])
        (self.root / 'models').mkdir()
        self.assertEqual(kernel_lab_inventory.check(self.root), ['admission: no model inventories found'])

    def test_workspace_inherited_contract_aliases_are_resolved(self):
        (self.root / "Cargo.toml").write_text('[workspace.dependencies]\ncontract-alias = { package = "selected-contract", path = "models/selected/contract" }\n')
        self.manifest('[dependencies]\ncontract-alias.workspace = true\n[build-dependencies]\ncontract-alias.workspace = true\n')
        (self.lab / "models.rs").write_text('[&contract_alias::kernels::CATALOG, &lab_shapes::CATALOG]')
        self.assertEqual(kernel_lab_inventory.check(self.root, self.models), [])

    def test_missing_aliased_contract_names_the_workspace_key(self):
        (self.root / "Cargo.toml").write_text('[workspace.dependencies]\ncontract-alias = { package = "selected-contract", path = "models/selected/contract" }\n')
        (self.lab / "models.rs").write_text('[&contract_alias::kernels::CATALOG, &lab_shapes::CATALOG]')
        line = 'contract-alias.workspace = true\n'
        for present in (('dependencies',), ('build-dependencies',), ()):
            with self.subTest(present=present):
                self.manifest(''.join(f'[{section}]\n' + (line if section in present else '')
                                      for section in ('dependencies', 'build-dependencies')))
                self.assertEqual(kernel_lab_inventory.check(self.root, self.models),
                                 [f'{section}: missing contract selected-contract (add `contract-alias.workspace = true`)'
                                  for section in ('dependencies', 'build-dependencies') if section not in present])
        (self.lab / "models.rs").write_text('[&lab_shapes::CATALOG]')
        self.assertEqual(kernel_lab_inventory.check(self.root, self.models)[0],
                         'models.rs: missing catalog contract_alias (append `&contract_alias::sm120::CATALOG,` before `&lab_shapes::CATALOG`)')

    def test_catalogs_in_sm120_modules_register(self):
        (self.lab / "models.rs").write_text('[&selected_contract::sm120::CATALOG, &lab_shapes::CATALOG]')
        self.assertEqual(kernel_lab_inventory.check(self.root, self.models), [])
        (self.lab / "models.rs").write_text('[&selected_contract::sm120::CATALOG, &selected_contract::kernels::CATALOG]')
        self.assertEqual(kernel_lab_inventory.check(self.root, self.models), ['models.rs: duplicate catalog selected_contract'])

    def test_current_registration(self):
        self.assertEqual(kernel_lab_inventory.check(), [])
        self.assertEqual(kernel_lab_inventory.check(self.root, self.models), [])

    def test_missing_and_extra_catalogs(self):
        (self.lab / "models.rs").write_text('[&extra_contract::kernels::CATALOG, &lab_shapes::CATALOG]')
        self.assertEqual(kernel_lab_inventory.check(self.root, self.models),
                         ['models.rs: missing catalog selected_contract (append `&selected_contract::sm120::CATALOG,` before `&lab_shapes::CATALOG`)',
                          'models.rs: extra catalog extra_contract (remove its CATALOG entry)'])

    def test_missing_and_extra_dependencies_in_each_section(self):
        for section in ('dependencies', 'build-dependencies'):
            other = 'build-dependencies' if section == 'dependencies' else 'dependencies'
            self.manifest(f'[{section}]\nextra-contract.workspace = true\n[{other}]\nselected-contract.workspace = true\n')
            self.assertEqual(kernel_lab_inventory.check(self.root, self.models),
                             [f'{section}: missing contract selected-contract (add `selected-contract.workspace = true`)',
                              f'{section}: extra contract extra-contract (remove its entry)'])

    def test_workspace_contract_dependency_without_an_inventory_is_extra(self):
        (self.root / "Cargo.toml").write_text('[workspace.dependencies]\norphan-contract = { path = "models/orphan/contract" }\n')
        self.manifest('[dependencies]\nselected-contract.workspace = true\norphan-contract.workspace = true\n[build-dependencies]\nselected-contract.workspace = true\n')
        self.assertEqual(kernel_lab_inventory.check(self.root, self.models),
                         ['dependencies: extra contract orphan-contract (remove its entry)',
                          'Cargo.toml: missing workspace dependency selected-contract '
                          '(add `selected-contract = { path = "models/selected/contract" }` to [workspace.dependencies])'])
        (self.root / "Cargo.toml").write_text('[workspace.dependencies]\nselected-contract = { path = "models/selected/contract" }\n')
        self.manifest('[dependencies]\nselected-contract.workspace = true\n[build-dependencies]\nselected-contract.workspace = true\n')
        self.assertEqual(kernel_lab_inventory.check(self.root, self.models), [])

    def test_duplicate_catalog_is_refused(self):
        (self.lab / "models.rs").write_text('[&selected_contract::kernels::CATALOG, &selected_contract::kernels::CATALOG]')
        self.assertEqual(kernel_lab_inventory.check(self.root, self.models), ['models.rs: duplicate catalog selected_contract'])


if __name__ == '__main__':
    unittest.main()

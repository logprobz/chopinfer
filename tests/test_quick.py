"""Changed-package selection includes build, dev and transitive dependents."""
import importlib.util
import contextlib
import io
import json
import os
import shutil
import subprocess
import pathlib
import re
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "ci"))
spec = importlib.util.spec_from_file_location(
    "quick", pathlib.Path(__file__).resolve().parents[1] / "ci/quick.py")
quick = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quick)
import change_scope
import check_inputs
import runner


class Selection(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "workspace_root": "/repo", "workspace_members": ["core", "host", "fixture", "other"],
            "packages": [{"id": name, "name": name,
                          "manifest_path": f"/repo/crates/{name}/Cargo.toml"}
                         for name in ("core", "host", "fixture", "other")],
            "resolve": {"nodes": [
                {"id": "core", "dependencies": []},
                {"id": "host", "dependencies": ["core"]},
                {"id": "fixture", "dependencies": ["host"]},
                {"id": "other", "dependencies": []}]}}

    def test_dependency_closure(self):
        self.assertEqual(quick.affected(self.metadata, ["crates/core/src/lib.rs"]),
                         ["core", "fixture", "host"])
        self.assertEqual(quick.affected(self.metadata, ["crates/host/build.rs"]),
                         ["fixture", "host"])

    def test_workspace_and_unowned_inputs_expand_to_every_member(self):
        for path in ("Cargo.lock", "ci/cpu.sh", "crates/deleted/src/lib.rs",
                     "crates/core-other/src/lib.rs", "crates/core/tests/support/shared.rs",
                     "crates/core/tests/common/mod.rs", "crates/runtime-bundle/src/rank_index.rs"):
            self.assertEqual(quick.affected(self.metadata, [path]),
                             ["core", "fixture", "host", "other"])

    def test_empty_diff_does_not_claim_tests_ran(self):
        self.assertEqual(quick.affected(self.metadata, []), [])

    def test_kernel_evidence_options_require_report_mode(self):
        for flag in ['--kernel-reports', '--previous-kernel-reports']:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                quick.main([flag, 'unused'])
            self.assertEqual(error.exception.code, 2)

    def model_metadata(self):
        roots = {"model-qwen38": "models/qwen38/compiler", "bundle-qwen38": "models/qwen38/bundle",
                 "model-mimo26": "models/mimo26/compiler", "bundle-mimo26": "models/mimo26/bundle",
                 "kernel-lab": "crates/kernel-lab"}
        return {"workspace_root": "/repo", "workspace_members": list(roots),
                "packages": [{"id": name, "name": name, "manifest_path": f"/repo/{path}/Cargo.toml"}
                             for name, path in roots.items()],
                "resolve": {"nodes": [{"id": name, "dependencies": []} for name in roots]}}

    def test_cross_crate_frontend_consumers(self):
        metadata = self.model_metadata()
        for path, consumer in (("models/qwen38/compiler/src/frontend-multimodal.json", "bundle-qwen38"),
                               ("models/mimo26/compiler/src/frontend.json", "bundle-mimo26")):
            with self.subTest(path=path):
                self.assertIn(consumer, quick.affected(metadata, [path]))

    def test_qwen_cuda_selects_kernel_lab(self):
        for header in ("qwen38_bundle.cuh", "qwen38_speculation.cuh"):
            with self.subTest(header=header):
                self.assertIn("kernel-lab", quick.affected(self.model_metadata(),
                              ["models/qwen38/bundle/cuda/" + header]))

    def test_dependency_kinds_and_renames_explain_conservative_dev_chains(self):
        nodes = self.metadata["resolve"]["nodes"]
        nodes[1]["deps"] = [{"name": "renamed_core", "pkg": "core",
                             "dep_kinds": [{"kind": "build", "target": "cfg(unix)"}]}]
        nodes[2]["deps"] = [{"name": "host", "pkg": "host", "dep_kinds": [{"kind": "dev"}]}]
        report = change_scope.Graph(self.metadata).select(["crates/core/src/lib.rs"])
        reason = report["reasons"]["fixture"][0]
        self.assertEqual([edge["kind"] for edge in reason["chain"]], ["owner", "Cargo build", "Cargo dev"])
        self.assertTrue(reason["conservative"])

    def test_incomplete_metadata_requires_workspace(self):
        self.metadata["resolve"]["nodes"].pop()
        report = change_scope.Graph(self.metadata).select(["crates/core/src/lib.rs"])
        self.assertTrue(report["workspace"])
        self.assertEqual(report["packages"], ["core", "fixture", "host", "other"])

    def test_transitive_includes_and_removed_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for package in self.metadata["packages"]:
                package["manifest_path"] = str(root / "crates" / package["name"] / "Cargo.toml")
            self.metadata["workspace_root"] = directory
            reader = root / "crates/other/tests/reader.rs"
            reader.parent.mkdir(parents=True)
            reader.write_text('#[path = "../../core/tests/shared.rs"] mod shared;')
            shared = root / "crates/core/tests/shared.rs"
            shared.parent.mkdir(parents=True)
            shared.write_text('const X: &str = include_str!("../../../input.json");')
            graph = change_scope.Graph(self.metadata)
            self.assertIn("other", graph.select(["input.json"])["packages"])
            self.assertTrue(any(len(r["chain"]) == 3 for r in graph.select(["input.json"])["reasons"]["other"]))
            reader.write_text("")
            old = {"crates/other/tests/reader.rs": '#[path = "../../core/tests/shared.rs"] mod shared;'}
            graph = change_scope.Graph(self.metadata, old_sources=old)
            self.assertTrue(any(len(r["chain"]) == 3 for r in graph.select(["input.json"])["reasons"]["other"]))

    def test_unresolved_include_broadens_instead_of_losing_a_consumer(self):
        with tempfile.TemporaryDirectory() as directory:
            self.metadata["workspace_root"] = directory
            for package in self.metadata["packages"]:
                package["manifest_path"] = f"{directory}/crates/{package['name']}/Cargo.toml"
            source = pathlib.Path(directory) / "crates/core/src/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text('include!(arbitrary_input!());')
            result = change_scope.Graph(self.metadata).select(["crates/host/src/lib.rs"])
            self.assertTrue(result["workspace"])
            self.assertIn("other", result["packages"])

    def test_checked_tooling_needs_no_rust_unless_rust_names_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            self.metadata["workspace_root"] = directory
            for package in self.metadata["packages"]:
                package["manifest_path"] = f"{directory}/crates/{package['name']}/Cargo.toml"
            for name in ("tools/qual/modules.json", "tools/qual/score.py", "tools/refrun/run.py",
                         "tools/dev/tool.py", "models/m/reference.py", "docs/guide.md", "README.md",
                         "NOTES.md", "Cargo.toml", "rust-toolchain.toml"):
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text("")
            source = root / "crates/host/tests/eval.rs"
            source.parent.mkdir(parents=True)
            source.write_text('const M: &str = "tools/qual/modules.json"; const R: &str = "/../../tools/refrun";\n'
                              'const N: &str = "NOTES.md";\n'
                              '// "tools/dev/tool.py" and "README.md" in a comment are no readers\n')
            # A package with no Cargo edge to host that includes host's naming source.
            include = root / "crates/other/tests/eval.rs"
            include.parent.mkdir(parents=True)
            include.write_text('#[path = "../../host/tests/eval.rs"]\nmod eval;\n')
            graph = change_scope.Graph(self.metadata)
            everything = ["core", "fixture", "host", "other"]
            # Root files other than Cargo's own inputs are tooling too.
            for path in ("tools/qual/score.py", "tools/dev/tool.py", "docs/guide.md", "README.md"):
                with self.subTest(path=path):
                    self.assertEqual(graph.select([path], checked={path})["packages"], [])
                    self.assertEqual(graph.select([path])["packages"], everything)
            # Rust names the file or its directory, a root file also without a
            # slash: the naming source's package, its dependents and whatever
            # includes the naming source.
            for path in ("tools/qual/modules.json", "tools/refrun/run.py", "NOTES.md"):
                for checked in ({path}, set()):
                    with self.subTest(path=path, checked=bool(checked)):
                        selection = graph.select([path], checked=checked)
                        self.assertEqual(selection["packages"], ["fixture", "host", "other"])
                        self.assertEqual(selection["fallback"], [])
                        self.assertEqual([step["kind"] for step in selection["reasons"]["host"][0]["chain"]],
                                         ["runtime literal", "test/source owner"])
            # Not tooling, or a file Cargo reads: still conservative.
            for path in ("models/m/reference.py", "Cargo.toml", "rust-toolchain.toml"):
                with self.subTest(path=path):
                    self.assertEqual(graph.select([path], checked={path})["packages"], everything)


class Preflight(unittest.TestCase):
    """quick refuses before it runs anything when its checks would fail on
    their environment."""

    class Ran(Exception):
        pass

    def subprocess_run(self, argv, *args, **kwargs):
        # The work directory's cleanup runs; anything else stops the check.
        if argv[0] != "chmod":
            raise self.Ran(argv)

    def check(self, packages, workspace=False, **variables):
        graph = Mock()
        graph.select.return_value = {"packages": packages, "workspace": workspace}
        args = Mock(all=False, list=False)
        env = {key: value for key, value in os.environ.items() if not key.endswith("_REFERENCE")}
        stderr = io.StringIO()
        with patch.dict(os.environ, {**env, **variables}, clear=True), \
             patch.object(quick, "tooling_checks", return_value=([], [], set())), \
             patch.object(quick.subprocess, "run", side_effect=self.subprocess_run) as run, \
             contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            try:
                status = quick.check(args, graph, ["crates/x/src/lib.rs"])
            except self.Ran:
                status = "ran"
        return status, stderr.getvalue(), [call.args[0] for call in run.call_args_list if call.args[0][0] != "chmod"]

    def test_a_missing_reference_variable_or_tmpdir_refuses_before_anything_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            for packages, workspace in ((["runtime-eval"], False), (["runtime-kv"], True)):
                with self.subTest(packages=packages, workspace=workspace):
                    status, stderr, run = self.check(packages, workspace, TMPDIR=directory)
                    self.assertEqual(status, 2)
                    for variable in ("CHOPINFER_QWEN38_LONG_REFERENCE", "CHOPINFER_QWEN38_LAYER_REFERENCE",
                                     "CHOPINFER_GLM5NEXT_LAYER_REFERENCE"):
                        self.assertIn(f"quick preflight: {variable} is unset", stderr)
                    self.assertIn("quick preflight failed: nothing ran", stderr)
                    self.assertEqual(run, [])
            status, stderr, run = self.check(["runtime-kv"], TMPDIR=str(pathlib.Path(directory) / "missing"))
            self.assertEqual(status, 2)
            self.assertIn("is not a directory: create it first", stderr)
            self.assertEqual(run, [])
            # Tests that read no reference start without them.
            status, stderr, run = self.check(["runtime-kv"], TMPDIR=directory)
            self.assertEqual((status, stderr), ("ran", ""))
            self.assertEqual([argv[:2] for argv in run], [["cargo", "fmt"]])


class AffectedChecks(unittest.TestCase):
    """Representative tooling edits select the gate's own checks for them."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.scanner = check_inputs.Scanner()

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def select(self, *paths):
        checks, gate_only, covered = quick.tooling_checks(list(paths), pathlib.Path(self.work.name), 4,
                                                          scanner=self.scanner)
        return [step.name for step in checks], gate_only, covered

    def test_tooling_edits_select_their_tests(self):
        for path, expected in (("ci/runner.py", {"ci-runner", "quick-selection"}),
                               ("ci/check_inputs.py", {"quick-selection"}),
                               ("tools/dev/benchmark-loop.py", {"benchmark-loop"}),
                               ("tools/dev/startup_timing.py", {"startup-timing"}),
                               ("tools/dev/test_startup_timing.py", {"startup-timing"}),
                               ("tools/refrun/run.py", {"refrun", "refrun-qwen38", "refrun-python-compile"}),
                               ("tools/qual/tapdiff.py", {"tapdiff"}),
                               ("tests/leak_guard.py", {"leak-guard"}),
                               ("docs/testing.md", {"project-layout"})):
            with self.subTest(path=path):
                checks, _, covered = self.select(path)
                self.assertLessEqual(expected, set(checks))
                self.assertEqual(covered, {path})

    def test_unrelated_checks_stay_out(self):
        checks, gate_only, _ = self.select("tools/dev/benchmark-loop.py")
        self.assertEqual(set(checks), {"benchmark-loop", "quick-selection"})
        self.assertEqual(gate_only, [])
        checks = self.select("crates/runtime-host/src/offload.rs")[0]
        self.assertFalse({"refrun", "release-gate", "profile-tools"} & set(checks))
        # Profile tools read kernel sources and need a scorer for their
        # control-rate checks, so their build and copy belong in this selection.
        checks = self.select("crates/kernels-sm120/cuda/linear/delta.cuh")[0]
        self.assertLessEqual({"profile-tools", "profile-cli-build", "preparation-fixture-build", "profile-copy"},
                             set(checks))

    def test_python_scorer_consumers_bring_a_fresh_copy(self):
        for path, name in (("tests/test_validation.py", "validation-report"),
                           ("tests/test_spec_bench.py", "spec-bench"),
                           ("tests/test_deploy_artifacts.py", "deploy-artifacts"),
                           ("tools/profile/test_qwen38_hc_linear_e5.py", "profile-tools")):
            with self.subTest(step=name), patch.dict(os.environ, CHOPINFER_EVAL="/stale/chopinfer"):
                checks, _, _ = quick.tooling_checks([path], pathlib.Path(self.work.name), 4, scanner=self.scanner)
                steps = {step.name: step for step in checks}
                self.assertLessEqual({"profile-cli-build", "preparation-fixture-build", "profile-copy", name},
                                     set(steps))
                self.assertIn("profile-copy", steps[name].after)
                self.assertEqual(steps[name].env["CHOPINFER_EVAL"],
                                 str(pathlib.Path(self.work.name) / "profile-target/debug/chopinfer"))

    def qualification_modules(self, path):
        checks, _, covered = quick.tooling_checks([path], pathlib.Path(self.work.name), 4, scanner=self.scanner)
        steps = {step.name: step for step in checks}
        self.assertEqual(covered, {path})
        run = steps["release-gate"]
        self.assertEqual(run.argv[:2], ["bash", "ci/qualification.sh"])
        # The run scores with the profile copy, which quick builds and copies first.
        self.assertEqual([name for name in steps if name in ("profile-cli-build", "preparation-fixture-build",
                                                             "profile-copy", "release-gate")],
                         ["profile-cli-build", "preparation-fixture-build", "profile-copy", "release-gate"])
        self.assertEqual(run.after, ("profile-copy",))
        return [run.argv[index + 1] for index, argument in enumerate(run.argv) if argument == "--module"]

    def test_qualification_modules_run_for_their_tests_and_sources(self):
        # A changed test module runs, not only compiles.
        self.assertIn("test_checks", self.qualification_modules("tools/qual/test_checks.py"))
        self.assertIn("test_verify_exact", self.qualification_modules("tools/qual/test_verify_exact.py"))
        # So do the modules that test a changed source.
        self.assertIn("test_verify_exact", self.qualification_modules("tools/qual/verify_exact.py"))
        # Only those: a narrow source selects its tests, not every module.
        modules = self.qualification_modules("tools/qual/color_probe.py")
        self.assertIn("test_color_probe", modules)
        self.assertNotIn("test_tapdiff", modules)

    def test_portable_shell_guards_run_when_changed(self):
        for path, name in (("tests/bundle-boundary.sh", "bundle-boundary"),
                           ("tests/model-neutral-host.sh", "model-neutral-host")):
            with self.subTest(path=path):
                checks, _, covered = self.select(path)
                self.assertIn(name, checks)
                self.assertEqual(covered, {path})

    def test_compile_steps_alone_do_not_cover_a_file(self):
        import cpu
        import qualification
        compile_only = runner.Step("qual-python-compile", ["bash", "-c", "python3 -m py_compile tools/x/*.py"])
        scanner = check_inputs.Scanner(change_scope.ROOT, files=["tools/x/tool.py"])
        with patch.object(cpu, "steps", return_value=([], [compile_only])), \
             patch.object(qualification, "modules", return_value=[]):
            checks, _, covered = quick.tooling_checks(["tools/x/tool.py"], pathlib.Path(self.work.name), 4,
                                                      scanner=scanner)
        self.assertEqual([step.name for step in checks], ["qual-python-compile"])
        self.assertEqual(covered, set())

    def test_fresh_build_checks_are_left_to_the_gate(self):
        checks, gate_only, covered = self.select("tests/model-geometry-recipe.sh")
        self.assertIn("geometry-recipe", gate_only)
        self.assertNotIn("geometry-recipe", checks)
        self.assertEqual(covered, {"tests/model-geometry-recipe.sh"})

    def test_selected_checks_wait_only_for_selected_steps(self):
        checks, _, _ = quick.tooling_checks(["models/qwen38/test_reference.py"], pathlib.Path(self.work.name), 4,
                                            scanner=self.scanner)
        names = {step.name for step in checks}
        self.assertIn("refrun-qwen38", names)
        # The setup its reference steps run after comes with them: the prerequisite
        # check and the profile copy whose chopinfer they score with.
        self.assertLessEqual({"reference-prerequisites", "profile-cli-build", "preparation-fixture-build",
                              "profile-copy"}, names)
        self.assertNotIn("refrun-mimo26", names)
        for step in checks:
            self.assertLessEqual(set(step.after), names)
            self.assertFalse(step.echo)
        self.assertIn("profile-copy", next(step for step in checks if step.name == "refrun-qwen38").after)


class CheckInputs(unittest.TestCase):
    FILES = {
        "tests/test_tool.py": 'ROOT = X / "tools" / "dev"\nsys.path.insert(0, str(ROOT))\nimport helper\n'
                              'from tools.qual import common\nDATA = "fixtures/cases.json"\n'
                              'PATTERN = "models/*/model.toml"\nWIDE = "*.json"\nrun(["bash", "tests/run.sh"])\n',
        "tools/dev/helper.py": "from . import sibling\n",
        "tools/dev/sibling.py": "",
        "tools/dev/unused.py": "",
        "tools/dev/notes.json": "",
        "tools/dev/other.sh": "",
        "tools/qual/__init__.py": "",
        "tools/qual/common.py": "",
        "tests/run.sh": 'python3 -B "$root/tools/dev/script.py"\n',
        "tools/dev/script.py": "",
        "fixtures/cases.json": "",
        "models/a/model.toml": "",
        "tools/unit/test_x.py": "",
        "tools/unit/data/x.bin": "",
        "tools/qual/test_mod.py": "",
        # Literal globs, loads by module name, helpers that start scripts, and tree walks.
        "tests/test_loads.py": 'import importlib\nfrom pathlib import Path\nfrom tools import driver\n'
                               'importlib.import_module("tools.worker")\n'
                               'for path in Path(".").glob("tools/plugins/*.py"):\n    pass\n'
                               'for path in Path("fixtures/tree").rglob("*.json"):\n    pass\n',
        "tools/__init__.py": "",
        "tools/driver.py": 'import subprocess\nsubprocess.run(["bash", "tools/start.sh"])\n',
        "tools/start.sh": "python3 tools/started.py\n",
        "tools/started.py": "",
        "tools/worker.py": "",
        "tools/plugins/entry.py": "from tools import shared\n",
        "tools/shared.py": "",
        "fixtures/tree/nested/one.json": "",
        # A load from a computed path covers the modules of the directories it names.
        "tests/test_computed.py": 'import importlib.util\nCI = ROOT / "ci"\n'
                                  'importlib.util.spec_from_file_location(name, CI / (name + ".py"))\n',
        "ci/loaded.py": "import tools.unloaded\n",
        "tools/unloaded.py": "",
    }

    def scanner(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = pathlib.Path(directory.name)
        for name, text in self.FILES.items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(text)
        return check_inputs.Scanner(root, files=self.FILES)

    def test_imports_scripts_and_named_data_close_transitively(self):
        inputs = self.scanner().inputs(["python3", "-B", "tests/test_tool.py"])
        for path in ("tests/test_tool.py", "tools/dev/helper.py", "tools/dev/sibling.py", "tools/qual/common.py",
                     "tools/qual/__init__.py", "tests/run.sh", "tools/dev/script.py", "fixtures/cases.json",
                     "models/a/model.toml", "tools/dev/notes.json"):
            self.assertTrue(inputs.covers(path), path)
        # A directory on the import path covers its data, not its unused modules; a bare glob covers nothing.
        for path in ("tools/dev/unused.py", "tools/dev/other.sh", "tools/unit/test_x.py", "crates/x/a.json"):
            self.assertFalse(inputs.covers(path), path)

    def test_globs_named_modules_helper_scripts_and_walked_trees(self):
        inputs = self.scanner().inputs(["python3", "-B", "tests/test_loads.py"])
        for path in ("tools/worker.py", "tools/plugins/entry.py", "tools/shared.py", "tools/driver.py",
                     "tools/start.sh", "tools/started.py", "fixtures/tree/nested/one.json"):
            self.assertTrue(inputs.covers(path), path)
        shell = self.scanner().inputs(["bash", "-c", "python3 tools/plugins/*.py"])
        self.assertTrue(shell.covers("tools/shared.py"))

    def test_computed_loads_cover_the_named_directories_modules(self):
        inputs = self.scanner().inputs(["python3", "-B", "tests/test_computed.py"])
        self.assertTrue(inputs.covers("ci/loaded.py"))
        # Covered, not scanned in turn.
        self.assertFalse(inputs.covers("tools/unloaded.py"))

    def test_the_model_inventory_check_covers_the_ci_modules_its_tests_load(self):
        import cpu
        with tempfile.TemporaryDirectory() as directory:
            _, independent = cpu.steps(pathlib.Path(directory) / "target", pathlib.Path(directory), 4, 4)
        step = next(step for step in independent if step.name == "model-inventory")
        inputs = check_inputs.Scanner().inputs(step.argv)
        for path in ("ci/contributor.py", "ci/model_checks.py", "models/mimo26/test_inventory.py"):
            self.assertTrue(inputs.covers(path), path)

    def test_discovered_trees_and_unittest_modules(self):
        scanner = self.scanner()
        tree = scanner.inputs(["python3", "-m", "unittest", "discover", "-s", "tools/unit", "-p", "test_*.py"])
        self.assertTrue(tree.covers("tools/unit/data/x.bin"))
        self.assertTrue(tree.covers("tools/unit/new_test.py"))
        named = scanner.inputs(["python3", "-m", "unittest", "tools.qual.test_mod", "-v"])
        self.assertTrue(named.covers("tools/qual/test_mod.py"))
        self.assertFalse(named.covers("tools/qual/common.py"))


class Contributor(unittest.TestCase):
    def setUp(self):
        import contributor
        sys.path.insert(0, str(change_scope.ROOT))
        self.module = contributor
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.scratch = pathlib.Path(self.temporary.name)
        self.env = dict(os.environ, CARGO_TARGET_DIR=str(self.scratch / "target"),
                        CHOPINFER_SCRATCH=str(self.scratch), CHOPINFER_CUDA_HOME=str(self.scratch / "no-cuda"))

    def test_discovered_plans_keep_shared_sdk_bookends(self):
        for mode in self.module.modes():
            with self.subTest(mode=mode):
                commands = self.module.steps(mode)[2:]
                first, last = commands[0], commands[-1]
                self.assertEqual(first["phase"], "compile")
                self.assertEqual(first["argv"][:4], ["cargo", "test", "-p", "model-compiler-sdk"])
                self.assertIn("--no-run", first["argv"])
                self.assertEqual(last["argv"][:6], ["cargo", "test", "-p", "model-compiler-sdk", "--test", "emitter_contract"])
                self.assertTrue(last["tests"])

    def probe(self, command, *args, **kwargs):
        return "rustc 1.89.0 (fixture)" if command[0] == "rustc" else "available"

    def test_doctor_reports_portable_setup_and_native_requirements(self):
        graph = Mock()
        graph.cuda_build_owners.return_value = []
        with patch.dict(os.environ, {"RUSTUP_AUTO_INSTALL":"1"}), \
             patch.object(self.module.shutil, "which", return_value="available"), \
             patch.object(self.module, "probe", side_effect=self.probe) as probes, \
             patch("tools.model_sdk.graph.metadata", return_value={}), \
             patch("tools.model_sdk.graph.selected_graph", return_value=(graph, ["compiler"])):
            for mode in ("qwen38", "mimo26"):
                result = self.module.preflight(mode, env=self.env)
                self.assertIsNone(result["nvcc"])
                graph.require_cuda_free.assert_called_with(self.module.roots(mode))
            self.assertEqual(os.environ["RUSTUP_AUTO_INSTALL"], "0")
            self.assertTrue(all(call.kwargs["env"]["RUSTUP_AUTO_INSTALL"] == "0" for call in probes.call_args_list))
            graph.cuda_build_owners.return_value = ["native-bundle"]
            with self.assertRaisesRegex(ValueError, "CUDA 13.2.86.*native-bundle"):
                self.module.preflight(None, env=self.env)

    def test_doctor_missing_tools_components_cache_and_assets(self):
        for tool in ("git", "cargo", "rustc", "cc"):
            with self.subTest(tool=tool), patch.object(self.module.shutil, "which", side_effect=lambda name, **_: None if name == tool else "available"):
                with self.assertRaisesRegex(ValueError, "missing " + tool):
                    self.module.preflight("sdk", env=self.env)
        with patch.object(self.module.shutil, "which", return_value="available"):
            for component in ("fmt", "clippy"):
                def probe(command, remedy, **kwargs):
                    if command[:2] == ["cargo", component]:
                        raise ValueError(remedy)
                    return self.probe(command)
                with self.subTest(component=component), patch.object(self.module, "probe", side_effect=probe):
                    with self.assertRaisesRegex(ValueError, "component"):
                        self.module.preflight("sdk", env=self.env)
            with patch.object(self.module, "probe", side_effect=self.probe), \
                 patch("tools.model_sdk.graph.metadata", side_effect=subprocess.CalledProcessError(101, ["cargo"])):
                with self.assertRaisesRegex(ValueError, "cargo fetch --locked"):
                    self.module.preflight("sdk", env=self.env)
            with patch.object(self.module, "probe", side_effect=self.probe):
                for mode in ("qwen38", "mimo26"):
                    with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "restore missing fixture inputs"):
                        self.module.preflight(mode, root=self.scratch / "checkout", env=self.env)

    def test_doctor_reports_the_gate_tmpdir_limit(self):
        long = self.scratch / ("t" * 80)
        long.mkdir()
        with patch.object(self.module.shutil, "which", return_value="available"), \
             patch.object(self.module, "probe", side_effect=self.probe):
            with self.assertRaisesRegex(ValueError, "more than 64"):
                self.module.preflight("sdk", env=dict(self.env, TMPDIR=str(long)))

    def test_mimo_fixtures_do_not_inherit_checkpoint_or_module_inputs(self):
        inputs = {key:"/unused/production" for key in (
            "CHOPINFER_MIMO_CHECKPOINT", "CHOPINFER_MIMO_TOKENIZER_DIR",
            "CHOPINFER_MIMO_FIXTURES", "CHOPINFER_PACKAGE_NATIVE_ROOT",
            "CHOPINFER_PACKAGE_SNAPSHOT")}
        with patch.dict(os.environ, inputs):
            env = self.module.environment()
        self.assertTrue(inputs.keys().isdisjoint(env))
        self.assertEqual(env["CARGO_NET_OFFLINE"], "true")

    def test_shell_reports_missing_python_before_inventory(self):
        checkout = self.scratch / "checkout"
        (checkout / "ci").mkdir(parents=True)
        script = checkout / "ci/quick.sh"
        script.write_bytes((change_scope.ROOT / "ci/quick.sh").read_bytes())
        binaries = self.scratch / "bin"
        binaries.mkdir()
        (binaries / "dirname").symlink_to(shutil.which("dirname"))
        env = dict(self.env, PATH=str(binaries), CHOPINFER_LOCAL_ENV=str(self.scratch / "absent.env"))
        result = subprocess.run(["/bin/bash", str(script), "--contributor", "qwen38", "--doctor"], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Python 3.11", result.stderr)
        self.assertNotIn("model_inventory", result.stderr)

    def test_gate_refuses_zero_tests_ignored_tests_and_command_failures(self):
        for output in ("", "test result: ok. 0 passed; 0 failed; 0 ignored", "test result: ok. 1 passed; 0 failed; 1 ignored"):
            with self.assertRaisesRegex(ValueError, "nonzero passing"):
                self.module.test_counts(output)
        self.assertEqual(self.module.test_counts("test result: ok. 2 passed; 0 failed; 0 ignored")["passed"], 2)
        declared = {("model-x-contract", "graph::complete_sweep")}
        output = ("test graph::sampled ... ok\ntest graph::complete_sweep ... ignored, about 8 minutes\n"
                  "test result: ok. 1 passed; 0 failed; 1 ignored")
        counts = self.module.test_counts(output, "model-x-contract", declared)
        self.assertEqual((counts["passed"], counts["ignored"], counts["exhaustive"]), (1, 1, ["graph::complete_sweep"]))
        # The same name in another package, or another ignored case beside it, still fails.
        with self.assertRaisesRegex(ValueError, "unexpected: graph::complete_sweep"):
            self.module.test_counts(output, "model-y-contract", declared)
        with self.assertRaisesRegex(ValueError, "unexpected: graph::other"):
            self.module.test_counts(output.replace("1 ignored", "2 ignored")
                                    + "\ntest graph::other ... ignored", "model-x-contract", declared)
        step = self.module.steps("qwen38")[-1]
        with patch.object(self.module.subprocess, "run", return_value=subprocess.CompletedProcess([], 7, "fixture failed")):
            with self.assertRaises(subprocess.CalledProcessError) as failure:
                self.module.execute(step, env=self.env)
            self.assertEqual(failure.exception.returncode, 7)

    def undeclared_ignored(self, declared):
        """(mode, step, test) of each #[ignore]d test a contributor step
        selects that `declared` does not name as an exhaustive comparison."""
        packages = {tomllib.loads(path.read_text())["package"]["name"]: path.parent
                    for path in [*change_scope.ROOT.glob("crates/*/Cargo.toml"),
                                 *change_scope.ROOT.glob("models/*/*/Cargo.toml")]}
        ignored_fn = re.compile(r"#\[ignore\b[^\]]*\]\s*(?:#\[[^\]]*\]\s*)*(?:async\s+)?fn (\w+)\(")
        found = []
        for mode in self.module.modes():
            for step in self.module.steps(mode):
                if not step["tests"]:
                    continue
                argv = step["argv"]
                package = argv[argv.index("-p") + 1]
                directory = packages[package]
                sources = []
                if "--lib" in argv or "--bin" in argv:
                    sources += directory.glob("src/**/*.rs")
                for i, word in enumerate(argv):
                    if word == "--test":
                        sources += [directory / "tests" / (argv[i + 1] + ".rs"),
                                    *directory.glob(f"tests/{argv[i + 1]}/**/*.rs")]
                exact = argv[argv.index("--exact") - 2] if "--exact" in argv else None
                names = {name for path in sources if path.is_file() for name in ignored_fn.findall(path.read_text())}
                if exact:
                    names &= {exact.rsplit("::", 1)[-1]}
                allowed = {test.rsplit("::", 1)[-1] for owner, test in declared if owner == package}
                found += [(mode, step["name"], name) for name in sorted(names - allowed)]
        return found

    @staticmethod
    def case_source(directory, argv, case):
        """The source file that defines the exact test `case` (a module path
        and a function) in the target the step selects, following inline
        modules, MOD.rs, MOD/mod.rs and #[path], or None."""
        if "--lib" in argv:
            root = directory / "src/lib.rs"
        elif "--bin" in argv:
            name = argv[argv.index("--bin") + 1]
            declared = {row["name"]: row["path"] for row in
                        tomllib.loads((directory / "Cargo.toml").read_text()).get("bin", []) if "path" in row}
            root = directory / declared.get(name, f"src/bin/{name}.rs")
        else:
            root = directory / "tests" / (argv[argv.index("--test") + 1] + ".rs")
        *modules, function = case.split("::")
        current = root
        folder = root.parent if root.name in ("lib.rs", "main.rs", "mod.rs") or root.parent.name == "tests" \
            else root.parent / root.stem
        if root.parent.name == "tests" or root.parent.name == "bin":
            folder = root.parent / root.stem
        for module in modules:
            if not current.is_file():
                return None
            text = current.read_text()
            if re.search(rf"\bmod {module}\s*\{{", text):
                folder = folder / module
                continue
            path = re.search(rf'#\[path\s*=\s*"([^"]+)"\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod {module}\s*;',
                             text)
            if path:
                current = (current.parent / path.group(1)).resolve()
                folder = current.parent / current.stem
            elif (folder / f"{module}.rs").is_file():
                current, folder = folder / f"{module}.rs", folder / module
            elif (folder / module / "mod.rs").is_file():
                current, folder = folder / module / "mod.rs", folder / module
            else:
                return None
        return current if current.is_file() and re.search(rf"\bfn {function}\(", current.read_text()) else None

    def test_contributor_exact_cases_name_tests_that_exist(self):
        # An exact case that names no test runs zero tests, which the
        # contributor rule refuses; a moved test must move its declaration.
        packages = {tomllib.loads(path.read_text())["package"]["name"]: path.parent
                    for path in [*change_scope.ROOT.glob("crates/*/Cargo.toml"),
                                 *change_scope.ROOT.glob("models/*/*/Cargo.toml")]}
        missing = []
        for mode in self.module.modes():
            for step in self.module.steps(mode):
                argv = step["argv"]
                if "--exact" in argv:
                    case = argv[argv.index("--exact") - 2]
                    if self.case_source(packages[argv[argv.index("-p") + 1]], argv, case) is None:
                        missing.append((mode, step["name"], case))
        self.assertEqual(missing, [])

    def test_contributor_selections_ignore_only_declared_exhaustive_comparisons(self):
        # A contributor run fails on any other ignored case, so an #[ignore]
        # added to a selected target must be declared in the exhaustive
        # profile (or kept out of the selection) when it lands.
        declared = self.module.exhaustive()
        self.assertIn(("model-mimo26-contract", "projections::variants::tests::graph::"
                       "complete_served_graphs_match_native_variant_specs_records_and_stages"), declared)
        self.assertEqual(self.undeclared_ignored(declared), [])

    def test_contributor_dispatch_uses_the_landed_validation_report(self):
        import validation
        sample = Selection()
        sample.setUp()
        report = {"obligations":[]}
        def output(*command):
            if command[0] == "cargo":
                return json.dumps(sample.metadata).encode()
            return b"revision" if command[1] == "rev-parse" else b""
        with patch.object(quick, "output", side_effect=output), \
             patch.object(self.module, "preflight", return_value={}), \
             patch.object(self.module, "run", return_value=0) as run, \
             patch.object(validation, "Catalog"), \
             patch.object(validation, "build_report", return_value=report) as build, \
             patch.object(quick.subprocess, "run") as cargo:
            for mode in ("qwen38", "mimo26"):
                self.assertEqual(quick.main(["--contributor", mode, "--doctor"]), 0)
                self.assertIs(run.call_args.args[1], report)
                self.assertTrue(run.call_args.args[0].doctor)
                self.assertEqual(run.call_args.args[0].contributor, mode)
            self.assertEqual(build.call_count, 2)
            cargo.assert_not_called()

    def test_contributor_modes_stay_bounded(self):
        """Contributor runs need no reference Python (Torch), weights or CUDA, and run
        no gate checks: only their declared Cargo fixtures."""
        import model_checks
        sample = Selection()
        sample.setUp()
        def output(*command):
            if command[0] == "cargo":
                return json.dumps(sample.metadata).encode()
            return b"revision" if command[1] == "rev-parse" else b"tools/refrun/run.py\0ci/runner.py\0"
        refused = AssertionError("contributor mode reached a gate check")
        with patch.object(quick, "output", side_effect=output), \
             patch.object(self.module, "preflight", return_value={}), \
             patch.object(self.module, "run", return_value=0) as run, \
             patch.object(quick, "tooling_checks", side_effect=refused), \
             patch.object(model_checks, "reference_python", side_effect=refused), \
             patch("validation.Catalog"), patch("validation.build_report", return_value={"obligations":[]}):
            for mode in self.module.modes():
                self.assertEqual(quick.main(["--contributor", mode]), 0)
                self.assertEqual(run.call_args.args[0].contributor, mode)
        graph = Mock()
        graph.cuda_build_owners.return_value = []
        env = {key: value for key, value in self.env.items() if key != "CHOPINFER_REFRUN_PYTHON"}
        with patch.object(self.module.shutil, "which", return_value="available"), \
             patch.object(self.module, "probe", side_effect=self.probe), \
             patch("tools.model_sdk.graph.metadata", return_value={}), \
             patch("tools.model_sdk.graph.selected_graph", return_value=(graph, ["compiler"])), \
             patch.object(model_checks, "reference_python", side_effect=refused):
            for mode in self.module.modes():
                self.module.preflight(mode, env=env)
            with self.assertRaises(AssertionError):
                self.module.preflight(None, env=env)
        for mode in self.module.modes():
            for step in self.module.steps(mode):
                self.assertEqual(step["argv"][0], "cargo")
                self.assertFalse(any("refrun" in argument or argument.endswith(".py") for argument in step["argv"]))

    def test_outstanding_scope_uses_policy_satisfaction_without_reclassifying(self):
        report = {"obligations":[
            {"id":"cpu", "evidence":{"selection":"selected", "satisfied":False}},
            {"id":"device", "evidence":{"selection":"selected", "satisfied":True}},
            {"id":"unused", "evidence":{"selection":"not_applicable"}},
        ]}
        self.assertEqual([row["id"] for row in self.module.outstanding(report)], ["cpu"])
        with self.assertRaisesRegex(ValueError, "outside"):
            self.module.external_path(str(change_scope.ROOT / "target"), "CARGO_TARGET_DIR")
        with patch.object(self.module.shutil, "which", return_value="available"), \
             patch.object(self.module, "probe", side_effect=self.probe):
            for key in ("CHOPINFER_SCRATCH", "TMPDIR"):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "outside"):
                    self.module.preflight("sdk", env=dict(self.env, **{key:str(change_scope.ROOT / "tmp")}))
        for mode in self.module.modes():
            steps = self.module.steps(mode)
            self.assertTrue(any(step["tests"] for step in steps))
            self.assertFalse(any("ci/cpu.sh" in step["argv"] for step in steps))


if __name__ == "__main__":
    unittest.main()

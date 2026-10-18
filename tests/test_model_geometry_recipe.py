"""Check that the recipe proof rejects code edits and changed lock records."""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import model_geometry_recipe
from model_geometry_recipe import (assert_contract_data_diff, assert_data_diff, assert_mimo_layouts,
                                   assert_tests_ran)
import model_inventory as inventory

ROOT = Path(__file__).resolve().parents[1]


def setUpModule():
    # Fixture repositories and the recipe under test must not inherit the
    # calling Git's repository: `git rebase -x` in a linked worktree exports
    # GIT_DIR, and a fixture's `git init` and commits would write it.
    environment = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1')
    isolated = patch.dict(os.environ, environment, clear=True)
    isolated.start()
    unittest.addModuleCleanup(isolated.stop)


class SignalCleanupTests(unittest.TestCase):
    def test_sigterm_removes_the_worktree_and_target(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / 'repo'
            root.mkdir()
            def git(*args):
                return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                                               stderr=subprocess.DEVNULL)
            git('init')
            fixture = 'fixtures/qwen38/geometries/synthetic-small'
            shutil.copytree(ROOT / fixture, root / fixture)
            geometry = root / 'models/qwen38/geometries'
            geometry.mkdir(parents=True)
            (geometry / 'supported.json').write_text('{"aliases":{"architecture":[]},"geometries":[]}')
            lock = root / 'models/qwen38/bundle/kernels.lock'
            lock.parent.mkdir()
            lock.write_text('[{"name":"old"}]')
            git('add', 'models/qwen38/geometries/supported.json',
                'models/qwen38/bundle/kernels.lock',
                *(f'{fixture}/{name}' for name in ('config.json', 'geometry.json', 'supported.json')))
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                'commit', '-m', 'Create fixture')
            scratch = base / 'scratch'
            scratch.mkdir()
            marker = base / 'ready'
            script = '''import json, os, signal, sys, tempfile
from pathlib import Path
import model_geometry_recipe as recipe
recipe.ROOT = Path(sys.argv[1])
recipe.shutil.which = lambda _: "fixture-tool"
Path.is_file = lambda _: True
run = recipe.run
def pause_build(*args, cwd=None, env=None):
    if args[0] == "cargo":
        Path(env["CARGO_TARGET_DIR"]).mkdir(exist_ok=True)
        Path(sys.argv[2]).write_text(json.dumps({"scratch": str(Path(cwd).parent), "target": env["CARGO_TARGET_DIR"]}))
        signal.pause()
    return run(*args, cwd=cwd, env=env)
recipe.run = pause_build
with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as target:
    recipe.recipe("qwen38", target_dir=Path(target))
'''
            env = dict(os.environ, TMPDIR=str(scratch),
                       CHOPINFER_QWEN38_TOKENIZER_DIR=str(base / 'tokenizer'),
                       PYTHONPATH=str(ROOT / 'tests'))
            child = subprocess.Popen([sys.executable, '-B', '-c', script, str(root), str(marker)],
                                     env=env, text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 10
                while not marker.exists() and child.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.exists())
                child.send_signal(signal.SIGTERM)
                output, _ = child.communicate(timeout=10)
                self.assertEqual(child.returncode, 143, output)
                for path in json.loads(marker.read_text()).values():
                    self.assertFalse(Path(path).exists())
                self.assertEqual(git('worktree', 'list', '--porcelain').count('worktree '), 1)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate()


class ContractStageRootTests(unittest.TestCase):
    """The contract-stage recipe checks, branches from and cleans up its own
    repository, whatever directory it is started from."""

    def test_the_recipe_uses_its_own_repository_from_any_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root, other, outside = base / 'repo', base / 'other', base / 'outside'
            for directory in (root, other, outside):
                directory.mkdir()
            def git(repository, *args):
                return subprocess.check_output(['git', '-C', str(repository), *args], text=True,
                                               stderr=subprocess.DEVNULL)
            def commit(repository, *paths):
                git(repository, 'add', *paths)
                git(repository, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-m', 'Create fixture')
            fixture = 'fixtures/glm5next/geometries/synthetic-small'
            supported = 'models/glm5next/geometries/supported.json'
            git(root, 'init')
            shutil.copytree(ROOT / fixture, root / fixture)
            (root / supported).parent.mkdir(parents=True)
            shutil.copyfile(ROOT / supported, root / supported)
            commit(root, fixture, supported)
            # Another clean checkout without the fixture.
            git(other, 'init')
            (other / 'README').write_text('another checkout\n')
            commit(other, 'README')
            scratch = base / 'scratch'
            scratch.mkdir()
            script = '''import sys
from pathlib import Path
import model_geometry_recipe as recipe
recipe.ROOT = Path(sys.argv[1])
recipe.shutil.which = lambda _: "fixture-tool"
run = recipe.run
def cargo(*args, cwd=None, env=None):
    if args[0] != "cargo":
        return run(*args, cwd=cwd, env=env)
    passed = "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\\n"
    if "--test" in args:
        return passed * 2 + env["CHOPINFER_GLM5NEXT_RECIPE_GEOMETRY"] + " TP4: planned\\n"
    return passed
recipe.run = cargo
recipe.contract_recipe("glm5next", "synthetic-small")
'''
            env = dict(os.environ, TMPDIR=str(scratch), PYTHONPATH=str(ROOT / 'tests'),
                       GIT_CEILING_DIRECTORIES=str(base))
            for start in (outside, other):
                with self.subTest(start=start.name):
                    result = subprocess.run([sys.executable, '-B', '-c', script, str(root)], cwd=start,
                                            env=env, text=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT)
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertIn('glm5next/synthetic-small: contract stage', result.stdout)
                    for repository in (root, other):
                        self.assertEqual(git(repository, 'worktree', 'list', '--porcelain').count('worktree '), 1)
                        self.assertEqual(git(repository, 'status', '--porcelain'), '')


def declared_model(name, stage, fixtures):
    """A model whose inventory declares geometry recipe legs."""
    recipe = inventory.GeometryRecipe(stage, tuple(fixtures), 'Recipe', ('suite',) if stage == 'bundle' else ())
    return inventory.Model(name, name.upper(), {}, geometry_recipe=recipe)


class LegTests(unittest.TestCase):
    """Each model's model.toml `[geometry_recipe]` declares its legs; each
    model's own test_inventory.py pins them."""
    MODELS = [declared_model('a', 'contract', ['small']), inventory.Model('b', None, {}),
              declared_model('c', 'bundle', ['small', 'wide']), declared_model('d', 'bundle', ['small'])]

    def test_legs_follow_each_models_stage_and_fixtures(self):
        legs = lambda *selection: model_geometry_recipe.legs(*selection, models=self.MODELS)
        self.assertEqual(legs('--ci'), [('c', 'small'), ('c', 'wide'), ('d', 'small')])
        self.assertEqual(legs('--contract'), [('a', 'small')])
        self.assertEqual(legs('c'), [('c', 'small'), ('c', 'wide')])
        self.assertEqual(legs('a'), [('a', 'small')])
        self.assertEqual(legs('--ci', 'other'), [('c', 'other'), ('d', 'other')])
        self.assertEqual(legs('d', 'other'), [('d', 'other')])
        for undeclared in ('b', 'e', '--all'):
            with self.subTest(undeclared), self.assertRaises(SystemExit) as caught:
                legs(undeclared)
            self.assertEqual(str(caught.exception), 'usage: model-geometry-recipe.sh --ci|--contract|a|c|d [fixture]')

    def test_the_inventorys_legs_run_once_each(self):
        declared = model_geometry_recipe.declared()
        expected = [(name, fixture) for name, model in declared.items() for fixture in model.geometry_recipe.fixtures]
        planned = model_geometry_recipe.legs('--ci') + model_geometry_recipe.legs('--contract')
        self.assertEqual(sorted(planned), sorted(expected))
        self.assertEqual(len(set(planned)), len(planned))
        for name in declared:
            self.assertEqual(model_geometry_recipe.legs(name), [leg for leg in expected if leg[0] == name])


class EmptyStageTests(unittest.TestCase):
    def test_a_stage_no_model_declares_runs_nothing_and_succeeds(self):
        """`--ci` or `--contract` with no declared legs, as when the last
        contract-stage model moves to the bundle stage, allocates no lane or
        target at any lane count."""
        def refuse(*args, **kwargs):
            raise AssertionError(f'allocated a lane or target: {args}')
        models = {'--ci': [declared_model('a', 'contract', ['small'])],
                  '--contract': [declared_model('c', 'bundle', ['small'])]}
        for selection, declared in models.items():
            for count in ('1', '2'):
                with self.subTest(selection=selection, lanes=count), \
                        patch.dict(os.environ, CHOPINFER_GEOMETRY_LANES=count), \
                        patch.object(model_geometry_recipe, 'run_lanes', refuse), \
                        patch.object(model_geometry_recipe, 'run_lane', refuse), \
                        patch.object(model_geometry_recipe.tempfile, 'TemporaryDirectory', refuse), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(model_geometry_recipe.main([selection], models=declared), 0)
                    self.assertEqual(output.getvalue(),
                                     f'geometry recipe {selection}: no model declares these legs, nothing to run\n')


class LaneTests(unittest.TestCase):
    def test_lanes_keep_a_models_legs_together_or_split_every_leg(self):
        planned = model_geometry_recipe.legs('--ci', models=LegTests.MODELS)
        self.assertEqual(model_geometry_recipe.lanes(planned, 1), [planned])
        self.assertEqual(model_geometry_recipe.lanes(planned, 2), [planned[:2], planned[2:]])
        self.assertEqual(model_geometry_recipe.lanes(planned, 3), [[leg] for leg in planned])

    def run_lanes(self, lane_program, base):
        """Run two lanes whose program is `lane_program`, from a fixture
        repository and with TMPDIR at base/scratch; return the child. Every Git
        call is recorded for `assert_git_calls`."""
        repository = base / 'repository'
        subprocess.run(['git', 'init', '-q', str(repository)], check=True)
        tools = base / 'tools'
        tools.mkdir()
        git = tools / 'git'
        git.write_text(f'''#!/bin/sh
printf '%s\\t%s\\n' "$(pwd -P)" "$*" >> {shlex.quote(str(base / 'git-calls'))}
exec {shlex.quote(shutil.which('git'))} "$@"
''')
        git.chmod(0o755)
        stub = base / 'scratch/lane.py'
        stub.write_text(lane_program)
        script = f'''import sys
from pathlib import Path
import model_geometry_recipe as recipe
recipe.ROOT = Path({str(repository)!r})
recipe.__file__ = {str(stub)!r}
sys.exit(recipe.run_lanes([[("qwen38", "a"), ("qwen38", "b")], [("mimo26", "c")]], 10))
'''
        env = dict(os.environ, TMPDIR=str(base / 'scratch'), PATH=str(tools) + os.pathsep + os.environ['PATH'],
                   PYTHONPATH=os.pathsep.join((str(ROOT / 'tests'), str(ROOT / 'ci'))))
        return subprocess.Popen([sys.executable, '-B', '-c', script], env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def assert_git_calls(self, base):
        """The lanes' only Git call prunes stale worktrees of the fixture."""
        calls = (base / 'git-calls').read_text().splitlines()
        self.assertEqual(calls, [f"{(base / 'repository').resolve()}\tworktree prune"])

    def test_each_lane_gets_its_own_target_and_a_share_of_the_build_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            scratch = base / 'scratch'
            scratch.mkdir()
            program = '''import json, os, sys
from pathlib import Path
print(json.dumps({"lane": sys.argv[2], "target": sys.argv[4], "jobs": os.environ["CARGO_BUILD_JOBS"],
                  "kernel": os.environ["CHOPINFER_KERNEL_BUILD_JOBS"], "empty": not any(Path(sys.argv[4]).iterdir())}))
'''
            child = self.run_lanes(program, base)
            output, _ = child.communicate(timeout=30)
            self.assertEqual(child.returncode, 0, output)
            rows = [json.loads(line) for line in output.splitlines() if line.startswith('{')]
            self.assertEqual(sorted(row['lane'] for row in rows), ['mimo26:c', 'qwen38:a,qwen38:b'])
            self.assertEqual(len({row['target'] for row in rows}), 2)
            self.assertTrue(all(row['jobs'] == row['kernel'] == '5' and row['empty'] for row in rows))
            self.assertEqual([path.name for path in scratch.iterdir()], ['lane.py'])
            self.assert_git_calls(base)

    def test_a_failed_lane_fails_the_recipe(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'scratch').mkdir()
            program = 'import sys\nsys.exit(3 if "mimo26" in sys.argv[2] else 0)\n'
            child = self.run_lanes(program, base)
            output, _ = child.communicate(timeout=30)
            self.assertEqual(child.returncode, 1, output)
            self.assertIn('mimo26/c (failed)', output)
            self.assert_git_calls(base)

    def test_sigterm_stops_every_lane_and_removes_their_targets(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            scratch = base / 'scratch'
            scratch.mkdir()
            program = '''import os, signal, sys, time
from pathlib import Path
# Publish the pid whole: the test reads each *.pid as soon as it appears.
pid = Path(os.environ["TMPDIR"], sys.argv[2].replace(":", "_") + ".pid")
Path(str(pid) + ".tmp").write_text(str(os.getpid()))
os.replace(str(pid) + ".tmp", pid)
time.sleep(60)
'''
            child = self.run_lanes(program, base)
            try:
                deadline = time.monotonic() + 10
                while len(list(scratch.glob('*.pid'))) < 2 and time.monotonic() < deadline:
                    time.sleep(0.02)
                pids = [int(path.read_text()) for path in scratch.glob('*.pid')]
                self.assertEqual(len(pids), 2)
                child.send_signal(signal.SIGTERM)
                output, _ = child.communicate(timeout=30)
                self.assertEqual(child.returncode, 143, output)
                for pid in pids:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
                self.assertEqual(sorted(path.suffix for path in scratch.iterdir()), ['.pid', '.pid', '.py'])
                self.assert_git_calls(base)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.communicate()


class ExecutedTests(unittest.TestCase):
    def test_counts_reject_empty_failed_and_skipped_suites(self):
        assert_tests_ran("test result: ok. 10 passed; 0 failed; 0 ignored;", 1)
        for output in ("", "test result: ok. 0 passed; 0 failed; 0 ignored;",
                       "test result: ok. 1 passed; 1 failed; 0 ignored;",
                       "test result: ok. 1 passed; 0 failed; 1 ignored;"):
            with self.subTest(output=output), self.assertRaises(AssertionError):
                assert_tests_ran(output, 1)

    def test_manifest_checks_the_root_and_each_topologys_formats(self):
        expected = ["mimo.ga-paged-bf16.v1", "mimo.ga-paged-fp8.v2.fixture"]
        formats = [{"layout": {"id": layout}} for layout in expected]
        manifest = {"kv_formats": formats, "topology_resources": [
            {"topology": "fixture-tp4", "kv_formats": formats}]}
        with contextlib.redirect_stdout(io.StringIO()):
            assert_mimo_layouts(manifest, expected)
        manifest["topology_resources"][0]["kv_formats"] = formats[:1]
        with self.assertRaises(AssertionError):
            assert_mimo_layouts(manifest, expected)



class DataDiffTests(unittest.TestCase):
    def test_lock_growth_preserves_existing_record_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            tree = Path(temp)
            def git(*args):
                subprocess.run(["git", "-C", str(tree), *args], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            git("init")
            lock = tree / "models/fixture/bundle/kernels.lock"
            lock.parent.mkdir(parents=True)
            before = json.dumps([{"name": "old", "digest": "unchanged"}], indent=2)
            lock.write_text(before)
            git("add", "models/fixture/bundle/kernels.lock")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                "commit", "-m", "Create fixture")
            added = before[:-2] + ',\n  {"name": "new"}\n]'
            lock.write_text(added)
            with contextlib.redirect_stdout(io.StringIO()):
                assert_data_diff(tree, "fixture", before)
            lock.write_text(added.replace("unchanged", "changed"))
            with self.assertRaisesRegex(AssertionError, "pre-existing"):
                assert_data_diff(tree, "fixture", before)
            lock.write_text(added)
            (tree / "build.rs").write_text("// forbidden\n")
            with self.assertRaises(AssertionError):
                assert_data_diff(tree, "fixture", before)
            git("add", "build.rs")
            with self.assertRaises(AssertionError):
                assert_data_diff(tree, "fixture", before)
            git("reset", "--", "build.rs")
            (tree / "build.rs").unlink()
            lock.write_text(json.dumps([{"name": "new"}]))
            with self.assertRaisesRegex(AssertionError, "pre-existing"):
                assert_data_diff(tree, "fixture", before)

    def test_lock_directory_families_grow_without_changing_records(self):
        with tempfile.TemporaryDirectory() as temp:
            tree = Path(temp)
            def git(*args):
                subprocess.run(["git", "-C", str(tree), *args], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            git("init")
            directory = tree / "models/fixture/bundle/kernels.lock.d"
            directory.mkdir(parents=True)
            rows = json.dumps([{"name": "rows_old", "digest": "unchanged"}], indent=2)
            own = json.dumps([{"name": "own_old"}], indent=2)
            (directory / "rows.lock").write_text(rows)
            (directory / "rows.provenance.json").write_text(rows)
            (directory / "own.lock").write_text(own)
            git("add", "models")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                "commit", "-m", "Create fixture")
            before = model_geometry_recipe.lock_state(tree, "fixture")
            self.assertEqual(set(before), {"rows.lock", "rows.provenance.json", "own.lock"})
            grown = rows[:-2] + ',\n  {"name": "recipe_1_rows"}\n]'
            def check():
                with contextlib.redirect_stdout(io.StringIO()):
                    assert_data_diff(tree, "fixture", before)
            # Untouched families stay untouched; changed ones only gain records.
            (directory / "rows.lock").write_text(grown)
            (directory / "rows.provenance.json").write_text(grown)
            check()
            (directory / "rows.lock").write_text(grown.replace("unchanged", "changed"))
            with self.assertRaisesRegex(AssertionError, "pre-existing"):
                check()
            (directory / "rows.lock").write_text(grown)
            (directory / "added.lock").write_text('[{"name": "recipe_1_added"}]')
            with self.assertRaisesRegex(AssertionError, "lock family file"):
                check()
            (directory / "added.lock").unlink()
            (directory / "rows.lock").write_text(rows)
            (directory / "rows.provenance.json").write_text(rows)
            with self.assertRaisesRegex(AssertionError, "did not add lock records"):
                check()

    def test_supported_changes_and_new_lock_tags_are_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            tree = Path(temp)
            def git(*args):
                subprocess.run(["git", "-C", str(tree), *args], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            git("init")
            lock = tree / "models/fixture/bundle/kernels.lock"
            lock.parent.mkdir(parents=True)
            before = '[{"name":"old"}]'
            lock.write_text(before)
            supported = tree / "models/fixture/geometries/supported.json"
            supported.parent.mkdir(parents=True)
            previous = {"geometries": [{"id": "old", "pinned_names": True}],
                        "aliases": {"architecture": ["Old"]},
                        "semantic_rules": {"activation": ["silu"]}}
            supported.write_text(json.dumps(previous))
            git("add", "models/fixture/bundle/kernels.lock",
                "models/fixture/geometries/supported.json")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                "commit", "-m", "Create fixture")
            entry = {"id": "recipe123", "kernel_tag": "recipe_123", "file": "new.json"}
            (supported.parent / "new.json").write_text("{}")
            actual = json.loads(json.dumps(previous))
            actual["geometries"].append(entry)
            actual["aliases"]["architecture"].append("Recipe")
            supported.write_text(json.dumps(actual))
            lock.write_text('[{"name":"old"},{"name":"recipe_123_rows"}]')
            def check():
                with contextlib.redirect_stdout(io.StringIO()):
                    assert_data_diff(tree, "fixture", before, supported_before=previous,
                                     entry=entry, alias="Recipe")
            check()
            for case in ("tag", "semantics", "pinned_names"):
                with self.subTest(case=case):
                    changed = json.loads(json.dumps(actual))
                    lock.write_text('[{"name":"old"},{"name":"recipe_123_rows"}]')
                    if case == "tag":
                        lock.write_text('[{"name":"old"},{"name":"unrelated_rows"}]')
                    elif case == "semantics":
                        changed["semantic_rules"]["activation"].append("relu")
                    else:
                        changed["geometries"][0]["pinned_names"] = False
                    supported.write_text(json.dumps(changed))
                    with self.assertRaisesRegex(AssertionError, "kernel tag" if case == "tag" else "supported.json"):
                        check()

    def test_existing_geometry_edits_are_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            tree = Path(temp)
            def git(*args):
                subprocess.run(["git", "-C", str(tree), *args], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            git("init")
            lock = tree / "models/fixture/bundle/kernels.lock"
            lock.parent.mkdir(parents=True)
            before = '[{"name":"old"}]'
            lock.write_text(before)
            geometries = tree / "models/fixture/geometries"
            geometries.mkdir(parents=True)
            previous = {"geometries": [{"id": "flash", "file": "flash.json"}],
                        "aliases": {"architecture": ["Old"]}}
            (geometries / "supported.json").write_text(json.dumps(previous))
            (geometries / "flash.json").write_text('{"max_position_embeddings":262144}')
            git("add", "models/fixture/bundle/kernels.lock",
                "models/fixture/geometries/supported.json", "models/fixture/geometries/flash.json")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                "commit", "-m", "Create fixture")
            entry = {"id": "recipe1", "kernel_tag": "recipe_1", "file": "new.json"}
            actual = json.loads(json.dumps(previous))
            actual["geometries"].append(entry)
            actual["aliases"]["architecture"].append("New")
            (geometries / "supported.json").write_text(json.dumps(actual))
            (geometries / "new.json").write_text("{}")
            lock.write_text('[{"name":"old"},{"name":"recipe_1_rows"}]')
            def check():
                with contextlib.redirect_stdout(io.StringIO()):
                    assert_data_diff(tree, "fixture", before, supported_before=previous,
                                     entry=entry, alias="New")
            check()
            (geometries / "flash.json").write_text('{"max_position_embeddings":1048576}')
            with self.assertRaisesRegex(AssertionError, "unexpected data-only paths"):
                check()
            (geometries / "flash.json").write_text('{"max_position_embeddings":262144}')
            (geometries / "extra.json").write_text("{}")
            with self.assertRaisesRegex(AssertionError, "unexpected data-only paths"):
                check()
            (geometries / "extra.json").unlink()
            entry["file"] = "flash.json"
            actual["geometries"][-1] = entry
            (geometries / "supported.json").write_text(json.dumps(actual))
            with self.assertRaisesRegex(AssertionError, "already exist"):
                check()

    def test_contract_stage_adds_only_the_geometry_and_its_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            tree = Path(temp)
            def git(*args):
                subprocess.run(["git", "-C", str(tree), *args], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            git("init")
            geometries = tree / "models/fixture/geometries"
            geometries.mkdir(parents=True)
            previous = {"geometries": [{"id": "flash", "file": "flash.json"}],
                        "aliases": {"architecture": ["Old"]}}
            (geometries / "supported.json").write_text(json.dumps(previous))
            (geometries / "flash.json").write_text('{"hidden_size":4096}')
            git("add", "models/fixture/geometries/supported.json", "models/fixture/geometries/flash.json")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                "commit", "-m", "Create fixture")
            entry = {"id": "recipe1", "kernel_tag": "recipe_1", "file": "new.json"}
            actual = json.loads(json.dumps(previous))
            actual["geometries"].append(entry)
            actual["aliases"]["architecture"].append("New")
            (geometries / "supported.json").write_text(json.dumps(actual))
            (geometries / "new.json").write_text("{}")
            def check():
                with contextlib.redirect_stdout(io.StringIO()):
                    assert_contract_data_diff(tree, "fixture", previous, entry, "New")
            check()
            (tree / "models/fixture/build.rs").write_text("// forbidden\n")
            with self.assertRaisesRegex(AssertionError, "unexpected data-only paths"):
                check()
            (tree / "models/fixture/build.rs").unlink()
            (geometries / "flash.json").write_text('{"hidden_size":2048}')
            with self.assertRaisesRegex(AssertionError, "unexpected data-only paths"):
                check()
            (geometries / "flash.json").write_text('{"hidden_size":4096}')
            changed = json.loads(json.dumps(actual))
            changed["aliases"]["architecture"].append("Extra")
            (geometries / "supported.json").write_text(json.dumps(changed))
            with self.assertRaisesRegex(AssertionError, "supported.json"):
                check()
            entry["file"] = "flash.json"
            actual["geometries"][-1] = entry
            (geometries / "supported.json").write_text(json.dumps(actual))
            with self.assertRaisesRegex(AssertionError, "already exists"):
                check()

    def test_checkpoint_registration_keeps_existing_entries(self):
        with tempfile.TemporaryDirectory() as temp:
            tree = Path(temp)
            def git(*args):
                subprocess.run(["git", "-C", str(tree), *args], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            git("init")
            bundle = tree / "models/fixture/bundle"
            bundle.mkdir(parents=True)
            before = '[{"name":"old"}]'
            (bundle / 'kernels.lock').write_text(before)
            registry = tree / 'models/fixture/checkpoints.json'
            previous = {"default": "old", "checkpoints": {"old": {"revision": "original"}}}
            registry.write_text(json.dumps(previous))
            git('add', 'models/fixture/bundle/kernels.lock', 'models/fixture/checkpoints.json')
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                'commit', '-m', 'Create fixture')
            (bundle / 'kernels.lock').write_text('[{"name":"old"},{"name":"new"}]')
            actual = json.loads(json.dumps(previous))
            actual['checkpoints']['new'] = {"revision": "new"}
            registry.write_text(json.dumps(actual))
            checkpoint_data = {'checkpoints.json': previous}
            with contextlib.redirect_stdout(io.StringIO()):
                assert_data_diff(tree, 'fixture', before, checkpoint_data)
            actual['checkpoints']['old']['revision'] = 'changed'
            registry.write_text(json.dumps(actual))
            with self.assertRaises(AssertionError):
                assert_data_diff(tree, 'fixture', before, checkpoint_data)
            actual['checkpoints']['old']['revision'] = 'original'
            actual['default'] = 'new'
            registry.write_text(json.dumps(actual))
            with self.assertRaises(AssertionError):
                assert_data_diff(tree, 'fixture', before, checkpoint_data)



class InheritedRepositoryTests(unittest.TestCase):
    def test_fixtures_leave_an_inherited_repository_unchanged(self):
        """Run the tests that create repositories, here and in the export
        recipe's tests, with GIT_DIR naming another repository, as `git rebase
        -x` does in a linked worktree: that repository keeps its commits,
        index, configuration and worktrees."""
        with tempfile.TemporaryDirectory() as temp:
            sentinel = Path(temp) / 'sentinel'
            sentinel.mkdir()
            def git(*args):
                return subprocess.check_output(['git', '-C', str(sentinel), *args], text=True,
                                               stderr=subprocess.DEVNULL)
            git('init', '-q')
            (sentinel / 'kept').write_text('kept\n')
            git('add', 'kept')
            git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'Sentinel')
            def state():
                return (git('log', '--all', '--format=%H %s'), git('status', '--porcelain', '--ignored'),
                        (sentinel / '.git/config').read_bytes(), git('worktree', 'list', '--porcelain'))
            before = state()
            scratch = Path(temp) / 'scratch'
            scratch.mkdir()
            env = dict(os.environ, GIT_DIR=str(sentinel / '.git'), PYTHONPATH=str(ROOT / 'tests'),
                       TMPDIR=str(scratch))
            selected = [f'test_model_geometry_recipe.{name}' for name in
                        ('SignalCleanupTests', 'ContractStageRootTests', 'DataDiffTests')]
            result = subprocess.run([sys.executable, '-B', '-m', 'unittest', *selected, 'test_model_export_recipe'],
                                    cwd=ROOT / 'tests', env=env, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            self.assertEqual(state(), before)
            self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()

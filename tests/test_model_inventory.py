#!/usr/bin/env python3
"""Bounded inventory admission, file checks and current consumer parity."""
import dataclasses
import importlib.util
import json
import pathlib
import os
import re
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model_inventory as inventory
import model_requirements

VALID = '''schema = "chopinfer-model-inventory-v1"
checkpoint_env = "M3"
[bundle]
package = "bundle-m3"
cdylib = ["bundle_m3"]
[compiler]
package = "model-m3"
bin = ["m3-package"]
'''
RECIPE = '''[contributor]
roots = ["model-m3"]
[[contributor.tests]]
name = "bounded test"
phase = "config"
package = "model-m3"
lib = true
'''
LABEL = "models/m3/model.toml"

# A bundle's conformance suite (bundle-runtime `conformance::suite!`)
# holds its serving startup tests; its scenario runner is only for the
# macro, so a bundle cannot run scenarios without startup.
SUITE = re.compile(r'\b(?:conformance::suite|conformance_suite)!')
DIRECT = re.compile(r'\bconformance::run\b|\bconformance::\{[^}]*\brun\b')
EXPORTS_API = re.compile(r'\bexport_bundle_api_v1!')


def test_module(source, path):
    """Whether `path` under the crate source `source` is a module file whose
    declaration, or an ancestor's, is `#[cfg(test)]`: `mod name;` right
    after the attribute in its parent file (`lib.rs`, `<dir>.rs` or
    `<dir>/mod.rs`). A file reached otherwise, as by `#[path]`, is not."""
    relative = path.relative_to(source)
    if relative.parts in (('lib.rs',), ('main.rs',)):
        return False
    name = relative.parent.name if relative.name == 'mod.rs' else relative.stem
    directory = path.parent.parent if relative.name == 'mod.rs' else path.parent
    if directory == source:
        parents = [source / 'lib.rs', source / 'main.rs']
    else:
        parents = [directory.with_suffix('.rs'), directory / 'mod.rs']
    declared = re.compile(r'#\[cfg\(test\)\]\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+' + re.escape(name) + r'\s*;')
    for parent in parents:
        if parent.is_file():
            if declared.search(parent.read_text()):
                return True
            return test_module(source, parent)
    return False


def exports_bundle_api(root, model):
    """Whether `model`'s bundle exports the bundle API, so a host can serve
    it. A skeleton bundle with no program yet exports nothing, and an export
    confined to a `#[cfg(test)]` module file (`test_module`) serves only the
    crate's own tests."""
    source = root / pathlib.PurePosixPath(model.directory) / 'bundle/src'
    return any(EXPORTS_API.search(path.read_text()) and not test_module(source, path)
               for path in sorted(source.rglob('*.rs')))


def tests_kernels_on_gpu(model):
    """Whether a released model's kernels have GPU tests: in kernel-lab's
    catalog, or in the model's own lab crate on kernel-harness."""
    return bool(model.kernel_lab.catalog or model.lab is not None)


def conformance_suite(root, model):
    """The one bundle source file of `model` that invokes the conformance
    suite; refuses none, several, or a direct call of its scenario runner."""
    bundle = pathlib.PurePosixPath(model.directory) / 'bundle/src'
    invoked = []
    for path in sorted((root / bundle).rglob('*.rs')):
        text = path.read_text()
        label = path.relative_to(root).as_posix()
        if DIRECT.search(text):
            raise AssertionError(f'{label}: run conformance scenarios through conformance::suite!, not conformance::run')
        invoked += [label] * len(SUITE.findall(text))
    if len(invoked) != 1:
        raise AssertionError(f'{bundle}: invoke conformance::suite! once in the bundle crate, found {invoked}')
    return invoked[0]


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name).resolve()
        for path in ("bundle/Cargo.toml", "compiler/Cargo.toml", "bundle/kernels.lock", "profiles.json", "asset", "reference.py"):
            self.write("models/m3/" + path, "")

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def load(self, text=VALID):
        self.write("models/m3/model.toml", text)
        return inventory.load(self.root)

    def refused(self, text, message, *, files=False, metadata=None):
        with self.assertRaises(inventory.InventoryError) as caught:
            models = self.load(text)
            if files:
                inventory.check_files(models, self.root)
            if metadata is not None:
                inventory.check_metadata(models, metadata, self.root)
        self.assertEqual(str(caught.exception), message)

    def metadata(self, extra_bins=(), members=None):
        def package(name, role, targets):
            return {"id": name, "name": name,
                    "manifest_path": str(self.root / "models/m3" / role / "Cargo.toml"),
                    "targets": [{"name": target, "kind": [kind]} for target, kind in targets]}
        packages = [
            package("bundle-m3", "bundle", [("bundle_m3", "cdylib"), ("bundle_m3", "lib")]),
            package("model-m3", "compiler", [("m3-package", "bin"), ("model_m3", "lib"), ("bounded", "test"),
                                             *((name, "bin") for name in extra_bins)]),
        ]
        return {"packages": packages,
                "workspace_members": members if members is not None else [p["id"] for p in packages]}

    def test_control_characters_in_model_names_are_refused(self):
        for char in ('\t', '\n', '\r', '\v', '\x1b', '\x7f', '\x85', '\u200b', '\u202e'):
            with self.subTest(char=repr(char)):
                name = 'm3' + char
                directory = self.root / 'models' / name
                for role in ('bundle', 'compiler'):
                    self.write(f'models/{name}/{role}/Cargo.toml', '')
                self.write(f'models/{name}/model.toml', VALID)
                with self.assertRaises(inventory.InventoryError) as caught:
                    inventory.load_model(self.root, directory)
                self.assertEqual(str(caught.exception),
                                 f"model.toml: model name {name!r} contains control characters")

    def test_control_characters_in_model_directory_parents_are_refused(self):
        for char in ('\t', '\n', '\x1b', '\x85', '\u202e'):
            with self.subTest(char=repr(char)):
                relative = 'nested' + char + '/m3'
                for role in ('bundle', 'compiler'):
                    self.write(f'{relative}/{role}/Cargo.toml', '')
                self.write(f'{relative}/model.toml', VALID)
                with self.assertRaises(inventory.InventoryError) as caught:
                    inventory.load_model(self.root, self.root / relative)
                self.assertEqual(str(caught.exception),
                                 f"model.toml: model directory {relative!r} contains control characters")

    def test_external_root_model_name_controls_are_refused(self):
        name = 'external\tmodel'
        root = self.root / name
        self.write(f'{name}/model.toml', VALID)
        with self.assertRaises(inventory.InventoryError) as caught:
            inventory.load_model(root, root)
        self.assertEqual(str(caught.exception),
                         f"model.toml: model name {name!r} contains control characters")

    def test_defaults_are_immutable_and_legacy_roles_match(self):
        models = self.load()
        model = models[0]
        self.assertEqual(model.checkpoint_env, "M3")
        self.assertFalse(model.release.enabled)
        self.assertEqual(model.release.bins, ())
        self.assertIsNone(model.contributor)
        self.assertEqual(model.checks.compile_inputs, ())
        self.assertFalse(model.kernel_lab.catalog)
        self.assertIsNone(model.geometry_recipe)
        self.assertEqual(inventory.release_plan(models), ())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            model.release.enabled = True
        inventory.check_metadata(models, self.metadata(), self.root)

    def test_closed_sections_and_rows(self):
        for section in ("release", "contributor", "checks", "kernel_lab", "geometry_recipe"):
            with self.subTest(section=section):
                self.refused(VALID + f"[{section}]\nunknown = true\n", f"{LABEL}.{section}: unknown keys ['unknown']")
        for section in ("release.assets", "contributor.tests", "checks.profile_targets"):
            with self.subTest(section=section):
                self.refused(VALID + f"[[{section}]]\nunknown = true\n", f"{LABEL}.{section}[0]: unknown keys ['unknown']")
        self.refused(VALID + RECIPE + 'cases = [{ name = "a", case = "a", shell = "x" }]\n',
                     f"{LABEL}.contributor.tests[0].cases[0]: unknown keys ['shell']")

    def test_boolean_types(self):
        for section, key in (("release", "enabled"), ("kernel_lab", "catalog")):
            with self.subTest(section=section):
                self.refused(VALID + f'[{section}]\n{key} = "yes"\n', f"{LABEL}.{section}.{key}: must be boolean")
        for key in ("compile", "lib"):
            with self.subTest(key=key):
                text = RECIPE.replace('lib = true\n', '') + f'{key} = "yes"\n'
                self.refused(VALID + text, f"{LABEL}.contributor.tests[0].{key}: must be boolean")

    def test_release_enabled_handling_and_target_selection(self):
        self.refused(VALID + '[release]\nenabled = true\n', f"models/m3/model.toml.release: enabled release has nothing selected")
        for text in ('[release]\n', '[release]\nenabled = false\nbins = ["m3-package"]\n'):
            self.assertEqual(inventory.release_plan(self.load(VALID + text)), ())
        models = self.load(VALID + '[release]\nenabled = true\nbins = ["m3-package"]\ncdylibs = ["bundle_m3"]\n')
        plan = inventory.release_plan(models)
        self.assertEqual([row.name for row in plan], ["m3"])
        self.assertEqual([(row.package, row.name, row.kind, inventory.destination(row)) for row in plan[0].targets],
                         [("model-m3", "m3-package", "bin", "bin/m3-package"),
                          ("bundle-m3", "bundle_m3", "cdylib", "lib/libbundle_m3.so")])
        self.assertEqual([(asset.source, asset.destination) for asset in plan[0].assets],
                         [("models/m3/bundle/kernels.lock", "models/m3/kernels.lock")])
        self.refused(VALID + '[release]\nbins = ["developer"]\n',
                     f"{LABEL}.release.bins[0]: target 'developer' needs exactly one owning role")

    def test_release_ships_both_kernel_lock_forms_and_refuses_other_entries(self):
        enabled = VALID + '[release]\nenabled = true\nbins = ["m3-package"]\n'
        family = self.root / 'models/m3/bundle/kernels.lock.d'
        family.mkdir()
        for name in ('row.provenance.json', 'row.lock', 'comm.lock'):
            self.write('models/m3/bundle/kernels.lock.d/' + name, '[]')
        expected = [('models/m3/bundle/kernels.lock', 'models/m3/kernels.lock'),
                    ('models/m3/bundle/kernels.lock.d/comm.lock', 'models/m3/kernels.lock.d/comm.lock'),
                    ('models/m3/bundle/kernels.lock.d/row.lock', 'models/m3/kernels.lock.d/row.lock'),
                    ('models/m3/bundle/kernels.lock.d/row.provenance.json', 'models/m3/kernels.lock.d/row.provenance.json')]
        models = self.load(enabled)
        self.assertEqual([(a.source, a.destination) for a in inventory.release_plan(models)[0].assets], expected)
        self.assertEqual(inventory.file_inputs(self.root, models[0])[:4], tuple(source for source, _ in expected[:4]))
        # The directory form alone ships without a kernels.lock file.
        (self.root / 'models/m3/bundle/kernels.lock').unlink()
        models = self.load(enabled)
        self.assertEqual([(a.source, a.destination) for a in inventory.release_plan(models)[0].assets], expected[1:])
        inventory.check_files(models, self.root)
        # Neither form: check_files reports the missing kernels.lock.
        for name in ('row.provenance.json', 'row.lock', 'comm.lock'):
            (family / name).unlink()
        family.rmdir()
        self.refused(enabled, f"{LABEL}.release.kernels_lock: missing file 'models/m3/bundle/kernels.lock'", files=True)
        family.mkdir()
        for name, message in (('notes.txt', 'notes.txt'), ('sub.lock', 'sub.lock')):
            with self.subTest(name=name):
                path = family / name
                path.mkdir() if name == 'sub.lock' else path.write_text('')
                self.refused(enabled, f"{LABEL}: bundle/kernels.lock.d/{message} is not a lock family file")
                path.rmdir() if name == 'sub.lock' else path.unlink()
        self.write('models/m3/elsewhere.lock', '[]')
        (family / 'linked.lock').symlink_to(self.root / 'models/m3/elsewhere.lock')
        self.refused(enabled, f"{LABEL}: bundle/kernels.lock.d/linked.lock is not a lock family file")
        (family / 'linked.lock').unlink()
        family.rmdir()
        family.symlink_to(self.root / 'models/m3', target_is_directory=True)
        self.refused(enabled, f"{LABEL}: bundle/kernels.lock.d must not be a symlink")

    def test_asset_paths_are_confined_even_when_traversal_exists(self):
        self.write('outside', 'exists')
        asset = '\n[release]\nassets = [{ source = "models/m3/asset", destination = "models/m3/extra" }]\n'
        for bad in ('models/m3/../../outside', '/outside', 'models/m3/-asset', 'models/m3/a b', 'models/m3/a\nfile', 'models\\m3\\asset'):
            for key in ('source', 'destination'):
                with self.subTest(key=key, path=bad):
                    original = 'models/m3/asset' if key == 'source' else 'models/m3/extra'
                    text = asset.replace(f'{key} = "{original}"', f'{key} = {json.dumps(bad)}')
                    self.refused(VALID + text, f"{LABEL}.release.assets[0].{key}: invalid path {bad!r}")
        self.refused(VALID + asset.replace('models/m3/extra', 'models/m4/extra'),
                     f"{LABEL}.release.assets[0].destination: must be under models/m3/")
        self.refused(VALID + asset.replace('models/m3/asset', 'outside'),
                     f"{LABEL}.release.assets[0].source: must be under models/m3/")

    def test_missing_files_and_symlinks_are_checked_separately(self):
        text = VALID + '[release]\nassets = [{ source = "models/m3/missing", destination = "models/m3/extra" }]\n'
        self.load(text)  # Loading does not read optional assets.
        self.refused(text, f"{LABEL}.release.assets[0].source: missing file 'models/m3/missing'", files=True)
        (self.root / 'models/m3/link').symlink_to(self.root / 'models/m3/asset')
        self.refused(text.replace('models/m3/missing', 'models/m3/link'),
                     f"{LABEL}.release.assets[0].source: symlink path 'models/m3/link'", files=True)
        (self.root / 'models/m3/linked').symlink_to(self.root / 'models/m3', target_is_directory=True)
        self.refused(text.replace('models/m3/missing', 'models/m3/linked/asset'),
                     f"{LABEL}.release.assets[0].source: symlink path 'models/m3/linked/asset'", files=True)

    def test_file_parent_and_duplicate_destination_collisions(self):
        prefix = VALID + '[release]\nenabled = true\nbins = ["m3-package"]\n'
        for dest in ('models/m3/kernels.lock', 'models/m3/kernels.lock/child'):
            with self.subTest(dest=dest):
                self.refused(prefix + f'assets = [{{ source = "models/m3/asset", destination = "{dest}" }}]\n',
                             f"{LABEL}.release.assets[0]: destination {dest!r} collides with {LABEL}.release.kernels_lock ('models/m3/kernels.lock')")
        self.refused(prefix + 'assets = [{ source = "models/m3/asset", destination = "models/m3/a" }, { source = "models/m3/asset", destination = "models/m3/a" }]\n',
                     f"{LABEL}.release.assets[1]: destination 'models/m3/a' collides with {LABEL}.release.assets[0] ('models/m3/a')")

    def test_disabled_assets_still_refuse_destination_collisions(self):
        self.refused(VALID + '[release]\nassets = [{ source = "models/m3/asset", destination = "models/m3/a" }, { source = "models/m3/asset", destination = "models/m3/a/child" }]\n',
                     f"{LABEL}.release.assets[1]: destination 'models/m3/a/child' collides with {LABEL}.release.assets[0] ('models/m3/a')")

    def test_extra_model_selection_and_cross_model_collisions(self):
        first = self.load(VALID + '[release]\nenabled = true\nbins = ["m3-package"]\n')
        for role in ('bundle', 'compiler'):
            self.write(f'models/m4/{role}/Cargo.toml', '')
        self.write('models/m4/model.toml', VALID.replace('m3', 'm4'))
        self.assertEqual([model.name for model in inventory.release_plan(inventory.load(self.root))], ['m3'])
        # Cargo permits identical binary names in distinct packages. The release cannot.
        self.write('models/m4/model.toml', VALID.replace('m3', 'm4').replace('bin = ["m4-package"]', 'bin = ["m3-package"]') + '[release]\nenabled = true\nbins = ["m3-package"]\n')
        with self.assertRaises(inventory.InventoryError) as caught:
            inventory.load(self.root)
        self.assertEqual(str(caught.exception), f"models/m4/model.toml.release.bins[0]: destination 'bin/m3-package' collides with {LABEL}.release.bins[0] ('bin/m3-package')")
        self.assertEqual(len(first), 1)

    def test_contributor_packages_and_extra_environment_ownership(self):
        self.refused(VALID + '[contributor]\n', f"{LABEL}.contributor: needs nonempty roots and tests")
        self.refused(VALID + RECIPE.replace('roots = ["model-m3"]', 'roots = ["model-compiler-sdk"]'),
                     f"{LABEL}.contributor.roots[0]: package 'model-compiler-sdk' is not a model role")
        self.refused(VALID + RECIPE.replace('package = "model-m3"\nlib', 'package = "model-compiler-sdk"\nlib'),
                     f"{LABEL}.contributor.tests[0].package: 'model-compiler-sdk' is not a model role")
        for key in ('CHOPINFER_PACKAGE_NATIVE_ROOT', 'CHOPINFER_M4_EXTRA', 'CHOPINFER_M3_CHECKPOINT', 'CHOPINFER_M3_TOKENIZER_DIR'):
            with self.subTest(key=key):
                text = RECIPE.replace('roots = ["model-m3"]', f'roots = ["model-m3"]\nclear_env = ["{key}"]')
                self.refused(VALID + text, f"{LABEL}.contributor.clear_env[0]: must be a model-owned extra with prefix 'CHOPINFER_M3_'")
        text = RECIPE.replace('roots = ["model-m3"]', 'roots = ["model-m3"]\nclear_env = ["CHOPINFER_M3_EXTRA"]')
        self.assertEqual(self.load(VALID + text)[0].contributor.clear_env, ('CHOPINFER_M3_EXTRA',))

    def test_contributor_exact_case_validation_and_duplicate_names(self):
        for case in ('a:b', 'a::', 'a b', 'a;exit', '1a'):
            with self.subTest(case=case):
                self.refused(VALID + RECIPE + f'case = "{case}"\n', f"{LABEL}.contributor.tests[0]: invalid exact test case {case!r}")
        self.refused(VALID + RECIPE + 'cases = ["a"]\ncompile = true\n', f"{LABEL}.contributor.tests[0]: compile cannot select exact cases")
        self.refused(VALID + RECIPE + 'cases = ["a"]\ncase = "a"\n', f"{LABEL}.contributor.tests[0]: case and cases are mutually exclusive")
        self.refused(VALID + RECIPE + 'cases = ["a", {name = "a", case = "b"}]\n', f"{LABEL}.contributor.tests[0]: duplicate test name 'a'")
        self.refused(VALID + RECIPE + RECIPE[RECIPE.index('[[contributor.tests]]'):], f"{LABEL}.contributor.tests[1]: duplicate test name 'bounded test'")

    def test_compile_cannot_have_exact_case(self):
        self.refused(VALID + RECIPE + 'case = "a::b"\ncompile = true\n',
                     f"{LABEL}.contributor.tests[0]: compile cannot select exact cases")

    def test_contributor_target_metadata_and_membership(self):
        for selection, expected in (('suites = ["missing"]', "suites[0]: package 'model-m3' has no test target 'missing'"),
                                    ('bins = ["missing"]', "bins[0]: package 'model-m3' has no bin target 'missing'")):
            with self.subTest(selection=selection):
                self.refused(VALID + RECIPE + selection + '\n', f"{LABEL}.contributor.tests[0].{expected}", metadata=self.metadata())
        data = self.metadata()
        data['packages'][1]['targets'] = [t for t in data['packages'][1]['targets'] if 'lib' not in t['kind']]
        self.refused(VALID + RECIPE, f"{LABEL}.contributor.tests[0].lib: package 'model-m3' has no library target", metadata=data)
        self.refused(VALID + RECIPE, f"{LABEL}.contributor.roots[0]: package 'model-m3' is not a workspace member",
                     metadata=self.metadata(members=['bundle-m3']))
        inventory.check_metadata(self.load(VALID + RECIPE + 'suites = ["bounded"]\nbins = ["m3-package"]\n'), self.metadata(), self.root)

    def test_a_kernel_test_crate_has_no_role_no_artifact_and_no_default_membership(self):
        def with_lab(targets=(("sm120", "test"),), default=False):
            data = self.metadata()
            data["packages"].append({"id": "m3-lab", "name": "m3-lab",
                                     "manifest_path": str(self.root / "models/m3/lab/Cargo.toml"),
                                     "targets": [{"name": name, "kind": [kind]} for name, kind in targets]})
            data["workspace_members"].append("m3-lab")
            data["workspace_default_members"] = [p["id"] for p in data["packages"] if default or p["id"] != "m3-lab"]
            return data
        # Without lab/, the member has no role.
        self.refused(VALID, "models/m3/lab/Cargo.toml: package m3-lab has no role in its model.toml", metadata=with_lab())
        self.write("models/m3/lab/Cargo.toml", "")
        self.assertEqual(self.load()[0].lab, "models/m3/lab/Cargo.toml")
        inventory.check_metadata(self.load(), with_lab(), self.root)
        self.refused(VALID, "models/m3/lab/Cargo.toml: a kernel test crate is not a default workspace member",
                     metadata=with_lab(default=True))
        self.refused(VALID, "models/m3/lab/Cargo.toml: a kernel test crate has no bin or cdylib target",
                     metadata=with_lab(targets=(("lab", "bin"),)))
        data = with_lab()
        data["workspace_default_members"].remove("model-m3")
        self.refused(VALID, "models/m3/compiler/Cargo.toml: package model-m3 is missing from the default workspace members",
                     metadata=data)
        # An external workspace may keep other test crates out of its defaults.
        inventory.check_metadata(self.load(), data, self.root, external=True)

    def test_compile_patterns_are_bounded_and_admit_new_files(self):
        self.write('models/outside.py', 'exists')
        for path in ('../outside.py', 'reference\\*.py', '/reference.py'):
            with self.subTest(path=path):
                self.refused(VALID + f'[checks]\ncompile_inputs = [{json.dumps(path)}]\n', f"{LABEL}.checks.compile_inputs[0]: invalid path {path!r}")
        self.refused(VALID + '[checks]\ncompile_inputs = ["*/reference.py"]\n', f"{LABEL}.checks.compile_inputs[0]: wildcard must be in the filename")
        self.refused(VALID + '[checks]\ncompile_inputs = ["missing*.py"]\n', f"{LABEL}.checks.compile_inputs[0]: pattern 'missing*.py' matches no files", files=True)
        model = self.load(VALID + '[checks]\ncompile_inputs = ["reference*.py"]\n')[0]
        self.assertEqual(inventory.compile_files(self.root, model), ('models/m3/reference.py',))
        self.write('models/m3/reference_x.py', '')
        self.assertEqual(inventory.compile_files(self.root, model), ('models/m3/reference.py', 'models/m3/reference_x.py'))
        (self.root / 'models/m3/reference_link.py').symlink_to(self.root / 'models/m3/reference.py')
        with self.assertRaises(inventory.InventoryError) as caught:
            inventory.compile_files(self.root, model)
        self.assertEqual(str(caught.exception), f"{LABEL}.checks.compile_inputs[0]: symlink path 'models/m3/reference_link.py'")

    def test_recursive_compile_globs_are_refused_during_load(self):
        for pattern in ('reference**.py', '**'):
            with self.subTest(pattern=pattern):
                self.refused(VALID + f'[checks]\ncompile_inputs = ["{pattern}"]\n',
                             f"{LABEL}.checks.compile_inputs[0]: recursive glob '**' is forbidden")

    def test_profile_duplicates_and_catalog_requirements(self):
        row = '[[checks.profile_targets]]\npackage = "model-m3"\nkind = "bin"\ntarget = "m3-package"\n'
        self.refused(VALID + row + row, f"{LABEL}.checks.profile_targets[1]: duplicate target 'm3-package'")
        self.refused(VALID + row.replace('m3-package', 'developer'), f"{LABEL}.checks.profile_targets[0]: undeclared target 'developer'")
        self.refused(VALID + '[kernel_lab]\ncatalog = true\n', f"{LABEL}.kernel_lab.catalog: needs a contract role")

    def test_geometry_recipe_legs_and_fixture_files(self):
        at = f"{LABEL}.geometry_recipe"
        recipe = '[geometry_recipe]\nstage = "contract"\nfixtures = ["small"]\narchitecture = "M3RecipeForCausalLM"\n'
        self.refused(VALID + recipe, f"{at}: needs a contract role")
        self.write("models/m3/contract/Cargo.toml", "")
        text = VALID + '[contract]\npackage = "model-m3-contract"\n' + recipe
        [model] = self.load(text)
        self.assertEqual(model.geometry_recipe, inventory.GeometryRecipe("contract", ("small",), "M3RecipeForCausalLM"))
        self.refused(text, f"{at}.fixtures[0]: missing file 'fixtures/m3/geometries/small/config.json'", files=True)
        for name in ("config.json", "geometry.json", "supported.json"):
            self.write("fixtures/m3/geometries/small/" + name, "{}")
        inventory.check_files(self.load(text), self.root)
        bundle = text.replace('"contract"', '"bundle"')
        self.refused(bundle, f"{at}: a bundle-stage recipe needs contract_tests and checkpoint_env")
        [model] = self.load(bundle + 'contract_tests = ["kernel_rank_catalog"]\n')
        self.assertEqual(model.geometry_recipe.contract_tests, ("kernel_rank_catalog",))
        self.refused(bundle.replace('checkpoint_env = "M3"\n', '') + 'contract_tests = ["kernel_rank_catalog"]\n',
                     f"{at}: a bundle-stage recipe needs contract_tests and checkpoint_env")
        for old, new, message in (
                ('stage = "contract"', 'stage = "gpu"', f"{at}.stage: must be one of ['bundle', 'contract']"),
                ('fixtures = ["small"]', 'fixtures = []', f"{at}.fixtures: needs at least one fixture"),
                ('fixtures = ["small"]', 'fixtures = ["../small"]', f"{at}.fixtures: must be a list of valid strings"),
                ('architecture = "M3RecipeForCausalLM"', 'architecture = "m3"',
                 f"{at}.architecture: must be an architecture class name"),
                ('architecture = "M3RecipeForCausalLM"\n', '', f"{at}.architecture: must be a nonempty printable string"),
                ('architecture = "M3RecipeForCausalLM"', 'architecture = "M3"\ncontract_tests = ["suite"]',
                 f"{at}.contract_tests: a contract-stage recipe runs its geometry_recipe test")):
            with self.subTest(new):
                self.refused(text.replace(old, new), message)

    def test_an_external_model_root_keeps_its_recipe_fixtures_model_relative(self):
        root = self.root / "external"
        recipe = '[geometry_recipe]\nstage = "contract"\nfixtures = ["small"]\narchitecture = "M3RecipeForCausalLM"\n'
        for path in ("bundle/Cargo.toml", "compiler/Cargo.toml", "contract/Cargo.toml", "bundle/kernels.lock",
                     "fixtures/m3/geometries/small/config.json"):
            self.write("external/" + path, "{}")
        self.write("external/model.toml", VALID + '[contract]\npackage = "model-m3-contract"\n' + recipe)
        model = inventory.load_model(root, root)
        self.assertEqual(inventory.fixture_root(model), "fixtures")
        with self.assertRaises(inventory.InventoryError) as caught:
            inventory.check_files([model], root)
        self.assertEqual(str(caught.exception), "./model.toml.geometry_recipe.fixtures[0]: "
                         "missing file 'fixtures/geometries/small/config.json'")
        for name in ("config.json", "geometry.json", "supported.json"):
            self.write("external/fixtures/geometries/small/" + name, "{}")
        inventory.check_files([model], root)
        [model] = self.load(VALID)
        self.assertEqual(inventory.fixture_root(model), "fixtures/m3")

    def test_role_schema_and_metadata_refusals(self):
        self.refused(VALID.replace('-v1', '-v2'), f"{LABEL}: schema must be 'chopinfer-model-inventory-v1'")
        self.refused(VALID.replace('"M3"', '"m3"'), f"{LABEL}: checkpoint_env must be an upper-case variable stem")
        self.refused(VALID.replace('["m3-package"]', '"m3-package"'), f"{LABEL}: compiler.bin: must be a list of valid strings")
        self.refused(VALID.replace('["m3-package"]', '["m3-package", "m3-package"]'), f"{LABEL}: compiler.bin: repeats a value")
        self.refused(VALID.split('[compiler]')[0], f"{LABEL}: [compiler] is required")
        self.refused(VALID + '[contract]\npackage = "contract-m3"\n', f"{LABEL}: [contract] declared but contract/Cargo.toml is missing")
        self.refused(VALID, "model-m3 target inventory differs: declared [('m3-package', 'bin')], observed [('extra', 'bin'), ('m3-package', 'bin')]", metadata=self.metadata(extra_bins=['extra']))

    def test_missing_inventory_and_undeclared_role(self):
        with self.assertRaises(inventory.InventoryError) as caught:
            inventory.load(self.root)
        self.assertEqual(str(caught.exception), f"{LABEL}: is missing")
        self.write('models/m3/contract/Cargo.toml', '')
        self.refused(VALID, f"{LABEL}: contract/Cargo.toml has no [contract] entry")

    def test_one_package_owner_and_exact_manifest_path(self):
        self.load()
        for role in ('bundle', 'compiler'):
            self.write(f'models/m4/{role}/Cargo.toml', '')
        self.write('models/m4/model.toml', VALID.replace('m3', 'm4').replace('package = "model-m4"', 'package = "model-m3"'))
        with self.assertRaises(inventory.InventoryError) as caught:
            inventory.load(self.root)
        self.assertEqual(str(caught.exception), 'package model-m3 is declared by models/m3/compiler/Cargo.toml and models/m4/compiler/Cargo.toml')

    def test_fixture_and_reference_paths_are_checked(self):
        for section, key, prefix in (('contributor', 'fixtures', ''), ('checks', 'reference_tests', 'models/m3/')):
            with self.subTest(section=section):
                text = RECIPE.replace('roots = ["model-m3"]', 'roots = ["model-m3"]\nfixtures = ["missing"]') if section == 'contributor' else '[checks]\nreference_tests = ["missing"]\n'
                self.refused(VALID + text, f"{LABEL}.{section}.{key}[0]: missing file {prefix + 'missing'!r}", files=True)
                text = text.replace('"missing"', '"../outside"')
                self.write('outside', 'exists')
                self.write('models/outside', 'exists')
                self.refused(VALID + text, f"{LABEL}.{section}.{key}[0]: invalid path '../outside'")

    def test_the_publisher_owns_every_repository_the_model_names(self):
        text = VALID.replace('checkpoint_env = "M3"\n', 'checkpoint_env = "M3"\npublisher = "m3-org"\n')
        self.write("models/m3/frontend.json", json.dumps({"tokenizer": {"repository": "m3-org/M3"}}))
        self.write("fixtures/m3/source.json", json.dumps([{"repository": "https://github.com/m3-org/code"}]))
        self.write("models/m3/notes.json", "not JSON")
        models = self.load(text)
        self.assertEqual(models[0].publisher, "m3-org")
        inventory.check_files(models, self.root)
        self.write("fixtures/m3/export.json", json.dumps({"pin": {"repository": "someone/M3-FP4"}}))
        self.refused(text, "fixtures/m3/export.json: repository 'someone/M3-FP4' is not m3-org's, the "
                           "publisher models/m3/model.toml declares", files=True)
        self.write("fixtures/m3/export.json", json.dumps({"repository": "m3-org-mirror/M3"}))
        self.refused(text, "fixtures/m3/export.json: repository 'm3-org-mirror/M3' is not m3-org's, the "
                           "publisher models/m3/model.toml declares", files=True)
        self.refused(VALID.replace('checkpoint_env = "M3"\n', 'checkpoint_env = "M3"\npublisher = "m3 org"\n'),
                     f"{LABEL}: publisher must be a Hugging Face organization name")

    def test_unknown_top_and_role_keys_are_closed(self):
        self.refused('extra = 1\n' + VALID, f"{LABEL}: unknown keys ['extra']")
        self.refused(VALID + 'unknown = true\n', f"{LABEL}: unknown keys in [compiler] ['unknown']")

    def released_facts(self, root, contributor, models):
        """Properties every released model's declarations keep. A model that
        declares `[release] enabled = false` until it qualifies is exempt,
        except that kernel-lab registers every model that opts into its
        catalog (`ci/kernel_lab_inventory.py`)."""
        declared = models
        models = [model for model in models if model.release.enabled]
        self.assertTrue(models)
        for model, release in zip(models, inventory.release_plan(models)):
            self.assertEqual(release.name, model.name)
            actual = []
            for selection in model.contributor.tests:
                for row in selection.expanded():
                    argv = ['cargo', 'test', '-p', row.package]
                    if row.lib:
                        argv += ['--lib']
                    for values, flag in ((row.bins, '--bin'), (row.suites, '--test')):
                        for target in values:
                            argv += [flag, target]
                    argv += ['--locked', '--offline']
                    if row.compile:
                        argv += ['--no-run']
                    elif row.case:
                        argv += [row.case, '--', '--exact']
                    actual.append(dict(name=row.name, phase=row.phase, argv=argv, tests=not row.compile))
            original = [step for step in contributor.steps(model.name) if step['argv'][:2] == ['cargo', 'test']]
            self.assertEqual([original[0], *actual, original[-1]], original)
            self.assertEqual(model.contributor.roots, tuple(name for name in contributor.roots(model.name) if name != 'model-compiler-sdk'))
            # Contributor fixtures are the model's own.
            for path in model.contributor.fixtures:
                self.assertTrue(path.startswith((model.directory + '/', f'fixtures/{model.name}/')), path)
            # Every kernel lock form the bundle keeps; the serving profiles
            # ship inside each package.
            locks = inventory.kernel_locks(model)
            self.assertTrue(locks)
            self.assertEqual([(asset.source, asset.destination) for asset in release.assets[:len(locks)]],
                             [(source, f'models/{model.name}/' + source.removeprefix(f'{model.directory}/bundle/'))
                              for source in locks])
            # The release ships every bundle library and the declared release
            # binaries. Each model's own test_inventory.py names them.
            bundle = {name for name, kind in model.packages['bundle'].targets if kind == 'cdylib'}
            compiler = {name for name, kind in model.packages['compiler'].targets if kind == 'bin'}
            self.assertEqual({t.name for t in release.targets if t.kind == 'cdylib'}, bundle)
            self.assertEqual({t.name for t in release.targets if t.kind == 'bin'}, set(model.release.bins))
            self.assertTrue(model.release.bins)
            self.assertLessEqual(set(model.release.bins), compiler)
            # The CPU profile build covers every released binary.
            profiled = {t.name for t in model.checks.profile_targets if t.kind == 'bin'}
            self.assertLessEqual(set(model.release.bins), profiled)
            self.assertIn('test_reference.py', model.checks.reference_tests)
            for path in inventory.compile_files(root, model):
                self.assertTrue(path.startswith(model.directory + '/'), path)
            qual = json.loads((root / model.directory / 'qual.json').read_text())
            self.assertEqual(qual['qualification']['checkpoint']['lock'], f'{model.directory}/checkpoint-files.json')
            self.assertTrue(tests_kernels_on_gpu(model))
        self.assertEqual(self.ci_module(root, 'kernel_lab_inventory').check(root, declared), [])

    def ci_module(self, root, name):
        spec = importlib.util.spec_from_file_location('inventory_' + name, root / 'ci' / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_a_released_model_missing_a_fact_still_fails(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        contributor = self.ci_module(root, 'contributor')
        # The full inventory, unreleased models included, passes unchanged.
        models = inventory.load(root)
        self.released_facts(root, contributor, models)
        first = next(model.name for model in models if model.release.enabled)
        broken = [dataclasses.replace(model, kernel_lab=dataclasses.replace(model.kernel_lab, catalog=False), lab=None)
                  if model.name == first else model for model in models]
        # Only one released model has neither a catalog nor a lab, and its check fails.
        with self.assertRaisesRegex(AssertionError, '^False is not true$'):
            self.released_facts(root, contributor, broken)
        # A model's own lab crate stands in for the catalog, and only it:
        # the released models without one keep the catalog.
        model = next(model for model in models if model.name == first)
        uncatalogued = dataclasses.replace(model, kernel_lab=dataclasses.replace(model.kernel_lab, catalog=False))
        self.assertTrue(tests_kernels_on_gpu(dataclasses.replace(uncatalogued, lab=f'{model.directory}/lab/Cargo.toml')))
        self.assertFalse(tests_kernels_on_gpu(dataclasses.replace(uncatalogued, lab=None)))
        self.assertTrue(all(model.kernel_lab.catalog for model in models if model.release.enabled and model.lab is None))

    def test_released_and_shipped_follow_the_release_flag(self):
        self.load(VALID + '[release]\nenabled = false\n')
        for role in ('bundle', 'compiler'):
            self.write(f'models/m4/{role}/Cargo.toml', '')
        for path in ('bundle/kernels.lock', 'profiles.json'):
            self.write('models/m4/' + path, '')
        self.write('models/m4/model.toml', VALID.replace('m3', 'm4') + '[release]\nenabled = true\nbins = ["m4-package"]\n')
        self.assertEqual(inventory.shipped(self.root), ('m3', 'm4'))
        self.assertEqual(inventory.released(self.root), ('m4',))
        self.write('models/m3/model.toml', VALID)
        self.assertEqual(inventory.released(self.root), ('m4',))

    def test_every_bundle_invokes_the_conformance_suite_once(self):
        model = self.load()
        [model] = model
        refused = 'models/m3/bundle/src: invoke conformance::suite! once in the bundle crate, found '
        with self.assertRaises(AssertionError) as caught:
            conformance_suite(self.root, model)
        self.assertEqual(str(caught.exception), refused + '[]')
        # Only a bundle that exports the bundle API can be served.
        self.assertFalse(exports_bundle_api(self.root, model))
        export = 'runtime_bundle_abi::export_bundle_api_v1!(\n    Model,\n);\n'
        # An export in a #[cfg(test)] module file serves only the crate's
        # tests; one outside it, or reached by a module that is not test-only,
        # is served and needs the suite.
        self.write('models/m3/bundle/src/api.rs', export)
        self.write('models/m3/bundle/src/lib.rs', '#[cfg(test)]\nmod api;\n')
        self.assertFalse(exports_bundle_api(self.root, model))
        self.write('models/m3/bundle/src/lib.rs', '#[cfg(test)]\nmod other;\nmod api;\n')
        self.assertTrue(exports_bundle_api(self.root, model))
        with self.assertRaises(AssertionError) as caught:
            conformance_suite(self.root, model)
        self.assertEqual(str(caught.exception), refused + '[]')
        self.write('models/m3/bundle/src/lib.rs', '#[cfg(test)]\nmod tests;\n')
        self.write('models/m3/bundle/src/tests.rs', 'mod api;\n')
        self.write('models/m3/bundle/src/tests/api.rs', export)
        (self.root / 'models/m3/bundle/src/api.rs').unlink()
        self.assertFalse(exports_bundle_api(self.root, model))
        self.write('models/m3/bundle/src/lib.rs', 'mod tests;\n')
        self.assertTrue(exports_bundle_api(self.root, model))
        (self.root / 'models/m3/bundle/src/tests/api.rs').unlink()
        (self.root / 'models/m3/bundle/src/tests.rs').unlink()
        self.write('models/m3/bundle/src/lib.rs', '#[cfg(test)]\nmod api {}\n' + export)
        self.assertTrue(exports_bundle_api(self.root, model))
        suite = 'conformance::suite! {\n    startup: Served {\n        tp4_ar: "tp4-ar",\n    }\n}\n'
        self.write('models/m3/bundle/src/tests/conformance.rs', suite)
        self.assertEqual(conformance_suite(self.root, model), 'models/m3/bundle/src/tests/conformance.rs')
        self.write('models/m3/bundle/src/lib.rs', 'bundle_runtime::conformance_suite! { startup: S { tp4_ar: "tp4-ar" } }\n')
        with self.assertRaises(AssertionError) as caught:
            conformance_suite(self.root, model)
        self.assertEqual(str(caught.exception), refused + "['models/m3/bundle/src/lib.rs', 'models/m3/bundle/src/tests/conformance.rs']")
        for direct in ('conformance::run::<S, Model>(Scenario::SlotReuse);\n',
                       'use bundle_runtime::fake_host::conformance::{\n    self, run, Scenario,\n};\n'):
            self.write('models/m3/bundle/src/lib.rs', direct)
            with self.assertRaises(AssertionError) as caught:
                conformance_suite(self.root, model)
            self.assertEqual(str(caught.exception), 'models/m3/bundle/src/lib.rs: run conformance scenarios through '
                                                    'conformance::suite!, not conformance::run')

    def test_real_declarations_preserve_all_consumer_facts(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        contributor = self.ci_module(root, 'contributor')
        models = inventory.load(root)
        inventory.check_files(models, root)
        for model in models:
            if model.release.enabled or exports_bundle_api(root, model):
                conformance_suite(root, model)
        # Each in-tree model names its author, whose checkpoints alone its JSON names.
        for model in models:
            self.assertIsNotNone(model.publisher, f'{model.directory}/model.toml: declare its publisher')
        self.assertEqual(inventory.shipped(root), tuple(model.name for model in models))
        self.assertEqual(inventory.released(root), tuple(model.name for model in models if model.release.enabled))
        # An unreleased model says so explicitly until it qualifies.
        for model in models:
            if not model.release.enabled:
                declared = tomllib.loads((root / model.directory / 'model.toml').read_text())
                self.assertIs(declared.get('release', {}).get('enabled'), False,
                              f'{model.directory}/model.toml: declare [release] enabled = false until it qualifies')
        self.released_facts(root, contributor, models)
        cleared_keys = {'CHOPINFER_PACKAGE_NATIVE_ROOT', 'CHOPINFER_PACKAGE_SNAPSHOT', 'CHOPINFER_PACKAGE_EXTRA'}
        for model in models:
            if model.contributor:
                cleared_keys.update(model.contributor.clear_env)
            if model.checkpoint_env:
                cleared_keys.update(f'CHOPINFER_{model.checkpoint_env}_{suffix}' for suffix in ('CHECKPOINT', 'TOKENIZER_DIR'))
        with patch.dict(os.environ, {key:'production-input' for key in cleared_keys}, clear=True):
            self.assertTrue(cleared_keys.isdisjoint(contributor.environment()))
        # tests/test_model_checks.py holds the CPU plan properties.
        self.ci_module(root, 'model_checks').plan(root, env={})


def load_tests(loader, tests, pattern):
    """These properties, then each model's own requirements
    (`models/<m>/test_inventory.py`, tests/model_requirements.py)."""
    tests.addTests(model_requirements.discover(loader))
    return tests


if __name__ == '__main__':
    unittest.main()

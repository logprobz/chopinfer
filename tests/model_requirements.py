"""Model-owned inventory requirements (`models/<m>/test_inventory.py`).

The shared inventory tests check properties over every model. The facts only a
model can state (its exact fixtures, release names, CPU reference steps and
profile targets, deployment policies, geometry recipe legs) live in that
model's own `test_inventory.py`, which tests/test_model_inventory.py
discovers. A new model adds its own file, or none, and edits no shared list.
Each file subclasses `ModelRequirements` and names its directory in `MODEL`;
the helpers return what the consumers (ci/model_checks.py, ci/contributor.py,
the release plan, tests/model_geometry_recipe.py) derive from it."""
import functools
import importlib.util
import json
import os
from pathlib import Path
import sys
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.model_sdk import inventory  # noqa: E402

PATTERN = 'models/*/test_inventory.py'


def discover(loader, root=ROOT):
    """Every model's own requirements, in directory order."""
    suite = unittest.TestSuite()
    for path in sorted(root.glob(PATTERN)):
        spec = importlib.util.spec_from_file_location(path.parent.name + '_test_inventory', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        owners = {value.MODEL for value in vars(module).values()
                  if isinstance(value, type) and issubclass(value, ModelRequirements) and value is not ModelRequirements}
        if owners != {path.parent.name}:
            raise AssertionError(f'{path.relative_to(root)}: MODEL {sorted(owners)} must name its own directory')
        suite.addTests(loader.loadTestsFromModule(module))
    return suite


@functools.cache
def ci_module(name):
    spec = importlib.util.spec_from_file_location('requirements_' + name, ROOT / 'ci' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelRequirements(unittest.TestCase):
    MODEL = None

    @property
    def model(self):
        return next(model for model in inventory.load(ROOT) if model.name == self.MODEL)

    @property
    def declared(self):
        """The model's model.toml as written."""
        return tomllib.loads((ROOT / self.model.directory / 'model.toml').read_text())

    def plan(self):
        return ci_module('model_checks').plan(ROOT, env={})

    def release(self):
        """The model's release record, or None when it is not released."""
        models = [model for model in inventory.load(ROOT) if model.release.enabled]
        return next((record for record in inventory.release_plan(models) if record.name == self.MODEL), None)

    def deployment_policy(self):
        """The model's deployment policy (`deploy/policy.json`, which its
        packages carry and its helper reports), as the image reads it."""
        spec = importlib.util.spec_from_file_location('requirements_deployment', ROOT / 'deploy/docker/deployment.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        path = ROOT / self.model.directory / 'deploy/policy.json'
        return module.policy(path.read_bytes(), str(path.relative_to(ROOT)))

    def cleared(self, keys):
        """The keys among `keys` that contributor checks clear."""
        with patch.dict(os.environ, {key: 'production-input' for key in keys}, clear=True):
            return set(keys) - ci_module('contributor').environment().keys()

    def geometry_legs(self, selection):
        """The model's (model, fixture) legs that tests/model-geometry-recipe.sh
        runs under `selection` (--ci or --contract)."""
        import model_geometry_recipe
        return [leg for leg in model_geometry_recipe.legs(selection) if leg[0] == self.MODEL]

    def qual(self):
        return json.loads((ROOT / self.model.directory / 'qual.json').read_text())

    def assert_references(self, *paths):
        """Each reference test runs as its own CPU step."""
        planned = self.plan()['references']
        for path in paths:
            kind = Path(path).stem.removeprefix('test_').removesuffix('reference').rstrip('_')
            label = 'refrun-' + (kind + '-' if kind else '') + self.MODEL
            self.assertIn((label, ['python3', '-B', '-m', 'unittest', f'{self.model.directory}/{path}', '-v']), planned)

    def assert_profiled(self, package, target, kind):
        """The CPU profile build builds `target` of `package`."""
        argv = self.plan()['profile']
        packages = [argv[i + 1] for i, word in enumerate(argv) if word == '-p']
        self.assertIn(package, packages)
        if kind == 'cdylib':
            # Libraries come first, before a single `--lib`.
            self.assertLess(argv.index(package), argv.index('--lib'))
        else:
            self.assertIn(target, [argv[i + 1] for i, word in enumerate(argv) if word == '--bin'])

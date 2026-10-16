"""Qwen's own inventory requirements (tests/model_requirements.py), run by
tests/test_model_inventory.py."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests'))
from model_requirements import ModelRequirements  # noqa: E402


class QwenInventory(ModelRequirements):
    MODEL = 'qwen38'

    def test_release_ships_the_package_command_and_bundle(self):
        release = self.release()
        self.assertIsNotNone(release, 'qwen38 is released')
        self.assertEqual({(target.name, target.kind) for target in release.targets},
                         {('qwen38-package', 'bin'), ('chopinfer-model-qwen38', 'bin'), ('bundle_qwen38', 'cdylib')})
        # The oracle is a developer binary and stays out of the release.
        self.assertIn(('qwen38-oracle', 'bin'), self.model.packages['compiler'].targets)
        self.assertEqual(self.model.release.assets, ())
        self.assertEqual([(asset.source, asset.destination) for asset in release.assets],
                         [('models/qwen38/bundle/kernels.lock', 'models/qwen38/kernels.lock'),
                          # The context-parallel family keeps the directory lock form.
                          ('models/qwen38/bundle/kernels.lock.d/context_parallel.lock',
                           'models/qwen38/kernels.lock.d/context_parallel.lock'),
                          ('models/qwen38/bundle/kernels.lock.d/context_parallel.provenance.json',
                           'models/qwen38/kernels.lock.d/context_parallel.provenance.json')])
        self.assertEqual(self.qual()['qualification']['checkpoint']['lock'], 'models/qwen38/checkpoint-files.json')
        policy = self.deployment_policy()
        self.assertEqual((policy['bundle_id'], policy['knob_prefixes'], policy['drafters']),
                         ('qwen38.flash.v2', ['QWEN38_'], ['mtp']))

    def test_contributor_fixtures_and_cleared_environment(self):
        self.assertEqual(self.model.contributor.fixtures, tuple('models/qwen38/compiler/fixtures/' + name for name in (
            'config.json', 'contributor-config.json', 'contributor-tensors.json', 'tensor-families.json',
            'checkpoints/index.json')))
        keys = {'CHOPINFER_QWEN38_NATIVE_MODULE', 'CHOPINFER_QWEN38_CHECKPOINT', 'CHOPINFER_QWEN38_TOKENIZER_DIR'}
        self.assertEqual(self.cleared(keys), keys)

    def test_cpu_checks(self):
        self.assert_references('test_reference.py', 'test_encoder_reference.py')
        self.assertEqual([path for path in self.plan()['compile'][3:] if path.startswith('models/qwen38/')], [])
        self.assertEqual([(t.package, t.name, t.kind) for t in self.model.checks.profile_targets],
                         [('model-qwen38', 'chopinfer-model-qwen38', 'bin'), ('model-qwen38', 'qwen38-package', 'bin'),
                          ('bundle-qwen38', 'bundle_qwen38', 'cdylib')])
        for target in self.model.checks.profile_targets:
            self.assert_profiled(target.package, target.name, target.kind)
        self.assertTrue(self.model.kernel_lab.catalog)

    def test_geometry_recipe_legs(self):
        recipe = self.model.geometry_recipe
        self.assertEqual(self.geometry_legs('--ci'), [('qwen38', 'synthetic-small'), ('qwen38', 'synthetic-wide')])
        self.assertEqual(recipe.architecture, 'QwenGeometryRecipeForConditionalGeneration')
        self.assertEqual(recipe.contract_tests, ('kernel_geometries', 'kernel_rank_catalog'))


if __name__ == '__main__':
    unittest.main()

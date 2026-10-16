"""MiMo's own inventory requirements (tests/model_requirements.py), run by
tests/test_model_inventory.py."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests'))
from model_requirements import ROOT, ModelRequirements  # noqa: E402


class MiMoInventory(ModelRequirements):
    MODEL = 'mimo26'

    def test_release_ships_the_package_command_and_bundle(self):
        release = self.release()
        self.assertIsNotNone(release, 'mimo26 is released')
        self.assertEqual({(target.name, target.kind) for target in release.targets},
                         {('mimo26-package', 'bin'), ('chopinfer-model-mimo26', 'bin'), ('bundle_mimo26', 'cdylib')})
        self.assertEqual(self.model.release.assets, ())
        self.assertEqual([(asset.source, asset.destination) for asset in release.assets],
                         [('models/mimo26/bundle/kernels.lock', 'models/mimo26/kernels.lock')])
        self.assertEqual(self.qual()['qualification']['checkpoint']['lock'], 'models/mimo26/checkpoint-files.json')
        policy = self.deployment_policy()
        self.assertEqual((policy['bundle_id'], policy['knob_prefixes'], policy['drafters']),
                         ('mimo.v26.flash.v2', ['MIMO_', 'ENCODER_'], ['dflash', 'mtp']))

    def test_contributor_fixtures_and_cleared_environment(self):
        self.assertEqual(self.model.contributor.fixtures, (
            'models/mimo26/compiler/fixtures/config.json', 'fixtures/mimo26/frontend/chat-renders.json',
            'fixtures/mimo26/multimodal/processor/processor-manifest.json'))
        keys = {'CHOPINFER_MIMO_FIXTURES', 'CHOPINFER_MIMO_CHECKPOINT', 'CHOPINFER_MIMO_TOKENIZER_DIR'}
        self.assertEqual(self.cleared(keys), keys)

    def test_cpu_checks(self):
        self.assert_references('test_reference.py', 'test_encoder_reference.py')
        directory = ROOT / self.model.directory
        inputs = [path.relative_to(ROOT).as_posix() for path in sorted(directory.glob('reference*.py'))]
        inputs.append('models/mimo26/encoder_reference.py')
        self.assertIn('models/mimo26/reference.py', inputs)
        compiled = [path for path in self.plan()['compile'][3:] if path.startswith('models/mimo26/')]
        self.assertEqual(compiled, inputs)
        self.assertEqual([(t.package, t.name, t.kind) for t in self.model.checks.profile_targets],
                         [('model-mimo26', 'chopinfer-model-mimo26', 'bin'), ('model-mimo26', 'mimo26-package', 'bin')])
        for target in self.model.checks.profile_targets:
            self.assert_profiled(target.package, target.name, target.kind)
        self.assertTrue(self.model.kernel_lab.catalog)

    def test_geometry_recipe_legs(self):
        recipe = self.model.geometry_recipe
        self.assertEqual(self.geometry_legs('--ci'), [('mimo26', 'synthetic-small')])
        self.assertEqual(recipe.architecture, 'MiMoGeometryRecipeForCausalLM')
        self.assertEqual(recipe.contract_tests, ('kernel_rank_catalog', 'kernel_cuda_geometry', 'kernel_engine',
                                                 'kernel_moe_build', 'kernel_moe_selection', 'kernel_comm',
                                                 'encoder_geometry_build'))


if __name__ == '__main__':
    unittest.main()

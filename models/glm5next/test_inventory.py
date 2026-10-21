"""GLM's own inventory requirements (tests/model_requirements.py), run by
tests/test_model_inventory.py."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests'))
from model_requirements import ROOT, ModelRequirements, inventory  # noqa: E402


class GlmInventory(ModelRequirements):
    MODEL = 'glm5next'

    def test_released_once_qualified(self):
        # Released after its full requalification through the release preview.
        self.assertIs(self.declared['release']['enabled'], True)
        self.assertIsNotNone(self.release())

    def test_release_targets_and_lock_directory(self):
        self.assertEqual(self.declared['release']['bins'], ['chopinfer-model-glm5next'])
        self.assertEqual(self.declared['release']['cdylibs'], ['bundle_glm5next'])
        release = self.release()
        self.assertEqual({(target.name, target.kind) for target in release.targets},
                         {('chopinfer-model-glm5next', 'bin'), ('bundle_glm5next', 'cdylib')})
        # The bundle keeps only the directory lock form; every family file ships.
        self.assertFalse((ROOT / 'models/glm5next/bundle/kernels.lock').exists())
        families = sorted(path.name for path in (ROOT / 'models/glm5next/bundle/kernels.lock.d').iterdir())
        self.assertTrue(families)
        self.assertEqual([(asset.source, asset.destination) for asset in release.assets],
                         [(f'models/glm5next/bundle/kernels.lock.d/{name}', f'models/glm5next/kernels.lock.d/{name}')
                          for name in families])
        self.assertEqual(self.qual()['qualification']['checkpoint']['lock'], 'models/glm5next/checkpoint-files.json')
        policy = self.deployment_policy()
        self.assertEqual((policy['bundle_id'], policy['knob_prefixes'], policy['drafters']),
                         ('glm5next.v1', ['GLM5NEXT_'], ['mtp']))

    def test_deploy_artifacts(self):
        # tests/test_deploy_artifacts.py checks these for released models.
        unit = (ROOT / 'deploy/systemd/chopinfer-glm5next.service').read_text()
        self.assertRegex(unit, r'CHOPINFER_EXPECTED_BUNDLE_ID=glm5next\.v1 /opt/chopinfer/bin/entrypoint\.sh')
        self.assertIn('EnvironmentFile=/etc/chopinfer/glm5next.env', unit)
        self.assertIn('ReadWritePaths=/var/lib/chopinfer', unit)
        self.assertIn('timeout 600s curl --retry 600', (ROOT / 'docs/developer-preview.md').read_text())
        # The packages carry the deployment policy, which
        # test_release_targets_and_lock_directory_once_enabled pins.

    def test_contributor_fixtures_and_cleared_environment(self):
        self.assertEqual(self.model.contributor.roots, ('model-glm5next',))
        self.assertEqual(self.model.contributor.fixtures, tuple(
            f'models/glm5next/contract/fixtures/{name}' for name in
            ('config.json',)))

    def test_cpu_checks(self):
        self.assert_references('test_reference.py', 'test_encoder_reference.py', 'test_nvfp4_reference.py')
        self.assertTrue(self.model.kernel_lab.catalog)
        keys = {'CHOPINFER_GLM5NEXT_CHECKPOINT', 'CHOPINFER_GLM5NEXT_TOKENIZER_DIR'}
        self.assertEqual(self.cleared(keys), keys)
        # The CPU profile build builds the helper; the bundle's CUDA build stays out of it.
        self.assertEqual([(t.package, t.name, t.kind) for t in self.model.checks.profile_targets],
                         [('model-glm5next', 'chopinfer-model-glm5next', 'bin')])
        for target in self.model.checks.profile_targets:
            self.assert_profiled(target.package, target.name, target.kind)

    def test_geometry_recipe_legs(self):
        # Contract stage: the bundle is compiled for the flash geometry at TP4.
        self.assertEqual(self.geometry_legs('--contract'), [('glm5next', 'synthetic-small')])
        self.assertEqual(self.geometry_legs('--ci'), [])
        self.assertEqual(self.model.geometry_recipe.architecture, 'GlmGeometryRecipeForConditionalGeneration')


if __name__ == '__main__':
    unittest.main()

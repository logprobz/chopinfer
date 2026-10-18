"""DeepSeek V4.1 Flash's own inventory requirements (tests/model_requirements.py),
run by tests/test_model_inventory.py."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests'))
from model_requirements import ROOT, ModelRequirements  # noqa: E402
from tools.model_sdk.graph import Graph  # noqa: E402


class DeepseekInventory(ModelRequirements):
    MODEL = 'deepseek41'

    def test_unreleased_until_qualified(self):
        # Released only after its full qualification through the release preview.
        self.assertIs(self.declared['release']['enabled'], False)
        self.assertIsNone(self.release())
        self.assertEqual(self.model.checkpoint_env, 'DEEPSEEK')

    def test_one_bundle_and_its_roles(self):
        self.assertEqual({role: package.name for role, package in self.model.packages.items()},
                         {'bundle': 'bundle-deepseek41', 'compiler': 'model-deepseek41',
                          'contract': 'model-deepseek41-contract'})
        self.assertEqual(self.declared['bundle']['cdylib'], ['bundle_deepseek41'])
        self.assertEqual(self.model.lab, 'models/deepseek41/lab/Cargo.toml')

    def test_the_checkpoint_pin_names_the_official_revision(self):
        pin = self.qual()['qualification']['checkpoint']
        self.assertEqual(pin['lock'], 'models/deepseek41/checkpoint-files.json')
        self.assertEqual((pin['repository'], pin['revision']),
                         ('deepseek-ai/DeepSeek-V4.1-Flash-1017', '1bae0e1088993b9b82de08f59de5be5a34e42170'))

    def test_the_kernel_test_crate_never_links_kernel_lab(self):
        # The model tests its kernels on kernel-harness from its lab crate,
        # by every edge including the dev edges its tests build.
        metadata = json.loads(subprocess.check_output(
            [os.environ.get('CARGO', 'cargo'), 'metadata', '--locked', '--offline', '--format-version', '1',
             '--all-features'], cwd=ROOT))
        graph = Graph(metadata)
        [lab] = graph.package_ids(['model-deepseek41-lab'])
        linked = {graph.packages[key]['name'] for key in graph.closure(lab, tests=True)}
        self.assertIn('kernel-harness', linked)
        self.assertFalse(linked & {'kernel-lab', 'kernel-lab-families'}, sorted(linked))
        self.assertNotIn(lab, metadata['workspace_default_members'])
        self.assertFalse(self.model.kernel_lab.catalog)

    def test_kv_bf16_judges_only_bf16_receipts(self):
        # The comparison configuration's prefix evidence must come from a
        # server running the BF16 KV layout: both prefix records and the
        # report's common serving identity.
        sys.path.insert(0, str(ROOT / 'tools' / 'qual'))
        import release_gate
        config = self.qual()['configurations']['kv-bf16']
        self.assertEqual(config['serve'], ['--kv-cache-dtype', 'bf16'])
        appended = config['append']['prefix_exact']
        self.assertEqual(sorted(appended), ['prefix_record_1', 'prefix_record_2', 'rules'])

        def record(kv):
            return {'server': {'health': {'serving': {'profile': {'kv_cache_dtype': kv}}}},
                    'serving_identity': {'profile': {'kv_cache_dtype': kv}}}
        for key, rules in appended.items():
            release_gate.judge_input(key, record('bf16'), rules, {})
            for kv in ('mixed', 'fp8'):
                with self.subTest(key=key, kv=kv), self.assertRaises(ValueError):
                    release_gate.judge_input(key, record(kv), rules, {})

    def test_geometry_recipe_legs(self):
        # The bundle serves an added geometry: a bundle-stage leg running
        # the contract's kernel suites.
        self.assertEqual(self.geometry_legs('--ci'), [('deepseek41', 'synthetic-small')])
        self.assertEqual(self.geometry_legs('--contract'), [])
        self.assertEqual(self.model.geometry_recipe.contract_tests,
                         ('kernel_compressor', 'kernel_latent', 'kernel_moe', 'kernel_mxfp8_fold'))
        self.assertEqual(self.model.geometry_recipe.architecture, 'DeepseekV41GeometryRecipeForCausalLM')


if __name__ == '__main__':
    unittest.main()

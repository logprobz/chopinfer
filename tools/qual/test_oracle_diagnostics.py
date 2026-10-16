"""Model-owned diagnostic adapters are selected for comparison and reuse."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.qual import oracle_diagnostics as oracle

ADAPTER = "def validate(a, b, scope):\n    raise ValueError('validated by ' + __file__)\n"


class AdapterDiscovery(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tree = Path(directory.name) / 'tree'
        self.outside = Path(directory.name) / 'outside'
        self.tree.mkdir()
        self.outside.mkdir()
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop('CHOPINFER_QUAL_MODEL_ROOT', None)

    def results(self, model):
        scope = {'model': model, 'policy_sha256': 'a' * 64}
        attachment = {'description': {'scope': scope}}
        with patch.object(oracle, 'in_tree_root', lambda model: self.tree / 'models' / model), \
             patch.object(oracle.wire, 'external', side_effect=lambda value: value), \
             patch.object(oracle.wire, 'load', return_value=attachment), \
             patch.object(oracle.wire, 'validate_scope', return_value={}), \
             patch.object(oracle, 'checkpoint', return_value='checkpoint'):
            return oracle.compare('reference', 'candidate', scope), oracle.reuse('candidate', scope)

    def assert_validated_by(self, results, path):
        for result in results:
            self.assertEqual(result['status'], 'insufficient')
            self.assertEqual(result['reason'], f'validated by {path.resolve()}')

    def test_third_model_validates_both_comparison_and_reuse(self):
        adapter = self.tree / 'models/third_fixture/diagnostic.py'
        adapter.parent.mkdir(parents=True)
        adapter.write_text(ADAPTER)
        self.assert_validated_by(self.results('third_fixture'), adapter)

    def test_an_outside_model_root_supplies_the_adapter(self):
        adapter = self.outside / 'diagnostic.py'
        adapter.write_text(ADAPTER)
        os.environ['CHOPINFER_QUAL_MODEL_ROOT'] = str(self.outside)
        self.assert_validated_by(self.results('outside_model'), adapter)

    def test_a_model_without_an_adapter_is_insufficient(self):
        (self.tree / 'models/bare').mkdir(parents=True)
        for model, root in (('bare', None), ('absent', None), ('absent', self.outside)):
            with self.subTest(model=model, root=root):
                if root is not None:
                    os.environ['CHOPINFER_QUAL_MODEL_ROOT'] = str(root)
                for result in self.results(model):
                    self.assertEqual(result['reason'], 'no model-owned diagnostic adapter for ' + model)

    def test_an_adapter_symlinked_into_the_root_is_refused(self):
        (self.tree / 'diagnostic.py').write_text(ADAPTER)
        (self.outside / 'diagnostic.py').symlink_to(self.tree / 'diagnostic.py')
        os.environ['CHOPINFER_QUAL_MODEL_ROOT'] = str(self.outside)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            oracle.model_adapter('outside_model')

    def test_in_tree_models_load_their_own_adapter(self):
        paths = sorted(oracle.in_tree_root('m').parent.glob('*/diagnostic.py'))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(model=path.parent.name):
                adapter = oracle.model_adapter(path.parent.name)
                self.assertEqual(Path(adapter.__file__), path.resolve())
                self.assertTrue(callable(adapter.validate))

    def test_model_names_cannot_select_an_arbitrary_python_module(self):
        for name in ('../other', 'a.b', '', '__init__'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                oracle.model_adapter(name)

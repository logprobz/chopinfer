"""Model-provided qualification modules load only from a regular file in
the model root."""
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import model_modules


class Loader(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / 'root'
        self.root.mkdir()
        self.outside = Path(directory.name) / 'outside'
        self.outside.mkdir()

    def test_a_regular_file_in_the_root_loads_as_a_new_module(self):
        (self.root / 'probe_module.py').write_text('VALUE = 7\n')
        module, _ = model_modules.load('probe_module', self.root)
        self.assertEqual(module.VALUE, 7)
        self.assertEqual(Path(module.__file__), self.root.resolve() / 'probe_module.py')

    def test_a_module_can_define_annotated_dataclasses(self):
        """dataclasses reads the defining module from sys.modules, which
        postponed annotations need."""
        (self.root / 'probe_module.py').write_text(
            'from __future__ import annotations\n'
            'from dataclasses import dataclass\n\n\n'
            '@dataclass\nclass Region:\n    name: str\n    count: int = 0\n')
        module, _ = model_modules.load('probe_module', self.root)
        self.assertEqual(module.Region('read').count, 0)
        self.assertIs(sys.modules[module.__name__], module)

    def test_each_load_is_a_new_privately_named_module(self):
        (self.root / 'probe_module.py').write_text('VALUE = []\n')
        first, second = (model_modules.load('probe_module', self.root)[0] for _ in range(2))
        self.assertIsNot(first, second)
        self.assertIsNot(first.VALUE, second.VALUE)
        self.assertNotEqual(first.__name__, second.__name__)
        for module in (first, second):
            self.assertTrue(module.__name__.startswith('_chopinfer_model_module_probe_module_'))

    def test_a_module_that_fails_to_run_is_not_registered(self):
        (self.root / 'probe_module.py').write_text('raise RuntimeError("probe")\n')
        before = set(sys.modules)
        with self.assertRaisesRegex(RuntimeError, 'probe'):
            model_modules.load('probe_module', self.root)
        self.assertEqual(set(sys.modules), before)

    def test_a_module_of_the_same_name_on_the_path_is_not_read(self):
        # common is a tools/qual module on sys.path.
        (self.root / 'common.py').write_text('VALUE = 7\n')
        import common
        module, _ = model_modules.load('common', self.root)
        self.assertEqual(module.VALUE, 7)
        self.assertFalse(hasattr(module, 'frontend'))
        self.assertIs(sys.modules['common'], common)
        self.assertFalse(hasattr(common, 'VALUE'))

    def test_the_digest_is_of_the_bytes_that_ran(self):
        """The loader reads the file once; a later change does not alter
        the digest it returned."""
        source = 'VALUE = 7\n'
        (self.root / 'probe_module.py').write_text(source)
        module, digest = model_modules.load('probe_module', self.root)
        (self.root / 'probe_module.py').write_text('VALUE = 8\n')
        self.assertEqual(digest, hashlib.sha256(source.encode()).hexdigest())
        self.assertEqual(module.VALUE, 7)

    def test_an_absent_module_is_none(self):
        self.assertIsNone(model_modules.load('probe_module', self.root))

    def test_a_symlink_is_refused(self):
        (self.outside / 'probe_module.py').write_text('VALUE = 7\n')
        (self.root / 'inside.py').write_text('VALUE = 7\n')
        for target in (self.outside / 'probe_module.py', self.root / 'inside.py', 'inside.py'):
            with self.subTest(target=str(target)):
                link = self.root / 'probe_module.py'
                link.unlink(missing_ok=True)
                link.symlink_to(target)
                with self.assertRaisesRegex(ValueError, 'is a symlink'):
                    model_modules.load('probe_module', self.root)

    def test_a_dangling_symlink_is_refused(self):
        (self.root / 'probe_module.py').symlink_to(self.outside / 'missing.py')
        with self.assertRaisesRegex(ValueError, 'is a symlink'):
            model_modules.load('probe_module', self.root)

    def test_a_name_that_leaves_the_root_is_refused(self):
        (self.outside / 'probe_module.py').write_text('VALUE = 7\n')
        for name in ('../outside/probe_module', 'outside/probe_module', '', 'probe.module', None):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'invalid model module name'):
                model_modules.load(name, self.root)

    def test_a_file_that_is_not_regular_is_refused(self):
        (self.root / 'probe_module.py').mkdir()
        with self.assertRaisesRegex(ValueError, 'not a regular file'):
            model_modules.load('probe_module', self.root)

    def test_an_oversized_module_is_refused(self):
        (self.root / 'probe_module.py').write_bytes(b'#' * (model_modules.MAX_BYTES + 1))
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            model_modules.load('probe_module', self.root)

    def test_a_root_that_is_no_directory_is_refused(self):
        for root in (self.root / 'missing', self.outside / 'file'):
            (self.outside / 'file').write_text('')
            with self.subTest(root=str(root)), self.assertRaisesRegex(ValueError, 'is not a directory'):
                model_modules.load('probe_module', root)

    def test_the_model_root_is_the_explicit_one_else_the_environment(self):
        with patch.dict(os.environ, {'CHOPINFER_QUAL_MODEL_ROOT': str(self.outside)}):
            self.assertEqual(model_modules.model_root(str(self.root)), self.root)
            self.assertEqual(model_modules.model_root(), self.outside)
        environment = {k: v for k, v in os.environ.items() if k != 'CHOPINFER_QUAL_MODEL_ROOT'}
        with patch.dict(os.environ, environment, clear=True):
            self.assertIsNone(model_modules.model_root())


if __name__ == '__main__':
    unittest.main()

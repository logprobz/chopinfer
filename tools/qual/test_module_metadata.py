"""tools/qual/modules.json is what module_metadata.py derives from the
module sources, so chopinfer eval validates declarations as declaration.py
does."""

import unittest

import module_metadata


class ModuleMetadataTests(unittest.TestCase):
    def test_the_shipped_metadata_is_current(self):
        self.assertEqual(module_metadata.OUTPUT.read_text(), module_metadata.encoded(),
                         "regenerate it: python3 -B tools/qual/module_metadata.py")

    def test_kinds_follow_the_override_rules(self):
        self.assertEqual([module_metadata.kind(default) for default in
                          (None, (True,), (3,), (0.5,), ("a",), ((1,),), ([1],), ({},), (None,), ({1},))],
                         ["number", "bool", "int", "number", "str", "list", "list", "dict", "null", "set"])


if __name__ == "__main__":
    unittest.main()

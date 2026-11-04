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

    def test_guard_constants_take_no_override(self):
        # A declaration that blanked these could adopt proof inputs in a
        # scoped refs check or hide a server's retired prefill controls.
        modules = module_metadata.metadata()["modules"]
        guards = {"refs": ("CODE_INPUTS", "KEPT_SECTIONS", "LOCK_KEYS", "PROOF_INPUTS", "SNAPSHOT_FIELDS"),
                  "agentic_server": ("RETIRED_FLAGS", "RETIRED_METRICS")}
        self.assertEqual({(module, name): modules[module][name] for module, names in guards.items() for name in names},
                         {(module, name): "set" for module, names in guards.items() for name in names})


if __name__ == "__main__":
    unittest.main()

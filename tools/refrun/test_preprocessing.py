"""Preprocessing bindings reject stale producer code and changed tensor bytes."""

import copy
from pathlib import Path
import tempfile
import unittest

from tools.model_sdk.diagnostic import canonical
from tools.refrun import keys, preprocessing


class BindingTests(unittest.TestCase):
    def test_changed_tensors_and_stale_code_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = preprocessing.write_tensor(root, "source.bin", b"\x01", "u8", (1,))
            rows = preprocessing.write_tensor(root, "rows.bin", preprocessing.packed([1.0], "f32"), "f32", (1,))
            record = {"schema": preprocessing.SCHEMA,
                      "binding": {"code": keys.repo_files(preprocessing.CODE), "requirements": {}},
                      "cases": [{"name": "sample", "source": source, "stages": {"patches": {"f32": rows}}}]}
            def seal(value):
                value["sha256"] = keys.sha256(canonical({k: v for k, v in value.items() if k != "sha256"}))
                return value
            seal(record)
            self.assertEqual(preprocessing.verify(record, root), record)
            stale = copy.deepcopy(record)
            stale["binding"]["code"][preprocessing.CODE[0]] = "0" * 64
            with self.assertRaisesRegex(ValueError, "stale preprocessing"):
                preprocessing.verify(seal(stale), root)
            (root / "rows.bin").write_bytes(preprocessing.packed([2.0], "f32"))
            with self.assertRaisesRegex(ValueError, "tensor digest"):
                preprocessing.verify(record, root)

    def test_manifest_digest_and_empty_capture_are_refused(self):
        for value in ({"schema": preprocessing.SCHEMA, "cases": []},
                      {"schema": preprocessing.SCHEMA, "cases": [{}], "sha256": "0" * 64}):
            with self.assertRaises(ValueError):
                preprocessing.verify(value, Path("."))


if __name__ == "__main__":
    unittest.main()

"""Append-only projection adoption guard regressions in isolated repositories."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


guard = module("projection_guard", ROOT / "tests/projection-adoptions.py")
fixtures = module("encoder_guard_tests", ROOT / "tests/test_encoder_adoptions.py")
fixtures.CHECK = ROOT / "tests/projection-adoptions.py"
fixtures.LEDGER = "models/example/bundle/projection-adoptions.json"
A = {"symbol": "legacy_a_v1", "generated": "chop_proj_dense_a_v1", "entry_sha256": "a" * 64}
B = {"symbol": "legacy_b_v1", "generated": "chop_proj_dense_b_v1", "entry_sha256": "b" * 64}


class ProjectionAdoptionsTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.repository = fixtures.Repository(Path(self.directory.name))
        self.repository.commit([A])
        self.repository.run("checkout", "-q", "-b", "branch")

    def tearDown(self):
        self.directory.cleanup()

    def test_append_preserves_approved_entries(self):
        self.assertEqual(self.repository.check().returncode, 0)
        self.repository.write([A, B])
        self.assertEqual(self.repository.check().returncode, 0)

    def test_superseding_record_appends_without_rewriting_history(self):
        self.repository.write([A, B, dict(A, generated="chop_proj_dense_new_v1", entry_sha256="c" * 64)])
        self.assertEqual(self.repository.check().returncode, 0)

    def test_rewrite_remove_and_reorder_fail(self):
        for entries in [[], [dict(A, entry_sha256="b" * 64)], [B, A]]:
            with self.subTest(entries=entries):
                self.repository.write(entries)
                self.assertEqual(self.repository.check().returncode, 1)

    def test_deleting_or_replacing_entire_ledger_fails(self):
        path = self.repository.root / fixtures.LEDGER
        path.unlink()
        self.assertEqual(self.repository.check().returncode, 1)
        path.symlink_to(self.repository.root / "README")
        self.assertEqual(self.repository.check().returncode, 1)

    def test_duplicate_invalid_and_noncanonical_entries_fail(self):
        for text in [fixtures.text([A, A]), fixtures.text([dict(A, extra=1)]),
                     fixtures.text([dict(A, generated="invalid")]),
                     fixtures.text([dict(A, entry_sha256="bad")]),
                     fixtures.text([dict(A, symbol=None)]), json.dumps([A]), "{", "{}"]:
            with self.subTest(text=text):
                self.repository.write(text)
                self.assertEqual(self.repository.check().returncode, 1)

    def test_placements_are_unique_for_active_records(self):
        placed = dict(A, placement={"cubin": "locked_cubin", "position": 0})
        self.repository.write([A, placed, dict(B, placement={"cubin": "locked_cubin", "position": 1})])
        self.assertEqual(self.repository.check().returncode, 0)
        for place in [{"cubin": "locked_cubin", "position": 0},
                      {"cubin": "locked_cubin", "position": -1},
                      {"cubin": "locked_cubin", "position": True},
                      {"cubin": "../escape", "position": 1}]:
            self.repository.write([A, placed, dict(B, placement=place)])
            self.assertEqual(self.repository.check().returncode, 1)

    def test_retired_symbols_name_an_active_replacement(self):
        placed = dict(A, placement={"cubin": "locked_cubin", "position": 0})
        new = dict(B, symbol="chop_proj_dense_b_v1", generated="chop_proj_dense_b_v1",
                   placement={"cubin": "locked_cubin", "position": 0})
        retire = {key: placed[key] for key in ("symbol", "generated", "entry_sha256")}
        retire["replaced_by"] = new["symbol"]
        self.repository.write([A, placed, retire, new])
        self.assertEqual(self.repository.check().returncode, 0)
        # A second retirement keeps both steps of history; the chain still ends active.
        newer = dict(new, symbol="chop_proj_dense_c_v1", generated="chop_proj_dense_c_v1")
        retire_new = {key: new[key] for key in ("symbol", "generated", "entry_sha256")}
        retire_new["replaced_by"] = newer["symbol"]
        self.repository.write([A, placed, retire, new, retire_new, newer])
        self.assertEqual(self.repository.check().returncode, 0)
        cycle = dict(retire_new, replaced_by=A["symbol"])
        missing = dict(retire, replaced_by="chop_proj_absent_v1")
        self_cycle = dict(retire, replaced_by=A["symbol"])
        for entries, problem in [
                ([A, placed, retire, new, cycle], "replaced in a cycle"),
                ([A, placed, self_cycle], "replaced in a cycle"),
                ([A, placed, missing], "replaced by missing chop_proj_absent_v1"),
                ([A, placed, retire], "replaced by missing chop_proj_dense_b_v1"),
                ([A, placed, dict(retire, placement=placed["placement"])], "invalid fields"),
                ([A, placed, dict(retire, replaced_by=None)], "non-string fields"),
                ([A, placed, dict(retire, replaced_by="bad")], "invalid identity")]:
            with self.subTest(entries=entries):
                self.repository.write(entries)
                result = self.repository.check()
                self.assertEqual(result.returncode, 1)
                self.assertIn(problem, result.stdout)

    def test_retired_records_free_their_slot(self):
        placed = dict(A, placement={"cubin": "locked_cubin", "position": 0})
        retire = dict(A, replaced_by=B["symbol"])
        self.repository.write([A, placed, retire, dict(B, placement=placed["placement"])])
        self.assertEqual(self.repository.check().returncode, 0)

    def test_new_ledger_passes(self):
        path = self.repository.root / "models/other/bundle/projection-adoptions.json"
        path.parent.mkdir(parents=True)
        path.write_text(fixtures.text([B]))
        self.assertEqual(self.repository.check().returncode, 0)

    def test_example_ledgers_are_checked(self):
        path = self.repository.root / "examples/demo/bundle/projection-adoptions.json"
        path.parent.mkdir(parents=True)
        path.write_text(fixtures.text([B]))
        self.assertEqual(self.repository.check().returncode, 0)
        path.write_text(json.dumps([B]))
        self.assertEqual(self.repository.check().returncode, 1)


if __name__ == "__main__":
    unittest.main()

"""Verify the producer interface, bounded reads and historical membership."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from kernel_report_fixture import HEADER, make_report, to_directory, write_report
import kernel_reports as kr


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path, self.report, self.bindings = make_report(self.root / "build")

    def read(self, report=None, **kwargs):
        if report is not None:
            write_report(self.path, report)
        reader = kr.Reader(kwargs.pop("limits", None))
        variants = reader.load(current=[self.path], bindings=self.bindings, **kwargs)
        return variants[0]

    def test_current_and_separate_declaration_completeness(self):
        for complete in (False, True):
            self.report["entry_declarations_complete"] = complete
            row = self.read(self.report)
            self.assertEqual(row["status"], "current", row["issues"])
            self.assertEqual(row["report"]["entry_declarations_complete"], complete)
            self.assertEqual(row["report"]["cubins"][0]["identity"], self.report["cubins"][0]["identity"])

    def test_all_local_evidence_is_verified(self):
        cubin = self.report["cubins"][0]
        paths = {"lock": self.report["lock_path"], "compiler": self.report["compiler_path"],
                 "cubin": cubin["path"], "sidecar": cubin["identity_path"],
                 "source": str(self.bindings["chopinfer"] / HEADER), "generated": str(self.path.parent / "entry.cu")}
        for name, value in paths.items():
            path = Path(value)
            original = path.read_bytes()
            for missing in (False, True):
                with self.subTest(name=name, missing=missing):
                    if missing:
                        path.unlink()
                    else:
                        path.write_bytes(original + b"mutation")
                    row = self.read()
                    self.assertEqual(row["status"], "stale")
                    self.assertTrue(row["issues"])
                    path.write_bytes(original)

    def test_structural_refusals(self):
        changes = [lambda r: r.update(schema="future"), lambda r: r.update(lock_status="unknown"),
                   lambda r: r.update(source_membership_complete=1),
                   lambda r: r["cubins"][0]["recipe"]["defines"].reverse(),
                   lambda r: r["cubins"][0]["recipe"].update(generated_sha256=[]),
                   lambda r: r["cubins"][0]["declarations"].append({"symbol": False}),
                   lambda r: r["cubins"][0]["membership"][0].update(file="0" * 64),
                   lambda r: r["cubins"][0]["membership"].append(r["cubins"][0]["membership"][0]),
                   lambda r: next(iter(r["files"].values()))["identity"].update(path="../escape"),
                   lambda r: next(iter(r["files"].values())).update(sha256="bad"),
                   lambda r: r["context"]["sources"].append(r["context"]["sources"][0])]
        for change in changes:
            report = deepcopy(self.report)
            change(report)
            row = self.read(report)
            self.assertEqual(row["status"], "invalid", row)
            self.assertTrue(row["issues"])

    def test_recipe_duplicate_fields_and_generated_hashes(self):
        for field, value in [("toolchain_version", "other"), ("flags", ["-O0"]),
                             ("generated_sha256", {"entry.cu": "0" * 64})]:
            report = deepcopy(self.report)
            cubin = report["cubins"][0]
            cubin["recipe"][field] = value
            cubin["recipe_sha256"] = kr.digest(kr.recipe_bytes(cubin["recipe"]))
            self.assertEqual(self.read(report)["status"], "invalid")

    def test_ordered_recipe_and_unordered_json_keys(self):
        self.path.write_text(json.dumps(self.report, sort_keys=True))
        self.assertEqual(self.read()["status"], "current")
        recipe = self.report["cubins"][0]["recipe"]
        before = kr.digest(kr.recipe_bytes(recipe))
        recipe["defines"].reverse()
        self.assertNotEqual(before, kr.digest(kr.recipe_bytes(recipe)))

    def test_nonfinite_and_duplicate_json_keys(self):
        for content in ['{"schema": 0, "schema": 1}', '{"schema": NaN}']:
            self.path.write_text(content)
            self.assertEqual(self.read()["status"], "invalid")

    def test_lock_status_and_incomplete_membership_or_ownership(self):
        for field, value in [("lock_status", status) for status in ("Skipped", "Updated", "Profiling")] + [
                ("source_membership_complete", False), ("source_ownership_complete", False), ("context", None)]:
            report = deepcopy(self.report)
            report[field] = value
            row = self.read(report)
            self.assertEqual(row["status"], "stale", row)
            self.assertTrue(row["issues"])

    def test_closure_hash_is_recomputed(self):
        report = deepcopy(self.report)
        report["cubins"][0]["membership"][0]["closure_key"] += "/different"
        self.assertIn("source closure differs", " ".join(self.read(report)["issues"]))

    def test_previous_membership_does_not_reopen_removed_files(self):
        for file in self.report["files"].values():
            Path(file["location"]).unlink()
        reader = kr.Reader()
        row = reader.load(previous=[self.path])[0]
        self.assertEqual(row["status"], "historical")
        self.assertFalse(row["content_current"])
        self.assertIn(HEADER, [f["identity"]["path"] for f in row["report"]["files"].values()])

    def test_lookup_roots_and_candidate_source_copy(self):
        report = deepcopy(self.report)
        report["context"]["sources"][0]["path"] += "/bundle/../../source"
        self.assertEqual(self.read(report)["status"], "current")
        report["context"]["sources"][0]["path"] = str(self.root / "wrong")
        self.assertIn("root/location differs", " ".join(self.read(report)["issues"]))
        write_report(self.path, self.report)
        other = self.root / "candidate" / HEADER
        other.parent.mkdir(parents=True)
        other.write_bytes(b"changed candidate")
        reader = kr.Reader()
        row = reader.load([self.path], bindings={"chopinfer": self.root / "candidate"})[0]
        # Obsolete ownership is refused before spending the closure budget.
        self.assertFalse(row["content_current"])
        self.assertEqual(reader.files.closure_bytes, 0)
        self.assertFalse(row["ownership_current"])
        self.assertIn("stale bound source", " ".join(row["issues"]))

    def test_files_directories_aliases_and_descendant_symlinks(self):
        alias = self.root / "alias"
        alias.symlink_to(self.path.parent, target_is_directory=True)
        reader = kr.Reader()
        rows = reader.load([alias, self.path, self.path.parent], bindings=self.bindings)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "current")
        link = self.path.parent / "nested"
        link.symlink_to(self.path.parent, target_is_directory=True)
        reader = kr.Reader()
        reader.load([alias], bindings=self.bindings)
        self.assertTrue(any("symlinked report descendant" in x for x in reader.gaps))
        self.path.unlink()
        self.path.symlink_to(self.path.parent / "fixture_v1.identity.json")
        reader = kr.Reader()
        self.assertEqual(reader.load([self.path]), [])
        self.assertTrue(reader.gaps)

    def test_profiling_filename_is_enumerated_without_release_coverage(self):
        profile = deepcopy(self.report)
        profile['lock_status'] = 'Profiling'
        write_report(self.path.parent / 'kernel-build-report.profile.json', profile)
        rows = kr.Reader().load([self.path.parent], bindings=self.bindings)
        self.assertEqual({r['status'] for r in rows}, {'current', 'stale'})
        self.assertEqual(len(rows), 2)

    def test_report_supplied_symlink_parents_are_refused(self):
        alias = self.root / "unrequested-alias"
        alias.symlink_to(self.path.parent, target_is_directory=True)
        self.report["compiler_path"] = str(alias / "nvcc")
        self.assertEqual(self.read(self.report)["status"], "stale")
        reader = kr.Reader()
        self.assertEqual(reader.load([alias], bindings=self.bindings)[0]["status"], "current")
        self.report["compiler_path"] = str(alias / ".." / "build" / "nvcc")
        self.assertIn("parent traversal", " ".join(self.read(self.report)["issues"]))

    def test_missing_empty_and_bounded_inputs_are_gaps(self):
        empty = self.root / "empty"
        empty.mkdir()
        for inputs, limits in [([empty], kr.Limits()), ([self.root / "missing"], kr.Limits()),
                               ([self.root], replace(kr.Limits(), entries=1)),
                               ([self.root], replace(kr.Limits(), depth=0)),
                               ([self.path], replace(kr.Limits(), reports=0))]:
            reader = kr.Reader(limits)
            reader.load(inputs)
            self.assertTrue(reader.gaps)
        for field in ("report_bytes", "total_bytes", "records", "source_bytes", "artifact_bytes", "closure_bytes", "files"):
            row = self.read(limits=replace(kr.Limits(), **{field: 1}))
            self.assertNotEqual(row["status"], "current", field)

    def test_realistic_report_and_streamed_compiler_sizes(self):
        # Legal JSON whitespace exercises the report byte limit without huge fixtures.
        self.path.write_text(json.dumps(self.report) + " " * 11_100_000)
        compiler = Path(self.report["compiler_path"])
        compiler.write_bytes(b"x" * 34_000_000)
        reader = kr.Reader()
        self.assertEqual(reader.files.read(compiler, reader.limits.artifact_bytes)[2], 34_000_000)
        self.assertIsNone(reader.files.cache[compiler][2])
        self.assertEqual(reader.files.json(self.path), self.report)

    def test_accumulated_variants_do_not_spend_the_current_closure_budget(self):
        paths = []
        for index in range(48):
            path, report, _ = make_report(self.root / f"a-old-{index:02}", content=b"old source")
            if index % 2:
                report["context"] = None
                report["source_ownership_complete"] = False
                write_report(path, report)
            paths.append(path)
        paths.append(self.path)
        for index in range(24):
            path = self.root / f"z-duplicate-{index:02}.json"
            write_report(path, self.report)
            paths.append(path)
        size = sum(f["bytes"] for f in self.report["files"].values())
        reader = kr.Reader(replace(kr.Limits(), closure_bytes=size))
        rows = reader.load(paths, bindings=self.bindings)
        self.assertEqual(sum(row["status"] == "current" for row in rows), 25)
        self.assertEqual(sum(row["status"] == "stale" for row in rows), 48)
        self.assertEqual(reader.files.closure_bytes, size)
        self.assertFalse(reader.gaps)

    def test_cached_closures_keep_artifact_and_membership_checks(self):
        other, report, _ = make_report(self.root / "other")
        reader = kr.Reader()
        self.assertEqual(reader.load([self.path], bindings=self.bindings)[0]["status"], "current")
        # Equivalent bytes in different build directories can share a closure hash.
        self.assertEqual(reader.load([other], bindings=self.bindings)[0]["status"], "current")
        size = sum(f["bytes"] for f in self.report["files"].values())
        self.assertEqual(reader.files.closure_bytes, size)
        Path(report["cubins"][0]["path"]).write_bytes(b"changed cubin")
        self.assertEqual(reader.load([other], bindings=self.bindings)[0]["status"], "stale")
        changed = deepcopy(self.report)
        changed["cubins"][0]["membership"][0]["closure_key"] += "/different"
        path = self.root / "changed.json"
        write_report(path, changed)
        row = reader.load([path], bindings=self.bindings)[0]
        self.assertIn("source closure differs", " ".join(row["issues"]))

    def test_reused_closure_does_not_trust_another_builds_artifacts(self):
        other, report, _ = make_report(self.root / "other")
        cubin = report["cubins"][0]
        paths = [report["lock_path"], report["compiler_path"], cubin["path"], cubin["identity_path"],
                 *(f["location"] for f in report["files"].values())]
        for value in paths:
            path = Path(value)
            original = path.read_bytes()
            for missing in (False, True):
                with self.subTest(path=path.name, missing=missing):
                    reader = kr.Reader()
                    self.assertEqual(reader.load([self.path], bindings=self.bindings)[0]["status"], "current")
                    if missing:
                        path.unlink()
                    else:
                        path.write_bytes(original + b"mutation")
                    self.assertEqual(reader.load([other], bindings=self.bindings)[0]["status"], "stale")
                    path.write_bytes(original)

    def test_directory_lock_is_current_with_provenance_drift(self):
        lock, provenance = to_directory(self.path, self.report)
        self.assertEqual(self.read()["status"], "current", self.read()["issues"])
        # Provenance is report-only: drift, a changed file or none at all stays current.
        self.report["lock_families"][0]["provenance_changed"] = ["fixture_v1"]
        for content in (b"[]", None):
            if content is None:
                provenance.unlink()
            else:
                provenance.write_bytes(content)
            row = self.read(self.report)
            self.assertEqual(row["status"], "current", row["issues"])
            self.assertEqual(row["report"]["lock_families"][0]["provenance_changed"], ["fixture_v1"])
        self.assertTrue(lock.is_file())

    def test_directory_lock_missing_or_changed_family(self):
        lock, _ = to_directory(self.path, self.report)
        original = lock.read_bytes()
        records = json.loads(original)
        changed = deepcopy(records)
        changed[0]["cubin_sha256"] = "0" * 64
        legacy = deepcopy(records)
        legacy[0]["source_closure_sha256"] = self.report["cubins"][0]["identity"]["source_closure_sha256"]
        cases = [(None, False, "symbolic links"), (original + b" ", False, "stale lock digest: family fixture"),
                 (json.dumps(changed).encode(), True, "lock identities differ: family fixture"),
                 (json.dumps(legacy).encode(), True, "lock identities differ: family fixture")]
        for content, rehash, issue in cases:
            with self.subTest(issue=issue, rehash=rehash):
                report = deepcopy(self.report)
                lock.unlink()
                if content is None:
                    # No-follow reads refuse a symlinked family lock.
                    lock.symlink_to(self.path)
                else:
                    lock.write_bytes(content)
                if rehash:
                    report["lock_families"][0]["lock_sha256"] = kr.digest(content)
                try:
                    row = self.read(report)
                finally:
                    lock.unlink()
                    lock.write_bytes(original)
                self.assertEqual(row["status"], "stale")
                self.assertIn(issue, " ".join(row["issues"]))
        lock.unlink()
        row = self.read(self.report)
        self.assertEqual(row["status"], "stale")
        self.assertTrue(row["issues"])

    def test_directory_lock_coverage_refusals(self):
        to_directory(self.path, self.report)
        family = self.report["lock_families"][0]
        changes = [lambda r: r["lock_families"].append(deepcopy(family) | {"family": "second"}),
                   lambda r: r["lock_families"].append(deepcopy(family)),
                   lambda r: r["lock_families"][0].update(cubins=["fixture_v1", "fixture_v1"]),
                   lambda r: r["lock_families"][0].update(cubins=["other_v1"]),
                   lambda r: r["lock_families"][0].update(cubins=[]),
                   lambda r: r["lock_families"][0].update(provenance_changed=["other_v1"]),
                   lambda r: r["lock_families"][0].update(family="../escape"),
                   lambda r: r["lock_families"][0].update(lock_sha256="bad"),
                   lambda r: r.update(lock_families=[]),
                   lambda r: r.update(lock_sha256="0" * 64)]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                report = deepcopy(self.report)
                change(report)
                row = self.read(report)
                self.assertEqual(row["status"], "invalid", row)
                self.assertTrue(row["issues"])
        self.assertIn("duplicate lock family coverage", " ".join(self.read(
            deepcopy(self.report) | {"lock_families": [family, family | {"family": "second"}]})["issues"]))

    def test_cache_detects_mutation(self):
        files = kr.Files(kr.Limits())
        files.read(self.path, files.limits.report_bytes)
        self.path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed during inspection"):
            files.read(self.path, files.limits.report_bytes)


if __name__ == "__main__":
    unittest.main()

"""Read real reports after the CPU gate's maintained cuda-probe SDK smoke.

Missing required current reports fail this gate. Extra stale feature variants
remain visible and do not invalidate a different, verified current variant.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ci"))
import change_scope
import kernel_reports as kr
import kernel_scope
import validation

REQUIRED = {"bundle-mimo26", "bundle-qwen38", "kernel-lab", "example-bundle"}
HEADERS = ["crates/kernels-sm120/cuda/" + h for h in ("common/ptx.cuh", "common/sigmoid.cuh", "row/row.cuh")]


def lock_digests(variant):
    """A report's lock digests: its lock file's, or each family's when it
    built against a lock directory."""
    lock = variant["lock"]
    return [lock["sha256"]] if lock["sha256"] else [f["sha256"] for f in lock.get("families") or []]


class RealReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = ("CHOPINFER_KERNEL_REPORT_TARGET", "CHOPINFER_SDK_REPORT_TARGET", "CHOPINFER_SDK_REPORT_WORKSPACE")
        missing = [name for name in names if not os.environ.get(name)]
        if missing:
            raise AssertionError("Run after the SDK smoke with explicit artifact locations: " + ", ".join(missing))
        cls.core, cls.sdk, cls.external = [Path(os.environ[name]) for name in names]
        cls.metadata = json.loads(subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--all-features", "--locked", "--offline"], cwd=ROOT))
        cls.graph = change_scope.Graph(cls.metadata)
        cls.catalog = validation.Catalog(ROOT, cls.metadata)
        cls.index = kernel_scope.Index(cls.metadata, cls.graph, cls.catalog, [cls.core, cls.sdk])

    def current(self, index):
        found = {}
        for key, row in index.consumers.items():
            name = key[1].split("@")[0]
            if name in REQUIRED:
                found.setdefault(name, []).extend(v for v in row["variants"] if v["status"] == "current")
        for name in sorted(REQUIRED):
            self.assertTrue(found.get(name), name + ": no current report; " + str(index.issues) + "; " + str([
                (key, [(v["path"], v["issues"]) for v in row["variants"]])
                for key, row in index.consumers.items() if key[1].split("@")[0] == name]))
        return found

    def test_real_current_consumers_and_completeness(self):
        found = self.current(self.index)
        self.assertGreater(max(Path(v["path"]).stat().st_size for v in found["kernel-lab"]), 8_000_000)
        for name, variants in found.items():
            for variant in variants:
                summary = self.index.describe(variant)
                raw = json.loads(Path(variant["path"]).read_bytes())
                self.assertEqual(summary["completeness"], {key: raw[key] for key in kr.FLAGS})
                self.assertEqual(summary["lock"]["status"], "Verified")
                if name == "example-bundle":
                    context = raw["context"]
                    self.assertEqual(context["consumer"]["source"], "external-model")
                    roots = {s["id"]: kr.lexical(s["path"]) for s in context["sources"]}
                    self.assertEqual(roots["external-model"], self.external.resolve())
        print("current kernel report variants: " + json.dumps({k: len(v) for k, v in sorted(found.items())}))

    def test_actual_shared_headers_select_every_required_consumer(self):
        self.current(self.index)
        for header in HEADERS:
            selection = self.graph.select([header], kernel_index=self.index, kernel_changes=self.catalog.classify(header))
            # A package may build several selections (bundle-qwen38's catalogs
            # and its context-parallel lock family); each affected one counts.
            hits = {}
            for c in selection["kernel_consumers"]:
                if c["affected_sources"]:
                    hits.setdefault(c["identity"]["package"].split("@")[0], []).append(c)
            self.assertTrue(REQUIRED <= hits.keys(), header)
            for name in REQUIRED:
                for row in hits[name]:
                    self.assertTrue(row["selected"])
                    self.assertTrue(row["affected_cubins"])
                    self.assertTrue(any(r["kind"] == "current membership" for r in row["reasons"]))
                    self.assertTrue(any(all(lock_digests(v)) and lock_digests(v) for v in row["variants"]))
            self.assertNotIn("example-bundle", selection["packages"])

    def test_stale_extra_and_missing_historical_sources_keep_current_evidence(self):
        found = self.current(self.index)
        # Each declared selection is its own consumer, including selections
        # with no current report. Exercise the fixture's exact identity instead
        # of whichever external consumer sorts first.
        sdk_reports = {kernel_scope.consumer_key(v["report"]["context"]): v
                       for v in found["example-bundle"]}
        for key, sdk in sorted(sdk_reports.items()):
            with self.subTest(consumer=key), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                stale = deepcopy(sdk["report"])
                compiler = root / "changed-compiler"
                compiler.write_bytes(b"stale compiler fixture")
                stale["compiler_path"] = str(compiler)
                stale_path = root / "kernel-build-report.json"
                stale_path.write_text(json.dumps(stale))
                historical = deepcopy(sdk["report"])
                for file in historical["files"].values():
                    file["location"] = str(root / "removed-source")
                previous = root / "previous.json"
                previous.write_text(json.dumps(historical))
                paths = [variants[0]["path"] for name, variants in found.items() if name != "example-bundle"]
                paths.append(sdk["path"])
                index = kernel_scope.Index(self.metadata, self.graph, self.catalog, [*paths, stale_path], [previous])
                self.current(index)
                report = self.graph.select([HEADERS[0]], kernel_index=index, kernel_changes=self.catalog.classify(HEADERS[0]))
                sdk_row = next(c for c in report["kernel_consumers"]
                               if tuple(c["identity"][k] for k in ("source", "package", "selection")) == key)
                self.assertTrue(sdk_row["current_evidence"])
                self.assertEqual({v["status"] for v in sdk_row["variants"]}, {"current", "stale", "historical"})
                self.assertTrue(any("stale compiler" in r["reason"] for r in sdk_row["reasons"]))
                self.assertTrue(any(r["kind"] == "previous membership" for r in sdk_row["reasons"]))

    def test_explicit_symlink_target_roots_are_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            aliases = [Path(temporary) / name for name in ("core", "sdk")]
            for alias, target in zip(aliases, (self.core, self.sdk)):
                alias.symlink_to(target.resolve(), target_is_directory=True)
            index = kernel_scope.Index(self.metadata, self.graph, self.catalog, aliases)
            found = self.current(index)
            baseline = self.current(self.index)
            self.assertEqual({k: sorted(v["path"] for v in values) for k, values in found.items()},
                             {k: sorted(v["path"] for v in values) for k, values in baseline.items()})


if __name__ == "__main__":
    unittest.main()

"""Required evidence cannot be removed by weakening declaration predicates."""
import unittest

import kinds


def rule(path, value=True, op="eq"):
    return {"path": path, "op": op, "value": value, "source": "test"}


def table_mapping(rss_kib, size_mib=193365, **changed):
    """An exempt mapping as a soak sample lists it: a read-only shared view
    of `size_mib` MiB holding `rss_kib` kB."""
    start = 0x7428b2600000
    return {"pid": 101, "start_time": 5001, "range": f"{start:x}-{start + size_mib * 2 ** 20:x}", "device": "103:01",
            "inode": 11, "offset": "00000000", "perms": "r--s", "size_kib": size_mib * 1024, "rss_kib": rss_kib,
            **changed}


def file_identities(files):
    """The declared files as a soak receipt names them; the first holds the
    table_mapping inode."""
    return [{"file": name, "path": f"/checkpoint/{name}", "device": "103:01", "inode": 11 + index}
            for index, name in enumerate(files)]


def soak_receipt():
    return {"passed": True, "completed": 10, "duration_seconds": 1380,
            "measured_duration_seconds": 1380, "concurrency": 32,
            "memory_scope": "serving_process_tree", "memory_growth_limit_mib": 512,
            "host_memory_growth_limit_mib": 512,
            "memory_samples": [{"elapsed_seconds": t, "mib": 100, "host_rss_mib": 200}
                               for t in range(0, 1381, 30)],
            "mixed_requests": {"media": 2, "structured": 2, "structured_truncated": 0},
            "prefix_mix": {"canary_passed": True, "hits_total": 1, "idle_blocks_used": 0,
                           "idle_active_sequences": 0}}


class CoverageContracts(unittest.TestCase):
    def score(self, module, value, check):
        return kinds.rescore(kinds.of(module), value, None, {"module": module, **check})["status"]

    def test_empty_generic_evidence_cannot_pass_a_zero_count(self):
        self.assertNotEqual(self.score("unknown", {"records": {}},
                                       {"rules": [rule(["records"], 0, "count_ge")]}), "pass")

    def test_a_passed_boolean_is_not_evidence(self):
        self.assertNotEqual(self.score("unknown", {"passed": True},
                                       {"rules": [rule(["passed"])]}), "pass")

    def test_soak_requires_full_duration_memory_and_mixes(self):
        check = {"args": ["--duration", "1380", "--concurrency", "32", "--prefix-mix",
                          "--media-mix", "--structured-mix"], "rules": [rule(["passed"])]}
        self.assertEqual(self.score("soak", soak_receipt(), check), "pass")
        changes = [{"duration_seconds": 600, "measured_duration_seconds": 600},
                   {"memory_samples": []}, {"memory_scope": "not_sampled_remote"},
                   {"concurrency": 1}, {"completed": 0},
                   {"mixed_requests": {"media": 0, "structured": 2, "structured_truncated": 0}}]
        for change in changes:
            with self.subTest(change=change):
                self.assertNotEqual(self.score("soak", {**soak_receipt(), **change}, check), "pass")
        for field in ("mib", "host_rss_mib"):
            value = soak_receipt()
            value["memory_samples"][-1][field] += 513
            self.assertNotEqual(self.score("soak", value, check), "pass")
        value = soak_receipt()
        value["memory_samples"] = value["memory_samples"][:2]
        self.assertNotEqual(self.score("soak", value, check), "pass")

    def test_verify_refuses_self_comparison_and_empty_or_partial_coverage(self):
        check = {"rules": [rule(["records"], 1, "count_ge")]}
        record = {"arm": "spec", "k": 3, "speculation": {"draft_tokens": 1, "verify_items": 1},
                  "coverage": {"rule": "every forced decode position", "expected": 1, "scored": 1,
                               "missing": [], "unexpected": [], "passed": True},
                  "exact": {"all": {"gate_against_ar": {"passed": True}}}}
        value = {"ar_k": 0, "ar_arm": "ar", "ar_speculation": {}, "records": {"candidate": record}}
        self.assertEqual(self.score("verify_exact", value, check), "pass")
        for change in ({"k": 0}, {"arm": "ar"}, {"speculation": {}},
                       {"coverage": {**record["coverage"], "expected": 0, "scored": 0}},
                       {"coverage": {**record["coverage"], "missing": [1]}}):
            with self.subTest(change=change):
                modified = {**value, "records": {"candidate": {**record, **change}}}
                self.assertNotEqual(self.score("verify_exact", modified, check), "pass")

    def test_declared_capacities_gate_rescore_even_when_recorded_verdicts_claim_pass(self):
        from test_verify_exact import RowCapacityTests
        check = {"args": ["--row-capacity", "8", "--ar-row-capacity", "8"],
                 "rules": [rule(["records"], 1, "count_ge")]}
        record = {"arm": "spec", "k": 3, "speculation": RowCapacityTests.observed(4),
                  "coverage": {"rule": "every forced decode position", "expected": 1, "scored": 1,
                               "missing": [], "unexpected": [], "passed": True},
                  "row_capacity_coverage": {"passed": True},
                  "exact": {"all": {"gate_against_ar": {"passed": True}}}}
        value = {"ar_k": 0, "ar_arm": "ar", "ar_speculation": RowCapacityTests.observed(),
                 "records": {"candidate": record}}
        self.assertEqual(self.score("verify_exact", value, check), "pass")
        for field, changed in (("ar_speculation", RowCapacityTests.observed(capacity=16)),
                               ("records", {"candidate": {**record, "speculation": {
                                   "draft_tokens": 1, "verify_items": 1}}})):
            self.assertNotEqual(self.score("verify_exact", {**value, field: changed}, check), "pass")

    def test_existing_numerical_records_require_observed_shapes_without_a_capacity_matrix(self):
        from test_verify_exact import RowCapacityTests
        check = {"args": ["--require-row-shapes"], "rules": [rule(["records"], 1, "count_ge")]}
        record = {"arm": "spec", "k": 3, "speculation": RowCapacityTests.observed(4, capacity=4),
                  "coverage": {"rule": "every forced decode position", "expected": 1, "scored": 1,
                               "missing": [], "unexpected": [], "passed": True},
                  "exact": {"all": {"gate_against_ar": {"passed": True}}}}
        value = {"ar_k": 0, "ar_arm": "ar", "ar_speculation": RowCapacityTests.observed(capacity=1),
                 "records": {"candidate": record}}
        self.assertEqual(self.score("verify_exact", value, check), "pass")
        self.assertNotEqual(self.score("verify_exact", {**value, "ar_speculation": {}}, check), "pass")
        missing = {**record, "speculation": {"draft_tokens": 1, "verify_items": 1}}
        self.assertNotEqual(self.score("verify_exact", {**value, "records": {"candidate": missing}}, check), "pass")

    def test_reference_inventory_cannot_relabel_an_image_as_audio(self):
        import declaration
        with self.assertRaisesRegex(ValueError, "declared modality"):
            declaration.check_media_evidence({"coverage": {"audio": ["probe"]}},
                                             [{"id": "probe", "kind": "image"}])

    def test_weak_predicates_cannot_override_the_soak_growth_verdict(self):
        value = soak_receipt()
        value["memory_samples"][-1]["mib"] += 513
        check = {"args": ["--duration", "1380"], "rules": [rule(["completed"], 1, "ge")]}
        self.assertNotEqual(self.score("soak", value, check), "pass")

    def test_unserved_media_does_not_waive_memory_or_served_mix_requirements(self):
        import common
        import declaration
        from qual_test_config import configure
        configure("qwen38")
        config = declaration.load(declaration.path("qwen38"))
        check = {"module": "soak", "args": ["--duration", "1380", "--media-mix", "--media-cases",
                 "image-blue-square.png", "--structured-mix"], "rules": [rule(["passed"])]}
        value = soak_receipt()
        value.update(schema="chopinfer-qualification/v1", server={"health": {"modalities": ["text"],
            "serving": {"checkpoint_identity": config["qualification"]["checkpoint"]["checkpoint_identity"],
                        "startup_nonce": "a" * 32}}})
        value["not_applicable"] = [common.not_applicable(value["server"], "image")]
        value["mixed_requests"]["media"] = 0
        with kinds.policy(declaration.path("qwen38"), check):
            self.assertEqual(self.score("soak", value, check), "not_applicable")
            import json
            import tempfile
            import time
            from pathlib import Path
            from unittest.mock import patch
            import release_gate
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "checks").mkdir()
                policy = root / "policy.json"
                policy.write_text(json.dumps(config))
                selected = {**check, "id": "soak", "receipt": "*-soak.json", "timeout_seconds": 1700}
                def capture(*args):
                    (root / "checks" / "test-soak.json").write_text(json.dumps(value))
                    return 0
                context = {"url": "unused"}
                with patch.object(kinds.of("soak"), "capture", side_effect=capture):
                    row = release_gate.run_check(selected, [], "full", context, root, policy,
                                                 time.monotonic() + 1700, False, None)
                self.assertEqual((row["status"], row["passed"], row["evaluated"]),
                                 ("not_applicable", None, False))
                self.assertNotIn("soak", context)
            value["memory_samples"][-1]["host_rss_mib"] += 513
            self.assertEqual(self.score("soak", value, check), "fail")
            value["memory_samples"][-1]["host_rss_mib"] -= 513
            value["mixed_requests"]["structured"] = 0
            self.assertNotEqual(self.score("soak", value, check), "not_applicable")

    def test_prefix_unserved_image_is_not_a_pass_or_an_omission(self):
        import common
        import declaration
        from qual_test_config import configure
        configure("qwen38")
        config = declaration.load(declaration.path("qwen38"))
        check = {"module": "prefix_exact", "rules": [rule(["passed"])]}
        value = {"schema": "chopinfer-qualification/v1", "passed": None, "skipped": [],
                 "scores": {"all": {"positions": 2}}, "checks": {"cache": True, "hit_equals_cold": True,
                  "chunked_equals_cold": True, "hit_vs_cold": True, "media": None},
                 "server": {"health": {"modalities": ["text"], "serving": {
                  "checkpoint_identity": config["qualification"]["checkpoint"]["checkpoint_identity"],
                  "startup_nonce": "a" * 32}}}}
        value["not_applicable"] = [common.not_applicable(value["server"], "image")]
        with kinds.policy(declaration.path("qwen38"), check):
            self.assertEqual(self.score("prefix_exact", value, check), "not_applicable")
            value["checks"]["hit_equals_cold"] = False
            self.assertNotEqual(self.score("prefix_exact", value, check), "not_applicable")
            value["checks"]["hit_equals_cold"] = True
            value["checks"]["media"] = True
            self.assertNotEqual(self.score("prefix_exact", value, check), "not_applicable")

    def test_cohort_selects_soak_duration_without_trusting_receipt(self):
        check = {"args": ["--concurrency", "32"], "profile_args": {
            "quick": ["--duration", "60"], "full": ["--duration", "1380"]},
            "rules": [rule(["passed"])]}
        value = soak_receipt()
        value.update(duration_seconds=60, measured_duration_seconds=60)
        value["memory_samples"] = value["memory_samples"][:3]
        self.assertNotEqual(self.score("soak", value, check), "pass")
        self.assertEqual(self.score("soak", value, {**check, "_cohort": "quick"}), "pass")



class DeclaredSoakWarmup(unittest.TestCase):
    """A model that fills a bounded cache during warm-up declares the warm-up
    and the cache's bound (soak.memory_rule); the rule still fails a steady
    leak and any growth beyond the bound."""
    DECLARED = {"soak": {"WARMUP_SECONDS": {"value": 180, "source": "test"},
                         "CACHE_ALLOWANCE_MIB": {"value": {"host": 2560, "device": 1344}, "source": "test"}}}

    def receipt(self, host, device=None, declared=True):
        value = soak_receipt()
        value["memory_samples"] = [{"elapsed_seconds": t, "mib": 100 + (device or (lambda t: 0))(t),
                                    "host_rss_mib": 200 + host(t)} for t in range(0, 1381, 30)]
        if declared:
            value.update(warmup_seconds=180, cache_allowance_mib={"host": 2560, "device": 1344})
        return value

    def score(self, value, declared=True):
        check = {"args": ["--duration", "1380", "--concurrency", "32", "--prefix-mix", "--media-mix",
                          "--structured-mix"], "rules": [rule(["passed"])],
                 **({"constants": self.DECLARED} if declared else {})}
        return CoverageContracts.score(self, "soak", value, check)

    def test_a_bounded_warm_up_passes_only_when_declared(self):
        # The 2026-10-11 final gate's soak shape: about 1 GiB of pinned graphs in two minutes, then flat.
        warm = lambda t: min(t, 120) * 1000 / 120
        self.assertEqual(self.score(self.receipt(warm)), "pass")
        self.assertNotEqual(self.score(self.receipt(warm, declared=False), declared=False), "pass")

    def test_a_steady_leak_after_the_warm_up_fails(self):
        leak = lambda t: max(0, t - 180) * 600 / 1200
        self.assertNotEqual(self.score(self.receipt(leak)), "pass")
        self.assertNotEqual(self.score(self.receipt(lambda t: 0, device=leak)), "pass")

    def test_growth_beyond_the_declared_bound_fails(self):
        self.assertNotEqual(self.score(self.receipt(lambda t: min(t, 120) * 3100 / 120)), "pass")
        self.assertNotEqual(self.score(self.receipt(lambda t: 0, device=lambda t: min(t, 120) * 1900 / 120)), "pass")

    def test_a_receipt_judged_under_another_rule_is_refused(self):
        value = self.receipt(lambda t: 0)
        value["cache_allowance_mib"] = {"host": 9999, "device": 1344}
        self.assertNotEqual(self.score(value), "pass")
        self.assertNotEqual(self.score(self.receipt(lambda t: 0), declared=False), "pass")


class DeclaredMappedViews(unittest.TestCase):
    """A model whose server maps large read-only checkpoint files declares
    the soak's HOST_MEMORY_SCOPE rss_less_mapped_views and the files: the
    rescore recomputes the judged host memory, VmRSS less the exempt
    mappings' Rss, from each sample, and refuses a receipt judged under
    another scope or other files, or a sample without consistent values."""
    FILES = ["tables/table-a.bin", "tables/table-b.bin"]

    def receipt(self, exempt=lambda t: t * 4, other=lambda t: 0, scope="rss_less_mapped_views", files=None):
        value = soak_receipt()
        value["memory_samples"] = []
        for t in range(0, 1381, 30):
            rss = 25000 + exempt(t) + other(t)
            value["memory_samples"].append({"elapsed_seconds": t, "mib": 100, "host_rss_mib": rss,
                                            "host_exempt_rss_mib": 12000 + exempt(t),
                                            "host_exempt_bound_mib": 193365,
                                            "host_exempt_mappings": [table_mapping((12000 + exempt(t)) * 1024)],
                                            "host_judged_mib": rss - 12000 - exempt(t)})
        if scope is not None:
            value["host_memory_scope"] = scope
        if scope == "rss_less_mapped_views" or files is not None:
            value["mapped_view_files"] = self.FILES if files is None else files
            value["mapped_view_file_identities"] = file_identities(value["mapped_view_files"])
        return value

    def rescore(self, value, scope="rss_less_mapped_views", files=None):
        constants = {}
        if scope is not None:
            constants["HOST_MEMORY_SCOPE"] = {"value": scope, "source": "test"}
        if scope == "rss_less_mapped_views" or files is not None:
            constants["MAPPED_VIEW_FILES"] = {"value": self.FILES if files is None else files, "source": "test"}
        check = {"module": "soak", "args": ["--duration", "1380", "--concurrency", "32", "--prefix-mix",
                                            "--media-mix", "--structured-mix"], "rules": [rule(["passed"])],
                 **({"constants": {"soak": constants}} if constants else {})}
        return kinds.rescore(kinds.of("soak"), value, None, check)

    def test_exempt_growth_passes_only_under_the_scope(self):
        result = self.rescore(self.receipt())
        self.assertEqual(result["status"], "pass")
        self.assertEqual({key: result["metrics"][key] for key in (
            "host_memory_scope", "mapped_view_files", "host_memory_growth_mib", "host_exempt_rss_mib",
            "host_exempt_bound_mib")},
            {"host_memory_scope": "rss_less_mapped_views", "mapped_view_files": self.FILES,
             "host_memory_growth_mib": 0, "host_exempt_rss_mib": 12000 + 5520, "host_exempt_bound_mib": 193365})
        # Judged by VmRSS the same growth fails.
        self.assertEqual(self.rescore(self.receipt(scope=None), scope=None)["status"], "fail")

    def test_growth_outside_the_exempt_mappings_fails(self):
        result = self.rescore(self.receipt(other=lambda t: t // 2))
        self.assertEqual((result["status"], result["metrics"]["host_memory_growth_mib"]), ("fail", 690))
        # The judged series is recomputed from VmRSS and the exempt Rss, not
        # read from the receipt's summary.
        value = self.receipt(other=lambda t: t // 2)
        value.update(host_memory_growth_mib=0, steady_host_memory_growth_mib=0, passed=True)
        self.assertEqual(self.rescore(value)["status"], "fail")

    def test_a_receipt_judged_under_another_scope_or_other_files_is_refused(self):
        for value, scope, files in ((self.receipt(scope=None), "rss_less_mapped_views", None),
                                    (self.receipt(scope="rss"), "rss_less_mapped_views", None),
                                    (self.receipt(scope="anonymous_shmem"), "rss_less_mapped_views", None),
                                    (self.receipt(files=self.FILES[:1]), "rss_less_mapped_views", None),
                                    (self.receipt(), "rss_less_mapped_views", self.FILES[::-1]),
                                    (self.receipt(exempt=lambda t: 0), None, None),
                                    (self.receipt(exempt=lambda t: 0), "rss", None)):
            with self.subTest(recorded=(value.get("host_memory_scope"), value.get("mapped_view_files")),
                              declared=(scope, files)):
                result = self.rescore(value, scope, files)
                self.assertEqual(result["status"], "insufficient")
                self.assertIn("another host memory scope or other mapped-view files", result["error"])

    def test_a_sample_without_consistent_values_is_insufficient(self):
        for key in ("host_exempt_rss_mib", "host_exempt_bound_mib", "host_judged_mib"):
            for missing in (None, float("nan"), -1, "absent"):
                value = self.receipt()
                if missing == "absent":
                    del value["memory_samples"][5][key]
                else:
                    value["memory_samples"][5][key] = missing
                with self.subTest(key=key, value=missing):
                    result = self.rescore(value)
                    self.assertEqual(result["status"], "insufficient")
                    self.assertIn("incomplete or inconsistent", result["error"])
        for key, change in (("host_judged_mib", -600), ("host_exempt_rss_mib", 200000),
                            ("host_exempt_bound_mib", -190000)):
            value = self.receipt()
            value["memory_samples"][5][key] += change
            with self.subTest(key=key, change=change):
                result = self.rescore(value)
                self.assertEqual(result["status"], "insufficient")
                self.assertIn("incomplete or inconsistent", result["error"])

    def test_exempt_bytes_need_the_mappings_each_sample_lists(self):
        # VmRSS 100, 400 and 700 MiB, 0, 300 and 600 MiB of it claimed
        # exempt within a 1024 MiB bound: judged flat.
        check = {"module": "soak", "args": ["--duration", "60", "--concurrency", "1"],
                 "constants": {"soak": {"HOST_MEMORY_SCOPE": {"value": "rss_less_mapped_views", "source": "test"},
                                        "MAPPED_VIEW_FILES": {"value": ["table.bin"], "source": "test"}}},
                 "rules": [rule(["passed"])]}

        def receipt(listed):
            return {"passed": True, "errors": [], "completed": 3, "duration_seconds": 60,
                    "measured_duration_seconds": 60, "concurrency": 1, "memory_scope": "serving_process_tree",
                    "host_memory_scope": "rss_less_mapped_views", "mapped_view_files": ["table.bin"],
                    "mapped_view_file_identities": file_identities(["table.bin"]),
                    "memory_samples": [{"elapsed_seconds": t, "mib": 100, "host_rss_mib": 100 + 10 * t,
                                        "host_exempt_rss_mib": 10 * t, "host_exempt_bound_mib": 1024,
                                        "host_judged_mib": 100, **listed(10 * t * 1024)} for t in (0, 30, 60)]}

        def score(listed):
            return kinds.rescore(kinds.of("soak"), receipt(listed), None, check)

        self.assertEqual(score(lambda kib: {"host_exempt_mappings": [table_mapping(kib, 1024)]})["status"], "pass")
        for label, listed in (
                ("no list", lambda kib: {}),
                ("an empty list", lambda kib: {"host_exempt_mappings": []}),
                ("a 1 MiB mapping", lambda kib: {"host_exempt_mappings": [table_mapping(min(kib, 1024), 1)]}),
                ("a writable device mapping", lambda kib: {"host_exempt_mappings": [
                    table_mapping(kib, 1024, perms="rw-s", device="00:05")]}),
                ("Rss over Size", lambda kib: {"host_exempt_mappings": [table_mapping(kib, 1024, size_kib=1)]}),
                ("another process later", lambda kib: {"host_exempt_mappings": [
                    table_mapping(kib, 1024, start_time=5001 if kib == 0 else 9001)]})):
            with self.subTest(label):
                result = score(listed)
                self.assertEqual(result["status"], "insufficient")
                self.assertIn("incomplete or inconsistent mapped-view memory evidence", result["error"])

    def test_exempt_mappings_must_be_of_the_declared_files(self):
        self.assertEqual(self.rescore(self.receipt())["status"], "pass")
        identities = file_identities(self.FILES)
        for label, change, message in (
                ("no identities", lambda value: value.pop("mapped_view_file_identities"), "names no device and inode"),
                ("another file's", lambda value: value.update(mapped_view_file_identities=identities[1:]),
                 "names no device and inode"),
                ("a malformed device", lambda value: value["mapped_view_file_identities"][0].update(device="0x67"),
                 "names no device and inode"),
                ("a mapping of another inode",
                 lambda value: [row["host_exempt_mappings"][0].update(inode=99) for row in value["memory_samples"]],
                 "at 0 s exempts a mapping of no declared file"),
                ("a mapping of another device",
                 lambda value: [row["host_exempt_mappings"][0].update(device="00:05")
                                for row in value["memory_samples"]],
                 "at 0 s exempts a mapping of no declared file")):
            value = self.receipt()
            change(value)
            with self.subTest(label):
                result = self.rescore(value)
                self.assertEqual(result["status"], "insufficient")
                self.assertIn(message, result["error"])
        # Device numbers compare as numbers, however smaps pads them.
        value = self.receipt()
        value["mapped_view_file_identities"][0]["device"] = "103:1"
        self.assertEqual(self.rescore(value)["status"], "pass")

    def test_invalid_declared_files_are_refused(self):
        for scope, files in (("rss_less_mapped_views", []), ("rss_less_mapped_views", ["/abs.bin"]),
                             ("rss_less_mapped_views", ["../up.bin"]), ("rss", self.FILES)):
            with self.subTest(scope=scope, files=files):
                result = self.rescore(self.receipt(scope=scope, files=files), scope, files)
                self.assertEqual(result["status"], "insufficient")
                self.assertIn("mapped-view files", result["error"])

    def test_an_unknown_declared_scope_is_refused(self):
        for scope in ("vmrss", "", 1, "anonymous_shmem"):
            with self.subTest(scope=scope):
                result = self.rescore(self.receipt(scope=scope), scope)
                self.assertEqual(result["status"], "insufficient")
                self.assertIn("host memory scope", result["error"])


if __name__ == "__main__":
    unittest.main()

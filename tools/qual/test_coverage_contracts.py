"""Required evidence cannot be removed by weakening declaration predicates."""
import unittest

import kinds


def rule(path, value=True, op="eq"):
    return {"path": path, "op": op, "value": value, "source": "test"}


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


if __name__ == "__main__":
    unittest.main()


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

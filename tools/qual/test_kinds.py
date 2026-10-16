"""The check-kind registry: every declared check has a kind, the kinds'
metrics hold what the declarations gate on, and a kind's score of a receipt
reproduces the metrics its module recorded."""

from qual_test_config import configure

import copy
import json
import unittest

import common
import declaration
import kinds
import release_gate
import structured
import teacher
from vectors import served_rows


def setUpModule():
    configure("qwen38")


# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "check-kind registry and rescoring; kinds.policy applies and restores each check's config"


def teacher_check(model):
    config = declaration.load(declaration.path(model))
    return config, next(check for check in config["checks"] if check["module"] == "teacher")


def teacher_receipt(config, texts, capture):
    """A teacher receipt as eval records a live run of `capture`
    against the pinned reference and checkpoint of `config`."""
    qualification = config["qualification"]
    # One row per text, with the served rows the capture's top-1 tokens come from.
    rows = [{"name": text["name"], "prompt_tokens": len(text["prompt_token_ids"]), "scored": len(served["top1"]),
             "latency_seconds": 0.1, **served_rows(text["prompt_token_ids"][text["score_from"]:], served["logprob"],
                                                   served["top1"])} for text, served in zip(texts, capture)]
    result = {"errors": [], "rows": rows, "skipped": [],
              "reference_sha256": qualification["references"]["teacher"]["sha256"]}
    _, passed = teacher.evaluate(capture, texts, result, variants=qualification["references"]["teacher"]["variants"])
    server = {"checkpoint": {"checkpoint_identity": qualification["checkpoint"]["checkpoint_identity"]}}
    return {"schema": "chopinfer-qualification/v1", "check": "teacher", "server": server,
            "mechanics_only": False, "passed": passed, **result}


class RegistryTests(unittest.TestCase):
    def test_every_declared_check_has_a_kind_that_names_its_gated_fields(self):
        for path in declaration.shipped():
            for check in declaration.load(path)["checks"]:
                with self.subTest(gate=path.parent.name, check=check["id"]):
                    kind = kinds.of(check["module"])
                    self.assertIn(kind.name, kinds.KINDS)
                    if kind is not kinds.RECEIPT:
                        # A documented kind's metrics cover every predicate its declarations hold.
                        heads = {rule["path"][0] for rule in check["rules"]}
                        self.assertLessEqual(heads, set(kind.metrics), check["rules"])

    def test_the_structured_case_set_is_the_modules(self):
        import argparse
        self.assertEqual(tuple(structured.cases(argparse.Namespace(reasoning_max_tokens=1))),
                         kinds.STRUCTURED_CASES)

    def test_the_shared_kinds_are_the_eval_contract(self):
        shared = [row["kind"] for row in kinds.describe() if row["shared"]]
        self.assertEqual(shared, ["teacher_forced", "media_forced", "greedy_pin", "determinism", "api", "structured"])
        for row in kinds.describe():
            self.assertTrue(all(row[key] for key in ("inputs", "receipt", "metrics", "verdict")), row)

    def test_a_module_without_a_kind_is_judged_by_its_predicates(self):
        check = {"rules": [{"path": ["rows", "*", "pass"], "op": "true_count_ge", "value": 2, "source": "x"},
                           {"path": ["first"], "op": "eq", "value": None, "source": "x"}]}
        kind = kinds.of("no_such_module")
        self.assertIs(kind, kinds.RECEIPT)
        value = {"first": None, "rows": [{"pass": True}, {"pass": True}], "other": 1}
        metrics = kind.score(value, check=check)
        self.assertEqual(metrics, {"first": None, "rows": value["rows"]})
        self.assertEqual(kind.judge(metrics, check)[0], "pass")
        self.assertEqual(kind.judge(dict(metrics, first=1), check)[0], "fail")

    def test_a_declared_waiver_turns_a_covered_failure_into_waived(self):
        rule = {"path": ["first"], "op": "le", "value": 0, "source": "test"}
        waiver = {"predicate": {"path": ["first"], "op": "le"}, "ceiling": 1, "reason": "a reason",
                  "decision": "operator, 2026-10-15"}
        check = {"module": "tapdiff", "rules": [rule], "waivers": [waiver]}
        kind = kinds.of("tapdiff")
        status, predicates = kind.judge({"first": 1}, check)
        self.assertEqual(status, "waived")
        self.assertEqual(predicates[0]["waiver"]["ceiling"], 1)
        self.assertEqual(kind.judge({"first": 2}, check)[0], "fail")
        self.assertEqual(kind.judge({"first": 0}, check)[0], "pass")
        self.assertEqual(kind.judge({"first": 1}, {"module": "tapdiff", "rules": [rule]})[0], "fail")
        # An empty list is refused like any invalid waiver (the one validator).
        with self.assertRaisesRegex(ValueError, "nonempty list of waivers"):
            kind.judge({"first": 1}, {**check, "waivers": []})
        # A teacher category waiver, applied by eval's judge.
        spread = lambda total: {"mean_abs_dlogprob": 0.1, "top1_decisive": 0.03, "top1_near_tie": 0.07,
                                "top1_disagreement": total}
        frozen = {**spread(0.1135), "limits": spread(0.1), "passed_by": {"top1_decisive": "limit",
                  "top1_near_tie": "limit", "top1_disagreement": None}}
        metrics = {**spread(0.0619), "limits": {**spread(0.1), "mean_abs_dlogprob": 0.15}, "passed": False,
                   "by_category": {"prose-frozen": frozen}, "failed_categories": ["prose-frozen"]}
        rules = [{"path": ["passed"], "op": "eq", "value": True, "source": "test"},
                 {"path": ["failed_categories"], "op": "eq", "value": [], "source": "test"}]
        category = {"category": "prose-frozen", "metric": "top1_disagreement", "ceiling": 0.1135,
                    "reason": "a reason", "decision": "operator, 2026-10-15"}
        teacher_kind = kinds.of("teacher")
        status, predicates = teacher_kind.judge(metrics, {"module": "teacher", "rules": rules, "waivers": [category]})
        self.assertEqual(status, "waived")
        self.assertEqual(predicates[1]["waiver"]["categories"][0]["measured"], 0.1135)
        self.assertEqual(teacher_kind.judge(metrics, {"module": "teacher", "rules": rules})[0], "fail")

    def test_metrics_judge_exactly_as_the_receipt(self):
        # Every predicate reads the same values from the metrics as from the receipt,
        # including whole-receipt and wildcard paths. (api and structured
        # recompute `passed`, so their metrics gain it.)
        value = {"rows": [{"name": "a", "status": "passed", "score": 0.1}], "summary": {"overall": {}}, "x": [3]}
        rules = [{"path": ["rows", "*", "score"], "op": "le", "value": 0.15, "source": "x"},
                 {"path": ["*"], "op": "count_ge", "value": 1, "source": "x"},
                 {"path": [], "op": "count_ge", "value": 3, "source": "x"}]
        for rule in rules:
            for module in ("teacher", "no_such_module"):
                with self.subTest(rule=rule, module=module):
                    kind = kinds.of(module)
                    check = {"rules": [rule]}
                    metrics = kind.score(value, check=check)
                    self.assertEqual(kind.judge(metrics, check)[1][0]["passed"], release_gate.judge(value, rule))

    def test_receipts_that_cannot_be_judged_are_refused_as_the_gate_refused_them(self):
        kind = kinds.of("api")
        for value, message in (([], "expected a JSON object receipt"),
                               ({"passed": True, "skipped": ["x"]}, "mechanics-only or skipped"),
                               ({"passed": True, "mechanics_only": True}, "mechanics-only or skipped"),
                               ({"passed": True, "errors": [{"name": "x"}]}, "check reported errors")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                kind.score(value, check={"rules": []})
        self.assertEqual(kind.score({"passed": True, "mechanics_only": True}, check={"rules": []}, mechanics=True),
                         {"passed": True})


class TeacherForcedTests(unittest.TestCase):
    """The teacher kind recomputes limits, categories and the verdict from
    a receipt's stored scores and the pinned reference."""

    @classmethod
    def setUpClass(cls):
        cls.config, cls.check = teacher_check("qwen38")
        path = common.input_path("teacher_reference", config=cls.config)
        cls.texts = json.loads(path.read_text())["texts"]
        pin = cls.config["qualification"]["references"]["teacher"]
        cls.reference = {"sha256": pin["sha256"], "gate": "exact", "variants": pin["variants"], "texts": cls.texts}
        cls.kind = kinds.of("teacher")

    def test_rescoring_reproduces_the_recorded_metrics(self):
        for variant in ("bf16", "exact"):
            with self.subTest(variant):
                value = teacher_receipt(self.config, self.texts, teacher.variant_scores(self.texts, variant))
                recorded = self.kind.score(value, check=self.check)
                self.assertEqual(self.kind.score(value, self.reference, self.check), recorded)
                self.assertEqual(recorded["limits"], {"mean_abs_dlogprob": 0.10, "top1_decisive": 0.08,
                                                      "top1_near_tie": 0.08, "top1_disagreement": 0.10})

    def test_rescoring_corrects_a_verdict_its_limits_do_not_support(self):
        value = teacher_receipt(self.config, self.texts, teacher.variant_scores(self.texts, "bf16"))
        tampered = copy.deepcopy(value)
        tampered["by_category"]["chat"].update(passed=False, limits={"mean_abs_dlogprob": 0.0,
                                                                     "top1_decisive": 0.0, "top1_near_tie": 0.0})
        tampered.update(failed_categories=["chat"], passed=False)
        self.assertEqual(self.kind.score(tampered, self.reference, self.check), self.kind.score(value, check=self.check))

    def test_a_failing_category_fails_on_rescore(self):
        worse = [{**text, "logprob": [value - 0.2 for value in text["logprob"]]} if text["category"] == "long"
                 else text for text in teacher.variant_scores(self.texts, "exact")]
        value = teacher_receipt(self.config, self.texts, worse)
        self.assertEqual(value["failed_categories"], ["long"])
        metrics = self.kind.score(value, self.reference, self.check)
        self.assertEqual(metrics["failed_categories"], ["long"])
        self.assertFalse(metrics["passed"])
        self.assertEqual(self.kind.judge(metrics, self.check)[0], "fail")

    def test_an_informational_receipt_keeps_its_verdict_as_a_diagnostic(self):
        value = teacher_receipt(self.config, self.texts, teacher.variant_scores(self.texts, "bf16"))
        value.update(diagnostic_passed=value["passed"], passed=False, informational_reason="context")
        metrics = self.kind.score(value, self.reference, self.check)
        self.assertEqual((metrics["passed"], metrics["diagnostic_passed"]), (False, True))

    def test_a_receipt_of_another_reference_is_insufficient(self):
        value = teacher_receipt(self.config, self.texts, teacher.variant_scores(self.texts, "bf16"))
        with self.assertRaises(kinds.Insufficient):
            self.kind.score(value, dict(self.reference, sha256="1" * 64), self.check)

    def test_the_reference_is_the_pinned_one(self):
        with kinds.policy(declaration.path("qwen38"), self.check):
            reference = self.kind.reference(self.check, self.config)
        self.assertEqual(reference["sha256"], self.config["qualification"]["references"]["teacher"]["sha256"])
        self.assertEqual([text["name"] for text in reference["texts"]], [text["name"] for text in self.texts])

    def test_the_policy_applies_and_restores_the_declared_constants(self):
        check = {"constants": {"teacher": {"LARGE_SET_MEAN": {"value": 0.2, "source": "x"}}}}
        before = teacher.LARGE_SET_MEAN
        with kinds.policy(declaration.path("qwen38"), check):
            self.assertEqual(teacher.LARGE_SET_MEAN, 0.2)
        self.assertEqual(teacher.LARGE_SET_MEAN, before)


class CoverageTests(unittest.TestCase):
    """Rescoring checks the receipt's coverage and counts before it trusts
    any stored score: missing coverage is insufficient evidence, not a pass."""

    @classmethod
    def setUpClass(cls):
        cls.config, cls.check = teacher_check("qwen38")
        path = common.input_path("teacher_reference", config=cls.config)
        cls.texts = json.loads(path.read_text())["texts"]
        cls.value = teacher_receipt(cls.config, cls.texts, teacher.variant_scores(cls.texts, "bf16"))

    def rescore(self, value, check=None, config=None):
        import checks
        return checks.rescore_one(value, check or self.check, config or self.config,
                                  declaration.path("qwen38"), False)

    def test_a_complete_receipt_rescores_to_pass(self):
        result = self.rescore(self.value)
        self.assertEqual(result["status"], "pass", result.get("error"))
        self.assertIn("mean_abs_dlogprob", result["recorded_fields"])

    def test_teacher_receipts_missing_coverage_are_insufficient(self):
        def one_text_less(value):
            value["rows"] = value["rows"][1:]

        changes = {
            "no limits": lambda value: value.pop("limits"),
            "no rows": lambda value: value.pop("rows"),
            "no categories": lambda value: value.pop("by_category"),
            "a category missing": lambda value: value["by_category"].pop("chat"),
            "scored inflated": lambda value: value.update(scored=100000),
            "a text missing": one_text_less,
            "a text miscounted": lambda value: value["rows"][0].update(scored=value["rows"][0]["scored"] + 1),
            "an inconsistent top-1 rate": lambda value: value.update(top1_disagreement=0.0),
        }
        for name, change in changes.items():
            with self.subTest(name):
                value = copy.deepcopy(self.value)
                change(value)
                self.assertEqual(self.rescore(value)["status"], "insufficient")

    def test_api_and_structured_receipts_need_their_cases(self):
        config = declaration.load(declaration.path("qwen38"))
        checks_by_id = {check["id"]: check for check in config["checks"]}
        bare = {"schema": "chopinfer-qualification/v1", "passed": True, "mechanics_only": False, "errors": [],
                "skipped": [], "summary": {"overall": {"validity_rate": 1.0, "refused": 0, "truncated": 0}}}
        for name in ("api", "structured"):
            with self.subTest(name):
                self.assertEqual(self.rescore(dict(bare, check=name), checks_by_id[name], config)["status"],
                                 "insufficient")

    def test_a_media_receipt_without_items_is_insufficient(self):
        config = declaration.load(declaration.path("mimo26"))
        check = next(check for check in config["checks"] if check["module"] == "multimodal")
        import checks
        value = {"schema": "chopinfer-qualification/v1", "check": "multimodal", "passed": True,
                 "mechanics_only": False, "errors": [], "skipped": [], "rows": [],
                 "exact_reference_sha256": config["qualification"]["references"]["media"]["sha256"],
                 "server": {"checkpoint": {"checkpoint_identity":
                                           config["qualification"]["checkpoint"]["checkpoint_identity"]}}}
        result = checks.rescore_one(value, check, config, declaration.path("mimo26"), False)
        self.assertEqual(result["status"], "insufficient", result.get("error"))


class MimoReferenceTests(unittest.TestCase):
    """MiMo's pinned teacher and media references, in whatever schema the
    checkout pins, score through the kinds: a receipt built from a spread
    variant rescores to pass, and missing coverage is insufficient."""

    @classmethod
    def setUpClass(cls):
        cls.config = declaration.load(declaration.path("mimo26"))
        cls.checks = {check["module"]: check for check in cls.config["checks"]}
        cls.identity = cls.config["qualification"]["checkpoint"]["checkpoint_identity"]

    def rescore(self, module, value):
        import checks
        return checks.rescore_one(value, self.checks[module], self.config, declaration.path("mimo26"), False)

    def test_the_teacher_reference_rescores(self):
        check = self.checks["teacher"]
        with kinds.policy(declaration.path("mimo26"), check):
            reference = kinds.of("teacher").reference(check, self.config)
            texts = reference["texts"]
            spread = next(variant for variant in reference["variants"] if variant != reference["gate"])
            value = teacher_receipt(self.config, texts, teacher.variant_scores(texts, spread))
        value["server"]["checkpoint"]["checkpoint_identity"] = self.identity
        result = self.rescore("teacher", value)
        self.assertEqual(result["status"], "pass", result.get("error"))
        self.assertEqual(result["metrics"], result["recorded_metrics"])
        value["rows"].pop()
        self.assertEqual(self.rescore("teacher", value)["status"], "insufficient")

    def media_receipt(self):
        import limits as rules
        import multimodal
        check = self.checks["multimodal"]
        with kinds.policy(declaration.path("mimo26"), check):
            reference = kinds.of("multimodal").reference(check, self.config)
        rows = []
        for case in reference["cases"]:
            gate = case["exact"][reference["gate"]]
            served = case["exact"][next(name for name in case["exact"] if name != reference["gate"])]
            differences = [abs(a - b) for a, b in zip(served["logprob"], gate["logprob"])]
            flips = multimodal.gate_flips(gate, served["top1"])
            scores = {"scored": len(differences), "mean_abs_dlogprob": sum(differences) / len(differences),
                      "max_abs_dlogprob": max(differences), **rules.top1_summary(flips, len(differences)),
                      "top1_flips": [{"offset": offset, "served": served["top1"][offset], **flip}
                                     for offset, flip in enumerate(flips) if flip is not None],
                      **served_rows(case["answer_token_ids"], served["logprob"], served["top1"])}
            scores.update(limits={}, passed=True)
            rows.append({"id": case["id"], "kind": case["kind"], "logprobs": scores, "greedy": {"passed": True},
                         "fixture_expectation_match": True, "token_count_match": True, "passed": True})
        value = {"schema": "chopinfer-qualification/v1", "check": "multimodal", "passed": True,
                 "mechanics_only": False, "errors": [], "skipped": [], "rows": rows,
                 "exact_reference_sha256": reference["sha256"],
                 "server": {"checkpoint": {"checkpoint_identity": self.identity}}}
        mixed = reference.get("mixed")
        if mixed:
            mixed_rows = []
            for case in mixed["cases"]:
                served = case["exact"]["bf16"]
                columns = {"logprob": served["logprob"], "top1": served["top1"],
                           **served_rows(case["answer_token_ids"], served["logprob"], served["top1"])}
                scores = kinds.call("score_capture", kinds.of("multimodal"), columns, case, check)["metrics"]
                mixed_rows.append({"id": case["id"], "kind": case["kind"], "capture": columns,
                    "prompt_token_ids": case["prompt_token_ids"] + case["answer_token_ids"],
                    "logprobs": scores, "prompt_token_ids_match": True, "passed": True})
            value["mixed"] = {"passed": True, "case_count": len(mixed_rows), "rows": mixed_rows,
                              "reference_sha256": mixed["sha256"]}
        return value

    def test_the_media_reference_rescores(self):
        import limits as rules
        value = self.media_receipt()
        result = self.rescore("multimodal", value)
        self.assertEqual(result["status"], "pass", result.get("error"))
        self.assertTrue(all(rules.within(row["logprobs"], row["logprobs"]["limits"])
                            for row in result["metrics"]["rows"]))
        worse = copy.deepcopy(value)
        worse["rows"][0]["logprobs"].update(mean_abs_dlogprob=5.0, max_abs_dlogprob=5.0)
        self.assertEqual(self.rescore("multimodal", worse)["status"], "fail")
        value["rows"].pop()
        self.assertEqual(self.rescore("multimodal", value)["status"], "insufficient")

    def test_mixed_receipt_requires_all_scores_and_prompt_positions(self):
        value = self.media_receipt()
        value["mixed"]["rows"][0]["prompt_token_ids"][0] += 1
        self.assertEqual(self.rescore("multimodal", value)["status"], "insufficient")
        value = self.media_receipt()
        value["mixed"]["rows"].pop()
        self.assertEqual(self.rescore("multimodal", value)["status"], "insufficient")
        value = self.media_receipt()
        value["mixed"]["rows"][0]["capture"]["logprob"][0] -= 10
        self.assertEqual(self.rescore("multimodal", value)["status"], "fail")

    def partial_media_receipt(self):
        import common
        import multimodal
        value = self.media_receipt()
        value["server"]["health"] = {"modalities": ["text", "image"], "serving": {
            "checkpoint_identity": self.identity, "startup_nonce": "a" * 32}}
        rows = []
        for row in value["rows"]:
            reason = common.not_applicable(value["server"], row["kind"])
            rows.append({"id": row["id"], "kind": row["kind"], "status": "not_applicable",
                         "reason": reason, "passed": None} if reason else row)
        for row in value.get("mixed", {}).get("rows", []):
            if reason := common.not_applicable(value["server"], row["kind"]):
                row.update(status="not_applicable", reason=reason, passed=None)
        return multimodal.applicable(value, rows)

    def test_partial_media_rescore_preserves_served_failures(self):
        import checks
        from pathlib import Path
        value = self.partial_media_receipt()
        def rescore(receipt):
            return checks.rescore_receipt(Path("standalone-multimodal.json"), receipt, "mimo26", "current")
        self.assertEqual(rescore(value)["status"], "not_applicable")
        failed = copy.deepcopy(value)
        row = next(row for row in failed["rows"] if row["kind"] == "image")
        row["passed"] = False
        failed["errors"] = [row]
        self.assertEqual(rescore(failed)["status"], "fail")
        # Do not trust the aggregate or the recorded row verdict, recompute scores.
        row = next(row for row in value["rows"] if row["kind"] == "image")
        row["logprobs"].update(mean_abs_dlogprob=5.0, max_abs_dlogprob=5.0)
        self.assertEqual(rescore(value)["status"], "fail")

    def test_unserved_image_probes_are_validated_without_passing(self):
        import common
        value = self.partial_media_receipt()
        value["server"]["health"]["modalities"] = ["text"]
        reason = common.not_applicable(value["server"], "image")
        value.update(rows=[{"id": "probe", "kind": "image", "status": "not_applicable",
                            "reason": reason, "passed": None}], not_applicable=[reason],
                     reference={"sha256": "reference"}, passed=False)
        reference = {"format": "image-probes", "sha256": "reference", "order": ["probe"],
                     "probes": {"probe": {"kind": "short"}}, "rule": {"near_tie": 0.3}}
        check = self.checks["multimodal"]
        with kinds.policy(declaration.path("mimo26"), check):
            result = kinds.rescore(kinds.of("multimodal"), value, reference, check)
            self.assertEqual(result["status"], "not_applicable")
            self.assertFalse(result["metrics"]["passed"])
            value["rows"] = []
            self.assertEqual(kinds.rescore(kinds.of("multimodal"), value, reference, check)["status"], "insufficient")

    def test_partial_media_rescore_requires_bound_complete_evidence(self):
        for change in [lambda v: v.pop("schema"), lambda v: v.pop("server"), lambda v: v.update(server="invalid"), lambda v: v["rows"].pop(),
                       lambda v: v.update(exact_reference_sha256="wrong"),
                       lambda v: v.update(not_applicable=["invented reason"]),
                       lambda v: v["server"]["health"].update(modalities=["text", "image", "audio", "video"])]:
            value = self.partial_media_receipt()
            change(value)
            with self.subTest(change=change):
                self.assertEqual(self.rescore("multimodal", value)["status"], "insufficient")



class StructuredTests(unittest.TestCase):
    def test_the_summary_is_recomputed_from_the_rows(self):
        rows = [{"name": "a", "status": "passed", "latency_seconds": 0.5, "tokens_per_second": 10.0},
                {"name": "a", "status": "truncated", "latency_seconds": 1.0},
                {"name": "b", "status": "failed", "latency_seconds": 2.0}]
        summary = {"overall": structured.summarize(rows),
                   "cases": {"a": structured.summarize(rows[:2]), "b": structured.summarize(rows[2:])}}
        value = {"passed": True, "rows": rows, "summary": {"overall": {"validity_rate": 1.0}}}
        self.assertEqual(kinds.of("structured").score(value, check={"rules": []})["summary"], summary)


class PairedKindTests(unittest.TestCase):
    def test_greedy_pin_rules(self):
        record = {"reference_sha256": "a", "prompt_tokens": 4, "max_tokens": 2, "concurrency": 1,
                  "requests": {"x": [1, 2]}, "finish_reasons": {"x": "length"}}
        kind = kinds.KINDS["greedy_pin"]
        same = kind.score(record, record)
        self.assertEqual(kind.judge(same, {"rule": "exact"})[0], "pass")
        other = dict(record, requests={"x": [1, 3]})
        differ = kind.score(other, record)
        self.assertEqual(differ["first_differences"], [{"request": "x", "position": 1, "ar_token": 2,
                                                        "other_token": 3}])
        self.assertEqual(kind.judge(differ, {"rule": "exact"})[0], "fail")
        self.assertEqual(kind.judge(differ, {"rule": "informational"})[0], "informational")
        with self.assertRaisesRegex(ValueError, "workload mismatch: max_tokens"):
            kind.score(dict(record, max_tokens=3), record)


if __name__ == "__main__":
    unittest.main()

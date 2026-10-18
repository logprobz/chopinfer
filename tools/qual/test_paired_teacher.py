"""Paired release gates reject regressions and incomparable captures."""

import copy
import contextlib
import io
import os
import sys
import json
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch
import checks
import declaration
import kinds
import paired_teacher as paired
import release_gate
import teacher
from vectors import scores, served_rows


class PairedTests(unittest.TestCase):
    def setUp(self):
        self.reference = {"sha256": "a" * 64, "gate": "exact", "variants": ["exact", "bf16"], "texts": []}
        for name, category in [(f"{prefix}-{index}", category) for prefix, category in
                               (("first", "coding"), ("second", "tools")) for index in range(4)]:
            ids = [1] + [7] * 40
            values = scores(ids[1:], [-1.0] * 40)
            self.reference["texts"].append({"name": name, "category": category, "prompt_token_ids": ids,
                                            "score_from": 1, "exact": values, "bf16": copy.deepcopy(values)})
        self.identity = {"checkpoint_identity": "b" * 64, "package_identity": "b" * 64, "metadata_identity": "b" * 64,
                         "version": "fixture-1", "git_commit": "c" * 40, "profile": {"tensor_parallel_size": 2}}
        self.accepted = self.capture(self.identity, 0.01)
        self.candidate = copy.deepcopy(self.accepted)
        self.policy = {"schema": "paired-teacher-margins/v1", "approval": "unit-test approval",
                       "rationale": "Synthetic boundary test", "reference_sha256": self.reference["sha256"],
                       "mean_margin": 0.005, "category_mean_margin": 0.01, "top1_delta": 0.01, "category_alpha": 0.05,
                       "min_clusters": 8, "allow_sparse_pooled": False, "tail_growth_nats": 0.5, "tail_net_allowance": 2}
        self.check = {"module": "paired_teacher", "rules": [{"path": ["passed"], "op": "eq", "value": True, "source": "test"}],
                      "constants": {"paired_teacher": {"POLICY": {"value": self.policy, "source": "test"}}}}
        self.reference["absolute_check"] = {"module": "teacher", "rules": copy.deepcopy(self.check["rules"])}

    def capture(self, identity, error, top1=None):
        errors = [error] * len(self.reference["texts"]) if isinstance(error, (int, float)) else error
        captured = [{"name": text["name"], "category": text["category"], "logprob": [-1 - value] * 40,
                     "top1": (top1 or {}).get(text["name"], [7] * 40)}
                    for text, value in zip(self.reference["texts"], errors, strict=True)]
        result = {"schema": "chopinfer-qualification/v1", "check": "teacher", "mechanics_only": False,
                  "at": "2002-01-01T00:00:00Z", "errors": [], "skipped": [], "reference_sha256": self.reference["sha256"], "rows": []}
        for text, capture in zip(self.reference["texts"], captured):
            result["rows"].append({"name": text["name"], "category": text["category"], "score_from": 1,
                                   "prompt_token_ids_sha256": teacher.ids_digest(text["prompt_token_ids"]),
                                   "prompt_tokens": 41, "scored": 40, "latency_seconds": 0.1,
                                   **served_rows([7] * 40, capture["logprob"], capture["top1"])})
        _, result["passed"] = teacher.evaluate(captured, self.reference["texts"], result, variants=["exact", "bf16"])
        result["server"] = {"health": {"serving": {key: identity[key] for key in
                                                   ("checkpoint_identity", "package_identity", "metadata_identity", "version")}}}
        result["server"]["health"]["serving"].update(source_commit=identity["git_commit"], startup_nonce="c" * 32,
                                                     profile=identity["profile"])
        return result

    def value(self):
        return {"schema": "chopinfer-qualification/v1", "policy_sha256": paired.digest(self.policy),
                "arms": {"accepted": self.accepted, "candidate": self.candidate},
                "qualified_server": copy.deepcopy(self.candidate["server"])}

    def run_gate(self):
        return kinds.rescore(kinds.KINDS["paired_teacher"], self.value(), self.reference, self.check)

    def test_margins_are_the_whole_policy(self):
        """A policy holds its approval, rationale, reference pin and margins;
        admission fields (identities, accepted pin, approval time) are not a
        policy any more."""
        self.assertEqual(paired.policy(self.check), self.policy)
        for change in ({"accepted_capture_sha256": "a" * 64}, {"status": "approved"}, {"approval": " "},
                       {"schema": "paired-teacher-policy/v3"}, {"min_clusters": 1}):
            check = copy.deepcopy(self.check)
            check["constants"]["paired_teacher"]["POLICY"]["value"].update(change)
            with self.subTest(change=change), self.assertRaises(kinds.Insufficient):
                paired.policy(check)

    def test_the_cli_compares_with_the_live_profile_s_margins(self):
        self.candidate["server"]["health"]["serving"]["profile"] = {
            "tensor_parallel_size": 2, "speculation": {"tokens": 0}}
        other = {**self.policy, "reference_sha256": "e" * 64}
        check = {**copy.deepcopy(self.check), "id": "paired_teacher"}
        check["constants"]["paired_teacher"] = {"PROFILE_POLICIES": {
            "value": {"tp2-ar": self.policy, "tp4-ar": other}, "source": "test"}}
        with tempfile.TemporaryDirectory() as temporary:
            accepted, candidate = (Path(temporary) / name for name in ("accepted.json", "candidate.json"))
            accepted.write_text(json.dumps(self.accepted))
            candidate.write_text(json.dumps(self.candidate))
            with patch.object(sys, "argv", ["paired_teacher.py", "--accepted", str(accepted), "--candidate",
                                            str(candidate), "--url", "http://127.0.0.1:1"]), patch.object(
                    paired.common, "qualification_config", return_value={"checks": [check]}), patch.object(
                    kinds.KINDS["paired_teacher"], "reference", return_value=self.reference), patch.object(
                    paired.common, "Client") as client, patch.object(paired.common, "bind_checkpoint"), \
                    patch.object(paired.common, "confirm_live"), patch.object(paired.common, "receipt") as receipt, \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exited:
                client.return_value.identity.return_value = copy.deepcopy(self.candidate["server"])
                paired.main()
        self.assertEqual(exited.exception.code, 0)
        recorded = receipt.call_args.args[2]
        self.assertEqual(recorded["policy_sha256"], paired.digest(self.policy))
        self.assertEqual(recorded["accepted_capture_sha256"], paired.digest(self.accepted))
        self.assertTrue(recorded["passed"])

    def test_a_context_parallel_profile_selects_its_own_key_when_declared(self):
        profile = {"tensor_parallel_size": 4, "speculation": {"tokens": 3, "drafter": "mtp"}}
        names = {"tp4-mtp", "tp4-ar", "tp4-mtp-cp2"}
        for size, expected in ((None, "tp4-mtp"), (1, "tp4-mtp"), (2, "tp4-mtp-cp2"), (4, "tp4-mtp")):
            with self.subTest(size=size):
                self.assertEqual(paired.policy_name({**profile, "context_parallel_size": size}, names), expected)
        self.assertEqual(paired.policy_name({**profile, "speculation": {"tokens": 0}, "context_parallel_size": 2},
                                            names), "tp4-ar")
        for size in (0, "2", 2.0, True):
            with self.subTest(size=size), self.assertRaisesRegex(kinds.Insufficient, "context-parallel size"):
                paired.policy_name({**profile, "context_parallel_size": size}, names)
        check = copy.deepcopy(self.check)
        policies = {name: {"name": name} for name in names}
        check["constants"]["paired_teacher"] = {"PROFILE_POLICIES": {"value": policies, "source": "test"}}
        server = {"health": {"serving": {"profile": {**profile, "context_parallel_size": 2},
                                         "checkpoint_identity": "a" * 64, "startup_nonce": "b" * 32}}}
        self.assertIs(paired.selected_policy(check, server), policies["tp4-mtp-cp2"])

    def test_unchanged_candidate_passes(self):
        result = self.run_gate()
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["metrics"]["affected_positions"], [])
        self.assertEqual(result["metrics"]["pooled"]["mean_gap_ci95"], [0, 0])

    def test_a_named_drafter_needs_its_own_declared_policy(self):
        check = copy.deepcopy(self.check)
        check["constants"]["paired_teacher"] = {"PROFILE_POLICIES": {
            "value": {"tp4-ar": self.policy, "tp4-block": self.policy}, "source": "test"}}
        profile = {"tensor_parallel_size": 4, "speculation": {"tokens": 5, "drafter": "block"}}
        server = {"health": {"serving": {"profile": profile,
                                        "checkpoint_identity": "a" * 64, "startup_nonce": "b" * 32}}}
        self.assertEqual(paired.policy(check, server), self.policy)
        del check["constants"]["paired_teacher"]["PROFILE_POLICIES"]["value"]["tp4-block"]
        with self.assertRaisesRegex(kinds.Insufficient, "no declared paired margins"):
            paired.policy(check, server)
        for drafter in (None, "", "none", "ar", "Bad", "../block", 7):
            with self.subTest(drafter=drafter), self.assertRaises(kinds.Insufficient):
                paired.profile_key({**profile, "speculation": {"tokens": 5, "drafter": drafter}})

    def test_regression_below_absolute_ceiling_fails(self):
        self.candidate = self.capture(self.identity, 0.025)
        self.assertTrue(self.candidate["passed"])
        result = self.run_gate()
        self.assertEqual(result["status"], "fail", result)
        self.assertEqual(result["metrics"]["failed_categories"], ["coding", "tools"])
        self.assertEqual(len(result["metrics"]["affected_positions"]), 320)

    def test_proposed_margin_matches_declaration(self):
        declaration = json.loads((Path(__file__).parent / "templates/paired-teacher.json").read_text())
        proposed = declaration["constants"]["paired_teacher"]["POLICY"]["value"]
        self.assertEqual(proposed["mean_margin"], self.policy["mean_margin"])
        self.assertEqual(set(proposed), set(self.policy))
        # A model adopting the template records the maintainer's approval first.
        self.assertEqual(proposed["approval"], "")

    def test_pooled_margin_accepts_small_change_and_rejects_larger_change(self):
        for error, expected in ((0.014, "pass"), (0.016, "fail")):
            with self.subTest(error=error):
                self.candidate = self.capture(self.identity, error)
                result = self.run_gate()
                self.assertEqual(result["status"], expected, result)
                self.assertEqual(result["metrics"]["failed_categories"], [])

    def test_incomparable_inputs_refused(self):
        mutations = [
            lambda c: c.update(reference_sha256="d" * 64),
            lambda c: c["rows"][0].update(score_from=2),
            lambda c: c["rows"][0].update(prompt_token_ids_sha256="d" * 64),
            lambda c: c["rows"][0].update(category="other"),
            lambda c: c["rows"].pop(),
            lambda c: c["rows"].reverse(),
            lambda c: c["rows"][0].update(served_rows_sha256="d" * 64),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.candidate = copy.deepcopy(self.accepted)
                mutate(self.candidate)
                self.assertEqual(self.run_gate()["status"], "insufficient")

    def test_an_accepted_capture_of_other_texts_is_refused(self):
        self.accepted["rows"][0]["name"] = "moved"
        self.assertEqual(self.run_gate()["status"], "insufficient")

    def test_policy_digest_cannot_change(self):
        value = self.value()
        self.policy["mean_margin"] = 0.1
        self.assertEqual(kinds.rescore(kinds.KINDS["paired_teacher"], value, self.reference, self.check)["status"], "insufficient")

    def test_absolute_gate_still_required(self):
        self.candidate = self.capture(self.identity, 0.2)
        self.policy["mean_margin"] = 0.5
        self.assertEqual(self.run_gate()["status"], "fail")

    def test_nonfinite_and_duplicate_json_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            for data in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":1e309}'):
                path.write_text(data)
                with self.assertRaises(ValueError):
                    paired.read_json(path)

    def test_release_gate_scores_with_its_policy_and_restores_the_environment(self):
        kind = kinds.KINDS["paired_teacher"]
        saved_policy = paired.POLICY
        for ambient in (None, "unrelated-policy.json"):
            for error, passed in ((0.01, True), (0.02, False)):
                with self.subTest(ambient=ambient, error=error), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    (root / "checks").mkdir()
                    self.candidate = self.capture(self.identity, error)
                    check = {"id": "paired_teacher", "args": [], "timeout_seconds": 10,
                             "receipt": "*-paired_teacher.json", **self.check}
                    config = declaration.load(declaration.MECHANICS)
                    config["checks"] = [check]
                    policy_path = root / "policy.json"
                    policy_path.write_text(json.dumps(config))

                    def capture(*args):
                        (root / "checks/run-paired_teacher.json").write_text(json.dumps(
                            {**self.value(), "passed": True}))
                        return 0

                    def reference(selected, loaded):
                        self.assertEqual(loaded, config)
                        self.assertEqual(selected["_cohort"], "full")
                        self.assertEqual(paired.POLICY, self.policy)
                        return self.reference

                    with patch.object(kind, "capture", side_effect=capture), \
                            patch.object(kind, "reference", side_effect=reference), patch.dict(os.environ):
                        os.environ.pop("CHOPINFER_QUAL_CONFIG", None)
                        if ambient is not None:
                            os.environ["CHOPINFER_QUAL_CONFIG"] = ambient
                        rows, context = [], {"_qualified_server": copy.deepcopy(self.candidate["server"])}
                        row = release_gate.run_check(check, rows, "full", context, root, policy_path,
                                                     time.monotonic() + 10, False, None)
                        self.assertEqual(os.environ.get("CHOPINFER_QUAL_CONFIG"), ambient)
                        self.assertIs(paired.POLICY, saved_policy)
                    self.assertEqual(row["passed"], passed, row)
                    self.assertEqual("paired_teacher" in context, passed)
                    self.assertEqual(row.get("error"), None if passed else "threshold predicate failed")

    def test_topology_rules_use_the_candidate_capture_identity(self):
        kind = kinds.KINDS["paired_teacher"]
        check = copy.deepcopy(self.check)
        check["rules"].append({"path": ["server", "health", "serving", "profile", "tensor_parallel_size"],
                               "op": "eq", "value": 2, "source": "test"})
        value = self.value()
        value["server"] = copy.deepcopy(self.candidate["server"])
        value.update(paired.compare(value, self.reference, check))
        scored = kinds.rescore(kind, value, self.reference, check)
        self.assertEqual(scored["status"], "pass", scored)
        self.assertEqual(scored["metrics"], kind.recorded(value, check))
        for changed in ("server", "qualified_server"):
            altered = copy.deepcopy(value)
            altered[changed]["health"]["serving"]["profile"]["tensor_parallel_size"] = 4
            result = kinds.rescore(kind, altered, self.reference, check)
            self.assertEqual(result["status"], "insufficient", result)
        other = copy.deepcopy(value)
        for server in (other["server"], other["qualified_server"], other["arms"]["candidate"]["server"]):
            server["health"]["serving"]["profile"]["tensor_parallel_size"] = 4
        result = kinds.rescore(kind, other, self.reference, check)
        self.assertEqual(result["status"], "fail", result)

    def test_strict_rescore_compares_the_recorded_absolute_gates(self):
        kind = kinds.KINDS["paired_teacher"]
        value = self.value()
        value.update(paired.compare(value, self.reference, self.check))

        def rescore(receipt):
            with patch.object(kind, "reference", return_value=self.reference):
                return checks.rescore_one(receipt, self.check, {}, declaration.MECHANICS, False)

        result = rescore(value)
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(checks.differing(result), [])
        self.assertEqual(result["recorded_metrics"], result["metrics"])
        for key in ("absolute", "absolute_failed_arms"):
            with self.subTest(key=key):
                changed = copy.deepcopy(value)
                changed[key] = {} if key == "absolute" else ["candidate"]
                result = rescore(changed)
                self.assertEqual(result["status"], "pass", result)
                self.assertEqual(checks.differing(result), [key])
                missing = copy.deepcopy(value)
                del missing[key]
                self.assertEqual(checks.differing(rescore(missing)), [key])

    def test_one_category_regresses_while_pooled_passes(self):
        self.accepted = self.capture(self.identity, 0.12)
        self.candidate = self.capture(self.identity, [0.135] * 4 + [0.005] * 4)
        result = self.run_gate()
        self.assertEqual(result["status"], "fail", result)
        self.assertTrue(result["metrics"]["pooled"]["passed"])
        self.assertEqual(result["metrics"]["failed_categories"], ["coding"])
        self.assertEqual(result["metrics"]["absolute_failed_arms"], [])

    def test_improvement_passes(self):
        self.candidate = self.capture(self.identity, 0.005)
        result = self.run_gate()
        self.assertEqual(result["status"], "pass", result)
        self.assertLess(result["metrics"]["pooled"]["mean_gap_ci95"][1], 0)

    def test_top1_regression_through_served_rows(self):
        # Eight independent coding texts, not eight trials from one text.
        for index in range(4):
            text = copy.deepcopy(self.reference["texts"][index])
            text["name"] = f"extra-{index}"
            self.reference["texts"].append(text)
        self.accepted = self.capture(self.identity, 0.01)
        top1 = {text["name"]: [8] + [7] * 39 for text in self.reference["texts"] if text["category"] == "coding"}
        self.candidate = self.capture(self.identity, 0.01, top1)
        self.assertTrue(self.candidate["passed"])
        result = self.run_gate()
        self.assertEqual(result["status"], "fail", result)
        self.assertTrue(result["metrics"]["by_category"]["coding"]["harm_rejected"])
        self.assertFalse(result["metrics"]["pooled"]["top1_passed"])
        self.assertEqual(result["metrics"]["by_category"]["coding"]["severity_added"], 8)
        self.assertEqual(len(result["metrics"]["affected_positions"]), 8)

    def test_decisive_to_near_tie_is_improvement(self):
        names = [text["name"] for text in self.reference["texts"][:4]]
        for text in self.reference["texts"][:4]:
            for variant in self.reference["variants"]:
                row = text[variant]
                row["topk_logits"][0][1] = row["topk_logprobs"][0][1] = -1.2
                row["margin"][0] = 0.2
        self.accepted = self.capture(self.identity, 0.01,
                                     {name: [8] + [7] * 39 for name in names})
        self.candidate = self.capture(self.identity, 0.01,
                                      {name: [900] + [7] * 39 for name in names})
        result = self.run_gate()
        self.assertEqual(result["status"], "pass", result)
        category = result["metrics"]["by_category"]["coding"]
        self.assertEqual((category["severity_added"], category["severity_removed"]), (0, 4))
        self.assertEqual(category["candidate"]["top1_near_ties"], 4)

    def test_realistic_neutral_fixture(self):
        fixture = json.loads((Path(__file__).resolve().parents[2] / "fixtures/paired-accuracy-neutral.json").read_text())
        names = [text["name"] for text in self.reference["texts"]]
        def top1(offsets):
            return {name: [8 if index in positions else 7 for index in range(40)]
                    for name, positions in zip(names, offsets, strict=True)}
        self.accepted = self.capture(self.identity, 0.01, top1(fixture["accepted_flip_offsets"]))
        self.candidate = self.capture(self.identity,
                                      [0.01 + value for value in fixture["error_changes"]], top1(fixture["candidate_flip_offsets"]))
        result = self.run_gate()
        self.assertEqual(result["status"], "pass", result)
        self.assertGreater(len(result["metrics"]["affected_positions"]), 0)

    def test_sparse_pooled_scope_requires_explicit_point_permission(self):
        positions = paired.capture_scores(self.accepted, self.reference)[:80]
        result = paired.scope(positions, positions, self.policy)
        self.assertEqual(result["interval_mode"], "insufficient")
        self.assertFalse(result["passed"])
        self.assertFalse(result["evidence_sufficient"])
        self.policy["allow_sparse_pooled"] = True
        result = paired.scope(positions, positions, self.policy)
        self.assertEqual(result["interval_mode"], "point_sparse")
        self.assertIsNone(result["mean_gap_ci95"])
        self.assertIsNone(result["top1_net_rate_ci95"])
        self.assertTrue(result["passed"])

    def test_bit_identical_scope_skips_bootstrap_and_pairs_once(self):
        positions = paired.capture_scores(self.accepted, self.reference)
        with patch.object(teacher, "clustered_interval", side_effect=AssertionError("unneeded bootstrap")), patch.object(
                paired.limits, "paired", wraps=paired.limits.paired) as scored:
            result = paired.scope(positions, positions, self.policy)
        self.assertTrue(result["bit_identical"])
        self.assertEqual(scored.call_count, 1)

    def test_rescore_pairs_once_per_scope(self):
        with patch.object(paired.limits, "paired", wraps=paired.limits.paired) as scored:
            self.assertEqual(self.run_gate()["status"], "pass")
        self.assertEqual(scored.call_count, 3)

    def test_tail_catches_growth_hidden_by_mean(self):
        accepted = [{"text": "t", "difference": 0.0, "disagrees": False, "flip": None, "served": 7}] * 1000
        candidate = copy.deepcopy(accepted)
        for index in range(3):
            candidate[index] = {**candidate[index], "difference": 0.6}
        result = paired.scope(candidate, accepted, {**self.policy, "allow_sparse_pooled": True})
        self.assertTrue(result["mean_passed"])
        self.assertEqual(result["tail"]["net_added"], 3)
        self.assertFalse(result["passed"])

    def test_text_sign_tail_and_holm_step_down(self):
        self.assertEqual(paired.sign_test(6, 0), 1 / 64)
        self.assertEqual(paired.sign_test(0, 6), 1)
        self.assertEqual(paired.sign_test(2, 3), 26 / 32)
        categories = {"one": {"harm_p": 0.04, "passed": True}, "two": {"harm_p": 0.045, "passed": True}}
        paired.holm(categories, 0.05)
        self.assertFalse(any(row["harm_rejected"] for row in categories.values()))

    def test_shipped_teacher_predicate_is_enforced(self):
        self.reference["absolute_check"]["rules"].append(
            {"path": ["mean_abs_dlogprob"], "op": "le", "value": 0.005, "source": "test shipped policy"})
        result = self.run_gate()
        self.assertEqual(result["status"], "insufficient", result)
        self.assertIn("accepted arm fails", result["error"])

    def test_recorded_mean_cannot_vouch_for_raw_scores(self):
        self.candidate["mean_abs_dlogprob"] += 0.001
        self.candidate["max_abs_dlogprob"] += 0.001
        for row in self.candidate["by_category"].values():
            row["mean_abs_dlogprob"] += 0.001
            row["max_abs_dlogprob"] += 0.001
        result = self.run_gate()
        self.assertEqual(result["status"], "fail", result)
        self.assertIn("mean_abs_dlogprob", result["metrics"]["absolute"]["candidate"]["summary_differences"])
        self.assertEqual(result["metrics"]["absolute"]["candidate"]["status"], "fail")

    def test_one_text_many_flips_is_one_harm_trial(self):
        positions = paired.capture_scores(self.accepted, self.reference)
        candidate = copy.deepcopy(positions)
        for index in range(8):
            candidate[index].update(flip={"decisive": True}, disagrees=True, served=8)
        category = paired.scope(candidate[:160], positions[:160], self.policy, pooled=False)
        self.assertEqual(category["severity_added"], 8)
        self.assertEqual((category["worsened_texts"], category["improved_texts"]), (1, 0))
        self.assertEqual(category["harm_p"], 0.5)
        categories = {str(index): copy.deepcopy(category) for index in range(7)}
        paired.holm(categories, self.policy["category_alpha"])
        self.assertTrue(all(row["passed"] for row in categories.values()))

    def test_text_harm_offsets_severity_within_a_text(self):
        accepted = paired.capture_scores(self.accepted, self.reference)
        candidate = copy.deepcopy(accepted)
        accepted[0].update(flip={"decisive": True}, disagrees=True, served=8)
        candidate[1].update(flip={"decisive": True}, disagrees=True, served=8)
        result = paired.scope(candidate, accepted, self.policy, pooled=False)
        self.assertEqual(result["tied_texts"], 8)
        self.assertEqual(result["harm_p"], 1)

    def test_sparse_pooled_rescore_is_insufficient(self):
        self.reference["texts"] = self.reference["texts"][:1]
        self.accepted = self.capture(self.identity, 0.01)
        self.candidate = copy.deepcopy(self.accepted)
        result = self.run_gate()
        self.assertEqual(result["status"], "insufficient", result)
        self.assertIn("too few texts", result["error"])
        self.policy["allow_sparse_pooled"] = True
        self.assertEqual(self.run_gate()["status"], "pass")

    def test_tail_counts_net_changes_not_gross_growth(self):
        accepted = [{"text": f"t-{index // 125}", "difference": 0.0, "disagrees": False,
                     "flip": None, "served": 7} for index in range(1000)]
        candidate = copy.deepcopy(accepted)
        for index in range(3):
            accepted[index]["difference"] = 0.6
            candidate[index + 3]["difference"] = 0.6
        result = paired.scope(candidate, accepted, self.policy)
        self.assertEqual((result["tail"]["added"], result["tail"]["removed"], result["tail"]["net_added"]), (3, 3, 0))
        self.assertTrue(result["passed"], result)

    def test_invalid_accepted_summary_is_insufficient(self):
        self.accepted["mean_abs_dlogprob"] += 0.001
        self.accepted["max_abs_dlogprob"] += 0.001
        for row in self.accepted["by_category"].values():
            row["mean_abs_dlogprob"] += 0.001
            row["max_abs_dlogprob"] += 0.001
        result = self.run_gate()
        self.assertEqual(result["status"], "insufficient", result)
        self.assertIn("accepted arm fails", result["error"])

    def test_rescore_verifies_arms_once(self):
        with patch.object(paired, "verify_arms", wraps=paired.verify_arms) as verified, patch.object(
                kinds, "call", wraps=kinds.call) as evaluated:
            result = self.run_gate()
        self.assertEqual(result["status"], "pass", result)
        self.assertEqual(result["recorded_fields"], ["arms", "policy_sha256"])
        self.assertEqual(verified.call_count, 1)
        self.assertEqual(evaluated.call_count, 6)

    def test_large_capture_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.json"
            path.write_text('{"rows":"' + 'x' * (21 << 20) + '"}')
            self.assertEqual(len(paired.read_json(path)["rows"]), 21 << 20)


if __name__ == "__main__":
    unittest.main()

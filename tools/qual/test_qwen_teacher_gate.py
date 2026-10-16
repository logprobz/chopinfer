"""Qwen3.8's teacher gate: the GPU runner's F32 reference, the BF16 and f64
spread that sets its limits, and the checkpoint and reference pins. The
live capture and its binding run in `chopinfer eval run`
(crates/runtime-eval/tests/teacher.rs)."""

from qual_test_config import configure

import copy
import json
import unittest

import common
import declaration
import teacher

CONFIG = declaration.load(declaration.path("qwen38"))
PIN = CONFIG["qualification"]["checkpoint"]
SPEC = CONFIG["qualification"]["references"]["teacher"]
REFERENCE = common.ROOT / SPEC["path"]
CHECK = next(check for check in CONFIG["checks"] if check["id"] == "teacher")
LONG = common.ROOT / CONFIG["qualification"]["references"]["long_context"]["path"]
LONG_CHECK = next(check for check in CONFIG["checks"] if check["id"] == "long_context")


def setUpModule():
    configure("qwen38")


# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "teacher gate references and rescoring"


def rule_value(path):
    return next(rule["value"] for rule in CHECK["rules"] if rule["path"] == path)


class SpreadTests(unittest.TestCase):
    def test_the_spread_sets_the_configured_limits(self):
        texts = json.loads(REFERENCE.read_text())["texts"]
        limits, spread = teacher.reference_limits(texts)
        bf16, f64 = teacher.summary(spread["bf16"]), teacher.summary(spread["f64"])
        # The runner's BF16 run is 0.066 mean and 2.6% top-1 from its F32 run over 4,176 positions;
        # its f64 run is 0.0017 and 3 flips, all in long_4k.
        self.assertEqual((bf16["scored"], f64["scored"]), (4176, 4176))
        self.assertAlmostEqual(bf16["mean_abs_dlogprob"], 0.0661, places=4)
        self.assertEqual(bf16["top1_disagreements"], 109)
        self.assertAlmostEqual(f64["mean_abs_dlogprob"], 0.0017, places=4)
        self.assertEqual(f64["top1_disagreements"], 3)
        # 4,176 positions take the 0.10 floor, which the spread rounded up to 0.05 equals.
        # Under the near-tie rule 64 of those flips are near ties: 45 decisive (1.1%).
        self.assertEqual((bf16["top1_near_ties"], bf16["top1_decisive_flips"]), (64, 45))
        self.assertEqual(limits, {"mean_abs_dlogprob": 0.10, "top1_decisive": 0.08, "top1_near_tie": 0.08,
                                  "top1_disagreement": 0.10})
        self.assertEqual(limits, {"mean_abs_dlogprob": rule_value(["mean_abs_dlogprob"]),
                                  "top1_decisive": rule_value(["top1_decisive"]),
                                  "top1_near_tie": rule_value(["top1_near_tie"]),
                                  "top1_disagreement": rule_value(["top1_disagreement"])})
        self.assertEqual(rule_value(["scored"]), 4176)

    def test_each_category_has_the_limits_of_its_own_spread(self):
        texts = json.loads(REFERENCE.read_text())["texts"]
        result = {}
        self.assertTrue(teacher.evaluate(teacher.variant_scores(texts, "exact"), texts, result)[1])
        counts = {name: rows["scored"] for name, rows in result["by_category"].items()}
        self.assertEqual(counts, {"chat": 899, "code": 753, "long": 512, "math": 779,
                                  "multilingual": 713, "prose": 379, "prose-frozen": 141})
        # Every category has fewer than 1,000 positions; chat's BF16 spread (0.151) rounds up to 0.20.
        for name, rows in result["by_category"].items():
            with self.subTest(name):
                self.assertAlmostEqual(rows["limits"]["mean_abs_dlogprob"], 0.20 if name == "chat" else 0.15)
                self.assertEqual(rows["limits"]["top1_decisive"], 0.08)
        self.assertAlmostEqual(result["reference_spread"]["bf16"]["by_category"]["chat"]["mean_abs_dlogprob"],
                               0.1508, places=4)

    def test_the_spread_variants_pass_and_a_bad_category_fails(self):
        texts = json.loads(REFERENCE.read_text())["texts"]
        for variant in ("bf16", "f64"):
            with self.subTest(variant):
                self.assertTrue(teacher.evaluate(teacher.variant_scores(texts, variant), texts, {})[1])
        # 0.2 nats off on every long position: inside the pooled limit, outside long's.
        worse = [{**text, "logprob": [value - 0.2 for value in text["logprob"]]} if text["category"] == "long"
                 else text for text in teacher.variant_scores(texts, "exact")]
        result = {}
        self.assertFalse(teacher.evaluate(worse, texts, result)[1])
        self.assertLessEqual(result["mean_abs_dlogprob"], result["limits"]["mean_abs_dlogprob"])
        self.assertEqual(result["failed_categories"], ["long"])

    def test_the_reference_scores_the_frozen_texts_and_is_bound(self):
        reference = common.pinned_reference("teacher", REFERENCE)
        frozen = json.loads((common.ROOT / "fixtures/qwen38/teacher-texts.json").read_text())["texts"]
        teacher.check_reference(reference["texts"], [text["name"] for text in frozen])
        self.assertEqual([teacher.ids_digest(text["prompt_token_ids"]) for text in reference["texts"]],
                         [text["prompt_token_ids_sha256"] for text in frozen])
        binding = common.pinned_binding("teacher", reference)
        self.assertEqual((binding["status"], binding["refset_sha256"]),
                         ("bound", reference["binding"]["refset_sha256"]))
        self.assertEqual({mode: entries["variant"] for mode, entries in reference["binding"]["modes"].items()},
                         {"f32": "exact", "bf16": "bf16", "f64": "f64"})
        self.assertEqual(reference["provenance"]["gate_variant"], teacher.GATE_VARIANT)


class PinTests(unittest.TestCase):
    def test_the_pin_matches_the_lock_and_the_reference(self):
        pin = common.checkpoint_pin()
        stated = json.loads(REFERENCE.read_text())["provenance"]["checkpoint"]
        self.assertEqual({key: stated[key] for key in common.PINNED_CHECKPOINT_KEYS},
                         {key: pin[key] for key in common.PINNED_CHECKPOINT_KEYS})
        frontend = json.loads(common.input_path("frontend").read_text())["tokenizer"]
        self.assertEqual((pin["repository"], pin["revision"]), (frontend["repository"], frontend["revision"]))
        self.assertEqual(rule_value(["server", "checkpoint", "checkpoint_identity"]), pin["checkpoint_identity"])
        self.assertEqual(rule_value(["reference_sha256"]), SPEC["sha256"])
        changed = copy.deepcopy(CONFIG)
        changed["qualification"]["checkpoint"]["source_lock_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differs from"):
            common.checkpoint_pin(changed)


class LongContextTests(unittest.TestCase):
    """The last 1,024 positions of long_32k, gated like the teacher texts
    (the check's --reference-name long_context)."""

    def test_the_spread_sets_the_floors(self):
        texts = json.loads(LONG.read_text())["texts"]
        self.assertEqual([(text["name"], len(text["prompt_token_ids"]), text["score_from"]) for text in texts],
                         [("long_32k", 32768, 31744)])
        limits, spread = teacher.reference_limits(texts, variants=("exact", "bf16", "f64"))
        bf16, f64 = teacher.summary(spread["bf16"]), teacher.summary(spread["f64"])
        self.assertAlmostEqual(bf16["mean_abs_dlogprob"], 0.0223, places=4)
        self.assertEqual((bf16["top1_disagreements"], f64["top1_disagreements"]), (7, 3))
        self.assertAlmostEqual(f64["mean_abs_dlogprob"], 0.0105, places=4)
        self.assertEqual(limits, {"mean_abs_dlogprob": 0.10, "top1_decisive": 0.08, "top1_near_tie": 0.08,
                                  "top1_disagreement": 0.10})
        rules = {tuple(rule["path"]): rule["value"] for rule in LONG_CHECK["rules"]}
        self.assertEqual([rules[(key,)] for key in ("mean_abs_dlogprob", "top1_decisive", "top1_near_tie",
                                                     "top1_disagreement", "scored")], [0.10, 0.08, 0.08, 0.10, 1024])
        self.assertEqual(rules[("reference_sha256",)], CONFIG["qualification"]["references"]["long_context"]["sha256"])


if __name__ == "__main__":
    unittest.main()

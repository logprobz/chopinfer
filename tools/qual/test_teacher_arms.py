"""teacher_arms.py: two bound teacher arms compared on the pinned reference."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from qual_test_config import setUpModule  # noqa: F401
import common
import teacher
import teacher_arms


def arm(config, reference, shift=0.0, nonce="a"):
    """A passing live teacher receipt and its capture whose logprobs are the
    reference's gate variant plus `shift`."""
    pin = common.checkpoint_pin(config)
    server = {"model": pin["served_model_name"],
              "health": {"serving": {"checkpoint_identity": pin["checkpoint_identity"], "startup_nonce": nonce * 32,
                                     "package_identity": "p" * 64, "version": "1.0", "source_commit": "c" * 40,
                                     "profile": {"tensor_parallel_size": 4, "kv_cache_dtype": "fp8"}}},
              "checkpoint": {"checkpoint_identity": pin["checkpoint_identity"], "startup_nonce": nonce * 32}}
    texts = []
    for text in reference:
        exact = text[teacher.GATE_VARIANT]
        texts.append({"name": text["name"], "category": text["category"],
                      "prompt_token_ids_sha256": teacher.ids_digest(text["prompt_token_ids"]),
                      "score_from": text["score_from"], "logprob": [x + shift for x in exact["logprob"]],
                      "top1": list(exact["top1"]), "margin": list(exact["margin"]),
                      "served_rows": [[[str(token), value]]
                                      for token, value in zip(exact["top1"], exact["logprob"])]})
    digest = config["qualification"]["references"]["teacher"]["sha256"]
    capture = {"schema": config["qualification"]["teacher_capture_schema"], "server": server,
               "reference_sha256": digest, "texts": texts}
    value = {"schema": "chopinfer-qualification/v1", "check": config["qualification"]["teacher_receipt"],
             "server": copy.deepcopy(server), "passed": True, "mechanics_only": False, "skipped": [], "errors": [],
             "reference_sha256": digest,
             "rows": [{"name": text["name"], "served_rows_sha256": common.rows_digest(text["served_rows"])}
                      for text in texts]}
    return value, capture


class TeacherArmsTests(unittest.TestCase):
    def setUp(self):
        self.config = common.qualification_config()
        self.reference = common.pinned_reference("teacher", common.input_path("teacher_reference",
                                                                                config=self.config),
                                                 self.config)["texts"]
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def write(self, name, value, capture, after=None):
        """The arm's files: the receipt binds the capture as written; `after`
        edits the capture once the receipt is written."""
        root = Path(self.directory.name)
        data = json.dumps(capture).encode()
        value = {**value, "capture_sha256": hashlib.sha256(data).hexdigest()}
        (root / f"{name}.json").write_text(json.dumps(value))
        if after:
            after(capture)
            data = json.dumps(capture).encode()
        (root / f"{name}-capture.json").write_bytes(data)
        return name, str(root / f"{name}.json"), str(root / f"{name}-capture.json")

    def test_two_bound_arms_report_their_differences(self):
        first = self.write("first", *arm(self.config, self.reference))
        second = self.write("second", *arm(self.config, self.reference, shift=-0.25, nonce="b"))
        result = teacher_arms.compare([first, second], self.config)
        self.assertTrue(result["passed"])
        self.assertEqual(result["scored"],
                         sum(len(text[teacher.GATE_VARIANT]["logprob"]) for text in self.reference))
        self.assertAlmostEqual(result["against_reference"]["first"]["pooled"]["mean_abs_dlogprob"], 0.0)
        self.assertAlmostEqual(result["mean_abs_dlogprob_difference"]["pooled"], 0.25)
        self.assertAlmostEqual(result["between_arms"]["pooled"]["mean_abs_logprob_difference"], 0.25)
        self.assertEqual(result["between_arms"]["pooled"]["top1_differs"], 0)

    def test_an_unbound_or_mismatched_arm_is_refused(self):
        first = self.write("first", *arm(self.config, self.reference))
        cases = {}
        value, capture = arm(self.config, self.reference, nonce="b")
        capture["texts"][0]["served_rows"][0] = [["0", 0.0]]
        cases["the capture does not hold the served rows"] = (value, capture)
        value, capture = arm(self.config, self.reference, nonce="b")
        value["server"]["rescored_from"] = capture["server"]["rescored_from"] = "elsewhere.json"
        cases["was not bound live"] = (value, capture)
        value, capture = arm(self.config, self.reference, nonce="b")
        value["passed"] = False
        cases["did not pass"] = (value, capture)
        value, capture = arm(self.config, self.reference, nonce="b")
        capture["texts"][0]["score_from"] += 1
        cases["different workloads"] = (value, capture)
        value, capture = arm(self.config, self.reference, nonce="b")
        value["reference_sha256"] = capture["reference_sha256"] = "0" * 64
        cases["pinned teacher reference"] = (value, capture)
        value, capture = arm(self.config, self.reference, nonce="b")
        value["server"]["health"]["serving"]["source_commit"] = "d" * 40
        capture["server"] = copy.deepcopy(value["server"])
        cases["different builds"] = (value, capture)
        for message, (value, capture) in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                teacher_arms.compare([first, self.write("second", value, capture)], self.config)

    def test_scores_outside_the_receipts_capture_or_workload_are_refused(self):
        def shifted(capture):
            for text in capture["texts"]:
                text["logprob"] = [x + 1 for x in text["logprob"]]
        first = self.write("first", *arm(self.config, self.reference))
        edited = self.write("second", *arm(self.config, self.reference, nonce="b"), after=shifted)
        with self.assertRaisesRegex(ValueError, "capture_sha256"):
            teacher_arms.compare([first, edited], self.config)
        # Both arms agreeing on a workload other than the pinned reference's.
        def moved(capture):
            for text in capture["texts"]:
                text["prompt_token_ids_sha256"] = "9" * 64
                text["score_from"] += 1
        arms = []
        for name, nonce in (("first", "a"), ("second", "b")):
            value, capture = arm(self.config, self.reference, nonce=nonce)
            moved(capture)
            arms.append(self.write(name, value, capture))
        with self.assertRaisesRegex(ValueError, "pinned reference's workload"):
            teacher_arms.compare(arms, self.config)


if __name__ == "__main__":
    unittest.main()

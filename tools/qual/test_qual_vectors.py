"""The shared check kinds against their conformance vectors
(tests/qual-vectors, docs/check-kinds.md): each vector's receipt, reference
and check must give its expected status and metrics. `chopinfer eval` runs
the same vectors."""

import json
import math
from pathlib import Path
import unittest

import kinds

VECTORS = Path(__file__).resolve().parents[2] / "tests/qual-vectors"
SHARED = ("teacher_forced", "media_forced", "greedy_pin", "determinism", "api", "structured")
# The contract clauses whose boundaries the vectors must pin (docs/check-kinds.md).
REQUIRED = {
    "teacher_forced": ("rounded_up exact multiple", "rounded_up binary overshoot", "limit inclusive",
                       "missing variant", "invalid reference scores", "top-1 near tie at limit",
                       "top-1 near tie above limit", "top-1 outside top-k", "near-tie rate inclusive",
                       "near-tie rate bound", "decisive rate bound", "near-tie limit from spread",
                       "decisive limit from spread", "flips recomputed", "flips required", "total rate inclusive",
                       "total rate bound", "total limit from spread", "total limit unrounded",
                       "total cap calibration",
                       "served tokens from rows", "served rows bound", "served rows required"),
    "media_forced": ("rounded_up exact multiple", "rounded_up binary overshoot", "limit inclusive",
                     "near tie at limit", "near tie above limit", "audio near tie at limit",
                     "audio near tie above limit", "outside top 5", "ended inside", "runner layout",
                     "missing variant", "invalid reference scores", "impossible scores",
                     "top-1 near tie at limit", "top-1 near tie above limit", "near-tie rate bound",
                     "total rate bound", "flips recomputed", "flips required", "served tokens from rows",
                     "served rows bound", "served rows required",
                     "probe near tie at limit", "probe near tie above limit", "probe margin not in reference top-k", "probe self gap", "probe forced at limit",
                     "probe forced above limit", "probe coverage", "probe impossible scores", "probe digest", "probe flip evidence"),
    "frontend_goldens": ("golden equality", "golden mismatch", "golden missing", "golden empty", "golden consistency", "golden binding", "golden counts", "golden pinned counts"),
    "references": ("unbound", "bound to another lock", "digest mismatch", "schema mismatch",
                   "provenance mismatch", "missing variant"),
}


def close(expected, actual, where):
    """Equal values, floats within 1e-12 relative (docs/check-kinds.md#conformance-vectors)."""
    if isinstance(expected, float) or isinstance(actual, float):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isclose(
            expected, actual, rel_tol=1e-12, abs_tol=1e-15), f"{where}: {actual!r} != {expected!r}"
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and expected.keys() == actual.keys(), f"{where}: keys differ"
        for key in expected:
            close(expected[key], actual[key], f"{where}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(expected) == len(actual), f"{where}: lengths differ"
        for index, (a, b) in enumerate(zip(expected, actual)):
            close(a, b, f"{where}[{index}]")
    else:
        assert expected == actual, f"{where}: {actual!r} != {expected!r}"


class VectorTests(unittest.TestCase):
    def test_every_shared_kind_has_vectors_of_each_status(self):
        for name in SHARED:
            with self.subTest(name):
                value = json.loads((VECTORS / f"{name}.json").read_text())
                self.assertEqual((value["schema"], value["kind"]), ("chopinfer-qual-vectors/v1", name))
                statuses = {vector["expected"]["status"] for vector in value["vectors"]}
                self.assertLessEqual({"pass", "insufficient"}, statuses)
                self.assertTrue(statuses & {"fail", "informational"})

    def test_the_vectors_cover_every_contract_boundary(self):
        # Each clause a second implementation could get wrong has vectors on
        # both sides; a vector names the clauses it covers.
        for name, clauses in REQUIRED.items():
            with self.subTest(name):
                value = json.loads((VECTORS / f"{name}.json").read_text())
                covered = {clause for vector in value["vectors"] for clause in vector.get("covers", [])}
                self.assertLessEqual(set(clauses), covered)

    def test_impossible_media_scores_are_insufficient(self):
        value = json.loads((VECTORS / "media_forced.json").read_text())
        vector = next(vector for vector in value["vectors"] if vector["name"] == "pass")
        changes = {"negative errors": {"mean_abs_dlogprob": -1, "max_abs_dlogprob": -1},
                   "negative top-1 rate": {"top1_disagreement": -1},
                   "top-1 rate above one": {"top1_disagreement": 1.5},
                   "a rate no count gives": {"top1_disagreement": 0.4},
                   "mean above max": {"mean_abs_dlogprob": 0.05, "max_abs_dlogprob": 0},
                   "NaN mean": {"mean_abs_dlogprob": float("nan")}}
        for label, change in changes.items():
            with self.subTest(label):
                receipt = json.loads(json.dumps(vector["receipt"]))
                receipt["rows"][0]["logprobs"].update(change)
                result = kinds.rescore(kinds.KINDS["media_forced"], receipt, vector["reference"], vector["check"])
                self.assertEqual(result["status"], "insufficient", result.get("error"))

    def test_impossible_reference_scores_are_insufficient(self):
        for name in ("teacher_forced", "media_forced"):
            value = json.loads((VECTORS / f"{name}.json").read_text())
            vector = next(vector for vector in value["vectors"] if vector["name"] == "pass")
            for label, logprob in (("positive", 0.5), ("NaN", float("nan"))):
                with self.subTest(kind=name, logprob=label):
                    reference = json.loads(json.dumps(vector["reference"]))
                    items = reference["texts"] if name == "teacher_forced" else reference["cases"]
                    scores = items[0][reference["gate"]] if name == "teacher_forced" else items[0]["exact"][reference["gate"]]
                    scores["logprob"][0] = logprob
                    result = kinds.rescore(kinds.KINDS[name], vector["receipt"], reference, vector["check"])
                    self.assertEqual(result["status"], "insufficient", result.get("error"))

    def test_the_reference_reading_vectors_reproduce(self):
        import vectors
        value = json.loads((VECTORS / "references.json").read_text())
        self.assertEqual((value["schema"], value["kind"]), ("chopinfer-qual-vectors/v1", "references"))
        for vector in value["vectors"]:
            with self.subTest(vector["name"]):
                result = vectors.read_reference(vector["pin"], vector["checkpoint"], vector["fixture"])
                self.assertEqual(result["status"], vector["expected"]["status"], result.get("error"))
                self.assertEqual(result.get("binding"), vector["expected"].get("binding"))
                if "error" in vector["expected"]:
                    self.assertIn(vector["expected"]["error"], result["error"])

    def test_the_vectors_reproduce(self):
        for path in sorted(VECTORS.glob("*.json")):
            value = json.loads(path.read_text())
            if value["kind"] == "references":
                continue
            kind = kinds.KINDS[value["kind"]]
            for vector in value["vectors"]:
                with self.subTest(kind=kind.name, vector=vector["name"]):
                    result = kinds.rescore(kind, vector["receipt"], vector.get("reference"), vector["check"],
                                           mechanics=vector.get("mechanics", False))
                    expected = vector["expected"]
                    self.assertEqual(result["status"], expected["status"], result.get("error"))
                    if "error" in expected:
                        self.assertIn(expected["error"], result.get("error", ""))
                    if "metrics" in expected:
                        close(expected["metrics"], result["metrics"], "metrics")


if __name__ == "__main__":
    unittest.main()

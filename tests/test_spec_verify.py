"""Check numerical gates while retaining greedy agreement as information."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


QUAL = Path(__file__).resolve().parents[1] / "tools" / "qual"
sys.path.insert(0, str(QUAL))
SPEC = importlib.util.spec_from_file_location("spec_verify", QUAL / "spec_verify.py")
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def record(token="a", *, arm="chopinfer-spec"):
    def group(value):
        return {"prompts_sha256": "fixture", "speculation": {"draft_tokens": int(arm == "chopinfer-spec"),
                        "verify_items": int(arm == "chopinfer-spec")}, "tokens_per_engine_step": 1,
                "requests": [{"tokens": [value], "logprobs": [-0.1], "top": [{value: -0.1}]}]}
    return {"run_id": "cpu", "output_tokens": 1, "arm": arm,
            "sets": {"copy": group("a"), "natural": group(token), "chat": group(token)}}


class SpecVerifyTests(unittest.TestCase):
    def compare(self, spec, ar, dlogprob=0.006, disagreement=0.01):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("spec.json", "ar.json")]
            for path, data in zip(paths, (spec, ar)):
                path.write_text(json.dumps(data))
            args = SimpleNamespace(spec=paths[0], ar=paths[1], maximum_abs_dlogprob=dlogprob,
                                   maximum_top1_disagreement=disagreement)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
                VERIFY.compare(args)
            return result.exception.code

    def test_verification_uses_unfiltered_logprobs(self):
        bodies = []
        def request(path, body):
            bodies.append(body)
            return {"choices": [{"logprobs": {"tokens": ["a"], "token_logprobs": [-0.1],
                     "top_logprobs": [{"a": -0.1}]}, "finish_reason": "length"}]}, 1.0
        VERIFY.completion(SimpleNamespace(request=request), "fixture", "Prompt", 1)
        self.assertEqual(bodies[0]["top_p"], 1)
        self.assertEqual(bodies[0]["logprobs"], 5)

    def test_greedy_agreement_is_informational(self):
        self.assertEqual(self.compare(record("b"), record(arm="chopinfer-ar")), 0)

    def test_same_arm_comparison_is_refused(self):
        with self.assertRaisesRegex(ValueError, "distinct speculative and AR arms"):
            self.compare(record(), record())

    def test_tighter_numerical_bounds_are_enforced(self):
        spec = record()
        spec["sets"]["copy"]["requests"][0]["logprobs"] = [-0.107]
        self.assertEqual(self.compare(spec, record(arm="chopinfer-ar")), 1)
        self.assertEqual(self.compare(spec, record(arm="chopinfer-ar"), dlogprob=0.008), 0)
        spec = record()
        spec["sets"]["copy"]["requests"][0]["top"] = [{"b": -0.1}]
        self.assertEqual(self.compare(spec, record(arm="chopinfer-ar")), 1)

    def test_missing_requests_or_sets_cannot_pass(self):
        spec = record()
        spec["sets"]["chat"]["requests"] = []
        with self.assertRaisesRegex(ValueError, "nonempty requests"):
            self.compare(spec, record(arm="chopinfer-ar"))
        spec = record()
        del spec["sets"]["chat"]
        self.assertIsInstance(self.compare(spec, record(arm="chopinfer-ar")), str)


if __name__ == "__main__":
    unittest.main()

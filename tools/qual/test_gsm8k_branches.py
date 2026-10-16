"""Offline checks of the GSM8K branch probe and the paired GSM8K rule."""

from qual_test_config import setUpModule  # noqa: F401

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import gsm8k_branches as probe

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "GSM8K branch probe and paired GSM8K rule"


PROMPT = [1, 2, 3]
# Each arm's generated tokens: they share [10, 11] and part at the third.
AR, SPEC = [10, 11, 12, 13], [10, 11, 14]


def retained(generated, messages="q", content="#### 1"):
    return {"body": {"messages": [{"role": "user", "content": messages}]},
            "response": {"prompt_token_ids": PROMPT, "choices": [
                {"token_ids": generated, "message": {"content": content}, "finish_reason": "stop"}]}}


class FakeServer:
    """Prompt logprobs whose last row holds the AR branch token 12 and
    `others`, as (logprob, rank) with rank None for an unranked token."""

    def __init__(self, others, ar=(-0.7, 1)):
        self.others = {12: ar, **others}
        self.bodies = []

    def request(self, path, body):
        self.bodies.append(body)
        ids = body["prompt"]
        rows = [None] + [{str(token): {"logprob": -0.1, "rank": 1}} for token in ids[1:]]
        rows[-1] = {str(token): {"logprob": value} | ({} if rank is None else {"rank": rank})
                    for token, (value, rank) in self.others.items()}
        return {"choices": [{"prompt_token_ids": ids, "prompt_logprobs": rows}]}, 0.1


class Run:
    """Two scored GSM8K arms on disk, their responses retained and bound by
    digest. `wrong` maps an arm to the items it answered wrong."""

    def __init__(self, directory, wrong, generated=None):
        self.paths = {}
        for arm in ("ar", "spec"):
            responses = Path(directory) / arm / "responses"
            responses.mkdir(parents=True)
            rows = []
            for index in range(200):
                correct = index not in wrong.get(arm, ())
                tokens = (generated or {}).get(arm, AR if arm == "ar" else SPEC)
                data = (json.dumps(retained(tokens, content="#### 1" if correct else "#### 2"), indent=2) + "\n").encode()
                (responses / f"{index}.json").write_bytes(data)
                rows.append({"index": index, "expected": "1", "actual": "1" if correct else "2", "rule": "####",
                             "finish_reason": "stop", "correct": correct,
                             "response_sha256": hashlib.sha256(data).hexdigest()})
            self.paths[arm] = Path(directory) / arm / "gsm8k.json"
            self.write(arm, {"examples_sha256": "e", "sampling": {"seed": 1}, "correct": sum(r["correct"] for r in rows),
                             "rows": rows})

    def write(self, arm, value):
        self.paths[arm].write_text(json.dumps(value))

    def receipt(self, arm):
        return json.loads(self.paths[arm].read_text())

    def probe(self, server):
        return probe.probe(server, "m", self.paths["ar"], self.paths["spec"])

    def pair(self, branches):
        return probe.pair(self.receipt("ar"), self.receipt("spec"), branches, floor=194)


class BranchTests(unittest.TestCase):
    def test_the_arms_part_at_their_first_differing_token(self):
        self.assertEqual(probe.branch(retained(AR), retained(SPEC)), (PROMPT, [10, 11], 12, 14))
        # An arm that ended where the other continued has no branch token.
        self.assertEqual(probe.branch(retained(AR), retained(AR[:2])), (PROMPT, [10, 11], 12, None))
        for spec, message in ((retained(AR), "same tokens"), (retained(SPEC, "other"), "different prompts")):
            with self.assertRaisesRegex(ValueError, message):
                probe.branch(retained(AR), spec)
        legacy = retained(SPEC)
        del legacy["response"]["choices"][0]["token_ids"]
        with self.assertRaisesRegex(ValueError, "no token IDs"):
            probe.branch(retained(AR), legacy)

    def probed(self, others, ar=(-0.7, 1), wrong=None):
        """Speculation loses item 45; the probe's result and the pair verdict."""
        server = FakeServer(others, ar)
        with tempfile.TemporaryDirectory() as directory:
            run = Run(directory, wrong or {"spec": [45]})
            result = run.probe(server)
            try:
                verdict = run.pair(result)
            except ValueError as error:
                verdict = str(error)
        return result, verdict, server

    def test_a_gap_within_the_limit_is_a_near_tie_branch(self):
        result, verdict, server = self.probed({14: (-0.9, 2)})
        item = result["items"][0]
        self.assertEqual((item["verdict"], item["branch_offset"], item["ar_token"], item["spec_token"]),
                         ("near_tie", 2, 12, 14))
        self.assertAlmostEqual(item["gap"], 0.2)
        self.assertEqual((item["ar_rank"], item["spec_rank"]), (1, 2))
        self.assertEqual((verdict["near_tie_lost"], verdict["decisive_lost"]), ([45], []))
        # The shared prefix and the AR token, teacher-forced with the top 20.
        self.assertEqual(server.bodies[0]["prompt"], PROMPT + [10, 11, 12])
        self.assertEqual(server.bodies[0]["prompt_logprobs"], 20)
        # The speculative token preferred by at most the limit is a near tie too.
        self.assertEqual(self.probed({14: (-0.5, 1)}, ar=(-0.7, 2))[0]["items"][0]["verdict"], "near_tie")

    def test_a_gap_beyond_the_limit_is_decisive(self):
        result, verdict, _ = self.probed({14: (-1.0000001, 2)})
        self.assertEqual(result["items"][0]["verdict"], "decisive")
        self.assertEqual(verdict["decisive_lost"], [45])

    def test_a_branch_token_outside_the_top_20_is_decisive_however_close(self):
        # The actual prompt token is always listed; unranked or past rank 20
        # it is outside the top 20, so a 0.055-nat gap does not exempt it.
        for ar, spec in (((-0.755, None), (-0.7, 1)), ((-0.755, 21), (-0.7, 1)), ((-0.7, 1), (-0.755, 21))):
            with self.subTest(ar=ar, spec=spec):
                item = self.probed({14: spec}, ar=ar)[0]["items"][0]
                self.assertEqual(item["verdict"], "decisive")
                self.assertIn("outside the top 20", item["reason"])
        item = self.probed({15: (-0.8, 2)})[0]["items"][0]
        self.assertEqual((item["verdict"], item["gap"]), ("decisive", None))

    def test_an_unmeasured_lost_item_makes_the_pair_insufficient(self):
        server = FakeServer({14: (-0.9, 2)})
        with tempfile.TemporaryDirectory() as directory:
            run = Run(directory, {"spec": [45]})
            (Path(directory) / "spec/responses/45.json").unlink()
            result = run.probe(server)
            self.assertEqual(result["items"][0]["verdict"], "unmeasured")
            # One lost item is within the one-loss tolerance only when measured.
            with self.assertRaisesRegex(ValueError, r"insufficient evidence: .*\[45\]"):
                run.pair(result)
        # A failed request measures nothing either.
        failed = FakeServer({14: (-0.9, 2)})
        failed.request = lambda path, body: (_ for _ in ()).throw(ValueError("HTTP 500"))
        with tempfile.TemporaryDirectory() as directory:
            run = Run(directory, {"spec": [45]})
            result = run.probe(failed)
            self.assertEqual((result["items"][0]["verdict"], result["items"][0]["reason"]), ("unmeasured", "HTTP 500"))
            with self.assertRaisesRegex(ValueError, "insufficient evidence"):
                run.pair(result)

    def test_responses_must_be_the_ones_the_receipts_scored(self):
        server = FakeServer({14: (-0.9, 2)})
        cases = {"replaced": lambda run, d: (Path(d) / "spec/responses/45.json").write_text(
                     json.dumps(retained(SPEC, content="#### 2"))),
                 "unbound": lambda run, d: run.write("spec", {**run.receipt("spec"), "rows": [
                     {key: value for key, value in row.items() if key != "response_sha256"}
                     for row in run.receipt("spec")["rows"]]})}
        for name, change in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                run = Run(directory, {"spec": [45]})
                change(run, directory)
                result = run.probe(server)
                self.assertEqual(result["items"][0]["verdict"], "unmeasured")
                self.assertIn("retained response", result["items"][0]["reason"])
                with self.assertRaisesRegex(ValueError, "insufficient evidence"):
                    run.pair(result)
        # A response bound by digest must still read as its scored row does.
        with tempfile.TemporaryDirectory() as directory:
            run = Run(directory, {"spec": [45]})
            value = run.receipt("spec")
            value["rows"][45]["actual"] = "3"
            run.write("spec", value)
            item = run.probe(server)["items"][0]
            self.assertEqual(item["verdict"], "unmeasured")
            self.assertIn("does not read as its scored row", item["reason"])

    def test_the_pair_judges_the_recorded_measurements_not_the_verdicts(self):
        server = FakeServer({14: (-0.9, 2)})
        with tempfile.TemporaryDirectory() as directory:
            run = Run(directory, {"spec": [45, 46]})
            result = run.probe(server)
            self.assertEqual(run.pair(result)["near_tie_lost"], [45, 46])
            for change, message in (
                    (lambda r: r["items"][0].update(spec_logprob=-1.7), "decisively"),
                    (lambda r: r["items"][0].update(spec_rank=21), "decisively"),
                    (lambda r: r["items"][0].update(responses={"ar": "0" * 64, "spec": "0" * 64}),
                     "other responses"),
                    (lambda r: r.update(near_tie_nats=0.5), "rule"),
                    (lambda r: r.update(top_logprobs=50), "rule")):
                tampered = json.loads(json.dumps(result))
                tampered["items"][1].update(spec_logprob=-1.7)
                change(tampered)
                with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                    run.pair(tampered)

    def test_the_probe_covers_every_discordant_item_and_binds_both_receipts(self):
        server = FakeServer({14: (-0.9, 2)})
        with tempfile.TemporaryDirectory() as directory:
            run = Run(directory, {"ar": [7], "spec": [45]})
            result = run.probe(server)
            self.assertEqual(result["ar"]["sha256"], probe.digest(run.paths["ar"]))
            self.assertEqual(result["items"][1]["responses"]["spec"], run.receipt("spec")["rows"][45]["response_sha256"])
            pair = run.pair(result)
        self.assertEqual((result["lost"], result["gained"]), ([45], [7]))
        self.assertEqual([(item["index"], item["outcome"], item["verdict"]) for item in result["items"]],
                         [(7, "gained", "near_tie"), (45, "lost", "near_tie")])
        self.assertEqual((pair["decisive_lost"], pair["near_tie_lost"]), ([], [45]))


if __name__ == "__main__":
    unittest.main()

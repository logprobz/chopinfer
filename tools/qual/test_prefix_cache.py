"""CPU checks for the prefix-cache qualification record and comparison."""

from qual_test_config import configure_receipts as setUpModule

import unittest
import copy
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import release_gate as gate
import declaration

from prefix_cache import pair_stats, scenarios, text


def row(tokens, logprobs):
    return {"tokens": tokens, "logprobs": logprobs}


def record():
    value = {"passed": True, "server": {"model": "fixture"}, "run_id": "same",
             "lengths": [1024, 32768, 131072], "answer_tokens": 8, "requests": []}
    for scenario, length, name, _ in scenarios(value["lengths"], "same", 8):
        value["requests"].append({"scenario": scenario, "length": length, "name": name,
            "messages_sha256": f"{scenario}-{length}-{name}", "tokens": ["a", "b"],
            "logprobs": [-0.5, -0.7], "prompt_tokens": length, "cached_tokens": 0,
            "ttft_seconds": 1.0})
    return value


class PrefixCacheTests(unittest.TestCase):
    def test_unrelated_and_unscored_floors_fail_cli_and_gate(self):
        baseline = record()
        candidate = copy.deepcopy(baseline)
        for row in candidate["requests"]:
            row.update(cached_tokens=16, ttft_seconds=0.5)
        variants = []
        floor = copy.deepcopy(baseline)
        floor["requests"][0]["messages_sha256"] = "unrelated"
        variants.append((candidate, floor))
        floor = copy.deepcopy(baseline)
        floor["requests"].reverse()
        variants.append((candidate, floor))
        floor = copy.deepcopy(baseline)
        floor["requests"][0]["logprobs"] = []
        variants.append((candidate, floor))
        floor = copy.deepcopy(baseline)
        floor["answer_tokens"] = 9
        variants.append((candidate, floor))
        floor = copy.deepcopy(baseline)
        divergent = copy.deepcopy(candidate)
        for value in (floor, divergent):
            for row in value["requests"]:
                row["tokens"] = ["x", "y"]
        variants.append((divergent, floor))
        floor = copy.deepcopy(baseline)
        floor["requests"].pop()
        variants.append((candidate, floor))
        floor = copy.deepcopy(baseline)
        floor["requests"][0]["logprobs"][0] = float("nan")
        variants.append((candidate, floor))
        config = declaration.load(declaration.path("mimo26"))
        check = next(c for c in config["checks"] if c["id"] == "prefix_cache")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / (name + ".json") for name in ("baseline", "candidate", "floor")]
            paths[0].write_text(json.dumps(baseline))
            for index, (warm, floor) in enumerate([(candidate, baseline), *variants]):
                paths[1].write_text(json.dumps(warm))
                paths[2].write_text(json.dumps(floor))
                result = subprocess.run([sys.executable, "-B", str(gate.ROOT / "tools/qual/prefix_cache.py"),
                    "compare", "--baseline", str(paths[0]), "--candidate", str(paths[1]), "--floor", str(paths[2])],
                    text=True, capture_output=True)
                self.assertEqual(result.returncode, 0 if index == 0 else 1, result.stdout + result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["passed"], index == 0)
                self.assertEqual(all(gate.judge(report, rule) for rule in check["rules"]), index == 0)
            config["checks"] = [check]
            for profile in config["profiles"].values():
                profile["checks"] = ["prefix_cache"]
            for valid in (True, False):
                paths[1].write_text(json.dumps(candidate))
                paths[2].write_text(json.dumps(baseline if valid else variants[0][1]))
                output = root / str(valid)
                with patch.object(gate.Client, "identity", return_value={"model": "fixture"}), contextlib.redirect_stdout(io.StringIO()):
                    code = gate.run(config, declaration.path("mimo26"), "full", "http://unused", None,
                        dict(zip(("prefix_baseline", "prefix_candidate", "prefix_floor"), map(str, paths))), output)
                self.assertEqual(code, 0 if valid else 1)
                summary = json.loads(next(output.glob("*/summary.json")).read_text())
                self.assertEqual(summary["qualified"], valid)

    def test_follow_up_ttft_carries_the_arms_host_load_label(self):
        """The compared TTFT is timing evidence only when both arms ran on an
        idle host; the label never changes `passed` or the TTFT rows."""
        import common
        idle = common.host_load([{"load_average": [1, 1, 1], "cpu_idle_ticks": 0, "cpu_ticks": 0},
                                 {"load_average": [1, 1, 1], "cpu_idle_ticks": 990, "cpu_ticks": 1000}], cpus=48)
        busy = common.host_load([{"load_average": [70, 70, 70], "cpu_idle_ticks": 0, "cpu_ticks": 0},
                                 {"load_average": [70, 70, 70], "cpu_idle_ticks": 80, "cpu_ticks": 1000}], cpus=48)
        baseline = record()
        candidate = copy.deepcopy(baseline)
        for row in candidate["requests"]:
            row.update(cached_tokens=16, ttft_seconds=0.5)
        reports = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label, (cold, warm) in {"idle": (idle, idle), "busy": (idle, busy), "old": (None, idle)}.items():
                for value, load in ((baseline, cold), (candidate, warm)):
                    value.pop("host_load", None)
                    if load is not None:
                        value["host_load"] = load
                (root / "baseline.json").write_text(json.dumps(baseline))
                (root / "candidate.json").write_text(json.dumps(candidate))
                result = subprocess.run([sys.executable, "-B", str(gate.ROOT / "tools/qual/prefix_cache.py"),
                    "compare", "--baseline", str(root / "baseline.json"), "--candidate", str(root / "candidate.json")],
                    text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                reports[label] = json.loads(result.stdout)
        self.assertEqual(reports["idle"]["host_load"]["timing"], "evidence")
        self.assertEqual(reports["busy"]["host_load"]["timing"], "informational")
        self.assertTrue(reports["busy"]["host_load"]["reasons"][0].startswith("candidate: "))
        self.assertEqual((reports["old"]["host_load"]["timing"], reports["old"]["host_load"]["reasons"]),
                         ("informational", ["baseline: no host load recorded"]))
        for report in reports.values():
            self.assertTrue(report["passed"])
            self.assertEqual(report["follow_up_ttft"], reports["idle"]["follow_up_ttft"])
            self.assertEqual(len(report["follow_up_ttft"]), 3)

    def test_empty_common_prefix_has_no_numeric_evidence(self):
        stats = pair_stats([row(["a"], [-0.5])], [row(["b"], [-0.5])])
        self.assertIsNone(stats["mean_abs_dlogprob"])
        self.assertIsNone(stats["max_abs_dlogprob"])

    def test_text_is_deterministic_and_salted(self):
        self.assertEqual(text(160, 3), text(160, 3))
        self.assertNotEqual(text(160, 3), text(160, 4))
        self.assertEqual(len(text(160, 3).splitlines()), 10)

    def test_scenarios_cover_follow_ups_system_prompts_and_partial_blocks(self):
        rows = list(scenarios([64], "run", 8))
        names = [(scenario, name) for scenario, _, name, _ in rows]
        self.assertEqual(names[:5], [("multi_turn", "turn1"), ("multi_turn", "turn2"),
                                     ("system", "question0"), ("system", "question1"),
                                     ("system", "question2")])
        self.assertEqual([name for scenario, _, name, _ in rows if scenario == "partial"],
                         ["tail0", "tail1", "tail2"])
        follow_up = rows[1][3]({"turn1": "amber"})
        self.assertEqual(follow_up[0], rows[0][3][0])
        self.assertEqual(follow_up[1], {"role": "assistant", "content": "amber"})

    def test_pair_stats_scores_only_the_common_prefix(self):
        left = [row([1, 2, 3], [-0.1, -0.2, -0.3]), row([4, 5], [-1.0, -2.0])]
        right = [row([1, 2, 9], [-0.1, -0.25, -5.0]), row([4, 5], [-1.0, -2.0])]
        stats = pair_stats(left, right)
        self.assertEqual(stats["exact_requests"], 1)
        self.assertEqual(stats["mean_common_prefix"], 2)
        self.assertAlmostEqual(stats["max_abs_dlogprob"], 0.05)
        self.assertAlmostEqual(stats["mean_abs_dlogprob"], 0.0125)


if __name__ == "__main__":
    unittest.main()

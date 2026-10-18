"""CPU checks of the KV layout served-bits equality check."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from qual_test_config import setUpModule  # noqa: F401
import kv_layout_equality
import release_preview as runner
from test_determinism import recorded

QUAL = Path(__file__).resolve().parent


def server(kv_cache_dtype, package="b"):
    """A bound server identity of one build, serving `kv_cache_dtype`."""
    serving = {
        "checkpoint_identity": "a" * 64, "package_identity": package * 64,
        "metadata_identity": "c" * 64, "startup_nonce": "d" * 32,
        "source_commit": "e" * 40, "bundle_id": "fixture.v1", "version": "1",
        "served_model_names": ["fixture"],
        "profile": {"name": "tp4", "tensor_parallel_size": 4, "kv_cache_dtype": kv_cache_dtype,
                    "max_sequences": 32, "prefix_cache": True,
                    "speculation": {"mode": "adaptive", "drafter": "fixture", "tokens": 5}}}
    return {"model": "fixture", "health": {"serving": serving},
            "checkpoint": {key: serving[key] for key in (
                "checkpoint_identity", "package_identity", "metadata_identity", "startup_nonce")}}


def records():
    """A native and a comparison record of the same answers in one run."""
    native = dict(recorded(image=False, arm="kv-native"), server=server("auto"))
    comparison = dict(recorded(image=False, arm="kv-comparison"), server=server("bf16", package="f"))
    return native, comparison


class CompareTests(unittest.TestCase):
    def test_identical_answers_from_two_layouts_pass(self):
        result = kv_layout_equality.compare(*records())
        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(result["equal"])
        self.assertIsNone(result["first_difference"])
        self.assertEqual(result["kv_cache_dtypes"], ["auto", "bf16"])
        # The package may follow the KV layout; both are reported.
        self.assertEqual(result["package_identities"], ["b" * 64, "f" * 64])
        self.assertEqual((result["requests"], result["tokens"]), (11, 29))

    def test_one_ulp_logprob_difference_fails_at_its_request_and_position(self):
        native, comparison = records()
        value = comparison["requests"][2]["logprobs"][1]
        comparison["requests"][2]["logprobs"][1] = math.nextafter(value, 0.0)
        result = kv_layout_equality.compare(native, comparison)
        self.assertFalse(result["passed"])
        self.assertEqual({key: result["first_difference"][key] for key in ("request", "field", "position")},
                         {"request": "fibonacci", "field": "logprobs", "position": 1})
        self.assertIn("kv-native and kv-comparison differ at fibonacci logprobs position 1", result["errors"][0])

    def test_a_token_difference_fails_at_its_first_position(self):
        native, comparison = records()
        comparison["requests"][0]["token_ids"][2] += 1
        result = kv_layout_equality.compare(native, comparison)
        self.assertFalse(result["passed"])
        self.assertEqual((result["first_difference"]["field"], result["first_difference"]["position"]),
                         ("token_ids", 2))

    def test_records_must_name_one_build_and_profile_in_one_run_with_two_layouts(self):
        native, comparison = records()
        for mutate, error in (
                (lambda value: value["server"]["health"]["serving"]["profile"].update(kv_cache_dtype="auto"),
                 "both records serve KV dtype auto"),
                (lambda value: value["server"]["health"]["serving"]["profile"].update(max_sequences=16),
                 "the records differ in profile max_sequences"),
                (lambda value: value["server"]["health"]["serving"].update(source_commit="9" * 40),
                 "the records differ in serving identity source_commit"),
                (lambda value: value.update(run_id="another"), "the records come from different runs")):
            changed = copy.deepcopy(comparison)
            mutate(changed)
            result = kv_layout_equality.compare(native, changed)
            self.assertFalse(result["passed"])
            self.assertIn(error, result["errors"])
            # A refused pair is never compared.
            self.assertNotIn("equal", result)
        # A record without a bound serving identity is refused.
        unbound = dict(comparison, server={"model": "fixture"})
        with self.assertRaises(SystemExit):
            kv_layout_equality.compare(native, unbound)


class CommandTests(unittest.TestCase):
    def test_compare_writes_a_receipt_and_exits_with_the_result(self):
        native, comparison = records()
        failing = copy.deepcopy(comparison)
        failing["requests"][0]["tokens"][0] += "x"
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for name, value in (("native", native), ("comparison", comparison), ("failing", failing)):
                paths.append(Path(directory) / f"{name}.json")
                paths[-1].write_text(json.dumps(value))
            receipts = Path(directory) / "receipts"
            env = dict(os.environ, CHOPINFER_QUAL_RECEIPTS=str(receipts))
            for other, code in ((paths[1], 0), (paths[2], 1)):
                result = subprocess.run([sys.executable, "-B", str(QUAL / "kv_layout_equality.py"), "compare",
                                         str(paths[0]), str(other)], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(json.loads(result.stdout.splitlines()[-1])["passed"], code == 0)
            written = sorted(receipts.glob("*-kv-layout-equality.json"))
            self.assertEqual(len(written), 2)
            value = json.loads(written[0].read_text())
            self.assertEqual(value["check"], "kv-layout-equality")


class RunnerTests(unittest.TestCase):
    def test_the_native_arm_replaces_the_configuration_kv_dtype_and_the_check_runs_offline(self):
        facts = runner.model_facts("glm5next")
        facts = dict(facts, full=["kv_layout_equality"],
                     checks={"kv_layout_equality": {"module": "kv_layout_equality",
                                                    "files": ["kv_layout_native", "kv_layout_comparison"]}})
        with patch.object(runner, "module_records", lambda name, model=None: (
                kv_layout_equality.RUNNER_ARMS, kv_layout_equality.RUNNER_RECORDS)):
            arms = runner.plan(facts, "full", "default", {"serve": ["--kv-cache-dtype", "bf16"],
                                                          "environment": {}, "features": []})
        by_name = {arm["arm"]: arm for arm in arms}
        self.assertEqual(by_name["kv-native"]["serve"], ["--kv-cache-dtype", "auto"])
        self.assertEqual(by_name["spec"]["serve"], ["--kv-cache-dtype", "bf16"])
        self.assertEqual([step["input"] for step in by_name["kv-native"]["steps"]], ["kv_layout_native"])
        self.assertIn("kv_layout_comparison", [step.get("input") for step in by_name["spec"]["steps"]])
        self.assertIn("kv_layout_equality", [step.get("check") for step in by_name["offline"]["steps"]])


if __name__ == "__main__":
    unittest.main()

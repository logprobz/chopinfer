"""CPU checks that the CLI and preview use the same numerical comparison."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import common
import spec_verify


class ComparisonTests(unittest.TestCase):
    def test_cli_matches_shared_comparison_and_exit_status(self):
        ar = {'arm': 'chopinfer-ar', 'run_id': 'test', 'output_tokens': 64, 'sampling': {'temperature': 0},
              'sets': {'copy': {'prompts_sha256': 'same', 'speculation': {},
                       'tokens_per_engine_step': 1,
                       'requests': [{'tokens': ['a', 'b'], 'logprobs': [0.0, 0.0],
                                     'top': [{'a': -0.25}, {'b': -0.5}]}]}}}
        for delta, passed in ((0.0, True), (0.15, True), (0.151, False)):
            spec = copy.deepcopy(ar)
            spec["arm"] = "chopinfer-spec"
            spec["sets"]["copy"]["speculation"] = {"draft_tokens": 1, "verify_items": 1}
            spec['sets']['copy']['requests'][0]['logprobs'] = [-delta, -delta]
            with tempfile.TemporaryDirectory() as tmp:
                paths = [Path(tmp) / name for name in ('ar.json', 'spec.json')]
                for path, value in zip(paths, (ar, spec)):
                    path.write_text(json.dumps(value))
                args = SimpleNamespace(ar=str(paths[0]), spec=str(paths[1]),
                                       maximum_abs_dlogprob=0.15, maximum_top1_disagreement=0.08)
                output = io.StringIO()
                with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
                    spec_verify.compare(args)
                self.assertEqual(stopped.exception.code, 0 if passed else 1)
                result = json.loads(output.getvalue())
                self.assertEqual(result, spec_verify.compare_records(spec, ar))
                self.assertEqual(result['passed'], passed)

    def test_ms_per_token_carries_both_records_host_load_label(self):
        stall = {"cpu": 0, "io": 0}
        idle = common.host_load([{"at": 0, "uptime_seconds": 0, "load_average": [1, 1, 1], "cpu_idle_ticks": 0, "cpu_ticks": 0,
                                  "pressure_some_us": stall},
                                 {"at": 1, "uptime_seconds": 1, "load_average": [1, 1, 1], "cpu_idle_ticks": 990, "cpu_ticks": 1000,
                                  "pressure_some_us": stall}], cpus=48)
        busy = common.host_load([{"at": 0, "uptime_seconds": 0, "load_average": [70, 70, 70], "cpu_idle_ticks": 0, "cpu_ticks": 0,
                                  "pressure_some_us": stall},
                                 {"at": 1, "uptime_seconds": 1, "load_average": [70, 70, 70], "cpu_idle_ticks": 80, "cpu_ticks": 1000,
                                  "pressure_some_us": stall}], cpus=48)
        spec, ar = copy_records([(["a", "b"], ["a", "b"])])
        for record in (spec, ar):
            record["sets"]["copy"]["ms_per_token"] = 20.0
        base = spec_verify.compare_records(spec, ar)
        for spec_load, ar_load, timing, reasons in (
                (idle, idle, "evidence", []),
                (busy, idle, "informational", ["spec: "]),
                (idle, None, "informational", ["ar: no host load recorded"])):
            with self.subTest(spec=spec_load is idle, ar=ar_load):
                left, right = dict(spec), dict(ar)
                for record, load in ((left, spec_load), (right, ar_load)):
                    if load is not None:
                        record["host_load"] = load
                result = spec_verify.compare_records(left, right)
                label = result["host_load"]
                self.assertEqual(label["timing"], timing)
                self.assertEqual(len(label["reasons"]), len(reasons))
                for reason, prefix in zip(label["reasons"], reasons):
                    self.assertTrue(reason.startswith(prefix), reason)
                # The label changes no verdict and no reported figure.
                self.assertIs(result["passed"], base["passed"])
                self.assertEqual(result["sets"], base["sets"])



def copy_records(requests):
    """A speculative and an AR record of one copy set, from (spec tokens, AR
    tokens) per request; every position has logprob 0 and itself as top-1."""
    def group(rows, drafting):
        return {"prompts_sha256": "same", "tokens_per_engine_step": 1,
                "speculation": {"draft_tokens": 1, "verify_items": 1} if drafting else {},
                "requests": [{"tokens": tokens, "logprobs": [0.0] * len(tokens),
                              "top": [{token: 0.0} for token in tokens]} for tokens in rows]}
    ar = {"arm": "chopinfer-ar", "run_id": "test", "output_tokens": 64, "sampling": {},
          "sets": {"copy": group([right for _, right in requests], False)}}
    spec = dict(ar, arm="chopinfer-spec", sets={"copy": group([left for left, _ in requests], True)})
    return spec, ar


class CoverageTests(unittest.TestCase):
    """A request scores only its common prefix and first divergence; the rest
    of it must show as unscored, per request and in aggregate."""

    def test_early_divergence_cannot_claim_complete_coverage(self):
        same = [f"t{index}" for index in range(64)]
        requests = [(same, same)] * 15 + [(["x"] + same[1:], same)]
        spec, ar = copy_records(requests)
        result = spec_verify.compare_records(spec, ar)
        copy_set = result["sets"]["copy"]
        # The aggregate bounds still pass: one divergence is one top-1
        # disagreement in 961 scored positions.
        self.assertTrue(result["passed"])
        self.assertEqual(copy_set["scored_positions"], 15 * 64 + 1)
        coverage = copy_set["coverage"]
        self.assertFalse(coverage["complete"])
        self.assertEqual((coverage["emitted_positions"], coverage["scored_positions"],
                          coverage["unscored_positions"]), (1024, 961, 63))
        self.assertEqual(coverage["complete_requests"], 15)
        self.assertAlmostEqual(coverage["fraction"], 961 / 1024)
        self.assertEqual(coverage["requests"][15], {"emitted": 64, "ar_emitted": 64, "common_prefix": 0,
                                                    "scored": 1, "unscored": 63})
        self.assertEqual(coverage["requests"][0], {"emitted": 64, "ar_emitted": 64, "common_prefix": 64,
                                                   "scored": 64, "unscored": 0})

    def test_identical_requests_cover_every_position(self):
        spec, ar = copy_records([(["a", "b"], ["a", "b"])] * 2)
        coverage = spec_verify.compare_records(spec, ar)["sets"]["copy"]["coverage"]
        self.assertTrue(coverage["complete"])
        self.assertEqual((coverage["scored_positions"], coverage["unscored_positions"], coverage["fraction"]),
                         (4, 0, 1.0))

    def test_a_position_the_control_never_emitted_is_not_scored(self):
        # The AR request stopped after "b": the speculative "c" has no AR
        # position to compare with, so it is neither scored nor counted in
        # the top-1 denominator.
        spec, ar = copy_records([(["a", "b", "c"], ["a", "b"])])
        copy_set = spec_verify.compare_records(spec, ar)["sets"]["copy"]
        self.assertEqual(copy_set["scored_positions"], 2)
        self.assertEqual(copy_set["top1_disagreement"], 0)
        self.assertEqual(copy_set["coverage"]["requests"][0], {"emitted": 3, "ar_emitted": 2, "common_prefix": 2,
                                                               "scored": 2, "unscored": 1})
        self.assertFalse(copy_set["coverage"]["complete"])
        # A shorter speculative request scores all of its own positions but
        # leaves the control's extra position unscored.
        spec, ar = copy_records([(["a", "b"], ["a", "b", "c"])])
        coverage = spec_verify.compare_records(spec, ar)["sets"]["copy"]["coverage"]
        self.assertEqual((coverage["scored_positions"], coverage["unscored_positions"], coverage["complete"]),
                         (2, 0, False))
        self.assertEqual(coverage["ar_unscored_positions"], 1)

    def test_the_kind_refuses_a_report_that_overstates_coverage(self):
        import kinds
        same = [f"t{index}" for index in range(64)]
        spec, ar = copy_records([(same, same)] * 15 + [(["x"] + same[1:], same)])
        report = spec_verify.compare_records(spec, ar)
        kind = kinds.KINDS["spec_verify"]
        self.assertTrue(kind.score(report)["passed"])
        # Reports written before the coverage block still score.
        legacy = copy.deepcopy(report)
        del legacy["sets"]["copy"]["coverage"]
        self.assertTrue(kind.score(legacy)["passed"])
        for change in (
                lambda coverage: coverage.update(complete=True),
                lambda coverage: coverage.update(unscored_positions=0),
                lambda coverage: coverage.update(complete_requests=16),
                lambda coverage: coverage["requests"][15].update(scored=64, unscored=0),
                lambda coverage: coverage["requests"].pop()):
            forged = copy.deepcopy(report)
            change(forged["sets"]["copy"]["coverage"])
            with self.subTest(forged["sets"]["copy"]["coverage"].get("complete")), \
                    self.assertRaisesRegex(kinds.Insufficient, "inconsistent coverage"):
                kind.score(forged)


def bound(record, identity="a" * 64, model="served", tokens=0, tensor_parallel_size=2, kv="bf16", package=None):
    """`record` as a bound arm: its server's pinned checkpoint and serving profile."""
    record = copy.deepcopy(record)
    nonce = "1" * 32 if record["arm"] == "chopinfer-spec" else "0" * 32
    checkpoint = {"checkpoint_identity": identity, "startup_nonce": nonce}
    if package is not None:
        checkpoint["package_identity"] = package
    record["server"] = {"model": model, "checkpoint": checkpoint}
    record["serving"] = {"tensor_parallel_size": tensor_parallel_size, "kv_cache_dtype": kv,
                         "speculation": {"mode": "adaptive" if tokens else "off", "tokens": tokens,
                                         "drafter": "mtp" if tokens else None}}
    return record


class IdentityTests(unittest.TestCase):
    """Both arms must come from one bound checkpoint, name and topology."""

    def setUp(self):
        self.ar = {'arm': 'chopinfer-ar', 'run_id': 'test', 'output_tokens': 64, 'sampling': {},
                   'sets': {'copy': {'prompts_sha256': 'same', 'speculation': {}, 'tokens_per_engine_step': 1,
                                     'requests': [{'tokens': ['a'], 'logprobs': [0.0], 'top': [{'a': 0.0}]}]}}}
        self.spec = copy.deepcopy(self.ar)
        self.spec["arm"] = "chopinfer-spec"
        self.spec["sets"]["copy"]["speculation"] = {"draft_tokens": 1, "verify_items": 1}

    def test_bound_arms_of_one_checkpoint_compare(self):
        result = spec_verify.compare_records(bound(self.spec, tokens=3), bound(self.ar))
        self.assertTrue(result["passed"])
        self.assertEqual((result["bound"]["checkpoint_identity"], result["bound"]["model"]), ("a" * 64, "served"))
        self.assertEqual(result["bound"]["serving"]["spec"]["speculation"]["tokens"], 3)
        self.assertIsNone(spec_verify.compare_records(self.spec, self.ar)["bound"])

    def test_every_served_kv_element_binds(self):
        # The host labels its served element bf16, fp8 or mixed (a
        # mixed-plane layout); both arms must serve the same one, from one
        # package, which fixes the layout the label selects.
        package = "c" * 64
        for kv in ("bf16", "fp8", "mixed"):
            with self.subTest(kv):
                result = spec_verify.compare_records(bound(self.spec, tokens=3, kv=kv, package=package),
                                                     bound(self.ar, kv=kv, package=package))
                self.assertEqual(result["bound"]["serving"]["ar"]["kv_cache_dtype"], kv)
        with self.assertRaisesRegex(ValueError, "no KV cache dtype"):
            spec_verify.compare_records(bound(self.spec, tokens=3, kv="fp4"), bound(self.ar, kv="fp4"))
        with self.assertRaises(ValueError):
            spec_verify.compare_records(bound(self.spec, tokens=3, kv="mixed", package=package),
                                        bound(self.ar, package=package))
        for spec, ar, message in (
                # A mixed-plane label alone names no layout.
                (bound(self.spec, tokens=3, kv="mixed"), bound(self.ar, kv="mixed"), "names no package identity"),
                (bound(self.spec, tokens=3, kv="mixed", package=package),
                 bound(self.ar, kv="mixed", package="d" * 64), "different packages"),
                (bound(self.spec, tokens=3, package=package), bound(self.ar), "different packages"),
                (bound(self.spec, tokens=3, package="short"), bound(self.ar), "malformed package identity")):
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                spec_verify.compare_records(spec, ar)

    def test_mismatched_arms_are_refused(self):
        for spec, ar, message in (
                (bound(self.spec, tokens=3, identity="b" * 64), bound(self.ar), "different checkpoints"),
                (bound(self.spec, tokens=3, model="other"), bound(self.ar), "different checkpoints or names"),
                (bound(self.spec, tokens=3), self.ar, "only one arm is bound"),
                (self.spec, bound(self.ar), "only one arm is bound"),
                (bound(self.spec, tokens=3, tensor_parallel_size=4), bound(self.ar), "different topologies"),
                (bound(self.spec), bound(self.ar), "speculative arm must draft"),
                (bound(self.spec, tokens=3), bound(self.ar, tokens=3), "control must not")):
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                spec_verify.compare_records(spec, ar)

    def test_incomplete_bound_records_are_refused(self):
        def without(record, *path):
            record = copy.deepcopy(record)
            parent = record
            for key in path[:-1]:
                parent = parent[key]
            del parent[path[-1]]
            return record

        def changed(record, value, *path):
            record = copy.deepcopy(record)
            parent = record
            for key in path[:-1]:
                parent = parent[key]
            parent[path[-1]] = value
            return record

        spec, ar = bound(self.spec, tokens=3), bound(self.ar)
        stripped = (spec, ar)
        for arm in range(2):
            stripped = tuple(without(record, "server", "checkpoint", "startup_nonce") if index == arm else record
                             for index, record in enumerate(stripped))
        stripped = tuple(without(without(record, "serving", "tensor_parallel_size"), "serving", "kv_cache_dtype")
                         for record in stripped)
        for (spec_record, ar_record), message in (
                (stripped, "no startup nonce"),
                ((without(spec, "server", "checkpoint", "startup_nonce"), ar), "no startup nonce"),
                ((changed(spec, "short", "server", "checkpoint", "checkpoint_identity"), ar), "no checkpoint identity"),
                ((spec, without(ar, "server", "model")), "no served model"),
                ((spec, without(ar, "serving")), "lacks the serving topology"),
                ((without(spec, "serving", "tensor_parallel_size"), without(ar, "serving", "tensor_parallel_size")),
                 "no tensor parallel size"),
                ((spec, changed(ar, "2", "serving", "tensor_parallel_size")), "no tensor parallel size"),
                ((without(spec, "serving", "kv_cache_dtype"), without(ar, "serving", "kv_cache_dtype")),
                 "no KV cache dtype"),
                ((changed(spec, {"mode": "off", "tokens": 3}, "serving", "speculation"), ar),
                 "no consistent speculation"),
                ((spec, without(ar, "serving", "speculation")), "no consistent speculation"),
                ((spec, changed(ar, "1" * 32, "server", "checkpoint", "startup_nonce")), "one server process")):
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                spec_verify.compare_records(spec_record, ar_record)

    def test_record_samples_host_load_around_its_requests(self):
        events = []

        def request(client, model, prompt, output_tokens):
            events.append("request")
            return {"tokens": ["x"], "logprobs": [-0.1], "top": [{}], "finish_reason": "length", "seconds": 0.1}

        def sample():
            events.append("sample")
            return {"sample": len(events)}

        args = SimpleNamespace(url="http://127.0.0.1:1", timeout=1, model=None, server_log=None, include_chat=False,
                               arm="chopinfer-spec", run_id="x", output_tokens=4)
        with mock.patch.object(spec_verify, "Client") as client, \
                mock.patch.object(spec_verify, "bind_checkpoint"), \
                mock.patch.object(spec_verify, "serving_profile", return_value={}), \
                mock.patch.object(spec_verify, "prompt_sets", return_value={"a": ["p", "q"], "b": ["r"]}), \
                mock.patch.object(spec_verify, "metrics_text", return_value=""), \
                mock.patch.object(spec_verify, "completion", request), \
                mock.patch.object(spec_verify, "confirm_live"), \
                mock.patch.object(spec_verify, "host_load_sample", sample), \
                mock.patch.object(spec_verify, "receipt") as receipt:
            client.return_value.identity.return_value = {"model": "fake"}
            spec_verify.record(args)
        self.assertEqual(events, ["sample", "request", "request", "request", "sample"])
        self.assertEqual(receipt.call_args.kwargs["load_samples"], [{"sample": 1}, {"sample": 5}])

    def test_record_refuses_a_server_not_bound_to_the_pin(self):
        import declaration
        gate = next(path for path in declaration.shipped() if path.name == declaration.NAME)
        pin = declaration.load(gate)["qualification"]["checkpoint"]
        server = {"url": "http://127.0.0.1:1", "model": pin["served_model_name"],
                  "health": {"serving": {"checkpoint_identity": "b" * 64, "startup_nonce": "c" * 32,
                                         "profile": {"tensor_parallel_size": 2}}}}
        args = SimpleNamespace(url=server["url"], timeout=1, model=None, server_log=None, include_chat=False,
                               arm="chopinfer-spec", run_id="x", output_tokens=4)
        with mock.patch.dict("os.environ", {"CHOPINFER_QUAL_CONFIG": str(gate)}), \
                mock.patch.object(spec_verify, "Client") as client:
            client.return_value.identity.return_value = server
            with self.assertRaisesRegex(SystemExit, "refused: the server loaded checkpoint b{64}, not the pinned"):
                spec_verify.record(args)
            client.return_value.request.assert_not_called()


if __name__ == '__main__':
    unittest.main()

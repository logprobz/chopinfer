"""CPU checks of the continuation scorer: every emitted position is scored on
its own history, coverage cannot be claimed for a request whose reference
rows read another history, and the entropy and KL reports follow the
reference rows."""

import copy
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import spec_continuation as continuation

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from tools.refrun import modes as reference_modes  # noqa: E402


def request(prompt, ids, logprob=-0.1):
    return {"prompt_ids": list(prompt), "ids": list(ids), "logprobs": [logprob] * len(ids),
            "finish_reason": "length"}


def served(model, identity, nonce, serving):
    """A record's server: bound to `identity` by a process with `nonce`, of
    one install (binary source commit, direct package and metadata)."""
    install = {"package_identity": "c" * 64, "metadata_identity": "e" * 64}
    return {"model": model, "checkpoint": {"checkpoint_identity": identity, "startup_nonce": nonce, **install},
            "health": {"serving": {"source_commit": "f" * 40, "checkpoint_identity": identity,
                                   "startup_nonce": nonce, "profile": copy.deepcopy(serving), **install}}}


def record(arm, sets):
    """An arm's record: name to (sampling, requests), bound to one checkpoint
    by its own server process of the shared install; the speculative arm
    drafted in every set and the control did not."""
    speculative = arm == "chopinfer-spec"
    drafting = {"draft_tokens": 1, "verify_items": 1} if speculative else {}
    serving = {"tensor_parallel_size": 4, "kv_cache_dtype": "bf16", "profile_name": "p", "max_sequences": 32,
               "speculation": {"mode": "fixed" if speculative else "off", "tokens": 7 if speculative else 0}}
    return {"arm": arm, "run_id": "cpu", "sets_sha256": "frozen",
            "server": served("served", "a" * 64, ("1" if speculative else "0") * 32, serving),
            "serving": serving,
            "sets": {name: {"sampling": dict(sampling), "requests": rows, "speculation": dict(drafting)}
                     for name, (sampling, rows) in sets.items()}}


# A pinned reference's binding block, as the reference runner's emit writes it.
BINDING = {"model": "m", "lock": {"path": "models/m/checkpoint-files.json", "variant": "v"},
           "key": {"checkpoint": {"source_lock_sha256": "1" * 64, "files_used": ["config.json"]},
                   "reference_code": {"files": {"m/modeling.py": "2" * 64}},
                   "adapter": {"files": {"models/m/reference.py": "3" * 64, "tools/refrun/run.py": "4" * 64}},
                   "environment": {"device": "sm_120", "torch": "t"}, "outputs": {"taps": None, "topk": 20}},
           "modes": {mode: {"mode": mode, "variant": variant, "dequant": dequant,
                            "policy_sha256": continuation.canonical_sha256(reference_modes.policy(mode, dequant))}
                     for mode, variant, dequant in (("f32", "exact", "rounded"), ("bf16", "bf16", "hf-bf16"),
                                                    ("f64", "f64", "exact"))}}


def mode_file(mode, entries, binding=None):
    """A reference-runner mode file of `entries` under the mode's pinned policy."""
    run = (binding or BINDING)["modes"][mode]
    return {"schema": "chopinfer.refrun.scores.v1", "policy": reference_modes.policy(run["mode"], run["dequant"]),
            "entries": entries}


def entry(name, row, logprob=-0.1, top=None, token_shift=0, mode="f32"):
    """A reference-runner entry that reads `row`'s own prompt and emitted IDs,
    keyed as BINDING's runs were. `top` gives each position's top-20 logprobs
    (default: the forced token holds nearly all the mass); `token_shift`
    forces other tokens."""
    prompt, ids = row["prompt_ids"], row["ids"]
    forced = [token + token_shift for token in ids]
    logprobs = top or [math.log(0.999)] + [math.log(0.001 / 19)] * 19
    topk_ids = [[token] + [token + 1000 + rank for rank in range(19)] for token in forced]
    key = {"model": BINDING["model"], **copy.deepcopy(BINDING["key"]),
           "mode": {"name": mode, "policy_sha256": BINDING["modes"][mode]["policy_sha256"]},
           "sequence": {"token_ids_sha256": continuation.ids_sha256(prompt), "score_from": len(prompt),
                        "answer": {"policy": "forced", "ids_sha256": continuation.ids_sha256(forced)}}}
    return {"name": name, "category": name.rsplit("-", 1)[0], "key": key,
            "key_sha256": continuation.canonical_sha256(key),
            "scores": {"score_from": len(prompt), "token": forced, "logprob": [logprob] * len(ids),
                       "top1": forced, "topk_ids": topk_ids, "topk_logits": [[-rank for rank in range(20)]] * len(ids),
                       "topk_logprobs": [logprobs] * len(ids)}}


def modes(value, **changes):
    """{mode: {name: entry}} for every request of a record, each mode the same
    rows unless `changes` names a mode's entry keyword arguments."""
    result = {}
    for mode in (continuation.GATE_MODE, *continuation.SPREAD_MODES):
        result[mode] = {}
        for name, group in value["sets"].items():
            for index, row in enumerate(group["requests"]):
                label = continuation.sequence_name(name, index)
                result[mode][label] = entry(label, row, mode=mode, **changes.get(mode, {}))
    return result


def gpu_execution(requests):
    """A reference run's execution record covering `requests` entries per mode."""
    return [{"schema": continuation.EXECUTION_SCHEMA, "exit": 0, "cuda_visible_devices": ["0", "1", "2", "3"],
             "gpus": [f"{index}, GPU, 1, 1 MiB, 0 MiB, 0 %" for index in range(4)],
             "entries": {mode: {"computed": requests, "cache_hits": 0, "cached": []}
                         for mode in continuation.MODE_VARIANTS}}]


def receipt(spec, ar, frozen=None):
    """The check's receipt over two scored arms, as `judge` writes it."""
    requests = sum(len(group["requests"]) for group in spec["_record"]["sets"].values())
    return {**continuation.compare(spec, ar, frozen), "spec": continuation.public(spec),
            "ar": continuation.public(ar),
            "execution": {"gpu": {"spec": gpu_execution(requests), "ar": gpu_execution(requests)},
                          "cpu_scoring": {"seconds": 0.1}}}


def without_frozen_inputs(config):
    """Why the frozen sets cannot be built under `config`: an input they read
    (the locked gsm8k examples under CHOPINFER_DATA) is absent, as in
    tests/qual-clean.sh. None when present; a present input must still match
    its lock."""
    import common
    with mock.patch.dict("os.environ", {"CHOPINFER_QUAL_CONFIG": str(config)}):
        for name in ("spec_teacher", "gsm8k"):
            path = common.resolve_input(name, optional=True)
            if path is None or not path.exists():
                return f"needs the locked {name} input (CHOPINFER_DATA)"
    return None


UNIFORM = [math.log(1 / 40)] * 20


class ContinuationTests(unittest.TestCase):
    def test_digest_matches_the_reference_runner(self):
        from tools.refrun import keys
        for ids in ([1, 2, 3], [151643], list(range(70))):
            self.assertEqual(continuation.ids_sha256(ids), keys.ids_sha256(ids))

    def test_workload_forces_each_request_on_its_own_prompt(self):
        value = record("chopinfer-spec", {"copy": (continuation.GREEDY, [request([5, 6], [7, 8, 9])])})
        workload = continuation.workload(value, "model")
        self.assertEqual(workload["sequences"], [{"name": "copy-000", "category": "copy", "token_ids": [5, 6],
                                                  "score_from": 2, "answer": {"policy": "forced",
                                                                              "ids": [7, 8, 9]}}])
        value["sets"]["sampled"] = {"sampling": continuation.SAMPLED, "requests": [request([5], [6])]}
        self.assertEqual([row["name"] for row in continuation.workload(value, "model", ["sampled"])["sequences"]],
                         ["sampled-000"])
        with self.assertRaisesRegex(ValueError, "no set"):
            continuation.workload(value, "model", ["open"])
        value["sets"]["copy"]["requests"][0]["ids"] = None
        with self.assertRaisesRegex(ValueError, "no prompt or emitted token IDs"):
            continuation.workload(value, "model")

    def test_early_divergence_scores_every_position_of_both_arms(self):
        # The arms diverge at their first token. spec_verify can score one
        # position of this request; here each arm scores all 64 of its own.
        prompt = [1, 2, 3]
        spec = record("chopinfer-spec", {"copy": (continuation.GREEDY, [request(prompt, [9] + list(range(10, 73)))])})
        ar = record("chopinfer-ar", {"copy": (continuation.GREEDY, [request(prompt, list(range(10, 74)))])})
        for value in (spec, ar):
            report = continuation.score(value, modes(value), BINDING)
            coverage = report["sets"]["copy"]["coverage"]
            self.assertEqual((coverage["emitted_positions"], coverage["scored_positions"], coverage["complete"]),
                             (64, 64, True))
            self.assertEqual(report["sets"]["copy"]["arm"]["mean_abs_dlogprob"], 0)
            self.assertEqual(report["sets"]["copy"]["arm"]["top1_disagreements"], 0)

    def test_a_reference_of_another_history_is_refused_not_shortened(self):
        value = record("chopinfer-spec", {"copy": (continuation.GREEDY, [request([1, 2], [3, 4]),
                                                                          request([1, 2], [5, 6])])})
        reference = modes(value)
        cases = {
            "the reference rows force other tokens": lambda rows: rows["copy-001"]["scores"].update(token=[5, 7]),
            "the reference read another history": lambda rows: rows["copy-001"]["key"]["sequence"].update(
                token_ids_sha256=continuation.ids_sha256([1, 9])),
            "no reference entry": lambda rows: rows.pop("copy-001"),
            "the reference rows do not cover every emitted token": lambda rows: rows["copy-001"]["scores"].update(
                logprob=[-0.1])}
        for reason, change in cases.items():
            for mode in reference:
                changed = copy.deepcopy(reference)
                change(changed[mode])
                with self.subTest(reason=reason, mode=mode):
                    report = continuation.score(value, changed, BINDING)
                    copy_set = report["sets"]["copy"]
                    self.assertEqual(copy_set["requests"][1]["refused"], reason)
                    self.assertEqual((copy_set["coverage"]["scored_positions"], copy_set["coverage"]["complete"],
                                      copy_set["coverage"]["complete_requests"]), (2, False, 1))
        short = copy.deepcopy(value)
        short["sets"]["copy"]["requests"][0]["logprobs"] = [-0.1]
        self.assertEqual(continuation.score(short, modes(short), BINDING)["sets"]["copy"]["requests"][0]["refused"],
                         "logprob rows differ from the emitted token IDs")

    def test_a_refused_request_is_never_complete_coverage(self):
        # Regression: a request that lost its emitted IDs
        # but kept its logprobs counted as complete, and the comparison passed.
        rows = [request([1, n], list(range(100, 164))) for n in range(17)]
        uniform = {"f32": {"top": UNIFORM}, "bf16": {"top": UNIFORM}}
        spec = record("chopinfer-spec", {"sampled": (continuation.SAMPLED, rows)})
        ar = record("chopinfer-ar", {"sampled": (continuation.SAMPLED, copy.deepcopy(rows))})
        reference = modes(spec, **uniform)
        for lost in (None, []):
            changed = copy.deepcopy(spec)
            changed["sets"]["sampled"]["requests"][-1]["ids"] = lost
            with self.subTest(ids=lost):
                report = continuation.score(changed, reference, BINDING)
                coverage = report["sets"]["sampled"]["coverage"]
                self.assertEqual(report["sets"]["sampled"]["requests"][-1]["refused"], "no emitted token IDs")
                self.assertEqual((coverage["complete"], coverage["complete_requests"], coverage["refused_requests"]),
                                 (False, 16, 1))
                result = continuation.compare(report, continuation.score(ar, modes(ar, **uniform), BINDING))
                self.assertEqual(result["status"], "insufficient")
                self.assertIn("sampled: incomplete coverage", result["insufficient"])
        # Neither may a request without logprobs or with a token that is not
        # an ID.
        for field, value in (("logprobs", None), ("ids", [100, "x"] + list(range(102, 164)))):
            changed = copy.deepcopy(spec)
            changed["sets"]["sampled"]["requests"][0][field] = value
            with self.subTest(field=field):
                self.assertFalse(continuation.score(changed, reference, BINDING)["sets"]["sampled"]["coverage"]["complete"])

    def test_greedy_flips_use_the_reference_top_twenty(self):
        value = record("chopinfer-spec", {"copy": (continuation.GREEDY, [request([1], [3, 4])])})
        reference = modes(value)
        rows = reference[continuation.GATE_MODE]["copy-000"]["scores"]
        # Position 0: the reference prefers 1003 by 0.25 nats (near tie);
        # position 1: a token outside its top 20 (decisive).
        rows["top1"] = [1003, 99]
        rows["topk_ids"] = [[1003, 3] + list(range(2000, 2018)), [99] + list(range(3000, 3019))]
        rows["topk_logits"] = [[0.0, -0.25] + [-1.0 - r for r in range(18)], [-float(r) for r in range(20)]]
        report = continuation.score(value, reference, BINDING)["sets"]["copy"]["arm"]
        self.assertEqual((report["top1_near_ties"], report["top1_decisive_flips"]), (1, 1))

    def test_high_entropy_positions_are_counted_and_scored(self):
        rows = [request([1, n], list(range(100, 164))) for n in range(16)]
        spec = record("chopinfer-spec", {"sampled": (continuation.SAMPLED, rows)})
        ar = record("chopinfer-ar", {"sampled": (continuation.SAMPLED, copy.deepcopy(rows))})
        for value in (spec, ar):
            for row in value["sets"]["sampled"]["requests"]:
                row["logprobs"] = [-0.09] * 64
        reference = {"spec": modes(spec, f32={"top": UNIFORM}, bf16={"top": UNIFORM}),
                     "ar": modes(ar, f32={"top": UNIFORM}, bf16={"top": UNIFORM})}
        scores = {arm: continuation.score(value, reference[arm], BINDING) for arm, value in (("spec", spec), ("ar", ar))}
        sampled = scores["spec"]["sets"]["sampled"]
        self.assertEqual(sampled["entropy"]["beyond_top20"]["positions"], 1024)
        self.assertEqual(sampled["entropy"]["top5"]["positions"], 0)
        self.assertAlmostEqual(sampled["kl"]["estimate"], 0.01)
        self.assertAlmostEqual(sampled["kl"]["total_variation_bound"], math.sqrt(0.01 / 2))
        self.assertNotIn("top1_disagreement", sampled["arm"])
        result = continuation.compare(scores["spec"], scores["ar"])
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["high_entropy_positions"], {"chopinfer-spec": 1024, "chopinfer-ar": 1024})
        self.assertTrue(result["passed"])
        # Fewer high-entropy positions than the large-set threshold leave the
        # distributional claim insufficient.
        few = copy.deepcopy(scores["spec"])
        few["_positions"]["sampled"] = few["_positions"]["sampled"][:999]
        self.assertEqual(continuation.compare(few, scores["ar"])["status"], "insufficient")

    def test_verdict_follows_the_control(self):
        rows = [request([1, n], list(range(100, 164))) for n in range(16)]
        spec = record("chopinfer-spec", {"sampled": (continuation.SAMPLED, rows)})
        ar = record("chopinfer-ar", {"sampled": (continuation.SAMPLED, copy.deepcopy(rows))})
        uniform = {"f32": {"top": UNIFORM}, "bf16": {"top": UNIFORM}}
        for row in spec["sets"]["sampled"]["requests"]:
            row["logprobs"] = [-0.5] * 64
        score_ar = continuation.score(ar, modes(ar, **uniform), BINDING)
        score_spec = continuation.score(spec, modes(spec, **uniform), BINDING)
        # The candidate sits 0.4 nats off where the control sits on the
        # reference: it fails.
        self.assertEqual(continuation.compare(score_spec, score_ar)["status"], "fail")
        # When the control misses the same limits, the set is insufficient.
        for row in ar["sets"]["sampled"]["requests"]:
            row["logprobs"] = [-0.5] * 64
        score_ar = continuation.score(ar, modes(ar, **uniform), BINDING)
        result = continuation.compare(score_spec, score_ar)
        self.assertEqual(result["status"], "insufficient")
        self.assertIn("sampled: the AR control misses the limits", result["insufficient"])

    def test_frozen_sets(self):
        import declaration
        gate = next(path for path in declaration.shipped() if path.name == declaration.NAME)
        reason = without_frozen_inputs(gate)
        if reason:
            self.skipTest(reason)
        with mock.patch.dict("os.environ", {"CHOPINFER_QUAL_CONFIG": str(gate)}):
            sets = continuation.prompt_sets()
        self.assertEqual({name: (len(group["requests"]), group["max_tokens"]) for name, group in sets.items()},
                         {"copy": (16, 64), "natural": (16, 64), "open": (16, 64), "sampled": (64, 128)})

    def test_both_arms_record_every_frozen_request(self):
        self.assertEqual(len(set(continuation.ENTROPY_PROMPTS)), 16)
        # Both arms must record every frozen request.
        frozen = {"copy": {"sampling": continuation.GREEDY, "max_tokens": 2,
                           "requests": [{"prompt": "a", "seed": None}, {"prompt": "b", "seed": None}]}}
        rows = [request([1], [2]), request([1], [3])]
        value = record("chopinfer-spec", {"copy": (continuation.GREEDY, rows)})
        control = record("chopinfer-ar", {"copy": (continuation.GREEDY, copy.deepcopy(rows))})
        for arm in (value, control):
            arm["sets_sha256"] = continuation.sets_digest(frozen)
        spec = continuation.score(value, modes(value), BINDING)
        ar = continuation.score(control, modes(control), BINDING)
        self.assertNotIn("not every frozen request", " ".join(continuation.compare(spec, ar, frozen)["insufficient"]))
        frozen["copy"]["requests"].append({"prompt": "c", "seed": None})
        with self.assertRaisesRegex(ValueError, "different frozen sets"):
            continuation.compare(spec, ar, frozen)
        for arm in (value, control):
            arm["sets_sha256"] = continuation.sets_digest(frozen)
        spec = continuation.score(value, modes(value), BINDING)
        ar = continuation.score(control, modes(control), BINDING)
        self.assertIn("chopinfer-spec copy: not every frozen request was recorded",
                      continuation.compare(spec, ar, frozen)["insufficient"])

    def test_record_asks_for_token_ids_and_twenty_alternatives(self):
        frozen = {"copy": {"sampling": dict(continuation.GREEDY), "max_tokens": 2,
                           "requests": [{"prompt": "a", "seed": None}]},
                  "sampled": {"sampling": dict(continuation.SAMPLED), "max_tokens": 2,
                              "requests": [{"prompt": "b", "seed": 1}]}}
        bodies, written = [], []

        def answer(path, body):
            bodies.append(body)
            return {"prompt_token_ids": [1, 2], "choices": [{
                "token_ids": [3, 4], "finish_reason": "length", "prompt_token_ids": [1, 2],
                "logprobs": {"tokens": ["c", "d"], "token_logprobs": [-0.1, -0.2],
                             "top_logprobs": [{"c": -0.1}, {"d": -0.2}]}}]}, 0.5

        import common
        import spec_metrics
        server = {"model": "served"}
        with mock.patch.object(continuation, "prompt_sets", return_value=frozen), \
                mock.patch.object(common, "Client") as client, \
                mock.patch.object(common, "bind_checkpoint"), \
                mock.patch.object(common, "serving_profile", return_value={"tensor_parallel_size": 4}), \
                mock.patch.object(common, "confirm_live"), \
                mock.patch.object(common, "receipt", side_effect=lambda *args: written.append(args)), \
                mock.patch.object(spec_metrics, "metrics_text", return_value=""):
            client.return_value.identity.return_value = server
            client.return_value.request.side_effect = answer
            continuation.record(SimpleNamespace(url="http://unused", timeout=1, model=None, server_log=None,
                                                arm="chopinfer-spec", run_id="cpu", concurrency=0))
        self.assertEqual([(body["logprobs"], body["return_token_ids"], body["temperature"], body.get("seed"))
                          for body in bodies], [(20, True, 0, None), (20, True, 1.0, 1)])
        (check, identity, result, mechanics), = written
        self.assertEqual((check, mechanics, result["arm"], result["sets_sha256"]),
                         ("spec-continuation-chopinfer-spec", False, "chopinfer-spec",
                          continuation.sets_digest(frozen)))
        row = result["sets"]["sampled"]["requests"][0]
        self.assertEqual((row["prompt_ids"], row["ids"], row["logprobs"], row["seed"]), ([1, 2], [3, 4], [-0.1, -0.2], 1))
        self.assertEqual(continuation.workload(result, "m")["sequences"][1]["answer"]["ids"], [3, 4])
        self.assertEqual([group["concurrency"] for group in result["sets"].values()], [1, 1])

    def test_reference_runs_of_one_arm_merge_and_refuse_duplicates(self):
        import json
        import tempfile
        value = record("chopinfer-spec", {"copy": (continuation.GREEDY, [request([1], [2])]),
                                          "sampled": (continuation.SAMPLED, [request([1], [3])])})
        reference = modes(value)
        with tempfile.TemporaryDirectory() as directory:
            runs = [Path(directory) / name for name in ("greedy", "sampled", "again")]
            for run, names in zip(runs, (["copy-000"], ["sampled-000"], ["copy-000"])):
                run.mkdir()
                for mode, entries in reference.items():
                    (run / f"{mode}.json").write_text(json.dumps(mode_file(mode, [entries[name] for name in names])))
            merged = continuation.load_modes(runs[:2], BINDING)
            self.assertEqual(continuation.score(value, merged, BINDING)["sets"]["sampled"]["coverage"]["complete"], True)
            with self.assertRaisesRegex(ValueError, "appears twice"):
                continuation.load_modes([runs[0], runs[2]], BINDING)

    def test_the_arms_must_be_two_bound_processes_of_one_profile(self):
        # Regression: one drafting process's capture,
        # labelled once as each arm, passed as candidate and control.
        rows = [request([1, n], list(range(100, 108))) for n in range(2)]
        spec = record("chopinfer-spec", {"copy": (continuation.GREEDY, rows)})
        ar = record("chopinfer-ar", {"copy": (continuation.GREEDY, copy.deepcopy(rows))})
        self.assertEqual(continuation.compare(continuation.score(spec, modes(spec), BINDING),
                                              continuation.score(ar, modes(ar), BINDING))["bound"]["model"], "served")

        def relabelled(value):
            value.update(server=copy.deepcopy(spec["server"]), serving=copy.deepcopy(spec["serving"]))
            value["sets"]["copy"]["speculation"] = {"draft_tokens": 64, "verify_items": 64}

        cases = {
            "one server process": relabelled,
            "only one arm is bound": lambda value: value.pop("server"),
            "different checkpoints": lambda value: value["server"]["checkpoint"].update(
                checkpoint_identity="b" * 64),
            "different checkpoints or names": lambda value: value["server"].update(model="other"),
            "different topologies": lambda value: value["serving"].update(tensor_parallel_size=2),
            "control must not": lambda value: value["serving"].update(speculation={"mode": "fixed", "tokens": 7}),
            "different profiles apart from speculation": lambda value: value["serving"].update(max_sequences=64),
            "AR control must not": lambda value: value["sets"]["copy"].update(
                speculation={"draft_tokens": 3, "verify_items": 1})}
        for message, change in cases.items():
            control = copy.deepcopy(ar)
            change(control)
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                continuation.compare(continuation.score(spec, modes(spec), BINDING), continuation.score(control, modes(control), BINDING))
        # Regression: a control of another binary or
        # package passed; both arms must name one install.
        install_cases = {
            "different installs": lambda value: value["server"]["health"]["serving"].update(source_commit="b" * 40),
            "different installs ": lambda value: (value["server"]["checkpoint"].update(package_identity="d" * 64),
                                                  value["server"]["health"]["serving"].update(
                                                      package_identity="d" * 64)),
            "different installs  ": lambda value: (value["server"]["checkpoint"].update(metadata_identity="9" * 64),
                                                   value["server"]["health"]["serving"].update(
                                                       metadata_identity="9" * 64)),
            "names no install identity": lambda value: value["server"]["health"]["serving"].pop("source_commit"),
            "names no install identity ": lambda value: value["server"].pop("health"),
            "names no install identity  ": lambda value: value["server"]["checkpoint"].pop("package_identity"),
            "disagrees with its own /health": lambda value: value["server"]["health"]["serving"].update(
                package_identity="d" * 64)}
        # spec_verify's server comparison requires one package first, so a
        # control whose direct package differs, or names none, is refused
        # there before the install comparison.
        package_cases = {"different installs ", "names no install identity  "}
        for message, change in install_cases.items():
            control = copy.deepcopy(ar)
            change(control)
            expected = message.strip()
            if message in package_cases:
                expected = f"{expected}|the arms served different packages"
            with self.subTest(message), self.assertRaisesRegex(ValueError, expected):
                continuation.compare(continuation.score(spec, modes(spec), BINDING),
                                     continuation.score(control, modes(control), BINDING))
        unbound = [copy.deepcopy(value) for value in (spec, ar)]
        for value in unbound:
            value.pop("server")
        with self.assertRaisesRegex(ValueError, "must be bound"):
            continuation.compare(*(continuation.score(value, modes(value), BINDING) for value in unbound))
        idle = copy.deepcopy(spec)
        idle["sets"]["copy"]["speculation"] = {}
        with self.assertRaisesRegex(ValueError, "must draft and verify"):
            continuation.compare(continuation.score(idle, modes(idle), BINDING), continuation.score(ar, modes(ar), BINDING))

    def test_comparison_refuses_mismatched_arms(self):
        value = record("chopinfer-spec", {"copy": (continuation.GREEDY, [request([1], [2])])})
        spec = continuation.score(value, modes(value), BINDING)
        for arm, run_id, sets_sha256, message in (
                ("chopinfer-spec", "cpu", "frozen", "speculative and AR arms"),
                ("chopinfer-ar", "other", "frozen", "speculative and AR arms"),
                ("chopinfer-ar", "cpu", "changed", "different frozen sets")):
            ar = dict(spec, arm=arm, run_id=run_id, sets_sha256=sets_sha256)
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                continuation.compare(spec, ar)



class BindingTests(unittest.TestCase):
    """A continuation reference counts only when it was computed as the
    gate's pinned reference was."""

    def test_entries_keyed_as_the_pinned_reference_are_scored(self):
        value = record("chopinfer-spec", {"copy": (continuation.GREEDY, [request([1], [2, 3])])})
        report = continuation.score(value, modes(value), BINDING)
        self.assertTrue(report["sets"]["copy"]["coverage"]["complete"])

    def test_an_entry_computed_otherwise_is_refused(self):
        value = record("chopinfer-spec", {"copy": (continuation.GREEDY, [request([1], [2, 3])])})

        def rekeyed(entry, change):
            change(entry["key"])
            entry["key_sha256"] = continuation.canonical_sha256(entry["key"])

        cases = {
            "adapter": lambda key: key["adapter"]["files"].update({"models/m/reference.py": "8" * 64}),
            "checkpoint lock": lambda key: key["checkpoint"].update(source_lock_sha256="9" * 64),
            "environment": lambda key: key["environment"].update(torch="other"),
            "reference code": lambda key: key["reference_code"]["files"].update({"m/modeling.py": "a" * 64}),
            "outputs": lambda key: key["outputs"].update(topk=5),
            "model": lambda key: key.update(model="other"),
            "mode policy": lambda key: key["mode"].update(policy_sha256=BINDING["modes"]["f64"]["policy_sha256"])}
        for field, change in cases.items():
            for mode in continuation.MODE_VARIANTS:
                reference = modes(value)
                rekeyed(reference[mode]["copy-000"], change)
                with self.subTest(field=field, mode=mode):
                    copy_set = continuation.score(value, reference, BINDING)["sets"]["copy"]
                    self.assertEqual(copy_set["requests"][0]["refused"],
                                     f"the reference was not computed as the pinned reference: {field}")
                    self.assertFalse(copy_set["coverage"]["complete"])
        reference = modes(value)
        reference["f32"]["copy-000"]["key_sha256"] = "0" * 64
        self.assertEqual(continuation.score(value, reference, BINDING)["sets"]["copy"]["requests"][0]["refused"],
                         "the reference was not computed as the pinned reference: key")

    def test_a_mode_file_is_read_only_as_its_pinned_run(self):
        # Regression: a file declaring a BF16 policy, with
        # entries of another model and checkpoint, was accepted as the F32
        # reference by its file name and the comparison passed.
        import json
        import tempfile
        value = record("chopinfer-spec", {"copy": (continuation.GREEDY, [request([1], [2, 3])])})
        reference = modes(value)
        documents = {
            "BF16 policy in the f32 file": {"f32": reference_modes.policy("bf16", "hf-bf16")},
            "another dequantization": {"f32": reference_modes.policy("f32", "exact")},
            "f64 policy in the bf16 file": {"bf16": reference_modes.policy("f64", "exact")},
            "no policy": {"f32": None}}
        for case, policies in documents.items():
            with tempfile.TemporaryDirectory() as directory, self.subTest(case):
                for mode, entries in reference.items():
                    document = mode_file(mode, list(entries.values()))
                    if mode in policies:
                        document["policy"] = policies[mode]
                    (Path(directory) / f"{mode}.json").write_text(json.dumps(document))
                mode = next(iter(policies))
                with self.assertRaisesRegex(ValueError, f"not the pinned reference's {mode} run"):
                    continuation.load_modes([directory], BINDING)
        # Entries of another model, checkpoint or mode stay refused even in a
        # file whose policy is right.
        for mode in reference:
            for entry_value in reference[mode].values():
                entry_value["key"].update(model="other-model", checkpoint={"source_lock_sha256": "b" * 64})
                entry_value["key_sha256"] = continuation.canonical_sha256(entry_value["key"])
        report = continuation.score(value, reference, BINDING)
        self.assertEqual(report["sets"]["copy"]["requests"][0]["refused"],
                         "the reference was not computed as the pinned reference: model")
        with self.assertRaisesRegex(ValueError, "modes"):
            continuation.score(value, {"f32": reference["f32"]}, BINDING)

    def test_captures_and_reference_must_be_the_gate_s_checkpoint(self):
        config = {"model": BINDING["model"], "qualification": {"checkpoint": {
            "checkpoint_identity": "a" * 64, "served_model_name": "served",
            "source_lock_sha256": BINDING["key"]["checkpoint"]["source_lock_sha256"]}}}
        bound = {"checkpoint_identity": "a" * 64, "model": "served"}
        continuation.authenticate(bound, BINDING, config)
        for change, message in (
                (lambda b, c: b.update(checkpoint_identity="b" * 64), "not the gate's pinned checkpoint"),
                (lambda b, c: b.update(model="other"), "not the gate's pinned checkpoint"),
                (lambda b, c: c["qualification"]["checkpoint"].update(source_lock_sha256="c" * 64),
                 "another model or checkpoint lock"),
                (lambda b, c: c.update(model="other"), "another model or checkpoint lock")):
            changed_bound, changed_config = copy.deepcopy(bound), copy.deepcopy(config)
            change(changed_bound, changed_config)
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                continuation.authenticate(changed_bound, BINDING, changed_config)

    def test_the_reference_command_follows_the_binding(self):
        with mock.patch.dict("os.environ", {"CHOPINFER_REFRUN_PYTHON": "/venv/python"}):
            command = continuation.reference_command(BINDING, "/ckpt", "/w.json", "/out", 2)
        self.assertEqual(command, ["/venv/python", "-B", "-m", "tools.refrun", "run", "--adapter",
                                   "models/m/reference.py", "--checkpoint", "/ckpt", "--lock",
                                   "models/m/checkpoint-files.json", "--lock-variant", "v", "--workload", "/w.json",
                                   "--modes", "f32,bf16", "--dequant", "f32=rounded", "--dequant", "bf16=hf-bf16",
                                   "--devices", "cuda:0,cuda:1", "--output", "/out"])
        unpinned = copy.deepcopy(BINDING)
        unpinned["modes"]["bf16"]["variant"] = "spread"
        with self.assertRaisesRegex(ValueError, "no bf16 run of its bf16 variant"):
            continuation.binding_modes(unpinned)


class ReferenceRunTests(unittest.TestCase):
    """`reference` runs the reference runner only after a GPU query that
    succeeded and listed the GPUs it ran on."""

    def run_reference(self, query):
        """reference() with nvidia-smi answering `query` (a CompletedProcess
        or an exception); its exit, the commands it ran and its output."""
        import json
        import subprocess
        import tempfile
        calls = []

        def child(command, *args, **kwargs):
            calls.append(command[0])
            if command[0] == "nvidia-smi":
                if isinstance(query, BaseException):
                    raise query
                return query
            output = arguments.out / "refrun"
            output.mkdir()
            (output / "run.json").write_text("{}")
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = root / "record.json"
            record.write_text(json.dumps({"arm": "chopinfer-spec"}))
            arguments = SimpleNamespace(record=record, out=root / "reference", checkpoint=root / "checkpoint")
            with mock.patch.object(continuation, "pinned_binding", return_value={"model": "m"}), \
                    mock.patch.object(continuation, "workload", return_value={}), \
                    mock.patch.object(continuation, "reference_command", return_value=["refrun"]), \
                    mock.patch.object(continuation, "entry_origins", return_value={}), \
                    mock.patch.object(continuation.subprocess, "run", side_effect=child), \
                    mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}):
                with self.assertRaises(SystemExit) as stopped:
                    continuation.reference(arguments)
            path = arguments.out / "execution.json"
            written = json.loads(path.read_text()) if path.is_file() else None
        return stopped.exception.code, calls, written

    def test_a_failed_gpu_query_refuses_before_the_reference_runs(self):
        # Regression: nvidia-smi exiting 18 went on
        # to the reference runner, recorded gpus null and exited 0.
        import subprocess
        listed = subprocess.CompletedProcess(["nvidia-smi"], 0, "0, GPU, 1, 1 MiB, 0 MiB, 0 %\n", "")
        code, calls, written = self.run_reference(listed)
        self.assertEqual((code, calls), (0, ["nvidia-smi", "refrun"]))
        self.assertEqual(written["gpus"], ["0, GPU, 1, 1 MiB, 0 MiB, 0 %"])
        for query in (subprocess.CompletedProcess(["nvidia-smi"], 18, "", "failed"),
                      subprocess.CompletedProcess(["nvidia-smi"], 0, "\n", ""),
                      FileNotFoundError("nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 30)):
            with self.subTest(query=query):
                code, calls, written = self.run_reference(query)
                self.assertIsInstance(code, str)
                self.assertIn("refused: the GPU query", code)
                self.assertEqual((calls, written), (["nvidia-smi"], None))


def without_exact_equality(path):
    """Whether a previewed declaration serves speculation without exact
    AR/speculative equality (its C1 greedy rule is informational)."""
    import json
    value = json.loads(Path(path).read_text())
    return value["preview"]["spec_greedy"]["rule"] == "informational"


class GateTests(unittest.TestCase):
    """spec-continuation.v1 as declared: it gates every previewed profile
    without exact AR/speculative equality, and only those."""

    @classmethod
    def setUpClass(cls):
        import declaration
        cls.declared = {}
        for path in sorted(ROOT.glob("models/*/qual.json")):
            import json
            if "preview" not in json.loads(path.read_text()):
                continue
            config = declaration.load(path)
            cls.declared[path] = (config, {check["id"]: check for check in config["checks"]})

    def gate(self):
        path = next(path for path in self.declared if without_exact_equality(path))
        return path, self.declared[path][1].get("spec_continuation")

    def test_profiles_without_exact_equality_declare_the_gate(self):
        self.assertTrue(any(without_exact_equality(path) for path in self.declared))
        for path, (config, checks) in self.declared.items():
            with self.subTest(declaration=str(path.relative_to(ROOT))):
                gated = without_exact_equality(path)
                self.assertEqual("spec_continuation" in checks, gated)
                self.assertEqual("spec_continuation" in config["profiles"]["full"]["checks"], gated)
                if not gated:
                    continue
                check = checks["spec_continuation"]
                self.assertEqual(check["module"], "spec_continuation")
                self.assertEqual(sorted(check["files"]), ["continuation_ar", "continuation_ar_reference",
                                                          "continuation_spec", "continuation_spec_reference"])
                self.assertTrue(check["stdout_json"])
                self.assertIn({"path": ["passed"], "op": "eq", "value": True,
                               "source": "docs/numerics.md#speculative-continuations"}, check["rules"])
                self.assertIn({"path": ["rule"], "op": "eq", "value": continuation.RULE,
                               "source": "docs/numerics.md#speculative-continuations"}, check["rules"])
                for arm, drafting in (("continuation_spec", ("ge", 1)), ("continuation_ar", ("eq", 0))):
                    rules = {tuple(rule["path"]): (rule["op"], rule["value"]) for rule in check["input_rules"][arm]}
                    self.assertEqual(rules[("sets_sha256",)][0], "eq")
                    self.assertRegex(rules[("sets_sha256",)][1], "^[0-9a-f]{64}$")
                    self.assertEqual(rules[("serving", "speculation", "tokens")], drafting)
                    self.assertEqual(rules[("arm",)], ("eq", "chopinfer-" + arm.split("_")[1]))

    def test_the_declared_digest_is_the_frozen_sets(self):
        gated = [(path, checks["spec_continuation"]) for path, (_, checks) in self.declared.items()
                 if "spec_continuation" in checks]
        self.assertTrue(gated)
        for path, check in gated:
            with self.subTest(declaration=str(path.relative_to(ROOT))):
                reason = without_frozen_inputs(path)
                if reason:
                    self.skipTest(reason)
                with mock.patch.dict("os.environ", {"CHOPINFER_QUAL_CONFIG": str(path)}):
                    digest = continuation.sets_digest(continuation.prompt_sets())
                for arm in ("continuation_spec", "continuation_ar"):
                    rules = {tuple(rule["path"]): (rule["op"], rule["value"]) for rule in check["input_rules"][arm]}
                    self.assertEqual(rules[("sets_sha256",)], ("eq", digest))

    def judged(self, value):
        """The declared check's verdict over a receipt, as rescore gives it."""
        import kinds
        _, check = self.gate()
        self.assertIsNotNone(check, "no declared spec_continuation check")
        return kinds.rescore(kinds.of(check["module"]), value, None, check)["status"]

    def arms(self, spec_logprob=-0.09, ar_logprob=-0.09, sampled=16):
        rows = [request([1, n], list(range(100, 164))) for n in range(sampled)]
        greedy = [request([2, n], list(range(200, 216))) for n in range(4)]
        values = {}
        for arm, logprob in (("chopinfer-spec", spec_logprob), ("chopinfer-ar", ar_logprob)):
            value = record(arm, {"copy": (continuation.GREEDY, copy.deepcopy(greedy)),
                                 "sampled": (continuation.SAMPLED, copy.deepcopy(rows))})
            for row in value["sets"]["sampled"]["requests"]:
                row["logprobs"] = [logprob] * 64
            values[arm] = value
        return values["chopinfer-spec"], values["chopinfer-ar"]

    def scored(self, value, **changes):
        return continuation.score(value, modes(value, **changes), BINDING)

    UNIFORM_MODES = {"f32": {"top": UNIFORM}, "bf16": {"top": UNIFORM}}

    def test_a_candidate_within_the_limits_passes(self):
        spec, ar = self.arms()
        self.assertEqual(self.judged(receipt(self.scored(spec, **self.UNIFORM_MODES),
                                             self.scored(ar, **self.UNIFORM_MODES))), "pass")

    def test_a_candidate_failure_where_the_control_passed_fails(self):
        spec, ar = self.arms(spec_logprob=-0.5)
        value = receipt(self.scored(spec, **self.UNIFORM_MODES), self.scored(ar, **self.UNIFORM_MODES))
        self.assertEqual(value["failed_sets"], ["sampled"])
        self.assertEqual(self.judged(value), "fail")

    def test_an_insufficient_control_is_never_a_pass(self):
        spec, ar = self.arms(spec_logprob=-0.5, ar_logprob=-0.5)
        value = receipt(self.scored(spec, **self.UNIFORM_MODES), self.scored(ar, **self.UNIFORM_MODES))
        self.assertIn("sampled: the AR control misses the limits", value["insufficient"])
        self.assertEqual(self.judged(value), "insufficient")
        # A control within the limits beside a candidate that misses them
        # where the control does too still fails on the other set.
        spec, ar = self.arms(spec_logprob=-0.5)
        value = receipt(self.scored(spec, **self.UNIFORM_MODES), self.scored(ar, **self.UNIFORM_MODES))
        self.assertEqual(self.judged(value), "fail")

    def test_missing_coverage_is_insufficient(self):
        spec, ar = self.arms()
        reference = modes(spec, **self.UNIFORM_MODES)
        del reference["f32"]["sampled-003"]
        value = receipt(continuation.score(spec, reference, BINDING), self.scored(ar, **self.UNIFORM_MODES))
        self.assertIn("sampled: incomplete coverage", value["insufficient"])
        self.assertEqual(self.judged(value), "insufficient")
        # Too few high-entropy positions is missing coverage too.
        spec, ar = self.arms(sampled=15)
        self.assertEqual(self.judged(receipt(self.scored(spec, **self.UNIFORM_MODES),
                                             self.scored(ar, **self.UNIFORM_MODES))), "insufficient")

    def test_a_verdict_without_its_gpu_execution_or_with_a_forged_status_is_insufficient(self):
        spec, ar = self.arms()
        value = receipt(self.scored(spec, **self.UNIFORM_MODES), self.scored(ar, **self.UNIFORM_MODES))
        for change in (lambda v: v["execution"]["gpu"].update(spec=[None]),
                       lambda v: v["execution"]["gpu"]["ar"][0].update(exit=1),
                       lambda v: v["execution"]["gpu"]["spec"][0].update(gpus=None),
                       lambda v: v["execution"]["gpu"]["ar"][0].update(gpus=[]),
                       lambda v: v["execution"]["gpu"]["ar"][0]["entries"]["bf16"].update(computed=1),
                       lambda v: v["execution"].pop("cpu_scoring"),
                       lambda v: v.update(status="pass", passed=True, failed_sets=["copy"]),
                       lambda v: v["spec"]["sets"]["copy"]["coverage"].update(complete=False)):
            forged = copy.deepcopy(value)
            change(forged)
            with self.subTest(forged=change):
                self.assertEqual(self.judged(forged), "insufficient")


class OfflineCheckTests(unittest.TestCase):
    def test_only_an_offline_module_is_judged_without_a_server(self):
        import checks
        self.assertTrue(checks.offline_module("spec_continuation"))
        self.assertFalse(checks.offline_module("spec_verify"))



class CheckRunTests(unittest.TestCase):
    """The declared check run end to end by checks.py without a server, over
    recorded arms and reference directories keyed as the pinned reference."""

    def runner_inputs(self, directory, path, spec_logprob=-0.09):
        import json
        with mock.patch.dict("os.environ", {"CHOPINFER_QUAL_CONFIG": str(path)}):
            binding = continuation.pinned_binding()
            frozen = continuation.prompt_sets()
        config = self.declared[path][0]
        pin = config["qualification"]["checkpoint"]
        inputs = {}
        for arm, logprob in (("spec", spec_logprob), ("ar", -0.09)):
            sets = {}
            for name, group in frozen.items():
                greedy = group["sampling"]["temperature"] == 0
                rows = [request([7, index], [300 + index] * (8 if greedy else 16),
                                logprob=-0.1 if greedy else logprob) for index in range(len(group["requests"]))]
                sets[name] = (group["sampling"], rows)
            value = record(f"chopinfer-{arm}", sets)
            value.update(passed=True, sets_sha256=continuation.sets_digest(frozen),
                         serving={"tensor_parallel_size": 4, "kv_cache_dtype": "bf16",
                                  "speculation": {"mode": "fixed" if arm == "spec" else "off",
                                                  "tokens": 7 if arm == "spec" else 0}})
            value["server"] = served(pin["served_model_name"], pin["checkpoint_identity"],
                                     ("1" if arm == "spec" else "0") * 32, value["serving"])
            record_path = directory / f"{arm}.json"
            record_path.write_text(json.dumps(value))
            reference = directory / f"{arm}-reference"
            (reference / "refrun").mkdir(parents=True)
            for mode in continuation.MODE_VARIANTS:
                entries = []
                for name, group in value["sets"].items():
                    for index, row in enumerate(group["requests"]):
                        label = continuation.sequence_name(name, index)
                        made = entry(label, row, mode=mode, top=None if name != "sampled" else UNIFORM)
                        made["key"].update({"model": binding["model"], **copy.deepcopy(binding["key"]),
                                            "mode": {"name": binding["modes"][mode]["mode"],
                                                     "policy_sha256": binding["modes"][mode]["policy_sha256"]}})
                        made["key_sha256"] = continuation.canonical_sha256(made["key"])
                        entries.append(made)
                (reference / "refrun" / f"{mode}.json").write_text(json.dumps(mode_file(mode, entries, binding)))
            requests = sum(len(group["requests"]) for group in value["sets"].values())
            (reference / "execution.json").write_text(json.dumps(gpu_execution(requests)[0]))
            inputs[f"continuation_{arm}"] = str(record_path)
            inputs[f"continuation_{arm}_reference"] = str(reference)
        return inputs

    setUpClass = classmethod(GateTests.setUpClass.__func__)
    gate = GateTests.gate

    def check_run(self, spec_logprob=-0.09, change=None):
        import json
        import tempfile
        import checks
        path, _ = self.gate()
        reason = without_frozen_inputs(path)
        if reason:
            self.skipTest(reason)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            inputs = directory / "inputs.json"
            given = self.runner_inputs(directory, path, spec_logprob)
            if change is not None:
                change(given)
            inputs.write_text(json.dumps(given))
            config = self.declared[path][0]
            args = SimpleNamespace(model=config["model"], id="spec_continuation", url=None, offline=True,
                                   served_model=config["qualification"]["checkpoint"]["served_model_name"],
                                   cohort="full", inputs=inputs, with_receipt=[], receipts=directory / "receipts")
            with mock.patch("sys.stdout"):
                code = checks.check_command(args)
            result = json.loads(next((directory / "receipts").glob("*/check.json")).read_text())
            row = result.get("row") or {}
            value = json.loads(Path(row["receipt"]).read_text()) if row.get("receipt") else None
            if value is None and row.get("log"):
                result["log_text"] = Path(row["log"]).read_text()
        return code, result, value

    def test_the_declared_check_passes_offline_and_records_its_gpu_execution(self):
        code, result, value = self.check_run()
        self.assertEqual((code, result["status"]), (0, "pass"), result.get("error") or result.get("row"))
        self.assertIsNone(result["server"])
        self.assertEqual(value["execution"]["gpu"]["spec"][0]["schema"], continuation.EXECUTION_SCHEMA)
        self.assertIn("cpu_scoring", value["execution"])

    def test_the_declared_check_fails_a_candidate_the_control_passes(self):
        code, result, value = self.check_run(spec_logprob=-0.5)
        self.assertEqual((code, result["status"]), (1, "fail"))
        self.assertEqual(value["failed_sets"], ["sampled"])

    def test_the_declared_check_refuses_a_request_that_lost_its_ids(self):
        # The lost-IDs regression, through the public compare path:
        # natural-015 loses its emitted IDs but keeps its logprobs.
        def lose(inputs):
            import json
            value = json.loads(Path(inputs["continuation_spec"]).read_text())
            value["sets"]["natural"]["requests"][15]["ids"] = None
            Path(inputs["continuation_spec"]).write_text(json.dumps(value))
        code, result, value = self.check_run(change=lose)
        self.assertEqual((code, result["status"]), (1, "fail"))
        self.assertIn("natural: incomplete coverage", result["row"]["error"])
        self.assertEqual(value["status"], "insufficient")
        self.assertEqual(value["spec"]["sets"]["natural"]["requests"][15]["refused"], "no emitted token IDs")

    def test_the_declared_check_refuses_records_of_another_checkpoint(self):
        def other(inputs):
            import json
            for arm in ("continuation_spec", "continuation_ar"):
                value = json.loads(Path(inputs[arm]).read_text())
                value["server"]["checkpoint"]["checkpoint_identity"] = "b" * 64
                Path(inputs[arm]).write_text(json.dumps(value))
        code, result, _ = self.check_run(change=other)
        self.assertEqual((code, result["status"]), (1, "fail"))

    def test_the_declared_check_refuses_a_control_of_another_install(self):
        # Regression: distinct source commits and direct
        # packages in the two arms passed the declared offline check.
        def different_install(inputs):
            import json
            for arm, commit, package in (("spec", "a" * 40, "c" * 64), ("ar", "b" * 40, "d" * 64)):
                path = Path(inputs["continuation_" + arm])
                value = json.loads(path.read_text())
                value["server"]["checkpoint"]["package_identity"] = package
                value["server"]["health"]["serving"].update(source_commit=commit, package_identity=package)
                path.write_text(json.dumps(value))
        code, result, _ = self.check_run(change=different_install)
        self.assertEqual((code, result["status"]), (1, "fail"))
        self.assertIn("check exited", result["row"]["error"])
        # The arms' direct packages differ, which the shared server
        # comparison refuses before the install comparison.
        self.assertIn("the arms served different packages", result["log_text"])

    def test_a_live_check_cannot_run_offline(self):
        import checks
        path, _ = self.gate()
        config = self.declared[path][0]
        args = SimpleNamespace(model=config["model"], id="spec_verify", url=None, offline=True, served_model=None,
                               cohort="full", inputs=None, with_receipt=[], receipts=None)
        with self.assertRaisesRegex(ValueError, "needs a live server"):
            checks.check_command(args)


if __name__ == "__main__":
    unittest.main()

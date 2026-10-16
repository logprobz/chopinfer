"""Gate input parity and synthetic model configuration, without serving."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import common
import declaration
import evals
import logprobs_parity
import prefix_exact
import spec_bench
import structured
import verify_exact


# Pin declared checks and cohorts, including available checks outside cohorts.
# Source links are validated separately so moving a doc cannot change a gate.
# Pins include the explicit 128K full gate, the adopted HF media digests and
# the two full KV offload checks. Reversing only those media digest updates and
# offload checks reproduces the pre-integration policy hashes; existing
# numerical rules are unchanged. The first pin also holds that declaration's
# full spec_continuation check (2026-10-15); without it the policy hashes to
# the previous pin, 2d904f49ed52a333f3d0e9500c3880c6c327b645015a001e0208a54ab830b918.
# It also takes the audio withdrawal: restoring only its multimodal rule
# values (the exact media reference digest and the mixed case count 5)
# reproduces the spec_continuation pin,
# a9e05356ce8fb010e310000560e5471d6e3aeefc73683d5b3c64a45c39697b1e.
POLICY_HASHES = {'mimo26': 'd8280c1ab2804d3da4f40f93dae0cb54e7ec225a1928bebc0550978b8fe27cab',
                 'qwen38': 'f84f7167b4f7c41613d224afe55e23dec288788acaa671a50f20befd07596462'}
# The cohort budgets the hashes were pinned with, before cohorts lost their
# overall deadline (budget_seconds). Restored only to compare with the pins.
RETIRED_BUDGETS = {'quick': 900, 'full': 172800}


def policy_without_sources(value):
    if isinstance(value, dict):
        return {key: policy_without_sources(item) for key, item in value.items() if key != "source"}
    if isinstance(value, list):
        return [policy_without_sources(item) for item in value]
    return value


class ModelConfigTests(unittest.TestCase):
    def test_floor_only_long_context_keeps_existing_limits_and_full_coverage(self):
        found = []
        for path in sorted((common.ROOT / 'models').glob('*/qual.json')):
            raw = json.loads(path.read_text())
            for name, check in raw['checks'].items():
                policy = check.get('constants', {}).get('teacher', {})
                if not policy.get('ALLOW_FLOOR_ONLY', {}).get('value'):
                    continue
                found.append((raw['model'], name))
                self.assertEqual(name, 'long_context_128k')
                old = raw['checks']['long_context']
                self.assertEqual({k: v for k, v in policy.items() if k != 'ALLOW_FLOOR_ONLY'},
                                 old['constants']['teacher'])
                keep = lambda rules: [r for r in rules if r['path'] != ['reference_sha256']]
                self.assertEqual(keep(check['rules']), keep(old['rules']))
                self.assertIn(name, raw['cohorts']['full']['checks'])
                self.assertNotIn(name, raw['cohorts']['quick']['checks'])
                self.assertEqual(raw['references'][name]['variants'], ['exact'])
                for config, spec in raw['configurations'].items():
                    if 'long_context' in spec.get('checks', []):
                        self.assertIn(name, spec['checks'], config)
                        self.assertIn(name, spec['keep_references'], config)
                        self.assertEqual(spec['append'][name], spec['append']['long_context'])
        self.assertEqual(len(found), 1)

    def test_gate_documentation_sources_exist(self):
        def sources(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "source":
                        yield item
                    else:
                        yield from sources(item)
            elif isinstance(value, list):
                for item in value:
                    yield from sources(item)

        for path in declaration.shipped():
            for source in sources(json.loads(path.read_text())):
                with self.subTest(config=str(path), source=source):
                    file, _, anchor = source.partition("#")
                    document = common.ROOT / file
                    self.assertTrue(document.is_file(), source)
                    if anchor:
                        headings = re.findall(r"^#{1,6} (.+)$", document.read_text(), re.MULTILINE)
                        slugs = [re.sub(r"[^\w -]", "", heading.lower()).replace(" ", "-")
                                 for heading in headings]
                        self.assertIn(anchor, slugs, source)

    def test_existing_gates_keep_their_checks_and_inputs(self):
        for name, expected in POLICY_HASHES.items():
            path = declaration.path(name)
            config = declaration.load(path)
            # MiMo's teacher, exact verify and multimodal checks gate against the MOPD references;
            # Qwen3.8's teacher against the GPU reference runner's F32 reference.
            self.assertFalse([check['id'] for check in config['checks'] if check.get('informational_reason')])
            # The checks a cohort runs: qual.json also declares checks only its
            # other configurations run.
            selected = {name for profile in config["profiles"].values() for name in profile["checks"]}
            self.assertFalse([profile for profile in config["profiles"].values() if "budget_seconds" in profile])
            profiles = {profile: {"budget_seconds": RETIRED_BUDGETS[profile], **policy}
                        for profile, policy in config["profiles"].items()}
            gate = {"profiles": profiles, "checks": [check for check in config["checks"] if check["id"] in selected]}
            actual = hashlib.sha256(json.dumps(policy_without_sources(gate), sort_keys=True).encode()).hexdigest()
            self.assertEqual(actual, expected)
            with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(path),
                    "CHOPINFER_DATA": "/data/example", "CHOPINFER_MIMO_FIXTURES": "/data/fixtures"}):
                self.assertEqual(common.resolve_input("receipts"), Path(f"/data/example/qual/{name}"))
                # Each model's API shapes and chat renders are its own captures.
                own = "fixtures/qwen38/frontend" if name == "qwen38" else "fixtures/mimo26/frontend"
                self.assertNotIn("api_samples", config["inputs"])
                self.assertEqual(common.resolve_input("chat_renders"), common.ROOT / own / "chat-renders.json")
                self.assertEqual(common.resolve_input("gsm8k"), Path("/data/fixtures/gsm8k"))
                needle = (Path("/data/example/qwen38-refs/prompts/prompts.json") if name == "qwen38"
                          else Path("/data/fixtures/functional"))
                self.assertEqual(common.resolve_input("needle"), needle)
                self.assertEqual(common.resolve_input("spec_teacher"), common.ROOT / "fixtures/mimo26/teacher-texts.json")
                self.assertEqual(common.resolve_input("media_cases"), common.ROOT / "fixtures/mimo26/capture/multimodal_cases.py")
                self.assertEqual(common.thinking_kwargs(False), {"enable_thinking": False})
                self.assertEqual(config["qualification"]["chars_per_token"], 4.3)
                depth = json.loads(path.read_text())["preview"]["speculation"]["tokens"]
                self.assertEqual(config["qualification"]["verify_buckets"],
                                 [width for width in (2, 4, 8) if width <= depth + 1])
                # Qwen3.8's exact input is its pinned F32 teacher reference.
                exact = (common.ROOT / "fixtures/qwen38/teacher-reference.json" if name == "qwen38"
                         else common.ROOT / "fixtures/mimo26/teacher-exact-reference.json")
                self.assertEqual(common.resolve_input("exact"), exact)

    def test_verify_inputs_use_the_models_checkpoint_and_reference(self):
        for model in ("mimo26", "qwen38"):
            config = declaration.load(declaration.path(model))
            check = next(check for check in config["checks"] if check["id"] == "verify_exact")
            pin = config["qualification"]["checkpoint"]
            exact = common.input_path("exact", config=config)
            for arm in ("verify_ar", "verify_candidate"):
                rules = {tuple(rule["path"]): rule for rule in check["input_rules"][arm]}
                self.assertEqual(rules[("server", "checkpoint", "checkpoint_identity")]["value"],
                                 pin["checkpoint_identity"], (model, arm))
                self.assertEqual(rules[("reference_sha256",)]["value"],
                                 hashlib.sha256(exact.read_bytes()).hexdigest(), (model, arm))

    def test_hard_reference_predicates_match_the_adopted_lock(self):
        import refs
        for model in ("mimo26", "qwen38"):
            config = declaration.load(declaration.path(model))
            lock = refs.load_lock(config)
            references = config["qualification"]["references"]
            compared = 0
            for check in config["checks"]:
                if check.get("informational_reason"):
                    continue
                predicates = check["rules"] + [rule for rules in check.get("input_rules", {}).values()
                                               for rule in rules]
                for rule in predicates:
                    if rule["path"] not in (["reference_sha256"], ["exact_reference_sha256"],
                                            ["reference", "sha256"]):
                        continue
                    if check["module"] == "multimodal":
                        name = config["qualification"]["media"]["reference"]
                    elif check["module"] == "teacher":
                        args = check["args"]
                        name = args[args.index("--reference-name") + 1] if "--reference-name" in args else "teacher"
                    else:
                        self.assertEqual(check["module"], "verify_exact")
                        exact = common.input_path("exact", config=config)
                        name, = [name for name, spec in references.items() if common.declared_path(spec["path"]) == exact]
                    adopted = lock["entries"]["reference:" + name]["members"]["."]
                    with self.subTest(model=model, check=check["id"], reference=name):
                        self.assertEqual(adopted, references[name]["sha256"])
                        self.assertEqual(rule["op"], "eq")
                        self.assertEqual(rule["value"], adopted)
                    compared += 1
            self.assertGreater(compared, 0)

    def test_qwen_image_predicate_accepts_current_and_refuses_old_reference(self):
        import copy
        import kinds
        import multimodal
        import refs
        config = declaration.load(declaration.path("qwen38"))
        check = next(row for row in config["checks"] if row["id"] == "multimodal")
        spec = config["qualification"]["references"]["image_probes"]
        adopted = refs.load_lock(config)["entries"]["reference:image_probes"]["members"]["."]
        raw = (common.ROOT / spec["path"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), adopted)
        fixture = json.loads(raw)
        probes = {name: multimodal.reference_probe(name, item, fixture["rule"], fixture["token_bytes"])
                  for name, item in fixture["probes"].items()}
        reference = {"format": "image-probes", "sha256": adopted, "order": list(probes),
                     "probes": probes, "rule": fixture["rule"]}
        # Perfect synthetic scores isolate the declared pin from numerical tolerances.
        rows = []
        for name, probe in probes.items():
            row = {"id": name, "kind": "image", "passed": True, "fixture_expectation_match": True,
                   "token_count_match": True}
            if probe["kind"] == "free-text":
                row["teacher_forced"] = {"passed": True, "scored": probe["scored"],
                    "mean_abs_dlogprob": 0.0, "top1_disagreement": 0.0,
                    "near_tie_report": {"available": True, "flips": []}}
            else:
                row["greedy"] = {"passed": True, "exact": True, "divergence": None}
            rows.append(row)
        value = {"passed": True, "reference": {"sha256": adopted}, "rows": rows,
                 "server": {"checkpoint": config["qualification"]["checkpoint"]}}
        kind = kinds.of("multimodal")
        with kinds.policy(declaration.path("qwen38"), check):
            result = kinds.rescore(kind, value, reference, check)
            self.assertEqual(result["status"], "pass", result)
            old = copy.deepcopy(value)
            old["reference"]["sha256"] = "e36e01d00fa50358073a9b64f5b917ff5fcee0b48017a4d4bec3452f063becf8"
            self.assertEqual(kinds.rescore(kind, old, reference, check)["status"], "insufficient")
            self.assertEqual(kind.judge(kind.score(old, check=check), check)[0], "fail")

    def test_synthetic_frontend_keyword_and_buckets_reach_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frontend = root / "frontend.json"
            frontend.write_text(json.dumps({"reasoning": {"thinking_keywords": ["ponder"]}}))
            config = root / "gate.json"
            config.write_text(json.dumps({"receipts": str(root / "receipts"), "inputs": {"frontend": str(frontend), "exact": str(root / "reference.json")},
                "qualification": {"verify_buckets": [2, 3], "chars_per_token": 2.0}}))
            with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(config)}):
                self.assertEqual(common.thinking_kwargs(True), {"ponder": True})
                self.assertEqual(evals.chat([], 32, False)["chat_template_kwargs"], {"ponder": False})
                for _, endpoint, body in logprobs_parity.cases():
                    if "chat_template_kwargs" in body:
                        self.assertEqual(body["chat_template_kwargs"], {"ponder": False})
                for _, body, _ in structured.cases(argparse.Namespace(reasoning_max_tokens=32)).values():
                    if "chat_template_kwargs" in body:
                        self.assertEqual(list(body["chat_template_kwargs"]), ["ponder"])
                body = spec_bench.workload("m", "run", "gsm8k", 1, 0, "Question?", "Solve", 32)
                self.assertEqual(body["chat_template_kwargs"], {"ponder": True})
                self.assertEqual(verify_exact.draft_count(7, 20), 2)
                self.assertEqual(verify_exact.exact_path(None), root / "reference.json")
                with patch.object(evals, "haystack", return_value=["text"]) as haystack:
                    evals.long_item(["text"], "multikey", 100, 0)
                self.assertEqual(haystack.call_args.args[2], 188)
                with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "CHOPINFER_QUAL_CONFIG"):
                    common.input_path("exact")

    def test_a_frontend_that_refuses_disabled_thinking_gets_no_disabled_switch(self):
        """Answer-only requests omit the switch and budget for reasoning when
        the frontend refuses disabled thinking."""
        import prefix_exact
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frontend = root / "frontend.json"
            config = root / "gate.json"
            config.write_text(json.dumps({"receipts": str(root / "receipts"), "inputs": {"frontend": str(frontend)},
                                          "qualification": {"verify_buckets": [2], "chars_per_token": 2.0}}))
            with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(config)}):
                for allowed, kwargs in ((True, {"ponder": False}), (False, {})):
                    reasoning = {"thinking_keywords": ["ponder"]} | ({} if allowed else {"allow_disabled": False})
                    frontend.write_text(json.dumps({"reasoning": reasoning}))
                    self.assertEqual(common.thinking_can_be_disabled(), allowed)
                    self.assertEqual(common.answer_kwargs(), kwargs)
                    cases = structured.cases(argparse.Namespace(reasoning_max_tokens=32))
                    for name, (_, body, _) in cases.items():
                        if "messages" in body and name != "reasoning_json_schema":
                            self.assertEqual(body["chat_template_kwargs"], kwargs, name)
                            self.assertEqual(body.get("max_tokens"), None if allowed else 32, name)
                    self.assertEqual(cases["reasoning_json_schema"][1]["chat_template_kwargs"], {"ponder": True})
                    client = Mock()
                    client.request.return_value = ({"tokens": [1]}, 0.0)
                    prefix_exact.tokenize(client, "m", [])
                    self.assertEqual(client.request.call_args.args[1]["chat_template_kwargs"], kwargs)

    def test_qwen_exact_input_is_the_bound_teacher_reference_with_full_score_coverage(self):
        config = declaration.load(declaration.path("qwen38"))
        path = common.input_path("exact", config=config)
        self.assertEqual(path, common.input_path("teacher_reference", config=config))
        self.assertNotIn("exact", config["qualification"]["references"])
        reference = json.loads(path.read_bytes())
        self.assertEqual(reference["schema"], config["qualification"]["prefix"]["reference_schema"])
        self.assertEqual(sum(len(text["exact"]["logprob"]) for text in reference["texts"]), 4176)
        for text in reference["texts"]:
            count = len(text["prompt_token_ids"]) - text["score_from"]
            for field in ("logprob", "top1", "margin"):
                self.assertEqual(len(text["exact"][field]), count)

"""Expanded topology checks consume their actual typed inputs and reports."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import declaration
import kinds
import release_gate
import checks


def selected():
    result = []
    for path in declaration.shipped():
        base = json.loads(path.read_text())
        for name, spec in base.get("configurations", {}).items():
            if spec.get("serving", {}).get("tensor_parallel_size") == 2 and all(
                    check in spec["checks"] for check in ("teacher", "paired_teacher")):
                config = declaration.load(declaration.named(base["model"] + "." + name))
                result.append((config, spec))
    return result


class ConfigurationInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = selected()
        if len(cls.configs) < 3:
            raise AssertionError("no declared two-rank teacher/paired/prefix configuration tested")

    def test_teacher_accepts_plain_server_log_and_keeps_output_topology_gates(self):
        for config, spec in self.configs:
            check = next(c for c in config["checks"] if c["id"] == "teacher")
            with tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / "server.log"
                log.write_text("plain startup diagnostics, not a JSON receipt\n")
                result = {}
                release_gate.prerequisites(config, ["teacher"], "full", {
                    "server_log": str(log), "url": "http://unused", "model": "served"}, result)
                self.assertIn("server_log", result["input_sha256"])
            rules = [rule for rule in check["rules"] if rule["path"][:4] == ["server", "health", "serving", "profile"]]
            self.assertEqual(len(rules), 2 if "speculation_tokens" in spec["serving"] else 1)
            value = {"server": {"health": {"serving": {"profile": {
                "tensor_parallel_size": spec["serving"]["tensor_parallel_size"],
                "speculation": {"tokens": spec["serving"].get("speculation_tokens", 3)}}}}}}
            self.assertTrue(all(kinds.predicate(value, rule) for rule in rules))
            value["server"]["health"]["serving"]["profile"]["tensor_parallel_size"] = 4
            self.assertFalse(all(kinds.predicate(value, rule) for rule in rules))

    def test_paired_preflight_checks_both_real_input_names_and_profiles(self):
        for config, spec in self.configs:
            check = next(c for c in config["checks"] if c["id"] == "paired_teacher")
            with tempfile.TemporaryDirectory() as directory:
                context = {"model": "served", "url": "http://unused"}
                value = {"server": {"model": "served", "health": {"serving": {"profile": {
                    "tensor_parallel_size": spec["serving"]["tensor_parallel_size"],
                    "speculation": {"tokens": spec["serving"].get("speculation_tokens", 3)}}}}}}
                for key in ("stored_teacher", "teacher"):
                    path = Path(directory) / (key + ".json")
                    path.write_text(json.dumps(value));context[key] = str(path)
                release_gate.prerequisites(config, [check["id"]], "full", context, {})
                for key in ("stored_teacher", "teacher"):
                    for field, bad in [("tensor_parallel_size", 4)] + ([("speculation", {"tokens": 1})] if "speculation_tokens" in spec["serving"] else []):
                        changed = copy.deepcopy(value)
                        changed["server"]["health"]["serving"]["profile"][field] = bad
                        Path(context[key]).write_text(json.dumps(changed))
                        with self.subTest(key=key, field=field), self.assertRaises(ValueError):
                            release_gate.prerequisites(config, [check["id"]], "full", context, {})
                        Path(context[key]).write_text(json.dumps(value))

    def test_prefix_report_uses_its_bound_identity_and_rejects_other_profiles(self):
        for config, spec in self.configs:
            if "prefix_exact" not in spec["checks"]:
                continue
            check = next(c for c in config["checks"] if c["id"] == "prefix_exact")
            kind = kinds.of(check["module"])
            value = {"passed": True, "skipped": [], "scores": {"all": {"positions": 1}},
                     "checks": {key: True for key in ("cache", "hit_equals_cold", "chunked_equals_cold", "hit_vs_cold")},
                     "serving_identity": {"profile": {"tensor_parallel_size": 2, "speculation": {"tokens": spec["serving"].get("speculation_tokens", 3)}}}}
            self.assertEqual(kind.judge(kind.score(value, check=check), check)[0], "pass")
            for field, bad in [("tensor_parallel_size", 4)] + ([("speculation", {"tokens": 1})] if "speculation_tokens" in spec["serving"] else []):
                changed = copy.deepcopy(value);changed["serving_identity"]["profile"][field] = bad
                self.assertEqual(kind.judge(kind.score(changed, check=check), check)[0], "fail")
            del value["serving_identity"]
            with self.assertRaises(KeyError):
                kind.judge(kind.score(value, check=check), check)

    def test_comparators_gate_each_bound_input_and_use_actual_report_paths(self):
        exercised = set()
        for config, spec in self.configs:
            for check in config['checks']:
                if check['id'] not in ('spec_verify', 'verify_exact'):
                    continue
                exercised.add(check['id'])
                with tempfile.TemporaryDirectory() as directory:
                    context = {'url': 'http://unused', 'model': 'served'}
                    values = {}
                    for index, key in enumerate(check['files']):
                        value = {'server': {'model': 'served', 'health': {'serving': {
                            'profile': {'tensor_parallel_size': 2},
                            'startup_nonce': str(index + 1) * 32}}}}
                        value.update(passed=True, texts=[{}] * 35, k=index,
                                     sets={name: {'requests': [{}] * 16} for name in ('copy', 'natural')})
                        value['server']['checkpoint'] = {
                            'checkpoint_identity': config['qualification']['checkpoint']['checkpoint_identity']}
                        if check['id'] == 'verify_exact':
                            value['reference_sha256'] = next(r['value'] for r in check['input_rules'][key]
                                                             if r['path'] == ['reference_sha256'])
                        path = Path(directory) / (key + '.json')
                        path.write_text(json.dumps(value)); context[key] = str(path); values[key] = value
                    summary = {'inputs': context, 'server': {'model': 'served'}}
                    release_gate.prerequisites(config, [check['id']], 'full', context, summary)
                    self.assertIsNone(checks.input_verdict({}, check, {}, None, summary))
                    for key, value in values.items():
                        changed = copy.deepcopy(value)
                        changed['server']['health']['serving']['profile']['tensor_parallel_size'] = 4
                        Path(context[key]).write_text(json.dumps(changed))
                        with self.subTest(check=check['id'], input=key), self.assertRaises(ValueError):
                            release_gate.prerequisites(config, [check['id']], 'full', context, {})
                        self.assertEqual(checks.input_verdict({}, check, {}, None, summary)['status'], 'insufficient')
                        Path(context[key]).write_text(json.dumps(value))
                    # Verify comparisons carry topology in the two authenticated
                    # inputs; speculative comparisons also report both profiles.
                    self.assertFalse(any(r['path'][0] == 'server' for r in check['rules']))
                    if check['id'] == 'spec_verify':
                        rules = [r for r in check['rules'] if r['path'][0] == 'bound']
                        report = {'bound': {'serving': {arm: {'tensor_parallel_size': 2}
                                                       for arm in ('ar', 'spec')}}}
                        self.assertTrue(rules)
                        self.assertTrue(all(kinds.predicate(report, r) for r in rules))
                        report['bound']['serving']['spec']['tensor_parallel_size'] = 4
                        self.assertFalse(all(kinds.predicate(report, r) for r in rules))
        self.assertEqual(exercised, {'spec_verify', 'verify_exact'})


if __name__ == "__main__":
    unittest.main()

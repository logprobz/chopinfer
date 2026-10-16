"""checks.py: one declared check against a live server, and offline
rescores of stored gate runs and receipts."""

import contextlib
import copy
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import checks
import declaration
import release_gate as gate


def fake_worker(passed, receipt=True, exits=None):
    """subprocess.run for gate workers: each writes a receipt named for its
    check and exits with its code in `exits` (default 0)."""
    def execute(command, **kwargs):
        name = command[5]
        if receipt:
            value = {"schema": "chopinfer-qualification/v1", "check": name, "passed": passed.get(name, True),
                     "skipped": [], "errors": [], "mechanics_only": True}
            (Path(kwargs["env"]["CHOPINFER_QUAL_RECEIPTS"]) / f"fresh-{name}.json").write_text(json.dumps(value))
        return subprocess.CompletedProcess(command, (exits or {}).get(name, 0))
    return execute


def command(*arguments):
    """checks.py's exit status and stdout for `arguments`."""
    output = io.StringIO()
    with patch.object(sys, "argv", ["checks.py", *map(str, arguments)]), contextlib.redirect_stdout(output), \
            contextlib.redirect_stderr(io.StringIO()):
        status = checks.main()
    return status, output.getvalue()


def documented_exits():
    """command -> {exit status: meaning} from the exit status table of docs/check-kinds.md."""
    text = (Path(checks.__file__).resolve().parents[2] / "docs/check-kinds.md").read_text()
    table = {}
    for line in text.splitlines():
        cells = [cell.strip(" `") for cell in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] in ("check", "rescore") and cells[1].isdigit():
            table.setdefault(cells[0], {})[int(cells[1])] = cells[2]
    return table


class PairedLiveContextTests(unittest.TestCase):
    def test_single_check_binds_its_live_parent_identity(self):
        config = declaration.load(declaration.MECHANICS)
        config["checks"][0]["module"] = "paired_teacher"
        server = {"model": "fixture/tp4", "health": {"serving": {"startup_nonce": "a" * 32}}}
        def execute(check, rows, cohort, context, *args):
            self.assertIs(context["_qualified_server"], server)
            row = {"passed": True}
            rows.append(row)
            return row
        with tempfile.TemporaryDirectory() as directory, patch.object(checks, "load", return_value=(config, declaration.MECHANICS)), \
                patch.object(checks.Client, "identity", return_value=server), patch.object(gate, "run_check", side_effect=execute):
            status, _ = command("check", "api", "--model", "mechanics", "--url", "http://unused", "--receipts", directory)
        self.assertEqual(status, 0)


class RescoreTests(unittest.TestCase):
    def gate_run(self, root, passed=None, exits=None, variant=None):
        """A mechanics gate run of the shipped fixture policy (as the
        checkpoint variant `variant`, if given); its summary."""
        config = declaration.load(declaration.MECHANICS)
        if variant is not None:
            config = {**config, "variant": variant}
        with patch.object(gate.Client, "identity", return_value={"model": "fixture/tp4"}), \
             patch.object(gate.subprocess, "check_output", return_value="revision\n"), \
             patch.object(gate.subprocess, "run", side_effect=fake_worker(passed or {}, exits=exits)), \
             contextlib.redirect_stdout(io.StringIO()):
            gate.run(config, declaration.MECHANICS, "quick", "http://unused", None, {}, root, True)
        return next(root.glob("*/summary.json"))

    def rescore(self, *paths, limits="current", strict=True):
        with tempfile.TemporaryDirectory() as scratch:
            report = Path(scratch) / "report.json"
            status, output = command("rescore", "--model", "mechanics", "--limits", limits, "--json", report,
                                     *(["--require-recorded"] if strict else []), *paths)
            return status, json.loads(report.read_text()) if report.is_file() else None, output

    def test_a_stored_run_rescores_to_its_recorded_verdicts(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = self.gate_run(Path(temporary), {"structured": False})
            for limits in ("current", "recorded"):
                with self.subTest(limits=limits):
                    status, report, _ = self.rescore(summary.parent, limits=limits)
                    self.assertEqual(status, 0)
                    self.assertEqual([(row["id"], row["recorded"], row["status"], row["differing_metrics"])
                                      for row in report["checks"]],
                                     [("api", "pass", "pass", []), ("structured", "fail", "fail", [])])
                    self.assertEqual(report["gates"], [{"source": str(summary), "recorded_passed": False,
                                                        "passed": False}])
            # Without --require-recorded the exit status is the verdict's.
            self.assertEqual(self.rescore(summary, strict=False)[0], 1)

    def test_a_run_rescores_only_against_its_own_variant(self):
        """A MODEL.VARIANT gate run is rescored against that variant's
        declaration, and a run of the model's own against the model's."""
        with tempfile.TemporaryDirectory() as temporary:
            base = self.gate_run(Path(temporary) / "base")
            variant = self.gate_run(Path(temporary) / "variant", variant="demo")
            config = declaration.load(declaration.MECHANICS)
            for selected, run, refused in (({**config, "variant": "demo"}, variant, None),
                                           (config, variant, "a gate run of fixture.demo, not mechanics"),
                                           ({**config, "variant": "demo"}, base, "a gate run of fixture, not mechanics")):
                with self.subTest(selected=selected.get("variant"), run=run.parent.parent.name), \
                        patch.object(checks, "load", return_value=(selected, declaration.MECHANICS)):
                    if refused is None:
                        status, report, _ = self.rescore(run.parent)
                        self.assertEqual(status, 0)
                        self.assertEqual(report["gates"][0]["source"], str(run))
                    else:
                        error = io.StringIO()
                        with patch.object(sys, "argv", ["checks.py", "rescore", "--model", "mechanics",
                                                        str(run.parent)]), \
                                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(error):
                            self.assertEqual(checks.main(), 2)
                        self.assertIn(refused, error.getvalue())

    def test_current_limits_change_a_verdict(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = self.gate_run(Path(temporary))
            config = declaration.load(declaration.MECHANICS)
            config["checks"][0]["rules"][0]["value"] = False
            with patch.object(checks, "load", return_value=(config, declaration.MECHANICS)):
                status, report, _ = self.rescore(summary)
            self.assertEqual(status, 1)
            self.assertEqual([(row["recorded"], row["status"]) for row in report["checks"]],
                             [("pass", "fail"), ("pass", "pass")])
            # The run's own policy still judges as recorded.
            self.assertEqual(self.rescore(summary, limits="recorded")[0], 0)

    def test_a_receipt_changed_after_the_run_is_insufficient(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = self.gate_run(Path(temporary))
            receipt = next(summary.parent.glob("checks/*api.json"))
            receipt.write_text(receipt.read_text() + " ")
            status, report, _ = self.rescore(summary, strict=False)
            self.assertEqual(status, 3)
            self.assertEqual(report["checks"][0]["status"], "insufficient")
            self.assertEqual(self.rescore(summary)[0], 1)

    def test_a_moved_run_reads_its_receipts_where_they_are(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = self.gate_run(Path(temporary) / "before")
            moved = Path(temporary) / "after"
            summary.parent.parent.rename(moved)
            status, report, _ = self.rescore(moved / summary.parent.name)
            self.assertEqual(status, 0)
            self.assertTrue(all(row["receipt"].startswith(str(moved)) for row in report["checks"]))

    def test_a_receipt_under_a_former_check_id_is_read_under_the_current_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "20261001T000000Z-former-api.json"
            receipt.write_text(json.dumps({"schema": "chopinfer-qualification/v1", "check": "former-api",
                                           "passed": True, "mechanics_only": True, "errors": [], "skipped": []}))
            with patch.object(declaration, "renamed", return_value={"former-api": "api"}):
                status, report, _ = self.rescore(receipt, strict=False)
            self.assertEqual(status, 0)
            self.assertEqual((report["checks"][0]["id"], report["checks"][0]["recorded_id"]), ("api", "former-api"))
            # A single receipt has no recorded policy.
            self.assertEqual(self.rescore(receipt, limits="recorded")[0], 2)

    def test_a_worker_that_exited_nonzero_keeps_its_failure(self):
        # The API worker wrote a passing receipt, then exited 1: the gate failed it.
        with tempfile.TemporaryDirectory() as temporary:
            summary = self.gate_run(Path(temporary), exits={"api": 1})
            recorded = json.loads(summary.read_text())
            self.assertEqual(recorded["checks"][0].get("exit_code"), 1)
            self.assertFalse(recorded["passed"])
            status, report, _ = self.rescore(summary, strict=False)
            api = report["checks"][0]
            self.assertEqual((api["id"], api["status"], api["receipt_status"]), ("api", "fail", "pass"))
            self.assertEqual(api["error"], "check exited 1")
            self.assertFalse(report["gates"][0]["passed"])
            self.assertEqual(status, 1)
            self.assertEqual(self.rescore(summary)[0], 0)

    def test_a_linked_receipt_cannot_erase_a_failed_worker_exit(self):
        # The same failed exit, with the receipt linked on the row (receipt and digest).
        with tempfile.TemporaryDirectory() as temporary:
            summary = self.gate_run(Path(temporary), exits={"api": 1})
            value = json.loads(summary.read_text())
            receipt = next(summary.parent.glob("checks/*api.json"))
            value["checks"][0].update(receipt=str(receipt), sha256=gate.digest(receipt))
            summary.write_text(json.dumps(value))
            status, report, _ = self.rescore(summary, strict=False)
            api = report["checks"][0]
            self.assertEqual((api["status"], api["receipt_status"], api["error"]), ("fail", "pass", "check exited 1"))
            self.assertFalse(report["gates"][0]["passed"])
            self.assertEqual(status, 1)

    def test_a_check_the_current_cohort_selects_must_be_in_the_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary = self.gate_run(Path(temporary))
            value = json.loads(summary.read_text())
            value["checks"] = [row for row in value["checks"] if row["id"] != "api"]
            summary.write_text(json.dumps(value))
            status, report, _ = self.rescore(summary, strict=False)
            self.assertEqual([(row["id"], row["recorded"], row["status"]) for row in report["checks"]],
                             [("api", "absent", "insufficient"), ("structured", "pass", "pass")])
            self.assertFalse(report["gates"][0]["passed"])
            self.assertEqual(status, 3)
            # A check a later declaration adds to the cohort is required the same way.
            summary = self.gate_run(Path(temporary) / "full")
            config = declaration.load(declaration.MECHANICS)
            config["checks"].append(dict(config["checks"][0], id="api_again"))
            config["profiles"]["quick"]["checks"].append("api_again")
            with patch.object(checks, "load", return_value=(config, declaration.MECHANICS)):
                status, report, _ = self.rescore(summary, strict=False)
            self.assertEqual(report["checks"][-1]["id"], "api_again")
            self.assertEqual(report["checks"][-1]["status"], "insufficient")
            self.assertFalse(report["gates"][0]["passed"])

    def test_a_reference_file_is_the_record_the_kind_scores_against(self):
        # Determinism scores one arm's record against the other arm's; the
        # gate reads the other arm from the check's reference_file.
        import test_determinism
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right, differing = root / "ar.json", root / "spec.json", root / "spec-differs.json"
            left.write_text(json.dumps(test_determinism.recorded(arm="ar")))
            right.write_text(json.dumps(test_determinism.recorded(arm="spec")))
            value = test_determinism.recorded(arm="spec")
            for row in value["requests"]:
                if row["name"] == "capital":
                    row["logprobs"][0] -= 1e-6
            differing.write_text(json.dumps(value))
            config = declaration.load(declaration.MECHANICS)
            config["checks"].append({
                "id": "determinism", "module": "determinism", "args": ["compare", "{ar}", "{spec}"],
                "timeout_seconds": 60, "json_file": "{spec}", "reference_file": "{ar}", "files": ["ar", "spec"],
                "rules": [{"path": ["passed"], "op": "eq", "value": True,
                           "source": "docs/check-kinds.md#determinism-determinismv1"}]})
            config["profiles"]["quick"]["checks"].append("determinism")
            path = root / "declaration.json"
            path.write_text(json.dumps(config))
            rows = {}
            for name, spec in (("equal", right), ("differs", differing)):
                with patch.object(gate.Client, "identity", return_value={"model": "fixture/tp4"}), \
                     patch.object(gate.subprocess, "check_output", return_value="revision\n"), \
                     patch.object(gate.subprocess, "run", side_effect=fake_worker({})), \
                     contextlib.redirect_stdout(io.StringIO()):
                    gate.run(config, path, "quick", "http://unused", None, {"ar": str(left), "spec": str(spec)},
                             root / name, True)
                summary = next((root / name).glob("*/summary.json"))
                rows[name] = (summary, json.loads(summary.read_text())["checks"][-1])
            summary, row = rows["equal"]
            self.assertEqual((row["id"], row["passed"], row["receipt"]), ("determinism", True, str(right)))
            self.assertEqual((row["reference_file"], row["reference_sha256"]), (str(left), gate.digest(left)))
            self.assertFalse(rows["differs"][1]["passed"])
            self.assertEqual(rows["differs"][1]["error"], "threshold predicate failed")
            with patch.object(checks, "load", return_value=(config, path)):
                status, report, _ = self.rescore(summary, strict=False)
                self.assertEqual(report["checks"][-1]["status"], "pass")
                self.assertEqual(self.rescore(rows["differs"][0], strict=False)[1]["checks"][-1]["status"], "fail")
                # The other arm's record must be the one the gate read.
                left.write_text(left.read_text() + " ")
                status, report, _ = self.rescore(summary, strict=False)
            self.assertEqual(status, 3)
            self.assertEqual(report["checks"][-1]["status"], "insufficient")

    def test_a_reference_record_must_match_its_declared_pair_paths_and_be_admitted(self):
        import test_determinism
        config = declaration.load(declaration.MECHANICS)
        config["checks"].append({
            "id": "determinism", "module": "determinism", "args": ["compare", "{ar}", "{spec}"],
            "timeout_seconds": 60, "json_file": "{spec}", "reference_file": "{ar}", "files": ["ar", "spec"],
            "reference_equal": [["run_id"], ["server", "health", "serving", "source_commit"]],
            "rules": [{"path": ["passed"], "op": "eq", "value": True,
                       "source": "docs/check-kinds.md#determinism-determinismv1"}]})
        config["profiles"]["quick"]["checks"].append("determinism")
        check = config["checks"][-1]

        def records(**changes):
            ar, spec = test_determinism.recorded(arm="ar"), test_determinism.recorded(arm="spec")
            for value in (ar, spec):
                value["server"] = {"model": "fixture/tp4", "health": {"serving": {"source_commit": "c" * 40}}}
            for key, change in changes.items():
                change(ar)
            return ar, spec

        cases = {"same": records(),
                 "another run": records(run=lambda ar: ar.update(run_id="other")),
                 "another build": records(build=lambda ar: ar["server"]["health"]["serving"].update(
                     source_commit="d" * 40)),
                 "no build": records(build=lambda ar: ar["server"]["health"]["serving"].pop("source_commit")),
                 "unqualified options": records(options=lambda ar: ar["server"]["health"]["serving"].update(
                     model_options={"x": 1}, model_options_qualified=False))}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "declaration.json"
            path.write_text(json.dumps(config))
            for name, (ar, spec) in cases.items():
                with self.subTest(case=name):
                    left, right = root / f"{name}-ar.json", root / f"{name}-spec.json"
                    left.write_text(json.dumps(ar))
                    right.write_text(json.dumps(spec))
                    with patch.object(gate.Client, "identity", return_value={"model": "fixture/tp4"}), \
                         patch.object(gate.subprocess, "check_output", return_value="revision\n"), \
                         patch.object(gate.subprocess, "run", side_effect=fake_worker({})), \
                         contextlib.redirect_stdout(io.StringIO()):
                        gate.run(config, path, "quick", "http://unused", None, {"ar": str(left), "spec": str(right)},
                                 root / name, True)
                    row = json.loads(next((root / name).glob("*/summary.json")).read_text())["checks"][-1]
                    self.assertEqual(row["passed"], name == "same", row.get("error"))
                    # Rescore applies the same pair rules to the stored records.
                    stored = {**row, "receipt": str(right), "sha256": gate.digest(right),
                              "reference_file": str(left), "reference_sha256": gate.digest(left)}
                    status = checks.rescore_row(root / name / "summary.json", stored, check, config, path, True,
                                                None)["status"]
                    self.assertEqual(status, "pass" if name == "same" else "fail")

    def test_a_moved_run_reads_its_run_local_reference_file_where_it_is(self):
        # Records inside the run directory move with it; an external one does not.
        import test_determinism
        config = declaration.load(declaration.MECHANICS)
        check = {"id": "determinism", "module": "determinism", "args": ["compare", "{ar}", "{spec}"],
                 "timeout_seconds": 60, "json_file": "{spec}", "reference_file": "{ar}", "files": ["ar", "spec"],
                 "rules": [{"path": ["passed"], "op": "eq", "value": True,
                            "source": "docs/check-kinds.md#determinism-determinismv1"}]}
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            (run / "checks").mkdir(parents=True)
            ar, spec = run / "checks/ar.json", run / "checks/spec.json"
            ar.write_text(json.dumps(test_determinism.recorded(arm="ar")))
            spec.write_text(json.dumps(test_determinism.recorded(arm="spec")))
            policy = run / "policy.json"
            policy.write_text(json.dumps(config))
            row = {"id": "determinism", "passed": True, "receipt": str(spec), "sha256": gate.digest(spec),
                   "reference_file": str(ar), "reference_sha256": gate.digest(ar), "log": str(run / "determinism.log")}
            self.assertEqual(checks.rescore_row(run / "summary.json", row, check, config, policy, True, None)["status"],
                             "pass")
            moved = Path(temporary) / "moved"
            run.rename(moved)
            self.assertEqual(checks.rescore_row(moved / "summary.json", row, check, config, moved / "policy.json",
                                                True, None)["status"], "pass")
            (moved / "checks/ar.json").write_text((moved / "checks/ar.json").read_text() + " ")
            self.assertEqual(checks.rescore_row(moved / "summary.json", row, check, config, moved / "policy.json",
                                                True, None)["status"], "insufficient")

    def test_rescore_applies_the_declarations_input_rules_the_live_preflight_does(self):
        # Records the current preflight refuses fail rescore too, though the
        # run scored them; an input the run did not record, or that changed
        # since, leaves nothing to judge.
        import test_determinism
        config = declaration.load(declaration.MECHANICS)
        check = {"id": "determinism", "module": "determinism", "args": ["compare", "{ar}", "{spec}"],
                 "timeout_seconds": 60, "json_file": "{spec}", "reference_file": "{ar}",
                 "files": ["ar", "spec", "approval"],
                 "input_rules": {
                     "ar": [{"path": ["server", "model"], "op": "eq", "value": "{model}", "source": "x"}],
                     "spec": [{"path": ["image", "included"], "op": "eq", "value": True, "source": "x"}],
                     "approval": [{"path": ["approved"], "op": "eq", "value": True, "source": "x"}]},
                 "rules": [{"path": ["passed"], "op": "eq", "value": True,
                            "source": "docs/check-kinds.md#determinism-determinismv1"}]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def rescored(name, image=True, model="served", approved=True, summary=None, changed=False, earlier=False):
                run = root / name
                (run / "checks").mkdir(parents=True)
                ar, spec, approval = run / "checks/ar.json", run / "checks/spec.json", root / f"{name}-approval.json"
                ar.write_text(json.dumps(test_determinism.recorded(arm="ar", image=image)))
                spec.write_text(json.dumps(test_determinism.recorded(arm="spec", image=image)))
                approval.write_text(json.dumps({"approved": approved}))
                policy = run / "policy.json"
                policy.write_text(json.dumps(config))
                inputs = {"ar": str(ar), "spec": str(spec), "approval": str(approval)}
                summary = summary or {"server": {"model": model}, "inputs": inputs,
                                      "input_sha256": {key: gate.digest(Path(value)) for key, value in inputs.items()}}
                context = {**inputs, "model": model}
                try:
                    gate.prerequisites({"checks": [check]}, ["determinism"], "quick", context, {})
                except ValueError as error:
                    preflight = str(error)
                else:
                    preflight = None
                if earlier:
                    summary = {**summary, "inputs": {k: v for k, v in inputs.items() if k != "approval"},
                               "earlier": {"approval": str(approval)}}
                if changed:
                    approval.write_text(json.dumps({"approved": approved}) + " ")
                row = {"id": "determinism", "passed": True, "receipt": str(spec), "sha256": gate.digest(spec),
                       "reference_file": str(ar), "reference_sha256": gate.digest(ar),
                       "log": str(run / "determinism.log")}
                result = checks.rescore_row(run / "summary.json", row, check, config, policy, True, None,
                                            summary=summary)
                return preflight, result["status"], result.get("error")

            self.assertEqual(rescored("control")[:3], (None, "pass", None))
            self.assertEqual(rescored("recorded-earlier", earlier=True)[:3], (None, "pass", None))
            for name, change, refused in [("no-image", {"image": False}, "spec: input predicate failed"),
                                          ("another-model", {"model": "other"}, "ar: input predicate failed"),
                                          ("unapproved", {"approved": False}, "approval: input predicate failed")]:
                with self.subTest(case=name):
                    preflight, status, error = rescored(name, **change)
                    self.assertIn(refused, preflight)
                    self.assertEqual(status, "fail")
                    self.assertIn(refused, error)
            for name, change in [("changed", {"changed": True}),
                                 ("unrecorded", {"summary": {"server": {"model": "served"}}})]:
                with self.subTest(case=name):
                    self.assertEqual(rescored(name, **change)[1], "insufficient")

    def test_the_shipped_renames_are_declared(self):
        self.assertEqual(declaration.renamed("qwen38"), {"qwen38-teacher": "teacher"})
        self.assertEqual(declaration.renamed("mimo26"), {})


class RecordedInputMapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "teacher.json"
        self.value = {"passed": True, "model": "served", "mechanics_only": False, "errors": []}
        self.path.write_text(json.dumps(self.value))
        self.check = {"input_rules": {"teacher": [
            {"path": ["passed"], "op": "eq", "value": True, "source": "fixture"},
            {"path": ["model"], "op": "eq", "value": "{model}", "source": "fixture"}]}}
        self.summary = {"server": {"model": "served"}, "inputs": {}, "earlier": {"teacher": str(self.path)},
                        "input_sha256": {"teacher": gate.digest(self.path)}}

    def verdict(self, summary=None, check=None, value=None, named=None):
        return checks.input_verdict({}, self.check if check is None else check,
                                    self.value if value is None else value, named,
                                    self.summary if summary is None else summary)

    def test_explicit_input_prior_receipt_and_identical_duplicates_agree(self):
        for location in ("inputs", "earlier", "both"):
            with self.subTest(location=location):
                summary = copy.deepcopy(self.summary)
                if location != "earlier": summary["inputs"]["teacher"] = str(self.path)
                if location == "inputs": summary["earlier"] = {}
                self.assertIsNone(self.verdict(summary))

    def test_conflicting_duplicate_paths_are_refused_in_either_direction(self):
        foreign = self.path.with_name("other.json")
        foreign.write_bytes(self.path.read_bytes())
        for left, right in ((self.path, foreign), (foreign, self.path)):
            with self.subTest(left=left):
                summary = {**self.summary, "inputs": {"teacher": str(left)}, "earlier": {"teacher": str(right)}}
                result = self.verdict(summary)
                self.assertIsInstance(result, dict)
                self.assertEqual(result["status"], "insufficient")
                self.assertIn("conflicting", result["error"])

    def test_malformed_recorded_maps_are_refused(self):
        for field in ("inputs", "earlier", "input_sha256"):
            for malformed in (None, [], False, "path", {"teacher": None}, {1: "path"}):
                with self.subTest(field=field, value=malformed):
                    result = self.verdict({**self.summary, field: malformed})
                    self.assertEqual(result["status"], "insufficient")
                    self.assertIn("maps", result["error"])

    def test_missing_digest_missing_file_and_changed_bytes_are_refused(self):
        for name in ("digest", "file", "changed"):
            with self.subTest(case=name):
                summary = copy.deepcopy(self.summary)
                self.path.write_text(json.dumps(self.value))
                if name == "digest": summary["input_sha256"] = {}
                elif name == "file": self.path.unlink()
                else: self.path.write_bytes(self.path.read_bytes() + b" ")
                self.assertEqual(self.verdict(summary)["status"], "insufficient")

    def test_earlier_inputs_keep_the_live_quality_and_predicate_rules(self):
        for fields in ({"passed": False}, {"model": "other"}, {"mechanics_only": True}, {"errors": ["failure"]}):
            with self.subTest(fields=fields):
                value = {**self.value, **fields}
                self.path.write_text(json.dumps(value))
                summary = {**self.summary, "input_sha256": {"teacher": gate.digest(self.path)}}
                with self.assertRaises(ValueError):
                    gate.judge_input("teacher", value, self.check["input_rules"]["teacher"], {"model": "served"})
                self.assertEqual(self.verdict(summary)["status"], "fail")

    def test_json_and_reference_inputs_keep_their_bound_record_routes(self):
        self.path.unlink()
        for key in ("json_file", "reference_file"):
            with self.subTest(key=key):
                check = {**self.check, key: "{teacher}"}
                self.assertIsNone(self.verdict(check=check, named=self.value))
                changed = {**self.value, "passed": False}
                self.assertEqual(self.verdict(check=check, value=changed, named=changed)["status"], "fail")


class CheckCommandTests(unittest.TestCase):
    def run_check(self, root, passed):
        with patch.object(gate.Client, "identity", return_value={"model": "fixture/tp4"}), \
             patch.object(gate.subprocess, "check_output", return_value="revision\n"), \
             patch.object(gate.subprocess, "run", side_effect=fake_worker({"api": passed})):
            status, output = command("check", "api", "--model", "mechanics", "--url", "http://unused",
                                     "--receipts", root)
        result = json.loads(Path(json.loads(output.splitlines()[-1])["receipt"]).read_text())
        return status, result

    def test_one_check_runs_through_its_kind(self):
        with tempfile.TemporaryDirectory() as temporary:
            status, result = self.run_check(Path(temporary), True)
            self.assertEqual(status, 0)
            self.assertEqual((result["check"], result["kind"], result["status"]), ("api", "api", "pass"))
            self.assertTrue(all(rule["passed"] for rule in result["row"]["predicates"]))
            self.assertTrue(Path(result["row"]["receipt"]).is_file())
            status, result = self.run_check(Path(temporary), False)
            self.assertEqual((status, result["status"], result["row"]["error"]), (1, "fail", "threshold predicate failed"))

    def test_an_undeclared_check_is_a_usage_error(self):
        status, _ = command("check", "absent", "--model", "mechanics", "--url", "http://unused")
        self.assertEqual(status, 2)

    def test_the_documented_exit_statuses_are_the_commands(self):
        # docs/check-kinds.md states each command's exit statuses; a CLI that
        # follows it must return the same code for the same case.
        table = documented_exits()
        self.assertEqual(table["check"], {0: "pass, waived, informational or not applicable", 1: "fail",
                                          2: "usage or validation error"})
        self.assertEqual(table["rescore"], {0: "every verdict passes", 1: "a verdict fails",
                                            2: "usage or validation error",
                                            3: "insufficient evidence and no failure"})
        with tempfile.TemporaryDirectory() as temporary:
            bad = Path(temporary) / "inputs.json"
            bad.write_text("{broken")
            for arguments in (["check", "absent", "--model", "mechanics", "--url", "http://unused"],
                              ["check", "api", "--model", "no-such-model", "--url", "http://unused"],
                              ["check", "api", "--model", "mechanics", "--url", "http://unused", "--inputs", bad],
                              ["check", "api", "--model", "mechanics", "--url", "http://unused",
                               "--with", "api=/no/such/receipt"],
                              ["rescore", "--model", "mechanics", bad]):
                with self.subTest(arguments=arguments):
                    self.assertEqual(command(*arguments)[0], 2)

    def test_list_names_each_declared_check_kind(self):
        status, output = command("list", "--model", "qwen38", "--json")
        self.assertEqual(status, 0)
        kinds = {row["id"]: row["kind"] for row in json.loads(output)["checks"]}
        self.assertEqual((kinds["teacher"], kinds["api"], kinds["structured"], kinds["soak"]),
                         ("teacher_forced", "api", "structured", "soak"))



class EvalsCompareGateTests(unittest.TestCase):
    """Each shipped configuration whose full cohort selects evals_compare,
    given the evals_candidate receipt an earlier check run recorded under its
    own run ID, passes as the gate judges it: through `checks.py check`,
    `checks.py rescore` of its receipt and the correctness runner's replay of
    the recorded row. Its other arm is a stored one (stored_<check>) or one
    recorded elsewhere and held to the compare's input rules; another check
    may share its receipt pattern."""

    IDENTITY = {"model": "synthetic", "health": {"serving": {"version": "1.0", "source_commit": "c" * 40}}}

    @staticmethod
    def declared():
        """(model, configuration) of each previewed configuration whose full
        cohort selects evals_compare."""
        import release_preview as preview
        found = []
        for model in preview.declaration.previewed(preview.ROOT):
            facts = preview.model_facts(model)
            for configuration in preview.configurations(model):
                if "evals_compare" in preview.configuration_facts(facts, configuration)["full"]:
                    found.append((model, configuration))
        return found

    def arm(self, root, run, arm):
        """A passing evals arm of 1,319 items: its results file and receipt."""
        results = root / "cache/runs" / run / f"{arm}.jsonl"
        results.parent.mkdir(parents=True, exist_ok=True)
        results.write_text("".join(json.dumps({"id": f"gsm8k-{index}", "suite": "gsm8k", "group": "gsm8k",
                                               "body_sha256": f"{index:064x}", "correct": True, "extracted": "1",
                                               "content": "#### 1", "finish_reason": "stop"}) + "\n"
                                   for index in range(1319)))
        path = root / arm / f"20261014T000000Z-{run}-evals-{arm}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schema": "chopinfer-qualification/v1", "check": f"evals-{arm}", "passed": True,
            "mechanics_only": False, "skipped": [], "errors": [], "server": copy.deepcopy(self.IDENTITY),
            "run_id": run, "arm": arm, "items": 1319, "suites": {"gsm8k": {"count": 1319}}, "datasets": {},
            "workload_sha256": "w", "concurrency": 1, "long_concurrency": 1, "top_logprobs": 0, "limit": None,
            "at": "t", "results": str(results), "results_sha256": hashlib.sha256(results.read_bytes()).hexdigest()}))
        return path

    def recorded_arm(self, root, key, rules, inputs):
        """An arm recorded elsewhere (input `key`) that holds its input rules."""
        values = {tuple(rule["path"]): rule["value"].format_map(inputs) if isinstance(rule["value"], str)
                  else rule["value"] for rule in rules}
        path = self.arm(root, values.get(("run_id",), f"{key}-run"), values.get(("arm",), key))
        value = json.loads(path.read_text())
        for keys, required in values.items():
            parent = value
            for part in keys[:-1]:
                parent = parent.setdefault(part, {})
            parent[keys[-1]] = required
        path.write_text(json.dumps(value))
        return path

    def test_each_declared_compare_passes_dispatch_rescore_and_replay(self):
        import release_preview as preview
        declared = self.declared()
        self.assertTrue(declared)
        styles = set()
        for model, configuration in declared:
            name = model if configuration == "default" else f"{model}.{configuration}"
            with self.subTest(name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config, _ = checks.load(name)
                check = next(entry for entry in config["checks"] if entry["id"] == "evals_compare")
                candidate = self.arm(root, "candidate-check-run", "candidate")
                stored = preview.stored_inputs(check)
                rules = check.get("input_rules", {})
                inputs = {key: f"{key}-value" for key in re.findall(r"\{(\w+)\}", json.dumps(rules))
                          if key not in rules and key != "model"}
                for key, key_rules in rules.items():
                    inputs[key] = str(self.recorded_arm(root, key, key_rules, inputs))
                inputs.update({key: str(self.arm(root, "stored-run", "candidate")) for key in stored})
                styles.add("stored" if stored else "recorded")
                labels = [f"evals_compare/{store}" for store in preview.STORES] if stored else ["evals_compare"]
                (root / "inputs.json").write_text(json.dumps(inputs))
                args = SimpleNamespace(model=name, id="evals_compare", inputs=root / "inputs.json",
                                       with_receipt=[f"evals_candidate={candidate}"], receipts=root / "checks",
                                       cohort="full", url="http://127.0.0.1:1", served_model="synthetic")
                with patch.object(checks, "Client", lambda *a, **k: SimpleNamespace(identity=lambda *a, **k:
                                                                                     self.IDENTITY)), \
                        contextlib.redirect_stdout(io.StringIO()):
                    code = checks.check_command(args)
                check_run = next((root / "checks").glob("*/check.json"))
                result = json.loads(check_run.read_text())
                self.assertEqual((code, result["status"]), (0, "pass"), result["row"])
                receipt = Path(result["row"]["receipt"])
                value = json.loads(receipt.read_text())
                self.assertEqual((value["items"], value["arm_runs"]["candidate"]), (1319, "candidate-check-run"))

                status, _ = command("rescore", "--model", name, "--check", "evals_compare",
                                    "--json", root / "rescore.json", receipt)
                rows = json.loads((root / "rescore.json").read_text())["checks"]
                self.assertEqual((status, [(row["id"], row["status"]) for row in rows]),
                                 (0, [("evals_compare", "pass")]), rows)
                # Without --check the receipt is found as before, by its
                # receipt pattern, which another check may share.
                sharing = [entry["id"] for entry in config["checks"] if entry.get("receipt") == check["receipt"]]
                status, _ = command("rescore", "--model", name, "--json", root / "found.json", receipt)
                self.assertEqual(status, 0 if sharing == ["evals_compare"] else 2)

                for label in labels:
                    directory = root / "spec/checks" / label.replace("/", "-")
                    (directory / "run").mkdir(parents=True)
                    (directory / "run/check.json").write_text(json.dumps(result))
                    (directory / "inputs.json").write_text(json.dumps(inputs))
                    (directory / "checks.log.exit").write_text("0\n")
                facts = preview.model_facts(model)
                replayed = []
                run = SimpleNamespace(replay=True, inputs=inputs, used_features=set(), facts=facts,
                                      install=root / "install",
                                      row=lambda config, arm, label, status, **fields: replayed.append(
                                          (label, status, fields.get("error"))),
                                      stores={store: SimpleNamespace(receipt=lambda *a: None)
                                              for store in preview.STORES})
                context = {"facts": preview.configuration_facts(facts, configuration), "config_name": configuration,
                           "records": {}, "checks": {"evals_candidate": str(candidate)}, "suite": "full",
                           "fill": {"url": "http://127.0.0.1:1"}, "env": dict(os.environ)}
                preview.run_check(run, context, {"check": "evals_compare"}, root / "spec", "spec")
                self.assertEqual(replayed, [(label, "pass", None) for label in labels])
        # Both kinds of other arm, and a receipt pattern another check shares, are covered.
        self.assertEqual(styles, {"stored", "recorded"})

    def test_rescore_check_refuses_a_receipt_of_another_check(self):
        model, configuration = self.declared()[0]
        name = model if configuration == "default" else f"{model}.{configuration}"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / "20261014T000000Z-evals-compare.json"
            receipt.write_text(json.dumps({"schema": "chopinfer-qualification/v1", "check": "evals-compare",
                                           "passed": True, "items": 1}))
            # A file another check's receipt pattern names, a receipt whose
            # own check is another declared check, and an undeclared check.
            status, _ = command("rescore", "--model", name, "--check", "evals_candidate", receipt)
            self.assertEqual(status, 2)
            named = root / "20261014T000000Z-evals-candidate.json"
            named.write_text(json.dumps({"schema": "chopinfer-qualification/v1", "check": "evals_compare",
                                         "passed": True, "items": 1}))
            status, _ = command("rescore", "--model", name, "--check", "evals_candidate", named)
            self.assertEqual(status, 2)
            status, _ = command("rescore", "--model", name, "--check", "absent", receipt)
            self.assertEqual(status, 2)


class RescoreSelectorTests(unittest.TestCase):
    """`rescore --check ID` names the check of a receipt whose own identity
    leaves it open; it never overrides one the receipt already has."""

    STAMP = "20261014T000000Z"

    @staticmethod
    def shipped():
        """(declaration name, config) of the first previewed declaration with
        a check that declares no receipt pattern and a check whose pattern a
        stamped file name matches alone."""
        for name in declaration.previewed():
            config, _ = checks.load(name)
            patternless = [check["id"] for check in config["checks"]
                           if "receipt" not in check and not check.get("informational_reason")]
            if patternless and any("receipt" in check for check in config["checks"]):
                return name, config, patternless[0]
        raise AssertionError("no shipped declaration has a patternless check")

    def test_a_patternless_selector_cannot_take_a_receipt_its_file_name_identifies(self):
        name, config, patternless = self.shipped()
        patterns = {check["id"]: check["receipt"] for check in config["checks"] if "receipt" in check}
        owner, file_name = next((check, pattern.replace("*", self.STAMP)) for check, pattern in patterns.items()
                                if [other for other, candidate in patterns.items()
                                    if fnmatch.fnmatch(pattern.replace("*", self.STAMP), candidate)] == [check])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = root / file_name
            # A module-style check name, no declared check's ID.
            receipt.write_text(json.dumps({"schema": "chopinfer-qualification/v1", "check": "module-receipt",
                                           "passed": True, "mechanics_only": False, "errors": [], "items": 1}))
            status, _ = command("rescore", "--model", name, "--json", root / "found.json", receipt)
            self.assertNotEqual(status, 2)
            self.assertEqual(json.loads((root / "found.json").read_text())["checks"][0]["id"], owner)
            status, _ = command("rescore", "--model", name, "--check", patternless, receipt)
            self.assertEqual(status, 2)
            status, _ = command("rescore", "--model", name, "--check", owner, "--json", root / "owner.json",
                                receipt)
            self.assertNotEqual(status, 2)
            self.assertEqual(json.loads((root / "owner.json").read_text())["checks"][0]["id"], owner)
            # A capture no identity names is judged as the patternless check named.
            capture = root / "capture.receipt"
            self.assertFalse([check for check, pattern in patterns.items() if fnmatch.fnmatch(capture.name, pattern)])
            capture.write_text(receipt.read_text())
            status, _ = command("rescore", "--model", name, "--check", patternless, "--json",
                                root / "capture.json", capture)
            self.assertNotEqual(status, 2)
            self.assertEqual(json.loads((root / "capture.json").read_text())["checks"][0]["id"], patternless)

if __name__ == "__main__":
    unittest.main()

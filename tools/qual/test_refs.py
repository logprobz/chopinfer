"""CPU evidence for gate input authentication and reference lock adoption."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import difflib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import common
import declaration
import refs
from qual_test_config import clean_checkout


class ReferenceLockTests(unittest.TestCase):
    def make_lock(self, root, config):
        value = refs.snapshot(config, root)
        path = refs.lock_path(config["model"], root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path, value

    def test_object_records_do_not_confine_byte_pins(self):
        sha = 'a' * 64
        source = {'env': 'CHOPINFER_TEST_LAYER_INPUT', 'path': 'run'}
        lock = {'schema': 'chopinfer-qual-references/v1', 'model': 'synthetic',
                'entries': {'other': {'path': source, 'directory': False, 'members': {'.': sha}}},
                'objects': {sha: {'schema': 'binary', 'production': {'producer': 'runner-layer-reference'}}}}
        config = {'model': 'synthetic', 'inputs': {'other': source}}
        with patch.dict(os.environ, {}, clear=True):
            refs.validate_lock(lock, config)
            missing = copy.deepcopy(lock)
            missing['entries']['other']['members']['.'] = 'b' * 64
            with self.assertRaisesRegex(ValueError, 'invalid or missing object'):
                refs.validate_lock(missing, config)
            unnamed = copy.deepcopy(lock)
            del unnamed['objects'][sha]['schema']
            with self.assertRaisesRegex(ValueError, 'missing schema'):
                refs.validate_lock(unnamed, config)

    def test_external_reference_paths_keep_byte_pins_and_workload_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = root / "external"
            external.mkdir()
            fixture = external / "reference.json"
            fixture.write_text('{"schema": "synthetic.reference.v1"}')
            workload = external / "reference.workload.json"
            workload.write_text('{"schema": "synthetic.workload.v1"}')
            source = {"env": "CHOPINFER_TEST_EXTERNAL_REFERENCE", "path": "reference.json"}
            config = {"model": "synthetic", "inputs": {}, "qualification": {
                "references": {"teacher": {"path": source, "workload": {**source, "path": "reference.workload.json"}}}}}
            with patch.dict(os.environ, {source["env"]: str(external)}):
                self.make_lock(root, config)
                self.assertEqual(common.reference_path("teacher", config, root), fixture)
                self.assertEqual(common.pinned_kind(fixture, config, root), "teacher")
                from kinds import source as historical_source
                self.assertEqual(historical_source(fixture, "unavailable-git-revision", external=True), fixture)
                self.assertEqual(refs.sources(config)["workload:teacher"],
                                 {**source, "path": "reference.workload.json"})
                self.assertEqual(refs.verify_input(config, "reference:teacher", fixture, root),
                                 fixture.read_bytes())
                fixture.write_text('{"schema": "tampered"}')
                with self.assertRaisesRegex(ValueError, "sha256 differs"):
                    refs.verify_input(config, "reference:teacher", fixture, root)
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "needs CHOPINFER_TEST_EXTERNAL_REFERENCE"):
                    common.reference_path("teacher", config, root)

    def test_portable_integer_fixture_must_match_the_adopted_rows(self):
        fixture = {"images": [{"width": 3, "height": 2, "grid": [1,2,3], "integer_patch_sha256": "digest"}],
                   "overrides": {}, "versions": {"torch": "version"}, "checkpoint_config_sha256": "config"}
        record = {"cases": [{"source": {"shape": [2,3,3]}, "grid": [1,2,3],
                             "stages": {"patches": {"integer": {"sha256": "digest"}}}}],
                  "binding": {"processor": {"overrides": {}}, "environment": {"torch": "version"},
                              "configs": {"preprocessor_config.json": "config"}}}
        refs.verify_preprocessing_fixture(fixture, record)
        changed = copy.deepcopy(fixture)
        changed["images"][0]["integer_patch_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "portable integer fixture"):
            refs.verify_preprocessing_fixture(changed, record)

    def test_tampered_legacy_fixture_names_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "sample.json"
            path.write_text('[{"prompt": "original"}]\n')
            config = {"model": "synthetic", "inputs": {"api_samples": "sample.json"}}
            self.make_lock(root, config)
            self.assertEqual(refs.verify_input(config, "api_samples", path, root), path.read_bytes())
            path.write_text('[{"prompt": "tampered"}]\n')
            with self.assertRaisesRegex(ValueError, "lock entry api_samples: sha256 differs"):
                refs.verify_input(config, "api_samples", path, root)

    def test_a_layer_reference_is_read_only_as_a_declared_byte_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference"
            reference.mkdir()
            (reference / "run.json").write_text('{}')
            config = {"model": "synthetic", "inputs": {"layer_reference": "reference"}}
            self.make_lock(root, config)
            refs.verify_input(config, "layer_reference", reference, root)
            (reference / "run.json").write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "sha256 differs"):
                refs.verify_input(config, "layer_reference", reference, root)
            # No operator evidence policy admits an undeclared run under a lock.
            undeclared = {"model": "synthetic", "inputs": {}}
            refs.lock_path("synthetic", root).write_text(json.dumps(refs.snapshot(undeclared, root)))
            with self.assertRaisesRegex(ValueError, "lock entry layer_reference: not adopted"):
                refs.verify_input(undeclared, "layer_reference", reference, root)

    def test_directory_members_cannot_be_added_or_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "media"
            directory.mkdir()
            (directory / "image.png").write_bytes(b"image")
            config = {"model": "synthetic", "inputs": {"media": "media"}}
            self.make_lock(root, config)
            (directory / "extra.png").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "lock entry media:.*extra.png"):
                refs.verify_input(config, "media", directory, root)

    def test_adopt_refuses_refresh_and_lists_changed_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "sample.json"
            fixture.write_text("{}")
            config = {"model": "synthetic", "inputs": {"sample": str(fixture)}}
            lock, original = self.make_lock(root, config)
            fixture.write_text('{"changed": true}')
            changed = refs.snapshot(config, root, previous=original)
            errors = io.StringIO()
            with patch("declaration.load", return_value=config), \
                    patch.object(refs, "lock_path", return_value=lock), \
                    patch.object(refs, "snapshot", return_value=changed), \
                    patch("sys.argv", ["refs", "adopt", "--model", "synthetic"]), redirect_stderr(errors):
                with self.assertRaises(SystemExit) as refusal:
                    refs.main()
            self.assertEqual(refusal.exception.code, 2)
            self.assertIn("changed: sample/.", errors.getvalue())
            self.assertIn("refuse changed lock", errors.getvalue())
            self.assertEqual(json.loads(lock.read_text()), original)

    def test_adopt_refuses_an_unbound_numerical_reference_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "reference.json"
            fixture.write_text("{}")
            config = {"model": "synthetic", "inputs": {}, "qualification": {
                "references": {"teacher": {"path": str(fixture)}}}}
            lock = Path(temporary) / "lock.json"
            errors = io.StringIO()
            with patch("declaration.load", return_value=config), patch.object(refs, "lock_path", return_value=lock), \
                    patch("sys.argv", ["refs", "adopt", "--model", "synthetic"]), redirect_stderr(errors):
                with self.assertRaises(SystemExit):
                    refs.main()
            self.assertIn("lock entry reference:teacher: . is not runner-bound; runner binding required",
                          errors.getvalue())
            self.assertFalse(lock.exists())

    def test_gate_admission_refuses_a_reference_object_without_its_runner_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "reference.json"
            fixture.write_text("{}")
            config = {"model": "synthetic", "inputs": {}, "qualification": {
                "references": {"teacher": {"path": "reference.json"}}}}
            _, lock = self.make_lock(root, config)
            sha = lock["entries"]["reference:teacher"]["members"]["."]
            runner = {"producer": "runner-binding", "refset_sha256": "a" * 64,
                      "assembled_fixture_sha256": "b" * 64, "workload_sha256": "c" * 64}
            invalid = [None, {"producer": "legacy-unknown"}, {**runner, "producer": "runner-workload"},
                       *({**runner, field: "A" * 64} for field in runner if field != "producer"),
                       *({key: value for key, value in runner.items() if key != field} for field in runner)]
            for production in invalid:
                with self.subTest(production=production):
                    lock["objects"][sha].pop("production", None)
                    if production is not None:
                        lock["objects"][sha]["production"] = production
                    refs.lock_path("synthetic", root).write_text(json.dumps(lock))
                    refs.validate_lock(lock, config, root)
                    for options in ({"policy_only": True}, {"optional": False}):
                        with self.assertRaisesRegex(ValueError, "reference:teacher: . is not runner-bound"):
                            refs.verify_config(config, root, **options)
            lock["objects"][sha]["production"] = runner
            refs.lock_path("synthetic", root).write_text(json.dumps(lock))
            refs.verify_config(config, root, optional=False)

    def test_readoption_keeps_old_records_and_scopes_and_pins_new_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first.json", root / "second.json"
            first.write_text("{}")
            second.write_text("[]")
            directory = root / "media"
            directory.mkdir()
            (directory / "read.png").write_bytes(b"read")
            (directory / "unread.txt").write_bytes(b"unread")
            config = {"model": "synthetic",
                      "inputs": {"first": "first.json", "second": "second.json", "media": "media"}}
            original = refs.snapshot(config, root)
            self.assertEqual(set(original["entries"]["media"]["members"]), {"read.png", "unread.txt"})
            # An earlier adoption's record and a directory scope written into the lock.
            original["objects"][refs.digest(first)]["production"] = {"producer": "legacy-unknown",
                                                                    "adoption_date": "2026-09-24"}
            original["entries"]["media"]["scope"] = ["read.png"]
            del original["entries"]["media"]["members"]["unread.txt"]
            refs.validate_lock(original, config, root)
            second.write_text("[1]")
            updated = refs.snapshot(config, root, previous=original)
            self.assertEqual(updated["objects"][refs.digest(first)], original["objects"][refs.digest(first)])
            self.assertEqual(updated["objects"][refs.digest(second)], {"schema": "legacy-json-array"})
            self.assertEqual(updated["entries"]["media"]["scope"], ["read.png"])
            self.assertEqual(set(updated["entries"]["media"]["members"]), {"read.png"})
            refs.validate_lock(updated, config, root)
            unscoped = copy.deepcopy(updated)
            unscoped["entries"]["media"]["scope"] = ["read.png", "unread.txt"]
            with self.assertRaisesRegex(ValueError, "lock entry media: directory scope differs from its members"):
                refs.validate_lock(unscoped, config, root)

    def test_committed_locks_verify_without_optional_environment(self):
        for model in ("mimo26", "qwen38"):
            config = declaration.load(declaration.path(model))
            refs.verify_config(config)

    def test_an_unproven_object_records_only_its_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "sample.json"
            fixture.write_text("{}")
            (root / "receipt.json").write_text(json.dumps({"files": {"sample.json": refs.digest(fixture)}}))
            config = {"model": "synthetic", "inputs": {"sample": "sample.json"}}
            _, lock = self.make_lock(root, config)
            self.assertEqual(lock["objects"][refs.digest(fixture)], {"schema": "legacy-json-object"})

    def test_readoption_verifies_candidate_workload_before_pinning_new_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = declaration.load(declaration.path("mimo26"))
            config = {"model": "synthetic", "inputs": {}, "qualification": {
                "checkpoint": original["qualification"]["checkpoint"], "references": {
                    "teacher": copy.deepcopy(original["qualification"]["references"]["teacher"])}}}
            spec = config["qualification"]["references"]["teacher"]
            fixture = root / spec["path"]
            fixture.parent.mkdir(parents=True)
            shutil.copyfile(common.ROOT / spec["path"], fixture)
            workload = root / "candidate-workload.json"
            source = common.reference_workload(common.ROOT / spec["path"])
            workload.write_text(json.dumps(json.loads(source.read_text()), indent=4))
            spec["workload"] = str(workload)
            self.assertNotEqual(refs.digest(workload), refs.digest(source))
            lock = refs.lock_path("synthetic", root)
            lock.parent.mkdir(parents=True)
            candidate = refs.snapshot(config, root)
            old_sha = refs.digest(source)
            candidate["objects"][old_sha] = copy.deepcopy(candidate["objects"][refs.digest(workload)])
            candidate["entries"]["workload:teacher"]["members"]["."] = old_sha
            lock.write_text(json.dumps(candidate))
            value = json.loads(fixture.read_text())
            with patch.dict(os.environ, {"CHOPINFER_QUAL_MODEL_ROOT": str(root)}):
                with self.assertRaisesRegex(ValueError, "workload:teacher: sha256 differs"):
                    common.verified_binding("teacher", value, config)
                adopted = refs.snapshot(config, root)
                self.assertEqual(adopted["entries"]["workload:teacher"]["members"]["."], refs.digest(workload))
                self.assertEqual(adopted["objects"][refs.digest(workload)]["production"]["producer"], "runner-workload")

    def test_frontend_entries_keep_their_records_and_byte_pins(self):
        config = declaration.load(declaration.path("mimo26"))
        lock = json.loads(refs.lock_path("mimo26").read_text())
        for key in ("tokenizer_goldens", "chat_renders"):
            # The adoption record of an unproven frontend capture stays information.
            self.assertEqual(lock["entries"][key]["producer_policy"], "legacy-authenticated")
        lock["entries"]["chat_renders"].pop("reason")
        lock["entries"]["chat_renders"].pop("producer_policy")
        refs.validate_lock(lock, config)
        lock["entries"]["chat_renders"]["members"]["."] = "0" * 64
        with self.assertRaisesRegex(ValueError, "lock entry chat_renders: member . references an invalid"):
            refs.validate_lock(lock, config)

    def test_directory_scope_excludes_unread_goldens(self):
        config = declaration.load(declaration.path("qwen38"))
        lock = json.loads(refs.lock_path("qwen38").read_text())
        self.assertNotIn("weight-plan-goldens.json", lock["entries"]["fixture_root"]["members"])
        self.assertNotIn("README.md", lock["entries"]["fixture_root"]["members"])
        self.assertNotIn("media_cases", lock["entries"])
        self.assertNotIn("media_images", lock["entries"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / config["inputs"]["fixture_root"]
            for member in lock["entries"]["fixture_root"]["members"]:
                target = fixture / member
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(common.ROOT / config["inputs"]["fixture_root"] / member, target)
            lock_file = refs.lock_path("qwen38", root)
            lock_file.parent.mkdir(parents=True)
            lock_file.write_text(json.dumps(lock))
            (fixture / "weight-plan-goldens.json").write_bytes(b"unrelated changed golden")
            refs.verify_input(config, "fixture_root", fixture, root)
            extra = fixture / "frontend/unadopted.json"
            extra.write_text("{}")
            refs.verify_input(config, "fixture_root", fixture, root)
            with self.assertRaisesRegex(ValueError, "member (frontend/)?unadopted.json is not adopted"):
                refs.fixture_bytes(extra, config, root)
            (fixture / "frontend/chat-renders.json").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "fixture_root: sha256 differs for frontend/chat-renders.json"):
                refs.read_member(config, "fixture_root", fixture / "frontend/chat-renders.json", root)



class AdoptionKeepsTheLockTests(unittest.TestCase):
    """`adopt` replaces the entries the declaration names and the objects
    only they pinned; it keeps every other section, record and object of the
    current lock byte for byte, or refuses."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "sample.json").write_text("{}")
        (self.root / "shared.json").write_text("[]")
        (self.root / "own.json").write_text('{"own": true}')
        self.config = {"model": "synthetic", "inputs": {"sample": "sample.json", "shared": "shared.json"}}
        lock = refs.snapshot(self.config, self.root)
        self.sample = refs.digest(self.root / "sample.json")
        own, shared = refs.digest(self.root / "own.json"), refs.digest(self.root / "shared.json")
        # A configuration's own entry (an object only it pins) and one that
        # pins the same bytes as a base entry; an operator record; an
        # earlier adoption's object nothing pins any more.
        lock["configurations"] = {"demo": {"entries": {
            "sample": {"path": "own.json", "directory": False, "members": {".": own}},
            "shared": {"path": "shared.json", "directory": False, "members": {".": shared}}}}}
        lock["objects"][own] = {"schema": "legacy-json-object",
                                "production": {"producer": "legacy-unknown", "adoption_date": "2026-10-01"}}
        lock["entries"]["operator:layer_reference"] = {"path": {"input": "layer_reference"},
                                                      "producer_policy": "operator-evidence", "reason": "r"}
        self.orphan = "e" * 64
        lock["objects"][self.orphan] = {"schema": "chop-frontend-openai-v1"}
        # An entry's own fields beside its pin.
        lock["entries"]["sample"].update(role="frontend-reference", producer_policy="legacy-authenticated",
                                         reason="no producer run")
        refs.validate_lock(lock, self.config, self.root)
        self.lock = refs.lock_path("synthetic", self.root)
        self.lock.parent.mkdir(parents=True)
        # Committed locks keep their own key order (configurations last here).
        self.lock.write_text(json.dumps(lock, indent=2) + "\n")
        self.assertNotEqual(self.lock.read_text(), json.dumps(lock, indent=2, sort_keys=True) + "\n")
        self.previous = lock

    def adopt(self, *options):
        errors = io.StringIO()
        with patch("declaration.load", return_value=self.config), patch("declaration.named", return_value="x"), \
                patch.object(refs, "lock_path", return_value=self.lock), \
                patch.object(refs, "root_path", return_value=self.root), \
                patch("sys.argv", ["refs", "adopt", "--model", "synthetic", *options]), \
                redirect_stderr(errors), redirect_stdout(io.StringIO()) as output:
            try:
                refs.main()
                code = 0
            except SystemExit as stopped:
                code = stopped.code
        return code, errors.getvalue(), output.getvalue()

    def kept(self, value):
        return {"configurations": value.get("configurations"),
                "operator": value["entries"].get("operator:layer_reference"),
                "sample fields": {field: value["entries"]["sample"].get(field)
                                  for field in ("role", "producer_policy", "reason")},
                "objects": {sha: value["objects"].get(sha) for sha in self.previous["objects"]
                            if sha != self.sample}}

    def test_an_unchanged_declaration_adopts_the_same_bytes(self):
        # Before the fix the snapshot dropped the configurations, the
        # operator record and the objects only they or no entry pinned.
        before = self.lock.read_bytes()
        code, errors, _ = self.adopt()
        self.assertEqual((code, errors), (0, ""))
        self.assertEqual(self.lock.read_bytes(), before)

    def test_a_changed_input_replaces_only_its_own_entry_and_object(self):
        (self.root / "sample.json").write_text('{"changed": true}')
        code, errors, _ = self.adopt("--replace-sha256", refs.digest(self.lock))
        self.assertEqual(code, 0, errors)
        self.assertIn("changed: sample/.", errors)
        value = json.loads(self.lock.read_text())
        new = refs.digest(self.root / "sample.json")
        self.assertEqual(value["entries"]["sample"]["members"], {".": new})
        self.assertNotIn(self.sample, value["objects"])
        self.assertIn(f"objects: 1 no longer pinned: {self.sample[:12]}", errors)
        self.assertEqual(json.dumps(self.kept(value), sort_keys=True),
                         json.dumps(self.kept(self.previous), sort_keys=True))
        refs.validate_lock(value, self.config, self.root)
        # The rewrite touches only the changed member and its object, in
        # the lock's own key order.
        before = (json.dumps(self.previous, indent=2) + "\n").splitlines()
        lines = [line for line in difflib.unified_diff(before, self.lock.read_text().splitlines(), lineterm="", n=0)
                 if line[:1] in "+-" and line[:3] not in ("---", "+++")]
        new = refs.digest(self.root / "sample.json")
        self.assertEqual(sorted(line[1:].strip() for line in lines if line.startswith("-")),
                         sorted([f'".": "{self.sample}"', f'"{self.sample}": {{', '"schema": "legacy-json-object"', "},"]))
        self.assertEqual(sorted(line[1:].strip() for line in lines if line.startswith("+")),
                         sorted([f'".": "{new}"', f'"{new}": {{', '"schema": "legacy-json-object"', "},"]))

    def test_an_entry_s_own_fields_do_not_move_to_another_path(self):
        (self.root / "moved.json").write_text("{}")
        moved = {**self.config, "inputs": {**self.config["inputs"], "sample": "moved.json"}}
        with self.assertRaisesRegex(ValueError, "lock entry sample: its path changed; adopt cannot carry its "
                                                "producer_policy, reason, role"):
            refs.adoption(moved, self.previous, self.root)

    def test_a_dry_run_reports_and_writes_nothing(self):
        before = self.lock.read_bytes()
        (self.root / "sample.json").write_text('{"changed": true}')
        code, errors, output = self.adopt("--dry-run")
        self.assertEqual(code, 0, errors)
        self.assertIn("changed: sample/.", errors)
        self.assertIn("dry run", output)
        self.assertEqual(self.lock.read_bytes(), before)

    def test_a_lock_adopt_cannot_keep_whole_is_refused(self):
        def section(value):
            value["exports"] = {}
        def configuration_key(value):
            value["configurations"]["demo"]["note"] = "x"
        def missing_object(value):
            del value["objects"][value["configurations"]["demo"]["entries"]["sample"]["members"]["."]]
        def other_model(value):
            value["model"] = "other"
        for change, refusal in ((section, "unknown section exports"),
                                (configuration_key, "configuration demo: adopt can keep only its entries"),
                                (missing_object, "configuration demo entry sample: object .* is missing"),
                                (other_model, "another schema or model")):
            previous = copy.deepcopy(self.previous)
            change(previous)
            with self.subTest(change=change.__name__):
                with self.assertRaisesRegex(ValueError, refusal):
                    refs.adoption(self.config, previous, self.root)
        self.lock.write_text(json.dumps({**self.previous, "exports": {}}))
        before = self.lock.read_bytes()
        code, errors, _ = self.adopt("--replace-sha256", refs.digest(self.lock))
        self.assertEqual(code, 2)
        self.assertIn("unknown section exports", errors)
        self.assertEqual(self.lock.read_bytes(), before)

    def test_a_runner_proven_object_keeps_its_own_fields_or_is_refused(self):
        # Regression: snapshot rebuilds a runner-proven object's
        # record from its proof, and adoption dropped the record's other
        # fields although a configuration and another entry still pin it.
        # The first shipped model whose teacher reference is a runner-bound
        # file in the tree.
        original = next(config for config in (declaration.load(path) for path in declaration.shipped()
                                              if path.name == declaration.NAME)
                        if isinstance(config["qualification"].get("references", {}).get("teacher", {}).get("path"), str)
                        and (common.ROOT / config["qualification"]["references"]["teacher"]["path"]).is_file())
        spec = copy.deepcopy(original["qualification"]["references"]["teacher"])
        config = {"model": "synthetic", "inputs": {"exact": spec["path"]}, "qualification": {
            "checkpoint": original["qualification"]["checkpoint"], "references": {"teacher": spec}}}
        for source in (common.ROOT / spec["path"], common.reference_workload(common.ROOT / spec["path"])):
            copied = self.root / source.relative_to(common.ROOT)
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copied)
        previous = refs.snapshot(config, self.root)
        sha = previous["entries"]["reference:teacher"]["members"]["."]
        self.assertEqual(previous["entries"]["exact"]["members"]["."], sha)
        self.assertEqual(previous["objects"][sha]["production"]["producer"], "runner-binding")
        previous["objects"][sha]["reason"] = "kept"
        previous["objects"][sha]["production"]["adoption_date"] = "2026-10-01"
        previous["configurations"] = {"demo": {"entries": {
            "reference:teacher": copy.deepcopy(previous["entries"]["reference:teacher"])}}}
        refs.validate_lock(previous, config, self.root)
        refs.gate_reference_producers(config, previous)
        value = refs.adoption(config, previous, self.root)
        self.assertEqual(value["objects"][sha], previous["objects"][sha])
        self.assertEqual(refs.encoded(value, previous), refs.encoded(previous, previous))
        for field, change in (("refset_sha256", lambda record: record["production"].update(refset_sha256="0" * 64)),
                              ("producer", lambda record: record["production"].update(producer="legacy-unknown")),
                              ("schema", lambda record: record.update(schema="other"))):
            conflicting = copy.deepcopy(previous)
            change(conflicting["objects"][sha])
            with self.subTest(conflict=field):
                with self.assertRaisesRegex(ValueError, f"object {sha}: the lock's record differs from what "
                                                        f"adoption proved in {field}"):
                    refs.adoption(config, conflicting, self.root)
                # Through adopt: refused, and the lock is not written.
                self.lock.write_bytes(refs.encoded(conflicting, conflicting))
                before = self.lock.read_bytes()
                self.config = config
                code, errors, _ = self.adopt("--replace-sha256", refs.digest(self.lock))
                self.assertEqual(code, 2)
                self.assertIn(f"in {field}", errors)
                self.assertEqual(self.lock.read_bytes(), before)

    def preprocessing_reference(self, environment):
        """A preprocessing reference of one case under `pre/case`, its record
        binding `environment`; the config declaring it; its tensors' digests."""
        from tools.model_sdk.diagnostic import canonical
        from tools.refrun import keys, preprocessing
        case = self.root / "pre/case"
        case.mkdir(parents=True, exist_ok=True)
        source = preprocessing.write_tensor(case, "source.bin", b"\x01", "u8", (1,))
        rows = preprocessing.write_tensor(case, "rows.bin", preprocessing.packed([1.0], "f32"), "f32", (1,))
        record = {"schema": preprocessing.SCHEMA,
                  "binding": {"code": keys.repo_files(preprocessing.CODE), "requirements": {},
                              "environment": environment},
                  "cases": [{"name": "sample", "kind": "audio", "source": source,
                             "stages": {"patches": {"f32": rows}}}]}
        record["sha256"] = keys.sha256(canonical(record))
        (case / "reference.json").write_text(json.dumps(record, indent=2) + "\n")
        config = {**self.config, "inputs": {**self.config["inputs"], "media_preprocessing_reference": "pre"}}
        return config, record, (source["sha256"], rows["sha256"])

    def adopt_preprocessing(self):
        """A lock holding the preprocessing reference recorded under one
        environment, its tensors' records with fields of their own; then the
        same tensors recorded under another (the record alone changes)."""
        self.config, first, tensors = self.preprocessing_reference({"torch": "1"})
        previous = json.loads(json.dumps(refs.adoption(self.config, self.previous, self.root)))
        for sha in tensors:
            self.assertEqual(previous["objects"][sha]["production"],
                             {"producer": "runner-preprocessing-binding", "reference_sha256": first["sha256"]})
            previous["objects"][sha]["reason"] = "kept"
            previous["objects"][sha]["production"]["adoption_date"] = "2026-10-01"
        self.lock.write_bytes(refs.encoded(previous, self.previous))
        self.previous = previous
        _, second, same = self.preprocessing_reference({"torch": "2"})
        self.assertEqual(same, tensors)
        self.assertNotEqual(second["sha256"], first["sha256"])
        return tensors, second

    def test_a_fresh_proof_of_the_same_bytes_renews_their_record(self):
        # Before the fix a changed preprocessing record over unchanged
        # tensors was refused (reference_sha256 differs), so the entry and
        # its objects had to be dropped from the lock by hand.
        tensors, second = self.adopt_preprocessing()
        previous = self.previous
        manifest = previous["entries"]["media_preprocessing_reference"]["members"]["case/reference.json"]
        code, errors, _ = self.adopt("--replace-sha256", refs.digest(self.lock))
        self.assertEqual(code, 0, errors)
        self.assertIn("media_preprocessing_reference/case/rows.bin: producer metadata changed", errors)
        # The new record replaces the old one, and each tensor's production
        # names it; everything else stays.
        path = self.root / "pre/case/reference.json"
        expected = copy.deepcopy(previous)
        expected["entries"]["media_preprocessing_reference"]["members"]["case/reference.json"] = refs.digest(path)
        del expected["objects"][manifest]
        expected["objects"][refs.digest(path)] = {"schema": refs.schema(path), "production": {
            "producer": "runner-preprocessing-binding", "reference_sha256": second["sha256"]}}
        for sha in tensors:
            expected["objects"][sha]["production"]["reference_sha256"] = second["sha256"]
        value = json.loads(self.lock.read_text())
        self.assertEqual(value, expected)
        self.assertEqual(value["objects"][tensors[0]],
                         {"schema": "legacy-bin", "reason": "kept",
                          "production": {"producer": "runner-preprocessing-binding",
                                         "reference_sha256": second["sha256"], "adoption_date": "2026-10-01"}})
        refs.validate_lock(value, self.config, self.root)
        # Adopting again leaves the renewed lock as it is.
        before = self.lock.read_bytes()
        self.assertEqual(self.adopt()[:2], (0, ""))
        self.assertEqual(self.lock.read_bytes(), before)
        # Only the proof's own fields change: another producer is refused.
        conflicting = copy.deepcopy(previous)
        conflicting["objects"][tensors[1]]["production"]["producer"] = "legacy-unknown"
        with self.assertRaisesRegex(ValueError, f"object {tensors[1]}: the lock's record differs from what "
                                                f"adoption proved in producer"):
            refs.adoption(self.config, conflicting, self.root)

    def test_a_fresh_proof_does_not_renew_bytes_a_kept_configuration_pins(self):
        tensors, _ = self.adopt_preprocessing()
        previous = copy.deepcopy(self.previous)
        # The configuration shares one tensor; the other alone would renew.
        previous["configurations"]["demo"]["entries"]["media_preprocessing_reference"] = {
            "path": "pre", "directory": True, "members": {"case/rows.bin": tensors[1]}, "scope": ["case/rows.bin"]}
        self.lock.write_bytes(refs.encoded(previous, self.previous))
        before = self.lock.read_bytes()
        with self.assertRaisesRegex(ValueError, f"object {tensors[1]}: the lock's record differs from what "
                                                f"adoption proved in reference_sha256"):
            refs.adoption(self.config, previous, self.root)
        code, errors, _ = self.adopt("--replace-sha256", refs.digest(self.lock))
        self.assertEqual(code, 2)
        self.assertIn("in reference_sha256", errors)
        self.assertEqual(self.lock.read_bytes(), before)

    def test_a_fresh_proof_of_other_bytes_is_refused(self):
        tensors, _ = self.adopt_preprocessing()
        before = self.lock.read_bytes()
        from tools.refrun import preprocessing
        (self.root / "pre/case/rows.bin").write_bytes(preprocessing.packed([2.0], "f32"))
        with self.assertRaisesRegex(ValueError, "preprocessing tensor digest mismatch"):
            refs.adoption(self.config, self.previous, self.root)
        code, errors, _ = self.adopt("--replace-sha256", refs.digest(self.lock))
        self.assertEqual(code, 2)
        self.assertIn("preprocessing tensor digest mismatch", errors)
        self.assertEqual(self.lock.read_bytes(), before)

    def test_a_configuration_of_the_model_lock_is_not_adopted_as_its_own_lock(self):
        config = {**self.config, "variant": "demo"}
        errors = io.StringIO()
        with patch("declaration.load", return_value=config), patch("declaration.named", return_value="x"), \
                patch("declaration.configurations", return_value={"demo": {}}), \
                patch.object(refs, "lock_path", return_value=self.lock), \
                patch("sys.argv", ["refs", "adopt", "--model", "synthetic.demo"]), redirect_stderr(errors):
            with self.assertRaises(SystemExit) as refusal:
                refs.main()
        self.assertEqual(refusal.exception.code, 2)
        self.assertIn("configuration demo is pinned in the model lock's configurations", errors.getvalue())


class GateEntryPointTests(unittest.TestCase):
    def environment(self):
        # A clean host still has the chopinfer binary, which validates declarations.
        return {**{key: value for key, value in os.environ.items()
                   if not key.startswith("CHOPINFER_") or key == "CHOPINFER_EVAL"},
                "CHOPINFER_LOCAL_ENV": str(Path(tempfile.gettempdir()) / "absent-chopinfer-local.env")}

    def test_list_inputs_on_a_clean_host_keeps_unset_paths_null(self):
        # A clean host has no local.env either, so run from a copy without
        # the checkout's: release_gate.py would read it before the absent
        # CHOPINFER_LOCAL_ENV.
        with clean_checkout() as checkout:
            for model in ("mimo26", "qwen38", "mechanics"):
                result = subprocess.run([sys.executable, "-B", str(checkout / "tools/qual/release_gate.py"),
                                         "--model", model, "--list-inputs"], env=self.environment(),
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                paths = json.loads(result.stdout)["paths"]
                if model == "qwen38":
                    self.assertNotIn("teacher_long", paths)
                    self.assertIsNone(paths["needle"])

    def test_list_inputs_refuses_a_tampered_locked_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = declaration.load(declaration.path("mimo26"))
            config["qualification"]["mechanics_only"] = True
            for location in ("fixtures/mimo26/teacher-texts.json", "fixtures/mimo26/frontend/chat-renders.json",
                             "fixtures/mimo26/frontend/tokenizer-golden.json"):
                target = root / location
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(common.ROOT / location, target)
            lock = refs.lock_path("mimo26", root)
            lock.parent.mkdir(parents=True)
            shutil.copyfile(refs.lock_path("mimo26"), lock)
            path = root / "gate.json"
            path.write_text(json.dumps(config))
            command = [sys.executable, "-B", str(common.ROOT / "tools/qual/release_gate.py"),
                       "--config", str(path), "--model-root", str(root), "--list-inputs"]
            result = subprocess.run(command, env=self.environment(), capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            (root / "fixtures/mimo26/frontend/chat-renders.json").write_text("{}")
            result = subprocess.run(command, env=self.environment(), capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn("lock entry chat_renders: sha256 differs", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_gate_execution_validates_the_lock_before_http(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = declaration.load(declaration.path("mimo26"))
            config["qualification"]["mechanics_only"] = True
            committed = json.loads(refs.lock_path("mimo26").read_text())
            lock_file = refs.lock_path("mimo26", root)
            lock_file.parent.mkdir(parents=True)
            policy = root / "gate.json"
            policy.write_text(json.dumps(config))
            unbound = copy.deepcopy(committed)
            sha = unbound["entries"]["reference:teacher"]["members"]["."]
            unbound["objects"][sha].pop("production")
            missing = copy.deepcopy(committed)
            del missing["entries"]["reference:teacher"]
            for lock, refusal in ((missing, "lock entry reference:teacher: not adopted"),
                                  (unbound, "lock entry reference:teacher: . is not runner-bound")):
                with self.subTest(refusal=refusal):
                    lock_file.write_text(json.dumps(lock))
                    result = subprocess.run([sys.executable, "-B", str(common.ROOT / "tools/qual/release_gate.py"),
                                             "--config", str(policy), "--model-root", str(root),
                                             "--url", "http://127.0.0.1:1", "--receipts", str(root / "receipts")],
                                            env=self.environment(), capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(refusal, result.stderr)
                    self.assertFalse((root / "receipts").exists())

    def test_worker_reports_missing_selected_input_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = declaration.load(declaration.path("mimo26"))
            frontend = root / config["inputs"]["frontend"]
            frontend.parent.mkdir(parents=True)
            shutil.copyfile(common.ROOT / config["inputs"]["frontend"], frontend)
            lock = refs.lock_path("mimo26", root)
            lock.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(refs.lock_path("mimo26"), lock)
            policy = root / "gate.json"
            policy.write_text(json.dumps(config))
            environment = {**self.environment(), "CHOPINFER_QUAL_CONFIG": str(policy),
                           "CHOPINFER_QUAL_MODEL_ROOT": str(root), "CHOPINFER_QUAL_RECEIPTS": str(root / "receipts")}
            # The media check's worker reads its pinned media reference first;
            # the teacher check is captured by chopinfer eval run.
            result = subprocess.run([sys.executable, "-B", str(common.ROOT / "tools/qual/release_gate.py"),
                                     "--worker", str(policy), "multimodal", "--url", "http://127.0.0.1:1"],
                                    env=environment, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn("refused: lock entry", result.stderr)
            self.assertIn("missing input", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_capture_comparison_authenticates_each_reference_read(self):
        import frontend_goldens as compare
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "fixtures/qwen38/frontend"
            reference.mkdir(parents=True)
            path = reference / "chat-renders.json"
            shutil.copyfile(common.ROOT / "fixtures/qwen38/frontend/chat-renders.json", path)
            lock = refs.lock_path("qwen38", root)
            lock.parent.mkdir(parents=True)
            shutil.copyfile(refs.lock_path("qwen38"), lock)
            with patch.dict(os.environ, {"CHOPINFER_QUAL_MODEL_ROOT": str(root),
                                          "CHOPINFER_QUAL_CONFIG": str(declaration.path("qwen38"))}):
                self.assertTrue(compare.load(reference, "chat-renders.json", reference=True))
                path.write_text("[]")
                with self.assertRaisesRegex(ValueError, "frontend_capture: sha256 differs for chat-renders.json"):
                    compare.load(reference, "chat-renders.json", reference=True)

    def test_layer_inputs_must_be_declared_byte_pins(self):
        import release_gate
        config = declaration.load(declaration.path("qwen38"))
        config["checks"] = [{"id": "layers", "args": [], "files": ["layer_reference"]}]
        config["inputs"].pop("layer_reference", None)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = {"layer_reference": str(root)}
            lock = json.loads(refs.lock_path("qwen38").read_text())
            lock["entries"].pop("layer_reference", None)
            target = refs.lock_path("qwen38", root)
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(lock))
            # The lock's former operator evidence record admits nothing.
            self.assertIn("operator:layer_reference", lock["entries"])
            with self.assertRaisesRegex(ValueError, "lock entry layer_reference: not adopted"):
                release_gate.prerequisites(config, ["layers"], "full", context, {}, model_root=root)

    def test_pinned_reference_uses_the_same_lock_root_and_refuses_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = declaration.load(declaration.path("mimo26"))
            spec = config["qualification"]["references"]["teacher"]
            fixture = root / spec["path"]
            fixture.parent.mkdir(parents=True)
            shutil.copyfile(common.ROOT / spec["path"], fixture)
            lock = refs.lock_path("mimo26", root)
            lock.parent.mkdir(parents=True)
            shutil.copyfile(refs.lock_path("mimo26"), lock)
            with patch.dict(os.environ, {"CHOPINFER_QUAL_MODEL_ROOT": str(root)}):
                self.assertEqual(common.reference_root(), root)
                common.pinned_reference("teacher", fixture, config)
                fixture.write_text("{}")
                with self.assertRaisesRegex(ValueError, "lock entry reference:teacher: sha256 differs"):
                    common.pinned_reference("teacher", fixture, config)
                lock.unlink()
                with self.assertRaisesRegex(ValueError, "reference lock is missing"):
                    common.pinned_reference("teacher", fixture, config)


class ConfigurationLockTests(unittest.TestCase):
    """A configuration reads its model's lock: the model's entries, with its
    own (configurations.<name>.entries) where they differ, for the inputs it
    declares."""

    def test_a_configuration_reads_its_own_entries_of_the_model_lock(self):
        compared = 0
        for path in declaration.shipped():
            if path.name != declaration.NAME:
                continue
            model = path.parent.name
            lock = json.loads(refs.lock_path(model).read_text())
            for name, spec in lock.get("configurations", {}).items():
                own = spec.get("entries", {})
                if "teacher_reference" not in own or own["teacher_reference"] == lock["entries"]["teacher_reference"]:
                    continue
                with self.subTest(model=model, config=name):
                    config = declaration.load(declaration.materialized(model, name))
                    loaded = refs.load_lock(config)
                    self.assertEqual(loaded["entries"]["teacher_reference"], own["teacher_reference"])
                    self.assertNotEqual(own["teacher_reference"], lock["entries"]["teacher_reference"])
                    self.assertEqual({key for key in loaded["entries"] if not key.startswith("operator:")},
                                     set(refs.sources(config)))
                    self.assertNotIn("configurations", loaded)
                    self.assertEqual(refs.load_lock(declaration.load(path))["entries"]["teacher_reference"],
                                     lock["entries"]["teacher_reference"])
                    compared += 1
        self.assertGreater(compared, 0)


class VariantDeclarationTests(unittest.TestCase):
    """models/<model>/qual.<variant>.json declares the model <model> and
    its variant, and keeps its own required reference lock beside qual.json."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = next(path for path in declaration.shipped() if path.name == declaration.NAME).parent.name
        value = json.loads(declaration.path(self.model).read_text())
        value.pop("configurations", None)
        self.path = declaration.path(self.model, self.root, "demo")
        self.path.parent.mkdir(parents=True)
        self.path.write_text(json.dumps({**value, "variant": "demo"}))

    def test_the_file_names_the_model_and_the_variant(self):
        self.assertEqual(declaration.variants(self.model, self.root), ["demo"])
        self.assertEqual(declaration.named(f"{self.model}.demo"), declaration.path(self.model, variant="demo"))
        config = declaration.load(self.path)
        self.assertEqual((config["model"], config["variant"]), (self.model, "demo"))
        value = json.loads(self.path.read_text())
        for field, stated, message in (("model", f"{self.model}-demo", f"model: must be {self.model}"),
                                       ("variant", "other", "variant: must be demo")):
            with self.subTest(field=field):
                self.path.write_text(json.dumps({**value, field: stated}))
                with self.assertRaisesRegex(declaration.DeclarationError, message):
                    declaration.load(self.path)

    def test_its_lock_is_required(self):
        config = declaration.load(self.path)
        self.assertEqual(refs.lock_path(self.model, self.root, "demo"),
                         self.root / f"models/{self.model}/qual-references.demo.lock.json")
        with patch.dict(os.environ, {"CHOPINFER_QUAL_MODEL_ROOT": str(self.root)}):
            with self.assertRaisesRegex(ValueError, "reference lock is missing"):
                refs.load_lock(config)
            with self.assertRaisesRegex(ValueError, "reference lock is missing"):
                common.input_bytes(next(iter(config["inputs"])), config=config)
            lock = json.loads(refs.lock_path(self.model, declaration.ROOT).read_text())
            refs.lock_path(self.model, variant="demo").write_text(json.dumps({**lock, "model": f"{self.model}-demo"}))
            with self.assertRaisesRegex(ValueError, "reference lock: model differs from declaration"):
                refs.load_lock(config)

    def test_a_declaration_outside_the_reference_root_is_refused(self):
        model = self.root / f"models/{self.model}/qual.json"
        model.write_text(declaration.path(self.model).read_text())
        environment = {key: value for key, value in os.environ.items() if key != "CHOPINFER_QUAL_MODEL_ROOT"}
        for path in (model, self.path):
            with self.subTest(path.relative_to(self.root).as_posix()), \
                    patch.dict(os.environ, {**environment, "CHOPINFER_QUAL_CONFIG": str(path)}, clear=True):
                with self.assertRaisesRegex(ValueError, "not the declaration under this workspace or the reference root"):
                    common.qualification_config()
                with self.assertRaisesRegex(ValueError, "not the declaration under this workspace or the reference root"):
                    common.input_bytes("frontend")
        with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(self.path),
                                     "CHOPINFER_QUAL_MODEL_ROOT": str(self.root)}):
            config = common.qualification_config()
            with self.assertRaisesRegex(ValueError, "reference lock is missing"):
                refs.load_lock(config)
        with tempfile.TemporaryDirectory() as other, \
                self.assertRaisesRegex(ValueError, "not the declaration under this workspace or the reference root"):
            declaration.require_rooted(self.path, Path(other))
        declaration.require_rooted(declaration.path(self.model), declaration.ROOT)
        declaration.require_rooted(declaration.MECHANICS, Path(self.root))

    def test_a_variant_the_workspace_declares_needs_its_lock_under_any_root(self):
        """A workspace that declares the variant (and not its model) still
        requires its lock."""
        config = declaration.load(self.path)
        with tempfile.TemporaryDirectory() as other, patch.object(refs, "ROOT", self.root), \
                self.assertRaisesRegex(ValueError, "reference lock is missing: qual-references.demo.lock.json"):
            refs.load_lock(config, Path(other))

    def test_an_unlocked_config_does_not_vouch_for_a_locked_input(self):
        config = declaration.load(declaration.path(self.model))
        lock = refs.load_lock(config)
        key, path = next((key, common.resolve_input(key, optional=True, config=config))
                         for key, entry in sorted(lock["entries"].items())
                         if key in config["inputs"] and not entry.get("directory", True)
                         and (common.resolve_input(key, optional=True, config=config) or Path()).is_file())
        value = json.loads(declaration.path(self.model).read_text())
        value["model"] = "unlocked-demo"
        unlocked = self.root / "elsewhere" / declaration.NAME
        unlocked.parent.mkdir()
        unlocked.write_text(json.dumps(value))
        self.assertIsNone(refs.load_lock(declaration.load(unlocked)))
        read = Path.read_bytes
        altered = read(path) + b"\n "
        with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(unlocked)}), \
                patch.object(Path, "read_bytes", lambda self: altered if self == path else read(self)), \
                self.assertRaisesRegex(ValueError, f"lock entry {key}: sha256 differs"):
            refs.fixture_bytes(path)

    def test_a_model_declaration_must_name_its_directory(self):
        import release_gate
        value = json.loads(declaration.path(self.model).read_text())
        value.pop("configurations", None)
        value["model"] = "audit-unknown-model"
        value.setdefault("qualification", {})["mechanics_only"] = False
        variant = self.root / f"models/{self.model}/qual.demo.json"
        variant.write_text(json.dumps({**value, "variant": "demo"}))
        message = f"model: must be {self.model}, the model its directory names"
        with self.assertRaisesRegex(ValueError, message):
            declaration.load(variant)
        path = self.root / f"models/{self.model}/qual.json"
        path.write_text(json.dumps(value))
        with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(path), "CHOPINFER_QUAL_MODEL_ROOT": str(self.root)}):
            with self.assertRaisesRegex(ValueError, message):
                common.qualification_config()
            with self.assertRaisesRegex(ValueError, message):
                common.input_bytes("frontend")
            with self.assertRaisesRegex(ValueError, message):
                refs.fixture_bytes(common.ROOT / "README.md")
        with patch("sys.argv", ["release_gate.py", "--config", str(path), "--model-root", str(self.root),
                                "--list-inputs"]), \
                redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit) as stopped:
            release_gate.main()
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn(message, error.getvalue())
        self.assertEqual(declaration.load(declaration.path(self.model))["model"], self.model)

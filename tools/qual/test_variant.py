"""External variants: another checkpoint of a supported model qualified from
an overlay kept outside the tree, with its lock, checkpoint lock and
references beside it (tools/qual/README.md#external-variants)."""
from qual_test_config import setUpModule  # noqa: F401

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import common
import declaration
import refs
import release_gate as gate

QUALIFICATION_RUNS_ONCE = "external variants built in temporary directories from the shipped declarations"
ROOT = Path(__file__).resolve().parents[2]
VARIANT = "requant"


def rehomed_model():
    """The first model (by name) whose qual.json pins one checkpoint lock (no
    registry) and a runner-bound teacher reference whose workload is in the
    repository: the reference a synthetic variant re-homes."""
    for path in sorted(ROOT.glob(f"models/*/{declaration.NAME}")):
        value = json.loads(path.read_text())
        pin = value["qualification"]["checkpoint"]
        teacher = value.get("references", {}).get("teacher")
        if ("registry" in pin or teacher is None or not isinstance(teacher["path"], str)
                or "workload" in teacher or not refs.lock_path(path.parent.name).is_file()):
            continue
        fixture = json.loads((ROOT / teacher["path"]).read_text())
        if fixture.get("binding") and common.reference_workload(ROOT / teacher["path"]).is_file():
            return path.parent.name
    raise unittest.SkipTest("no model pins a bound teacher reference with its workload in the repository")


def teacher_only(capabilities):
    """A base's capabilities narrowed to what a teacher-only variant covers."""
    narrowed = {}
    for key, value in capabilities.items():
        if key == "unqualified_media":
            continue
        narrowed[key] = (value if key == "text" else [] if key == "media" else "none" if key == "speculation"
                         else False)
    return narrowed


def restated(fixture, checkpoint):
    """`fixture` with its provenance naming `checkpoint` (such as another
    repository and revision), its binding's fixture and refset digests
    restated over it: its scores stay bound to the runs that wrote them."""
    from tools.refrun import cache
    fixture["provenance"]["checkpoint"].update(checkpoint)
    binding = fixture["binding"]
    binding["assembled_fixture_sha256"] = cache.fixture_sha256(fixture)
    binding["refset_sha256"] = cache.digest(cache.refset(binding["model"], binding["workload_sha256"],
                                                         cache.pairs(binding["modes"]),
                                                         binding["assembled_fixture_sha256"]))
    return fixture


def synthetic_variant(directory, model=None, variant=VARIANT, overlay=None, adopt=True, checkpoint=None):
    """An external variant of `model` under the variant root `directory`:
    the model's checkpoint lock copied beside the overlay, and its bound
    teacher reference re-homed there (its binding names the copy, so it
    still verifies, against the variant's own lock root) with its
    workload. With `checkpoint` (pin fields such as another repository and
    revision) the pin and the reference's provenance name that checkpoint
    instead of the model's. The overlay runs the teacher check; `overlay`
    changes it (None values remove a field). With `adopt`, its lock is
    adopted as `refs adopt` adopts it. Returns (model, the overlay as
    written)."""
    model = model or rehomed_model()
    base = json.loads(declaration.path(model).read_text())
    target = Path(directory) / model
    target.mkdir(parents=True, exist_ok=True)
    pin = dict(base["qualification"]["checkpoint"])
    lock = f"{variant}.checkpoint-files.json"
    shutil.copyfile(ROOT / pin["lock"], target / lock)
    spec = base["references"]["teacher"]
    fixture = json.loads((ROOT / spec["path"]).read_text())
    fixture["binding"]["lock"]["path"] = f"{model}/{lock}"
    if checkpoint:
        pin.update(checkpoint)
        restated(fixture, checkpoint)
    data = (json.dumps(fixture, indent=1) + "\n").encode()
    reference = f"{variant}.teacher-reference.json"
    (target / reference).write_bytes(data)
    shutil.copyfile(common.reference_workload(ROOT / spec["path"]), common.reference_workload(target / reference))
    pinned = {**{key: value for key, value in spec.items() if key != "path"}, "path": reference,
              "sha256": hashlib.sha256(data).hexdigest()}
    value = {"schema": "chopinfer-qual-variant/v1", "model": model, "checkpoint": f"{model.upper()}_REQUANT",
             "checks": ["teacher"], "quick": ["teacher"], "media": False,
             "references": {"teacher": pinned},
             "inputs": {key: reference for key, location in base["inputs"].items() if location == spec["path"]},
             "qualification": {"checkpoint": {**pin, "lock": lock}}}
    if "capabilities" in base:
        value["capabilities"] = teacher_only(base["capabilities"])
    for key, item in (overlay or {}).items():
        if item is None:
            value.pop(key, None)
        else:
            value[key] = item
    (target / f"{variant}.json").write_text(json.dumps(value, indent=1) + "\n")
    if adopt:
        with selected(directory):
            config = declaration.load(declaration.named(f"{model}.{variant}"))
            locked = refs.external_adoption(config)
            refs.validate_lock(locked, config)
            refs.gate_reference_producers(config, locked)
            (target / f"{variant}.lock.json").write_text(json.dumps(locked, indent=2, sort_keys=True) + "\n")
    return model, value


@contextlib.contextmanager
def selected(directory):
    """`directory` selected as the variant root, as `--variant-dir` selects it."""
    with patch.dict(os.environ, {declaration.VARIANTS_ENV: str(Path(directory).resolve())}):
        yield


class ExternalVariantTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "variants"
        self.model, self.overlay = synthetic_variant(self.root)
        self.name = f"{self.model}.{VARIANT}"

    def load(self):
        return declaration.load(declaration.named(self.name))

    def test_it_is_read_from_its_root_and_resolves_what_it_owns_there(self):
        with selected(self.root):
            config = self.load()
            self.assertTrue(declaration.external(config))
            self.assertEqual((config["model"], config["variant"]), (self.model, VARIANT))
            # It inherits no base reference: its own teacher alone.
            references = config["qualification"]["references"]
            self.assertEqual(set(references), {"teacher"})
            self.assertEqual(references["teacher"]["path"],
                             {"env": declaration.VARIANTS_ENV, "path": f"{self.model}/{VARIANT}.teacher-reference.json"})
            pin = common.checkpoint_pin(config)
            self.assertEqual(pin["lock"], f"{self.model}/{VARIANT}.checkpoint-files.json")
            path = common.reference_path("teacher", config)
            self.assertEqual(path, self.root / self.model / f"{VARIANT}.teacher-reference.json")
            value = common.pinned_reference("teacher", path, config)
            self.assertEqual(common.verified_binding("teacher", value, config)["status"], "bound")
            refs.verify_config(config, optional=True)
            # The rules and limits are the base declaration's.
            base = declaration.load(declaration.path(self.model))
            teacher = next(check for check in config["checks"] if check["id"] == "teacher")
            self.assertEqual(teacher["constants"] if "constants" in teacher else None,
                             next(check for check in base["checks"] if check["id"] == "teacher").get("constants"))
            source = refs.variant_source(config)
        digest = lambda name: hashlib.sha256((self.root / self.model / name).read_bytes()).hexdigest()  # noqa: E731
        self.assertEqual(source, {
            "root": str(self.root),
            "overlay": {"path": f"{self.model}/{VARIANT}.json", "sha256": digest(f"{VARIANT}.json")},
            "lock": {"path": f"{self.model}/{VARIANT}.lock.json", "sha256": digest(f"{VARIANT}.lock.json")},
            "checkpoint_lock": {"path": f"{self.model}/{VARIANT}.checkpoint-files.json",
                                "sha256": digest(f"{VARIANT}.checkpoint-files.json")}})

    def test_without_its_root_it_is_not_read(self):
        with patch.dict(os.environ):
            os.environ.pop(declaration.VARIANTS_ENV, None)
            self.assertEqual(declaration.named(self.name), declaration.path(self.model, variant=VARIANT))
            self.assertFalse(declaration.named(self.name).exists())
        with selected(self.root):
            config = self.load()
        with patch.dict(os.environ):
            os.environ.pop(declaration.VARIANTS_ENV, None)
            for read in (common.checkpoint_pin, refs.load_lock, refs.variant_source):
                with self.subTest(read.__name__), self.assertRaisesRegex(ValueError, "is an external variant"):
                    read(config)

    def test_its_lock_is_required(self):
        (self.root / self.model / f"{VARIANT}.lock.json").unlink()
        with selected(self.root):
            config = self.load()
            with self.assertRaisesRegex(ValueError, "reference lock is missing"):
                refs.load_lock(config)
            with self.assertRaisesRegex(ValueError, "reference lock is missing"):
                common.input_bytes("frontend", config=config)
        done = subprocess.run([sys.executable, "-B", str(ROOT / "tools/qual/release_gate.py"), "--model", self.name,
                               "--variant-dir", str(self.root), "--url", "http://127.0.0.1:1"],
                              capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertIn("reference lock is missing", done.stderr)

    def test_a_kept_base_reference_must_name_the_variant_s_checkpoint(self):
        """Keeping the base's teacher reference, bound to the model's own
        checkpoint lock, is refused when the variant's lock is adopted."""
        root = self.root.parent / "kept"
        with self.assertRaisesRegex(ValueError, "stale_reference: lock: the teacher reference is bound to another "
                                                "checkpoint lock than the pin"):
            synthetic_variant(root, self.model, overlay={"references": {}, "keep_references": ["teacher"],
                                                         "inputs": None})

    def test_refused_overlay_fields_and_names_surface_as_declaration_errors(self):
        root = self.root.parent / "refused"
        cases = [({"rules": {"teacher": []}}, "rules: an external variant cannot state it"),
                 ({"waivers": {}}, "waivers: an external variant cannot state it"),
                 ({"serve": ["--x"]}, "serve: unknown field")]
        for overlay, message in cases:
            with self.subTest(message):
                synthetic_variant(root, self.model, overlay=overlay, adopt=False)
                with selected(root), self.assertRaisesRegex(declaration.DeclarationError, message):
                    self.load()
        configured = sorted(declaration.configurations(self.model))
        if configured:
            synthetic_variant(root, self.model, variant=configured[0], adopt=False)
            with selected(root), self.assertRaisesRegex(declaration.DeclarationError,
                                                        "is a variant the tree declares"):
                declaration.load(declaration.named(f"{self.model}.{configured[0]}"))

    def test_a_gate_run_records_its_source_and_runs_eval_under_its_root(self):
        with selected(self.root):
            config = self.load()
            check = next(check for check in config["checks"] if check["id"] == "teacher")
            source = gate.eval_source(config, check)
            self.assertEqual(source, (ROOT / "models" / self.model, VARIANT, self.root))
            command = gate.eval_command(check, "quick", {"model": "served", "url": "http://127.0.0.1:1"}, source,
                                        Path("report"))
            self.assertEqual(command[command.index("--variant"):command.index("--variant") + 4],
                             ["--variant", VARIANT, "--variant-dir", str(self.root)])
            expected = refs.variant_source(config)
            with tempfile.TemporaryDirectory() as receipts, \
                    patch.object(gate.Client, "identity", return_value={"model": "served", "health": None}), \
                    patch.object(gate, "run_check", side_effect=lambda check, rows, *rest: rows.append(
                        {"id": check["id"], "passed": False, "error": "not run here"}) or rows[-1]), \
                    contextlib.redirect_stdout(io.StringIO()):
                code = gate.run(config, declaration.named(self.name), "quick", "http://127.0.0.1:1", None, {},
                                Path(receipts))
                summary = json.loads(next(Path(receipts).glob("*/summary.json")).read_text())
        self.assertEqual(code, 1)
        self.assertEqual(summary["variant_source"], expected)
        self.assertEqual(summary["config"]["variant_source"], config["variant_source"])

    def test_tools_take_the_variant_root_as_an_option(self):
        env = {key: value for key, value in os.environ.items() if key != declaration.VARIANTS_ENV}
        for arguments in (["list", "--model", self.name, "--json"], ["gate", "--model", self.name, "--plan"]):
            with self.subTest(arguments[0]):
                done = subprocess.run([sys.executable, "-B", "-m", "tools.qual", *arguments, "--variant-dir",
                                       str(self.root)], cwd=ROOT, env=env, capture_output=True, text=True,
                                      timeout=300)
                self.assertEqual(done.returncode, 0, done.stderr)
        listed = subprocess.run([sys.executable, "-B", "-m", "tools.qual", "list", "--model", self.name, "--json",
                                 "--variant-dir", str(self.root)], cwd=ROOT, env=env, capture_output=True,
                                text=True, timeout=300)
        self.assertEqual([row["id"] for row in json.loads(listed.stdout)["checks"]], ["teacher"])

    def test_refs_adopt_writes_the_variant_lock_beside_its_overlay(self):
        lock = self.root / self.model / f"{VARIANT}.lock.json"
        before = lock.read_bytes()
        lock.unlink()
        done = subprocess.run([sys.executable, "-B", "-m", "tools.qual", "refs", "adopt", "--model", self.name,
                               "--variant-dir", str(self.root)], cwd=ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(json.loads(lock.read_bytes()), json.loads(before))
        # Every input the variant shares with the model has the model lock's record.
        model_lock = json.loads(refs.lock_path(self.model).read_text())
        value = json.loads(lock.read_text())
        shared = [key for key, entry in value["entries"].items()
                  if model_lock["entries"].get(key, {}).get("path") == entry["path"]]
        self.assertTrue(shared)
        for key in shared:
            self.assertEqual(value["entries"][key], model_lock["entries"][key], key)
        owned = value["entries"]["reference:teacher"]["members"]["."]
        self.assertEqual(value["objects"][owned]["production"]["producer"], "runner-binding")

    def test_a_changed_owned_input_is_refused_when_read(self):
        path = self.root / self.model / f"{VARIANT}.teacher-reference.json"
        path.write_bytes(path.read_bytes() + b" ")
        with selected(self.root):
            config = self.load()
            with self.assertRaisesRegex(ValueError, "lock entry reference:teacher: sha256 differs"):
                common.pinned_reference("teacher", path, config)


def safetensors(path, name):
    """A one-tensor safetensors file."""
    import struct
    header = json.dumps({name: {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(8))


class VariantToolTests(unittest.TestCase):
    """`python3 -m tools.qual variant`: the checkpoint lock, the pin, the
    reference skeleton and adoption, all outside the tree."""

    def setUp(self):
        import variant
        self.tool = variant
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.temporary = Path(temporary.name).resolve()
        self.root = self.temporary / "variants"
        self.model = rehomed_model()

    def arguments(self, command, *extra):
        return [command, "--model", self.model, "--variant", VARIANT, "--dir", str(self.root),
                "--checkpoint", str(self.temporary / "checkpoint"), "--repository", "someone/requant",
                "--revision", "a" * 40, *extra]

    def checkpoint(self):
        checkpoint = self.temporary / "checkpoint"
        for name in ("indexed.safetensors", "superseded.safetensors", "audio/model.safetensors",
                     ".hidden/x.safetensors"):
            safetensors(checkpoint / name, name)
        for name in ("config.json", "tokenizer.json", "README.md"):
            (checkpoint / name).write_text(json.dumps({"name": name}))
        (checkpoint / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"w": "indexed.safetensors"}}))
        return checkpoint

    def test_the_lock_covers_what_a_direct_load_opens_and_is_never_replaced(self):
        """Shards the index no longer names and subdirectory files are
        locked, as `chopinfer verify` and the checkpoint identity count
        them; hidden directories and files no loader reads are not."""
        from tools.refrun import stamps
        checkpoint = self.checkpoint()
        with patch.object(stamps, "RACY_NS", 1):
            path = self.tool.write_lock(self.root / self.model / self.tool.lock_name(VARIANT), checkpoint,
                                        self.temporary / "stamps")
            self.assertEqual(path, self.root / self.model / f"{VARIANT}.checkpoint-files.json")
            lock = json.loads(path.read_text())
            self.assertEqual(sorted(lock), ["audio/model.safetensors", "config.json", "indexed.safetensors",
                                            "model.safetensors.index.json", "superseded.safetensors",
                                            "tokenizer.json"])
            for entry in lock.values():
                self.assertEqual(set(entry), {"byte_length", "sha256"})
            with self.assertRaises(FileExistsError):
                self.tool.write_lock(path, checkpoint, self.temporary / "stamps")

    def test_the_identity_is_the_engines(self):
        """Every .safetensors file of the root and its non-hidden immediate
        subdirectories, by name, length and header."""
        root = self.temporary / "identity"
        safetensors(root / "model.safetensors", "a")
        before = self.tool.identity(root)
        safetensors(root / ".hidden/x.safetensors", "b")
        (root / "config.json").write_text("{}")
        self.assertEqual(self.tool.identity(root), before)
        safetensors(root / "audio/model.safetensors", "c")
        self.assertNotEqual(self.tool.identity(root), before)

    def test_the_pin_names_the_checkpoint_and_its_lock_beside_the_overlay(self):
        from tools.refrun import keys, stamps
        checkpoint = self.checkpoint()
        with patch.object(stamps, "RACY_NS", 1), patch.object(stamps, "default_cache", return_value=None), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            self.assertEqual(self.tool.main(self.arguments("lock")), 0)
            self.assertEqual(self.tool.main(self.arguments("pin", "--served-model-name", "Requant")), 0)
        value = json.loads(printed.getvalue().split("\n", 1)[1])
        lock = keys.read_lock(self.root / self.model / f"{VARIANT}.checkpoint-files.json")
        self.assertEqual(value, {"lock": f"{VARIANT}.checkpoint-files.json", "repository": "someone/requant",
                                 "revision": "a" * 40, "served_model_name": "Requant",
                                 "source_lock_sha256": keys.checkpoint(lock, [])["source_lock_sha256"],
                                 "checkpoint_identity": self.tool.identity(checkpoint)})
        (checkpoint / "config.json").write_text("{}")
        with patch.object(stamps, "RACY_NS", 1), self.assertRaisesRegex(ValueError, "checkpoint lock"):
            self.tool.pin(SimpleNamespace(model=self.model, variant=VARIANT, dir=self.root, checkpoint=checkpoint,
                                          repository="someone/requant", revision="a" * 40,
                                          served_model_name=None), cache=None)

    def test_nothing_is_written_into_the_tree(self):
        with contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            self.tool.main([*self.arguments("lock")[:6], str(ROOT / "target/variants"),
                            *self.arguments("lock")[7:]])
        self.assertIn("is in the tree", error.getvalue())
        self.assertFalse((ROOT / "target/variants").exists())

    def test_the_skeleton_keeps_the_base_layout_without_its_scores(self):
        value = {"repository": "someone/requant", "revision": "a" * 40, "served_model_name": "Requant",
                 "source_lock_sha256": "1" * 64, "checkpoint_identity": "2" * 64, "lock": "x"}
        document, workload = self.tool.skeleton(self.model, "teacher", value)
        spec, path = self.tool.base_reference(self.model, "teacher")
        base = json.loads(path.read_text())
        self.assertEqual(document["schema"], base["schema"])
        self.assertEqual(document["provenance"]["checkpoint"],
                         {key: value[key] for key in declaration.CHECKPOINT_KEYS})
        self.assertEqual(workload, common.reference_workload(path))
        self.assertNotIn("binding", document)
        self.assertEqual(common.workload_sha256(document["texts"]), spec["workload_sha256"])
        for text in document["texts"]:
            self.assertFalse(set(text) & set(spec["variants"]))
        kept = spec["variants"][:1]
        document, _ = self.tool.skeleton(self.model, "teacher", value, kept)
        self.assertEqual(list(document["provenance"]["variants"]), kept)

    def emitted(self):
        """The model's checkpoint lock as the variant's, and its teacher
        reference re-homed onto it as emit would have written it, with the
        pin `variant pin` would print for it."""
        base = json.loads(declaration.path(self.model).read_text())
        pin = dict(base["qualification"]["checkpoint"], lock=self.tool.lock_name(VARIANT))
        (self.root / self.model).mkdir(parents=True)
        shutil.copyfile(ROOT / base["qualification"]["checkpoint"]["lock"],
                        self.root / self.model / self.tool.lock_name(VARIANT))
        spec = base["references"]["teacher"]
        fixture = json.loads((ROOT / spec["path"]).read_text())
        fixture["binding"]["lock"]["path"] = f"{self.model}/{self.tool.lock_name(VARIANT)}"
        emitted = self.temporary / "emitted"
        emitted.mkdir()
        (emitted / "teacher-reference.json").write_text(json.dumps(fixture, indent=1) + "\n")
        shutil.copyfile(common.reference_workload(ROOT / spec["path"]), emitted / "teacher-reference.workload.json")
        return pin, emitted

    def adopt(self, emitted, *extra):
        return self.tool.main(self.arguments("adopt", "--reference", "teacher", "--emitted", str(emitted),
                                             "--check", "teacher", *extra))

    def test_adopt_writes_the_overlay_its_references_and_lock_and_verifies_them(self):
        pin, emitted = self.emitted()
        with patch.object(self.tool, "pin", return_value=pin), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.adopt(emitted, "--profile-name", "requant-profile"), 0)
        directory = self.root / self.model
        self.assertEqual(sorted(path.name for path in directory.iterdir()),
                         sorted([f"{VARIANT}.json", f"{VARIANT}.lock.json", f"{VARIANT}.checkpoint-files.json",
                                 f"{VARIANT}.teacher-reference.json", f"{VARIANT}.teacher-reference.workload.json"]))
        overlay = json.loads((directory / f"{VARIANT}.json").read_text())
        base = json.loads(declaration.path(self.model).read_text())
        self.assertEqual((overlay["checks"], overlay["serving"], overlay["qualification"]["checkpoint"]),
                         (["teacher"], {"profile_name": "requant-profile"}, pin))
        self.assertEqual(overlay["references"]["teacher"]["path"], f"{VARIANT}.teacher-reference.json")
        redirected = {key for key, location in base["inputs"].items()
                      if location == base["references"]["teacher"]["path"]}
        self.assertEqual(set(overlay.get("inputs", {})), redirected)
        if "capabilities" in base:
            self.assertEqual(overlay["capabilities"]["text"], base["capabilities"]["text"])
            self.assertEqual((overlay["capabilities"]["media"], overlay["media"]), ([], False))
        with selected(self.root):
            config = declaration.load(declaration.named(f"{self.model}.{VARIANT}"))
            self.tool.verify(config)
            self.assertEqual(refs.load_lock(config)["model"], self.model)
        # Adopting again replaces nothing.
        with patch.object(self.tool, "pin", return_value=pin), \
                contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            self.adopt(emitted)
        self.assertIn("refusing to replace", error.getvalue())

    def test_a_failed_adoption_leaves_nothing_behind(self):
        pin, emitted = self.emitted()
        workload = emitted / "teacher-reference.workload.json"
        value = json.loads(workload.read_text())
        value["sequences"] = value["sequences"][:1]
        workload.write_text(json.dumps(value))
        with patch.object(self.tool, "pin", return_value=pin), \
                contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            self.adopt(emitted)
        self.assertIn("stale_reference: workload", error.getvalue())
        self.assertEqual([path.name for path in (self.root / self.model).iterdir()],
                         [self.tool.lock_name(VARIANT)])
        # A pin whose lock digest is not its lock's is refused too.
        pin, emitted = dict(pin, source_lock_sha256="0" * 64), self.temporary / "emitted"
        shutil.copyfile(common.reference_workload(self.tool.base_reference(self.model, "teacher")[1]), workload)
        with patch.object(self.tool, "pin", return_value=pin), \
                contextlib.redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
            self.adopt(emitted)
        self.assertIn("checkpoint pin differs", error.getvalue())
        self.assertEqual([path.name for path in (self.root / self.model).iterdir()],
                         [self.tool.lock_name(VARIANT)])


class PreviewTests(unittest.TestCase):
    """The correctness runner serves an external variant it is asked for and
    never stores its run as a model's last good one."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "variants"
        import release_preview
        self.runner = release_preview
        model = next((name for name in sorted(declaration.previewed(ROOT)) if name == rehomed_model()), None)
        if model is None:
            self.skipTest("the re-homed model declares no preview")
        self.model, self.overlay = synthetic_variant(self.root, model)

    def test_an_external_variant_is_served_from_its_checkpoint_and_lock(self):
        with selected(self.root):
            declared = self.runner.configurations(self.model)
            self.assertNotIn(VARIANT, declared)
            self.assertNotIn(VARIANT, self.runner.selected_configurations(self.model, [], "full"))
            config = self.runner.selected_configurations(self.model, [VARIANT], "full")[VARIANT]
            facts = self.runner.configuration_facts(self.runner.model_facts(self.model), VARIANT)
        self.assertEqual((config["checkpoint"], config["screens"], config["external"]),
                         (f"CHOPINFER_{self.overlay['checkpoint']}_CHECKPOINT", False, True))
        self.assertEqual(config["manifest"], str(self.root / self.model / f"{VARIANT}.checkpoint-files.json"))
        self.assertEqual(facts["full"], ["teacher"])
        with patch.dict(os.environ):
            os.environ.pop(declaration.VARIANTS_ENV, None)
            with self.assertRaisesRegex(ValueError, f"declares no configuration {VARIANT}"):
                self.runner.selected_configurations(self.model, [VARIANT], "full")

    def test_an_external_variant_never_moves_last_good(self):
        declared = sorted(self.runner.configurations(self.model))
        result = {"model": self.model, "suite": "full", "passed": True, "selected_checks": [],
                  "configurations": declared, "rows": [{"id": f"{name}/x", "status": "pass"} for name in declared]}
        self.assertTrue(self.runner.whole_model_full_pass(result, self.model))
        external = dict(result, external_variants=[VARIANT])
        self.assertFalse(self.runner.whole_model_full_pass(external, self.model))
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(self.runner, "VALIDATION", Path(temporary)):
            message = self.runner.Store("last-good", self.model).update(Path(temporary), external)
        self.assertIn("not updated (an external variant ran)", message)


class ValidationTests(unittest.TestCase):
    def test_ci_validation_never_selects_an_external_variant(self):
        sys.path.insert(0, str(ROOT / "ci"))
        self.addCleanup(sys.path.remove, str(ROOT / "ci"))
        import validation
        with tempfile.TemporaryDirectory() as temporary:
            model, _ = synthetic_variant(Path(temporary), adopt=False)
            with selected(temporary):
                variants = validation.declared_variants(ROOT, model)
        self.assertNotIn(VARIANT, variants)


if __name__ == "__main__":
    unittest.main()

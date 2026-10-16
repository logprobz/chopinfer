"""Reference cache bindings at gate time. A reference without a binding keeps
its checks and the gates report it unbound, but adoption and gate admission
refuse it as a numerical reference. A bound reference's bytes are
pinned, so its gate reads the binding it records without verifying it
against the reference cache again; it must name the gate's checkpoint lock.
Adoption (`common.verified_binding`) verifies a binding and refuses a stale
one naming the differing field."""

from qual_test_config import needs_qwen_probe_workload, configure, fixture_lock

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import common
import declaration
import multimodal
import refs
from test_checkpoint_binding import (CONFIG, ENTRY_POINTS, PIN, TEACHER, ReplayServer, live, run_entry, run_main)

refcache = common.refcache()
from tools.refrun import keys, modes, workload  # noqa: E402

URL = "http://127.0.0.1:8101"
QWEN_CONFIG = declaration.load(declaration.path("qwen38"))
QWEN_PIN = QWEN_CONFIG["qualification"]["checkpoint"]
IMAGES = common.ROOT / "fixtures/qwen38/image-probe-references.json"


def setUpModule():
    configure()


# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "reference cache bindings with copied gates in temporary roots"


def write_workload(path, model, sequences):
    """A runner workload of forced (name, token IDs, score_from) sequences."""
    path.write_text(json.dumps({"schema": workload.SCHEMA, "model": model, "sequences": [
        {"name": name, "category": "probe", "token_ids": ids, "score_from": start} for name, ids, start in sequences]}))
    return path


def synthetic_run(model, lock, sequences, scores):
    """Synthetic f32 runner entries of (name, token IDs, score_from) under
    `lock`, each scored by `scores(name)`, keyed with the model's reference adapter."""
    sections = {"checkpoint": keys.checkpoint(lock, ["config.json"]),
                "reference_code": keys.reference_code({f"{model}/modeling.py": "f" * 64}, transformers="5.16.1",
                                                  transformers_sha256="9" * 64),
                "adapter": keys.adapter(keys.repo_files([f"models/{model}/reference.py", *keys.NUMERICS])),
                "environment": {"torch": "2.14.0+cu130", "cuda_runtime": "13.0", "cublas": "13.1.1.3",
                                "device": "sm_120", "env_freeze_sha256": "e" * 64},
                "outputs": keys.outputs(20)}
    policy = modes.policy("f32")
    entries = []
    for name, ids, start in sequences:
        sequence = {"token_ids_sha256": keys.ids_sha256(ids), "score_from": start,
                    "answer": {"policy": "forced", "ids_sha256": keys.ids_sha256([]), "max_new_tokens": None,
                               "stop_ids": None}, "media": []}
        document = keys.document(model=model, sequence=sequence, mode=keys.mode(policy), **sections)
        entries.append({"name": name, "key_sha256": keys.key_sha256(document), "key": document,
                        "scores": scores(name)})
    return {"policy": policy, "entries": entries}


def teacher_run(value, lock_variant, drift=0.0):
    """Synthetic f32 runner entries holding the reference's exact scores."""
    lock = keys.read_lock(common.ROOT / PIN["lock"])[lock_variant]
    texts = {text["name"]: text for text in value["texts"]}

    def scores(name):
        copied = {field: list(texts[name]["exact"][field]) for field in ("logprob", "top1", "margin")}
        copied["logprob"][0] += drift
        return copied
    return synthetic_run("mimo26", lock, teacher_sequences(value), scores)


def reference_bindings(models=None):
    """Each previewed model's pinned references by declared path, with the
    refset each is bound to (None when unbound), read through the gate's pin:
    a binding must name the gate's checkpoint lock and declared workload and
    verify."""
    bindings = {}
    for model in models or declaration.previewed():
        config = declaration.load(declaration.path(model))
        bindings[model] = {}
        for kind, spec in sorted(config.get("qualification", {}).get("references", {}).items()):
            path = spec["path"]
            try:
                value = common.pinned_reference(kind, common.reference_path(kind, config), config)
                status = common.pinned_binding(kind, value, config)
            except ValueError as error:
                raise ValueError(f"{model} {kind} reference {path}: {error}") from error
            key = path if isinstance(path, str) else json.dumps(path, sort_keys=True, separators=(",", ":"))
            bindings[model][key] = status["refset_sha256"]
    return bindings


def gate_read(kind, path, config):
    """The pinned reference `kind` at `path` read as the gate reads it, as
    (0, {"reference_binding": its binding}), or (the refusal, None)."""
    try:
        value = common.pinned_reference(kind, path, config)
        return 0, {"reference_binding": common.pinned_binding(kind, value, config)}
    except ValueError as error:
        return f"refused: {error}", None


def teacher_sequences(value):
    return [(text["name"], text["prompt_token_ids"], text["score_from"]) for text in value["texts"]]


class BoundTeacher:
    """The pinned MiMo teacher reference bound to synthetic entries in a
    cache under $CHOPINFER_DATA, its workload, and a gate config pinning its
    bytes and declaring that workload."""

    def __init__(self, directory, lock_variant="mopd", cached_drift=0.0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        store = refcache.Store(self.directory / "refcache/v1")
        value = json.loads(TEACHER.read_text())
        for entry in teacher_run(value, lock_variant, cached_drift)["entries"]:
            store.put(entry["key"], entry["scores"])
        self.workload = write_workload(self.directory / "workload.json", "mimo26", teacher_sequences(value))
        self.value = refcache.bind(value, {"f32": teacher_run(value, lock_variant)}, str(self.workload), PIN["lock"],
                                   lock_variant, {"f32": "exact"}, store)
        self.path = self.directory / "teacher-reference.json"
        self.path.write_text(json.dumps(self.value))
        config = copy.deepcopy(CONFIG)
        spec = config["qualification"]["references"]["teacher"]
        spec.update(path=str(self.path), sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                    workload=str(self.workload))
        self.config_value = config
        self.config = self.directory / "mimo26.json"
        self.config.write_text(json.dumps(config))

    def teacher(self):
        """The gate-time read of this reference, as the teacher check reads
        it: (0, the binding it reports) or (the refusal, None)."""
        with mock.patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(self.config),
                                          "CHOPINFER_DATA": str(self.directory)}), fixture_lock(self.config_value, self.directory):
            return gate_read("teacher", self.path, self.config_value)

    def gate_bindings(self):
        """MiMo's reference bindings with this config as its gate."""
        real = declaration.load

        def load(path):
            return self.config_value if Path(path) == declaration.path("mimo26") else real(path)
        # The cache is this one; $CHOPINFER_DATA keeps the other references' workloads.
        with mock.patch.object(declaration, "load", side_effect=load), \
                mock.patch.object(refcache, "default_root", return_value=self.directory / "refcache/v1"), \
                fixture_lock(self.config_value, self.directory):
            return reference_bindings(("mimo26",))


class BoundImages:
    """Qwen3.8's image probe references bound under `lock_path`/`lock_variant`,
    and a Qwen gate config pinning them."""

    def __init__(self, directory, lock_path, lock_variant=None, data=None):
        """`data` keeps the workload under that $CHOPINFER_DATA, declared by
        an environment-relative path."""
        self.directory = Path(directory)
        self.directory.mkdir(parents=True)
        # The tracked probes, without the sequence fields the synthetic entries do not score.
        value = json.loads(IMAGES.read_text())
        value.pop("binding", None)
        for item in value["probes"].values():
            for field in ("prompt_token_ids", "score_from", "answer_token_ids"):
                item.pop(field, None)
        lock = keys.read_lock(common.ROOT / lock_path)
        lock = lock[lock_variant] if lock_variant else lock
        sequences = [(name, [1, 2, 3], 1) for name in value["probes"]]
        run = synthetic_run("qwen38", lock, sequences, lambda name: {"logprob": [-0.5, -0.25]})
        # The canonical workload beside the reference unless the gate declares one under `data`.
        self.path = self.directory / "image-probe-references.json"
        self.workload = (Path(data) / "probes/workload.json" if data
                         else self.directory / "image-probe-references.workload.json")
        self.workload.parent.mkdir(parents=True, exist_ok=True)
        write_workload(self.workload, "qwen38", sequences)
        self.value = refcache.bind(value, {"f32": run}, str(self.workload), lock_path, lock_variant)
        self.path.write_text(json.dumps(self.value))
        config = copy.deepcopy(QWEN_CONFIG)
        config["qualification"]["references"]["image_probes"] = {
            "path": str(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "schema": value["schema"], "checkpoint_keys": ["revision"]}
        if data:
            config["qualification"]["references"]["image_probes"]["workload"] = {
                "env": "CHOPINFER_DATA", "path": "probes/workload.json"}
        self.config_value = config
        self.config = self.directory / "qwen38.json"
        self.config.write_text(json.dumps(config))

    def multimodal(self):
        """multimodal.py with the declared image-probes format against a live pinned Qwen server."""
        _, _, patches = ENTRY_POINTS["multimodal"]
        client = (lambda url, timeout=None: ReplayServer(url, model=QWEN_PIN["served_model_name"],
                                                         identity=QWEN_PIN["checkpoint_identity"]))
        with mock.patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(self.config)}), fixture_lock(self.config_value, self.directory):
            return run_main(multimodal, ["multimodal.py", "--url", URL], client, patches)


class GateBindingTests(unittest.TestCase):
    @needs_qwen_probe_workload
    def test_todays_references_report_their_binding(self):
        """Each tracked reference is reported with the refset its binding
        block names, or unbound without one."""
        def refset(path):
            return (json.loads(common.declared_path(path).read_text()).get("binding") or {}).get("refset_sha256")
        references = CONFIG["qualification"]["references"]
        code, value = gate_read("teacher", common.reference_path("teacher", CONFIG), CONFIG)
        self.assertEqual(code, 0)
        receipts = {"teacher": value}
        code, value = run_entry("multimodal", [], live())
        self.assertEqual(code, 0)
        receipts["media"] = value
        for kind, value in receipts.items():
            stated = refset(references[kind]["path"])
            self.assertEqual((value["reference_binding"]["status"], value["reference_binding"]["refset_sha256"]),
                             ("bound" if stated else "unbound", stated), kind)
        bindings = reference_bindings()
        for model in bindings:
            references = declaration.load(declaration.path(model))["qualification"]["references"]
            expected = {(spec["path"] if isinstance(spec["path"], str) else
                         json.dumps(spec["path"], sort_keys=True, separators=(",", ":"))): refset(spec["path"])
                        for spec in references.values()}
            self.assertEqual(bindings[model], expected)

    def test_a_bound_reference_is_read_and_the_receipt_names_its_refset(self):
        with tempfile.TemporaryDirectory() as directory:
            bound = BoundTeacher(directory)
            code, value = bound.teacher()
        self.assertEqual(code, 0)
        self.assertEqual(value["reference_binding"], {
            "status": "bound", "refset_sha256": bound.value["binding"]["refset_sha256"], "cache": None})

    STALE = {
        "mode_policy": ({}, lambda: mock.patch.dict(modes.FLAGS, {"cudnn_benchmark": True})),
        "adapter": ({}, lambda: mock.patch.object(keys, "NUMERICS", (*keys.NUMERICS, "tools/refrun/taps.py"))),
        "scores": ({"cached_drift": 2.0 ** -30}, lambda: mock.patch.dict(os.environ, {})),
    }

    def test_a_pinned_reference_is_not_verified_against_the_cache_on_each_read(self):
        # Reference code, mode policy and cached scores that changed after the
        # binding do not refuse the pinned bytes at gate time (flag F5).
        for field, (options, change) in self.STALE.items():
            with self.subTest(field), tempfile.TemporaryDirectory() as directory:
                bound = BoundTeacher(directory, **options)
                with change():
                    code, value = bound.teacher()
                self.assertEqual(code, 0, code)
                self.assertEqual(value["reference_binding"]["refset_sha256"], bound.value["binding"]["refset_sha256"])

    def test_adoption_refuses_a_stale_binding_naming_the_field(self):
        for field, (options, change) in self.STALE.items():
            with self.subTest(field), tempfile.TemporaryDirectory() as directory:
                bound = BoundTeacher(directory, **options)
                with change(), mock.patch.object(refcache, "default_root", return_value=bound.directory / "refcache/v1"), \
                        fixture_lock(bound.config_value, bound.directory), \
                        self.assertRaisesRegex(ValueError, f"^stale_reference: {field}:"):
                    common.verified_binding("teacher", bound.value, bound.config_value)

    def test_a_binding_to_another_checkpoint_lock_is_refused_at_gate_time(self):
        with tempfile.TemporaryDirectory() as directory:
            # Verifies against its own lock, which is not the gate's pin.
            bound = BoundTeacher(directory, lock_variant="rl")
            code, value = bound.teacher()
        self.assertIn("refused: stale_reference: lock:", str(code))
        self.assertIsNone(value)

    @needs_qwen_probe_workload
    def test_a_binding_to_another_checkpoint_lock_is_refused(self):
        # The review's case: self-consistent, but bound to the RL lock, not the gate's pin.
        with tempfile.TemporaryDirectory() as directory:
            bound = BoundTeacher(directory)
            self.assertEqual(bound.gate_bindings()["mimo26"][str(bound.path)],
                             bound.value["binding"]["refset_sha256"])
            bound = BoundTeacher(Path(directory) / "rl", lock_variant="rl")
            with self.assertRaisesRegex(ValueError, f"mimo26 teacher reference {bound.path}: stale_reference: lock"):
                bound.gate_bindings()

    def test_adoption_and_gate_admission_require_a_runner_binding(self):
        # Gate reads do not verify a pinned binding again, so a numerical
        # reference without one is neither adopted nor admitted.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound = BoundTeacher(root / "teacher")
            spec = copy.deepcopy(bound.config_value["qualification"]["references"]["teacher"])
            config = {"model": "synthetic", "inputs": {}, "qualification": {
                "checkpoint": bound.config_value["qualification"]["checkpoint"], "references": {"teacher": spec}}}
            path = refs.lock_path("synthetic", root)
            path.parent.mkdir(parents=True)
            with mock.patch.object(refcache, "default_root", return_value=bound.directory / "refcache/v1"):
                lock = refs.snapshot(config, root)
            refs.gate_reference_producers(config, lock)
            path.write_text(json.dumps(lock))
            refs.verify_config(config, root, optional=False)
            unbound = copy.deepcopy(bound.value)
            del unbound["binding"]
            bound.path.write_text(json.dumps(unbound))
            spec["sha256"] = hashlib.sha256(bound.path.read_bytes()).hexdigest()
            lock = refs.snapshot(config, root)
            refs.validate_lock(lock, config, root)
            refusal = "lock entry reference:teacher: . is not runner-bound; runner binding required"
            with self.assertRaisesRegex(ValueError, refusal):
                refs.gate_reference_producers(config, lock)
            path.write_text(json.dumps(lock))
            for options in ({"policy_only": True}, {"optional": False}):
                with self.subTest(**options), self.assertRaisesRegex(ValueError, refusal):
                    refs.verify_config(config, root, **options)

    def test_a_workload_changed_after_binding_is_refused_at_adoption(self):
        with tempfile.TemporaryDirectory() as directory:
            bound = BoundTeacher(directory)
            document = json.loads(bound.workload.read_text())
            document["sequences"][0]["token_ids"] = document["sequences"][0]["token_ids"][:-1]
            bound.workload.write_text(json.dumps(document))
            code, value = bound.teacher()
            self.assertEqual(code, 0, code)
            with self.assertRaisesRegex(ValueError, "^stale_reference: workload:"):
                common.verified_binding("teacher", bound.value, bound.config_value, authenticate_workload=False)

    @needs_qwen_probe_workload
    def test_qwens_image_probe_reference_is_pinned(self):
        self.assertIn("fixtures/qwen38/image-probe-references.json", reference_bindings()["qwen38"])

    def test_qwens_image_binding_must_name_the_pinned_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            # Self-consistent, but bound to MiMo's lock rather than Qwen's pin.
            foreign = BoundImages(Path(directory) / "foreign", PIN["lock"], "mopd")
            code, value = foreign.multimodal()
            self.assertIn("refused: stale_reference: lock:", str(code))
            self.assertIsNone(value)
            pinned = BoundImages(Path(directory) / "pinned", QWEN_PIN["lock"])
            code, value = pinned.multimodal()
            self.assertEqual(code, 0, code)
            self.assertEqual(value["reference_binding"]["refset_sha256"], pinned.value["binding"]["refset_sha256"])
            # The gate reads the pinned bytes without the workload; adoption needs it.
            pinned.workload.unlink()
            code, value = pinned.multimodal()
            self.assertEqual(code, 0, code)
            with self.assertRaisesRegex(ValueError, "^stale_reference: workload:"):
                common.verified_binding("image_probes", pinned.value, pinned.config_value,
                                        authenticate_workload=False)

    def test_a_workload_under_an_environment_variable_is_read_there(self):
        # Media too large for the repository keep the workload under $CHOPINFER_DATA.
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / "data"
            bound = BoundImages(Path(directory) / "fixture", QWEN_PIN["lock"], data=data)
            with mock.patch.dict(os.environ, {"CHOPINFER_DATA": str(data)}):
                code, value = bound.multimodal()
                self.assertEqual(code, 0, code)
                self.assertEqual(value["reference_binding"]["refset_sha256"], bound.value["binding"]["refset_sha256"])
                self.assertEqual(common.verified_binding("image_probes", bound.value, bound.config_value,
                                                         authenticate_workload=False)["refset_sha256"],
                                 bound.value["binding"]["refset_sha256"])
            environment = {key: value for key, value in os.environ.items() if key != "CHOPINFER_DATA"}
            with mock.patch.dict(os.environ, environment, clear=True), self.assertRaisesRegex(
                    ValueError, "^stale_reference: workload: the bound image_probes reference's workload needs "
                    "CHOPINFER_DATA"):
                common.verified_binding("image_probes", bound.value, bound.config_value, authenticate_workload=False)


if __name__ == "__main__":
    unittest.main()


class FloorOnlyBindingTests(unittest.TestCase):
    @staticmethod
    def bound_text_reference():
        for declaration_path in declaration.shipped():
            config = declaration.load(declaration_path)
            for kind, spec in config['qualification'].get('references', {}).items():
                original = common.declared_path(spec['path'])
                if original is None or not original.is_file():
                    continue
                value = json.loads(original.read_text())
                texts, binding = value.get('texts', []), value.get('binding', {})
                mode = binding.get('modes', {}).get('f32', {})
                workload_path = (common.declared_path(spec['workload']) if 'workload' in spec
                                 else common.reference_workload(original))
                if (texts and mode.get('variant') == 'exact' and mode.get('dequant') == 'exact'
                        and sum(len(text.get('exact', {}).get('logprob', [])) for text in texts) >= 1000
                        and workload_path is not None and workload_path.is_file()):
                    return copy.deepcopy(config), kind, value, workload_path
        raise AssertionError('no bound F32 text reference with at least 1000 positions')

    def test_floor_only_readoption_pins_a_refreshed_workload(self):
        # Adoption verifies a floor-only candidate against its own workload,
        # not the previous lock's pin of it; a gate read still checks that pin.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound = BoundTeacher(root / "floor")
            value = copy.deepcopy(bound.value)
            value.pop("binding")
            value["provenance"]["variants"] = {"exact": value["provenance"]["variants"]["exact"]}
            for text in value["texts"]:
                for variant in ("bf16", "f64"):
                    text.pop(variant, None)
            value = refcache.bind(value, {"f32": teacher_run(value, "mopd")}, str(bound.workload), PIN["lock"],
                                  "mopd", {"f32": "exact"})
            bound.path.write_text(json.dumps(value))
            spec = copy.deepcopy(bound.config_value["qualification"]["references"]["teacher"])
            spec.update(variants=["exact"], sha256=hashlib.sha256(bound.path.read_bytes()).hexdigest())
            config = {"model": "synthetic", "inputs": {}, "qualification": {
                "checkpoint": copy.deepcopy(PIN), "references": {"teacher": spec}}}
            with mock.patch.dict(os.environ, {"CHOPINFER_QUAL_MODEL_ROOT": str(root)}), \
                    mock.patch.object(refcache, "default_root", return_value=bound.directory / "refcache/v1"):
                initial = refs.snapshot(config, root)
                path = refs.lock_path("synthetic", root)
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(initial))
                self.assertEqual(common.pinned_binding("teacher", value, config)["status"], "bound")
                # A formatting-only refresh: the same logical workload in other bytes.
                old = bound.workload.read_bytes()
                bound.workload.write_text(json.dumps(json.loads(old), indent=4))
                self.assertNotEqual(refs.digest(bound.workload), hashlib.sha256(old).hexdigest())
                with self.assertRaisesRegex(ValueError, "lock entry workload:teacher: sha256 differs"):
                    common.pinned_binding("teacher", value, config)
                updated = refs.snapshot(config, root, previous=initial)
            self.assertEqual(updated["entries"]["workload:teacher"]["members"]["."], refs.digest(bound.workload))
            self.assertEqual(updated["entries"]["reference:teacher"], initial["entries"]["reference:teacher"])
            refs.validate_lock(updated, config, root)
            refs.gate_reference_producers(config, updated)

    def test_floor_only_keeps_pin_workload_and_producer_authentication(self):
        import teacher
        config, kind, value, source_workload = self.bound_text_reference()
        spec = config['qualification']['references'][kind]
        pin = config['qualification']['checkpoint']
        bound = value.pop('binding')
        variants = value['provenance']['variants']
        value['provenance']['variants'] = {'exact': variants['exact']}
        for text in value['texts']:
            for variant in variants:
                if variant != 'exact':
                    text.pop(variant)
        policy = modes.policy('f32')
        by_name = {text['name']: text for text in value['texts']}
        run = {'policy': policy, 'entries': []}
        for name, index, expected in bound['modes']['f32']['entries']:
            key = keys.document(model=bound['model'], sequence=bound['sequences'][index],
                                mode=keys.mode(policy), **bound['key'])
            self.assertEqual(keys.key_sha256(key), expected)
            run['entries'].append({'name': name, 'key': key, 'key_sha256': expected,
                                   'scores': by_name[name]['exact']})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'floor-reference.json'
            workload_path = Path(directory) / 'floor-reference.workload.json'
            workload_path.write_bytes(source_workload.read_bytes())
            spec.update(path=str(path), workload=str(workload_path), variants=['exact'])

            def write(runner):
                fixture = refcache.bind(value, {'f32': runner}, str(workload_path), pin['lock'],
                                        pin.get('pin'), {'f32': 'exact'})
                path.write_text(json.dumps(fixture))
                spec['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                return fixture

            write(run)
            with mock.patch.object(teacher, 'ALLOW_FLOOR_ONLY', True), fixture_lock(config, directory):
                loaded = common.pinned_reference(kind, path, config)
                self.assertTrue(teacher.floor_only_reference(loaded, ('exact',)))
                saved = path.read_bytes()
                path.write_bytes(saved + b'\n')
                with self.assertRaisesRegex(ValueError, 'sha256|pinned'):
                    common.pinned_reference(kind, path, config)
                path.write_bytes(saved)
                workload_path.write_text('{}')
                with self.assertRaises(ValueError):
                    common.pinned_reference(kind, path, config)
                workload_path.write_bytes(source_workload.read_bytes())
                path.unlink()
                with self.assertRaises((ValueError, OSError)):
                    common.pinned_reference(kind, path, config)
            # Rebinding and repinning a different producer must still fail.
            changed = copy.deepcopy(run)
            for entry in changed['entries']:
                files = entry['key']['adapter']['files']
                adapter = next(name for name in files if name not in keys.NUMERICS)
                files[adapter] = '0' * 64
                entry['key_sha256'] = keys.key_sha256(entry['key'])
            write(changed)
            with mock.patch.object(teacher, 'ALLOW_FLOOR_ONLY', True), fixture_lock(config, directory):
                with self.assertRaisesRegex(ValueError, 'adapter|producer'):
                    common.pinned_reference(kind, path, config)

"""Explicit gate selection for qualification CPU tests."""
from contextlib import contextmanager
import copy
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import common
import declaration

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.model_sdk import inventory  # noqa: E402


def released(path):
    """Whether the declaration at `path` (models/MODEL/...) is a release
    declaration: its model is released (tools/model_sdk/inventory.py), and
    it names no variant. A model that declares `[release] enabled = false`
    until it qualifies is exempt from the shipped-model facts. A variant
    (`qual.<variant>.json`, whose `variant` field declaration.load requires)
    qualifies another checkpoint or serving configuration as MODEL.VARIANT;
    it is not the model's release declaration, which keeps every release
    fact."""
    path = Path(path)
    if json.loads(path.read_text()).get("variant") is not None:
        return False
    return path.parent.name in inventory.released(path.parents[2])


def configure_receipts():
    receipts = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(receipts.cleanup)
    selected = patch.dict(os.environ, {"CHOPINFER_QUAL_RECEIPTS": receipts.name})
    selected.start()
    unittest.addModuleCleanup(selected.stop)


def configure(name="mimo26"):
    configure_receipts()
    selected = patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(declaration.named(name))})
    selected.start()
    unittest.addModuleCleanup(selected.stop)


def setUpModule():
    configure()


def qwen_probe_workload():
    """Whether the workload of Qwen3.8's bound image probe reference is
    present: its pixel rows are too large for the repository and stay under
    $CHOPINFER_DATA, so a test that verifies the real binding needs them."""
    spec = declaration.load(declaration.path("qwen38"))["qualification"]["references"]["image_probes"]
    path = common.declared_path(spec["workload"])
    return path is not None and path.is_file()


needs_qwen_probe_workload = unittest.skipUnless(qwen_probe_workload(), "no Qwen3.8 image probe workload "
                                                "under $CHOPINFER_DATA")


@contextmanager
def clean_checkout():
    """A copy of this checkout's tracked sources, with HEAD, and without
    local.env. The qualification tools read the checkout's local.env before
    CHOPINFER_LOCAL_ENV (common.py, release_preview.sh), so a test that sets
    the environment file, or simulates a host without one, runs the tools
    from this copy. tests/qual-clean.sh copies the same way."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1")
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "src"
        git = ["git", "-C", str(common.ROOT)]
        subprocess.run(["git", "clone", "--quiet", "--shared", "--no-checkout", str(common.ROOT), str(source)],
                       env=env, check=True)
        subprocess.run(["git", "-C", str(source), "read-tree", "HEAD"], env=env, check=True)
        files = subprocess.run([*git, "ls-files", "-z"], env=env, check=True, capture_output=True).stdout
        packed = subprocess.run(["tar", "--null", "-T", "-", "-c"], cwd=common.ROOT, input=files,
                                check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(source)], input=packed, check=True)
        assert not (source / "local.env").exists()
        yield source


@contextmanager
def fixture_lock(config, directory=None):
    """A scoped lock of synthetic bytes for tests of the separate pin/binding contracts."""
    import refs
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        lookup = refs.lock_path
        model = config["model"]
        lock = copy.deepcopy(json.loads(lookup(model).read_text()))
        declared = refs.sources(config)
        lock["entries"] = {key: entry for key, entry in lock["entries"].items()
                           if key in declared or key.startswith("operator:")}
        for key, location in declared.items():
            path = common.declared_path(location, common.reference_root())
            entry = lock["entries"].get(key, {"directory": False, "members": {}})
            entry["path"] = location
            if path is not None and path.exists():
                entry["directory"] = path.is_dir()
                members = {}
                selected = refs.scope(config, key) or entry.get("scope")
                for member in refs.files(path, selected):
                    if not member.is_file():
                        continue
                    sha = refs.digest(member)
                    relative = member.relative_to(path).as_posix() if path.is_dir() else "."
                    members[relative] = sha
                    original = next(iter(entry.get("members", {}).values()), None)
                    obj = lock["objects"].get(original)
                    if obj is None:
                        obj = {"schema": "synthetic"}
                    lock["objects"][sha] = copy.deepcopy(obj)
                entry["members"] = members
                if path.is_dir():
                    if selected is not None:
                        entry["scope"] = selected
                    else:
                        entry.pop("scope", None)
            lock["entries"][key] = entry
        target = Path(temporary) / "qual-references.lock.json"
        target.write_text(json.dumps(lock))
        with patch.object(refs, "lock_path", side_effect=lambda selected, root=None:
                          target if selected == model else lookup(selected, root)):
            yield


@contextmanager
def capture_fixture(directory):
    """A selected qualification config and byte lock for a synthetic frontend capture."""
    import refs
    import shutil
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fixtures = root / "fixtures"
        reference = fixtures / "frontend"
        reference.mkdir(parents=True)
        for path in Path(directory).glob("*.json"):
            shutil.copyfile(path, reference / path.name)
        samples = reference / "api-samples.json"
        if samples.exists():
            from frontend_goldens import shape
            (reference / "api-shapes.json").write_text(json.dumps(
                {name: shape(value) for name, value in json.loads(samples.read_text()).items()}))
        config = {"schema": "chopinfer-release-gates/v1", "model": "synthetic",
                  "inputs": {"fixture_root": str(fixtures), "frontend_capture": str(reference),
                             "tokenizer_goldens": str(reference / "tokenizer-golden.json")},
                  "checks": [{"id": "frontend", "module": "frontend_goldens", "args": [], "timeout_seconds": 60,
                              "stdout_json": True, "rules": [{"path": [name, "agrees"], "op": "eq", "value": True,
                               "source": "docs/check-kinds.md"} for name in ("chat-renders.json", "tokenizer-golden.json")]}],
                  "qualification": {"mechanics_only": True}}
        if not (reference / "tokenizer-golden.json").exists():
            config["inputs"].pop("tokenizer_goldens")
        lock = refs.lock_path("synthetic", root)
        lock.parent.mkdir(parents=True)
        lock.write_text(json.dumps(refs.snapshot(config, root)))
        policy = root / "policy.json"
        policy.write_text(json.dumps(config))
        with patch.dict(os.environ, {"CHOPINFER_QUAL_CONFIG": str(policy), "CHOPINFER_QUAL_MODEL_ROOT": str(root)}):
            yield reference


def fixture_evidence(config, key, path):
    """Authenticate a synthetic reference against a lock of its fixture bytes."""
    import refs
    value = copy.deepcopy(config)
    value["inputs"][key] = str(path)
    with fixture_lock(value):
        refs.verify_input(value, key, path)

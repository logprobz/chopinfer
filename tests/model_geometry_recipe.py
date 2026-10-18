"""Prove that a compiled geometry can be added with data alone, without a GPU.

Each model declares its legs in its model.toml `[geometry_recipe]` table
(tools/model_sdk/inventory.py): its stage, its fixtures under
fixtures/<model>/geometries/, the architecture alias of the added geometry and,
at the bundle stage, the contract crate's test suites."""
import copy
import hashlib
import json
import os
import signal
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile
import time

from data_only_recipe import (assert_tests_ran, changed_paths, detached_worktree, kept_lock_records, probe,
                              require_build_tools, require_clean, run, tokenizer_dir)
from model_inventory import load


ROOT = Path(__file__).resolve().parents[1]
# The checkout whose inventory declares the legs. The recipe branches from
# ROOT's HEAD, the same checkout unless a test points ROOT at a fixture.
INVENTORY = ROOT
FP8_SCALES = [[0.5, 0.25], [1.0, 0.5]]
FP8_LAYOUT = "mimo.ga-paged-fp8.v2." + hashlib.sha256(
    b"".join(struct.pack("<f", scale) for pair in FP8_SCALES for scale in pair)).hexdigest()


def lock_state(tree, model):
    """The bundle's kernel lock before a recipe build: the text of a
    monolithic `kernels.lock`, or `{file name: text}` of a lock directory's
    family files (`kernels.lock.d`, `*.lock` and `*.provenance.json`)."""
    bundle = Path(tree) / f"models/{model}/bundle"
    if (bundle / "kernels.lock").is_file():
        return (bundle / "kernels.lock").read_text()
    directory = bundle / "kernels.lock.d"
    assert directory.is_dir(), f"models/{model}/bundle has no kernels.lock or kernels.lock.d"
    return {path.name: path.read_text() for path in sorted(directory.iterdir()) if path.is_file()}


def lock_growth(tree, model, before, paths):
    """(lock paths, kept records, added records) of the lock rewrite: every
    family file keeps its existing records' bytes and order, and a lock
    directory gains no family file."""
    if isinstance(before, str):
        lock = f"models/{model}/bundle/kernels.lock"
        assert lock in paths, "new geometry did not add lock records"
        old, added = kept_lock_records(before, (Path(tree) / lock).read_text())
        return {lock}, old, added
    prefix = f"models/{model}/bundle/kernels.lock.d/"
    changed = {p for p in paths if p.startswith(prefix)}
    assert changed, "new geometry did not add lock records"
    assert {p[len(prefix):] for p in changed} <= set(before), "a new geometry added a lock family file"
    old, added = [], []
    for path in sorted(changed):
        kept, grown = kept_lock_records(before[path[len(prefix):]], (Path(tree) / path).read_text())
        old += kept
        added += grown
    return changed, old, added


def assert_data_diff(tree, model, before, checkpoint_data=None, *, supported_before=None, entry=None, alias=None):
    prefix = f"models/{model}/geometries/"
    paths = changed_paths(tree)
    locks, old, added = lock_growth(tree, model, before, paths)
    allowed = {*locks, *(f"models/{model}/{name}" for name in checkpoint_data or {})}
    if entry is not None:
        geometry_file = prefix + entry["file"]
        allowed.update((prefix + "supported.json", geometry_file))
        additions = {geometry_file}
        if model == "qwen38":
            config_file = prefix + Path(entry["file"]).stem + "-config.json"
            allowed.add(config_file)
            additions.add(config_file)
        existing = set(run("git", "ls-tree", "-r", "--name-only", "HEAD", "--", prefix, cwd=tree).splitlines())
        assert not (additions & existing), "new geometry files already exist"
        assert additions <= set(paths), "appended geometry files are missing"
    assert paths and set(paths) <= allowed, f"unexpected data-only paths: {paths}"
    for name, previous in (checkpoint_data or {}).items():
        actual = json.loads((tree / f"models/{model}/{name}").read_text())
        if name == "checkpoints.json":
            assert actual["default"] == previous["default"]
            actual, previous = actual["checkpoints"], previous["checkpoints"]
        assert all(actual.get(key) == value for key, value in previous.items()), name
        assert len(actual) > len(previous), "checkpoint data did not gain entries"
    assert all(Path(p).suffix == ".json" or p in locks for p in paths), paths
    if supported_before is not None:
        expected = copy.deepcopy(supported_before)
        expected["geometries"].append(entry)
        expected["aliases"]["architecture"].append(alias)
        actual = json.loads((tree / prefix / "supported.json").read_text())
        assert actual == expected, "supported.json changed beyond the appended geometry and alias"
    assert added, "lock did not gain records"
    if entry is not None:
        assert all(entry["kernel_tag"] in name for name in added), \
            "added lock record does not carry the new geometry's kernel tag"
    print("Asserted data-only diff:", flush=True)
    for path in paths:
        print(f"  {path}", flush=True)
    print(f"Lock: {len(old)} byte-identical existing records; {len(added)} added records", flush=True)


def assert_mimo_layouts(manifest, expected):
    for resources in [manifest, *manifest.get("topology_resources", [])]:
        actual = [format["layout"]["id"] for format in resources["kv_formats"]]
        assert actual == expected, resources["kv_formats"]
    print(f"Prepared KV layouts at every topology: {expected}", flush=True)


def qwen_refusals(helper, config, flash, scratch, env, geometry_id):
    unlisted_context = config["text_config"]["max_position_embeddings"] // 2
    flash_field = "head_dim" if config["text_config"]["head_dim"] != flash["text_config"]["head_dim"] else "hidden_size"
    cases = [
        ("text_config.max_position_embeddings", unlisted_context,
         f"unsupported geometry: flash: text_config.{flash_field} expected {flash['text_config'][flash_field]}, "
         f"received {config['text_config'][flash_field]}; {geometry_id}: "
         f"text_config.max_position_embeddings expected {config['text_config']['max_position_embeddings']}, "
         f"received {unlisted_context}; supported ids: flash, {geometry_id}"),
        ("text_config.layer_types", ["unknown"],
         'unsupported semantic text_config.layer_types: expected one of ["linear_attention","full_attention"], received ["unknown"]'),
        ("text_config.hidden_act", "relu",
         'unsupported semantic text_config.hidden_act: expected one of ["silu"], received "relu"'),
        ("text_config.rope_parameters.rope_type", "unknown",
         'unsupported semantic text_config.rope_parameters.rope_type: expected one of ["default"], received "unknown"'),
        ("quantization_config", {"quant_method": "unknown"},
         "unsupported semantic quantization_config: recipe class unknown is not supported"),
        ("text_config.indexer_n_heads", 12,
         "unsupported engine shape text_config.indexer_n_heads: sparse indexer requires heads in {4,8,16}, received 12"),
    ]
    refused = scratch / "refused"
    refused.mkdir()
    # No weights or tokenizer files: each refusal must happen before loading.
    for field, value, expected in cases:
        changed = copy.deepcopy(config)
        parent = changed
        parts = field.split(".")
        for part in parts[:-1]:
            parent = parent[part]
        parent[parts[-1]] = value
        (refused / "config.json").write_text(json.dumps(changed))
        for ranks in (2, 4):
            reply = probe(helper, refused, ranks, env)
            assert reply["result"] == "probe" and not reply["compatible"], reply
            reason = reply["reason"]
            assert reason == expected, reason
            print(f"TP{ranks} refusal: {reason}", flush=True)


def file_pins(root):
    pins = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        pins[path.relative_to(root).as_posix()] = {
            "byte_length": path.stat().st_size, "sha256": digest.hexdigest()}
    return pins


def mimo_registration(tree, checkpoint, config, supported, scratch, geometry_id):
    declarations_path = tree / "models/mimo26/checkpoints.json"
    locks_path = tree / "models/mimo26/checkpoint-files.json"
    declarations = json.loads(declarations_path.read_text())
    locks = json.loads(locks_path.read_text())
    previous = {declarations_path.name: copy.deepcopy(declarations),
                locks_path.name: copy.deepcopy(locks)}
    registered = "geometry-recipe"
    declarations["checkpoints"][registered] = {
        "repository": "fixture/geometry-recipe", "revision": "0" * 39 + "1",
        "served_model_name": "MiMoGeometryRecipe",
        "fp8_kv_scales": FP8_SCALES, "fp8_kv_geometry": geometry_id}
    locks[registered] = file_pins(checkpoint)
    uncalibrated = scratch / "uncalibrated"
    shutil.copytree(checkpoint, uncalibrated, copy_function=os.link)
    (uncalibrated / "config.json").unlink()
    uncalibrated_config = dict(config, _name_or_path="fixture/uncalibrated")
    (uncalibrated / "config.json").write_text(json.dumps(uncalibrated_config))
    declarations["checkpoints"]["geometry-recipe-uncalibrated"] = {
        "repository": "fixture/geometry-recipe", "revision": "0" * 39 + "f",
        "served_model_name": "MiMoGeometryRecipeUncalibrated"}
    locks["geometry-recipe-uncalibrated"] = dict(locks[registered])
    config_bytes = (uncalibrated / "config.json").read_bytes()
    locks["geometry-recipe-uncalibrated"]["config.json"] = {
        "byte_length": len(config_bytes), "sha256": hashlib.sha256(config_bytes).hexdigest()}
    base = json.loads((tree / "models/mimo26/geometries/v26-flash.json").read_text())
    compact = lambda value: json.dumps(value, separators=(",", ":"))
    unlisted_context = config["max_position_embeddings"] * 2
    cases = [("unlisted", "max_position_embeddings", unlisted_context,
              "unsupported geometry: v26-flash: hybrid_layer_pattern expected "
              f"{compact(base['hybrid_layer_pattern'])}, received {compact(config['hybrid_layer_pattern'])}; "
              f"{geometry_id}: max_position_embeddings expected {config['max_position_embeddings']}, "
              f"received {unlisted_context}; supported ids: v26-flash, {geometry_id}")]
    for label, field, value in [("activation", "hidden_act", "relu"),
                                ("projection", "attention_projection_layout", "separate_qkv"),
                                ("rope", "rope_parameters.rope_type", "unknown")]:
        cases.append((label, field, value,
                      f"unsupported {field}: expected {compact(supported['semantic_rules'][field])}, received {compact(value)}"))
    cases.append(("quantization", "quantization_config", {"quant_method": "unknown"},
                  "unsupported quantization_config.quant_method: expected "
                  f"{compact(supported['quantization_recipe_classes'])}, received \"unknown\""))
    domain = supported["engine_domain"]["v_head_dim"]
    cases.append(("domain", "v_head_dim", 64,
                  f"unsupported v_head_dim: expected {compact(domain['values'])} ({domain['reason']}), received 64"))
    refusals = []
    for number, (label, field, value, expected) in enumerate(cases, 2):
        root = scratch / f"refused-{label}"
        root.mkdir()
        changed = copy.deepcopy(config)
        parent = changed
        parts = field.split(".")
        for part in parts[:-1]:
            parent = parent[part]
        parent[parts[-1]] = value
        (root / "config.json").write_text(json.dumps(changed))
        for name in ("model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"):
            shutil.copyfile(checkpoint / name, root / name)
        name = f"geometry-recipe-{label}"
        declarations["checkpoints"][name] = {
            "repository": "fixture/geometry-recipe", "revision": f"{number:040x}",
            "served_model_name": f"MiMoGeometryRecipe-{label}"}
        locks[name] = file_pins(root)
        refusals.append((name, root, expected))
    declarations_path.write_text(json.dumps(declarations, indent=2) + "\n")
    locks_path.write_text(json.dumps(locks, indent=2) + "\n")
    return previous, declarations, refusals, uncalibrated


def mimo_refusals(helper, declarations, refusals, env):
    # The local identity's refusal follows the registered ones'.
    names = [declarations["default"], *sorted(name for name in declarations["checkpoints"]
                                             if name != declarations["default"]), "local"]
    for name, root, expected in refusals:
        assert not list(root.rglob("*.safetensors")), "refusal checkpoint contains tensor headers"
        for ranks in (2, 4):
            reply = probe(helper, root, ranks, env)
            assert reply["result"] == "probe" and not reply["compatible"], reply
            reason = reply["reason"]
            assert reason.startswith("not a supported MiMo V2.6 Flash checkpoint ("), reason
            start = reason.index(name + ": ") + len(name) + 2
            end = min([reason.find("; " + other + ": ", start) for other in names
                       if reason.find("; " + other + ": ", start) >= 0] or [len(reason) - 1])
            assert reason[start:end] == expected, reason
            print(f"TP{ranks} refusal: {name}: {expected}", flush=True)


def assert_contract_data_diff(tree, model, supported_before, entry, alias):
    """A contract-stage geometry adds its geometry file and supported.json entry only."""
    prefix = f"models/{model}/geometries/"
    tracked = run("git", "diff", "HEAD", "--name-only", cwd=tree).splitlines()
    untracked = run("git", "ls-files", "--others", "--exclude-standard", cwd=tree).splitlines()
    paths = sorted(set(tracked + untracked))
    geometry_file = prefix + entry["file"]
    existing = set(run("git", "ls-tree", "-r", "--name-only", "HEAD", "--", prefix, cwd=tree).splitlines())
    assert geometry_file not in existing, "new geometry file already exists"
    assert paths == sorted([prefix + "supported.json", geometry_file]), f"unexpected data-only paths: {paths}"
    expected = copy.deepcopy(supported_before)
    expected["geometries"].append(entry)
    expected["aliases"]["architecture"].append(alias)
    actual = json.loads((tree / prefix / "supported.json").read_text())
    assert actual == expected, "supported.json changed beyond the appended geometry and alias"
    print("Asserted data-only diff:", flush=True)
    for path in paths:
        print(f"  {path}", flush=True)


def declared(models=None):
    """{model: Model} for every model that declares geometry recipe legs, in
    directory order."""
    return {model.name: model for model in (load(INVENTORY) if models is None else models)
            if model.geometry_recipe}


def declaration(model):
    found = declared().get(model)
    if found is None:
        raise SystemExit(f"usage: model-geometry-recipe.sh {'|'.join(declared())}")
    return found


def contract_recipe(model, fixture_name, target_dir=None):
    """The recipe for a model whose bundle cannot yet build and serve an added
    geometry (no bundle program yet, or one compiled for fixed geometries): the
    geometry is added as data, then the contract crate admits, catalogs, plans
    and refuses by field. It stops at the contract stage."""
    declared_model = declaration(model)
    packages = declared_model.packages
    alias = declared_model.geometry_recipe.architecture
    for tool in ("git", "cargo"):
        if not shutil.which(tool):
            raise SystemExit(f"missing required tool: {tool}")
    if run("git", "status", "--porcelain", cwd=ROOT).strip():
        raise SystemExit("geometry recipe requires a clean committed HEAD")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    started = time.monotonic()
    stem = model.upper()
    with tempfile.TemporaryDirectory(prefix="chopinfer-geometry-", dir=os.environ.get("TMPDIR")) as temp:
        scratch = Path(temp)
        tree = scratch / "tree"
        try:
            run("git", "worktree", "add", "--detach", str(tree), "HEAD", cwd=ROOT)
            env = os.environ.copy()
            env.update(CARGO_TARGET_DIR=str(target_dir or scratch / "target"), CARGO_INCREMENTAL="0")
            env.pop(f"CHOPINFER_{stem}_GEOMETRY_DIRS", None)
            env.pop("KERNELS_LOCK_UPDATE", None)
            fixture = tree / f"fixtures/{model}/geometries/{fixture_name}"
            geometries = tree / f"models/{model}/geometries"
            shutil.copyfile(fixture / "geometry.json", geometries / f"{fixture_name}.json")
            supported = json.loads((geometries / "supported.json").read_text())
            supported_before = copy.deepcopy(supported)
            entry = json.loads((fixture / "supported.json").read_text())["geometries"][0]
            nonce = hashlib.sha256(scratch.name.encode()).hexdigest()[:12]
            entry["id"] = f"recipe{nonce}"
            entry["kernel_tag"] = f"recipe_{nonce}"
            entry["file"] = f"{fixture_name}.json"
            print(f"Geometry {entry['id']}, kernel tag {entry['kernel_tag']}", flush=True)
            supported["geometries"].append(entry)
            supported["aliases"]["architecture"].append(alias)
            (geometries / "supported.json").write_text(json.dumps(supported, indent=2) + "\n")
            checkpoint = scratch / "checkpoint"
            checkpoint.mkdir()
            config = json.loads((fixture / "config.json").read_text())
            config["architectures"] = [alias]
            (checkpoint / "config.json").write_text(json.dumps(config))
            env[f"CHOPINFER_{stem}_RECIPE_CONFIG"] = str(checkpoint / "config.json")
            env[f"CHOPINFER_{stem}_RECIPE_GEOMETRY"] = entry["id"]
            locked = ("--locked", "--offline")
            print("Admit, catalog, plan and refuse the data-only geometry", flush=True)
            result = run("cargo", "test", "-p", packages["contract"].name, "--lib", "--test", "geometry_recipe",
                         *locked, "--", "--nocapture", cwd=tree, env=env)
            print(result, flush=True)
            assert_tests_ran(result, 2)
            assert f"{entry['id']} TP4:" in result, "the recipe geometry was not planned"
            result = run("cargo", "test", "-p", packages["compiler"].name, "--lib", *locked, cwd=tree, env=env)
            print(result, flush=True)
            assert_tests_ran(result, 1)
            assert_contract_data_diff(tree, model, supported_before, entry, alias)
        finally:
            if tree.exists():
                run("git", "worktree", "remove", "--force", str(tree), cwd=ROOT)
    print(f"geometry recipe {model}/{fixture_name}: contract stage; the bundle does not serve an added geometry yet: "
          f"{time.monotonic() - started:.3f}s (exit 0)", flush=True)


def recipe(model, fixture_name="synthetic-small", target_dir=None):
    declared_model = declaration(model)
    declared_recipe = declared_model.geometry_recipe
    if declared_recipe.stage == "contract":
        return contract_recipe(model, fixture_name, target_dir)
    bundle, compiler, contract = (declared_model.packages[role].name for role in ("bundle", "compiler", "contract"))
    tokenizer = tokenizer_dir(declared_model.checkpoint_env)
    mask = None
    if model == "mimo26":
        draft_dir = os.environ.get("CHOPINFER_MIMO_DFLASH_DIR")
        assets = os.environ.get("CHOPINFER_MIMO_CHECKPOINT", tokenizer)
        mask = Path(draft_dir) / "mask_embedding.pt" if draft_dir else Path(assets) / "dflash/mask_embedding.pt"
        if not mask.is_file():
            raise SystemExit("provide the pinned mask_embedding.pt through CHOPINFER_MIMO_DFLASH_DIR or the tokenizer/checkpoint directory's dflash subdirectory")
    require_build_tools()
    require_clean(ROOT, "geometry recipe")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    started = time.monotonic()
    with detached_worktree(ROOT, "chopinfer-geometry-", target_dir) as (scratch, tree, env):
        fixture = tree / f"fixtures/{model}/geometries/{fixture_name}"
        geometries = tree / f"models/{model}/geometries"
        before = lock_state(tree, model)
        geometry_file = "geometry.json"
        shutil.copyfile(fixture / geometry_file, geometries / f"{fixture_name}.json")
        if model == "qwen38":
            shutil.copyfile(fixture / "config.json", geometries / f"{fixture_name}-config.json")
        supported = json.loads((geometries / "supported.json").read_text())
        supported_before = copy.deepcopy(supported)
        entry = json.loads((fixture / "supported.json").read_text())["geometries"][0]
        # Derive both names from the unique scratch directory so committed
        # fixture-specific admission cannot satisfy a new run.
        nonce = hashlib.sha256(scratch.name.encode()).hexdigest()[:12]
        entry["id"] = f"recipe{nonce}"
        entry["kernel_tag"] = f"recipe_{nonce}"
        print(f"Geometry {entry['id']}, kernel tag {entry['kernel_tag']}", flush=True)
        entry["file"] = f"{fixture_name}.json"
        if model == "mimo26":
            assert entry.get("encoders"), "MiMo recipe fixture must enable vision and audio encoders"
        supported["geometries"].append(entry)
        alias = declared_recipe.architecture
        supported["aliases"]["architecture"].append(alias)
        (geometries / "supported.json").write_text(json.dumps(supported, indent=2) + "\n")
        cargo = ("cargo",)
        locked = ("--locked", "--offline")
        build_env = dict(env, KERNELS_LOCK_UPDATE="1")
        print("Build the bundle with data-only geometry additions", flush=True)
        run(*cargo, "build", "-p", bundle, "--lib", *locked, cwd=tree, env=build_env)
        placement_result = run(*cargo, "test", "-p", bundle,
                               "--test", "projection_assembly", *locked, cwd=tree, env=env)
        print(placement_result, flush=True)
        assert_tests_ran(placement_result, 1)
        suites = declared_recipe.contract_tests
        selectors = [argument for suite in suites for argument in ("--test", suite)]
        catalog_result = run(*cargo, "test", "-p", contract, *selectors,
                             *locked, cwd=tree, env=env)
        print(catalog_result, flush=True)
        assert_tests_ran(catalog_result, len(suites))
        if model == "qwen38":
            result = run(*cargo, "test", "-p", compiler, "--example", "geometry-checkpoint",
                         *locked, cwd=tree, env=env)
            print(result, flush=True)
            assert_tests_ran(result, 1)
        run(*cargo, "build", "-p", compiler, "--example", "geometry-checkpoint", *locked, cwd=tree, env=env)
        checkpoint = scratch / "checkpoint"
        generator = Path(env["CARGO_TARGET_DIR"]) / "debug/examples/geometry-checkpoint"
        inputs = [str(fixture / "config.json"), tokenizer]
        if mask:
            inputs.append(str(mask))
        if model == "qwen38":
            print("Check Flash encoder padding and plan destinations", flush=True)
            flash_config = tree / "models/qwen38/compiler/fixtures/config.json"
            width = json.loads(flash_config.read_text())["vision_config"]["intermediate_size"]
            assert width == json.loads((tree / "models/qwen38/geometries/flash.json").read_text())["vision_config"]["intermediate_size"]
            assert width % 128, "Flash plan check must exercise encoder row padding"
            print(run(str(generator), str(flash_config), tokenizer,
                      str(scratch / "flash-checkpoint"), env=env), flush=True)
        print(run(str(generator), *inputs, str(checkpoint), env=env), flush=True)
        config = json.loads((checkpoint / "config.json").read_text())
        config["architectures"] = [alias]
        (checkpoint / "config.json").write_text(json.dumps(config))
        checkpoint_data = None
        if model == "mimo26":
            checkpoint_data, declarations, refusals, uncalibrated = mimo_registration(
                tree, checkpoint, config, supported, scratch, entry["id"])
        run(*cargo, "build", "-p", compiler, "--bin", f"chopinfer-model-{model}",
            *locked, cwd=tree, env=env)
        env[f"CHOPINFER_{model.upper()}_GEOMETRY_JOURNAL_DIR"] = str(scratch / "journals")
        result = run(*cargo, "test", "-p", bundle, "--lib", *locked,
                     "every_geometry_topology_runs_the_fake_host_journal_and_lifecycle",
                     "--", "--nocapture", cwd=tree, env=env)
        print(result, flush=True)
        assert_tests_ran(result, 1)
        for ranks in (2, 4):
            assert list((scratch / "journals").glob(f"*{entry['id']}*tp{ranks}*")), "missing geometry journal"
        if model == "mimo26":
            result = run(*cargo, "test", "-p", bundle, "--lib", *locked,
                         "geometry_encoders_run_tp2_tp4_media_lifecycle", "--", "--nocapture",
                         cwd=tree, env=env)
            print(result, flush=True)
            assert_tests_ran(result, 1)
            result = run(*cargo, "test", "-p", bundle, "--lib", *locked,
                         "compiled_fp8_layouts_select_their_scales_at_create", "--", "--nocapture",
                         cwd=tree, env=env)
            print(result, flush=True)
            assert_tests_ran(result, 1)
        if model == "qwen38":
            for diagnostic in (scratch / "journals").glob("*/diagnostic.json"):
                reproduction = json.loads(diagnostic.read_text())["reproduction"]
                assert any(part.startswith("CHOPINFER_QWEN38_GEOMETRY_JOURNAL_DIR=") for part in reproduction), reproduction
                assert "fake_host_tests::every_geometry_topology_runs_the_fake_host_journal_and_lifecycle" in reproduction, reproduction
        helper = Path(env["CARGO_TARGET_DIR"]) / f"debug/chopinfer-model-{model}"
        for ranks in (2, 4):
            reply = probe(helper, checkpoint, ranks, env)
            assert reply == {"result": "probe", "compatible": True, "reason": "compatible metadata"}, reply
            reply = json.loads(run(str(helper), "prepare", "--direct", "--model", str(checkpoint),
                                   "--tensor-parallel-size", str(ranks), "--modalities", "all",
                                   "--cache-dir", str(scratch / f"cache-tp{ranks}"), env=env))
            assert reply["result"] == "prepared-direct", reply
            assert reply["tensor_parallel_size"] == ranks, reply
            assert entry["id"] in reply["profile"]["profile"]["topology"], reply
            assert Path(reply["bundle"]).is_dir(), reply
            frontend = json.loads((Path(reply["bundle"]) / "components/frontend.json").read_text())
            text_config = config.get("text_config", config)
            assert frontend["vocabulary"]["size"] == text_config["vocab_size"], frontend
            assert frontend["context"]["maximum_total_tokens"] == text_config["max_position_embeddings"], frontend
            assert frontend["limits"]["maximum_generated_tokens"] == text_config["max_position_embeddings"], frontend
            if model == "mimo26":
                manifest = json.loads((Path(reply["bundle"]) / "manifest.json").read_text())
                assert_mimo_layouts(manifest, ["mimo.ga-paged-bf16.v1", FP8_LAYOUT])
                uncalibrated_reply = json.loads(run(
                    str(helper), "prepare", "--direct", "--model", str(uncalibrated),
                    "--tensor-parallel-size", str(ranks), "--modalities", "all",
                    "--cache-dir", str(scratch / f"uncalibrated-cache-tp{ranks}"), env=env))
                assert uncalibrated_reply["result"] == "prepared-direct", uncalibrated_reply
                manifest = json.loads((Path(uncalibrated_reply["bundle"]) / "manifest.json").read_text())
                assert_mimo_layouts(manifest, ["mimo.ga-paged-bf16.v1"])
                assert reply["profile"]["profile"]["speculation"]["drafter"] == ("dflash" if ranks == 4 else "none"), reply
            for identity in ("package_identity", "checkpoint_identity", "metadata_identity"):
                assert len(reply[identity]) == 64, reply
            print(f"TP{ranks}: helper probe, authentication and emit_direct passed", flush=True)
        if model == "qwen38":
            flash = json.loads((geometries / "flash.json").read_text())
            qwen_refusals(helper, config, flash, scratch, env, entry["id"])
        elif model == "mimo26":
            mimo_refusals(helper, declarations, refusals, env)
        assert_data_diff(tree, model, before, checkpoint_data,
                         supported_before=supported_before, entry=entry, alias=alias)
    print(f"geometry recipe {model}/{fixture_name}: {time.monotonic() - started:.3f}s (exit 0)", flush=True)


def legs(selection, fixture=None, models=None):
    """(model, fixture) pairs for `--ci` (bundle-stage legs), `--contract`
    (models whose recipe stops at the contract stage while their bundle cannot
    serve an added geometry) or one model, from the declarations of `models`
    (default: the inventory) in directory order."""
    recipes = declared(models)
    stage = {"--ci": "bundle", "--contract": "contract"}.get(selection)
    if stage is None and selection not in recipes:
        raise SystemExit(f"usage: model-geometry-recipe.sh --ci|--contract|{'|'.join(recipes)} [fixture]")
    selected = [model for model in recipes.values()
                if model.geometry_recipe.stage == stage or model.name == selection]
    return [(model.name, name) for model in selected
            for name in ([fixture] if fixture else model.geometry_recipe.fixtures)]


def lanes(planned, count):
    """Group legs into lanes that run at once. Legs in one lane run in turn and
    share one target, so the second reuses the first's dependency builds: two
    lanes keep each model's legs together, three or more give every leg its own."""
    if count <= 1:
        return [planned]
    if count >= len(planned):
        return [[leg] for leg in planned]
    grouped = {}
    for leg in planned:
        grouped.setdefault(leg[0], []).append(leg)
    return list(grouped.values())


def run_lane(lane, target):
    for model, fixture in lane:
        recipe(model, fixture, target)


def run_lanes(groups, budget):
    """Run each lane as a supervised child with its own temporary target and an
    equal share of the build budget: Cargo jobs and kernel-build's nvcc pool."""
    sys.path.insert(0, str(ROOT / "ci"))
    import runner
    share = str(max(budget // len(groups), 1))
    env = dict(os.environ, CARGO_BUILD_JOBS=share, CHOPINFER_KERNEL_BUILD_JOBS=share)
    with tempfile.TemporaryDirectory(prefix="chopinfer-geometry-lanes-", dir=os.environ.get("TMPDIR")) as scratch:
        steps = []
        for index, lane in enumerate(groups):
            target = Path(scratch) / f"target-{index}"
            target.mkdir()
            name = "+".join(f"{model}/{fixture}" for model, fixture in lane)
            argv = [sys.executable, "-B", __file__, "--lane", ",".join(f"{model}:{fixture}" for model, fixture in lane),
                    "--target", str(target)]
            steps.append(runner.Step(name, argv, env=env))
        print(f"geometry recipe: {len(groups)} lanes, {share} build jobs each", flush=True)
        try:
            results = runner.run(steps, len(steps), Path(scratch) / "logs", out=sys.stdout, label="geometry lane",
                                 grace=120)
        except runner.Interrupted as interruption:
            return 128 + interruption.signum
        finally:
            # A stopped lane may leave its worktree registered.
            subprocess.run(["git", "worktree", "prune"], cwd=ROOT, check=False)
    failed = [f"{result.name} ({result.status})" for result in results if result.status != "ok"]
    if failed:
        print("geometry recipe failed: " + ", ".join(failed), flush=True)
    return 1 if failed else 0


def main(selection, models=None):
    """Run the legs `legs(*selection)` plans, from the declarations of `models`
    (default: the inventory); return the exit status. A stage no model
    declares, such as `--contract` once every model reaches the bundle stage,
    runs nothing and succeeds."""
    planned = legs(*selection, models=models)
    if not planned:
        print(f"geometry recipe {selection[0]}: no model declares these legs, nothing to run", flush=True)
        return 0
    groups = lanes(planned, int(os.environ.get("CHOPINFER_GEOMETRY_LANES", "2")))
    if len(groups) > 1:
        return run_lanes(groups, int(os.environ.get("CHOPINFER_GEOMETRY_JOBS", os.cpu_count() or 8)))
    # One lane: every leg reuses dependencies from the one before while each
    # geometry still builds in its own clean worktree. Remove the shared target
    # after all legs or SIGTERM.
    with tempfile.TemporaryDirectory(prefix="chopinfer-geometry-target-", dir=os.environ.get("TMPDIR")) as target:
        run_lane(groups[0], Path(target))
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if sys.argv[1:2] == ["--lane"] and len(sys.argv) == 5 and sys.argv[3] == "--target":
        run_lane([tuple(leg.split(":")) for leg in sys.argv[2].split(",")], Path(sys.argv[4]))
        sys.exit(0)
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: model-geometry-recipe.sh <model> [fixture]")
    sys.exit(main(sys.argv[1:]))

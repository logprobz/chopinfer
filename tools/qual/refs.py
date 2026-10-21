"""Adopt and authenticate the legacy inputs actually read by qualification gates."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))

CODE_INPUTS = ("media_cases", "media_images")
LOCK_KEYS = ("schema", "model", "entries", "objects")
# The fields of an entry `snapshot` writes; adopt keeps any other field.
SNAPSHOT_FIELDS = ("path", "directory", "members", "scope")
# Lock sections adopt keeps as they are: each configuration's own entries.
KEPT_SECTIONS = ("configurations",)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def files(path, scope=None):
    path = Path(path)
    if scope is not None:
        return [path / member for member in scope]
    if path.is_dir():
        return sorted(member for member in path.rglob("*") if member.is_file())  # check-inputs: walks fixtures
    return [path]


def schema(path):
    if path.suffix == ".json":
        value = json.loads(path.read_bytes())
        return value.get("schema", "legacy-json-object") if isinstance(value, dict) else "legacy-json-array"
    return "legacy-" + (path.suffix.lstrip(".") or "text")


def root_path(root=None):
    from common import reference_root
    return reference_root(root)


def lock_path(model, root=None, variant=None):
    """A model's reference lock, or a checkpoint variant's beside it."""
    root = root_path(root)
    name = "qual-references.lock.json" if variant is None else f"qual-references.{variant}.lock.json"
    return root / "models" / model / name


def external_lock_path(config):
    """An external variant's reference lock, beside its overlay:
    `<variant root>/<model>/<variant>.lock.json`."""
    import declaration
    return declaration.require_variant_root(config) / config["model"] / f"{config['variant']}.lock.json"


def config_lock_path(config, root=None):
    """The lock `adopt`, `verify` and `status` read and write for `config`:
    an external variant's beside its overlay, a variant file's beside it,
    else the model's (which also holds each configuration's entries)."""
    import declaration
    if declaration.external(config):
        return external_lock_path(config)
    if config.get("variant") is None:
        return lock_path(config["model"], root)
    return lock_path(config["model"], root, config["variant"])


def variant_source(config):
    """What receipts record of the external variant `config` came from: the
    variant root and the overlay, its lock and the checkpoint lock it pins,
    each by its path under that root and its SHA-256 (the overlay's as it
    was expanded). None for a config the tree declares."""
    import declaration
    if not declaration.external(config):
        return None
    root = declaration.require_variant_root(config)
    lock = external_lock_path(config).relative_to(root).as_posix()
    checkpoint = config["qualification"]["checkpoint"]["lock"]
    return {"root": str(root),
            "overlay": {"path": config["variant_source"]["overlay"], "sha256": config["variant_source"]["sha256"]},
            "lock": {"path": lock, "sha256": digest(root / lock)},
            "checkpoint_lock": {"path": checkpoint, "sha256": digest(root / checkpoint)}}


def reference_sibling(source, name):
    """A sibling path preserving an external reference's environment base."""
    path = Path(source.get("path", "") if isinstance(source, dict) else source)
    sibling = str(path.with_name(name))
    return {**source, "path": sibling} if isinstance(source, dict) else sibling


def sources(config):
    from common import declared_path
    result = {key: value for key, value in config["inputs"].items() if key not in CODE_INPUTS}
    for kind, spec in config.get("qualification", {}).get("references", {}).items():
        result["reference:" + kind] = spec["path"]
        source = spec["path"]
        path = Path(source.get("path", "") if isinstance(source, dict) else source)
        workload = spec.get("workload", reference_sibling(source, path.stem + ".workload.json"))
        resolved = declared_path(workload)
        # A reference without a workload file beside it declares none.
        if "workload" in spec or resolved is not None and resolved.is_file():
            result["workload:" + kind] = workload
        if kind == "media":
            result["media_capture"] = reference_sibling(source, "reference.json")
            result["media_manifest"] = reference_sibling(source, "manifest.json")
    needle = config.get("qualification", {}).get("needle", {})
    if needle.get("format") == "prompt-set":
        result["needle_frozen"] = needle["frozen"]
    return {key: value for key, value in result.items() if value is not None}


def scope(config, key):
    """The members of directory input `key` the declaration selects: the
    needle's context files. Other directory inputs pin the members their
    lock entry scopes, else every file."""
    if key == "needle" and config.get("qualification", {}).get("needle", {}).get("format") == "context-files":
        return ["manifest.json", *[f"context-{length}.json" for length in config["qualification"]["needle"]["lengths"]]]
    return None


def validate_lock(lock, config, root=None):
    from common import declared_path
    root = root_path(root)
    if not isinstance(lock, dict) or lock.get("schema") != "chopinfer-qual-references/v1":
        raise ValueError("reference lock: unsupported schema")
    if lock.get("model") != config.get("model"):
        raise ValueError("reference lock: model differs from declaration")
    entries, objects = lock.get("entries"), lock.get("objects")
    if not isinstance(entries, dict) or not isinstance(objects, dict):
        raise ValueError("reference lock: entries and objects must be objects")
    for sha, obj in objects.items():
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("reference lock: invalid object digest")
        if not isinstance(obj, dict) or not isinstance(obj.get("schema"), str) or not obj["schema"]:
            raise ValueError(f"reference lock object {sha}: missing schema")
    declared = sources(config)
    for key, entry in entries.items():
        label = f"lock entry {key}"
        if not isinstance(entry, dict):
            raise ValueError(label + ": must be an object")
        if key.startswith("operator:"):
            # A record of former operator evidence policy: it pins no bytes
            # and no reader admits anything by it.
            if "directory" in entry or "members" in entry or "scope" in entry:
                raise ValueError(label + ": operator evidence cannot masquerade as a byte pin")
            continue
        if key not in declared or entry.get("path") != declared[key]:
            raise ValueError(label + ": declaration differs")
        directory = entry.get("directory")
        members = entry.get("members")
        if type(directory) is not bool or not isinstance(members, dict):
            raise ValueError(label + ": directory must be boolean and members must be an object")
        path = declared_path(declared[key], root)
        if path is not None and path.exists() and directory != path.is_dir():
            raise ValueError(label + ": file/directory type differs from source")
        if "scope" in entry:
            selected = entry["scope"]
            if (not directory or not isinstance(selected, list) or not all(isinstance(member, str) for member in selected)
                    or set(members) != set(selected)):
                raise ValueError(label + ": directory scope differs from its members")
        if scope(config, key) not in (None, entry.get("scope")):
            raise ValueError(label + ": directory scope differs from the declaration")
        if not directory and set(members) != {"."}:
            raise ValueError(label + ": file must have exactly one dot member")
        for member, sha in members.items():
            if directory and (not isinstance(member, str) or not member or member == "." or
                              Path(member).is_absolute() or ".." in Path(member).parts):
                raise ValueError(label + ": invalid directory member")
            if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha) or sha not in objects:
                raise ValueError(label + f": member {member} references an invalid or missing object")
    for key in declared:
        if key not in entries:
            raise ValueError(f"lock entry {key}: not adopted")
    return lock


def load_lock(config, root=None):
    """The validated lock of the config's model or checkpoint variant;
    required for a model or a variant with a declaration in this workspace
    or the reference root, so every read of their inputs is authenticated."""
    import declaration
    model = config.get("model", "")
    variant = config.get("variant")
    if declaration.external(config):
        # An external variant's lock is beside its overlay, and required.
        path = external_lock_path(config)
        if not path.is_file():
            raise ValueError(f"reference lock is missing: {path}")
        return validate_lock(json.loads(path.read_text()), config, root)
    if variant is not None and lock_path(model, root).is_file() and any(
            variant in declaration.configurations(model, tree) for tree in (ROOT, root_path(root))):
        # A configuration reads its model's lock: the model's entries, with
        # the configuration's own where they differ, for the inputs it
        # declares. (An exported configuration ships its own lock alone.)
        path = lock_path(model, root)
        lock = json.loads(path.read_text())
        entries = {**lock.get("entries", {}),
                   **lock.get("configurations", {}).get(variant, {}).get("entries", {})}
        declared = sources(config)
        lock = {**{key: value for key, value in lock.items() if key != "configurations"},
                "entries": {key: entry for key, entry in entries.items()
                            if key in declared or key.startswith("operator:")}}
        return validate_lock(lock, config, root)
    path = lock_path(model, root) if variant is None else lock_path(model, root, variant)
    if not path.exists():
        if any(declaration.path(model, tree).is_file()
               or (variant is not None and declaration.path(model, tree, variant).is_file())
               for tree in (ROOT, root_path(root))):
            raise ValueError("reference lock is missing: " + path.name)
        return None
    return validate_lock(json.loads(path.read_text()), config, root)


def verify_preprocessing_fixture(fixture, record):
    expected = {(c["source"]["shape"][1], c["source"]["shape"][0]):
                (c["grid"], c["stages"]["patches"]["integer"]["sha256"]) for c in record["cases"]}
    actual = {(c["width"], c["height"]): (c["grid"], c["integer_patch_sha256"]) for c in fixture["images"]}
    if (actual != expected or fixture["overrides"] != record["binding"]["processor"]["overrides"]
            or fixture["versions"] != record["binding"]["environment"]
            or fixture["checkpoint_config_sha256"] != record["binding"]["configs"]["preprocessor_config.json"]):
        raise ValueError("portable integer fixture differs from adopted preprocessing reference")

def verify_mixed_generation(value, generated):
    """Every forced answer is the full text of the recorded F32 continuation."""
    expected = {case["id"]: case for case in generated["cases"]}
    stops = value["provenance"]["generation"]["generation"]["stop_ids"]
    if set(expected) != {"single-image", "two-images", "two-resized-images", "image-video"}:
        raise ValueError("mixed generated answer case set differs")
    for case in value["cases"]:
        if case["id"] not in expected:
            continue
        ids = expected[case["id"]]["answer_token_ids"]
        if len(ids) < 33 or ids[-1] not in stops or any(token in stops for token in ids[:-1]):
            raise ValueError("mixed generated answer is short or unfinished")
        if case["answer_token_ids"] != ids[:-1] or case["terminal_stop_id"] != ids[-1]:
            raise ValueError("mixed forced answer differs from full F32 generation")
        entries = generated["binding"]["modes"]["f32"]["entries"]
        greedy_key = next(key for name, index, key in entries if name == case["id"]
                          and generated["binding"]["sequences"][index]["answer"]["policy"] == "greedy")
        if case["generation_key_sha256"] != greedy_key:
            raise ValueError("mixed answer generation key differs")


def snapshot(config, root=None, *, previous=None, only=None):
    """The lock of every input `config` declares (of those whose lock keys
    `only` names, when given): each file's or scoped directory member's
    SHA-256. A directory keeps the scope of its `previous` entry at the same
    path. Bound references, their workloads and preprocessing records are
    verified before their bytes are pinned, and their objects record what
    proved them; other objects keep the record `previous` holds for the same
    bytes."""
    from common import binding_names_pin, declared_path, verified_binding
    root = root_path(root)
    wanted = lambda key: only is None or key in only  # noqa: E731
    proven = {}
    for kind, spec in config.get("qualification", {}).get("references", {}).items():
        if not wanted("reference:" + kind):
            continue
        path = declared_path(spec["path"], root)
        value = json.loads(path.read_bytes())
        if value.get("binding"):
            # Candidate bytes are proved by the runner binding before adoption
            # pins them, against the candidate workload rather than the previous
            # lock's pin of it (which a gate read of a floor-only reference checks).
            binding_names_pin(kind, value["binding"], config)
            status = verified_binding(kind, value, config, authenticate_workload=False)
            binding = value["binding"]
            proven[digest(path)] = {
                "producer": "runner-binding", "refset_sha256": status["refset_sha256"],
                "assembled_fixture_sha256": binding["assembled_fixture_sha256"],
                "workload_sha256": binding["workload_sha256"],
            }
            workload = declared_path(sources(config)["workload:" + kind], root)
            proven[digest(workload)] = {
                "producer": "runner-workload", "refset_sha256": status["refset_sha256"],
                "workload_sha256": binding["workload_sha256"],
            }
    generation = config.get("inputs", {}).get("media_mixed_generation")
    if generation and wanted("media_mixed_generation"):
        from common import reference_binding
        capture_path = declared_path(generation, root)
        generated = json.loads(capture_path.read_bytes())
        workload = declared_path(config["inputs"]["media_mixed_generation_workload"], root)
        status = reference_binding(generated, workload=workload)
        if status["status"] != "bound" or set(generated["binding"]["modes"]) != {"f32"}:
            raise ValueError("mixed answer generation needs a bound F32 reference")
        fixture = json.loads(declared_path(config["qualification"]["references"]["media_mixed"]["path"], root).read_bytes())
        if fixture["provenance"]["generation_reference_sha256"] != digest(capture_path):
            raise ValueError("mixed answer generation digest differs")
        verify_mixed_generation(fixture, generated)
        proven[digest(capture_path)] = {"producer": "runner-binding", "refset_sha256": status["refset_sha256"],
            "assembled_fixture_sha256": generated["binding"]["assembled_fixture_sha256"],
            "workload_sha256": generated["binding"]["workload_sha256"]}
        proven[digest(workload)] = {"producer": "runner-workload", "refset_sha256": status["refset_sha256"],
                                   "workload_sha256": generated["binding"]["workload_sha256"]}
    preprocessing_path = config.get("inputs", {}).get("media_preprocessing_reference")
    if preprocessing_path is not None and wanted("media_preprocessing_reference"):
        from tools.refrun import preprocessing
        directory = declared_path(preprocessing_path, root)
        manifests = sorted(directory.glob("*/reference.json")) if directory is not None else []
        if not manifests:
            raise ValueError("lock entry media_preprocessing_reference: missing input")
        for manifest in manifests:
            record = preprocessing.verify(json.loads(manifest.read_bytes()), manifest.parent)
            fixture_path = config.get("inputs", {}).get("media_preprocessing_integer_fixture")
            if fixture_path and record["cases"][0]["kind"] == "image":
                fixture = json.loads(declared_path(fixture_path, root).read_bytes())
                verify_preprocessing_fixture(fixture, record)
            production = {"producer": "runner-preprocessing-binding", "reference_sha256": record["sha256"]}
            proven[digest(manifest)] = production
            for case in record["cases"]:
                tensors = [case["source"], *(tensor for stage in case["stages"].values() for tensor in stage.values())]
                for tensor in tensors:
                    proven[tensor["sha256"]] = production
    ab_path = config.get("inputs", {}).get("media_preprocessing_ab")
    if ab_path and wanted("media_preprocessing_ab"):
        from tools.refrun import preprocessing_ab
        ab_path = declared_path(ab_path, root)
        record = json.loads(ab_path.read_bytes())
        for name, sha in record["code"].items():
            if digest(root / name) != sha:
                raise ValueError(f"stale preprocessing A/B producer: {name}")
        preprocessing_ab.require_effective_probes(record["sources"], record["resized_images"])
        probes = declared_path(config["inputs"]["media_preprocessing_ab_probes"], root)
        if digest(probes) != record["probe_manifest_sha256"]:
            raise ValueError("preprocessing A/B probe manifest differs")
        production = {"producer": "runner-preprocessing-ab-binding", "reference_sha256": digest(ab_path)}
        proven[digest(ab_path)] = production
        proven[digest(probes)] = production
    entries, objects = {}, {}
    old_entries = (previous or {}).get("entries", {})
    old_objects = (previous or {}).get("objects", {})
    for key, location in sources(config).items():
        if not wanted(key):
            continue
        path = declared_path(location, root)
        if path is None or not path.exists():
            raise ValueError(f"lock entry {key}: missing input")
        members = {}
        selected = scope(config, key)
        if selected is None and path.is_dir() and old_entries.get(key, {}).get("path") == location:
            selected = old_entries[key].get("scope")
        for member in files(path, selected):
            sha = digest(member)
            relative = member.relative_to(path).as_posix() if path.is_dir() else "."
            members[relative] = sha
            if sha in proven:
                objects[sha] = {"schema": schema(member), "production": proven[sha]}
            else:
                objects[sha] = old_objects.get(sha) or {"schema": schema(member)}
        entries[key] = {"path": location, "directory": path.is_dir(), "members": members}
        if selected is not None:
            entries[key]["scope"] = selected
    return {"schema": "chopinfer-qual-references/v1", "model": config["model"],
            "entries": entries, "objects": objects}


def adoption(config, previous=None, root=None):
    """The lock `adopt` writes: `snapshot` of the inputs `config` declares,
    with everything else `previous` holds kept byte for byte. The snapshot
    replaces the declared entries and the objects only they pinned; the
    configurations section, operator records, every other object (those a
    configuration's entries pin, and records nothing pins any more) and an
    entry's own fields beside its pin (its role, producer policy, reason)
    stay. An object only the replaced entries pinned takes a fresh proof of
    the same bytes (`kept_record`). A previous lock adopt cannot keep whole
    (another schema or model, an unknown section, a configuration with more
    than entries or whose object is missing, an entry with its own fields
    whose path changed, an object a kept entry or configuration pins whose
    record contradicts what adoption proved of its bytes) is refused."""
    value = snapshot(config, root, previous=previous)
    if previous is None:
        return value
    if (previous.get("schema"), previous.get("model")) != (value["schema"], value["model"]):
        raise ValueError("reference lock: the current lock has another schema or model; adopt does not replace it")
    unknown = sorted(set(previous) - set(LOCK_KEYS) - set(KEPT_SECTIONS))
    if unknown:
        raise ValueError(f"reference lock: adopt cannot keep the unknown section {', '.join(unknown)}")
    pinned = set()
    for name, configuration in previous.get("configurations", {}).items():
        if not isinstance(configuration, dict) or set(configuration) != {"entries"} \
                or not isinstance(configuration["entries"], dict):
            raise ValueError(f"reference lock configuration {name}: adopt can keep only its entries")
        for key, entry in configuration["entries"].items():
            members = entry.get("members") if isinstance(entry, dict) else None
            if not isinstance(members, dict):
                raise ValueError(f"reference lock configuration {name} entry {key}: members must be an object")
            for sha in members.values():
                if sha not in previous["objects"]:
                    raise ValueError(f"reference lock configuration {name} entry {key}: object {sha} is missing")
                pinned.add(sha)
    for key, entry in value["entries"].items():
        old = previous["entries"].get(key)
        own = {field: data for field, data in (old or {}).items() if field not in SNAPSHOT_FIELDS}
        if own and old.get("path") != entry["path"]:
            raise ValueError(f"lock entry {key}: its path changed; adopt cannot carry its "
                             f"{', '.join(sorted(own))} to the new path, so restate them in the lock first")
        entry.update(copy.deepcopy(own))
    for section in KEPT_SECTIONS:
        if section in previous:
            value[section] = copy.deepcopy(previous[section])
    value["entries"].update({key: copy.deepcopy(entry) for key, entry in previous["entries"].items()
                             if key.startswith("operator:")})
    # The objects an entry or configuration adopt keeps pins: their records
    # stand for those pins too, so a fresh proof cannot rewrite them.
    pinned |= {sha for key, entry in previous["entries"].items() if key.startswith("operator:")
               and isinstance(entry.get("members"), dict) for sha in entry["members"].values()}
    replaced = {sha for key, entry in previous["entries"].items() if not key.startswith("operator:")
                for sha in entry.get("members", {}).values()} - pinned
    for sha, record in previous["objects"].items():
        if sha in value["objects"]:
            value["objects"][sha] = kept_record(sha, value["objects"][sha], record, renew=sha in replaced)
        elif sha not in replaced:
            value["objects"][sha] = copy.deepcopy(record)
    return value


def external_adoption(config, root=None):
    """The lock of the external variant `config`: the model lock's record of
    every input the variant shares with the model (the same lock key at the
    same path, entry and objects as the model lock holds them), and a
    snapshot of everything else, its references always: each is proved by
    its runner binding against the variant's own checkpoint pin, so a kept
    base reference bound to another checkpoint is refused."""
    model_lock = json.loads(lock_path(config["model"], root).read_text())
    declared = sources(config)
    shared = {key for key, location in declared.items()
              if not key.startswith(("reference:", "workload:"))
              and model_lock["entries"].get(key, {}).get("path") == location}
    lock = snapshot(config, root, previous=model_lock, only=set(declared) - shared)
    for key in sorted(shared):
        entry = copy.deepcopy(model_lock["entries"][key])
        lock["entries"][key] = entry
        for sha in entry["members"].values():
            lock["objects"][sha] = copy.deepcopy(model_lock["objects"][sha])
    return lock


def kept_record(sha, proved, record, renew=False):
    """The record of object `sha` after adoption: the lock's `record` with
    what `snapshot` `proved` of the bytes, its schema and production. Fields
    the proof does not name stay; a record whose schema, producer or a
    production field differs from the proof is refused. With `renew` (only
    entries this adoption replaces pinned the bytes) the proof's other
    production fields replace the record's: the same bytes proved again by
    the same producer, as when a pin bump changes the requirements a
    preprocessing reference binds."""
    for field in ("schema", "production"):
        if field in proved and field in record and not isinstance(proved[field], dict) \
                and proved[field] != record[field]:
            raise ValueError(f"lock object {sha}: the lock's record differs from what adoption proved in {field}")
    proof, before = proved.get("production"), record.get("production")
    if isinstance(proof, dict) and isinstance(before, dict):
        clash = sorted(field for field in proof if field in before and before[field] != proof[field]
                       and (not renew or field == "producer"))
        if clash:
            raise ValueError(f"lock object {sha}: the lock's record differs from what adoption proved in "
                             f"{', '.join(clash)}")
    elif isinstance(proof, dict) and before is not None:
        raise ValueError(f"lock object {sha}: the lock's record differs from what adoption proved in production")
    result = {**copy.deepcopy(record), **{field: data for field, data in proved.items() if field != "production"}}
    if isinstance(proof, dict):
        result["production"] = {**copy.deepcopy(before or {}), **proof}
    return result


def encoded(value, previous=None):
    """The lock's bytes: two-space JSON in `previous`'s key order, keys it
    lacks sorted after them, so an adoption rewrites only what it changes."""
    def ordered(new, old):
        if isinstance(new, dict):
            old = old if isinstance(old, dict) else {}
            keys = [key for key in old if key in new] + sorted(key for key in new if key not in old)
            return {key: ordered(new[key], old.get(key)) for key in keys}
        if isinstance(new, list):
            old = old if isinstance(old, list) and len(old) == len(new) else [None] * len(new)
            return [ordered(item, before) for item, before in zip(new, old)]
        return new
    return (json.dumps(ordered(value, previous), indent=2) + "\n").encode()


def gate_reference_producers(config, lock):
    """Admit the numerical references `lock` pins: every object a declared
    reference entry pins must record the runner binding `snapshot` verified
    before pinning it. An unbound reference is refused here, at adoption and
    at gate admission; the binding is not verified again on each read."""
    for kind in config.get("qualification", {}).get("references", {}):
        key = "reference:" + kind
        for member, sha in lock["entries"][key]["members"].items():
            production = lock["objects"][sha].get("production")
            if not isinstance(production, dict) or production.get("producer") != "runner-binding" or not all(
                    isinstance(production.get(field), str) and re.fullmatch(r"[0-9a-f]{64}", production[field])
                    for field in ("refset_sha256", "assembled_fixture_sha256", "workload_sha256")):
                raise ValueError(f"lock entry {key}: {member} is not runner-bound; runner binding required")


def verify_input(config, key, path, root=None):
    """Return authenticated bytes for a file, or verify the locked directory members."""
    if key in CODE_INPUTS:
        return None
    lock = load_lock(config, root)
    if lock is None:
        return None
    entry = lock["entries"].get(key)
    if entry is None:
        raise ValueError(f"lock entry {key}: not adopted")
    if entry["path"] != sources(config).get(key):
        raise ValueError(f"lock entry {key}: declaration differs")
    path = Path(path)
    if not path.exists():
        raise ValueError(f"lock entry {key}: missing input")
    if not entry["directory"]:
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["members"].get("."):
            raise ValueError(f"lock entry {key}: sha256 differs for .")
        return data
    actual = {member.relative_to(path).as_posix(): digest(member)
              for member in files(path, entry.get("scope")) if member.is_file()}
    if actual != entry["members"]:
        changed = sorted(key for key in actual.keys() | entry["members"].keys()
                         if actual.get(key) != entry["members"].get(key))
        raise ValueError(f"lock entry {key}: sha256 differs for {', '.join(changed)}")
    return None


def read_member(config, key, path, root=None):
    from common import declared_path
    root = root_path(root)
    lock = load_lock(config, root)
    if lock is None:
        return Path(path).read_bytes()
    entry = lock["entries"][key]
    base = declared_path(sources(config)[key], root)
    path = Path(path)
    member = path.relative_to(base).as_posix() if entry["directory"] else "."
    if member not in entry["members"]:
        raise ValueError(f"lock entry {key}: member {member} is not adopted")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["members"][member]:
        raise ValueError(f"lock entry {key}: sha256 differs for {member}")
    return data


def fixture_bytes(path, config=None, root=None):
    """Authenticate declared fixture reads; standalone diagnostic files keep their own checks."""
    import declaration
    from common import declared_path, qualification_config
    root = root_path(root)
    configs = [config] if config is not None else [declaration.load(file) for file in declaration.shipped()]
    if config is None:
        import os
        if os.environ.get("CHOPINFER_QUAL_CONFIG"):
            configs.insert(0, qualification_config())
    path = Path(path).absolute()
    unadopted = None
    for candidate in configs:
        for key, location in sources(candidate).items():
            base = declared_path(location, root)
            if base is not None and (path == base.absolute() or base.absolute() in path.parents):
                lock = load_lock(candidate, root)
                if lock is None:
                    # An unlocked config does not vouch for a file a locked
                    # one declares.
                    continue
                if path != base.absolute():
                    member = path.relative_to(base.absolute()).as_posix()
                    if member not in lock["entries"][key]["members"]:
                        unadopted = (key, member)
                        continue
                return read_member(candidate, key, path, root)
    if unadopted:
        key, member = unadopted
        raise ValueError(f"lock entry {key}: member {member} is not adopted")
    return path.read_bytes()


def verify_config(config, root=None, *, policy_only=False, optional=True):
    """Validate the config's lock and admit its numerical references
    (`gate_reference_producers`), then (unless `policy_only`) authenticate
    every declared input it pins."""
    from common import declared_path
    root = root_path(root)
    lock = load_lock(config, root)
    if lock is None:
        return
    gate_reference_producers(config, lock)
    if policy_only:
        return
    for key, location in sources(config).items():
        path = declared_path(location, root)
        if path is None or not path.exists():
            if optional:
                continue
            raise ValueError(f"lock entry {key}: missing input")
        verify_input(config, key, path, root)


def changes(previous, value):
    lines = [f"{section}: section dropped" for section in sorted(set(previous or {}) - set(value))]
    old_entries = (previous or {}).get("entries", {})
    for key in sorted(old_entries.keys() | value["entries"].keys()):
        old, new = old_entries.get(key, {}), value["entries"].get(key, {})
        old_members, new_members = old.get("members", {}), new.get("members", {})
        for member in sorted(old_members.keys() | new_members.keys()):
            if old_members.get(member) != new_members.get(member):
                lines.append(f"{key}/{member}: {old_members.get(member, 'missing')} -> "
                             f"{new_members.get(member, 'missing')}")
        if old != new and old_members == new_members:
            lines.append(f"{key}: entry metadata changed")
        for member, sha in new_members.items():
            if sha in (previous or {}).get("objects", {}) and (
                    previous["objects"][sha] != value["objects"][sha]):
                lines.append(f"{key}/{member}: producer metadata changed")
    dropped = sorted(set((previous or {}).get("objects", {})) - set(value["objects"]))
    if dropped:
        lines.append(f"objects: {len(dropped)} no longer pinned: {', '.join(sha[:12] for sha in dropped)}")
    return lines


def main():
    import declaration
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("adopt", "verify", "status"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--replace-sha256", help="Required old lock digest when refreshing changed entries")
    parser.add_argument("--dry-run", action="store_true",
                        help="adopt: report what the lock would become; write nothing")
    parser.add_argument("--variant-dir", type=Path,
                        help=f"the root of external variants (default ${declaration.VARIANTS_ENV})")
    args = parser.parse_args()
    declaration.select_variant_root(args.variant_dir)
    try:
        config = declaration.load(declaration.named(args.model))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if (args.action == "adopt" and not declaration.external(config)
            and config.get("variant") in declaration.configurations(config["model"])):
        parser.error(f"configuration {config['variant']} is pinned in the model lock's configurations; "
                     f"adopt writes only a model's, a variant file's or an external variant's lock")
    path = config_lock_path(config)
    try:
        if args.action == "adopt":
            previous = json.loads(path.read_text()) if path.exists() else None
            value = external_adoption(config) if declaration.external(config) else adoption(config, previous)
            validate_lock(value, config)
            gate_reference_producers(config, value)
            for line in changes(previous, value):
                print("changed: " + line, file=sys.stderr)
            data = encoded(value, previous)
            if args.dry_run:
                same = path.exists() and path.read_bytes() == data
                print(f"{args.model}: adopt dry run: the lock would be "
                      f"{'unchanged' if same else 'written'} ({hashlib.sha256(data).hexdigest()}); nothing written")
                return
            if path.exists() and path.read_bytes() != data and args.replace_sha256 != digest(path):
                parser.error("refuse changed lock; supply --replace-sha256 with the current lock digest")
            path.write_bytes(data)
        else:
            if not path.exists():
                parser.error("reference lock is missing")
            verify_config(config, optional=False)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"{args.model}: {args.action} verified {digest(path)}")


if __name__ == "__main__":
    main()

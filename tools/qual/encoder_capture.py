"""Capture the declared encoder workload against its pinned reference runs.

The producer runs on a diagnostic server, then encoder_parity judges the
capture while that same process is still serving. Reference production is
separate; this command only reads the admitted reference files.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

from common import ROOT, checkpoint_pin, declared_path, qualification_config
import declaration
import encoder_declarations
import tap_binding
from tools.model_sdk import diagnostic as wire
from tools.refrun import keys

SCHEMA = "chopinfer.encoder-parity-inputs.v1"
MODES = ("exact", "policy", "reference-bf16")
DTYPES = {"f32": 4, "bf16": 2, "i64": 8, "i32": 4, "u32": 4}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def member(root, name):
    path = Path(name)
    require(not path.is_absolute() and path.parts and all(p not in (".", "..") for p in path.parts),
            f"invalid reference member: {name}")
    return wire.external(root / path)


def inventory(root):
    """Closed regular-file inventory, with no links or special entries."""
    root = wire.external(Path(root).absolute())
    result = {}
    for directory, subdirs, names in os.walk(root, followlinks=False):  # check-inputs: walks outside the repository
        for name in subdirs:
            wire.external(Path(directory) / name)
        for name in names:
            path = Path(directory) / name
            result[path.relative_to(root).as_posix()] = wire.identity(path)
    require(result, f"{root}: empty reference")
    return result


def tree_digest(files):
    return keys.sha256(wire.canonical(files))


def specification_path(model):
    require(declaration.plain(model), "invalid encoder model")
    return ROOT / "models" / model / "encoder-parity-inputs.json"


def configured(model):
    return specification_path(model).is_file()


def tensor(root, value):
    if value is None:
        return None
    path = member(root, value["path"])
    identity = wire.identity(path)
    require(value["dtype"] in DTYPES and identity["bytes"] == math.prod(value["shape"]) * DTYPES[value["dtype"]],
            f"{path}: tensor size differs")
    return {"dtype": value["dtype"], "shape": value["shape"], "sha256": identity["sha256"]}


def preflight(config):
    spec = wire.decode_json(specification_path(config["model"]).read_bytes())
    require(spec.get("schema") == SCHEMA and spec.get("model") == config["model"],
            "encoder reference input schema or model differs")
    root = wire.external(declared_path(spec["root"]).absolute())
    workload_path = member(root, spec["workload"]["path"])
    require(wire.identity(workload_path)["sha256"] == spec["workload"]["sha256"], "encoder workload changed")
    workload = wire.read_json(workload_path)
    require(workload.get("model") == config["model"] and workload.get("schema") == "chopinfer.refrun.workload.v1",
            "encoder workload model or schema differs")
    sequences = workload["sequences"]
    names = [s["name"] for s in sequences]
    require(names and len(set(names)) == len(names), "empty or duplicate encoder sequences")
    media, payloads = {}, {}
    for sequence in sequences:
        require(sequence.get("media"), "encoder sequence has no media")
        media[sequence["name"]] = []
        for item in sequence["media"]:
            value = {"kind": item["kind"], "processor": item["processor"]}
            for key in ("tensor", "grid"):
                value[key] = tensor(workload_path.parent, item.get(key))
                if item.get(key):
                    name = item[key]["path"]
                    payloads[name] = wire.identity(member(workload_path.parent, name))
            media[sequence["name"]].append(value)
    require(tree_digest(payloads) == spec["workload_files_sha256"], "encoder workload payloads changed")
    requests = []
    require([r["name"] for r in spec["requests"]] == names, "encoder request coverage differs from workload")
    for request in spec["requests"]:
        path = member(root, request["path"])
        require(wire.identity(path)["sha256"] == request["sha256"], "encoder request bytes changed")
        body = wire.read_json(path)
        require(body.get("max_tokens") == 1 and body.get("temperature") == 0, "encoder request budget changed")
        requests.append((request["name"], "/v1/chat/completions", body))
    encoders = encoder_declarations.for_model(config["model"])
    # Encoders of media served but not qualified in-tree need no reference;
    # the subject records each with its declared reason.
    withdrawn = {d["path"].relative_to(ROOT).as_posix(): reason
                 for d, reason in encoder_declarations.unqualified(encoders, config.get("capabilities"))}
    declarations = {d["path"].relative_to(ROOT).as_posix(): d for d in encoders}
    declarations = {path: d for path, d in declarations.items() if path not in withdrawn}
    references, bindings, seen = [], {}, set()
    for reference in spec["references"]:
        path = member(root, reference["path"])
        files = inventory(path)
        require(tree_digest(files) == reference["files_sha256"], f"{path}: reference files changed")
        run = wire.read_json(path / "run.json")
        declared = run["declaration"]
        require(declared["path"] in declarations and declared["path"] not in seen, "encoder declaration coverage differs")
        seen.add(declared["path"])
        encoder = declarations[declared["path"]]
        require(hashlib.sha256((ROOT / declared["path"]).read_bytes()).hexdigest() == declared["sha256"], "encoder declaration changed")
        require(run["git_head"] == reference["source"], "encoder reference source differs")
        require(set(run.get("unchecked_pins", [])) <= set(spec.get("extra_checkpoint_files", {})),
                "unadmitted encoder checkpoint inputs")
        expected = [s["name"] for s in sequences if any(m["kind"] in encoder["kinds"] for m in s["media"])]
        require(expected, "declared encoder has no workload items")
        for mode in MODES:
            require(run["modes"][mode]["repeat_bitwise_equal"] is True, "encoder repeat was not bitwise equal")
            value = wire.read_json(path / f"{mode}.json")
            require(value.get("schema") == "chopinfer.refrun.encoder-rows.v1", "encoder mode schema differs")
            document = value["mode"]
            require(document["name"] == mode and document["arithmetic"] == {
                "exact": "f64", "policy": "pairs" if encoder["precision"] == "f32-pairs.v1" else "legacy",
                "reference-bf16": "bf16"}[mode], "encoder reference arithmetic differs")
            require(document["precision"] == (encoder["precision"] if mode == "policy" else None),
                    "encoder reference precision differs")
            require([entry["name"] for entry in value["entries"]] == expected, "encoder mode sequence coverage differs")
            for entry in value["entries"]:
                key = entry["key"]
                require(keys.key_sha256(key) == entry["key_sha256"], "encoder key digest differs")
                require(key["schema"] == keys.ENCODER_SCHEMA and key["model"] == config["model"]
                        and key["declaration"] == declared, "encoder key identity differs")
                require(key["checkpoint"]["source_lock_sha256"] == checkpoint_pin(config)["source_lock_sha256"],
                        "encoder checkpoint lock differs")
                require(key["mode"] == {"name": mode, "policy_sha256": keys.sha256(wire.canonical(document))},
                        "encoder mode key differs")
                require(key["media"] == [m for m in media[entry["name"]] if m["kind"] in encoder["kinds"]],
                        "encoder mode scored different media")
                for name, sha in key["code"]["files"].items():
                    source = subprocess.check_output(["git", "show", f"{reference['source']}:{name}"], cwd=ROOT)
                    require(hashlib.sha256(source).hexdigest() == sha, "encoder producer source bytes differ")
        references.append(path)
        bindings[reference["path"]] = reference["files_sha256"]
    require(seen == set(declarations), "missing declared encoder reference")
    # The extra config was pinned by the original preprocessing producer,
    # although the historical encoder checkpoint lock did not cover it.
    for name, sha in spec.get("extra_checkpoint_files", {}).items():
        metadata = tomllib.loads((ROOT / "models" / config["model"] / "model.toml").read_text())
        checkpoint = Path(os.environ[f"CHOPINFER_{metadata['checkpoint_env']}_CHECKPOINT"]).resolve()
        require(wire.identity(member(checkpoint, name))["sha256"] == sha, "encoder extra checkpoint file changed")
    subject = {"schema": SCHEMA, "model": config["model"], "inputs_sha256": keys.sha256(wire.canonical(spec)),
               "references": bindings, "workload_sha256": spec["workload"]["sha256"],
               "workload_files_sha256": spec["workload_files_sha256"]}
    if withdrawn:
        subject["unqualified"] = withdrawn
    return {"workload": workload_path, "payloads": payloads, "requests": requests, "references": references,
            "subject": subject}


def assemble(run, prepared, capture, config):
    require([r["name"] for r in capture["requests"]] == [r[0] for r in prepared["requests"]],
            "encoder capture request coverage differs")
    require(capture.get("subject") == prepared["subject"], "encoder capture reference binding differs")
    references = []
    for index, path in enumerate(prepared["references"]):
        target = run / "reference" / str(index)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(path, target_is_directory=True)
        references.append(target.relative_to(run).as_posix())
    shutil.copyfile(prepared["workload"], run / "workload.json")
    for name in prepared["payloads"]:
        path = run / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with wire.opened(member(prepared["workload"].parent, name)) as (source, _), path.open("xb") as destination:
            shutil.copyfileobj(source, destination)
    value = {"schema": "chopinfer.encoder-parity-run.v1", "model": config["model"], "engine": "engine",
             "workload": "workload.json", "references": references, "source": capture["server"]["source_commit"],
             "subject": prepared["subject"]}
    wire.write_new(run / "parity.json", value)
    return value


def admit(run, manifest, config):
    prepared = preflight(config)
    require(manifest.get("subject") == prepared["subject"], "encoder parity reference binding differs")
    capture = wire.read_json(run / tap_binding.MANIFEST)
    require(capture.get("subject") == prepared["subject"], "encoder capture reference binding differs")
    require(manifest["source"] == capture["server"]["source_commit"], "encoder capture source differs")
    require([r["name"] for r in capture["requests"]] == [r[0] for r in prepared["requests"]],
            "encoder capture request coverage differs")
    require(manifest["engine"] == tap_binding.STEPS and manifest["workload"] == "workload.json",
            "encoder parity paths differ")
    require([str((run / p).resolve()) for p in manifest["references"]] == [str(p) for p in prepared["references"]],
            "encoder parity references differ")
    require(wire.identity(run / "workload.json")["sha256"] == prepared["subject"]["workload_sha256"],
            "captured encoder workload changed")
    for name, identity in prepared["payloads"].items():
        require(wire.identity(member(run, name)) == identity, "captured encoder media changed")
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "record"))
    parser.add_argument("--url")
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--taps-dir", type=Path)
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    try:
        config = qualification_config()
        prepared = preflight(config)
        if args.command == "record":
            require(args.url and args.server_log and args.taps_dir and args.run, "record needs live capture arguments")
            captured = tap_binding.capture(args.url, args.server_log, args.taps_dir, prepared["requests"], args.run,
                                           config, subject=prepared["subject"])
            require(preflight(config)["subject"] == prepared["subject"], "encoder inputs changed during capture")
            assemble(args.run, prepared, captured, config)
        print(json.dumps({"subject": prepared["subject"], "requests": [r[0] for r in prepared["requests"]]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(f"encoder capture refused: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

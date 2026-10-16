"""Qwen3.8's ModelOpt mixed exports as reference-runner and gate inputs.

    python3 -B models/qwen38/reference_exports.py lock EXPORT --checkpoint DIR
    python3 -B models/qwen38/reference_exports.py pin EXPORT --checkpoint DIR
    python3 -B models/qwen38/reference_exports.py skeleton EXPORT --checkpoint DIR --output S.json
    python3 -B models/qwen38/reference_exports.py adopt EXPORT --checkpoint DIR --emitted DIR

EXPORT is a directory of models/qwen38/exports/ whose `source.json` names
the repository and revision the export was published at; adding one needs no
code. The export is gated as the checkpoint variant EXPORT of qwen38
(`qual.EXPORT.json`, tools/qual/README.md#model-declarations).

- `lock` writes the export's source lock, `models/qwen38/exports/EXPORT/
  checkpoint-files.json`: byte length and SHA-256 of `config.json`, the shard
  index, every shard it names, every other safetensors file a direct load
  opens (so `chopinfer verify --manifest` covers the served weights) and the
  tokenizer and chat-template files. Digests go through the runner's stamp cache
  (tools/refrun/stamps.py), so a later run verifies without rehashing. It
  refuses to replace a lock.
- `pin` prints the checkpoint pin a gate holds a server and a reference to:
  repository, revision, served model name, lock, its digest and the
  checkpoint identity `serve --model` reports (the SHA-256 of every
  safetensors file's name, length and header, as the engine computes it).
- `skeleton` writes a teacher-reference skeleton over the FP8 reference's
  frozen teacher workload (`fixtures/qwen38/teacher-reference.workload.json`)
  with the export's pin as provenance, for the runner's `emit` with
  `--modes f32=exact,bf16,f64` over models/qwen38/reference_mixed.py runs.
- `adopt` publishes an emitted teacher reference in this tree as gate
  input: the fixture and its canonical workload under
  `fixtures/qwen38/exports/EXPORT/`, and the export's variant declaration
  `models/qwen38/qual.EXPORT.json` (model `qwen38`, variant `EXPORT`) with
  its reference lock `models/qwen38/qual-references.EXPORT.lock.json`. The
  declaration keeps only the teacher, spec_verify and verify_exact checks
  of models/qwen38/qual.json, pinned to the export. The lock adopts the
  reference by its runner binding and repeats models/qwen38's records of
  every shared input. `adopt` refuses before writing if any of these exists,
  and verifies them as the gates read them. Select the gate as
  `release_gate.py --model qwen38.EXPORT` (`chopinfer eval ... --variant
  EXPORT`).

The served model name is the export's repository ID: serve it with
`--served-model-name REPOSITORY`.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/qual"))
import common  # noqa: E402
import declaration  # noqa: E402
from tools.refrun import keys, stamps  # noqa: E402

SOURCE = "source.json"
METADATA = ("chat_template.jinja", "config.json", "merges.txt", "model.safetensors.index.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json")
INDEX = "model.safetensors.index.json"
GATE = ROOT / "models/qwen38/qual.json"
WORKLOAD = "fixtures/qwen38/teacher-reference.workload.json"
TEACHER_SCHEMA = "chopinfer.qwen38.teacher-reference.v2"
# What an export declaration keeps of models/qwen38/qual.json.
CHECKS = ("teacher", "spec_verify", "verify_exact")
INPUTS = ("fixture_root", "frontend", "gsm8k", "spec_teacher")
QUALIFICATION = ("teacher_reference_kind", "teacher_receipt", "teacher_capture_schema", "verify_buckets",
                 "chars_per_token")
VARIANTS_ORDER = ["exact", "bf16", "f64"]
CODE = "HF transformers 5.16.1 Qwen4ExpForConditionalGeneration (models/qwen38/reference_mixed.py)"
VARIANTS = {
    "exact": f"{CODE} in F32 on the GPU runner (tools/refrun), quantized weights as the exact product of codes "
             "and scales rounded once to F32 (dequant rounded), F32 norm weights as stored: the gate variant",
    "bf16": f"{CODE} in BF16, quantized weights' F32 product rounded to BF16 (dequant hf-bf16), F32 norm "
            "weights kept F32, F32 log-softmax: spread variant",
    "f64": f"{CODE} in f64 with HF's own F32 steps (each run's layout.upstream_f32), dequant exact: spread "
           "variant",
}
POSITIONS = ("entry i scores the token at score_from + i from the logits of the row before it; top1 is the "
             "highest logit (ties to the lower ID), margin its logit minus the second's, and the top-k columns "
             "are ranked by logit")


def exports(root=None):
    """Every export with a source declaration,
    `models/qwen38/exports/EXPORT/source.json` (the repository and revision
    it was exported to), by name."""
    found = {}
    for path in sorted(Path(root or ROOT).glob(f"models/qwen38/exports/*/{SOURCE}")):
        source = json.loads(path.read_text())
        if (not isinstance(source, dict) or set(source) != {"repository", "revision"}
                or not isinstance(source["repository"], str) or not isinstance(source["revision"], str)
                or not re.fullmatch(r"[0-9a-f]{40}", source["revision"])):
            raise ValueError(f"{path}: expected the export's repository and its 40-hex revision")
        found[path.parent.name] = source
    return found


def lock_path(export):
    return f"models/qwen38/exports/{export}/checkpoint-files.json"


def safetensors_names(root):
    """The safetensors files a direct load opens (model-compiler-sdk's
    `safetensors_names`): those of `root` and of its non-hidden immediate
    subdirectories, named `dir/file`, whether or not the index names them."""
    names = []
    for entry in os.scandir(root):
        if entry.is_dir():
            if not entry.name.startswith("."):
                names += [f"{entry.name}/{inner.name}" for inner in os.scandir(entry.path) if not inner.is_dir()]
        else:
            names.append(entry.name)
    return sorted(name for name in names if name.endswith(".safetensors"))


def identity(root):
    """The engine's checkpoint identity (runtime-weights `header_identity`
    over every file of `safetensors_names`), by name."""
    root, files = Path(root), []
    for name in safetensors_names(root):
        path = root / name
        with path.open("rb") as stream:
            (length,) = struct.unpack("<Q", stream.read(8))
            header = stream.read(length)
        if len(header) != length:
            raise ValueError(f"{name}: truncated safetensors header")
        files.append((name.encode(), path.stat().st_size, header))
    if not files:
        raise ValueError(f"{root} holds no safetensors files")
    digest = hashlib.sha256(b"chopinfer.checkpoint-headers.v1")
    for name, size, header in sorted(files):
        digest.update(struct.pack("<Q", len(name)) + name + struct.pack("<Q", size)
                      + struct.pack("<Q", len(header)) + header)
    return digest.hexdigest()


def write_lock(export, checkpoint, cache=stamps.default_cache()):
    root = Path(checkpoint)
    for required in ("config.json", INDEX):
        if not (root / required).is_file():
            raise ValueError(f"{root} has no {required}")
    # The index's shards and every other safetensors file the engine opens
    # (an export may keep superseded shards beside the index).
    shards = set(json.loads((root / INDEX).read_text())["weight_map"].values()) | set(safetensors_names(root))
    files = sorted({*(name for name in METADATA if (root / name).exists()), *shards})
    with ThreadPoolExecutor(stamps.THREADS) as pool:
        digests = pool.map(lambda name: stamps.digest(root / name, cache), files)
        lock = {name: {"byte_length": stamp["size"], "sha256": sha256} for name, (sha256, stamp) in zip(files, digests)}
    path = ROOT / lock_path(export)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(lock, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def pin(export, checkpoint, cache=stamps.default_cache()):
    """The export's checkpoint pin, after verifying the checkpoint's config
    and index against its lock."""
    lock = keys.read_lock(ROOT / lock_path(export))
    stamps.verify(checkpoint, lock, ["config.json", INDEX], cache)
    source = exports()[export]
    return {"lock": lock_path(export), "repository": source["repository"], "revision": source["revision"],
            "served_model_name": source["repository"],
            "source_lock_sha256": keys.checkpoint(lock, [])["source_lock_sha256"],
            "checkpoint_identity": identity(checkpoint)}


def skeleton(export, checkpoint):
    """A teacher-reference skeleton of the frozen teacher texts, provenance
    the export's pin."""
    value = pin(export, checkpoint)
    sequences = json.loads((ROOT / WORKLOAD).read_text())["sequences"]
    return {"schema": TEACHER_SCHEMA,
            "provenance": {"source": "tools/refrun emit of models/qwen38/reference_mixed.py runs "
                                     "(models/qwen38/reference_exports.py)",
                           "checkpoint": {key: value[key] for key in declaration.CHECKPOINT_KEYS},
                           "workload": Path(WORKLOAD).name, "gate_variant": "exact", "variants": VARIANTS,
                           "positions": POSITIONS},
            "texts": [{"name": s["name"], "category": s["category"], "prompt_token_ids": s["token_ids"],
                       "score_from": s["score_from"]} for s in sequences]}


def replaced(value, old, new):
    """`value` with every string equal to a key of `old` replaced."""
    if isinstance(value, dict):
        return {key: replaced(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [replaced(item, old, new) for item in value]
    return new[old.index(value)] if isinstance(value, str) and value in old else value


def fixture_path(export):
    return f"fixtures/qwen38/exports/{export}/teacher-reference.json"


def declaration_path(export):
    """The export's variant declaration, relative to the root."""
    return declaration.path("qwen38", "", export).as_posix()


def reference_lock_path(export):
    """The export's variant reference lock, relative to the root
    (refs.lock_path)."""
    return f"models/qwen38/qual-references.{export}.lock.json"


def derived(base, export, value, sha256, workload_sha256):
    """The export's variant declaration from `base` (models/qwen38/qual.json): the
    teacher, spec_verify and verify_exact checks with their cohorts, change
    classes and inputs only (the FP8-only references and checks are left
    out), the export's checkpoint pin `value`, its teacher reference
    (`sha256`) as `references.teacher` and the exact input, every rule that
    held the FP8 checkpoint identity or teacher digest holding the export's,
    both spec_verify arms and the comparison bound to the export's
    checkpoint, a GSM8K floor and its own receipts."""
    if workload_sha256 != base["references"]["teacher"]["workload_sha256"]:
        raise ValueError(f"the {export} reference does not score the frozen teacher workload")
    old = [base["qualification"]["checkpoint"]["checkpoint_identity"], base["references"]["teacher"]["sha256"]]
    new = [value["checkpoint_identity"], sha256]
    checks = {name: replaced(copy.deepcopy(base["checks"][name]), old, new) for name in CHECKS}
    bound = {"path": ["server", "checkpoint", "checkpoint_identity"], "op": "eq",
             "value": value["checkpoint_identity"], "source": "tools/qual/README.md#checkpoint-binding"}
    for arm in ("spec_ar", "spec_candidate"):
        checks["spec_verify"]["input_rules"][arm].append(bound)
    checks["spec_verify"]["rules"].append({**bound, "path": ["bound", "checkpoint_identity"]})
    reference = fixture_path(export)
    preview = base["preview"]
    return {
        "schema": base["schema"], "model": "qwen38", "variant": export,
        "cohorts": {name: {"checks": [check for check in cohort["checks"] if check in CHECKS]}
                    for name, cohort in base["cohorts"].items()},
        "change_classes": {change: [check for check in names if check in CHECKS]
                           for change, names in base["change_classes"].items()},
        "checks": checks,
        "references": {"teacher": {"path": reference, "sha256": sha256, "schema": TEACHER_SCHEMA,
                                   "workload_sha256": workload_sha256, "variants": VARIANTS_ORDER}},
        "inputs": {**{key: base["inputs"][key] for key in INPUTS},
                   "exact": reference, "teacher_reference": reference},
        "qualification": {**{key: base["qualification"][key] for key in QUALIFICATION}, "checkpoint": dict(value)},
        "receipts": {"env": "CHOPINFER_DATA", "path": f"qual/qwen38-{export}"},
        "limitations": [f"The {value['repository']} export at {value['revision'][:9]}: teacher, "
                        "spec_verify, verify_exact and the GSM8K screen only, text only; its vision encoders, "
                        "long-context, frontend and service checks are not gated by this declaration."],
        "preview": {"profile": "tp2-mtp", "speculation": preview["speculation"],
                    "spec_greedy": preview["spec_greedy"], "numerical_reference": preview["numerical_reference"],
                    "gsm8k_floor": preview["gsm8k_floor"]},
    }


def adopt(export, checkpoint, emitted):
    """Publishes the emitted teacher reference of `export` (the fixture and
    its canonical workload in the directory `emitted`) with the export's
    variant declaration and reference lock, in this tree:
    fixtures/qwen38/exports/EXPORT/, models/qwen38/qual.EXPORT.json and
    models/qwen38/qual-references.EXPORT.lock.json.
    Refuses before writing anything if any of them exists. The lock adopts
    the reference (runner binding) and its workload (runner workload) and
    keeps models/qwen38's lock records of every shared input unchanged:
    entries, digests, scopes and object records. Everything is verified as the
    gates read it; on any failure the files written here are removed."""
    sys.path.insert(0, str(ROOT / "tools/qual"))
    import refs
    emitted = Path(emitted)
    source = emitted / "teacher-reference.json"
    targets = [ROOT / fixture_path(export), ROOT / fixture_path(export).replace(".json", ".workload.json"),
               ROOT / declaration_path(export), ROOT / reference_lock_path(export)]
    present = [str(target.relative_to(ROOT)) for target in targets if target.exists()]
    if present:
        raise ValueError(f"refusing to replace published {export} evidence: {present}")
    data = source.read_bytes()
    fixture = json.loads(data)
    if fixture.get("schema") != TEACHER_SCHEMA or "binding" not in fixture:
        raise ValueError(f"{source} is not a bound {TEACHER_SCHEMA} reference")
    base = json.loads(GATE.read_text())
    value = derived(base, export, pin(export, checkpoint), hashlib.sha256(data).hexdigest(),
                    common.workload_sha256(fixture["texts"]))
    written = []
    try:
        for target, payload in zip(targets, (data, (emitted / "teacher-reference.workload.json").read_bytes(),
                                             (json.dumps(value, indent=1) + "\n").encode())):
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                written.append(target)
                stream.write(payload)
        config = declaration.load(ROOT / declaration_path(export))
        lock = export_lock(config, json.loads(refs.lock_path("qwen38").read_text()))
        with targets[3].open("x") as stream:
            written.append(targets[3])
            stream.write(json.dumps(lock, indent=2, sort_keys=True) + "\n")
        verify(config)
    except BaseException:
        for target in reversed(written):
            target.unlink(missing_ok=True)
        raise
    return [str(target.relative_to(ROOT)) for target in written]


def export_lock(config, base_lock, root=None):
    """The export's variant reference lock: refs.snapshot with the model's lock as
    the previous one, whose records of every shared input it must repeat."""
    import refs
    lock = refs.snapshot(config, root, previous=base_lock)
    refs.validate_lock(lock, config, root)
    refs.gate_reference_producers(config, lock)
    for key, entry in lock["entries"].items():
        if key in base_lock["entries"] and entry["path"] == base_lock["entries"][key]["path"]:
            if entry != base_lock["entries"][key] or any(
                    lock["objects"][sha] != base_lock["objects"][sha] for sha in entry.get("members", {}).values()):
                raise ValueError(f"lock entry {key}: differs from the model's adopted record")
    return lock


def verify(config, root=None):
    """The export's inputs as the gates read them: every lock entry, the
    pinned reference with its bound runner binding, and the checkpoint pin."""
    import refs
    refs.verify_config(config, root, optional=False)
    common.checkpoint_pin(config)
    reference = common.reference_root(root) / config["qualification"]["references"]["teacher"]["path"]
    value = common.pinned_reference("teacher", reference, config)
    if common.verified_binding("teacher", value, config)["status"] != "bound":
        raise ValueError(f"{reference}: the runner binding does not verify")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("lock", "pin", "skeleton", "adopt"))
    parser.add_argument("export", choices=sorted(exports()))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="skeleton: the file to create")
    parser.add_argument("--emitted", type=Path, help="adopt: the directory emit wrote the teacher reference to")
    args = parser.parse_args()
    if args.command == "lock":
        print(write_lock(args.export, args.checkpoint))
    elif args.command == "pin":
        print(json.dumps(pin(args.export, args.checkpoint), indent=1))
    elif args.command == "skeleton":
        if args.output is None:
            parser.error("skeleton needs --output")
        with args.output.open("x") as stream:
            json.dump(skeleton(args.export, args.checkpoint), stream, indent=1)
            stream.write("\n")
        print(args.output)
    else:
        if args.emitted is None:
            parser.error("adopt needs --emitted")
        print("\n".join(adopt(args.export, args.checkpoint, args.emitted)))


if __name__ == "__main__":
    main()

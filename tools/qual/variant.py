"""Qualify another checkpoint of a supported model from outside the tree.

    python3 -m tools.qual variant lock --model M --variant V --dir DIR --checkpoint C
    python3 -m tools.qual variant pin --model M --variant V --dir DIR --checkpoint C --repository R --revision REV
        [--served-model-name N]
    python3 -m tools.qual variant skeleton --model M --variant V --dir DIR --checkpoint C --repository R
        --revision REV [--served-model-name N] --reference NAME... --output OUT
    python3 -m tools.qual variant adopt --model M --variant V --dir DIR --checkpoint C --repository R
        --revision REV [--served-model-name N] --reference NAME... --emitted OUT --check ID...
        [--quick ID...] [--keep-reference NAME...] [--from-default KEY...] [--profile-name P]
        [--checkpoint-env KEY]

An external variant (tools/qual/README.md#external-variants) is the overlay
`DIR/M/V.json` of the model's `models/M/qual.json`, with everything it owns
beside it: its checkpoint lock, its references and their workloads, and its
reference lock. Nothing here writes into the tree.

- `lock` writes the checkpoint lock `DIR/M/V.checkpoint-files.json`: byte
  length and SHA-256 of the checkpoint's config, shard index, tokenizer and
  template files, every shard the index names and every other safetensors
  file a direct load opens (so `chopinfer verify --manifest` covers the
  served weights). Digests go through the runner's stamp cache. It refuses
  to replace a lock.
- `pin` prints the checkpoint pin a gate holds a server and a reference to:
  repository, revision, served model name (default the repository), the
  lock, its digest and the checkpoint identity `serve --model` reports.
- `skeleton` writes `OUT/NAME-reference.skeleton.json` for each base
  reference NAME the model's qual.json pins: its schema, workload and text
  layout, the scores removed and the variant's pin as its provenance, for the
  runner's `emit --lock-root DIR --lock DIR/M/V.checkpoint-files.json`
  over the base reference's workload. It prints what emit takes.
- `adopt` takes each emitted `OUT/NAME-reference.json` (with the workload
  emit wrote beside it) into `DIR/M/` as `V.NAME-reference.json`, writes the
  overlay `V.json` (the checks given, the references, the inputs that read
  a base reference redirected to the variant's, the pin, and the base
  capabilities narrowed to what those checks cover) and its lock
  `V.lock.json`, then verifies all of it as the gates read it. The lock
  repeats the model lock's record of every input the variant shares with
  the model, and pins what the variant owns. It refuses before writing if
  any target exists, and removes what it wrote on a failure.

Select the variant with `--variant-dir DIR` (or `CHOPINFER_QUAL_VARIANTS`)
and `--model M.V`; the server serves it as `--served-model-name N`.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
import declaration  # noqa: E402

INDEX = "model.safetensors.index.json"
# Checkpoint files a lock covers when present, besides the weights: what the
# frontend, the tokenizer and the loaders read.
METADATA = ("chat_template.jinja", "config.json", "generation_config.json", "hf_quant_config.json", INDEX,
            "merges.txt", "preprocessor_config.json", "processor_config.json", "special_tokens_map.json",
            "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "video_preprocessor_config.json",
            "vocab.json")
# The capability each module's hard check covers (eval declaration's obligations).
COVERS = {"teacher": "text", "frontend_goldens": "frontend", "api": "api", "structured": "structured",
          "prefix_exact": "prefix_cache", "preempt_compare": "preemption", "needle": "long_context",
          "soak": "stability"}


def lock_name(variant):
    """The checkpoint lock's name beside the overlay."""
    return f"{variant}.checkpoint-files.json"


def reference_name(variant, name):
    return f"{variant}.{name}-reference.json"


def directory(args):
    """`DIR/M`, where the variant's files live."""
    if not declaration.plain(args.model) or not declaration.plain(args.variant):
        raise ValueError("--model and --variant must be plain names")
    if not declaration.path(args.model).is_file():
        raise ValueError(f"{args.model} has no qual.json in this tree")
    root = Path(args.dir).resolve()
    if root == ROOT or ROOT in root.parents:
        raise ValueError(f"{root} is in the tree; an external variant lives outside it")
    return root / args.model


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


def write_lock(target, checkpoint, cache=None):
    """The checkpoint lock of `checkpoint`, written to `target` (refused when
    it exists)."""
    from tools.refrun import stamps
    root = Path(checkpoint)
    if not (root / "config.json").is_file():
        raise ValueError(f"{root} has no config.json")
    shards = set(safetensors_names(root))
    if (root / INDEX).is_file():
        shards |= set(json.loads((root / INDEX).read_text())["weight_map"].values())
    files = sorted({*(name for name in METADATA if (root / name).is_file()), *shards})
    cache = stamps.default_cache() if cache is None else cache
    with ThreadPoolExecutor(stamps.THREADS) as pool:
        digests = pool.map(lambda name: stamps.digest(root / name, cache), files)
        lock = {name: {"byte_length": stamp["size"], "sha256": sha256} for name, (sha256, stamp) in zip(files, digests)}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(lock, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return target


def pin(args, cache=None):
    """The variant's checkpoint pin, after verifying the checkpoint's config
    and index against its lock. Its lock path is the overlay's own (beside
    it); the expanded declaration names it under the variant root."""
    from tools.refrun import keys, stamps
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision or ""):
        raise ValueError("--revision must be the 40-hex revision the checkpoint was published at")
    if not args.repository:
        raise ValueError("--repository is required")
    lock = keys.read_lock(directory(args) / lock_name(args.variant))
    checked = [name for name in ("config.json", INDEX) if name in lock]
    stamps.verify(args.checkpoint, lock, checked, stamps.default_cache() if cache is None else cache)
    return {"lock": lock_name(args.variant), "repository": args.repository, "revision": args.revision,
            "served_model_name": args.served_model_name or args.repository,
            "source_lock_sha256": keys.checkpoint(lock, [])["source_lock_sha256"],
            "checkpoint_identity": identity(args.checkpoint)}


def base_reference(model, name):
    """The base declaration's pin of reference `name` and its bytes, as the
    gates read them (the model's own lock authenticates them)."""
    value = json.loads(declaration.path(model).read_text())
    spec = value.get("references", {}).get(name)
    if spec is None:
        raise ValueError(f"{model}'s qual.json pins no reference {name}")
    path = common.declared_path(spec["path"])
    if path is None or not path.is_file():
        raise ValueError(f"the {name} reference of {model} is not available here")
    return spec, path


def skeleton(model, name, value, variants=None):
    """A skeleton of the base reference `name` for the pin `value`: its
    schema, provenance layout and texts without their scores (the pinned
    variants), the variant's checkpoint as its provenance, and of the base's
    variants those `variants` names (default all)."""
    spec, path = base_reference(model, name)
    fixture = json.loads(path.read_bytes())
    scores = set(spec.get("variants", []))
    provenance = {key: item for key, item in fixture.get("provenance", {}).items()}
    provenance["checkpoint"] = {key: value[key] for key in declaration.CHECKPOINT_KEYS}
    if variants is not None and "variants" in provenance:
        unknown = sorted(set(variants) - set(provenance["variants"]))
        if unknown:
            raise ValueError(f"the {name} reference has no variant {', '.join(unknown)}")
        listed = provenance["variants"]
        provenance["variants"] = ({key: listed[key] for key in listed if key in variants}
                                  if isinstance(listed, dict) else [key for key in listed if key in variants])
    workload = Path(common.declared_path(spec["workload"])) if "workload" in spec else common.reference_workload(path)
    return {"schema": fixture["schema"], "provenance": provenance,
            "texts": [{key: item for key, item in text.items() if key not in scores} for text in fixture["texts"]]}, \
        workload


def capabilities(base, checks):
    """The base capabilities narrowed to what `checks` (the variant's IDs)
    cover; `eval declaration` judges the coverage."""
    claimed = {key: value for key, value in base["capabilities"].items() if key != "unqualified_media"}
    modules = {base["checks"][check]["module"] for check in checks
               if "informational_reason" not in base["checks"][check]}
    for module, capability in COVERS.items():
        if claimed.get(capability) is True and module not in modules:
            claimed[capability] = False
    if "multimodal" not in modules:
        claimed["media"] = []
    if not {"spec_verify", "verify_exact"} <= modules:
        claimed["speculation"] = "none"
    if "kv_offload" not in modules:
        claimed["kv_offload"] = False
    return claimed


def overlay(args, value, references):
    """The overlay `adopt` writes: the variant's checks, its references (by
    name: (pin, path beside the overlay)), the base inputs that read a base
    reference's file redirected to the variant's, and its pin."""
    base = json.loads(declaration.path(args.model).read_text())
    unknown = [check for check in [*args.check, *args.quick] if check not in base["checks"]]
    if unknown:
        raise ValueError(f"{args.model} declares no check {', '.join(unknown)}")
    quick = args.quick or [check for check in base["cohorts"]["quick"]["checks"] if check in args.check]
    inputs = {}
    for name, (spec, file) in references.items():
        old = base["references"][name]["path"]
        for key, location in base["inputs"].items():
            if location == old:
                inputs[key] = file
    key = args.checkpoint_env or re.sub(r"[^A-Z0-9]", "_", f"{args.model}_{args.variant}".upper())
    result = {"schema": "chopinfer-qual-variant/v1", "model": args.model, "checkpoint": key,
              "checks": list(args.check), "quick": quick}
    if args.profile_name:
        result["serving"] = {"profile_name": args.profile_name}
    if args.from_default:
        result["from_default"] = list(args.from_default)
    result["references"] = {name: spec for name, (spec, _) in references.items()}
    if args.keep_reference:
        result["keep_references"] = list(args.keep_reference)
    if inputs:
        result["inputs"] = inputs
    result["qualification"] = {"checkpoint": value}
    if "capabilities" in base:
        result["capabilities"] = capabilities(base, args.check)
        result["media"] = bool(result["capabilities"]["media"])
    return result


def verify(config):
    """The variant as the gates read it: its lock, every input it owns (and
    each shared one present here), the checkpoint pin, and each pinned
    reference with its runner binding, held to the variant's pin."""
    import refs
    lock = refs.load_lock(config)
    refs.gate_reference_producers(config, lock)
    for key, location in refs.sources(config).items():
        path = common.declared_path(location)
        if path is not None and path.exists():
            refs.verify_input(config, key, path)
        elif isinstance(location, dict) and location.get("env") == declaration.VARIANTS_ENV:
            raise ValueError(f"lock entry {key}: missing input")
    common.checkpoint_pin(config)
    for kind in config["qualification"]["references"]:
        path = common.reference_path(kind, config)
        value = common.pinned_reference(kind, path, config)
        if common.verified_binding(kind, value, config)["status"] != "bound":
            raise ValueError(f"{path}: the runner binding does not verify")


def adopt(args):
    """Writes the emitted references, the overlay and its lock into `DIR/M/`
    and verifies them, or writes nothing."""
    import refs
    target = directory(args)
    if not args.reference:
        raise ValueError("adopt needs at least one --reference")
    if not args.check:
        raise ValueError("adopt needs the checks the variant runs (--check)")
    value = pin(args)
    emitted = Path(args.emitted)
    references, payloads = {}, []
    for name in args.reference:
        spec, _ = base_reference(args.model, name)
        source = emitted / f"{name}-reference.json"
        data = source.read_bytes()
        fixture = json.loads(data)
        if fixture.get("schema") != spec["schema"] or "binding" not in fixture:
            raise ValueError(f"{source} is not a bound {spec['schema']} reference")
        workload = common.workload_sha256(fixture["texts"])
        if "workload_sha256" in spec and workload != spec["workload_sha256"]:
            raise ValueError(f"{source} does not score the {name} reference's workload")
        variants = fixture.get("provenance", {}).get("variants")
        file = reference_name(args.variant, name)
        pinned = {"path": file, "sha256": hashlib.sha256(data).hexdigest(), "schema": spec["schema"],
                  "workload_sha256": workload,
                  "variants": list(variants) if variants is not None else list(spec.get("variants", []))}
        if "checkpoint_keys" in spec:
            pinned["checkpoint_keys"] = list(spec["checkpoint_keys"])
        references[name] = (pinned, file)
        payloads += [(target / file, data),
                     (target / common.reference_workload(file).name,
                      common.reference_workload(source).read_bytes())]
    value_overlay = overlay(args, value, references)
    payloads.append((target / f"{args.variant}.json", (json.dumps(value_overlay, indent=1) + "\n").encode()))
    lock_target = target / f"{args.variant}.lock.json"
    present = [str(path) for path, _ in payloads if path.exists()] + ([str(lock_target)] if lock_target.exists() else [])
    if present:
        raise ValueError(f"refusing to replace {args.model}.{args.variant} evidence: {present}")
    written = []
    previous = os.environ.get(declaration.VARIANTS_ENV)
    declaration.select_variant_root(args.dir)
    try:
        for path, data in payloads:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                written.append(path)
                stream.write(data)
        config = declaration.load(declaration.named(f"{args.model}.{args.variant}"))
        lock = refs.external_adoption(config)
        refs.validate_lock(lock, config)
        refs.gate_reference_producers(config, lock)
        with lock_target.open("x") as stream:
            written.append(lock_target)
            stream.write(json.dumps(lock, indent=2, sort_keys=True) + "\n")
        verify(config)
    except BaseException:
        for path in reversed(written):
            path.unlink(missing_ok=True)
        raise
    finally:
        if previous is None:
            os.environ.pop(declaration.VARIANTS_ENV, None)
        else:
            os.environ[declaration.VARIANTS_ENV] = previous
    return [str(path) for path in written]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("lock", "pin", "skeleton", "adopt"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--dir", required=True, type=Path, help="the variant root, outside the tree")
    parser.add_argument("--checkpoint", required=True, type=Path, help="the checkpoint directory")
    parser.add_argument("--repository", help="the repository the checkpoint was published at")
    parser.add_argument("--revision", help="its 40-hex revision")
    parser.add_argument("--served-model-name", help="the name the server serves (default the repository)")
    parser.add_argument("--reference", action="append", default=[],
                        help="a reference of the model's qual.json the variant re-runs (skeleton, adopt)")
    parser.add_argument("--output", type=Path, help="skeleton: the directory to write skeletons to")
    parser.add_argument("--variants", help="skeleton: the reference variants to keep, comma separated "
                                           "(default the base reference's)")
    parser.add_argument("--emitted", type=Path, help="adopt: the directory emit wrote NAME-reference.json to")
    parser.add_argument("--check", action="append", default=[], help="adopt: a declared check the variant runs")
    parser.add_argument("--quick", action="append", default=[],
                        help="adopt: a check of its quick cohort (default those of the base's quick cohort)")
    parser.add_argument("--keep-reference", action="append", default=[],
                        help="adopt: a base reference kept (it must name the variant's checkpoint)")
    parser.add_argument("--from-default", action="append", default=[],
                        help="adopt: an input the default configuration's run records")
    parser.add_argument("--profile-name", help="adopt: the served profile name")
    parser.add_argument("--checkpoint-env", help="adopt: KEY of CHOPINFER_KEY_CHECKPOINT (default MODEL_VARIANT)")
    args = parser.parse_args(argv)
    try:
        if args.command == "lock":
            print(write_lock(directory(args) / lock_name(args.variant), args.checkpoint))
        elif args.command == "pin":
            print(json.dumps(pin(args), indent=1))
        elif args.command == "skeleton":
            if args.output is None or not args.reference:
                parser.error("skeleton needs --output and at least one --reference")
            value, target = pin(args), directory(args)
            rows = []
            for name in args.reference:
                document, workload = skeleton(args.model, name, value,
                                              args.variants.split(",") if args.variants else None)
                path = args.output / f"{name}-reference.skeleton.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x") as stream:
                    json.dump(document, stream, indent=1)
                    stream.write("\n")
                rows.append({"reference": name, "skeleton": str(path), "workload": str(workload),
                             "output": str(args.output / f"{name}-reference.json"),
                             "lock_root": str(target.parent), "lock": str(target / lock_name(args.variant))})
            print(json.dumps(rows, indent=1))
        else:
            if args.emitted is None:
                parser.error("adopt needs --emitted")
            print("\n".join(adopt(args)))
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.exit(main())

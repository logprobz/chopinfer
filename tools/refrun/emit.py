"""Assemble a bound gate fixture from reference runs.

    python3 -m tools.refrun emit --skeleton SKELETON.json --runs DIR [DIR ...] \\
        --workload WORKLOAD.json --lock models/<m>/checkpoint-files.json [--lock-variant V] \\
        --modes f32=exact,bf16,f64 --output fixtures/<m>/<stem>.json

The skeleton is the fixture without scores: its schema, provenance and items
(`texts` by name, `cases` by id or a `probes` mapping, as `cache.items`
reads them), each named like a workload sequence. A model's capture script
writes it; this module adds the scores, so the fixture copies the runs and
nothing else.

`--runs` are runner output directories; each `<label>.json` in them is one
mode's run, and a mode split over several directories (one set per GPU job)
is joined. `--modes` selects the run labels to bind, each as `LABEL` or
`LABEL=VARIANT` for the fixture variant holding its scores. For every bound
sequence the item gets, under each variant, the `--fields` of the forced
entry's scores (top-k columns cut to `--topk`). A generated answer becomes
the item's `answer_token_ids`, and with `--greedy-fields` its step scores go
under `greedy`.

The workload is copied to `<stem>.workload.json` beside the fixture with the
media tensor files it names, the canonical path the gates read
(tools/qual/README.md, reference cache bindings), and the fixture is bound
to that copy (`cache.bind`); the refset is written to the cache.

This module does not import torch.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

from tools.model_sdk.diagnostic import canonical, decode_json
from tools.refrun import cache, keys, run

FIELDS = ("logprob", "top1", "margin", "topk_ids", "topk_logits", "topk_logprobs")
TOPK = 5


def read_runs(directories):
    """{label: run document} of every mode run in `directories`. A label
    found in several directories is joined: one policy, and an entry listed
    twice (by name and key) must hold the same scores."""
    runs = {}
    for directory in directories:
        for path in sorted(Path(directory).glob("*.json")):
            document = decode_json(path.read_bytes())
            if not isinstance(document, dict) or document.get("schema") != run.SCHEMA:
                continue
            joined = runs.setdefault(path.stem, {"policy": document["policy"], "entries": []})
            if joined["policy"] != document["policy"]:
                raise ValueError(f"{path}: {path.stem} ran under another policy than in an earlier directory")
            for entry in document["entries"]:
                twin = next((e for e in joined["entries"]
                             if (e["name"], e["key_sha256"]) == (entry["name"], entry["key_sha256"])), None)
                if twin is None:
                    joined["entries"].append(entry)
                elif canonical(twin["scores"]) != canonical(entry["scores"]):
                    raise ValueError(f"{path}: {entry['name']} scored differently in another directory")
    return runs


def ordered(document, sequences, label):
    """The run's entries of the workload `sequences` in the order a single
    run scores them: each generated answer's greedy entry first, then every
    forced entry in workload order. A workload sequence the run lacks is
    refused; entries of other sequences are left out."""
    def policy(entry):
        return entry["key"]["sequence"]["answer"]["policy"]

    found = {}
    for entry in document["entries"]:
        slot = found.setdefault((entry["name"], policy(entry)), entry)
        if slot["key_sha256"] != entry["key_sha256"]:
            raise ValueError(f"{label}: {entry['name']} has two {policy(entry)} entries with different keys")
    greedy = [found[(name, "greedy")] for name, _ in sequences if (name, "greedy") in found]
    missing = [name for name, _ in sequences if (name, "forced") not in found]
    if missing:
        raise ValueError(f"{label}: no forced entry of {missing}")
    return {"policy": document["policy"], "entries": greedy + [found[(name, "forced")] for name, _ in sequences]}


def columns(scores, fields, topk):
    """The `fields` of one entry's scores, top-k rows cut to `topk`."""
    absent = [field for field in fields if field not in scores]
    if absent:
        raise ValueError(f"the scores have no {absent}")
    return {field: [row[:topk] for row in scores[field]] if field in cache.TOPK else scores[field]
            for field in fields}


def assemble(skeleton, runs, variants, fields=FIELDS, topk=TOPK, greedy_fields=()):
    """The skeleton with each run's scores under its variant, and generated
    answers with their optional greedy step scores."""
    value = json.loads(json.dumps(skeleton))
    stated = cache.items(value)
    for label, document in runs.items():
        variant = variants.get(label, label)
        for entry in document["entries"]:
            item = stated.get(entry["name"])
            if item is None:
                raise ValueError(f"{label}: {entry['name']} is not an item of the skeleton")
            if entry["key"]["sequence"]["answer"]["policy"] == "greedy":
                item["answer_token_ids"] = entry["scores"]["answer"]
                if greedy_fields:
                    item["greedy"] = {"answer": entry["scores"]["answer"],
                                      **columns(entry["scores"], greedy_fields, topk)}
            else:
                item[variant] = columns(entry["scores"], fields, topk)
    return value


def copy_workload(source, target):
    """Copy a workload file and the media tensor files it names (relative to
    it) to `target`. The copy loads to the same sequences."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if source != target:
        document = json.loads(source.read_text())
        for spec in document["sequences"]:
            for item in spec.get("media", []):
                for tensor in (item["tensor"], item.get("grid")):
                    if tensor is None:
                        continue
                    payload, copy = source.parent / tensor["path"], target.parent / tensor["path"]
                    # A workload beside its fixture already has its payloads in place.
                    if copy.exists() and copy.samefile(payload):
                        continue
                    copy.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(payload, copy)
        shutil.copyfile(source, target)
    if cache.workload_sha256(source) != cache.workload_sha256(target):
        raise ValueError(f"{target} does not load to the sequences of {source}")
    return target


def canonical_workload(fixture):
    """`<stem>.workload.json` beside a fixture, where the gates read its workload."""
    fixture = Path(fixture)
    return fixture.with_name(fixture.stem + ".workload.json")


def emit(skeleton, directories, workload_path, output, lock, lock_variant, modes, fields=FIELDS, topk=TOPK,
         greedy_fields=(), store=None):
    """Write the fixture `output` bound to the runs in `directories` and its
    workload beside it. `lock` is the lock's repo path, `modes` the
    `--modes` items. Returns what was written."""
    labels = dict(item.partition("=")[::2] for item in modes)
    variants = {label: variant or label for label, variant in labels.items()}
    if len(set(variants.values())) != len(variants):
        raise ValueError(f"--modes names a variant twice: {variants}")
    found = read_runs(directories)
    absent = sorted(set(labels) - set(found))
    if absent:
        raise ValueError(f"no run of {absent} in {[str(d) for d in directories]}")
    workload_path = copy_workload(workload_path, canonical_workload(output))
    _, sequences = cache.workload_sequences(workload_path)
    runs = {label: ordered(found[label], sequences, label) for label in labels}
    value = assemble(skeleton, runs, variants, fields, topk, greedy_fields)
    value = cache.bind(value, runs, str(workload_path), lock, lock_variant, variants, store)
    data = (json.dumps(value) + "\n").encode()
    Path(output).write_bytes(data)
    return {"fixture": str(output), "sha256": keys.sha256(data), "workload": str(workload_path),
            "refset_sha256": value["binding"]["refset_sha256"],
            "entries": {label: len(document["entries"]) for label, document in runs.items()}}


def arguments(parser):
    parser.add_argument("--skeleton", required=True, type=Path, help="the fixture without scores")
    parser.add_argument("--runs", required=True, nargs="+", type=Path, help="runner output directories")
    parser.add_argument("--workload", required=True, type=Path, help="the workload the runs scored")
    parser.add_argument("--lock", required=True, type=Path, help="the checkpoint lock, in the repository")
    parser.add_argument("--lock-variant", help="the lock's checkpoint variant")
    parser.add_argument("--modes", required=True, help="run labels to bind, each LABEL or LABEL=VARIANT")
    parser.add_argument("--fields", default=",".join(FIELDS), help="score columns each variant copies")
    parser.add_argument("--topk", type=int, default=TOPK, help="top-k entries kept per row")
    parser.add_argument("--greedy-fields", help="score columns of generated answers' steps, under `greedy`")
    parser.add_argument("--output", required=True, type=Path, help="the fixture, written with its binding")
    parser.add_argument("--cache", type=Path, default=cache.default_root(),
                        help="reference cache for the refset (default $CHOPINFER_DATA/refcache/v1)")
    parser.add_argument("--no-cache", dest="cache", action="store_const", const=None,
                        help="write no refset")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m tools.refrun emit",
                                     description="Assemble a bound gate fixture from reference runs.")
    arguments(parser)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    lock = args.lock.resolve().relative_to(keys.ROOT).as_posix() if args.lock.is_absolute() else args.lock.as_posix()
    report = emit(decode_json(args.skeleton.read_bytes()), args.runs, args.workload, args.output, lock,
                  args.lock_variant, args.modes.split(","), tuple(args.fields.split(",")), args.topk,
                  tuple(args.greedy_fields.split(",")) if args.greedy_fields else (),
                  cache.Store(args.cache) if args.cache is not None else None)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

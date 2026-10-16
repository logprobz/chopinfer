"""The reference runner's CLI: run, cache gc, diff-entries, verify-binding.

    python3 -m tools.refrun run --adapter models/<m>/reference.py \\
        --checkpoint DIR --lock models/<m>/checkpoint-files.json \\
        --workload WORKLOAD.json --modes f32,bf16,f64 --output DIR

`run` runs upstream reference code over a workload and writes per-mode
scores. Every checkpoint file it reads is checked against the lock by sha256
before it is read, and found unchanged before each output is written
(stamps.py). Each (sequence, mode) is looked up in the reference cache
(cache.py) first; a hit is not computed again, and every computed entry is
stored. The checkpoint is loaded only for the first miss, once
(place.Store), and each mode with a miss builds its model in turn. Greedy
answers are generated in the first mode and then forced in every mode. In
each mode the first computed sequence runs twice and its two score documents
must be byte-identical. Output: <output>/<mode>.json with every entry's
sequence name and category, key document, key_sha256 and scores, and
<output>/run.json with provenance.

With --taps, every mode's forced pass of each sequence also writes the
adapter's taps (taps.py) to <output>/taps/<mode>/step<NNNN>, one step per
sequence in workload order. Taps are not cached, so --taps computes every
forced pass, and a cached entry must then equal it byte for byte. The
repeat of the first sequence runs without taps, so the repeat check also
shows that taps leave the scores unchanged.

`cache gc` lists (and with --delete removes) entries and refsets no
tracked fixture binds and older than 14 days; `diff-entries` compares two
entries or two refsets; `verify-binding` checks fixtures' bindings.

Only `run` imports torch.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.model_sdk.diagnostic import canonical, decode_json
from tools.refrun import cache, keys, modes, stamps, workload

SCHEMA = "chopinfer.refrun.scores.v1"


def load_adapter(path):
    path = Path(path)
    relative = path.resolve().relative_to(keys.ROOT).as_posix()
    spec = importlib.util.spec_from_file_location(f"refrun_adapter_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, relative


def device_label(devices):
    import torch
    if all(d.startswith("cpu") for d in devices):
        return "cpu"
    capabilities = {torch.cuda.get_device_capability(d) for d in devices}
    if len(capabilities) != 1:
        raise ValueError(f"devices differ in compute capability: {sorted(capabilities)}")
    major, minor = capabilities.pop()
    return f"sm_{major}{minor}"


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=keys.ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_arguments(parser):
    parser.add_argument("--adapter", required=True, help="repo path of models/<m>/reference.py")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path, help="the checkpoint's source lock")
    parser.add_argument("--lock-variant", help="the lock's checkpoint variant (mimo26: mopd or rl)")
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--modes", default="f32,bf16,f64",
                        help="modes in run order; MODE:RULE adds MODE under dequantization RULE")
    parser.add_argument("--dequant", action="append", default=[], metavar="MODE=RULE",
                        help="override a mode's dequantization rule (exact, rounded, hf-bf16)")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stamps", type=Path, default=stamps.default_cache(),
                        help="checkpoint stamp cache (default $CHOPINFER_DATA/refcache/v1/stamps)")
    parser.add_argument("--cache", type=Path, default=cache.default_root(),
                        help="reference cache (default $CHOPINFER_DATA/refcache/v1)")
    parser.add_argument("--no-cache", dest="cache", action="store_const", const=None,
                        help="neither read nor write the reference cache")
    parser.add_argument("--recompute", action="store_true",
                        help="compute cache hits too; each must equal its entry byte for byte")
    parser.add_argument("--taps", action="store_true", help="write the adapter's taps (taps.py)")
    parser.add_argument("--tap-sites", help="comma-separated site globs to tap, or all (default: taps.DEFAULT_SITES)")
    parser.add_argument("--tap-rows", help="comma-separated positions to write (default: taps.rows)")
    parser.add_argument("--tap-bytes", type=int, help="byte budget per sequence (default: taps.BUDGET)")


def parse_args(argv):
    """The arguments of `run`."""
    parser = argparse.ArgumentParser(description="Run upstream reference code over a workload.")
    run_arguments(parser)
    return parser.parse_args(argv)


def tap_capture(args, reference, mode):
    """The mode's taps, or None without --taps."""
    if not args.taps:
        return None
    from tools.refrun import taps
    if reference.taps is None:
        raise ValueError("--taps: the adapter declares no tap map")
    sites = (taps.DEFAULT_SITES if args.tap_sites is None else
             None if args.tap_sites == "all" else tuple(args.tap_sites.split(",")))
    return taps.Taps(reference.model, reference.taps(reference.model), args.output / "taps" / mode,
                     sites, taps.BUDGET if args.tap_bytes is None else args.tap_bytes)


def mode_policies(names, dequant=()):
    """(label, policy) per item of `--modes`: a mode name, or MODE:RULE for
    the mode under another dequantization rule, labelled MODE-RULE, so one
    load can run both rules. `--dequant MODE=RULE` sets the rule of a plain
    MODE item instead."""
    overrides = dict(item.split("=", 1) for item in dequant)
    policies = []
    for item in names.split(","):
        name, _, rule = item.partition(":")
        if rule:
            policies.append((f"{name}-{rule}", modes.policy(name, rule)))
        else:
            policies.append((name, modes.policy(name, overrides.pop(name, None))))
    if overrides:
        raise ValueError(f"--dequant names modes not run: {sorted(overrides)}")
    labels = [label for label, _ in policies]
    if len(set(labels)) != len(labels):
        raise ValueError(f"--modes repeats a mode: {labels}")
    return policies


def run(args):
    import torch
    from tools.refrun import adapter as interface, place, quant, score

    devices = args.devices.split(",")
    modes.prepare_environment(cuda=any(d.startswith("cuda") for d in devices))
    reference_adapter, adapter_path = load_adapter(args.adapter)
    model_name, sequences = workload.load(args.workload)
    if model_name != reference_adapter.MODEL:
        raise ValueError(f"workload is for {model_name}, adapter for {reference_adapter.MODEL}")
    policies = mode_policies(args.modes, args.dequant)
    args.output.mkdir(parents=True, exist_ok=False)
    store = cache.Store(args.cache) if args.cache is not None else None

    lock = keys.read_lock(args.lock)
    if args.lock_variant is not None:
        lock = lock[args.lock_variant]
    # Verify before reading: a file read first could be changed back to its
    # locked bytes before it is verified.
    root = args.checkpoint
    first = ["config.json", place.INDEX] if (root / place.INDEX).exists() else ["config.json"]
    verified = stamps.verify(root, lock, first, args.stamps)
    extra = getattr(reference_adapter, "EXTRA_SHARDS", {})
    verified.update(stamps.verify(root, lock, [*place.shard_files(root), *extra], args.stamps))
    checkpoint = place.Checkpoint(root, extra)
    unverified = sorted(set(checkpoint.read) - set(verified))
    if unverified:
        raise RuntimeError(f"checkpoint files changed during verification: {unverified}")
    units = reference_adapter.units(checkpoint)
    files = sorted({"config.json", *checkpoint.files([n for u in units for n in u.tensors])})
    for device in devices:
        quant.check_codecs(device)
    placement = place.plan([(u.module, checkpoint.stored_bytes(u.tensors)) for u in units if not u.host],
                           devices)
    common = {
        "model": model_name,
        "checkpoint": keys.checkpoint(lock, files),
        "reference_code": keys.reference_code(
            {label: keys.file_sha256(path) for label, path in reference_adapter.reference_code(checkpoint).items()}),
        "adapter": keys.adapter(keys.repo_files([adapter_path, *keys.NUMERICS])),
        "environment": keys.environment(device_label(devices)),
        "outputs": keys.outputs(score.TOPK),
    }
    provenance = {"argv": sys.argv, "git_head": git_head(), "placement": placement.document(),
                  "load": None, "cache": None if store is None else str(store.root),
                  "devices": [torch.cuda.get_device_name(d) if d.startswith("cuda") else d for d in devices],
                  "modes": {}}
    weights = []

    def loaded():
        """The checkpoint tensors, loaded at the first cache miss."""
        if not weights:
            weights.append(place.Store(checkpoint, placement, units))
            provenance["load"] = {"bytes": weights[0].read_bytes, "read_seconds": weights[0].read_seconds,
                                  "seconds": weights[0].load_seconds}
        return weights[0]

    forced = list(sequences)
    for index, (label, policy) in enumerate(policies):
        started = time.perf_counter()
        seconds = {}

        def timed(sequence, score_one):
            began = time.perf_counter()
            scores = score_one(sequence)
            seconds[sequence.name] = seconds.get(sequence.name, 0.0) + time.perf_counter() - began
            print(json.dumps({"mode": label, "sequence": sequence.name, "seconds": round(seconds[sequence.name], 1)}),
                  file=sys.stderr, flush=True)
            return scores
        dtype = modes.torch_dtype(policy["dtype"])
        built, tapped, entries, computed, repeated = [], [], [], [], []
        tap_rows = None if args.tap_rows is None else [int(p) for p in args.tap_rows.split(",")]

        def reference():
            if not built:
                built.append(reference_adapter.build(interface.BuildConfig(checkpoint, policy, dtype, loaded()),
                                                     placement))
                interface.check_tensor_names(built[0], loaded().tensors)
                quant.configure(built[0].model, dtype, policy["dequant"])
                place.attach(built[0].model, placement)
                tapped.append(tap_capture(args, built[0], label))
            return built[0]

        def scored(sequence, compute, recompute=args.recompute):
            """The sequence's scores from the cache, or computed (and then
            also when `recompute` asks to check a hit)."""
            document = keys.document(sequence=keys.sequence(sequence), mode=keys.mode(policy), **common)
            key = keys.key_sha256(document)
            hit = store.get(key) if store is not None else None
            scores = hit["scores"] if hit is not None and not recompute else timed(sequence, compute)
            if hit is None or recompute:
                computed.append((document, scores, sequence.name))
            entries.append({"name": sequence.name, "category": sequence.category, "key_sha256": key,
                            "key": document, "scores": scores})
            return scores

        def teacher_forced(sequence):
            model = reference()
            if tapped[0] is None:
                scores = score.teacher_forced(model, sequence, policy, placement.first)
            else:
                with tapped[0].step(sequence.name, sequence.forced_ids, tap_rows,
                                    {item.kind for item in sequence.media}):
                    scores = score.teacher_forced(model, sequence, policy, placement.first)
            if not repeated:
                repeat = score.teacher_forced(reference(), sequence, policy, placement.first)
                if canonical(scores) != canonical(repeat):
                    raise RuntimeError(f"mode {label}: {sequence.name} scored differently on a repeat")
                repeated.append(sequence.name)
            return scores

        if index == 0:
            for position, sequence in enumerate(forced):
                if sequence.answer.policy == "greedy":
                    steps = scored(sequence, lambda s: score.greedy(reference(), s, policy, placement.first)[1])
                    forced[position] = sequence.forced(steps["answer"])
        for sequence in forced:
            scored(sequence, teacher_forced, args.recompute or args.taps)
        stamps.confirm(root, verified)
        if store is not None:
            for document, scores, name in computed:
                store.put(document, scores, {"name": name, "git_head": provenance["git_head"], "argv": sys.argv,
                                             "devices": provenance["devices"], "created": time.time()})
        (args.output / f"{label}.json").write_bytes(canonical(
            {"schema": SCHEMA, "policy": policy, "entries": entries}) + b"\n")
        if built:
            provenance["layout"], provenance["replacements"] = built[0].layout, built[0].replacements
        if tapped and tapped[0] is not None:
            tapped[0].remove()
            provenance["taps"] = tapped[0].names
        built.clear()
        tapped.clear()
        quant.release_scratch()
        if any(d.startswith("cuda") for d in devices):
            for device in devices:
                torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        provenance["modes"][label] = {"seconds": time.perf_counter() - started, "sequence_seconds": seconds,
                                      "repeat_bitwise_equal": True if repeated else None,
                                      "entries": len(entries), "computed": len(computed),
                                      "cache_hits": len(entries) - len(computed)}
    (args.output / "run.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return provenance


def store_argument(parser):
    parser.add_argument("--cache", type=Path, default=cache.default_root(),
                        help="reference cache (default $CHOPINFER_DATA/refcache/v1)")


def collect(args):
    if args.cache is None:
        raise SystemExit("no reference cache: set CHOPINFER_DATA or pass --cache")
    report = cache.gc(cache.Store(args.cache), args.repo, args.ref, delete=args.delete)
    print(json.dumps(report, indent=2))


def diff(args):
    if args.cache is None:
        raise SystemExit("no reference cache: set CHOPINFER_DATA or pass --cache")
    print(json.dumps(cache.diff_entries(cache.Store(args.cache), args.a, args.b), indent=2))


def verify_binding(args):
    store = cache.Store(args.cache) if args.cache is not None and args.cache.exists() else None
    if args.workload is not None and len(args.fixture) != 1:
        raise SystemExit("--workload checks one fixture")
    stale = False
    for path in args.fixture:
        try:
            result = cache.verify(decode_json(path.read_bytes()), args.repo, args.checkpoint, store, args.workload)
        except cache.StaleReference as error:
            result, stale = {"status": "stale_reference", "field": error.field, "detail": str(error)}, True
        print(json.dumps({"fixture": str(path), **result}, sort_keys=True))
    if stale:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m tools.refrun", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_arguments(commands.add_parser("run", help="score a workload through the cache"))
    gc_parser = commands.add_parser("cache", help="reference cache maintenance").add_subparsers(
        dest="action", required=True).add_parser("gc", help="entries no tracked fixture binds, older than 14 days")
    store_argument(gc_parser)
    gc_parser.add_argument("--repo", type=Path, default=keys.ROOT)
    gc_parser.add_argument("--ref", action="append", default=[],
                           help="another git ref whose fixtures keep their refsets, besides main and the keep file")
    gc_parser.add_argument("--delete", action="store_true", help="remove them (default: list only)")
    diff_parser = commands.add_parser("diff-entries", help="compare two entries or two refsets")
    store_argument(diff_parser)
    diff_parser.add_argument("a")
    diff_parser.add_argument("b")
    verify_parser = commands.add_parser("verify-binding", help="check fixtures' reference bindings")
    store_argument(verify_parser)
    verify_parser.add_argument("--repo", type=Path, default=keys.ROOT)
    verify_parser.add_argument("--checkpoint", type=Path, help="checkpoint directory, to check reference code")
    verify_parser.add_argument("--workload", type=Path, help="the workload the fixture's binding scored")
    verify_parser.add_argument("fixture", nargs="+", type=Path)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    {"run": run, "cache": collect, "diff-entries": diff, "verify-binding": verify_binding}[args.command](args)


if __name__ == "__main__":
    main()

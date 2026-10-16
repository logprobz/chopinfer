#!/usr/bin/env python3
"""Fixed-state decode replay: capacity, slot placement and batch composition.

One teacher-forced decode bucket is replayed from identical inputs in arms
that change one thing: a repeat (run-to-run determinism), the served
capacity, the slots and pages its requests occupy, or the ordered lengths
of the filler requests sharing its steps. `prepare` and `compare` run
without devices; `run` needs the idle GPUs of one topology.
"""
import argparse
import datetime
import hashlib
import importlib.util
from itertools import repeat
import json
import math
import operator
import os
from pathlib import Path
import sys
import time
from array import array

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnostics

spec = importlib.util.spec_from_file_location(
    "fixed_batch_logits", Path(__file__).resolve().parent / "fixed-batch-logits.py")
fixed_batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed_batch)

FORMAT = "fixed-state-replay-v1"
TEST = fixed_batch.TEST
MIXED_DECODE = 5
DRY_RUN = "fixed trace dry run: "
# Arms in run order: (name, trace, served capacity as a multiple of the
# bucket). Repeats follow their first arm, so a later arm's comparison
# always has a determinism control at the same capacity.
ARMS = [("a1", "a", 1), ("a1-repeat", "a", 1), ("a2", "a", 2), ("a2-repeat", "a", 2),
        ("holders2", "holders", 2), ("b1", "b", 1), ("b2", "b", 2)]
# (question, reference arm, other arm, rows): bitwise questions compare
# every row both arms share; composition compares the probe rows.
COMPARISONS = [("determinism", "a1", "a1-repeat", "shared"),
               ("determinism", "a2", "a2-repeat", "shared"),
               ("capacity", "a1", "a2", "shared"),
               ("capacity", "b1", "b2", "shared"),
               ("placement", "a2", "holders2", "shared"),
               ("composition", "a1", "b1", "probes")]


def select_arms(names):
    """The arms named in the comma-separated `names` (all when empty), in
    run order, and the comparisons both of whose arms run."""
    names = names.split(",") if names else [arm for arm, _, _ in ARMS]
    arms = [arm for arm in ARMS if arm[0] in names]
    if not arms or len(arms) != len(names):
        raise ValueError("unknown or repeated arm in --arms")
    comparisons = [c for c in COMPARISONS if c[1] in names and c[2] in names]
    return arms, comparisons


def history_key(tokens):
    """A row's identity across arms: its full token history."""
    return hashlib.sha256(array("I", tokens).tobytes()).hexdigest()


def probe_texts(reference, probes, decode_steps):
    """The first `probes` texts by name that are long enough to force
    `decode_steps` decode rows after a prompt of at least four tokens, and the
    longest text, whose tokens fill the other requests."""
    texts = sorted(reference["texts"], key=lambda text: text["name"])
    source = max(texts, key=lambda text: len(text["prompt_token_ids"]))
    chosen = [text for text in texts if text is not source
              and len(text["prompt_token_ids"]) >= decode_steps + 5
              and text["score_from"] <= len(text["prompt_token_ids"]) - decode_steps - 1]
    if len(chosen) < probes:
        raise ValueError(f"the reference has {len(chosen)} texts that can force "
                         f"{decode_steps} decode steps, not {probes}")
    return chosen[:probes], source


def filler_lengths(fillers, shortest, longest):
    """Evenly spread prompt lengths, in filler order."""
    if fillers == 1:
        return [shortest]
    return [shortest + (longest - shortest) * f // (fillers - 1) for f in range(fillers)]


def requests(reference, rows, probes, decode_steps, shortest, longest, reverse):
    """The bucket's requests in batch order, probes and fillers interleaved:
    each its full token stream and prompt length, and for a probe its scored
    targets. `reverse` gives the fillers the same lengths in reverse order."""
    texts, source = probe_texts(reference, probes, decode_steps)
    fillers = rows - probes
    if fillers < 0 or probes < 1:
        raise ValueError("a bucket holds its probes and at least zero fillers")
    lengths = filler_lengths(fillers, shortest, longest)
    if reverse:
        lengths.reverse()
    tokens = source["prompt_token_ids"]
    span = len(tokens) - longest - decode_steps
    if span < fillers:
        raise ValueError("the filler source is too short for distinct filler windows")
    probe_rows = []
    for text in texts:
        stream = text["prompt_token_ids"]
        prompt = len(stream) - 1 - decode_steps
        # The prefill row predicts the token at `prompt`; decode step k
        # forces the token at `prompt + k` and predicts its successor.
        targets = [dict(position=position, token=stream[position],
                        exact=text["exact"]["logprob"][position - text["score_from"]]
                        if position >= text["score_from"] else None)
                   for position in range(prompt, prompt + decode_steps + 1)]
        probe_rows.append(dict(text=text["name"], stream=list(stream[:prompt + decode_steps]),
                               prompt=prompt, targets=targets))
    # A filler's window ends at the same token whatever its length, so its
    # forced decode tokens are the same in both compositions.
    ends = [f * span // fillers + longest + decode_steps for f in range(fillers)]
    filler_rows = [dict(filler=f, prompt=lengths[f],
                        stream=list(tokens[ends[f] - lengths[f] - decode_steps:ends[f]]))
                   for f in range(fillers)]
    # Probes and fillers alternate while both remain.
    out = []
    for index in range(max(probes, fillers)):
        out += probe_rows[index:index + 1] + filler_rows[index:index + 1]
    return out


def holder_requests(reference, holders, holder_tokens, decode_steps, longest):
    """Short requests that are prefilled and then stay idle."""
    tokens = max(reference["texts"], key=lambda text: len(text["prompt_token_ids"]))["prompt_token_ids"]
    span = len(tokens) - longest - decode_steps
    return [dict(holder=h, prompt=holder_tokens,
                 stream=list(tokens[span - 1 - h:span - 1 - h + holder_tokens]))
            for h in range(holders)]


def build_trace(rows, holders, vocabulary, decode_steps, prefill_tokens):
    """A version-5 trace: holders, then the bucket's requests, prefilled
    whole in steps of at most `prefill_tokens`; then `decode_steps` steps
    forcing one token for each bucket request in batch order."""
    slots = holders + rows
    steps, step, used = [], [], 0
    for slot, request in enumerate(slots):
        prompt = request["stream"][:request["prompt"]]
        if len(prompt) > prefill_tokens:
            raise ValueError("a prompt exceeds the prefill step budget")
        # Holders never share a prefill step with the bucket, so they claim
        # the lowest slots first.
        if step and (used + len(prompt) > prefill_tokens
                     or (slot == len(holders) and holders)):
            steps.append(step)
            step, used = [], 0
        step.append(dict(slot=slot, prefill=True, inputs=prompt, drafts=0, kept=len(prompt),
                         history=prompt, emitted=0))
        used += len(prompt)
    steps.append(step)
    for offset in range(decode_steps):
        items = []
        for index, request in enumerate(rows):
            slot = len(holders) + index
            end = request["prompt"] + offset + 1
            items.append(dict(slot=slot, prefill=False, inputs=[request["stream"][end - 1]],
                              drafts=0, kept=1, history=request["stream"][:end],
                              emitted=offset + 1))
        steps.append(items)
    for request in slots:
        if any(token >= vocabulary for token in request["stream"]):
            raise ValueError("a reference token is outside the vocabulary")
    return dict(version=MIXED_DECODE, concurrency=len(slots), vocabulary=vocabulary,
                steps=[dict(mode="greedy", items=items) for items in steps])


def row_plan(rows):
    """Scored probe rows by history key: the target each row's logits predict."""
    plan = {}
    for request in rows:
        if "text" not in request:
            continue
        for offset, target in enumerate(request["targets"]):
            key = history_key(request["stream"][:request["prompt"] + offset])
            plan[key] = dict(text=request["text"], step=offset, **target)
    return plan


def geometry(rows, decode_steps):
    """Per decode step, the ordered sequence lengths the bucket's attention
    partitions over."""
    return [[request["prompt"] + offset + 1 for request in rows] for offset in range(decode_steps)]


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, separators=(",", ":"))
        stream.write("\n")


def make_traces(args, destination):
    reference = json.loads(args.reference.read_text())
    common = dict(rows=args.rows, probes=args.probes, decode_steps=args.decode_steps,
                  shortest=args.shortest, longest=args.longest)
    a = requests(reference, reverse=False, **common)
    b = requests(reference, reverse=True, **common)
    holders = holder_requests(reference, args.rows, args.holder_tokens, args.decode_steps, args.longest)
    traces, plans = {}, {}
    for name, rows, idle in (("a", a, []), ("b", b, []), ("holders", a, holders)):
        trace = build_trace(rows, idle, args.vocabulary, args.decode_steps, args.prefill_tokens)
        path = destination / f"trace-{name}.json"
        write_json(path, trace)
        traces[name] = dict(path=str(path), sha256=diagnostics.digest(path),
                            requests=trace["concurrency"],
                            final_tokens=max(len(i["history"]) for s in trace["steps"] for i in s["items"]),
                            steps=len(trace["steps"]),
                            rows=sum(len(s["items"]) for s in trace["steps"]))
        plans[name] = row_plan(rows)
    if plans["a"] != plans["b"] or plans["a"] != plans["holders"]:
        raise ValueError("probe rows must be identical in every trace")
    lengths = {name: geometry(rows, args.decode_steps) for name, rows in (("a", a), ("b", b))}
    if any(x == y for x, y in zip(lengths["a"], lengths["b"])):
        raise ValueError("composition arm does not change the ordered lengths")
    page_tokens = args.page_tokens
    pages = max(t["requests"] * -(-t["final_tokens"] // page_tokens) + t["requests"]
                for t in traces.values())
    write_json(destination / "probes.json", plans["a"])
    return traces, pages, lengths


def arm_config(prepared, arm, trace, capacity, output):
    return dict(bundle=prepared["bundle"], topology=prepared["topology"],
                checkpoint=prepared["checkpoint"], trace=trace["path"], trace_input=True,
                output=str(output), reference=None, control=None,
                concurrency=trace["requests"], verify=False, owned=False,
                max_sequences=capacity, mixed=True, kv_pages=prepared["kv_pages"],
                source=dict(build=prepared["source"], binary_sha256=prepared["binary_sha256"]))


def arm_environment(parent, devices, ranks, overrides, dry_run=False):
    # No inherited runtime switches: packages run their defaults, apart from
    # the preparation's recorded overrides.
    env = {key: value for key, value in parent.items() if not key.startswith("CHOPINFER_")}
    env.update(overrides)
    env.update(CUDA_VISIBLE_DEVICES="" if dry_run else devices,
               CHOPINFER_KERNEL_GPUS=",".join(str(rank) for rank in range(ranks)))
    if dry_run:
        env["CHOPINFER_BATCH_REPLAY_DRY_RUN"] = "1"
    return env


def dry_run(prepared, destination):
    """Checks every arm's configuration, package, checkpoint and trace in
    the recorder without opening a device."""
    results = {}
    for arm, trace, multiple in prepared["arms"]:
        config = arm_config(prepared, arm, prepared["traces"][trace],
                            multiple * prepared["bucket"], destination / "dry-run" / arm)
        path = destination / f"dry-run-{arm}.json"
        write_json(path, config)
        env = arm_environment(os.environ, "", prepared["ranks"], prepared["environment"],
                              dry_run=True)
        env["CHOPINFER_BATCH_REPLAY_CONFIG"] = str(path)
        log = destination / f"dry-run-{arm}.log"
        fixed_batch.command([prepared["binary"], "--ignored", "--exact", TEST, "--test-threads=1",
                             "--nocapture"], log, env, 600)
        # The test harness may print its own text before ours on the line.
        lines = [line.split(DRY_RUN, 1)[1] for line in log.read_text().splitlines() if DRY_RUN in line]
        if len(lines) != 1:
            raise RuntimeError(f"{arm}: the recorder did not report one dry run")
        result = json.loads(lines[0])
        identity = result["identity"]
        if (identity.get("format") != "fixed-batch-mixed-v1"
                or identity.get("max_sequences") != config["max_sequences"]
                or identity.get("kv_pages") != prepared["kv_pages"]
                or identity.get("trace_sha256") != prepared["traces"][trace]["sha256"]):
            raise RuntimeError(f"{arm}: dry run identity differs from its configuration")
        results[arm] = result
    return results


def prepare(args):
    revision = fixed_batch.source()
    root = diagnostics.external(args.root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / ("prepared-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S"))
    destination.mkdir()
    traces, pages, lengths = make_traces(args, destination)
    write_json(destination / "lengths.json", lengths)
    if args.binary:
        binary = destination / "replay-tests"
        binary.write_bytes(diagnostics.external(args.binary).read_bytes())
        binary.chmod(0o755)
    else:
        binary = fixed_batch.build_recorder(destination)
    if fixed_batch.source() != revision:
        raise RuntimeError("source changed during preparation")
    bundle = diagnostics.external(args.bundle)
    arms, comparisons = select_arms(args.arms)
    prepared = dict(format=FORMAT, source=revision, binary=str(binary),
                    binary_sha256=diagnostics.digest(binary), bundle=str(bundle),
                    manifest_sha256=diagnostics.digest(bundle / "manifest.json"),
                    checkpoint=str(args.checkpoint), topology=args.topology, ranks=args.ranks,
                    bucket=args.rows, probes=args.probes, decode_steps=args.decode_steps,
                    kv_pages=pages, traces=traces, arms=arms, comparisons=comparisons,
                    environment=fixed_batch.parse_overrides(args.env),
                    reference=str(args.reference), reference_sha256=fixed_batch.digest(args.reference))
    prepared["dry_run"] = dry_run(prepared, destination)
    write_json(destination / "prepared.json", prepared)
    print("prepared:", destination / "prepared.json", flush=True)


def check_prepared(prepared):
    if prepared.get("format") != FORMAT:
        raise ValueError("not a fixed-state replay preparation")
    if diagnostics.digest(Path(prepared["binary"])) != prepared["binary_sha256"]:
        raise ValueError("prepared executable changed")
    if diagnostics.digest(Path(prepared["bundle"]) / "manifest.json") != prepared["manifest_sha256"]:
        raise ValueError("prepared package manifest changed")
    for trace in prepared["traces"].values():
        if diagnostics.digest(Path(trace["path"])) != trace["sha256"]:
            raise ValueError("prepared trace changed")


def run(args):
    prepared_path = diagnostics.external(args.prepared)
    prepared = json.loads(prepared_path.read_text())
    check_prepared(prepared)
    devices = args.devices.split(",")
    if (len(devices) != prepared["ranks"] or len(set(devices)) != len(devices)
            or not all(d.isdigit() for d in devices)):
        raise ValueError("one distinct GPU ordinal per rank required")
    directory = prepared_path.parent / ("run-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S"))
    directory.mkdir()
    deadline = time.monotonic() + args.minutes * 60
    write_json(directory / "start.json", dict(prepared=str(prepared_path), devices=fixed_batch.gpu_state(args.devices),
                                              minutes=args.minutes))
    arms, error = {}, None
    try:
        for arm, trace, multiple in prepared["arms"]:
            config = arm_config(prepared, arm, prepared["traces"][trace],
                                multiple * prepared["bucket"], directory / arm)
            path = directory / f"{arm}.json"
            write_json(path, config)
            env = arm_environment(os.environ, args.devices, prepared["ranks"],
                                  prepared["environment"])
            env["CHOPINFER_BATCH_REPLAY_CONFIG"] = str(path)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("deadline reached before arm " + arm)
            started = time.monotonic()
            fixed_batch.command([prepared["binary"], "--ignored", "--exact", TEST, "--test-threads=1",
                                 "--nocapture"], directory / f"{arm}.log", env, remaining)
            receipt = json.loads((directory / arm / "receipt.json").read_text())
            expected = prepared["traces"][trace]
            if (receipt.get("passed") is not True or receipt.get("steps") != expected["steps"]
                    or receipt.get("rows") != expected["rows"] or receipt.get("mixed") is not True
                    or receipt.get("max_sequences") != config["max_sequences"]
                    or receipt.get("kv_pages") != prepared["kv_pages"]):
                raise RuntimeError(f"{arm}: incomplete replay receipt")
            arms[arm] = dict(seconds=time.monotonic() - started, rank_rows=receipt["rank_rows"],
                             memory_after_steps=receipt["memory_after_steps"])
            fixed_batch.gpu_state(args.devices)
    except Exception as failure:
        error = str(failure)
        raise
    finally:
        write_json(directory / "result.json", dict(passed=error is None, error=error, arms=arms,
                   devices_after=fixed_batch.gpu_state(args.devices)))
        print("result:", directory, "passed:", error is None, flush=True)
    if args.compare_minutes and deadline - time.monotonic() >= args.compare_minutes * 60:
        compare(SimpleArgs(run=directory))
    elif args.compare_minutes:
        print("compare skipped for time; run: compare --run", directory, flush=True)


class SimpleArgs:
    def __init__(self, **values):
        self.__dict__.update(values)


def recorded_rows(root):
    """(key, frame step, element type, width, byte offset) for each row of
    one recorded arm, in record order."""
    rows, offset = [], 0
    with (root / "frames.jsonl").open() as frames:
        next(frames)
        for step, line in enumerate(frames):
            for row in json.loads(line)["rows"]:
                rows.append((history_key(row["metadata"]["history"]), step, row["element_type"],
                             row["width"], offset, row["byte_count"]))
                offset += row["byte_count"]
    if offset != (root / "rows.bin").stat().st_size:
        raise ValueError(f"{root}: frames and logits bytes disagree")
    keys = [row[0] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{root}: two rows share a history")
    return {row[0]: row[1:] for row in rows}


def logits(raw, element_type):
    values = array("f")
    if element_type == 1:
        values.frombytes(raw)
    else:
        widened = bytearray(2 * len(raw))
        widened[2::4] = raw[0::2]
        widened[3::4] = raw[1::2]
        values.frombytes(bytes(widened))
    if sys.byteorder != "little":
        values.byteswap()
    return values


def log_softmax_at(values, token):
    top = max(values)
    total = math.fsum(map(math.exp, map(operator.sub, values, repeat(top, len(values)))))
    return values[token] - top - math.log(total)


class Arm:
    def __init__(self, root):
        self.root = root
        self.rows = recorded_rows(root)
        self.stream = (root / "rows.bin").open("rb")

    def close(self):
        self.stream.close()

    def read(self, key):
        _, element_type, _, offset, count = self.rows[key]
        self.stream.seek(offset)
        return self.stream.read(count), element_type


def compare_pair(reference, other, keys, probes):
    """Bitwise and numerical differences of the rows both arms record; per
    probe step (0 is the prefill row) the mean absolute target logprob
    difference over all of the step's probe rows."""
    identical, first, worst, per_step = 0, None, 0.0, {}
    for key in keys:
        left, element_type = reference.read(key)
        right, _ = other.read(key)
        bucket = per_step.setdefault(probes[key]["step"], dict(
            rows=0, identical=0, dlogprob=0.0, max_abs_logit=0.0, top1_differs=0)) if key in probes else {}
        bucket["rows"] = bucket.get("rows", 0) + 1
        if left == right:
            identical += 1
            bucket["identical"] = bucket.get("identical", 0) + 1
            continue
        a, b = logits(left, element_type), logits(right, element_type)
        # Bytes can differ where values compare equal (+0 and -0): such a
        # row is not identical and adds no numeric difference.
        difference = max(map(abs, map(operator.sub, a, b)))
        worst = max(worst, difference)
        frame = reference.rows[key][0]
        if first is None or frame < first["frame"]:
            size = 4 if element_type == 1 else 2
            first = dict(frame=frame, probe=probes.get(key, {}).get("text"),
                         element=next(i for i in range(len(left) // size)
                                      if left[i * size:(i + 1) * size] != right[i * size:(i + 1) * size]))
        if key in probes:
            target = probes[key]["token"]
            bucket["dlogprob"] += abs(log_softmax_at(a, target) - log_softmax_at(b, target))
            bucket["max_abs_logit"] = max(bucket["max_abs_logit"], difference)
            bucket["top1_differs"] += int(a.index(max(a)) != b.index(max(b)))
    for bucket in per_step.values():
        bucket["dlogprob"] /= bucket["rows"]
    return dict(rows=len(keys), identical=identical, first_difference=first, max_abs_logit=worst,
                probe_steps={str(step): per_step[step] for step in sorted(per_step)})


def probe_error(arm, probes):
    """Mean absolute logprob error of an arm's scored probe rows."""
    errors = []
    for key, probe in probes.items():
        if probe["exact"] is None or key not in arm.rows:
            continue
        raw, element_type = arm.read(key)
        errors.append(abs(log_softmax_at(logits(raw, element_type), probe["token"]) - probe["exact"]))
    return dict(positions=len(errors), mean_abs_dlogprob=math.fsum(errors) / len(errors) if errors else None)


def compare(args):
    directory = diagnostics.external(args.run)
    result = json.loads((directory / "result.json").read_text())
    if result.get("passed") is not True:
        raise ValueError("the replay run did not pass")
    prepared = json.loads((directory.parent / "prepared.json").read_text())
    probes = json.loads((directory.parent / "probes.json").read_text())
    arms = {arm: Arm(directory / arm) for arm, _, _ in prepared["arms"]}
    try:
        report = compare_arms(directory, prepared, probes, arms)
    finally:
        for record in arms.values():
            record.close()
    write_json(directory / "compare.json", report)
    print("compare:", directory / "compare.json", flush=True)
    return report


def compare_arms(directory, prepared, probes, arms):
    report = dict(format=FORMAT, run=str(directory), comparisons=[],
                  environment=prepared.get("environment", {}),
                  probe_error={arm: probe_error(record, probes) for arm, record in arms.items()})
    # Every arm recorded its whole trace: a shortened replay would compare
    # only the rows it reached.
    traces = {arm: trace for arm, trace, _ in prepared["arms"]}
    for arm, record in arms.items():
        expected = prepared["traces"][traces[arm]]["rows"]
        if len(record.rows) != expected:
            raise ValueError(f"{arm}: {len(record.rows)} rows recorded, its trace has {expected}")
    for question, left, right, rows in prepared["comparisons"]:
        ours, theirs = arms[left].rows.keys(), arms[right].rows.keys()
        if rows == "probes":
            keys = sorted(key for key in ours & theirs if key in probes)
            if len(keys) != len(probes):
                raise ValueError(f"{left}/{right}: probe rows are missing")
        elif traces[left] == traces[right]:
            # Repeats and capacities replay one trace: the same histories.
            if ours != theirs:
                raise ValueError(f"{left}/{right}: the arms recorded different histories")
            keys = sorted(ours)
        else:
            # Placement: the holders' trace adds the holders' own prefill
            # rows to every row of the other trace.
            if not ours <= theirs:
                raise ValueError(f"{left}/{right}: rows of {left} are missing from {right}")
            keys = sorted(ours)
        outcome = compare_pair(arms[left], arms[right], keys, probes)
        outcome.update(question=question, reference=left, other=right,
                       unpaired_rows=len(theirs - ours),
                       bitwise_equal=outcome["identical"] == outcome["rows"])
        report["comparisons"].append(outcome)
        print(f"{question:12s} {left:>10s} vs {right:<10s} rows {outcome['rows']:5d} "
              f"identical {outcome['identical']:5d} max|dlogit| {outcome['max_abs_logit']:.6g}", flush=True)
    for arm, error in report["probe_error"].items():
        print(f"probe error {arm:>10s}: {error['mean_abs_dlogprob']} over {error['positions']} positions",
              flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare", help="build the recorder, write traces and dry-run every arm without devices")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True,
                   help="teacher reference JSON: texts with prompt_token_ids, score_from and exact logprobs")
    p.add_argument("--bundle", type=Path, required=True, help="direct package directory")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--topology", required=True)
    p.add_argument("--ranks", type=int, default=4)
    p.add_argument("--vocabulary", type=int, required=True)
    p.add_argument("--page-tokens", type=int, required=True)
    p.add_argument("--rows", type=int, default=64, help="decode bucket rows")
    p.add_argument("--probes", type=int, default=32)
    p.add_argument("--decode-steps", type=int, default=40)
    p.add_argument("--shortest", type=int, default=256, help="shortest filler prompt")
    p.add_argument("--longest", type=int, default=2048, help="longest filler prompt")
    p.add_argument("--holder-tokens", type=int, default=8)
    p.add_argument("--prefill-tokens", type=int, default=8192)
    p.add_argument("--binary", type=Path, help="an already built recorder instead of building one")
    p.add_argument("--arms", help="comma-separated arms to run, default all; comparisons need both arms")
    p.add_argument("--env", action="append", default=[], metavar="CHOPINFER_NAME=VALUE",
                   help="a runtime switch every arm runs with (dispatch switches only)")
    r = sub.add_parser("run", help="replay every arm on idle GPUs")
    r.add_argument("--prepared", type=Path, required=True)
    r.add_argument("devices")
    r.add_argument("--minutes", type=int, default=22)
    r.add_argument("--compare-minutes", type=int, default=0,
                   help="compare after the replays when at least this many minutes remain")
    c = sub.add_parser("compare", help="bitwise and numerical comparison of a run's arms")
    c.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "run":
        if not 1 <= args.minutes <= 25:
            parser.error("run requires a 1..25 minute budget")
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()

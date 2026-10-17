#!/usr/bin/env python3
"""Where each serving step's time goes, from an Nsight Systems SQLite export
of a server (CUDA trace) and, optionally, its `CHOPINFER_STEP_TRACE` lines.

Host steps come from the engine thread, the thread issuing the most CUDA API
calls. A step ends at its sampler collect: the last device-to-host copy that
is waited on (cuMemcpyDtoH*, then cuStreamSynchronize) after the step's long
logits wait (a cuStreamSynchronize of at least `--long-sync-us`) and before
the next step's first cuGraphLaunch. Each kernel belongs to the step of the
API call that launched it (correlationId), so graph-node kernels belong to
the step whose cuGraphLaunch issued them. At graph granularity, whole-graph
records (CUPTI_ACTIVITY_KIND_GRAPH_TRACE) stand in for their kernels.

Per step and device:
  span         first kernel start to last kernel end;
  busy         union of kernel intervals (PDL overlap counted once);
  inner gap    span minus busy, idle device time inside the step;
  idle before  the previous step's last kernel end to this step's first
               kernel start: device idle between steps.
Each kernel is charged the advance it causes in the running latest end (as
serving_step_nsys.py does), summed per kernel class (`--class`).

Collectives: the k-th collective kernel of a step on every device is one
instance. A device arrives when its earlier kernels in the step have ended;
the instance's intrinsic time is the latest exit minus the latest arrival,
and a device's skew wait is the latest arrival minus its own.

Host phases per step: pre_launch (step start to the first graph launch:
commit, scheduling, batch preparation, uploads), launch_api, logits_wait,
after_logits (sampling issue, wait and collect), prepare_kv (from the first
cuMemset* to the last cuStreamSynchronize before the first graph launch, the
host KV pool's page zeroing).

The step trace reader takes JSON lines (`chopinfer.step.v1`), each optionally
prefixed by a unix time in ns and a tab, and summarizes the decode steps
(no prefill tokens) inside `--window-ns`, if given.

Usage: serving_step_breakdown.py [EXPORT.sqlite] [--step-trace FILE]
       [--window-ns START END] [--class LABEL=PREFIX[,PREFIX...]]...
       [--long-sync-us N] [--skip N] [--json OUT]
"""

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict

# Shared kernel families (kernels-sm120 symbol prefixes), first match wins.
CLASSES = [
    ("comm", ("chop_comm_",)),
    ("moe", ("chop_moe_",)),
    ("proj_gemm", ("chop_proj_", "chop_gemm_")),
    ("linear_attn", ("chop_channel_decay_", "chop_delta_", "chop_linear_")),
    ("attention", ("chop_latent_", "chop_pooled_indexer_", "chop_attn_", "chop_attention_")),
    ("hyper_conn", ("chop_sinkhorn_hc_",)),
    ("graph", ("<graph>",)),
]
SYNC = "cuStreamSynchronize"
LAUNCHES = ("cuGraphLaunch", "cuLaunchKernelEx", "cuLaunchKernel")
TOP = 15


def kernel_class(name, classes):
    for label, prefixes in classes:
        if name.startswith(prefixes):
            return label
    return "other"


def load(connection):
    """(engine-thread API calls, kernels) with names resolved."""
    names = dict(connection.execute("SELECT id, value FROM StringIds"))
    api = connection.execute(
        "SELECT start, end, globalTid, correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"
    ).fetchall()
    calls = defaultdict(int)
    for row in api:
        calls[row[2]] += 1
    engine = max(calls, key=calls.get)
    api = [(start, end, correlation, names[name]) for start, end, thread, correlation, name in api
           if thread == engine]
    kernels = [(start, end, device, correlation, names[name]) for start, end, device, correlation, name in
               connection.execute("SELECT start, end, deviceId, correlationId, shortName "
                                  "FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:
        kernels += [(start, end, device, correlation, "<graph>") for start, end, device, correlation in
                    connection.execute("SELECT start, end, deviceId, correlationId "
                                       "FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE")]
    kernels.sort()
    return api, kernels


def host_steps(api, long_sync_ns):
    """[(start, end, calls)] between consecutive sampler collects."""
    boundaries = []
    waiting = pending_copy = False
    collect = None
    for start, end, _, name in api:
        if name == SYNC and end - start >= long_sync_ns:
            if waiting and collect is not None:
                boundaries.append(collect)
            waiting, pending_copy, collect = True, False, None
        elif not waiting:
            continue
        elif name.startswith("cuMemcpyDtoH"):
            pending_copy = True
        elif name == SYNC and pending_copy:
            collect, pending_copy = end, False
        elif name == "cuGraphLaunch":
            if collect is not None:
                boundaries.append(collect)
            waiting, collect = False, None
    steps = []
    for first, last in zip(boundaries, boundaries[1:]):
        steps.append((first, last, [call for call in api if first <= call[0] < last]))
    return steps


def union(intervals):
    total, current = 0, None
    for start, end in sorted(intervals):
        if current is None or start > current[1]:
            if current is not None:
                total += current[1] - current[0]
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    return total + (current[1] - current[0] if current else 0)


def host_phases(calls, first, last):
    syncs = [(start, end) for start, end, _, name in calls if name == SYNC]
    graphs = [start for start, _, _, name in calls if name == "cuGraphLaunch"]
    logits = max(syncs, key=lambda sync: sync[1] - sync[0]) if syncs else (last, last)
    launch = graphs[0] if graphs else logits[0]
    memsets = [start for start, _, _, name in calls if name.startswith("cuMemset")]
    prepare = 0
    if memsets:
        ends = [end for start, end in syncs if memsets[0] <= start and end <= launch]
        prepare = max(ends) - memsets[0] if ends else 0
    return {
        "period": last - first,
        "pre_launch": launch - first,
        "launch_api": sum(end - start for start, end, _, name in calls if name in LAUNCHES),
        "logits_wait": logits[1] - logits[0],
        "after_logits": last - logits[1],
        "prepare_kv": prepare,
        "graph_launches": len(graphs),
        "kernel_launches": sum(1 for call in calls if call[3] in LAUNCHES[1:]),
        "memsets": len(memsets),
    }


def device_step(kernels, classes):
    """One device's kernels of one step: timing, charges and collectives."""
    kernels = sorted(kernels)
    charges = defaultdict(float)
    by_name = defaultdict(lambda: [0, 0.0])
    collectives = []
    latest = kernels[0][0]
    for start, end, _, _, name in kernels:
        advance = max(0, end - latest)
        label = kernel_class(name, classes)
        charges[label] += advance
        by_name[name][0] += 1
        by_name[name][1] += advance
        if label == "comm":
            collectives.append((latest, end))
        latest = max(latest, end)
    first, last = kernels[0][0], max(kernel[1] for kernel in kernels)
    busy = union([(kernel[0], kernel[1]) for kernel in kernels])
    return {"start": first, "end": last, "kernels": len(kernels), "span": last - first, "busy": busy,
            "inner_gap": last - first - busy, "charges": dict(charges), "names": dict(by_name),
            "collectives": collectives}


def collective_times(per_device):
    """(instances, intrinsic, mean skew wait) of one step, or None when it
    ran none or the devices ran different numbers of collectives."""
    lists = [data["collectives"] for data in per_device.values()]
    if not lists or not lists[0] or len({len(entry) for entry in lists}) != 1:
        return None
    intrinsic = skew = 0
    for instance in zip(*lists):
        arrivals = [arrive for arrive, _ in instance]
        intrinsic += max(0, max(exit for _, exit in instance) - max(arrivals))
        skew += statistics.mean(max(arrivals) - arrive for arrive in arrivals)
    return len(lists[0]), intrinsic, skew


def analyze(connection, long_sync_us=1000, skip=1, classes=CLASSES):
    api, kernels = load(connection)
    steps = host_steps(api, long_sync_us * 1000)[skip:]
    owner = {correlation: index for index, (_, _, calls) in enumerate(steps) for _, _, correlation, _ in calls}
    launched = [defaultdict(list) for _ in steps]
    for kernel in kernels:
        index = owner.get(kernel[3])
        if index is not None:
            launched[index][kernel[2]].append(kernel)
    records = []
    previous_end = {}
    for index, (first, last, calls) in enumerate(steps):
        per_device = {device: device_step(rows, classes) for device, rows in sorted(launched[index].items())}
        for device, data in per_device.items():
            data["idle_before"] = data["start"] - previous_end[device] if device in previous_end else None
            previous_end[device] = data["end"]
        records.append({"host": host_phases(calls, first, last), "devices": per_device,
                        "collectives": collective_times(per_device)})
    return records


def us(values):
    values = [value for value in values if value is not None]
    return round(statistics.median(values) / 1000, 1) if values else None


def summarize(records):
    """Medians per step in microseconds (charges are means, so they add up)."""
    if not records:
        return {"steps": 0}
    report = {"steps": len(records), "host_us": {}, "devices": {}}
    for key in records[0]["host"]:
        values = [record["host"][key] for record in records]
        count = key in ("graph_launches", "kernel_launches", "memsets")
        report["host_us"][key] = statistics.median(values) if count else us(values)
    report["page_work_step_fraction"] = round(
        sum(1 for record in records if record["host"]["memsets"]) / len(records), 3)
    report["prepare_kv_mean_us"] = round(
        statistics.mean(record["host"]["prepare_kv"] for record in records) / 1000, 1)
    devices = sorted({device for record in records for device in record["devices"]})
    for device in devices:
        rows = [record["devices"][device] for record in records if device in record["devices"]]
        labels = sorted({label for row in rows for label in row["charges"]})
        totals = defaultdict(lambda: [0, 0.0])
        for row in rows:
            for name, (count, charge) in row["names"].items():
                totals[name][0] += count
                totals[name][1] += charge
        report["devices"][str(device)] = {
            "kernels": statistics.median(row["kernels"] for row in rows),
            "span_us": us([row["span"] for row in rows]),
            "busy_us": us([row["busy"] for row in rows]),
            "inner_gap_us": us([row["inner_gap"] for row in rows]),
            "idle_before_us": us([row["idle_before"] for row in rows]),
            "charged_us": {label: round(statistics.mean(row["charges"].get(label, 0) for row in rows) / 1000, 1)
                           for label in labels},
            "top_kernels": [
                {"kernel": name, "per_step": round(count / len(rows), 1),
                 "charged_us": round(charge / len(rows) / 1000, 1)}
                for name, (count, charge) in sorted(totals.items(), key=lambda item: -item[1][1])[:TOP]
            ],
        }
    collectives = [record["collectives"] for record in records if record["collectives"]]
    if collectives:
        report["collectives"] = {
            "instances": statistics.median(entry[0] for entry in collectives),
            "intrinsic_us": us([entry[1] for entry in collectives]),
            "mean_skew_wait_us": us([entry[2] for entry in collectives]),
        }
    return report


TRACE_HOST = ("period_us", "execute_us", "bundle_execute_us", "device_wait_us", "outside_us", "schedule_us",
              "commit_us", "gap_us", "boundary_us", "upload_us", "launch_us", "sampling_us")


def step_trace(lines, window=None):
    """Decode-step summary of `chopinfer.step.v1` lines inside `window`
    ((start, end) unix ns, which needs time-prefixed lines)."""
    steps = []
    for line in lines:
        stamp, tab, body = line.partition("\t")
        if not tab:
            stamp, body = None, line
        if '"chopinfer.step.v1"' not in body:
            continue
        if window is not None and (stamp is None or not window[0] <= int(stamp) <= window[1]):
            continue
        step = json.loads(body)
        if step.get("prefill_tokens", 0) == 0:
            steps.append(step)
    if not steps:
        return {"steps": 0}
    composition = defaultdict(int)
    for step in steps:
        composition[f'decode {step["decode_rows"]}, verify {step["verify_items"]}x{step["verify_rows"]}'] += 1
    return {
        "steps": len(steps),
        "composition": dict(sorted(composition.items(), key=lambda item: -item[1])),
        "median_host_us": {key: statistics.median(step["host"][key] for step in steps)
                           for key in TRACE_HOST if all(key in step["host"] for step in steps)},
    }


def class_option(text):
    label, _, prefixes = text.partition("=")
    if not label or not prefixes:
        raise argparse.ArgumentTypeError("expected LABEL=PREFIX[,PREFIX...]")
    return label, tuple(prefixes.split(","))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sqlite", nargs="?")
    parser.add_argument("--step-trace")
    parser.add_argument("--window-ns", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument("--class", dest="classes", action="append", type=class_option, default=[],
                        help="kernel class LABEL=PREFIX[,PREFIX...], tried before the defaults")
    parser.add_argument("--long-sync-us", type=float, default=1000)
    parser.add_argument("--skip", type=int, default=1, help="leading steps to drop (default 1)")
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    if not args.sqlite and not args.step_trace:
        parser.error("give an export, --step-trace, or both")
    report = {}
    if args.sqlite:
        connection = sqlite3.connect(args.sqlite)
        report["nsys"] = summarize(analyze(connection, args.long_sync_us, args.skip, args.classes + CLASSES))
    if args.step_trace:
        with open(args.step_trace, errors="replace") as lines:
            report["step_trace"] = step_trace(lines, args.window_ns)
    text = json.dumps(report, indent=1)
    if args.json:
        with open(args.json, "w") as output:
            output.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

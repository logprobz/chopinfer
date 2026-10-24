#!/usr/bin/env python3
"""Where each serving step's time goes, from an Nsight Systems SQLite export
of a server (CUDA trace) and, optionally, its `CHOPINFER_STEP_TRACE` lines.

Steps are found by structure, not by how long anything waits. The engine
thread is the thread issuing the most CUDA API calls. Its launch calls
(cuGraphLaunch, cuLaunchKernel, cuLaunchKernelEx, cuLaunchCooperativeKernel)
are sampler launches when their kernels run on a sampler stream, a stream
carrying a kernel whose name starts with `--sampler-kernel` (the host
sampler's `chop_sampling_`), and bundle launches otherwise. A sampler collect
is a device-to-host copy on a sampler stream (CUPTI_ACTIVITY_KIND_MEMCPY),
complete when the engine thread's next synchronize call returns. A step ends
at the last collect before the next bundle launch. A step counts only if it
holds, in order, a bundle launch, a sampler launch after its last bundle
launch, and a collect; the others are reported as `rejected_steps`.

Each kernel belongs to the step of the API call that launched it
(correlationId), so graph-node kernels belong to the step whose cuGraphLaunch
issued them. Per step and device:
  span         first kernel start to last kernel end;
  busy         union of kernel intervals (PDL overlap counted once);
  inner gap    span minus busy, idle device time inside the step;
  idle before  the previous step's last kernel end to this step's first
               kernel start: device idle between steps.
At graph granularity (CUPTI_ACTIVITY_KIND_GRAPH_TRACE) a graph's record is
its elapsed envelope, not kernel time: it is reported as `graph_elapsed`, and
busy and inner gap are null (unavailable without node tracing) for a device
whose step holds a graph record.
Each kernel (or graph record) is charged the advance it causes in the
running latest end (as serving_step_nsys.py does), summed per kernel class
(`--class`); idle time inside the step goes to the record after it.

Collectives: the k-th collective kernel (`comm` class) of a step on every
device is one instance. Its dependency is the previous kernel on the same
stream; a device arrives at the later of that kernel's end and its own
collective's start, and its launch gap is how long its collective started
after that dependency ended. The instance's intrinsic time is the latest
exit minus the latest arrival, a device's skew wait the latest arrival minus
its own. An instance whose collective has no earlier kernel on its stream in
the step has an unknown dependency (arrival bounded by its start) and is
counted in `unknown_dependency`; cross-stream dependencies are not seen.

Host phases per step (they add up to the period):
  pre_launch    step start to the first bundle launch: commit, scheduling,
                batch preparation, uploads;
  bundle_issue  first bundle launch to the end of the last one;
  wait          end of the last bundle launch to the first sampler launch,
                of which device_wait is spent in synchronize calls;
  sampling      first sampler launch to the step's end (issue and collect).
Bundle launches include the commit's state-copy kernels, so pre_launch ends
at the first GPU work the step submits. Also launch_api (time in bundle
launch calls) and prepare_kv: from the first cuMemset* before the first
bundle launch to the last cuStreamSynchronize before it (the host KV pool's
page zeroing; the bundle's uploads wait on events), zero without memsets.

The step trace reader takes JSON lines (`chopinfer.step.v1`), each optionally
prefixed by a unix time in ns and a tab, and summarizes the decode steps
(no prefill tokens) inside `--window-ns`, if given.

Usage: serving_step_breakdown.py [EXPORT.sqlite] [--step-trace FILE]
       [--window-ns START END] [--class LABEL=PREFIX[,PREFIX...]]...
       [--sampler-kernel PREFIX] [--skip N] [--json OUT]
"""

import argparse
import json
import sqlite3
import sys
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
    ("sampling", ("chop_sampling_",)),
    ("graph", ("<graph>",)),
]
GRAPH = "<graph>"
SAMPLER = "chop_sampling_"
LAUNCHES = ("cuGraphLaunch", "cuLaunchKernel", "cuLaunchKernelEx", "cuLaunchCooperativeKernel")
SYNCS = ("cuStreamSynchronize", "cuEventSynchronize", "cuCtxSynchronize")
DTOH = 2  # CUPTI copyKind
TOP = 15


def kernel_class(name, classes):
    for label, prefixes in classes:
        if name.startswith(prefixes):
            return label
    return "other"


def base(name):
    """The driver API name without its version or stream suffix
    (cuMemcpyDtoHAsync_v2, cuStreamSynchronize_ptsz)."""
    return name.split("_", 1)[0]


def is_launch(name):
    return base(name) in LAUNCHES


def is_sync(name):
    return base(name) in SYNCS


def load(connection):
    """(engine-thread API calls, kernels, sampler copy correlations)."""
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
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    kernels = [(start, end, device, correlation, names[name], stream)
               for start, end, device, correlation, name, stream in
               connection.execute("SELECT start, end, deviceId, correlationId, shortName, streamId "
                                  "FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:
        kernels += [(start, end, device, correlation, GRAPH, stream) for start, end, device, correlation, stream in
                    connection.execute("SELECT start, end, deviceId, correlationId, streamId "
                                       "FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE")]
    kernels.sort()
    copies = []
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        copies = connection.execute("SELECT correlationId, deviceId, streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                                    "WHERE copyKind = ?", (DTOH,)).fetchall()
    return api, kernels, copies


def classify(api, kernels, copies, sampler_prefix):
    """Per engine-thread call: 'bundle', 'sampler', 'collect', 'sync',
    'memset' or None."""
    sampler_streams = {(device, stream) for _, _, device, _, name, stream in kernels
                       if name.startswith(sampler_prefix)}
    sampler_launches = {correlation for _, _, device, correlation, name, stream in kernels
                        if (device, stream) in sampler_streams}
    collects = {correlation for correlation, device, stream in copies if (device, stream) in sampler_streams}
    kinds = []
    for _, _, correlation, name in api:
        if is_launch(name):
            kinds.append("sampler" if correlation in sampler_launches else "bundle")
        elif base(name).startswith("cuMemcpyDtoH") and correlation in collects:
            kinds.append("collect")
        elif is_sync(name):
            kinds.append("sync")
        elif name.startswith("cuMemset"):
            kinds.append("memset")
        else:
            kinds.append(None)
    return kinds


def host_steps(api, kinds):
    """([(start, end, calls, kinds)], rejected): steps between the ends of
    the last sampler collect before each bundle launch."""
    boundaries = []
    collect_end = None  # end of the latest completed collect since the last bundle launch
    pending = None  # a collect copy waiting for its synchronize call
    for (start, end, _, _), kind in zip(api, kinds):
        if kind == "collect":
            pending = end
        elif kind == "sync" and pending is not None:
            collect_end, pending = end, None
        elif kind == "bundle":
            if pending is not None:  # a collect without a wait completes at its copy call
                collect_end, pending = pending, None
            if collect_end is not None:
                boundaries.append(collect_end)
            collect_end = None
    steps, rejected = [], 0
    index = 0
    for first, last in zip(boundaries, boundaries[1:]):
        while index < len(api) and api[index][0] < first:
            index += 1
        stop = index
        while stop < len(api) and api[stop][0] < last:
            stop += 1
        calls, step_kinds = api[index:stop], kinds[index:stop]
        if valid(step_kinds):
            steps.append((first, last, calls, step_kinds))
        else:
            rejected += 1
    return steps, rejected


def valid(kinds):
    """A bundle launch, then a sampler launch after the last bundle launch,
    then a collect."""
    bundles = [index for index, kind in enumerate(kinds) if kind == "bundle"]
    if not bundles:
        return False
    after = kinds[bundles[-1] + 1:]
    if "sampler" not in after:
        return False
    return "collect" in after[after.index("sampler"):]


def host_phases(calls, kinds, first, last):
    bundles = [index for index, kind in enumerate(kinds) if kind == "bundle"]
    sampler = next(index for index in range(bundles[-1] + 1, len(kinds)) if kinds[index] == "sampler")
    launch_start, launch_end = calls[bundles[0]][0], calls[bundles[-1]][1]
    sampling_start = calls[sampler][0]
    memsets = [index for index in range(bundles[0]) if kinds[index] == "memset"]
    prepare = 0
    if memsets:
        ends = [calls[index][1] for index in range(memsets[0], bundles[0])
                if base(calls[index][3]) == "cuStreamSynchronize"]
        prepare = max(ends) - calls[memsets[0]][0] if ends else 0
    return {
        "period": last - first,
        "pre_launch": launch_start - first,
        "bundle_issue": launch_end - launch_start,
        "wait": sampling_start - launch_end,
        "device_wait": sum(calls[index][1] - calls[index][0] for index in range(bundles[-1] + 1, sampler)
                           if kinds[index] == "sync"),
        "sampling": last - sampling_start,
        "launch_api": sum(calls[index][1] - calls[index][0] for index in bundles),
        "prepare_kv": prepare,
        "graph_launches": sum(1 for index in bundles if base(calls[index][3]) == "cuGraphLaunch"),
        "kernel_launches": sum(1 for index in bundles if base(calls[index][3]) != "cuGraphLaunch"),
        "sampler_launches": kinds.count("sampler"),
        "collects": kinds.count("collect"),
        "memsets": len(memsets),
    }


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


def device_step(kernels, classes):
    """One device's kernels of one step: timing, charges and collectives."""
    kernels = sorted(kernels)
    charges = defaultdict(float)
    by_name = defaultdict(lambda: [0, 0.0])
    collectives = []
    stream_end = {}
    latest = kernels[0][0]
    for start, end, _, _, name, stream in kernels:
        advance = max(0, end - latest)
        label = kernel_class(name, classes)
        charges[label] += advance
        by_name[name][0] += 1
        by_name[name][1] += advance
        if label == "comm":
            dependency = stream_end.get(stream)
            collectives.append({
                "arrive": start if dependency is None else max(start, dependency),
                "exit": end,
                "launch_gap": None if dependency is None else max(0, start - dependency),
            })
        stream_end[stream] = max(stream_end.get(stream, end), end)
        latest = max(latest, end)
    first, last = kernels[0][0], max(kernel[1] for kernel in kernels)
    graphs = [(kernel[0], kernel[1]) for kernel in kernels if kernel[4] == GRAPH]
    busy = None if graphs else union([(kernel[0], kernel[1]) for kernel in kernels])
    return {"start": first, "end": last, "kernels": len(kernels) - len(graphs), "graphs": len(graphs),
            "span": last - first, "busy": busy, "inner_gap": None if busy is None else last - first - busy,
            "graph_elapsed": union(graphs) if graphs else None,
            "charges": dict(charges), "names": dict(by_name), "collectives": collectives}


def collective_times(per_device):
    """Per step: instances, intrinsic, mean skew wait, mean launch gap and the
    instances with an unknown dependency; None when the step ran none or the
    devices ran different numbers of collectives."""
    lists = [data["collectives"] for data in per_device.values()]
    if not lists or not lists[0] or len({len(entry) for entry in lists}) != 1:
        return None
    intrinsic = skew = gap = 0
    unknown = 0
    for instance in zip(*lists):
        arrivals = [entry["arrive"] for entry in instance]
        intrinsic += max(0, max(entry["exit"] for entry in instance) - max(arrivals))
        skew += statistics.mean(max(arrivals) - arrive for arrive in arrivals)
        gaps = [entry["launch_gap"] for entry in instance]
        if None in gaps:
            unknown += 1
        else:
            gap += statistics.mean(gaps)
    return {"instances": len(lists[0]), "intrinsic": intrinsic, "skew_wait": skew, "launch_gap": gap,
            "unknown_dependency": unknown}


def analyze(connection, skip=1, classes=CLASSES, sampler_prefix=SAMPLER):
    """(per-step records, rejected step count)."""
    api, kernels, copies = load(connection)
    kinds = classify(api, kernels, copies, sampler_prefix)
    steps, rejected = host_steps(api, kinds)
    steps = steps[skip:]
    owner = {correlation: index for index, (_, _, calls, _) in enumerate(steps) for _, _, correlation, _ in calls}
    launched = [defaultdict(list) for _ in steps]
    for kernel in kernels:
        index = owner.get(kernel[3])
        if index is not None:
            launched[index][kernel[2]].append(kernel)
    records = []
    previous_end = {}
    for index, (first, last, calls, step_kinds) in enumerate(steps):
        per_device = {device: device_step(rows, classes) for device, rows in sorted(launched[index].items())}
        for device, data in per_device.items():
            data["idle_before"] = data["start"] - previous_end[device] if device in previous_end else None
            previous_end[device] = data["end"]
        records.append({"host": host_phases(calls, step_kinds, first, last), "devices": per_device,
                        "collectives": collective_times(per_device)})
    return records, rejected


def us(values):
    values = [value for value in values if value is not None]
    return round(statistics.median(values) / 1000, 1) if values else None


COUNTS = ("graph_launches", "kernel_launches", "sampler_launches", "collects", "memsets")


def summarize(records, rejected=0):
    """Medians per step in microseconds (charges are means, so they add up);
    null where a figure is unavailable."""
    report = {"steps": len(records), "rejected_steps": rejected}
    if not records:
        return report
    report["host_us"] = {}
    for key in records[0]["host"]:
        values = [record["host"][key] for record in records]
        report["host_us"][key] = statistics.median(values) if key in COUNTS else us(values)
    report["page_work_step_fraction"] = round(
        sum(1 for record in records if record["host"]["memsets"]) / len(records), 3)
    report["prepare_kv_mean_us"] = round(
        statistics.mean(record["host"]["prepare_kv"] for record in records) / 1000, 1)
    report["devices"] = {}
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
            "graphs": statistics.median(row["graphs"] for row in rows),
            "span_us": us([row["span"] for row in rows]),
            "busy_us": us([row["busy"] for row in rows]),
            "inner_gap_us": us([row["inner_gap"] for row in rows]),
            "graph_elapsed_us": us([row["graph_elapsed"] for row in rows]),
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
            "instances": statistics.median(entry["instances"] for entry in collectives),
            "intrinsic_us": us([entry["intrinsic"] for entry in collectives]),
            "mean_skew_wait_us": us([entry["skew_wait"] for entry in collectives]),
            "mean_launch_gap_us": us([entry["launch_gap"] for entry in collectives]),
            "unknown_dependency": statistics.median(entry["unknown_dependency"] for entry in collectives),
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
    parser.add_argument("--sampler-kernel", default=SAMPLER, help=f"sampler kernel prefix (default {SAMPLER})")
    parser.add_argument("--skip", type=int, default=1, help="leading steps to drop (default 1)")
    parser.add_argument("--json")
    parser.add_argument("--route-classes", help="route tag export from the profiled revision")
    parser.add_argument("--route-allow", help="justified route allowlist")
    args = parser.parse_args(argv)
    if not args.sqlite and not args.step_trace:
        parser.error("give an export, --step-trace, or both")
    report = {}
    if args.sqlite:
        connection = sqlite3.connect(args.sqlite)
        records, rejected = analyze(connection, args.skip, args.classes + CLASSES, args.sampler_kernel)
        report["nsys"] = summarize(records, rejected)
    if args.step_trace:
        with open(args.step_trace, errors="replace") as lines:
            report["step_trace"] = step_trace(lines, args.window_ns)
    text = json.dumps(report, indent=1)
    if args.json:
        with open(args.json, "w") as output:
            output.write(text + "\n")
    print(text)
    if args.route_classes:
        import subprocess
        from pathlib import Path
        command = [sys.executable, "-B", str(Path(__file__).resolve().parents[1] / "route_guard/check.py"), "--classes", args.route_classes]
        if args.sqlite: command += ["--nsys", args.sqlite]
        if args.step_trace: command += ["--steps", args.step_trace]
        if args.route_allow: command += ["--allow", args.route_allow]
        result = subprocess.run(command, check=False)
        if result.returncode: raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

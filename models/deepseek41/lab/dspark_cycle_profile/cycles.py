"""Join completed engine steps to Nsight and charge overlapping kernels once."""
import bisect
import collections
import json
from pathlib import Path
import sqlite3
import statistics

EMBED = "chop_row_embedding_deepseek41_tp4_v1"
MARKOV = "chop_deepseek41_dspark_markov_v1"
PHASES = ("target", "drafter", "catch_up", "sampling", "other_device")


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def overlap(a, b):
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def classify(rows, live_rows, verify_items):
    """Graph markers label complete graphs; eager intervals remain explicit."""
    draft_graphs = {r["graphId"] for r in rows if r["name"] == MARKOV
                    and r["graphId"] is not None}
    catch_graphs = {r["graphId"] for r in rows if "dspark_quantize_w15360_" in r["name"]
                    and r["graphId"] is not None and r["graphId"] not in draft_graphs}
    embeddings = [r for r in rows if r["name"] == EMBED]
    assert embeddings, "target embedding missing: incomplete capture or clock mismatch"
    target_begin = embeddings[-1]
    assert target_begin["gridX"] == live_rows, "target geometry differs from engine trace"
    heads = [r for r in rows if "head_f32_" in r["name"]
             and r["name"].endswith("_cached_reduce_v1") and r["start"] >= target_begin["start"]]
    assert len(heads) == 1, "target head missing or multiple target forwards in one step"
    target_end = heads[0]["end"]
    target_graphs = {r["graphId"] for r in rows
                     if target_begin["start"] <= r["start"] < target_end
                     and r["graphId"] is not None}
    assert not target_graphs & (draft_graphs | catch_graphs), "ambiguous graph phase"
    marks = [r for r in rows if r["name"] == MARKOV]
    assert bool(marks) == bool(verify_items), "drafter markers differ from verify items"
    draft_begin = embeddings[0]["start"] if marks else target_begin["start"]
    draft_end = max((h["end"] for h in rows if "head_f32_" in h["name"]
                     and h["name"].endswith("_cached_reduce_v1")
                     and h["start"] < target_begin["start"]), default=draft_begin)
    # Outside graphs, only known projection families before drafting count
    # as catch-up. Other kernels retain their own category and names.
    def eager_catch(name):
        return any(x in name for x in ("dspark_quantize_", "rope_compressed_positions",
                   "rope_window_positions", "latent_window_rope", "latent_window_store",
                   "latent_ceil32", "rmsnorm_deepseek41_f32_w5120")) or (
                       "mxfp8_" in name and any(x in name for x in ("_n5120_k15360_", "_n512_k5120_")))
    def phase(row):
        name, graph = row["name"], row["graphId"]
        if name.startswith(("chop_sampling_", "chop_dspark_argmax_")) or "dspark_" in name and "acceptance_" in name:
            return "sampling"
        if graph in draft_graphs:
            return "drafter"
        if graph in catch_graphs:
            return "catch_up"
        if graph in target_graphs or target_begin["start"] <= row["start"] < target_end:
            return "target"
        if marks and draft_begin <= row["start"] < target_begin["start"]:
            # Eager draft forward ends at its head. Hash/gather kernels
            # between draft and target remain other device work.
            if row["start"] < draft_end:
                return "drafter"
        if row["start"] < draft_begin and eager_catch(name):
            return "catch_up"
        return "other_device"
    return phase


def charge(rows, begin, end, phase):
    frontier = begin
    totals = dict.fromkeys(PHASES, 0)
    names = collections.Counter()
    busy = []
    for row in rows:
        a, b = max(begin, row["start"]), min(end, row["end"])
        if b <= a:
            continue
        amount = max(0, b - max(a, frontier))
        frontier = max(frontier, b)
        category = phase(row)
        totals[category] += amount
        if category == "other_device":
            names[row["name"]] += amount
        busy.append((a, b))
    busy = union(busy)
    assert sum(totals.values()) == sum(b-a for a, b in busy)
    gaps = []
    frontier = begin
    for a, b in busy:
        if a > frontier:
            gaps.append((frontier, a))
        frontier = b
    if frontier < end:
        gaps.append((frontier, end))
    totals["no_recorded_kernel"] = sum(b-a for a, b in gaps)
    assert sum(totals.values()) == end-begin
    return {k: v/1e6 for k, v in totals.items()}, gaps, names


def analyze(path, directory, *, historical=False):
    path, directory = Path(path), Path(directory)
    db = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA temp_store=MEMORY")
    epoch = db.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    rows = [dict(r) for r in db.execute("""SELECT k.start,k.end,k.deviceId,k.graphId,k.gridX,
        s.value name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
        ORDER BY k.start,k.end""")]
    assert {r["deviceId"] for r in rows} == {0, 1, 2, 3}
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    host = {table: union([(r[0], r[1]) for r in db.execute(f"SELECT start,end FROM {table}")])
            for table in ("CUPTI_ACTIVITY_KIND_DRIVER", "CUPTI_ACTIVITY_KIND_RUNTIME", "OSRT_API")
            if table in tables}
    copies = collections.defaultdict(list)
    for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table in tables:
            for r in db.execute(f"SELECT start,end,deviceId FROM {table}"):
                copies[r[2]].append((r[0], r[1]))
    copies = {device: union(intervals) for device, intervals in copies.items()}
    diagnostics = [dict(r) for r in db.execute("SELECT * FROM DIAGNOSTIC_EVENT")] if "DIAGNOSTIC_EVENT" in tables else []
    cases = {}
    for concurrency in (8, 16):
        source = directory / f"C{concurrency}" / "steps.jsonl"
        if historical and not source.exists():
            continue
        steps = [json.loads(line) for line in source.read_text().splitlines()]
        pairs = [(a, b) for a, b in zip(steps, steps[1:])
                 if b["batch"] == a["batch"] + 1 and not a["prefill_items"] and not b["prefill_items"]
                 and all(s["decode_rows"] + s["verify_items"] == concurrency for s in (a, b))]
        pairs = pairs[4:]
        assert len(pairs) >= 8, "fewer than eight complete full-concurrency cycles"
        devices = {}
        for device in range(4):
            rr = [r for r in rows if r["deviceId"] == device]
            starts = [r["start"] for r in rr]
            cycles = []
            for previous, step in pairs:
                begin, end = (s["completed_unix_ns"] - epoch for s in (previous, step))
                left, right = bisect.bisect_left(starts, begin), bisect.bisect_left(starts, end)
                current = rr[left:right]
                assert current and max(r["end"] for r in current) <= end, "clock or cycle boundary mismatch"
                capacity = step["row_capacity"]
                phase = classify(current, capacity, step["verify_items"])
                if not historical and device == 0 and step["verify_items"]:
                    assert any("dspark_" in r["name"] and "acceptance_" in r["name"] for r in current), "device acceptance kernel missing"
                amounts, gaps, names = charge(current, begin, end, phase)
                copy_ms = overlap(gaps, copies.get(device, []))/1e6
                cycles.append(dict(batch=step["batch"], rows=step["live_rows"], verify_items=step["verify_items"],
                    catch_up_rows=step.get("phase", {}).get("catch_up_rows"),
                    begin_ns=begin, end_ns=end, elapsed_ms=(end-begin)/1e6, phase_ms=amounts,
                    host_api_overlap_with_gaps_ms={k: overlap(gaps, v)/1e6 for k, v in host.items()},
                    copy_or_memset_overlap_with_gaps_ms=copy_ms,
                    no_recorded_device_activity_ms=amounts["no_recorded_kernel"]-copy_ms,
                    host_timers_us=step.get("host", {}),
                    other_kernels={name: value/1e6 for name, value in names.items()}))
            devices[str(device)] = dict(cycles=cycles, count=len(cycles),
                mean_cycle_ms=statistics.mean(c["elapsed_ms"] for c in cycles),
                phase_mean_ms={k: statistics.mean(c["phase_ms"][k] for c in cycles) for k in cycles[0]["phase_ms"]})
        cases[str(concurrency)] = dict(devices=devices)
    assert len(cases) == (1 if historical else 2)
    return dict(schema="chopinfer.cycle-attribution.v1", source=str(path), epoch_ns=epoch,
                cases=cases, capture_diagnostics=diagnostics, synthetic="CPU_FIXTURE" in tables,
                method="Engine completion-to-completion intervals, UTC aligned; exclusive kernel charge end-max(start,frontier).",
                limits=["No recorded kernel does not prove host idle; copies and untraced work may occupy gaps.",
                        "Host API overlap includes blocking and concurrent threads; categories overlap and must not be added.",
                        "Host sampling and commit timers are separate from GPU phase charges.",
                        "Profiler overhead and any collection warnings limit timing conclusions."])


def markdown(result):
    lines = ["Instrumented DSpark cycle attribution. These are profiling timings.", "",
             "| Policy | C | GPU | Cycles | Cycle ms | Target | Drafter | Catch-up | Sampling | Other | No kernel |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for server in result["servers"]:
        for c, case in server["attribution"]["cases"].items():
            for device, row in case["devices"].items():
                values = [row["phase_mean_ms"][k] for k in (*PHASES, "no_recorded_kernel")]
                lines.append(f"| {server['config']} | {c} | {device} | {row['count']} | {row['mean_cycle_ms']:.3f} | "
                             + " | ".join(f"{v:.3f}" for v in values) + " |")
    lines += ["", "The JSON retains individual rows, catch-up rows, host timers, API overlap with gaps, fallback rates, acceptance, process faults and collection diagnostics.",
              "No-kernel gaps are not a CPU acceptance or host-idle measurement. Review capture warnings before drawing timing conclusions."]
    return "\n".join(lines) + "\n"

"""Bracket the host-to-profiler clock offset with independent host clocks."""
import bisect
import json


def calibrate(metadata, directory, rows, points):
    epoch = metadata["utcEpochNs"]
    system = metadata.get("systemClockNs")
    if system is None:
        return dict(offset_ns=0, method="export UTC epoch", anchors=0)
    base = epoch - system
    precise = directory / "clock-anchors.json"
    if precise.exists():
        samples = json.loads(precise.read_text())
        lo = min(s["unix_before_ns"] - s["monotonic_ns"] - base for s in samples)
        hi = max(s["unix_after_ns"] - s["monotonic_ns"] - base for s in samples)
        method = "bracketed realtime/monotonic samples"
    else:
        # Older captures retained /proc/uptime at centisecond precision.
        samples = [s for p in directory.glob("C*/summary.json")
                   for s in json.loads(p.read_text())["host_load"]["samples"]]
        if not samples:
            return dict(offset_ns=0, method="export UTC epoch", anchors=0)
        offsets = [round(s["at"] * 1e9) - round(s["uptime_seconds"] * 1e9) - base
                   for s in samples]
        lo, hi = max(v - 10_000_000 for v in offsets), min(offsets)
        method = "recorded uptime bracket refined by sampling completion gaps"
    assert lo <= hi, "host clock anchors do not admit a constant offset"
    independent = [lo, hi]
    rr = [r for r in rows if r["deviceId"] == 0]
    starts = [r["start"] for r in rr]
    accepted = sorted(r["end"] for r in rr if "dspark_" in r["name"]
                      and "acceptance_" in r["name"])
    sampled = sorted(r["end"] for r in rr if r["name"].startswith("chop_sampling_")
                     and "_finish_" in r["name"])
    count = 0
    used = set()
    for point in points:
        end = point["completed_unix_ns"] - epoch
        markers = accepted if point["verify_items"] else sampled
        index = bisect.bisect_right(markers, end - independent[0]) - 1
        assert index >= 0, f"clock anchor for batch {point['batch']} has no completion marker"
        done = markers[index]
        assert done not in used, "one sampling completion matched multiple engine steps"
        used.add(done)
        following = bisect.bisect_left(starts, done)
        # Sampling is complete before the engine publishes this step. The
        # next kernel begins the following cycle. A single offset must put
        # every selected completion in this gap; never fit separate cycles.
        if following < len(starts):
            lo = max(lo, end - starts[following])
        hi = min(hi, end - done)
        count += 1
    assert count >= 8 and lo <= hi, "completion gaps disagree with the host clock bracket"
    return dict(offset_ns=(lo + hi) // 2, method=method, anchors=count,
                independent_offset_bounds_ns=independent, offset_bounds_ns=[lo, hi])

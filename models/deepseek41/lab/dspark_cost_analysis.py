"""Attribute graph admission to completed steps, including completion tails."""
import collections
import json


def records(text):
    for line in text.splitlines():
        try:
            yield json.loads(line[line.index("{"):])
        except ValueError:
            continue


def capture_hits(text, concurrency):
    pending, steps, misses = [], 0, []
    tails, eager_tails, eager_full, graph_full = 0, [], [], 0
    shapes = collections.defaultdict(lambda: dict(steps=0, eager_steps=0, eager_execution_us=0))
    for r in records(text):
        if r.get("schema") == "chopinfer.graph-cache.v1":
            pending.append(r)
        elif r.get("schema") == "chopinfer.step.v1":
            items = r["decode_rows"] + r["verify_items"]
            if not r["prefill_items"] and r["verify_items"]:
                eager = any(g.get("event") == "target-eager" for g in pending)
                captures = [g for g in pending if g.get("event") == "capture" and g["family"] == "target"]
                if items == concurrency:
                    steps += 1
                    graph_full += not eager
                    if captures:
                        misses.append(dict(batch=r.get("batch"), captures=captures))
                    if eager:
                        eager_full.append(r.get("execution_us", 0))
                elif items < concurrency:
                    tails += 1
                    key = (r.get("row_capacity"), items)
                    shapes[key]["steps"] += 1
                    if eager:
                        us = r.get("execution_us", 0)
                        eager_tails.append(us)
                        shapes[key]["eager_steps"] += 1
                        shapes[key]["eager_execution_us"] += us
            pending = []
    return dict(full_verify_steps=steps, target_capture_steps=len(misses),
                hit_rate=None if not steps else 1 - len(misses) / steps,
                capture_misses=misses, full_graph_steps=graph_full,
                full_eager_steps=len(eager_full), tail_verify_steps=tails,
                tail_eager_steps=len(eager_tails),
                tail_eager_rate=None if not tails else len(eager_tails) / tails,
                tail_eager_execution_ms=sum(eager_tails) / 1000,
                tail_eager_mean_ms=None if not eager_tails else sum(eager_tails) / len(eager_tails) / 1000,
                tail_shapes=[dict(rows=rows, items=items, **data) for (rows, items), data in sorted(shapes.items())],
                time_scope="Completed whole-step execution time, not isolated GPU target time.")


def replay_retention(text, capacity=2048, ranks=4):
    """Replay observed capture requests atomically without discarding warm keys.

    This checks admission and geometry coverage, not device outputs or timings.
    The recorded source has one target bank. Bank-aware traces must identify
    their bank explicitly before using this replay for multiple context banks.
    """
    held, denied, first, evictions = set(), collections.Counter(), {}, []
    recaptures = 0
    for line, r in enumerate(records(text)):
        if r.get("schema") != "chopinfer.graph-cache.v1":
            continue
        event = r.get("event")
        if event == "eviction-begin":
            evictions.append(r)
        if event not in ("capture", "target-eager", "eager"):
            continue
        family, width, items = r["family"], r["width"], r["items"]
        key = (family, width, items)
        if event == "capture":
            first.setdefault(key, dict(record=line, epoch=r["epoch"]))
            if key in held and r["rank"] == 0 and r["segment"] == 0:
                recaptures += 1
        handles = (3 if family == "target" else 1) * ranks
        used = sum((3 if k[0] == "target" else 1) * ranks for k in held)
        if key not in held:
            if used + handles <= capacity:
                held.add(key)
            elif r["rank"] == 0 and r.get("segment", 0) == 0:
                denied[key] += 1
    return dict(capacity=capacity, retained_handles=sum((3 if k[0] == "target" else 1) * ranks for k in held),
                observed_evictions=evictions, avoided_recapture_groups=recaptures,
                protected_32_by_16=("target", 32, 16) in held,
                first_32_by_16=first.get(("target", 32, 16)),
                denied=[dict(family=f, rows=w, items=i, attempts=n) for (f, w, i), n in sorted(denied.items())],
                bank_scope="Single target bank in the recorded native snapshot.")

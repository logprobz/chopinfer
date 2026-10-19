"""Check complete pair-job coverage and summarize measured query or MMA evidence."""
import copy
import json
import math
from pathlib import Path
import re
import sys

ROWS = [1, 2, 4, 6, 8, 16, 32]
OFFSETS = [0, 32768, 131072]


def records(text, marker):
    return [json.loads(line.split(marker, 1)[1]) for line in text.splitlines() if marker in line]


def finite(value):
    assert isinstance(value, (int, float)) and math.isfinite(value), value


def validate(kind, texts):
    assert len(texts) == 2
    result = []
    for shard, text in enumerate(texts):
        summaries = re.findall(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored", text)
        assert summaries == [("ok", "1", "0", "0")], summaries
        rows = records(text, "QUERY_CASE " if kind == "query" else "MMA_BLOCK ")
        for row in rows:
            for field in ["old_ms", "new_ms"]:
                finite(row[field])
                assert row[field] > 0
            assert row["exclusive"] is True
        if kind == "query":
            assert [r["rows"] for r in rows] == ROWS[shard::2]
            assert all(r["comparisons"] == 16 and r["negative_control_rejected"] is True for r in rows)
            expected = len(rows) * 16
            assert re.findall(r"QUERY_COMPLETE shard=(\d+) comparisons=(\d+)", text) == [(str(shard), str(expected))]
        else:
            expected = {(False, i, n, 0, ratio) for i, n in enumerate(ROWS) if i % 2 == shard for ratio in [1, 2]}
            expected |= {(True, first, 64, offset, ratio) for offset in OFFSETS for ratio in [1, 2]
                         for first in range(shard * 4096, (shard + 1) * 4096, 64)}
            keys = [(r["prefill"], r["first"], r["rows"], r["offset"], r["ratio"]) for r in rows]
            assert len(keys) == len(set(keys)) and set(keys) == expected
            assert re.findall(r"MMA_COMPLETE shard=(\d+) blocks=(\d+)", text) == [(str(shard), str(len(expected)))]
            for r in rows:
                assert r["serving"] is False and r["samples"] > 0
                control = not (r["prefill"] and r["first"] == 0 and r["offset"] == 0 and r["ratio"] == 2)
                assert r["negative_control_rejected"] == control
                for field in ["squared_error", "max_abs_error", "signed_error", "max_bound_fraction"]:
                    assert len(r[field]) == 2
                    for x in r[field]:
                        finite(x)
                assert all(0 <= x <= 1 for x in r["max_bound_fraction"])
                assert r["reference_squared"] >= 0
                assert r["score_bit_mismatches"] >= 0 and r["selection_mismatches"] >= 0
        result.extend(rows)
    if kind == "query":
        return {"query_shapes": len(result), "bitwise_comparisons": sum(r["comparisons"] for r in result),
                "rows": sorted(result, key=lambda r: r["rows"]), "gpu_measured": True}
    chunks = []
    for offset in OFFSETS:
        for ratio in [1, 2]:
            group = [r for r in result if r["prefill"] and r["offset"] == offset and r["ratio"] == ratio]
            assert sum(r["rows"] for r in group) == 8192
            samples = sum(r["samples"] for r in group)
            reference = sum(r["reference_squared"] for r in group)
            squared = [sum(r["squared_error"][arm] for r in group) for arm in [0, 1]]
            chunks.append({"offset": offset, "ratio": ratio, "rows": 8192, "f64_samples": samples,
                           "old_ms": sum(r["old_ms"] for r in group), "new_ms": sum(r["new_ms"] for r in group),
                           "score_bit_mismatches": sum(r["score_bit_mismatches"] for r in group),
                           "selection_mismatches": sum(r["selection_mismatches"] for r in group),
                           "rmse": [math.sqrt(x / samples) for x in squared],
                           "relative_l2": [math.sqrt(x / reference) if reference else 0 for x in squared],
                           "max_abs_error": [max(r["max_abs_error"][arm] for r in group) for arm in [0, 1]],
                           "mean_error": [sum(r["signed_error"][arm] for r in group) / samples for arm in [0, 1]],
                           "max_bound_fraction": [max(r["max_bound_fraction"][arm] for r in group) for arm in [0, 1]]})
    return {"blocks": len(result), "prefill_chunks": chunks, "decode": [r for r in result if not r["prefill"]],
            "eligible_for_serving": False, "reason": "The quantizer-valid arithmetic witness exceeds the current build's error.",
            "gpu_measured": True}


def synthetic(kind):
    texts = []
    for shard in [0, 1]:
        lines = []
        if kind == "query":
            for n in ROWS[shard::2]:
                lines.append("QUERY_CASE " + json.dumps(dict(rows=n, old_ms=1, new_ms=0.1, comparisons=16,
                                                            negative_control_rejected=True, exclusive=True)))
            lines.append(f"QUERY_COMPLETE shard={shard} comparisons={16*len(lines)}")
        else:
            keys = [(False, i, n, 0, ratio) for i, n in enumerate(ROWS) if i % 2 == shard for ratio in [1, 2]]
            keys += [(True, first, 64, offset, ratio) for offset in OFFSETS for ratio in [1, 2]
                     for first in range(shard * 4096, (shard + 1) * 4096, 64)]
            for prefill, first, n, offset, ratio in keys:
                control = not (prefill and first == 0 and offset == 0 and ratio == 2)
                row = dict(prefill=prefill, first=first, rows=n, offset=offset, ratio=ratio, old_ms=1, new_ms=0.1,
                           exclusive=True, serving=False, samples=8, squared_error=[0, 0.1], max_abs_error=[0, 0.1],
                           signed_error=[0, 0.1], max_bound_fraction=[0, 0.1], reference_squared=1,
                           score_bit_mismatches=1, selection_mismatches=0, negative_control_rejected=control)
                lines.append("MMA_BLOCK " + json.dumps(row))
            lines.append(f"MMA_COMPLETE shard={shard} blocks={len(keys)}")
        texts.append("test gpu_case ... " + "\n".join(lines) + "\ntest result: ok. 1 passed; 0 failed; 0 ignored\n")
    return texts


def self_test():
    for kind in ["query", "mma"]:
        texts = synthetic(kind)
        summary = validate(kind, texts)
        assert summary["query_shapes" if kind == "query" else "blocks"] == (7 if kind == "query" else 782)
        for changed in [texts[0].replace("0 failed", "1 failed"), "\n".join(texts[0].splitlines()[1:]),
                        texts[0].replace('"old_ms": 1', '"old_ms": NaN', 1),
                        texts[0].replace('"negative_control_rejected": true', '"negative_control_rejected": false', 1)]:
            bad = copy.copy(texts)
            bad[0] = changed
            try:
                validate(kind, bad)
            except AssertionError:
                pass
            else:
                raise AssertionError("post-check accepted incomplete or invalid evidence")
    print("post-check self-test passed: prefixed markers, complete shards, failures and nonfinite timings")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        kind, directory = sys.argv[1:]
        assert kind in ["query", "mma"]
        root = Path(directory)
        result = validate(kind, [(root / f"{kind}-{shard}.log").read_text() for shard in [0, 1]])
        (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"{kind} complete; summary: {root / 'summary.json'}")

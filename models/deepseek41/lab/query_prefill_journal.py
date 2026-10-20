"""Prove query fusion arguments, launch deltas and unchanged host lifetimes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from audit_prefill_gate_journals import normalized
from verify_hc_journal import align_buffers, align_streams, config, launches, require

NORM = "chop_deepseek41_q_norm_f32_v1"
CEIL = "chop_deepseek41_dense_ceil_f32_w1280_prefill_w8_v1"
FUSED = "chop_prefill_query_norm_ceil_cached_v1"
REREAD = "chop_prefill_query_norm_ceil_reread_v1"


def project(entries, rows):
    result, count, at = [], 0, 0
    while at < len(entries):
        norm = entries[at]
        if rows != 8192 or norm["symbol"] != NORM:
            result.append(norm)
            at += 1
            continue
        require(at + 1 < len(entries), "incomplete norm/ceil pair")
        ceil = entries[at + 1]
        require(ceil["symbol"] == CEIL, "ceil must immediately consume the norm")
        require(norm["config"] == ceil["config"] == config(rows, 1, 256), "original grid changed")
        a, b = norm["args"], ceil["args"]
        require(len(a) == 9 and len(b) == 6, "original ABI changed")
        require(a[2:4] == ["NullDevicePointer"] * 2, "unexpected norm residual or gate")
        require(a[4] == b[0] and a[5] == b[3] and a[7:] == b[4:],
                "norm/ceil output, live or status binding differs")
        result.append(dict(symbol=FUSED, config=norm["config"],
                           args=[a[0], a[1], a[4], "NullDevicePointer",
                                 b[1], b[2], a[5], a[7], a[8], a[6]]))
        count += 1
        at += 2
    return result, count


def audit(before, after, ranks, rows):
    left, right = before.splitlines(), after.splitlines()
    resolves = lambda lines: Counter(l.split()[2] for l in lines if l.startswith("resolve "))
    a, b = resolves(left), resolves(right)
    require(not a-b and b-a == Counter({FUSED: ranks, REREAD: ranks}), "module entry delta differs")
    require(right.count("release") - left.count("release") == 3*ranks, "resource release delta differs")
    # Resource and opaque handle normalization is independent of launch rewriting.
    a, _ = normalized(left)
    b, _ = normalized(right)
    require([l for l in a if not l.startswith("launch ")]
            == [l for l in b if not l.startswith("launch ")],
            "allocations, copies, events, capture or lifecycle changed")
    a = launches(before)
    b = launches(align_streams(before, after))
    require(a.keys() == b.keys(), "stream set differs")
    sites = 0
    for stream, old in a.items():
        expected, count = project(old, rows)
        require(expected == b[stream], f"launch operands or ordering changed on stream {stream}")
        sites += count
    require(sites == (40*ranks if rows == 8192 else 0), "wrong full-chunk boundary count")
    return {"ranks": ranks, "rows": rows, "launches_saved_per_rank": sites//ranks,
            "before_launches_per_rank": sum(map(len, a.values()))//ranks,
            "after_launches_per_rank": sum(map(len, b.values()))//ranks,
            "all_other_records_equal": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for ranks in (1, 2, 4):
        for rows in (64, 8192):
            name = f"query-tp{ranks}-rows{rows}.journal"
            old, new = args.before/name, args.after/name
            metadata = lambda p: json.loads(p.with_name(p.name+".handles.json").read_text())
            before = old.read_text()
            after = align_buffers(metadata(old), metadata(new), new.read_text())
            report = audit(before, after, ranks, rows)
            negatives = 0
            if rows == 8192:
                line = next(l for l in after.splitlines() if l.startswith("launch ") and FUSED in l)
                for field in ("grid_x", "block_x", "offset", "length"):
                    changed, count = re.subn(r"("+field+r": )(\d+)",
                        lambda m: m[1]+str(int(m[2])+1), line, count=1)
                    require(count == 1, "negative control did not mutate a field")
                    try:
                        audit(before, after.replace(line, changed, 1), ranks, rows)
                    except ValueError:
                        negatives += 1
                    else:
                        raise ValueError("changed launch escaped the proof")
            reports.append(dict(report, negative_controls=negatives))
    args.output.write_text(json.dumps(reports, indent=2) + "\n")
    print("QUERY_PREFILL_JOURNAL pairs=6 negatives=12 full_chunk_saved_per_rank=40 PASS")


if __name__ == "__main__":
    main()

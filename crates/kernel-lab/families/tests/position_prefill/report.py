#!/usr/bin/env python3
"""Validate complete paired prefill coverage and sum per-kernel chunk times."""
import copy
import json
import math
from pathlib import Path
import re
import sys

OFFSETS = (0, 8192, 24576, 122880)


def markers(text, marker):
    result = []
    for line in text.splitlines():
        at = line.find(marker + " ")
        if at >= 0:
            result.append(json.loads(line[at + len(marker) + 1:]))
    return result


def validate(logs):
    cases = {}
    for shard, text in enumerate(logs):
        assert re.findall(r"test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;", text) == [("ok", "4", "0", "0")]
        assert not re.search(r"\bskipp(?:ed|ing)\b", text, re.I)
        assert not re.search(r"^test .* \.\.\. FAILED\s*$|^failures:|^fatal runtime error:|^thread .* panicked", text, re.M)
        assert text.count("arithmetic negative control rejected on both old and live GPU paths") == 1
        assert text.count("completed 129 poisoned score/select failure comparisons") == 1
        assert re.findall(r"PREFILL_COMPLETE shard=(\d+) cases=(\d+) rows_per_case=(\d+)", text) == [(str(shard), "16", "4096")]
        expected = {(offset, bf16, ratio) for offset in OFFSETS for bf16 in (False, True) for ratio in (1, 2)}
        observed = markers(text, "PREFILL_CASE")
        assert len(observed) == 16
        for case in observed:
            key = (case["offset"], case["bf16"], case["ratio"])
            assert key in expected
            expected.remove(key)
            assert case["shard"] == shard and case["first"] == shard * 4096
            assert case["rows"] == 4096 and case["chunk_rows"] == 8192
            assert case["bitwise"] is True and case["dense_blocks"] == 64
            compact = case["ratio"] == 1
            assert case["candidate_counts"] == ([2048, 1024] if compact else [])
            assert case["compact_rows"] == (4096 if compact else 0)
            assert case["negative_controls"] == (3 if compact else 1)
            assert len(case["kernel_ms"]) == (20 if compact else 5)
            assert all(math.isfinite(ms) and ms > 0 for ms in case["kernel_ms"].values())
            assert len(case["dense_score_select_ms"]) == 2
            assert all(math.isfinite(ms) and ms > 0 for ms in case["dense_score_select_ms"])
            for name in ("candidate_ms", "compact_score_select_ms"):
                assert len(case[name]) == 2 and all(len(pair) == 2 for pair in case[name])
                assert all(math.isfinite(ms) and (ms > 0 if compact else ms == 0) for pair in case[name] for ms in pair)
            assert math.isfinite(case["wall_ms"]) and case["wall_ms"] > 0
            cases[(shard,) + key] = case
        assert not expected
        blocks = markers(text, "PREFILL_BLOCK")
        expected_blocks = {(offset, bf16, ratio, first) for offset in OFFSETS for bf16 in (False, True) for ratio in (1, 2) for first in range(shard * 4096, (shard + 1) * 4096, 64)}
        assert len(blocks) == 1024
        for block in blocks:
            key = (block["offset"], block["bf16"], block["ratio"], block["first"])
            assert key in expected_blocks
            expected_blocks.remove(key)
            assert block["shard"] == shard and block["rows"] == 64 and block["bitwise"] is True
        assert not expected_blocks
    chunks = []
    for offset in OFFSETS:
        for bf16 in (False, True):
            for ratio in (1, 2):
                pair = [cases[(shard, offset, bf16, ratio)] for shard in (0, 1)]
                assert pair[0]["kernel_ms"].keys() == pair[1]["kernel_ms"].keys()
                chunks.append({
                    "offset": offset, "bf16": bf16, "ratio": ratio, "rows": 8192,
                    "dense_score_select_ms": [sum(c["dense_score_select_ms"][arm] for c in pair) for arm in (0, 1)],
                    "candidate_ms": [[sum(c["candidate_ms"][budget][arm] for c in pair) for arm in (0, 1)] for budget in (0, 1)],
                    "compact_score_select_ms": [[sum(c["compact_score_select_ms"][budget][arm] for c in pair) for arm in (0, 1)] for budget in (0, 1)],
                    "kernel_ms": {key: sum(c["kernel_ms"][key] for c in pair) for key in pair[0]["kernel_ms"]},
                })
    return {"shard_cases": 32, "dense_blocks": 2048, "chunks": chunks, "bitwise": True}


def fixture():
    logs = []
    for shard in (0, 1):
        lines = ["test arithmetic ... arithmetic negative control rejected on both old and live GPU paths",
                 "test failures ... completed 129 poisoned score/select failure comparisons"]
        for offset in OFFSETS:
            for bf16 in (False, True):
                for ratio in (1, 2):
                    case = {"offset": offset, "shard": shard, "bf16": bf16, "ratio": ratio,
                            "first": shard * 4096, "rows": 4096, "chunk_rows": 8192,
                            "dense_blocks": 64, "candidate_counts": [2048, 1024] if ratio == 1 else [],
                            "compact_rows": 4096 if ratio == 1 else 0, "negative_controls": 3 if ratio == 1 else 1,
                            "bitwise": True, "dense_score_select_ms": [1., 1.],
                            "candidate_ms": [[float(ratio == 1)] * 2 for _ in range(2)],
                            "compact_score_select_ms": [[float(ratio == 1)] * 2 for _ in range(2)],
                            "kernel_ms": {str(i): 1. for i in range(20 if ratio == 1 else 5)}, "wall_ms": 1.}
                    lines.append("libtest prefix: PREFILL_CASE " + json.dumps(case))
                    for first in range(shard * 4096, (shard + 1) * 4096, 64):
                        lines.append("libtest prefix: PREFILL_BLOCK " + json.dumps({"offset": offset, "shard": shard, "bf16": bf16, "ratio": ratio, "first": first, "rows": 64, "bitwise": True}))
        lines.extend([f"libtest prefix: PREFILL_COMPLETE shard={shard} cases=16 rows_per_case=4096",
                      "test result: ok. 4 passed; 0 failed; 0 ignored; 5 filtered out;"])
        logs.append("\n".join(lines))
    return logs


def self_test():
    valid = fixture()
    assert len(validate(valid)["chunks"]) == 16
    replacements = [("PREFILL_BLOCK", "OMITTED_BLOCK"), ("\"negative_controls\": 3", "\"negative_controls\": 0"),
                    ("4 passed; 0 failed", "3 passed; 1 failed"), ("\"wall_ms\": 1.0", "\"wall_ms\": NaN")]
    for before, after in replacements:
        bad = copy.copy(valid)
        bad[0] = bad[0].replace(before, after, 1)
        try:
            validate(bad)
        except (AssertionError, KeyError):
            pass
        else:
            raise AssertionError(f"accepted invalid log: {before}")
    print("prefill parser self-test passed: prefixed markers accepted; omissions, failures, missing controls and nonfinite times rejected")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    else:
        root = Path(sys.argv[1])
        report = validate([(root / f"gpu-{shard}.log").read_text() for shard in (0, 1)])
        (root / "prefill-report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("PREFILL_VERIFIED chunks=16 rows_per_chunk=8192 shard_cases=32 dense_blocks=2048")

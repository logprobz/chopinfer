"""Extract a measured C1 cost/acceptance replay without storing traces in source."""
import argparse
import hashlib
import json
from pathlib import Path

import analysis


def window(directory):
    receipts = list((directory / "decode-c1/receipts").glob("*-bench.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["passed"] and receipt["concurrencies"] == [1]
    samples = receipt["host_load"]["samples"]
    begin, end = samples[0]["at"] * 1e9, samples[-1]["at"] * 1e9
    blocks, rows = [], []
    log = directory / "server.log"
    for line in log.read_text().splitlines():
        at = line.find("{")
        if at < 0:
            continue
        try:
            row = json.loads(line[at:])
        except ValueError:
            continue
        if row.get("schema") == "chopinfer.deepseek41.drafter-confidence.v1":
            blocks.append(row)
        elif row.get("schema") == "chopinfer.step.v1":
            if begin <= row["completed_unix_ns"] <= end:
                assert len(blocks) == row["verify_items"]
                assert sum(b["verify_rows"] for b in blocks) == row["verify_rows"]
                assert all(b["completion"] == "committed" for b in blocks)
                rows.append(dict(row, blocks=blocks))
            blocks = []
    assert rows
    return receipt, rows, [receipts[0], log]


def extract(run, output):
    ar, ar_rows, ar_sources = window(run / "head/tp4-ar")
    draft, draft_rows, draft_sources = window(run / "head-device-results/tp4-dspark")
    audit = analysis.coverage(draft_rows, draft["speculation"]["delta"])
    assert audit["per_item_exact"], "lower bounds must match independent exact counters"
    ar_us = [r["execution_us"] for r in ar_rows if not r["prefill_items"]]
    assert len(ar_us) >= 100
    assert all(r["decode_rows"] == 1 for r in ar_rows if not r["prefill_items"])
    observations = []
    for row in draft_rows:
        if row["prefill_items"]:
            continue
        assert row["verify_items"] == 1 and row["decode_rows"] == 0
        block, = row["blocks"]
        observations.append(dict(
            micros=row["execution_us"], drafts=block["emitted_positions"],
            accepted=block["outcomes"].count("accepted"), batch=row["batch"],
            request=block["request_id"], live_rows=row["live_rows"],
            reported_capacity=row["row_capacity"]))
    assert len(observations) >= 64
    result = dict(schema="chopinfer.low-concurrency-replay.v1", concurrency=1,
                  ar_micros=ar_us, observations=observations, audit=audit,
                  sampled_recipe=draft["sampling"],
                  served_ar=ar["decode"][0]["cohort"],
                  served_draft=draft["decode"][0]["cohort"],
                  sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                           for p in ar_sources + draft_sources],
                  scope="Recorded C1 cost feedback with exact item acceptance, replayed through adaptive planning. "
                        "Observed feedback is retained even if the planner proposes another positive width. "
                        "This is an offline no-fallback regression, not a new served timing result or a C4 measurement.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "low-concurrency.json").write_text(json.dumps(result, indent=2) + "\n")
    print("LOW_CONCURRENCY_TRACE_OK", len(ar_us), len(observations), audit["per_item_exact"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    extract(args.run, args.output)

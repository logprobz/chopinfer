"""Compare fixed-workload speculative decoding receipts from AR and speculative arms."""

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from datetime import datetime, timezone
import json

from common import compared_timing, receipt_root, startup


ARMS = ("chopinfer-ar", "chopinfer-spec")


def latest(directory, arm, label, run_id):
    for path in reversed(sorted(Path(directory).glob(f"*-spec-bench-{arm}.json"))):
        data = json.loads(path.read_text())
        if data.get("label") == label and data.get("run_id") == run_id:
            return path, data
    return None, None


def cohort(receipt, category, concurrency):
    return next((row for row in receipt["cohorts"]
                 if row["category"] == category and row["concurrency"] == concurrency), None)


def fmt(value, digits=2):
    return "n/a" if value is None else f"{value:,.{digits}f}"


def position_rates(receipt):
    drafted = {}
    accepted = {}
    for row in receipt["cohorts"]:
        counters = row["speculation"]
        if not counters["available"]:
            continue
        for position, count in counters["drafted_positions"].items():
            drafted[int(position)] = drafted.get(int(position), 0) + count
        for position, count in counters["accepted_positions"].items():
            accepted[int(position)] = accepted.get(int(position), 0) + count
    return {position: accepted.get(position, 0) / count if count else None
            for position, count in sorted(drafted.items())}


def compare(receipts, output):
    """Write a table only when every arm passed the identical request bodies."""
    data = {arm: receipt for arm, (_, receipt) in receipts.items()}
    if any(not row["passed"] for row in data.values()):
        raise ValueError("both arm receipts must pass before comparison")
    settings = {(row["workload_sha256"], row["output_tokens"], tuple(row["concurrencies"]))
                for row in data.values()}
    if len(settings) != 1:
        raise ValueError("arm workloads differ: request hashes or token budgets do not match")
    if any(any(request["completion_tokens"] != row["output_tokens"]
                   or request["finish_reason"] != "length"
                   for group in row["cohorts"] for request in group["requests"])
           for row in data.values()):
        raise ValueError("a request missed its exact output length")
    # Throughput and latency count as evidence only when both arms' cohorts
    # ran on an idle host (common.HOST_LOAD_RULE).
    timing = compared_timing({arm: data[arm].get("host_load") for arm in ARMS})
    reasons = f" ({'; '.join(timing['reasons'])})" if timing["reasons"] else ""
    lines = [f"# Speculation comparison ({data['chopinfer-ar']['label']}, {data['chopinfer-ar']['run_id']})",
             "", f"Workload SHA-256: `{data['chopinfer-ar']['workload_sha256']}`. "
             "Every request used greedy decoding and reached the fixed output length.", "",
             f"Throughput and latency timing: {timing['timing']}{reasons}.", "",
             "| Workload | C | Metric | Chopinfer AR | Chopinfer spec |",
             "|---|---:|---|---:|---:|"]
    for category in ("gsm8k", "chat"):
        for concurrency in data["chopinfer-ar"]["concurrencies"]:
            groups = [cohort(data[arm], category, concurrency) for arm in ARMS]
            if any(group is None for group in groups):
                raise ValueError(f"missing {category} C{concurrency} cohort")
            if any(len(group["requests"]) != concurrency for group in groups):
                raise ValueError(f"{category} C{concurrency} request count differs")
            for label, extract, digits in (
                ("aggregate tok/s", lambda row: row["aggregate_tokens_per_second"], 1),
                ("mean ms/token", lambda row: row["mean_ms_per_token"], 2),
                ("accepted / drafted", lambda row: row["speculation"].get("acceptance_rate"), 3),
                ("tokens / verify step", lambda row: row["speculation"].get("tokens_per_step"), 3),
                ("tokens / verify step within output",
                 lambda row: row["speculation"].get("acceptance_length_within_output"), 3),
                ("emitted tokens / step", lambda row: row["speculation"].get("emitted_tokens_per_step"), 3),
            ):
                values = [fmt(extract(group), digits) for group in groups]
                lines.append(f"| {category} | {concurrency} | {label} | " + " | ".join(values) + " |")
    lines += ["", "Mean acceptance length includes one bonus token.", "",
              "| Draft position | Chopinfer spec |", "|---:|---:|"]
    for position, rate in sorted(position_rates(data["chopinfer-spec"]).items()):
        lines.append(f"| {position} | {fmt(rate, 3)} |")
    lines += ["", "| Receipt | Endpoint |", "|---|---|"]
    lines += [f"| `{receipts[arm][0].name}` | {data[arm]['server']['url']} |" for arm in ARMS]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="exclusive")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--receipts-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    args.receipts_dir = args.receipts_dir or receipt_root()
    args.output_dir = args.output_dir or receipt_root()
    startup(directory=args.output_dir)
    found = {arm: latest(args.receipts_dir, arm, args.label, args.run_id) for arm in ARMS}
    missing = [arm for arm, (_, data) in found.items() if data is None]
    if missing:
        raise SystemExit(f"missing receipt(s) for {args.run_id}: {', '.join(missing)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = compare(found, args.output_dir / f"spec-{stamp}-{args.run_id}.md")
    print(path)


if __name__ == "__main__":
    main()

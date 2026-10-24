#!/usr/bin/env python3
"""One arm of the step-overlap A/B (CHOPINFER_STEP_OVERLAP), and the comparison.

`run` drives a server started with CHOPINFER_STEP_TRACE=1: C1 requests one at
a time, then each batched concurrency `--repeats` times, all greedy and
streaming with logprobs=1 as bench.py's decode does, keeping every request's
token IDs and logprobs. It reads the server log's chopinfer.step.v1 lines
written during each pass and summarizes the host timings of the steps that
ran the whole cohort. `compare` puts an overlap-off and an overlap-on arm
side by side: throughput, the host-side step boundary, and greedy equality.
Each arm records the host load before each pass and at its end
(common.host_load); the comparison's timing is informational unless both arms
ran on an idle host, which never changes whether it passes.
C1 outputs must be identical; batched outputs are compared against the off
arm's agreement with its own repeat, since batch composition there follows
arrival timing. Every request must end its stream with [DONE], a finish
reason and usage, and no error; C1 compares token IDs, logprobs, text,
finish reasons and usage.

    python3 tools/qual/step_overlap_ab.py run --url URL --arm mimo-ar-off --log SERVER_LOG --out ARM.json
    python3 tools/qual/step_overlap_ab.py compare OFF.json ON.json
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.request

from common import Client, arguments, compared_timing, host_load, host_load_sample

RECORDED = ("index", "token_ids", "logprobs", "text", "finish_reason", "usage", "done",
            "error", "complete", "ttft_s")
HOST_FIELDS = ("boundary_us", "gap_us", "period_us", "outside_us", "schedule_us",
               "execute_us", "commit_us", "device_wait_us", "deferred_us")


def prompt(index):
    return f"Continue this numbered list from {index}: 1, 2, 3, 4,"


def read_stream(client, body):
    """The stream's (arrival, event) pairs and whether it ended with [DONE];
    an error event or a failed request is returned rather than raised."""
    headers = {"Content-Type": "application/json"}
    if key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(client.url + "/v1/completions",
                                     data=json.dumps(body).encode(), headers=headers)
    events, done = [], False
    try:
        with urllib.request.urlopen(request, timeout=client.timeout) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                events.append((time.monotonic(), json.loads(data)))
    except Exception as error:  # the arm records the failure
        return events, done, f"{type(error).__name__}: {error}"
    return events, done, None


def stream_request(client, model, index, output_tokens):
    body = {"model": model, "prompt": prompt(index), "max_tokens": output_tokens,
            "temperature": 0, "logprobs": 1, "return_token_ids": True, "stream": True,
            "stream_options": {"include_usage": True}}
    started = time.monotonic()
    events, done, error = read_stream(client, body)
    return collect(index, started, events, done, error)


def collect(index, started, events, done, error):
    ids, logprobs, arrivals, text = [], [], [], []
    finish, usage = None, None
    for at, event in events:
        if event.get("error"):
            error = error or json.dumps(event["error"])
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            chunk = choice.get("token_ids") or []
            ids.extend(chunk)
            logprobs.extend((choice.get("logprobs") or {}).get("token_logprobs") or [])
            text.append(choice.get("text") or "")
            finish = choice.get("finish_reason") or finish
            if chunk:
                arrivals.append((at, len(chunk)))
    complete = done and error is None and finish is not None and usage is not None
    return {"index": index, "token_ids": ids, "logprobs": logprobs, "text": "".join(text),
            "finish_reason": finish, "usage": usage, "done": done, "error": error,
            "complete": complete,
            "ttft_s": arrivals[0][0] - started if arrivals else None, "arrivals": arrivals}


def cohort(requests):
    """Tokens after each request's first, over the span from the first first
    token to the last token, as bench.py's cohort rate."""
    streams = [row["arrivals"] for row in requests if row["arrivals"]]
    if not streams:
        return {"tokens": 0, "seconds": 0.0, "tokens_per_second": None}
    start = min(stream[0][0] for stream in streams)
    end = max(stream[-1][0] for stream in streams)
    tokens = sum(n for stream in streams for _, n in stream[1:])
    return {"tokens": tokens, "seconds": end - start,
            "tokens_per_second": tokens / (end - start) if end > start else None}


def step_lines(text):
    steps = []
    for line in text.splitlines():
        if "chopinfer.step.v1" not in line:
            continue
        try:
            record = json.loads(line[line.index("{"):])
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("schema") == "chopinfer.step.v1":
            steps.append(record)
    return steps


def summarize_steps(steps, rows):
    """Median host timings of the steps with `rows` decode or verify items and
    no prefill, the steady cohort; and of every step."""
    def medians(selected):
        out = {"steps": len(selected)}
        for field in HOST_FIELDS:
            values = [step["host"][field] for step in selected
                      if step.get("host") and step["host"].get(field) is not None]
            out[field] = statistics.median(values) if values else None
        return out
    steady = [step for step in steps if step.get("prefill_tokens") == 0
              and step.get("decode_rows", 0) + step.get("verify_items", 0) == rows]
    return {"steady": medians(steady), "all": medians(steps)}


def run(args):
    client = Client(args.url, timeout=args.timeout)
    identity = client.identity(args.model)
    model = identity["model"]
    log = Path(args.log)
    stream_request(client, model, 900, 8)
    passes = []

    def one_pass(concurrency, indices):
        offset = log.stat().st_size
        started = time.monotonic()
        if concurrency == 1:
            requests = [stream_request(client, model, index, args.output_tokens) for index in indices]
        else:
            with ThreadPoolExecutor(concurrency) as pool:
                requests = list(pool.map(
                    lambda index: stream_request(client, model, index, args.output_tokens), indices))
        wall = time.monotonic() - started
        time.sleep(0.2)
        with log.open("rb") as handle:
            handle.seek(offset)
            steps = step_lines(handle.read().decode(errors="replace"))
        per_token = [(row["arrivals"][-1][0] - row["arrivals"][0][0]) * 1000
                     / max(1, sum(n for _, n in row["arrivals"][1:]))
                     for row in requests if len(row["arrivals"]) > 1]
        return {"concurrency": concurrency, "wall_s": wall,
                "cohort": None if concurrency == 1 else cohort(requests),
                "ms_per_token": statistics.median(per_token) if per_token else None,
                "host": summarize_steps(steps, concurrency),
                "fallback_launches": sum(step["route"]["fallback_launches"] for step in steps) if steps and all("route" in step for step in steps) else None,
                "incomplete": sum(not row["complete"] for row in requests),
                "requests": [{key: row[key] for key in RECORDED} for row in requests]}

    load = []
    for concurrency in args.concurrencies:
        load.append(host_load_sample())
        if concurrency == 1:
            passes.append(one_pass(1, range(args.c1_requests)))
        else:
            for _ in range(args.repeats):
                passes.append(one_pass(concurrency, range(concurrency)))
    result = {"arm": args.arm, "server": identity, "output_tokens": args.output_tokens,
              "passes": passes, "host_load": host_load([*load, host_load_sample()])}
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    for row in passes:
        rate = row["cohort"]["tokens_per_second"] if row["cohort"] else None
        steady = row["host"]["steady"]
        speed = row["ms_per_token"]
        print(f"{args.arm} C{row['concurrency']}: "
              + (f"{speed:.2f} ms/token" if speed else "no tokens")
              + (f", {rate:.0f} tok/s" if rate else "")
              + f", boundary {steady['boundary_us']} us, period {steady['period_us']} us, fallback launches {row['fallback_launches']}"
              + f" over {steady['steps']} steps")


COMPARED = ("token_ids", "logprobs", "text", "finish_reason", "usage")


def outputs(row):
    return {request["index"]: request for request in row["requests"]}


def agreement(left, right):
    """Complete requests whose token IDs, logprobs, text, finish reason and
    usage are all identical, of those in both."""
    left, right = outputs(left), outputs(right)
    shared = sorted(set(left) & set(right))
    return sum(left[index].get("complete", False) and right[index].get("complete", False)
               and bool(left[index]["token_ids"])
               and all(left[index].get(key) == right[index].get(key) for key in COMPARED)
               for index in shared), len(shared)


def incomplete(arm):
    return sum(not request.get("complete", False)
               for row in arm["passes"] for request in row["requests"])


def coverage_errors(off, on):
    """Why the two arms did not run the same workload, if they did not: the
    same concurrencies, the same number of passes of each, and in every pass
    the same nonempty set of request indices, each once."""
    errors = []
    concurrencies = {name: sorted({row["concurrency"] for row in arm["passes"]})
                     for name, arm in (("off", off), ("on", on))}
    if not concurrencies["off"]:
        errors.append("the off arm ran nothing")
    if concurrencies["off"] != concurrencies["on"]:
        errors.append(f"concurrencies differ: {concurrencies}")
    for concurrency in sorted(set(concurrencies["off"]) | set(concurrencies["on"])):
        passes = {name: [row for row in arm["passes"] if row["concurrency"] == concurrency]
                  for name, arm in (("off", off), ("on", on))}
        if len(passes["off"]) != len(passes["on"]):
            errors.append(f"C{concurrency}: {len(passes['off'])} off passes, {len(passes['on'])} on")
        indices = None
        for name, rows in passes.items():
            for row in rows:
                seen = [request["index"] for request in row["requests"]]
                if not seen:
                    errors.append(f"C{concurrency}: an empty {name} pass")
                if len(seen) != len(set(seen)):
                    errors.append(f"C{concurrency}: duplicate request indices in the {name} arm")
                if indices is None:
                    indices = set(seen)
                elif set(seen) != indices:
                    errors.append(f"C{concurrency}: the {name} arm ran other requests")
    return errors


def compare_arms(off, on):
    rows = []
    # Both arms must have run the same workload with every stream complete
    # before anything counts as equal.
    coverage = coverage_errors(off, on)
    strict_ok = not coverage and incomplete(off) == 0 and incomplete(on) == 0
    for concurrency in sorted({row["concurrency"] for row in off["passes"]}):
        off_passes = [row for row in off["passes"] if row["concurrency"] == concurrency]
        on_passes = [row for row in on["passes"] if row["concurrency"] == concurrency]
        if not on_passes:
            continue
        equal, total = agreement(off_passes[0], on_passes[0])
        self_equal = agreement(off_passes[0], off_passes[1]) if len(off_passes) > 1 else None
        if concurrency == 1 and equal != total:
            strict_ok = False

        def median_of(passes, getter):
            values = [getter(row) for row in passes]
            values = [value for value in values if value is not None]
            return statistics.median(values) if values else None
        entry = {"concurrency": concurrency, "equal": equal, "compared": total,
                 "off_self_equal": self_equal,
                 # Informational: batched on/off agreement below the off
                 # arm's agreement with its own repeat.
                 "below_off_repeat": self_equal is not None and equal < self_equal[0]}
        for name, passes in (("off", off_passes), ("on", on_passes)):
            entry[name] = {
                "ms_per_token": median_of(passes, lambda row: row["ms_per_token"]),
                "fallback_launches": median_of(passes, lambda row: row.get("fallback_launches")),
                "tokens_per_second": median_of(
                    passes, lambda row: row["cohort"]["tokens_per_second"] if row["cohort"] else None),
                **{field: median_of(passes, lambda row, field=field: row["host"]["steady"][field])
                   for field in HOST_FIELDS}}
        rows.append(entry)
    return {"off": off["arm"], "on": on["arm"], "passed": strict_ok, "coverage": coverage,
            "incomplete": {"off": incomplete(off), "on": incomplete(on)}, "rows": rows,
            "host_load": compared_timing({"off": off.get("host_load"), "on": on.get("host_load")})}


def table(result):
    timing = result["host_load"]
    lines = [f"{result['off']} against {result['on']}",
             f"timing: {timing['timing']}" + (f" ({'; '.join(timing['reasons'])})" if timing["reasons"] else ""), "",
             "| C | ms/token off | on | tok/s off | on | boundary us off | on | period us off | on "
             "| deferred us on | fallback launches off | on | equal on/off | off/off |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]

    def cell(value, digits=0):
        return "" if value is None else f"{value:.{digits}f}"
    for row in result["rows"]:
        self_equal = row["off_self_equal"]
        lines.append(" | ".join([
            f"| {row['concurrency']}", cell(row["off"]["ms_per_token"], 2), cell(row["on"]["ms_per_token"], 2),
            cell(row["off"]["tokens_per_second"]), cell(row["on"]["tokens_per_second"]),
            cell(row["off"]["boundary_us"]), cell(row["on"]["boundary_us"]),
            cell(row["off"]["period_us"]), cell(row["on"]["period_us"]),
            cell(row["on"]["deferred_us"]),
            cell(row["off"].get("fallback_launches")), cell(row["on"].get("fallback_launches")),
            f"{row['equal']}/{row['compared']}",
            "" if self_equal is None else f"{self_equal[0]}/{self_equal[1]}"]) + " |")
    return "\n".join(lines)


def main():
    parser = arguments(__doc__)
    parser.add_argument("command", choices=("run", "compare"))
    parser.add_argument("paths", nargs="*", help="compare: the off arm's and the on arm's JSON")
    parser.add_argument("--arm", help="run: this arm's label")
    parser.add_argument("--log", help="run: the server log the step trace goes to")
    parser.add_argument("--out", help="run: where the arm's JSON goes")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 16, 32])
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--c1-requests", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.c1_requests < 1 or args.repeats < 1 or args.output_tokens < 1:
        parser.error("--c1-requests, --repeats and --output-tokens must be positive")
    if (not args.concurrencies or any(value < 1 for value in args.concurrencies)
            or len(set(args.concurrencies)) != len(args.concurrencies)):
        parser.error("--concurrencies must be distinct positive values")
    if args.command == "run":
        if not (args.arm and args.log and args.out):
            parser.error("run needs --arm, --log and --out")
        run(args)
        return 0
    if len(args.paths) != 2:
        parser.error("compare needs the off and the on arm")
    off, on = (json.loads(Path(path).read_text()) for path in args.paths)
    result = compare_arms(off, on)
    print(table(result))
    for row in result["rows"]:
        if row["below_off_repeat"]:
            print(f"note: C{row['concurrency']} on/off agreement {row['equal']} is below the off arm's "
                  f"own repeat {row['off_self_equal'][0]}; batch composition or adaptive speculation "
                  "follows timing, so compare with further repeats before reading it as a difference")
    print(json.dumps({key: result[key] for key in ("off", "on", "passed", "coverage", "incomplete", "host_load")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

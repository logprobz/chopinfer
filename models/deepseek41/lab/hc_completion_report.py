"""Require the complete C1/C8 HC completion matrix, memcheck, timing and peer parity."""
import argparse
import json
import math
from pathlib import Path
import re
from collections import defaultdict

VARIANTS = ["tail8", "tail16", "tail32", "split8", "split16"]
CONFIGURATIONS = ["served", "other"]
SPECIALS = ["sum_nan", "sum_inf", "sum_subnormal", "negative_zero", "weight_nan",
            "gamma_inf", "gamma_last_nan", "random"]
CONTROLS = [f"chop_hc_completion_{m}_{layout}_w8_v1" for layout, m in
            [("tail", "clearstatus"), ("tail", "wrongfloor"), ("split", "clearstatus"),
             ("split", "wrongfloor"), ("split", "reversefold")]]
SELECTORS = {
    "memcheck": "replay::memcheck_real_c1_c8_on_gpu",
    "matrix": "replay::real_c1_c8_bitwise_invariances_and_controls_on_gpu",
    "timing": "timing::paired_cold_hot_served_configuration_on_gpu",
    "artifacts": "artifacts::frozen_modules_and_cases_on_cpu",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def result(text):
    found = re.findall(r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;", text)
    require(len(found) == 1 and found[0][1:] in (("1", "0", "0"), ("0", "1", "0")),
            "exactly one executed test required")
    require(not re.search(r"\bskipp(?:ed|ing)\b", text, re.I), "skipped test")
    return found[0][0] == "ok"


def listing(text, selector):
    require(re.findall(r"^([^\r\n]+): test$", text, re.M) == [selector]
            and re.search(r"^1 test, 0 benchmarks$", text, re.M), "selector must list exactly one test")


def markers(text, name):
    """Markers anywhere in a line: libtest prefixes the first output line."""
    return [json.loads(v) for v in re.findall(re.escape(name) + r" (\{[^\r\n]*\})", text)]


def first_by_rows(coverage):
    seen = {}
    for case in coverage["cases"]:
        seen.setdefault(case["rows"], case)
    return seen


def labels(coverage):
    """Every invariance and control label the matrix must report once."""
    wanted = set()
    for case in coverage["cases"]:
        for variant in VARIANTS:
            for configuration in CONFIGURATIONS:
                label = f"{case['id']}:{variant}:{configuration}"
                for replay in (0, 1):
                    wanted.add((label, f"graph_replay{replay}"))
    for rows, case in first_by_rows(coverage).items():
        for variant in VARIANTS:
            for configuration in CONFIGURATIONS:
                label = f"{case['id']}:{variant}:{configuration}"
                for chunk in (1, 3):
                    for index in range((rows + chunk - 1) // chunk):
                        wanted.add((label, f"chunk{chunk}:{index}"))
                names = ["reverse_position", "concurrent_scratch", "sticky", "negative_byte",
                         *{f"live{n}" for n in (0, rows - 1)}, *SPECIALS]
                for name in names:
                    wanted.add((label, name))
        for prefix in CONTROLS:
            wanted.add((f"{case['id']}:{prefix}", "mutation_detected"))
    return wanted


def matrix(value, text, coverage, memcheck=False):
    executed = result(text)
    cases = {c["id"]: c for c in coverage["cases"]}
    if memcheck:
        cases = {c["id"]: c for c in memcheck_cases(coverage)}
        require("ERROR SUMMARY: 0 errors" in text, "clean memcheck summary missing")
    else:
        require(value["corpus_sha256"] == coverage["corpus_sha256"], "wrong corpus")
    require(cases and value["cases"] == len(cases) and value["memcheck"] is memcheck, "wrong or empty case set")
    configurations = ["served"] if memcheck else CONFIGURATIONS
    expected = {(c, v, k) for c in cases for v in VARIANTS for k in configurations}
    actual = [(r["case"], r["variant"], r["configuration"]) for r in value["reports"]]
    require(len(actual) == len(expected) and set(actual) == expected, "missing or duplicate comparison")
    if not memcheck:
        require(len(actual) == coverage["matrix_reports"], "declared report count differs")
    for r in value["reports"]:
        case = cases[r["case"]]
        served = r["configuration"] == "served"
        require(r["rows"] == case["rows"] and r["alias"] is served
                and r["widen"] is (case["widening"] if served else not case["widening"]), "configuration differs")
        require(r["passed"] is (not r["failures"]) and re.fullmatch("[0-9a-f]{64}", r["output_sha256"]),
                "invalid bitwise result")
    checks = [(r["case"], r["name"]) for r in value["checks"]]
    wanted = set() if memcheck else labels(coverage)
    require(len(checks) == len(wanted) and set(checks) == wanted, "missing or duplicate invariance/control")
    captured = value["captured"]
    if not memcheck:
        require(len(captured) == len(cases) and {r["case"] for r in captured} == cases.keys()
                and all(r["outputs"] >= 6 and r["passed"] is (not r["failures"]) for r in captured),
                "missing captured-output comparison")
        require(value["negatives"] == len(CONTROLS) * 2, "mutation control count differs")
    else:
        require(not captured and value["negatives"] == 0, "memcheck runs only the comparisons")
    per_case = len(VARIANTS) * len(configurations)
    require(sorted((r["case"], r["rows"], r["reports"]) for r in markers(text, "HC_COMPLETION_CASE"))
            == sorted((c["id"], c["rows"], per_case) for c in cases.values()), "executed case markers differ")
    passed = all(r["passed"] for name in ("reports", "checks", "captured") for r in value[name])
    summary = (f"HC_COMPLETION_RESULT memcheck={str(memcheck).lower()} cases={len(cases)} reports={len(expected)} "
               f"checks={len(checks)} negatives={value['negatives']} passed={str(passed).lower()}")
    require(value["passed"] is passed and executed is passed
            and re.findall(r"HC_COMPLETION_RESULT[^\r\n]*", text) == [summary], "matrix summary differs")
    return dict(passed=passed, cases=len(cases), reports=len(actual), checks=len(checks), captured=len(captured),
                failures={k: [r for r in value[k] if not r["passed"]] for k in ("reports", "checks", "captured")})


def memcheck_cases(coverage):
    picked = []
    for rows in (1, 8):
        picked.extend([c for c in coverage["cases"] if c["rows"] == rows][:2])
    return picked


def step_totals(records, keys):
    """Per captured step (the case id before its site), sum every boundary;
    then average the steps. `keys` names the grouping fields."""
    groups = defaultdict(lambda: [0.0, 0.0, 0])
    for r in records:
        row = groups[tuple(r[k] for k in keys) + (r["case"].rsplit(":", 2)[0],)]
        row[0] += r["reference_us"]
        row[1] += r["candidate_us"]
        row[2] += 1
    totals = defaultdict(list)
    for (*key, _), v in groups.items():
        totals[tuple(key)].append(v)
    return [dict(zip(keys, key), groups=len(items), boundaries=[x[2] for x in items],
                 reference_us_per_step=sum(x[0] for x in items) / len(items),
                 candidate_us_per_step=sum(x[1] for x in items) / len(items),
                 saved_us_per_step=sum(x[0] - x[1] for x in items) / len(items),
                 launches_saved_per_step=sum(x[2] for x in items) / len(items))
            for key, items in sorted(totals.items())]


def timing(value, text, coverage):
    require(result(text), "timing test failed")
    cases = {c["id"]: c for c in coverage["cases"]}
    expected = {(c, v, h) for c in cases for v in VARIANTS for h in (False, True)}
    got = [(r["case"], r["variant"], r["hot"]) for r in value["records"]]
    require(value["cases"] == len(cases) and value["corpus_sha256"] == coverage["corpus_sha256"]
            and len(got) == coverage["timing_records"] and len(got) == len(expected) and set(got) == expected,
            "missing timing case")
    for r in value["records"]:
        case = cases[r["case"]]
        require(r["rows"] == case["rows"] and r["widen"] is case["widening"] and r["rounds"] == coverage["rounds"]
                and r["warmup"] == coverage["warmup"] and r["reference_launches"] == 2
                and r["candidate_launches"] == 1, "timing plan differs")
        require(all(isinstance(r[k], (int, float)) and math.isfinite(r[k]) and r[k] > 0
                    for k in ("reference_us", "candidate_us", "paired_ratio")), "invalid event timing")
    done = markers(text, "HC_COMPLETION_TIMING_CASE")
    require(sorted((r["case"], r["rows"], r["records"]) for r in done)
            == sorted((c["id"], c["rows"], 2 * len(VARIANTS)) for c in cases.values()), "timing markers differ")
    require(re.findall(r"HC_COMPLETION_TIMING_RESULT[^\r\n]*", text)
            == [f"HC_COMPLETION_TIMING_RESULT cases={len(cases)} records={len(expected)} PASS"], "timing summary differs")
    return step_totals(value["records"], ("rows", "variant", "hot"))


def peers(a, b):
    key = lambda r: (r["case"], r["variant"], r["configuration"])
    x = {key(r): r["output_sha256"] for r in a["reports"]}
    y = {key(r): r["output_sha256"] for r in b["reports"]}
    require(x and x.keys() == y.keys(), "peer cases differ")
    return [list(k) for k in sorted(x) if x[k] != y[k]]


def telemetry(value, coverage):
    records = value["timing_rounds"]
    summaries = [r for r in records if r.get("kind") == "paired_summary"]
    samples = [r for r in records if r.get("kind") != "paired_summary"]
    require(value["case_count"] == len(coverage["cases"]) and len(summaries) == coverage["timing_records"]
            and len(samples) == 2 * coverage["rounds"] * coverage["timing_records"], "incomplete timing telemetry")
    flags = sorted({f for r in samples for f in r.get("gpu_state", {}).get("flags", [])})
    return dict(measurement_label=value["measurement_label"], flags=flags, samples=len(samples),
                information_only=value["measurement_label"] != "exclusive-device" or bool(flags))


def analyze(root, out):
    coverage = json.loads((root / "snapshot/cubins/coverage.json").read_text())
    results, matrices, failed = [], [], False
    for gpu in (0, 1):
        row = dict(gpu=gpu)
        try:
            row["memcheck"] = matrix(json.loads((out / f"memcheck-{gpu}.json").read_text()),
                                     (out / f"memcheck-{gpu}.log").read_text(), coverage, memcheck=True)
            m = json.loads((out / f"matrix-{gpu}.json").read_text())
            matrices.append(m)
            row["matrix"] = matrix(m, (out / f"matrix-{gpu}.log").read_text(), coverage)
            row["timing"] = timing(json.loads((out / f"timing-{gpu}.json").read_text()),
                                   (out / f"timing-{gpu}.log").read_text(), coverage)
            receipts = list((out / f"receipts-{gpu}/hc_completion_timing").glob("*.json"))
            require(len(receipts) == 1, "expected one telemetry receipt")
            row["timing_conditions"] = telemetry(json.loads(receipts[0].read_text()), coverage)
            failed |= not (row["matrix"]["passed"] and row["memcheck"]["passed"])
        except (ValueError, KeyError, OSError) as e:
            row["invalid"] = str(e)
            failed = True
        results.append(row)
    peer = peers(*matrices) if len(matrices) == 2 else ["missing device"]
    failed |= bool(peer)
    value = dict(results=results, peer_mismatches=peer, passed=not failed,
                 scope="Exact real-input lab only: every captured C1/C8 boundary against the served pair. No serving promotion by itself.")
    (out / "analysis.json").write_text(json.dumps(value, indent=2) + "\n")
    print("HC_COMPLETION_PAIR " + json.dumps(dict(passed=not failed, cases=len(coverage["cases"]),
                                                  reports_per_gpu=coverage["matrix_reports"],
                                                  timing_records_per_gpu=coverage["timing_records"],
                                                  peer_mismatches=len(peer))))
    return int(failed)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(analyze(a.root, a.output))

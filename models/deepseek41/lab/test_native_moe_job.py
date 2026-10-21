"""CPU checks of native_moe_job's analyzers on synthetic logs of the
format the frozen test binary prints."""
import copy
import json
import unittest

import native_moe_job as job


def timing(ref=2.0, new=1.0):
    rounds = [[[ref, ref, ref], [new, new, new]] for _ in range(5)]
    return {"arms": ["reference", "candidate"], "ms": [ref, new], "median_round_ratio": new / ref,
            "round_samples_ms": rounds, "replays_per_sample": 1}


def stage(failures=0):
    return {"rows": 192, "elements": 192 * 1152, "failures": failures, "worst_ratio": 0.01, "rmse": 1e-6,
            "max_abs": 1e-5, "min_cosine": 0.999999, "cosine_failures": 0}


def report_only():
    return {"native": stage(), "exact": stage(), "rmse_ratio": 1.0, "max_abs_ratio": 1.0, "no_larger_error": True, "gate": False}


def qualified(live, oracle, target=True):
    variants = [{"kind": "GateUp", "tile": "t", "loop": "served", "bitwise": True}] * job.TILE_VARIANTS
    if target:
        variants += [{"kind": "GateUp", "tile": "t", "loop": "lean", "bitwise": True}] * (job.VARIANTS - job.TILE_VARIANTS)
    row = {"live": live, "variants": variants,
           "serving_gate_matches_raw": True, "failures": [], "passed": True}
    if oracle:
        row["oracle"] = {"sampled_tokens": min(32, live), "routes": 6 * min(32, live), "gate_up": stage(), "down": stage(),
                         "chain_down": stage(), "exact_down": stage(), "omit_first_rejected": True,
                         "omit_first": stage(failures=100),
                         "report_only": {"down_vs_exact": report_only(), "ordered_sum_vs_exact": report_only()}}
    return row


def synthetic_log(memcheck, oracle, timed, probe):
    lines = ["running 1 test", "test synthetic::native_moe_synthetic_gpu ... "]
    if probe:
        lines.append("native_moe_probe " + json.dumps({"executed_outputs": 4 * 32 * 128 + 8 * 128, "failures": 0}))
    lives = job.MEMCHECK_LIVES if memcheck else (8192,)
    for case in job.CASES:
        for live in lives:
            row = qualified(live, oracle, case in job.TARGET_CASES)
            row.update(case=case, memcheck=memcheck)
            if oracle:
                row["invariance"] = {"prefixes": [{"live": l, "bitwise_prefix": True} for l in (4097, 65, 1)],
                                     "graph_bitwise": True, "repeat_bitwise": True}
            if timed:
                row["timing"] = {p: timing() for p in ("gate_up", "down", "both")}
            lines.append("native_moe_synthetic " + json.dumps(row))
    lines.append("native_moe_synthetic_done " + json.dumps({"cases": len(job.CASES) * len(lives), "failed": 0, "memcheck": memcheck}))
    if memcheck:
        lines.append("========= ERROR SUMMARY: 0 errors")
    lines.append("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 9 filtered out")
    return "\n".join(lines)


def captured_log(worker, oracle, timed):
    lines = ["running 1 test"]
    n = 0
    for layer in job.LAYERS:
        for rank in (worker, worker + 2):
            for prompt, position, rows in job.CHUNKS:
                row = qualified(rows, oracle)
                row.update(layer=layer, rank=rank, prompt=prompt, position=position, rows=rows, worker=worker,
                           authentication_failures=[], ordered_sum_model_reproduces_capture=True)
                if timed and rows == 8192:
                    tiles = [{"kind": "GateUp", "tile": str(i), "serving_tile": i in (0, 3), "timing": timing()} for i in range(5)]
                    row["timing"] = {**{p: timing() for p in ("gate_up", "down", "both")}, "tiles": tiles}
                lines.append(("test captured::native_moe_captured_gpu ... " if n == 0 else "") + "native_moe_captured " + json.dumps(row))
                n += 1
    lines.append("native_moe_captured_done " + json.dumps({"worker": worker, "cases": n, "failed": 0}))
    lines.append("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 9 filtered out")
    return "\n".join(lines)


def mutate(text, marker, change):
    out = []
    for line in text.splitlines():
        if marker + " " in line:
            head, body = line.split(marker + " ", 1)
            row = json.loads(body)
            change(row)
            line = head + marker + " " + json.dumps(row)
        out.append(line)
    return "\n".join(out)


# `native-moe-test --list` of the binary sealed in the first staged snapshot.
SEALED_LIST = [
    "captured::native_moe_captured_cpu_admission",
    "captured::native_moe_captured_gpu",
    "oracle::tests::bound_and_cosine_reject_a_dropped_chain",
    "oracle::tests::dot_matches_a_naive_decode",
    "oracle::tests::ordered_sum_adds_routes_in_order_then_the_addend",
    "probe::probe_cases_isolate_one_product_and_cover_the_domain",
    "samples_cover_first_and_last",
    "synthetic::cpu_plan",
    "synthetic::native_moe_synthetic_gpu",
]


def libtest_list(argv):
    """libtest's --list under a filter: --exact keeps whole-name matches."""
    binary = argv.index("/staged/bin/native-moe-test")
    args = argv[binary + 1:]
    name = next(a for a in args if not a.startswith("--"))
    names = [n for n in SEALED_LIST if (n == name if "--exact" in args else name in n)]
    return "".join(f"{n}: test\n" for n in names) + f"\n{len(names)} test{'' if len(names) == 1 else 's'}, 0 benchmarks\n"


class Filters(unittest.TestCase):
    def test_every_command_selects_exactly_its_test(self):
        from pathlib import Path
        plan = {"sanitizer": "/sanitizer"}
        for phase, tests in job.PHASE_TESTS.items():
            for test in tests:
                argv = job.command(Path("/staged"), plan, phase, test)
                job.check_listed(libtest_list(argv), job.TESTS[test])
        for name in job.CPU_TESTS:
            job.check_listed(libtest_list(job.test_args(Path("/staged"), name)), name)

    def test_rejects_a_filter_that_selects_nothing_or_more(self):
        with self.assertRaises(ValueError):
            job.check_listed("\n0 tests, 0 benchmarks\n", "synthetic::native_moe_synthetic_gpu")
        bare = libtest_list(["/staged/bin/native-moe-test", "native_moe_synthetic_gpu", "--exact"])
        with self.assertRaises(ValueError):
            job.check_listed(bare, "native_moe_synthetic_gpu")
        loose = libtest_list(["/staged/bin/native-moe-test", "native_moe"])
        with self.assertRaises(ValueError):
            job.check_listed(loose, "native_moe")


class Analyzers(unittest.TestCase):
    def test_accepts_complete_phases(self):
        job.analyze_synthetic(synthetic_log(True, False, False, True), True, False, False, True)
        job.analyze_synthetic(synthetic_log(False, True, False, True), False, True, False, True)
        job.analyze_synthetic(synthetic_log(False, False, True, False), False, False, True, False)
        for worker in (0, 1):
            rows = job.analyze_captured(captured_log(worker, True, False), worker, True, False)
            self.assertEqual(len(rows), 24)
            job.analyze_captured(captured_log(worker, False, True), worker, False, True)
        parsed = [{"synthetic": job.analyze_synthetic(synthetic_log(False, True, False, True), False, True, False, True),
                   "captured": job.analyze_captured(captured_log(0, True, False), 0, True, False)}]
        summary = job.summarize(parsed)
        self.assertEqual(summary["no_larger_error"]["down_vs_exact"]["cases"], 27)

    def test_timing_summary_scales_to_a_chunk(self):
        parsed = [{"captured": job.analyze_captured(captured_log(w, False, True), w, False, True)} for w in (0, 1)]
        summary = job.summarize(parsed)
        self.assertEqual(summary["timing"]["both"]["cases"], 16)
        self.assertAlmostEqual(summary["timing"]["per_chunk_40_layers_ms"]["native"], 40.0)

    def test_rejects_missing_failed_or_unexecuted_work(self):
        good = synthetic_log(False, True, False, True)
        bad = [
            good.replace("test result: ok. 1 passed", "test result: ok. 0 passed"),
            "\n".join(l for l in good.splitlines() if "e128-skewed" not in l),
            mutate(good, "native_moe_synthetic", lambda r: r["oracle"]["gate_up"].update(failures=1)),
            mutate(good, "native_moe_synthetic", lambda r: r["oracle"]["down"].update(cosine_failures=1)),
            mutate(good, "native_moe_synthetic", lambda r: r["oracle"].update(omit_first_rejected=False)),
            mutate(good, "native_moe_synthetic", lambda r: r["invariance"]["prefixes"][1].update(bitwise_prefix=False)),
            mutate(good, "native_moe_synthetic", lambda r: r["variants"][2].update(bitwise=False)),
            mutate(good, "native_moe_synthetic", lambda r: r["variants"][-1].update(bitwise=False)),
            mutate(good, "native_moe_synthetic", lambda r: r["variants"].pop() if r["case"] != "e128-skewed" else None),
            mutate(good, "native_moe_synthetic", lambda r: r["variants"].append(r["variants"][-1]) if r["case"] == "e128-skewed" else None),
            mutate(good, "native_moe_synthetic", lambda r: r.update(serving_gate_matches_raw=False)),
            mutate(good, "native_moe_probe", lambda r: r.update(failures=1)),
            mutate(good, "native_moe_synthetic_done", lambda r: r.update(cases=2)),
        ]
        for text in bad:
            with self.assertRaises((ValueError, KeyError)):
                job.analyze_synthetic(text, False, True, False, True)
        memcheck = synthetic_log(True, False, False, True)
        with self.assertRaises(ValueError):
            job.analyze_synthetic(memcheck.replace("ERROR SUMMARY: 0 errors", "ERROR SUMMARY: 2 errors"), True, False, False, True)
        cap = captured_log(1, True, False)
        for text in [
            mutate(cap, "native_moe_captured", lambda r: r.update(authentication_failures=["x"])),
            mutate(cap, "native_moe_captured", lambda r: r.update(worker=0)),
            "\n".join(l for l in cap.splitlines() if '"layer": 39' not in l),
        ]:
            with self.assertRaises(ValueError):
                job.analyze_captured(text, 1, True, False)

    def test_rejects_inconsistent_timing(self):
        t = timing()
        job.validate_timing(t)
        broken = copy.deepcopy(t)
        broken["ms"][1] = 0.5
        with self.assertRaises(ValueError):
            job.validate_timing(broken)
        broken = copy.deepcopy(t)
        broken["round_samples_ms"].pop()
        with self.assertRaises(ValueError):
            job.validate_timing(broken)


if __name__ == "__main__":
    unittest.main()

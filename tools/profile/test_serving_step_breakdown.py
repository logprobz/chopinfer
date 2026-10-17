import contextlib
import io
import json
import os
import sqlite3
import tempfile
import unittest

from serving_step_breakdown import CLASSES, analyze, main, step_trace, summarize

US = 1000  # trace times are in ns


def export(graph_trace=False):
    """Three steps 2,000 us apart on two devices; the engine thread is tid 1.

    Each step: (the middle one only) a page memset and its sync, two graph
    launches, a 1,000 us logits wait, then the sampler's copy and its wait,
    which ends the step 1,100 us after its base.
    """
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE StringIds(id INTEGER, value TEXT);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME("
        "start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
        "start INTEGER, end INTEGER, deviceId INTEGER, correlationId INTEGER, shortName INTEGER);"
    )
    names = ["cuMemsetD8Async", "cuStreamSynchronize", "cuGraphLaunch", "cuMemcpyDtoHAsync_v2",
             "chop_proj_a", "chop_comm_x", "chop_moe_b", "cuEventRecord"]
    ids = {name: index for index, name in enumerate(names)}
    connection.executemany("INSERT INTO StringIds VALUES (?, ?)", list(enumerate(names)))
    api, kernels = [], []
    correlation = 0

    def call(start, end, name, thread=1):
        nonlocal correlation
        correlation += 1
        api.append((start * US, end * US, thread, correlation, ids[name]))
        return correlation

    for base in (0, 2000, 4000, 6000):
        if base == 2000:
            call(base, base + 1, "cuMemsetD8Async")
            call(base + 5, base + 8, "cuStreamSynchronize")
        first = call(base + 10, base + 12, "cuGraphLaunch")
        second = call(base + 12, base + 14, "cuGraphLaunch")
        call(base + 20, base + 1020, "cuStreamSynchronize")
        call(base + 1030, base + 1032, "cuMemcpyDtoHAsync_v2")
        call(base + 1035, base + 1100, "cuStreamSynchronize")
        call(base + 1500, base + 1501, "cuEventRecord", thread=2)
        # Device 0: the collective launched early under PDL; device 1 arrives
        # 50 us later. Both leave the collective at +500.
        kernels += [
            (base + 15, base + 300, 0, first, ids["chop_proj_a"]),
            (base + 250, base + 500, 0, first, ids["chop_comm_x"]),
            (base + 600, base + 900, 0, second, ids["chop_moe_b"]),
            (base + 15, base + 350, 1, first, ids["chop_proj_a"]),
            (base + 360, base + 500, 1, first, ids["chop_comm_x"]),
            (base + 600, base + 950, 1, second, ids["chop_moe_b"]),
        ]
    connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)", api)
    if graph_trace:
        connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE("
                           "start INTEGER, end INTEGER, deviceId INTEGER, correlationId INTEGER)")
        graphs = [(start, end, device, number) for start, end, device, number, _ in kernels
                  if _ == ids["chop_moe_b"]]
        connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (?, ?, ?, ?)",
                               [(start * US, end * US, device, number) for start, end, device, number in graphs])
    else:
        connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
                               [(start * US, end * US, device, number, name)
                                for start, end, device, number, name in kernels])
    connection.commit()
    return connection


class ServingStepBreakdownTest(unittest.TestCase):
    def test_steps_end_at_the_sampler_collect(self):
        records = analyze(export(), skip=0)
        # Collects at +1100 bound the steps: 1,100 to 3,100 and 3,100 to 5,100
        # us; the last collect has no next graph launch to close it.
        self.assertEqual(len(records), 2)
        host = records[0]["host"]
        self.assertEqual(host["period"], 2000 * US)
        self.assertEqual(host["pre_launch"], 910 * US)
        self.assertEqual(host["launch_api"], 4 * US)
        self.assertEqual(host["logits_wait"], 1000 * US)
        self.assertEqual(host["after_logits"], 80 * US)
        self.assertEqual(host["prepare_kv"], 8 * US)
        self.assertEqual((host["graph_launches"], host["memsets"]), (2, 1))
        self.assertEqual(records[1]["host"]["prepare_kv"], 0)

    def test_device_timing_charges_and_collectives(self):
        records = analyze(export(), skip=0)
        zero, one = records[0]["devices"][0], records[0]["devices"][1]
        self.assertEqual((zero["span"], zero["busy"], zero["inner_gap"]), (885 * US, 785 * US, 100 * US))
        # The idle 100 us before the MoE kernel is charged to it.
        self.assertEqual(zero["charges"], {"proj_gemm": 285 * US, "comm": 200 * US, "moe": 400 * US})
        self.assertEqual(one["charges"], {"proj_gemm": 335 * US, "comm": 150 * US, "moe": 450 * US})
        self.assertIsNone(zero["idle_before"])
        self.assertEqual(records[1]["devices"][0]["idle_before"], 1115 * US)
        self.assertEqual(records[1]["devices"][1]["idle_before"], 1065 * US)
        # Arrivals 2,300 and 2,350 us, both exit at 2,500.
        self.assertEqual(records[0]["collectives"], (1, 150 * US, 25 * US))

    def test_summary_in_microseconds(self):
        report = summarize(analyze(export(), skip=0))
        self.assertEqual(report["steps"], 2)
        self.assertEqual(report["host_us"]["period"], 2000.0)
        self.assertEqual(report["host_us"]["graph_launches"], 2)
        self.assertEqual(report["page_work_step_fraction"], 0.5)
        self.assertEqual(report["prepare_kv_mean_us"], 4.0)
        zero = report["devices"]["0"]
        self.assertEqual((zero["span_us"], zero["busy_us"], zero["idle_before_us"]), (885.0, 785.0, 1115.0))
        self.assertEqual(zero["charged_us"], {"comm": 200.0, "moe": 400.0, "proj_gemm": 285.0})
        self.assertEqual(zero["top_kernels"][0], {"kernel": "chop_moe_b", "per_step": 1.0, "charged_us": 400.0})
        self.assertEqual(report["collectives"], {"instances": 1, "intrinsic_us": 150.0, "mean_skew_wait_us": 25.0})

    def test_class_overrides_come_first(self):
        records = analyze(export(), skip=0, classes=[("routed", ("chop_moe_",)), ("other", ("chop_proj_",))] + CLASSES)
        self.assertEqual(records[0]["devices"][0]["charges"], {"other": 285 * US, "comm": 200 * US, "routed": 400 * US})

    def test_graph_granularity_records_stand_in_for_kernels(self):
        records = analyze(export(graph_trace=True), skip=0)
        zero = records[0]["devices"][0]
        self.assertEqual((zero["kernels"], zero["busy"]), (1, 300 * US))
        self.assertEqual(zero["charges"], {"graph": 300 * US})
        self.assertIsNone(records[0]["collectives"])

    def test_step_trace_window_and_composition(self):
        def line(stamp, prefill, decode, verify_items, verify_rows, period):
            step = {"schema": "chopinfer.step.v1", "prefill_tokens": prefill, "decode_rows": decode,
                    "verify_items": verify_items, "verify_rows": verify_rows,
                    "host": {"period_us": period, "boundary_us": period // 100}}
            return f"{stamp}\t{json.dumps(step)}\n"

        lines = [
            "100\tstartup readiness: 1s\n",
            line(100, 0, 8, 0, 0, 1000),
            line(200, 2048, 7, 0, 0, 9000),
            line(300, 0, 8, 0, 0, 3000),
            line(400, 0, 0, 8, 32, 5000),
            line(900, 0, 8, 0, 0, 7000),
        ]
        report = step_trace(lines, (150, 500))
        self.assertEqual(report["steps"], 2)
        self.assertEqual(report["composition"], {"decode 8, verify 0x0": 1, "decode 0, verify 8x32": 1})
        self.assertEqual(report["median_host_us"], {"period_us": 4000, "boundary_us": 40})
        self.assertEqual(step_trace([text.split("\t", 1)[1] for text in lines[1:]])["steps"], 4)
        self.assertEqual(step_trace([], None), {"steps": 0})

    def test_main_writes_both_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            database = os.path.join(directory, "export.sqlite")
            with sqlite3.connect(database) as copy:
                export().backup(copy)
            trace = os.path.join(directory, "trace.tsv")
            with open(trace, "w") as stream:
                stream.write('5\t{"schema":"chopinfer.step.v1","prefill_tokens":0,"decode_rows":1,'
                             '"verify_items":0,"verify_rows":0,"host":{"period_us":13000}}\n')
            output = os.path.join(directory, "report.json")
            with contextlib.redirect_stdout(io.StringIO()):
                main([database, "--step-trace", trace, "--skip", "0", "--json", output])
            with open(output) as stream:
                report = json.load(stream)
        self.assertEqual(report["nsys"]["steps"], 2)
        self.assertEqual(report["step_trace"]["median_host_us"], {"period_us": 13000})


if __name__ == "__main__":
    unittest.main()

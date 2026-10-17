import contextlib
import io
import json
import os
import sqlite3
import tempfile
import unittest

from serving_step_breakdown import CLASSES, analyze, main, step_trace, summarize

US = 1000  # trace times are in ns
SAMPLER_STREAM = 94


class Export:
    """A synthetic Nsight export; times in us, the engine thread is tid 1."""

    def __init__(self):
        self.names, self.api, self.kernels, self.copies, self.graphs = {}, [], [], [], []
        self.correlation = 0

    def name(self, text):
        return self.names.setdefault(text, len(self.names))

    def call(self, start, end, name, thread=1):
        self.correlation += 1
        self.api.append((start * US, end * US, thread, self.correlation, self.name(name)))
        return self.correlation

    def kernel(self, start, end, device, correlation, name, stream):
        self.kernels.append((start * US, end * US, device, correlation, self.name(name), stream))

    def sampler_launch(self, start):
        correlation = self.call(start, start + 3, "cuLaunchKernelEx")
        self.kernel(start + 10, start + 20, 0, correlation, "chop_sampling_filtered_stats_f32_v1", SAMPLER_STREAM)

    def collect(self, start, wait):
        copy = self.call(start, start + 2, "cuMemcpyDtoHAsync_v2")
        self.copies.append((copy, 0, SAMPLER_STREAM))
        self.call(start + 5, start + 5 + wait, "cuStreamSynchronize")

    def step(self, base, logits_wait=1000, collect_wait=65, eager=False, page_wait=None, kernels=True):
        """Bundle launches at +10 and +12, the logits wait from +20, then the
        sampler: one launch with its kernel, the collect copy and its wait."""
        if page_wait is not None:
            self.call(base, base + 1, "cuMemsetD8Async")
            self.call(base + 5, base + 5 + page_wait, "cuStreamSynchronize")
            base += 5 + page_wait
        # The bundle's input upload waits on an event, not the KV stream.
        self.call(base + 6, base + 7, "cuEventSynchronize")
        self.call(base + 7, base + 8, "cuMemcpyHtoDAsync_v2")
        launch = "cuLaunchKernelEx" if eager else "cuGraphLaunch"
        first = self.call(base + 10, base + 12, launch)
        second = self.call(base + 12, base + 14, launch)
        self.call(base + 20, base + 20 + logits_wait, "cuStreamSynchronize")
        end = base + 20 + logits_wait
        self.sampler_launch(end + 5)
        self.collect(end + 10, collect_wait)
        self.call(base + 1500, base + 1501, "cuEventRecord", thread=2)
        if kernels:
            # Device 0's collective launches early under PDL; device 1's
            # starts 10 us after its dependency. Both leave at +500.
            self.kernel(base + 15, base + 300, 0, first, "chop_proj_a", 7)
            self.kernel(base + 250, base + 500, 0, first, "chop_comm_x", 7)
            self.kernel(base + 600, base + 900, 0, second, "chop_moe_b", 7)
            self.kernel(base + 15, base + 350, 1, first, "chop_proj_a", 8)
            self.kernel(base + 360, base + 500, 1, first, "chop_comm_x", 8)
            self.kernel(base + 600, base + 950, 1, second, "chop_moe_b", 8)
        return first, second

    def build(self):
        connection = sqlite3.connect(":memory:")
        connection.executescript(
            "CREATE TABLE StringIds(id INTEGER, value TEXT);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME("
            "start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, deviceId INTEGER, "
            "correlationId INTEGER, shortName INTEGER, streamId INTEGER);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY("
            "correlationId INTEGER, deviceId INTEGER, streamId INTEGER, copyKind INTEGER);"
        )
        connection.executemany("INSERT INTO StringIds VALUES (?, ?)",
                               [(index, name) for name, index in self.names.items()])
        connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?, ?, ?)", self.api)
        connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)", self.kernels)
        connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, 2)", self.copies)
        if self.graphs:
            connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start INTEGER, end INTEGER, "
                               "deviceId INTEGER, correlationId INTEGER, streamId INTEGER)")
            connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES (?, ?, ?, ?, ?)",
                                   self.graphs)
        connection.commit()
        return connection


def standard(spacing=2000, **options):
    export = Export()
    for step in range(4):
        export.step(step * spacing, **options)
    return export.build()


class ServingStepBreakdownTest(unittest.TestCase):
    def test_steps_end_at_the_sampler_collect(self):
        records, rejected = analyze(standard(), skip=0)
        # Collects end at +1100 and bound the steps 1,100-3,100 and
        # 3,100-5,100 us; the last collect has no bundle launch after it.
        self.assertEqual((len(records), rejected), (2, 0))
        self.assertEqual(records[0]["host"], {
            "period": 2000 * US, "pre_launch": 910 * US, "bundle_issue": 4 * US, "wait": 1011 * US,
            "device_wait": 1000 * US, "sampling": 75 * US, "launch_api": 4 * US, "prepare_kv": 0,
            "graph_launches": 2, "kernel_launches": 0, "sampler_launches": 1, "collects": 1, "memsets": 0})

    def test_long_sampler_waits_keep_their_steps(self):
        # A 1.2 ms collect wait, longer than the logits wait.
        records, _ = analyze(standard(spacing=3000, collect_wait=1200), skip=0)
        self.assertEqual([record["host"]["period"] for record in records], [3000 * US, 3000 * US])
        self.assertEqual(records[0]["host"]["sampling"], 1210 * US)

    def test_a_short_logits_wait_does_not_merge_steps(self):
        export = Export()
        for base, wait in ((0, 1000), (2000, 500), (4000, 1000), (6000, 1000)):
            export.step(base, logits_wait=wait)
        records, _ = analyze(export.build(), skip=0)
        self.assertEqual([record["host"]["period"] for record in records], [1500 * US, 2500 * US])
        self.assertEqual(records[0]["host"]["device_wait"], 500 * US)

    def test_steps_out_of_order_are_rejected(self):
        export = Export()
        for base in (0, 2000, 4000, 6000):
            export.step(base)
        # A step whose sampler launch comes before its bundle work.
        export.sampler_launch(9000)
        export.call(9010, 9012, "cuGraphLaunch")
        export.collect(9020, 5)
        export.step(10000)
        records, rejected = analyze(export.build(), skip=0)
        self.assertEqual((len(records), rejected), (3, 1))

    def test_eager_steps_and_page_preparation(self):
        export = Export()
        for base in (0, 3000, 6000, 9000):
            # Page zeroing waits 1.5 ms, longer than the 1 ms logits wait.
            export.step(base, eager=True, page_wait=1500 if base == 3000 else None)
        records, _ = analyze(export.build(), skip=0)
        host = records[0]["host"]
        # The step starts at 1,100 us; zeroing 3,000-4,505, launches at 4,515.
        self.assertEqual((host["pre_launch"], host["bundle_issue"]), (3415 * US, 4 * US))
        self.assertEqual((host["device_wait"], host["prepare_kv"]), (1000 * US, 1505 * US))
        self.assertEqual((host["kernel_launches"], host["graph_launches"], host["memsets"]), (2, 0, 1))
        # Its collect ends at 5,605 us; the next launches are at 6,010.
        self.assertEqual(records[1]["host"]["pre_launch"], 405 * US)

    def test_device_timing_charges_and_collectives(self):
        records, _ = analyze(standard(), skip=0)
        zero, one = records[0]["devices"][0], records[0]["devices"][1]
        # Device 0 also runs the sampler kernel at 3,035-3,045 us.
        self.assertEqual((zero["span"], zero["busy"], zero["inner_gap"]), (1030 * US, 795 * US, 235 * US))
        self.assertIsNone(zero["graph_elapsed"])
        # Idle time inside the step is charged to the kernel after it.
        self.assertEqual(zero["charges"], {"proj_gemm": 285 * US, "comm": 200 * US, "moe": 400 * US,
                                           "sampling": 145 * US})
        self.assertEqual(one["charges"], {"proj_gemm": 335 * US, "comm": 150 * US, "moe": 450 * US})
        self.assertIsNone(zero["idle_before"])
        self.assertEqual(records[1]["devices"][0]["idle_before"], 970 * US)
        self.assertEqual(records[1]["devices"][1]["idle_before"], 1065 * US)
        # Arrivals at the dependencies' ends, 2,300 and 2,360 us; exits 2,500.
        self.assertEqual(records[0]["collectives"], {"instances": 1, "intrinsic": 140 * US, "skew_wait": 30 * US,
                                                     "launch_gap": 5 * US, "unknown_dependency": 0})

    def collective(self, extra=()):
        export = Export()
        for base in (0, 2000, 4000, 6000):
            first, _ = export.step(base, kernels=False)
            export.kernel(base + 15, base + 100, 0, first, "chop_proj_a", 7)
            export.kernel(base + 15, base + 1000, 1, first, "chop_proj_a", 8)
            export.kernel(base + 1000, base + 1100, 0, first, "chop_comm_x", 7)
            export.kernel(base + 1000, base + 1100, 1, first, "chop_comm_x", 8)
            for start, end, device, stream in extra:
                export.kernel(base + start, base + end, device, first, "chop_other", stream)
        records, _ = analyze(export.build(), skip=0)
        return records[0]["collectives"]

    def test_collectives_arrive_at_their_dependency_or_their_start(self):
        # Device 0's input is ready at 100 us but its collective starts at
        # 1,000, with device 1's: no one waits; device 0 has a launch gap.
        self.assertEqual(self.collective(), {"instances": 1, "intrinsic": 100 * US, "skew_wait": 0,
                                             "launch_gap": 450 * US, "unknown_dependency": 0})

    def test_unrelated_streams_are_not_dependencies(self):
        # A kernel on another stream of device 0 runs past the collective.
        result = self.collective(extra=[(900, 1200, 0, 9)])
        self.assertEqual((result["intrinsic"], result["skew_wait"]), (100 * US, 0))

    def test_collectives_without_a_dependency_are_counted(self):
        export = Export()
        for base in (0, 2000, 4000, 6000):
            first, _ = export.step(base, kernels=False)
            export.kernel(base + 15, base + 100, 0, first, "chop_comm_x", 7)
            export.kernel(base + 15, base + 120, 1, first, "chop_comm_x", 8)
        records, _ = analyze(export.build(), skip=0)
        self.assertEqual(records[0]["collectives"], {"instances": 1, "intrinsic": 105 * US, "skew_wait": 0,
                                                     "launch_gap": 0, "unknown_dependency": 1})

    def test_graph_records_are_elapsed_time_not_kernel_time(self):
        def device_one(graph_trace):
            export = Export()
            for base in (0, 2000, 4000, 6000):
                first, _ = export.step(base, kernels=False)
                if graph_trace:
                    export.graphs.append(((base + 15) * US, (base + 1015) * US, 1, first, 7))
                else:
                    export.kernel(base + 15, base + 115, 1, first, "chop_proj_a", 7)
                    export.kernel(base + 915, base + 1015, 1, first, "chop_moe_b", 7)
            records, _ = analyze(export.build(), skip=0)
            return records[0]["devices"][1]

        node = device_one(False)
        self.assertEqual((node["busy"], node["inner_gap"], node["graph_elapsed"]), (200 * US, 800 * US, None))
        graph = device_one(True)
        self.assertEqual((graph["busy"], graph["inner_gap"], graph["graph_elapsed"]), (None, None, 1000 * US))
        self.assertEqual((graph["kernels"], graph["graphs"], graph["charges"]), (0, 1, {"graph": 1000 * US}))

    def test_summary_in_microseconds(self):
        records, rejected = analyze(standard(), skip=0)
        report = summarize(records, rejected)
        self.assertEqual((report["steps"], report["rejected_steps"]), (2, 0))
        self.assertEqual(report["host_us"]["period"], 2000.0)
        self.assertEqual(report["host_us"]["graph_launches"], 2)
        self.assertEqual((report["page_work_step_fraction"], report["prepare_kv_mean_us"]), (0.0, 0.0))
        zero = report["devices"]["0"]
        self.assertEqual((zero["span_us"], zero["busy_us"], zero["idle_before_us"]), (1030.0, 795.0, 970.0))
        self.assertIsNone(zero["graph_elapsed_us"])
        self.assertEqual(zero["charged_us"], {"comm": 200.0, "moe": 400.0, "proj_gemm": 285.0, "sampling": 145.0})
        self.assertEqual(zero["top_kernels"][0], {"kernel": "chop_moe_b", "per_step": 1.0, "charged_us": 400.0})
        self.assertEqual(report["collectives"], {"instances": 1, "intrinsic_us": 140.0, "mean_skew_wait_us": 30.0,
                                                 "mean_launch_gap_us": 5.0, "unknown_dependency": 0})
        self.assertEqual(summarize([], 3), {"steps": 0, "rejected_steps": 3})

    def test_class_overrides_come_first(self):
        records, _ = analyze(standard(), skip=0,
                             classes=[("routed", ("chop_moe_",)), ("other", ("chop_proj_",))] + CLASSES)
        self.assertEqual(records[0]["devices"][1]["charges"],
                         {"other": 335 * US, "comm": 150 * US, "routed": 450 * US})

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
                standard().backup(copy)
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

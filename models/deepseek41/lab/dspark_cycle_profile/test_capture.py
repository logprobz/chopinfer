"""CPU regression tests for capture ownership, alignment and attribution."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

import capture
import cycles
import families
from sqlite_fixture import create, HEAD_ROWS

SOURCE = Path(__file__).resolve().parents[4]


def steps(directory):
    for c in (8, 16):
        rows = [dict(batch=i, prefill_items=0, decode_rows=0, verify_items=c,
                     live_rows=32, row_capacity=32, completed_unix_ns=10**18+i*2_000_000+c*10**9,
                     phase=dict(catch_up_rows=c*2), host=dict(sampling_us=3))
                for i in range(20)]
        target = directory/f"C{c}"
        target.mkdir()
        (target/"steps.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))


class Attribution(unittest.TestCase):
    def test_adaptive_widths_keep_exact_row_witnesses_with_padded_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            steps(root)
            for c in (8, 16):
                path = root/f"C{c}/steps.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                widths = (c, 20, 24, 32)
                for index, row in enumerate(rows):
                    width = widths[index % len(widths)]
                    verify = min(c, width-c)
                    row.update(live_rows=width, row_capacity=width,
                               verify_items=verify, decode_rows=c-verify)
                path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            create(root/"profile.sqlite", root)
            result = cycles.analyze(root/"profile.sqlite", root)
            for c, case in result["cases"].items():
                for device in case["devices"].values():
                    self.assertEqual(set(device["target_by_rows"]), {c, "20", "24", "32"})
            # A wrong exact width in the same embedding bucket must fail.
            db = sqlite3.connect(root/"profile.sqlite")
            name, = db.execute("SELECT id FROM StringIds WHERE value=?", (HEAD_ROWS,)).fetchone()
            db.execute("UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET gridX=21 WHERE shortName=? AND gridX=20", (name,))
            db.commit()
            db.close()
            with self.assertRaisesRegex(AssertionError, "exact target row"):
                cycles.analyze(root/"profile.sqlite", root)

    def test_target_families_partition_exclusive_kernel_charges(self):
        kernels = {
            "chain_before_gateup_test_stream_small_v1": 4,
            "chain_before_down_test_stream_small_v1": 2,
            "chop_comm_example": 1,
            "chop_latent_sources_test": 3,
            "unfamiliar_kernel": 5,
        }
        grouped = families.summarize(kernels)
        self.assertEqual(sum(grouped.values()), 15)
        self.assertEqual(grouped["MoE gate/up"], 4)
        self.assertEqual(grouped["Other row, norm and Engram"], 5)
        self.assertEqual(families.family("chop_deepseek41_moe_bandwidth_direct_e384_k6_gate_small_v1"), "MoE gate/up")
        self.assertEqual(families.family("chop_deepseek41_moe_floor_e384_k6_down_finish_v1"), "MoE down")
        self.assertEqual(families.family("chop_deepseek41_hc_round_chain_w4_v1"), "HC")

    def test_clock_offset_uses_independent_host_anchors(self):
        for measured in (True, False):
            with self.subTest(measured=measured), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                steps(root)
                create(root/"profile.sqlite", root)
                db = sqlite3.connect(root/"profile.sqlite")
                epoch = db.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
                db.execute("ALTER TABLE TARGET_INFO_SESSION_START_TIME ADD COLUMN systemClockNs INTEGER")
                db.execute("UPDATE TARGET_INFO_SESSION_START_TIME SET utcEpochNs=?,systemClockNs=?",
                           (epoch-7_000_000_000, 1_000_000))
                db.commit()
                db.close()
                if measured:
                    (root/"clock-anchors.json").write_text(json.dumps([
                        dict(unix_before_ns=epoch, unix_after_ns=epoch, monotonic_ns=1_000_000)]))
                else:
                    for c in (8, 16):
                        (root/f"C{c}/summary.json").write_text(json.dumps(dict(host_load=dict(
                            samples=[dict(at=(epoch+offset)/1e9, uptime_seconds=0.0)
                                     for offset in (-800_000, 8_900_000)]))))
                report = cycles.analyze(root/"profile.sqlite", root)
                alignment = report["clock_alignment"]
                self.assertGreaterEqual(alignment["anchors"], 30)
                self.assertLess(abs(alignment["offset_ns"]-7_000_000_000), 1_000_000)
                for case in report["cases"].values():
                    for device in case["devices"].values():
                        self.assertEqual(device["count"], 15)
                        self.assertAlmostEqual(device["phase_mean_ms"]["target"], .4)

    def test_missing_unstarted_session_keeps_the_startup_error(self):
        cases = [(started, failed, missing) for started, failed in (
            (False, True), (False, False), (True, True)) for missing in (False, True)]
        for started, startup_failed, named in cases:
            with self.subTest(started=started, startup_failed=startup_failed, named=named):
                with tempfile.TemporaryDirectory() as tmp:
                    runner = SimpleNamespace(directory=Path(tmp), output=Path(tmp), settings={},
                                             dry=False, env={}, remaining=lambda x: x)
                    session = capture.Session(runner, "cpu")
                    session.started = started
                    message = (f"Session is not found: {session.name}" if named
                               else "There are no active sessions.")
                    def result(argv, **_):
                        return subprocess.CompletedProcess(
                            argv, 1 if "shutdown" in argv else 0, "",
                            message if "shutdown" in argv else "")
                    with patch("session.subprocess.run", side_effect=result):
                        if startup_failed and not started:
                            session.close(startup_failed=True)
                        else:
                            with self.assertRaises(AssertionError):
                                session.close(startup_failed=startup_failed)
                    self.assertTrue(session.closed)
        with tempfile.TemporaryDirectory() as tmp:
            runner = object.__new__(capture.Runner)
            runner.directory = Path(tmp)
            runner.session = SimpleNamespace(close=lambda **_: (_ for _ in ()).throw(
                RuntimeError("cleanup failed")))
            with self.assertRaisesRegex(ValueError, "original startup error"):
                try:
                    raise ValueError("original startup error")
                finally:
                    runner.close_session()
            receipt = json.loads((Path(tmp)/"cleanup-error.json").read_text())
            self.assertEqual(receipt["primary"], "original startup error")
            self.assertEqual(receipt["cleanup"], "cleanup failed")

    def test_server_wrapper_owns_its_group_under_each_launcher_state(self):
        for options in ({}, {"process_group": 0}, {"start_new_session": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                pidfile = Path(tmp) / "server-group.json"
                result = subprocess.run(
                    [sys.executable, "-B", capture.__file__, "server-child", str(pidfile),
                     sys.executable, "-c",
                     "import json,os; print(json.dumps([os.getpid(),os.getpgrp()]))"],
                    capture_output=True, text=True, timeout=10, **options)
                self.assertEqual(result.returncode, 0, result.stderr)
                identity = json.loads(pidfile.read_text())
                self.assertEqual(identity["pid"], identity["pgid"])
                self.assertEqual(json.loads(result.stdout), [identity["pid"], identity["pgid"]])
                self.assertGreater(identity["start_ticks"], 0)

    def test_fast_http_steps_do_not_create_overlapping_fake_cycles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);steps(root)
            for c in (8,16):
                path=root/f"C{c}/steps.jsonl"
                rows=[json.loads(l) for l in path.read_text().splitlines()]
                for i,r in enumerate(rows):
                    r["completed_unix_ns"]=10**18+c*10**9+i*500_000
                path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            create(root/"profile.sqlite",root)
            report=cycles.analyze(root/"profile.sqlite",root)
            for case in report["cases"].values():
                for device in case["devices"].values():
                    self.assertEqual(device["count"],15)
                    self.assertAlmostEqual(device["mean_cycle_ms"],.5)

    def test_three_target_pieces_overlap_and_host_gaps(self):
        for eager in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp);steps(root)
                create(root/"profile.sqlite", root, eager=eager)
                report = cycles.analyze(root/"profile.sqlite", root)
                self.assertTrue(report["synthetic"])
                for case in report["cases"].values():
                    for device in case["devices"].values():
                        self.assertEqual(device["count"], 15)
                        phase = device["phase_mean_ms"]
                        self.assertAlmostEqual(phase["target"], .4)
                        self.assertAlmostEqual(phase["drafter"], .08)
                        self.assertAlmostEqual(phase["catch_up"], .09)
                        self.assertAlmostEqual(phase["sampling"], .08)
                        self.assertAlmostEqual(sum(phase.values()), 2)
                        target = device["target_by_rows"]["32"]
                        self.assertEqual(target["count"], 15)
                        self.assertAlmostEqual(target["mean_ms"], .4)
                        self.assertAlmostEqual(sum(target["kernels_mean_ms"].values()), .4)
                        row = device["cycles"][0]
                        self.assertAlmostEqual(row["host_api_overlap_with_gaps_ms"]["OSRT_API"], .03)
                        self.assertAlmostEqual(row["copy_or_memset_overlap_with_gaps_ms"], .02)
                        self.assertAlmostEqual(row["no_recorded_device_activity_ms"],
                                               phase["no_recorded_kernel"]-.02)

    def test_zero_draft_fallback_is_measured_without_inventing_drafter_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);steps(root)
            for c in (8,16):
                path=root/f"C{c}/steps.jsonl"
                rows=[json.loads(l) for l in path.read_text().splitlines()]
                for r in rows:
                    r.update(verify_items=0,decode_rows=c,live_rows=c,row_capacity=c)
                path.write_text("".join(json.dumps(r)+"\n" for r in rows))
            create(root/"profile.sqlite",root)
            report=cycles.analyze(root/"profile.sqlite",root)
            for case in report["cases"].values():
                for device in case["devices"].values():
                    self.assertEqual(device["phase_mean_ms"]["drafter"],0)
                    self.assertEqual(device["phase_mean_ms"]["catch_up"],0)

    def test_shifted_clock_and_missing_target_are_rejected(self):
        for shift in (10**9, -10**9):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);steps(root);create(root/"profile.sqlite",root)
                for path in root.glob("C*/steps.jsonl"):
                    rr=[json.loads(l) for l in path.read_text().splitlines()]
                    for r in rr:r["completed_unix_ns"]+=shift
                    path.write_text("".join(json.dumps(r)+"\n" for r in rr))
                with self.assertRaises(AssertionError):cycles.analyze(root/"profile.sqlite",root)


def dry_run(root):
    root.mkdir()
    for fault in ("", "start", "stop", "export", "geometry", "missing-acceptance"):
        stage=root/(fault or "success");stage.mkdir()
        config=dict(source=str(SOURCE), binary="unused", package="unused", checkpoint="unused",
                    native_revision="0"*40)
        (stage/"config.json").write_text(json.dumps(config))
        env=dict(os.environ, X7_PROFILE_FAIL=fault, PYTHONDONTWRITEBYTECODE="1")
        process=subprocess.run([sys.executable,"-B",str(Path(capture.__file__)),str(stage),
                                str(stage/"run"),"--dry-run"],env=env,capture_output=True,text=True,timeout=100)
        (stage/"driver.log").write_text(process.stdout+process.stderr)
        if fault:
            assert process.returncode != 0,(fault,process.stdout)
            assert not (stage/"run/summary.json").exists()
        else:
            assert process.returncode == 0,process.stdout+process.stderr
            report=json.loads((stage/"run/summary.json").read_text())
            assert report["case_count"]==4 and report["server_count"]==2 and report["bitwise"]
            assert [s["config"] for s in report["servers"]]==["uniform3","adaptive"]
        sessions=list((stage/"run/fixture-sessions").glob("*.json"))
        captured_both = fault not in ("start", "stop")
        assert len(sessions) == (2 if captured_both else 1), "post-processing dropped a capture"
        if captured_both:
            receipt = json.loads((stage/"run/captures.json").read_text())
            assert receipt["case_count"] == 4 and receipt["server_count"] == 2
        for session in sessions:
            value=json.loads(session.read_text())
            assert value["closed"] and value["starts"]==value["stops"]==1
            identity=json.loads((Path(value["directory"])/"server-group.json").read_text())
            try:os.kill(identity["pid"],0)
            except ProcessLookupError:pass
            else:raise AssertionError("server survived cleanup")
    print("PROFILE_DRY_RUN_OK cases=4 sessions=2 faults=5")


def adaptive_dry_run(root):
    root.mkdir()
    config = dict(source=str(SOURCE), binary="unused", package="unused", checkpoint="unused",
                  native_revision="0"*40, profile_policies=["adaptive"], full_workload_warmup=True)
    (root/"config.json").write_text(json.dumps(config))
    process = subprocess.run(
        [sys.executable, "-B", capture.__file__, str(root), str(root/"run"), "--dry-run"],
        capture_output=True, text=True, timeout=100)
    (root/"driver.log").write_text(process.stdout+process.stderr)
    assert process.returncode == 0, process.stdout+process.stderr
    result = json.loads((root/"run/summary.json").read_text())
    assert result["server_count"] == 1 and result["case_count"] == 2
    server, = result["servers"]
    assert server["config"] == "adaptive"
    assert server["session"]["starts"] == server["session"]["stops"] == 1
    warm = server["full_workload_warmup"]
    assert len(warm) == 4
    assert [(w["concurrency"], w["repeat"]) for w in warm] == [(8, 0), (8, 1), (16, 0), (16, 1)]
    assert all(w["tokens_per_request"] == 192 and w["bitwise"]["bitwise"]
               and not w["collected_by_nsight"] and "majflt" in w["faults"] for w in warm)
    assert {c["concurrency"] for c in server["cohorts"]} == {8, 16}
    print("ADAPTIVE_PROFILE_DRY_RUN_OK cases=2 sessions=1 full_warm_waves=4")


def replay(sqlite, log, output):
    db=sqlite3.connect(Path(sqlite).resolve().as_uri()+"?mode=ro&immutable=1",uri=True)
    epoch=db.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    lo,hi=db.execute("SELECT min(start),max(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    rows=[]
    for line in Path(log).read_text().splitlines():
        if '"schema":"chopinfer.step.v1"' in line:
            row=json.loads(line[line.index("{"):])
            if lo < row["completed_unix_ns"]-epoch < hi:
                rows.append(row)
    assert len(rows)>=70
    target=Path(output)/"C8";target.mkdir(parents=True)
    (target/"steps.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    result=cycles.analyze(Path(sqlite).resolve(),Path(output),historical=True)
    assert not result["synthetic"]
    (Path(output)/"attribution.json").write_text(json.dumps(result,indent=2)+"\n")
    counts=[d["count"] for d in result["cases"]["8"]["devices"].values()]
    assert all(n>=60 for n in counts)
    print("HISTORICAL_CYCLE_REPLAY_OK",counts)


if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--dry-run":
        dry_run(Path(sys.argv[2]))
    elif len(sys.argv)>1 and sys.argv[1]=="--adaptive-dry-run":
        adaptive_dry_run(Path(sys.argv[2]))
    elif len(sys.argv)>1 and sys.argv[1]=="--replay":
        replay(*sys.argv[2:])
    else:
        unittest.main(verbosity=2)

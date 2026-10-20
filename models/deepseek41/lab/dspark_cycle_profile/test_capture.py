"""CPU regression tests for capture ownership, alignment and attribution."""
import json
import os
from pathlib import Path
import subprocess
import sqlite3
import sys
import tempfile
import unittest

import capture
import cycles
from sqlite_fixture import create

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
        assert len(sessions)==(2 if not fault else 1)
        for session in sessions:
            value=json.loads(session.read_text())
            assert value["closed"] and value["starts"]==value["stops"]==1
            identity=json.loads((Path(value["directory"])/"server-group.json").read_text())
            try:os.kill(identity["pid"],0)
            except ProcessLookupError:pass
            else:raise AssertionError("server survived cleanup")
    print("PROFILE_DRY_RUN_OK cases=4 sessions=2 faults=5")


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
    elif len(sys.argv)>1 and sys.argv[1]=="--replay":
        replay(*sys.argv[2:])
    else:
        unittest.main(verbosity=2)

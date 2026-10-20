"""Small synthetic graph and eager activities in the native SQLite schema."""
import json
import os
from pathlib import Path
import sqlite3

from cycles import EMBED, MARKOV

HEAD = "chop_deepseek41_head_f32_test_cached_reduce_v1"
ACCEPT = "chop_deepseek41_dspark_host_acceptance_u32_v1"


def create(path, directory, *, eager=False):
    steps = []
    for concurrency in (8, 16):
        steps += [json.loads(line) for line in
                  (Path(directory) / f"C{concurrency}/steps.jsonl").read_text().splitlines()]
    epoch = min(s["completed_unix_ns"] for s in steps) - 1_000_000_000
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE TARGET_INFO_SESSION_START_TIME(utcEpochNs INTEGER);
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,
            graphId INTEGER,gridX INTEGER,shortName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER);
        CREATE TABLE OSRT_API(start INTEGER,end INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER);
        CREATE TABLE CPU_FIXTURE(value INTEGER);
    """)
    db.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES(?)", (epoch,))
    names = [EMBED, MARKOV, HEAD, ACCEPT, "chop_deepseek41_dspark_quantize_w15360_bf16_v1",
             "chop_deepseek41_mxfp8_ticket_n5120_k15360_g1_f32_v2",
             "target_middle", "target_tail", "chop_sampling_filtered_finish_f32_v1",
             "chop_dspark_argmax_f32_v1"]
    db.executemany("INSERT INTO StringIds VALUES(?,?)", enumerate(names, 1))
    previous = None
    for step in steps:
        completed = step["completed_unix_ns"]
        # The HTTP fixture can finish steps faster on an idle CPU. Keep
        # every fabricated kernel within half of the observed step period.
        gap = completed - previous if previous is not None else 1_800_000
        assert gap > 0
        span = min(900_000, gap // 2)
        origin = completed - epoch - span
        at = lambda offset: origin + offset * span // 900_000
        previous = completed
        width = step["row_capacity"]
        for device in range(4):
            target, draft, catch = 100 + device*10, 200 + device, 300 + device
            rows = [(300_000, 310_000, target, width, 1),
                    (310_000, 410_000, target, 1, 7),
                    (400_000, 620_000, target+1, 1, 7),
                    (600_000, 680_000, target+2, 1, 8),
                    (680_000, 700_000, target+2, 1, 3)]
            if step["verify_items"]:
                rows += [(10_000, 20_000, catch, 1, 5), (20_000, 100_000, catch, 1, 6),
                         (120_000, 130_000, draft, 5*step["verify_items"], 1),
                         (130_000, 180_000, draft, 1, 2), (180_000, 200_000, draft, 1, 10),
                         (200_000, 220_000, draft, 1, 3)]
            sample = 4 if step["verify_items"] and device == 0 else 9
            if os.environ.get("X7_PROFILE_FAIL") != "missing-acceptance" or sample != 4:
                rows.append((720_000, 780_000, None, 1, sample))
            if os.environ.get("X7_PROFILE_FAIL") == "geometry":
                rows[0] = (*rows[0][:3], width+1, 1)
            db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?)",
                [(at(a), at(b), device, None if eager else graph, grid, name)
                 for a, b, graph, grid, name in rows])
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?)",
                       (at(230_000), at(250_000), device))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?)",
                   (at(220_000), at(300_000)))
        db.execute("INSERT INTO OSRT_API VALUES(?,?)", (at(230_000), at(260_000)))
    db.commit()
    db.close()

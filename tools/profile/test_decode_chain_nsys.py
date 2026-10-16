import sqlite3
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import unittest

from decode_chain_nsys import main, summarize


class DecodeChainNsysTest(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.connection.executescript(
            "CREATE TABLE TARGET_INFO_GPU(id INTEGER, cuDevice INTEGER);"
            "CREATE TABLE StringIds(id INTEGER, value TEXT);"
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
            "deviceId INTEGER, graphId INTEGER, graphNodeId INTEGER, "
            "start INTEGER, end INTEGER, shortName INTEGER);"
        )
        self.connection.executemany(
            "INSERT INTO TARGET_INFO_GPU VALUES (?, ?)",
            [(9 - rank, rank) for rank in range(4)],
        )
        self.connection.executemany(
            "INSERT INTO StringIds VALUES (?, ?)", [(0, "norm"), (1, "gemm")]
        )
        rows = []
        for rank in range(4):
            for replay in range(6):
                start = replay * 100_000 + rank * 1_000
                rows.extend(
                    [(9 - rank, 10 + rank, 100, start, start + 10_000, 0),
                     (9 - rank, 10 + rank, 101, start + 8_000, start + 14_000, 1)]
                )
        self.connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?)", rows
        )

    def tearDown(self):
        self.connection.close()

    def test_rank_mapping_and_overlap_gaps(self):
        report = summarize(self.connection, [0, 1, 2, 3])
        self.assertEqual(report["trace_critical_path_us"], 17)
        for rank, data in enumerate(report["ranks"]):
            self.assertEqual((data["rank"], data["gpu_ordinal"]), (rank, rank))
            self.assertEqual(data["trace_graph_us"], 14)
            self.assertEqual(data["stages"][1]["gap_before_us"], -2)
            self.assertEqual(data["stages"][1]["tail_after_previous_us"], 4)

    def test_wrappers_forward_gpu_lists_to_build(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / 'model.safetensors.index.json').write_text('{}')
            cargo = temp / 'cargo'
            cargo.write_text('#!/bin/sh\nprintf "%s" "$CHOPINFER_KERNEL_GPUS" > "$CAPTURE"\nexit 19\n')
            cargo.chmod(0o755)
            scripts = temp / 'tools/profile'
            scripts.mkdir(parents=True)
            for phase, size in [('decode', '1'), ('prefill', '2048')]:
                # An isolated script root cannot source the user's local.env
                # and replace our stub cargo with a real GPU build or profiler.
                script = scripts / f'{phase}_chain_nsys.sh'
                script.write_text((root / f'tools/profile/{phase}_chain_nsys.sh').read_text())
                for devices in (None, '2,3', '7,6,5,4,3,2,1,0'):
                    env = dict(os.environ, PATH=str(temp) + ':' + os.environ['PATH'],
                               TMPDIR=directory, CHOPINFER_LOCAL_ENV=str(temp / 'absent'),
                               CHOPINFER_MIMO_CHECKPOINT=directory, CAPTURE=str(temp / 'capture'))
                    env.pop('CHOPINFER_KERNEL_GPUS', None)
                    if devices is not None:
                        env['CHOPINFER_KERNEL_GPUS'] = devices
                    result = subprocess.run(['bash', str(script),
                                             '1', size, str(temp / 'out')], env=env,
                                            capture_output=True, text=True)
                    # Stop at the stubbed build, before any profiler or GPU use.
                    self.assertEqual(result.returncode, 19, result.stderr)
                    self.assertEqual((temp / 'capture').read_text(), devices or '0,1,2,3')

    def test_cli_accepts_subset_and_reordered_ordinals(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'trace.sqlite'
            output = Path(directory) / 'report.json'
            self.connection.commit()
            with sqlite3.connect(database) as destination:
                self.connection.backup(destination)
            for ordinals in ('2,3', '3,1,0,2'):
                with patch.object(sys, 'argv', ['profile', str(database), '--layer', '1',
                                               '--batch', '1', '--ordinals', ordinals,
                                               '--output', str(output)]):
                    main()
                report = json.loads(output.read_text())
                self.assertEqual([row['gpu_ordinal'] for row in report['ranks']],
                                 [int(value) for value in ordinals.split(',')])

    def test_incomplete_replay_is_rejected(self):
        self.connection.execute(
            "DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=9 AND start=508000"
        )
        with self.assertRaisesRegex(ValueError, "six complete graph launches"):
            summarize(self.connection, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()

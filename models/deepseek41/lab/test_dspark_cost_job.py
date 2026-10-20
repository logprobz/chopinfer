import json
import contextlib
from pathlib import Path
import tempfile
import unittest
import dspark_cost_job as job
import dspark_cost_stage as sealing
from dspark_cost_analysis import replay_retention


class CostJobTests(unittest.TestCase):
    def test_drafter_comparison_covers_preparation_and_stops_before_timing(self):
        class Fixture(job.Runner):
            def gpu_guard(self, *args, **kwargs):
                pass

            @contextlib.contextmanager
            def serving(self, config, epoch):
                self.fixture_policy = config
                self.directory = self.output / f"{epoch}-{config}"
                self.directory.mkdir()
                self.log = self.directory / "server.log"
                self.log.write_text("")
                yield

            def prepare(self):
                markers = list(self.output.glob("draft-check-*"))
                expected = self.env["CHOPINFER_DSPARK_DRAFT_CHUNK_ITEMS"] == "6" and self.fixture_policy == "uniform3"
                assert len(markers) == int(expected)
                assert self.env["CHOPINFER_DSPARK_GRAPH_HANDLES_PER_RANK"] == "512"
                return []

            def qualify_tails(self):
                return [dict(concurrency=c) for c in (3, 5, 7, 14, 15)]

            def cohort(self, config, concurrency):
                assert not list(self.output.glob("draft-check-*"))
                path = self.directory / f"C{concurrency}"
                path.mkdir()
                (path / "timed.log").write_text("")
                return dict(concurrency=concurrency, host_load=dict(idle=True))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(dict(
                source=str(Path(__file__).resolve().parents[3]))))
            result = Fixture(root, root / "output", dry=True).run()
            self.assertEqual(result["case_count"], 10)
            self.assertEqual(result["tail_case_count"], 10)

    def test_handle_and_observed_allocation_limits_remain_bounded(self):
        graph = dict(schema="chopinfer.graph-cache.v1", capacity=2048,
                     peak=1540, released=0)
        self.assertEqual(job.capture_memory(json.dumps(graph))["peak_handles"], 1540)
        graph["capacity"] = 2049
        with self.assertRaises(AssertionError):
            job.capture_memory(json.dumps(graph))
        held = job.GRAPH_ALLOCATION_LIMIT + 1
        with self.assertRaisesRegex(AssertionError, "2 GiB"):
            job.capture_memory(f"graph capture memory rank 0: {held} bytes free before, 0 after, {held} held")

    def test_seal_removes_write_permissions_from_root_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "install/src").mkdir(parents=True)
            (root / "install/cache/p").mkdir(parents=True)
            (root / "install/src/.commit").write_text("0" * 40)
            (root / "NATIVE_REVISION").write_text("0" * 40)
            (root / "install/CHECKPOINT").write_text(str(root / "checkpoint"))
            (root / "install/PACKAGE").write_text(str(root / "install/cache/p"))
            for name in ("cpu-validation", "native-preflight", "timeout-audit", "config"):
                path = root / f"{name}.json"
                path.write_text(json.dumps(dict(passed=True)))
                path.chmod(0o666)
            sealing.seal(root)
            receipt = json.loads((root / "staged-job.json").read_text())
            sealing.stage.verify(root, receipt["manifest_sha256"])
            for path in sealing.stage.files(root):
                self.assertFalse(path.stat().st_mode & 0o022)

    def test_memory_reports_allocations_without_calling_them_a_bound(self):
        text = "\n".join([
            "graph capture memory rank 0: 100 bytes free before, 60 after, 40 held",
            "graph capture memory rank 0: 60 bytes free before, 55 after, 5 held",
            json.dumps(dict(schema="chopinfer.graph-cache.v1", capacity=1536,
                            peak=12, released=0)),
        ])
        result = job.capture_memory(text)
        self.assertEqual(result["capture_allocated_bytes_by_rank"], {"0": 45})
        self.assertEqual(result["memory_samples_by_rank"], {"0": 2})
        self.assertEqual(result["peak_handles"], 12)
        with self.assertRaises(AssertionError):
            job.capture_memory(text.replace("5 held", "6 held"))

    def test_retention_replay_preserves_a_warmed_key_after_tail_pressure(self):
        def graph(event, width, items, epoch=0):
            return dict(schema="chopinfer.graph-cache.v1", event=event,
                family="target", rank=0, segment=0, width=width, items=items, epoch=epoch)
        text = "\n".join(map(json.dumps, [
            graph("capture", 32, 16), graph("target-eager", 31, 14),
            graph("eviction-begin", None, None), graph("capture", 31, 14, 1),
            graph("capture", 32, 16, 1),
        ]))
        replay = replay_retention(text, capacity=12)
        self.assertTrue(replay["protected_32_by_16"])
        self.assertEqual(replay["retained_handles"], 12)
        self.assertEqual(replay["avoided_recapture_groups"], 1)
        self.assertEqual(replay["first_32_by_16"]["epoch"], 0)
        self.assertEqual(replay["denied"][0]["rows"], 31)

    def test_eager_tail_costs_use_completed_steps_and_preserve_full_capture_guard(self):
        text = "\n".join(map(json.dumps, [
            dict(schema="chopinfer.graph-cache.v1", event="target-eager", family="target"),
            dict(schema="chopinfer.step.v1", prefill_items=0, decode_rows=10, verify_items=4,
                 row_capacity=31, execution_us=88092),
            dict(schema="chopinfer.graph-cache.v1", event="capture", family="target", width=32, items=16),
            dict(schema="chopinfer.step.v1", prefill_items=0, decode_rows=0, verify_items=16, batch=933),
        ]))
        result = job.capture_hits(text, 16)
        self.assertEqual(result["target_capture_steps"], 1)
        self.assertEqual(result["capture_misses"][0]["batch"], 933)
        self.assertEqual(result["tail_eager_steps"], 1)
        self.assertEqual(result["tail_eager_rate"], 1)
        self.assertEqual(result["tail_eager_mean_ms"], 88.092)

    def test_target_misses_belong_to_their_own_completed_step(self):
        records = [
            dict(schema="chopinfer.graph-cache.v1", event="capture", family="target"),
            dict(schema="chopinfer.step.v1", prefill_items=0, decode_rows=0, verify_items=7),
            dict(schema="chopinfer.step.v1", prefill_items=0, decode_rows=0, verify_items=8),
            dict(schema="chopinfer.graph-cache.v1", event="capture", family="catch-up"),
            dict(schema="chopinfer.step.v1", prefill_items=0, decode_rows=0, verify_items=8),
            dict(schema="chopinfer.graph-cache.v1", event="capture", family="target"),
            dict(schema="chopinfer.step.v1", prefill_items=0, decode_rows=1, verify_items=7),
        ]
        result = job.capture_hits("\n".join(map(json.dumps, records)), 8)
        self.assertEqual(result["full_verify_steps"], 3)
        self.assertEqual(result["target_capture_steps"], 1)
        self.assertAlmostEqual(result["hit_rate"], 2 / 3)
        self.assertIsNone(job.capture_hits("", 8)["hit_rate"])


if __name__ == "__main__":
    unittest.main()

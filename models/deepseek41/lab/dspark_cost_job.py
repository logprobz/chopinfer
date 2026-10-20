"""Compare whole and chunked drafts with production exact-width warmup."""
import argparse
import collections
import json
from pathlib import Path
import re
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
MATRIX = HERE / "verify_width_matrix"
sys.path.insert(0, str(MATRIX if MATRIX.exists() else HERE.parent))
import run as matrix

GRAPH_HANDLES_PER_RANK = 512
GRAPH_ALLOCATION_LIMIT = 2 << 30


def capture_memory(text):
    per_rank = collections.Counter()
    count = collections.Counter()
    minimum_free = {}
    for rank, before, after, held in re.findall(
        r"graph capture memory rank (\d+): (\d+) bytes free before, (\d+) after, (\d+) held", text
    ):
        assert max(0, int(before) - int(after)) == int(held)
        per_rank[rank] += int(held)
        count[rank] += 1
        minimum_free[rank] = min(minimum_free.get(rank, int(after)), int(after))
    graphs = matrix.records(text, "chopinfer.graph-cache.v1")
    peaks = [g["peak"] for g in graphs]
    assert all(g["capacity"] <= GRAPH_HANDLES_PER_RANK * 4 for g in graphs)
    assert all(n <= GRAPH_ALLOCATION_LIMIT for n in per_rank.values()), "graph capture allocation exceeds 2 GiB per rank"
    return dict(capture_allocated_bytes_by_rank=dict(per_rank),
                memory_samples_by_rank=dict(count),
                minimum_free_bytes_by_rank=minimum_free,
                capacities=sorted({g["capacity"] for g in graphs}),
                peak_handles=max(peaks, default=0),
                releases=max((g["released"] for g in graphs), default=0),
                limit_handles=GRAPH_HANDLES_PER_RANK * 4,
                allocation_limit_bytes_per_rank=GRAPH_ALLOCATION_LIMIT,
                bytes_scope="Sum of observed CUDA free-memory deltas at captures, including recaptures; not an allocator upper bound.")


def capture_hits(text, concurrency):
    captures, steps, misses = [], 0, 0
    for line in text.splitlines():
        if "{" not in line:
            continue
        try:
            r = json.loads(line[line.index("{"):])
        except ValueError:
            continue
        if r.get("schema") == "chopinfer.graph-cache.v1" and r.get("event") == "capture":
            captures.append(r)
        elif r.get("schema") == "chopinfer.step.v1":
            if not r["prefill_items"] and r["decode_rows"] + r["verify_items"] == concurrency and r["verify_items"]:
                steps += 1
                misses += any(g["family"] == "target" for g in captures)
            captures = []
    return dict(full_verify_steps=steps, target_capture_steps=misses,
                hit_rate=None if not steps else 1 - misses / steps)


class Runner(matrix.Runner):
    def __init__(self, stage, output, dry=False):
        super().__init__(stage, output, "cost", dry)
        self.settings["startup_timeout"] = 240
        self.chunk = 32
        self.env["CHOPINFER_DSPARK_GRAPH_HANDLES_PER_RANK"] = str(GRAPH_HANDLES_PER_RANK)
        self.deadline = self.started + (120 if dry else 1140)

    def argv(self, config, port):
        argv = super().argv(config, port)
        if not self.dry:
            argv[argv.index("--verify-warmup") + 1] = "on"
            argv[argv.index("--max-sequences") + 1] = "32"
        return argv

    def prepare(self):
        receipts = []
        for c in (8, 16):
            for repeat in range(2):
                begin = self.log.stat().st_size
                before = matrix.faults(self.server.pid)
                requests = self.wave(c, matrix.TOKENS)
                after = self.scrape()
                time.sleep(.02 if self.dry else .2)
                assert after == self.scrape()
                key = str(c)
                if key not in self.reference:
                    self.reference[key] = requests
                parity = self.exact.compare({key: self.reference[key]}, {key: requests})
                assert parity["bitwise"], parity
                text = self.log.read_bytes()[begin:].decode()
                checks = matrix.records(text, "chopinfer.dspark-draft-parity.v1")
                assert all(r["bitwise"] and r["positions"] == 5 and r["ranks"] == 4 for r in checks)
                if not self.dry and self.chunk == 6 and self.policy == "uniform3":
                    assert any(r["items"] == c for r in checks), "diverse workload missed whole-versus-chunked drafter check"
                receipts.append(dict(concurrency=c, repeat=repeat, bitwise=True,
                    drafter_bitwise_checks=checks,
                    faults=matrix.fault_delta(before, matrix.faults(self.server.pid))))
        return receipts

    def run(self):
        self.gpu_guard("before", idle=True)
        servers = []
        try:
            for epoch, (config, chunk) in enumerate([
                ("adaptive", 32), ("uniform3", 32),
                ("uniform3", 6), ("adaptive", 6),
            ]):
                self.chunk, self.policy = chunk, config
                self.env["CHOPINFER_DSPARK_DRAFT_CHUNK_ITEMS"] = str(chunk)
                marker = self.output / f"draft-check-{epoch}"
                if chunk == 6:
                    marker.write_text("compare complete five-position blocks during warmup\n")
                    self.env["CHOPINFER_DSPARK_DRAFT_CHECK"] = str(marker)
                else:
                    self.env.pop("CHOPINFER_DSPARK_DRAFT_CHECK", None)
                with self.serving(config, epoch):
                    matrix.write(self.directory / "policy.json", dict(
                        policy=config, draft_chunk_items=chunk,
                        graph_handles_per_rank=GRAPH_HANDLES_PER_RANK,
                        graph_allocation_limit_bytes_per_rank=GRAPH_ALLOCATION_LIMIT))
                    startup = self.log.read_text()
                    checks = matrix.records(startup, "chopinfer.dspark-draft-parity.v1")
                    if chunk == 6 and not self.dry:
                        assert {8, 16} <= {r["items"] for r in checks}, "missing same-input drafter comparison"
                        assert all(r["bitwise"] and r["positions"] == 5 and r["ranks"] == 4 for r in checks)
                    memory = capture_memory(startup)
                    if not self.dry:
                        assert memory["memory_samples_by_rank"].keys() == {"0", "1", "2", "3"}
                        assert memory["capacities"] == [GRAPH_HANDLES_PER_RANK * 4], "configured graph capacity differs"
                        assert memory["releases"] == 0, "startup evicted warmed geometries"
                    # Fixed-policy warmup qualifies diverse draft bits without
                    # teaching adaptive's cost estimator the comparison overhead.
                    if config == "adaptive":
                        marker.unlink(missing_ok=True)
                    preparation = self.prepare()
                    checks = matrix.records(self.log.read_text(), "chopinfer.dspark-draft-parity.v1")
                    marker.unlink(missing_ok=True)
                    prepared_memory = capture_memory(self.log.read_text())
                    assert prepared_memory["releases"] == 0, "workload warmup evicted warmed geometries"
                    cohorts = [self.cohort(config, c) for c in ((8, 16) if epoch % 2 == 0 else (16, 8))]
                    for cohort in cohorts:
                        c = cohort["concurrency"]
                        text = (self.directory / f"C{c}/timed.log").read_text()
                        cohort["graph_hits"] = capture_hits(text, c)
                        if not self.dry:
                            assert cohort["graph_hits"]["target_capture_steps"] == 0, "full-batch target captured during timing"
                    text = self.log.read_text()
                    final_memory = capture_memory(text)
                    assert final_memory["releases"] == 0, "timing evicted warmed geometries"
                    servers.append(dict(policy=config, chunk_items=chunk,
                        cohorts=cohorts, full_warmup=preparation,
                        drafter_bitwise_checks=checks, startup_graphs=memory,
                        prepared_graphs=prepared_memory, final_graphs=final_memory,
                        resident_cache=matrix.cache_ready(text, required=not self.dry)))
                self.gpu_guard(f"after-{epoch}")
        finally:
            if self.server is not None:
                matrix.stop(self.server)
            self.gpu_guard("final")
        assert len(servers) == 4 and sum(len(s["cohorts"]) for s in servers) == 8
        result = dict(schema="chopinfer.dspark-cost.v1", status="passed",
            dry_run=self.dry, server_count=4, case_count=8, bitwise=True,
            timing_valid=all(c["host_load"]["idle"] for s in servers for c in s["cohorts"]),
            servers=servers, elapsed_seconds=time.monotonic()-self.started)
        matrix.write(self.output / "summary.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    def interrupted(number, _frame):
        raise InterruptedError(f"signal {number}")
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, interrupted)
    runner = Runner(args.stage, args.output, args.dry_run)
    try:
        result = runner.run()
        print(json.dumps({k: result[k] for k in ("status", "server_count", "case_count", "elapsed_seconds", "timing_valid")}))
    except BaseException as error:
        matrix.write(runner.output / "failure.json", dict(error=repr(error)))
        raise


if __name__ == "__main__":
    main()

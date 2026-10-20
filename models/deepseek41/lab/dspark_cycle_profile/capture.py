"""Capture two policies sequentially, once per process, without changing serving."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
MATRIX = HERE.parent / "verify_width_matrix"
if not MATRIX.exists():
    MATRIX = HERE.parent
sys.path.insert(0, str(MATRIX))
import run as width

from session import Session
import cycles


class Runner(width.Runner):
    def __init__(self, stage, output, dry=False):
        super().__init__(stage, output, "profile", dry)
        if not dry:
            self.deadline = self.started + 1020
        self.session = None
        self.env["X7_PROFILE_FIXTURE"] = "1" if dry else "0"
        self.env["X7_PROFILE_FIXTURE_ROOT"] = str(self.output / "fixture-sessions")

    def argv(self, config, port):
        command = super().argv(config, port)
        self.session = Session(self, config)
        return self.session.launch(command)

    @contextlib.contextmanager
    def profiled(self, config, epoch):
        try:
            with super().serving(config, epoch):
                launcher = self.server
                assert self.session.pidfile.is_file(), "server identity missing"
                identity = json.loads(self.session.pidfile.read_text())
                assert identity["pid"] == identity["pgid"]
                self.server = SimpleNamespace(pid=identity["pid"])
                try:
                    self.session.start()
                    yield
                finally:
                    try:
                        self.session.close()
                    finally:
                        self.server = launcher
        finally:
            if self.session is not None:
                self.session.close()

    def run(self):
        self.gpu_guard("before", idle=True)
        servers = []
        try:
            for epoch, config in enumerate(("uniform3", "adaptive")):
                with self.profiled(config, epoch):
                    cohorts = [self.cohort(config, c) for c in (8, 16)]
                    ready = width.cache_ready(self.log.read_text())
                # Export only after capture has stopped and its server exited.
                sqlite = self.session.export()
                attribution = cycles.analyze(sqlite, self.directory)
                assert attribution["synthetic"] == self.dry
                width.write(self.directory / "attribution.json", attribution)
                assert self.session.starts == self.session.stops == 1
                servers.append(dict(config=config, cohorts=cohorts, resident_cache=ready,
                                    attribution=attribution,
                                    session=self.session.receipt()))
                self.gpu_guard(f"after-{epoch}", idle=True)
            assert len(servers) == 2 and sum(len(s["cohorts"]) for s in servers) == 4
        finally:
            self.gpu_guard("final")
        result = dict(schema="chopinfer.dspark-cycle-profile.v1", status="passed",
                      dry_run=self.dry, native_revision=self.settings["native_revision"],
                      server_count=2, case_count=4, bitwise=True, servers=servers,
                      elapsed_seconds=time.monotonic() - self.started,
                      timing="Instrumented phase attribution, not a throughput benchmark",
                      cpu_acceptance="Host sampling/commit timers are reported separately; they are not GPU phase times.")
        width.write(self.output / "summary.json", result)
        (self.output / "summary.md").write_text(cycles.markdown(result))
        return result


def child(pidfile, command):
    # Nsight can already make the child a group leader, which forbids setsid.
    # Cleanup needs a private process group, whether inherited or created here.
    if os.getpgrp() != os.getpid():
        os.setsid()
    assert os.getpgrp() == os.getpid()
    width.write(Path(pidfile), dict(pid=os.getpid(), pgid=os.getpgrp(),
                                   start_ticks=width.faults(os.getpid())["start_ticks"]))
    os.execv(command[0], command)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "server-child":
        child(sys.argv[2], sys.argv[3:])
        return
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
        print(json.dumps({k: result[k] for k in ("status", "case_count", "elapsed_seconds")}))
    except BaseException as error:
        width.write(runner.output / "failure.json", dict(status="failed", error=repr(error)))
        raise


if __name__ == "__main__":
    main()

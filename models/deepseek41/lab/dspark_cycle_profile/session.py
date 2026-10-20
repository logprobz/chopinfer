"""Own one Nsight session and its one server process."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class Session:
    def __init__(self, runner, config):
        self.runner = runner
        self.name = f"x7-cycle-{config}-{os.getpid()}-{time.time_ns()}"
        self.pidfile = runner.directory / "server-group.json"
        store = Path(runner.settings.get("profile_store", runner.output / "captures"))
        self.prefix = store / self.name / "capture"
        self.prefix.parent.mkdir(parents=True)
        self.starts = self.stops = 0
        self.started = self.closed = False
        self.tool = ([sys.executable, "-B", str(Path(__file__).with_name("nsys_fixture.py"))]
                     if runner.dry else [runner.settings.get("nsys_path", "nsys")])

    def launch(self, command):
        owned = [sys.executable, "-B", str(Path(__file__).with_name("capture.py")),
                 "server-child", str(self.pidfile), *command]
        return self.tool + ["launch", "--session-new", self.name, "--trace=cuda,osrt",
                            "--cuda-graph-trace=node", "--cuda-flush-interval=100",
                            "--wait=all", *owned]

    def call(self, action, extra=(), timeout=90, cleanup=False):
        args = self.tool + [action]
        if action != "export":
            args += ["--session", self.name]
        args += list(map(str, extra))
        result = subprocess.run(args, env=self.runner.env, capture_output=True, text=True,
                                timeout=timeout if cleanup else self.runner.remaining(timeout))
        (self.runner.directory / f"nsys-{action}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, (action, result.returncode, result.stderr[-2000:])
        return result

    def start(self):
        assert self.starts == 0 and not self.closed
        self.starts += 1
        # Even a failed start can leave collection active; cleanup stops it once.
        self.started = True
        self.call("start", ["--sample=none", "--cpuctxsw=none", "--output", self.prefix], 30)

    def close(self):
        if self.closed:
            return
        self.closed = True
        errors = []
        if self.started:
            self.stops += 1
            try:
                self.call("stop", timeout=90, cleanup=True)
            except BaseException as error:
                errors.append(str(error))
        if self.pidfile.is_file():
            identity = json.loads(self.pidfile.read_text())
            pid = identity["pid"]
            try:
                parts = (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
                assert int(parts[19]) == identity["start_ticks"], "server PID reused"
                os.killpg(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except FileNotFoundError:
                pass
        try:
            self.call("shutdown", ["--kill=sigkill"], timeout=15, cleanup=True)
        except BaseException as error:
            errors.append(str(error))
        if self.pidfile.is_file():
            identity = json.loads(self.pidfile.read_text())
            try:
                parts = (Path("/proc") / str(identity["pid"]) / "stat").read_text().rsplit(") ", 1)[1].split()
                assert int(parts[19]) == identity["start_ticks"], "server PID reused"
                os.killpg(identity["pgid"], signal.SIGKILL)
            except (ProcessLookupError, FileNotFoundError):
                pass
        assert not errors, errors

    def export(self):
        reports = list(self.prefix.parent.glob("*.nsys-rep"))
        assert len(reports) == 1, "missing or ambiguous Nsight report"
        path = self.prefix.with_suffix(".sqlite")
        self.call("export", ["--type=sqlite", "--output", path, reports[0]], 120)
        assert path.is_file()
        return path

    def receipt(self):
        return dict(name=self.name, starts=self.starts, stops=self.stops,
                    server_identity=json.loads(self.pidfile.read_text()),
                    capture_prefix=str(self.prefix), closed=self.closed)

"""CPU-only Nsight lifecycle fixture, selected only by capture --dry-run."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from sqlite_fixture import create


def main():
    assert os.environ.get("X7_PROFILE_FIXTURE") == "1"
    args = sys.argv[1:]
    action = args[0]
    def option(key):
        return args[args.index(key) + 1]
    if action == "export":
        if os.environ.get("X7_PROFILE_FAIL") == action:
            return 2
        receipt = json.loads(Path(args[-1]).read_text())
        create(Path(option("--output")), Path(receipt["directory"]))
        return 0
    name = option("--session-new" if action == "launch" else "--session")
    root = Path(os.environ["X7_PROFILE_FIXTURE_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / (name + ".json")
    if action == "launch":
        for other in root.glob("*.json"):
            assert json.loads(other.read_text())["closed"], "overlapping servers"
        command = args[args.index("--wait=all") + 1:]
        pidfile = Path(command[command.index("server-child") + 1])
        state = dict(name=name, directory=str(pidfile.parent), starts=0, stops=0, closed=False)
        path.write_text(json.dumps(state))
        process = subprocess.Popen(command)
        return process.wait()
    state = json.loads(path.read_text())
    if action == "start":
        assert state["starts"] == 0 and not state["closed"]
        state["starts"] += 1
        state["output"] = option("--output")
    elif action == "stop":
        assert state["starts"] == 1 and state["stops"] == 0
        state["stops"] += 1
        Path(state["output"]).with_suffix(".nsys-rep").write_text(json.dumps(state))
    elif action == "shutdown":
        assert not state["closed"]
        state["closed"] = True
        pidfile = Path(state["directory"]) / "server-group.json"
        if pidfile.is_file():
            try:
                os.killpg(json.loads(pidfile.read_text())["pgid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
    else:
        raise AssertionError(action)
    path.write_text(json.dumps(state))
    return 2 if os.environ.get("X7_PROFILE_FAIL") == action else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Seal and execute the bounded whole-machine drafter comparison."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
MATRIX = HERE / "verify_width_matrix"
sys.path.insert(0, str(MATRIX if MATRIX.exists() else HERE.parent))
import stage
from run import digest, write


def seal(root):
    for name in ("cpu-validation", "native-preflight", "timeout-audit"):
        assert json.loads((root / f"{name}.json").read_text())["passed"], name
    stage.configure(root)
    destination = root / "harness/cost"
    destination.mkdir(exist_ok=True)
    for name in ("dspark_cost_stage.py", "dspark_cost_job.py"):
        shutil.copyfile(HERE / name, destination / name)
    for directory in (root / "install", root / "harness"):
        for path in [directory, *directory.rglob("*")]:
            path.chmod(path.stat().st_mode & ~0o022)
    selected = stage.files(root)
    for path in [root, *selected]:
        path.chmod(path.stat().st_mode & ~0o022)
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(f"{digest(p)}  {p.relative_to(root)}\n" for p in selected))
    pin = digest(manifest)
    script = destination / "dspark_cost_stage.py"
    job = root / "x7-prewarm-drafter.job"
    job.write_text(f"""# timeout 1200
# Whole-machine sampled C8/C16 adaptive and K3, 250 W on every GPU.
set -euo pipefail
source "$HOME/.config/chopinfer/local.env"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONDONTWRITEBYTECODE=1
unset CHOPINFER_KERNEL_GPU CHOPINFER_KERNEL_GPUS CHOPINFER_KERNEL_REQUIRE
exec timeout --signal=TERM --kill-after=15s 1170s python3 -B {shlex.quote(str(script))} {shlex.quote(str(root))} --execute --manifest {pin}
""")
    job.chmod(0o644)
    subprocess.run(["bash", "-n", str(job)], check=True)
    stage.verify(root, pin)
    receipt = dict(job=str(job), sha256=digest(job), manifest_sha256=pin,
                   graph_handles_per_rank=512, graph_allocation_limit_bytes_per_rank=2 << 30,
                   native_revision=(root / "NATIVE_REVISION").read_text().strip(),
                   queued=False, server_count=4, case_count=8, timeout_seconds=1200)
    write(root / "staged-job.json", receipt)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seal", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.seal:
        seal(root)
        return
    stage.verify(root, args.manifest)
    if args.verify:
        return
    os.environ["X7_JOB_STARTED_MONOTONIC"] = str(time.monotonic())
    settings = json.loads((root / "config.json").read_text())
    source = Path(settings["source"])
    check = subprocess.run([sys.executable, "-B", "-m", "tools.refrun.stamps",
        "--checkpoint", settings["checkpoint"], "--lock", "models/deepseek41/checkpoint-files.json"],
        cwd=source, capture_output=True, text=True, timeout=30)
    assert check.returncode == 0, check.stderr
    output = root / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}")
    output.parent.mkdir(exist_ok=True)
    os.execv(sys.executable, [sys.executable, "-B",
        str(HERE / "dspark_cost_job.py"), str(root), str(output)])


if __name__ == "__main__":
    main()

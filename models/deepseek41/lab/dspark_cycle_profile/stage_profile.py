"""Seal one whole-machine profiling job. This never submits a job."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
MATRIX = HERE.parent / "verify_width_matrix"
if not MATRIX.exists():
    MATRIX = HERE.parent
sys.path.insert(0, str(MATRIX))
import stage as freezing
from run import digest, write


def seal(root):
    for name in ("cpu-validation", "native-preflight", "timeout-audit"):
        assert json.loads((root / f"{name}.json").read_text())["passed"], name
    freezing.configure(root)
    settings = json.loads((root / "config.json").read_text())
    nsys = shutil.which("nsys")
    assert nsys is not None
    version = subprocess.run([nsys, "--version"], check=True, capture_output=True, text=True).stdout.strip()
    settings.update(nsys_path=nsys, nsys_version=version,
                    profile_store=str(Path("/tmp") / "chopinfer-x7-cycle-captures"))
    write(root / "config.json", settings)
    destination = root / "harness/profile"
    destination.mkdir(exist_ok=True)
    for path in HERE.iterdir():
        if path.suffix in (".py", ".md"):
            shutil.copyfile(path, destination / path.name)
    selected = freezing.files(root)
    for path in [root, root / "install", root / "harness", *selected,
                 *(root / "install").rglob("*"), *(root / "harness").rglob("*")]:
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o022)
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(f"{digest(p)}  {p.relative_to(root)}\n" for p in selected))
    manifest.chmod(0o644)
    pin = digest(manifest)
    job = root / "x7-cycle-profile.job"
    job.write_text(f'''# timeout 1200
# Whole-machine DSpark profiling, TP4, 250 W on every GPU.
set -euo pipefail
source "$HOME/.config/chopinfer/local.env"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONDONTWRITEBYTECODE=1
unset CHOPINFER_KERNEL_GPU CHOPINFER_KERNEL_GPUS CHOPINFER_KERNEL_REQUIRE
exec timeout --signal=TERM --kill-after=15s 1170s python3 -B {shlex.quote(str(destination / "stage_profile.py"))} {shlex.quote(str(root))} --execute --manifest {pin}
''')
    job.chmod(0o644)
    subprocess.run(["bash", "-n", str(job)], check=True)
    freezing.verify(root, pin)
    receipt = dict(job=str(job), sha256=digest(job), manifest_sha256=pin, queued=False,
                   native_revision=settings["native_revision"], cases=4, sessions=2,
                   order=["K3 C8", "K3 C16", "fixed adaptive C8", "fixed adaptive C16"])
    write(root / "staged-job.json", receipt)
    print(json.dumps(receipt, indent=2))


def execute(root, pin):
    os.environ["X7_JOB_STARTED_MONOTONIC"] = str(time.monotonic())
    freezing.verify(root, pin)
    settings = json.loads((root / "config.json").read_text())
    result = subprocess.run([settings["nsys_path"], "--version"], capture_output=True,
                            text=True, check=True, timeout=10)
    assert result.stdout.strip() == settings["nsys_version"], "Nsight version changed"
    output = root / "runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}")
    output.parent.mkdir(exist_ok=True)
    check = subprocess.run([sys.executable, "-B", "-m", "tools.refrun.stamps",
        "--checkpoint", settings["checkpoint"], "--lock", "models/deepseek41/checkpoint-files.json"],
        cwd=settings["source"], capture_output=True, text=True, timeout=30)
    output.with_suffix(".checkpoint.log").write_text(check.stdout + check.stderr)
    assert check.returncode == 0
    print(f"Output: {output}", flush=True)
    os.execv(sys.executable, [sys.executable, "-B", str(HERE / "capture.py"), str(root), str(output)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.seal:
        seal(root)
    elif args.verify:
        freezing.verify(root, args.manifest)
    else:
        execute(root, args.manifest)


if __name__ == "__main__":
    main()

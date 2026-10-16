#!/usr/bin/env bash
# Build a self-contained comm qualification directory. Cargo and CUDA must
# already be configured. Usage: prepare_comm_latency.sh OUTPUT_DIRECTORY
set -Eeuo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
prepared=$(realpath -m -- "${1:?output directory outside the repository}")
case "$prepared/" in "$repo/"*) echo 'prepare outside the repository' >&2; exit 2 ;; esac
cd "$repo"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$repo/target}" CARGO_INCREMENTAL=0
unset CHOPINFER_KERNEL_GPU CHOPINFER_KERNEL_GPUS CHOPINFER_KERNEL_REQUIRE KERNELS_LOCK_UPDATE CHOPINFER_KERNEL_LINEINFO
mkdir -p "$prepared/bin" "$prepared/verify"
cargo test -p kernel-lab --release --locked --offline --no-run --test comm_latency --test comm > "$prepared/verify/prepare-binaries.log" 2>&1
cargo build -p kernel-lab --release --locked --offline --bin kernel-lab > "$prepared/verify/prepare-probe.log" 2>&1
python3 - "$prepared" "$CARGO_TARGET_DIR" <<'PY'
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path
root, target = map(Path, sys.argv[1:])
log = (root / 'verify/prepare-binaries.log').read_text()
for test in ('comm_latency', 'comm'):
    paths = re.findall(r'Executable tests/' + test + r'\.rs \(([^)]+)\)', log)
    if len(paths) != 1:
        raise SystemExit(f'{test}: expected one executable, found {paths}')
    shutil.copy2(paths[0], root / 'bin' / test)
shutil.copy2(target / 'release/kernel-lab', root / 'bin/kernel-lab')
shutil.copy2('crates/kernel-lab/tools/run_comm_latency.sh', root / 'run-gpu.sh')
files = ('comm', 'comm_latency', 'kernel-lab')
(root / 'bin/SHA256SUMS').write_text(''.join(
    hashlib.sha256((root / 'bin' / name).read_bytes()).hexdigest() + '  ' + name + '\n'
    for name in files))
(root / 'bin/source.txt').write_text(
    subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True)
    + subprocess.check_output(['git', 'status', '--short'], text=True))
PY
"$prepared/bin/comm_latency" --list > "$prepared/verify/prepared-tests.txt"
"$prepared/bin/comm" --list >> "$prepared/verify/prepared-tests.txt"
(cd "$prepared/bin" && sha256sum -c SHA256SUMS)

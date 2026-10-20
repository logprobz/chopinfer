#!/usr/bin/env bash
# Install beside the frozen binaries. The queue file contains absolute paths.
set -euo pipefail
l4_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
l4_mode=${1:-run}
case "$l4_mode" in run|--dry-run) ;; *) exit 2;; esac
export PYTHONDONTWRITEBYTECODE=1
unset CHOPINFER_KERNEL_GPU CHOPINFER_KERNEL_GPUS CHOPINFER_KERNEL_REQUIRE KERNELS_LOCK_UPDATE
l4_start=$(date +%s)
l4_out="$l4_root/runs/run-$l4_start-$$"
mkdir -p "$l4_out"
exec > >(tee "$l4_out/job.log") 2>&1
l4_pids=()
l4_gpu_started=0
power() {
 nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,power.limit --format=csv,noheader,nounits > "$l4_out/power-$1.csv"
 python3 - "$l4_out/power-$1.csv" <<'POWER'
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
assert len(rows) == 2 and all(float(r[1]) == 250 for r in rows), rows
POWER
}
finish() {
 l4_rc=$?
 trap - EXIT
 trap '' INT TERM HUP
 set +e
 for l4_pid in "${l4_pids[@]}"; do kill -TERM -- "-$l4_pid" 2>/dev/null; done
 for l4_pid in "${l4_pids[@]}"; do kill -KILL -- "-$l4_pid" 2>/dev/null; wait "$l4_pid" 2>/dev/null; done
 l4_journal=0
 if [ "$l4_gpu_started" -eq 1 ]; then
  power after || l4_rc=1
  journalctl -k --since "@$l4_start" --no-pager > "$l4_out/kernel.log" 2> "$l4_out/journal.err"
  l4_journal=$?
  rg 'NVRM:.*Xid' "$l4_out/kernel.log" > "$l4_out/xids.txt"
  if [ "$l4_journal" -ne 0 ] || [ -s "$l4_out/xids.txt" ]; then l4_rc=1; fi
 fi
 printf '{"exit_code":%d,"journal_exit":%d,"gpu_launched":%d}\n' "$l4_rc" "$l4_journal" "$l4_gpu_started" > "$l4_out/verdict.json"
 exit "$l4_rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
sha256sum --check --quiet "$l4_root/stage.sha256"
python3 -B "$l4_root/snapshot/indexer_prefill_tile_job.py" --stage "$l4_root/stage.json" --output "$l4_out/preflight.json"
PYTHONPATH="$l4_root/snapshot" python3 -B -m unittest discover -s "$l4_root/snapshot" -p test_indexer_prefill_tile_job.py -v
if [ "$l4_mode" = --dry-run ]; then
 echo "PREFILL_TILE_DRY_RUN_COMPLETE gpu_launched=0 output=$l4_out"
 exit 0
fi
: "${CUDA_VISIBLE_DEVICES:?runner must allocate two idle GPUs}"
export CHOPINFER_KERNEL_REQUIRE=1
export CHOPINFER_INDEX_PREFILL_ARTIFACTS="$l4_root/snapshot"
export CHOPINFER_INDEX_PREFILL_TILE_ARTIFACTS="$l4_root/snapshot"
l4_gpu_started=1
power before
for l4_shard in 0 1; do
 CHOPINFER_KERNEL_GPU="$l4_shard" CHOPINFER_L4_SHARD="$l4_shard" \
  setsid timeout --signal=TERM --kill-after=5s 820s /usr/bin/bash "$l4_root/snapshot/worker.sh" "$l4_root" "$l4_out" "$l4_shard" &
 l4_pids+=("$!")
done
l4_failed=0
for l4_pid in "${l4_pids[@]}"; do wait "$l4_pid" || l4_failed=1; done
l4_pids=()
[ "$l4_failed" -eq 0 ]
python3 -B "$l4_root/snapshot/indexer_prefill_tile_job.py" --run "$l4_out" --output "$l4_out/comparison.json"
echo "PREFILL_TILE_PAIR_COMPLETE cases=94 memcheck_cases=20 raw=376 pipelines=376 partitions=94 controls=188 timing_pairs=176 raw_timing_samples=1056"

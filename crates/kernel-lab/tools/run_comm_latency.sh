#!/usr/bin/env bash
# Copied beside bin/ by prepare_comm_latency.sh. No checkout is needed.
# Usage: run-gpu.sh RANKS BITWISE_TEST SKEW_TEST VERIFY_TEST, on assigned idle GPUs only.
set -Eeuo pipefail
ranks=${1:?rank count}
bitwise=${2:?bitwise test name}
skew=${3:?skew test name}
verify=${4:?verify rows test name}
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$root"
if [[ ! -x bin/comm_latency || ! -x bin/comm || ! -x bin/kernel-lab ]]; then
  echo 'run the prepared copy beside bin/' >&2
  exit 2
fi
case "$ranks" in
  2) devices=0,1 ;;
  4) devices=0,1,2,3 ;;
  *) exit 2 ;;
esac
# CUDA_VISIBLE_DEVICES comes from the caller.
unset CHOPINFER_KERNEL_GPU CHOPINFER_KERNEL_SHARED_DEVICE CHOPINFER_KERNEL_BENCH
export CHOPINFER_KERNEL_GPUS=$devices CHOPINFER_KERNEL_REQUIRE=1 CHOPINFER_KERNEL_TIMEOUT_SECS=300
stamp=$(timeout 5 date -u +%Y%m%dT%H%M%SZ)
out=$root/run-$stamp-tp$ranks
timeout 5 mkdir -p "$out/receipts" "$out/data"
export CHOPINFER_KERNEL_RECEIPTS=$out/receipts CHOPINFER_COMM_LAT_OUT=$out/data
export CHOPINFER_COMM_LAT_CONTROL=0 CHOPINFER_COMM_LAT_ROWS=1,2
step() {
  local label=$1 budget=$2 rc=0
  shift 2
  timeout --kill-after=15 "$budget" "$@" > "$out/$label.log" 2>&1 || rc=$?
  printf '%s exit %s\n' "$label" "$rc" | timeout 5 tee -a "$out/status.txt"
  return "$rc"
}
step artifacts 10 bash -c 'cd "$1/bin" && sha256sum -c SHA256SUMS' _ "$root"
step list 10 "$root/bin/comm_latency" --list
for test in "$bitwise" "$skew" "$verify" decode_ll_flag_reuse_g1 decode_v2_after_each_family_g1 decode_flat_tree_negative_g1 decode_latency_g4; do
  timeout 5 grep -Fxq -- "$test: test" "$out/list.log"
done
step probe 120 "$root/bin/kernel-lab" probe
step bitwise 180 "$root/bin/comm_latency" --exact "$bitwise" --nocapture --test-threads=1
step verify-rows 180 "$root/bin/comm_latency" --exact "$verify" --nocapture --test-threads=1
step skew 180 "$root/bin/comm_latency" --exact "$skew" --nocapture --test-threads=1
step flag-reuse 90 "$root/bin/comm_latency" --exact decode_ll_flag_reuse_g1 --nocapture --test-threads=1
step transitions 240 "$root/bin/comm_latency" --exact decode_v2_after_each_family_g1 --nocapture --test-threads=1
step wrong-tree 90 "$root/bin/comm_latency" --exact decode_flat_tree_negative_g1 --nocapture --test-threads=1
step invariance 90 env CHOPINFER_INVARIANCE_FILTER=allreduce_decode_norm_bf16_v2,allreduce_decode_combine_norm_bf16_v1,allreduce_combine_norm_bf16_v1 "$root/bin/comm" --exact invariance_comm_rows invariance_comm_batch --nocapture --test-threads=1
step latency 240 env CHOPINFER_KERNEL_BENCH=1 "$root/bin/comm_latency" --exact decode_latency_g4 --nocapture --test-threads=1
printf 'Comm qualification finished: %s\n' "$out"

#!/usr/bin/env bash
set -euo pipefail
l4_root=$1
l4_out=$2
l4_shard=$3
l4_test=indexer_mma::prefill::tile::early::gpu_early_selection_real_edges_and_timing
CHOPINFER_L4_MEMCHECK=1 CHOPINFER_KERNEL_RECEIPTS="$l4_out/memcheck-receipts-$l4_shard" \
 timeout --signal=TERM --kill-after=5s 180s /usr/local/cuda/bin/compute-sanitizer \
 --tool memcheck --error-exitcode 99 --target-processes all --log-file "$l4_out/sanitizer-$l4_shard.log" \
 "$l4_root/snapshot/prefill-tile-tests" "$l4_test" --exact --include-ignored --test-threads=1 --nocapture > "$l4_out/memcheck-$l4_shard.log" 2>&1
CHOPINFER_KERNEL_RECEIPTS="$l4_out/receipts-$l4_shard" \
 "$l4_root/snapshot/prefill-tile-tests" "$l4_test" --exact --include-ignored --test-threads=1 --nocapture > "$l4_out/gpu-$l4_shard.log" 2>&1

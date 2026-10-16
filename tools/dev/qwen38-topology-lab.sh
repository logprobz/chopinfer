#!/usr/bin/env bash
# Lean kernel-lab GPU validation of the Qwen3.8 TP2 topology on one
# GPU: every per-rank kernel entry the bundle launches at that rank count,
# against the f64 and exact oracles, with the smallest cases that launch each
# entry and oracles on deterministic output samples (kernel_lab::qwen38_lean).
# TP4 is not run here; its full tests are the default kernel-lab run.
#
# First it builds the release test binaries (not counted in the budget). Then
# it runs the test steps in sequence under one budget (default 900 s): each
# step gets `timeout --kill-after=30s` of the time left. It prints each step's
# elapsed time and a final PASS or FAIL line, and exits non-zero on a failed,
# skipped or timed-out step. Run it on idle GPUs.
#
#   tools/dev/qwen38-topology-lab.sh --ranks 2 [--gpu N] [--budget SECONDS]
#       [--logs-dir DIR] [--dry-run]
#
# The log goes to DIR (default $CHOPINFER_DATA/logs). --dry-run prints the
# commands without building or running. $CARGO overrides the cargo binary.
set -Eeuo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
for local_env in "$repo/local.env" "${CHOPINFER_LOCAL_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/chopinfer/local.env}"; do
    if [[ -f $local_env ]]; then source "$local_env"; break; fi
done
usage() { sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
ranks=
gpu=${CHOPINFER_KERNEL_GPU:-0}
budget=900
logs=
dry_run=0
while (($#)); do
    case "$1" in
        --ranks) ranks=${2-}; shift 2 ;;
        --gpu) gpu=${2-}; shift 2 ;;
        --budget) budget=${2-}; shift 2 ;;
        --logs-dir) logs=${2-}; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
done
case $ranks in
    2) ;;
    *) echo "--ranks must be 2" >&2; exit 2 ;;
esac
[[ $gpu =~ ^[0-9]+$ ]] || { echo "--gpu must be a GPU ordinal" >&2; exit 2; }
[[ $budget =~ ^[0-9]+$ ]] && ((budget > 0)) || { echo "--budget must be positive seconds" >&2; exit 2; }
cargo=${CARGO:-cargo}
cargo_test=("$cargo" test --release -p kernel-lab --locked --offline)
targets=(ops qwen38_gemm qwen38_rows qwen38_ple_items qwen38_speculation_rows
    qwen38_bundle_rows qwen38_moe_tail)
build=("${cargo_test[@]}" --no-run)
for target in "${targets[@]}"; do build+=(--test "$target"); done

# name, test target, exact test names.
steps=(
    "qsa|ops|sparse_attention::qsa_indexer_and_sparse_attention_match_oracles sparse_attention::qsa_catalog_covers_every_phase sparse_attention::qsa_lean_cases_launch_every_entry_and_split"
    "gated|ops|attention::qwen38::qwen38_gated_rows_match_oracle attention::qwen38::qwen38_output_gate_keeps_sigmoid_tails attention::qwen38::qwen38_gated_rows_cover_the_topology"
    "linear|ops|linear::linear_conv_matches_oracle_gpu linear::linear_gated_norm_matches_oracle_gpu linear::linear_delta_decode_verify_match_oracle_gpu linear::linear_delta_prefill_matches_oracle_gpu linear::linear_layer_chains_replay_deterministically_gpu linear::linear_prefill_state_carry_gpu"
    "moe|ops|moe::qwen38::qwen38_moe_decode_matches_oracles moe::qwen38::qwen38_moe_prefill_matches_oracles moe::qwen38::qwen38_expert_block_scales_promote_like_decode moe::qwen38::qwen38_shared_gate_keeps_sigmoid_tails"
    "moe-tail|qwen38_moe_tail|qwen38_moe_tail_arrangements"
    "gemm-own|qwen38_gemm|qwen38_gemm_g1"
    "gemm-fragment|ops|gemm::gemm_qwen38_shapes gemm::gemm_qwen38_replay"
    "ngram|qwen38_rows|ngram_g1"
    "ngram-items|qwen38_ple_items|qwen38_ple_items_match_per_item_launches"
    "speculation|qwen38_speculation_rows|qwen38_speculation_rows_match_references"
    "bundle|qwen38_bundle_rows|qwen38_prefix_copy_round_trips_a_slot_through_scattered_pages qwen38_bundle_rows_match_cpu_references"
)
step_command() {
    local target=$1 names=$2
    local -a words
    read -r -a words <<<"$names"
    command=("${cargo_test[@]}" --test "$target" -- --exact --test-threads=1 --nocapture "${words[@]}")
}
export CHOPINFER_QWEN38_RANKS=$ranks CHOPINFER_KERNEL_GPU=$gpu
unset CHOPINFER_KERNEL_GPUS CHOPINFER_KERNEL_BENCH

if ((dry_run)); then
    echo "export CHOPINFER_QWEN38_RANKS=$ranks CHOPINFER_KERNEL_GPU=$gpu"
    echo "# build, not counted in the ${budget} s budget"
    echo "${build[*]}"
    for step in "${steps[@]}"; do
        IFS='|' read -r name target names <<<"$step"
        step_command "$target" "$names"
        echo "# $name"
        echo "${command[*]}"
    done
    exit 0
fi

logs=${logs:-${CHOPINFER_DATA:?set CHOPINFER_DATA or pass --logs-dir}/logs}
mkdir -p "$logs"
log=$logs/qwen38-topology-lab-tp$ranks-$(date -u +%Y%m%dT%H%M%SZ).log
exec > >(tee -a "$log") 2>&1
echo "qwen38 topology lab: TP$ranks on GPU $gpu, budget ${budget} s, log $log, $(date -u +%FT%TZ)"
cd "$repo"

echo "== build (not counted)"
build_start=$SECONDS
if ! "${build[@]}"; then
    echo "FAIL: TP$ranks release test build failed"
    exit 1
fi
echo "   build took $((SECONDS - build_start)) s"

start=$SECONDS
failed=
for step in "${steps[@]}"; do
    IFS='|' read -r name target names <<<"$step"
    left=$((budget - (SECONDS - start)))
    if ((left <= 0)); then
        failed="$name (budget spent before it started)"
        break
    fi
    step_command "$target" "$names"
    echo "== $name: ${command[*]}"
    step_start=$SECONDS
    output=$(mktemp)
    status=0
    timeout --kill-after=30s "$left" "${command[@]}" 2>&1 | tee "$output" || status=${PIPESTATUS[0]}
    elapsed=$((SECONDS - step_start))
    if ((status == 124 || status == 137)); then
        failed="$name (timed out after $elapsed s)"
    elif ((status != 0)); then
        failed="$name (exit $status)"
    elif grep -q "skipping GPU test" "$output"; then
        failed="$name (GPU tests skipped)"
    elif ! grep -Eq "test result: ok\. [1-9][0-9]* passed" "$output"; then
        failed="$name (no test ran)"
    fi
    rm -f "$output"
    echo "   $name: ${elapsed} s, $((SECONDS - start)) s of ${budget} s used"
    [[ -z $failed ]] || break
done

total=$((SECONDS - start))
if [[ -n $failed ]]; then
    echo "FAIL: TP$ranks kernel lab in ${total} s: $failed; log $log"
    exit 1
fi
echo "PASS: TP$ranks kernel lab in ${total} s of ${budget} s; log $log"

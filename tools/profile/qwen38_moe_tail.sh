#!/usr/bin/env bash
# Kernel-lab runner for the Qwen3.8 MoE tail prototype. Run only on an idle GPU:
#
#   tools/profile/qwen38_moe_tail.sh check <gpu>   correctness, any allowed time
#   tools/profile/qwen38_moe_tail.sh bench <gpu>   timings, benchmark window
#
# `bench` also runs the correctness check first and refuses a GPU that has
# another compute process. Receipts go to qwen38_moe_tail/ under the kernel-lab
# receipts directory (see crates/kernel-lab/README.md).
set -Eeuo pipefail

if [[ $# -ne 2 || ! $1 =~ ^(check|bench)$ || ! $2 =~ ^[0-9]+$ ]]; then
    printf 'usage: %s <check|bench> <gpu>\n' "$0" >&2
    exit 2
fi
mode=$1
gpu=$2
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
for local_env in "$root/local.env" "${CHOPINFER_LOCAL_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/chopinfer/local.env}"; do
    if [[ -f $local_env ]]; then source "$local_env"; break; fi
done
: "${CARGO_HOME:=$HOME/.cargo}"
: "${TMPDIR:=${CHOPINFER_SCRATCH:-/tmp}}"
export CARGO_HOME TMPDIR CARGO_NET_OFFLINE=true
cd "$root"

# The host-side weight banks are large; optimize the CPU code that packs them.
run() {
    CHOPINFER_KERNEL_GPU=$gpu cargo test \
        --config 'profile.dev.package.kernel-core.opt-level=3' \
        --config 'profile.dev.package.kernel-lab.opt-level=3' \
        -p kernel-lab --test qwen38_moe_tail --locked --offline -- --nocapture
}

exclusive() {
    local uuid others
    uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu")
    others=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader |
        awk -F', ' -v u="$uuid" '$1 == u { print $2 }')
    if [[ -n $others ]]; then
        printf 'GPU %s has other compute processes: %s\n' "$gpu" "$others" >&2
        return 1
    fi
    printf 'GPU %s (%s) has no other compute process at %s\n' "$gpu" "$uuid" "$(date -u +%FT%TZ)"
}

run
if [[ $mode == bench ]]; then
    for routing in uniform clustered; do
        exclusive
        CHOPINFER_KERNEL_BENCH=1 CHOPINFER_QWEN_TAIL_ROUTING=$routing run
    done
    exclusive
fi

#!/usr/bin/env bash
set -Eeuo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "${root}"
for local_env in "$root/local.env" "${CHOPINFER_LOCAL_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/chopinfer/local.env}"; do
    if [[ -f $local_env ]]; then source "$local_env"; break; fi
done
# Tokenizer goldens need only the checkpoint's tokenizer files. Tests that read
# checkpoint weights are slow and stay opt-in: CHOPINFER_CI_CHECKPOINT_TESTS=1.
# Each model's variable stem is its models/<m>/model.toml checkpoint_env.
checkpoint_envs=$(python3 -B tests/model_inventory.py checkpoint-env)
for model in ${checkpoint_envs}; do
    checkpoint_var=CHOPINFER_${model}_CHECKPOINT
    tokenizer_var=CHOPINFER_${model}_TOKENIZER_DIR
    if [[ -n ${!checkpoint_var:-} ]]; then
        export "$tokenizer_var=${!tokenizer_var:-${!checkpoint_var}}"
    fi
    # Other checkpoints of the model (any layout), colon-separated.
    extra_var=CHOPINFER_${model}_EXTRA_CHECKPOINTS
    if [[ ${CHOPINFER_CI_CHECKPOINT_TESTS:-0} != 1 ]]; then unset "$checkpoint_var" "$extra_var"; fi
done
: "${CARGO_HOME:=$HOME/.cargo}"
: "${CARGO_NET_OFFLINE:=true}"
: "${TMPDIR:=${CHOPINFER_SCRATCH:-/tmp}}"
export CARGO_HOME CARGO_NET_OFFLINE TMPDIR
# Bound independent test cases on shared development machines.
export RUST_TEST_THREADS=${RUST_TEST_THREADS:-8}
# The CI Rust dev and test profile (ci/rust-profile.env).
while IFS='=' read -r name value; do
    if [[ $name =~ ^[A-Z_]+$ ]]; then export "$name=${!name:-$value}"; fi
done < "$root/ci/rust-profile.env"

export CHOPINFER_ROUTE_SWEEP=cpu
# ci/cpu.py runs every check as supervised steps: see docs/contributing.md#checks.
exec python3 -B ci/cpu.py "$@"

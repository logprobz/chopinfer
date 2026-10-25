#!/usr/bin/env bash
# Iteration check only. ci/cpu.sh runs every check.
set -Eeuo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$root"
for local_env in "$root/local.env" "${CHOPINFER_LOCAL_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/chopinfer/local.env}"; do
    if [[ -f $local_env ]]; then source "$local_env"; break; fi
done
if ! command -v python3 >/dev/null 2>&1; then
    printf '%s\n' 'quick: Python 3.11 or newer is required; install it and select python3 in PATH' >&2
    exit 2
fi
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
    printf '%s\n' 'quick: select Python 3.11 or newer in PATH' >&2
    exit 2
fi
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
export CHOPINFER_ROUTE_SWEEP=quick
# The CI Rust dev and test profile (ci/rust-profile.env).
while IFS='=' read -r name value; do
    if [[ $name =~ ^[A-Z_]+$ ]]; then export "$name=${!name:-$value}"; fi
done < "$root/ci/rust-profile.env"
exec nice -n 19 python3 -B ci/quick.py "$@"

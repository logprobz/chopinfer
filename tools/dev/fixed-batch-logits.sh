#!/usr/bin/env bash
set -Eeuo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo"
for local_env in "$repo/local.env" "${CHOPINFER_LOCAL_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/chopinfer/local.env}"; do
    if [[ -f $local_env ]]; then source "$local_env"; break; fi
done
export CARGO_NET_OFFLINE=true CARGO_BUILD_JOBS=8 CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-${CHOPINFER_DATA:?set CHOPINFER_DATA in local.env}/c16-host/target}
# prepare-compare builds only the recorder for two existing direct packages.
# run-compare needs idle GPUs, like run.
exec nice -n 19 python3 -B tools/dev/fixed-batch-logits.py "$@"

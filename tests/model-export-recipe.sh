#!/usr/bin/env bash
set -Eeuo pipefail
# The recipe's Git commands read this checkout only: not a repository or
# index inherited through GIT_* (`git rebase -x` exports GIT_DIR), nor the
# caller's global and system configuration.
for name in $(compgen -e -X '!GIT_*'); do unset "$name"; done
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ ${1:-} != --ci ]]; then
for local_env in "$root/local.env" "${CHOPINFER_LOCAL_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/chopinfer/local.env}"; do
    if [[ -f $local_env ]]; then source "$local_env"; break; fi
done
fi
export CARGO_INCREMENTAL=0
exec python3 -B "$root/tests/model_export_recipe.py" "$@"

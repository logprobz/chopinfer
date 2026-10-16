#!/usr/bin/env bash
set -Eeuo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
for env_file in "$root/local.env" "${CHOPINFER_LOCAL_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/chopinfer/local.env}"; do
    if [[ -f $env_file ]]; then source "$env_file"; break; fi
done
cd "$root"
exec python3 -B "$root/tools/qual/release_preview.py" "$@"

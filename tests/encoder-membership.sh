#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
shopt -s nullglob
gates=("$root"/models/*/contract/tests/encoder-membership.sh)
((${#gates[@]} > 0)) || { echo 'no encoder membership gates' >&2; exit 1; }
for gate in "${gates[@]}"; do bash "$gate"; done

#!/usr/bin/env bash
set -Eeuo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "${root}"
cargo=${CARGO:-cargo}
"${cargo}" test --locked --offline -p runtime-server -p runtime-engine -p runtime-kv --test metrics_contract

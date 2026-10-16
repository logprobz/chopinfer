#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd -P)
cd "$root"
CHOPINFER_MIMO26_GEOMETRY_DIRS="$root/fixtures/mimo26/geometries/shared-audio" \
  cargo test --locked --offline -p model-mimo26-contract --test encoder_membership

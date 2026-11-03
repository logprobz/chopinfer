#!/usr/bin/env bash
# Produce or stage a recorded release. Qualification remains a separate gate.
set -Eeuo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$root"
stage_from=
if [[ ${1:-} == --stage-from ]]; then
    [[ $# == 3 ]] || { echo 'usage: tools/release/build.sh [--stage-from PREBUILT_DIRECTORY] OUTPUT_DIRECTORY' >&2; exit 2; }
    stage_from=$2
    shift 2
fi
[[ $# == 1 ]] || { echo 'usage: tools/release/build.sh [--stage-from PREBUILT_DIRECTORY] OUTPUT_DIRECTORY' >&2; exit 2; }
if [[ -z $stage_from ]]; then
    : "${CARGO_TARGET_DIR:?set CARGO_TARGET_DIR to an external build scratch directory}"
    result=$(mktemp "${TMPDIR:-/tmp}/chopinfer-release-result.XXXXXX")
    trap 'rm -f "$result"' EXIT
    python3 -B tools/release/provenance.py --scratch "$CARGO_TARGET_DIR" > "$result"
    stage_from=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["prebuilt"])' "$result")
    printf 'Prebuilt: %s\n' "$stage_from"
fi
python3 -B tools/release/staging.py "$stage_from" "$1"

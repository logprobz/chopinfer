#!/usr/bin/env bash
# Use the same model declarations and mechanics fixture locally and in CI.
# ci/qualification.py runs one process per module and gate config; a module
# that declares QUALIFICATION_RUNS_ONCE pins its own config and runs once.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${PYTHON:-python3}" -B ci/qualification.py "$@"

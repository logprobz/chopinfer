#!/usr/bin/env bash
# Entry point for tools/dev/heavy.py, the host-wide queue for heavy CPU
# commands; see that file and docs/contributing.md#shared-build-hosts. It is a shell script
# because tools/journal_proof/capture.py runs `bash $CHOPINFER_HEAVY_WRAPPER
# COMMAND...`. exec keeps this process's PID and process group.
exec python3 -B "$(dirname -- "${BASH_SOURCE[0]}")/heavy.py" "$@"

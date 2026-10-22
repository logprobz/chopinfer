#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

binary=${CHOPINFER_BINARY:-/opt/chopinfer/bin/chopinfer}
# The release build checks BUILD_INFO against the binary. Reading it here
# keeps image identification available without the host's CUDA driver mount.
info=$(dirname -- "$binary")/../BUILD_INFO
if [[ ${1:-} == --version && $# == 1 && -f $info ]]; then
    exec python3 -c '
import sys
info = dict(line.rstrip().split("=", 1) for line in open(sys.argv[1]))
print("chopinfer " + info["version"] + " (" + info["revision"] + ")")
' "$info"
fi
case ${1:-} in
    --version|--help|verify) exec "$binary" "$@" ;;
esac

# Checkpoint verification can take minutes on a cold volume. Forward startup
# signals to that child too, before serve takes over this PID.
child=
stop_startup() {
    trap 'exit 0' TERM INT
    if [[ -n $child ]]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" || true
    fi
    exit 0
}
trap stop_startup TERM INT
checked() {
    "$@" &
    child=$!
    local status=0
    wait "$child" || status=$?
    child=
    return "$status"
}

# Every installed helper reports its model's deployment policy, so the
# environment is checked against each installed model's variables before any
# checkpoint is prepared.
deployment=$(dirname -- "${BASH_SOURCE[0]}")/deployment.py
python3 "$deployment" environment "$binary" "$@"

model=${CHOPINFER_MODEL:-/data/model}
cache=${CHOPINFER_MODEL_CACHE:-/data/model-cache}
ranks=${CHOPINFER_TENSOR_PARALLEL_SIZE:-4}
[[ -d $model ]] || { echo "CHOPINFER_MODEL: $model is not a checkpoint directory" >&2; exit 1; }
mkdir -p -- "$cache"

# Optional offline check of every checkpoint file's SHA-256.
if [[ ${CHOPINFER_VERIFY_MODEL:-0} == 1 ]]; then
    verify=("$binary" verify --model "$model")
    [[ ! -v CHOPINFER_VERIFY_MANIFEST ]] || verify+=(--manifest "$CHOPINFER_VERIFY_MANIFEST")
    checked "${verify[@]}"
fi

given() {
    local argument
    for argument in "${arguments[@]}"; do [[ $argument == "$1" ]] && return 0; done
    return 1
}
arguments=("$@")

# The served modalities select the package, so the helper prepares the one
# serve selects: --modalities as an argument, else CHOPINFER_MODALITIES.
modalities=${CHOPINFER_MODALITIES:-auto}
for ((index=0; index<${#arguments[@]}; index++)); do
    if [[ ${arguments[index]} == --modalities ]]; then
        modalities=${arguments[index+1]:-}
    fi
done

# The model family is whichever installed helper prepares the checkpoint. The
# helper packages the native module installed with it, so the package always
# matches this image. serve --model repeats the same preparation, reuses the
# cached package and refuses a checkpoint that more than one helper accepts.
scratch=$(mktemp -d)
trap 'rm -rf -- "$scratch"' EXIT
helpers=("$(dirname -- "$binary")"/chopinfer-model-*)
prepared=
for helper in "${helpers[@]}"; do
    [[ -f $helper && -x $helper ]] || continue
    if checked "$helper" prepare --direct --model "$model" --tensor-parallel-size "$ranks" \
        --cache-dir "$cache" --modalities "$modalities" >"$scratch/prepared" \
        2>"$scratch/stderr"; then
        cat -- "$scratch/stderr" >&2
        prepared=$helper
        break
    fi
    { printf '%s: ' "${helper##*/}"; cat -- "$scratch/stderr"; } >>"$scratch/refusals"
done
if [[ -z $prepared ]]; then
    echo "no installed model helper prepares the checkpoint $model for $ranks ranks" >&2
    [[ ! -f $scratch/refusals ]] || cat -- "$scratch/refusals" >&2
    exit 1
fi

# Read only data from the helper's reply and the prepared package, whose
# deployment policy governs the bundle it serves. Never source package
# contents. The drafter is the one the package declares.
python3 "$deployment" prepared "$binary" "$scratch/prepared" "$@"
rm -rf -- "$scratch"
trap - EXIT

# Native serve selects the checkpoint's profile and owns every serving default.
# Container environment variables are explicit overrides; CLI flags win.
overrides=()
for pair in \
    MAX_SEQUENCES:max-sequences MAX_BATCH_TOKENS:max-batch-tokens \
    PREFILL_CHUNK_TOKENS:prefill-chunk-tokens QUEUE_DEPTH:queue-depth \
    GPU_MEMORY_UTILIZATION:gpu-memory-utilization KV_CACHE_DTYPE:kv-cache-dtype \
    MODALITIES:modalities \
    MAX_CONTEXT_TOKENS:max-context-tokens DRAIN_TIMEOUT_MS:drain-timeout-ms \
    SPECULATIVE_TOKENS:speculative-tokens SPECULATION:speculation DRAFT_COST:draft-cost; do
    variable=CHOPINFER_${pair%%:*}
    flag=--${pair#*:}
    if [[ -v $variable ]] && ! given "$flag"; then
        overrides+=("$flag" "${!variable}")
    fi
done
# A container must listen beyond its own loopback for published ports to work.
given --listen || overrides+=(--listen "${CHOPINFER_LISTEN:-0.0.0.0:8000}")
exec "$binary" serve --model "$model" --tensor-parallel-size "$ranks" \
    --model-cache-dir "$cache" --architecture sm_120f "${overrides[@]}" "$@"

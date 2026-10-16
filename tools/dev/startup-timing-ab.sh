#!/usr/bin/env bash
# Startup timing of direct serving (`chopinfer serve --model`) for one host
# binary, checkpoint and model cache, recorded under a scenario label with
# the measurement conditions.
#
#   tools/dev/startup-timing-ab.sh local --host BIN --model DIR --cache-dir DIR \
#     --scenario LABEL --conditions TEXT [--ranks N] [--load-threads N]
#     [--port N] [--timeout-seconds N] [--dry-run]
#
# BIN is the chopinfer of a private install: its model helper beside it and
# the bundle module beside it or in ../lib, nothing group or world writable
# (tools/dev/redeploy-common.sh install_serving_binaries makes one). For an
# A/B, run it once per binary or cache with the same conditions. It needs
# idle GPUs, starts the server, waits for health, stops it and prints the
# startup lines.
set -Eeuo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
for local_env in "$repo/local.env" "${CHOPINFER_LOCAL_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/chopinfer/local.env}"; do
    if [[ -f $local_env ]]; then source "$local_env"; break; fi
done
data=${CHOPINFER_DATA:-$HOME/chopinfer-data}
state=$data/startup-timing-ab

usage() { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

(($#)) || { usage >&2; exit 2; }
command=$1
shift
case $command in
    prepare|run)
        echo "$command is gone with the prepared packages; time direct serving with: $0 local --host BIN --model DIR ..." >&2
        exit 2 ;;
esac
port=8391
dry_run=0
local_host=
local_model=
local_cache=
scenario=
conditions=
ranks=4
load_threads=
ready_timeout=3600
while (($#)); do
    case "$1" in
        --host) local_host=${2:?missing host binary}; shift 2 ;;
        --model) local_model=${2:?missing checkpoint}; shift 2 ;;
        --cache-dir) local_cache=${2:?missing cache}; shift 2 ;;
        --scenario) scenario=${2:?missing scenario label}; shift 2 ;;
        --conditions) conditions=${2:?record hardware, filesystem and page-cache conditions}; shift 2 ;;
        --ranks) ranks=${2:?missing rank count}; shift 2 ;;
        --load-threads) load_threads=${2:?missing thread count}; shift 2 ;;
        --timeout-seconds) ready_timeout=${2:?missing timeout}; shift 2 ;;
        --port) port=${2:?missing port}; shift 2 ;;
        --before-ref|--no-preread) printf '%s is gone with the prepare and run modes; time each binary with local --host BIN\n' "$1" >&2; exit 2 ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ $port =~ ^[0-9]+$ ]] && ((port >= 1 && port <= 65535)) || { echo 'port must be 1..65535' >&2; exit 2; }

now_ms() { date +%s%3N; }

gpu_apps() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '; }

start_and_time() {
    local model=$1 label=$2 binary=$3 log pid started deadline healthy_ms
    log=$state/logs/$model-$label-$run_id.log
    started=$(now_ms)
    nohup "$binary" "${arguments[@]}" >"$log" 2>&1 </dev/null &
    pid=$!
    deadline=$(( $(date +%s) + ready_timeout ))
    until grep -Fq '"status":"listening"' "$log" && \
        curl -fsS --connect-timeout 1 --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "$model $label: server exited before health; see $log" >&2
            tail -20 "$log" >&2
            return 1
        fi
        if (( $(date +%s) >= deadline )); then
            echo "$model $label: not healthy within ${ready_timeout}s; see $log" >&2
            kill -TERM "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
            return 1
        fi
        sleep 0.5
    done
    healthy_ms=$(( $(now_ms) - started ))
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 120); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    for _ in $(seq 120); do grep -qx "$pid" <<<"$(gpu_apps)" || break; sleep 0.5; done
    {
        printf '%s %s: serve to healthy %d.%03ds (log %s)\n' "$model" "$label" \
            "$(( healthy_ms / 1000 ))" "$(( healthy_ms % 1000 ))" "$log"
        grep -E '^startup ' "$log" \
            | sed 's/^/    /' || true
    } | tee -a "$results"
}

local_run() {
    [[ -x $local_host && -d $local_model && -n $local_cache && -n $conditions ]] || {
        echo 'local needs --host, --model, --cache-dir and --conditions' >&2; exit 2;
    }
    [[ $scenario =~ ^[a-zA-Z0-9_-]{1,64}$ && $ranks =~ ^(2|4)$ && $ready_timeout =~ ^[1-9][0-9]*$ ]] || {
        echo 'use a bounded scenario label, TP2/TP4 and a positive timeout' >&2; exit 2;
    }
    local_host=$(realpath -- "$local_host")
    arguments=(serve --model "$local_model" --model-cache-dir "$local_cache"
        --tensor-parallel-size "$ranks" --listen "127.0.0.1:$port")
    [[ -z $load_threads ]] || arguments+=(--load-threads "$load_threads")
    if ((dry_run)); then
        printf 'scenario=%s conditions=%s\n' "$scenario" "$conditions"
        printf '%q ' "$local_host" "${arguments[@]}"; printf '\n'
        return
    fi
    [[ -z $(gpu_apps) ]] || { echo 'local timing requires idle GPUs; another GPU process is running' >&2; exit 1; }
    run_id=$(date -u +%Y%m%dT%H%M%SZ)
    mkdir -p -- "$state/logs"
    results=$state/local-$scenario-$run_id.txt
    {
        printf 'scenario=%s conditions=%s checkpoint=%s cache=%s load_threads=%s\n' "$scenario" "$conditions" "$local_model" "$local_cache" "${load_threads:-default}"
        sha256sum -- "$local_host"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    } >"$results"
    # This mode neither builds nor clears the cache or filesystem page cache.
    # `serve --model` loads the checkpoint directly: select a fresh cache and
    # an evicted checkpoint for cold, repeat A for warm, then select B and A.
    start_and_time local "$scenario" "$local_host"
    printf 'results: %s\n' "$results"
}

case $command in
    -h|--help) usage ;;
    local) local_run ;;
    *) usage >&2; exit 2 ;;
esac

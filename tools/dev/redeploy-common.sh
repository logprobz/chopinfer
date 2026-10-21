# Shared dev redeploy operations: privately install the declared artifacts,
# prepare the package, stop this port's own server, and wait for health.
# Sourced; it only defines functions. The caller supplies paths and options.

now_ms() { date +%s%3N; }
duration() { printf '%d.%03d' "$(( $1 / 1000 ))" "$(( $1 % 1000 ))"; }
show_command() { printf '  '; printf '%q ' "$@"; printf '\n'; }

# The host's helper discovery protocol uses the chopinfer-model- prefix.
# Libraries and additional binaries come from the selected inventory targets.
redeploy_install_plan() {
    local view=$1 model=$2 helper
    helper=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../release" && pwd -P)/plan.py
    python3 -B "$helper" --view "$view" --model "$model"
}

# Fresh files renamed into place let a running server keep its binaries.
# Usage: install_serving_binaries RELEASE_DIR DEST MODEL
install_serving_binaries() {
    local release=$1 dest=$2 model=$3 source destination path plan
    plan=$(redeploy_install_plan serving "$model") || return
    mkdir -p -- "$dest/bin" "$dest/lib"
    chmod 0755 -- "$dest" "$dest/bin" "$dest/lib"
    while IFS=$'\t' read -r source destination; do
        path=$dest/$destination
        [[ -f $release/$source ]] || { echo "missing build output $release/$source" >&2; return 1; }
        install -m 0755 -- "$release/$source" "$path.tmp.$$"
        mv -f -- "$path.tmp.$$" "$path"
    done <<< "$plan"
    if find "$dest" \( -perm -g+w -o -perm -o+w \) -print -quit | grep -q .; then
        echo "private install $dest has a group or world writable path" >&2
        return 1
    fi
}

install_redeploy_extras() {
    local release=$1 dest=$2 model=$3 source destination plan
    plan=$(redeploy_install_plan extras "$model") || return
    while IFS=$'\t' read -r source destination; do
        [[ -n $source ]] || continue
        install -m 0755 -- "$release/$source" "$dest/$destination"
    done <<< "$plan"
}

# The direct package `serve --model` serves, emitted ahead of the server stop
# so the timed health wait does not include it. Sets package_dir from the
# helper's JSON line. The helper arguments are in helper_args.
prepare_direct() {
    local started result
    started=$(now_ms)
    result=$("${helper_args[@]}")
    package_dir=$(python3 -c '
import json, sys
lines = [line for line in sys.stdin.read().splitlines() if line.strip()]
record = json.loads(lines[-1])
if record.get("result") != "prepared-direct":
    raise SystemExit("helper did not prepare a direct package: " + lines[-1])
print(record["bundle"])
' <<<"$result")
    prepare_ms=$(( $(now_ms) - started ))
}

# An explicit package for a choice the helper does not make, served with
# `serve --bundle P --checkpoint C`. The packager arguments are in
# package_args and name --output "$package_dir".
prepare_explicit() {
    local started
    started=$(now_ms)
    "${package_args[@]}"
    prepare_ms=$(( $(now_ms) - started ))
}

# Whether the checkpoint's safetensors index names vision tower tensors, as
# the family helpers check before planning the encoders. Returns 2 when the
# index is unreadable.
checkpoint_has_vision() {
    python3 - "$1" <<'PY'
import json
import pathlib
import sys
try:
    names = json.loads((pathlib.Path(sys.argv[1]) / 'model.safetensors.index.json').read_text())['weight_map']
except (OSError, ValueError, KeyError):
    raise SystemExit(2)
raise SystemExit(0 if any(name.startswith(('visual.', 'model.visual.')) for name in names) else 1)
PY
}

# Prints the removal message of a redeploy flag that only made sense for rank
# images or installed bundles, or nothing for any other word.
removed_flag_message() {
    case $1 in
        --rank-dir|--drafter-rank-dir|--encoder-rank-dir)
            echo "$1 is gone: serving loads the checkpoint directly; pass --checkpoint DIR" ;;
        --install-root)
            echo '--install-root is gone: pass --model-cache-dir DIR for the direct package and --work-dir DIR for the private binaries' ;;
        --encoders)
            echo '--encoders is gone: serve --model plans the vision encoders whenever the checkpoint has them' ;;
        --ngram-placement)
            echo '--ngram-placement is gone: the weight plan keeps the n-gram table in host memory at TP2 and on the GPU otherwise' ;;
        --package-max-sequences)
            echo '--package-max-sequences is gone: the direct package needs no declared limit; pass --max-sequences N' ;;
        --fresh-package)
            echo '--fresh-package is gone: the model cache keys the direct package by checkpoint and options; pass --model-cache-dir DIR for a fresh cache' ;;
        --check-reuse)
            echo '--check-reuse is gone: --prepare-only prepares or reuses the direct package without a GPU' ;;
        --architecture)
            echo '--architecture is gone: it selected the installation, which serve --model does not use' ;;
        --topology)
            echo '--topology is gone: --ranks N selects the topology and its profile' ;;
    esac
}

# A pidfd pins the exact kernel process while the token and start time prove
# this pidfile belongs to a server started by this script.
stop_owned_server() {
    [[ -e $pidfile ]] || return 0
    python3 -B - "$pidfile" "$repo/tools/dev" "${release_timeout:-120}" "${serve_args[@]}" <<'PY'
import importlib.util
import os
import pathlib
import signal
import sys
import time

spec = importlib.util.spec_from_file_location("redeploy_wait", pathlib.Path(sys.argv[2]) / "redeploy-wait.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
deadline = time.monotonic() + int(sys.argv[3])

def released(pid, pidfd):
    print(f"Waiting for PID {pid} exit and GPU release (timeout {sys.argv[3]}s)", file=sys.stderr)
    try:
        helper.wait_released(pid, pidfd, sys.argv[4:], deadline)
    except TimeoutError as error:
        raise SystemExit(str(error)) from None

path = pathlib.Path(sys.argv[1])
fields = path.read_text().split()
if len(fields) != 3:
    raise SystemExit(f'invalid pidfile {path}; refusing to signal any process')
pid, started, token = int(fields[0]), fields[1], fields[2]
try:
    pidfd = os.pidfd_open(pid)
except ProcessLookupError:
    released(pid, None)
    path.unlink()
    raise SystemExit(0)

def state():
    return helper.process_state(pid)

try:
    current = state()
    if current is None or current[0] == 'Z':
        released(pid, pidfd)
        path.unlink()
        raise SystemExit(0)
    environment = helper.process_environment(pid)
    if environment is None:
        released(pid, pidfd)
        path.unlink()
        raise SystemExit(0)
    if current[1] != started or f'CHOPINFER_DEV_REDEPLOY_TOKEN={token}'.encode() not in environment:
        raise SystemExit(f'pidfile {path} does not identify its original server; refusing to signal pid {pid}')
    try:
        signal.pidfd_send_signal(pidfd, signal.SIGTERM)
    except ProcessLookupError:
        pass
    grace = min(time.monotonic() + 5, time.monotonic() + int(sys.argv[3]) / 2)
    while time.monotonic() < grace:
        current = state()
        if current is None or current[0] == 'Z':
            break
        time.sleep(min(0.1, max(0, grace - time.monotonic())))
    else:
        try:
            signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        except ProcessLookupError:
            pass
    released(pid, pidfd)
    path.unlink()
finally:
    os.close(pidfd)
PY
}

# Starts the server, waits for health, prints the timings and runs the
# optional smoke completion. A failure stops the server it started. The
# body keeps column 0 so its here-documents end where they did.
serve_until_healthy() {
stop_owned_server
if python3 - "$port" <<'PY'
import socket
import sys
try:
    socket.create_connection(('127.0.0.1', int(sys.argv[1])), timeout=0.5).close()
except OSError:
    raise SystemExit(1)
PY
then
    echo "another process is listening on port $port; refusing to claim it" >&2
    exit 1
fi

startup_started=$(now_ms)
token=$(python3 -c 'import secrets; print(secrets.token_hex(16))')
CHOPINFER_DEV_REDEPLOY_TOKEN=$token nohup "$host_bin" "${serve_args[@]}" \
    >"$server_log" 2>&1 </dev/null {lock_fd}>&- &
server_pid=$!
server_started=1
cleanup_failure() {
    code=$?
    trap - EXIT
    if ((code != 0 && server_started)); then stop_owned_server || true; fi
    exit "$code"
}
trap cleanup_failure EXIT
starttime=$(python3 - "$server_pid" <<'PY'
import pathlib
import sys
print(pathlib.Path(f'/proc/{sys.argv[1]}/stat').read_text().rsplit(')', 1)[1].split()[19])
PY
)
printf '%s %s %s\n' "$server_pid" "$starttime" "$token" >"$pidfile.tmp.$$"
mv -f -- "$pidfile.tmp.$$" "$pidfile"

deadline=$(( $(date +%s) + startup_timeout ))
while ! { grep -Fq '"status":"listening"' "$server_log" && \
    curl -fsS --connect-timeout 1 --max-time 2 "$url/health" >/dev/null 2>&1; }; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "server exited before health; see $server_log" >&2
        tail -40 "$server_log" >&2
        exit 1
    fi
    if (( $(date +%s) >= deadline )); then
        echo "server did not become healthy within ${startup_timeout}s; see $server_log" >&2
        tail -40 "$server_log" >&2
        exit 1
    fi
    sleep 0.5
done
models=$(curl -fsS --max-time 5 "$url/v1/models")
model=$(printf '%s' "$models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
healthy_ms=$(( $(now_ms) - startup_started ))
total_ms=$(( $(now_ms) - total_started ))

printf '\nRedeploy timings (seconds):\n'
printf '  build: %s\n  install: %s\n  prepare: %s\n  serve to healthy: %s\n  total to healthy: %s\n' \
    "$(duration "$build_ms")" "$(duration "$install_ms")" "$(duration "$prepare_ms")" \
    "$(duration "$healthy_ms")" "$(duration "$total_ms")"
printf 'Healthy: %s (model %s)\nPackage: %s\nLog: %s\nPID file: %s\n' \
    "$url" "$model" "$package_dir" "$server_log" "$pidfile"

if ((smoke)); then
    payload=$(python3 - "$model" <<'PY'
import json
import sys
print(json.dumps({'model': sys.argv[1], 'prompt': 'Hello.', 'max_tokens': 4, 'temperature': 0}))
PY
)
    auth=()
    [[ -z ${OPENAI_API_KEY:-} ]] || auth=(-H "Authorization: Bearer $OPENAI_API_KEY")
    curl -fsS --max-time 120 -H 'Content-Type: application/json' "${auth[@]}" \
        -d "$payload" "$url/v1/completions" | python3 -c '
import json, sys
result = json.load(sys.stdin)
choices = result.get("choices", [])
if not choices or not isinstance(choices[0].get("text"), str):
    raise SystemExit("smoke completion returned no text choice")
print("Smoke completion passed:", repr(choices[0]["text"]))
'
fi
grep -E '^startup (provider_discovery|direct_[a-z_]*|weight_plan|weight_upload|module load|bundle create|KV pool acquisition|first graph capture|generation warmup|readiness):' "$server_log" | sed 's/^/  /' || true
server_started=0
}

validate_redeploy_options() {
    [[ $port =~ ^[0-9]+$ ]] && ((port >= 1 && port <= 65535)) || { echo 'port must be 1..65535' >&2; exit 2; }
    [[ $startup_timeout =~ ^[0-9]+$ ]] && ((startup_timeout >= 1)) || { echo 'startup timeout must be positive' >&2; exit 2; }
    [[ $release_timeout =~ ^[0-9]+$ ]] && ((release_timeout >= 1)) || { echo 'release timeout must be positive' >&2; exit 2; }
    [[ -z $max_sequences ]] || { [[ $max_sequences =~ ^[0-9]+$ ]] && ((max_sequences >= 1)); } || {
        echo 'max sequences must be positive' >&2
        exit 2
    }
    [[ -z $load_threads ]] || { [[ $load_threads =~ ^[0-9]+$ ]] && ((load_threads >= 1)); } || {
        echo 'load threads must be positive' >&2
        exit 2
    }
    [[ -z $prefix_cache || $prefix_cache == on || $prefix_cache == off ]] || { echo 'prefix cache must be on or off' >&2; exit 2; }
    [[ -z $kv_cache_dtype || $kv_cache_dtype == auto || $kv_cache_dtype == bf16 || $kv_cache_dtype == fp8 ]] || { echo 'KV cache dtype must be auto, bf16 or fp8' >&2; exit 2; }
    [[ $listen_host == 127.0.0.1 || $listen_host == 0.0.0.0 ]] || { echo 'listen host must be 127.0.0.1 or 0.0.0.0' >&2; exit 2; }
}

redeploy_serving_options() {
    serving_options=(--listen "$listen_host:$port")
    [[ -z $devices ]] || serving_options+=(--devices "$devices")
    [[ -z $max_sequences ]] || serving_options+=(--max-sequences "$max_sequences")
    [[ -z $max_batch_tokens ]] || serving_options+=(--max-batch-tokens "$max_batch_tokens")
    [[ -z $prefill_chunk_tokens ]] || serving_options+=(--prefill-chunk-tokens "$prefill_chunk_tokens")
    [[ -z $queue_depth ]] || serving_options+=(--queue-depth "$queue_depth")
    [[ -z $kv_memory_fraction ]] || serving_options+=(--kv-memory-fraction "$kv_memory_fraction")
    [[ -z $kv_cache_dtype ]] || serving_options+=(--kv-cache-dtype "$kv_cache_dtype")
    [[ -z $prefix_cache ]] || serving_options+=(--prefix-cache "$prefix_cache")
    [[ -z $load_threads ]] || serving_options+=(--load-threads "$load_threads")
}

# A model supplies only additional value options; common options set the
# existing script variables so preparation and serving retain their defaults.
parse_redeploy_options() {
    local extra_options=$1 option name variable message
    local value_options='ranks checkpoint model-cache-dir work-dir drafter activation-policy load-threads devices logs-dir port listen-host max-sequences max-batch-tokens prefill-chunk-tokens queue-depth kv-memory-fraction startup-timeout release-timeout prefix-cache kv-cache-dtype'
    shift
    while (($#)); do
        message=$(removed_flag_message "$1")
        if [[ -n $message ]]; then printf '%s\n' "$message" >&2; exit 2; fi
        case "$1" in
            --smoke) smoke=1; shift ;;
            --prepare-only) prepare_only=1; shift ;;
            --dry-run) dry_run=1; shift ;;
            --help|-h) usage; exit 0 ;;
            *)
                option=$1
                name=${option#--}
                if [[ $option != --* || ! $name =~ ^[a-z][a-z0-9-]*$ || " $value_options $extra_options serve-arg " != *" $name "* ]]; then
                    printf 'unknown option: %s\n' "$option" >&2; usage >&2; exit 2
                fi
                (($# >= 2)) || { printf 'missing value for %s\n' "$option" >&2; exit 2; }
                if [[ $name == serve-arg ]]; then
                    extra_serve_args+=("$2")
                else
                    variable=${name//-/_}
                    printf -v "$variable" '%s' "$2"
                fi
                shift 2 ;;
        esac
    done
}

redeploy_run() {
    local model=$1 label=$2
    umask 077
    mkdir -p -- "$logs_dir" "$run_root" "$model_cache_dir"
    exec {lock_fd}>"$logs_dir/$label-redeploy-$port.lock"
    flock -n "$lock_fd" || { echo 'another redeploy is using this port' >&2; exit 1; }

    total_started=$(now_ms)
    cd "$repo"

    build_started=$(now_ms)
    cargo build "${build_args[@]}"
    build_ms=$(( $(now_ms) - build_started ))

    install_started=$(now_ms)
    install_serving_binaries "$release_dir" "$install_dir" "$model"
    install_redeploy_extras "$release_dir" "$install_dir" "$model"
    install_ms=$(( $(now_ms) - install_started ))

    if ((explicit)); then
        prepare_explicit
    else
        prepare_direct
    fi
    if ((prepare_only)); then
        printf 'prepared %s\n' "$package_dir"
        exit 0
    fi

    serve_until_healthy
}

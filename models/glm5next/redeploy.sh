#!/usr/bin/env bash
set -Eeuo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
source "$repo/tools/dev/redeploy-common.sh"
for local_env in "$repo/local.env" "${CHOPINFER_LOCAL_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/chopinfer/local.env}"; do
    if [[ -f $local_env ]]; then source "$local_env"; break; fi
done
data=${CHOPINFER_DATA:-$HOME/chopinfer-data}
ranks=4
checkpoint=${CHOPINFER_GLM5NEXT_CHECKPOINT:-}
model_cache_dir=$data/model-cache
work_dir=$data/dev-redeploy
logs_dir=$data/logs
port=8441
listen_host=127.0.0.1
startup_timeout=600
release_timeout=${CHOPINFER_DEV_RELEASE_TIMEOUT:-120}
smoke=0
dry_run=0
prepare_only=0
drafter=
activation_policy=
load_threads=
devices=
context_parallel_size=
modalities=
owner_record=
# Serve options pass through only when given; the profile supplies the rest.
max_sequences=
max_batch_tokens=
prefill_chunk_tokens=
queue_depth=
kv_memory_fraction=
kv_cache_dtype=
prefix_cache=
extra_serve_args=()

usage() {
    cat <<'EOF'
Usage: models/glm5next/redeploy.sh [options]
Builds the host, chopinfer-model-glm5next and bundle-glm5next, copies them
into a private install, prepares the direct package in the model cache, then
serves `chopinfer serve --model CHECKPOINT` with the profile the helper
selects (tp4-mtp, for every checkpoint layout) as defaults. Only --dry-run
works until models/glm5next/model.toml lists the chopinfer-model-glm5next
helper.
With CHOPINFER_TAPS_DIR set it builds bundle-glm5next's diagnostic-taps
feature, which writes the encoder taps there (encoder_parity's build).
  --checkpoint DIR   Default $CHOPINFER_GLM5NEXT_CHECKPOINT (a checkpoint of
                     any admitted layout).
  --model-cache-dir DIR
                     Where the direct package goes (default
                     $CHOPINFER_DATA/model-cache).
  --work-dir DIR     Per-run private binaries under runs/<id>/ (default
                     $CHOPINFER_DATA/dev-redeploy).
  --logs-dir DIR
  --ranks 4          Tensor parallel size; glm5next serves TP4 only.
  --devices LIST     Serve on these visible ordinals.
  --drafter none|mtp Default mtp, the profile's MTP with three draft tokens.
                     none serves AR (--speculative-tokens 0). A --serve-arg
                     --speculative-tokens overrides the depth up to the
                     package's maximum.
  --context-parallel-size 1|2|4
                     Default 1. Create refuses 2 and 4 until the bundle's
                     context-parallel lowering passes its GPU check.
  --modalities LIST  Default the helper's auto (text and image when the
                     checkpoint has a vision tower). Reaches both the
                     helper's prepare and serve, so `--modalities text`
                     prepares and serves the text package (text benchmarks).
  --activation-policy auto|checkpoint|bf16  --load-threads N
  --port N --listen-host HOST
  --max-sequences N --max-batch-tokens N --prefill-chunk-tokens N
  --queue-depth N --kv-memory-fraction F
  --kv-cache-dtype auto|bf16|fp8  --prefix-cache on|off
                     Each overrides the profile's value.
  --serve-arg ARG    Append one serve argument; repeat for a flag and its value.
  --startup-timeout N  Default 600 s (direct loading reads 200 to 330 GB).
  --release-timeout N  Wait for old PID exit and GPU release (default 120 s).
  --smoke            Run one short greedy completion after health.
  --prepare-only     Build, install and prepare the package, print
                     `prepared <package dir>`, then exit without stopping or
                     starting a server (no GPU work).
  --dry-run          Print the complete redeploy sequence without changing state.
  --stop             Stop only the server this script's PID file for --port
                     under --logs-dir names (start time and token checked),
                     wait for its exit and GPU release, and exit. It takes
                     the same per-port lock as a start (waiting up to the
                     release timeout), so a stop never races a start. With
                     --dry-run it only prints what it would stop.
  --owner-record FILE
                     With --stop: stop only if the PID file still holds this
                     copy of it (a job's record of its own start); a record
                     of another start is left alone.

The private install comes from the model's own inventory entry (its
chopinfer-model- helper and bundle cdylib), not the release plan; nothing
is promoted to release and no running service is touched.
EOF
}

stop_only=0
arguments=()
for argument in "$@"; do
    if [[ $argument == --stop ]]; then stop_only=1; else arguments+=("$argument"); fi
done
parse_redeploy_options "context-parallel-size modalities owner-record" "${arguments[@]}"
[[ -z $owner_record ]] || ((stop_only)) || { echo '--owner-record applies to --stop' >&2; exit 2; }

[[ $ranks == 4 ]] || { echo 'glm5next serves TP4 only: pass --ranks 4' >&2; exit 2; }
[[ -n $drafter ]] || drafter=mtp
case $drafter in
    none|mtp) ;;
    *) echo 'drafter must be none or mtp' >&2; exit 2 ;;
esac
case $context_parallel_size in
    ''|1|2|4) ;;
    *) echo 'context-parallel size must be 1, 2 or 4' >&2; exit 2 ;;
esac
case $activation_policy in
    ''|auto|checkpoint|bf16) ;;
    *) echo 'activation policy must be auto, checkpoint or bf16' >&2; exit 2 ;;
esac
((stop_only)) || [[ -n $checkpoint ]] || { echo 'set --checkpoint or CHOPINFER_GLM5NEXT_CHECKPOINT' >&2; exit 2; }
validate_redeploy_options
model_cache_dir=$(realpath -m -- "$model_cache_dir")
work_dir=$(realpath -m -- "$work_dir")
logs_dir=$(realpath -m -- "$logs_dir")

export CARGO_HOME=${CARGO_HOME:-$HOME/.cargo}
export CARGO_NET_OFFLINE=true
export TMPDIR=${TMPDIR:-${CHOPINFER_SCRATCH:-/tmp}}

target_dir=${CARGO_TARGET_DIR:-$repo/target}
[[ $target_dir == /* ]] || target_dir=$repo/$target_dir
release_dir=$target_dir/release
run_id=$(date -u +%Y%m%dT%H%M%S)-$$
run_root=$work_dir/runs/$run_id
install_dir=$run_root/install
host_bin=$install_dir/bin/chopinfer
server_log=$logs_dir/glm5next-redeploy-$run_id.log
pidfile=$logs_dir/glm5next-redeploy-$port.pid
url=http://127.0.0.1:$port

speculation_set=0
for argument in "${extra_serve_args[@]}"; do
    [[ $argument == --speculative-tokens ]] && speculation_set=1
    # A serve-only selection would leave the prepared package another one.
    if [[ $argument == --modalities* ]]; then
        echo 'pass --modalities, not --serve-arg --modalities: the helper prepares the same package' >&2
        exit 2
    fi
done
# The helper chooses the package and the profile from the checkpoint's
# quantization layout; there is no explicit glm5next package path.
explicit=0
redeploy_serving_options
[[ -z $context_parallel_size ]] || serving_options+=(--context-parallel-size "$context_parallel_size")
if ((!speculation_set)); then
    [[ $drafter != none ]] || serving_options+=(--speculative-tokens 0)
fi
serving_options+=("${extra_serve_args[@]}")

package_dir='(named by the helper)'
helper_args=("$install_dir/bin/chopinfer-model-glm5next" prepare --direct --model "$checkpoint"
    --tensor-parallel-size "$ranks" --cache-dir "$model_cache_dir")
[[ -z $activation_policy ]] || helper_args+=(--activation-policy "$activation_policy")
[[ -z $modalities ]] || helper_args+=(--modalities "$modalities")
serve_args=(serve --model "$checkpoint" --tensor-parallel-size "$ranks"
    --model-cache-dir "$model_cache_dir")
[[ -z $activation_policy ]] || serve_args+=(--activation-policy "$activation_policy")
[[ -z $modalities ]] || serve_args+=(--modalities "$modalities")
serve_args+=("${serving_options[@]}")
features=runtime-host/linux-cuda-host
# The encoder taps run only in a diagnostic bundle; a serving one refuses
# to create while CHOPINFER_TAPS_DIR is set.
if [[ -n ${CHOPINFER_TAPS_DIR:-} ]]; then
    features+=,bundle-glm5next/diagnostic-taps
fi
build_args=(-p bundle-glm5next -p runtime-host -p model-glm5next
    --release --lib --bin chopinfer --bin chopinfer-model-glm5next
    --features "$features" --locked --offline)

if ((stop_only)); then
    if ((dry_run)); then
        # Nothing is locked, read beyond the record or signalled.
        printf 'would stop the server named by %s%s, then wait up to %ss for its exit and GPU release\n' \
            "$pidfile" "${owner_record:+ if it still equals $owner_record}" "$release_timeout"
        exit 0
    fi
    mkdir -p -- "$logs_dir"
    exec {lock_fd}>"$logs_dir/glm5next-redeploy-$port.lock"
    flock -w "$release_timeout" "$lock_fd" || { echo "a redeploy holds port $port's lock; not stopping" >&2; exit 1; }
    if [[ -n $owner_record ]]; then
        if [[ ! -e $pidfile ]]; then
            echo "no server is recorded in $pidfile; nothing of this owner to stop"
            exit 0
        fi
        if ! cmp -s -- "$owner_record" "$pidfile"; then
            echo "$pidfile records another start than $owner_record; leaving it running"
            exit 0
        fi
    fi
    stop_owned_server
    exit 0
fi

if ((dry_run)); then
    printf 'checkpoint: %s\nmodel cache: %s\nprivate install: %s\npackage: %s\nlog: %s\npidfile: %s\n' \
        "$checkpoint" "$model_cache_dir" "$install_dir" "$package_dir" "$server_log" "$pidfile"
    show_command cargo build "${build_args[@]}"
    printf '  copy chopinfer and chopinfer-model-glm5next to %s/bin and libbundle_glm5next.so to %s/lib, mode 0755\n' \
        "$install_dir" "$install_dir"
    printf '  prepare the direct package outside the health wait:\n'
    show_command "${helper_args[@]}"
    printf '  stop only the server identified by %s, wait up to %ss for PID exit and GPU release, then start:\n' "$pidfile" "$release_timeout"
    show_command "$host_bin" "${serve_args[@]}"
    printf '  wait for %s/health and %s/v1/models, timeout %ss\n' "$url" "$url" "$startup_timeout"
    ((smoke)) && printf '  POST one greedy completion to %s/v1/completions\n' "$url"
    exit 0
fi

# The development install: the host, the model's one chopinfer-model-
# helper and its bundle cdylibs, as `source<TAB>destination` rows read from
# the model's validated inventory entry. Refuses while the inventory lists
# no helper.
development_install_plan() {
    python3 -B - "$repo" <<'PY'
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / 'tools'))
from model_sdk import inventory
model = inventory.load_model(root, root / 'models' / 'glm5next')
targets = [inventory.Target(package.name, name, kind) for package in model.packages.values()
           for name, kind in sorted(package.targets)]
helpers = [t for t in targets if t.kind == 'bin' and t.name.startswith('chopinfer-model-')]
libraries = [t for t in targets if t.kind == 'cdylib']
if len(helpers) != 1 or not libraries:
    raise SystemExit('glm5next has no chopinfer-model-glm5next helper yet (models/glm5next/model.toml); '
                     'only --dry-run works')
print('chopinfer\tbin/chopinfer')
for target in [*helpers, *libraries]:
    destination = inventory.destination(target)
    print(f'{pathlib.Path(destination).name}\t{destination}')
PY
}

# Fresh files renamed into place, as install_serving_binaries installs a
# released model's plan.
install_development_binaries() {
    local release=$1 dest=$2 source destination path
    mkdir -p -- "$dest/bin" "$dest/lib"
    chmod 0755 -- "$dest" "$dest/bin" "$dest/lib"
    while IFS=$'\t' read -r source destination; do
        path=$dest/$destination
        [[ -f $release/$source ]] || { echo "missing build output $release/$source" >&2; return 1; }
        install -m 0755 -- "$release/$source" "$path.tmp.$$"
        mv -f -- "$path.tmp.$$" "$path"
    done <<< "$development_plan"
    if find "$dest" \( -perm -g+w -o -perm -o+w \) -print -quit | grep -q .; then
        echo "private install $dest has a group or world writable path" >&2
        return 1
    fi
}

# redeploy_run with the development install in place of the release plan's.
development_plan=$(development_install_plan) || exit 2
[[ -d $checkpoint && -r $checkpoint/config.json ]] || { echo "checkpoint directory is unreadable: $checkpoint" >&2; exit 1; }
umask 077
mkdir -p -- "$logs_dir" "$run_root" "$model_cache_dir"
exec {lock_fd}>"$logs_dir/glm5next-redeploy-$port.lock"
flock -n "$lock_fd" || { echo 'another redeploy is using this port' >&2; exit 1; }
total_started=$(now_ms)
cd "$repo"
build_started=$(now_ms)
cargo build "${build_args[@]}"
build_ms=$(( $(now_ms) - build_started ))
install_started=$(now_ms)
install_development_binaries "$release_dir" "$install_dir"
install_ms=$(( $(now_ms) - install_started ))
prepare_direct
if ((prepare_only)); then
    printf 'prepared %s\n' "$package_dir"
    exit 0
fi
serve_until_healthy

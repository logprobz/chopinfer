#!/usr/bin/env bash
set -Eeuo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
source "$repo/tools/dev/redeploy-common.sh"
for local_env in "$repo/local.env" "${CHOPINFER_LOCAL_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/chopinfer/local.env}"; do
    if [[ -f $local_env ]]; then source "$local_env"; break; fi
done
data=${CHOPINFER_DATA:-$HOME/chopinfer-data}
ranks=4
checkpoint=${CHOPINFER_QWEN38_CHECKPOINT:-}
model_cache_dir=$data/model-cache
work_dir=$data/dev-redeploy
logs_dir=$data/logs
port=8431
listen_host=127.0.0.1
startup_timeout=300
release_timeout=${CHOPINFER_DEV_RELEASE_TIMEOUT:-120}
smoke=0
dry_run=0
prepare_only=0
drafter=none
image_positions=
activation_policy=
load_threads=
devices=
# Serve options pass through only when given; the profile supplies the rest.
max_sequences=
max_batch_tokens=
prefill_chunk_tokens=
queue_depth=
gpu_memory_utilization=
kv_cache_dtype=
prefix_cache=
extra_serve_args=()

usage() {
    cat <<'EOF'
Usage: tools/dev/qwen38-redeploy.sh [options]
Builds the host, chopinfer-model-qwen38 and bundle-qwen38, copies them into a
private install, prepares the direct package in the model cache, then serves
`chopinfer serve --model CHECKPOINT` with the model's profile as defaults.
  --checkpoint DIR   Default $CHOPINFER_QWEN38_CHECKPOINT.
  --model-cache-dir DIR
                     Where the direct package goes (default
                     $CHOPINFER_DATA/model-cache).
  --work-dir DIR     Per-run private binaries and explicit packages under
                     runs/<id>/ (default $CHOPINFER_DATA/dev-redeploy).
  --logs-dir DIR
  --ranks 2|4        Tensor parallel size (default 4).
  --devices LIST     Serve on these visible ordinals.
  --drafter none|mtp Default none serves AR (--speculative-tokens 0). mtp
                     serves the tp2-mtp or tp4-mtp profile, which the helper
                     selects when the checkpoint's precision executes MTP.
                     A --serve-arg --speculative-tokens overrides the depth.
  --image-positions grid-3axis|sequential
                     Image row positions; sequential is the pinned vLLM's rule,
                     for comparison, and serves an explicit qwen38-package.
                     The vision encoders are planned whenever the checkpoint
                     has them.
  --activation-policy auto|checkpoint|bf16  --load-threads N
  --port N --listen-host HOST
  --max-sequences N --max-batch-tokens N --prefill-chunk-tokens N
  --queue-depth N --gpu-memory-utilization F
  --kv-cache-dtype auto|bf16|fp8  --prefix-cache on|off
                     Each overrides the profile's value.
  --serve-arg ARG    Append one serve argument; repeat for a flag and its value.
  --startup-timeout N
  --release-timeout N  Wait for old PID exit and GPU release (default 120 s).
  --smoke            Run one short greedy completion after health.
  --prepare-only     Build, install and prepare the package, print
                     `prepared <package dir>`, then exit without stopping or
                     starting a server (no GPU work).
  --dry-run          Print the complete redeploy sequence without changing state.
EOF
}

parse_redeploy_options "image-positions" "$@"

case $ranks in
    2|4) ;;
    *) echo 'ranks must be 2 or 4' >&2; exit 2 ;;
esac
case $drafter in
    none|mtp) ;;
    *) echo 'drafter must be none or mtp' >&2; exit 2 ;;
esac
case $image_positions in
    ''|grid-3axis|sequential) ;;
    *) echo 'image positions must be grid-3axis or sequential' >&2; exit 2 ;;
esac
case $activation_policy in
    ''|auto|checkpoint|bf16) ;;
    *) echo 'activation policy must be auto, checkpoint or bf16' >&2; exit 2 ;;
esac
[[ -n $checkpoint ]] || { echo 'set --checkpoint or CHOPINFER_QWEN38_CHECKPOINT' >&2; exit 2; }
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
server_log=$logs_dir/qwen38-redeploy-$run_id.log
pidfile=$logs_dir/qwen38-redeploy-$port.pid
url=http://127.0.0.1:$port

speculation_set=0
for argument in "${extra_serve_args[@]}"; do
    [[ $argument == --speculative-tokens ]] && speculation_set=1
done
# The helper chooses the profile and the package from the checkpoint. Choices
# it does not make go through an explicit qwen38-package, served with
# --bundle and --checkpoint under the matching profile.
explicit=0
[[ $image_positions == sequential ]] && explicit=1
redeploy_serving_options
if ((!speculation_set)); then
    if [[ $drafter == none ]]; then
        serving_options+=(--speculative-tokens 0)
    fi
fi
serving_options+=("${extra_serve_args[@]}")

if ((explicit)); then
    profile=tp$ranks-ar; [[ $drafter == none ]] || profile=tp$ranks-mtp
    package_dir=$run_root/package
    package_args=("$install_dir/bin/qwen38-package" --checkpoint "$checkpoint"
        --module "$install_dir/lib/libbundle_qwen38.so" --output "$package_dir"
        --drafter "$drafter")
    [[ -z $activation_policy ]] || package_args+=(--activation-policy "$activation_policy")
    if [[ $image_positions == sequential ]]; then
        package_args+=(--encoders on --image-positions sequential)
    elif checkpoint_has_vision "$checkpoint"; then
        package_args+=(--encoders on)
    elif ((!dry_run)); then
        package_args+=(--encoders off)
    else
        package_args+=(--encoders on)
    fi
    serve_args=(serve --profile "$repo/models/qwen38/profiles.json#$profile"
        --bundle "$package_dir" --checkpoint "$checkpoint" "${serving_options[@]}")
else
    package_dir='(named by the helper)'
    helper_args=("$install_dir/bin/chopinfer-model-qwen38" prepare --direct --model "$checkpoint"
        --tensor-parallel-size "$ranks" --cache-dir "$model_cache_dir")
    [[ -z $activation_policy ]] || helper_args+=(--activation-policy "$activation_policy")
    serve_args=(serve --model "$checkpoint" --tensor-parallel-size "$ranks"
        --model-cache-dir "$model_cache_dir")
    [[ -z $activation_policy ]] || serve_args+=(--activation-policy "$activation_policy")
    serve_args+=("${serving_options[@]}")
fi
features=runtime-host/linux-cuda-host
# Per-layer taps run only in a diagnostic bundle; a serving one refuses them.
if [[ -n ${CHOPINFER_TAPS_DIR:-}${CHOPINFER_QWEN38_DUMP_DIR:-} ]]; then
    features+=,bundle-qwen38/diagnostic-taps
fi
build_args=(-p bundle-qwen38 -p runtime-host -p model-qwen38
    --release --lib --bin chopinfer --bin chopinfer-model-qwen38 --bin qwen38-package
    --features "$features" --locked --offline)

if ((dry_run)); then
    printf 'checkpoint: %s\nmodel cache: %s\nprivate install: %s\npackage: %s\nlog: %s\npidfile: %s\n' \
        "$checkpoint" "$model_cache_dir" "$install_dir" "$package_dir" "$server_log" "$pidfile"
    show_command cargo build "${build_args[@]}"
    printf '  copy chopinfer, chopinfer-model-qwen38 and qwen38-package to %s/bin and libbundle_qwen38.so to %s/lib, mode 0755\n' \
        "$install_dir" "$install_dir"
    if ((explicit)); then
        printf '  package the choices the helper does not make:\n'
        show_command "${package_args[@]}"
    else
        printf '  prepare the direct package outside the health wait:\n'
        show_command "${helper_args[@]}"
    fi
    printf '  stop only the server identified by %s, wait up to %ss for PID exit and GPU release, then start:\n' "$pidfile" "$release_timeout"
    show_command "$host_bin" "${serve_args[@]}"
    printf '  wait for %s/health and %s/v1/models, timeout %ss\n' "$url" "$url" "$startup_timeout"
    ((smoke)) && printf '  POST one greedy completion to %s/v1/completions\n' "$url"
    exit 0
fi

[[ -d $checkpoint && -r $checkpoint/config.json ]] || { echo "checkpoint directory is unreadable: $checkpoint" >&2; exit 1; }
redeploy_run qwen38 qwen38

# Shared settings for the Qwen3.8 capture scripts. Source it; do not run it.
#
capture_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$capture_dir/../../.." && pwd -P)
for local_env in "$repo_dir/local.env" "${CHOPINFER_LOCAL_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/chopinfer/local.env}"; do
    if [[ -f $local_env ]]; then source "$local_env"; break; fi
done

QWEN38_REVISION=236dfdf285828023ca3bcd3f37366c58a3469b13
: "${CHOPINFER_QWEN38_CHECKPOINT:?set CHOPINFER_QWEN38_CHECKPOINT to the 236dfdf2 snapshot directory}"
: "${CHOPINFER_DATA:?set CHOPINFER_DATA}"
QWEN38_REFS=${QWEN38_REFS:-$CHOPINFER_DATA/qwen38-refs}

qwen38_require_snapshot() {
    [[ $(basename -- "$CHOPINFER_QWEN38_CHECKPOINT") == "$QWEN38_REVISION" ]] || {
        echo "CHOPINFER_QWEN38_CHECKPOINT is not snapshot $QWEN38_REVISION" >&2; return 1; }
    local name
    for name in config.json model.safetensors.index.json tokenizer.json chat_template.jinja preprocessor_config.json; do
        [[ -r $CHOPINFER_QWEN38_CHECKPOINT/$name ]] || { echo "missing $name in the checkpoint" >&2; return 1; }
    done
}

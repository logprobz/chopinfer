#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 || $# -gt 3 || ! $1 =~ ^(1|5)$ || ! $2 =~ ^(2048|8192)$ ]]; then
    printf 'usage: %s <layer:1|5> <tokens:2048|8192> [output-prefix]\n' "$0" >&2
    exit 2
fi
layer=$1
tokens=$2
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
for local_env in "$root/local.env" "${CHOPINFER_LOCAL_ENV:-${XDG_CONFIG_HOME:-$HOME/.config}/chopinfer/local.env}"; do
    if [[ -f $local_env ]]; then source "$local_env"; break; fi
done
output=${3:-${TMPDIR:-/tmp}/chopinfer-prefill-l${layer}-t${tokens}}
mkdir -p -- "$(dirname -- "$output")"

# The test plans the layers' weights from the checkpoint and reads them in
# memory; no rank images are needed.
: "${CHOPINFER_MIMO_CHECKPOINT:?set CHOPINFER_MIMO_CHECKPOINT to the MiMo checkpoint}"
: "${CHOPINFER_KERNEL_GPUS:=0,1,2,3}"
: "${CARGO_HOME:=$HOME/.cargo}"
: "${TMPDIR:=${CHOPINFER_SCRATCH:-/tmp}}"
export CHOPINFER_MIMO_CHECKPOINT CHOPINFER_KERNEL_GPUS CARGO_HOME TMPDIR
export CARGO_NET_OFFLINE=true
[[ -f $CHOPINFER_MIMO_CHECKPOINT/model.safetensors.index.json ]] || {
    printf 'missing model.safetensors.index.json in %s\n' "$CHOPINFER_MIMO_CHECKPOINT" >&2
    exit 2
}

build_log=$(mktemp "${TMPDIR}/prefill-chain-build.XXXXXX")
trap 'rm -f -- "$build_log"' EXIT
cd "$root"
cargo test -p kernel-lab --test prefill_chain --release --locked --offline \
    --no-run --message-format=json > "$build_log"
test_binary=$(python3 - "$build_log" <<'PY'
import json
import sys

matches = []
with open(sys.argv[1], encoding="utf-8") as stream:
    for line in stream:
        message = json.loads(line)
        if (message.get("reason") == "compiler-artifact"
                and message.get("target", {}).get("name") == "prefill_chain"
                and message.get("executable")):
            matches.append(message["executable"])
if len(matches) != 1:
    raise SystemExit(f"expected one prefill_chain test binary, found {len(matches)}")
print(matches[0])
PY
)

export CHOPINFER_MIMO_PREFILL_LAYER=$layer CHOPINFER_MIMO_PREFILL_TOKENS=$tokens
if ! timeout -s KILL 600 /usr/local/cuda/bin/nsys profile \
    --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
    --export=sqlite --force-overwrite=true --output="$output" \
    "$test_binary" --exact prefill_chain_graph_replay --nocapture \
    > "${output}.nsys.log" 2>&1; then
    tail -n 30 "${output}.nsys.log" >&2
    exit 1
fi
if ! grep -q 'test prefill_chain_graph_replay ... ok' "${output}.nsys.log"; then
    tail -n 30 "${output}.nsys.log" >&2
    exit 1
fi
printf 'prefill_chain_graph_replay: eager and graph bitwise checks passed\n'
python3 "$root/tools/profile/decode_chain_nsys.py" "${output}.sqlite" \
    --layer "$layer" --tokens "$tokens" --ordinals "$CHOPINFER_KERNEL_GPUS" \
    --output "${output}.json"

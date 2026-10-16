#!/usr/bin/env bash
# Build the frozen Qwen3.8 prompt set on the host (CPU only, under a minute).
#   bash fixtures/qwen38/capture/prompts.sh [--check]
# Writes ${QWEN38_REFS}/prompts/prompts.json and fixtures/qwen38/prompts.json.
# --check verifies the environment and inputs, then writes nothing.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
qwen38_require_snapshot
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
: "${CHOPINFER_MIMO_CHECKPOINT:?the needle haystack is decoded with the MiMo tokenizer}"
: "${CHOPINFER_MIMO_FIXTURES:?the needle haystack comes from the MiMo functional fixtures}"
if [[ ${1:-} == --check ]]; then
    "${CHOPINFER_REFRUN_PYTHON:?set the reference Python}" -c 'import tokenizers, transformers; print("tokenizers", tokenizers.__version__, "transformers", transformers.__version__)'
    for file in "$CHOPINFER_MIMO_CHECKPOINT/tokenizer.json" "$CHOPINFER_MIMO_FIXTURES/functional/context-524288.json" \
        "$CHOPINFER_DATA/evals/wikitext.json" "$CHOPINFER_DATA/evals/manifest.json"; do
        [[ -r $file ]] || { echo "missing $file" >&2; exit 1; }
    done
    echo "prompts: inputs present; would write $QWEN38_REFS/prompts/prompts.json"
    exit 0
fi
"${CHOPINFER_REFRUN_PYTHON:?set the reference Python}" "$capture_dir/prompts.py" --output "$QWEN38_REFS"

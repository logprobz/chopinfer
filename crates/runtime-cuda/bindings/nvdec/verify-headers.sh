#!/usr/bin/env bash
set -Eeuo pipefail

NVDEC_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

check_hash() {
  local name=$1
  local expected=$2
  local path=${NVDEC_ROOT}/${name}
  [[ -f "${path}" ]] || {
    echo "NVDEC header is missing: ${path}" >&2
    exit 1
  }
  local actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ "${actual}" == "${expected}" ]] || {
    echo "NVDEC header ${name} SHA-256 differs from pinned provenance" >&2
    exit 1
  }
}

check_hash dynlink_cuviddec.h 1244165ea5521d4be865995eb1c49b8910f755d0ca35004e1c4010ef7f612143
check_hash dynlink_nvcuvid.h 2141dade75f3178520092dce2c02df9b47c4b19036b56bb9316bd83a07bf16f1
echo "verified pinned NVDEC headers"

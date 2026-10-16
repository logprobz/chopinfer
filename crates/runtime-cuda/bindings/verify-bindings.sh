#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_HEADER_SHA256=673873a292718591d75ea3c2e65330b1f8f8278eac3c96b32245418fb0817851
EXPECTED_WRAPPER_SHA256=776abee6aa7027474a1076f50375e09ed31da011cb9a557896d56addef797d7a
EXPECTED_ALLOWLIST_SHA256=e0fbcd20cf7f3b3ecee8eee543205bccd967776a1ea5d17c0ac314a456eb2415
EXPECTED_RAW_SHA256=538bddca3bb64608bef57ba8f27b6842094a7752fb817545491c8611b6400c6a
EXPECTED_CUDA_VERSION=13020

BINDING_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
HEADER=${1:-/usr/local/cuda/targets/x86_64-linux/include/cuda.h}
MODE=${2:-full}
WRAPPER=${BINDING_ROOT}/bindings/cuda-driver-wrapper.h
ALLOWLIST=${BINDING_ROOT}/bindings/allowlist-functions.txt
RAW=${BINDING_ROOT}/bindings/cuda-driver-13.2.rs.inc

check_hash() {
  local label=$1
  local expected=$2
  local path=$3
  [[ -f "${path}" ]] || {
    echo "${label} is missing: ${path}" >&2
    exit 1
  }
  local actual
  actual=$(sha256sum "${path}" | awk '{print $1}')
  [[ "${actual}" == "${expected}" ]] || {
    echo "${label} SHA-256 differs from pinned provenance" >&2
    exit 1
  }
}

check_hash "CUDA header" "${EXPECTED_HEADER_SHA256}" "${HEADER}"
actual_version=$(awk '$1 == "#define" && $2 == "CUDA_VERSION" {print $3; exit}' "${HEADER}")
[[ "${actual_version}" == "${EXPECTED_CUDA_VERSION}" ]] || {
  echo "CUDA header version differs from pinned CUDA_VERSION 13020" >&2
  exit 1
}
check_hash "CUDA bindgen wrapper" "${EXPECTED_WRAPPER_SHA256}" "${WRAPPER}"
check_hash "CUDA function allowlist" "${EXPECTED_ALLOWLIST_SHA256}" "${ALLOWLIST}"

if [[ "${MODE}" == "--inputs-only" ]]; then
  exit 0
fi
[[ "${MODE}" == "full" ]] || {
  echo "verify-bindings accepts only optional --inputs-only mode" >&2
  exit 1
}
check_hash "generated CUDA raw binding" "${EXPECTED_RAW_SHA256}" "${RAW}"
echo "verified pinned CUDA binding inputs and generated raw module"

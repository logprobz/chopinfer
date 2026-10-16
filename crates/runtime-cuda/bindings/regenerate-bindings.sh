#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_BINDGEN_VERSION='bindgen 0.72.1'
EXPECTED_LIBCLANG_PACKAGE='1:18.1.3-1ubuntu1'
CUDA_INCLUDE=/usr/local/cuda/targets/x86_64-linux/include

BINDING_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
HEADER=${1:-${CUDA_INCLUDE}/cuda.h}
MODE=${2:---check}
ALLOWLIST=${BINDING_ROOT}/bindings/allowlist-functions.txt
WRAPPER=${BINDING_ROOT}/bindings/cuda-driver-wrapper.h
CHECKED_IN=${BINDING_ROOT}/bindings/cuda-driver-13.2.rs.inc

[[ "$(bindgen --version)" == "${EXPECTED_BINDGEN_VERSION}" ]] || {
  echo "binding generation requires exact bindgen-cli 0.72.1" >&2
  exit 1
}
actual_libclang=$(dpkg-query -W -f='${Version}' libclang-18-dev)
[[ "${actual_libclang}" == "${EXPECTED_LIBCLANG_PACKAGE}" ]] || {
  echo "binding generation requires exact Ubuntu libclang-18-dev 18.1.3-1ubuntu1" >&2
  exit 1
}
bash "${BINDING_ROOT}/bindings/verify-bindings.sh" "${HEADER}" --inputs-only
LC_ALL=C sort -c -u "${ALLOWLIST}" || {
  echo "CUDA function allowlist must be sorted and unique" >&2
  exit 1
}

GENERATION_ROOT=$(mktemp -d)
trap 'rm -rf -- "${GENERATION_ROOT}"' EXIT
RAW_OUTPUT=${GENERATION_ROOT}/bindgen.rs
NORMALIZED=${GENERATION_ROOT}/cuda-driver-13.2.rs.inc

bindgen_args=()
previous=''
while IFS= read -r function; do
  [[ "${function}" =~ ^cu[A-Za-z0-9_]+$ ]] || {
    echo "invalid CUDA function allowlist entry: ${function}" >&2
    exit 1
  }
  [[ "${function}" != "${previous}" ]] || {
    echo "duplicate CUDA function allowlist entry: ${function}" >&2
    exit 1
  }
  previous=${function}
  bindgen_args+=(--allowlist-function "^${function}$")
done <"${ALLOWLIST}"

for name in \
  CUdeviceptr_v2 CUdeviceptr CUdevice_v1 CUdevice CUcontext CUmodule CUfunction \
  CUlibrary CUkernel CUevent CUstream CUgraph CUgraphExec CUmemoryPool \
  CUdevice_attribute_enum CUdevice_attribute CUjit_option_enum CUjit_option \
  CUlibraryOption_enum CUlibraryOption cudaError_enum CUresult \
  CUmemPool_attribute_enum CUmemPool_attribute CUstreamCaptureMode_enum \
  CUstreamCaptureMode CUctx_st CUmod_st CUfunc_st CUlib_st CUkern_st \
  CUevent_st CUstream_st CUgraph_st CUgraphExec_st CUmemPoolHandle_st; do
  bindgen_args+=(--blocklist-type "^${name}$")
done

raw_lines=(
  'pub type CUdeviceptr = u64;'
  'pub type CUdevice = ::core::ffi::c_int;'
  'pub enum CUctx_st {}'
  'pub enum CUmod_st {}'
  'pub enum CUfunc_st {}'
  'pub enum CUlib_st {}'
  'pub enum CUkern_st {}'
  'pub enum CUevent_st {}'
  'pub enum CUstream_st {}'
  'pub enum CUgraph_st {}'
  'pub enum CUgraphExec_st {}'
  'pub enum CUmemPoolHandle_st {}'
  'pub type CUcontext = *mut CUctx_st;'
  'pub type CUmodule = *mut CUmod_st;'
  'pub type CUfunction = *mut CUfunc_st;'
  'pub type CUlibrary = *mut CUlib_st;'
  'pub type CUkernel = *mut CUkern_st;'
  'pub type CUevent = *mut CUevent_st;'
  'pub type CUstream = *mut CUstream_st;'
  'pub type CUgraph = *mut CUgraph_st;'
  'pub type CUgraphExec = *mut CUgraphExec_st;'
  'pub type CUmemoryPool = *mut CUmemPoolHandle_st;'
  'pub type CUdevice_attribute = ::core::ffi::c_uint;'
  'pub type CUjit_option = ::core::ffi::c_uint;'
  'pub type CUlibraryOption = ::core::ffi::c_uint;'
  'pub type CUresult = ::core::ffi::c_uint;'
  'pub type CUmemPool_attribute = ::core::ffi::c_uint;'
  'pub type CUstreamCaptureMode = ::core::ffi::c_uint;'
)
for line in "${raw_lines[@]}"; do
  bindgen_args+=(--raw-line "${line}")
done

LIBCLANG_PATH=/usr/lib/llvm-18/lib bindgen "${WRAPPER}" \
  --output "${RAW_OUTPUT}" \
  --formatter prettyplease \
  --rust-target 1.89 \
  --rust-edition 2024 \
  --use-core \
  --no-layout-tests \
  --no-doc-comments \
  --sort-semantically \
  --merge-extern-blocks \
  --default-enum-style consts \
  --no-prepend-enum-name \
  --allowlist-type '^runtime_cuda_constants$' \
  --allowlist-var '^RUNTIME_[A-Z0-9_]+$' \
  --generate functions,types,vars \
  "${bindgen_args[@]}" \
  -- -I"${CUDA_INCLUDE}"

awk '
  /^unsafe extern "C" \{$/ && !reviewed {
    print "#[cfg_attr("
    print "    all(target_os = \"linux\", not(feature = \"fake-driver-tests\")),"
    print "    link(name = \"libcuda.so.1\", kind = \"dylib\", modifiers = \"+verbatim\")"
    print ")]"
    print "// SAFETY-REVIEWED: generated from the pinned CUDA header and closed allowlist."
    reviewed=1
  }
  { print }
  END { if (!reviewed) exit 1 }
' "${RAW_OUTPUT}" >"${NORMALIZED}"

case "${MODE}" in
  --check)
    cmp "${NORMALIZED}" "${CHECKED_IN}" >/dev/null || {
      echo "checked-in CUDA raw binding differs from pinned bindgen output" >&2
      exit 1
    }
    echo "regenerated CUDA raw binding matches checked-in module"
    ;;
  --write)
    cp "${NORMALIZED}" "${CHECKED_IN}"
    echo "updated checked-in CUDA raw binding"
    ;;
  *)
    echo "regenerate-bindings accepts only --check or --write" >&2
    exit 1
    ;;
esac

# CUDA Driver binding provenance

The private raw binding is generated from immutable inputs:

- Reproduction base: `nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:44a9504c6dfb50b1241464241b02a93871928f373de6f5a644cf5fe9f080aa63`
- Compiler packages: `13.2.86-1`, installed by `tools/release/setup-toolchain.sh`
- Historical builder: `voipmonitor/vllm@sha256:a8a7a05fe8a8599b2ff86cfa2814e4a3b1ec1a28fb154b496f891115f28a6c84`.
  The official base above is the reproduction recipe; byte equality still
  requires the check below.
- CUDA header: `/usr/local/cuda/targets/x86_64-linux/include/cuda.h`
- CUDA header version: `CUDA_VERSION 13020`
- CUDA header SHA-256: `673873a292718591d75ea3c2e65330b1f8f8278eac3c96b32245418fb0817851`
- generator: `bindgen-cli 0.72.1` (`bindgen 0.72.1`)
- parser: Ubuntu `libclang-18-dev=1:18.1.3-1ubuntu1`
- wrapper SHA-256: `776abee6aa7027474a1076f50375e09ed31da011cb9a557896d56addef797d7a`
- function allowlist SHA-256: `e0fbcd20cf7f3b3ecee8eee543205bccd967776a1ea5d17c0ac314a456eb2415`
- generated raw module SHA-256: `538bddca3bb64608bef57ba8f27b6842094a7752fb817545491c8611b6400c6a`

`allowlist-functions.txt` is the complete function boundary.
`cuda-driver-wrapper.h` derives eight named constants from the pinned CUDA
header. `regenerate-bindings.sh` blocklists CUDA's broad enum and handle types,
supplies their reviewed C ABI aliases as fixed `--raw-line` arguments, and
generates functions and constant values from the header with the closed
allowlist.

The exact reproducibility check is:

```sh
bash crates/runtime-cuda/bindings/regenerate-bindings.sh \
  /usr/local/cuda/targets/x86_64-linux/include/cuda.h --check
```

The exact update command is the same command with `--write`. The script checks
the complete bindgen version string, exact libclang package version, header
hash and version, wrapper hash, and sorted function allowlist before invoking
the frozen bindgen flags recorded in the script. It generates into a temporary
directory, inserts only the reviewed `libcuda.so.1` link attribute and
`SAFETY-REVIEWED` marker, then byte-compares the result with
`cuda-driver-13.2.rs.inc`. That link attribute names the driver on Linux
except under `fake-driver-tests`, where the no-device stub is the driver and a
build must not reach the real one.

The normal Linux build does not require or invoke bindgen. It runs
`verify-bindings.sh`, which checks all pinned input hashes and the generated
module hash. `ffi.rs` includes the raw module inside private `ffi::raw` and
keeps every ergonomic wrapper private to `runtime-cuda`.

The explicit event timing addition uses the existing C-helper boundary in
`csrc/tensor_map.c`: `runtime_cuda_event_elapsed` calls `cuEventElapsedTime`.
The helper declaration and definition compile against the same pinned header.
`cuda_host_abi_fake_driver_test.c` checks argument order, returned milliseconds,
and error propagation with a fake `cuEventElapsedTime`. The generated raw
binding, allowlist, wrapper, and their hashes remain unchanged. Full bindgen
regeneration was unavailable for this addition because the pinned bindgen and
libclang packages were absent. The normal provenance check and C compilation
both passed; this is not a claim that regeneration ran.

The lendable device memory addition uses the same C-helper boundary in
`csrc/tensor_map.c`. `runtime_cuda_vmm_granularity`, `runtime_cuda_vmm_create`,
`runtime_cuda_vmm_export`, `runtime_cuda_vmm_import`,
`runtime_cuda_vmm_set_access` and `runtime_cuda_vmm_release` call
`cuMemGetAllocationGranularity`, `cuMemCreate`, `cuMemAddressReserve`,
`cuMemMap`, `cuMemSetAccess`, `cuMemExportToShareableHandle`,
`cuMemImportFromShareableHandle`, `cuMemUnmap`, `cuMemAddressFree` and
`cuMemRelease` with POSIX file descriptor handles. The helper declarations and
definitions compile against the same pinned header.
`cuda_host_abi_fake_driver_test.c` checks the call order, allocation
properties, access flags, and the cleanup of every partial step on each
failure with fake driver functions. The no-device stub backs the helpers with
host memory so the Rust fake-driver tests can share bytes across mappings. The
generated raw binding, allowlist, wrapper, and their hashes remain unchanged.
Full bindgen regeneration was unavailable for this addition for the same
reason as the event timing addition. The normal provenance check and C
compilation both passed; this is not a claim that regeneration ran.

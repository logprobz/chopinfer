#!/usr/bin/env bash
# Run as root inside nvidia/cuda:13.2.1-devel-ubuntu24.04.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
apt-get update
apt-get install -y --no-install-recommends --allow-downgrades \
    build-essential ca-certificates curl git python3 \
    cuda-nvcc-13-2=13.2.86-1 cuda-crt-13-2=13.2.86-1 \
    cuda-cccl-13-2=13.2.86-1 cuda-cudart-13-2=13.2.86-1 \
    cuda-cudart-dev-13-2=13.2.86-1 cuda-driver-dev-13-2=13.2.86-1 \
    cuda-cuobjdump-13-2=13.2.86-1 cuda-nvdisasm-13-2=13.2.86-1
# CPU tests and host linking need the driver SONAME, supplied only by a stub.
mkdir -p /opt/cuda-link-stub
ln -sf /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so /opt/cuda-link-stub/libcuda.so.1
rust_version=$(python3 -c 'import tomllib; print(tomllib.load(open("rust-toolchain.toml", "rb"))["toolchain"]["channel"])')
bash tools/release/install-rust.sh "$rust_version"

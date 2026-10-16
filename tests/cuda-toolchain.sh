#!/usr/bin/env bash
# Every host kernel compilation must use the selected, checked CUDA toolkit.
set -Eeuo pipefail
work=$(mktemp -d "${TMPDIR:-/tmp}/chopinfer-cuda-toolchain.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
export CHOPINFER_CUDA_HOME
CHOPINFER_CUDA_HOME=$(realpath "${CHOPINFER_CUDA_HOME:-/usr/local/cuda}")
cat > "$work/nvcc-wrapper" <<'WRAPPER'
#!/usr/bin/env bash
set -Eeuo pipefail
if [[ $1 != "$CHOPINFER_CUDA_HOME/bin/nvcc" ]]; then
    echo 'kernel compilation did not use the selected CUDA toolkit' >&2
    exit 1
fi
exec "$@"
WRAPPER
chmod +x "$work/nvcc-wrapper"
CHOPINFER_NVCC_WRAPPER="$work/nvcc-wrapper" cargo build -p runtime-kernels --locked --offline

#!/usr/bin/env bash
# Build a distributable alpha with locked, embedded cubins. No GPU is needed.
set -Eeuo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$root"
stage_from=
if [[ ${1:-} == --stage-from ]]; then
    [[ $# == 3 ]] || { echo "usage: tools/release/build.sh [--stage-from RELEASE_DIRECTORY] OUTPUT_DIRECTORY" >&2; exit 2; }
    stage_from=$2
    shift 2
fi
[[ $# == 1 ]] || { echo "usage: tools/release/build.sh [--stage-from RELEASE_DIRECTORY] OUTPUT_DIRECTORY" >&2; exit 2; }
[[ $(uname -sm) == 'Linux x86_64' ]] || { echo 'Linux x86_64 required' >&2; exit 1; }
[[ ${KERNELS_LOCK_UPDATE:-0} != 1 && ${CHOPINFER_KERNEL_LINEINFO:-0} != 1 ]] || {
    echo 'Release builds forbid kernel lock updates and profiling bypasses' >&2; exit 1;
}
export KERNELS_LOCK_UPDATE=0 CHOPINFER_KERNEL_LINEINFO=0
export CARGO_BUILD_JOBS=8 CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0
export CARGO_NET_OFFLINE=true
: "${CARGO_TARGET_DIR:?set CARGO_TARGET_DIR to a build scratch directory}"
mkdir -p "$CARGO_TARGET_DIR"
export CARGO_TARGET_DIR=$(cd "$CARGO_TARGET_DIR" && pwd -P)
version=$(python3 - <<'PY'
import re, tomllib
version = tomllib.load(open('Cargo.toml', 'rb'))['workspace']['package']['version']
if not re.fullmatch(r'(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)-alpha\.(0|[1-9][0-9]*)', version):
    raise SystemExit('workspace version must be a SemVer alpha release')
print(version)
PY
)
export CHOPINFER_BUILD_REVISION=${CHOPINFER_BUILD_REVISION:-$(git rev-parse --verify HEAD)}
[[ $CHOPINFER_BUILD_REVISION =~ ^[0-9a-f]{40}$ ]] || { echo 'Full Git revision required' >&2; exit 1; }
toolchain=$(python3 -c 'import tomllib; print(tomllib.load(open("rust-toolchain.toml", "rb"))["toolchain"]["channel"])')
[[ $(rustc --version) == "rustc $toolchain "* ]] || { echo 'Pinned Rust toolchain required' >&2; exit 1; }
cuda_home=${CHOPINFER_CUDA_HOME:-/usr/local/cuda}
"$cuda_home/bin/nvcc" --version | grep -F 'V13.2.86'
# Match the driver's SONAME for CPU-only linking, without shipping a stub.
stub="$CARGO_TARGET_DIR/cuda-link-stub"
mkdir -p "$stub"
ln -sf "$cuda_home/targets/x86_64-linux/lib/stubs/libcuda.so" "$stub/libcuda.so.1"
export LIBRARY_PATH="$stub${LIBRARY_PATH:+:$LIBRARY_PATH}"
mkdir -p "$1"
out=$(cd "$1" && pwd -P)
name="chopinfer-$version-linux-x86_64-cuda13.2"
[[ ! -e $out/$name.tar.gz ]] || { echo 'Archive already exists' >&2; exit 1; }
stage=$(mktemp -d "$out/.stage.XXXXXX")
trap 'rm -rf "$stage"' EXIT
# Both lock forms: kernels.lock files and the family files of kernels.lock.d.
find models \( -name kernels.lock -o -path '*/kernels.lock.d/*' \) -type f -print0 | sort -z | xargs -0 sha256sum > "$stage/kernel-locks.before"
cargo metadata --format-version 1 --no-deps --locked --offline > "$stage/metadata.json"
python3 -B tools/release/plan.py --metadata "$stage/metadata.json" --view stage > "$stage/staging-plan"
if [[ -z $stage_from ]]; then
    nice -n 19 cargo build --release --locked --offline -p runtime-host --no-default-features \
        --features linux-cuda-host --bin chopinfer --bin chopinfer-host
    # A separate package without the CUDA host features; it links no serving code.
    nice -n 19 cargo build --release --locked --offline -p chopinfer-top --bin chopinfer-top
    python3 -B tools/release/plan.py --metadata "$stage/metadata.json" --view build-bins > "$stage/bin-args"
    mapfile -t bin_args < "$stage/bin-args"
    if ((${#bin_args[@]})); then
        nice -n 19 cargo build --release --locked --offline "${bin_args[@]}"
    fi
    python3 -B tools/release/plan.py --metadata "$stage/metadata.json" --view build-libs > "$stage/lib-args"
    mapfile -t lib_args < "$stage/lib-args"
    if ((${#lib_args[@]})); then
        nice -n 19 cargo build --release --locked --offline "${lib_args[@]}"
    fi
    stage_from=$CARGO_TARGET_DIR/release
fi
sha256sum --check "$stage/kernel-locks.before"
package="$stage/$name"
mkdir -p "$package/bin" "$package/lib"
while IFS=$'\t' read -r kind source destination mode; do
    case $kind in
        binary) install -D -m "$mode" -- "$stage_from/$source" "$package/$destination" ;;
        asset)
            mkdir -p -- "$(dirname -- "$package/$destination")"
            cp -- "$source" "$package/$destination"
            ;;
        eval)
            # Preserve authenticated eval discovery and its checkout paths.
            # The third column names a checkpoint variant, or is - for the
            # model's own declaration.
            variant=()
            [[ $destination == - ]] || variant=(--variant "$destination")
            LD_LIBRARY_PATH="$stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
                "$package/bin/chopinfer" eval files --model "$source" "${variant[@]}" > "$stage/eval-files"
            while IFS= read -r file; do install -D -m 644 "$file" "$package/$file"; done < "$stage/eval-files"
            ;;
    esac
done < "$stage/staging-plan"
install -D -m 644 fixtures/GSM8K-LICENSE "$package/fixtures/GSM8K-LICENSE"
cp LICENSE "$package/"
python3 tools/release/notices.py "$package/THIRD_PARTY_NOTICES"
cp tools/release/INSTALL.md "$package/README.md"
printf 'version=%s\nrevision=%s\nrust=%s\ncuda=13.2\nnvcc=13.2.86\n' \
    "$version" "$CHOPINFER_BUILD_REVISION" "$toolchain" > "$package/BUILD_INFO"
for binary in chopinfer chopinfer-host; do
    [[ $(LD_LIBRARY_PATH="$stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$package/bin/$binary" --version) == "$binary $version ($CHOPINFER_BUILD_REVISION)" ]]
done
[[ $("$package/bin/chopinfer-top" --version) == "chopinfer-top $version" ]]
(cd "$package" && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS && sha256sum --check SHA256SUMS)
tar -C "$stage" -czf "$out/$name.tar.gz" "$name"
(cd "$out" && sha256sum "$name.tar.gz" > SHA256SUMS && sha256sum --check SHA256SUMS)
printf 'Artifact: %s\n' "$out/$name.tar.gz"

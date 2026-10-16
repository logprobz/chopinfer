#!/usr/bin/env bash
# Fixed upstream installer for the supported Linux x86_64 release target.
set -Eeuo pipefail
[[ $# == 1 && $(uname -sm) == 'Linux x86_64' ]] || exit 2
installer_dir=$(mktemp -d)
trap 'rm -rf "$installer_dir"' EXIT
installer="$installer_dir/rustup-init"
curl --proto '=https' --tlsv1.2 -fsS \
    https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init -o "$installer"
printf '%s  %s\n' 20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c "$installer" | sha256sum --check
chmod 700 "$installer"
"$installer" -y --profile minimal --default-toolchain "$1" --component clippy,rustfmt

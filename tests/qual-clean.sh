#!/usr/bin/env bash
set -Eeuo pipefail
# Git below reads this checkout only: not a repository or index inherited
# through GIT_* (`git rebase -x` exports GIT_DIR), nor the caller's global and
# system configuration.
for name in $(compgen -e -X '!GIT_*'); do unset "$name"; done
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$root"
clean_home=$(mktemp -d "${TMPDIR:-/tmp}/chopinfer-qual-clean.XXXXXX")
trap 'rm -rf -- "$clean_home"' EXIT
# Copy tracked sources so every loader sees a checkout without local.env.
# Keep HEAD available for tests that inspect Git.
source_copy="$clean_home/src"
git clone --quiet --shared --no-checkout "$root" "$source_copy"
git -C "$source_copy" read-tree HEAD
git ls-files -z | tar --null -T - -c | tar -x -C "$source_copy"
[[ ! -e $source_copy/local.env ]]
# Build scripts can embed source paths, so keep their artifacts with this copy.
build_target="$clean_home/target"
cd "$source_copy"
# Only the repository's CI Rust profile reaches the builds below.
mapfile -t rust_profile < <(grep -E '^[A-Z_]+=' ci/rust-profile.env)

# Each pass runs one process per module (ci/qualification.py) without a gate
# config; --receipts inherit keeps writes under $CHOPINFER_DATA visible to the
# missing-data check below.
run_pass() {
    local label=$1 data=$2
    shift 2
    env -i PATH="$PATH" HOME="$clean_home" XDG_CONFIG_HOME="$clean_home/.config" \
        TMPDIR="${TMPDIR:-/tmp}" CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}" \
        RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}" CARGO_NET_OFFLINE=true CARGO_INCREMENTAL=0 \
        CARGO_TARGET_DIR="$build_target" PYTHONPATH="$source_copy/tools/qual" \
        CHOPINFER_LOCAL_ENV="$clean_home/missing.env" CHOPINFER_DATA="$data" "${rust_profile[@]}" \
        python3 -B ci/qualification.py --no-config --receipts inherit --jobs "${CHOPINFER_CI_JOBS:-8}" \
        --log "$clean_home/$label.log" "$@"
    awk -v label="$label" '
        / \.\.\. skipped / {
            reason = $0
            sub(/^.* \.\.\. skipped /, "", reason)
            counts[reason]++
        }
        END {
            for (reason in counts)
                printf "qual-clean %s skips: %d %s\n", label, counts[reason], reason
        }
    ' "$clean_home/$label.log"
    [[ ! -e $clean_home/missing-data ]]
    git ls-files --others -z > "$clean_home/untracked"
    if [[ -s $clean_home/untracked ]]; then
        printf 'qual-clean %s: unexpected checkout files\n' "$label" >&2
        git ls-files --others >&2
        return 1
    fi
}

# Build the scorer in this checkout and target, just as the validation catalog does.
env -i PATH="$PATH" HOME="$clean_home" XDG_CONFIG_HOME="$clean_home/.config" \
    TMPDIR="${TMPDIR:-/tmp}" CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}" \
    RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}" CARGO_NET_OFFLINE=true CARGO_INCREMENTAL=0 \
    CARGO_TARGET_DIR="$build_target" CHOPINFER_LOCAL_ENV="$clean_home/missing.env" "${rust_profile[@]}" \
    cargo build -p runtime-host --bin chopinfer --locked --offline

run_pass empty "" --module test_bench --module test_fill_batch --module test_needle \
    --module test_logprobs_parity --module test_prefix_cache --module test_release_gate \
    --module test_step_overlap_ab --module test_review_config
run_pass missing "$clean_home/missing-data"

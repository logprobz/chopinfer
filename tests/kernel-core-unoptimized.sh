#!/usr/bin/env bash
# kernel-core's NaN-order test at opt-level 0. The CI profile
# (ci/rust-profile.env) optimizes kernel-core like every other package, so the
# workspace tests build without this test (crates/kernel-core/build.rs). This
# builds crates/kernel-core/tests/rotary.rs unoptimized in a target of its own
# below the shared one, kept warm between runs, and requires the test to pass.
set -Eeuo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$root"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$root/target}/kernel-core-unoptimized"
export CARGO_PROFILE_DEV_OPT_LEVEL=0 CARGO_PROFILE_TEST_OPT_LEVEL=0
test=legacy_f64_orders_keep_their_nan_operand
log=$(cargo test -p kernel-core --test rotary --locked --offline -- --exact "$test" 2>&1) || {
    printf '%s\n' "$log"
    exit 1
}
printf '%s\n' "$log"
if ! grep -qx "test $test ... ok" <<<"$log"; then
    echo "kernel-core-unoptimized: $test did not run at opt-level 0" >&2
    exit 1
fi

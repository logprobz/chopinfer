#!/usr/bin/env bash
set -Eeuo pipefail
# Git below reads this checkout only: not a repository or index inherited
# through GIT_* (`git rebase -x` exports GIT_DIR), nor the caller's global and
# system configuration.
for name in $(compgen -e -X '!GIT_*'); do unset "$name"; done
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
runtime_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
repository_root=$(git -C "${runtime_root}" rev-parse --show-toplevel)
[[ "${runtime_root}" == "${repository_root}" ]]
test -f "${runtime_root}/README.md"
cd "${runtime_root}"
cargo metadata --locked --offline --no-deps --format-version 1 >/dev/null
python3 -B tests/sdk_test_support.py
test -f docs/README.md
for doc in docs/*.md; do
    [[ $doc == docs/README.md ]] || grep -qF "(${doc#docs/})" docs/README.md \
        || { echo "docs/README.md does not list $doc" >&2; exit 1; }
done

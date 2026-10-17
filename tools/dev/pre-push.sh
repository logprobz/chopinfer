#!/usr/bin/env bash
# Git pre-push check. Every push to the public repository publishes, so this
# refuses a pushed ref whose history does not start at origin/main's root
# commit, the deletion of main or a tag, and anything tests/leak_guard.py
# rejects in what the push sends: each file version and commit message not
# already on origin, and each annotated tag.
#
# Install it from a checkout's .git/hooks/pre-push (docs/contributing.md):
#   #!/bin/sh
#   exec bash "$(git rev-parse --show-toplevel)/tools/dev/pre-push.sh" "$@"
set -euo pipefail

top=$(git rev-parse --show-toplevel)
guard=(python3 -B "$top/tests/leak_guard.py")
public=refs/remotes/origin/main
roots=$(git rev-list --max-parents=0 "$public" 2>/dev/null || true)
if [ -z "$roots" ]; then
    echo "pre-push: $public is missing; run git fetch origin first" >&2
    exit 1
fi
# What origin already has was checked when it was pushed.
bases=()
while read -r ref; do
    bases+=(--base "$ref")
done < <(git for-each-ref --format='%(refname)' refs/remotes/origin/)

status=0
refuse() {
    echo "pre-push: $*" >&2
    status=1
}

while read -r local_ref local_sha remote_ref remote_sha; do
    if [[ $local_sha =~ ^0+$ ]]; then
        case $remote_ref in
            refs/heads/main | refs/tags/*) refuse "refusing to delete $remote_ref" ;;
        esac
        continue
    fi
    commit=$local_sha
    kind=$(git cat-file -t "$local_sha")
    if [ "$kind" = tag ]; then
        "${guard[@]}" --tag "$local_sha" </dev/null || refuse "$remote_ref: the tag failed the leak guard"
        if ! commit=$(git rev-parse --verify --quiet "$local_sha^{commit}"); then
            refuse "$remote_ref does not peel to a commit"
            continue
        fi
    elif [ "$kind" != commit ]; then
        refuse "$remote_ref names a $kind, not a commit"
        continue
    fi
    if [ "$(git rev-list --max-parents=0 "$commit")" != "$roots" ]; then
        refuse "$local_ref has history that does not start at origin/main's root; rebase it onto origin/main"
        continue
    fi
    since=("${bases[@]}")
    if [[ ! $remote_sha =~ ^0+$ ]] && git cat-file -e "$remote_sha^{commit}" 2>/dev/null; then
        since+=(--base "$remote_sha")
    fi
    "${guard[@]}" --rev "$commit" --messages "$commit" "${since[@]}" </dev/null ||
        refuse "$local_ref failed the leak guard"
done
exit "$status"

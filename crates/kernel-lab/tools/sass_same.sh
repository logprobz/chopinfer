#!/usr/bin/env bash
# Compare the SASS of named entries between two cubins, for changes that
# must leave existing entries untouched (for example new entries added to a
# cubin). Usage: sass_same.sh BASE.cubin NEW.cubin [SYMBOL|OLD=NEW...]; with
# no symbols, every chop_* entry of BASE. OLD=NEW compares entry OLD of BASE
# with entry NEW of NEW, for an entry renamed with an unchanged body: the
# name in each `Function :` header is normalized. Exits nonzero if either
# cubin is unreadable, if there is no entry to compare, or if any entry
# differs or has no function body in either cubin. test_sass_same.sh checks
# this.
set -euo pipefail
(($# >= 2)) || { echo "usage: sass_same.sh BASE.cubin NEW.cubin [SYMBOL|OLD=NEW...]" >&2; exit 2; }
base=$1 new=$2
shift 2
for cubin in "$base" "$new"; do
  [[ -f $cubin ]] || { echo "no cubin $cubin" >&2; exit 2; }
done
if (($# == 0)); then
  listing=$(cuobjdump -symbols "$base") || { echo "cannot list the entries of $base" >&2; exit 2; }
  mapfile -t symbols < <(grep -o 'chop_[A-Za-z0-9_]*' <<< "$listing" | sort -u)
else
  symbols=("$@")
fi
((${#symbols[@]} > 0)) || { echo "no chop_* entry in $base" >&2; exit 2; }
# The function's SASS without addresses and blank lines, from its
# `Function :` header on, with the header's name replaced by `ENTRY`;
# empty when the cubin has no body for it (cuobjdump then still prints the
# file's headers and exits 0).
sass() {
  cuobjdump -sass -fun "$1" "$2" 2>/dev/null |
    sed -e 's#/\*[0-9a-f]*\*/##g' -e '/^[[:space:]]*$/d' |
    awk -v name="$1" '$0 ~ "Function : " name "$" {inside = 1; sub("Function : " name "$", "Function : ENTRY")}
      inside {text = text $0 "\n"; if (/;/) body = 1}
      END {if (body) printf "%s", text}' || true
}
status=0
for symbol in "${symbols[@]}"; do
  old=${symbol%%=*} renamed=${symbol#*=}
  a=$(sass "$old" "$base")
  b=$(sass "$renamed" "$new")
  if [[ -z $a ]]; then
    echo "missing $old in $base"
    status=1
  elif [[ -z $b ]]; then
    echo "missing $renamed in $new"
    status=1
  elif [[ $a == "$b" ]]; then
    echo "same $symbol"
  else
    echo "differs $symbol"
    status=1
  fi
done
exit "$status"

#!/usr/bin/env bash
# CPU checks of sass_same.sh on two small cubins built here: identical and
# changed entries, renamed entries (OLD=NEW), a missing cubin, a file that is
# no cubin, and a symbol that neither cubin holds. Needs nvcc and cuobjdump (PATH or
# /usr/local/cuda/bin).
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PATH=$PATH:/usr/local/cuda/bin
work=$(mktemp -d "${TMPDIR:-/tmp}/sass-same.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
cat > "$work/a.cu" << 'CU'
extern "C" __global__ void chop_same_v1(int* p) { p[threadIdx.x] = 1; }
extern "C" __global__ void chop_changed_v1(int* p) { p[threadIdx.x] = 2; }
CU
sed 's/= 2;/= 3 * p[0];/' "$work/a.cu" > "$work/b.cu"
sed 's/chop_same_v1/chop_renamed_v1/' "$work/a.cu" > "$work/c.cu"
for name in a b c; do
  nvcc -cubin -arch=sm_120 -o "$work/$name.cubin" "$work/$name.cu"
done
echo "not a cubin" > "$work/text.cubin"
failures=0
# expect STATUS OUTPUT-PATTERN ARGS...: sass_same.sh ARGS exits STATUS
# (0, or nonzero for "fail") and prints a line matching OUTPUT-PATTERN.
expect() {
  local want=$1 pattern=$2 got=0 output
  shift 2
  output=$(bash "$here/sass_same.sh" "$@" 2>&1) || got=$?
  local ok=1
  if [[ $want == fail ]]; then ((got != 0)) || ok=0; else ((got == 0)) || ok=0; fi
  grep -q -- "$pattern" <<< "$output" || ok=0
  if ((ok == 0)); then
    echo "FAIL sass_same.sh $* -> exit $got, expected $want and /$pattern/:"
    sed 's/^/  /' <<< "$output"
    failures=$((failures + 1))
  fi
}
a=$work/a.cubin b=$work/b.cubin c=$work/c.cubin
expect 0 "same chop_same_v1" "$a" "$b" chop_same_v1
expect fail "differs chop_changed_v1" "$a" "$b" chop_same_v1 chop_changed_v1
expect fail "differs chop_changed_v1" "$a" "$b"
expect 0 "same chop_changed_v1" "$a" "$a"
expect fail "missing chop_absent_v1" "$a" "$a" chop_absent_v1
expect fail "no cubin" "$work/absent.cubin" "$a"
expect fail "no cubin" "$a" "$work/absent.cubin" chop_same_v1
expect fail "cannot list\|no chop_" "$work/text.cubin" "$a"
expect fail "missing chop_same_v1" "$a" "$work/text.cubin" chop_same_v1
expect 0 "same chop_same_v1=chop_renamed_v1" "$a" "$c" chop_same_v1=chop_renamed_v1 chop_changed_v1
expect fail "differs chop_changed_v1=chop_renamed_v1" "$a" "$c" chop_changed_v1=chop_renamed_v1
expect fail "missing chop_renamed_v1 in" "$a" "$a" chop_same_v1=chop_renamed_v1
expect fail "missing chop_renamed_v1 in" "$a" "$c" chop_renamed_v1=chop_same_v1
expect fail "missing chop_same_v1 in" "$a" "$c" chop_same_v1
((failures == 0)) || exit 1
echo "sass_same.sh: all checks passed"

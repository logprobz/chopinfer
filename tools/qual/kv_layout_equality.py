#!/usr/bin/env python3
"""Served-bits equality of two KV storage layouts of the same values.

A model whose comparison KV layout stores the same values as its served one
(BF16 pages holding exactly what the native formats decode to) must serve
the same bits from both. The correctness runner records the determinism
request set (tools/qual/determinism.py: C1 greedy, 48 output tokens with
top-5 logprobs, a long prompt over two prefill chunks, token-ID sends and
prefix hits) on two servers of one configuration that differ only in
`--kv-cache-dtype`: the configuration's own server (`spec`, serving its
comparison layout) and a `kv-native` server, which serves `auto`, the
package's preferred layout. `compare` requires every request's text, finish
reason, token IDs, tokens and selected and top logprobs to be bit-identical
across the two records, and names the first differing request, field and
position.

Both records must come from one run and one build: the same checkpoint,
metadata, source commit, bundle, version and served model, and the same
effective profile in every field but `kv_cache_dtype`, which must differ.
The package identity may differ with the KV layout; both are reported.

    python3 tools/qual/kv_layout_equality.py compare NATIVE_RECORD COMPARISON_RECORD
"""

import argparse
import json
from pathlib import Path
import sys

from common import receipt
import determinism
from prefix_exact import serving_identity

_RECORD = ["record", "--url", "{url}", "--model", "{name}", "--timeout", "120", "--run-id", "{run_id}",
           "--server-log", "{server_log}", "{image}"]
# The correctness runner's arms and records (tools/qual/README.md#kv-layout-equality):
# the kv-native server replaces the configuration's --kv-cache-dtype with
# auto; the configuration's own server records the comparison layout.
RUNNER_ARMS = {"kv-native": ["--kv-cache-dtype", "auto"]}
RUNNER_RECORDS = {"kv_layout_native": ("kv-native", "determinism.py", [*_RECORD, "--arm", "kv-native"]),
                  "kv_layout_comparison": ("spec", "determinism.py", [*_RECORD, "--arm", "kv-comparison"])}
RUNNER_CHECK_ARM = "offline"
# The serving identity fields that follow the KV layout.
LAYOUT_FIELDS = ("package_identity",)


def identities(native, comparison):
    """The two records' serving identities, refused unless they name one
    build and profile that differ only in KV storage."""
    left, right = serving_identity(native), serving_identity(comparison)
    errors = []
    if native.get("run_id") != comparison.get("run_id") or not native.get("run_id"):
        errors.append("the records come from different runs")
    for key in sorted((left.keys() | right.keys()) - {"profile", *LAYOUT_FIELDS}):
        if left.get(key) != right.get(key):
            errors.append(f"the records differ in serving identity {key}")
    profiles = left["profile"], right["profile"]
    for key in sorted((profiles[0].keys() | profiles[1].keys()) - {"kv_cache_dtype"}):
        if profiles[0].get(key) != profiles[1].get(key):
            errors.append(f"the records differ in profile {key}")
    dtypes = [profile["kv_cache_dtype"] for profile in profiles]
    if dtypes[0] == dtypes[1]:
        errors.append(f"both records serve KV dtype {dtypes[0]}")
    return {"native": left, "comparison": right, "kv_cache_dtypes": dtypes}, errors


def compare(native, comparison):
    """The check's result for two determinism records."""
    served, errors = identities(native, comparison)
    result = {"kv_cache_dtypes": served["kv_cache_dtypes"],
              "package_identities": [served[side].get("package_identity") for side in ("native", "comparison")],
              "requests": len(native.get("requests") or []),
              "tokens": sum(len(row.get("token_ids") or []) for row in native.get("requests") or [])}
    if not errors:
        metrics = determinism.compare(native, comparison)
        result |= {"equal": metrics["equal"], "first_difference": metrics.get("first_difference"),
                   "max_abs_difference": metrics.get("max_abs_difference")}
        if not metrics["passed"]:
            errors.append(determinism.describe(metrics))
    return dict(result, passed=not errors, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("compare",))
    parser.add_argument("native", type=Path, help="the record of the package's preferred KV layout")
    parser.add_argument("comparison", type=Path, help="the record of the comparison KV layout")
    args = parser.parse_args()
    records = []
    try:
        records = [json.loads(path.read_text()) for path in (args.native, args.comparison)]
        result = compare(*records)
    except (OSError, ValueError, KeyError, SystemExit) as error:
        result = {"passed": False, "errors": [str(error)]}
    servers = {side: (value.get("server") if isinstance(value, dict) else None)
               for side, value in zip(("native", "comparison"), records)}
    # Compares saved records. No timed work, so no host load samples: the
    # receipt's timing reads informational.
    receipt("kv-layout-equality", {"records": servers}, dict(result, native=str(args.native),
                                                             comparison=str(args.comparison)))
    print(json.dumps({key: result.get(key) for key in ("passed", "errors", "first_difference")}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

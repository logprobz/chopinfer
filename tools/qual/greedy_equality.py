#!/usr/bin/env python3
"""Greedy token-for-token equality of an autoregressive and a speculative run.

The verify-drift gate (verify_exact.py) compares verify and AR rows with the
exact reference and so tolerates two paths that pick different wrong tokens
at a position. This check does not: `chopinfer eval capture --kind
greedy_pin` records every reference text's prompt prefix unforced and greedy
with the emitted token IDs; `compare` fails at the first request and position
where two records differ, and names it, so its logits and state can be
compared next.

    python3 tools/qual/greedy_equality.py compare AR_RECEIPT SPEC_RECEIPT
"""

import argparse
import json
from pathlib import Path
import sys


def compare(args):
    ar, other = (json.loads(Path(path).read_text()) for path in (args.ar, args.other))
    import kinds
    result = kinds.rescore(kinds.KINDS["greedy_pin"], other, ar, {"rule": "exact"})
    metrics = result.get("metrics") or {}
    differences = metrics.get("first_differences", [])
    print(json.dumps({"ar": args.ar, "other": args.other, "requests": len(ar["requests"]),
                      "ar_tokens": sum(len(ids) for ids in ar["requests"].values()),
                      **metrics, "first_difference": differences[0] if differences else None,
                      "status": result["status"], "error": result.get("error")}, indent=1))
    raise SystemExit(0 if result["status"] == "pass" else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compare",))
    parser.add_argument("ar", help="the AR receipt")
    parser.add_argument("other", help="the speculative receipt")
    compare(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())

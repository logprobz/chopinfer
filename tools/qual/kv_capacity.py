"""Record the serving KV capacity a running server measured at startup.

The pool's token capacity is read from `/metrics` (`chop_kv_capacity_tokens`
and `chop_kv_block_capacity`, counted across a context-parallel group) and
the serving shape from `/health` (profile, tensor- and context-parallel size,
KV dtype, sequences, speculation, served modalities: a package serving images
leaves less of the pool). With `--server-log` the startup line
`serving KV capacity: P pages, T tokens` (and, above size 1, the
`context parallel: N ranks per group, each holding P pages of S tokens`
line) must agree with the gauges and the health: the same tokens and pages,
a group line exactly when the health reports a group, of that size, and
tokens equal to pages times page tokens times the group size.
`--page-tokens S` also checks the block size against the package's pages.
With
`--planned TOKENS` the receipt records the measured capacity against a
planner's estimate; the estimate is context, not a pass condition.

    python3 -B tools/qual/kv_capacity.py --url URL [--server-log LOG] [--page-tokens S] [--planned TOKENS] [--label NAME]
"""

import re

from common import Client, arguments, receipt
from spec_metrics import metrics_text

CAPACITY_LINE = re.compile(r"serving KV capacity: (\d+) pages, (\d+) tokens")
GROUP_LINE = re.compile(r"context parallel: (\d+) ranks per group, each holding (\d+) pages of (\d+) tokens")


def gauges(text):
    """The unlabelled samples of a Prometheus scrape, by name."""
    values = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.partition(" ")
        if "{" not in name:
            try:
                values[name] = float(value.split()[0])
            except (ValueError, IndexError):
                continue
    return values


def startup_lines(text):
    """The last startup's capacity line and context-parallel line of a log."""
    capacity = group = None
    for line in text.splitlines():
        if match := CAPACITY_LINE.search(line):
            capacity = {"pages": int(match[1]), "tokens": int(match[2])}
            group = None
        elif match := GROUP_LINE.search(line):
            group = {"size": int(match[1]), "pages_per_rank": int(match[2]), "page_tokens": int(match[3])}
    return capacity, group


def geometry_errors(tokens, blocks, size, page_tokens=None):
    """Contradictions between the gauge's token and block counts, the
    context-parallel size and, when known, the package's page tokens: a
    logical block is one page on each of `size` ranks."""
    if not tokens or not blocks:
        return []
    if tokens % blocks:
        return [f"{tokens} capacity tokens are not a whole number of {blocks} blocks"]
    block_tokens = tokens // blocks
    if block_tokens % size:
        return [f"{block_tokens}-token blocks do not split into pages on {size} ranks"]
    if page_tokens is not None and block_tokens != page_tokens * size:
        return [f"{block_tokens}-token blocks are not {page_tokens}-token pages on {size} ranks"]
    return []


def log_errors(capacity, group, tokens, blocks, size):
    """Contradictions between the startup log's lines and the gauges and
    health: the same pages and tokens, and a group line exactly when the
    health reports a size above 1, naming that size and those pages."""
    if capacity is None:
        return ["the server log has no serving KV capacity line"]
    errors = []
    if tokens and capacity["tokens"] != int(tokens):
        errors.append(f"the log's {capacity['tokens']} tokens differ from the gauge's {int(tokens)}")
    if blocks and capacity["pages"] != int(blocks):
        errors.append(f"the log's {capacity['pages']} pages differ from the gauge's {int(blocks)} blocks")
    if group is None:
        if size > 1:
            errors.append(f"the log names no context-parallel group of {size}")
        return errors
    if size == 1:
        errors.append(f"the log names a context-parallel group of {group['size']}, /health reports none")
    elif group["size"] != size:
        errors.append(f"the log's group of {group['size']} differs from /health's {size}")
    if group["pages_per_rank"] != capacity["pages"]:
        errors.append(f"the log's {group['pages_per_rank']} pages per rank differ from its {capacity['pages']} pages")
    if capacity["pages"] * group["page_tokens"] * group["size"] != capacity["tokens"]:
        errors.append(f"{capacity['pages']} pages of {group['page_tokens']} tokens on {group['size']} ranks "
                      f"are not the log's {capacity['tokens']} tokens")
    return errors


def measure(metrics, health, log=None, planned=None, page_tokens=None):
    """The capacity record and its errors."""
    values = gauges(metrics)
    errors = []
    tokens = values.get("chop_kv_capacity_tokens")
    blocks = values.get("chop_kv_block_capacity")
    if not tokens or not blocks:
        errors.append("the scrape has no chop_kv_capacity_tokens or chop_kv_block_capacity")
    profile = ((health or {}).get("serving") or {}).get("profile") or {}
    if not profile:
        errors.append("/health reports no serving profile")
    size = profile.get("context_parallel_size") or 1
    tokens = int(tokens) if tokens else None
    blocks = int(blocks) if blocks else None
    errors += geometry_errors(tokens, blocks, size, page_tokens)
    fp8 = values.get("chop_fp8_kv_capacity_tokens")
    if fp8 is not None and tokens and int(fp8) != tokens:
        errors.append(f"the FP8 gauge's {int(fp8)} tokens differ from the pool's {tokens}")
    record = {
        "capacity_tokens": tokens,
        "block_capacity": blocks,
        "block_tokens": tokens // blocks if tokens and blocks else None,
        "fp8_capacity_tokens": int(fp8) if fp8 is not None else None,
        "profile": profile.get("name"),
        "tensor_parallel_size": profile.get("tensor_parallel_size"),
        "context_parallel_size": size,
        "kv_cache_dtype": profile.get("kv_cache_dtype"),
        "max_sequences": profile.get("max_sequences"),
        "speculation": profile.get("speculation"),
        "model_options": ((health or {}).get("serving") or {}).get("model_options"),
        "modalities": (health or {}).get("modalities"),
    }
    if log is not None:
        capacity, group = startup_lines(log)
        record["startup"] = {"capacity": capacity, "context_parallel": group}
        errors += log_errors(capacity, group, tokens, blocks, size)
    if planned is not None:
        record["planned_tokens"] = planned
        record["measured_over_planned"] = tokens / planned if tokens else None
    return record, errors


def main():
    parser = arguments(__doc__)
    parser.add_argument("--server-log", help="The server's stdout, to check its startup capacity line")
    parser.add_argument("--planned", type=int, help="A planner's token estimate for the same configuration")
    parser.add_argument("--label", help="A name for the measured configuration, kept in the receipt")
    parser.add_argument("--page-tokens", type=int, help="The package's tokens per KV page, to check the block size")
    args = parser.parse_args()
    if args.planned is not None and args.planned <= 0:
        parser.error("--planned must be positive")
    if args.page_tokens is not None and args.page_tokens <= 0:
        parser.error("--page-tokens must be positive")
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    log = None
    if args.server_log:
        with open(args.server_log, errors="replace") as handle:
            log = handle.read()
    record, errors = measure(metrics_text(client), server.get("health"), log, args.planned, args.page_tokens)
    record["label"] = args.label
    # No timed work, so no host load samples: the receipt's timing reads informational.
    receipt("kv-capacity", server, {"passed": not errors, "errors": errors, **record}, args.mechanics_only)
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()

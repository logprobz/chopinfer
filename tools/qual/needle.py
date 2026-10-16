"""Run a model's frozen token-ID needle prompts through streaming completions.

The gate config (CHOPINFER_QUAL_CONFIG) names the prompts and the answer
rule. `inputs.needle` is the prompt source and `qualification.needle`
describes it:

- `format` `context-files` (the default): a directory with `manifest.json`
  and one `context-{length}.json` per context, as MiMo's functional
  fixtures;
- `format` `prompt-set`: a frozen prompt set whose `needles` rows hold
  `token_ids`, `max_tokens`, `total_context` and `expected`, authenticated
  against the SHA256 of the file and of each row's token IDs in the repo's
  `frozen` digest file (Qwen3.8's `qwen38-refs/prompts/prompts.json` and
  `fixtures/qwen38/prompts.json`).

`lengths` are the contexts it may run (all by default), and `match` is the
answer rule: `whitespace` (substring after removing whitespace, or literal
with --strict) or `alphanumeric` (substring of the upper-case letters and
digits, ignoring spaces and punctuation).
"""

import hashlib
import json
import time

from common import input_path, Client, arguments, qualification_config, receipt, response_text, ROOT


FIXTURE = None
LENGTHS = (8192, 131072, 524288, 1048576)
DEFAULT_SPEC = {"format": "context-files", "lengths": list(LENGTHS), "match": "whitespace"}


def spec():
    """The needle declaration of the selected gate config."""
    if FIXTURE is not None:
        return DEFAULT_SPEC
    return {**DEFAULT_SPEC, **qualification_config()["qualification"].get("needle", {})}


def answer_matches(expected, output, strict=False, match="whitespace"):
    """Keep substring matching under the declared rule."""
    if match == "alphanumeric":
        def normalized(text):
            return "".join(ch for ch in text.upper() if ch.isalnum())
        return normalized(expected) in normalized(output)
    if match != "whitespace":
        raise ValueError(f"unknown needle answer rule {match}")
    if not strict:
        expected = "".join(expected.split())
        output = "".join(output.split())
    return expected in output


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def context_files(root, length):
    """(prompt, max_tokens, expected) of a context-files fixture."""
    from refs import fixture_bytes
    manifest = json.loads(fixture_bytes(root / "manifest.json"))
    expected_by_length = {item["total_context"]: item["expected"] for item in manifest["contexts"]}
    frozen = json.loads(fixture_bytes(root / f"context-{length}.json"))
    if len(frozen["prompt"]) + frozen["max_tokens"] != frozen["total_context"]:
        raise ValueError("frozen context length is inconsistent")
    if expected_by_length[length] != frozen["expected"]:
        raise ValueError("manifest expected string differs from frozen prompt")
    return frozen["prompt"], frozen["max_tokens"], frozen["expected"]


def prompt_set(path, frozen_path):
    """The authenticated needle rows of a frozen prompt set, by context."""
    from refs import fixture_bytes
    blob = fixture_bytes(path)
    frozen = json.loads(fixture_bytes(frozen_path))
    if sha256(blob) != frozen["full_file"]["sha256"]:
        raise ValueError(f"{path} differs from {frozen_path}")
    digests = {row["name"]: row["token_ids_sha256"] for row in frozen["needles"]}
    rows = {}
    for row in json.loads(blob)["needles"]:
        if sha256(json.dumps(row["token_ids"]).encode()) != digests.get(row["name"]):
            raise ValueError(f"needle {row['name']} token IDs differ from the frozen digest")
        if len(row["token_ids"]) + row["max_tokens"] != row["total_context"]:
            raise ValueError(f"needle {row['name']} context length is inconsistent")
        rows[row["total_context"]] = (row["token_ids"], row["max_tokens"], row["expected"])
    return rows


def loader(declared):
    """A function from a context length to (prompt, max_tokens, expected)."""
    root = FIXTURE or input_path("needle")
    if declared["format"] == "context-files":
        return lambda length: context_files(root, length)
    if declared["format"] == "prompt-set":
        rows = prompt_set(root, ROOT / declared["frozen"])
        return lambda length: rows[length]
    raise ValueError(f"unknown needle format {declared['format']}")


def main():
    parser = arguments(__doc__, model_facts=True)
    parser.add_argument("--lengths", type=lambda value: tuple(int(n) for n in value.split(",")),
                        help="comma-separated contexts to run (default: every context the gate declares)")
    parser.add_argument("--strict", action="store_true",
                        help="with the whitespace rule, compare answers as literal substrings")
    args = parser.parse_args()
    declared = spec()
    lengths = args.lengths or tuple(declared["lengths"])
    if not set(lengths) <= set(declared["lengths"]):
        parser.error(f"--lengths must be among {tuple(declared['lengths'])}")
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    rows = []
    for length in lengths:
        try:
            prompt, budget, expected = loader(declared)(length)
            prompt = [1] * 16 if args.mechanics_only else prompt
            budget = 8 if args.mechanics_only else budget
            started = time.monotonic()
            events, latency = client.request("/v1/completions", {
                "model": server["model"], "prompt": prompt, "max_tokens": budget,
                "temperature": 0, "stream": True}, stream=True)
            first = next((at for at, event in events if response_text([(at, event)])), None)
            output = response_text(events)
            rows.append({"context": length, "prompt_tokens": len(prompt), "output": output,
                         "latency_seconds": latency,
                         "ttft_seconds": first - started if first is not None else None,
                         "passed": bool(events) and (args.mechanics_only or answer_matches(
                             expected, output, strict=args.strict, match=declared["match"]))})
        except Exception as error:
            rows.append({"context": length, "passed": False, "error": str(error)})
    passed = all(row["passed"] for row in rows)
    receipt("needle", server, {"passed": passed, "skipped": ["needle answer"] if args.mechanics_only else [],
                              "rule": declared["match"],
                              "errors": [row for row in rows if not row["passed"]], "rows": rows}, args.mechanics_only)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

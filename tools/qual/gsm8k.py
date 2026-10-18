"""Score the frozen 200-item GSM8K screen.

The screen passes with at least the model's declared `preview.gsm8k_floor`
items correct (CHOPINFER_QUAL_CONFIG); the receipt records the floor. A model
that declares no preview, screened at bring-up before a floor is adopted,
names the floor with --floor, which the receipt records as explicit. A
scored screen binds the server to the gate's checkpoint pin (and its
startup log, when given), confirms every response came from that process,
and records the topology and speculation it served with.

`gsm8k.py rescore --output DIR RECEIPT[=RESPONSES]...` re-reads the responses retained
with --response-dir (default: RESPONSES is `responses` beside the receipt) under the
current answer rule and writes rescored copies under DIR; the receipts are not modified.
A receipt that records no floor is judged at the declared floor of the model whose served
name it records, which the copy then records; one of an unknown name, or of another model than
CHOPINFER_QUAL_CONFIG selects, is refused.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import os
import re
import sys

from common import (template_control, input_path, Client, ROOT, arguments, bind_checkpoint, confirm_live, qualification_config, receipt,
                    serving_profile)
import declaration


FIXTURE = None


MARKED = re.compile(r"####\s*(?:\\?\$\s*)?(-?[\d,]+(?:\.\d+)?)")
BOXED = re.compile(r"\\boxed\s*\{")
# A boxed answer is a signed plain number, optionally after `$` or `\$`. Anything else in the
# box (percent, units, fractions, words) leaves it unparsed: such forms can change the value.
BOXED_ANSWER = re.compile(r"\s*(?:\\?\$\s*)?(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*")


def number(text):
    try:
        return str(Decimal(text.replace(",", "")).normalize())
    except InvalidOperation:
        return None


def boxed(text):
    """The plain number in the last \\boxed{...}, or None."""
    starts = list(BOXED.finditer(text))
    if not starts:
        return None
    depth, start = 1, starts[-1].end()
    for end in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            break
    else:
        return None
    match = BOXED_ANSWER.fullmatch(text[start:end].replace("{,}", ","))
    return number(match[1]) if match else None


def parse(text):
    """The final answer and the rule that read it: the last `#### NUMBER`, else the last \\boxed{}."""
    text = text or ""
    if marked := MARKED.findall(text):
        value = number(marked[-1])
        return value, "####" if value is not None else None
    value = boxed(text)
    return value, "boxed" if value is not None else None


def extract(text):
    return parse(text)[0]


def answer_rules(rows):
    """How many answered rows each rule read; `unparsed` counts rows neither rule read."""
    counts = {"####": 0, "boxed": 0, "unparsed": 0}
    for row in rows:
        if "error" not in row:
            counts[row.get("rule") or "unparsed"] += 1
    return counts


def screen_floor(explicit=None):
    """The floor of the selected declaration: its declared
    `preview.gsm8k_floor` (a model's, or a checkpoint variant's own in its
    qual.VARIANT.json), or `explicit` (--floor) for a declaration without
    one. A declared floor cannot be replaced."""
    config = qualification_config()
    model = config["model"]
    if config.get("variant") is not None:
        # The selected file is the variant's declaration (declaration.load
        # requires its name to match), whose preview the gate config drops.
        model = f"{model}.{config['variant']}"
        value = json.loads(Path(os.environ["CHOPINFER_QUAL_CONFIG"]).read_text())
    else:
        value = declaration.previewed().get(model)
    declared = (value or {}).get("preview", {}).get("gsm8k_floor")
    if explicit is not None:
        if declared is not None:
            raise ValueError(f"{model} declares its GSM8K floor ({declared}); --floor is for a model without one")
        if not 1 <= explicit <= 200:
            raise ValueError("--floor must be between 1 and 200")
        return explicit
    if declared is None:
        raise ValueError(f"{model} declares no preview, so no GSM8K floor; name one with --floor")
    return declared


def receipt_floor(value):
    """The floor a receipt is judged at: the one it records, else the declared floor of the
    previewed model whose profiles name the served model the receipt records."""
    if "floor" in value:
        return value["floor"]
    name = (value.get("server") or {}).get("model")
    previewed = declaration.previewed()
    models = [model for model in previewed
              if json.loads((ROOT / "models" / model / "profiles.json").read_text())["served_model_name"] == name]
    if len(models) != 1:
        raise ValueError(f"the receipt records no floor, and served model {name!r} names no single "
                         "previewed model whose floor applies")
    if os.environ.get("CHOPINFER_QUAL_CONFIG") and qualification_config()["model"] != models[0]:
        raise ValueError(f"the receipt was served by {models[0]}, but CHOPINFER_QUAL_CONFIG selects "
                         f"{qualification_config()['model']}")
    return previewed[models[0]]["preview"]["gsm8k_floor"]


def screen_passed(rows, errors, mechanics_only, floor=None):
    """Whether the screen passed: every item answered and at least `floor`
    (default: the declared floor) correct."""
    if errors:
        return False
    if mechanics_only:
        return True
    floor = screen_floor() if floor is None else floor
    return len(rows) == 200 and sum(row["correct"] for row in rows) >= floor


def main():
    parser = arguments(__doc__, model_facts=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--response-dir", type=Path, help="Retain raw responses without changing request bodies")
    parser.add_argument("--floor", type=int,
                        help="items that must be correct, for a model that declares no preview floor")
    parser.add_argument("--server-log", type=Path,
                        help="The server's startup log; when given, it must be the live server's")
    args = parser.parse_args()
    floor = screen_floor(args.floor)
    from refs import fixture_bytes
    root = FIXTURE or input_path("gsm8k")
    manifest = json.loads(fixture_bytes(root / "manifest.json"))
    raw = fixture_bytes(root / "examples.json")
    examples = json.loads(raw)
    if (hashlib.sha256(raw).hexdigest() != manifest["examples_sha256"]
            or len(examples) != 200 or manifest["expected_count"] != 200):
        raise ValueError("frozen GSM8K examples changed")
    if args.concurrency < 1 or not 1 <= args.limit <= 200:
        parser.error("concurrency and limit must be positive")
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    serving = None
    if not args.mechanics_only:
        # Scores count only from the pinned checkpoint, served by one live process.
        try:
            bind_checkpoint(server, args.server_log, args.url)
            serving = serving_profile(server)
        except ValueError as error:
            raise SystemExit(f"refused: {error}")
    if args.response_dir:
        args.response_dir.mkdir(parents=True, exist_ok=False)
    selected = examples[:min(args.limit, 2) if args.mechanics_only else args.limit]

    def run(example):
        body = {"model": server["model"], "messages": [
            {"role": "system", "content": manifest["instruction"]},
            {"role": "user", "content": "1+1?" if args.mechanics_only else example["question"]}],
            "temperature": manifest["temperature"], "top_p": 1, "seed": manifest["seed"],
            "max_tokens": 1 if args.mechanics_only else manifest["max_tokens"],
            **template_control(manifest["chat_template_kwargs"]),
            # Retained responses carry their token IDs for gsm8k_branches.py.
            "return_token_ids": True}
        try:
            answer, latency = client.request("/v1/chat/completions", body)
            retained = {}
            if args.response_dir:
                # The row binds the retained bytes, so a later probe reads these answers.
                data = (json.dumps({"body": body, "response": answer, "latency_seconds": latency}, indent=2)
                        + "\n").encode()
                (args.response_dir / f"{example['index']}.json").write_bytes(data)
                retained["response_sha256"] = hashlib.sha256(data).hexdigest()
            choice = answer["choices"][0]
            actual, rule = parse(choice["message"].get("content"))
            return {"index": example["index"], "expected": example["expected"],
                    "actual": actual, "rule": rule, "finish_reason": choice["finish_reason"],
                    "latency_seconds": latency,
                    "correct": actual == example["expected"] and choice["finish_reason"] != "length", **retained}
        except Exception as error:
            return {"index": example["index"], "error": str(error), "correct": False}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(run, selected))
    errors = [row for row in rows if "error" in row]
    correct = sum(row["correct"] for row in rows)
    confirm_live(client, server)
    passed = screen_passed(rows, errors, args.mechanics_only, floor)
    # Its per-request seconds are diagnostics of an accuracy check, not timing evidence, so no host load
    # samples bound them and the receipt's timing reads informational.
    receipt("gsm8k", server, {"passed": passed, "serving": serving, "skipped": ["model quality"] if args.mechanics_only else [],
                              "correct": correct, "floor": floor,
                              "floor_source": "declared" if args.floor is None else "explicit",
                              "count": len(rows), "errors": errors,
                              "answer_rules": answer_rules(rows),
                              "sampling": {"temperature": manifest["temperature"], "top_p": 1, "seed": manifest["seed"]},
                              "examples_sha256": manifest["examples_sha256"], "rows": rows}, args.mechanics_only)
    raise SystemExit(0 if passed else 1)


def rescore_receipt(path, responses):
    """A copy of a receipt scored by parse(). Rows are re-read from retained responses; an answer
    the receipt already read must read the same. Without responses a row keeps its reading, which
    is final when it has an answer and unread (listed) for a stopped row without one."""
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get("check") != "gsm8k" or value.get("mechanics_only"):
        raise ValueError(f"{path}: not a scored GSM8K receipt")
    floor = receipt_floor(value)
    rows, unread = [], []
    for row in value["rows"]:
        row = dict(row)
        if "error" in row:
            rows.append(row)
            continue
        if responses:
            choice = json.loads((responses / f"{row['index']}.json").read_text())["response"]["choices"][0]
            actual, rule = parse(choice["message"].get("content"))
            if choice["finish_reason"] != row["finish_reason"] or row["actual"] not in (None, actual):
                raise ValueError(f"{path}: response {row['index']} does not match its receipt row")
            row.update(actual=actual, rule=rule)
        else:
            if "rule" not in row:  # receipts before the boxed rule read only `####`
                row["rule"] = "####" if row["actual"] is not None else None
            if row["actual"] is None and row["finish_reason"] == "stop":
                unread.append(row["index"])
        row["correct"] = row["actual"] == row["expected"] and row["finish_reason"] != "length"
        rows.append(row)
    before = {row["index"]: row["correct"] for row in value["rows"]}
    value["rescore"] = {"source": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
                        "responses": str(responses) if responses else None,
                        "previous_correct": value["correct"], "previous_passed": value["passed"],
                        "gained": sorted(row["index"] for row in rows if row["correct"] and not before[row["index"]]),
                        "lost": sorted(row["index"] for row in rows if before[row["index"]] and not row["correct"]),
                        "unread_indices": sorted(unread)}
    value.update(rows=rows, correct=sum(row["correct"] for row in rows), answer_rules=answer_rules(rows),
                 floor=floor, passed=screen_passed(rows, value["errors"], False, floor))
    return value


def rescore(argv):
    parser = argparse.ArgumentParser(prog="gsm8k.py rescore", description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for the rescored copies")
    parser.add_argument("--root", type=Path, help="Copies mirror receipt paths below this directory "
                        "(default: the receipts' common parent)")
    parser.add_argument("receipts", nargs="+", metavar="RECEIPT[=RESPONSES]")
    args = parser.parse_args(argv)
    pairs = []
    for item in args.receipts:
        path, _, responses = item.partition("=")
        path = Path(path).resolve()
        if responses and not Path(responses).is_dir():
            parser.error(f"{responses}: no such response directory for {path}")
        responses = Path(responses).resolve() if responses else path.parent / "responses"
        pairs.append((path, responses if responses.is_dir() else None))
    root = (args.root.resolve() if args.root
            else Path(os.path.commonpath([path.parent for path, _ in pairs])))
    print("| receipt | model | old | new | passed | #### | boxed | unparsed | gained | unread |")
    print("|---|---|---:|---:|---|---:|---:|---:|---|---|")
    failed = False
    for path, responses in pairs:
        try:
            value = rescore_receipt(path, responses)
            target = args.output / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x") as output:
                json.dump(value, output, indent=2, sort_keys=True)
                output.write("\n")
        except (OSError, ValueError, KeyError) as error:
            failed = True
            print(f"| {path} | error: {error} |||||||||")
            continue
        rules, note = value["answer_rules"], value["rescore"]
        print(f"| {target.relative_to(args.output)} | {value['server'].get('model')} | "
              f"{note['previous_correct']} | {value['correct']} | {value['passed']} | {rules['####']} | "
              f"{rules['boxed']} | {rules['unparsed']} | {note['gained']} | "
              f"{note['unread_indices'] if not responses else 'responses read'} |")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    rescore(sys.argv[2:]) if sys.argv[1:2] == ["rescore"] else main()

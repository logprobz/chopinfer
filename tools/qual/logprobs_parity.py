"""Capture and compare generated-token logprobs of two OpenAI-compatible servers.

`capture` sends a fixed request set to one endpoint and writes a receipt with
the raw non-streaming responses and streamed chunks. `compare` reads two such
receipts (for example vLLM with --reasoning-parser and Chopinfer) and reports,
case by case, which tokens each server lists in `logprobs`, on which stream
chunks they arrive, and how the top-logprob lists are ordered.

    python3 tools/qual/logprobs_parity.py capture --mechanics-only --arm vllm-ar --url http://127.0.0.1:8102
    python3 tools/qual/logprobs_parity.py capture --arm chopinfer --url http://127.0.0.1:8311
    python3 tools/qual/logprobs_parity.py compare A.json B.json
"""

import argparse
import json
import math
from pathlib import Path
import sys

from common import startup, answer_control, bind_checkpoint, confirm_live, Client, receipt


QUESTION = "What is 17 + 25? Reply with the number only."
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


def cases():
    """Each case runs non-streaming, streaming, and streaming with token IDs."""
    # Thinking off, or the frontend's default where it refuses that.
    no_thinking = answer_control()
    user = [{"role": "user", "content": QUESTION}]
    chat = {"messages": user, "max_tokens": 1024, "temperature": 0, "seed": 1,
            "logprobs": True, "top_logprobs": 3}
    return [
        ("chat-thinking", "chat", chat),
        ("chat-no-thinking", "chat", {**chat, **no_thinking}),
        ("chat-hidden-reasoning", "chat", {**chat, "include_reasoning": False}),
        ("chat-top-zero", "chat", {**chat, **no_thinking, "top_logprobs": 0}),
        ("chat-sampled", "chat", {**chat, **no_thinking, "temperature": 1.0, "seed": 7,
                                  "top_logprobs": 2}),
        ("chat-stop-string", "chat", {
            **chat, **no_thinking, "stop": [","],
            "messages": [{"role": "user", "content": "Count from 1 to 5, separated by commas."}]}),
        ("chat-tool-call", "chat", {
            **chat, **no_thinking, "tools": TOOLS, "tool_choice": "auto",
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}]}),
        ("completion", "completion", {"prompt": "1, 2, 3, 4,", "max_tokens": 8,
                                      "temperature": 0, "logprobs": 2}),
        ("completion-echo", "completion", {"prompt": "1, 2, 3, 4,", "max_tokens": 4,
                                           "temperature": 0, "logprobs": 2, "echo": True}),
        ("completion-sampled", "completion", {"prompt": "Once upon a time", "max_tokens": 8,
                                              "temperature": 1.0, "seed": 7, "logprobs": 2}),
    ]


def capture(args):
    client = Client(args.url, timeout=args.timeout)
    identity = client.identity(args.model)
    if not args.mechanics_only:
        bind_checkpoint(identity, args.server_log, args.url)
    results, errors = [], []
    for name, endpoint, body in cases():
        path = "/v1/chat/completions" if endpoint == "chat" else "/v1/completions"
        body = {**body, "model": identity["model"]}
        row = {"case": name, "endpoint": endpoint, "request": body}
        try:
            row["json"], _ = client.request(path, body)
            for mode, extra in (("stream", {}), ("stream_ids", {"return_token_ids": True})):
                events, _ = client.request(path, {**body, "stream": True, **extra}, stream=True)
                row[mode] = [event for _, event in events]
        except RuntimeError as error:
            errors.append(f"{name}: {error}")
        results.append(row)
    if not args.mechanics_only:
        confirm_live(client, identity)
    # No timed work, so no host load samples: the receipt's timing reads informational.
    receipt(f"logprobs-parity-{args.arm}", identity,
            {"arm": args.arm, "cases": results, "passed": not errors, "errors": errors,
             "responders": sorted(client.responders, key=str)}, args.mechanics_only)
    return 1 if errors else 0


def rows(logprobs, endpoint):
    """(label, logprob, top labels) per token, or None when logprobs are null."""
    if logprobs is None:
        return None
    if endpoint == "chat":
        return [(row["token"], row.get("logprob"), [top["token"] for top in row.get("top_logprobs") or []])
                for row in logprobs.get("content") or []]
    return [(label, value, list(top or {})) for label, value, top in
            zip(logprobs["tokens"], logprobs["token_logprobs"], logprobs["top_logprobs"])]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_logprobs(probs, endpoint, requested, allow_null):
    if not isinstance(probs, dict):
        return ["requested logprobs are missing"]
    errors = []
    if endpoint == "completion":
        arrays = [probs.get(key) for key in
                  ("tokens", "token_logprobs", "top_logprobs", "text_offset")]
        if any(not isinstance(array, list) for array in arrays) or len({len(a) for a in arrays}) != 1:
            return ["logprob parallel arrays differ"]
        entries = zip(arrays[0], arrays[1], arrays[2])
    else:
        content = probs.get("content")
        if not isinstance(content, list):
            return ["logprob content is missing"]
        entries = ((r.get("token"), r.get("logprob"), r.get("top_logprobs")) for r in content)
    for index, (label, score, top) in enumerate(entries):
        if not isinstance(label, str):
            errors.append("token label is missing")
        if allow_null and index == 0 and score is None:
            continue
        if not finite(score):
            errors.append("token score is missing or nonfinite")
        if endpoint == "completion":
            valid_top = isinstance(top, dict)
            labels = list(top) if valid_top else []
            scores = list(top.values()) if valid_top else []
        else:
            valid_top = isinstance(top, list)
            labels = [entry.get("token") for entry in top] if valid_top else []
            scores = [entry.get("logprob") for entry in top] if valid_top else []
        if (not valid_top or len(labels) < requested or len(set(labels)) != len(labels)
                or any(not isinstance(label, str) for label in labels)
                or any(not finite(score) for score in scores)):
            errors.append("top-logprob evidence is missing or invalid")
    return errors


def summarize(row):
    try:
        return summarize_valid(row)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        return {"error": str(error) or "malformed capture"}


def hidden_summary(row):
    """Validate the observable chat response when reasoning hides all scores."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def usage(value):
        require(isinstance(value, dict) and all(type(value.get(key)) is int and value[key] > 0
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')), 'missing token usage')
        require(value['total_tokens'] == value['prompt_tokens'] + value['completion_tokens'], 'invalid token usage')
        return {key: value[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}

    def choice(value):
        require(isinstance(value, dict) and isinstance(value.get('choices'), list)
                and len(value['choices']) == 1, 'missing or ambiguous chat choice')
        result = value['choices'][0]
        require(type(result.get('index')) is int and result['index'] == 0, 'invalid choice index')
        require('logprobs' in result and result['logprobs'] is None, 'hidden reasoning must not expose scores')
        return result

    require(row['request'].get('logprobs') is True, 'hidden case did not request logprobs')
    response = row['json']; selected = choice(response)
    require(response.get('object') == 'chat.completion', 'invalid chat response object')
    message = selected['message']
    require(message.get('role') == 'assistant' and isinstance(message.get('content'), str), 'missing assistant content')
    require(not message.get('reasoning') and not message.get('reasoning_content'), 'hidden reasoning was exposed')
    require(selected.get('finish_reason') in ('stop', 'length'), 'missing terminal finish reason')
    counts = usage(response['usage'])
    visible = {'text': message['content'], 'finish_reason': selected['finish_reason'], 'streams': {}}
    for mode in ('stream', 'stream_ids'):
        chunks = row.get(mode)
        require(isinstance(chunks, list) and bool(chunks), f'{mode}: capture is missing')
        content, prompt_ids, terminal, observed_usage = [], [], None, None
        for index, chunk in enumerate(chunks):
            require(chunk.get('object') == 'chat.completion.chunk', f'{mode}: invalid chunk object')
            if chunk.get('usage') is not None:
                observed_usage = usage(chunk['usage'])
                require(observed_usage == counts, f'{mode}: token usage differs')
            if chunk.get('choices') == []:
                require(terminal is not None and observed_usage is not None and index == len(chunks) - 1,
                        f'{mode}: empty nonterminal chunk')
                continue
            streamed = choice(chunk); delta = streamed['delta']
            require(isinstance(delta, dict) and terminal is None, f'{mode}: malformed or post-terminal delta')
            if index == 0:
                require(delta.get('role') == 'assistant', f'{mode}: missing assistant role')
            require(not delta.get('reasoning') and not delta.get('reasoning_content'), f'{mode}: reasoning exposed')
            if 'content' in delta:
                require(isinstance(delta['content'], str), f'{mode}: invalid content delta')
                content.append(delta['content'])
            if streamed.get('finish_reason') is not None:
                terminal = streamed['finish_reason']
            if 'prompt_token_ids' in chunk:
                ids = chunk['prompt_token_ids']
                require(isinstance(ids, list) and len(ids) == counts['prompt_tokens']
                        and all(type(token) is int and token >= 0 for token in ids), f'{mode}: invalid prompt IDs')
                prompt_ids.append(ids)
        require(terminal == selected['finish_reason'] and ''.join(content) == message['content'],
                f'{mode}: incomplete content or finish coverage')
        require(len(prompt_ids) == (1 if mode == 'stream_ids' else 0), f'{mode}: prompt ID coverage differs')
        visible['streams'][mode] = {'text': ''.join(content), 'finish_reason': terminal,
                                    'usage': observed_usage, 'prompt_token_ids': prompt_ids}
    return {'errors': [], **{key: counts[key] for key in ('prompt_tokens', 'completion_tokens')},
            'echo': False, 'labels': None, 'values': None, 'top': None, 'text': message['content'],
            'text_offset': None, 'stream': [], 'stream_ids': [], 'hidden': visible}


def summarize_valid(row):
    """What one server returned for one case, reduced to comparable facts."""
    endpoint = row["endpoint"]
    if "json" not in row:
        return {"error": True}
    choice = row["json"]["choices"][0]
    request = row.get("request", {})
    if endpoint == 'chat' and request.get('include_reasoning') is False:
        return hidden_summary(row)
    requested = request.get("top_logprobs", 0) if endpoint == "chat" else request.get("logprobs", 0)
    echo = bool(request.get("echo")) and endpoint == "completion"
    errors = validate_logprobs(choice.get("logprobs"), endpoint, requested, echo)
    listed = rows(choice.get("logprobs"), endpoint)
    summary = {
        "errors": errors,
        "completion_tokens": row["json"]["usage"]["completion_tokens"],
        "prompt_tokens": row["json"]["usage"].get("prompt_tokens", 0),
        "echo": row.get("request", {}).get("echo", False),
        "labels": None if listed is None else [label for label, _, _ in listed],
        "values": None if listed is None else [value for _, value, _ in listed],
        "top": None if listed is None else [top for _, _, top in listed],
        "text": choice.get("text") if endpoint == "completion" else choice["message"].get("content"),
        "text_offset": (choice.get("logprobs") or {}).get("text_offset"),
    }
    for mode in ("stream", "stream_ids"):
        chunks = []
        position = 0
        if not row.get(mode):
            errors.append(f"{mode}: capture is missing")
        for chunk in row.get(mode, []):
            for streamed in chunk.get("choices", []):
                listed = rows(streamed.get("logprobs"), endpoint)
                if listed is None:
                    continue
                errors.extend(f"{mode}: {error}" for error in validate_logprobs(
                    streamed.get("logprobs"), endpoint, requested, echo and position == 0))
                position += len(listed)
                delta = streamed.get("delta") or {}
                kind = ("finish" if streamed.get("finish_reason") else
                        "tool" if delta.get("tool_calls") else
                        "reasoning" if delta.get("reasoning") or delta.get("reasoning_content") else
                        "content" if delta.get("content") or streamed.get("text") else "empty")
                chunks.append({"kind": kind, "labels": [label for label, _, _ in listed],
                               "top": [top for _, _, top in listed],
                               "text_offset": (streamed.get("logprobs") or {}).get("text_offset")})
        if not position:
            errors.append(f"{mode}: no scored rows")
        summary[mode] = chunks
    return summary


def compare_case(first, second):
    """Differences between two servers' summaries of one case."""
    if first.get("error") or second.get("error"):
        return ["a request failed"]
    notes = [f"{name}: {error}" for name, value in (("first", first), ("second", second))
             for error in value.get("errors", [])]
    if (first["echo"], first["prompt_tokens"], first["completion_tokens"]) != (
            second["echo"], second["prompt_tokens"], second["completion_tokens"]):
        notes.append("prompt boundary or token usage differs")
    for name, value in (("first", first), ("second", second)):
        expected = value["completion_tokens"] + (value["prompt_tokens"] if value["echo"] else 0)
        if value["labels"] is not None and len(value["labels"]) != expected:
            notes.append(f"{name}: {len(value['labels'])} logprob rows for "
                         f"{expected} {'prompt and completion' if value['echo'] else 'completion'} tokens")
    if (first["labels"] is None) != (second["labels"] is None):
        notes.append("logprobs are null in one server only")
        return notes
    if first["labels"] is None:
        if first.get('hidden') != second.get('hidden'):
            notes.append('hidden-reasoning observable response differs')
        return notes
    if first["labels"] != second["labels"]:
        index = next((i for i, (a, b) in enumerate(zip(first["labels"], second["labels"])) if a != b),
                     min(len(first["labels"]), len(second["labels"])))
        notes.append(f"generated tokens differ from row {index}: "
                     f"{first['labels'][index:index + 3]} vs {second['labels'][index:index + 3]}")
        return notes
    for index, (a, b) in enumerate(zip(first["top"], second["top"])):
        if a != b:
            notes.append(f"row {index}: top-logprob lists differ: {a} vs {b}")
            break
    if first["text_offset"] != second["text_offset"]:
        notes.append(f"text offsets differ: {first['text_offset']} vs {second['text_offset']}")
    for mode in ("stream", "stream_ids"):
        for key in ("top", "text_offset"):
            a_values = [item for chunk in first[mode] for item in chunk.get(key) or []]
            b_values = [item for chunk in second[mode] for item in chunk.get(key) or []]
            if a_values != b_values:
                notes.append(f"{mode}: {key} differ")
        a = [label for chunk in first[mode] for label in chunk["labels"]]
        b = [label for chunk in second[mode] for label in chunk["labels"]]
        if a != b:
            index = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            notes.append(f"{mode}: {len(a)} vs {len(b)} streamed rows of {len(first['labels'])}, "
                         f"first difference at streamed row {index}: {a[index:index + 3]} vs "
                         f"{b[index:index + 3]}")
    return notes


def compare(args):
    captures = [json.loads(Path(path).read_text()) for path in args.receipts]
    expected = {name: (endpoint, request) for name, endpoint, request in cases()}
    errors = []
    for capture in captures:
        names = [row.get("case") for row in capture.get("cases", [])]
        if len(names) != len(expected) or set(names) != set(expected):
            errors.append("expected each required case exactly once")
        if capture.get("passed") is not True or capture.get("errors"):
            errors.append("capture did not pass")
    by_case = [{row["case"]: row for row in capture.get("cases", [])} for capture in captures]
    report = {}
    for name, (endpoint, request) in expected.items():
        if any(name not in arm for arm in by_case):
            report[name] = {"notes": ["required case is missing"]}
            continue
        for arm in by_case:
            row = arm[name]
            actual = {key: value for key, value in row.get("request", {}).items() if key != "model"}
            if row.get("endpoint") != endpoint or actual != request:
                errors.append(f"{name}: request differs from required case")
        first, second = summarize(by_case[0][name]), summarize(by_case[1][name])
        values = [value.get("values") or [] for value in (first, second)]
        report[name] = {"notes": compare_case(first, second),
                        "max_abs_logprob_difference": max(
                            (abs(a - b) for a, b in zip(*values) if finite(a) and finite(b)),
                            default=None),
                        "rows": [None if value.get("labels") is None else len(value["labels"])
                                 for value in (first, second)],
                        "streamed_rows": {mode: [sum(len(chunk["labels"]) for chunk in value.get(mode, []))
                                                 for value in (first, second)]
                                          for mode in ("stream", "stream_ids")}}
    arms = [capture.get("arm") for capture in captures]
    print(json.dumps({"arms": arms, "cases": report, "errors": errors,
                      "mechanics_only": any(capture.get("mechanics_only") for capture in captures),
                      "passed": not errors and all(not case["notes"] for case in report.values()),
                      "comparison": "structural coverage; numerical differences are informational"}, indent=2))
    return 0 if not errors and all(not case["notes"] for case in report.values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("capture")
    runner.add_argument("--arm", required=True)
    runner.add_argument("--url", default="http://127.0.0.1:8101")
    runner.add_argument("--model")
    runner.add_argument("--timeout", type=float, default=300)
    runner.add_argument("--server-log", type=Path,
                        help="Optional startup log, checked against the live pinned process")
    runner.add_argument("--mechanics-only", action="store_true",
                        help="Capture an unbound diagnostic endpoint; never qualifies a model")
    checker = commands.add_parser("compare")
    checker.add_argument("receipts", nargs=2)
    args = parser.parse_args(argv)
    startup(model_facts=args.command == "capture")
    return capture(args) if args.command == "capture" else compare(args)


if __name__ == "__main__":
    sys.exit(main())

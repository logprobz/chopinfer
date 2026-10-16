"""Structured-output cases, their local validator and the mechanics capture.

A model's structured check is captured by `chopinfer eval run` (the shared
structured kind, docs/check-kinds.md), which reads only a model's qual.json.
This module keeps the structured.v1 request builders and validator, which
soak.py's structured mix and the tests read, and the mechanics capture
(`--mechanics-only`) for the release-gates/v1 configs eval does not read,
such as the mechanics fixture (tools/qual/gates/mechanics.json).

Each case is one request to an OpenAI-compatible server. Outputs are checked
with a local validator for the JSON Schema subset used here, choice
membership and regex full match. HTTP 400 is recorded as "refused" and
finish_reason "length" as "truncated"; neither counts as a grammar violation.
Validity is valid / (valid + invalid).

    python3 tools/qual/structured.py --url URL --mechanics-only [--cases A,B]
    python3 tools/qual/structured.py --self-test
"""

import json
import re

from common import answer_kwargs, thinking_can_be_disabled, thinking_kwargs, Client, arguments, receipt


PERSON = {
    "type": "object", "additionalProperties": False,
    "required": ["name", "age", "status", "kind", "address", "tags", "active", "nickname", "note"],
    "properties": {
        "name": {"type": "string", "maxLength": 40},
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "status": {"type": "string", "enum": ["active", "inactive", "pending"]},
        "kind": {"const": "person"},
        "address": {"type": "object", "additionalProperties": False, "required": ["city", "zip"],
                    "properties": {"city": {"type": "string", "maxLength": 40},
                                   "zip": {"type": "string", "maxLength": 10}}},
        "tags": {"type": "array", "items": {"type": "string", "maxLength": 20},
                 "minItems": 1, "maxItems": 3},
        "active": {"type": "boolean"},
        "nickname": {"anyOf": [{"type": "string", "maxLength": 20}, {"type": "null"}]},
        "note": {"type": "null"},
    },
}
PERSON_PROMPT = ("Return a JSON record for a fictional person named Ada, aged 36, living in Paris "
                 "(zip 75001), with a status, a few tags, whether she is active, and an optional nickname.")
CHOICES = ["positive", "negative", "neutral"]
PHONE = r"\d{3}-\d{4}"
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["city"],
                       "properties": {"city": {"type": "string"},
                                      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}}}},
    {"type": "function", "function": {
        "name": "get_time", "description": "Get the current time in a timezone.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["timezone"],
                       "properties": {"timezone": {"type": "string"}}}}},
]
PARAMETERS = {tool["function"]["name"]: tool["function"]["parameters"] for tool in TOOLS}
TYPES = {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
         "string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool),
         "null": lambda v: v is None,
         "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool))
         or (isinstance(v, float) and v.is_integer()),
         "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)}


def schema_errors(value, schema, path="$"):
    """Return the violations of the supported JSON Schema subset, empty when valid."""
    if "anyOf" in schema:
        if all(schema_errors(value, option, path) for option in schema["anyOf"]):
            return [f"{path}: matches no anyOf branch"]
    if "const" in schema and (value != schema["const"] or type(value) is not type(schema["const"])):
        return [f"{path}: expected const {schema['const']!r}"]
    if "enum" in schema and not any(value == item and type(value) is type(item) for item in schema["enum"]):
        return [f"{path}: {value!r} not in enum"]
    kind = schema.get("type")
    if kind is not None and not any(TYPES[name](value) for name in ([kind] if isinstance(kind, str) else kind)):
        return [f"{path}: expected {kind}, got {type(value).__name__}"]
    errors = []
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        errors += [f"{path}: missing {key!r}" for key in schema.get("required", []) if key not in value]
        if schema.get("additionalProperties") is False:
            errors += [f"{path}: unexpected {key!r}" for key in value if key not in properties]
        for key, child in value.items():
            if key in properties:
                errors += schema_errors(child, properties[key], f"{path}.{key}")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        for index, child in enumerate(value):
            errors += schema_errors(child, schema.get("items", {}), f"{path}[{index}]")
    elif isinstance(value, str):
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']}")
    elif TYPES["number"](value):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above {schema['maximum']}")
    return errors


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_json(schema):
    def check(content, message):
        value = json.loads(content)
        require(isinstance(value, dict), "output is not a JSON object")
        if schema is not None:
            errors = schema_errors(value, schema)
            require(not errors, "; ".join(errors[:5]))
    return check


def check_choice(content, message):
    require(content in CHOICES, f"{content!r} is not an offered choice")


def check_regex(content, message):
    require(re.fullmatch(PHONE, content) is not None, f"{content!r} does not fully match {PHONE}")


def check_tools(named=None):
    def check(content, message):
        calls = message.get("tool_calls") or []
        require(calls, "no tool_calls")
        for call in calls:
            name = call["function"]["name"]
            require(name in PARAMETERS, f"tool {name!r} was not offered")
            require(named is None or name == named, f"called {name!r}, expected {named!r}")
            arguments = json.loads(call["function"]["arguments"])
            require(isinstance(arguments, dict), f"{name} arguments are not a JSON object")
            errors = schema_errors(arguments, PARAMETERS[name])
            require(not errors, f"{name}: " + "; ".join(errors[:5]))
    return check


def cases(args):
    """Return name -> (path, body without model/sampling fields, validator)."""
    def chat(prompt, thinking=False, **extra):
        if not thinking and not thinking_can_be_disabled():
            # The frontend always reasons: the answer follows its reasoning.
            extra.setdefault("max_tokens", args.reasoning_max_tokens)
        return {"messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": thinking_kwargs(True) if thinking else answer_kwargs(), **extra}

    def schema_format(strict):
        return {"type": "json_schema", "json_schema": {"name": "person", "schema": PERSON, "strict": strict}}

    completion_prompt = PERSON_PROMPT + "\nJSON:"
    weather = "What's the weather in Paris right now, in celsius? Also, what time is it in Tokyo?"
    chat_path, completion_path = "/v1/chat/completions", "/v1/completions"
    return {
        "chat_json_object": (chat_path, chat("Return a JSON object describing Paris with keys name and country.",
                                             response_format={"type": "json_object"}), check_json(None)),
        "chat_json_schema_strict": (chat_path, chat(PERSON_PROMPT, response_format=schema_format(True)),
                                    check_json(PERSON)),
        "chat_json_schema_loose": (chat_path, chat(PERSON_PROMPT, response_format=schema_format(False)),
                                   check_json(PERSON)),
        "chat_so_json": (chat_path, chat(PERSON_PROMPT, structured_outputs={"json": PERSON}), check_json(PERSON)),
        "chat_so_choice": (chat_path, chat("Classify the sentiment of: 'I love this phone.'",
                                           structured_outputs={"choice": CHOICES}), check_choice),
        "chat_so_regex": (chat_path, chat("Give a phone number in the form 555-0123.",
                                          structured_outputs={"regex": PHONE}), check_regex),
        "chat_so_json_object": (chat_path, chat("Return only a JSON object with exactly two string fields: name (Paris) and country (France).",
                                                structured_outputs={"json_object": True}), check_json(None)),
        "completion_so_json": (completion_path, {"prompt": completion_prompt, "structured_outputs": {"json": PERSON}},
                               check_json(PERSON)),
        "completion_json_schema": (completion_path, {"prompt": completion_prompt,
                                                     "response_format": schema_format(True)}, check_json(PERSON)),
        "tool_required": (chat_path, chat(weather, tools=TOOLS, tool_choice="required"), check_tools()),
        "tool_named": (chat_path, chat(weather, tools=TOOLS, tool_choice={
            "type": "function", "function": {"name": "get_time"}}), check_tools("get_time")),
        "reasoning_json_schema": (chat_path, chat(PERSON_PROMPT, thinking=True, response_format=schema_format(True),
                                                  max_tokens=args.reasoning_max_tokens), check_json(PERSON)),
    }


def assemble(events, chat):
    """Rebuild a non-streamed response shape from stream chunks."""
    content, calls, finish, usage = [], {}, None, None
    for _, event in events:
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta", {}) if chat else choice
            content.append((delta.get("content") if chat else delta.get("text")) or "")
            for part in delta.get("tool_calls") or []:
                call = calls.setdefault(part.get("index", 0), {"function": {"name": "", "arguments": ""}})
                call["id"] = part.get("id") or call.get("id")
                function = part.get("function") or {}
                call["function"]["name"] += function.get("name") or ""
                call["function"]["arguments"] += function.get("arguments") or ""
    message = {"content": "".join(content), "tool_calls": [calls[key] for key in sorted(calls)]}
    choice = {"finish_reason": finish, **({"message": message} if chat else {"text": message["content"]})}
    return {"choices": [choice], "usage": usage or {}}


def run_case(client, name, path, body, validate, stream):
    chat = path.endswith("/chat/completions")
    row = {"name": name + ("+stream" if stream else ""), "stream": stream}
    if stream:
        body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    try:
        value, latency = client.request(path, body, stream=stream)
    except RuntimeError as error:
        text = str(error)
        return {**row, "status": "refused" if text.startswith("HTTP 400") else "failed", "error": text}
    if stream:
        value = assemble(value, chat)
    choice = (value.get("choices") or [{}])[0]
    tokens = (value.get("usage") or {}).get("completion_tokens")
    row.update(latency_seconds=round(latency, 4), completion_tokens=tokens,
               tokens_per_second=round(tokens / latency, 2) if tokens and latency > 0 else None,
               finish_reason=choice.get("finish_reason"))
    message = choice.get("message") or {}
    content = message.get("content") if chat else choice.get("text")
    if choice.get("finish_reason") == "length":
        return {**row, "status": "truncated", "output": (content or "")[-200:]}
    try:
        validate(content or "", message)
        return {**row, "status": "passed"}
    except (ValueError, KeyError, TypeError) as error:
        return {**row, "status": "failed", "error": f"{type(error).__name__}: {error}",
                "output": (content or "")[:500]}


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))] if ordered else None


def summarize(rows):
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("passed", "failed", "refused", "truncated")}
    judged = counts["passed"] + counts["failed"]
    latencies = [row["latency_seconds"] for row in rows if "latency_seconds" in row]
    speeds = [row["tokens_per_second"] for row in rows if row.get("tokens_per_second")]
    return {**counts, "validity_rate": round(counts["passed"] / judged, 4) if judged else None,
            "latency_p50_seconds": percentile(latencies, 0.5), "latency_p90_seconds": percentile(latencies, 0.9),
            "mean_tokens_per_second": round(sum(speeds) / len(speeds), 2) if speeds else None}


def self_test():
    """Exercise the local validator without a server."""
    good = {"name": "Ada", "age": 36, "status": "active", "kind": "person",
            "address": {"city": "Paris", "zip": "75001"}, "tags": ["math"], "active": True,
            "nickname": None, "note": None}
    bad = [
        {k: v for k, v in good.items() if k != "age"}, {**good, "extra": 1}, {**good, "age": 151},
        {**good, "age": 3.5}, {**good, "age": True}, {**good, "status": "gone"}, {**good, "kind": "robot"},
        {**good, "address": {"city": "Paris"}}, {**good, "address": {"city": "Paris", "zip": "1", "x": 0}},
        {**good, "tags": []}, {**good, "tags": ["a", "b", "c", "d"]}, {**good, "tags": [1]},
        {**good, "name": "x" * 41}, {**good, "active": 1}, {**good, "nickname": 5}, {**good, "note": "x"},
    ]
    checks = [schema_errors(good, PERSON) == [],
              schema_errors({**good, "nickname": "Countess", "age": 36.0}, PERSON) == []]
    checks += [bool(schema_errors(value, PERSON)) for value in bad]
    cases_ok = [(check_json(PERSON), json.dumps(good), {}), (check_json(None), "{}", {}),
                (check_choice, "neutral", {}), (check_regex, "555-0123", {}),
                (check_tools("get_time"), "", {"tool_calls": [{"function": {
                    "name": "get_time", "arguments": '{"timezone": "Asia/Tokyo"}'}}]})]
    cases_bad = [(check_json(None), "[1]", {}), (check_json(None), "{", {}), (check_choice, "Positive", {}),
                 (check_regex, "555-01234", {}), (check_regex, "tel 555-0123", {}),
                 (check_tools(), "", {"tool_calls": []}),
                 (check_tools(), "", {"tool_calls": [{"function": {"name": "get_stock", "arguments": "{}"}}]}),
                 (check_tools(), "", {"tool_calls": [{"function": {"name": "get_weather", "arguments": "{}"}}]}),
                 (check_tools("get_time"), "", {"tool_calls": [{"function": {
                     "name": "get_weather", "arguments": '{"city": "Paris"}'}}]})]
    for validate, content, message in cases_ok:
        try:
            validate(content, message)
            checks.append(True)
        except ValueError:
            checks.append(False)
    for validate, content, message in cases_bad:
        try:
            validate(content, message)
            checks.append(False)
        except ValueError:
            checks.append(True)
    events = [(0, {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "a", "function": {
                  "name": "get_time", "arguments": '{"time'}}]}}]}),
              (0, {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'zone": "UTC"}'}}]},
                                "finish_reason": "tool_calls"}]}),
              (0, {"choices": [], "usage": {"completion_tokens": 9}})]
    rebuilt = assemble(events, chat=True)
    checks.append(rebuilt["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == '{"timezone": "UTC"}'
                  and rebuilt["usage"]["completion_tokens"] == 9)
    failed = [index for index, ok in enumerate(checks) if not ok]
    print(json.dumps({"self_test": "passed" if not failed else "failed", "checks": len(checks),
                      "failed_indexes": failed}))
    return not failed


def main():
    parser = arguments(__doc__, model_facts=True)
    parser.add_argument("--cases", help="Comma-separated case names; defaults to all")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--stream", action="store_true", help="Also run each case streamed")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reasoning-max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--self-test", action="store_true", help="Check the local validator and exit")
    args = parser.parse_args()
    if args.self_test:
        raise SystemExit(0 if self_test() else 1)
    if not args.mechanics_only:
        parser.error("structured.py runs only the mechanics capture; chopinfer eval run captures a model's "
                     "structured check")
    catalog = cases(args)
    names = args.cases.split(",") if args.cases else list(catalog)
    if unknown := [name for name in names if name not in catalog]:
        parser.error(f"unknown cases {unknown}; choose from {list(catalog)}")
    if args.repeat < 1:
        parser.error("repeat must be positive")
    client = Client(args.url, args.timeout, retries=1)
    server = client.identity(args.model)
    rows = []
    for name in names:
        path, body, validate = catalog[name]
        body = {"model": server["model"], "temperature": args.temperature,
                "max_tokens": args.max_tokens, **body}
        # Fixture outputs are 8 tokens long.
        body["max_tokens"] = 8
        if args.seed is not None:
            body["seed"] = args.seed
        for _ in range(args.repeat):
            for stream in (False, True) if args.stream else (False,):
                row = run_case(client, name, path, body, validate, stream)
                rows.append(row)
                print(json.dumps(row), flush=True)
    per_case = {}
    for row in rows:
        per_case.setdefault(row["name"], []).append(row)
    summary = {"overall": summarize(rows), "cases": {name: summarize(group) for name, group in per_case.items()}}
    print(json.dumps({"summary": summary}))
    errors = [{"name": row["name"], "error": row.get("error", "")} for row in rows if row["status"] == "failed"]
    passed = not errors
    receipt("structured", server, {
        "passed": passed, "skipped": ["output validity (8-token outputs)"],
        "errors": errors, "summary": summary, "rows": rows,
        "settings": {"cases": names, "repeat": args.repeat, "stream": args.stream, "max_tokens": args.max_tokens,
                     "reasoning_max_tokens": args.reasoning_max_tokens, "temperature": args.temperature,
                     "seed": args.seed}}, True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

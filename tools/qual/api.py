"""API mechanics: the api.v1 request shapes against a fixture server.

A model's api check is captured by `chopinfer eval run` (the shared api
kind, docs/check-kinds.md), which reads only a model's qual.json. This module
keeps the mechanics capture (`--mechanics-only`) for the release-gates/v1
configs eval does not read: the mechanics fixture (tools/qual/gates/
mechanics.json) and external declarations such as
examples/external-model/qualification.json. It sends the same requests as
eval's mechanics capture and checks response shapes only: no model
reasoning and no tool round trip. The validators keep their full checks
(`validate_tool_stream` outside a mechanics run too), which runtime-eval's
own validators are tested against.

    python3 tools/qual/api.py --url URL --mechanics-only
"""

import json
import math
import urllib.request

from common import thinking_kwargs, Client, arguments, receipt, response_text

# The tool round trip's final answer, after the model's default reasoning:
# a constant a model declaration's api check may set, which `chopinfer eval`
# applies (the mechanics capture sends no tool round trip).
TOOL_RESULT_MAX_TOKENS = 1024





def main():
    parser = arguments(__doc__, model_facts=True)
    args = parser.parse_args()
    if not args.mechanics_only:
        parser.error("api.py runs only the mechanics capture; chopinfer eval run captures a model's api check")
    client = Client(args.url, args.timeout)
    server = client.identity(args.model)
    errors = []
    rows = []

    def check(name, body, validate, path=None):
        value = None
        try:
            value, latency = client.request(path or ("/v1/chat/completions" if "messages" in body else "/v1/completions"), body,
                                            stream=body.get("stream", False))
            validate(value)
            rows.append({"name": name, "latency_seconds": latency, "passed": True})
            return value
        except Exception as error:
            # Keep the response a check rejected: a failure cannot be explained without it.
            errors.append({"name": name, "error": str(error), **({"response": value} if value is not None else {})})
            rows.append({"name": name, "passed": False})
            return None

    model = server["model"]
    for chat in (False, True):
        # vLLM refuses n > 1 with greedy sampling, so multiple choices sample.
        body = {"model": model, "n": 3, "seed": 41, "temperature": 0.7, "max_tokens": 3, "ignore_eos": True}
        body.update({"messages": [{"role": "user", "content": "Hello"}]} if chat else {"prompt": "Hello"})
        single = check(f"choice_single_{chat}", dict(body, n=1), lambda value: require(len(value["choices"]) == 1, "missing choice"))
        expected_prompt = single["usage"]["prompt_tokens"] if single else None
        for stream in (False, True):
            request = dict(body, stream=stream)
            if stream:
                request["stream_options"] = {"include_usage": True}
            check(f"multiple_choices_{chat}_{stream}", request,
                  lambda value, stream=stream, prompt=expected_prompt: validate_choices(value, stream, 3, prompt, 9))
    for chat in (False, True):
        for stream in (False, True):
            request = {"model": model, "return_token_ids": True, "temperature": 0, "max_tokens": 3,
                       "ignore_eos": True, "stream": stream}
            request.update({"messages": [{"role": "user", "content": "Hello"}]} if chat else {"prompt": "Hello"})
            if stream:
                request["stream_options"] = {"include_usage": True}
            check(f"generated_token_ids_{chat}_{stream}", request,
                  lambda value, stream=stream, chat=chat: validate_token_ids(value, stream, chat))
    encoded = check("tokenize", {"model": model, "prompt": "Hello", "add_special_tokens": False},
                    lambda value: require(value["count"] == len(value["tokens"]), "token count differs"), path="/tokenize")
    if encoded is not None and encoded["tokens"]:
        for prompt in (["Hello", "Hello"], [encoded["tokens"], encoded["tokens"]]):
            single = check("batch_single", {"model": model, "prompt": prompt[0], "max_tokens": 1, "temperature": 0}, lambda value: require(bool(value["choices"]), "missing choice"))
            expected_prompt = 2 * single["usage"]["prompt_tokens"] if single else None
            for stream in (False, True):
                request = {"model": model, "prompt": prompt, "n": 2, "seed": 41, "temperature": 0.7, "max_tokens": 3,
                           "ignore_eos": True, "stream": stream}
                if stream:
                    request["stream_options"] = {"include_usage": True}
                check(f"batch_{isinstance(prompt[0], list)}_{stream}", request,
                      lambda value, stream=stream, prompt=expected_prompt: validate_choices(value, stream, 4, prompt, 12))
    for special in (False, True):
        for chat in (False, True):
            request = {"model": model, "add_special_tokens": special, "return_token_ids": True, "max_tokens": 1, "temperature": 0}
            request.update({"messages": [{"role": "user", "content": "Hello"}]} if chat else {"prompt": "Hello"})
            check(f"add_special_tokens_{chat}_{special}", request,
                  lambda value, chat=chat: require(bool(value.get("prompt_token_ids") if chat else value["choices"][0].get("prompt_token_ids")), "missing prompt IDs"))
    if encoded is not None:
        check("detokenize", {"model": model, "tokens": encoded["tokens"]},
              lambda value: require(value["prompt"] == "Hello", "tokenizer round trip differs"), path="/detokenize")
    if encoded is not None and encoded["tokens"]:
        stop_id = encoded["tokens"][0]
        check("stop_token_ids", {"model": model, "prompt": "Hello", "max_tokens": 3,
                                 "temperature": 0, "logit_bias": {str(stop_id): 100},
                                 "stop_token_ids": [stop_id]},
              lambda value: require(value["usage"]["completion_tokens"] == 1
                                    and value["choices"][0]["text"] == ""
                                    and value["choices"][0]["finish_reason"] == "stop", "token stop differs"))
    check("generation_prompt_control", {"model": model, "messages": [{"role": "user", "content": "Hello"}],
                                        "max_tokens": 1, "add_generation_prompt": False, "temperature": 0},
          lambda value: require(bool(value.get("choices")), "generation prefix control failed"))
    check("hidden_reasoning", {"model": model, "messages": [{"role": "user", "content": "What is 17 times 23?"}],
                                "chat_template_kwargs": thinking_kwargs(True), "include_reasoning": False,
                                "max_tokens": 8, "temperature": 0},
          lambda value: require(not (value["choices"][0]["message"].get("reasoning") or
                                    value["choices"][0]["message"].get("reasoning_content")), "reasoning was not suppressed"))
    check("ignore_eos", {"model": model, "prompt": "Hello", "max_tokens": 3, "ignore_eos": True,
                          "skip_special_tokens": False, "temperature": 0, "stop_token_ids": []},
          lambda value: require(value["usage"]["completion_tokens"] == 3 and value["choices"][0]["finish_reason"] == "length", "ignore_eos did not reach the length bound"))
    check("nullable_flags", {"model": model, "messages": [{"role": "user", "content": "Hello"}],
                              "max_tokens": 1, "temperature": 0, "stream": None, "logprobs": None, "top_logprobs": 0},
          lambda value: require(bool(value.get("choices")), "nullable flags were refused"))
    check("unknown_extension", {"model": model, "prompt": "Hello", "max_tokens": 1, "temperature": 0, "parity_future_extension": True},
          lambda value: require(bool(value.get("choices")), "unknown extension was not ignored"))
    check("sampling_extensions", {"model": model, "prompt": "Hello", "max_tokens": 3,
                                  "min_tokens": 2, "temperature": 0, "seed": -1, "top_k": -1, "user": "api-qualification"},
          lambda value: require(2 <= value["usage"]["completion_tokens"] <= 3, "minimum token bound differs"))
    check("chat_token_limit_precedence", {"model": model, "messages": [{"role": "user", "content": "Hello"}],
                                          "max_tokens": 1, "max_completion_tokens": 3, "min_tokens": 2, "temperature": 0},
          lambda value: require(2 <= value["usage"]["completion_tokens"] <= 3, "max_completion_tokens did not take precedence"))
    check("continuous_usage", {"model": model, "prompt": "Hello", "max_tokens": 3, "temperature": 0, "stream": True,
                                "stream_options": {"include_usage": True, "continuous_usage_stats": True}},
          validate_continuous_usage)
    stream = check("stream_usage", {"model": model, "messages": [{"role": "user", "content": "Say hello."}],
                                    "stream": True, "stream_options": {"include_usage": True},
                                    "max_tokens": 24, "temperature": 0},
                   lambda events: (require(any(event.get("object") == "chat.completion.chunk" for _, event in events), "missing chat chunk"),
                                   require(any(event.get("choices") == [] and "usage" in event for _, event in events), "missing usage chunk")))
    check("stop", {"model": model, "messages": [{"role": "user", "content": "Write a short sentence ending with END."}],
                   "stop": ["END"], "max_tokens": 48, "temperature": 0},
          lambda value: require("END" not in (value["choices"][0]["message"].get("content") or ""), "stop string leaked"))
    check("completion_logprobs", {"model": model, "prompt": "Paris is", "max_tokens": 5,
                                   "temperature": 0, "logprobs": 5},
          lambda value: require({"text_offset", "token_logprobs", "tokens", "top_logprobs"} <= set(value["choices"][0]["logprobs"]),
                                "completion logprob shape differs"))
    # Both reasoning switches are sent; a mechanics run reads each answer's
    # message without judging the model's reasoning.
    for thinking in (True, False):
        check(f"reasoning_{thinking}", {"model": model, "messages": [{"role": "user", "content": "What is 17 times 23?"}],
                                        "chat_template_kwargs": thinking_kwargs(thinking), "max_tokens": 8,
                                        "temperature": 0}, lambda value: value["choices"][0]["message"])
    tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    messages = [{"role": "user", "content": "Paris?"}]
    check("stream_tool_arguments", {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto",
                                    "stream": True, "max_tokens": 1, "temperature": 0},
          lambda value: validate_tool_stream(value, True, None))
    check("tool_call", {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto", "max_tokens": 1,
                        "temperature": 0}, lambda value: None)
    try:
        with urllib.request.urlopen(args.url.rstrip("/") + "/metrics", timeout=args.timeout) as response:
            validate_metrics(response.read().decode())
        rows.append({"name": "metrics_compatibility", "passed": True})
    except Exception as error:
        errors.append({"name": "metrics_compatibility", "error": str(error)})
        rows.append({"name": "metrics_compatibility", "passed": False})
    passed = not errors
    # Its per-request seconds are diagnostics of an accuracy check, not timing evidence, so no host load
    # samples bound them and the receipt's timing reads informational.
    receipt("api", server, {"passed": passed, "skipped": ["model reasoning and tool round trip (fixture context is 128 tokens)"],
                           "errors": errors, "rows": rows,
                           "stream_text": response_text(stream, chat=True) if stream else None}, True)
    raise SystemExit(0 if passed else 1)


def validate_metrics(text):
    samples = [line for line in text.splitlines() if line and not line.startswith("#")]
    counters = ["vllm:request_success_total", "vllm:prompt_tokens_total", "vllm:generation_tokens_total"]
    gauges = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc"]
    histograms = ["vllm:time_to_first_token_seconds", "vllm:inter_token_latency_seconds", "vllm:e2e_request_latency_seconds"]
    for name in counters + gauges + [name + suffix for name in histograms for suffix in ("_bucket", "_sum", "_count")]:
        series = [line for line in samples if line.startswith(name + "{")]
        require(series, f"missing compatibility series {name}")
        for line in series:
            require('model_name="' in line and 'engine="' in line, f"missing labels: {line}")
            require(float(line.rsplit(" ", 1)[1]) >= 0, f"negative metric: {line}")
    require(any('finished_reason="' in line for line in samples if line.startswith("vllm:request_success_total{")), "missing finish reason label")
    for name in histograms:
        require(any(line.startswith(name + "_bucket{") and 'le="+Inf"' in line for line in samples), f"missing infinite bucket {name}")


def validate_token_ids(value, stream, chat):
    if stream:
        values = [event for _, event in value]
        require(all("error" not in event for event in values), f"token stream failed: {values}")
        ids = [token for event in values for choice in event.get("choices", []) for token in (choice.get("token_ids") or [])]
        prompts = ([event.get("prompt_token_ids") for event in values] if chat else
                   [choice.get("prompt_token_ids") for event in values for choice in event.get("choices", [])])
        require(any(prompts), "missing prompt IDs")
        usage = values[-1]["usage"]
    else:
        ids = value["choices"][0]["token_ids"]
        require(value.get("prompt_token_ids") if chat else value["choices"][0].get("prompt_token_ids"), "missing prompt IDs")
        usage = value["usage"]
    require(len(ids) == usage["completion_tokens"], "generated token IDs differ from usage")
    require(all(isinstance(token, int) and token >= 0 for token in ids), "invalid token IDs")


def validate_tool_stream(events, mechanics_only, unstreamed):
    """Every chunk's choice is choice 0, and tool indexes count up from 0.
    Outside a mechanics run: a call finishing `tool_calls`, each call with a
    unique ID and one name, sent before its arguments, which are complete
    before the finish and form a JSON object; the calls' names and arguments
    equal those of `unstreamed()`, the same request served whole. The
    arguments may arrive in one delta or several."""
    calls = []
    terminal = None
    for _, event in events:
        require("error" not in event, "tool stream failed")
        for choice in event.get("choices", []):
            index = choice.get("index")
            require(type(index) is int and index == 0, "tool stream has another choice")
            terminal = choice.get("finish_reason") or terminal
            for delta in choice.get("delta", {}).get("tool_calls", []):
                index = delta["index"]
                require(type(index) is int and 0 <= index <= len(calls), "invalid tool index")
                if index == len(calls):
                    calls.append({"arguments": "", "unnamed": False, "late": False})
                call = calls[index]
                identifier = delta.get("id")
                if identifier is not None:
                    require(isinstance(identifier, str) and identifier, "invalid tool ID")
                    require(all(other.get("id") != identifier for other in calls) and "id" not in call,
                            "tool ID repeated")
                    call["id"] = identifier
                function = delta.get("function", {})
                name = function.get("name")
                if name is not None and name != "":
                    require(isinstance(name, str), "invalid tool name")
                    require("name" not in call, "tool name repeated")
                    call["name"] = name
                part = function.get("arguments") or ""
                call["arguments"] += part
                if part:
                    call["unnamed"] = call["unnamed"] or "name" not in call
                    call["late"] = call["late"] or bool(terminal)
    if not mechanics_only:
        require(calls and terminal == "tool_calls", "missing streaming tool call")
        for call in calls:
            require(call.get("id") and call.get("name"), "missing tool identity")
            require(not call["unnamed"], "tool arguments came before the tool name")
            require(not call["late"], "tool arguments came with or after the finish")
            require(json_object(call["arguments"]) is not None, "tool arguments are not a JSON object")
        whole = unstreamed()["choices"][0]["message"].get("tool_calls") or []
        require(same_json([[call["name"], json_object(call["arguments"])] for call in calls],
                          [[call["function"]["name"], json_object(call["function"]["arguments"])] for call in whole]),
                "streamed tool calls differ from the unstreamed response")


def same_json(left, right):
    """JSON values equal with their types: an integer never equals a float
    (1 is not 1.0) and a boolean never equals a number."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same_json(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(map(same_json, left, right))
    return left == right


def json_object(text):
    """`text` parsed when it is a JSON object, otherwise None. As runtime-eval
    parses JSON: repeated keys, non-finite numbers and integers outside 64
    bits are refused."""
    def refuse(_):
        raise ValueError("refused")

    def pairs(items):
        keys = [key for key, _ in items]
        return dict(items) if len(set(keys)) == len(keys) else refuse(keys)

    def integer(text):
        value = int(text)
        return value if -2**63 <= value < 2**64 else refuse(text)

    def real(text):
        value = float(text)
        return value if math.isfinite(value) else refuse(text)
    try:
        value = json.loads(text, parse_constant=refuse, object_pairs_hook=pairs, parse_int=integer,
                           parse_float=real)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def validate_choices(value, stream, count, prompt_tokens, completion_tokens):
    if stream:
        values = [event for _, event in value]
        choices = [choice for event in values for choice in event.get("choices", [])
                   if choice.get("finish_reason") is not None]
        usage = values[-1]["usage"]
        require(values[-1]["choices"] == [], "missing aggregate usage chunk")
        require(len({event["id"] for event in values}) == 1, "response IDs differ")
    else:
        choices, usage = value["choices"], value["usage"]
    require(sorted(choice["index"] for choice in choices) == list(range(count)), "choice indexes differ")
    require(all(choice["finish_reason"] == "length" for choice in choices), "choice finish reason differs")
    require(usage["prompt_tokens"] == prompt_tokens, "prompt usage duplicated")
    require(usage["completion_tokens"] == completion_tokens, "completion usage differs")
    require(usage["total_tokens"] == prompt_tokens + completion_tokens, "total usage differs")


def validate_continuous_usage(events):
    values = [event for _, event in events]
    require(values and values[-1].get("choices") == [], "missing final usage chunk")
    previous = 0
    for value in values:
        usage = value.get("usage")
        require(isinstance(usage, dict), "missing continuous usage")
        count = usage["completion_tokens"]
        require(count >= previous, "usage regressed")
        require(usage["total_tokens"] == usage["prompt_tokens"] + count, "usage total differs")
        previous = count
    details = values[-1]["usage"].get("prompt_tokens_details")
    if details is not None:
        require(0 <= details["cached_tokens"] <= values[-1]["usage"]["prompt_tokens"], "invalid cache usage")


def require(condition, message):
    if not condition:
        raise ValueError(message)


if __name__ == "__main__":
    main()

"""DeepSeek V4.1 Flash cases for the differential frontend corpus.

The checkpoint ships no chat template: its prompt-format reference is the
Python encoder `encoding/encoding.py`, which `PIN` names with its digest.
`python -m tools.refrun.frontend_corpus generate --model deepseek41` renders
these cases with that encoder (`$CHOPINFER_DEEPSEEK_CHECKPOINT`, revision
1bae0e10), through `encoder_render`, the mapping of a request onto
`encode_messages` that `models/deepseek41/contract/src/frontend.rs` serves.
The module states the pin, the transport translation, the generation endings,
how a history turn reads back, and DeepSeek's own axes and fixed cases.
"""

import copy
import json

PIN = {
    "revision": "1bae0e1088993b9b82de08f59de5be5a34e42170",
    "source": "encoding/encoding.py",
    "source_sha256": "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2",
    "checkpoint_env": "CHOPINFER_DEEPSEEK_CHECKPOINT",
}

# What the export's OpenedV1 states: history reasoning_content on every
# assistant turn and the client's tool objects whole.
TRANSPORT = {
    "reasoning_content_fallback": True,
    "tool_call_reasoning_fallback": False,
    "reads_tool_objects": True,
    "tool_references": False,
    "extra_roles": [],
}

# The request fields the encoder renders beyond the HF input.
INPUT_FIELDS = ("response_format",)

# Tool-call arguments reach the encoder as the text sent: it reads them
# itself, up to twice (lines 533-544).
RAW_TOOL_ARGUMENTS = True

# The protocol efforts the encoder does not name, as deepseek-recipe maps
# them and the served format reads them: minimal as low, medium and xhigh
# as high.
EFFORT_ALIASES = {"minimal": "low", "medium": "high", "xhigh": "high"}

EOS = "<｜end▁of▁sentence｜>"

# Every assistant turn ends with the end-of-sentence token (encoding.py
# line 60), so a history render holds it; these apply only without it.
ENDINGS = {"tool_calls": [EOS], "answer": [EOS]}

# Refused by this frontend: each runs in one row of otherwise ordinary values.
ISOLATED = {"developer:present"}

# The markers the vendor parser refuses inside reasoning or content
# (encoding.py lines 981-983).
SPECIAL = ("<｜begin▁of▁sentence｜>", EOS, "<think>", "</think>", "｜DSML｜")


def thinking(keywords):
    """The mode the served frontend selects: the first declared thinking
    keyword the request sets, else thinking (`default_enabled`)."""
    for keyword in ("enable_thinking", "thinking"):
        if isinstance(keywords.get(keyword), bool):
            return keywords[keyword]
    return True


def encoder_message(message):
    """One HF message as the encoder reads it: a media item back in OpenAI's
    `image_url` form, which `_extract_image` (lines 305-320) reads."""
    message = copy.deepcopy(message)
    if isinstance(message.get("content"), list):
        parts = []
        for part in message["content"]:
            if isinstance(part, dict) and part.get("type") in ("image", "video", "audio"):
                kind = part["type"]
                part = {"type": f"{kind}_url", f"{kind}_url": {"url": part.get(kind)}}
            parts.append(part)
        message["content"] = parts
    return message


def encoder_render(namespace, messages, tools, add_generation_prompt, continue_final_message, keywords,
                   response_format=None):
    """The prompt `encode_messages` (lines 810-844) writes for one HF input,
    mapped as the served format maps a request:
    - the mode from the thinking keywords, and the effort (the keyword
      `reasoning_effort`, which a top-level effort sets, with a protocol
      alias read as the effort it names) in thinking mode only;
    - `preserve_thinking` turns off `drop_thinking`;
    - the offered tools and `response_format` go to the leading system
      message, an empty one when the request has none (README: tools are
      defined on the system message);
    - a continued final assistant turn renders without its end token
      (`wo_eos`); the encoder writes the generation header itself, so
      `add_generation_prompt` changes nothing."""
    mode = "thinking" if thinking(keywords) else "chat"
    effort = keywords.get("reasoning_effort") if mode == "thinking" else None
    effort = EFFORT_ALIASES.get(effort, effort) if isinstance(effort, str) else effort
    messages = [encoder_message(message) for message in messages]
    if continue_final_message:
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("continue_final_message requires a final assistant message")
        messages[-1]["wo_eos"] = True
    if tools or response_format is not None:
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": ""})
        if tools:
            if messages[0].get("tools"):
                raise ValueError("tools are offered both by the request and by its system message")
            messages[0]["tools"] = tools
        messages[0]["response_format"] = response_format
    return namespace["encode_messages"](
        messages, thinking_mode=mode, drop_thinking=not keywords.get("preserve_thinking", False),
        reasoning_effort=effort)


def expected_turn(message, context):
    """What the vendor's `parse_message_from_completion_text` (lines
    946-990) reads back from a generated assistant turn: the reasoning in
    thinking mode (the prompt leaves `<think>` open), and the content. A
    turn with tool calls reads back only through the DSML parser family,
    which the declaration does not use yet, so it reads no turn back; nor
    does a turn whose text holds a marker the vendor parser refuses."""
    if message.get("tool_calls"):
        return None
    reasoning = message.get("reasoning_content") if context["prompt"].endswith("<think>") else None
    content = message.get("content")
    if isinstance(content, list):
        content = "\n\n".join(part.get("text", "") for part in content
                              if isinstance(part, dict) and part.get("type") == "text")
    content = content or ""
    reasoning = reasoning or ""
    if any(marker in text for marker in SPECIAL for text in (content, reasoning)):
        return None
    return {"reasoning": reasoning, "content": content, "tool_calls": []}


RESPONSE_FORMATS = {
    "json-schema": {"type": "json_schema", "json_schema": {
        "name": "answer", "strict": True,
        "schema": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city"], "additionalProperties": False}}},
    "json-object": {"type": "json_object"},
    "text": {"type": "text"},
}


def features(declaration):
    return {"response-format": ["absent", *RESPONSE_FORMATS]}


def apply_feature(request, axis, value, rng):
    """Sets the request's `response_format`."""
    if axis != "response-format" or value == "absent":
        return ()
    request["response_format"] = copy.deepcopy(RESPONSE_FORMATS[value])
    return {f"response-format:{value}"}


def user(text):
    return {"role": "user", "content": text}


def call(identifier, name, arguments):
    return {"id": identifier, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the weather for a specific location",
    "parameters": {"type": "object", "properties": {
        "location": {"type": "string", "description": "The city name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}},
        "required": ["location"]}}}


def cases(declaration):
    # The encoder's test pair 1 (encoding/tests/test_input_1.json) as a
    # served request: tools on the leading system message, a call without
    # an id and its result, thinking at the default effort.
    yield "vendor-pair-1", {"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        user("What's the weather like in Beijing?"),
        {"role": "assistant", "reasoning_content": "The user wants the weather in Beijing. I should call get_weather.",
         "content": "", "tool_calls": [call("call_0", "get_weather", {"location": "Beijing", "unit": "celsius"})]},
        {"role": "tool", "tool_call_id": "call_0", "content": "{\"temperature\": 22, \"condition\": \"sunny\"}"},
        {"role": "assistant", "reasoning_content": "Got the weather data.",
         "content": "The weather in Beijing is sunny at 22°C."},
    ], "tools": [WEATHER]}, {"fixed:vendor-pair-1", "ends:assistant", "tools:wrapped"}
    # A later system message counts as a user turn: the header follows it,
    # and reasoning before it drops.
    yield "later-system-message", {"messages": [
        user("q1"), {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "system", "content": "Answer in French from now on."},
    ]}, {"fixed:later-system-message", "system:late"}
    # Consecutive user messages and a result followed by user text join
    # into one user turn (merge_tool_messages, lines 215-226).
    yield "joined-user-turns", {"messages": [
        user("first"), user("second"),
        {"role": "assistant", "content": "", "tool_calls": [call("call_a", "get_weather", {"location": "Paris"})]},
        {"role": "tool", "tool_call_id": "call_a", "content": "18"}, user("And tomorrow?"),
    ], "tools": [WEATHER]}, {"fixed:joined-user-turns", "tools:wrapped"}
    # Reasoning kept by preserve_thinking, and an effort the encoder maps.
    yield "preserved-reasoning-at-max", {"messages": [
        user("q1"), {"role": "assistant", "content": "a1", "reasoning_content": "r1"}, user("q2"),
    ], "chat_template_kwargs": {"preserve_thinking": True}, "reasoning_effort": "max"}, {
        "fixed:preserved-reasoning-at-max", "preserve_thinking:true"}
    # The response format on a request without a system message, and with
    # tools: both go to an inserted empty system message.
    yield "response-format-with-tools", {"messages": [user("Weather in Oslo as JSON.")],
                                         "tools": [WEATHER], "response_format": RESPONSE_FORMATS["json-schema"]}, {
        "fixed:response-format-with-tools", "response-format:json-schema", "tools:wrapped"}
    # Arguments as the encoder writes them: strings raw, other values as
    # Python's JSON, keys in their order.
    yield "dsml-argument-values", {"messages": [
        user("Go."),
        {"role": "assistant", "content": "", "tool_calls": [call("call_v", "f", {
            "s": "a \"quoted\" <tag>", "n": 1.5, "i": -0, "b": True, "z": None, "l": [1, "x"], "o": {"k": "v"}})]},
        {"role": "tool", "tool_call_id": "call_v", "content": "ok"},
    ], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]}, {
        "fixed:dsml-argument-values", "tools:wrapped"}
    # The image placeholder in client text, which the encoder refuses in
    # string content and reasoning (lines 355-367) and in a text block
    # (lines 342-348).
    placeholder = "<｜deepseek_image｜>"
    yield "image-placeholder-in-content", {"messages": [user(f"see {placeholder}")]}, {
        "fixed:image-placeholder-in-content"}
    yield "image-placeholder-in-reasoning", {"messages": [
        user("q"), {"role": "assistant", "content": "a", "reasoning_content": f"r {placeholder}"}, user("q2"),
    ]}, {"fixed:image-placeholder-in-reasoning"}
    yield "image-placeholder-in-text-part", {"messages": [
        {"role": "user", "content": [{"type": "text", "text": f"see {placeholder}"}]}]}, {
        "fixed:image-placeholder-in-text-part", "user:parts"}
    # Arguments that are JSON text of JSON text, which the encoder reads
    # twice (lines 533-544).
    yield "double-encoded-arguments", {"messages": [
        user("Go."),
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_d", "type": "function",
            "function": {"name": "f", "arguments": json.dumps(json.dumps({"x": 1, "y": "z"}))}}]},
        {"role": "tool", "tool_call_id": "call_d", "content": "ok"},
    ], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]}, {
        "fixed:double-encoded-arguments", "tools:wrapped"}
    # List content on a system and an assistant turn: its text parts joined
    # (lines 380-393).
    text = lambda value: {"type": "text", "text": value}
    yield "list-content-on-system-and-assistant", {"messages": [
        {"role": "system", "content": [text("Be brief."), text("Be kind.")]}, user("q1"),
        {"role": "assistant", "content": [text("a"), text("b")]}, user("q2"),
    ]}, {"fixed:list-content-on-system-and-assistant"}
    # A developer message before the last user turn in thinking mode, which
    # the encoder drops unread (lines 756-761) and this format refuses.
    yield "developer-before-last-user", {"messages": [
        {"role": "developer", "content": "Be terse."}, user("q1"),
        {"role": "assistant", "content": "a1"}, user("q2"),
    ]}, {"fixed:developer-before-last-user", "developer:present"}
    # Protocol efforts the encoder does not name, top-level and nested.
    yield "medium-effort", {"messages": [user("q")], "reasoning_effort": "medium"}, {"fixed:medium-effort"}
    yield "nested-xhigh-effort", {"messages": [user("q")], "chat_template_kwargs": {"reasoning_effort": "xhigh"}}, {
        "fixed:nested-xhigh-effort"}
    # Argument text the encoder reads itself: spacing kept in a value it
    # reads once, a null, three JSON string layers, and a repeated key,
    # whose last value Python keeps.
    raw = lambda identifier, arguments: {"id": identifier, "type": "function",
                                         "function": {"name": "f", "arguments": arguments}}
    yield "raw-argument-text", {"messages": [
        user("Go."),
        {"role": "assistant", "content": "", "tool_calls": [
            raw("call_l", "[1,  2]"), raw("call_n", "null"),
            raw("call_t", json.dumps(json.dumps(json.dumps({"x": 1})))), raw("call_r", "{\"x\":1,\"y\":3,\"x\":2}")]},
        {"role": "tool", "tool_call_id": "call_l", "content": "ok"},
        {"role": "tool", "tool_call_id": "call_n", "content": "ok"},
        {"role": "tool", "tool_call_id": "call_t", "content": "ok"},
        {"role": "tool", "tool_call_id": "call_r", "content": "ok"},
    ], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]}, {
        "fixed:raw-argument-text", "tools:wrapped"}
    # A function object holding a `function` key is still wrapped whole.
    yield "function-key-in-function", {"messages": [user("q")], "tools": [
        {"type": "function", "function": {"name": "f", "function": {"name": "g"}}}]}, {
        "fixed:function-key-in-function", "tools:wrapped"}
    # A namespace on a function object or a system message's tool object,
    # which the encoder writes as namespace::name and this format refuses.
    yield "namespaced-function-object", {"messages": [user("q")], "tools": [
        {"type": "function", "function": {"name": "f", "namespace": {"name": "ns", "description": "Tools of ns."},
                                          "description": "F.", "parameters": {"type": "object"}}}]}, {
        "fixed:namespaced-function-object", "tool-namespace:present"}
    yield "namespaced-system-tool", {"messages": [
        {"role": "system", "content": "s", "tools": [
            {"type": "function", "namespace": "ns", "function": {"name": "f", "parameters": {"type": "object"}}}]},
        user("q"),
    ]}, {"fixed:namespaced-system-tool", "tool-namespace:present"}

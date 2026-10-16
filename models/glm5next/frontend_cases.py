"""GLM 5.3 Flash cases for the HF differential corpus.

`python -m tools.refrun.frontend_corpus generate --model glm5next` renders
these with the pinned template (`$CHOPINFER_GLM5NEXT_CHECKPOINT`, revision
eb9eb208). The module states the template pin, the transport translation the
HF input follows, the generation endings of each turn shape, how a history
turn reads back through the parser, and GLM's own feature axes and fixed
cases: tool_reference results, including the three chats of the
tool_reference handoff.
"""

import json

PIN = {
    "revision": "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    "template_sha256": "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5",
    "checkpoint_env": "CHOPINFER_GLM5NEXT_CHECKPOINT",
}

# What the export's OpenedV1 states: history reasoning_content on every
# assistant turn, the client's tool objects whole, and tool_reference parts.
TRANSPORT = {
    "reasoning_content_fallback": True,
    "tool_call_reasoning_fallback": False,
    "reads_tool_objects": True,
    "tool_references": True,
    "extra_roles": [],
}

# A history turn ends where the next role token begins (template lines
# 140-164), so a generated turn ends at one of these stop tokens.
ENDINGS = {"tool_calls": ["<|observation|>"], "answer": ["<|user|>", "<|endoftext|>"]}

# Refused before the render or by this frontend: each runs in one row of
# otherwise ordinary values.
ISOLATED = {"developer:present", "message-tools:present"} | {
    f"tool-reference:{value}" for value in ("name", "tool_name", "both-keys", "unknown", "no-tools",
                                            "deferred-eager", "mixed-text", "text-first")}


def visible_text(content):
    """The template's `visible_text` of text content (lines 54-74)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content
                       if isinstance(part, dict) and part.get("type") == "text")
    return ""


def expected_turn(message, context=None):
    """What the parser reads back from an assistant history turn (template
    lines 140-164): the reasoning, else the text inside the content's
    `<think>`, and the stripped content. None when reasoning markup stays in
    the content, which a generated turn never carries. GLM always opens
    `<think>`, so the context (prompt, request) changes nothing."""
    content = visible_text(message.get("content"))
    reasoning = message.get("reasoning_content")
    if not isinstance(reasoning, str):
        reasoning = None
        if "</think>" in content:
            reasoning = content.split("</think>")[0].split("<think>")[-1]
            content = content.split("</think>")[-1]
    if "<think>" in content or "</think>" in content:
        return None
    calls = [{"name": call["function"]["name"],
              "arguments": json.dumps(call["function"]["arguments"], ensure_ascii=False)}
             for call in message.get("tool_calls") or []]
    return {"reasoning": reasoning or "", "content": content.strip(), "tool_calls": calls}


def features(declaration):
    return {"tool-reference": ["none", "name", "tool_name", "both-keys", "unknown", "no-tools", "deferred-eager",
                               "mixed-text", "text-first"]}


def apply_feature(request, axis, value, rng):
    """Turns the first tool result into tool_reference parts."""
    if axis != "tool-reference" or value == "none":
        return ()
    result = next((message for message in request["messages"] if message.get("role") == "tool"), None)
    if result is None:
        return ()
    tools = request.get("tools") or []
    names = [tool["function"]["name"] for tool in tools if isinstance(tool, dict) and "function" in tool]
    if not names:
        return ()
    name = names[0]
    reference = {"type": "tool_reference", "name": name}
    if value == "tool_name":
        reference = {"type": "tool_reference", "tool_name": name}
    elif value == "both-keys":
        reference["tool_name"] = name
    elif value == "unknown":
        reference["name"] = "missing_tool"
    parts = [reference]
    if value == "deferred-eager":
        request["tools"] = tools + [{"type": "function", "function": {
            "name": "lookup_later", "description": "Deferred.", "parameters": {"type": "object"},
            "defer_loading": True}}]
        parts = [reference, {"type": "tool_reference", "name": "lookup_later"},
                 {"type": "tool_reference", "name": name}]
    elif value == "mixed-text":
        parts = [reference, {"type": "text", "text": " loaded"}]
    elif value == "text-first":
        parts = [{"type": "text", "text": "loaded "}, reference]
    elif value == "no-tools":
        del request["tools"]
    result["content"] = parts
    return {f"tool-reference:{value}", "part:tool_reference"}


def tool_name_to_name(request):
    """The `tool_name` alias as the `name` the template reads."""
    for message in request["messages"]:
        if isinstance(message.get("content"), list):
            for part in message["content"]:
                if isinstance(part, dict) and part.get("type") == "tool_reference" and "tool_name" in part:
                    part.setdefault("name", part.pop("tool_name"))
                    part.pop("tool_name", None)
    return request


# The tool_reference handoff chats (dropped card branch df5255694, goldens.py).
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city, in °C.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name, e.g. 北京"},
        "days": {"type": "integer", "minimum": 1, "maximum": 14},
        "metric": {"type": "boolean"},
        "ratio": {"type": "number", "default": 0.5}}, "required": ["city"]},
    "strict": True}}
PING = {"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}
SEARCH = {"type": "function", "function": {
    "name": "tool_search", "description": "Find tools by keyword.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
LOOKUP = {"type": "function", "function": {
    "name": "lookup", "description": "Look up a record.",
    "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}}, "strict": True,
    "defer_loading": True}}
CONVERT = {"type": "function", "function": {
    "defer_loading": True, "name": "convert", "description": "Convert °C to °F.",
    "parameters": {"type": "object", "properties": {"celsius": {"type": "number"}}}}}


def call(identifier, name, arguments):
    return {"id": identifier, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


def user(text):
    return {"role": "user", "content": text}


def tool(identifier, text):
    return {"role": "tool", "tool_call_id": identifier, "content": text}


def assistant(content, reasoning=None, calls=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    return message


def references(identifier, *names):
    """A tool result of `tool_reference` parts, one per name."""
    return {"role": "tool", "tool_call_id": identifier,
            "content": [{"type": "tool_reference", "name": name} for name in names]}


def cases(declaration):
    yield "tool-references-deferred-and-eager", {"messages": [
        user("Convert 18 °C and look up record 7."),
        assistant("", "Find the tools first.", [call("call_s", "tool_search", {"query": "convert lookup"})]),
        references("call_s", "convert", "missing", "get_weather", "lookup", "convert"),
        assistant("", "Both are loaded.", [
            call("call_c", "convert", {"celsius": 18}), call("call_l", "lookup", {"id": 7})]),
        tool("call_c", "64.4"),
        tool("call_l", "{\"id\": 7, \"name\": \"seven\"}"),
    ], "tools": [SEARCH, LOOKUP, WEATHER, CONVERT]}, {
        "fixed:tool-references-deferred-and-eager", "part:tool_reference", "tool-reference:deferred-eager",
        "tool-reference:unknown"}
    yield "tool-references-reordered-with-text", {"messages": [
        user("Find both."),
        assistant("", "Search twice.", [
            call("call_1", "tool_search", {"query": "convert"}),
            call("call_2", "tool_search", {"query": "weather"}),
            call("call_3", "ping", {}),
        ]),
        tool("call_3", "pong"),
        references("call_2", "get_weather"),
        references("call_1", "convert"),
    ], "tools": [SEARCH, WEATHER, PING, CONVERT], "reasoning_effort": "low"}, {
        "fixed:tool-references-reordered-with-text", "part:tool_reference", "tool-reference:name",
        "results:reordered"}
    yield "tool-reference-to-an-unoffered-tool", {"messages": [
        user("Find a calendar tool."),
        assistant("", None, [call("call_s", "tool_search", {"query": "calendar"})]),
        references("call_s", "calendar"),
    ], "tools": [SEARCH, LOOKUP]}, {
        "fixed:tool-reference-to-an-unoffered-tool", "part:tool_reference", "tool-reference:unknown"}
    # The sort fallbacks of an observation block (lines 211-223): every result
    # answers a call, but a call has no id, or two calls share one.
    weather = [call("call_a", "get_weather", {"city": "Paris"}), call("", "ping", {})]
    yield "sort-fallback-call-without-id", {"messages": [
        user("Weather?"), assistant("", "Check.", weather), tool("call_a", "18"),
    ], "tools": [WEATHER, PING]}, {"fixed:sort-fallback-call-without-id", "sort:call-without-id"}
    yield "tool-result-without-a-call", {"messages": [
        user("Weather?"), tool("call_a", "18"), user("And?"),
    ], "tools": [WEATHER]}, {"fixed:tool-result-without-a-call", "sort:no-calls"}
    shared = [call("call_a", "get_weather", {"city": "Paris"}), call("call_a", "ping", {})]
    yield "sort-fallback-duplicate-call-ids", {"messages": [
        user("Weather?"), assistant("", "Check.", shared), tool("call_a", "18"),
    ], "tools": [WEATHER, PING]}, {"fixed:sort-fallback-duplicate-call-ids", "sort:duplicate-call-ids"}
    # An answer stripped with Python's whitespace (line 154), which includes
    # the separators \x1c to \x1f.
    yield "answer-stripped-of-separator-controls", {"messages": [
        user("Weather?"), assistant("\x1f\x1c It is mild. \x1d\x1e", "Check."), user("And tomorrow?"),
    ]}, {"fixed:answer-stripped-of-separator-controls", "assistant-content:python-whitespace"}
    # A continued final turn whose text the render keeps (an inline <think>
    # first), and one it does not (text before the <think>, which the
    # template drops), which transformers refuses.
    yield "continue-inline-think-kept", {"messages": [
        user("Weather?"), assistant("<think>Check.</think>It is"),
    ], "continue_final_message": True}, {"fixed:continue-inline-think-kept", "ends:continue-inline-think"}
    yield "continue-inline-think-dropped", {"messages": [
        user("Weather?"), assistant("Draft. <think>Check.</think> It is"),
    ], "continue_final_message": True}, {"fixed:continue-inline-think-dropped", "ends:continue-text-dropped"}
    # A continued final turn whose content is a list: transformers continues
    # its last text part, so the turn continues when the render keeps that
    # part (after the <think>, or empty), and is refused when the template
    # drops text of it (text before the <think>).
    text = lambda value: {"type": "text", "text": value}
    for name, parts, tags in (
            ("continue-parts-inline-kept", [text("Draft. <think>Check.</think>"), text(" It is")],
             "ends:continue-parts-kept"),
            ("continue-parts-empty-last", [text("Draft. <think>Check.</think>Answer"), text("")],
             "ends:continue-parts-empty-last"),
            ("continue-parts-last-dropped", [text("It is"), text(" Draft. <think>Check.</think> more")],
             "ends:continue-parts-dropped"),
            ("continue-parts-single-dropped", [text("Draft. <think>Check.</think> It is")],
             "ends:continue-parts-dropped")):
        yield name, {"messages": [user("Weather?"), assistant(parts)], "continue_final_message": True}, {
            f"fixed:{name}", tags, "assistant-content:parts"}

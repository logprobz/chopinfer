"""Qwen3.8 cases for the HF differential corpus.

`python -m tools.refrun.frontend_corpus generate --model qwen38` renders these
with the pinned template (`$CHOPINFER_QWEN38_CHECKPOINT`, the revision and
digest `fixtures/qwen38/frontend/provenance.json` records). The module states
the template pin, the transport translation the HF input follows, how a
history turn reads back through the parser, and Qwen's own axes and fixed
cases: `add_vision_id` and each `raise_exception` path of the template.
"""

import json

PIN = {
    "revision": "236dfdf285828023ca3bcd3f37366c58a3469b13",
    "template_sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "checkpoint_env": "CHOPINFER_QWEN38_CHECKPOINT",
}

# History reasoning_content only on turns with tool calls
# (provenance.json tool_call_reasoning_content_fallback).
TRANSPORT = {
    "reasoning_content_fallback": False,
    "tool_call_reasoning_fallback": True,
    "reads_tool_objects": False,
    "tool_references": False,
    "extra_roles": [],
}

# The history render holds the stop token (<|im_end|>).
ENDINGS = None

# Values the template refuses (the developer role, a late system message)
# or this frontend refuses: each runs in one row of otherwise ordinary values.
ISOLATED = {"developer:present", "system:late", "message-tools:present"}


def render_content(content):
    """The template's `render_content` of text content, trimmed (lines 3-41,
    103)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
    return ""


# The generation prompt's end with thinking disabled (template lines 165-166).
CHAT_PROMPT_END = "<think>\n\n</think>\n\n"


def expected_turn(message, context=None):
    """What the parser reads back from an assistant history turn (template
    lines 110-146). After a thinking prompt (`<think>\\n`): the trimmed
    reasoning and its newline, then the content after the `\\n\\n` answer
    prefix, up to the calls. After a chat prompt, which already closed an
    empty think block: the content alone, up to the calls."""
    thinking = not (context and context["prompt"].endswith(CHAT_PROMPT_END))
    reasoning = message.get("reasoning_content")
    reasoning = reasoning.strip() if isinstance(reasoning, str) else ""
    content = render_content(message.get("content"))
    calls = message.get("tool_calls") or []
    answer = ("\n\n" if thinking else "") + content
    if calls:
        answer = answer + "\n\n" if content else ""
    return {"reasoning": reasoning + "\n" if thinking else "", "content": answer,
            "tool_calls": [{"name": call["function"]["name"],
                            "arguments": json.dumps(call["function"]["arguments"], ensure_ascii=False)}
                           for call in calls]}


def features(declaration):
    return {"add_vision_id": ["absent", "true", "false"]}


def apply_feature(request, axis, value, rng):
    if axis != "add_vision_id" or value == "absent":
        return ()
    request.setdefault("chat_template_kwargs", {})["add_vision_id"] = value == "true"
    return {f"add_vision_id:{value}"}


IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
VIDEO = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA=="}}


def cases(declaration):
    """Each `raise_exception` path, and the tool-response user turn the
    template skips when it looks for the last query (lines 88-101)."""
    question = {"role": "user", "content": "Hi"}
    yield "raise-system-image", {"messages": [
        {"role": "system", "content": [IMAGE, {"type": "text", "text": "Be brief."}]}, question]}, {
        "fixed:raise-system-image", "raise:system-image"}
    yield "raise-system-video", {"messages": [
        {"role": "system", "content": [VIDEO, {"type": "text", "text": "Be brief."}]}, question]}, {
        "fixed:raise-system-video", "raise:system-video"}
    yield "raise-unexpected-item", {"messages": [
        {"role": "user", "content": [{"type": "file", "file": {"file_id": "f"}}]}]}, {
        "fixed:raise-unexpected-item", "raise:unexpected-item"}
    yield "raise-unexpected-content", {"messages": [{"role": "user", "content": 7}]}, {
        "fixed:raise-unexpected-content", "raise:unexpected-content"}
    yield "raise-no-messages", {"messages": []}, {"fixed:raise-no-messages", "raise:no-messages"}
    yield "raise-no-user-query", {"messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "<tool_response>\n42\n</tool_response>"}]}, {
        "fixed:raise-no-user-query", "raise:no-user-query"}
    yield "raise-unexpected-role", {"messages": [question, {"role": "critic", "content": "Fine."}]}, {
        "fixed:raise-unexpected-role", "raise:unexpected-role"}
    yield "numbered-image-and-video", {"messages": [
        {"role": "user", "content": [IMAGE, VIDEO, {"type": "text", "text": "Compare them."}]}],
        "chat_template_kwargs": {"add_vision_id": True}}, {
        "fixed:numbered-image-and-video", "media:image_url", "media:video_url", "add_vision_id:true"}
    # Round trips after a chat prompt (enable_thinking false).
    chat = {"enable_thinking": False}
    yield "round-chat-answer", {"messages": [question, {"role": "assistant", "content": "Done."}],
                                "chat_template_kwargs": chat}, {
        "fixed:round-chat-answer", "thinking:enable_thinking=false"}
    yield "round-chat-call", {"messages": [question, {"role": "assistant", "content": "Checking.", "tool_calls": [
        {"id": "call_0", "type": "function", "function": {"name": "lookup", "arguments": "{\"id\": 7}"}}]}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {
            "type": "object", "properties": {"id": {"type": "integer"}}}}}],
        "chat_template_kwargs": chat}, {"fixed:round-chat-call", "thinking:enable_thinking=false"}
    # A final turn with an empty tool_calls list and only reasoning_content:
    # the transport forwards reasoning_content on a turn with a tool_calls
    # list, so its history writes the reasoning a chat prompt closed empty.
    yield "round-chat-empty-calls-reasoning", {"messages": [
        {"role": "user", "content": "Go."},
        {"role": "assistant", "content": "a", "tool_calls": [], "reasoning_content": "fallback"}],
        "chat_template_kwargs": chat, "add_generation_prompt": False}, {
        "fixed:round-chat-empty-calls-reasoning", "thinking:enable_thinking=false",
        "reasoning-field:reasoning_content", "calls:empty-list"}
    # A tool result with no turn before it: loop.previtem is undefined, so no
    # user turn opens (lines 148-150).
    yield "tool-first", {"messages": [
        {"role": "tool", "tool_call_id": "call_0", "content": "42"}, question]}, {
        "fixed:tool-first", "results:first"}
    # Python's whitespace, which `| trim` strips (lines 70, 103, 115), includes
    # the separators U+001C to U+001F.
    yield "python-whitespace", {"messages": [
        {"role": "system", "content": "\x1cBe brief.\x1f"},
        {"role": "user", "content": "\x1dWeather?\x1e"},
        {"role": "assistant", "content": "\x1cChecking.\x1f", "reasoning": "\x1dAsk the tool.\x1e",
         "tool_calls": [{"id": "call_0", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "\x1c18\x1f"},
        {"role": "user", "content": "Thanks."}]}, {
        "fixed:python-whitespace", "whitespace:python"}
    yield "tool-response-user-turn", {"messages": [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": "", "reasoning": "Ask the tool.",
         "tool_calls": [{"id": "call_0", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "user", "content": "<tool_response>\n18\n</tool_response>"},
        {"role": "assistant", "content": "It is 18.", "reasoning": "Report it."},
        {"role": "user", "content": "Thanks."}],
        "chat_template_kwargs": {"preserve_thinking": False}}, {
        "fixed:tool-response-user-turn", "user:tool-response"}
    # GIF and WebP images, which the image modality declares beside PNG and
    # JPEG.
    for kind in ("gif", "webp"):
        yield f"image-{kind}", {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64,AA=="}},
            {"type": "text", "text": "Describe."}]}]}, {f"fixed:image-{kind}", f"image:{kind}"}

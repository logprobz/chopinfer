"""MiMo v2.6 cases for the HF differential corpus.

`python -m tools.refrun.frontend_corpus generate --model mimo26` renders these
with the pinned template (`$CHOPINFER_MIMO_CHECKPOINT`, the MOPD revision
and the digest `fixtures/mimo26/frontend/provenance.json` records). The
module states the template pin, the transport translation the HF input
follows, how a history turn reads back through the parser, MiMo's own axis:
custom tool calls (`tool_call.custom`, template lines 41-46), and the
processor's rewrite of the template's text. A top-level disabling effort
needs no normalization here: the shared HF input passes the reasoning
controls as the transport does.
"""

import json

PIN = {
    "revision": "2479e2d0029eca9a34cc7e7f55a121925f81908e",
    "template_sha256": "11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059",
    "checkpoint_env": "CHOPINFER_MIMO_CHECKPOINT",
}

# MiMo's chat format reads no history reasoning_content, no client tool
# objects and no tool_reference parts (`Mimo26Format::opened`,
# models/mimo26/contract/src/frontend.rs).
TRANSPORT = {
    "reasoning_content_fallback": False,
    "tool_call_reasoning_fallback": False,
    "reads_tool_objects": False,
    "tool_references": False,
    "extra_roles": [],
}

# The history render holds the stop token (<|im_end|>).
ENDINGS = None

ISOLATED = {"tool-call:custom"}


def reference_text(text, arguments):
    """What MiMo reads of the template's text: HF's processor replaces each
    `<|video_pad|>` the template writes (line 21) with `<|mimo_video_start|>`,
    the timestamped frames and `<|mimo_video_end|>`
    (fixtures/mimo26/capture/frontend_media.py, `prepare`;
    models/mimo26/media_workload.py). The frontend check writes each media
    item as one pad, so the reference keeps one pad between the processor's
    markers."""
    return text.replace("<|video_pad|>", "<|mimo_video_start|><|video_pad|><|mimo_video_end|>")


def render_content(content):
    """The template's `render_content` of text content (lines 9-27)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def expected_turn(message, context=None):
    """What the parser reads back from an assistant history turn (template
    lines 60-68): the reasoning between `<think>` and `</think>`, the content
    as written, and the calls. None when the content carries reasoning
    markup, which a generated turn never does. A chat prompt closes an empty
    `<think></think>` (lines 92-94), and a turn round-trips under it only
    without reasoning, so it reads back the same."""
    reasoning = message.get("reasoning_content")
    content = render_content(message.get("content"))
    if "<think>" in content or "</think>" in content:
        return None
    return {"reasoning": reasoning if isinstance(reasoning, str) else "",
            "content": content,
            "tool_calls": [{"name": call["function"]["name"],
                            "arguments": json.dumps(call["function"]["arguments"], ensure_ascii=False)}
                           for call in message.get("tool_calls") or []]}


def features(declaration):
    return {"tool-call": ["function", "custom"]}


def apply_feature(request, axis, value, rng):
    """Turns the first tool call into a custom call with raw input."""
    if axis != "tool-call":
        return ()
    for message in request["messages"]:
        for call in message.get("tool_calls") or []:
            if value == "function":
                return {"tool-call:function"}
            function = call.pop("function")
            call["type"] = "custom"
            call["custom"] = {"name": function["name"], "input": "raw input <&>"}
            return {"tool-call:custom"}
    return ()


def cases(declaration):
    """A user message with media that also carries tools (template lines
    78-86): its media and its tool list both render."""
    yield "message-tools-with-media", {"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        {"type": "text", "text": "Which tool fits?"}], "tools": [{"type": "function", "function": {
            "name": "lookup", "description": "Look a record up.",
            "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}}}}]}]}, {
        "fixed:message-tools-with-media", "message-tools:present", "media:image_url"}
    # GIF and WebP images, which the image modality declares beside PNG and
    # JPEG.
    for kind in ("gif", "webp"):
        yield f"image-{kind}", {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/{kind};base64,AA=="}},
            {"type": "text", "text": "Describe."}]}]}, {f"fixed:image-{kind}", f"image:{kind}"}

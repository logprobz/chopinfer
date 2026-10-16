"""Render and tokenizer goldens for the GLM 5.3 Flash frontend declaration.

Renders every case with transformers' `apply_chat_template` from the pinned
`chat_template.jinja` (revision eb9eb208) and tokenizes with the pinned
`tokenizer.json`, on CPU. Nothing here loads weights.

    python goldens.py CHECKPOINT_DIR OUTPUT_DIR [--gate GATE_DIR] [--check]

CHECKPOINT_DIR holds chat_template.jinja, tokenizer.json and
tokenizer_config.json. OUTPUT_DIR receives chat-goldens.json and
tokenizer-golden.json; `--check` compares instead of writing.

GATE_DIR (fixtures/glm5next/frontend) receives the frontend_goldens gate's
chat reference, chat-renders.json: the text cases here and the conversations
of the chat teacher reference (models/glm5next/reference_workloads.py CHATS
and CHATS_EXTENDED), each with its HF render and token ids. The image cases
stay out: a served /tokenize expands the image pad, which a template render
does not (media_preprocessing gates the expansion). The gate's tokenizer
reference is tokenizer-golden.json here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
TEMPLATE_SHA256 = "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5"

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city, in °C.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 北京"},
                "days": {"type": "integer", "minimum": 1, "maximum": 14},
                "metric": {"type": "boolean"},
                "ratio": {"type": "number", "default": 0.5},
            },
            "required": ["city"],
        },
        "strict": True,
    },
}
PING = {"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}
CLOCK = {"type": "function", "function": {"name": "clock", "description": "The time."}}


def call(identifier, name, arguments):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": arguments}}


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


TOOL_TURN = assistant(
    "\n  Let me check.  \n",
    "The user wants weather.",
    [
        call("call_a", "get_weather", {"city": "Paris <&>", "days": 3, "metric": True, "ratio": 0.25}),
        call("call_b", "get_weather", {"city": "北京", "days": 1, "extra": {"unit": "c", "hours": [1, 2.5, None]}, "note": None}),
        call("call_c", "ping", {}),
    ],
)


def with_ids(messages):
    """`messages` as an OpenAI request carries them: each tool call with an
    id (call_0, call_1, ...) and each tool result answering the oldest
    unanswered call. The teacher conversations leave them out; the
    template does not render them there."""
    result, pending, count = [], [], 0
    for message in messages:
        message = dict(message)
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                calls.append({"id": f"call_{count}", **call})
                pending.append(f"call_{count}")
                count += 1
            message["tool_calls"] = calls
        elif message["role"] == "tool" and "tool_call_id" not in message:
            message["tool_call_id"] = pending.pop(0)
        result.append(message)
    return result


def cases():
    hello = [user(" Hello! ")]
    yield "effort-default", hello, [], {}
    for effort in ["low", "high", "max"]:
        yield f"effort-{effort}", hello, [], {"reasoning_effort": effort}
    yield "system-multi-turn", [
        {"role": "system", "content": " You are terse. "},
        user("What is 2+2?"),
        assistant("  4  ", " Add them. "),
        user("And 3+3?"),
        assistant("6"),
        user("Thanks.\n"),
    ], [], {"reasoning_effort": "high"}
    yield "late-and-empty-system", [
        user("first"),
        assistant("one", "r1"),
        {"role": "system", "content": ""},
        {"role": "system", "content": "Now answer in French."},
        user("second"),
    ], [], {}
    yield "tools-calls-and-results", [
        {"role": "system", "content": "Use tools."},
        user("Weather in Paris and Beijing?"),
        TOOL_TURN,
        tool("call_a", "18 °C"),
        tool("call_b", "{\"temp\": 21}"),
        tool("call_c", " pong "),
        assistant("Paris 18 °C, Beijing 21 °C.", "Report both."),
        user("Thanks"),
    ], [WEATHER, PING, CLOCK], {"reasoning_effort": "low"}
    yield "tool-results-reordered", [
        user("Weather?"),
        TOOL_TURN,
        tool("call_c", "pong"),
        tool("call_a", "18"),
        tool("call_b", "21"),
    ], [WEATHER, PING], {}
    yield "tool-results-unmatched-keep-order", [
        user("Weather?"),
        TOOL_TURN,
        tool("call_c", "pong"),
        tool("call_x", "18"),
    ], [WEATHER, PING], {}
    yield "inline-reasoning", [
        user("q"),
        assistant("<think>old</think>  answer "),
        assistant("pre <think>a <think>b</think>mid</think> end"),
        assistant("kept</think>tail", "given"),
        user("next"),
    ], [], {}
    yield "clear-thinking", [
        user("q1"),
        assistant("a1", "r1"),
        user("q2"),
        assistant("", "r2", [call("call_p", "ping", {})]),
        tool("call_p", "pong"),
        assistant("a2", "r3"),
    ], [PING], {"clear_thinking": True}
    yield "tool-definition-forms", [user("Find it.")], [
        {"type": "function", "function": {
            "description": "find", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            "name": "lookup", "strict": False}},
        {"type": "function", "function": {"name": "nullable", "description": None, "parameters": {"type": "object"}}},
        {"type": "function", "function": {
            "name": "titled", "title": "Lookup records", "description": "t", "parameters": {"type": "object"},
            "x-meta": {"a": [1, 2.5, None, True], "ü": "ß"}}},
        {"type": "function", "function": {
            "name": "later", "description": "deferred", "parameters": {"type": "object"}, "defer_loading": True}},
        {"type": "function", "function": {"name": "eager", "parameters": {"type": "object"}, "defer_loading": False}},
        {"type": "function", "function": {"name": "zero", "defer_loading": 0}},
        {"type": "function", "function": {"name": "flagged", "defer_loading": "yes"}},
    ], {}
    yield "all-tools-deferred", [user("Find it.")], [
        {"type": "function", "function": {"name": "later", "parameters": {"type": "object"}, "defer_loading": True}},
    ], {}
    yield "tool-call-without-content", [
        user("Ping."),
        assistant("", None, [call("call_p", "ping", {})]),
        tool("call_p", "pong"),
    ], [PING], {"reasoning_effort": "high"}


def image_cases():
    image = {"type": "image_url", "image_url": {"url": "file:///probe.png"}}
    yield "image-one", [
        {"role": "user", "content": [image, {"type": "text", "text": "What is this?"}]},
    ], [], {}
    yield "image-two-multi-turn", [
        {"role": "user", "content": [{"type": "text", "text": "Compare "}, image, {"type": "text", "text": " and "}, image]},
        assistant("They match.", "Same picture."),
        {"role": "user", "content": [{"type": "text", "text": "And now?"}, image]},
    ], [], {"reasoning_effort": "low"}


TOKENIZER_TEXTS = [
    "Hello, world!",
    " leading and  double  spaces ",
    "line one\nline two\r\n\ttabbed\n\n\n",
    "数字 12345 和 3.14159，中文标点！",
    "émoji 🙂 and 👩‍👩‍👧 family",
    "def f(x):\n    return x ** 2  # square\n",
    "I'm sure they've done it, haven't they? 'Quotes' \"double\"",
    "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>",
    "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
    "<|begin_of_image|><|image|><|end_of_image|><|observation|><tool_response>ok</tool_response>",
    "/nothink <|endoftext|> trailing",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    " non breaking　spaces​",
    "ÅÄÖ åäö Ø ß ﬁ ﬀ İ ı",
    "",
    "3.14159 2.71828 -0.001 1e10 0x1F 0b1010 1,234,567.89 50% $12.99",
    "مرحبا بالعالم שלום עולם Привет, мир! 한국어 텍스트 日本語のテキスト",
    "URL https://example.com/a/b?c=d&e=f#g and email someone@example.org",
    "Ｆｕｌｌｗｉｄｔｈ ①②③ \U0001d400\U0001d401 zero\u200bwidth joiner\u200d end",
    "<|system|><|user|><|assistant|><|observation|>",
    "<think></think><tool_call></tool_call><arg_key></arg_key><arg_value></arg_value>",
    "<tool_response></tool_response><|begin_of_image|><|end_of_image|><|image|>",
    "[gMASK][sMASK][MASK]<sop><eop><|endoftext|>",
    "{\"temperature\": 18, \"condition\": \"cloudy\", \"hours\": [1, 2.5, null], \"ok\": true}",
    "| a | b |\n|---|---|\n| 1 | 2 |\n\n- item\n  - nested\n\n```python\nprint('x')\n```",
    "        \t\t   \n \r\n  trailing    ",
    "请用一句话解释什么是机器学习。机器学习是让计算机从数据中自动发现规律。",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gate", type=Path, help="also write (or check) the gate's chat-renders.json here")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    template = (args.checkpoint / "chat_template.jinja").read_bytes().decode()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    def render(case_id, messages, tools, kwargs):
        text = tokenizer.apply_chat_template(
            messages, tools=tools or None, chat_template=template, tokenize=False,
            add_generation_prompt=True, **kwargs,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return {"id": case_id, "messages": messages, "tools": tools,
                "chat_template_kwargs": kwargs, "rendered_text": text, "token_ids": ids}

    chat = [render(*case) for case in cases()]
    chat += [render(*case) for case in image_cases()]
    texts = []
    for text in TOKENIZER_TEXTS:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        texts.append({"text": text, "token_ids": ids, "decoded": tokenizer.decode(ids)})
    outputs = {
        args.output / "chat-goldens.json": chat,
        args.output / "tokenizer-golden.json": texts,
    }
    if args.gate:
        sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
        import reference_workloads
        conversations = [render(name, with_ids(spec["messages"]), spec.get("tools", []), spec.get("kwargs", {}))
                         for name, spec in (reference_workloads.CHATS | reference_workloads.CHATS_EXTENDED).items()]
        for row, spec in zip(conversations, (reference_workloads.CHATS | reference_workloads.CHATS_EXTENDED).values()):
            bare = render(row["id"], spec["messages"], spec.get("tools", []), spec.get("kwargs", {}))
            if bare["rendered_text"] != row["rendered_text"]:
                raise SystemExit(f"{row['id']}: the call ids change the render")
        outputs[args.gate / "chat-renders.json"] = [render(*case) for case in cases()] + conversations
    digest = hashlib.sha256(template.encode()).hexdigest()
    if digest != TEMPLATE_SHA256:
        raise SystemExit(f"chat_template.jinja is {digest}, not revision {REVISION}'s {TEMPLATE_SHA256}")
    failed = False
    for path, value in outputs.items():
        text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        if args.check:
            if path.read_text() != text:
                print(f"{path.name}: differs")
                failed = True
        else:
            path.write_text(text)
            print(f"wrote {path} ({len(value)} cases)")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

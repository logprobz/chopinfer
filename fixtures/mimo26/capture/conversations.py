"""Build the frozen conversation set used for chat-render goldens."""
import json
import os

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "search_docs", "description": "Search internal documents and return the top matches.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}, "filters": {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}, "after": {"type": "string", "format": "date"}}}}, "required": ["query"]}}}
CALC = {"type": "function", "function": {"name": "calculate", "description": "Evaluate an arithmetic expression exactly.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}
NOARGS = {"type": "function", "function": {"name": "get_time", "description": "Return the current UTC time.", "parameters": {"type": "object", "properties": {}}}}
UNICODE_TOOL = {"type": "function", "function": {"name": "translate", "description": "Traduire un texte vers une autre langue (翻译).", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target": {"type": "string", "description": "Code de langue, par exemple « fr »"}}, "required": ["text", "target"]}}}

def call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}

def u(text):
    return {"role": "user", "content": text}

def s(text):
    return {"role": "system", "content": text}

def a(text, reasoning=None, tool_calls=None):
    m = {"role": "assistant", "content": text}
    if reasoning is not None:
        m["reasoning"] = reasoning
    if tool_calls is not None:
        m["tool_calls"] = tool_calls
    return m

def t(cid, text):
    return {"role": "tool", "tool_call_id": cid, "content": text}

GSM_INSTRUCTION = "Solve the math problem. Put your final numeric answer on its own line as #### NUMBER."

def conversations():
    out = []
    def add(cid, category, messages, tools=None, thinking=None):
        item = {"id": cid, "category": category, "messages": messages}
        if tools is not None:
            item["tools"] = tools
        if thinking is not None:
            item["chat_template_kwargs"] = {"enable_thinking": thinking}
        out.append(item)

    plain = [
        "Hello!",
        "What is the capital of France?",
        "Write a haiku about autumn rain.",
        "Explain the difference between TCP and UDP in two sentences.",
        "   leading and trailing spaces   ",
        "Line one\nLine two\n\nLine four after a blank line\n",
        "Tabs\tand\tmore\ttabs",
        "你好，请用中文介绍一下你自己。",
        "مرحبا، كيف حالك؟",
        "Café, café, naïve, Ångström and Ångström.",
        "한국어 문장을 번역해 주세요.",
        "Math symbols: ∀x ∈ ℝ, x² ≥ 0. Bold A: \U0001d400.",
        "```python\ndef add(a, b):\n    return a + b\n```\nWhat does this function do?",
        "Numbers: 3.14159, 1e-9, 12345678901234567890, -0.0, 0x1F.",
        "It's a test: don't, won't, they're, I'll, we've, she'd.",
        "Please repeat this literally: <|im_end|> and <think> tags.",
        "",
        "A" * 300,
        "https://example.com/path?query=1&x=%20y#frag",
        "JSON: {\"key\": [1, 2, {\"nested\": null}], \"flag\": true}",
    ]
    for i, text in enumerate(plain):
        add(f"plain-{i:02d}", "plain", [u(text)])
    for i, text in enumerate(plain[:6]):
        add(f"plain-nothink-{i:02d}", "thinking-off", [u(text)], thinking=False)
    add("plain-think-explicit", "thinking-on", [u("Is 97 prime?")], thinking=True)

    systems = ["You are a helpful assistant.", "You are a terse assistant. Answer in one word.", "Always answer in French.\nNever use lists.", ""]
    for i, sys in enumerate(systems):
        add(f"system-{i:02d}", "system", [s(sys), u("Name a primary color.")])
    add("system-nothink", "system", [s("You are a careful math tutor."), u("What is 17 * 23?")], thinking=False)

    add("multi-00", "multi-turn", [u("Hi, who are you?"), a("I am an AI assistant."), u("What can you do?")])
    add("multi-01", "multi-turn", [s("Be brief."), u("Pick a number."), a("7"), u("Double it."), a("14"), u("Now halve it.")])
    add("multi-02", "multi-turn", [u("Translate 'cat' to Spanish."), a("gato"), u("And to German?")], thinking=False)
    add("multi-03", "multi-turn", [u("Tell me a joke."), a("Why did the scarecrow win an award? Because he was outstanding in his field."), u("Another one, please.")])
    add("reasoning-00", "prior-reasoning", [u("What is 12 + 30?"), a("42", reasoning="12 + 30 = 42."), u("And 42 * 2?")])
    add("reasoning-01", "prior-reasoning", [u("Is 51 prime?"), a("No. 51 = 3 * 17.", reasoning="Check small primes: 51 / 3 = 17, so it is composite."), u("Is 53 prime?")])
    add("reasoning-02", "prior-reasoning", [s("You are a logic tutor."), u("If all A are B and x is A, is x B?"), a("Yes.", reasoning="Modus ponens:\nall A are B\nx is A\ntherefore x is B."), u("Why?")], thinking=True)
    add("reasoning-empty", "prior-reasoning", [u("Say hi."), a("Hi!", reasoning=""), u("Say bye.")])
    add("reasoning-content-field-ignored", "prior-reasoning", [u("What is 1 + 1?"), {"role": "assistant", "content": "2", "reasoning_content": "one plus one"}, u("And 2 + 2?")])

    add("tools-00", "tools", [u("What's the weather in Paris?")], tools=[WEATHER])
    add("tools-01", "tools", [s("You are a research assistant."), u("Find documents about GPU scheduling from 2025.")], tools=[WEATHER, SEARCH, CALC])
    add("tools-02", "tools", [u("What time is it?")], tools=[NOARGS])
    add("tools-03", "tools", [u("Traduis 'bonjour' en japonais.")], tools=[UNICODE_TOOL])
    add("tools-nothink", "tools", [u("Compute 2**10.")], tools=[CALC], thinking=False)
    add("toolcall-00", "tool-call", [u("What's the weather in Paris?"), a("", tool_calls=[call("call_1", "get_weather", {"city": "Paris"})]), t("call_1", "{\"temperature\": 18, \"condition\": \"cloudy\"}")], tools=[WEATHER])
    add("toolcall-01", "tool-call", [u("Weather in Tokyo in fahrenheit, and compute 3*7."), a("Let me check both.", reasoning="Two independent calls.", tool_calls=[call("c1", "get_weather", {"city": "Tokyo", "unit": "fahrenheit"}), call("c2", "calculate", {"expression": "3*7"})]), t("c1", "72F, sunny"), t("c2", "21")], tools=[WEATHER, CALC])
    add("toolcall-02", "tool-call", [u("Search for 'kernel fusion' with tags gpu and cuda, limit 5."), a("", tool_calls=[call("c9", "search_docs", {"query": "kernel fusion", "limit": 5, "filters": {"tags": ["gpu", "cuda"], "after": "2025-01-01"}})]), t("c9", "[{\"title\": \"Fusing norms\", \"score\": 0.91}]"), u("Summarize the top result.")], tools=[SEARCH])
    add("toolcall-03", "tool-call", [u("What time is it?"), a("", tool_calls=[call("t0", "get_time", {})]), t("t0", "2026-09-23T06:00:00Z")], tools=[NOARGS])
    add("toolcall-04", "tool-call", [u("Translate hello to French."), a("", tool_calls=[call("x1", "translate", {"text": "hello 世界", "target": "fr"})]), t("x1", "bonjour le monde")], tools=[UNICODE_TOOL], thinking=False)
    add("toolcall-05", "tool-call", [u("Weather in Oslo?"), a("", tool_calls=[call("w", "get_weather", {"city": "Oslo"})]), t("w", "-3C, snow"), a("It is -3C with snow in Oslo."), u("And in Rome?")], tools=[WEATHER])
    add("toolcall-06", "tool-call", [u("Compute these."), a("", tool_calls=[call("n1", "calculate", {"expression": "1/3"})]), t("n1", "0.3333333333333333")], tools=[CALC])
    add("toolcall-07", "tool-call", [u("Numbers and booleans as arguments."), a("", tool_calls=[call("b1", "search_docs", {"query": "x", "limit": 0, "filters": {"tags": [], "after": None}})]), t("b1", "[]")], tools=[SEARCH])
    add("tool-result-multiline", "tool-result", [u("Run the search."), a("", tool_calls=[call("m", "search_docs", {"query": "multi"})]), t("m", "line 1\nline 2\n\n  indented line 4"), u("Thanks.")], tools=[SEARCH])

    examples = json.load(open(os.environ["CHOPINFER_MIMO_FIXTURES"] + "/gsm8k/examples.json"))
    for ex in examples[:6]:
        add(f"gsm8k-{ex['index']:03d}", "gsm8k", [u(f"{GSM_INSTRUCTION}\n\n{ex['question']}")], thinking=True)
    return out

if __name__ == "__main__":
    convs = conversations()
    print(len(convs))

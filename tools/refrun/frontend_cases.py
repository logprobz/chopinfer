"""Seeded, model-neutral chat requests for the HF differential corpus.

Each case is an exact OpenAI chat request body, as `runtime-server` accepts
it (or refuses it), and the feature tags (`axis:value`) it exercises. The
shared axes cover messages, assistant turns, tool calls and results, tool
definitions and request flags; a model's `models/<m>/frontend_cases.py` adds
its own axes and fixed cases. Rows are chosen greedily so that every pair of
axis values appears in some row (pairwise), from `random.Random(seed)`, so
one seed always yields the same cases.

Efforts, thinking keywords, served media and limits come from the model's
frontend declaration (`models/<m>/compiler/src/frontend.json`, plus
`frontend-multimodal.json` where the model has one), never from a copy here.
"""

import copy
import json
import random
from pathlib import Path

# Tool-call argument values by tag, each with the JSON schema type that lets
# a schema-typed parser read the value back as it was sent.
ARGUMENTS = {
    "string": ("Paris <&>", "string"),
    "int": (3, "integer"),
    "float": (0.25, "number"),
    "float-1e-05": (1e-05, "number"),
    "float-1.0": (1.0, "number"),
    "float-1e+16": (1e16, "number"),
    "bool": (True, "boolean"),
    "null": (None, "null"),
    "object": ({"unit": "c", "hours": [1, 2.5, None], "ok": False}, "object"),
    "array": ([1, "two", 3.5, None], "array"),
    "non-ascii": ("北京 °C ü", "string"),
}

# The inline sources a render vector's media use: a render places media and
# never reads them (crates/runtime-host/src/frontend_check.rs).
MEDIA_PARTS = {
    "image_url": {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
    "video_url": {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA=="}},
    "audio_url": {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AA=="}},
    "input_audio": {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
}

USER_TEXT = {
    "string": "What is the weather in Paris?",
    "empty": "",
    "padded": "  Weather?  \n",
    "non-ascii": "天气怎么样？ Ünïcödé ½",
}

ASSISTANT_CONTENT = {
    "none": None,
    "empty": "",
    "text": "  It is 18 °C.  ",
    "whitespace": " \n ",
    "inline-think": "<think>inline plan</think> It is mild.",
}

DEFER = {"true": True, "false": False, "string-true": "true"}

RESULT_TEXTS = ["18 °C", "{\"temp\": 21}", " pong "]
CALL_TOOLS = ["get_weather", "convert", "ping"]


def load_declaration(model_dir):
    """The host sections of a model's frontend declaration: its
    `frontend.json` without `format`, with the multimodal section of
    `frontend-multimodal.json` when the model has one."""
    source = Path(model_dir) / "compiler" / "src"
    declaration = json.loads((source / "frontend.json").read_text())
    declaration.pop("format", None)
    multimodal = source / "frontend-multimodal.json"
    if multimodal.exists():
        declaration["multimodal"] = json.loads(multimodal.read_text())["multimodal"]
    return declaration


def load_parser(model_dir):
    """The `format.parser` section of a model's frontend declaration: the
    markers its completions are parsed by."""
    declaration = json.loads((Path(model_dir) / "compiler" / "src" / "frontend.json").read_text())
    return declaration.get("format", {}).get("parser")


def served_part_types(declaration):
    modalities = (declaration.get("multimodal") or {}).get("modalities", [])
    return [part for modality in modalities for part in modality["part_types"]]


def stop_texts(declaration):
    """The text of each declared stop token, in declaration order."""
    vocabulary = declaration["vocabulary"]
    names = {token["id"]: token["text"] for token in vocabulary["boundary_tokens"]}
    return [names[identifier] for identifier in vocabulary["stop_token_ids"]]


def undeclared_effort(declaration):
    reasoning = declaration["reasoning"]
    taken = {effort["name"] for effort in reasoning["efforts"]} | {reasoning.get("disable_value")}
    return next(name for name in ("minimal", "turbo", "extreme") if name not in taken)


# The top-level thinking object's types, for a declaration that switches
# thinking by it.
THINKING_OBJECT_TYPES = ("enabled", "adaptive", "disabled")


def shared_axes(declaration):
    """The model-neutral axes and their values, in a fixed order."""
    reasoning = declaration["reasoning"]
    thinking = ["absent"] + [f"{keyword}={value}" for keyword in reasoning["thinking_keywords"]
                             for value in ("true", "false")]
    if reasoning.get("switch") == "thinking-object":
        thinking += [f"object={kind}" for kind in THINKING_OBJECT_TYPES]
    return {
        "system": ["absent", "first", "late", "empty", "whitespace"],
        "developer": ["absent", "present"],
        "user": ["string", "parts", "empty", "padded", "non-ascii"],
        "message-tools": ["absent", "present"],
        "media": ["none", *MEDIA_PARTS],
        "assistant-content": list(ASSISTANT_CONTENT),
        "reasoning-field": ["absent", "reasoning", "reasoning_content", "both"],
        "calls": ["0", "1", "3"],
        "call-id": ["present", "empty", "missing"],
        "arguments-text": ["json", "invalid-json", "empty"],
        "arg": list(ARGUMENTS),
        "results": ["in-order", "reordered", "unmatched", "duplicate", "missing-id", "empty-id"],
        "result-content": ["string", "parts", "empty"],
        "tools": ["wrapped", "absent", "bare"],
        "tool-field": ["plain", "strict", "no-description", "null-description", "no-parameters",
                       "null-parameters", "key-order"],
        "defer": ["absent", "true", "false", "string-true"],
        "effort": ["absent", *[effort["name"] for effort in reasoning["efforts"]], "disable", "undeclared"],
        "effort-via": ["field", "kwargs"],
        "thinking": thinking,
        "clear_thinking": ["absent", "true", "false"],
        "preserve_thinking": ["absent", "true", "false"],
        "ends": ["user", "tool", "assistant", "continue", "continue-calls"],
        "add_generation_prompt": ["absent", "false", "true"],
    }


def pairwise_rows(axes, rng, candidates=8):
    """Rows (axis -> value) covering every pair of values of two axes,
    built greedily: each row starts from an uncovered pair and fills the
    other axes with the value covering the most uncovered pairs."""
    names = list(axes)
    values = [list(axes[name]) for name in names]
    uncovered = {(i, a, j, b) for i in range(len(names)) for j in range(i + 1, len(names))
                 for a in range(len(values[i])) for b in range(len(values[j]))}
    rows = []
    while uncovered:
        seeds = sorted(uncovered)
        best, best_gain = None, -1
        for _ in range(candidates):
            i, a, j, b = seeds[rng.randrange(len(seeds))]
            row = {i: a, j: b}
            order = [k for k in range(len(names)) if k not in row]
            rng.shuffle(order)
            for k in order:
                scores = []
                for c in range(len(values[k])):
                    gain = sum(((min(k, m), c if k < m else row[m], max(k, m), row[m] if k < m else c)
                                in uncovered) for m in row)
                    scores.append(gain)
                top = max(scores)
                row[k] = rng.choice([c for c, score in enumerate(scores) if score == top])
            gain = sum((i, row[i], j, row[j]) in uncovered
                       for i in range(len(names)) for j in range(i + 1, len(names)))
            if gain > best_gain:
                best, best_gain = row, gain
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                uncovered.discard((i, best[i], j, best[j]))
        rows.append({names[k]: values[k][best[k]] for k in range(len(names))})
    return rows


def exercised(tag, request):
    """Whether `request` carries what `tag` names, for the axes whose tag a
    request alone shows; None for the others."""
    axis, _, value = tag.partition(":")
    messages = request.get("messages") or []
    tools = [tool for tool in request.get("tools") or [] if isinstance(tool, dict)]
    functions = [tool.get("function", tool) for tool in tools]
    if axis == "tools":
        wrapped = [isinstance(tool.get("function"), dict) for tool in tools]
        return {"absent": not tools, "wrapped": bool(tools) and all(wrapped),
                "bare": bool(tools) and not any(wrapped)}.get(value)
    if axis == "tool-field":
        if not functions:
            return False
        first = functions[0]
        return {"plain": True, "strict": first.get("strict") is True, "no-description": "description" not in first,
                "null-description": "description" in first and first["description"] is None,
                "no-parameters": "parameters" not in first,
                "null-parameters": "parameters" in first and first["parameters"] is None,
                "key-order": next(iter(first), None) != "name"}.get(value)
    if axis == "defer":
        wanted = json.dumps(DEFER.get(value))
        return any("defer_loading" in function and json.dumps(function["defer_loading"]) == wanted
                   for function in functions)
    systems = [(index, message) for index, message in enumerate(messages) if message.get("role") == "system"]
    if axis == "system":
        contents = [message.get("content") for _, message in systems]
        return {"first": bool(systems) and systems[0][0] == 0 and bool(str(contents[0]).strip()),
                "late": any(index > 0 for index, _ in systems), "empty": "" in contents,
                "whitespace": any(isinstance(text, str) and text and not text.strip() for text in contents)}.get(value)
    if axis == "developer":
        return any(message.get("role") == "developer" for message in messages)
    if axis == "message-tools":
        return any(message.get("tools") for message in messages)
    if axis == "media":
        return any(isinstance(part, dict) and part.get("type") == value
                   for message in messages if isinstance(message.get("content"), list)
                   for part in message["content"])
    return None


# Spellings of a JSON schema type the served coercion reads as that type
# (frontend-format chatml_decode.rs `push_type`).
TYPE_ALIASES = {
    **dict.fromkeys(("str", "text", "varchar", "char", "enum"), "string"),
    **dict.fromkeys(("int", "int32", "int64", "uint", "uint32", "uint64", "long", "short", "unsigned"), "integer"),
    **dict.fromkeys(("float", "float32", "float64", "double"), "number"),
    "bool": "boolean", "dict": "object", **dict.fromkeys(("arr", "list", "sequence"), "array"),
}


# The JSON types the served coercion reads a value's text as, other than a
# string; it keeps the text of any other type name as a string.
NON_TEXT_TYPES = ("integer", "number", "boolean", "object", "array", "null")


def parameter_schema(parameters, name):
    """The schema of the parameter `name` under a tool's `parameters`, as
    the served parser resolves it (frontend-format chatml_decode.rs
    `parameter_schema_v1`): the declared property, else free text when
    pattern properties may type it, none when `additionalProperties` is
    false, the additional-property schema when it is one, and otherwise free
    text. None stands for free text or no schema; both read as a string."""
    if not isinstance(parameters, dict):
        return None
    properties = parameters.get("properties")
    if isinstance(properties, dict) and name in properties:
        return properties[name]
    if "patternProperties" in parameters:
        return None
    additional = parameters.get("additionalProperties")
    return additional if isinstance(additional, dict) else None


def schema_types(schema):
    """The JSON types a schema names, as the served coercion collects them
    (`schema_types`): its `type`, its `enum` values' types and those of its
    `anyOf`, `oneOf` and `allOf` branches. Empty when it names none."""
    types = []

    def add(kind):
        kind = kind.strip().lower()
        kind = TYPE_ALIASES.get(kind, kind)
        if kind not in types:
            types.append(kind)

    def collect(schema):
        if not isinstance(schema, dict):
            return
        kind = schema.get("type")
        for name in [kind] if isinstance(kind, str) else kind if isinstance(kind, list) else []:
            if isinstance(name, str):
                add(name)
        for value in schema.get("enum") if isinstance(schema.get("enum"), list) else []:
            add(json_type(value))
        for keyword in ("anyOf", "oneOf", "allOf"):
            for branch in schema.get(keyword) if isinstance(schema.get(keyword), list) else []:
                collect(branch)

    collect(schema)
    return types


def argument_loss(value, parameters, name):
    """Why the served parser cannot read the argument `name` back as `value`
    from its text, or None when it can: the parser reads a value by the one
    type its schema names, and reads it as a string when the schema names
    none. `("union", detail)` when the schema names several types, so the
    text can stand for more than one value; `("untyped", detail)` for a value
    that is not a string under no type; `("mismatch", detail)` for a value
    of another type than the one named. An integer has the type `number`,
    whose coercion reads its decimal text back as that integer, and a type
    name the coercion reads as no other JSON type is a string."""
    types = []
    for kind in schema_types(parameter_schema(parameters, name)):
        kind = kind if kind in NON_TEXT_TYPES else "string"
        if kind not in types:
            types.append(kind)
    kind = json_type(value)
    if len(types) > 1:
        return "union", f"{name} has the types {', '.join(types)}"
    if not types:
        return None if kind == "string" else ("untyped", f"{name} has no type, so its {kind} reads as a string")
    if types[0] != kind and (types[0], kind) != ("number", "integer"):
        return "mismatch", f"{name} is a {types[0]}, not the {kind} it holds"
    return None


def text_type(kind):
    """Whether the served coercion keeps a value of the type name `kind` as
    raw text: a string type or alias, or a name it reads as no other JSON
    type (chatml_decode.rs `text_type_v1`)."""
    return isinstance(kind, str) and TYPE_ALIASES.get(kind.strip().lower(), kind.strip().lower()) not in NON_TEXT_TYPES


def admits_string(schema):
    """Whether a value of `schema` could be a string, so the served value
    scan reads its text raw (chatml_decode.rs `admits_string_v1`)."""
    if not isinstance(schema, dict):
        return True
    kind = schema.get("type")
    if isinstance(kind, str):
        return text_type(kind)
    if isinstance(kind, list):
        return any(text_type(name) for name in kind)
    for keyword in ("anyOf", "oneOf"):
        if isinstance(schema.get(keyword), list):
            return any(admits_string(branch) for branch in schema[keyword])
    if isinstance(schema.get("enum"), list):
        return any(isinstance(value, str) for value in schema["enum"])
    if "const" in schema:
        return isinstance(schema["const"], str)
    return not any(keyword in schema for keyword in ("properties", "items", "minimum", "maximum"))


def json_scanned(function, name):
    """Whether the served value scan reads the argument `name` of a call to
    `function` (the offered function object) as JSON, where a marker inside
    a string literal is content (chatml_decode.rs `json_arguments_v1`): the
    tool constrains its arguments (`strict` is not false and it declares
    parameters) and the argument's schema admits no string. Every other
    value is raw text, which ends at the first marker that ends a value."""
    parameters = function.get("parameters")
    if function.get("strict") is False or parameters is None:
        return False
    schema = parameter_schema(parameters, name)
    return isinstance(schema, dict) and not admits_string(schema)


# The tool grammars with a function element, whose parameters a function
# close ends.
FUNCTION_GRAMMARS = ("xml-function-parameters-v1", "xml-typed-parameters-v1")


def raw_markers(parser):
    """The bare markers that end a raw (string) parameter value in the
    declared tool grammar: the parameter open and close and the function
    close in the function-parameter grammar, the parameter close alone in
    the typed one, and the value close and the call close in the key-value
    one."""
    if parser is None:
        return []
    grammar = parser.get("tool_grammar")
    if grammar == "xml-typed-parameters-v1":
        names = ("parameter_close",)
    elif grammar in FUNCTION_GRAMMARS:
        names = ("parameter_open", "parameter_close", "function_close")
    else:
        names = ("parameter_close", "tool_call_close")
    return [parser[name].strip() for name in names if parser.get(name, "").strip()]


def strings_of(value):
    """Every string inside a JSON value, keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings_of(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings_of(item)


def json_type(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return "array" if isinstance(value, list) else "object"


class _Builder:
    """Turns one pairwise row into a request body and the tags it realized."""

    def __init__(self, declaration, row, rng):
        self.declaration = declaration
        self.row = row
        self.rng = rng
        self.tags = set()

    def tag(self, axis, value=None):
        self.tags.add(f"{axis}:{self.row[axis] if value is None else value}")

    def user(self):
        form, media = self.row["user"], self.row["media"]
        if form == "parts" or media != "none":
            parts = [{"type": "text", "text": "Look at this "}]
            if media != "none":
                parts.append(copy.deepcopy(MEDIA_PARTS[media]))
                self.tag("media")
                if media not in served_part_types(self.declaration):
                    self.tag("media-unserved", media)
            parts.append({"type": "text", "text": " and say what it is."})
            self.tag("user", "parts")
            return {"role": "user", "content": parts}
        self.tag("user")
        return {"role": "user", "content": USER_TEXT[form]}

    def message_tools(self, message):
        if self.row["message-tools"] == "present":
            message["tools"] = [{"type": "function", "function": {
                "name": "lookup", "description": "Look a record up.",
                "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}}}}]
            self.tag("message-tools")
        return message

    def calls(self):
        """The assistant turn's calls and the tools that describe them. A
        continued final message has none (`continue-calls` keeps them)."""
        count = 0 if self.row["ends"] == "continue" else int(self.row["calls"])
        kinds = [self.row["arg"]] + [self.rng.choice(sorted(ARGUMENTS)) for _ in range(count - 1)]
        calls, tools = [], []
        id_form = self.row["call-id"]
        for index in range(count):
            name = CALL_TOOLS[index]
            value, schema = ARGUMENTS[kinds[index]]
            arguments = {"value": value, "city": "Paris"} if index == 0 else {"value": value}
            properties = {key: {"type": json_type(item)} for key, item in arguments.items()}
            call = {"id": f"call_{index}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}
            text = self.row["arguments-text"]
            if index == 0 and text != "json":
                call["function"]["arguments"] = {"invalid-json": "{\"value\": ", "empty": ""}[text]
            if index == 0 and id_form == "empty":
                call["id"] = ""
            elif index == 0 and id_form == "missing":
                del call["id"]
            calls.append(call)
            tools.append({"name": name, "description": f"The {name} tool.",
                          "parameters": {"type": "object", "properties": properties}})
            self.tag("arg", kinds[index])
            assert properties["value"]["type"] == schema
        self.tag("calls", str(count))
        if count:
            self.tag("call-id")
            self.tag("arguments-text")
        return calls, tools

    def assistant(self, calls):
        message = {"role": "assistant", "content": ASSISTANT_CONTENT[self.row["assistant-content"]]}
        self.tag("assistant-content")
        field = self.row["reasoning-field"]
        if field in ("reasoning", "both"):
            message["reasoning"] = "Plan the answer."
        if field in ("reasoning_content", "both"):
            message["reasoning_content"] = " Check the units. "
        self.tag("reasoning-field")
        if calls:
            message["tool_calls"] = calls
        return message

    def results(self, calls):
        identifiers = [call.get("id") for call in calls]
        form = self.row["results"]
        if form == "reordered" and len(calls) < 2:
            form = "in-order"
        order = list(range(len(calls)))
        if form == "reordered":
            order = order[::-1]
        messages = []
        for index in order:
            messages.append({"role": "tool", "tool_call_id": identifiers[index] or "",
                             "content": RESULT_TEXTS[index]})
        if form == "unmatched":
            messages[-1]["tool_call_id"] = "call_x"
        elif form == "duplicate":
            messages.append(dict(messages[0], content="again"))
        elif form == "missing-id":
            del messages[0]["tool_call_id"]
        elif form == "empty-id":
            messages[0]["tool_call_id"] = ""
        self.tags.add(f"results:{form}")
        content = self.row["result-content"]
        if content == "parts":
            messages[0]["content"] = [{"type": "text", "text": messages[0]["content"]},
                                      {"type": "text", "text": " (cached)"}]
        elif content == "empty":
            messages[0]["content"] = ""
        self.tag("result-content")
        return messages

    def tools(self, called):
        form = self.row["tools"]
        functions = list(called)
        extra = {"name": "clock", "description": "The time.", "parameters": {"type": "object", "properties": {}}}
        defer = self.row["defer"]
        if defer != "absent":
            extra["defer_loading"] = DEFER[defer]
        functions.append(extra)
        field = self.row["tool-field"]
        first = functions[0]
        if field == "strict":
            first["strict"] = True
        elif field == "no-description":
            first.pop("description")
        elif field == "null-description":
            first["description"] = None
        elif field == "no-parameters":
            first.pop("parameters")
        elif field == "null-parameters":
            first["parameters"] = None
        elif field == "key-order":
            functions[0] = dict(reversed(list(first.items())))
        self.tag("tools")
        if form == "absent":
            return None
        # The tool fields are tagged only when the tools are sent.
        self.tag("tool-field")
        if defer != "absent":
            self.tag("defer")
        if form == "bare":
            return functions
        return [{"type": "function", "function": function} for function in functions]

    def flags(self, request):
        kwargs = {}
        effort = self.row["effort"]
        if effort != "absent":
            reasoning = self.declaration["reasoning"]
            value = {"disable": reasoning.get("disable_value"),
                     "undeclared": undeclared_effort(self.declaration)}.get(effort, effort)
            if self.row["effort-via"] == "kwargs":
                kwargs["reasoning_effort"] = value
            else:
                request["reasoning_effort"] = value
            self.tag("effort-via")
            self.tag("effort")
        thinking = self.row["thinking"]
        if thinking != "absent":
            keyword, value = thinking.split("=")
            if keyword == "object":
                request["thinking"] = {"type": value}
            else:
                kwargs[keyword] = value == "true"
            self.tag("thinking")
        for keyword in ("clear_thinking", "preserve_thinking"):
            if self.row[keyword] != "absent":
                kwargs[keyword] = self.row[keyword] == "true"
                self.tag(keyword)
        if kwargs:
            request["chat_template_kwargs"] = kwargs

    def build(self):
        row = self.row
        messages = []
        system = row["system"]
        if system in ("first", "empty", "whitespace"):
            messages.append({"role": "system",
                             "content": {"first": "You are terse.", "empty": "", "whitespace": "  \n "}[system]})
        if row["developer"] == "present":
            messages.append({"role": "developer", "content": "Answer in English."})
            self.tag("developer")
        messages.append(self.message_tools(self.user()))
        calls, called = self.calls()
        ends = row["ends"]
        if ends == "tool" and not calls:
            ends = "user"
        assistant = self.assistant(calls)
        late = {"role": "system", "content": "Now answer in French."}
        if ends in ("assistant", "continue", "continue-calls"):
            if system == "late":
                messages += [{"role": "assistant", "content": "One moment."}, late,
                             {"role": "user", "content": "Go on."}]
            messages.append(assistant)
        else:
            if ends == "tool" and system == "late":
                messages.append(late)
            messages.append(assistant)
            if calls:
                messages += self.results(calls)
            if ends == "user":
                if system == "late":
                    messages.append(late)
                messages.append({"role": "user", "content": "Thanks. And tomorrow?"})
        if system != "absent":
            self.tag("system")
        self.tags.add(f"ends:{ends}")
        request = {"messages": messages}
        tools = self.tools(called)
        if tools is not None:
            request["tools"] = tools
        self.flags(request)
        if ends in ("continue", "continue-calls"):
            request["continue_final_message"] = True
            request["add_generation_prompt"] = False
        elif row["add_generation_prompt"] != "absent":
            request["add_generation_prompt"] = row["add_generation_prompt"] == "true"
            self.tag("add_generation_prompt")
        limits = self.declaration["limits"]
        assert len(messages) <= limits["maximum_messages"] and len(tools or []) <= limits["maximum_tools"]
        return request


# Values that stand for an absent field: they carry no tag.
NEUTRAL = ("absent", "none")


def isolated_values(declaration):
    """Shared values the transport refuses, or a model may refuse, before the
    render: each runs in one row whose other values are all ordinary, so a
    refusal never hides the rest of a row."""
    reasoning = declaration["reasoning"]
    isolated = {"tools:bare", "call-id:missing", "results:missing-id", "effort:undeclared", "effort:disable",
                "ends:continue-calls",
                "arguments-text:invalid-json", "arguments-text:empty", "message-tools:present"}
    served = served_part_types(declaration)
    isolated |= {f"media:{part}" for part in MEDIA_PARTS if part not in served}
    if not reasoning.get("allow_disabled", True):
        isolated |= {f"thinking:{keyword}=false" for keyword in reasoning["thinking_keywords"]}
        if reasoning.get("switch") == "thinking-object":
            isolated.add("thinking:object=disabled")
    return isolated


# The values an isolated value runs among: none the transport or a model is
# known to refuse, and the calls, results and tools most isolated values act on.
CLEAN_ROW = {"ends": "user", "calls": "1", "media": "none", "user": "string", "message-tools": "absent",
             "developer": "absent", "tools": "wrapped", "tool-field": "plain", "arguments-text": "json",
             "call-id": "present", "results": "in-order", "effort": "absent", "thinking": "absent",
             "add_generation_prompt": "absent"}

# A plain row ending in an assistant turn of each tool-call, reasoning and
# content shape, so every shape has a round-trip vector.
ROUND_ROW = {"ends": "assistant", "media": "none", "user": "string", "message-tools": "absent", "developer": "absent",
             "tools": "wrapped", "tool-field": "plain", "arguments-text": "json", "call-id": "present",
             "effort": "absent", "thinking": "absent", "add_generation_prompt": "absent"}
ROUND_SHAPES = [{"calls": calls, "reasoning-field": reasoning, "assistant-content": content}
                for calls in ("0", "1", "3") for reasoning in ("absent", "reasoning") for content in ("empty", "text")]


def effort_cases(declaration):
    """Fixed cases at the boundaries of the reasoning controls, which pairwise
    rows set one at a time: a top-level `reasoning_effort` with a nested one
    that differs (every ordered pair of declared efforts, and each declared
    effort over an undeclared one), and the top-level disabling value, alone
    and over a declared effort. Each is `(id, request, tags)`; its
    `fixed:<id>` tag keeps it through minimization."""
    reasoning = declaration["reasoning"]
    efforts = [effort["name"] for effort in reasoning["efforts"]]
    undeclared = undeclared_effort(declaration)
    pairs = [(top, nested) for top in efforts for nested in efforts if nested != top]
    pairs += [(top, undeclared) for top in efforts]
    pairs += [(reasoning["disable_value"], None), (reasoning["disable_value"], efforts[0])]

    def tag(value):
        return {reasoning["disable_value"]: "disable", undeclared: "undeclared"}.get(value, value)

    for top, nested in pairs:
        identifier = f"s-effort-{tag(top)}" + (f"-over-{tag(nested)}" if nested is not None else "")
        request = {"messages": [{"role": "user", "content": "Go."}], "reasoning_effort": top}
        tags = {f"effort:{tag(top)}", f"fixed:{identifier}"}
        if nested is None:
            tags.add("effort-via:field")
        else:
            request["chat_template_kwargs"] = {"reasoning_effort": nested}
            tags |= {"effort-via:both", f"nested-effort:{tag(nested)}"}
        yield identifier, request, tags


# Tool parameter schemas and an argument of each, by the tag `schema:<form>`
# names: forms the parser reads back (typed additional properties, a
# required name outside the properties, untyped strings, string enums) and
# accepted forms it cannot (unions, untyped values that are not strings).
SCHEMA_FORMS = {
    "additional-integer": ({"type": "object", "additionalProperties": {"type": "integer"}}, {"x": 42}),
    "additional-object": ({"type": "object", "additionalProperties": {"type": "object"}},
                          {"x": {"a": [1, True, None], "b": "text"}}),
    "additional-string": ({"type": "object", "additionalProperties": {"type": "string"}}, {"x": "42"}),
    "additional-required": ({"type": "object", "properties": {"y": {"type": "string"}}, "required": ["x", "y"],
                             "additionalProperties": {"type": "integer"}}, {"x": 7, "y": "z"}),
    "additional-any": ({"type": "object", "additionalProperties": True}, {"x": 42}),
    "pattern-integer": ({"type": "object", "patternProperties": {"^x$": {"type": "integer"}}}, {"x": 42}),
    "untyped-string": ({"type": "object", "properties": {"x": {}}}, {"x": "text"}),
    "untyped-integer": ({"type": "object", "properties": {"x": {}}}, {"x": 42}),
    "enum-string": ({"type": "object", "properties": {"x": {"enum": ["a", "b"]}}}, {"x": "a"}),
    "number-integer": ({"type": "object", "properties": {"x": {"type": "number"}}}, {"x": 42}),
    "additional-number-integer": ({"type": "object", "additionalProperties": {"type": "number"}}, {"x": 42}),
    "union-string": ({"type": "object", "properties": {"x": {"type": ["string", "integer"]}}, "required": ["x"]},
                     {"x": "42"}),
    "union-integer": ({"type": "object", "properties": {"x": {"type": ["string", "integer"]}}, "required": ["x"]},
                      {"x": 42}),
    "any-of": ({"type": "object", "properties": {"x": {"anyOf": [{"type": "integer"}, {"type": "null"}]}}},
               {"x": None}),
}

# String argument values by the tag `value:<form>`: empty, newlines at
# either end (a template's own framing is newlines too), CRLF, and a
# decomposed character a normalizing tokenizer composes.
STRING_VALUES = {
    "empty": "", "lf": "\n", "crlf": "\r\n", "boundary-newlines": "\n\ntext\n\n", "leading-lf": "\ntext",
    "trailing-lf": "text\n", "inner-crlf": "a\r\nb\r\n", "decomposed": "e\u0301 \u5317\u4eac",
}


# String values whose text looks like JSON, under each string type name
# the coercion reads as a string, by the tag `string-type:<name>`: a value
# the parser must keep as raw text, not scan as JSON.
STRING_TYPE_VALUES = {
    "string": "\"quoted", "str": "\"\u00e9", "text": "{\"a\": 1}", "varchar": "[1, 2]", "char": "42",
    "enum": "\"x\"", "str-decomposed": "\"e\u0301",
}


def argument_cases(parser=None):
    """Fixed cases whose last turn is one call to `f`, ending the request,
    for the round trip: each schema form (`SCHEMA_FORMS`), each string value
    (`STRING_VALUES`), a JSON-looking value under each string type name
    (`STRING_TYPE_VALUES`), and each marker that ends a raw value in the declared
    tool grammar (`raw_markers` of `parser`, the model's format.parser) as a
    string's whole value. Each is `(id, request, tags)`."""
    string = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    forms = [(f"s-schema-{form}", parameters, arguments, f"schema:{form}")
             for form, (parameters, arguments) in SCHEMA_FORMS.items()]
    forms += [(f"s-value-{form}", string, {"x": value}, f"value:{form}") for form, value in STRING_VALUES.items()]
    forms += [(f"s-string-type-{name}", {"type": "object", "properties": {"x": {"type": name.split("-")[0]}},
                                         "required": ["x"]}, {"x": value}, f"string-type:{name}")
              for name, value in STRING_TYPE_VALUES.items()]
    forms += [(f"s-value-marker-{index}", string, {"x": marker}, "value:marker")
              for index, marker in enumerate(raw_markers(parser))]
    for identifier, parameters, arguments, tag in forms:
        call = {"id": "call_0", "type": "function",
                "function": {"name": "f", "arguments": json.dumps(arguments, ensure_ascii=False)}}
        request = {"messages": [{"role": "user", "content": "Go."},
                                {"role": "assistant", "content": "", "tool_calls": [call]}],
                   "tools": [{"type": "function", "function": {"name": "f", "parameters": parameters}}],
                   "add_generation_prompt": False}
        yield identifier, request, {tag, f"fixed:{identifier}", "ends:assistant", "calls:1", "tools:wrapped"}


def _case(declaration, cases_module, model_axes, row, rng):
    builder = _Builder(declaration, row, rng)
    request = builder.build()
    for axis in model_axes:
        builder.tags.update(cases_module.apply_feature(request, axis, row[axis], rng) or ())
    # A model feature may take away what a shared tag names (a reference
    # case that drops the offered tools).
    tags = {tag for tag in builder.tags if exercised(tag, request) is not False}
    if not request.get("tools"):
        tags.add("tools:absent")
    return request, tags


def generate(declaration, cases_module=None, seed=0, parser=None):
    """Every case for one model: pairwise rows over the ordinary values of
    the shared axes and the model's own, a row for each value no row
    realized, a row for each isolated value, the shared fixed cases
    (`effort_cases`, and `argument_cases` with the model's format.parser
    section `parser`), then the model's fixed cases.
    Each case is `{"id", "request", "features"}`, its features sorted."""
    rng = random.Random(seed)
    axes = shared_axes(declaration)
    model_axes = {}
    isolated = isolated_values(declaration)
    if cases_module is not None:
        if hasattr(cases_module, "features"):
            model_axes = cases_module.features(declaration)
        isolated |= set(getattr(cases_module, "ISOLATED", ()))
    axes.update(model_axes)
    ordinary = {axis: [value for value in values if f"{axis}:{value}" not in isolated]
                for axis, values in axes.items()}
    rows = pairwise_rows(ordinary, rng)
    built = [_case(declaration, cases_module, model_axes, row, rng) for row in rows]
    for axis, values in axes.items():
        for value in values:
            tag = f"{axis}:{value}"
            if value in NEUTRAL or (tag not in isolated and any(tag in tags for _, tags in built)):
                continue
            start = rng.randrange(len(rows))
            for offset in range(len(rows)):
                base = rows[(start + offset) % len(rows)]
                row = {**base, **(CLEAN_ROW if tag in isolated else {}), axis: value}
                request, tags = _case(declaration, cases_module, model_axes, row, rng)
                if tag in tags:
                    built.append((request, tags))
                    break
            else:
                raise ValueError(f"no row realizes {tag}")
    for shape in ROUND_SHAPES:
        row = {**rows[rng.randrange(len(rows))], **ROUND_ROW, **shape}
        built.append(_case(declaration, cases_module, model_axes, row, rng))
    cases = [{"id": f"g{index:04d}", "request": request, "features": sorted(tags)}
             for index, (request, tags) in enumerate(built)]
    # Fixed shared cases follow the seeded ones, so they leave those as they are.
    for case_id, request, features in (*effort_cases(declaration), *argument_cases(parser)):
        cases.append({"id": case_id, "request": request, "features": sorted(features)})
    if cases_module is not None and hasattr(cases_module, "cases"):
        for case_id, request, features in cases_module.cases(declaration):
            cases.append({"id": case_id, "request": request, "features": sorted(features)})
    identifiers = [case["id"] for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("case ids must be unique")
    return cases

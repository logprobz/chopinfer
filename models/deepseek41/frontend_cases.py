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
import math

PIN = {
    "revision": "1bae0e1088993b9b82de08f59de5be5a34e42170",
    "source": "encoding/encoding.py",
    "source_sha256": "13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2",
    "checkpoint_env": "CHOPINFER_DEEPSEEK_CHECKPOINT",
}

# The published recipe the corpus is compared with at generation, by
# recipe_reference.py under CHOPINFER_DEEPSEEK_RECIPE_PYTHON
# (tools/refrun/README.md, "Frontend corpus recipe oracle").
RECIPE = {
    "package": "deepseek-recipe",
    "version": "0.2.0",
    "wheel": "deepseek_recipe-0.2.0-cp310-abi3-manylinux_2_28_x86_64.whl",
    "wheel_sha256": "259812bf382bc82668b672707c5922095cc4f5f60a6c514ff89640bb62a746f7",
    "installed_sha256": "84ef54ed1b08cd105614efc36d0722c4f8b741f080ca4268bd90d8898b8856b0",
    "python_env": "CHOPINFER_DEEPSEEK_RECIPE_PYTHON",
    "oracle": "recipe_reference.py",
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

# The request fields the encoder renders beyond the HF input, as request
# mapping reads them.
INPUT_FIELDS = ("response_format", "tool_choice")

# What a forced tool choice's prompt ends with (the declaration's
# tools.choice forced_prefill): the opening of the calls block, as
# deepseek-recipe 0.2.0 prefills it (dsv41.rs lines 307-310).
FORCED_PREFILL = "\n\n<｜DSML｜ calls>\n"

# Tool-call arguments reach the encoder as the text sent: it reads them
# itself, up to twice (lines 533-544).
RAW_TOOL_ARGUMENTS = True

# The protocol efforts the encoder does not name, as deepseek-recipe 0.2.0
# renders them and the served format reads them (reasoning.rs lines 37-48,
# dsv41.rs line 24): minimal as low, medium and xhigh as high, ultra as max.
EFFORT_ALIASES = {"minimal": "low", "medium": "high", "xhigh": "high", "ultra": "max"}

EOS = "<｜end▁of▁sentence｜>"

# Every assistant turn ends with the end-of-sentence token (encoding.py
# line 60), so a history render holds it; these apply only without it.
ENDINGS = {"tool_calls": [EOS], "answer": [EOS]}

# Refused by this frontend: each runs in one row of otherwise ordinary values.
# A strict first tool beside the shared rows' non-strict clock tool is
# refused, as deepseek-recipe 0.2.0 refuses strict and non-strict tools
# together.
ISOLATED = {"tool-field:strict"}

# The markers the vendor parser refuses inside reasoning or content
# (encoding.py lines 981-983).
SPECIAL = ("<｜begin▁of▁sentence｜>", EOS, "<think>", "</think>", "｜DSML｜")


def thinking(keywords):
    """The mode the served frontend selects: the first declared thinking
    keyword the request sets, else thinking (`default_enabled`). The
    declaration switches by the top-level thinking object, which the HF
    input carries as both keywords, and drops the client's own keywords."""
    for keyword in ("enable_thinking", "thinking"):
        if isinstance(keywords.get(keyword), bool):
            return keywords[keyword]
    return True


def encoder_message(message):
    """One HF message as the encoder reads it: a media item back in OpenAI's
    `image_url` form, which `_extract_image` (lines 305-320) reads. A file
    part's data URL is an image, as deepseek-recipe 0.2.0 maps it
    (convert.rs lines 332-353), from OpenAI's nested file object or the
    recipe's flat fields; a file ID is refused, as the recipe refuses it."""
    message = copy.deepcopy(message)
    if isinstance(message.get("content"), list):
        parts = []
        for part in message["content"]:
            if isinstance(part, dict) and part.get("type") in ("image", "video", "audio"):
                kind = part["type"]
                part = {"type": f"{kind}_url", f"{kind}_url": {"url": part.get(kind)}}
            elif isinstance(part, dict) and part.get("type") == "file":
                file = part.get("file") if isinstance(part.get("file"), dict) else part
                if file.get("file_id") is not None or not isinstance(file.get("file_data"), str):
                    raise ValueError("file must have file_data and no file_id")
                part = {"type": "image_url", "image_url": {"url": file["file_data"]}}
            parts.append(part)
        message["content"] = parts
    return message


def recipe_tool(tool, encoder):
    """An offered tool as deepseek-recipe 0.2.0 maps it (convert.rs lines
    265-274, client_tools.rs lines 24-37, v4/mod.rs lines 45-52): its name,
    its description or "", its parameters as sent (None when absent or
    null) and whether it is strict, every other field dropped;
    `recipe_function` checks the parameters. A
    tool in a namespace (on the tool, else on its function) is named and
    described as the encoder's `tools_from_openai_format` (lines 84-98)
    names it, the transport's qualified name (`namespace` in
    frontend-divergences.json), and then mapped the same way."""
    sent = tool.get("function") or {}
    function = {"name": sent.get("name"), "description": sent.get("description") or ""}
    wrapped = {"type": "function", "function": function}
    if tool.get("namespace") is not None:
        wrapped["namespace"] = tool["namespace"]
    elif sent.get("namespace") is not None:
        function["namespace"] = sent["namespace"]
    if "namespace" in wrapped or "namespace" in function:
        function = encoder["tools_from_openai_format"]([wrapped])[0]
    return {"name": function["name"], "description": function.get("description") or "",
            "parameters": sent.get("parameters"), "strict": sent.get("strict") is True}


def recipe_function(tool):
    """The tool object the encoder reads for a `recipe_tool`, its parameters
    checked as deepseek-recipe 0.2.0 checks them (validation.rs lines
    209-240): a strict tool's normalized (`strict_tool_parameters`), and a
    non-strict tool's present ones declaring type object, else `{}`."""
    parameters = tool["parameters"]
    if tool["strict"]:
        parameters = strict_tool_parameters(parameters)
    elif parameters is not None:
        kind = parameters.get("type") if isinstance(parameters, dict) else None
        if kind != "object":
            raise ValueError(f"Invalid schema for function '{tool['name']}': schema must be a JSON Schema of "
                             f"'type: \"object\"', got 'type: {compact_json(kind)}'.")
    else:
        parameters = {}
    return {"type": "function", "function": {
        "name": tool["name"], "description": tool["description"], "parameters": parameters}}


def compact_json(value):
    """`value` as serde_json's `Display` writes it."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# Strict JSON Schema normalization, as deepseek-recipe 0.2.0 normalizes a
# strict tool's parameters and checks a strict json_schema response format
# (json_schema.rs lines 1-287, validation.rs lines 200-264), and as
# models/deepseek41/contract/src/frontend/strict_schema.rs ports it: a node
# is read as the recipe deserializes it, with serde's messages, then
# normalized, and written in the recipe's key order.

NO_NODE = "one of `type`, `anyOf`, `$ref` field is required"
SCHEMA_KINDS = ("object", "string", "number", "integer", "boolean", "array", "null")
SIMPLE_KINDS = ("string", "number", "integer", "boolean", "null")
STRING_FORMATS = ("email", "hostname", "ipv4", "ipv6", "uuid")
STRING_KEYWORDS = ("pattern", "format", "enum")
NUMERIC_KEYWORDS = ("const", "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf")
I64 = range(-(1 << 63), 1 << 63)
U64 = range(0, 1 << 64)


class SchemaError(ValueError):
    """A strict schema the recipe refuses."""


def float_text(value):
    """A float as the recipe's serde_json writes it in a refusal (the `zmij`
    crate's format), as strict_schema.rs writes it: the shortest digits that
    read back, ties to even as `repr` and serde_json break them, in fixed
    notation with at least one fraction digit when the decimal exponent is
    from -5 through 15 (`0.00001`, `100.0`), else in exponent notation with
    a signed exponent (`1e-6`, `1.5e+16`)."""
    if value != value or value in (float("inf"), float("-inf")):
        return {"nan": "NaN"}.get(repr(value), repr(value))
    sign = "-" if math.copysign(1.0, value) < 0 else ""
    mantissa, _, exponent = repr(abs(value)).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits, exponent = whole + fraction, int(exponent or 0) + len(whole) - 1
    while len(digits) > 1 and digits[0] == "0":
        digits, exponent = digits[1:], exponent - 1
    digits = digits.rstrip("0") or "0"
    if digits == "0":
        exponent = 0
    if not -5 <= exponent <= 15:
        fraction = f".{digits[1:]}" if len(digits) > 1 else ""
        return f"{sign}{digits[0]}{fraction}e{'-' if exponent < 0 else '+'}{abs(exponent)}"
    if exponent < 0:
        return f"{sign}0.{'0' * (-exponent - 1)}{digits}"
    if exponent + 1 >= len(digits):
        return f"{sign}{digits}{'0' * (exponent + 1 - len(digits))}.0"
    return f"{sign}{digits[:exponent + 1]}.{digits[exponent + 1:]}"


def invalid_type(value, expected):
    """serde's `invalid type` refusal, the value as serde_json spells it and
    a string quoted as JSON, as strict_schema.rs writes it."""
    if value is None:
        unexpected = "null"
    elif isinstance(value, bool):
        unexpected = f"boolean `{'true' if value else 'false'}`"
    elif isinstance(value, int) and (value in I64 or value in U64):
        unexpected = f"integer `{value}`"
    elif isinstance(value, (int, float)):
        unexpected = f"floating point `{float_text(float(value))}`"
    elif isinstance(value, str):
        unexpected = f"string {json.dumps(value, ensure_ascii=False)}"
    elif isinstance(value, list):
        unexpected = "sequence"
    else:
        unexpected = "map"
    return SchemaError(f"invalid type: {unexpected}, expected {expected}")


def unknown_variant(name, variants):
    return SchemaError(f"unknown variant `{name}`, expected one of {', '.join(f'`{variant}`' for variant in variants)}")


def schema_variant(value, variants):
    """A unit enum value: a variant name, or a map of one variant name to
    null."""
    if isinstance(value, str):
        if value not in variants:
            raise unknown_variant(value, variants)
        return value
    if isinstance(value, dict):
        if len(value) != 1:
            raise SchemaError("invalid value: map, expected map with a single key")
        [(name, unit)] = value.items()
        name = schema_variant(name, variants)
        if unit is not None:
            raise invalid_type(unit, "unit")
        return name
    raise invalid_type(value, "string or map")


def schema_string(value):
    if not isinstance(value, str):
        raise invalid_type(value, "a string")
    return value


def schema_sequence(value, read):
    if not isinstance(value, list):
        raise invalid_type(value, "a sequence")
    return [read(item) for item in value]


def schema_nodes(value):
    if not isinstance(value, dict):
        raise invalid_type(value, "a map")
    return [(key, read_schema_node(item)) for key, item in value.items()]


def schema_number(value, kind):
    if kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise invalid_type(value, "a JSON number")
    elif isinstance(value, int) and not isinstance(value, bool) and value not in I64 and value in U64:
        raise SchemaError(f"invalid value: integer `{value}`, expected i64")
    elif isinstance(value, bool) or not isinstance(value, int) or value not in I64:
        raise invalid_type(value, "i64")
    return value


def labeled(name, read, value):
    try:
        return read(value)
    except SchemaError as error:
        raise SchemaError(f"field `{name}`: {error}") from None


def optional(value, read):
    return None if value is None else read(value)


def read_schema_node(value):
    """`JsonSchema`: a `type` array, a single type, an `anyOf` list or a
    `$ref`, by the field present, as ("types" | "type" | "anyOf" | "ref",
    ...)."""
    if not isinstance(value, dict):
        raise SchemaError(NO_NODE)
    description = lambda: labeled("description", lambda item: optional(item, schema_string), value.get("description"))
    if isinstance(value.get("type"), list):
        types = labeled("type", lambda item: schema_sequence(item, lambda kind: schema_variant(kind, SIMPLE_KINDS)),
                        value["type"])
        return ("types", types, description())
    if "type" in value:
        return ("type", read_typed_node(value))
    if "anyOf" in value:
        return ("anyOf", labeled("anyOf", lambda item: schema_sequence(item, read_typed_node), value["anyOf"]),
                description())
    if "$ref" in value:
        return ("ref", labeled("$ref", schema_string, value["$ref"]))
    raise SchemaError(NO_NODE)


def schema_tag(tag):
    if not isinstance(tag, str):
        raise invalid_type(tag, "variant identifier")
    if tag not in SCHEMA_KINDS:
        raise unknown_variant(tag, SCHEMA_KINDS)
    return tag


def read_typed_node(value):
    """`JsonSchemaType`, tagged by `type`: the tag first, then the type's
    own fields in their arrival order, every other key dropped."""
    if isinstance(value, list):
        if not value:
            raise SchemaError("missing field `type`")
        schema_tag(value[0])
        raise invalid_type(value, "internally tagged enum JsonSchemaType")
    if not isinstance(value, dict):
        raise invalid_type(value, "internally tagged enum JsonSchemaType")
    if "type" not in value:
        raise SchemaError("missing field `type`")
    kind = schema_tag(value["type"])
    node = {"kind": kind, "description": None, "properties": None, "$def": None, "additionalProperties": None,
            "required": None, "keywords": {}, "items": None}
    items = ()
    for key, item in value.items():
        if key == "description":
            node["description"] = optional(item, schema_string)
        elif kind == "object" and key in ("properties", "$def"):
            node[key] = optional(item, schema_nodes)
        elif kind == "object" and key == "additionalProperties":
            if item is not None and not isinstance(item, bool):
                raise invalid_type(item, "a boolean")
            node[key] = item
        elif kind == "object" and key == "required":
            node[key] = optional(item, lambda names: schema_sequence(names, schema_string))
        elif kind == "array" and key == "items":
            items = (read_schema_node(item),)
        elif item is None:
            continue
        elif kind == "string" and key in STRING_KEYWORDS:
            node["keywords"][key] = (schema_string(item) if key == "pattern" else
                                     schema_variant(item, STRING_FORMATS) if key == "format" else
                                     schema_sequence(item, schema_string))
        elif kind in ("number", "integer") and key in NUMERIC_KEYWORDS:
            node["keywords"][key] = schema_number(item, kind)
    if kind == "array":
        if not items:
            raise SchemaError("missing field `items`")
        node["items"] = items[0]
    order = (*STRING_KEYWORDS, *NUMERIC_KEYWORDS)
    node["keywords"] = dict(sorted(node["keywords"].items(), key=lambda entry: order.index(entry[0])))
    return node


def normalize_schema_node(node, allow_empty_object):
    """`JsonSchema::normalize` (json_schema.rs lines 158-166)."""
    if node[0] == "type":
        normalize_typed_node(node[1], allow_empty_object)
    elif node[0] == "anyOf":
        for item in node[1]:
            normalize_typed_node(item, allow_empty_object)


def normalize_typed_node(node, allow_empty_object):
    """`JsonSchemaType::normalize` (json_schema.rs lines 188-253)."""
    if node["kind"] == "array":
        normalize_schema_node(node["items"], False)
    if node["kind"] != "object":
        return
    properties = node["properties"] or []
    if not allow_empty_object and not properties:
        raise SchemaError("An object with no properties is not allowed.")
    for _, value in properties:
        normalize_schema_node(value, False)
    for _, value in node["$def"] or []:
        normalize_schema_node(value, False)
    names = [name for name, _ in properties]
    if node["required"] is not None:
        seen = set()
        for key in node["required"]:
            if key not in names:
                raise SchemaError(f"Required property '{key}' not found in object")
            if key in seen:
                raise SchemaError(f"Duplicate required property '{key}'")
            seen.add(key)
        if len(seen) != len(names):
            raise SchemaError("Required properties must match all properties in the object")
    if node["additionalProperties"] is True:
        raise SchemaError("additionalProperties should not be true")
    node.update(properties=properties, required=names, additionalProperties=False)


def write_schema_node(node):
    """The node as the recipe serializes it."""
    if node[0] == "type":
        return write_typed_node(node[1])
    if node[0] == "ref":
        return {"$ref": node[1]}
    written = {"type": node[1]} if node[0] == "types" else {"anyOf": [write_typed_node(item) for item in node[1]]}
    if node[2] is not None:
        written["description"] = node[2]
    return written


def write_typed_node(node):
    written = {"type": node["kind"]}
    description = {} if node["description"] is None else {"description": node["description"]}
    if node["kind"] == "object":
        written["properties"] = (None if node["properties"] is None else
                                 {key: write_schema_node(value) for key, value in node["properties"]})
        written["additionalProperties"] = node["additionalProperties"]
        written["required"] = node["required"]
        written.update(description)
        if node["$def"] is not None:
            written["$def"] = {key: write_schema_node(value) for key, value in node["$def"]}
    elif node["kind"] == "array":
        written["items"] = write_schema_node(node["items"])
        written.update(description)
    else:
        written.update(description)
        written.update(node["keywords"])
    return written


def normalize_strict_schema(schema, allow_empty_object, read_error, top_level_error):
    """`normalize_strict_schema` (json_schema.rs lines 273-287)."""
    try:
        node = read_schema_node(schema)
    except SchemaError as error:
        raise SchemaError(f"{read_error}{error}") from None
    normalize_schema_node(node, allow_empty_object)
    if node[0] != "type" or node[1]["kind"] != "object":
        raise SchemaError(top_level_error)
    return write_schema_node(node)


def strict_tool_parameters(parameters):
    """A strict tool's parameters normalized, a missing schema being the
    empty object (validation.rs lines 214-225)."""
    if parameters is None:
        parameters = {"type": "object", "properties": {}, "additionalProperties": False, "required": []}
    return normalize_strict_schema(parameters, True, "Invalid tool parameters schema : ",
                                   "The top level of the function parameters schema must be an object.")


def strict_response_schema(schema):
    """A strict json_schema's schema checked as the recipe checks it
    (validation.rs lines 249-260)."""
    return normalize_strict_schema(schema, False, "Invalid json schema: ",
                                   "The top level of the json schema must be an object.")


# The format's refusals of tools and response formats deepseek-recipe 0.2.0
# refuses (output_constraint.rs lines 34-46), and of a json_schema without
# the schema its request schema requires.
STRICT_WITH_NON_STRICT = "Cannot use strict tool with non-strict tool in the same request"
FORMAT_WITH_STRICT_TOOLS = "response_format should not be used with tools"
SCHEMA_REQUIRED = "response_format.json_schema.schema is required"


def recipe_format(response_format):
    """What deepseek-recipe 0.2.0 renders for a response format (dsv41.rs
    lines 253-257): json_object as `{"type": "json_object"}`, json_schema
    as its schema, which a strict one must normalize (`strict_response_schema`),
    and nothing for text and regex."""
    if not isinstance(response_format, dict):
        return None
    kind = response_format.get("type")
    if kind == "json_object":
        return {"type": "json_object"}
    if kind != "json_schema":
        return None
    fields = response_format.get("json_schema")
    fields = fields if isinstance(fields, dict) else {}
    if fields.get("schema") is None:
        raise ValueError(SCHEMA_REQUIRED)
    if fields.get("strict") is True:
        strict_response_schema(fields["schema"])
    return fields["schema"]


# The text that starts a system message deepseek-recipe strips
# (chat_completion/request/convert.rs lines 373-394), and the whitespace
# Rust's `trim_start` strips after it.
BILLING_HEADER = "x-anthropic-billing-header"
RUST_WHITESPACE = ("\t\n\x0b\x0c\r \x85\xa0\u1680" + "".join(map(chr, range(0x2000, 0x200b)))
                   + "\u2028\u2029\u202f\u205f\u3000")


def strip_billing_header(text):
    """System text without a leading billing header: the text from the first
    newline or `You are`, whichever comes first, without leading whitespace,
    or nothing without either."""
    if not text.startswith(BILLING_HEADER):
        return text
    starts = [index for index in (text.find("\n"), text.find("You are")) if index >= 0]
    return text[min(starts):].lstrip(RUST_WHITESPACE) if starts else ""


def text_part(part):
    return isinstance(part, dict) and part.get("type") == "text"


def strip_billing_parts(parts):
    """`strip_billing_header` over a system message's parts, as the recipe
    strips their text joined with blank lines, each image its token
    (convert.rs lines 291-357), and as frontend.rs's
    `strip_billing_header_blocks` does: the cut falls in the first part when
    it holds a newline or "You are", else at the blank line after it, and
    the whitespace after it runs through blank text parts to the next text
    or image."""
    if not parts or not text_part(parts[0]) or not parts[0].get("text", "").startswith(BILLING_HEADER):
        return parts
    kept = strip_billing_header(parts[0]["text"])
    if kept or len(parts) == 1:
        return [{**parts[0], "text": kept}, *parts[1:]]
    rest = list(parts[1:])
    while rest and text_part(rest[0]):
        text = rest[0].get("text", "").lstrip(RUST_WHITESPACE)
        if text:
            return [{**rest[0], "text": text}, *rest[1:]]
        rest.pop(0)
    return rest


def recipe_message(message):
    """One HF message as deepseek-recipe maps it for the encoder: a developer
    message is a system message, as the recipe maps a Responses developer
    message for V4.1 (responses/request/convert.rs lines 421-446), and a
    system message's billing header is removed. Its images stay in place,
    as the recipe reads a V4.1 system message's (convert.rs lines
    428-441)."""
    if message.get("role") not in ("system", "developer"):
        return message
    message = {**message, "role": "system"}
    content = message.get("content")
    if isinstance(content, list):
        message["content"] = strip_billing_parts(content)
    elif isinstance(content, str):
        message["content"] = strip_billing_header(content)
    return message


def empty_system(message):
    """Whether a message is a system message deepseek-recipe drops: its
    text, parts joined with each image as its token, is empty (convert.rs
    lines 428-440)."""
    if message.get("role") != "system":
        return False
    content = message.get("content")
    if isinstance(content, list):
        return not content or (len(content) == 1 and text_part(content[0]) and not content[0].get("text"))
    return not content


def encoder_render(namespace, messages, tools, add_generation_prompt, continue_final_message, keywords,
                   response_format=None, tool_choice=None):
    """The prompt `encode_messages` (lines 810-844) writes for one HF input,
    with the request mapped into messages as deepseek-recipe maps it (plan
    V4; `mapping` in frontend-divergences.json) and the served format
    serves it:
    - the mode from the top-level thinking object, else the top-level
      effort (`none` disables), as deepseek-recipe 0.2.0 resolves it
      (convert.rs lines 49-53, reasoning.rs lines 51-78) and the HF input
      carries it; the thinking keywords and the effort in
      `chat_template_kwargs` are ignored, as the recipe ignores that object;
      the effort, with a protocol alias read as the effort it names, applies
      in thinking mode only, and an enabled object over an effort of `none`
      thinks at the default effort;
    - `preserve_thinking` turns off `drop_thinking`;
    - a developer message is a system message, a system message's billing
      header is removed, and a system message left without text or images
      is dropped; a system message's images render in place;
    - a tool choice of none renders without the offered tools, a named one
      with only its tool, and a required or named one ends the prompt with
      the calls block's opening and is refused in thinking mode, as the
      recipe maps it (convert.rs lines 254-283, client_tools.rs lines
      83-104, dsv41.rs lines 307-310); without offered tools any choice but
      none is auto (convert.rs lines 262-264);
    - each tool left is the recipe's function object, a strict one's
      parameters normalized; strict and non-strict tools together, and a
      response format beside strict tools, are refused (recipe 0.2.0);
    - the tools and the response format go to the leading system message,
      an empty one when the request has none (README: tools are defined on
      the system message) and they render something; the response format
      renders as recipe 0.2.0 renders it (`recipe_format`), nothing for
      text and regex, and the encoder renders nothing for one Python reads
      as false;
    - a continued final assistant turn renders without its end token
      (`wo_eos`); the encoder writes the generation header itself, so
      `add_generation_prompt` changes nothing; in thinking mode a continued
      turn with no content and no calls keeps its reasoning open, as recipe
      0.2.0 renders a prefix turn (dsv41.rs lines 206-230), so the `</think>`
      the encoder closes it with is removed (`R4-final-assistant-turn`)."""
    mode = "thinking" if thinking(keywords) else "chat"
    effort = keywords.get("reasoning_effort") if mode == "thinking" else None
    effort = EFFORT_ALIASES.get(effort, effort) if isinstance(effort, str) else effort
    messages = [encoder_message(message) for message in map(recipe_message, messages) if not empty_system(message)]
    tools = [recipe_tool(tool, namespace) for tool in tools or []]
    named = tool_choice.get("function", {}).get("name") if isinstance(tool_choice, dict) else None
    forced = bool(tools) and (tool_choice == "required" or named is not None)
    if tool_choice == "none":
        tools = []
    elif named is not None:
        tools = [tool for tool in tools if tool["name"] == named]
    functions = [recipe_function(tool) for tool in tools]
    rendered_format = recipe_format(response_format)
    if any(tool["strict"] for tool in tools):
        if not all(tool["strict"] for tool in tools):
            raise ValueError(STRICT_WITH_NON_STRICT)
        if response_format is not None:
            raise ValueError(FORMAT_WITH_STRICT_TOOLS)
    if forced and mode == "thinking":
        raise ValueError("thinking mode does not support a required or named tool_choice")
    if continue_final_message:
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("continue_final_message requires a final assistant message")
        messages[-1]["wo_eos"] = True
    rendered_format = rendered_format if rendered_format else None
    if functions or rendered_format is not None:
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": ""})
        if functions:
            if messages[0].get("tools"):
                raise ValueError("tools are offered both by the request and by its system message")
            messages[0]["tools"] = functions
        if rendered_format is not None:
            messages[0]["response_format"] = rendered_format
    prompt = namespace["encode_messages"](
        messages, thinking_mode=mode, drop_thinking=not keywords.get("preserve_thinking", False),
        reasoning_effort=effort)
    if continue_final_message and mode == "thinking" and reasoning_alone(messages[-1]):
        if not prompt.endswith("</think>"):
            raise RuntimeError("the encoder did not close the continued turn's reasoning last")
        prompt = prompt.removesuffix("</think>")
    return prompt + FORCED_PREFILL if forced else prompt


def reasoning_alone(message):
    """Whether an assistant turn has no content, its parts' text joined as
    the encoder joins it (lines 387-393), and no calls."""
    content = message.get("content")
    if isinstance(content, list):
        content = "\n\n".join(part.get("text", "") for part in content if text_part(part))
    return not content and not message.get("tool_calls")


def turn_arguments(text):
    """A call's arguments as the encoder writes them (`encode_arguments_to_dsml`,
    lines 524-555): the text read as JSON up to twice, and an object
    whose every value is written as itself, else one string parameter
    named `arguments` holding the text."""
    value = text
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except ValueError:
            break
    return value if isinstance(value, dict) else {"arguments": text}


def expected_turn(message, context):
    """What the vendor's `parse_message_from_completion_text` (lines
    946-990) reads back from a generated assistant turn: the reasoning in
    thinking mode (the prompt leaves `<think>` open), the content, and each
    call's arguments as `decode_dsml_to_arguments` (lines 156-177) writes
    them from the DSML the encoder wrote, the object `json.dumps` writes.
    None for a turn whose text holds a marker the vendor parser refuses, and
    for a turn whose content ends with a newline before its calls: the
    vendor parser keeps it, while the served parser drops every newline
    before the calls block (`call-open-newlines`)."""
    reasoning = message.get("reasoning_content") if context["prompt"].endswith("<think>") else None
    content = message.get("content")
    if isinstance(content, list):
        content = "\n\n".join(part.get("text", "") for part in content
                              if isinstance(part, dict) and part.get("type") == "text")
    content = content or ""
    reasoning = reasoning or ""
    if any(marker in text for marker in SPECIAL for text in (content, reasoning)):
        return None
    if message.get("tool_calls") and content.endswith("\n"):
        return None
    calls = [{"name": call["function"]["name"],
              "arguments": json.dumps(turn_arguments(call["function"]["arguments"]), ensure_ascii=False)}
             for call in message.get("tool_calls") or []]
    return {"reasoning": reasoning, "content": content, "tool_calls": calls}


RESPONSE_FORMATS = {
    "json-schema": {"type": "json_schema", "json_schema": {
        "name": "answer", "strict": True,
        "schema": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                   "required": ["city", "days"], "additionalProperties": False}}},
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


def tool_image_cases():
    """Image results and the vendor's source order, including sorting across user parts."""
    def image(data):
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}

    def text(value):
        return {"type": "text", "text": value}

    def result(identifier, parts):
        return {"role": "tool", "tool_call_id": identifier, "content": parts}

    initial = [user("Look."), {"role": "assistant", "content": "", "tool_calls": [
        call("a", "look", {}), call("b", "look", {})]}]
    for name, results, order in [
        ("tool-image-single", [result("a", [image("AA==")])], ["AA=="]),
        ("tool-images-several", [result("a", [image("AA=="), image("AQ==")])], ["AA==", "AQ=="]),
        ("tool-images-mixed-text", [result("a", [text("before"), image("AA=="),
            text("between"), image("AQ=="), text("after")])], ["AA==", "AQ=="]),
        ("tool-images-sorted", [result("b", [text("second"), image("AA==")]),
            result("a", [text("first"), image("AQ=="), image("Ag==")])], ["AQ==", "Ag==", "AA=="]),
        ("tool-images-with-user", [result("b", [text("second"), image("AQ==")]),
            user([image("Ag==")]), result("a", [text("first"), image("Aw=="), image("BA=="), text("end")])],
            ["AA==", "Aw==", "BA==", "Ag==", "AQ=="]),
    ]:
        messages = copy.deepcopy(initial)
        if name == "tool-images-with-user":
            messages[0] = user([image("AA==")])
        yield name, {"messages": messages + results, "thinking": {"type": "disabled"}}, [
            f"data:image/png;base64,{data}" for data in order]


def image_format_cases():
    """Images as GIF and WebP data URLs and under other spellings of the JPEG
    type, and file parts carrying an image data URL, in user and tool
    messages, which the encoder renders as any image; and the file parts
    this format refuses."""
    def image(mime, data="AA=="):
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}

    def file(data_url, **fields):
        return {"type": "file", "file": {"file_data": data_url, **fields}}

    text = {"type": "text", "text": "Describe."}
    looked = [user("Look."), {"role": "assistant", "content": "", "tool_calls": [call("a", "look", {})]}]
    chat = {"thinking": {"type": "disabled"}}

    def result(parts):
        return {"role": "tool", "tool_call_id": "a", "content": parts}

    yield "image-gif", {"messages": [user([image("image/gif"), text])]}, {"image:gif"}
    yield "image-webp", {"messages": [user([image("image/webp"), text])]}, {"image:webp"}
    yield "image-jpg-spellings", {"messages": [user([image("image/jpg"), image("image/JPG", "AQ=="), text])]}, {
        "image:jpg"}
    yield "tool-images-gif-webp", {"messages": looked + [result([image("image/gif"), image("image/webp", "AQ==")])],
                                   **chat}, {"tool-results:images", "image:gif", "image:webp"}
    yield "file-image", {"messages": [user([file("data:image/png;base64,AA==", filename="a.png"), text])]}, {
        "file:image"}
    yield "tool-file-image", {"messages": looked + [result([file("data:image/webp;base64,AA==")])], **chat}, {
        "tool-results:images", "file:image"}
    yield "file-id", {"messages": [user([{"type": "file", "file": {"file_id": "file-1"}}])]}, {"file:id"}
    yield "file-pdf", {"messages": [user([file("data:application/pdf;base64,AA==")])]}, {"file:non-image"}
    yield "file-flat", {"messages": [user([{"type": "file", "file_data": "data:image/png;base64,AA=="}])]}, {
        "file:flat"}


def cases(declaration):
    for name, request, _ in tool_image_cases():
        yield name, request, {f"fixed:{name}", "tool-results:images"}
    for name, request, tags in image_format_cases():
        yield name, request, {f"fixed:{name}", *tags}
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
        "fixed:preserved-reasoning-at-max", "preserve_thinking:true", "reasoning:rendered-without-tools"}
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
        "fixed:image-placeholder-in-content", "client-text:image-placeholder"}
    yield "image-placeholder-in-reasoning", {"messages": [
        user("q"), {"role": "assistant", "content": "a", "reasoning_content": f"r {placeholder}"}, user("q2"),
    ]}, {"fixed:image-placeholder-in-reasoning", "client-text:image-placeholder"}
    yield "image-placeholder-in-text-part", {"messages": [
        {"role": "user", "content": [{"type": "text", "text": f"see {placeholder}"}]}]}, {
        "fixed:image-placeholder-in-text-part", "user:parts", "client-text:image-placeholder"}
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
    # A developer message is a system message (`developer-as-system` in
    # frontend-divergences.json): before the last user turn in thinking
    # mode, which the encoder would drop unread as a developer message
    # (lines 756-761); leading, carrying the offered tools and the response
    # format; later, followed by the generation header; and empty, dropped.
    yield "developer-before-last-user", {"messages": [
        {"role": "developer", "content": "Be terse."}, user("q1"),
        {"role": "assistant", "content": "a1"}, user("q2"),
    ]}, {"fixed:developer-before-last-user", "developer:present"}
    yield "developer-message", {"messages": [{"role": "developer", "content": "Answer in JSON."}, user("Oslo?")],
                                "tools": [WEATHER], "response_format": RESPONSE_FORMATS["json-object"]}, {
        "fixed:developer-message", "developer:present", "tools:wrapped", "response-format:json-object"}
    yield "later-developer-message", {"messages": [
        user("q1"), {"role": "assistant", "content": "a1"},
        {"role": "developer", "content": "Answer in French from now on."},
    ]}, {"fixed:later-developer-message", "developer:present"}
    yield "empty-developer-message", {"messages": [{"role": "developer", "content": ""}, user("q")]}, {
        "fixed:empty-developer-message", "developer:present"}
    # Protocol efforts the encoder does not name.
    yield "medium-effort", {"messages": [user("q")], "reasoning_effort": "medium"}, {"fixed:medium-effort"}
    yield "minimal-effort", {"messages": [user("q")], "reasoning_effort": "minimal"}, {"fixed:minimal-effort"}
    yield "xhigh-effort", {"messages": [user("q")], "reasoning_effort": "xhigh"}, {"fixed:xhigh-effort"}
    yield "ultra-effort", {"messages": [user("q")], "reasoning_effort": "ultra"}, {"fixed:ultra-effort"}
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
    # Mapping rules (`mapping` in frontend-divergences.json). R1: reasoning
    # before the last user turn drops, and a turn without any writes
    # </think> alone, as the encoder renders it.
    yield "earlier-reasoning-dropped", {"messages": [
        user("q1"), {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        user("q2"), {"role": "assistant", "content": "a2"}, user("q3"),
    ]}, {"fixed:earlier-reasoning-dropped"}
    # R1: a turn after the last user turn keeps its reasoning, without
    # offered tools too.
    yield "reasoning-after-last-user", {"messages": [
        user("Weather in Oslo?"),
        {"role": "assistant", "content": "", "reasoning_content": "Call the tool.",
         "tool_calls": [call("call_o", "get_weather", {"location": "Oslo"})]},
        {"role": "tool", "tool_call_id": "call_o", "content": "18"},
    ]}, {"fixed:reasoning-after-last-user", "reasoning:rendered-without-tools"}
    # R2: each offered tool is the recipe's function object: name,
    # description or "", parameters or {}, every other field dropped.
    yield "recipe-tool-schema", {"messages": [user("q")], "tools": [
        {"type": "function", "function": {"parameters": {"type": "object"}, "strict": False, "name": "a",
                                          "defer_loading": True, "description": "A."}},
        {"type": "function", "function": {"name": "b", "description": None, "parameters": None}},
        {"type": "function", "function": {"name": "c"}},
        {"type": "function", "function": {"name": "d", "function": {"name": "g"}}}]}, {
        "fixed:recipe-tool-schema", "tools:wrapped"}
    # R2: a strict tool's parameters are the recipe's normalized schema:
    # every property required, additionalProperties false, keywords outside
    # its type set dropped, and no parameters the empty object.
    lookup = {"type": "function", "function": {"name": "lookup", "description": "Look a record up.", "strict": True,
                                               "parameters": {
        "type": "object", "title": "Lookup", "$defs": {"unused": {"type": "string"}},
        "properties": {"id": {"type": "integer", "minimum": 1, "description": "The id."},
                       "tags": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 3},
                       "filter": {"type": "object", "properties": {
                           "kind": {"enum": ["a", "b"], "type": "string"},
                           "since": {"anyOf": [{"type": "string", "format": "uuid"}, {"type": "null"}]}}}}}}}
    clock = {"type": "function", "function": {"name": "now", "strict": True}}
    yield "strict-tool-schemas", {"messages": [user("Look up record 7.")], "tools": [lookup, clock]}, {
        "fixed:strict-tool-schemas", "tools:wrapped"}
    # R2: a strict schema the recipe refuses, a non-strict tool's parameters
    # without type object, strict and non-strict tools together, and a
    # response format beside strict tools are refused; a named choice
    # leaves only its tool, so the strict tool it drops no longer counts.
    partial = {"type": "function", "function": {"name": "f", "strict": True, "parameters": {
        "type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a"]}}}
    yield "strict-tool-partial-required", {"messages": [user("q")], "tools": [partial]}, {
        "fixed:strict-tool-partial-required", "tools:wrapped"}
    yield "non-object-tool-parameters", {"messages": [user("q")], "tools": [
        {"type": "function", "function": {"name": "f", "parameters": {"type": "string"}}}]}, {
        "fixed:non-object-tool-parameters", "tools:wrapped"}
    yield "strict-and-non-strict-tools", {"messages": [user("q")], "tools": [clock, WEATHER]}, {
        "fixed:strict-and-non-strict-tools", "tools:wrapped"}
    yield "strict-tool-with-response-format", {"messages": [user("q")], "tools": [clock],
                                               "response_format": RESPONSE_FORMATS["text"]}, {
        "fixed:strict-tool-with-response-format", "tools:wrapped", "response-format:text"}
    yield "strict-mix-named-choice", {"messages": [user("Weather in Oslo?")], "tools": [clock, WEATHER],
                                      "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                                      "thinking": {"type": "disabled"}}, {
        "fixed:strict-mix-named-choice", "tools:wrapped"}
    # A named choice leaves only its tool, so a schema the recipe refuses on
    # a tool it drops is not checked (`named-choice-schema-scope`).
    yield "named-choice-skips-schema", {"messages": [user("Weather in Oslo?")], "tools": [partial, WEATHER],
                                        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                                        "thinking": {"type": "disabled"}}, {
        "fixed:named-choice-skips-schema", "tools:wrapped"}
    # A strict schema keeps a number's digits as sent, where the recipe
    # rounds one beyond u64 to a float, and refuses a node written as a
    # sequence, which the recipe reads by position (`strict-schema-numbers`,
    # `strict-schema-positional-nodes`).
    yield "strict-large-number", {"messages": [user("q")], "tools": [
        {"type": "function", "function": {"name": "f", "strict": True, "parameters": {
            "type": "object", "properties": {"x": {"type": "number", "maximum": 18446744073709551617}}}}}]}, {
        "fixed:strict-large-number", "tools:wrapped", "strict-schema:large-number"}
    yield "strict-positional-node", {"messages": [user("q")], "tools": [
        {"type": "function", "function": {"name": "f", "strict": True, "parameters": {
            "type": "object", "properties": {"x": {"anyOf": [["string", None, None, None, None]]}}}}}]}, {
        "fixed:strict-positional-node", "tools:wrapped", "strict-schema:positional-node"}
    # A non-strict tool's parameters are checked only for type object; the
    # recipe validates them against its JSON Schema meta-schema, which has
    # no type `str` (`non-strict-schema-validation`).
    yield "non-strict-type-alias", {"messages": [user("q")], "tools": [
        {"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"x": {"type": "str"}}}}}]}, {
        "fixed:non-strict-type-alias", "tools:wrapped", "tool-schema:type-alias"}
    # R3: a response format renders as the recipe renders it: text and regex
    # nothing (and need no system message), json_object its type, json_schema its
    # schema, a strict one normalizing as a strict tool's does. A schema the
    # encoder reads as false renders nothing, and a missing one is refused.
    yield "response-format-text", {"messages": [user("q")], "response_format": RESPONSE_FORMATS["text"]}, {
        "fixed:response-format-text", "response-format:text"}
    yield "response-format-json-object", {"messages": [user("Answer in JSON.")],
                                          "response_format": RESPONSE_FORMATS["json-object"]}, {
        "fixed:response-format-json-object", "response-format:json-object"}
    yield "response-format-json-schema", {"messages": [user("Weather in Oslo as JSON.")],
                                          "response_format": RESPONSE_FORMATS["json-schema"]}, {
        "fixed:response-format-json-schema", "response-format:json-schema"}
    for name, schema in (("empty", {}), ("false", False)):
        yield f"response-format-{name}-schema", {"messages": [user("q")], "response_format": {
            "type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}}, {
            f"fixed:response-format-{name}-schema", "json-schema:falsy"}
    yield "response-format-regex", {"messages": [user("Answer yes or no.")],
                                    "response_format": {"type": "regex", "regex": "(yes|no)"}}, {
        "fixed:response-format-regex", "response-format:regex"}
    yield "response-format-without-schema", {"messages": [user("q")], "response_format": {
        "type": "json_schema", "json_schema": {"name": "answer"}}}, {"fixed:response-format-without-schema"}
    yield "response-format-null-schema", {"messages": [user("q")], "response_format": {
        "type": "json_schema", "json_schema": {"name": "answer", "schema": None}}}, {
        "fixed:response-format-null-schema"}
    # json_object without the word json in a message, which the recipe
    # refuses and this format renders (`json-word-check`).
    yield "json-object-without-json-word", {"messages": [user("Weather in Oslo?")],
                                            "response_format": RESPONSE_FORMATS["json-object"]}, {
        "fixed:json-object-without-json-word", "response-format:json-object"}
    yield "strict-response-format-partial-required", {"messages": [user("q")], "response_format": {
        "type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {
            "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"]}}}}, {"fixed:strict-response-format-partial-required"}
    # R4: system messages alone render as the encoder renders them: a
    # later system message is followed by the generation header (lines
    # 713, 731-739), so two are served, and an empty one is dropped (R5).
    yield "system-messages-only", {"messages": [
        {"role": "system", "content": "Be brief."}, {"role": "system", "content": "Answer in French."},
    ]}, {"fixed:system-messages-only", "system:late"}
    yield "system-messages-with-empty", {"messages": [
        {"role": "system", "content": "Be brief."}, {"role": "system", "content": "Answer in French."},
        {"role": "system", "content": ""},
    ]}, {"fixed:system-messages-with-empty", "system:late", "system:empty"}
    # R4: a final assistant turn ends with the end-of-sentence token and no
    # header follows, so it is refused unless continued.
    yield "final-assistant-turn", {"messages": [user("q"), {"role": "assistant", "content": "a"}]}, {
        "fixed:final-assistant-turn", "ends:assistant"}
    # A turn whose content holds a special token, which the vendor parser
    # refuses (lines 981-983), so it reads no turn back
    # (`special-token-in-turn`).
    yield "special-token-in-assistant-turn", {"messages": [
        user("q"), {"role": "assistant", "content": "<think>inline plan</think> It is mild."}]}, {
        "fixed:special-token-in-assistant-turn", "ends:assistant", "assistant-content:inline-think"}
    # A turn whose content ends with a newline before its calls, which the
    # served parser reads back without it (`call-open-newlines`).
    yield "content-newline-before-calls", {"messages": [
        user("Weather in Oslo?"),
        {"role": "assistant", "content": "Calling.\n", "tool_calls": [call("call_n", "get_weather", {"location": "Oslo"})]},
    ], "tools": [WEATHER]}, {"fixed:content-newline-before-calls", "ends:assistant", "tools:wrapped",
                             "assistant-content:newline-before-calls"}
    # Turns the served parser reads back byte for byte where the recipe's
    # parser does not (`parser` in frontend-divergences.json): newlines
    # around reasoning and answer text, and an answer in a JSON fence under
    # json_object.
    yield "newline-edged-turn", {"messages": [
        user("q"), {"role": "assistant", "content": "\n\nIt is mild.", "reasoning_content": "\nCheck the units.\n"}]}, {
        "fixed:newline-edged-turn", "ends:assistant", "turn-text:edge-newlines"}
    yield "json-fenced-answer", {"messages": [
        user("Weather in Oslo as JSON."), {"role": "assistant", "content": "```json\n{\"temp\": 18}\n```"}],
        "response_format": RESPONSE_FORMATS["json-object"]}, {
        "fixed:json-fenced-answer", "ends:assistant", "response-format:json-object", "answer:json-fence"}
    # A special token's spelling in history text, which the served
    # tokenizer reads as that token and the recipe as plain text
    # (`special-token-spellings`).
    yield "special-token-in-history", {"messages": [
        user("q1"), {"role": "assistant", "content": "<think>inline plan</think> It is mild."}, user("q2")]}, {
        "fixed:special-token-in-history", "assistant-content:inline-think"}
    yield "continued-final-assistant-turn", {"messages": [user("q"), {"role": "assistant", "content": "a"}],
                                             "continue_final_message": True, "add_generation_prompt": False}, {
        "fixed:continued-final-assistant-turn", "ends:continue"}
    # R4: `prefix: true` on the final assistant turn continues it, as the
    # recipe sends a continuation; in thinking mode a turn of reasoning
    # alone keeps its reasoning open, and one with content closes it.
    yield "prefix-continues-reasoning", {"messages": [
        user("What is 17 times 23?"),
        {"role": "assistant", "content": "", "reasoning_content": "17 times 23 is", "prefix": True}]}, {
        "fixed:prefix-continues-reasoning", "ends:prefix"}
    yield "prefix-continues-answer", {"messages": [
        user("What is 17 times 23?"),
        {"role": "assistant", "content": "The answer is", "reasoning_content": "17 times 23 is 391.",
         "prefix": True}]}, {"fixed:prefix-continues-answer", "ends:prefix"}
    # A `prefix` continuation stays served beside offered tools, calls in
    # history and json_object, which the recipe refuses
    # (`prefix-beside-tools-or-json-object`).
    yield "prefix-beside-tools", {"messages": [
        user("Weather in Oslo?"),
        {"role": "assistant", "content": "Let me check", "reasoning_content": "The user wants Oslo's weather.",
         "prefix": True}], "tools": [WEATHER]}, {"fixed:prefix-beside-tools", "ends:prefix", "tools:wrapped"}
    yield "prefix-after-tool-results", {"messages": [
        user("Weather in Oslo?"),
        {"role": "assistant", "content": "", "tool_calls": [call("call_a", "get_weather", {"location": "Oslo"})]},
        {"role": "tool", "tool_call_id": "call_a", "content": "18"},
        {"role": "assistant", "content": "It is", "prefix": True}]}, {"fixed:prefix-after-tool-results", "ends:prefix"}
    yield "prefix-beside-json-object", {"messages": [
        user("Answer in JSON: what is 17 times 23?"),
        {"role": "assistant", "content": "{\"answer\": ", "prefix": True}],
        "response_format": RESPONSE_FORMATS["json-object"]}, {
        "fixed:prefix-beside-json-object", "ends:prefix", "response-format:json-object"}
    # R5: an empty system message is dropped, leading, later or as empty
    # parts; the one carrying the offered tools is still inserted.
    yield "empty-system-messages-dropped", {"messages": [
        {"role": "system", "content": ""}, user("q1"), {"role": "assistant", "content": "a1"},
        {"role": "system", "content": [{"type": "text", "text": ""}]}, user("q2"),
    ]}, {"fixed:empty-system-messages-dropped", "system:empty"}
    yield "empty-system-message-with-tools", {"messages": [{"role": "system", "content": ""}, user("q")],
                                              "tools": [WEATHER]}, {
        "fixed:empty-system-message-with-tools", "system:empty", "tools:wrapped"}
    # R5: a leading billing header is removed first, up to the first newline
    # or "You are", and the leading whitespace after it; a system message of
    # the header alone is empty and dropped.
    header = "x-anthropic-billing-header: cc_version=2.1.3; cc_entrypoint=cli;"
    yield "billing-header-stripped", {"messages": [
        {"role": "system", "content": f"{header}\n  You are a coding agent.\nBe brief."}, user("q")]}, {
        "fixed:billing-header-stripped"}
    yield "billing-header-before-you-are", {"messages": [
        {"role": "system", "content": f"{header}You are a coding agent.\nBe brief."}, user("q")]}, {
        "fixed:billing-header-before-you-are"}
    yield "billing-header-in-parts", {"messages": [
        {"role": "system", "content": [text(header), text("You are a coding agent.")]}, user("q")]}, {
        "fixed:billing-header-in-parts"}
    yield "billing-header-alone", {"messages": [{"role": "system", "content": header}, user("q")]}, {
        "fixed:billing-header-alone"}
    # System and developer images, which the image declaration admits as the
    # recipe reads V4.1's (`images` in frontend-divergences.json): in place
    # in the system text, alone, which keeps the message, and after a
    # removed billing header.
    image = lambda data: {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}
    yield "system-image", {"messages": [
        {"role": "system", "content": [text("Describe images briefly."), image("AA==")]},
        user([image("AQ=="), text("What changed?")])]}, {"fixed:system-image", "system:image"}
    yield "developer-image-alone", {"messages": [{"role": "developer", "content": [image("AA==")]},
                                                 user("What is this?")]}, {
        "fixed:developer-image-alone", "developer:present", "system:image"}
    yield "billing-header-before-system-image", {"messages": [
        {"role": "system", "content": [text(header), text(" "), image("AA==")]}, user("q")]}, {
        "fixed:billing-header-before-system-image", "system:image"}
    # R6: a tool choice of none renders no tools; required and a named
    # choice end the prompt with the calls block's opening (a named one
    # renders only its tool), and are refused in thinking mode.
    chat = {"thinking": {"type": "disabled"}}
    second = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}
    yield "tool-choice-none", {"messages": [user("q")], "tools": [WEATHER], "tool_choice": "none"}, {
        "fixed:tool-choice-none", "tools:wrapped"}
    yield "tool-choice-required", {"messages": [user("Weather in Oslo?")], "tools": [WEATHER, second],
                                   "tool_choice": "required", **chat}, {
        "fixed:tool-choice-required", "tools:wrapped"}
    yield "tool-choice-named", {"messages": [user("Weather in Oslo?")], "tools": [WEATHER, second],
                                "tool_choice": {"type": "function", "function": {"name": "get_weather"}}, **chat}, {
        "fixed:tool-choice-named", "tools:wrapped"}
    yield "tool-choice-required-thinking", {"messages": [user("Weather in Oslo?")], "tools": [WEATHER],
                                            "tool_choice": "required"}, {
        "fixed:tool-choice-required-thinking", "tools:wrapped"}
    # Without offered tools, required and a named choice are auto, in
    # thinking mode too.
    yield "tool-choice-required-without-tools", {"messages": [user("q")], "tools": [], "tool_choice": "required"}, {
        "fixed:tool-choice-required-without-tools"}
    yield "tool-choice-named-without-tools", {"messages": [user("q")], "tool_choice": {
        "type": "function", "function": {"name": "get_weather"}}}, {"fixed:tool-choice-named-without-tools"}
    # The thinking switch: the top-level object, which overrides the
    # effort, then the effort; the template keywords are ignored.
    yield "thinking-object-disabled", {"messages": [user("q")], "thinking": {"type": "disabled"},
                                       "chat_template_kwargs": {"enable_thinking": True, "thinking": True}}, {
        "fixed:thinking-object-disabled", "thinking:object=disabled"}
    yield "thinking-object-adaptive-over-none", {"messages": [user("q")], "thinking": {"type": "adaptive"},
                                                 "reasoning_effort": "none"}, {
        "fixed:thinking-object-adaptive-over-none", "thinking:object=adaptive"}
    yield "thinking-object-enabled-at-max", {"messages": [user("q")],
                                             "thinking": {"type": "enabled", "budget_tokens": 1024},
                                             "reasoning_effort": "max"}, {
        "fixed:thinking-object-enabled-at-max", "thinking:object=enabled"}
    yield "template-thinking-ignored", {"messages": [user("q")], "chat_template_kwargs": {
        "enable_thinking": False, "thinking": False, "reasoning_effort": "low"}}, {"fixed:template-thinking-ignored"}
    yield "effort-none-disables", {"messages": [user("q")], "reasoning_effort": "none"}, {
        "fixed:effort-none-disables"}
    # Namespaces (`namespace` in frontend-divergences.json): on a function
    # object, as an object with a description or as a name; on the tool
    # object; spelled in the name; a call by its qualified name in history;
    # and a named choice of a qualified tool.
    namespaced = {"type": "function", "function": {
        "name": "f", "namespace": {"name": "ns", "description": "Tools of ns."},
        "description": "F.", "parameters": {"type": "object"}}}
    yield "namespaced-function-object", {"messages": [user("q")], "tools": [namespaced]}, {
        "fixed:namespaced-function-object", "tools:wrapped", "namespace:present"}
    yield "namespace-name-on-function", {"messages": [user("q")], "tools": [
        {"type": "function", "function": {"name": "f", "namespace": "ns", "parameters": {"type": "object"}}}]}, {
        "fixed:namespace-name-on-function", "tools:wrapped", "namespace:present"}
    yield "namespace-on-tool-object", {"messages": [user("q")], "tools": [
        {"type": "function", "namespace": {"name": "ns", "description": "Tools of ns."},
         "function": {"name": "f", "description": "F.", "parameters": {"type": "object"}}}]}, {
        "fixed:namespace-on-tool-object", "tools:wrapped", "namespace:present"}
    yield "qualified-tool-name", {"messages": [user("q")], "tools": [
        {"type": "function", "function": {"name": "ns::f", "parameters": {"type": "object"}}}]}, {
        "fixed:qualified-tool-name", "tools:wrapped", "namespace:present"}
    yield "qualified-call-in-history", {"messages": [
        user("Go."),
        {"role": "assistant", "content": "", "tool_calls": [call("call_q", "ns::f", {"x": "v"})]},
        {"role": "tool", "tool_call_id": "call_q", "content": "ok"}, user("Again."),
    ], "tools": [namespaced]}, {"fixed:qualified-call-in-history", "tools:wrapped", "namespace:present"}
    # Refused by the encoder's assertions and here by name: a namespace
    # other than the one the name spells, and a separator inside a part.
    yield "conflicting-namespace", {"messages": [user("q")], "tools": [
        {"type": "function", "namespace": "b", "function": {"name": "a::f", "parameters": {"type": "object"}}}]}, {
        "fixed:conflicting-namespace", "tools:wrapped", "namespace:present"}
    yield "separator-in-namespace", {"messages": [user("q")], "tools": [
        {"type": "function", "function": {"name": "f", "namespace": "a::b", "parameters": {"type": "object"}}}]}, {
        "fixed:separator-in-namespace", "tools:wrapped", "namespace:present"}
    yield "named-choice-of-qualified-tool", {"messages": [user("q")], "tools": [namespaced, second],
                                             "tool_choice": {"type": "function", "function": {"name": "ns::f"}},
                                             **chat}, {"fixed:named-choice-of-qualified-tool", "tools:wrapped", "namespace:present"}
    # A system message's own tools, which the recipe's request messages do
    # not carry and this format refuses.
    yield "system-message-tools", {"messages": [
        {"role": "system", "content": "s", "tools": [WEATHER]}, user("q"),
    ]}, {"fixed:system-message-tools", "message-tools:present"}

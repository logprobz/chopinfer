# frontend-format

The library a model's chat format is written with. A model renders its
prompt in Rust, ported from its chat template with the toolkit below, and
declares in its frontend component only how its completions read: the
`format.parser` section. The library serves the two over `chop.frontend.v1`:
it renders through the model's format, builds the grammar a tool choice
imposes, and parses completions back into reasoning, answer text and tool
calls with the shared parsers.

The library is opt-in. A bundle exports its model's format with one line at
its crate root:

```rust
frontend_format::export_chat_format_v1!(Format);
```

A bundle with a format of its own throughout, parsers included, implements
`runtime_bundle_abi::frontend::FrontendV1` instead, exports it with
`runtime_bundle_abi::export_frontend_v1!`, and links nothing from here. A
completion syntax neither parser family below reads gets a new named family
here, never a flag on an existing family; a bundle outside this tree that
cannot wait for an SDK revision with that family implements `FrontendV1`.

The library depends only on the ABI documents (`runtime-bundle-abi`), the
declaration (`runtime-frontend-contract`), `runtime-grammar` and serde. It
links no transport, host or device code, and names no model. Model tests and
benchmarks call `tool_constraint_v1`, `parse_completion_v1` and
`DeclaredParserV1` directly.

## The format section

The host reads a component's model-neutral sections and passes its `format`
object unread. This library reads it as `FormatDescriptorV1`. The object
holds exactly one member, `parser` (`FrontendParserDeclarationV1`), and
nothing else: a template or media labels are the renderer's code, and a
`format` that carries them is refused.

Opening the component checks the section against the host sections: the
reserved markers, each stop token's boundary text, the reasoning end markers
and thinking keywords it reads, and a tool-name byte set that cannot begin a
name delimiter.

## The `format.parser` section

The shared streaming parser, whole-text parser and tool-choice grammars read
one input: the tool grammar (`xml-function-parameters-v1`,
`xml-key-value-arguments-v1` or `xml-typed-parameters-v1`), the JSON encoding of parsed arguments, the
reasoning open and close, `answer_prefix`, the tool-call, function and
parameter markers with `parameter_value_open` and `tool_call_separator`,
`drop_blank_tool_call_content`, and the reserved markers. `turn_open` (the
boundary tokens that open a turn) and `assistant_header` are for the
qualification re-parser. Markers keep the strings and whitespace the template
writes; the parsers read most of them without that whitespace.

```json
"format": {
  "parser": {
    "tool_grammar": "xml-key-value-arguments-v1",
    "json_encoding": "python-json-v1",
    "reasoning_open": "<think>",
    "reasoning_close": "</think>",
    "answer_prefix": "",
    "tool_call_open": "<tool_call>",
    "tool_call_close": "</tool_call>",
    "tool_call_separator": "",
    "function_open": "",
    "function_name_close": "",
    "function_close": "",
    "parameter_open": "<arg_key>",
    "parameter_name_close": "</arg_key>",
    "parameter_value_open": "<arg_value>",
    "parameter_close": "</arg_value>",
    "drop_blank_tool_call_content": false,
    "reserved_markers": [],
    "turn_open": ["<|assistant|>", "<|user|>"],
    "assistant_header": "<|assistant|>"
  }
}
```

`answer_prefix`, `tool_call_separator`, the three function markers,
`parameter_value_open`, `parameter_json_name_close`,
`drop_blank_tool_call_content` and `reserved_markers` may be left out and
default to empty. The reasoning,
tool-call and parameter markers may not be blank. The function markers are
present for function and typed parameters and empty for key-value
arguments, which alone has `parameter_value_open`; typed parameters alone
have `parameter_json_name_close`, which neither begins nor is begun by
`parameter_name_close`. Each stop token has a boundary text, each
`turn_open` entry is a boundary token, and `assistant_header` begins with
one of them.

## The parser families

The three tool grammars are named families every model's parser section
picks from:

- **Function parameters** (`xml-function-parameters-v1`): a call is the
  tool-call open, a function element around the name, one parameter element
  per argument whose name sits in its open tag, and the closes.
- **Key-value arguments** (`xml-key-value-arguments-v1`): the name follows
  the call open directly, and each argument is a key element then a value
  element. There is no function element.
- **Typed parameters** (`xml-typed-parameters-v1`): the function element of
  function parameters, whose parameter open tag ends in one of two closes
  that type the value: `parameter_name_close` for raw text, read as the
  string it spells whatever the schema says, and `parameter_json_name_close`
  for JSON, read as the value it spells (a repeated key taking its last
  value; JSON that does not read stays text). The call open is read with
  the whitespace written before it, so text before a call keeps none of
  it. A forced call writes string values raw and every other value as
  JSON with the JSON close.

A string argument is raw text up to its close, and any other argument is
JSON for its schema. A type the coercion reads as a string, by name or by
an alias such as `str`, `text`, `varchar`, `char` or `enum`, or a type
name it reads as no other JSON type, is a string type here too. A
function parameter's value is every byte between
its markers less the whitespace the declared name close writes after its
bare marker and the declared close writes before its own, once each: a
template that writes `>\n` and `\n</parameter>` frames values in one
newline, one that writes `>` and `</parameter>` frames nothing. A
parameter's schema is its declared property, else free text when pattern
properties may type it, none when `additionalProperties` is false, the
additional-property schema when it is one, and otherwise free text; the
value scan, the coercion and the tool-choice grammar all resolve it so.
The grammar writes each declared or required name at most once, and an
additional or pattern property under none of those names, so every name it
writes resolves to the schema it wrote the value by. Two additional
properties may share a name; the parser keeps the last value, which the
same schema typed.
Tool regions stream as they are generated, and each
call's arguments reach the client only once they cannot change. When
generation ends inside a call, at a stop, a client stop or the length limit,
the calls whose arguments are complete are delivered and the unfinished
argument is dropped. A reserved marker anywhere the parser does not expect
it refuses the completion. A whole-text parse (`parse_completion_v1`) reads
a completion as one token carrying all of it and agrees with the incremental
parse.

The tool-choice grammar a request's `tool_choice` imposes writes calls in
the same family, so a forced call reads back as the parser reads any call.

A declaration may also state how a tool choice changes the render, in the
host-read `tools.choice` section: `none_omits_tools`,
`named_renders_its_tool_only`, a `forced_prefill` that ends the prompt of a
`required` or named choice, and `forced_refused_in_thinking`. The host
applies it around the render. The prefill must begin the parser section's
`tool_call_open`; the completion's parser and the forced choice's
constraint read it as the completion's start. A declaration without the
section renders every offered tool for every choice, and the constraint
alone forces a call.

A `tools.namespaces` section (`{"separator": "::"}`) qualifies function
names: a namespace, the separator, then the name, each part a name of
`name_bytes` within `maximum_name_bytes`, so a qualified name may reach twice
that length and the separator. The host names a tool in a namespace by its qualified name,
and the parser and `validate_function_name_v1` read a call's qualified name
whole. The separator is ASCII punctuation outside `name_bytes` that begins
no marker ending a name.

## A model's own chat format

A model's renderer implements `ChatFormatV1`:

```rust
pub trait ChatFormatV1: Sized + Send + Sync + 'static {
    fn open(descriptor: &FormatDescriptorV1) -> Result<Self, String>;
    fn opened(&self) -> OpenedV1;
    fn render(&self, request: FormatRenderRequestV1<'_>)
        -> Result<FormatRenderedV1, FrontendErrorV1>;
}
```

`open` reads the checked declaration (the host sections and
`format.parser`) and refuses one it cannot render or would not follow. Each
in-tree format compares the whole parser section with the one its template
writes (its contract's `parser_declaration()`), since the export parses and
constrains completions with it and the renderer writes the same markers and
JSON encoding itself. It also refuses reasoning controls its template does
not follow, and a served media kind it writes no labels for or whose pad
differs. `opened` is the export's open answer: what the render reads beyond
the common request fields. `render` returns the prompt, its segments and
the request's media in segment order, the mode and the effort.

`ChatFormatExportV1<F>` serves `F` over `chop.frontend.v1`. Renders go
through `F`. Tool constraints and parsers come from `format.parser` and the
shared parsers, so a format writes no parser.
`export_chat_format_v1!(Type)` exports it from a bundle's crate root, and
`chat_format_export_v1::<Type>()` is the in-process export for tests.

The export, not the renderer, decides `reasoning_open`. It computes the flag
from the rendered prompt with the parser section's reasoning markers
(`toolkit::reasoning_open_v1`), replacing whatever the render returned. A
thinking prompt leaves reasoning open, except a continued final assistant
turn whose reasoning the prompt already closed. The host starts the
completion in reasoning and gates constraints on that flag, so the parser
and constraints agree with the prompt the same way for every model. The
shared parser reads no parser state, so the export refuses a render that
leaves some.

## The toolkit

`frontend_format::toolkit` holds the model-neutral pieces a renderer is
written with:

- prompt writing under `maximum_prompt_bytes` (`append_v1`), name checks,
  and media segments: a private placeholder per item, client text screened
  for it and for media pads, and the finished prompt split into text and
  media segments;
- the conversation as a renderer reads it from a request, and the checks it
  passes before anything is written (`chat_message_v1`, `validate_chat_v1`);
- bounded JSON reading and the declared JSON encodings
  (`parse_json_object_v1`, `write_json_v1`, `json_to_string_v1`);
- Python semantics a ported template relies on: `tojson_v1` (`json.dumps`
  under HF's `tojson` arguments), `python_strip_v1` and `python_rstrip_v1`
  (Python's whitespace, which includes `\x1c` to `\x1f`, unlike Rust's
  `trim`), `python_split_v1` and `python_capitalize_v1` (titlecase, final
  sigma);
- tool objects as a template writes them (`json_truthy`,
  `defers_loading_v1`, `write_tool_object_v1`);
- the reasoning controls (`select_mode_v1`, `preserve_thinking_v1`,
  `reasoning_open_v1`) and the refusal messages, including
  `TOOL_REFERENCES_UNSUPPORTED_V1` for a format that reads no
  `tool_reference` part.

`tests/python_semantics.json` holds inputs with the outputs Python gave for
them, with the interpreter and Python version that wrote it.
`tools/refrun/python_semantics.py` writes it, and `--check` reports whether
it is current. The toolkit test reproduces every row. The same script writes
`src/toolkit/unicode15.rs`, the Unicode 15.0 classes `capitalize` reads
(unassigned, Cased and Case_Ignorable), and refuses to run under a Python
that reads another Unicode version. With them the helpers agree with Python
3.12 on every Unicode scalar value: `capitalize` gives an unassigned code
point no case mapping.

The toolkit keeps what a port calls. A Python semantic no port uses (such as
`str()` or `repr` of a JSON value) is not kept for later: the port whose
template needs it adds it, with its rows in `python_semantics.py`.

## Test support

The `test-support` feature adds what format and host tests share: the
parser fixture declarations (`test_support::fixtures::KEY_VALUE` and
`FUNCTION_PARAMETERS`, one on each parser family), `FixtureFormatV1`, a
minimal chat format with one documented layout for tests that need a prompt
but assert nothing about a model's, and the chat-capture readers model tests
use.

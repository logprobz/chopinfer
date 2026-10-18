"""The HF differential corpus of a model's chat frontend.

    python -m tools.refrun.frontend_corpus generate --model M [--output DIR]
    python -m tools.refrun.frontend_corpus minimize --model M --corpus DIR --output DIR
    python -m tools.refrun.frontend_corpus check --model M
    python -m tools.refrun.frontend_corpus rounds --model M --corpus DIR

`generate` renders every seeded case (tools/refrun/frontend_cases.py plus
`models/M/frontend_cases.py`) with HF `apply_chat_template` at the model's
pinned template, or with the model's pinned Python encoder where it ships one
instead (`Reference.python_encoder`), tokenizes it the way the frontend
goldens do, and records reference refusals, template line coverage, feature
tags and parser round-trip vectors. The full corpus goes to `$CHOPINFER_DATA/frontend-corpus/M/` by
default. `minimize` keeps a subset that still runs every template line, every
feature tag, every reference-refusal class, every divergence case, one
round-trip vector per turn shape and ending, and every round trip of a fixed
case (a `fixed:<id>` tag). `check` re-renders the committed
corpus (`fixtures/M/frontend/`) and enforces the coverage and divergence gate
against `models/M/frontend-divergences.json`. `rounds` recomputes the
round-trip vectors of a corpus directory.

Each HF input comes from the model's declared transport translation
(`TRANSPORT` in its cases module): which reasoning fields reach history, tool
calls' arguments parsed into objects, and the tool objects the transport
carries. The reasoning controls reach the template as the transport's
protocol passes them, for every model alike (`hf_input`): that is protocol
normalization, while a divergence table states where a template differs.
`frontend.messages_for_hf` stays the translation of the legacy capture
fixtures only. A cases module may define `reference_text(text,
arguments)`, the rewrite its model's processor makes of the template's text
(`Reference.with_reference_text`); the reference is then the rewritten text.

Files (INTERFACES section 3): `corpus.json` holds render vectors with
`"raw": true` and either `rendered_text` and `token_ids` or a `refusal`;
`corpus-rounds.json` holds streaming vectors with an optional `separator`;
`corpus-coverage.json` holds the coverage metadata, including the reference
environment (the Python, transformers, jinja2 and tokenizers versions and
the Unicode version), which `check` warns about when its own differs. A full
corpus also holds `reference.json` (each case's HF outcome) and
`rounds-report.json` (why a case has no round-trip vector). CPU only; no
weights are loaded.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import unicodedata
import unittest
import warnings

from tools.refrun import frontend_cases
from tools.refrun.template_coverage import PythonSourceCoverage, TemplateCoverage

ROOT = Path(__file__).resolve().parents[2]
COVERAGE_SCHEMA = "chop-frontend-corpus-coverage-v1"
DIVERGENCES_SCHEMA = "chop-frontend-divergences-v1"
# A mapping entry records how a request maps into what the reference renders
# where a model has a second authority for that mapping: the rule, the
# authority it follows, its source, and the committed cases that show it.
MAPPING_KEYS = ("id", "follows", "rule", "source", "cases")
# The request fields a render vector may carry besides `messages`
# (crates/runtime-host/src/frontend_check.rs REQUEST_FIELDS_V1).
REQUEST_FIELDS = ("tools", "tool_choice", "chat_template_kwargs", "reasoning_effort", "thinking",
                  "response_format", "add_generation_prompt", "continue_final_message")
# Template keywords the transport reads as absent when sent as null
# (crates/runtime-server/src/frontend.rs TYPED_TEMPLATE_KEYWORDS_V1).
TYPED_KEYWORDS = ("enable_thinking", "thinking", "reasoning_effort", "preserve_thinking", "clear_thinking",
                  "add_vision_id")
# Every key a raw render vector may carry (lane T2's reader is strict).
VECTOR_KEYS = frozenset(("id", "raw", "rendered_text", "token_ids", "refusal", "messages", *REQUEST_FIELDS))
TRANSPORT_KEYS = ("reasoning_content_fallback", "tool_call_reasoning_fallback", "reads_tool_objects",
                  "tool_references", "extra_roles")
# What an OpenAI media part becomes in a template's content list.
HF_MEDIA = {"image_url": ("image", "image_url"), "video_url": ("video", "video_url"),
            "audio_url": ("audio", "audio_url")}


class CorpusError(Exception):
    """A corpus that cannot be built as declared."""


class ReferenceEnvironmentWarning(UserWarning):
    """A check runs another reference environment than its corpus recorded."""


# Models ---------------------------------------------------------------------

class Model:
    """One model's cases module, declaration and paths."""

    def __init__(self, name, root=ROOT, cases=None, declaration=None, parser=None):
        self.name = name
        self.root = Path(root)
        self.directory = self.root / "models" / name
        self.cases = cases if cases is not None else load_module(self.directory / "frontend_cases.py")
        self.declaration = (declaration if declaration is not None
                            else frontend_cases.load_declaration(self.directory))
        # The declaration's format.parser section, whose markers end a raw
        # value; None leaves delimiter collisions unchecked.
        self.parser = (parser if parser is not None or declaration is not None
                       else frontend_cases.load_parser(self.directory))
        missing = [key for key in TRANSPORT_KEYS if key not in self.cases.TRANSPORT]
        if missing:
            raise CorpusError(f"{name}: TRANSPORT lacks {', '.join(missing)}")

    @property
    def pin(self):
        return self.cases.PIN

    @property
    def transport(self):
        return self.cases.TRANSPORT

    @property
    def endings(self):
        return self.cases.ENDINGS

    @property
    def reference_text(self):
        """The cases module's `reference_text(text, arguments)`, or None."""
        return getattr(self.cases, "reference_text", None)

    @property
    def fixtures(self):
        return self.root / "fixtures" / self.name / "frontend"

    @property
    def divergences_path(self):
        return self.directory / "frontend-divergences.json"

    def hf_input(self, request):
        """The HF input of `request` under this model's transport and
        reasoning controls (`hf_input`), with each request field the cases
        module lists in `INPUT_FIELDS` passed through as sent: a reference
        that renders a field a template never reads (a Python encoder's
        `response_format`) receives it. A cases module that sets
        `RAW_TOOL_ARGUMENTS` receives tool-call arguments as the text sent,
        for a reference that decodes them itself."""
        arguments = hf_input(request, self.transport, self.declaration["reasoning"], self.reasoning_field,
                             not getattr(self.cases, "RAW_TOOL_ARGUMENTS", False))
        for field in getattr(self.cases, "INPUT_FIELDS", ()):
            if field in request:
                arguments[field] = copy.deepcopy(request[field])
        return arguments

    @property
    def reasoning_field(self):
        """The request and response field assistant reasoning travels in."""
        return (self.declaration.get("response") or {}).get("reasoning_field", "reasoning")

    def full_cases(self, seed):
        """The full corpus's cases: the shared generator's and the model's."""
        return frontend_cases.generate(self.declaration, self.cases, seed, self.parser)

    def checkpoint(self):
        """The directory of the pinned template and tokenizer files
        (`reference_variables`)."""
        variables = reference_variables(self.pin)
        value = next((os.environ[name] for name in variables if os.environ.get(name)), None)
        if not value:
            raise CorpusError(f"{' and '.join(variables)} are unset")
        return Path(value)


def reference_variables(pin):
    """The variables that name the directory of a model's pinned template and
    tokenizer files, in the order they are read: its tokenizer directory,
    which ci/cpu.sh exports from the checkpoint before it unsets the
    checkpoint, then its checkpoint (`checkpoint_env`)."""
    checkpoint = pin["checkpoint_env"]
    return [checkpoint.removesuffix("_CHECKPOINT") + "_TOKENIZER_DIR", checkpoint]


def load_module(path):
    spec = importlib.util.spec_from_file_location(f"frontend_cases_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_divergences(path):
    """A divergence table, or an empty one when the model has none yet."""
    if path is None or not Path(path).exists():
        return {"schema": DIVERGENCES_SCHEMA, "lines": [], "inputs": [], "mapping": []}
    table = json.loads(Path(path).read_text())
    if table.get("schema") != DIVERGENCES_SCHEMA:
        raise CorpusError(f"{path}: schema must be {DIVERGENCES_SCHEMA}")
    for entry in table.get("inputs", []):
        if entry.get("kind") not in ("refused", "normalized", "no-round-trip"):
            raise CorpusError(f"{path}: inputs entry {entry.get('id')} has an unknown kind")
        if not entry.get("when"):
            raise CorpusError(f"{path}: inputs entry {entry.get('id')} needs a when list")
        if entry.get("ending_only") and entry["kind"] != "refused":
            raise CorpusError(f"{path}: inputs entry {entry.get('id')} sets ending_only but refuses nothing")
    for entry in table.get("mapping", []):
        missing = [key for key in MAPPING_KEYS if not entry.get(key)]
        if missing:
            raise CorpusError(f"{path}: mapping entry {entry.get('id')} needs {', '.join(missing)}")
    table.setdefault("lines", [])
    table.setdefault("inputs", [])
    table.setdefault("mapping", [])
    return table


def matching_entries(table, features, kinds=("refused", "normalized")):
    """The inputs entries of `kinds` whose every `when` tag the case carries."""
    tags = set(features)
    return [entry for entry in table["inputs"] if entry["kind"] in kinds and set(entry["when"]) <= tags]


def yields_no_round_trip(table, features, refused):
    """Whether a case yields no round trip: it is refused (`refused`, by its
    vector or the reference) or matches a refused inputs entry, unless every
    refused entry it matches sets `ending_only`, a refusal of the
    conversation's final turn alone, which a round trip's prefix does not
    hold."""
    entries = matching_entries(table, features, ("refused",))
    if entries:
        return not all(entry.get("ending_only") for entry in entries)
    return refused


def case_divergence(table, features, failure=None):
    """What corpus-coverage.json records as a case's divergence: the id of the
    refused or normalized entry it matches, else, when its round trip fails
    (`failure`, a `RoundTripFailure`), of the no-round-trip entry its
    features and the failure's tags match; a list when it matches several;
    or None."""
    entries = matching_entries(table, features)
    if not entries and failure is not None:
        entries = matching_entries(table, set(features) | failure.tags, ("no-round-trip",))
    if not entries:
        return None
    return entries[0]["id"] if len(entries) == 1 else [entry["id"] for entry in entries]


# Transport translation ------------------------------------------------------

def hf_messages(messages, transport, reasoning_field="reasoning", decode_arguments=True):
    """The messages a template reads for `messages` as the transport carries
    them. An assistant turn's reasoning is `reasoning`, else, where the
    transport falls back to it, `reasoning_content`; a declaration whose
    reasoning field is `reasoning_content` reads that field alone. It
    reaches the template as `reasoning_content`. Tool-call arguments become
    objects, unless `decode_arguments` is false (a reference that reads the
    text as sent). Media parts become the template's media items; every
    other part is kept as sent."""
    result = []
    for message in copy.deepcopy(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            reasoning = message.pop("reasoning", None)
            fallback = message.pop("reasoning_content", None)
            calls = message.get("tool_calls") is not None
            if reasoning_field == "reasoning_content":
                reasoning = fallback
            elif reasoning is None and (transport["reasoning_content_fallback"]
                                        or (calls and transport["tool_call_reasoning_fallback"])):
                reasoning = fallback
            if reasoning is not None:
                message["reasoning_content"] = reasoning
            for call in (message.get("tool_calls") or []) if decode_arguments else ():
                function = call.get("function") if isinstance(call, dict) else None
                if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                    try:
                        function["arguments"] = json.loads(function["arguments"])
                    except ValueError:
                        pass
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            message["content"] = [hf_part(part) for part in message["content"]]
        result.append(message)
    return result


def hf_part(part):
    if not isinstance(part, dict):
        return part
    kind = part.get("type")
    if kind in HF_MEDIA:
        name, field = HF_MEDIA[kind]
        source = part.get(field)
        return {"type": name, name: source.get("url") if isinstance(source, dict) else source}
    if kind == "input_audio":
        source = part.get("input_audio")
        return {"type": "audio", "audio": source.get("data") if isinstance(source, dict) else source}
    return part


def hf_tools(tools, transport):
    """The offered tools as the template reads them: the client's objects
    whole when the transport carries them, else the function fields the
    transport reads (name, and description, parameters and strict when not
    null). The transport reads a null field as an absent one
    (runtime-server `FunctionDefinition`, `convert_tools`)."""
    if not tools:
        return None
    if transport["reads_tool_objects"]:
        return copy.deepcopy(tools)
    result = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            result.append(copy.deepcopy(tool))
            continue
        fields = {"name": function.get("name")}
        if function.get("description") is not None:
            fields["description"] = function["description"]
        if function.get("parameters") is not None:
            fields["parameters"] = copy.deepcopy(function["parameters"])
        if function.get("strict") is not None:
            fields["strict"] = function["strict"]
        result.append({"type": tool.get("type"), "function": fields})
    return result


def hf_input(request, transport, reasoning, reasoning_field="reasoning", decode_arguments=True):
    """The keyword arguments of one `apply_chat_template` call for `request`.

    The template keywords are the client's `chat_template_kwargs`, then the
    reasoning controls as the transport passes them to a template (frontend-
    format `select_mode_v1`, after vLLM): a top-level `reasoning_effort`
    replaces the nested one and, unless the client sent `enable_thinking`,
    sets it to whether the effort differs from the declaration's
    `disable_value` (`reasoning`, the declaration's reasoning section). This
    is protocol normalization, the same for every model; a model's
    divergence table states only where its template differs from it.

    A declaration whose reasoning `switch` is `thinking-object` switches by
    the top-level `thinking` object instead (runtime-server
    `thinking_object_switch`): the thinking keywords and the effort in
    `chat_template_kwargs` are dropped, the object's switch is both
    thinking keywords, and an effort of `none` under an enabled object
    names no effort."""
    keywords = {}
    template = request.get("chat_template_kwargs")
    by_object = reasoning.get("switch") == "thinking-object"
    ignored = ("enable_thinking", "thinking", "reasoning_effort") if by_object else ()
    if isinstance(template, dict):
        keywords.update({key: value for key, value in template.items()
                         if not (value is None and key in TYPED_KEYWORDS) and key not in ignored})
    effort = request.get("reasoning_effort")
    switch = thinking_object(request.get("thinking")) if by_object else None
    if switch is not None:
        keywords["enable_thinking"] = keywords["thinking"] = switch
        if switch and effort == "none":
            effort = None
    if effort is not None:
        keywords["reasoning_effort"] = effort
        keywords.setdefault("enable_thinking", effort != reasoning["disable_value"])
    continued = bool(request.get("continue_final_message"))
    generation = request.get("add_generation_prompt")
    return {"messages": hf_messages(request["messages"], transport, reasoning_field, decode_arguments),
            "tools": hf_tools(request.get("tools"), transport),
            "add_generation_prompt": (not continued) if generation is None else bool(generation),
            "continue_final_message": continued,
            "keywords": keywords}


def thinking_object(value):
    """The switch a top-level `thinking` object sets: True for a `type` of
    `enabled` or its alias `adaptive`, False for `disabled`, None for an
    absent or null object."""
    if value is None:
        return None
    kind = value.get("type") if isinstance(value, dict) else None
    if kind not in ("enabled", "adaptive", "disabled"):
        raise CorpusError(f"thinking object {value!r} is not one the transport reads")
    return kind != "disabled"


def vector_problems(vector):
    """Why a render vector is not one lane T2's raw reader accepts as written."""
    problems = [f"carries {key}, which a render vector may not" for key in sorted(set(vector) - VECTOR_KEYS)]
    problems += [f"{place} names a file:// source; a raw vector's media must be inline data: URLs"
                 for place in file_media(vector)]
    return problems


def file_media(request, kinds=(*HF_MEDIA, "input_audio")):
    """Where the request's media parts of `kinds` take their source from a
    file:// URL: the url of an image_url, video_url or audio_url part, or
    input_audio's data. Text that merely mentions such a URL is not media."""
    places = []
    for index, message in enumerate(request.get("messages") or []):
        content = message.get("content") if isinstance(message, dict) else None
        for number, part in enumerate(content if isinstance(content, list) else []):
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            source = part.get(kind) if kind in kinds else None
            field = "data" if kind == "input_audio" else "url"
            if isinstance(source, dict) and str(source.get(field, "")).startswith("file://"):
                places.append(f"message {index} part {number}")
    return places


def capture_normalization(request, reasoning_field="reasoning"):
    """What a streaming vector reader's capture normalization would change in
    `request` (crates/runtime-host/src/frontend_check.rs `VectorRequestV1::body`):
    a filled id or type, string arguments, a moved reasoning field or an
    inline media source. A round-trip request must have none."""
    changes = []
    for index, message in enumerate(request.get("messages") or []):
        if (message.get("role") == "assistant" and "reasoning_content" in message
                and reasoning_field != "reasoning_content"):
            changes.append(f"message {index}: reasoning_content may move to the served reasoning field")
        if message.get("role") == "tool" and "tool_call_id" not in message:
            changes.append(f"message {index}: an empty tool_call_id is filled")
        for call in message.get("tool_calls") or []:
            if "id" not in call or "type" not in call:
                changes.append(f"message {index}: a tool call's id or type is filled")
            if not isinstance((call.get("function") or {}).get("arguments"), str):
                changes.append(f"message {index}: a tool call's arguments become JSON text")
    # The reader rewrites a file:// url of these kinds (RENDERED_MEDIA_SOURCES_V1).
    changes += [f"{place}: a file:// media source becomes an inline one"
                for place in file_media(request, tuple(HF_MEDIA))]
    return changes


def request_of(vector):
    """The request fields of a render vector or a case."""
    return {key: vector[key] for key in ("messages", *REQUEST_FIELDS) if key in vector}


# The reference -----------------------------------------------------------------

def reference_environment(*packages):
    """What a reference renders and tokenizes with besides the template: the
    Python version, each named package's version and the Unicode version of
    Python's character database."""
    environment = {"python": platform.python_version()}
    environment.update((package, importlib.metadata.version(package)) for package in packages)
    environment["unicode"] = unicodedata.unidata_version
    return environment


class Reference:
    """Renders and tokenizes HF inputs with one template, recording the
    template lines each render runs.

    `render(**hf_input)` returns the prompt text; `tokenize(text)` its ids,
    and `normalize(text)` the text the tokenizer's normalizer reads, which
    the ids decode to. `environment` names the interpreter, packages and
    Unicode version the outcomes come from (`reference_environment`)."""

    def __init__(self, coverage, render, tokenize, environment=None, normalize=None):
        self.coverage = coverage
        self.render_text = render
        self.tokenize = tokenize
        self.environment = environment if environment is not None else reference_environment("jinja2")
        self.normalize = normalize if normalize is not None else (lambda text: text)

    @classmethod
    def hf(cls, checkpoint, source):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=True)
        coverage = TemplateCoverage(source)

        def render(messages, tools, add_generation_prompt, continue_final_message, keywords):
            return tokenizer.apply_chat_template(
                messages, tools=tools, chat_template=source, tokenize=False,
                add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message,
                **keywords)

        normalizer = tokenizer.backend_tokenizer.normalizer
        return cls(coverage, render, lambda text: list(tokenizer(text, add_special_tokens=False)["input_ids"]),
                   reference_environment("transformers", "jinja2", "tokenizers"),
                   normalizer.normalize_str if normalizer is not None else None)

    def outcome(self, arguments):
        """`{"rendered_text", "token_ids", "lines"}` or `{"refusal", "lines"}`,
        the refusal naming the exception class and its first message line."""
        with self.coverage.tracing() as hit:
            try:
                text = self.render_text(**arguments)
            except Exception as error:  # Every exception the render raises is the reference refusing.
                message = str(error).split("\n")[0][:200]
                return {"refusal": {"class": type(error).__name__, "message": message}, "lines": sorted(hit)}
        return {"rendered_text": text, "token_ids": self.tokenize(text), "lines": sorted(hit)}

    @classmethod
    def python_encoder(cls, checkpoint, source, render):
        """The reference of a model whose checkpoint ships a Python encoder
        instead of a chat template: `source` runs once into its own
        namespace (`PythonSourceCoverage`), and each render calls
        `render(namespace, **hf_input)`, the cases module's mapping of the HF
        input onto the encoder's functions. The tokenizer is the
        checkpoint's, as for a template."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        coverage = PythonSourceCoverage(source)
        normalizer = tokenizer.backend_tokenizer.normalizer
        return cls(coverage, lambda **arguments: render(coverage.namespace, **arguments),
                   lambda text: list(tokenizer(text, add_special_tokens=False)["input_ids"]),
                   reference_environment("transformers", "tokenizers"),
                   normalizer.normalize_str if normalizer is not None else None)

    def with_reference_text(self, reference_text):
        """This reference with `reference_text(text, arguments)` applied to
        every render, or this reference when it is None.

        A model's processor may rewrite the template's text before the model
        reads it, for example expanding a media pad into frames between
        markers of its own. A cases module that defines `reference_text`
        states that rewrite and cites the processor code it follows, so the
        reference is the text the model reads. It receives the template's
        text and the render's HF input, and must be the identity on text
        that holds nothing the processor rewrites."""
        if reference_text is None:
            return self
        render = self.render_text
        return Reference(self.coverage, lambda **arguments: reference_text(render(**arguments), arguments),
                         self.tokenize, self.environment, self.normalize)


def refusal_class(outcome):
    refusal = outcome["refusal"]
    return f"{refusal['class']}: {refusal['message']}"


def same_outcome(expected, actual):
    """Whether a vector's expectation and a reference outcome agree: the same
    text and ids, or both refusing."""
    if "refusal" in expected or "refusal" in actual:
        return "refusal" in expected and "refusal" in actual
    return expected["rendered_text"] == actual["rendered_text"] and expected["token_ids"] == actual["token_ids"]


def text_difference(expected, actual):
    index = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), min(len(expected), len(actual)))
    return (f"first differs at byte {index}: expected {expected[max(index - 30, 0):index + 50]!r}, "
            f"got {actual[max(index - 30, 0):index + 50]!r}")


def describe_outcome(outcome):
    if "refusal" in outcome:
        refusal = outcome["refusal"]
        return f"a refusal {refusal}" if refusal else "a refusal"
    return f"a render of {len(outcome['token_ids'])} tokens"


# Expected outcomes ---------------------------------------------------------------

def expected_vector(model, reference, case, outcome, table, refusals=None):
    """A case's render vector and the divergence it follows: the reference
    outcome, or the one matching inputs entry's. A case matching more than one
    entry gets the ids of all, and must refuse only when all of them refuse."""
    request = case["request"]
    vector = {"id": case["id"], "raw": True, **request}
    entries = matching_entries(table, case["features"])
    divergence = None
    if len(entries) == 1:
        entry = entries[0]
        divergence = entry["id"]
        if entry["kind"] == "refused":
            vector["refusal"] = dict(entry.get("outcome") or {})
            return vector, divergence
        normalized = normalized_request(model, entry, request)
        outcome = reference.outcome(model.hf_input(normalized))
    elif len(entries) > 1:
        divergence = [entry["id"] for entry in entries]
        if all(entry["kind"] == "refused" for entry in entries):
            vector["refusal"] = {}
            return vector, divergence
    if "refusal" in outcome:
        vector["refusal"] = dict((refusals or {}).get(case["id"], {}))
    else:
        vector["rendered_text"] = outcome["rendered_text"]
        vector["token_ids"] = outcome["token_ids"]
    return vector, divergence


def normalized_request(model, entry, request):
    name = entry.get("normalize")
    function = getattr(model.cases, name, None) if name else None
    if function is None:
        raise CorpusError(f"inputs entry {entry['id']} names no normalize function in the cases module")
    return function(copy.deepcopy(request))


# Round trips --------------------------------------------------------------------------

def canonical_messages(messages, transport, reasoning_field="reasoning"):
    """`messages` with each assistant turn's reasoning under the declared
    reasoning field alone, as the transport reads it, so a vector reader
    that moves `reasoning_content` has nothing to move."""
    result = []
    for message in copy.deepcopy(messages):
        if message.get("role") == "assistant":
            reasoning = hf_messages([message], transport, reasoning_field)[0].get("reasoning_content")
            message.pop("reasoning_content", None)
            message.pop("reasoning", None)
            if reasoning is not None:
                message[reasoning_field] = reasoning
        result.append(message)
    return result


def round_ineligible(request, transport):
    """Why a case is no round-trip candidate before any render, or None: its
    turn is not the one generation writes (no final assistant turn, a
    continued one, media), or the transport would not carry it into a
    vector as written (a result or call without its id, arguments that are
    not a JSON object's text, a call to a tool the request does not offer).
    A candidate's arguments outside the parser's lossless domain are not a
    reason here but a failure (`round_vectors`)."""
    messages = request["messages"]
    if not messages or messages[-1].get("role") != "assistant":
        return "ends without an assistant turn"
    if request.get("continue_final_message"):
        return "continues the final message"
    for message in messages:
        if isinstance(message.get("content"), list) and any(
                isinstance(part, dict) and part.get("type") in (*HF_MEDIA, "input_audio")
                for part in message["content"]):
            return "carries media"
        if message.get("role") == "tool" and not isinstance(message.get("tool_call_id"), str):
            return "a tool result has no tool_call_id"
        for call in message.get("tool_calls") or []:
            if not isinstance(call.get("id"), str) or call.get("type") != "function":
                return "a tool call has no id or type"
            if not isinstance(call.get("function", {}).get("arguments"), str):
                return "a tool call's arguments are not a string"
    offered = {}
    for tool in request.get("tools") or []:
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
            offered[tool["function"].get("name")] = tool["function"]
    for call in messages[-1].get("tool_calls") or []:
        function = offered.get(call["function"].get("name"))
        if function is None:
            return "the turn calls a tool the request does not offer"
        try:
            arguments = json.loads(call["function"]["arguments"])
        except ValueError:
            return "the turn's arguments are not JSON"
        if not isinstance(arguments, dict):
            return "the turn's arguments are not a JSON object"
    return None


class RoundTripFailure:
    """Why a candidate's round trip fails: `kind`, its class, and what
    failed. A no-round-trip entry matches it by the case's features and
    `tags`: `round-trip:<kind>` and the generation prompt's `mode:chat` or
    `mode:thinking`.

    The kinds are `prefix` (the generation prompt is not a prefix of the
    history render), `stop` (the render holds no stop token and the model
    declares no endings), `read-back` (the cases module reads no turn back),
    the argument losses of `frontend_cases.argument_loss` (`union`,
    `untyped`, `mismatch`), `delimiter` (a value the parser scans as raw
    text spells a marker that ends it) and `tokenizer` (the tokenizer's
    normalizer changes the completion's text; the case still yields its
    round trip, expecting the normalized turn, so the parser is checked on
    the text it reads)."""

    def __init__(self, kind, message, mode):
        self.kind = kind
        self.message = message
        self.mode = mode

    @property
    def tags(self):
        return {f"round-trip:{self.kind}", f"mode:{self.mode}"}

    def report(self):
        return {"failure": self.message, "tags": sorted(self.tags)}

    def __str__(self):
        return self.message


def thinking_mode(keywords, declaration):
    """The generation prompt's mode, `thinking` or `chat`, as the served
    frontend selects it (frontend-format `select_mode_v1`): the first
    declared thinking keyword the template receives a boolean for
    (`hf_input`, which passes a top-level effort's `enable_thinking`)
    decides, and with none the declaration's `default_enabled`."""
    reasoning = declaration["reasoning"]
    switch = next((keywords[keyword] for keyword in reasoning["thinking_keywords"]
                   if isinstance(keywords.get(keyword), bool)), reasoning.get("default_enabled", True))
    return "thinking" if switch else "chat"


def argument_failure(message, tools, parser, arguments_of=None):
    """The first argument of the turn's calls the parser cannot read back
    as sent, as `(kind, detail)`, or None. `tools` are the request's
    offered tools and `parser` the declaration's format.parser section;
    `arguments_of`, a cases module's `turn_arguments`, reads a call's
    arguments as its template writes them, where the HF input keeps them as
    text. The typed grammar writes every value with the close that types
    it, so only a raw string can lose its value, to a marker it spells."""
    functions = {tool["function"].get("name"): tool["function"] for tool in tools or []
                 if isinstance(tool, dict) and isinstance(tool.get("function"), dict)}
    markers = frontend_cases.raw_markers(parser)
    typed = (parser or {}).get("tool_grammar") == "xml-typed-parameters-v1"
    for call in message.get("tool_calls") or []:
        function = functions.get(call["function"]["name"]) or {}
        arguments = call["function"]["arguments"]
        if arguments_of is not None:
            arguments = arguments_of(arguments)
        for name, value in arguments.items():
            if typed:
                if not isinstance(value, str):
                    continue
            else:
                loss = frontend_cases.argument_loss(value, function.get("parameters"), name)
                if loss:
                    return loss
                # The scan reads a raw value to the first marker that ends
                # one; in a JSON value a marker inside a string literal is
                # content.
                if frontend_cases.json_scanned(function, name):
                    continue
            spelled = [marker for marker in markers
                       if any(marker in text for text in frontend_cases.strings_of(value))]
            if spelled:
                return "delimiter", f"{name} is raw text that spells the marker {spelled[0]!r}"
    return None


def turn_shape(message, keywords, declaration):
    """The turn's ending kind and its shape: calls, reasoning and content, and
    the generation prompt's `thinking_mode`."""
    calls = message.get("tool_calls") or []
    base = "tool_calls" if calls else "answer"
    reasoning = bool(message.get("reasoning_content"))
    content = bool(message.get("content"))
    mode = thinking_mode(keywords, declaration)
    return base, f"{base}/calls={min(len(calls), 2)}/reasoning={int(reasoning)}/content={int(content)}/{mode}"


def ending_name(text):
    return "".join(character for character in text if character.isalnum() or character == "_") or "stop"


def continuation(message):
    """A message that follows the turn: its tool results, or a user turn."""
    calls = message.get("tool_calls") or []
    if calls:
        return [{"role": "tool", "tool_call_id": call["id"], "content": "ok"} for call in calls]
    return [{"role": "user", "content": "Next."}]


def round_vectors(model, reference, case):
    """The round-trip vectors of one case and a report: `(vectors, infos,
    failure, reason)`. The cases module's `expected_turn(message, context)`
    reads the turn back; `context` holds the generation `prompt` the
    completion follows and the case's `request`. `reason` is set when the
    case is not a round-trip candidate at all (`round_ineligible`), and
    `failure`, a `RoundTripFailure`, when a candidate's turn does not read
    back as sent, which needs a `no-round-trip` entry.

    The lossless domain is a candidate whose history render, less the
    generation prompt, is the completion (cut at its stop token or ended by
    a declared ending), whose turn the cases module reads back, whose every
    argument has the one type its schema names (a string where it names
    none, a number for an integer) and, where the parser scans it as raw
    text, spells no marker that ends it, and whose completion the
    tokenizer's normalizer leaves as it is. Outside it only the tokenizer
    class yields vectors: a control that expects the normalized turn
    (`normalized_turn`)."""
    request = case["request"]
    transport = model.transport
    reason = round_ineligible(request, transport)
    if reason:
        return [], {}, None, reason
    fields = {key: value for key, value in request.items()
              if key not in ("messages", "add_generation_prompt", "continue_final_message")}
    prefix_request = {"messages": request["messages"][:-1], **fields}
    full_request = {"messages": request["messages"], **fields, "add_generation_prompt": False}
    full_input = model.hf_input(full_request)
    mode = thinking_mode(full_input["keywords"], model.declaration)
    prefix = reference.outcome(model.hf_input(prefix_request))
    full = reference.outcome(full_input)
    if "refusal" in prefix or "refusal" in full:
        return [], {}, None, "the reference refuses the prefix or the conversation"
    prompt, rendered = prefix["rendered_text"], full["rendered_text"]
    if not rendered.startswith(prompt):
        return [], {}, RoundTripFailure(
            "prefix", "the generation prompt is not a prefix of the turn's history render", mode), None
    suffix = rendered[len(prompt):]
    stops = frontend_cases.stop_texts(model.declaration)
    last = full_input["messages"][-1]
    base, shape = turn_shape(last, full_input["keywords"], model.declaration)
    found = [(suffix.find(stop), stop) for stop in stops if stop in suffix]
    completions = []
    if found:
        position, stop = min(found)
        end = position + len(stop)
        completions.append((stop, suffix[:end], suffix[end:]))
    elif model.endings is None:
        return [], {}, RoundTripFailure(
            "stop", "the turn's history render holds no stop token and the model declares no endings", mode), None
    else:
        declared = model.endings.get(base) or []
        if not declared:
            raise CorpusError(f"{model.name}: ENDINGS declares no ending for {base} turns")
        for ending in declared:
            if ending not in stops:
                raise CorpusError(f"{model.name}: ending {ending!r} is not a declared stop token")
            completions.append((ending, suffix + ending, None))
        check_continuation(model, reference, full_request, rendered, base, declared)
    expected = model.cases.expected_turn(last, {"prompt": prompt, "request": request})
    if expected is None:
        return [], {}, RoundTripFailure(
            "read-back", "the cases module reads no turn back from this history render", mode), None
    loss = argument_failure(last, request.get("tools"), model.parser, getattr(model.cases, "turn_arguments", None))
    if loss:
        kind, detail = loss
        return [], {}, RoundTripFailure(kind, f"the parser cannot read the turn's arguments back: {detail}",
                                        mode), None
    failure = None
    if any(reference.normalize(completion) != completion for _, completion, _ in completions):
        # The bytes sent are lost to the tokenizer, not to the parser: the
        # vectors become a control whose expectation is the turn the
        # normalized text writes, so the parser still reads it back.
        expected = normalized_turn(model, reference, case, full_request, prompt, suffix, completions)
        failure = RoundTripFailure(
            "tokenizer", "the tokenizer's normalizer changes the completion text the parser reads; "
            "its round trip reads back the normalized turn", mode)
    expect = {key: expected[key] for key in ("reasoning", "content") if expected.get(key)}
    if expected.get("tool_calls"):
        expect["tool_calls"] = expected["tool_calls"]
    expect["finish_reason"] = "tool_calls" if expected.get("tool_calls") else "stop"
    vector_request = {"messages": canonical_messages(request["messages"][:-1], transport, model.reasoning_field),
                      **fields}
    changes = capture_normalization(vector_request, model.reasoning_field)
    if changes:
        raise CorpusError(f"case {case['id']}: capture normalization would change its round-trip request: "
                          + "; ".join(changes))
    vectors, infos = [], {}
    prompt_ids = reference.tokenize(prompt)
    for ending, completion, separator in completions:
        identifier = f"{case['id']}-round-{ending_name(ending)}"
        ids = reference.tokenize(completion)
        vector = {"id": identifier, "request": vector_request, "token_ids": ids, "expect": expect}
        if separator is not None:
            vector["separator"] = separator
        vectors.append(vector)
        infos[identifier] = {"shape": shape, "ending": ending, "case": case["id"],
                             "boundary_merges": reference.tokenize(prompt + completion) != prompt_ids + ids}
    return vectors, infos, failure, None


def normalized_strings(value, normalize):
    """`value` with every string in it, keys included, normalized; a tool
    call's arguments text is normalized as the JSON value it holds."""
    if isinstance(value, str):
        return normalize(value)
    if isinstance(value, list):
        return [normalized_strings(item, normalize) for item in value]
    if isinstance(value, dict):
        result = {normalize(key): normalized_strings(item, normalize) for key, item in value.items()}
        function = result.get("function")
        if isinstance(function, dict) and isinstance(value["function"].get("arguments"), str):
            try:
                arguments = json.loads(value["function"]["arguments"])
            except ValueError:
                return result
            function["arguments"] = json.dumps(normalized_strings(arguments, normalize), ensure_ascii=False)
        return result
    return value


def normalized_turn(model, reference, case, request, prompt, suffix, completions):
    """The turn a round trip reads back when the tokenizer's normalizer
    rewrites its completion: the cases module's reading of the final turn
    with every string normalized. It holds only when that rewrite is all
    that changed: the normalized turn renders after the same prompt as the
    normalized completion, and both texts tokenize to the same ids, so the
    vector's ids are the original's and only its expectation changes."""
    messages = request["messages"][:-1] + [normalized_strings(request["messages"][-1], reference.normalize)]
    normalized = {**request, "messages": messages}
    outcome = reference.outcome(model.hf_input(normalized))
    rendered = outcome.get("rendered_text", "")
    if not rendered.startswith(prompt) or rendered[len(prompt):] != reference.normalize(suffix):
        raise CorpusError(f"case {case['id']}: the normalized turn does not render as the normalized completion")
    for _, completion, _ in completions:
        if reference.tokenize(reference.normalize(completion)) != reference.tokenize(completion):
            raise CorpusError(f"case {case['id']}: the normalized completion tokenizes to other ids")
    last = model.hf_input(normalized)["messages"][-1]
    expected = model.cases.expected_turn(last, {"prompt": prompt, "request": normalized})
    if expected is None:
        raise CorpusError(f"case {case['id']}: the cases module reads no normalized turn back")
    return expected


def check_continuation(model, reference, request, rendered, base, endings):
    """When the conversation continues past the turn, the render must go on
    with one of the endings declared for the turn's shape."""
    messages = request["messages"] + continuation(request["messages"][-1])
    outcome = reference.outcome(model.hf_input({**request, "messages": messages}))
    if "refusal" in outcome or not outcome["rendered_text"].startswith(rendered):
        return
    tail = outcome["rendered_text"][len(rendered):]
    if not any(tail.startswith(ending) for ending in endings):
        raise CorpusError(f"{model.name}: a {base} turn is followed by {tail[:40]!r}, "
                          f"which begins with none of its endings {endings}")


# generate ---------------------------------------------------------------------------------

def template_source(model, template=None):
    """The reference source and its digest: the checkpoint's chat template,
    or the file the pin names as `source` (a Python encoder) with its
    `source_sha256`. The model's own source must match the pin; one passed
    explicitly (a drill) is read as it is."""
    name = model.pin.get("source", "chat_template.jinja")
    pinned = model.pin["source_sha256"] if "source" in model.pin else model.pin["template_sha256"]
    path = Path(template) if template else model.checkpoint() / name
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if template is None and digest != pinned:
        raise CorpusError(f"{path.name} is {digest}, not the pinned {pinned} "
                          f"(revision {model.pin['revision']})")
    return data.decode(), digest


def model_reference(model, source):
    """The reference a model's pin names: its Python encoder through the
    cases module's `encoder_render` when the pin names a `source`, else HF
    at its chat template."""
    if "source" in model.pin:
        return Reference.python_encoder(model.checkpoint(), source, model.cases.encoder_render)
    return Reference.hf(model.checkpoint(), source)


def build_corpus(model, reference, cases, table, refusals=None):
    """Every case's vector, coverage and reference outcome, and the round
    trips: a dictionary of the corpus files' contents."""
    vectors, coverage_cases, outcomes = [], {}, {}
    rounds, round_infos, report = [], {}, {}
    for case in cases:
        outcome = reference.outcome(model.hf_input(case["request"]))
        vector, divergence = expected_vector(model, reference, case, outcome, table, refusals)
        problems = vector_problems(vector)
        if problems:
            raise CorpusError(f"case {case['id']}: " + "; ".join(problems))
        vectors.append(vector)
        coverage_cases[case["id"]] = {"features": case["features"], "lines": outcome["lines"],
                                      "reference": "refusal" if "refusal" in outcome else "render",
                                      "divergence": divergence}
        outcomes[case["id"]] = {key: value for key, value in outcome.items() if key != "lines"}
        if yields_no_round_trip(table, case["features"], "refusal" in vector):
            continue
        found, infos, failure, reason = round_vectors(model, reference, case)
        rounds += found
        round_infos.update(infos)
        if failure or reason:
            report[case["id"]] = failure.report() if failure else {"skipped": reason}
        if failure:
            coverage_cases[case["id"]]["divergence"] = case_divergence(table, case["features"], failure)
    return {"vectors": vectors, "cases": coverage_cases, "outcomes": outcomes, "rounds": rounds,
            "round_infos": round_infos, "report": report}


def coverage_document(model, reference, digest, seed, full_cases, cases, rounds):
    covered = set().union(*(set(case["lines"]) for case in cases.values())) if cases else set()
    return {
        "schema": COVERAGE_SCHEMA,
        "model": model.name,
        "revision": model.pin["revision"],
        "template_sha256": digest,
        "transport": {key: model.transport[key] for key in TRANSPORT_KEYS},
        "environment": reference.environment,
        "seed": seed,
        "full_cases": full_cases,
        "lines_with_code": sorted(reference.coverage.lines_with_code),
        "cases": cases,
        "rounds": rounds,
        "uncovered_lines": sorted(reference.coverage.lines_with_code - covered),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")


def generate(model, output, seed=0, divergences=None, template=None, refusals=None, reference=None):
    source, digest = template_source(model, template)
    if reference is None:
        reference = model_reference(model, source)
    reference = reference.with_reference_text(model.reference_text)
    table = read_divergences(divergences if divergences is not None else model.divergences_path)
    cases = model.full_cases(seed)
    corpus = build_corpus(model, reference, cases, table, refusals)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "corpus.json", corpus["vectors"])
    write_json(output / "corpus-rounds.json", corpus["rounds"])
    document = coverage_document(model, reference, digest, seed, len(cases), corpus["cases"],
                                 {key: {"shape": info["shape"], "ending": info["ending"]}
                                  for key, info in corpus["round_infos"].items()})
    write_json(output / "corpus-coverage.json", document)
    write_json(output / "reference.json", corpus["outcomes"])
    write_json(output / "rounds-report.json", {"rounds": corpus["round_infos"], "cases": corpus["report"]})
    return summary(document, corpus)


def summary(document, corpus=None):
    cases = document["cases"]
    features = set().union(*(set(case["features"]) for case in cases.values())) if cases else set()
    refusals = sum(case["reference"] == "refusal" for case in cases.values())
    lines = document["lines_with_code"]
    result = {"cases": len(cases), "lines_with_code": len(lines),
              "lines_covered": len(lines) - len(document["uncovered_lines"]),
              "uncovered_lines": document["uncovered_lines"], "features": len(features),
              "reference_refusals": refusals, "round_trip_vectors": len(document["rounds"]),
              "divergence_cases": sum(case["divergence"] is not None for case in cases.values())}
    if corpus is not None:
        failures = [key for key, value in corpus["report"].items() if "failure" in value]
        result["round_trip_failures"] = failures
    return result


# minimize ---------------------------------------------------------------------------

def read_corpus(directory):
    directory = Path(directory)
    vectors = json.loads((directory / "corpus.json").read_text())
    rounds_path = directory / "corpus-rounds.json"
    rounds = json.loads(rounds_path.read_text()) if rounds_path.exists() else []
    coverage = json.loads((directory / "corpus-coverage.json").read_text())
    if coverage.get("schema") != COVERAGE_SCHEMA:
        raise CorpusError(f"{directory}: corpus-coverage.json schema must be {COVERAGE_SCHEMA}")
    outcomes_path = directory / "reference.json"
    outcomes = json.loads(outcomes_path.read_text()) if outcomes_path.exists() else None
    return vectors, rounds, coverage, outcomes


def minimize_selection(coverage, outcomes):
    """The ids of a subset of the full corpus that keeps every template line
    and feature tag, every reference-refusal class, every divergence case and
    one round-trip vector per shape and ending. A case that matches several
    inputs entries has no declared outcome, so the corpus is refused until the
    entries' `when` lists tell its divergences apart."""
    cases = coverage["cases"]
    ambiguous = [key for key, case in cases.items() if isinstance(case["divergence"], list)]
    if ambiguous:
        raise CorpusError("cases match several inputs entries: " + ", ".join(
            f"{key} ({', '.join(cases[key]['divergence'])})" for key in ambiguous))
    eligible = list(cases)
    chosen = []

    def choose(key):
        if key not in chosen:
            chosen.append(key)

    for key in eligible:
        if cases[key]["divergence"] is not None:
            choose(key)
    classes = {}
    for key in eligible:
        outcome = (outcomes or {}).get(key, {})
        if "refusal" in outcome:
            classes.setdefault(refusal_class(outcome), key)
    for key in classes.values():
        choose(key)
    shapes = {}
    for identifier, info in coverage["rounds"].items():
        case = identifier.rsplit("-round-", 1)[0]
        if case in eligible:
            shapes.setdefault((info["shape"], info["ending"]), case)
    for key in shapes.values():
        choose(key)

    def units(key):
        return {("line", line) for line in cases[key]["lines"]} | {("tag", tag) for tag in cases[key]["features"]}

    wanted = set().union(*(units(key) for key in eligible)) if eligible else set()
    covered = set().union(*(units(key) for key in chosen)) if chosen else set()
    remaining = [key for key in eligible if key not in chosen]
    while wanted - covered:
        best = max(remaining, key=lambda key: (len(units(key) - covered), -remaining.index(key)))
        gain = units(best) - covered
        if not gain:
            break
        choose(best)
        covered |= gain
        remaining.remove(best)
    order = {key: index for index, key in enumerate(cases)}
    return sorted(chosen, key=order.__getitem__)


def is_fixed(features):
    """Whether a case is a fixed case (a `fixed:<id>` tag): an explicit
    regression, which minimization keeps with every round trip it yields."""
    return any(tag.startswith("fixed:") for tag in features)


def minimize(model, corpus, output):
    vectors, rounds, coverage, outcomes = read_corpus(corpus)
    if outcomes is None:
        raise CorpusError(f"{corpus}: minimize needs a full corpus with reference.json")
    report_path = Path(corpus) / "rounds-report.json"
    if not report_path.exists():
        raise CorpusError(f"{corpus}: minimize needs a full corpus with rounds-report.json")
    report = json.loads(report_path.read_text())["cases"]
    undeclared = [key for key, value in report.items()
                  if "failure" in value and coverage["cases"][key]["divergence"] is None]
    if undeclared:
        raise CorpusError("round trips fail without a no-round-trip entry: " + ", ".join(
            f"{key} ({report[key]['failure']})" for key in undeclared))
    chosen = minimize_selection(coverage, outcomes)
    keep = set(chosen)
    fixed = {key for key in keep if is_fixed(coverage["cases"][key]["features"])}
    shapes, selected_rounds = set(), []
    for vector in (vector for vector in rounds if vector["id"].rsplit("-round-", 1)[0] in keep):
        info = coverage["rounds"][vector["id"]]
        if (info["shape"], info["ending"]) not in shapes or vector["id"].rsplit("-round-", 1)[0] in fixed:
            shapes.add((info["shape"], info["ending"]))
            selected_rounds.append(vector)
    round_ids = {vector["id"] for vector in selected_rounds}
    document = dict(coverage)
    document["cases"] = {key: coverage["cases"][key] for key in chosen}
    document["rounds"] = {key: value for key, value in coverage["rounds"].items() if key in round_ids}
    covered = set().union(*(set(case["lines"]) for case in document["cases"].values())) if chosen else set()
    document["uncovered_lines"] = sorted(set(coverage["lines_with_code"]) - covered)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "corpus.json", [vector for vector in vectors if vector["id"] in keep])
    write_json(output / "corpus-rounds.json", selected_rounds)
    write_json(output / "corpus-coverage.json", document)
    return summary(document)


# check -------------------------------------------------------------------------------------

def check(model, fixtures=None, divergences=None, reference=None, template=None):
    """Every way the committed corpus fails the gate, as messages; empty when
    it passes."""
    fixtures = Path(fixtures) if fixtures else model.fixtures
    source, digest = template_source(model, template)
    if reference is None:
        reference = model_reference(model, source)
    reference = reference.with_reference_text(model.reference_text)
    table = read_divergences(divergences if divergences is not None else model.divergences_path)
    vectors, rounds, coverage, _ = read_corpus(fixtures)
    for difference in environment_differences(coverage.get("environment"), reference.environment):
        warnings.warn(f"{model.name}: {difference}; outcomes may differ for that reason alone",
                      ReferenceEnvironmentWarning, stacklevel=2)
    errors = []
    if coverage["template_sha256"] != digest:
        errors.append(f"corpus-coverage.json pins template {coverage['template_sha256']}, not {digest}")
    if coverage["transport"] != {key: model.transport[key] for key in TRANSPORT_KEYS}:
        errors.append("corpus-coverage.json records another transport than the cases module's TRANSPORT")
    lines_with_code = reference.coverage.lines_with_code
    if set(coverage["lines_with_code"]) != lines_with_code:
        errors.append("corpus-coverage.json lists other lines with code than the template has")
    ids = [vector["id"] for vector in vectors]
    if set(ids) != set(coverage["cases"]) or len(set(ids)) != len(ids):
        errors.append("corpus.json and corpus-coverage.json name different cases")
    outcomes = {}
    matched_reference = {}
    covered = set()
    for vector in vectors:
        identifier = vector["id"]
        recorded = coverage["cases"].get(identifier)
        if recorded is None:
            continue
        if vector.get("raw") is not True:
            errors.append(f"case {identifier}: a corpus vector must set raw")
        errors += [f"case {identifier}: {problem}" for problem in vector_problems(vector)]
        request = request_of(vector)
        for tag in recorded["features"]:
            if frontend_cases.exercised(tag, request) is False:
                errors.append(f"case {identifier}: is tagged {tag}, which its request does not carry")
        outcome = reference.outcome(model.hf_input(request))
        outcomes[identifier] = outcome
        covered |= set(outcome["lines"])
        if outcome["lines"] != recorded["lines"]:
            errors.append(f"case {identifier}: corpus-coverage.json records other lines than the render runs")
        expected = vector
        agrees = same_outcome(expected, outcome)
        matched_reference[identifier] = agrees
        entries = matching_entries(table, recorded["features"])
        if len(entries) > 1:
            errors.append(f"case {identifier}: matches more than one inputs entry "
                          f"({', '.join(entry['id'] for entry in entries)})")
            continue
        if not entries:
            if not agrees:
                detail = (text_difference(outcome["rendered_text"], expected["rendered_text"])
                          if "rendered_text" in outcome and "rendered_text" in expected else
                          f"the reference gives {describe_outcome(outcome)}, the corpus "
                          f"{describe_outcome(expected)}")
                errors.append(f"case {identifier}: differs from the reference and matches no inputs entry: {detail}")
            continue
        entry = entries[0]
        if entry["kind"] == "refused":
            declared = entry.get("outcome") or {}
            if "refusal" not in expected or any(expected["refusal"].get(key) != value
                                                for key, value in declared.items()):
                errors.append(f"case {identifier}: matches refused entry {entry['id']} but the corpus does not "
                              f"expect its refusal {declared}")
        else:
            normalized = reference.outcome(model.hf_input(normalized_request(model, entry, request)))
            if not same_outcome(expected, normalized):
                errors.append(f"case {identifier}: matches normalized entry {entry['id']} but the corpus does not "
                              f"expect the reference's render of the normalized request")
    for entry in table["inputs"]:
        if entry["kind"] == "no-round-trip":
            continue
        cases = [identifier for identifier in outcomes
                 if set(entry["when"]) <= set(coverage["cases"][identifier]["features"])]
        if not cases:
            errors.append(f"inputs entry {entry['id']} matches no committed case")
        elif all(matched_reference[identifier] for identifier in cases):
            errors.append(f"inputs entry {entry['id']} is stale: its cases now match the reference")
    errors += check_lines(table, lines_with_code, covered, coverage, reference)
    errors += [f"mapping entry {entry['id']} names case {case}, which the corpus does not hold"
               for entry in table["mapping"] for case in entry["cases"] if case not in outcomes]
    round_errors, failing = check_rounds(model, reference, vectors, rounds, coverage, table,
                                         model.full_cases(coverage["seed"]))
    errors += round_errors
    for identifier in outcomes:
        recorded = coverage["cases"][identifier]["divergence"]
        expected = case_divergence(table, coverage["cases"][identifier]["features"], failing.get(identifier))
        if recorded != expected:
            errors.append(f"case {identifier}: corpus-coverage.json records divergence {recorded!r}, not {expected!r}")
    errors += check_features(model, coverage, table)
    return errors


def environment_differences(recorded, current):
    """How the reference environment `current` differs from the one a
    corpus recorded, as messages; empty when they agree."""
    if not recorded:
        return [f"the corpus records no reference environment; this check runs {current}"]
    return [f"the corpus was generated with {key} {recorded.get(key)}, this check runs {current.get(key)}"
            for key in sorted(set(recorded) | set(current)) if recorded.get(key) != current.get(key)]


def check_lines(table, lines_with_code, covered, coverage, reference):
    errors = []
    listed = set()
    for entry in table["lines"]:
        run = sorted(set(entry["lines"]) & covered)
        if run:
            errors.append(f"lines entry {entry['id']} is stale: committed cases run lines {run}")
        listed |= set(entry["lines"])
    uncovered = sorted(lines_with_code - covered)
    if uncovered != coverage["uncovered_lines"]:
        errors.append(f"corpus-coverage.json records uncovered lines {coverage['uncovered_lines']}, not {uncovered}")
    for line in uncovered:
        if line not in listed:
            errors.append(f"template line {line} is run by no committed case and listed in no lines entry: "
                          f"{reference.coverage.line_text(line).strip()[:100]}")
    return errors


def expected_refusal(model, reference, case, table):
    """Whether a case's vector expects a refusal, as generate writes it."""
    entries = matching_entries(table, case["features"])
    if any(entry["kind"] == "refused" for entry in entries):
        return True
    request = case["request"]
    if len(entries) == 1:
        request = normalized_request(model, entries[0], request)
    return "refusal" in reference.outcome(model.hf_input(request))


def undeclared_round_trip(name, failure, table, features):
    """The error of a failed round trip that matches no no-round-trip entry,
    or several, by the case's features and the failure's tags; none when it
    matches exactly one."""
    entries = matching_entries(table, set(features) | failure.tags, ("no-round-trip",))
    if len(entries) == 1:
        return []
    matched = f"several no-round-trip entries ({', '.join(entry['id'] for entry in entries)})" if entries else \
        "no no-round-trip entry"
    return [f"{name}: {failure}, and it matches {matched}"]


def check_rounds(model, reference, vectors, rounds, coverage, table, full_cases):
    """The committed round trips are recomputed byte for byte, each with its
    separator and its recorded shape and ending. Every shape and ending the
    full corpus yields has a committed vector, and every round trip of a
    committed fixed case is committed. A case whose round trip fails,
    committed or only in the full corpus, must match exactly one
    no-round-trip entry, which goes stale unless a committed case fails as
    it names."""
    errors = []
    recomputed, infos, failing = {}, {}, {}
    by_id = {vector["id"]: vector for vector in vectors}
    for vector in vectors:
        features = coverage["cases"].get(vector["id"], {}).get("features", [])
        if yields_no_round_trip(table, features, "refusal" in vector):
            continue
        found, info, failure, _ = round_vectors(model, reference, {"id": vector["id"], "request": request_of(vector),
                                                                   "features": features})
        recomputed.update((round_vector["id"], round_vector) for round_vector in found)
        infos.update(info)
        if failure:
            failing[vector["id"]] = failure
            errors += undeclared_round_trip(f"case {vector['id']}", failure, table, features)
    for entry in table["inputs"]:
        if entry["kind"] != "no-round-trip":
            continue
        # An entry is exercised by a committed case whose round trip fails
        # the way it names.
        if not any(set(entry["when"]) <= set(coverage["cases"].get(key, {}).get("features", [])) | failure.tags
                   for key, failure in failing.items()):
            errors.append(f"inputs entry {entry['id']} is stale: no committed case fails its round trip that way")
    committed = set()
    present = {vector["id"] for vector in rounds}
    for identifier in sorted(recomputed):
        case = identifier.rsplit("-round-", 1)[0]
        if identifier not in present and is_fixed(coverage["cases"].get(case, {}).get("features", [])):
            errors.append(f"round trip {identifier}: fixed case {case} yields it, but it is not committed")
    for vector in rounds:
        expected = recomputed.get(vector["id"])
        if expected is None:
            errors.append(f"round trip {vector['id']}: no committed case yields it")
            continue
        if json.dumps(expected, sort_keys=True) != json.dumps(vector, sort_keys=True):
            errors.append(f"round trip {vector['id']}: differs from the recomputed vector")
        info = infos[vector["id"]]
        shape = {"shape": info["shape"], "ending": info["ending"]}
        committed.add((info["shape"], info["ending"]))
        recorded = coverage["rounds"].get(vector["id"])
        if recorded != shape:
            errors.append(f"round trip {vector['id']}: corpus-coverage.json records {recorded}, not {shape}")
    if {vector["id"] for vector in rounds} != set(coverage["rounds"]):
        errors.append("corpus-rounds.json and corpus-coverage.json name different round trips")
    required = set()
    for case in full_cases:
        if yields_no_round_trip(table, case["features"], expected_refusal(model, reference, case, table)):
            continue
        _, info, failure, _ = round_vectors(model, reference, case)
        required |= {(value["shape"], value["ending"]) for value in info.values()}
        if failure:
            errors += undeclared_round_trip(f"full corpus case {case['id']}", failure, table, case["features"])
    for shape, ending in sorted(required - committed):
        errors.append(f"no committed round trip has shape {shape} and ending {ending!r}, which the full corpus yields")
    return errors, failing


def check_features(model, coverage, table):
    """The committed cases keep every feature tag the full corpus exercises,
    regenerated from the recorded seed, and no full-corpus case matches
    several inputs entries."""
    errors = []
    full = set()
    for case in model.full_cases(coverage["seed"]):
        full |= set(case["features"])
        entries = matching_entries(table, case["features"])
        if len(entries) > 1:
            errors.append(f"full corpus case {case['id']} matches several inputs entries "
                          f"({', '.join(entry['id'] for entry in entries)})")
    kept = set().union(*(set(case["features"]) for case in coverage["cases"].values())) if coverage["cases"] else set()
    missing = sorted(full - kept)
    if missing:
        errors.append(f"the committed cases lack feature tags of the full corpus: {', '.join(missing)}")
    return errors


# The model reference test ----------------------------------------------------------

def corpus_test_case(model_name, root=ROOT):
    """The unittest case of `models/<m>/test_frontend_corpus.py`: `check`
    over the committed corpus, skipped when neither the tokenizer directory
    nor the checkpoint is set (`reference_variables`)."""

    class FrontendCorpus(unittest.TestCase):
        def test_the_committed_corpus_passes_the_gate(self):
            model = Model(model_name, root)
            variables = reference_variables(model.pin)
            if not any(os.environ.get(name) for name in variables):
                self.skipTest(f"{' and '.join(variables)} are unset")
            errors = check(model)
            self.assertEqual(errors, [], "\n".join(errors))

    FrontendCorpus.__name__ = FrontendCorpus.__qualname__ = f"FrontendCorpus_{model_name}"
    return FrontendCorpus


# CLI --------------------------------------------------------------------------------------

def default_output(model):
    data = os.environ.get("CHOPINFER_DATA")
    if not data:
        raise CorpusError("CHOPINFER_DATA is unset; pass --output")
    return Path(data) / "frontend-corpus" / model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("generate", "minimize", "check", "rounds"):
        command = commands.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--divergences", type=Path, help="a divergence table other than the model's")
        command.add_argument("--template", type=Path, help="an unpinned template to run instead (a drill)")
        if name == "generate":
            command.add_argument("--output", type=Path)
            command.add_argument("--seed", type=int, default=0)
            command.add_argument("--refusals", type=Path,
                                 help="JSON object of case id to the refusal the frontend gives a shared refusal")
        if name == "minimize":
            command.add_argument("--corpus", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
        if name == "check":
            command.add_argument("--fixtures", type=Path)
        if name == "rounds":
            command.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        model = Model(args.model)
        if args.command == "generate":
            refusals = json.loads(args.refusals.read_text()) if args.refusals else None
            result = generate(model, args.output or default_output(args.model), args.seed, args.divergences,
                              args.template, refusals)
        elif args.command == "minimize":
            result = minimize(model, args.corpus, args.output)
        elif args.command == "rounds":
            result = recompute_rounds(model, args.corpus, args.divergences, args.template)
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ReferenceEnvironmentWarning)
                errors = check(model, args.fixtures, args.divergences, template=args.template)
            for warning in caught:
                if issubclass(warning.category, ReferenceEnvironmentWarning):
                    print(f"WARN {warning.message}")
                else:
                    warnings.showwarning(warning.message, warning.category, warning.filename, warning.lineno)
            for error in errors:
                print(f"FAIL {error}")
            print(f"{args.model}: {'gate passes' if not errors else f'{len(errors)} gate failures'}")
            return 1 if errors else 0
    except CorpusError as error:
        print(f"frontend_corpus: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=1))
    return 0


def recompute_rounds(model, corpus, divergences=None, template=None, reference=None):
    """Recomputes the round-trip vectors of a corpus directory's cases."""
    source, _ = template_source(model, template)
    if reference is None:
        reference = model_reference(model, source)
    reference = reference.with_reference_text(model.reference_text)
    table = read_divergences(divergences if divergences is not None else model.divergences_path)
    corpus = Path(corpus)
    vectors, _, coverage, _ = read_corpus(corpus)
    rounds, infos, report = [], {}, {}
    for vector in vectors:
        features = coverage["cases"][vector["id"]]["features"]
        if yields_no_round_trip(table, features, "refusal" in vector):
            continue
        found, info, failure, reason = round_vectors(model, reference, {"id": vector["id"],
                                                                       "request": request_of(vector),
                                                                       "features": features})
        rounds += found
        infos.update(info)
        if failure or reason:
            report[vector["id"]] = failure.report() if failure else {"skipped": reason}
    coverage["rounds"] = {key: {"shape": info["shape"], "ending": info["ending"]} for key, info in infos.items()}
    write_json(corpus / "corpus-rounds.json", rounds)
    write_json(corpus / "corpus-coverage.json", coverage)
    return {"round_trip_vectors": len(rounds),
            "round_trip_failures": [key for key, value in report.items() if "failure" in value]}


if __name__ == "__main__":
    sys.exit(main())

"""The frontend corpus harness on a toy template under plain jinja2.

No transformers, torch or numpy: the toy template compiles with a jinja2
environment configured as transformers configures its own, and tokenizes one
id per character.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import tomllib
from types import SimpleNamespace
import unicodedata
import unittest
import unittest.mock
import warnings

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

from tools.refrun import frontend_cases
from tools.refrun import frontend_corpus as corpus
from tools.refrun.template_coverage import PythonSourceCoverage, TemplateCoverage

TOY = """\
{%- set effort = reasoning_effort if reasoning_effort in ['low', 'high'] else 'max' -%}
{{- '<sys>' + effort -}}
{%- for m in messages -%}
{%- if m.role == 'user' -%}
{{- '<user>' + m.content -}}
{%- elif m.role == 'assistant' -%}
{{- '<asst>' + (m.reasoning_content or '') + '|' + (m.content or '') -}}
{%- for c in m.tool_calls or [] -%}
{{- '<call>' + c.function.name + (c.function.arguments | tojson) -}}
{%- endfor -%}
{{- '<end>\\n' -}}
{%- elif m.role == 'tool' -%}
{{- '<tool>' + m.content -}}
{%- else -%}
{{- raise_exception('unknown role ' + m.role) -}}
{%- endif -%}
{%- endfor -%}
{%- if mode == 'legacy' -%}
{{- '<legacy>' + effort -}}
{%- endif -%}
{%- if add_generation_prompt -%}
{{- '<asst>' -}}
{%- endif -%}"""
LEGACY_LINE = 19


def toy_compile(source):
    """A jinja2 environment configured as transformers' chat-template one."""
    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=["jinja2.ext.loopcontrols"])
    environment.filters["tojson"] = lambda value, **_: json.dumps(value, ensure_ascii=False)
    environment.globals["raise_exception"] = raise_exception
    return environment.from_string(source)


def toy_reference(source=TOY):
    coverage = TemplateCoverage(source, compile=toy_compile)

    def render(messages, tools, add_generation_prompt, continue_final_message, keywords):
        return coverage.template.render(messages=messages, tools=tools,
                                        add_generation_prompt=add_generation_prompt, **keywords)

    return corpus.Reference(coverage, render, lambda text: [ord(character) for character in text])


def toy_declaration(stops=("<end>", "<user>")):
    tokens = [{"text": text, "id": index} for index, text in enumerate(stops)]
    return {"reasoning": {"efforts": [{"name": "low"}, {"name": "high"}], "disable_value": "none",
                          "thinking_keywords": ["enable_thinking"], "allow_disabled": True},
            "limits": {"maximum_messages": 4096, "maximum_tools": 128},
            "vocabulary": {"stop_token_ids": [token["id"] for token in tokens], "boundary_tokens": tokens}}


# Every in-tree model that has a frontend cases module, found rather than named.
CASES_MODELS = sorted(path.parent.name for path in corpus.ROOT.glob("models/*/frontend_cases.py"))

TRANSPORT = {"reasoning_content_fallback": True, "tool_call_reasoning_fallback": False,
             "reads_tool_objects": True, "tool_references": False, "extra_roles": []}


def toy_cases_module(endings=None):
    def expected_turn(message, context):
        assert context["prompt"].endswith("<asst>") and "messages" in context["request"], context
        return {"reasoning": message.get("reasoning_content") or "", "content": message.get("content") or "",
                "tool_calls": [{"name": call["function"]["name"],
                                "arguments": json.dumps(call["function"]["arguments"], ensure_ascii=False)}
                               for call in message.get("tool_calls") or []]}

    return SimpleNamespace(PIN={"revision": "toy", "template_sha256": "", "checkpoint_env": "TOY_CHECKPOINT"},
                           TRANSPORT=TRANSPORT, ENDINGS=endings, expected_turn=expected_turn)


def user(text="hi"):
    return {"role": "user", "content": text}


def call(name, arguments, identifier="call_0"):
    return {"id": identifier, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


TOOL = {"type": "function", "function": {"name": "f", "parameters": {
    "type": "object", "properties": {"n": {"type": "integer"}}}}}

CASES = [
    {"id": "low", "request": {"messages": [user()], "reasoning_effort": "low"}, "features": ["effort:low"]},
    {"id": "high", "request": {"messages": [user()], "reasoning_effort": "high"}, "features": ["effort:high"]},
    {"id": "undeclared", "request": {"messages": [user()], "reasoning_effort": "turbo"},
     "features": ["effort:undeclared"]},
    {"id": "int", "request": {"messages": [user(), {"role": "assistant", "content": "", "tool_calls": [
        call("f", {"n": 3})]}, {"role": "tool", "tool_call_id": "call_0", "content": "ok"}], "tools": [TOOL]},
     "features": ["arg:int"]},
    {"id": "float", "request": {"messages": [user(), {"role": "assistant", "content": "", "tool_calls": [
        call("f", {"n": 0.5})]}, {"role": "tool", "tool_call_id": "call_0", "content": "ok"}], "tools": [TOOL]},
     "features": ["arg:float"]},
    {"id": "answer", "request": {"messages": [user(), {"role": "assistant", "content": "Done.",
                                                       "reasoning": "r"}]},
     "features": ["ends:assistant"]},
    {"id": "called", "request": {"messages": [user(), {"role": "assistant", "content": "", "tool_calls": [
        call("f", {"n": 1})]}], "tools": [TOOL]}, "features": ["ends:assistant", "calls:1"]},
    {"id": "critic", "request": {"messages": [user(), {"role": "critic", "content": "no"}]},
     "features": ["role:critic"]},
    {"id": "legacy", "request": {"messages": [user()], "chat_template_kwargs": {"mode": "legacy"}},
     "features": ["mode:legacy"]},
]

WITHOUT_LEGACY = [case for case in CASES if case["id"] != "legacy"]
# The same features and lines as "low": minimization drops one of them.
REDUNDANT = {"id": "low-again", "request": {"messages": [user("hey")], "reasoning_effort": "low"},
             "features": ["effort:low"]}

UNDECLARED = {"id": "undeclared-effort", "kind": "refused", "when": ["effort:undeclared"],
              "outcome": {"parameter": "reasoning_effort"}, "reason": "the template remaps it to max"}
LEGACY = {"id": "legacy-mode", "lines": [LEGACY_LINE], "blocked_by": "mode is not forwarded",
          "reason": "the transport never sends mode"}


class ToyModel(corpus.Model):
    def __init__(self, root, cases=CASES, endings=None, stops=("<end>", "<user>")):
        super().__init__("toy", root, toy_cases_module(endings), toy_declaration(stops))
        self.cases_list = cases

    def full_cases(self, seed):
        return copy.deepcopy(self.cases_list)


class Harness(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        self.template = self.directory / "chat_template.jinja"
        self.use_template(TOY)

    def use_template(self, source):
        self.source = source
        self.template.write_text(source)

    def table(self, lines=(), inputs=()):
        path = self.directory / "divergences.json"
        path.write_text(json.dumps({"schema": corpus.DIVERGENCES_SCHEMA, "lines": list(lines),
                                    "inputs": list(inputs)}))
        return path

    def generate(self, model, table, name="full"):
        output = self.directory / name
        corpus.generate(model, output, divergences=table, template=self.template,
                        reference=toy_reference(self.source))
        return output

    def check(self, model, fixtures, table):
        return corpus.check(model, fixtures, table, reference=toy_reference(self.source), template=self.template)

    def committed(self, model, table, keep):
        """A committed corpus of the cases named in `keep`."""
        full = self.generate(model, table)
        vectors, rounds, coverage, _ = corpus.read_corpus(full)
        output = self.directory / "committed"
        output.mkdir(exist_ok=True)
        corpus.write_json(output / "corpus.json", [vector for vector in vectors if vector["id"] in keep])
        cases = {key: value for key, value in coverage["cases"].items() if key in keep}
        covered = set().union(*(set(case["lines"]) for case in cases.values()))
        kept_rounds = [vector for vector in rounds if vector["id"].rsplit("-round-", 1)[0] in keep]
        corpus.write_json(output / "corpus-rounds.json", kept_rounds)
        corpus.write_json(output / "corpus-coverage.json", {
            **coverage, "cases": cases,
            "rounds": {key: value for key, value in coverage["rounds"].items()
                       if key in {vector["id"] for vector in kept_rounds}},
            "uncovered_lines": sorted(set(coverage["lines_with_code"]) - covered)})
        return output


class GeneratorTests(unittest.TestCase):
    def test_one_seed_always_yields_the_same_cases(self):
        declaration = toy_declaration()
        first = frontend_cases.generate(declaration, None, seed=7)
        self.assertEqual(first, frontend_cases.generate(declaration, None, seed=7))
        self.assertNotEqual(first, frontend_cases.generate(declaration, None, seed=8))
        for case in first:
            json.dumps(case["request"])
            self.assertTrue(all(tag.count(":") >= 1 for tag in case["features"]), case["features"])

    def test_every_value_is_realized_and_isolated_values_run_alone(self):
        declaration = toy_declaration()
        cases = frontend_cases.generate(declaration, None, seed=0)
        tags = {tag for case in cases for tag in case["features"]}
        for axis, values in frontend_cases.shared_axes(declaration).items():
            for value in values:
                if value not in frontend_cases.NEUTRAL:
                    self.assertIn(f"{axis}:{value}", tags)
        isolated = frontend_cases.isolated_values(declaration)
        for case in cases:
            self.assertLessEqual(len(isolated & set(case["features"])), 1, case["features"])

    def test_every_tag_names_what_its_request_carries(self):
        self.assertTrue(CASES_MODELS, "no model has a frontend cases module")
        for name in CASES_MODELS:
            model = corpus.Model(name)
            for case in model.full_cases(0):
                for tag in case["features"]:
                    self.assertIsNot(frontend_cases.exercised(tag, case["request"]), False, (name, case["id"], tag))
                if any(tag.startswith(("tool-field:", "defer:")) for tag in case["features"]):
                    self.assertTrue(case["request"].get("tools"), (name, case["id"]))

    def test_pairwise_rows_cover_every_pair(self):
        axes = {"a": [1, 2, 3], "b": ["x", "y"], "c": [True, False], "d": [0, 1, 2]}
        rows = frontend_cases.pairwise_rows(axes, __import__("random").Random(0))
        names = list(axes)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                pairs = {(row[first], row[second]) for row in rows}
                self.assertEqual(len(pairs), len(axes[first]) * len(axes[second]))


class CoverageTests(unittest.TestCase):
    def test_executed_code_lines_map_to_template_lines(self):
        coverage = TemplateCoverage(TOY, compile=toy_compile)
        text, lines = coverage.render(messages=[user()], add_generation_prompt=True)
        self.assertEqual(text, "<sys>max<user>hi<asst>")
        self.assertEqual(lines, {1, 2, 3, 4, 5, 18, 21})
        self.assertTrue({7, 9, 13, 15, LEGACY_LINE} <= coverage.lines_with_code)
        # Constant output has no code of its own.
        self.assertNotIn(22, coverage.lines_with_code)
        _, legacy = coverage.render(messages=[user()], mode="legacy")
        self.assertIn(LEGACY_LINE, legacy)

    def test_a_loop_epilogue_or_an_else_branch_marks_no_line_it_did_not_run(self):
        coverage = TemplateCoverage(TOY, compile=toy_compile)
        answer = {"role": "assistant", "content": "a"}
        _, lines = coverage.render(messages=[answer], add_generation_prompt=False)
        self.assertIn(7, lines)
        self.assertNotIn(9, lines, "no call ran")
        self.assertNotIn(15, lines, "the else branch did not run")
        with self.assertRaises(Exception):
            coverage.render(messages=[{"role": "critic", "content": "no"}])
        with coverage.tracing() as hit:
            with self.assertRaises(Exception):
                coverage.template.render(messages=[{"role": "critic", "content": "no"}])
        self.assertIn(15, hit)
        self.assertNotIn(13, hit, "the tool branch did not run")


MACRO = """\
{%- macro pick(x) -%}
{%- if x == 1 -%}
{{- 'one' + x|string -}}
{%- elif x == 2 -%}
{{- 'two' + x|string -}}
{%- endif -%}
{%- endmacro -%}
{{- pick(value) -}}"""


# A model's pinned Python encoder in miniature: lines 1 and 3 run when the
# module loads, the functions' bodies when a render calls them.
TOY_ENCODER = """\
import json

PREFIX = "<sys>"


def encode(messages, response_format=None):
    prompt = PREFIX
    for message in messages:
        if message["role"] == "user":
            prompt += "<user>" + message["content"]
        elif message["role"] == "assistant":
            prompt += "<asst>" + (message.get("content") or "")
        else:
            raise ValueError("unknown role " + message["role"])
    if response_format is not None:
        prompt += json.dumps(response_format)
    return prompt + "<asst>"


def unused():
    return json.dumps({})
"""


def encoder_reference():
    coverage = PythonSourceCoverage(TOY_ENCODER)

    def render(messages, tools, add_generation_prompt, continue_final_message, keywords, response_format=None):
        return coverage.namespace["encode"](messages, response_format)

    return corpus.Reference(coverage, render, lambda text: [ord(character) for character in text])


class PythonEncoderTests(unittest.TestCase):
    def test_only_function_bodies_have_lines(self):
        coverage = PythonSourceCoverage(TOY_ENCODER)
        self.assertEqual(coverage.lines_with_code, {7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 21})
        self.assertEqual(coverage.line_text(14).strip(), 'raise ValueError("unknown role " + message["role"])')

    def test_a_render_records_the_lines_it_runs_and_a_raise_is_a_refusal(self):
        reference = encoder_reference()
        arguments = {"tools": None, "add_generation_prompt": True, "continue_final_message": False, "keywords": {}}
        outcome = reference.outcome({**arguments, "messages": [user()]})
        self.assertEqual(outcome["rendered_text"], "<sys><user>hi<asst>")
        self.assertEqual(outcome["lines"], [7, 8, 9, 10, 15, 17])
        refused = reference.outcome({**arguments, "messages": [{"role": "critic", "content": "no"}]})
        self.assertEqual(refused["refusal"], {"class": "ValueError", "message": "unknown role critic"})
        self.assertIn(14, refused["lines"])

    def test_listed_request_fields_reach_the_reference(self):
        request = {"messages": [user()], "response_format": {"type": "json_object"}}
        model = ToyModel(Path(tempfile.mkdtemp()))
        self.assertNotIn("response_format", model.hf_input(request))
        model.cases.INPUT_FIELDS = ("response_format",)
        arguments = model.hf_input(request)
        self.assertEqual(arguments["response_format"], {"type": "json_object"})
        self.assertEqual(encoder_reference().outcome(arguments)["rendered_text"],
                         '<sys><user>hi{"type": "json_object"}<asst>')

    def test_a_source_pin_names_the_file_and_its_digest(self):
        directory = Path(tempfile.mkdtemp())
        (directory / "encoder.py").write_text(TOY_ENCODER)
        model = ToyModel(directory)
        digest = hashlib.sha256(TOY_ENCODER.encode()).hexdigest()
        model.cases.PIN = {**model.cases.PIN, "source": "encoder.py", "source_sha256": digest}
        with unittest.mock.patch.dict(os.environ, {"TOY_CHECKPOINT": str(directory)}):
            self.assertEqual(corpus.template_source(model), (TOY_ENCODER, digest))
            model.cases.PIN["source_sha256"] = "0" * 64
            with self.assertRaisesRegex(corpus.CorpusError, "encoder.py is .*, not the pinned 0{64}"):
                corpus.template_source(model)


class MacroCoverageTests(unittest.TestCase):
    def test_a_macro_return_credits_no_skipped_sibling_branch(self):
        coverage = TemplateCoverage(MACRO, compile=toy_compile)
        self.assertEqual(coverage.lines_with_code, {1, 2, 3, 4, 5, 8})
        self.assertEqual(coverage.render(value=1), ("one1", {1, 2, 3, 8}))
        self.assertEqual(coverage.render(value=2), ("two2", {1, 2, 4, 5, 8}))
        # The elif condition runs and fails; neither branch body runs.
        self.assertEqual(coverage.render(value=3), ("", {1, 2, 4, 8}))


class TransportTests(unittest.TestCase):
    def test_reasoning_content_reaches_a_turn_without_tool_calls_when_forwarded(self):
        turn = {"role": "assistant", "content": "a", "reasoning_content": "kept"}
        forwarded = corpus.hf_messages([turn], TRANSPORT)[0]
        self.assertEqual(forwarded["reasoning_content"], "kept")
        tool_only = dict(TRANSPORT, reasoning_content_fallback=False, tool_call_reasoning_fallback=True)
        self.assertNotIn("reasoning_content", corpus.hf_messages([turn], tool_only)[0])
        called = dict(turn, tool_calls=[call("f", {"n": 1})])
        translated = corpus.hf_messages([called], tool_only)[0]
        self.assertEqual(translated["reasoning_content"], "kept")
        self.assertEqual(translated["tool_calls"][0]["function"]["arguments"], {"n": 1})
        both = dict(turn, reasoning="wins")
        self.assertEqual(corpus.hf_messages([both], TRANSPORT)[0]["reasoning_content"], "wins")

    def test_tool_objects_are_whole_only_when_the_transport_carries_them(self):
        tool = {"type": "function", "function": {"name": "f", "description": None, "parameters": {},
                                                 "defer_loading": True}}
        self.assertEqual(corpus.hf_tools([tool], TRANSPORT), [tool])
        read = corpus.hf_tools([tool], dict(TRANSPORT, reads_tool_objects=False))
        self.assertEqual(read, [{"type": "function", "function": {"name": "f", "parameters": {}}}])

    def test_null_and_absent_fields_differ_only_in_whole_tool_objects(self):
        absent = {"type": "function", "function": {"name": "f"}}
        null = {"type": "function", "function": {"name": "f", "description": None, "parameters": None,
                                                 "strict": None}}
        fields = dict(TRANSPORT, reads_tool_objects=False)
        self.assertEqual(corpus.hf_tools([null], fields), corpus.hf_tools([absent], fields))
        self.assertEqual(corpus.hf_tools([null], fields), [absent])
        self.assertNotEqual(corpus.hf_tools([null], TRANSPORT), corpus.hf_tools([absent], TRANSPORT))
        self.assertEqual(corpus.hf_tools([null], TRANSPORT), [null])


    def test_a_top_level_effort_replaces_the_nested_one_as_the_transport_passes_it(self):
        reasoning = toy_declaration()["reasoning"]

        def keywords(**request):
            return corpus.hf_input({"messages": [user()], **request}, TRANSPORT, reasoning)["keywords"]

        self.assertEqual(keywords(reasoning_effort="low", chat_template_kwargs={"reasoning_effort": "high"}),
                         {"reasoning_effort": "low", "enable_thinking": True})
        self.assertEqual(keywords(chat_template_kwargs={"reasoning_effort": "high"}), {"reasoning_effort": "high"})
        # The disabling value turns thinking off unless the client sent
        # enable_thinking; a null one is not sent.
        self.assertEqual(keywords(reasoning_effort="none", chat_template_kwargs={"reasoning_effort": "low"}),
                         {"reasoning_effort": "none", "enable_thinking": False})
        self.assertEqual(keywords(reasoning_effort="none", chat_template_kwargs={"enable_thinking": True}),
                         {"reasoning_effort": "none", "enable_thinking": True})
        self.assertEqual(keywords(reasoning_effort="none", chat_template_kwargs={"enable_thinking": None}),
                         {"reasoning_effort": "none", "enable_thinking": False})

    def test_the_reference_renders_the_top_level_effort_of_both_controls(self):
        model = ToyModel(Path(tempfile.mkdtemp()))
        both = {"messages": [user()], "reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "high"}}
        self.assertEqual(toy_reference().outcome(model.hf_input(both))["rendered_text"], "<sys>low<user>hi<asst>")

    def test_the_effort_cases_set_both_controls_and_the_disabling_value(self):
        cases = {case["id"]: case for case in frontend_cases.generate(toy_declaration(), None, seed=0)}
        self.assertEqual(cases["s-effort-low-over-high"]["request"]["chat_template_kwargs"],
                         {"reasoning_effort": "high"})
        self.assertEqual(cases["s-effort-low-over-high"]["features"],
                         ["effort-via:both", "effort:low", "fixed:s-effort-low-over-high", "nested-effort:high"])
        self.assertIn("s-effort-high-over-low", cases)
        self.assertIn("s-effort-high-over-undeclared", cases)
        self.assertEqual(cases["s-effort-disable"]["request"]["reasoning_effort"], "none")
        self.assertIn("effort:disable", cases["s-effort-disable-over-low"]["features"])


class VectorShapeTests(unittest.TestCase):
    def test_render_vectors_carry_only_raw_reader_keys_and_inline_media(self):
        good = {"id": "x", "raw": True, "messages": [user()], "tools": [], "rendered_text": "", "token_ids": []}
        self.assertEqual(corpus.vector_problems(good), [])
        self.assertEqual(corpus.vector_problems(dict(good, category="c")),
                         ["carries category, which a render vector may not"])
        media = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file:///x.png"}}]}
        self.assertEqual(len(corpus.vector_problems(dict(good, messages=[media]))), 1)

    def test_text_naming_a_file_url_is_valid_and_file_backed_media_is_not(self):
        text = {"messages": [user("Explain file:// URLs."), {"role": "assistant", "content": "file:///etc is local."}]}
        self.assertEqual(corpus.vector_problems({"id": "x", "raw": True, **text, "rendered_text": "file://",
                                                 "token_ids": []}), [])
        self.assertEqual(corpus.capture_normalization(text), [])
        for part in ({"type": "image_url", "image_url": {"url": "file:///x.png"}},
                     {"type": "audio_url", "audio_url": {"url": "file:///x.wav"}},
                     {"type": "input_audio", "input_audio": {"data": "file:///x.wav", "format": "wav"}}):
            request = {"messages": [{"role": "user", "content": [part]}]}
            self.assertEqual(corpus.vector_problems({"id": "x", "raw": True, **request}), [
                "message 0 part 0 names a file:// source; a raw vector's media must be inline data: URLs"])
        image = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file:///x.png"}}]}]}
        self.assertEqual(corpus.capture_normalization(image),
                         ["message 0 part 0: a file:// media source becomes an inline one"])

    def test_round_trip_requests_pass_capture_normalization_unchanged(self):
        canonical = corpus.canonical_messages(
            [user(), {"role": "assistant", "content": "a", "reasoning_content": "r",
                      "tool_calls": [call("f", {"n": 1})]},
             {"role": "tool", "tool_call_id": "call_0", "content": "ok"}], TRANSPORT)
        self.assertEqual(corpus.capture_normalization({"messages": canonical}), [])
        self.assertEqual(canonical[1]["reasoning"], "r")
        raw = {"messages": [{"role": "assistant", "content": "", "reasoning_content": "r",
                             "tool_calls": [{"function": {"name": "f", "arguments": {}}}]},
                            {"role": "tool", "content": ""}]}
        self.assertEqual(len(corpus.capture_normalization(raw)), 4)

    def test_every_generated_vector_obeys_the_raw_reader(self):
        for name in CASES_MODELS:
            for case in corpus.Model(name).full_cases(0):
                self.assertEqual(corpus.vector_problems({"id": case["id"], "raw": True, **case["request"]}), [],
                                 (name, case["id"]))


class ModelReadBackTests(unittest.TestCase):
    def test_every_cases_module_reads_a_turn_back_against_its_prompt(self):
        turn = {"role": "assistant", "content": "Done."}
        for name in CASES_MODELS:
            expected_turn = corpus.Model(name).cases.expected_turn
            read = expected_turn(turn, {"prompt": "", "request": {"messages": [user(), turn]}})
            self.assertEqual(set(read), {"reasoning", "content", "tool_calls"}, name)


class ReferenceDirectoryTests(unittest.TestCase):
    """ci/cpu.sh exports each model's tokenizer directory from its checkpoint
    and unsets the checkpoint, so the reference reads the tokenizer
    directory first."""

    def checkpoint(self, environment):
        """The toy model's reference directory under ENVIRONMENT's toy variables."""
        kept = {key: value for key, value in os.environ.items() if not key.startswith("TOY_")}
        with unittest.mock.patch.dict(os.environ, {**kept, **environment}, clear=True):
            return ToyModel(Path(tempfile.mkdtemp())).checkpoint()

    def test_the_tokenizer_directory_serves_without_the_checkpoint(self):
        self.assertEqual(self.checkpoint({"TOY_TOKENIZER_DIR": "/pinned"}), Path("/pinned"))
        self.assertEqual(self.checkpoint({"TOY_TOKENIZER_DIR": "/pinned", "TOY_CHECKPOINT": "/weights"}),
                         Path("/pinned"))
        self.assertEqual(self.checkpoint({"TOY_CHECKPOINT": "/weights"}), Path("/weights"))
        with self.assertRaisesRegex(corpus.CorpusError, "TOY_TOKENIZER_DIR and TOY_CHECKPOINT are unset"):
            self.checkpoint({})

    def test_every_model_reads_the_variables_ci_exports(self):
        """Each cases module's checkpoint variable has the stem of its
        model.toml checkpoint_env, which ci/cpu.sh exports a tokenizer
        directory for."""
        for name in CASES_MODELS:
            stem = tomllib.loads((corpus.ROOT / "models" / name / "model.toml").read_text())["checkpoint_env"]
            with self.subTest(model=name):
                self.assertEqual(corpus.reference_variables(corpus.Model(name).pin),
                                 [f"CHOPINFER_{stem}_TOKENIZER_DIR", f"CHOPINFER_{stem}_CHECKPOINT"])


class MinimizeTests(Harness):
    def test_minimization_keeps_lines_and_every_feature_tag(self):
        model = ToyModel(self.directory, cases=CASES + [REDUNDANT])
        full = self.generate(model, self.table())
        output = self.directory / "minimal"
        corpus.minimize(model, full, output)
        _, _, coverage, _ = corpus.read_corpus(full)
        _, _, minimal, _ = corpus.read_corpus(output)
        lines = set().union(*(set(case["lines"]) for case in coverage["cases"].values()))
        kept = set().union(*(set(case["lines"]) for case in minimal["cases"].values()))
        self.assertEqual(kept, lines)
        self.assertLess(len(minimal["cases"]), len(coverage["cases"]))
        # arg:int and arg:float run the same template lines; both stay.
        self.assertEqual(coverage["cases"]["int"]["lines"], coverage["cases"]["float"]["lines"])
        tags = {tag for case in minimal["cases"].values() for tag in case["features"]}
        self.assertTrue({"arg:int", "arg:float"} <= tags)
        self.assertIn("critic", minimal["cases"], "the refusal class stays")


OVERLAP = [
    {"id": "only-a", "request": {"messages": [user("a")]}, "features": ["flag:a"]},
    {"id": "only-b", "request": {"messages": [user("b")]}, "features": ["flag:b"]},
    {"id": "both", "request": {"messages": [user("ab")]}, "features": ["flag:a", "flag:b", "unique:tag"]},
]
ENTRY_A = {"id": "a", "kind": "refused", "when": ["flag:a"], "outcome": {}, "reason": "-"}
ENTRY_B = {"id": "b", "kind": "refused", "when": ["flag:b"], "outcome": {}, "reason": "-"}


class AmbiguityTests(Harness):
    def test_a_case_matching_several_entries_is_never_minimized_away(self):
        model = ToyModel(self.directory, cases=OVERLAP)
        table = self.table([{"id": "rest", "lines": sorted(toy_reference().coverage.lines_with_code),
                             "blocked_by": "-", "reason": "-"}], [ENTRY_A, ENTRY_B])
        full = self.generate(model, table)
        with self.assertRaisesRegex(corpus.CorpusError, r"both \(a, b\)"):
            corpus.minimize(model, full, self.directory / "minimal")
        # A hand-made subset without the overlap still fails: its tag and its
        # ambiguity are judged over the full corpus.
        committed = self.committed(model, table, {"only-a", "only-b"})
        errors = self.check(model, committed, table)
        self.assertTrue(any("full corpus case both matches several inputs entries" in error for error in errors),
                        errors)
        self.assertTrue(any("lack feature tags of the full corpus: unique:tag" in error for error in errors), errors)


class GateTests(Harness):
    def test_the_full_corpus_passes_with_its_divergences(self):
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY)
        table = self.table([LEGACY], [UNDECLARED])
        committed = self.committed(model, table, {case["id"] for case in WITHOUT_LEGACY})
        self.assertEqual(self.check(model, committed, table), [])

    def test_an_unlisted_uncovered_line_fails(self):
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY)
        table = self.table([], [UNDECLARED])
        committed = self.committed(model, table, {case["id"] for case in WITHOUT_LEGACY})
        errors = self.check(model, committed, table)
        self.assertTrue(any(f"template line {LEGACY_LINE} is run by no committed case" in error
                            for error in errors), errors)

    def test_a_lines_entry_a_committed_case_runs_is_stale(self):
        model = ToyModel(self.directory)
        table = self.table([LEGACY], [UNDECLARED])
        committed = self.committed(model, table, {case["id"] for case in CASES})
        errors = self.check(model, committed, table)
        self.assertTrue(any("lines entry legacy-mode is stale" in error for error in errors), errors)

    def test_inputs_entries_are_judged_by_outcome(self):
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY)
        table = self.table([LEGACY], [UNDECLARED])
        committed = self.committed(model, table, {case["id"] for case in WITHOUT_LEGACY})
        _, _, coverage, _ = corpus.read_corpus(committed)
        # An accepted and a refused effort run the same template lines.
        self.assertEqual(coverage["cases"]["low"]["lines"], coverage["cases"]["undeclared"]["lines"])
        self.assertEqual(self.check(model, committed, table), [])
        # The frontend now renders the undeclared effort as the reference does.
        vectors = json.loads((committed / "corpus.json").read_text())
        full = json.loads((self.directory / "full" / "reference.json").read_text())
        for vector in vectors:
            if vector["id"] == "undeclared":
                del vector["refusal"]
                vector.update(full["undeclared"])
        corpus.write_json(committed / "corpus.json", vectors)
        errors = self.check(model, committed, table)
        self.assertTrue(any("inputs entry undeclared-effort is stale" in error for error in errors), errors)

    def test_an_inputs_entry_without_a_committed_case_fails(self):
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY)
        never = {"id": "never", "kind": "refused", "when": ["effort:never"], "outcome": {}, "reason": "-"}
        table = self.table([LEGACY], [UNDECLARED, never])
        committed = self.committed(model, table, {case["id"] for case in WITHOUT_LEGACY})
        errors = self.check(model, committed, table)
        self.assertTrue(any("inputs entry never matches no committed case" in error for error in errors), errors)

    def test_a_tag_its_request_does_not_carry_fails(self):
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY)
        table = self.table([LEGACY], [UNDECLARED])
        committed = self.committed(model, table, {case["id"] for case in WITHOUT_LEGACY})
        coverage = json.loads((committed / "corpus-coverage.json").read_text())
        coverage["cases"]["low"]["features"].append("tool-field:no-description")
        corpus.write_json(committed / "corpus-coverage.json", coverage)
        errors = self.check(model, committed, table)
        self.assertIn("case low: is tagged tool-field:no-description, which its request does not carry", errors)

    def test_a_case_that_differs_without_an_entry_fails(self):
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY)
        table = self.table([LEGACY], [UNDECLARED])
        committed = self.committed(model, table, {case["id"] for case in WITHOUT_LEGACY})
        vectors = json.loads((committed / "corpus.json").read_text())
        vectors[0]["rendered_text"] += "x"
        corpus.write_json(committed / "corpus.json", vectors)
        errors = self.check(model, committed, table)
        self.assertTrue(any("case low: differs from the reference" in error for error in errors), errors)


class EnvironmentTests(Harness):
    def test_the_corpus_records_its_reference_environment_and_check_warns_on_another(self):
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY)
        table = self.table([LEGACY], [UNDECLARED])
        full = self.generate(model, table)
        recorded = json.loads((full / "corpus-coverage.json").read_text())["environment"]
        self.assertEqual(set(recorded), {"python", "jinja2", "unicode"})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(self.check(model, full, table), [])
        self.assertEqual([warning for warning in caught
                          if issubclass(warning.category, corpus.ReferenceEnvironmentWarning)], [])
        other = toy_reference(self.source)
        other.environment = dict(other.environment, unicode="0.0.0")
        with self.assertWarnsRegex(corpus.ReferenceEnvironmentWarning,
                                   f"generated with unicode {recorded['unicode']}, this check runs 0.0.0"):
            errors = corpus.check(model, full, table, reference=other, template=self.template)
        self.assertEqual(errors, [])


class ReferenceTextTests(Harness):
    def test_a_cases_module_rewrites_the_reference_as_its_processor_does(self):
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY)
        model.cases.reference_text = lambda text, arguments: text.replace("<tool>", "<tool><pad>")
        table = self.table([LEGACY], [UNDECLARED])
        committed = self.committed(model, table, {case["id"] for case in WITHOUT_LEGACY})
        vectors = {vector["id"]: vector for vector in json.loads((committed / "corpus.json").read_text())}
        self.assertIn("<tool><pad>ok", vectors["int"]["rendered_text"])
        self.assertEqual(vectors["int"]["token_ids"], [ord(c) for c in vectors["int"]["rendered_text"]])
        self.assertNotIn("<pad>", vectors["low"]["rendered_text"])
        self.assertEqual(self.check(model, committed, table), [])
        # The template's own text is not what the model reads.
        del model.cases.reference_text
        errors = self.check(model, committed, table)
        self.assertTrue(any("case int: differs from the reference" in error for error in errors), errors)


class RoundTripTests(Harness):
    def rounds(self, model):
        full = self.generate(model, self.table())
        return {vector["id"]: vector for vector in json.loads((full / "corpus-rounds.json").read_text())}

    def test_an_in_render_stop_token_cuts_the_completion_and_records_the_separator(self):
        rounds = self.rounds(ToyModel(self.directory))
        answer = rounds["answer-round-end"]
        self.assertEqual("".join(map(chr, answer["token_ids"])), "r|Done.<end>")
        self.assertEqual(answer["separator"], "\n")
        self.assertEqual(answer["expect"], {"reasoning": "r", "content": "Done.", "finish_reason": "stop"})
        self.assertEqual(answer["request"]["messages"], [user()])
        called = rounds["called-round-end"]
        self.assertEqual(called["expect"]["tool_calls"], [{"name": "f", "arguments": "{\"n\": 1}"}])
        self.assertEqual(called["expect"]["finish_reason"], "tool_calls")

    def test_declared_endings_are_appended_to_a_turn_without_a_stop_token(self):
        model = ToyModel(self.directory, endings={"answer": ["<user>", "<eos>"], "tool_calls": ["<tool>"]},
                         stops=("<user>", "<tool>", "<eos>"))
        rounds = self.rounds(model)
        self.assertEqual("".join(map(chr, rounds["answer-round-user"]["token_ids"])), "r|Done.<end>\n<user>")
        self.assertEqual("".join(map(chr, rounds["answer-round-eos"]["token_ids"])), "r|Done.<end>\n<eos>")
        self.assertNotIn("separator", rounds["answer-round-user"])
        self.assertIn("called-round-tool", rounds)

    def test_every_shape_and_ending_needs_a_committed_vector(self):
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY)
        table = self.table([LEGACY], [UNDECLARED])
        committed = self.committed(model, table, {case["id"] for case in WITHOUT_LEGACY})
        self.assertEqual(self.check(model, committed, table), [])
        coverage = json.loads((committed / "corpus-coverage.json").read_text())
        recorded = dict(coverage["rounds"])
        coverage["rounds"]["answer-round-end"] = {"shape": "answer/calls=0/reasoning=0/content=0/thinking",
                                                  "ending": "<end>"}
        corpus.write_json(committed / "corpus-coverage.json", coverage)
        errors = self.check(model, committed, table)
        self.assertTrue(any("round trip answer-round-end: corpus-coverage.json records" in error for error in errors),
                        errors)
        corpus.write_json(committed / "corpus-rounds.json", [])
        corpus.write_json(committed / "corpus-coverage.json", dict(coverage, rounds={}))
        errors = self.check(model, committed, table)
        for info in recorded.values():
            self.assertIn(f"no committed round trip has shape {info['shape']} and ending {info['ending']!r}, "
                          "which the full corpus yields", errors)

    def test_a_disabled_thinking_turn_is_a_shape_of_its_own(self):
        chat = {"messages": [user(), {"role": "assistant", "content": "Done."}],
                "chat_template_kwargs": {"enable_thinking": False}}
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY + [
            {"id": "chat", "request": chat, "features": ["ends:assistant", "thinking:enable_thinking=false"]}])
        full = self.generate(model, self.table())
        shapes = json.loads((full / "corpus-coverage.json").read_text())["rounds"]
        self.assertEqual(shapes["chat-round-end"]["shape"], "answer/calls=0/reasoning=0/content=1/chat")
        self.assertEqual(shapes["answer-round-end"]["shape"], "answer/calls=0/reasoning=1/content=1/thinking")
        corpus.minimize(model, full, self.directory / "minimal")
        kept = json.loads((self.directory / "minimal" / "corpus-coverage.json").read_text())["rounds"]
        self.assertIn("chat-round-end", kept)

    def test_the_first_supplied_thinking_keyword_decides_the_mode(self):
        declaration = dict(toy_declaration())
        declaration["reasoning"] = dict(declaration["reasoning"], thinking_keywords=["enable_thinking", "thinking"],
                                        default_enabled=True)
        reasoning = declaration["reasoning"]

        def mode(**request):
            keywords = corpus.hf_input({"messages": [user()], **request}, TRANSPORT, reasoning)["keywords"]
            return corpus.thinking_mode(keywords, declaration)

        # A top-level effort passes enable_thinking, the first keyword, so a
        # later thinking=false changes nothing.
        self.assertEqual(mode(reasoning_effort="low", chat_template_kwargs={"thinking": False}), "thinking")
        self.assertEqual(mode(chat_template_kwargs={"thinking": False}), "chat")
        self.assertEqual(mode(chat_template_kwargs={"enable_thinking": True, "thinking": False}), "thinking")
        self.assertEqual(mode(chat_template_kwargs={"enable_thinking": False, "thinking": True}), "chat")
        self.assertEqual(mode(reasoning_effort="none", chat_template_kwargs={"thinking": True}), "chat")
        self.assertEqual(mode(), "thinking")
        declaration["reasoning"] = dict(reasoning, default_enabled=False)
        self.assertEqual(corpus.thinking_mode({}, declaration), "chat")
        # The round trip of such a request has the thinking shape.
        both = {"id": "both", "request": {"messages": [user(), {"role": "assistant", "content": "Done."}],
                                          "reasoning_effort": "low", "chat_template_kwargs": {"thinking": False}},
                "features": ["ends:assistant"]}
        model = ToyModel(self.directory, cases=[both])
        model.declaration = dict(model.declaration, reasoning=dict(reasoning))
        shapes = json.loads((self.generate(model, self.table()) / "corpus-coverage.json").read_text())["rounds"]
        self.assertEqual(shapes["both-round-end"]["shape"], "answer/calls=0/reasoning=0/content=1/thinking")

    def test_an_ending_that_is_not_a_stop_token_is_refused(self):
        model = ToyModel(self.directory, endings={"answer": ["<bogus>"], "tool_calls": ["<tool>"]},
                         stops=("<user>", "<tool>"))
        with self.assertRaisesRegex(corpus.CorpusError, "not a declared stop token"):
            self.rounds(model)

    def test_an_ending_that_a_continuation_does_not_begin_with_is_refused(self):
        model = ToyModel(self.directory, endings={"answer": ["<eos>"], "tool_calls": ["<tool>"]},
                         stops=("<user>", "<tool>", "<eos>"))
        with self.assertRaisesRegex(corpus.CorpusError, "begins with none of its endings"):
            self.rounds(model)

    def test_text_naming_a_file_url_round_trips(self):
        case = {"id": "text-url", "request": {"messages": [user("Explain file:// URLs."),
                                                          {"role": "assistant", "content": "Local files."}]},
                "features": ["ends:assistant"]}
        rounds = self.rounds(ToyModel(self.directory, cases=[case]))
        self.assertEqual("".join(map(chr, rounds["text-url-round-end"]["token_ids"])), "|Local files.<end>")

    def test_an_undeclared_round_trip_failure_cannot_be_minimized_away(self):
        # A case with the lines and features of "answer" whose generation
        # prompt is not a prefix of its history render.
        marked = "{{- ('X' if messages[-1].get('content') == 'bad' else '') + '<sys>' + effort -}}"
        self.use_template(TOY.replace("{{- '<sys>' + effort -}}", marked))
        bad = copy.deepcopy(next(case for case in WITHOUT_LEGACY if case["id"] == "answer"))
        bad["id"] = "bad-prefix"
        bad["request"]["messages"][-1]["content"] = "bad"
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY + [bad])
        table = self.table([LEGACY], [UNDECLARED])
        full = self.generate(model, table)
        undeclared = "the generation prompt is not a prefix of the turn's history render, and it matches no " \
                     "no-round-trip entry"
        self.assertIn(f"case bad-prefix: {undeclared}", self.check(model, full, table))
        with self.assertRaisesRegex(corpus.CorpusError, "round trips fail without a no-round-trip entry: bad-prefix"):
            corpus.minimize(model, full, self.directory / "minimal")
        # A subset that leaves the case out still fails: the full corpus is
        # regenerated and its round trips are judged too.
        committed = self.committed(model, table, {case["id"] for case in WITHOUT_LEGACY})
        self.assertIn(f"full corpus case bad-prefix: {undeclared}", self.check(model, committed, table))
        # Declared, the failure is kept by minimization and the gate passes.
        entry = {"id": "bad", "kind": "no-round-trip", "when": ["ends:assistant"], "reason": "-"}
        table = self.table([LEGACY], [UNDECLARED, entry])
        full = self.generate(model, table)
        corpus.minimize(model, full, self.directory / "minimal")
        minimal = json.loads((self.directory / "minimal" / "corpus-coverage.json").read_text())
        self.assertIn("bad-prefix", minimal["cases"])
        self.assertEqual(self.check(model, self.directory / "minimal", table), [])

    def test_a_failed_prefix_needs_a_no_round_trip_entry(self):
        prefixless = TOY.replace("{{- '<asst>' -}}\n{%- endif -%}", "{{- '<gen>' -}}\n{%- endif -%}")
        self.template.write_text(prefixless)
        model = ToyModel(self.directory, cases=[case for case in CASES if case["id"] != "legacy"])
        table = self.table([LEGACY], [UNDECLARED])
        output = self.directory / "full"
        corpus.generate(model, output, divergences=table, template=self.template,
                        reference=toy_reference(prefixless))

        def check(table):
            return corpus.check(model, output, table, reference=toy_reference(prefixless), template=self.template)

        self.assertTrue(any("matches no no-round-trip entry" in error for error in check(table)))
        entry = {"id": "gen-prompt", "kind": "no-round-trip", "when": ["ends:assistant"], "reason": "-"}
        table = self.table([LEGACY], [UNDECLARED, entry])
        self.assertTrue(any("records divergence None, not 'gen-prompt'" in error for error in check(table)))
        corpus.generate(model, output, divergences=table, template=self.template,
                        reference=toy_reference(prefixless))
        self.assertEqual(check(table), [])
        coverage = json.loads((output / "corpus-coverage.json").read_text())
        self.assertEqual(coverage["cases"]["answer"]["divergence"], "gen-prompt")
        minimal = self.directory / "minimal"
        corpus.minimize(model, output, minimal)
        self.assertIn("answer", json.loads((minimal / "corpus-coverage.json").read_text())["cases"])


def calling(identifier, parameters, arguments, features=()):
    """A case ending in one call to `f`, whose tool has `parameters`."""
    tool = {"type": "function", "function": {"name": "f", "parameters": parameters}}
    return {"id": identifier, "request": {"messages": [user(), {"role": "assistant", "content": "", "tool_calls": [
        call("f", arguments)]}], "tools": [tool]}, "features": ["ends:assistant", *features]}


UNION = calling("union", {"type": "object", "properties": {"n": {"type": ["string", "integer"]}}}, {"n": "42"},
                ["schema:union"])
UNION_ENTRY = {"id": "union", "kind": "no-round-trip", "when": ["round-trip:union"], "reason": "-"}


class RoundTripDomainTests(Harness):
    """Accepted arguments outside the parser's lossless domain fail their
    round trip, and only an exercised no-round-trip entry of their class
    explains the failure."""

    def report(self, model, table):
        full = self.generate(model, table)
        return full, json.loads((full / "rounds-report.json").read_text())

    def test_supported_schema_forms_round_trip_with_their_types(self):
        cases = [calling("additional", {"type": "object", "additionalProperties": {"type": "integer"}}, {"n": 42}),
                 calling("additional-object", {"type": "object", "additionalProperties": {"type": "object"}},
                         {"n": {"a": [1, None]}}),
                 calling("untyped-string", {"type": "object", "properties": {"n": {}}}, {"n": "text"}),
                 calling("pattern-string", {"type": "object", "patternProperties": {"^n$": {"type": "integer"}}},
                         {"n": "text"}),
                 calling("enum", {"type": "object", "properties": {"n": {"enum": ["a", "b"]}}}, {"n": "a"}),
                 calling("number-integer", {"type": "object", "properties": {"n": {"type": "number"}}}, {"n": 42}),
                 calling("additional-number-integer", {"type": "object", "additionalProperties": {"type": "number"}},
                         {"n": 42}),
                 calling("alias", {"type": "object", "properties": {"n": {"type": "str"}}}, {"n": "text"}),
                 calling("unknown-type", {"type": "object", "properties": {"n": {"type": "unknown"}}}, {"n": "text"})]
        full, report = self.report(ToyModel(self.directory, cases=cases), self.table())
        self.assertEqual(report["cases"], {})
        rounds = {vector["id"]: vector for vector in json.loads((full / "corpus-rounds.json").read_text())}
        self.assertEqual(rounds["additional-round-end"]["expect"]["tool_calls"],
                         [{"name": "f", "arguments": "{\"n\": 42}"}])
        self.assertEqual(len(rounds), len(cases))

    def test_each_lossy_argument_fails_with_its_class(self):
        cases = [UNION,
                 calling("any-of", {"type": "object", "properties": {"n": {"anyOf": [{"type": "integer"},
                                                                                    {"type": "null"}]}}}, {"n": 3}),
                 calling("untyped", {"type": "object", "properties": {"n": {}}}, {"n": 3}),
                 calling("additional-any", {"type": "object", "additionalProperties": True}, {"n": 3}),
                 calling("pattern", {"type": "object", "patternProperties": {"^n$": {"type": "integer"}}},
                         {"n": 3}),
                 calling("no-parameters", None, {"n": 3}),
                 calling("mismatch", {"type": "object", "properties": {"n": {"type": "integer"}}}, {"n": "3"}),
                 calling("float-under-integer", {"type": "object", "properties": {"n": {"type": "integer"}}},
                         {"n": 1.5})]
        _, report = self.report(ToyModel(self.directory, cases=cases), self.table())
        kinds = {key: value["tags"] for key, value in report["cases"].items()}
        self.assertEqual(kinds, {
            "union": ["mode:thinking", "round-trip:union"], "any-of": ["mode:thinking", "round-trip:union"],
            "untyped": ["mode:thinking", "round-trip:untyped"],
            "additional-any": ["mode:thinking", "round-trip:untyped"],
            "pattern": ["mode:thinking", "round-trip:untyped"],
            "no-parameters": ["mode:thinking", "round-trip:untyped"],
            "mismatch": ["mode:thinking", "round-trip:mismatch"],
            "float-under-integer": ["mode:thinking", "round-trip:mismatch"]})

    def test_an_unexplained_accepted_schema_skip_fails_the_gate(self):
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY + [UNION])
        table = self.table([LEGACY], [UNDECLARED])
        full = self.generate(model, table)
        undeclared = ("case union: the parser cannot read the turn's arguments back: n has the types string, "
                      "integer, and it matches no no-round-trip entry")
        self.assertIn(undeclared, self.check(model, full, table))
        with self.assertRaisesRegex(corpus.CorpusError, "round trips fail without a no-round-trip entry: union"):
            corpus.minimize(model, full, self.directory / "minimal")
        # Left out of the committed corpus, it still fails from the full one.
        committed = self.committed(model, table, {case["id"] for case in WITHOUT_LEGACY})
        self.assertIn(f"full corpus {undeclared}", self.check(model, committed, table))
        # An entry of another class does not explain it.
        other = dict(UNION_ENTRY, id="delimiters", when=["round-trip:delimiter"])
        table = self.table([LEGACY], [UNDECLARED, other])
        self.assertIn(undeclared, self.check(model, self.generate(model, table), table))
        # Its own class does, and minimization keeps the case.
        table = self.table([LEGACY], [UNDECLARED, UNION_ENTRY])
        full = self.generate(model, table)
        self.assertEqual(self.check(model, full, table), [])
        corpus.minimize(model, full, self.directory / "minimal")
        minimal = json.loads((self.directory / "minimal" / "corpus-coverage.json").read_text())
        self.assertEqual(minimal["cases"]["union"]["divergence"], "union")
        self.assertEqual(self.check(model, self.directory / "minimal", table), [])
        # Without a committed case that fails that way, the entry is stale.
        committed = self.committed(model, table, {case["id"] for case in WITHOUT_LEGACY})
        self.assertIn("inputs entry union is stale: no committed case fails its round trip that way",
                      self.check(model, committed, table))

    def test_a_fixed_case_keeps_its_round_trip_through_minimization(self):
        twin = calling("twin", {"type": "object", "properties": {"n": {"type": "integer"}}}, {"n": 2},
                       ["fixed:twin"])
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY + [twin])
        table = self.table([LEGACY], [UNDECLARED])
        full = self.generate(model, table)
        coverage = json.loads((full / "corpus-coverage.json").read_text())
        # "twin" has the shape of "called", which comes first.
        self.assertEqual(coverage["rounds"]["twin-round-end"], coverage["rounds"]["called-round-end"])
        minimal = self.directory / "minimal"
        corpus.minimize(model, full, minimal)
        kept = {vector["id"] for vector in json.loads((minimal / "corpus-rounds.json").read_text())}
        self.assertLessEqual({"twin-round-end", "called-round-end"}, kept)
        self.assertEqual(self.check(model, minimal, table), [])
        rounds = [vector for vector in json.loads((minimal / "corpus-rounds.json").read_text())
                  if vector["id"] != "twin-round-end"]
        corpus.write_json(minimal / "corpus-rounds.json", rounds)
        minimized = json.loads((minimal / "corpus-coverage.json").read_text())
        del minimized["rounds"]["twin-round-end"]
        corpus.write_json(minimal / "corpus-coverage.json", minimized)
        self.assertEqual(self.check(model, minimal, table),
                         ["round trip twin-round-end: fixed case twin yields it, but it is not committed"])

    def test_a_raw_value_that_spells_a_marker_fails_as_a_delimiter(self):
        """Delimiter loss follows the parser's scan: a value it reads as raw
        text (a string type or alias, any value of a tool whose strict is
        false) ends at a marker it spells, while a marker inside a JSON
        value's string literal is content, so that witness round-trips."""
        parser = {"tool_grammar": "xml-key-value-arguments-v1", "parameter_close": "</v>",
                  "tool_call_close": "</call>"}
        cases = [calling("close", {"type": "object", "properties": {"n": {"type": "string"}}}, {"n": "a</v>"}),
                 calling("alias", {"type": "object", "properties": {"n": {"type": "str"}}}, {"n": "\"</v>\""}),
                 calling("nested", {"type": "object", "properties": {"n": {"type": "object"}}}, {"n": {"k": "</v>"}}),
                 calling("plain", {"type": "object", "properties": {"n": {"type": "string"}}}, {"n": "a<v>"})]
        loose = calling("loose", {"type": "object", "properties": {"n": {"type": "object"}}}, {"n": {"k": "</v>"}})
        loose["request"]["tools"][0]["function"]["strict"] = False
        model = ToyModel(self.directory, cases=cases + [loose])
        model.parser = parser
        full, report = self.report(model, self.table())
        self.assertEqual({key: value["tags"] for key, value in report["cases"].items()},
                         {key: ["mode:thinking", "round-trip:delimiter"] for key in ("close", "alias", "loose")})
        # The JSON witness is no delimiter loss: it yields its round trip.
        rounds = {vector["id"] for vector in json.loads((full / "corpus-rounds.json").read_text())}
        self.assertEqual(rounds, {"nested-round-end", "plain-round-end"})

    def tokenizer_reference(self):
        """The toy reference with a tokenizer that composes text (NFC)
        before it maps characters to ids, as a normalizing tokenizer does."""
        reference = toy_reference(self.source)
        reference.normalize = lambda text: unicodedata.normalize("NFC", text)
        reference.tokenize = lambda text: [ord(character) for character in unicodedata.normalize("NFC", text)]
        return reference

    def test_tokenizer_normalization_keeps_a_normalized_round_trip(self):
        """A completion the tokenizer normalizes is a declared byte loss, but
        it still yields a round trip: a control whose ids are the
        completion's and whose expectation is the normalized turn, which is
        the composed twin's own round trip. So a parser loss beyond the
        normalization cannot hide behind the tokenizer class."""
        string = {"type": "object", "properties": {"n": {"type": "str"}}}
        decomposed = calling("decomposed", string, {"n": "\"e\u0301"}, ["fixed:decomposed"])
        composed = calling("composed", string, {"n": "\"\u00e9"})
        model = ToyModel(self.directory, cases=[decomposed, composed])
        output = self.directory / "plain"
        corpus.generate(model, output, divergences=self.table(), template=self.template,
                        reference=toy_reference(self.source))
        self.assertEqual(json.loads((output / "rounds-report.json").read_text())["cases"], {})
        output = self.directory / "nfc"
        corpus.generate(model, output, divergences=self.table(), template=self.template,
                        reference=self.tokenizer_reference())
        report = json.loads((output / "rounds-report.json").read_text())["cases"]
        self.assertEqual(report["decomposed"]["tags"], ["mode:thinking", "round-trip:tokenizer"])
        rounds = {vector["id"]: vector for vector in json.loads((output / "corpus-rounds.json").read_text())}
        control, twin = rounds["decomposed-round-end"], rounds["composed-round-end"]
        self.assertEqual(control["expect"]["tool_calls"], [{"name": "f", "arguments": "{\"n\": \"\\\"\u00e9\"}"}])
        self.assertEqual({key: value for key, value in control.items() if key != "id"},
                         {key: value for key, value in twin.items() if key != "id"})
        # Declared, the gate passes only with the control committed.
        entry = {"id": "nfc", "kind": "no-round-trip", "when": ["round-trip:tokenizer"], "reason": "-"}
        table = self.table([], [entry])
        corpus.generate(model, output, divergences=table, template=self.template,
                        reference=self.tokenizer_reference())

        def check():
            return [error for error in corpus.check(model, output, table, reference=self.tokenizer_reference(),
                                                    template=self.template)
                    if "template line" not in error]

        self.assertEqual(check(), [])
        corpus.write_json(output / "corpus-rounds.json", [twin])
        coverage = json.loads((output / "corpus-coverage.json").read_text())
        del coverage["rounds"]["decomposed-round-end"]
        corpus.write_json(output / "corpus-coverage.json", coverage)
        self.assertIn("round trip decomposed-round-end: fixed case decomposed yields it, but it is not committed",
                      check())

    def test_a_normalized_turn_must_render_as_the_normalized_completion(self):
        """The control holds only when normalization is the whole change: a
        template that writes the normalized turn otherwise is refused."""
        self.use_template(TOY.replace("(m.content or '')", "(m.content or '') + ('!' if 'é' in "
                                                          "(m.tool_calls or [{}])[0].get('function', {})"
                                                          ".get('arguments', {}).get('n', '') else '')"))
        model = ToyModel(self.directory, cases=[
            calling("decomposed", {"type": "object", "properties": {"n": {"type": "string"}}}, {"n": "e\u0301"})])
        with self.assertRaisesRegex(corpus.CorpusError, "does not render as the normalized completion"):
            corpus.generate(model, self.directory / "nfc", divergences=self.table(), template=self.template,
                            reference=self.tokenizer_reference())

if __name__ == "__main__":
    unittest.main()

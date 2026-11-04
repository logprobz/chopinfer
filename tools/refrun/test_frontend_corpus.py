"""The frontend corpus harness on a toy template under plain jinja2.

No transformers, torch or numpy: the toy template compiles with a jinja2
environment configured as transformers configures its own, and tokenizes one
id per character.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
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


def toy_cases_module(endings=None, recipe=None):
    def expected_turn(message, context):
        assert context["prompt"].endswith("<asst>") and "messages" in context["request"], context
        return {"reasoning": message.get("reasoning_content") or "", "content": message.get("content") or "",
                "tool_calls": [{"name": call["function"]["name"],
                                "arguments": json.dumps(call["function"]["arguments"], ensure_ascii=False)}
                               for call in message.get("tool_calls") or []]}

    module = SimpleNamespace(PIN={"revision": "toy", "template_sha256": "", "checkpoint_env": "TOY_CHECKPOINT"},
                             TRANSPORT=TRANSPORT, ENDINGS=endings, expected_turn=expected_turn)
    if recipe is not None:
        module.RECIPE = recipe
    return module


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
    def __init__(self, root, cases=CASES, endings=None, stops=("<end>", "<user>"), recipe=None):
        super().__init__("toy", root, toy_cases_module(endings, recipe), toy_declaration(stops))
        self.cases_list = cases

    def full_cases(self, seed):
        return copy.deepcopy(self.cases_list)


class Harness(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
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

    def generate(self, model, table, name="full", oracle=None):
        output = self.directory / name
        corpus.generate(model, output, divergences=table, template=self.template,
                        reference=toy_reference(self.source), oracle=oracle)
        return output

    def check(self, model, fixtures, table):
        return corpus.check(model, fixtures, table, reference=toy_reference(self.source), template=self.template)

    def committed(self, model, table, keep, oracle=None):
        """A committed corpus of the cases named in `keep`."""
        full = self.generate(model, table, oracle=oracle)
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
        record = corpus.read_recipe(full)
        if record is not None:
            rounds = {vector["id"] for vector in kept_rounds}
            corpus.write_json(output / corpus.RECIPE_FILE, {
                **record, "cases": {key: row for key, row in record["cases"].items() if key in keep},
                "rounds": {key: row for key, row in record["rounds"].items() if key in rounds}})
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
        model = ToyModel(Path(self.enterContext(tempfile.TemporaryDirectory())))
        self.assertNotIn("response_format", model.hf_input(request))
        model.cases.INPUT_FIELDS = ("response_format",)
        arguments = model.hf_input(request)
        self.assertEqual(arguments["response_format"], {"type": "json_object"})
        self.assertEqual(encoder_reference().outcome(arguments)["rendered_text"],
                         '<sys><user>hi{"type": "json_object"}<asst>')

    def test_a_source_pin_names_the_file_and_its_digest(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
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

    def test_a_reasoning_content_field_is_read_alone(self):
        turn = {"role": "assistant", "content": "a", "reasoning": "ignored", "reasoning_content": "kept"}
        self.assertEqual(corpus.hf_messages([turn], TRANSPORT, "reasoning_content")[0]["reasoning_content"], "kept")
        alone = {"role": "assistant", "content": "a", "reasoning": "ignored"}
        self.assertNotIn("reasoning_content", corpus.hf_messages([alone], TRANSPORT, "reasoning_content")[0])

    def test_a_final_assistant_prefix_continues_the_turn_as_the_transport_reads_it(self):
        reasoning = toy_declaration()["reasoning"]
        turn = {"role": "assistant", "content": "a", "prefix": True}
        continued = corpus.hf_input({"messages": [user(), turn]}, TRANSPORT, reasoning)
        self.assertEqual((continued["continue_final_message"], continued["add_generation_prompt"]), (True, False))
        self.assertNotIn("prefix", continued["messages"][-1])
        unset = corpus.hf_input({"messages": [user(), dict(turn, prefix=False)]}, TRANSPORT, reasoning)
        self.assertEqual((unset["continue_final_message"], unset["add_generation_prompt"]), (False, True))
        self.assertEqual(corpus.round_ineligible({"messages": [user(), turn]}, TRANSPORT),
                         "continues the final message")

    def test_arguments_stay_text_for_a_reference_that_decodes_them(self):
        turn = {"role": "assistant", "content": "", "tool_calls": [call("f", {"n": 1})]}
        turn["tool_calls"][0]["function"]["arguments"] = "[1,  2]"
        kept = corpus.hf_messages([turn], TRANSPORT, decode_arguments=False)[0]
        self.assertEqual(kept["tool_calls"][0]["function"]["arguments"], "[1,  2]")
        self.assertEqual(corpus.hf_messages([turn], TRANSPORT)[0]["tool_calls"][0]["function"]["arguments"], [1, 2])

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

    def test_a_thinking_object_switch_reads_the_object_and_drops_the_keywords(self):
        reasoning = dict(toy_declaration()["reasoning"], switch="thinking-object",
                         thinking_keywords=["enable_thinking", "thinking"])

        def keywords(**request):
            return corpus.hf_input({"messages": [user()], **request}, TRANSPORT, reasoning)["keywords"]

        kwargs = {"enable_thinking": True, "thinking": True, "reasoning_effort": "high", "preserve_thinking": True}
        self.assertEqual(keywords(thinking={"type": "disabled"}, chat_template_kwargs=kwargs),
                         {"enable_thinking": False, "thinking": False, "preserve_thinking": True})
        # An explicit object overrides the effort that disables thinking.
        self.assertEqual(keywords(thinking={"type": "adaptive"}, reasoning_effort="none"),
                         {"enable_thinking": True, "thinking": True})
        self.assertEqual(keywords(thinking={"type": "enabled", "budget_tokens": 4}, reasoning_effort="low"),
                         {"enable_thinking": True, "thinking": True, "reasoning_effort": "low"})
        # Without one, the top-level effort switches, and the keywords are
        # ignored.
        self.assertEqual(keywords(reasoning_effort="none", chat_template_kwargs=kwargs),
                         {"reasoning_effort": "none", "enable_thinking": False, "preserve_thinking": True})
        self.assertEqual(keywords(thinking=None, chat_template_kwargs=kwargs), {"preserve_thinking": True})
        with self.assertRaises(corpus.CorpusError):
            keywords(thinking={"type": "on"})
        # A declaration without the switch ignores the object.
        plain = toy_declaration()["reasoning"]
        self.assertEqual(corpus.hf_input({"messages": [user()], "thinking": {"type": "disabled"}}, TRANSPORT,
                                         plain)["keywords"], {})

    def test_the_generator_sends_the_thinking_object_only_where_it_switches(self):
        declaration = toy_declaration()
        self.assertNotIn("object=enabled", frontend_cases.shared_axes(declaration)["thinking"])
        declaration = dict(declaration, reasoning=dict(declaration["reasoning"], switch="thinking-object"))
        cases = frontend_cases.generate(declaration, None, seed=0)
        for kind in frontend_cases.THINKING_OBJECT_TYPES:
            sent = [case for case in cases if f"thinking:object={kind}" in case["features"]]
            self.assertTrue(sent, kind)
            self.assertTrue(all(case["request"]["thinking"] == {"type": kind} for case in sent), kind)

    def test_the_reference_renders_the_top_level_effort_of_both_controls(self):
        model = ToyModel(Path(self.enterContext(tempfile.TemporaryDirectory())))
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
        # A frontend whose reasoning field is reasoning_content keeps it there.
        served = corpus.canonical_messages([user(), {"role": "assistant", "content": "a", "reasoning": "x",
                                                     "reasoning_content": "r"}], TRANSPORT, "reasoning_content")
        self.assertEqual(served[1], {"role": "assistant", "content": "a", "reasoning_content": "r"})
        self.assertEqual(corpus.capture_normalization({"messages": served}, "reasoning_content"), [])

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
            return ToyModel(Path(self.enterContext(tempfile.TemporaryDirectory()))).checkpoint()

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

    def test_mapping_entries_name_their_fields_and_committed_cases(self):
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY)
        table = self.table([LEGACY], [UNDECLARED])
        committed = self.committed(model, table, {case["id"] for case in WITHOUT_LEGACY})
        document = json.loads(table.read_text())
        rule = {"id": "rule", "follows": "a", "rule": "r", "source": "s", "cases": ["low", "absent-case"]}
        table.write_text(json.dumps({**document, "mapping": [rule]}))
        errors = self.check(model, committed, table)
        self.assertEqual([error for error in errors if "mapping" in error],
                         ["mapping entry rule names case absent-case, which the corpus does not hold"])
        table.write_text(json.dumps({**document, "mapping": [dict(rule, source="")]}))
        with self.assertRaisesRegex(corpus.CorpusError, "mapping entry rule needs source"):
            corpus.read_divergences(table)

    def test_a_case_that_differs_without_an_entry_fails(self):
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY)
        table = self.table([LEGACY], [UNDECLARED])
        committed = self.committed(model, table, {case["id"] for case in WITHOUT_LEGACY})
        vectors = json.loads((committed / "corpus.json").read_text())
        vectors[0]["rendered_text"] += "x"
        corpus.write_json(committed / "corpus.json", vectors)
        errors = self.check(model, committed, table)
        self.assertTrue(any("case low: differs from the reference" in error for error in errors), errors)


TOY_RECIPE = {"package": "toy-recipe", "version": "1.0", "wheel": "toy_recipe-1.0-py3-none-any.whl",
              "wheel_sha256": "a" * 64, "installed_sha256": "b" * 64, "python_env": "TOY_RECIPE_PYTHON",
              "oracle": "toy_oracle.py"}


def toy_parse(text):
    """A toy completion as a recipe parser reads it: reasoning to the bar,
    then content and each call's name and arguments."""
    reasoning, _, rest = text.removesuffix("<end>").partition("|")
    content, *calls = rest.split("<call>")
    calls = [{"name": call[:call.index("{")], "arguments": call[call.index("{"):]} for call in calls]
    return {"reasoning": reasoning, "content": content, "tool_calls": calls,
            "finish_reason": "tool_calls" if calls else "stop"}


class FakeOracle:
    """A recipe that renders as the toy reference does and parses the toy
    completion, except that it refuses the cases `refuses` names and gives
    `renders` (a case id to its outcome) and `parses` (a round trip id to its
    parse) as written. `environment` overrides what it reports installed."""

    def __init__(self, model, renders=None, parses=None, refuses=("undeclared",), environment=None):
        self.model = model
        self.renders = renders or {}
        self.parses = parses or {}
        self.refuses = refuses
        self.environment = {"package": TOY_RECIPE["package"], "version": TOY_RECIPE["version"],
                            "installed_sha256": TOY_RECIPE["installed_sha256"], "python": "3.12.0",
                            **(environment or {})}

    def run(self, cases, rounds):
        reference = toy_reference()
        outcomes = {}
        for case in cases:
            outcome = reference.outcome(self.model.hf_input(case["request"]))
            if case["id"] in self.renders:
                outcome = self.renders[case["id"]]
            elif case["id"] in self.refuses or "refusal" in outcome:
                outcome = {"refusal": {"status": 400, "message": "refused"}}
            else:
                outcome = {key: outcome[key] for key in ("rendered_text", "token_ids")}
            outcomes[case["id"]] = outcome
        parses = {vector["id"]: self.parses.get(vector["id"]) or toy_parse("".join(map(chr, vector["token_ids"])))
                  for vector in rounds}
        return {"environment": self.environment, "cases": outcomes, "rounds": parses}


def toy_render(text):
    return {"rendered_text": text, "token_ids": [ord(character) for character in text]}


def mapping_entry(identifier, follows, cases, recipe_when=None, **recipe):
    """A mapping entry; `recipe` sets its recipe_message or recipe_reviewed."""
    entry = {"id": identifier, "follows": follows, "rule": "r", "source": "s", "cases": cases}
    return {**entry, **({"recipe_when": recipe_when} if recipe_when else {}), **recipe}


class RecipeOracleTests(Harness):
    """The recipe oracle hook and gate, with a fake recipe."""

    def model(self, cases=WITHOUT_LEGACY, recipe=TOY_RECIPE):
        return ToyModel(self.directory, cases=cases, recipe=recipe)

    def tables(self, mapping=(), parser=(), inputs=(UNDECLARED,)):
        path = self.table([LEGACY], list(inputs))
        document = json.loads(path.read_text())
        path.write_text(json.dumps({**document, "mapping": list(mapping), "parser": list(parser)}))
        return path

    def corpus_of(self, model, table, oracle):
        return self.committed(model, table, {case["id"] for case in model.cases_list}, oracle)

    def record(self, directory):
        return json.loads((directory / corpus.RECIPE_FILE).read_text())

    def reviews(self, directory, *keys):
        """A recipe_reviewed of the differences of the cases or round trips
        `keys` names in the corpus at `directory`."""
        vectors, rounds, coverage, _ = corpus.read_corpus(directory)
        found = {difference.key: difference.fingerprint for difference in corpus.recipe_differences(
            self.record(directory), coverage, vectors, rounds)}
        return {key: found[key] for key in keys}

    def test_an_agreeing_recipe_is_recorded_with_its_pin_and_passes(self):
        model = self.model()
        table = self.tables()
        committed = self.corpus_of(model, table, FakeOracle(model))
        record = self.record(committed)
        self.assertEqual(record["recipe"], {**{key: TOY_RECIPE[key] for key in corpus.RECIPE_RECORDED},
                                            "python": "3.12.0"})
        rows = [*record["cases"].values(), *record["rounds"].values()]
        self.assertTrue(rows and all(row["agrees"] for row in rows), rows)
        self.assertEqual(record["cases"]["critic"], {
            "request_sha256": hashlib.sha256(json.dumps(
                next(case["request"] for case in CASES if case["id"] == "critic")).encode()).hexdigest(),
            "agrees": True, "refusal": {"status": 400, "message": "refused"}})
        self.assertEqual(set(record["rounds"]), {"answer-round-end", "called-round-end"})
        self.assertEqual(self.check(model, committed, table), [])
        (committed / corpus.RECIPE_FILE).unlink()
        self.assertEqual(self.check(model, committed, table),
                         ["the cases module pins toy-recipe 1.0, but the corpus has no corpus-recipe.json"])

    def test_a_difference_must_match_exactly_one_entry(self):
        model = self.model()
        oracle = FakeOracle(model, renders={"low": toy_render("<sys>low!<user>hi<asst>")})
        committed = self.corpus_of(model, self.tables(), oracle)
        self.assertEqual(self.record(committed)["cases"]["low"]["difference"], "render")
        rule = mapping_entry("low-rule", "chat_template.jinja", ["low"], ["effort:low", "recipe:render"],
                             recipe_reviewed=self.reviews(committed, "low"))
        table = self.tables([rule])
        self.assertEqual(self.check(model, committed, table), [])
        twin = dict(rule, id="twin")
        self.assertIn("case low: the recipe renders other text, and several entries explain it (low-rule, twin)",
                      self.check(model, committed, self.tables([rule, twin])))
        # The same text as other ids is a class of its own, and another
        # difference than the reviewed one.
        oracle = FakeOracle(model, renders={"low": {**toy_render("<sys>low<user>hi<asst>"), "token_ids": [0]}})
        committed = self.corpus_of(model, table, oracle)
        self.assertIn("case low: the recipe gives other token ids, and no inputs or mapping entry explains it",
                      self.check(model, committed, table))

    def test_a_refusal_is_explained_by_the_cause_its_message_names(self):
        pinned = {"id": "pinned", "request": {"messages": [user("p")]}, "features": ["fixed:pinned"]}
        model = self.model(cases=WITHOUT_LEGACY + [pinned])
        role = {"refusal": {"status": 400, "message": "unknown variant `critic`, expected `user`"}}
        oracle = FakeOracle(model, renders={"low": role}, refuses=("undeclared", "pinned"))
        roles = mapping_entry("roles", "chat_template.jinja", ["low"], ["effort:low", "recipe:refuses"],
                              recipe_message="^unknown variant `[a-z]+`")
        fixed = mapping_entry("fixed", "chat_template.jinja", ["pinned"], ["fixed:pinned", "recipe:refuses"],
                              recipe_message="^refused$")
        table = self.tables([roles, fixed])
        committed = self.corpus_of(model, table, oracle)
        self.assertEqual(self.check(model, committed, table), [])
        # The same case refused for another cause carries the same tags, and
        # no entry explains it.
        json_word = {"refusal": {"status": 400, "message": "Prompt must contain the word 'json'"}}
        oracle = FakeOracle(model, renders={"low": json_word}, refuses=("undeclared", "pinned"))
        committed = self.corpus_of(model, table, oracle)
        self.assertEqual(self.check(model, committed, table), [
            "case low: the recipe refuses it ({'status': 400, 'message': \"Prompt must contain the word 'json'\"}), "
            "and no inputs or mapping entry explains it",
            "mapping entry roles is stale: it explains no committed difference from the recipe"])
        # Nor does a review of it: a refusal is explained by its message.
        reviewed = mapping_entry("reviewed", "chat_template.jinja", ["low"], ["effort:low", "recipe:refuses"],
                                 recipe_reviewed=self.reviews(committed, "low"))
        self.assertIn("case low: the recipe refuses it ({'status': 400, 'message': \"Prompt must contain the word "
                      "'json'\"}), and no inputs or mapping entry explains it",
                      self.check(model, committed, self.tables([reviewed, fixed])))
        # An entry that would explain a refusal by its message or tags alone
        # is refused.
        for tags, recipe, error in (
                (["case:generated", "recipe:refuses"], {"recipe_message": "."}, "by its message alone"),
                (["recipe:refuses"], {"recipe_message": "."}, "by its message alone"),
                (["effort:low", "recipe:refuses"], {}, "sets recipe_when but neither recipe_message nor "
                                                       "recipe_reviewed")):
            broad = mapping_entry("broad", "chat_template.jinja", ["low"], tags, **recipe)
            with self.subTest(tags=tags, recipe=recipe), self.assertRaisesRegex(
                    corpus.CorpusError, f"mapping entry broad (would explain a refusal )?{error}"):
                corpus.read_divergences(self.tables([broad, fixed]))

    def test_only_a_review_explains_a_render_an_acceptance_or_a_parse(self):
        model = self.model()
        renders = {"low": toy_render("<sys>low!<user>hi<asst>"), "high": toy_render("<sys>max!<user>hi<asst>")}
        parse = {"reasoning": "", "content": "", "finish_reason": "tool_calls",
                 "tool_calls": [{"name": "f", "arguments": "{\"n\": true}"}]}
        oracle = FakeOracle(model, renders=renders, parses={"called-round-end": parse}, refuses=())
        committed = self.corpus_of(model, self.tables(), oracle)
        reviewed = self.reviews(committed, "low", "high", "undeclared", "called-round-end")
        efforts = mapping_entry("efforts", "chat_template.jinja", ["low"], ["recipe:render"],
                                recipe_reviewed={key: reviewed[key] for key in ("low", "high")})
        accepted = dict(UNDECLARED, recipe_when=["effort:undeclared", "recipe:accepts"],
                        recipe_reviewed={"undeclared": reviewed["undeclared"]})
        bools = {"id": "bools", "recipe_when": ["calls:1", "parse:tool_calls"], "reason": "-",
                 "recipe_reviewed": {"called-round-end": reviewed["called-round-end"]}}
        table = self.tables([efforts], [bools], [accepted])
        self.assertEqual(self.check(model, committed, table), [])
        # A render the entry did not review is not explained, though its
        # tags hold, nor is an acceptance or a parse; a message pattern
        # explains no render.
        partial = mapping_entry("efforts", "chat_template.jinja", ["low"], ["recipe:render"],
                                recipe_reviewed={"low": reviewed["low"]})
        maxes = mapping_entry("maxes", "chat_template.jinja", ["high"], ["effort:high", "recipe:render"],
                              recipe_message=".")
        errors = self.check(model, committed, self.tables([partial, maxes], [dict(bools, recipe_reviewed={
            "answer-round-end": reviewed["called-round-end"]})], [dict(accepted, recipe_reviewed={
                "low": reviewed["undeclared"]})]))
        self.assertIn("case high: the recipe renders other text, and no inputs or mapping entry explains it", errors)
        self.assertIn("case undeclared: the recipe renders it where the corpus refuses, and no inputs or mapping "
                      "entry explains it", errors)
        self.assertIn("round trip called-round-end: the recipe's parse differs in tool_calls, and no parser entry "
                      "explains it", errors)
        # A parse rule explains no refusal of the round trip's request.
        oracle = FakeOracle(model, renders=renders, parses={
            "called-round-end": {"refusal": {"status": 400, "message": "unrelated"}}}, refuses=())
        refused = self.corpus_of(model, table, oracle)
        self.assertIn("round trip called-round-end: the recipe's parse differs in refused, and no parser entry "
                      "explains it", self.check(model, refused, table))

    def test_a_review_binds_one_difference_and_fails_once_it_changes(self):
        model = self.model()
        oracle = FakeOracle(model, renders={"low": toy_render("<sys>low!<user>hi<asst>")})
        committed = self.corpus_of(model, self.tables(), oracle)
        vector = next(vector for vector in json.loads((committed / "corpus.json").read_text())
                      if vector["id"] == "low")
        fingerprint = corpus.difference_fingerprint(self.record(committed)["cases"]["low"],
                                                    corpus.corpus_outcome(vector))
        self.assertEqual(self.reviews(committed, "low"), {"low": fingerprint})
        unexplained = "case low: the recipe renders other text, and no inputs or mapping entry explains it"
        # A review of another case's difference explains nothing here.
        reviewed = mapping_entry("reviewed", "chat_template.jinja", ["low"], ["effort:low", "recipe:render"],
                                 recipe_reviewed={"absent": fingerprint})
        self.assertEqual(self.check(model, committed, self.tables([reviewed])), [
            unexplained, "mapping entry reviewed is stale: it explains no committed difference from the recipe"])
        table = self.tables([dict(reviewed, recipe_reviewed={"low": fingerprint})])
        self.assertEqual(self.check(model, committed, table), [])
        # Another render of the same case, of the same class and tags, is
        # not the reviewed difference.
        oracle = FakeOracle(model, renders={"low": toy_render("<sys>low?<user>hi<asst>")})
        committed = self.corpus_of(model, table, oracle)
        errors = self.check(model, committed, table)
        self.assertEqual(errors[:2], [
            unexplained, "mapping entry reviewed is stale: it explains no committed difference from the recipe"])
        self.assertRegex(errors[2], "^mapping entry reviewed reviewed the difference of low from the recipe, which "
                                    "the corpus no longer holds: it has the fingerprint [0-9a-f]{64}$")
        self.assertEqual(len(errors), 3)
        # Minimization refuses a review of a case the full corpus lacks.
        table = self.tables([dict(reviewed, recipe_reviewed={"low": fingerprint, "gone": fingerprint})])
        full = self.generate(model, table, oracle=FakeOracle(model, renders={
            "low": toy_render("<sys>low!<user>hi<asst>")}))
        with self.assertRaisesRegex(corpus.CorpusError, "mapping entry reviewed reviewed gone, which the full corpus "
                                                        "does not hold"):
            corpus.minimize(model, full, self.directory / "minimal", table)

    def test_message_patterns_and_reviews_have_their_shape(self):
        rule = mapping_entry("rule", "chat_template.jinja", ["low"], ["effort:low", "recipe:refuses"])
        for change, error in (
                ({"recipe_message": "(unclosed"}, "has a recipe_message that is not a regular expression"),
                ({"recipe_message": ""}, "has a recipe_message that is not a regular expression"),
                ({"recipe_reviewed": {"low": "x"}}, "has a recipe_reviewed that is not an object of case or round "
                                                    "trip ids to fingerprints"),
                ({"recipe_reviewed": {}}, "has a recipe_reviewed that is not an object"),
                ({"recipe_when": None, "recipe_message": "x"}, "sets recipe_message or recipe_reviewed without "
                                                               "recipe_when")):
            with self.subTest(change=change), self.assertRaisesRegex(corpus.CorpusError, error):
                corpus.read_divergences(self.tables([{key: value for key, value in {**rule, **change}.items()
                                                      if value is not None}]))

    def test_an_inputs_entry_explains_what_the_recipe_accepts(self):
        model = self.model()
        committed = self.corpus_of(model, self.tables(), FakeOracle(model, refuses=()))
        self.assertEqual(self.record(committed)["cases"]["undeclared"]["difference"], "accepts")
        explained = dict(UNDECLARED, recipe_when=["effort:undeclared", "recipe:accepts"],
                         recipe_reviewed=self.reviews(committed, "undeclared"))
        table = self.tables(inputs=[explained])
        self.assertEqual(self.check(model, committed, table), [])

    def test_an_unattributed_difference_fails_the_gate_and_minimization(self):
        model = self.model()
        table = self.tables()
        committed = self.corpus_of(model, table, FakeOracle(model, refuses=("undeclared", "high")))
        unexplained = ("case high: the recipe refuses it ({'status': 400, 'message': 'refused'}), and no inputs "
                       "or mapping entry explains it")
        self.assertEqual(self.check(model, committed, table), [unexplained])
        with self.assertRaisesRegex(corpus.CorpusError, "differences from the recipe match no entry or several: "
                                                        + unexplained.replace("(", r"\(").replace(")", r"\)")):
            corpus.minimize(model, self.directory / "full", self.directory / "minimal", table)
        # A parser entry explains no render difference, and is stale itself.
        parser = {"id": "p", "recipe_when": ["effort:high"], "recipe_message": "^refused$", "reason": "-"}
        self.assertEqual(self.check(model, committed, self.tables(parser=[parser])), [
            unexplained, "parser entry p is stale: it explains no committed difference from the recipe"])

    def test_mapping_entries_are_judged_by_the_authority_they_follow(self):
        model = self.model()
        model.cases.PIN = {**model.cases.PIN, "source": "encoding.py", "source_sha256": ""}
        reference = mapping_entry("R1", "encoding.py", ["low", "high"])
        recipe = mapping_entry("R2", "recipe", ["low", "high"])
        table = self.tables([reference, recipe])
        committed = self.corpus_of(model, table, FakeOracle(model))
        self.assertEqual(self.check(model, committed, table), [
            "mapping entry R1 follows encoding.py where the recipe differs, but every case it names agrees with "
            "the recipe"])
        # Once one of its cases differs, R1 holds, and R2, which names it, fails.
        committed = self.corpus_of(model, table, FakeOracle(model, renders={"high": toy_render("<sys>max<user>hi")}))
        reference = dict(reference, recipe_when=["effort:high", "recipe:render"],
                         recipe_reviewed=self.reviews(committed, "high"))
        table = self.tables([reference, recipe])
        self.assertEqual(self.check(model, committed, table),
                         ["mapping entry R2 follows the recipe, but its case high differs from it"])

    def test_another_installed_recipe_is_refused(self):
        model = self.model()
        table = self.tables()
        for key, value in (("version", "0.9"), ("installed_sha256", "c" * 64), ("package", "other")):
            with self.subTest(key=key), self.assertRaisesRegex(
                    corpus.CorpusError, f"TOY_RECIPE_PYTHON has {key} '{value}' installed, not the pinned"):
                self.generate(model, table, "refused", FakeOracle(model, environment={key: value}))
        self.assertFalse((self.directory / "refused").exists(), "a refused oracle writes no corpus")
        with unittest.mock.patch.dict(os.environ), self.assertRaisesRegex(
                corpus.CorpusError, "TOY_RECIPE_PYTHON is unset: the toy-recipe oracle runs under it"):
            os.environ.pop("TOY_RECIPE_PYTHON", None)
            self.generate(model, table)
        with self.assertRaisesRegex(corpus.CorpusError, "a recipe oracle needs the cases module's RECIPE pin"):
            self.generate(self.model(recipe=None), table, oracle=FakeOracle(model))
        # A record of another recipe than the pinned one fails the gate.
        committed = self.corpus_of(model, table, FakeOracle(model))
        model.cases.RECIPE = dict(TOY_RECIPE, version="1.1")
        self.assertEqual(self.check(model, committed, table),
                         ["corpus-recipe.json records version '1.0', not the pinned '1.1'"])

    def test_round_trips_compare_arguments_as_json_values(self):
        model = self.model()
        table = self.tables()
        parse = {"reasoning": "", "content": "", "finish_reason": "tool_calls",
                 "tool_calls": [{"name": "f", "arguments": "{ \"n\" : 1.0 }"}]}
        committed = self.corpus_of(model, table, FakeOracle(model, parses={"called-round-end": parse}))
        self.assertTrue(self.record(committed)["rounds"]["called-round-end"]["agrees"])
        self.assertEqual(self.check(model, committed, table), [])
        parse["tool_calls"][0]["arguments"] = "{\"n\": true}"
        committed = self.corpus_of(model, table, FakeOracle(model, parses={"called-round-end": parse}))
        self.assertEqual(self.record(committed)["rounds"]["called-round-end"]["differences"], ["tool_calls"])
        unexplained = ("round trip called-round-end: the recipe's parse differs in tool_calls, and no parser entry "
                       "explains it")
        self.assertEqual(self.check(model, committed, table), [unexplained])
        # An inputs or mapping entry explains no parse; a parser entry does,
        # by its review, where the case's features, the fields and the
        # prompt's mode hold.
        reviewed = self.reviews(committed, "called-round-end")
        rule = mapping_entry("rule", "s", ["called"], ["calls:1"], recipe_reviewed=reviewed)
        self.assertIn(unexplained, self.check(model, committed, self.tables([rule])))
        parser = {"id": "bools", "recipe_when": ["calls:1", "parse:tool_calls", "mode:thinking"], "reason": "-",
                  "recipe_reviewed": reviewed}
        self.assertEqual(self.check(model, committed, self.tables(parser=[parser])), [])
        # A refused request and a failed parse are differences too.
        committed = self.corpus_of(model, table, FakeOracle(model, parses={
            "called-round-end": {"refusal": {"status": 400, "message": "no"}},
            "answer-round-end": {"error": {"class": "RuntimeError", "message": "decode"}}}))
        rounds = self.record(committed)["rounds"]
        self.assertEqual((rounds["called-round-end"]["differences"], rounds["answer-round-end"]["differences"]),
                         (["refused"], ["error"]))
        # A parser entry's message pattern reads a refused request's message;
        # a failed parse, like any other parse, is explained by its review.
        refused = {"id": "refused", "recipe_when": ["calls:1", "parse:refused"], "recipe_message": "^no$",
                   "reason": "-"}
        failed = {"id": "failed", "recipe_when": ["parse:error"], "reason": "-",
                  "recipe_reviewed": self.reviews(committed, "answer-round-end")}
        self.assertEqual(self.check(model, committed, self.tables(parser=[refused, failed])), [])
        self.assertIn("round trip answer-round-end: the recipe's parse differs in error, and no parser entry "
                      "explains it", self.check(model, committed, self.tables(parser=[refused, dict(
                          failed, recipe_reviewed=None, recipe_when=["ends:assistant", "parse:error"],
                          recipe_message="decode")])))
        self.assertEqual(self.check(model, committed, self.tables(parser=[dict(refused, recipe_message="^n$"),
                                                                          failed])), [
            "round trip called-round-end: the recipe's parse differs in refused, and no parser entry explains it",
            "parser entry refused is stale: it explains no committed difference from the recipe"])

    def test_canonical_arguments_are_json_values(self):
        same = corpus.canonical_arguments
        self.assertEqual(same('{"a": [1, 2.50], "b": {"y": 1e2, "x": -0.0}}'),
                         same('{"b": {"x": 0, "y": 100}, "a": [1.0, 2.5]}'))
        self.assertNotEqual(same('{"a": true}'), same('{"a": 1}'))
        self.assertNotEqual(same('{"a": null}'), same('{"a": 0}'))
        self.assertNotEqual(same('{"a": "1"}'), same('{"a": 1}'))
        self.assertEqual(same("not json"), ["text", "not json"])
        self.assertNotEqual(same('"x"'), same("x"))

    def test_canonical_numbers_keep_every_digit(self):
        same = corpus.canonical_arguments
        # Beyond 28 digits, where the default decimal context rounds.
        self.assertNotEqual(same('{"n": 123456789012345678901234567890}'),
                            same('{"n": 123456789012345678901234567891}'))
        self.assertNotEqual(same('{"n": 0.12345678901234567890123456789012}'),
                            same('{"n": 0.12345678901234567890123456789013}'))
        self.assertNotEqual(same('{"n": 1234567890123456789012345678901234567890e-10}'),
                            same('{"n": 123456789012345678901234567890.1234567891}'))
        self.assertEqual(same('{"n": 123456789012345678901234567890.000}'),
                         same('{"n": 1234567890123456789012345678900e-1}'))
        self.assertEqual(same('{"n": 1.5e400}'), same('{"n": 15E399}'))
        self.assertEqual(same('{"n": 0.0e-7}'), same('{"n": -0}'))
        self.assertNotEqual(same('{"n": -1e2}'), same('{"n": 1e2}'))

    def test_minimization_keeps_one_case_per_entry_and_the_rows_of_kept_cases(self):
        model = self.model(cases=WITHOUT_LEGACY + [REDUNDANT])
        oracle = FakeOracle(model, renders={"low": toy_render("<sys>low!<user>hi<asst>"),
                                            "low-again": toy_render("<sys>low!<user>hey<asst>")})
        full = self.generate(model, self.tables(), oracle=oracle)
        table = self.tables([mapping_entry("lows", "chat_template.jinja", ["low"], ["effort:low", "recipe:render"],
                                           recipe_reviewed=self.reviews(full, "low", "low-again"))])
        full = self.generate(model, table, oracle=oracle)
        self.assertEqual(set(self.record(full)["cases"]), {case["id"] for case in model.cases_list})
        minimal = self.directory / "minimal"
        corpus.minimize(model, full, minimal, table)
        record, coverage = self.record(minimal), json.loads((minimal / "corpus-coverage.json").read_text())
        self.assertEqual(set(record["cases"]), set(coverage["cases"]))
        self.assertEqual(set(record["rounds"]), set(coverage["rounds"]))
        self.assertIn("low", record["cases"])
        self.assertNotIn("low-again", record["cases"], "one case per entry stays")
        self.assertEqual(self.check(model, minimal, table), [])
        (full / corpus.RECIPE_FILE).unlink()
        with self.assertRaisesRegex(corpus.CorpusError, "pins RECIPE, so minimize needs the full corpus's "
                                                        "corpus-recipe.json"):
            corpus.minimize(model, full, minimal, table)

    def test_a_record_that_no_longer_holds_against_the_corpus_fails(self):
        model = self.model()
        committed = self.corpus_of(model, self.tables(), FakeOracle(model, renders={"low": toy_render("<sys>low!")}))
        table = self.tables([mapping_entry("low-rule", "chat_template.jinja", ["low"], ["effort:low", "recipe:render"],
                                           recipe_reviewed=self.reviews(committed, "low"))])
        self.assertEqual(self.check(model, committed, table), [])
        record = self.record(committed)
        changed = json.loads(json.dumps(record))
        changed["cases"]["high"]["sha256"] = "0" * 64
        changed["cases"]["low"]["recipe"] = toy_render("<sys>low<user>hi<asst>")
        changed["rounds"]["answer-round-end"]["sha256"] = "0" * 64
        del changed["cases"]["int"]
        corpus.write_json(committed / corpus.RECIPE_FILE, changed)
        errors = self.check(model, committed, table)
        self.assertEqual(errors[:4], [
            "corpus-recipe.json and corpus.json name different cases",
            "case low: corpus-recipe.json records the difference 'render', but the recipe's outcome gives None",
            "case high: corpus-recipe.json records another render than the corpus's",
            "round trip answer-round-end: corpus-recipe.json records another parse than the turn the vector expects"])
        # The changed row is no longer the reviewed difference.
        self.assertIn("case low: the recipe renders other text, and no inputs or mapping entry explains it", errors)

    def test_recipe_tags_need_a_recipe_pin_and_their_shape(self):
        model = self.model(recipe=None)
        parser = {"id": "p", "recipe_when": ["calls:1", "recipe:parse"], "recipe_message": "x", "reason": "-"}
        table = self.tables(parser=[parser])
        committed = self.corpus_of(model, table, None)
        self.assertEqual(self.check(model, committed, table),
                         ["parser entry p sets recipe_when, but the cases module pins no RECIPE"])
        corpus.write_json(committed / corpus.RECIPE_FILE, {"schema": corpus.RECIPE_SCHEMA, "cases": {}, "rounds": {}})
        self.assertIn("corpus-recipe.json is committed, but the cases module pins no RECIPE",
                      self.check(model, committed, self.tables()))
        with self.assertRaisesRegex(corpus.CorpusError, "parser entry p needs reason"):
            corpus.read_divergences(self.tables(parser=[{"id": "p", "recipe_when": ["x"]}]))
        with self.assertRaisesRegex(corpus.CorpusError, "inputs entry undeclared-effort has a recipe_when that is "
                                                        "not a list of tags"):
            corpus.read_divergences(self.tables(inputs=[dict(UNDECLARED, recipe_when="recipe:accepts")]))
        with self.assertRaisesRegex(corpus.CorpusError, "toy: RECIPE lacks oracle"):
            self.model(recipe={key: value for key, value in TOY_RECIPE.items() if key != "oracle"})

    def test_rows_bind_the_requests_and_token_ids_they_were_computed_from(self):
        model = self.model()
        table = self.tables([mapping_entry("high-rule", "chat_template.jinja", ["high"],
                                           ["effort:high", "recipe:refuses"], recipe_message="^refused$")])
        oracle = FakeOracle(model, refuses=("undeclared", "high"))
        stale = self.record(self.corpus_of(model, table, oracle))
        # A strict tool the template never reads renders as before; the
        # recipe may read it, so an old row says nothing of the new request.
        strict = {"type": "function", "function": {**TOOL["function"], "strict": True}}
        model.cases_list = copy.deepcopy(model.cases_list)
        for case in model.cases_list:
            if case["id"] in ("low", "high", "called"):
                case["request"]["tools"] = [strict]
        committed = self.corpus_of(model, table, oracle)
        self.assertEqual(self.check(model, committed, table), [])
        corpus.write_json(committed / corpus.RECIPE_FILE, stale)
        self.assertEqual(self.check(model, committed, table), [
            "case low: corpus-recipe.json was computed from another request than the corpus's",
            "case high: corpus-recipe.json was computed from another request than the corpus's",
            "case called: corpus-recipe.json was computed from another request than the corpus's",
            "round trip called-round-end: corpus-recipe.json was computed from another request or other token ids "
            "than the round trip's"])
        # Other completion ids alone unbind a round trip's row too.
        vector = json.loads((committed / "corpus-rounds.json").read_text())[0]
        row = self.record(committed)["rounds"][vector["id"]]
        self.assertIsNone(corpus.round_row_problem(vector, row))
        self.assertIn("another request or other token ids",
                      corpus.round_row_problem({**vector, "token_ids": vector["token_ids"][:-1]}, row))

    def test_an_internal_recipe_failure_is_no_refusal(self):
        model = self.model()
        table = self.tables()
        for oracle in (FakeOracle(model, renders={"low": {"refusal": {"status": 500, "message": "panic"}}}),
                       FakeOracle(model, renders={"low": {"refusal": {"message": "no status"}}}),
                       FakeOracle(model, parses={"called-round-end": {"refusal": {"status": 503, "message": "x"}}})):
            with self.subTest(oracle=oracle.renders or oracle.parses), self.assertRaisesRegex(
                    corpus.CorpusError, "the recipe fails internally, not with a client error, on "
                                        "(case low|round trip called-round-end)"):
                self.generate(model, table, "failed", oracle)
        # A record that holds one as a refusal fails the gate.
        committed = self.corpus_of(model, table, FakeOracle(model))
        record = self.record(committed)
        record["cases"]["critic"]["refusal"]["status"] = 500
        corpus.write_json(committed / corpus.RECIPE_FILE, record)
        self.assertEqual(self.check(model, committed, table), [
            "case critic: corpus-recipe.json records an internal recipe failure as its refusal"])

    def test_entries_of_the_recipe_sections_need_distinct_ids(self):
        parser = {"id": "shared", "recipe_when": ["calls:1", "recipe:parse"], "recipe_message": "x", "reason": "-"}
        rule = mapping_entry("shared", "chat_template.jinja", ["high"], ["effort:high", "recipe:render"],
                             recipe_message="x")
        with self.assertRaisesRegex(corpus.CorpusError, "inputs, mapping and parser entries share the id shared"):
            corpus.read_divergences(self.tables([rule], [parser]))
        with self.assertRaisesRegex(corpus.CorpusError, "share the id undeclared-effort"):
            corpus.read_divergences(self.tables([mapping_entry("undeclared-effort", "s", ["low"])]))
        # Within one section too.
        content = dict(parser, recipe_when=["calls:1", "parse:content"])
        with self.assertRaisesRegex(corpus.CorpusError, "share the id shared"):
            corpus.read_divergences(self.tables(parser=[parser, content]))

    def test_a_model_without_a_pin_leaves_no_record_to_minimize(self):
        model = self.model()
        table = self.tables()
        full = self.generate(model, table, oracle=FakeOracle(model))
        self.assertTrue((full / corpus.RECIPE_FILE).exists())
        unpinned = self.model(recipe=None)
        self.generate(unpinned, table)
        self.assertFalse((full / corpus.RECIPE_FILE).exists())
        corpus.minimize(unpinned, full, self.directory / "minimal", table)
        self.assertFalse((self.directory / "minimal" / corpus.RECIPE_FILE).exists())

    def test_the_subprocess_oracle_runs_the_pinned_script_under_its_interpreter(self):
        model = self.model()
        model.directory.mkdir(parents=True)
        received = self.directory / "received.json"
        (model.directory / TOY_RECIPE["oracle"]).write_text(
            "import json, sys\n"
            "payload = json.loads(sys.stdin.buffer.read())\n"
            f"open({str(received)!r}, 'w').write(json.dumps(payload))\n"
            "refused = {'refusal': {'status': 400, 'message': 'no'}}\n"
            "sys.stdout.buffer.write(json.dumps({'environment': {'package': 'toy-recipe', 'version': '1.0', "
            f"'installed_sha256': {TOY_RECIPE['installed_sha256']!r}, 'python': '3.0'}}, "
            "'cases': {case['id']: refused for case in payload['cases']}, "
            "'rounds': {vector['id']: refused for vector in payload['rounds']}}).encode())\n")
        variables = {"TOY_RECIPE_PYTHON": sys.executable, "TOY_CHECKPOINT": str(self.directory)}
        with unittest.mock.patch.dict(os.environ, variables):
            full = self.generate(model, self.tables())
        payload = json.loads(received.read_text())
        self.assertEqual(payload["checkpoint"], str(self.directory))
        self.assertEqual([case["id"] for case in payload["cases"]], [case["id"] for case in WITHOUT_LEGACY])
        self.assertEqual(payload["cases"][0]["request"], WITHOUT_LEGACY[0]["request"])
        self.assertEqual(set(payload["rounds"][0]), {"id", "request", "token_ids"})
        record = self.record(full)
        self.assertEqual(record["recipe"]["python"], "3.0")
        self.assertEqual(record["cases"]["low"]["difference"], "refuses")
        self.assertEqual(record["rounds"]["answer-round-end"]["differences"], ["refused"])


class EndingRefusalTests(unittest.TestCase):
    def test_only_ending_refusals_keep_a_round_trip(self):
        ending = {"id": "e", "kind": "refused", "when": ["ends:assistant"], "ending_only": True}
        media = {"id": "m", "kind": "refused", "when": ["media:x"]}
        table = {"inputs": [ending, media]}
        self.assertFalse(corpus.yields_no_round_trip(table, ["ends:assistant"], True))
        self.assertTrue(corpus.yields_no_round_trip(table, ["ends:assistant", "media:x"], True))
        self.assertTrue(corpus.yields_no_round_trip(table, ["media:x"], False))
        self.assertTrue(corpus.yields_no_round_trip(table, [], True))
        self.assertFalse(corpus.yields_no_round_trip(table, [], False))

    def test_ending_only_belongs_to_a_refusal(self):
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "divergences.json"
        path.write_text(json.dumps({"schema": corpus.DIVERGENCES_SCHEMA, "inputs": [
            {"id": "n", "kind": "no-round-trip", "when": ["x"], "ending_only": True}]}))
        with self.assertRaisesRegex(corpus.CorpusError, "sets ending_only but refuses nothing"):
            corpus.read_divergences(path)


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


class TypedArgumentTests(unittest.TestCase):
    PARSER = {"tool_grammar": "xml-typed-parameters-v1", "parameter_open": "<p name=\"",
              "parameter_close": "</p>", "function_close": "</f>"}

    def test_a_typed_value_is_lost_only_to_a_marker_a_raw_string_spells(self):
        def turn(arguments):
            return {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": arguments}}]}

        tools = [{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"n": {"type": "string"}}}}}]
        # The close types every value, whatever its schema names.
        self.assertIsNone(corpus.argument_failure(turn({"n": 1, "u": [1, "</p>"]}), tools, self.PARSER))
        self.assertEqual(corpus.argument_failure(turn({"n": "a</p>"}), tools, self.PARSER),
                         ("delimiter", "n is raw text that spells the marker '</p>'"))
        # A cases module reads arguments the HF input keeps as text.
        self.assertIsNone(corpus.argument_failure(turn("{\"n\": 1}"), tools, self.PARSER, json.loads))
        # Only the parameter close ends a raw value: another parameter's
        # open and the function close inside one are text.
        self.assertIsNone(corpus.argument_failure(turn({"n": "a <p name=\"m\"> b </f>"}), tools, self.PARSER))

    def test_a_marker_holding_an_id_only_spelling_ends_no_raw_value(self):
        """A marked render's value spells such a marker in ordinary tokens,
        which the parser does not read as the marker."""
        def turn(arguments):
            return {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": arguments}}]}

        tools = [{"type": "function", "function": {"name": "f", "parameters": {
            "type": "object", "properties": {"n": {"type": "string"}}}}}]
        parser = {"tool_grammar": "xml-function-parameters-v1", "parameter_open": "<｜M｜ p=\"",
                  "parameter_close": "</｜M｜ p>\n", "function_close": "</f>"}
        value = turn({"n": "x</｜M｜ p>y"})
        self.assertEqual(corpus.argument_failure(value, tools, parser),
                         ("delimiter", "n is raw text that spells the marker '</｜M｜ p>'"))
        self.assertIsNone(corpus.argument_failure(value, tools, parser, id_only=["｜M｜"]))
        self.assertIsNone(corpus.argument_failure(turn({"n": "<｜M｜ p=\""}), tools, parser, id_only=["｜M｜"]))
        # A marker that holds no ID-only spelling still ends the value.
        self.assertEqual(corpus.argument_failure(turn({"n": "x</f>"}), tools, parser, id_only=["｜M｜"]),
                         ("delimiter", "n is raw text that spells the marker '</f>'"))


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


# The toy template as a Python encoder that marks the special tokens it
# writes, as a cases module's `encoder_render` returns `(text, ranges)`.
TOY_MARKED_ENCODER = """\
import json


class TemplateError(Exception):
    pass


def encode(messages, add_generation_prompt, effort=None):
    text, ranges = "", []

    def write(piece, special=False):
        nonlocal text
        if special:
            ranges.append((len(text), len(text) + len(piece)))
        text += piece

    write("<sys>", True)
    write(effort if effort in ("low", "high") else "max")
    for message in messages:
        if message["role"] == "user":
            write("<user>", True)
            write(message["content"])
        elif message["role"] == "assistant":
            write("<asst>", True)
            write((message.get("reasoning_content") or "") + "|" + (message.get("content") or ""))
            for call in message.get("tool_calls") or []:
                write("<call>", True)
                write(call["function"]["name"] + json.dumps(call["function"]["arguments"], ensure_ascii=False))
            write("<end>", True)
            write("\\n")
        elif message["role"] == "tool":
            write("<tool>", True)
            write(message["content"])
        else:
            raise TemplateError("unknown role " + message["role"])
    if add_generation_prompt:
        write("<asst>", True)
    return text, ranges
"""

# The toy tokenizer's added vocabulary: ids past every code point, so no
# character's id is one of them.
ADDED = {text: 0x110000 + index for index, text in enumerate(
    ("<sys>", "<user>", "<asst>", "<call>", "<end>", "<tool>", "<eos>"))}


def text_only(text, ranges, generation):
    """A rewrite of a marked render to its text alone, tokenized whole."""
    return text


def marked_reference(rewrite=None, normalize=None):
    """The toy marked encoder's reference: one id per character of plain
    text, after `normalize`, and an `ADDED` id per range. `rewrite(text,
    ranges, add_generation_prompt)` changes what the render returns."""
    coverage = PythonSourceCoverage(TOY_MARKED_ENCODER)

    def render(messages, tools, add_generation_prompt, continue_final_message, keywords):
        text, ranges = coverage.namespace["encode"](messages, add_generation_prompt, keywords.get("reasoning_effort"))
        return rewrite(text, ranges, add_generation_prompt) if rewrite else (text, ranges)

    def plain(region):
        return [ord(character) for character in (normalize or (lambda text: text))(region)]

    return corpus.Reference(coverage, render, plain, normalize=normalize,
                            tokenize_marked=lambda text, ranges: corpus.marked_token_ids(text, ranges, plain, ADDED))


GENERATION = {"tools": None, "add_generation_prompt": True, "continue_final_message": False, "keywords": {}}


class MarkedTokenizationTests(unittest.TestCase):
    def test_a_range_is_its_added_token_and_its_spelling_in_plain_text_is_characters(self):
        def plain(region):
            regions.append(region)
            return [ord(character) for character in region]

        regions = []
        ids = corpus.marked_token_ids("<end>a<end><user>", [(6, 11), [11, 17]], plain, ADDED)
        self.assertEqual(ids, [*map(ord, "<end>a"), ADDED["<end>"], ADDED["<user>"]])
        # Every region is encoded, the empty ones between and after ranges too.
        self.assertEqual(regions, ["<end>a", "", ""])
        with self.assertRaisesRegex(corpus.CorpusError, r"special range \(0, 1\) spells '<', which is no added token"):
            corpus.marked_token_ids("<end>", [(0, 1)], plain, ADDED)

    def test_special_ranges_are_ascending_disjoint_nonempty_string_indices(self):
        self.assertEqual(corpus.special_ranges("abcd", [[0, 2], (2, 4)]), ((0, 2), (2, 4)))
        self.assertEqual(corpus.special_ranges("", []), ())
        for ranges in ([(1, 3), (2, 4)], [(2, 3), (0, 1)], [(1, 1)], [(0, 5)], [(-1, 1)], [(False, True)],
                       [(0, 1, 2)], [(0.0, 1)], ["ab"]):
            with self.assertRaisesRegex(corpus.CorpusError, "is not an ascending, disjoint, nonempty"):
                corpus.special_ranges("abcd", ranges)
        with self.assertRaisesRegex(corpus.CorpusError, "are NoneType, not a list of pairs"):
            corpus.special_ranges("abcd", None)
        # Indices count code points, not UTF-8 bytes.
        self.assertEqual(corpus.marked_token_ids("\u00e9<end>", [(1, 6)], lambda region: [ord(c) for c in region],
                                                 ADDED), [0xE9, ADDED["<end>"]])

    def test_ranges_within_a_cut_are_relative_and_none_straddles_it(self):
        ranges = ((0, 2), (4, 6), (6, 9))
        self.assertEqual(corpus.ranges_within(ranges, 0, 6, "case c"), ((0, 2), (4, 6)))
        self.assertEqual(corpus.ranges_within(ranges, 6, 10, "case c"), ((0, 3),))
        self.assertIsNone(corpus.ranges_within(None, 0, 6, "case c"))
        with self.assertRaisesRegex(corpus.CorpusError, r"case c: the special range \(4, 6\) straddles the cut at 5"):
            corpus.ranges_within(ranges, 0, 5, "case c")
        with self.assertRaisesRegex(corpus.CorpusError, r"the special range \(0, 2\) straddles the cut at 1"):
            corpus.ranges_within(ranges, 1, 9, "case c")

    def test_only_plain_text_is_normalized(self):
        reference = corpus.Reference(None, None, None, normalize=lambda text: unicodedata.normalize("NFC", text))
        self.assertEqual(reference.normalized("e\u0301<x>e\u0301", ((2, 5),)), ("\u00e9<x>\u00e9", ((1, 4),)))
        self.assertEqual(reference.normalized("e\u0301", None), ("\u00e9", None))

    @unittest.skipUnless(importlib.util.find_spec("tokenizers"), "the tokenizers package is not installed")
    def test_the_plain_tokenizer_reads_an_added_spelling_as_text(self):
        from tokenizers import Tokenizer, models

        vocabulary = {character: index for index, character in enumerate(dict.fromkeys("<end>hi"))}
        tokenizer = Tokenizer(models.BPE(vocabulary, []))
        tokenizer.add_special_tokens(["<end>"])
        self.assertEqual(tokenizer.encode("hi<end>").ids, [5, 6, 7])
        self.assertEqual(corpus.plain_tokenizer(tokenizer).encode("hi<end>").ids, [5, 6, 0, 1, 2, 3, 4])
        marked = corpus.marked_tokenizer(tokenizer)
        self.assertEqual(marked("hi<end><end>", [(7, 12)]), [5, 6, 0, 1, 2, 3, 4, 7])
        # The copy leaves the tokenizer its added tokens.
        self.assertEqual(tokenizer.encode("<end>").ids, [7])


class MarkedRenderTests(Harness):
    """A Python encoder's marked render: its ranges are added tokens, the
    text between them is plain, and its round trips cut by its ranges."""

    def test_an_outcome_tokenizes_a_marked_render_and_a_text_render_whole(self):
        reference = marked_reference()
        outcome = reference.outcome({**GENERATION, "messages": [user("a<user>b")]})
        self.assertEqual(outcome["rendered_text"], "<sys>max<user>a<user>b<asst>")
        self.assertEqual(outcome["special_ranges"], ((0, 5), (8, 14), (22, 28)))
        self.assertEqual(outcome["token_ids"], [ADDED["<sys>"], *map(ord, "max"), ADDED["<user>"],
                                                *map(ord, "a<user>b"), ADDED["<asst>"]])
        # A render that returns its text is tokenized whole, as before.
        outcome = marked_reference(text_only).outcome({**GENERATION, "messages": [user("a<user>b")]})
        self.assertEqual(outcome, {"rendered_text": "<sys>max<user>a<user>b<asst>",
                                   "token_ids": list(map(ord, "<sys>max<user>a<user>b<asst>")),
                                   "lines": outcome["lines"]})
        refused = reference.outcome({**GENERATION, "messages": [{"role": "critic", "content": "no"}]})
        self.assertEqual(refused["refusal"], {"class": "TemplateError", "message": "unknown role critic"})

    def test_a_marked_render_needs_a_reference_that_tokenizes_one(self):
        reference = marked_reference()
        for broken in (corpus.Reference(reference.coverage, reference.render_text, reference.tokenize),
                       reference.with_reference_text(lambda text, arguments: text),
                       marked_reference(lambda text, ranges, generation: [text, ranges]),
                       marked_reference(lambda text, ranges, generation: (text, [(0, 3)]))):
            with self.assertRaises(corpus.CorpusError):
                broken.outcome({**GENERATION, "messages": [user()]})

    def test_a_marked_corpus_writes_no_ranges(self):
        model = ToyModel(self.directory, cases=WITHOUT_LEGACY)
        output = self.directory / "marked"
        corpus.generate(model, output, divergences=self.table(), template=self.template,
                        reference=marked_reference())
        vectors = {vector["id"]: vector for vector in json.loads((output / "corpus.json").read_text())}
        self.assertEqual(vectors["low"]["token_ids"][0], ADDED["<sys>"])
        for name in ("corpus.json", "corpus-rounds.json", "reference.json", "rounds-report.json"):
            self.assertNotIn("special_ranges", (output / name).read_text(), name)
        # The same text rendered as text writes the files it always did.
        marked = output
        output = self.directory / "text"
        corpus.generate(model, output, divergences=self.table(), template=self.template,
                        reference=marked_reference(text_only))
        plain = self.generate(model, self.table(), name="template")
        for name in ("corpus.json", "corpus-rounds.json", "reference.json", "rounds-report.json"):
            self.assertEqual((output / name).read_text(), (plain / name).read_text(), name)
        self.assertNotEqual((marked / "corpus.json").read_text(), (plain / "corpus.json").read_text())

    def test_round_trips_tokenize_the_prompt_and_completion_with_their_ranges(self):
        spelled = {"id": "spelled", "request": {"messages": [user(), {"role": "assistant", "reasoning": "r",
                                                                      "content": "a<end>b<user>"}]},
                   "features": ["ends:assistant"]}
        model = ToyModel(self.directory, cases=[spelled])
        vectors, infos, failure, reason = corpus.round_vectors(model, marked_reference(), spelled)
        self.assertEqual((failure, reason), (None, None))
        (vector,) = vectors
        # The completion ends at the stop token's range, not at its spelling
        # in the content, which stays characters.
        self.assertEqual(vector["token_ids"], [*map(ord, "r|a<end>b<user>"), ADDED["<end>"]])
        self.assertEqual(vector["separator"], "\n")
        self.assertEqual(vector["expect"]["content"], "a<end>b<user>")
        # The prompt is tokenized with its own ranges, so its ids and the
        # completion's are those of the whole marked text.
        self.assertFalse(infos["spelled-round-end"]["boundary_merges"])
        # Rendered as text, the spelling is the first stop token.
        vectors, infos, _, _ = corpus.round_vectors(model, marked_reference(text_only), spelled)
        self.assertEqual(vectors[0]["token_ids"], list(map(ord, "r|a<end>")))
        self.assertEqual(vectors[0]["separator"], "b<user><end>\n")

    def test_a_declared_ending_is_appended_as_its_added_token(self):
        spelled = {"id": "spelled", "request": {"messages": [user(), {"role": "assistant", "content": "a<user>b"}]},
                   "features": ["ends:assistant"]}
        model = ToyModel(self.directory, cases=[spelled], endings={"answer": ["<user>", "<eos>"],
                                                                   "tool_calls": ["<tool>"]},
                         stops=("<user>", "<tool>", "<eos>"))
        vectors, _, failure, _ = corpus.round_vectors(model, marked_reference(), spelled)
        self.assertIsNone(failure)
        self.assertEqual({vector["id"]: vector["token_ids"] for vector in vectors},
                         {f"spelled-round-{name}": [*map(ord, "|a<user>b"), ADDED["<end>"], ord("\n"), ADDED[ending]]
                          for name, ending in (("user", "<user>"), ("eos", "<eos>"))})
        self.assertTrue(all("separator" not in vector for vector in vectors))

    def test_a_generation_prompt_must_mark_what_its_history_render_marks(self):
        answer = next(case for case in CASES if case["id"] == "answer")
        model = ToyModel(self.directory, cases=[answer])
        unmarked = marked_reference(lambda text, ranges, generation: (text, ranges[:-1] if generation else ranges))
        with self.assertRaisesRegex(corpus.CorpusError,
                                    "case answer: its generation prompt has other special ranges than its history"):
            corpus.round_vectors(model, unmarked, answer)

    def test_an_id_only_marker_in_a_raw_value_ends_it_only_in_a_text_render(self):
        string = {"type": "object", "properties": {"n": {"type": "string"}}}
        cases = [calling("close", string, {"n": "a</v>"}), calling("call-close", string, {"n": "a</call>"})]
        model = ToyModel(self.directory, cases=cases)
        model.parser = {"tool_grammar": "xml-key-value-arguments-v1", "parameter_close": "</v>",
                        "tool_call_close": "</call>"}
        model.declaration["vocabulary"]["id_only_markers"] = ["</v>"]

        def failures(reference):
            return {case["id"]: getattr(corpus.round_vectors(model, reference, case)[2], "kind", None)
                    for case in cases}

        self.assertEqual(failures(marked_reference()), {"close": None, "call-close": "delimiter"})
        # Tokenized whole, the value's spelling is the marker's token.
        self.assertEqual(failures(marked_reference(text_only)), {"close": "delimiter", "call-close": "delimiter"})
        del model.declaration["vocabulary"]["id_only_markers"]
        self.assertEqual(failures(marked_reference()), {"close": "delimiter", "call-close": "delimiter"})

    def test_a_marked_completion_the_normalizer_rewrites_keeps_its_normalized_round_trip(self):
        string = {"type": "object", "properties": {"n": {"type": "str"}}}
        decomposed = calling("decomposed", string, {"n": "\"e\u0301"})
        composed = calling("composed", string, {"n": "\"\u00e9"})
        model = ToyModel(self.directory, cases=[decomposed, composed])
        reference = marked_reference(normalize=lambda text: unicodedata.normalize("NFC", text))
        control, _, failure, _ = corpus.round_vectors(model, reference, decomposed)
        self.assertEqual(failure.kind, "tokenizer")
        twin, _, failure, _ = corpus.round_vectors(model, reference, composed)
        self.assertIsNone(failure)
        self.assertEqual(control[0]["token_ids"][-1], ADDED["<end>"])
        self.assertEqual({key: value for key, value in control[0].items() if key != "id"},
                         {key: value for key, value in twin[0].items() if key != "id"})


if __name__ == "__main__":
    unittest.main()

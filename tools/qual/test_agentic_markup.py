"""The agentic raw re-parse reads each shipped frontend's declared markup."""

from qual_test_config import setUpModule

import argparse
from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import agentic_mixed
import common
import declaration
from agentic_metrics import RAW_CALLS_AGREE, RAW_MALFORMED_CALL, RAW_REASONING_ONLY
from agentic_mixed import (TOOL_GRAMMARS, Markup, calls_class, function_parameters, last_turn_class,
                           load_markup, raw_calls, raw_class, raw_turn, walk_step)

# Audited to run once rather than under every gate config (ci/qualification.py).
QUALIFICATION_RUNS_ONCE = "raw re-parse of synthetic turns in each model's committed frontend markup"

ROOT = Path(__file__).resolve().parents[2]
TOOLS = [{'type': 'function', 'function': {'name': 'read', 'parameters': {
    'type': 'object', 'properties': {'path': {'type': 'string'}, 'offset': {'type': 'integer'},
                                     'filter': {'type': 'object'}}}}}]
# Completions the served parser's test reads too, with the calls it returns.
SHARED_CASES = ROOT / 'crates/frontend-format/tests/function_parameter_cases.json'
# A typed-grammar declaration (frontend-format's TYPED_PARAMETERS fixture
# markup), each turn opener and the stop its own boundary token.
TYPED_PARSER = {
    'tool_grammar': 'xml-typed-parameters-v1', 'json_encoding': 'python-json-v1',
    'reasoning_open': '<think>', 'reasoning_close': '</think>',
    'tool_call_open': '\n\n<calls>\n', 'tool_call_close': '</calls>',
    'function_open': '<invoke name="', 'function_name_close': '">\n', 'function_close': '</invoke>\n',
    'parameter_open': '<parameter name="', 'parameter_name_close': '" string="true">',
    'parameter_json_name_close': '" string="false">', 'parameter_close': '</parameter>\n',
    'turn_open': ['A[', 'U['], 'assistant_header': 'A[',
}
TYPED_FRONTEND = {'format': {'parser': TYPED_PARSER}, 'vocabulary': {
    'boundary_tokens': [{'id': 0, 'text': 'A['}, {'id': 1, 'text': 'U['}, {'id': 2, 'text': '[EOS]'}],
    'stop_token_ids': [2]}}


def declared_frontend(model):
    path = declaration.path(model)
    if not path.is_file():
        return None
    declared = json.loads(path.read_text())['inputs'].get('frontend')
    return ROOT / declared if isinstance(declared, str) else None


def frontends():
    """(model, frontend) of every inventoried model whose qualification
    declares a frontend with a `format.parser` section."""
    for inventory in sorted(ROOT.glob('models/*/model.toml')):
        model = inventory.parent.name
        path = declared_frontend(model)
        frontend = json.loads(path.read_text()) if path else {}
        if 'parser' in frontend.get('format', {}):
            yield model, frontend


def markups():
    """(model, Markup) of every declared frontend (`frontends`)."""
    for model, frontend in frontends():
        yield model, Markup(frontend)


# The value of each `format.parser` field a declaration may leave out.
PARSER_DEFAULTS = {
    'answer_prefix': '', 'tool_call_separator': '', 'function_open': '', 'function_name_close': '',
    'function_close': '', 'parameter_value_open': '', 'drop_blank_tool_call_content': False,
    'reserved_markers': [],
}


def parser_section(frontend):
    """The `format.parser` section a declaration states, with its left-out
    fields at their defaults."""
    return {**copy.deepcopy(PARSER_DEFAULTS), **copy.deepcopy(frontend['format']['parser'])}


def parser_frontend(frontend):
    """FRONTEND with its format replaced by its `format.parser` section
    (`parser_section`), as a migrated declaration carries it."""
    return {**frontend, 'format': {'parser': parser_section(frontend)}}


def attributes(markup):
    """What the raw re-parse reads of a Markup."""
    return vars(markup)


def stop_texts(frontend):
    """The text of each declared stop token."""
    boundary = {token['id']: token['text'] for token in frontend['vocabulary']['boundary_tokens']}
    return [boundary[token] for token in frontend['vocabulary']['stop_token_ids']]


def undeclared_ending(frontend, markup):
    """A boundary token that is no declared stop token and no markup the
    re-parse reads."""
    markers = set(markup.turn_open) | set(markup.call_markup) | {
        markup.assistant, markup.reasoning_open, markup.reasoning_close, *stop_texts(frontend)}
    return next(token['text'] for token in frontend['vocabulary']['boundary_tokens']
                if not any(token['text'] in marker or marker in token['text'] for marker in markers))


def grammar(name):
    return next(markup for _, markup in markups() if markup.grammar == name)


class Counted(str):
    """A string that counts the characters its slices, strips and partitions
    copy."""
    copied = 0

    @staticmethod
    def copy(value):
        Counted.copied += len(value)
        return Counted(value)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return Counted.copy(value) if isinstance(key, slice) else value

    def strip(self, *args):
        return Counted.copy(super().strip(*args))

    def lstrip(self, *args):
        return Counted.copy(super().lstrip(*args))

    def rstrip(self, *args):
        return Counted.copy(super().rstrip(*args))

    def partition(self, separator):
        return tuple(Counted.copy(part) for part in super().partition(separator))


class DeclaredMarkup(unittest.TestCase):
    def test_every_declared_frontend_has_a_supported_grammar(self):
        found = {markup.grammar for _, markup in markups()}
        self.assertTrue(found)
        self.assertLessEqual(found, set(TOOL_GRAMMARS))

    def test_function_parameter_frontends_keep_their_markup(self):
        """The constants the re-parse used before it read the declaration."""
        for model, markup in markups():
            if markup.grammar != 'xml-function-parameters-v1':
                continue
            with self.subTest(model=model):
                self.assertEqual(markup.turn_open, ('<|im_start|>',))
                self.assertEqual(markup.assistant, '<|im_start|>assistant')
                self.assertEqual((markup.reasoning_open, markup.reasoning_close), ('<think>', '</think>'))
                self.assertLessEqual({'<|im_end|>', '<|endoftext|>'}, set(markup.end))
                self.assertEqual(markup.call_markup, ('<tool_call>', '</tool_call>', '<function=', '</function>'))
                region = '\n<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>tail'
                self.assertEqual(markup.region(region, TOOLS), ('read', [('path', '\na.py\n')], 'tail'))
                framed = f'{markup.value_lead}a.py{markup.value_trail}'
                self.assertTrue(markup.same_value(framed, 'a.py'))
                # Only the declared framing is markup: a value's own newlines
                # are its bytes.
                own = f'{markup.value_lead}\na.py\r\n{markup.value_trail}'
                self.assertTrue(markup.same_value(own, '\na.py\r\n'))
                self.assertFalse(markup.same_value(own, 'a.py'))

    def test_a_value_is_framed_by_the_whitespace_its_markers_declare(self):
        """A template that writes `>\\n` and `\\n</parameter>` frames each
        value in one newline; one that writes `>` and `</parameter>` frames
        nothing, so a newline value is its own bytes."""
        frames = set()
        for model, frontend in frontends():
            markup = Markup(frontend)
            if markup.grammar != 'xml-function-parameters-v1':
                continue
            parser = frontend['format']['parser']
            with self.subTest(model=model):
                self.assertEqual(parser['parameter_name_close'], markup.parameter_name_close + markup.value_lead)
                self.assertEqual(parser['parameter_close'].rstrip(), markup.value_trail + markup.parameter_close)
                self.assertEqual(markup.same_value('\n', '\n'), not markup.value_lead and not markup.value_trail)
                frames.add((markup.value_lead, markup.value_trail))
        # The declared frontends hold both kinds.
        self.assertEqual(frames, {('\n', '\n'), ('', '')})


class ParserDeclaration(unittest.TestCase):
    """A declaration's `format.parser` gives the shared re-parser the same
    markup with its left-out fields written at their defaults."""

    def test_each_model_parser_section_reads_with_its_defaults(self):
        models = set()
        for model, frontend in frontends():
            migrated = parser_frontend(frontend)
            with self.subTest(model=model):
                self.assertNotIn('template', migrated['format'])
                self.assertEqual(attributes(Markup(migrated)), attributes(Markup(frontend)))
                self.assertEqual(agentic_mixed.library_grammar(migrated), agentic_mixed.library_grammar(frontend))
                models.add(model)
        # Every inventoried model whose qualification declares a frontend is
        # compared, so none is skipped.
        declaring = {inventory.parent.name for inventory in ROOT.glob('models/*/model.toml')
                     if declared_frontend(inventory.parent.name)}
        self.assertTrue(models)
        self.assertEqual(models, declaring)

    def test_a_key_value_section_is_the_shared_example(self):
        """The `format.parser` example the Rust chat-format lanes share is
        the section of an inventoried key-value-grammar frontend."""
        sections = [parser_section(frontend) for _, frontend in frontends()
                    if agentic_mixed.library_grammar(frontend) == 'xml-key-value-arguments-v1']
        self.assertIn({
            'tool_grammar': 'xml-key-value-arguments-v1', 'json_encoding': 'python-json-v1',
            'reasoning_open': '<think>', 'reasoning_close': '</think>', 'answer_prefix': '',
            'tool_call_open': '<tool_call>', 'tool_call_close': '</tool_call>', 'tool_call_separator': '',
            'function_open': '', 'function_name_close': '', 'function_close': '',
            'parameter_open': '<arg_key>', 'parameter_name_close': '</arg_key>',
            'parameter_value_open': '<arg_value>', 'parameter_close': '</arg_value>',
            'drop_blank_tool_call_content': False, 'reserved_markers': [],
            'turn_open': sorted(['<|system|>', '<|user|>', '<|assistant|>', '<|observation|>']),
            'assistant_header': '<|assistant|>'}, sections)

    def test_a_parser_section_is_read_as_declared(self):
        """Its turn openers and header as stated, not derived; a turn opener
        must be a boundary token, and the key-value grammar needs its value
        open."""
        _, frontend = next(frontends())
        section = parser_section(frontend)
        declared = {**frontend, 'format': {'parser': {**section, 'assistant_header': section['turn_open'][0]}}}
        self.assertEqual(Markup(declared).assistant, section['turn_open'][0])
        unknown = {**frontend, 'format': {'parser': {**section, 'turn_open': [*section['turn_open'], '<|nope|>']}}}
        with self.assertRaisesRegex(ValueError, r"turn_open \['<\|nope\|>'\] are not boundary tokens"):
            Markup(unknown)
        key_value = parser_frontend(next(frontend for _, frontend in frontends() if
                                         agentic_mixed.library_grammar(frontend) == 'xml-key-value-arguments-v1'))
        del key_value['format']['parser']['parameter_value_open']
        with self.assertRaisesRegex(ValueError, 'needs parameter_value_open'):
            Markup(key_value)
        unsupported = parser_frontend(frontend)
        unsupported['format']['parser']['tool_grammar'] = 'synthetic-grammar-v1'
        self.assertIsNone(agentic_mixed.library_grammar(unsupported))
        with self.assertRaisesRegex(ValueError, 'unsupported tool grammar'):
            Markup(unsupported)


class DeclaredEnds(unittest.TestCase):
    """A raw turn ends with any declared stop token, as the served decoder
    ends it, and with nothing else."""

    def openings(self, markup, text):
        """(opening, text) with TEXT inside a thinking block the assistant
        header opens, and one the generation opens."""
        return ((markup.assistant + '\n' + markup.reasoning_open + '\n', text),
                (markup.assistant + '\n', markup.reasoning_open + text))

    def test_every_declared_stop_token_ends_a_turn(self):
        for model, frontend in frontends():
            markup = Markup(frontend)
            for stop in stop_texts(frontend):
                for opening, text in self.openings(markup, 'Still thinking' + stop):
                    with self.subTest(model=model, stop=stop, opening=opening):
                        turn = raw_turn(markup, opening, text)
                        self.assertEqual(turn, ('Still thinking', False, ''))
                        self.assertEqual(last_turn_class(markup, *turn), RAW_REASONING_ONLY)
            with self.subTest(model=model):
                self.assertEqual(markup.end, tuple(stop_texts(frontend)))

    def test_an_undeclared_ending_is_no_end_token(self):
        for model, frontend in frontends():
            markup = Markup(frontend)
            ending = undeclared_ending(frontend, markup)
            for opening, text in self.openings(markup, 'Still thinking' + ending):
                with self.subTest(model=model, ending=ending, opening=opening):
                    self.assertEqual(raw_turn(markup, opening, text), 'no end token')


class Resolution(unittest.TestCase):
    def test_the_served_model_id_selects_its_declared_frontend(self):
        args = argparse.Namespace(frontend=None, dry_run=False, no_token_evidence=False)
        environment = {k: v for k, v in os.environ.items() if k != 'CHOPINFER_QUAL_CONFIG'}
        with patch.dict(os.environ, environment, clear=True):
            for model, markup in markups():
                for served in json.loads(declared_frontend(model).read_text())['models']:
                    with self.subTest(model=model):
                        found, record = load_markup(args, served['id'])
                        self.assertIs(type(found), Markup)
                        self.assertEqual(record, {
                            'sha256': hashlib.sha256(declared_frontend(model).read_bytes()).hexdigest(),
                            'tool_grammar': markup.grammar, 'reparser_sha256': None})
            with self.assertRaisesRegex(ValueError, 'pass --frontend'):
                load_markup(args, 'undeclared-model')
        for flags in ({'dry_run': True}, {'no_token_evidence': True}):
            self.assertEqual(load_markup(argparse.Namespace(**{**vars(args), **flags}), 'undeclared-model'), (None, None))


# A model's own re-parser (agentic_markup.py) of a synthetic line format: the
# reasoning ends at `[/reason]`, and each call is `[call]` and a JSON object
# of its name and arguments.
OWN_MARKUP = textwrap.dedent("""\
    import json

    from agentic_metrics import RAW_MALFORMED_CALL, RAW_REASONING_ONLY


    class Markup:
        def __init__(self, frontend):
            lines = frontend['format']['lines']
            self.grammar = 'synthetic-lines-v1'
            self.assistant, self.close, self.call = lines['assistant'], lines['reasoning_close'], lines['call']
            self.end = lines['end']

        def turn(self, opening, text):
            if not opening.endswith(self.assistant):
                return 'no assistant header'
            if not text.endswith(self.end):
                return 'no end token'
            reasoning, close, after = text[:-len(self.end)].partition(self.close)
            return reasoning, bool(close), after

        def last_class(self, reasoning, closed, after):
            if not closed:
                return RAW_REASONING_ONLY if reasoning.strip() else 'empty'
            return RAW_MALFORMED_CALL if after.strip() == self.call else 'output after reasoning'

        def calls(self, after, tools=()):
            content, *regions = after.split(self.call)
            calls = []
            for region in regions:
                try:
                    call = json.loads(region)
                except ValueError:
                    return None
                calls.append((call['name'], [(k, json.dumps(v)) for k, v in call['arguments'].items()]))
            return content, calls

        def same_value(self, raw, value):
            return json.loads(raw) == value
    """)
LINES = {'assistant': '[assistant]', 'reasoning_close': '[/reason]', 'call': '[call]', 'end': '[end]'}
SERVED = 'synthetic-served-model'


class Evidence:
    """The token IDs of one turn: 1 the prompt tail, 2 the generation."""
    ids, prompt_tail = [2], [1]

    def complete(self):
        return True


class ReParserSelection(unittest.TestCase):
    """A format that declares a library tool grammar is re-parsed by the
    shared `Markup`; any other by the model's own agentic_markup.py from its
    model root; one with neither is refused."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tree = Path(directory.name)
        self.root = self.tree / 'models' / 'synthetic'
        self.root.mkdir(parents=True)
        self.frontend = self.root / 'frontend.json'
        self.frontend.write_text(json.dumps({'models': [{'id': SERVED}], 'format': {'lines': LINES}}))
        environment = {k: v for k, v in os.environ.items()
                       if k not in ('CHOPINFER_QUAL_CONFIG', 'CHOPINFER_QUAL_MODEL_ROOT')}
        self.enterContext(patch.dict(os.environ, environment, clear=True))

    def args(self, **flags):
        return argparse.Namespace(**{'frontend': str(self.frontend), 'model_root': None, 'dry_run': False,
                                     'no_token_evidence': False, **flags})

    def own(self):
        (self.root / 'agentic_markup.py').write_text(OWN_MARKUP)

    def inventory(self, *models):
        """Inventory MODELS in the synthetic tree, each declaring the
        synthetic frontend."""
        for model in models:
            (self.tree / 'models' / model).mkdir(exist_ok=True)
            (self.tree / 'models' / model / 'model.toml').write_text('')
            declaration.path(model, self.tree).write_text(
                json.dumps({'inputs': {'frontend': 'models/synthetic/frontend.json'}}))

    def test_a_model_root_supplies_the_re_parser(self):
        self.own()
        markup, record = load_markup(self.args(model_root=str(self.root)), SERVED)
        self.assertNotIsInstance(markup, Markup)
        self.assertEqual(markup.grammar, 'synthetic-lines-v1')
        self.assertEqual(record, {'sha256': hashlib.sha256(self.frontend.read_bytes()).hexdigest(),
                                  'tool_grammar': 'synthetic-lines-v1',
                                  'reparser_sha256': hashlib.sha256(OWN_MARKUP.encode()).hexdigest()})
        with patch.dict(os.environ, {'CHOPINFER_QUAL_MODEL_ROOT': str(self.root)}):
            self.assertEqual(load_markup(self.args(), SERVED)[0].grammar, 'synthetic-lines-v1')

    def test_the_receipt_records_the_re_parser_digest_and_no_path(self):
        self.own()
        with tempfile.TemporaryDirectory() as directory, patch.object(common, 'RECEIPTS', Path(directory)), \
                redirect_stdout(io.StringIO()):
            code = agentic_mixed.main(['--dry-run', '--frontend', str(self.frontend), '--model-root', str(self.root),
                                       '--levels', '1', '--seconds', '.3', '--warmup-waves', '0', '--no-decode',
                                       '--min-samples', '2'])
            receipts = [p.read_text() for p in Path(directory).glob('*.json')]
        self.assertEqual(code, 0)
        sweep = next(json.loads(r) for r in receipts if json.loads(r)['receipt_kind'] == 'sweep')
        self.assertEqual(sweep['frontend'], {'sha256': hashlib.sha256(self.frontend.read_bytes()).hexdigest(),
                                             'tool_grammar': 'synthetic-lines-v1',
                                             'reparser_sha256': hashlib.sha256(OWN_MARKUP.encode()).hexdigest()})
        for receipt in receipts:
            self.assertNotIn(str(self.tree), receipt)

    def test_the_cli_prints_a_refused_selection_and_keeps_its_partial_receipt(self):
        """On stderr, with both remedies; the receipt names only the error
        type, never the model root."""
        outside = self.tree / 'agentic_markup.py'
        outside.write_text(OWN_MARKUP)
        for linked in (False, True):
            if linked:
                (self.root / 'agentic_markup.py').symlink_to(outside)
            with self.subTest(linked=linked), tempfile.TemporaryDirectory() as directory, \
                    patch.object(common, 'RECEIPTS', Path(directory)), redirect_stdout(io.StringIO()), \
                    redirect_stderr(io.StringIO()) as stderr:
                code = agentic_mixed.main(['--dry-run', '--frontend', str(self.frontend), '--model-root', str(self.root),
                                           '--levels', '1', '--seconds', '.3', '--warmup-waves', '0', '--no-decode'])
                receipts = [p.read_text() for p in Path(directory).glob('*.json')]
                self.assertEqual(code, 1)
                for expected in (['is a symlink'] if linked else
                                 ['no library tool grammar', 'format.parser.tool_grammar', 'agentic_markup.py']):
                    self.assertIn(expected, stderr.getvalue())
                self.assertEqual([json.loads(r)['receipt_kind'] for r in receipts], ['partial'])
                self.assertEqual(json.loads(receipts[0])['aborted'], 'runner error: ReParserUnavailable')
                self.assertNotIn(str(self.tree), receipts[0])

    def test_the_raw_re_parse_reads_the_model_re_parser(self):
        self.own()
        markup, _ = load_markup(self.args(model_root=str(self.root)), SERVED)
        call = '[call]' + json.dumps({'name': 'read', 'arguments': {'path': 'a.py', 'offset': 5}})
        answer = {'reasoning': 'Plan', 'content': 'Reading.', 'tool_calls': [
            {'function': {'name': 'read', 'arguments': json.dumps({'path': 'a.py', 'offset': 5})}}]}
        wrong = {**answer, 'tool_calls': [{'function': {'name': 'read', 'arguments': '{"path": "b.py", "offset": 5}'}}]}
        cases = [('Still thinking[end]', None, RAW_REASONING_ONLY), ('Plan[/reason][call][end]', None, RAW_MALFORMED_CALL),
                 ('Plan[/reason]Done.[end]', None, 'output after reasoning'), ('Plan[/reason]Done.', None, 'no end token'),
                 (f'Plan[/reason]Reading.{call}[end]', answer, RAW_CALLS_AGREE),
                 (f'Plan[/reason]Reading.{call}[end]', wrong, 'calls differ'),
                 ('Plan[/reason]Reading.[call]{[end]', answer, 'malformed call region')]
        for text, given, expected in cases:
            with self.subTest(text=text, answer=given is not None):
                decoded = {1: 'Read a.py.[assistant]', 2: text}
                with patch.object(agentic_mixed, 'MARKUP', markup), \
                        patch('agentic_mixed.detokenize', lambda client, model, ids: decoded[ids[0]]):
                    self.assertEqual(raw_class(None, SERVED, Evidence(), given, TOOLS), expected)

    def test_an_inventoried_model_supplies_its_re_parser(self):
        """In-tree, the model root is the directory of the inventoried model
        whose frontend declares the served model."""
        self.own()
        self.inventory('synthetic')
        with patch.object(common, 'ROOT', self.tree):
            for args in (self.args(), self.args(frontend=None)):
                with self.subTest(frontend=args.frontend):
                    self.assertEqual(load_markup(args, SERVED)[0].grammar, 'synthetic-lines-v1')

    def test_models_sharing_a_library_frontend_select_it(self):
        """Two inventoried models that declare the same library frontend
        file name one frontend; no model root is needed."""
        _, frontend = next(frontends())
        self.frontend.write_text(json.dumps({**frontend, 'models': [{'id': SERVED}]}))
        self.inventory('synthetic', 'synthetic_other')
        with patch.object(common, 'ROOT', self.tree):
            markup, record = load_markup(self.args(frontend=None), SERVED)
        self.assertIs(type(markup), Markup)
        self.assertEqual(record['sha256'], hashlib.sha256(self.frontend.read_bytes()).hexdigest())

    def test_models_sharing_a_native_frontend_need_a_model_root(self):
        """Its re-parser is a model's own, so it needs one owning model."""
        self.own()
        self.inventory('synthetic', 'synthetic_other')
        with patch.object(common, 'ROOT', self.tree):
            for args in (self.args(), self.args(frontend=None)):
                with self.subTest(frontend=args.frontend), \
                        self.assertRaisesRegex(ValueError, 'no single inventoried model.*pass --model-root'):
                    load_markup(args, SERVED)
            markup, _ = load_markup(self.args(frontend=None, model_root=str(self.root)), SERVED)
        self.assertEqual(markup.grammar, 'synthetic-lines-v1')

    def test_a_library_grammar_selects_the_shared_re_parser(self):
        """Even beside a model's own re-parser, as declared or with its
        left-out fields written out."""
        self.own()
        for model, frontend in frontends():
            for form, declared in (('declared', frontend), ('defaults', parser_frontend(frontend))):
                with self.subTest(model=model, form=form):
                    self.frontend.write_text(json.dumps(declared))
                    markup, record = load_markup(self.args(model_root=str(self.root)), SERVED)
                    self.assertIs(type(markup), Markup)
                    self.assertEqual(markup.grammar, parser_section(frontend)['tool_grammar'])
                    self.assertEqual(record['tool_grammar'], markup.grammar)
                    self.assertIsNone(record['reparser_sha256'])
        # Every library grammar is a declared frontend's, or the typed one
        # this file declares (`TypedGrammar`).
        self.assertEqual({agentic_mixed.library_grammar(frontend) for _, frontend in frontends()}
                         | {agentic_mixed.library_grammar(TYPED_FRONTEND)}, set(TOOL_GRAMMARS))

    def test_a_format_with_neither_is_refused(self):
        """Naming both remedies: a library grammar or agentic_markup.py."""
        _, frontend = next(frontends())
        parser = parser_frontend(frontend)
        parser['format']['parser']['tool_grammar'] = 'synthetic-grammar-v1'
        for declared in ({'models': [{'id': SERVED}], 'format': {'lines': LINES}}, parser):
            self.frontend.write_text(json.dumps(declared))
            for args in (self.args(model_root=str(self.root)), self.args()):
                with self.subTest(format=list(declared['format']), root=args.model_root), \
                        self.assertRaisesRegex(ValueError, 'no library tool grammar') as refused:
                    load_markup(args, SERVED)
                for remedy in ('format.parser.tool_grammar', 'agentic_markup.py', *TOOL_GRAMMARS):
                    self.assertIn(remedy, str(refused.exception))
                if args.model_root is None:
                    self.assertIn('--model-root', str(refused.exception))
                else:
                    self.assertIn(str(self.root.resolve()), str(refused.exception))

    def test_a_linked_re_parser_is_refused(self):
        outside = self.tree / 'agentic_markup.py'
        outside.write_text(OWN_MARKUP)
        (self.root / 'agentic_markup.py').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'is a symlink'):
            load_markup(self.args(model_root=str(self.root)), SERVED)


class FunctionParameterGrammar(unittest.TestCase):
    """xml-function-parameters-v1: the served parser runs a function's
    arguments to its first function close, so a tool close inside a value
    (here inside a nested JSON string) is content."""

    def test_a_tool_close_inside_a_value_is_content(self):
        for model, markup in markups():
            if markup.grammar != 'xml-function-parameters-v1':
                continue
            with self.subTest(model=model):
                value = json.dumps({'pattern': '</tool_call>'})
                after = (f'\n\n<tool_call>\n<function=read>\n<parameter=filter>\n{value}\n</parameter>\n'
                         '</function>\n</tool_call>')
                answer = {'reasoning': 'Plan', 'content': '', 'tool_calls': [
                    {'function': {'name': 'read', 'arguments': json.dumps({'filter': json.loads(value)})}}]}
                self.assertEqual(calls_class(markup, 'Plan', True, after, answer, TOOLS), RAW_CALLS_AGREE)
                # A region the function close never ends is still malformed.
                self.assertIsNone(raw_calls(markup, '<tool_call>\n<function=read>\n</tool_call>', TOOLS))

    def function_parameter_markups(self):
        return [(model, markup) for model, markup in markups() if markup.grammar == 'xml-function-parameters-v1']

    def test_closing_markers_inside_json_strings_are_content(self):
        """The W1-QUAL reproductions, and escaped quotes, a tool close and
        arrays with a raw string between JSON values: a value whose schema
        rules out a string is JSON, as the served parser reads it."""
        tools = [{'type': 'function', 'function': {'name': 'read', 'parameters': {'type': 'object', 'properties': {
            'filter': {'type': 'object'}, 'list': {'type': 'array'}, 'text': {'type': 'string'}}}}}]
        mixed = [('filter', '{"m": "a\\"</parameter>\\\\", "n": ["</function>\\n</tool_call>"]}'), ('text', 'plain'),
                 ('list', '["<parameter=y>\\n", {"k": "\\n</parameter>\\n"}]')]
        cases = [[('filter', json.dumps({'pattern': inner}))] for inner in ('</parameter>', '</function>', '<parameter=x>')]
        for model, markup in self.function_parameter_markups():
            for parameters in cases + [mixed]:
                with self.subTest(model=model, parameters=parameters):
                    # Each value in the framing the model's markers declare.
                    framed = [(key, f'{markup.value_lead}{value}{markup.value_trail}') for key, value in parameters]
                    after = '\n\n<tool_call>\n<function=read>\n' + ''.join(
                        f'<parameter={key}>{value}</parameter>\n' for key, value in framed) + '</function>\n</tool_call>'
                    raw = framed
                    self.assertEqual(raw_calls(markup, after, tools), ('\n\n', [('read', raw)]))
                    arguments = {key: value if key == 'text' else json.loads(value) for key, value in parameters}
                    answer = {'reasoning': 'Plan', 'content': '', 'tool_calls': [
                        {'function': {'name': 'read', 'arguments': json.dumps(arguments)}}]}
                    self.assertEqual(calls_class(markup, 'Plan', True, after, answer, tools), RAW_CALLS_AGREE)

    def test_a_string_alias_keeps_its_value_raw(self):
        """A string type written as an alias (`str`, `text`, `enum`...) or
        an unknown name is raw text to the served parser, as `string` is;
        an alias of another type is JSON."""
        for kind in ('string', 'str', ' STR ', 'text', 'varchar', 'char', 'enum', ['str', 'null'], 'unknown'):
            with self.subTest(kind=kind):
                self.assertTrue(agentic_mixed.admits_string({'type': kind}))
                tools = [{'type': 'function', 'function': {'name': 'f', 'parameters': {
                    'type': 'object', 'properties': {'x': {'type': kind}}}}}]
                self.assertFalse(agentic_mixed.json_valued(tools, 'f', 'x'))
        for kind in ('int', 'float', 'bool', 'dict', 'list', 'object', ['integer', 'null']):
            with self.subTest(kind=kind):
                self.assertFalse(agentic_mixed.admits_string({'type': kind}))

    def test_raw_values_end_at_the_first_close(self):
        """A string value, a value of an undeclared function and a value of
        a tool whose `strict` is false end at the first parameter close, and
        their arguments at the first function close, as before."""
        loose = [{'type': 'function', 'function': {'name': 'read', 'strict': False, 'parameters': TOOLS[0]['function']['parameters']}}]
        value = json.dumps({'pattern': '</parameter>'})
        for model, markup in self.function_parameter_markups():
            for tools in ((), loose):
                with self.subTest(model=model, tools=tools):
                    after = f'<tool_call>\n<function=read>\n<parameter=filter>\n{value}\n</parameter>\n</function>\n</tool_call>'
                    self.assertIsNone(raw_calls(markup, after, tools))
            with self.subTest(model=model):
                after = '<tool_call>\n<function=read>\n<parameter=path>\na "b</parameter>c\n</parameter>\n</function>\n</tool_call>'
                self.assertIsNone(raw_calls(markup, after, TOOLS))
                # The served parser ends these arguments inside the value.
                after = '<tool_call>\n<function=read>\n<parameter=path>\nx</function>y\n</parameter>\n</function>\n</tool_call>'
                self.assertEqual(raw_calls(markup, after, TOOLS), ('', [('read', [('path', '\nx</function>y\n')])]))

    def test_many_closes_inside_one_json_string_parse_in_linear_time(self):
        """Each function close inside the string is checked from where the
        walk to the previous one stopped, so the walk takes a bounded number
        of steps per character (about two here), not one pass per close."""
        value = json.dumps({'x': '</function>' * 20000})
        after = f'<tool_call>\n<function=read>\n<parameter=filter>\n{value}\n</parameter>\n</function>\n</tool_call>'
        for model, markup in self.function_parameter_markups():
            with self.subTest(model=model):
                steps = 0

                def step(*args):
                    nonlocal steps
                    steps += 1
                    if steps > 3 * len(after):
                        raise AssertionError('the function-close walk is not linear')
                    return walk_step(*args)
                with patch('agentic_mixed.walk_step', step):
                    self.assertEqual(raw_calls(markup, after, TOOLS), ('', [('read', [('filter', f'\n{value}\n')])]))
                self.assertGreater(steps, len(value))

    def test_many_parameters_copy_each_value_once(self):
        """The parameter walk reads one body by offsets and copies only the
        values it returns, so many small parameters cost the body once."""
        values = 0
        for model, markup in self.function_parameter_markups():
            for json_key in (lambda key: True, lambda key: False):
                with self.subTest(model=model, json=json_key('')):
                    body = Counted('\n' + '<parameter=value>\n{}\n</parameter>\n' * 4000)
                    Counted.copied = 0
                    parameters = function_parameters(markup, body, json_key)
                    self.assertEqual(parameters, [('value', '\n{}\n')] * 4000)
                    values = sum(len(value) for _, value in parameters)
                    self.assertLessEqual(Counted.copied, values)


class SharedCases(unittest.TestCase):
    """The re-parse classes each completion the served parser's test reads
    (`SHARED_CASES`, crates/runtime-frontend/tests/function_parameter_tool_grammar.rs)
    against the content and calls that parser returns for it."""

    def test_each_case_has_its_recorded_class(self):
        shared = json.loads(SHARED_CASES.read_text())
        parser = shared['parser']
        # Each turn opener is its own boundary token.
        texts = sorted(parser['turn_open']) + [shared['stop']]
        markup = Markup({'format': {'parser': parser}, 'vocabulary': {
            'boundary_tokens': [{'id': index, 'text': text} for index, text in enumerate(texts)],
            'stop_token_ids': [len(texts) - 1]}})
        self.assertEqual(markup.grammar, 'xml-function-parameters-v1')
        classes = set()
        for case in shared['cases']:
            with self.subTest(case=case['name']):
                tools = () if case.get('undeclared') else shared['tools']
                answer = {'reasoning': 'plan', 'content': case['content'], 'tool_calls': [
                    {'function': {'name': name, 'arguments': arguments}} for name, arguments in case['calls']]}
                turn = raw_turn(markup, 'U[q]A[<think>', f"plan</think>{case['text']}{shared['stop']}")
                self.assertEqual(calls_class(markup, *turn, answer, tools), case['qual'])
                classes.add(case['qual'])
        self.assertEqual(classes, {RAW_CALLS_AGREE, 'calls differ', 'malformed call region'})


class TypedGrammar(unittest.TestCase):
    """xml-typed-parameters-v1: one calls block, opened after a blank line,
    of `<invoke name="N">` elements whose parameters' open tags say whether
    the value is raw text (`string="true"`) or JSON (`string="false"`)."""

    @classmethod
    def setUpClass(cls):
        cls.markup = Markup(TYPED_FRONTEND)

    def turn(self, after):
        return raw_turn(self.markup, 'U[q]A[<think>', f'Plan</think>{after}[EOS]')

    def invoke(self, name, *parameters):
        text = f'<invoke name="{name}">\n'
        for key, json_value, value in parameters:
            text += f'<parameter name="{key}" string="{"false" if json_value else "true"}">{value}</parameter>\n'
        return text + '</invoke>\n'

    def test_calls_in_one_block_agree_with_their_parsed_answer(self):
        # A raw value runs to its own close past a parameter open and a
        # function close; a JSON string may spell the parameter close.
        raw = 'a <parameter name="x" string="true"> b </invoke>'
        after = ('Reading.\n\n<calls>\n'
                 + self.invoke('read', ('path', False, raw), ('filter', True, '{"k": "</parameter>"}'))
                 + self.invoke('read', ('offset', True, '5')) + '</calls>')
        answer = {'reasoning': 'Plan', 'content': 'Reading.', 'tool_calls': [
            {'function': {'name': 'read', 'arguments': json.dumps({'path': raw, 'filter': {'k': '</parameter>'}})}},
            {'function': {'name': 'read', 'arguments': json.dumps({'offset': 5})}}]}
        self.assertEqual(calls_class(self.markup, *self.turn(after), answer, TOOLS), RAW_CALLS_AGREE)
        # The open tag types the value, whatever the schema says.
        typed = {**answer, 'tool_calls': [answer['tool_calls'][0],
                                          {'function': {'name': 'read', 'arguments': '{"offset": "5"}'}}]}
        self.assertEqual(calls_class(self.markup, *self.turn(after), typed, TOOLS), 'calls differ')

    def test_a_number_beyond_binary64_stays_text(self):
        after = '\n\n<calls>\n' + self.invoke('read', ('offset', True, '1e400')) + '</calls>'
        text = {'reasoning': 'Plan', 'content': '', 'tool_calls': [
            {'function': {'name': 'read', 'arguments': '{"offset": "1e400"}'}}]}
        self.assertEqual(calls_class(self.markup, *self.turn(after), text, TOOLS), RAW_CALLS_AGREE)

    def test_calls_outside_one_opened_block_are_malformed(self):
        call = self.invoke('read', ('path', False, 'a.py'))
        for after in (f'Done.<calls>\n{call}</calls>',
                      f'Done.{call}',
                      f'\n\n<calls>\n{call}</calls>\n\n<calls>\n{call}</calls>',
                      f'\n\n<calls>\n{call}</calls>trailing',
                      '\n\n<calls>\n<invoke name="read">\n<parameter name="path" string="true">a.py</invoke>\n</calls>',
                      '\n\n<calls>\n<invoke name="read">\n<parameter name="path" string="maybe">a</parameter>\n'
                      '</invoke>\n</calls>'):
            with self.subTest(after=after):
                self.assertIsNone(self.markup.calls(after, TOOLS))
        self.assertEqual(self.markup.calls('Done.', TOOLS), ('Done.', []))

    def test_last_turn_classes(self):
        self.assertEqual(last_turn_class(self.markup, *self.turn('\n\n<calls>\n</calls>')), RAW_MALFORMED_CALL)
        call = '\n\n<calls>\n' + self.invoke('read', ('path', False, 'a.py')) + '</calls>'
        self.assertEqual(last_turn_class(self.markup, *self.turn(call)), 'output after reasoning')
        self.assertEqual(raw_turn(self.markup, 'U[q]A[<think>', 'Plan <invoke name="read"></think>Done.[EOS]'),
                         'call markup')


class KeyValueGrammar(unittest.TestCase):
    """xml-key-value-arguments-v1: `<tool_call>NAME<arg_key>K</arg_key>
    <arg_value>V</arg_value>...</tool_call>` after an assistant header that
    opens the thinking block."""

    @classmethod
    def setUpClass(cls):
        cls.markup = grammar('xml-key-value-arguments-v1')
        cls.opening = '<|user|>Read a.py.<|assistant|><think>'

    def turn(self, text):
        return raw_turn(self.markup, self.opening, text)

    def test_a_call_agrees_with_its_parsed_answer(self):
        text = ('Read it.</think><tool_call>read<arg_key>path</arg_key><arg_value>a.py</arg_value>'
                '<arg_key>offset</arg_key><arg_value>5</arg_value></tool_call><|observation|>')
        answer = {'reasoning': 'Read it.', 'content': '', 'tool_calls': [
            {'function': {'name': 'read', 'arguments': json.dumps({'path': 'a.py', 'offset': 5})}}]}
        self.assertEqual(calls_class(self.markup, *self.turn(text), answer, TOOLS), RAW_CALLS_AGREE)
        wrong = {**answer, 'tool_calls': [{'function': {'name': 'read', 'arguments': '{"path": "b.py", "offset": 5}'}}]}
        self.assertEqual(calls_class(self.markup, *self.turn(text), wrong, TOOLS), 'calls differ')

    def test_last_turn_classes(self):
        self.assertEqual(last_turn_class(self.markup, *self.turn('Still thinking<|user|>')), RAW_REASONING_ONLY)
        headless = '</think><tool_call><arg_key>path</arg_key><arg_value>a.py</arg_value></tool_call><|observation|>'
        self.assertEqual(last_turn_class(self.markup, *self.turn('Plan' + headless)), RAW_MALFORMED_CALL)
        self.assertEqual(last_turn_class(self.markup, *self.turn('Plan</think>Done.<|user|>')), 'output after reasoning')
        self.assertEqual(self.turn('Plan</think>Done.'), 'no end token')
        self.assertEqual(self.turn('Plan</think><|assistant|>Done.<|user|>'), 'turn markup')
        self.assertEqual(raw_turn(self.markup, '<|user|>Read a.py.', 'Plan</think>Done.<|user|>'), 'no assistant header')

    def test_closing_markers_inside_json_strings_are_content(self):
        value = json.dumps({'pattern': '</arg_value></tool_call>'})
        after = f'<tool_call>read<arg_key>filter</arg_key><arg_value>{value}</arg_value></tool_call>'
        self.assertEqual(raw_calls(self.markup, after, TOOLS), ('', [('read', [('filter', value)])]))
        # A raw (string) value ends at its first close.
        raw = '<tool_call>read<arg_key>path</arg_key><arg_value>a</arg_value>b</arg_value></tool_call>'
        self.assertIsNone(raw_calls(self.markup, raw, TOOLS))

    def test_malformed_regions(self):
        for after in ('<tool_call>read<arg_key>path</arg_key>a.py</tool_call>',
                      '<tool_call>read<arg_key>path</arg_key><arg_value>a.py</tool_call>',
                      '<tool_call>read<arg_key>path</arg_key><arg_value>a</arg_value>x</tool_call>',
                      '<tool_call>read</tool_call>trailing<tool_call>read</tool_call>',
                      'text <arg_key> before'):
            with self.subTest(after=after):
                self.assertIsNone(raw_calls(self.markup, after, TOOLS))
        self.assertEqual(raw_calls(self.markup, 'Done.<tool_call>read</tool_call>\n<tool_call>read</tool_call>', TOOLS),
                         ('Done.', [('read', []), ('read', [])]))

    def test_many_parameters_copy_each_value_once(self):
        """The region walk reads the text by offsets and copies only the
        name, keys, values and rest it returns, so many small parameters cost
        the region once."""
        for key, value in (('filter', '{"m": "</arg_value>"}'), ('path', 'a.py')):
            with self.subTest(json=key == 'filter'):
                text = Counted(' read\n' + f'<arg_key> {key} </arg_key>\n<arg_value>{value}</arg_value>\n' * 4000
                               + '</tool_call>after')
                Counted.copied = 0
                region = self.markup.region(text, TOOLS)
                self.assertEqual(region, ('read', [(key, value)] * 4000, 'after'))
                name, parameters, rest = region
                returned = len(name) + sum(len(k) + len(v) for k, v in parameters) + len(rest)
                self.assertLessEqual(Counted.copied, returned)


if __name__ == '__main__':
    unittest.main()

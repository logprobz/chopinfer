"""Write the Python string and JSON semantics a chat template relies on.

A ported chat template writes values the way Jinja does under HF's
`apply_chat_template`: `tojson` is `json.dumps` with HF's arguments
(transformers `chat_template_utils`), `| trim` and `.strip()` are
`str.strip`, `.split(sep)` is `str.split`, and `| capitalize` is
`str.capitalize`. This script records inputs with the outputs this Python
gives for them in `crates/frontend-format/tests/python_semantics.json`,
which the frontend-format toolkit test reproduces row by row.

Every JSON value input is stored as the JSON text Python reads with
`json.loads`, so a number keeps the spelling the Rust reader sees. The file
records which interpreter wrote it and its Python and Unicode versions.

    python3 -m tools.refrun.python_semantics            # rewrite the file
    python3 -m tools.refrun.python_semantics --check    # fail when stale

It also writes the Unicode 15.0 character classes the toolkit pins its
`capitalize` to, as `crates/frontend-format/src/toolkit/unicode15.rs`: the
unassigned code points, and the Cased and Case_Ignorable sets Python's
final-sigma rule reads. `--check`
compares both files. Python's Unicode version must be the pinned one.

Run it with `$CHOPINFER_REFRUN_PYTHON` when that is set, so the rows come
from the reference runner's interpreter; plain python3 otherwise. It needs
only the standard library.
"""

import argparse
import json
import os
import platform
import sys
import unicodedata
from pathlib import Path

SCHEMA = 'chop-python-semantics-v1'
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'crates' / 'frontend-format' / 'tests' / 'python_semantics.json'
TABLES = ROOT / 'crates' / 'frontend-format' / 'src' / 'toolkit' / 'unicode15.rs'
UNICODE = '15.0.0'

# JSON values, as the text a request carries them in.
VALUES = [
    'null', 'true', 'false', '0', '-0', '7', '-12', '18446744073709551617',
    '1.0', '-0.0', '0.5', '1e-05', '1e+16', '1E16', '123.456', '2.5e-7',
    '""', '"plain"', '"it\'s"', '"say \\"hi\\""', '"both \' and \\""',
    '"tab\\tnew\\nline\\r"', '"\\u0000\\u001f\\u007f"', '"back\\\\slash"',
    '"caf\\u00e9 \\u4e2d\\u6587 \\ud83d\\ude00"', '"nbsp\\u00a0soft\\u00adzw\\u200b"',
    '"ls\\u2028ps\\u2029bom\\ufeffpua\\ue000"', '"html <b>&amp;</b> \'q\'"',
    '[]', '{}', '[1, "two", 3.0, null, true, false]',
    '{"b": 1, "a": [1.5, {"z": null}], "c": "x"}',
    '{"name": "lookup", "description": "Find \\u00e9", "parameters": {"type": "object", '
    '"properties": {"city": {"type": "string"}}, "required": ["city"]}, "strict": true}',
    '[[], {}, [[]], {"k": {}}]',
    # Unassigned code points, nested: U+0378, the noncharacter U+FFFF,
    # U+1C89 (assigned only after Unicode 15.0) and the astral U+E0080 and
    # U+10FFFF.
    '["\\u0378", "\\uffff", "\\u1c89", "\\udb40\\udc80", "\\udbff\\udfff"]',
    '{"\\u0378": {"k\\u0379": ["\\u0380"]}}',
    # One of each category Cc, Cf, the code points around the surrogates
    # and private use (Cs cannot occur), Co, Zl, Zp and Zs.
    '["\\u0007\\u0085", "\\u200e\\u061c\\u00ad", "\\ud7a3\\ud7ff\\uf900", '
    '"\\ue000\\uf8ff\\udb80\\udc00", "\\u2028", "\\u2029", "\\u00a0\\u2007\\u3000 "]',
]

# Every value is written with HF's default arguments (the templates call
# `tojson` and `tojson(ensure_ascii=False)`, which are the same) and with
# `ensure_ascii`; the containers and one non-ASCII string with the other
# forms HF's filter accepts.
FULL_ARGS = [{}, {'ensure_ascii': True}]
STRUCTURED = [text for text in VALUES if text[0] in '[{'] + ['"caf\\u00e9 \\u4e2d\\u6587 \\ud83d\\ude00"']
TOJSON_ARGS = [
    {'ensure_ascii': False},
    {'indent': 2},
    {'indent': 0},
    {'indent': '\t'},
    {'separators': [',', ':']},
    {'separators': [', ', ': '], 'indent': 2},
    {'sort_keys': True},
    {'ensure_ascii': True, 'indent': 4, 'sort_keys': True},
]

# Float spellings, by the repr Python gives the double each one reads as,
# which json.dumps writes.
FLOATS = [
    '1.0', '-1.0', '0.0', '-0.0', '1e-05', '1e-5', '0.0001', '0.00001', '1e-07',
    '1e+16', '1e16', '1e15', '9999999999999998.0', '1e22', '1.5e300',
    '1.7976931348623157e308', '5e-324', '2.2250738585072014e-308', '0.1',
    '0.30000000000000004', '123456789.125', '-2.5e-10', '3.0e2', '1.23000',
]

# Every character str.isspace() accepts, in code point order.
WHITESPACE = ''.join(map(chr, [
    0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680,
    *range(0x2000, 0x200b), 0x2028, 0x2029, 0x202f, 0x205f, 0x3000]))
STRIPS = [
    '', '   ', 'plain', '  padded  ', '\n\nlines\n', '\t tab \t',
    WHITESPACE + 'unicode' + WHITESPACE, '\x1cseparators\x1f', '\u200bzero width\u200b',
    '\ufeffbom\ufeff', 'inner  space', '\r\n', 'x\u3000',
]

SPLITS = [
    ('a,b,,c', ','), ('', ','), (',', ','), ('no separator', ','),
    ('<think>plan</think>answer', '</think>'), ('a</think>b</think>c', '</think>'),
    ('</think>', '</think>'), ('aaa', 'aa'), ('x\ny\n', '\n'), ('caf\u00e9\u00e9', '\u00e9'),
]

# Titlecase differs from uppercase for the first character of several of
# these, and a final sigma lowercases by the characters before it.
CAPITALIZES = [
    '', 'low', 'high', 'max', 'LOW', 'mIxEd case', '1st', ' lead', '\u00e9t\u00e9',
    '\u00df', '\u01c6emal', '\ufb01ne', '\u10d0\u10d1', '\u1f80x',
    '\u1fb3', '\u1fb2', '\u1fb7', '\u0587', '\u0130stanbul',
    'O\u03a3', '\u03a3\u03a3 \u03a3a', '\u0149', 'xx\u0130', '\u019bx', '\u0264',
    # Characters Unicode 15.0 does not assign have no case mapping, and they
    # are neither cased nor case-ignorable around a final sigma.
    '\u1c89x', 'x\u1c89', 'x\ua7cb\ua7dc', 'a\u03a3\u1c89', '\u0378\u03a3',
    # The case-ignorable characters a final sigma looks past.
    "a\u03a3'b", "a'\u03a3", 'a\u03a3.', 'a\u0345\u03a3', 'a\u03a3\u0345',
]


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    """HF's `tojson` filter (transformers `chat_template_utils`)."""
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                      separators=separators, sort_keys=sort_keys)


def document(interpreter):
    loads = json.loads
    return {
        'schema': SCHEMA,
        'description': (
            'Python outputs a ported chat template relies on, written by '
            'tools/refrun/python_semantics.py and reproduced by the frontend-format '
            'toolkit test. JSON value inputs are the text json.loads reads.'
        ),
        'interpreter': interpreter,
        'python': platform.python_version(),
        'unicode': unicodedata.unidata_version,
        'tojson': [
            {'input': text, 'args': args,
             'output': tojson(loads(text), **{k: tuple(v) if k == 'separators' else v
                                              for k, v in args.items()})}
            for args, texts in [*((args, VALUES) for args in FULL_ARGS),
                                *((args, STRUCTURED) for args in TOJSON_ARGS)]
            for text in texts
        ],
        'float_repr': [{'input': text, 'output': repr(loads(text))} for text in FLOATS],
        'strip': [{'input': text, 'strip': text.strip(), 'rstrip': text.rstrip()}
                  for text in STRIPS],
        'split': [{'input': text, 'separator': separator, 'output': text.split(separator)}
                  for text, separator in SPLITS],
        'capitalize': [{'input': text, 'output': text.capitalize()} for text in CAPITALIZES],
    }


def classes():
    """The pinned Unicode classes, each as inclusive code point ranges."""
    def cased(ch):
        return ch.islower() or ch.isupper() or ch.istitle()

    def case_ignorable(ch):
        # A final sigma looks past case-ignorable characters: one after the
        # sigma leaves it final unless a cased character follows.
        if cased(ch):
            return ('A\u03a3' + ch).lower()[1] == '\u03c2'
        return ('A\u03a3' + ch + 'B').lower()[1] == '\u03c3'

    def scalar(code):
        return not 0xd800 <= code <= 0xdfff

    tests = {
        'UNASSIGNED_V1': lambda ch: unicodedata.category(ch) == 'Cn',
        'CASED_V1': cased,
        'CASE_IGNORABLE_V1': case_ignorable,
    }
    tables = {}
    for name, test in tests.items():
        ranges, start = [], None
        for code in range(0x110000):
            member = scalar(code) and test(chr(code))
            if member and start is None:
                start = code
            if not member and start is not None:
                ranges.append((start, code - 1))
                start = None
        if start is not None:
            ranges.append((start, 0x10ffff))
        tables[name] = ranges
    return tables


def render_tables():
    """The classes as the toolkit's Rust tables."""
    lines = [
        f'//! Unicode {UNICODE} character classes, as Python {platform.python_version()} reads them.',
        '//! Written by `tools/refrun/python_semantics.py`; do not edit.',
        '//!',
        '//! Each table lists inclusive, ascending code point ranges. Surrogates are',
        '//! in none: they cannot occur in a Rust string.',
        '',
    ]
    docs = {
        'UNASSIGNED_V1': 'The code points Unicode 15.0 does not assign (category Cn).',
        'CASED_V1': 'The cased characters: Lowercase, Uppercase or titlecase.',
        'CASE_IGNORABLE_V1': 'The case-ignorable characters a final sigma looks past.',
    }
    for name, ranges in classes().items():
        lines.append(f'/// {docs[name]}')
        lines.append('#[rustfmt::skip]')
        lines.append(f'pub(super) const {name}: [(u32, u32); {len(ranges)}] = [')
        for index in range(0, len(ranges), 4):
            lines.append('    ' + ' '.join(f'(0x{a:x}, 0x{b:x}),' for a, b in ranges[index:index + 4]))
        lines.append('];')
        lines.append('')
    return '\n'.join(lines[:-1]) + '\n'


def render(interpreter):
    """The document as ASCII JSON with one row per line."""
    lines = ['{']
    items = list(document(interpreter).items())
    for index, (key, value) in enumerate(items):
        comma = ',' if index + 1 < len(items) else ''
        if isinstance(value, list):
            rows = [' ' + json.dumps(row, ensure_ascii=True) for row in value]
            lines.append(f' {json.dumps(key)}: [')
            lines.append(',\n'.join(rows))
            lines.append(f' ]{comma}')
        else:
            lines.append(f' {json.dumps(key)}: {json.dumps(value, ensure_ascii=True)}{comma}')
    lines.append('}')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true',
                        help='fail when the file differs from what this Python writes')
    args = parser.parse_args(argv)
    if unicodedata.unidata_version != UNICODE:
        raise SystemExit(f'this Python reads Unicode {unicodedata.unidata_version}; the '
                         f'toolkit pins Unicode {UNICODE} (Python 3.12)')
    reference = os.environ.get('CHOPINFER_REFRUN_PYTHON')
    interpreter = ('CHOPINFER_REFRUN_PYTHON'
                   if reference and Path(reference).resolve() == Path(sys.executable).resolve()
                   else 'python3')
    text = render(interpreter)
    if args.check:
        current = args.out.read_text(encoding='utf-8')
        # The interpreter name is provenance; the rows must match.
        recorded = json.loads(current)
        expected = json.loads(text)
        recorded.pop('interpreter', None)
        expected.pop('interpreter', None)
        if recorded != expected:
            raise SystemExit(f'{args.out} is stale: rerun tools/refrun/python_semantics.py')
        if TABLES.read_text(encoding='utf-8') != render_tables():
            raise SystemExit(f'{TABLES} is stale: rerun tools/refrun/python_semantics.py')
        return
    args.out.write_text(text, encoding='utf-8')
    TABLES.write_text(render_tables(), encoding='utf-8')


if __name__ == '__main__':
    main()

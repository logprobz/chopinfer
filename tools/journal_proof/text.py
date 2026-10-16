"""Text journals: pinned journals a test builds without a fake device.

No device recorded their lines, so there is no history, create or handle
mapping to compare. A text journal is IDENTICAL when its base and head
bodies are equal byte for byte. Otherwise it is PROVEN only under a declared
line-edit policy, `text_edits` in the spec's journal policy:

    {"remove": {"line": N, "text": "..."}}   one-based base line N, exact text
    {"add": {"line": M, "text": "..."}}      one-based head line M, exact text

The base without its removed lines must equal the head without its added
lines; every entry applies exactly once and nothing else may differ.
`specs/text_edits.py` writes such a policy from two bodies for a reviewer to
read; the proof never derives one.
"""
from common import need
from engine import Unproven, require

EDIT_KINDS = ('remove', 'add')


def validate(edits):
    """The shape of a `text_edits` list, before any body is read."""
    need(isinstance(edits, list), 'text_edits must be a list')
    seen = set()
    for entry in edits:
        need(isinstance(entry, dict) and len(entry) == 1 and next(iter(entry)) in EDIT_KINDS,
             'a text edit is one {"remove": ...} or {"add": ...} object')
        kind, edit = next(iter(entry.items()))
        need(isinstance(edit, dict) and set(edit) == {'line', 'text'} and type(edit['line']) is int
             and edit['line'] > 0 and isinstance(edit['text'], str), f'invalid text {kind} edit')
        need((kind, edit['line']) not in seen, f'text {kind} edit repeats line {edit["line"]}')
        seen.add((kind, edit['line']))


def compare(base, head, edits):
    """The verdict of one text journal: (verdict, counts). Raises Unproven
    when the bodies differ beyond the declared edits."""
    validate(edits)
    if not edits:
        if base != head:
            first = next(i for i in range(max(len(base), len(head))) if base[i:i + 1] != head[i:i + 1])
            raise Unproven(f'text journal body differs at line {first + 1} and declares no text_edits')
        return 'IDENTICAL', {}
    removed = {e['remove']['line']: e['remove']['text'] for e in edits if 'remove' in e}
    added = {e['add']['line']: e['add']['text'] for e in edits if 'add' in e}
    for side, lines, declared in (('base', base, removed), ('head', head, added)):
        for line, text in declared.items():
            require(line <= len(lines) and lines[line - 1] == text, f'{side} line {line} is not the declared text')
    kept_base = [line for i, line in enumerate(base, 1) if i not in removed]
    kept_head = [line for i, line in enumerate(head, 1) if i not in added]
    if kept_base != kept_head:
        first = next(i for i in range(max(len(kept_base), len(kept_head)))
                     if kept_base[i:i + 1] != kept_head[i:i + 1])
        raise Unproven(f'text journal differs beyond its declared edits at kept line {first + 1}')
    return 'PROVEN', {'text_removed_lines': len(removed), 'text_added_lines': len(added)}

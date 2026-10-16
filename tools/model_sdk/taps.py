"""Model tap declarations (`models/<m>/taps.json`) and the tap names of
engine dumps.

A declaration lists a model's engine taps in program order: each entry's
`name` is the tap's name in the shared grammar (`<scope>.<site>`, see
tools/refrun/taps.py), `{nn}` a two-digit layer and `layers` (optional) the
layers it applies to; the first matching entry applies. `v1` is the tap's
name in the model's older dump schema (`v1_schema`), which readers translate
through `canonical`.
"""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'chopinfer.taps.v1'


def pattern(name):
    """A compiled `{nn}` name pattern."""
    return re.compile(r'(?P<nn>\d\d)'.join(map(re.escape, name.split('{nn}'))))


class Declaration:
    def __init__(self, path):
        self.path = Path(path)
        value = json.loads(self.path.read_text())
        self.model, self.v1_schema = value['model'], value.get('v1_schema')
        self.entries = [(pattern(entry['name']), pattern(entry['v1']) if 'v1' in entry else None, entry)
                        for entry in value['taps']]

    @staticmethod
    def applies(match, entry):
        if match is None:
            return False
        nn = match.groupdict().get('nn')
        return 'layers' not in entry or nn is not None and int(nn) in entry['layers']

    def entry(self, name):
        """The entry of a tap name, or None."""
        for name_pattern, _, entry in self.entries:
            if self.applies(name_pattern.fullmatch(name), entry):
                return entry
        return None

    def canonical(self, v1):
        """(name, entry) of an older dump's tap name, or None when it is not
        declared (a tap of an older build)."""
        for _, v1_pattern, entry in self.entries:
            match = v1_pattern.fullmatch(v1) if v1_pattern else None
            if self.applies(match, entry):
                return entry['name'].replace('{nn}', match.groupdict().get('nn') or ''), entry
        return None


def declarations(root=ROOT):
    """The declarations of the models under `root`."""
    return [Declaration(path) for path in sorted(Path(root).glob('models/*/taps.json'))]


def model_declaration(model, root=ROOT):
    """The declaration of `model`, or None."""
    return next((d for d in declarations(root) if model is not None and d.model == model), None)


def v1_declaration(schema, root=ROOT):
    """The declaration whose older dump schema is `schema`, or None."""
    return next((d for d in declarations(root) if schema is not None and d.v1_schema == schema), None)

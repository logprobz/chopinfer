"""A qualification module's top-level constants, read from its source
without importing it. module_metadata.py derives tools/qual/modules.json
from them; checks.py reads a check module's RUNNER_CHECK_ARM. Kept apart
from module_metadata.py, whose listing of every module here
ci/check_inputs.py counts as an input of whatever imports it."""

import ast
from pathlib import Path

TOOLS = Path(__file__).parent


def module_constants(module):
    """A qualification module's top-level names, each with its literal default
    or None when computed. Read from source; the module is not imported."""
    constants = {}
    for node in ast.parse((TOOLS / (module + ".py")).read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            if isinstance(target, ast.Name):
                try:
                    constants[target.id] = (ast.literal_eval(node.value),)
                except ValueError:
                    constants[target.id] = None
    return constants

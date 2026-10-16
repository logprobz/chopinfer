"""Write tools/qual/modules.json: each qualification module a check may name
and the type its constant overrides must have, read from the module sources.
`chopinfer eval declaration`, the one implementation of the declaration
rules, reads this file;
test_module_metadata.py keeps it current.

    python3 -B tools/qual/module_metadata.py
"""

import ast
import json
from pathlib import Path
import sys

TOOLS = Path(__file__).parent

SCHEMA = "chopinfer-qual-modules/v1"
OUTPUT = Path(__file__).with_name("modules.json")


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


def check_module(name):
    """Whether `name` is a qualification module a check may name: a module
    of this directory, not a test."""
    return name.isidentifier() and not name.startswith("test_") and (TOOLS / (name + ".py")).is_file()


def kind(default):
    """The JSON type an override of a constant with `default` must have
    (crates/runtime-eval declaration.rs): a computed default takes a number."""
    if default is None:
        return "number"
    value, = default
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple)):
        return "list"
    if isinstance(value, dict):
        return "dict"
    if value is None:
        return "null"
    # No JSON value has this type, so no override is valid.
    return type(value).__name__


def metadata():
    """Every module a check may name, with its top-level names' kinds."""
    modules = sorted(path.stem for path in TOOLS.glob("*.py") if check_module(path.stem))
    return {"schema": SCHEMA, "modules": {module: {name: kind(default) for name, default in
                                                   sorted(module_constants(module).items())}
                                          for module in modules}}


def encoded():
    return json.dumps(metadata(), indent=1, sort_keys=True) + "\n"


if __name__ == "__main__":
    OUTPUT.write_text(encoded())
    print(f"wrote {OUTPUT}", file=sys.stderr)

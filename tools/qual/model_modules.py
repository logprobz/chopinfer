"""Qualification modules a model provides, read from its model root.

A model root is the model's own directory: `models/<m>` in this tree, or the
tree `--model-root` or CHOPINFER_QUAL_MODEL_ROOT (release_gate.py) names for
a model outside it. A module NAME is the file `NAME.py` directly in that
root, such as `agentic_markup.py` (agentic_mixed.py) and `diagnostic.py`
(oracle_diagnostics.py). It must be a regular file there: a symlink is
refused, so the module cannot come from outside the root. Each load runs the module as a new module object rather than an import
by name, so a module of the same name on sys.path neither replaces it nor is
replaced. It is registered in sys.modules under its own private name
(`_chopinfer_model_module_<NAME>_<id>`) for its lifetime, as code such as
dataclasses expects of a defining module. It may import the modules of
tools/qual, which are on sys.path.
"""
import hashlib
import os
from pathlib import Path
import stat
import sys
import types
import uuid

# The largest module source read.
MAX_BYTES = 1 << 20


def model_root(explicit=None):
    """The model root an outside model names: EXPLICIT (`--model-root`),
    else CHOPINFER_QUAL_MODEL_ROOT; None when neither is set."""
    value = explicit or os.environ.get("CHOPINFER_QUAL_MODEL_ROOT")
    return Path(value) if value else None


def source(name, root):
    """(path, bytes) of module NAME at model root ROOT, or None when ROOT
    holds no `NAME.py`."""
    if not (isinstance(name, str) and name.isidentifier()):
        raise ValueError(f"invalid model module name {name!r}")
    root = Path(root).resolve()
    path = root / f"{name}.py"
    try:
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except (FileNotFoundError, NotADirectoryError) as error:
        raise ValueError(f"model root {root} is not a directory") from error
    try:
        try:
            info = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"{path} is a symlink: a model module must be a regular file in its model root")
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{path} is not a regular file")
        file = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=directory)
        with os.fdopen(file, "rb") as stream:
            # The file the name led to when it was opened.
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError(f"{path} is not a regular file")
            data = stream.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError(f"{path} exceeds {MAX_BYTES} bytes")
    finally:
        os.close(directory)
    return path, data


def load(name, root):
    """(module, SHA-256 of the source it ran) of module NAME of the model at
    model root ROOT (`source`), run as a new module object, or None when
    ROOT holds no `NAME.py`. The digest is of the bytes compiled, read once."""
    found = source(name, root)
    if found is None:
        return None
    path, data = found
    module = types.ModuleType(f"_chopinfer_model_module_{name}_{uuid.uuid4().hex}")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    try:
        exec(compile(data, str(path), "exec"), module.__dict__)
    except BaseException:
        del sys.modules[module.__name__]
        raise
    return module, hashlib.sha256(data).hexdigest()

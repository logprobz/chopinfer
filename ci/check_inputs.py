"""The repository files a Python or shell check reads, derived from its command.

A check's inputs start from its argv: script paths, `python -m` modules, the
directory of `unittest discover -s DIR` and paths or globs inside `bash -c`
text. They close transitively over the Python and shell files the check
runs or imports: the modules a scanned file imports (resolved against its
directory, the root and the directories it names; dotted module names in
string literals count, as for importlib.import_module) and the Python and
shell files it names in a string literal, glob or shell token. Every other
file, directory or glob a scanned file names is a data input. A named
directory covers the data files directly in it, or below it when the file
walks trees (rglob, os.walk or a `**` pattern); its Python and shell files
are covered where they are imported or named. A file that loads a module from
a computed path (spec_from_file_location, import_module, runpy and the like
with no literal module) is taken to load any Python file in the directories
it names; those files are covered but not scanned in turn. A discovered test directory covers everything below it.

The closure is a bounded literal scan, like change_scope's Rust scan: a path
built at run time from names it does not contain is not found, so ci/cpu.sh
remains the full gate."""
import ast
import os
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCANNED = (".py", ".sh")
# The check definitions name every check's script. A check that imports them,
# or runs the gate on fixtures, is not each of those checks.
DEFINITIONS = ("ci/cpu.py",)
# Calls that load a module, with the position of the argument naming it.
LOADERS = {"spec_from_file_location": 1, "import_module": 0, "run_path": 0, "run_module": 0, "__import__": 0,
           "SourceFileLoader": 1}
MODULE_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
MAX_BYTES = 2_000_000
SHELL_TOKEN = re.compile(r"[A-Za-z0-9_.*/${}-]+")
SHELL_ROOT = re.compile(r"^(?:\$\{?[A-Za-z_]+\}?/)+")


def glob_expression(pattern):
    """A shell glob as a regular expression: `*` stays within one path part, `**` crosses parts."""
    parts = re.split(r"(\*\*/?|\*|\?)", pattern)
    special = {"**/": "(?:.*/)?", "**": ".*", "*": "[^/]*", "?": "[^/]"}
    return re.compile("".join(special.get(part, re.escape(part)) for part in parts))


def unittest_names(argv, index):
    """Whether argv[index] is a test name after `-m unittest` (not discover's or an option)."""
    if "unittest" not in argv[:index] or "discover" in argv or argv[index].startswith("-"):
        return False
    start = argv.index("unittest")
    return start > 0 and argv[start - 1] == "-m" and index > start and not argv[index].endswith(".py")


class Inputs:
    """What one check covers: files, directories (their own files, or whole trees) and globs."""

    def __init__(self):
        self.files, self.directories, self.trees, self.globs = set(), set(), set(), set()
        # Directories whose data files are covered at any depth.
        self.data_trees = set()
        # Python modules a scanned file imports.
        self.modules = set()
        # Modules it may load from a computed path: covered, not scanned.
        self.loaded = set()

    def covers(self, path):
        # A named directory is often an import path or a working directory; imports and named
        # scripts already reach the code used from it.
        data = not path.endswith(SCANNED)
        return (path in self.files or data and pathlib.PurePosixPath(path).parent.as_posix() in self.directories
                or data and any(path.startswith(tree + "/") for tree in self.data_trees)
                or any(path.startswith(tree + "/") for tree in self.trees)
                or any(glob_expression(pattern).fullmatch(path) for pattern in self.globs))


class Scanner:
    def __init__(self, root=ROOT, files=None):
        self.root = pathlib.Path(root)
        if files is None:
            listed = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                                             cwd=self.root)
            files = [name.decode() for name in listed.split(b"\0") if name]
        self.tracked = set(files)
        self.directories = {parent.as_posix() for name in self.tracked
                            for parent in pathlib.PurePosixPath(name).parents if parent.as_posix() != "."}
        self.cache, self.globbed = {}, {}

    def resolve(self, value, bases, found):
        """Add the repository paths `value` names, relative to any base, to `found`."""
        value = value.strip()
        if len(value) < 2 or len(value) > 200 or any(char in value for char in " \n{}%") or value.startswith("-"):
            return
        # A pattern names its directory: "*" or "*/x" alone would cover the tree.
        if "*" in value and ("/" not in value.strip("/") or "*" in value.strip("/").split("/")[0]):
            return
        for base in bases:
            path = os.path.normpath(os.path.join(base, value.lstrip("/")))
            if path in (".", "") or path.startswith(".."):
                continue
            if "*" in path:
                matched = self.matches(path)
                if matched:
                    found.globs.add(path)
                    found.files.update(name for name in matched if name.endswith(SCANNED))
            elif path in self.tracked:
                found.files.add(path)
            elif path in self.directories:
                found.directories.add(path)

    def matches(self, pattern):
        if pattern not in self.globbed:
            expression = glob_expression(pattern)
            self.globbed[pattern] = [name for name in self.tracked if expression.fullmatch(name)]
        return self.globbed[pattern]

    def module(self, name, bases, whole=False):
        """Repository files of the dotted module `name`, and of its package parents
        unless `whole`."""
        found = set()
        parts = name.split(".")
        for base in bases:
            for count in range(len(parts) if whole else 1, len(parts) + 1):
                stem = os.path.normpath(os.path.join(base, *parts[:count]))
                for candidate in (stem + ".py", stem + "/__init__.py"):
                    if candidate in self.tracked:
                        found.add(candidate)
        return found

    def read(self, path):
        full = self.root / path
        if full.is_symlink() or not full.is_file() or full.stat().st_size > MAX_BYTES:
            return None
        return full.read_text(errors="replace")

    def references(self, path):
        """The Inputs one Python or shell file names directly."""
        if path in self.cache:
            return self.cache[path]
        found = self.cache[path] = Inputs()
        text = self.read(path)
        if text is None:
            return found
        bases = [pathlib.PurePosixPath(path).parent.as_posix(), "."]
        if path.endswith(".sh"):
            self.shell(text, bases, found)
        elif path.endswith(".py"):
            try:
                tree = ast.parse(text)
            except SyntaxError:
                return found
            strings, imports, loads = [], [], []
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    strings.append(node.value)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(node)
                elif isinstance(node, ast.Call):
                    name = getattr(node.func, "attr", getattr(node.func, "id", None))
                    if name in LOADERS and len(node.args) > LOADERS[name]:
                        loads.append(node.args[LOADERS[name]])
            # A path is often joined from parts (ROOT / "tools" / "qual" / "x.py"), so names also
            # resolve against the directories the file names; those may be on its import path too.
            for _ in range(3):
                known = set(found.directories)
                for value in strings:
                    self.resolve(value, bases, found)
                if found.directories == known:
                    break
                bases = bases[:2] + sorted(found.directories)
            for value in strings:
                if MODULE_NAME.fullmatch(value):
                    found.modules.update(self.module(value, bases, whole=True))
            if re.search(r"\b(?:os\.walk|rglob)\(", text) or any("**" in value for value in strings):
                found.data_trees |= found.directories
            # A load from a computed path may reach any module in the directories the file names.
            for argument in loads:
                named = Inputs()
                for node in ast.walk(argument):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        self.resolve(node.value, bases, named)
                        if MODULE_NAME.fullmatch(node.value):
                            named.files.update(self.module(node.value, bases, whole=True))
                if not any(name.endswith(".py") for name in named.files):
                    found.loaded.update(name for name in self.tracked if name.endswith(".py")
                                        and pathlib.PurePosixPath(name).parent.as_posix() in found.directories)
                    break
            for node in imports:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        found.modules.update(self.module(alias.name, bases))
                else:
                    scope = [pathlib.PurePosixPath(path).parents[node.level - 1].as_posix()] if node.level else bases
                    prefix = node.module or ""
                    for alias in node.names:
                        found.modules.update(self.module(prefix + "." + alias.name if prefix else alias.name, scope))
                    if prefix:
                        found.modules.update(self.module(prefix, scope))
        found.files.discard(path)
        found.modules.discard(path)
        return found

    def shell(self, text, bases, found):
        for token in SHELL_TOKEN.findall(text):
            token = SHELL_ROOT.sub("", token)
            if "$" not in token and "{" not in token and ("/" in token or "." in token):
                self.resolve(token, bases, found)

    def inputs(self, argv, cwd="."):
        """The Inputs of a check started as `argv` from `cwd` (relative to the root)."""
        result = Inputs()
        bases = [cwd]
        named = Inputs()
        for index, argument in enumerate(argv):
            previous = argv[index - 1] if index else None
            if previous == "-m" and argument != "unittest" or unittest_names(argv, index):
                named.files.update(self.module(argument, bases))
            elif previous == "-s" and "discover" in argv:
                tree = os.path.normpath(os.path.join(cwd, argument))
                result.trees.add(tree)
                named.files.update(name for name in self.tracked if name.startswith(tree + "/"))
            else:
                self.shell(argument, bases, named)
        result.files |= named.files
        result.directories |= named.directories
        result.globs |= named.globs
        pending = list(named.files)
        while pending:
            path = pending.pop()
            if path in result.modules:
                continue
            result.files.add(path)
            if not path.endswith(SCANNED):
                continue
            result.modules.add(path)
            found = self.references(path)
            result.files |= found.files | found.loaded
            result.directories |= found.directories
            result.data_trees |= found.data_trees
            result.globs |= found.globs
            pending += found.modules
            if path not in DEFINITIONS:
                pending += [script for script in found.files if script.endswith(SCANNED)]
        return result

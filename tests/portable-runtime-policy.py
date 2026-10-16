#!/usr/bin/env python3
"""Verify the portable-crate dependency and ahead-of-time route policies.

Two policies are checked.

The portable-crate policy says the portable crates must not depend on CUDA or
NCCL. It is checked against every manifest key and value outside the prose
fields, the resolved lockfile closure, and the import and route surface of the
crate sources.

The ahead-of-time policy says production code carries one owned route with no
runtime compilation, runtime autotuning, backend fallback, or dynamic kernel
loading. It is checked against every workspace manifest and every Rust and
native source file under the crates tree.

Both policies read declarations rather than raw text. A forbidden term is
allowed in exactly one shape: a payload-free enum variant declared in the
workspace, and plain path references to that variant. That shape records that
a route is rejected or that an external identity is declared; it cannot invoke
anything. Every other shape - a dependency, an import, a type, a call, a
variant carrying a payload, or a reference through an unknown owner - fails.
Comments and string literals are removed before the source scan, so prose
about an absent route is not a dependency on it, while `link` attributes are
inspected because their literals do name a native library.

Any file the scanner cannot read, tokenize, or balance is a scan failure, not
a pass.
"""

import argparse
import bisect
import dataclasses
import functools
import pathlib
import re
import sys
import tomllib

PORTABLE_CRATES = (
    "bundle-recurrent-fixture",
    "bundle-tp4-fixture",
    "encoder-spec",
    "projection-spec",
    "model-compiler-sdk",
    "runtime-engine",
    "runtime-grammar",
    "runtime-kernel-contract",
    "runtime-kv",
    "runtime-media",
    "runtime-model-api",
    "runtime-sampling",
    "runtime-scheduler",
    "runtime-tokenizer",
    "runtime-weights",
)

ACCELERATOR_TERMS = ("cudarc", "cuda-sys", "nccl")

AHEAD_OF_TIME_TERMS = (
    "NVRTC",
    "RuntimeJit",
    "RuntimeAutotune",
    "BackendFallback",
    "DynamicKernelLoader",
)

RUST_SUFFIXES = (".rs",)
NATIVE_SUFFIXES = (".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp")

# Manifest fields that describe the crate to a reader rather than binding it to
# anything. A term named here is prose, so it is exempt; every other manifest
# key and string value is in scope, including tables this script does not model.
PROSE_FIELDS = frozenset(
    (
        "authors",
        "categories",
        "description",
        "documentation",
        "homepage",
        "keywords",
        "license",
        "license-file",
        "readme",
        "repository",
    )
)

LINK_ATTRIBUTES = ("link", "link_name")

PORTABLE_POLICY = "portable-crate-dependency"
AHEAD_OF_TIME_POLICY = "ahead-of-time-route"


class PolicyError(Exception):
    """The scanner could not evaluate the tree, so no verdict is available."""


@dataclasses.dataclass(frozen=True)
class Policy:
    portable_crates: tuple[str, ...] = PORTABLE_CRATES
    accelerator_terms: tuple[str, ...] = ACCELERATOR_TERMS
    ahead_of_time_terms: tuple[str, ...] = AHEAD_OF_TIME_TERMS
    crates_directory: str = "crates"
    workspace_manifest: str = "Cargo.toml"
    lockfile: str = "Cargo.lock"


@dataclasses.dataclass(frozen=True)
class Violation:
    policy: str
    path: str
    line: int
    symbol: str
    term: str
    detail: str


def render(violation: Violation) -> str:
    location = violation.path
    if violation.line:
        location = f"{location}:{violation.line}"
    return (
        f"{location}: {violation.policy}: {violation.symbol}"
        f" matches the forbidden term {violation.term}: {violation.detail}"
    )


@functools.lru_cache(maxsize=65_536)
def normalize(text: str) -> str:
    return re.sub(r"[-_]", "", text).lower()


def matched_term(text: str, terms) -> str | None:
    candidate = normalize(text)
    for term in terms:
        if normalize(term) in candidate:
            return term
    return None


_TOKEN_RE = re.compile(
    r"""
      (?P<space>\s+)
    | (?P<line_comment>//[^\n]*)
    | (?P<block_open>/\*)
    | (?P<raw_string>(?:b|c|rb|br)?r\#*")
    | (?P<string>(?:b|c)?")
    | (?P<quote>')
    | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<number>[0-9][0-9A-Za-z_]*)
    | (?P<punct>::|->|=>|[^\s])
    """,
    re.VERBOSE,
)

_OPENERS = {"{": "}", "(": ")", "[": "]"}
_CLOSERS = {"}": "{", ")": "(", "]": "["}

_ENUM_SENTINEL = object()


@dataclasses.dataclass
class Token:
    kind: str
    value: str
    start: int


class Unit:
    """A tokenized source file with its declaration structure resolved."""

    def __init__(self, path: pathlib.Path, relative: str, dialect: str, text: str):
        self.path = path
        self.relative = relative
        self.dialect = dialect
        self.text = text
        self.newlines = [index for index, char in enumerate(text) if char == "\n"]
        self.tokens = tokenize(text, relative, dialect)
        self.attribute_indices: set[int] = set()
        self.attribute_spans: list[tuple[int, int]] = []
        self.variant_declarations: dict[int, tuple[str | None, bool]] = {}
        self.self_names: list[str | None] = [None] * len(self.tokens)
        self.unit_variants: dict[str, set[str]] = {}
        if dialect == "rust":
            self._collect_attributes()
        self._analyze()

    def line_of(self, offset: int) -> int:
        return bisect.bisect_right(self.newlines, offset) + 1

    def _collect_attributes(self) -> None:
        tokens = self.tokens
        count = len(tokens)
        index = 0
        while index < count:
            token = tokens[index]
            if token.kind == "punct" and token.value == "#":
                cursor = index + 1
                if cursor < count and tokens[cursor].kind == "punct" and tokens[cursor].value == "!":
                    cursor += 1
                if cursor < count and tokens[cursor].kind == "punct" and tokens[cursor].value == "[":
                    depth = 0
                    scan = cursor
                    while scan < count:
                        candidate = tokens[scan]
                        if candidate.kind == "punct" and candidate.value == "[":
                            depth += 1
                        elif candidate.kind == "punct" and candidate.value == "]":
                            depth -= 1
                            if depth == 0:
                                break
                        scan += 1
                    if scan >= count:
                        raise PolicyError(
                            f"{self.relative}:{self.line_of(token.start)}:"
                            " attribute brackets are unterminated"
                        )
                    self.attribute_spans.append((index, scan))
                    self.attribute_indices.update(range(index, scan + 1))
                    index = scan + 1
                    continue
            index += 1

    def _resolve_impl_type(self, index: int) -> str | None:
        tokens = self.tokens
        count = len(tokens)
        cursor = index + 1
        depth = 0
        name = None
        while cursor < count:
            if cursor in self.attribute_indices:
                cursor += 1
                continue
            token = tokens[cursor]
            if token.kind == "punct":
                if token.value == "<":
                    depth += 1
                elif token.value == ">":
                    depth = max(0, depth - 1)
                elif depth == 0 and token.value in ("{", ";"):
                    break
                elif token.value in _OPENERS and depth == 0:
                    depth += 1
                elif token.value in _CLOSERS and depth > 0:
                    depth -= 1
            elif token.kind == "ident" and depth == 0:
                if token.value == "where":
                    break
                if token.value not in ("for", "dyn", "impl", "mut", "const"):
                    name = token.value
            cursor += 1
        return name

    def _analyze(self) -> None:
        tokens = self.tokens
        count = len(tokens)
        stack: list[dict] = []
        pending: tuple[str, str | None] | None = None
        index = 0
        while index < count:
            if index in self.attribute_indices:
                index += 1
                continue
            token = tokens[index]
            top = stack[-1] if stack else None
            current_self = top["self"] if top else None
            self.self_names[index] = current_self

            if token.kind == "ident":
                if top is not None and top["enum"] is not _ENUM_SENTINEL and top["variant"]:
                    following = tokens[index + 1] if index + 1 < count else None
                    is_unit = following is None or (
                        following.kind == "punct" and following.value in (",", "}", "=")
                    )
                    self.variant_declarations[index] = (top["enum"], is_unit)
                    if is_unit and top["enum"]:
                        self.unit_variants.setdefault(top["enum"], set()).add(token.value)
                    top["variant"] = False
                    index += 1
                    continue
                if token.value == "enum":
                    following = index + 1
                    while following < count and following in self.attribute_indices:
                        following += 1
                    name = (
                        tokens[following].value
                        if following < count and tokens[following].kind == "ident"
                        else None
                    )
                    pending = ("enum", name)
                elif token.value == "impl" and self.dialect == "rust":
                    pending = ("impl", self._resolve_impl_type(index))
                index += 1
                continue

            if token.kind == "punct":
                if token.value in _OPENERS:
                    entry = {
                        "open": token.value,
                        "self": current_self,
                        "enum": _ENUM_SENTINEL,
                        "variant": False,
                    }
                    if token.value == "{" and pending is not None:
                        kind, name = pending
                        if name:
                            entry["self"] = name
                        if kind == "enum":
                            entry["enum"] = name
                            entry["variant"] = True
                    if token.value == "{":
                        pending = None
                    stack.append(entry)
                elif token.value in _CLOSERS:
                    if not stack or stack[-1]["open"] != _CLOSERS[token.value]:
                        raise PolicyError(
                            f"{self.relative}:{self.line_of(token.start)}:"
                            f" unbalanced '{token.value}'"
                        )
                    stack.pop()
                elif token.value == ",":
                    if top is not None and top["enum"] is not _ENUM_SENTINEL:
                        top["variant"] = True
                elif token.value == ";":
                    pending = None
            index += 1
        if stack:
            raise PolicyError(f"{self.relative}: delimiters are unbalanced at end of file")


def tokenize(text: str, relative: str, dialect: str) -> list[Token]:
    tokens: list[Token] = []
    position = 0
    length = len(text)
    while position < length:
        match = _TOKEN_RE.match(text, position)
        if match is None:
            raise PolicyError(f"{relative}: unrecognized input at offset {position}")
        kind = match.lastgroup
        start = match.start()
        position = match.end()
        if kind == "space":
            continue
        if kind == "line_comment":
            continue
        if kind == "block_open":
            position = _skip_block_comment(text, position, relative, start, dialect)
            continue
        if kind == "raw_string":
            hashes = match.group().count("#")
            terminator = '"' + "#" * hashes
            end = text.find(terminator, position)
            if end < 0:
                raise PolicyError(f"{relative}: raw string literal is unterminated")
            position = end + len(terminator)
            tokens.append(Token("literal", text[start:position], start))
            continue
        if kind == "string":
            position = _skip_quoted(text, position, '"', relative)
            tokens.append(Token("literal", text[start:position], start))
            continue
        if kind == "quote":
            position, quoted_kind = _skip_quote_or_lifetime(text, position, relative, dialect)
            tokens.append(Token(quoted_kind, text[start:position], start))
            continue
        tokens.append(Token(kind, match.group(), start))
    return tokens


def _skip_block_comment(text: str, position: int, relative: str, start: int, dialect: str) -> int:
    depth = 1
    cursor = position
    while depth:
        close = text.find("*/", cursor)
        if close < 0:
            raise PolicyError(f"{relative}: block comment is unterminated")
        if dialect == "rust":
            opening = text.find("/*", cursor)
            if 0 <= opening < close:
                depth += 1
                cursor = opening + 2
                continue
        depth -= 1
        cursor = close + 2
    return cursor


def _skip_quoted(text: str, position: int, closer: str, relative: str) -> int:
    cursor = position
    length = len(text)
    while cursor < length:
        char = text[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == closer:
            return cursor + 1
        cursor += 1
    raise PolicyError(f"{relative}: {closer} literal is unterminated")


def _skip_quote_or_lifetime(text: str, position: int, relative: str, dialect: str):
    length = len(text)
    if position >= length:
        raise PolicyError(f"{relative}: quote is unterminated")
    char = text[position]
    if char == "\\":
        return _skip_quoted(text, position, "'", relative), "literal"
    if position + 1 < length and text[position + 1] == "'":
        return position + 2, "literal"
    if dialect == "rust" and (char.isalpha() or char == "_"):
        cursor = position
        while cursor < length and (text[cursor].isalnum() or text[cursor] == "_"):
            cursor += 1
        return cursor, "lifetime"
    return _skip_quoted(text, position, "'", relative), "literal"


def _read_text(path: pathlib.Path, relative: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise PolicyError(f"{relative}: cannot be read: {error}") from error
    except UnicodeDecodeError as error:
        raise PolicyError(f"{relative}: is not valid UTF-8: {error}") from error


def _load_manifest(path: pathlib.Path, relative: str) -> dict:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise PolicyError(f"{relative}: cannot be read: {error}") from error
    try:
        return tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise PolicyError(f"{relative}: is not parseable TOML: {error}") from error


def _manifest_symbols(node, trail: tuple[str, ...] = ()):
    """Yield (label, text) for every manifest key and string value in scope.

    Every table is walked, not a fixed list of dependency tables, so a name
    that binds the crate to something cannot hide in a table this script does
    not model. Prose fields are the only exemption.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            if key in PROSE_FIELDS:
                continue
            label = ".".join(trail + (key,))
            yield label, key
            yield from _manifest_symbols(value, trail + (key,))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _manifest_symbols(item, trail + (str(index),))
    elif isinstance(node, str):
        yield ".".join(trail) or "(root)", node


def _scan_manifest(path: pathlib.Path, relative: str, terms, policy_name):
    manifest = _load_manifest(path, relative)
    violations = []
    for label, name in _manifest_symbols(manifest):
        term = matched_term(name, terms)
        if term is not None:
            violations.append(
                Violation(
                    policy=policy_name,
                    path=relative,
                    line=0,
                    symbol=name,
                    term=term,
                    detail=f"declared by {label}",
                )
            )
    return violations, manifest


def _is_declaration_reference(unit: Unit, index: int, terms, unit_variants) -> bool:
    tokens = unit.tokens
    if index < 2:
        return False
    separator = tokens[index - 1]
    owner = tokens[index - 2]
    if separator.kind != "punct" or separator.value != "::":
        return False
    if owner.kind != "ident":
        return False
    name = owner.value
    if name == "Self":
        name = unit.self_names[index - 2]
        if not name:
            return False
    elif matched_term(name, terms) is not None:
        return False
    if index + 1 < len(tokens):
        following = tokens[index + 1]
        if following.kind == "punct" and following.value in ("(", "{", "!", "::"):
            return False
    return tokens[index].value in unit_variants.get(name, ())


def _scan_unit(unit: Unit, terms, policy_name, unit_variants):
    violations = []
    for index, token in enumerate(unit.tokens):
        if token.kind != "ident":
            continue
        term = matched_term(token.value, terms)
        if term is None:
            continue
        declaration = unit.variant_declarations.get(index)
        if declaration is not None:
            enum_name, is_unit = declaration
            if is_unit:
                continue
            detail = f"enum {enum_name or '(anonymous)'} variant carries a payload"
        elif _is_declaration_reference(unit, index, terms, unit_variants):
            continue
        else:
            detail = "is not a payload-free declared enum variant or a reference to one"
        violations.append(
            Violation(
                policy=policy_name,
                path=unit.relative,
                line=unit.line_of(token.start),
                symbol=token.value,
                term=term,
                detail=detail,
            )
        )
    for start, end in unit.attribute_spans:
        head = None
        for index in range(start, end + 1):
            if unit.tokens[index].kind == "ident":
                head = unit.tokens[index].value
                break
        if head not in LINK_ATTRIBUTES:
            continue
        for index in range(start, end + 1):
            token = unit.tokens[index]
            if token.kind != "literal":
                continue
            term = matched_term(token.value, terms)
            if term is not None:
                violations.append(
                    Violation(
                        policy=policy_name,
                        path=unit.relative,
                        line=unit.line_of(token.start),
                        symbol=token.value,
                        term=term,
                        detail=f"named by a {head} attribute",
                    )
                )
    return violations


def _collect_units(root: pathlib.Path, crates: pathlib.Path) -> dict[str, Unit]:
    units: dict[str, Unit] = {}
    for path in sorted(crates.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix
        if suffix in RUST_SUFFIXES:
            dialect = "rust"
        elif suffix in NATIVE_SUFFIXES:
            dialect = "native"
        else:
            continue
        relative = path.relative_to(root).as_posix()
        units[relative] = Unit(path, relative, dialect, _read_text(path, relative))
    return units


def _lock_graph(path: pathlib.Path, relative: str) -> dict[str, list[str]]:
    manifest = _load_manifest(path, relative)
    packages = manifest.get("package")
    if not isinstance(packages, list):
        raise PolicyError(f"{relative}: has no package array")
    graph: dict[str, list[str]] = {}
    for entry in packages:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise PolicyError(f"{relative}: holds a package entry without a name")
        dependencies = entry.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise PolicyError(f"{relative}: {entry['name']} dependencies is not an array")
        graph.setdefault(entry["name"], []).extend(
            dependency.split(" ", 1)[0]
            for dependency in dependencies
            if isinstance(dependency, str)
        )
    return graph


def _lock_closure(graph: dict[str, list[str]], name: str) -> set[str]:
    seen: set[str] = set()
    stack = [name]
    while stack:
        current = stack.pop()
        for dependency in graph.get(current, ()):
            if dependency not in seen:
                seen.add(dependency)
                stack.append(dependency)
    return seen


def verify(root, policy: Policy | None = None) -> list[Violation]:
    policy = policy or Policy()
    root = pathlib.Path(root)
    if not root.is_dir():
        raise PolicyError(f"{root}: is not a directory")
    crates = root / policy.crates_directory
    if not crates.is_dir():
        raise PolicyError(f"{policy.crates_directory}: is not a directory under {root}")

    if not (root / policy.workspace_manifest).is_file():
        raise PolicyError(f"{policy.workspace_manifest}: is missing")

    violations: list[Violation] = []
    source_roots = [crates]
    workspace = tomllib.loads(_read_text(root / policy.workspace_manifest, policy.workspace_manifest))
    members = workspace.get("workspace", {}).get("members", [])
    models = root / "models"
    if models.is_dir():
        source_roots.append(models)
    elif any(member.startswith("models/") for member in members):
        raise PolicyError("models: declared workspace sources are missing")
    units = {}
    for source_root in source_roots:
        collected = _collect_units(root, source_root)
        if not collected:
            raise PolicyError(f"{source_root}: no source files scanned")
        units.update(collected)
        print(f"portable source coverage: {source_root.relative_to(root)}: {len(collected)} files", file=sys.stderr)
    unit_variants: dict[str, set[str]] = {}
    for unit in units.values():
        for enum_name, variants in unit.unit_variants.items():
            unit_variants.setdefault(enum_name, set()).update(variants)

    manifest_paths = [policy.workspace_manifest]
    manifest_paths.extend(
        sorted(path.relative_to(root).as_posix() for directory in source_roots for path in directory.rglob("Cargo.toml"))
    )
    for relative in manifest_paths:
        found, _ = _scan_manifest(
            root / relative, relative, policy.ahead_of_time_terms, AHEAD_OF_TIME_POLICY
        )
        violations.extend(found)

    # The workspace manifest carries the inherited dependency table, so a
    # portable crate that writes `workspace = true` inherits whatever it names.
    found, _ = _scan_manifest(
        root / policy.workspace_manifest,
        policy.workspace_manifest,
        policy.accelerator_terms,
        PORTABLE_POLICY,
    )
    violations.extend(found)

    portable_packages = []
    for crate in policy.portable_crates:
        directory = crates / crate
        manifest_relative = f"{policy.crates_directory}/{crate}/Cargo.toml"
        if not (root / manifest_relative).is_file():
            raise PolicyError(f"{manifest_relative}: is missing")
        sources = directory / "src"
        if not sources.is_dir():
            raise PolicyError(f"{policy.crates_directory}/{crate}/src: is not a directory")
        found, manifest = _scan_manifest(
            root / manifest_relative,
            manifest_relative,
            policy.accelerator_terms,
            PORTABLE_POLICY,
        )
        violations.extend(found)
        package = manifest.get("package", {})
        name = package.get("name") if isinstance(package, dict) else None
        if not isinstance(name, str):
            raise PolicyError(f"{manifest_relative}: declares no package name")
        portable_packages.append(name)
        prefix = f"{policy.crates_directory}/{crate}/src/"
        for relative, unit in units.items():
            if relative.startswith(prefix):
                violations.extend(
                    _scan_unit(unit, policy.accelerator_terms, PORTABLE_POLICY, unit_variants)
                )

    for unit in units.values():
        violations.extend(
            _scan_unit(unit, policy.ahead_of_time_terms, AHEAD_OF_TIME_POLICY, unit_variants)
        )

    lockfile = root / policy.lockfile
    if not lockfile.is_file():
        raise PolicyError(f"{policy.lockfile}: is missing")
    graph = _lock_graph(lockfile, policy.lockfile)
    for name in portable_packages:
        if name not in graph:
            raise PolicyError(f"{policy.lockfile}: does not resolve {name}")
        for dependency in sorted(_lock_closure(graph, name)):
            term = matched_term(dependency, policy.accelerator_terms)
            if term is not None:
                violations.append(
                    Violation(
                        policy=PORTABLE_POLICY,
                        path=policy.lockfile,
                        line=0,
                        symbol=dependency,
                        term=term,
                        detail=f"is in the resolved dependency closure of {name}",
                    )
                )
    for dependency in sorted(graph):
        term = matched_term(dependency, policy.ahead_of_time_terms)
        if term is not None:
            violations.append(
                Violation(
                    policy=AHEAD_OF_TIME_POLICY,
                    path=policy.lockfile,
                    line=0,
                    symbol=dependency,
                    term=term,
                    detail="is a resolved workspace package",
                )
            )

    violations.sort(key=lambda item: (item.path, item.line, item.symbol, item.policy))
    return violations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        default=None,
        help="workspace root to verify (defaults to the chopinfer tree holding this script)",
    )
    arguments = parser.parse_args(argv)
    root = (
        pathlib.Path(arguments.root)
        if arguments.root
        else pathlib.Path(__file__).resolve().parent.parent
    )
    try:
        violations = verify(root)
    except PolicyError as error:
        print(f"policy scan failed: {error}", file=sys.stderr)
        return 2
    if violations:
        for violation in violations:
            print(render(violation), file=sys.stderr)
        print(f"{len(violations)} policy violations", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

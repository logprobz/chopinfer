"""Conservative package reachability, with reasons kept separate from test coverage.

Cargo's all-feature graph finds potential consumers. Following a dev edge
transitively deliberately over-selects quick checks; it does not prove that a
particular test executable links that dependency.
"""
from collections import deque
from fnmatch import fnmatchcase
import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = "ci/source_dependencies.toml"
# Unowned tooling that a selected Python or shell check covers needs no Rust
# packages, unless a Rust source or a declared edge names it.
TOOLING = ("ci/", "docs/", "tests/", "tools/")
# Root files Cargo reads; every other root file counts as tooling.
CARGO_ROOT_FILES = ("Cargo.toml", "Cargo.lock", "rust-toolchain", "rust-toolchain.toml")


def is_tooling(path):
    """Whether `path` is under a tooling directory or is a root file Cargo does not read."""
    return path.startswith(TOOLING) or ("/" not in path and path not in CARGO_ROOT_FILES)


def declarations(root=ROOT):
    value = tomllib.loads((pathlib.Path(root) / REGISTRY).read_text())
    if value.get("schema") != "chopinfer-source-dependencies-v1":
        raise ValueError("unsupported source dependency schema")
    for edge in value["edges"]:
        if not all(edge.get(k) for k in ("source", "consumers", "kind", "reader")):
            raise ValueError("incomplete source dependency")
    return value["edges"]


def relative(path, root):
    try:
        return pathlib.Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def code_mask(source):
    return re.sub(r'r(?P<hash>#{0,8})".*?"(?P=hash)|"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/',
                  lambda match: " " * len(match[0]), source, flags=re.DOTALL)


def comment_mask(source):
    return re.sub(r'r(?P<hash>#{0,8})".*?"(?P=hash)|"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/',
                  lambda match: match[0] if match[0][0] in 'r"' else " " * len(match[0]), source, flags=re.DOTALL)


def include_arguments(source):
    """Read balanced macro arguments, including multiline concat!/env! forms."""
    masked = code_mask(source)
    for match in re.finditer(r'include(?:_str|_bytes)?!\s*\(', masked):
        start = match.end()
        depth, quoted, escaped = 1, False, False
        for index in range(start, min(len(source), start + 4096)):
            char = source[index]
            if quoted:
                if char == '"' and not escaped:
                    quoted = False
                escaped = char == "\\" and not escaped
            elif char == '"':
                quoted = True
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    yield source[start:index]
                    break
        else:
            yield "<unresolved macro>"


class Graph:
    def __init__(self, metadata, edges=None, scan=True, old_sources=None):
        self.root = pathlib.Path(metadata.get("workspace_root", ROOT))
        members = set(metadata.get("workspace_members", []))
        self.packages = {p["id"]: p for p in metadata.get("packages", []) if p.get("id") in members}
        self.by_name = {p["name"]: key for key, p in self.packages.items()}
        self.directories = {key: relative(pathlib.Path(p["manifest_path"]).parent, self.root)
                            for key, p in self.packages.items()}
        self.edges = edges if edges is not None else declarations()
        self.reverse = {key: [] for key in members}
        self.issues = []
        self.reads = {}
        # Repository files and directories Rust code names in a string literal.
        self.literals = {}
        if len(self.by_name) != len(self.packages):
            self.issues.append("ambiguous Cargo package names; use --workspace")
        nodes = (metadata.get("resolve") or {}).get("nodes", [])
        if (not members or members != self.packages.keys()
                or not members <= {n["id"] for n in nodes}
                or None in self.directories.values()):
            self.issues.append("incomplete Cargo workspace/resolve graph; use --workspace")
        for node in nodes:
            if node["id"] not in members:
                continue
            deps = node.get("deps")
            if deps is None:
                deps = [{"pkg": dep, "dep_kinds": [{"kind": "unknown"}]} for dep in node["dependencies"]]
            for dep in deps:
                if dep["pkg"] not in members:
                    continue
                for kind in dep.get("dep_kinds", [{"kind": "unknown"}]):
                    label = kind.get("kind") or "normal"
                    self.reverse[dep["pkg"]].append((node["id"], "Cargo " + label,
                                                    kind.get("target") or "all targets"))
        if scan and self.root.is_dir():
            self.scan_sources(old_sources or {})

    def owners(self, path):
        matches = {key: directory for key, directory in self.directories.items()
                   if directory is not None and (path == directory or path.startswith(directory + "/"))}
        depth = max((len(d) for d in matches.values()), default=-1)
        return sorted(key for key, directory in matches.items() if len(directory) == depth)

    def scan_sources(self, old_sources):
        # A bounded literal scan supplements the registry. Build scripts are never executed.
        sources = {}
        for directory in self.directories.values():
            if directory is None:
                continue
            for path in sorted((self.root / directory).rglob("*.rs")):
                if path.is_symlink() or path.stat().st_size > 2_000_000 or len(sources) >= 8000:
                    self.issues.append("source scan bound or symlink: " + str(path.relative_to(self.root)))
                    continue
                sources[path.relative_to(self.root).as_posix()] = path.read_text()
        for name, source in old_sources.items():
            sources[name] = sources.get(name, "") + "\n" + source
        # Root files count as tooling, and Rust may name one without a slash.
        root_files = {path.name for path in self.root.iterdir() if path.is_file()}
        for name, source in sources.items():
            masked = code_mask(source)
            patterns = [r'include(?:_str|_bytes)?!\s*\(\s*"([^"\n]+)"', r'#\[path\s*=\s*"([^"\n]+)"\]']
            for pattern in patterns:
                for match in re.finditer(pattern, source):
                    if masked[match.start()] == " ":
                        continue
                    value = match[1]
                    path = relative(self.root / pathlib.Path(name).parent / value, self.root)
                    if path:
                        self.reads.setdefault(path, []).append((name, "literal include"))
                    else:
                        self.issues.append("include outside workspace: " + name)
            owners = self.owners(name)
            manifest_root = self.directories[owners[0]] if owners else ""
            # Simple manifest-relative file joins and concatenated includes.
            for value in re.findall(r'(?:root|manifest_dir|env!\("CARGO_MANIFEST_DIR"\)\))\.join\("([^"\n]+)"\)', source):
                path = relative(self.root / manifest_root / value, self.root)
                if path and (self.root / path).is_file():
                    self.reads.setdefault(path, []).append((name, "manifest-relative input"))
            for value in re.findall(r'concat!\(\s*env!\("CARGO_MANIFEST_DIR"\),\s*"([^"\n]+)"', source):
                path = relative(self.root / manifest_root / value.lstrip("/"), self.root)
                if path:
                    self.reads.setdefault(path, []).append((name, "manifest-relative include"))
            # Generated CubinSpec headers rooted at the repository are literal inputs.
            for value in re.findall(r'#include \\"([^"\n]+)\\"', source):
                if (self.root / value).is_file():
                    self.reads.setdefault(value, []).append((name, "generated CUDA header"))
            # Paths read at run time, such as tooling a test starts, appear as literals:
            # relative to the root, the source file or the manifest.
            strings = comment_mask(source)
            for value in re.findall(r'"([A-Za-z0-9_./-]*/[A-Za-z0-9_./-]*)"', strings):
                for base in ("", pathlib.Path(name).parent.as_posix(), manifest_root):
                    path = relative(self.root / base / value.lstrip("/"), self.root)
                    if path and path != "." and (self.root / path).exists():
                        self.literals.setdefault(path, []).append(name)
            for value in re.findall(r'"([A-Za-z0-9_.-]+)"', strings):
                if value in root_files:
                    self.literals.setdefault(value, []).append(name)
            build = sources.get(manifest_root + "/build.rs", "")
            for expression in include_arguments(source):
                generated_env = re.fullmatch(r'\s*env!\("([A-Z_]+)"\)\s*', expression)
                generated = generated_env and ("rustc-env=" + generated_env[1]) in build
                if not expression.lstrip().startswith('"') and 'OUT_DIR' not in expression and 'CARGO_MANIFEST_DIR' not in expression and not generated:
                    self.issues.append("unresolved include expression: " + name)

    def literal_readers(self, path):
        """Rust sources that name `path`, or a directory holding it, in a string literal."""
        return sorted({name for literal, names in self.literals.items()
                       if path == literal or path.startswith(literal + "/") for name in names})

    def unread(self, path):
        """Whether `path` is tooling no Rust source names and no declared edge reads."""
        return (is_tooling(path) and path not in self.reads and not self.literal_readers(path)
                and not any(fnmatchcase(path, edge["source"]) for edge in self.edges))

    def select(self, paths, broad=False, *, checked=(), kernel_index=None, kernel_changes=()):
        """`checked` names changed paths that a selected non-Rust check covers."""
        reasons = {}
        fallback = []
        paths = sorted(set(paths))
        if self.issues and paths:
            fallback.extend(self.issues)
        for path in paths:
            if ("/tests/support/" in path or "/tests/common/" in path):
                fallback.append("shared test/source fallback: " + path)
            owners = self.owners(path)
            # Unowned tooling that Rust sources name is theirs to test: the
            # sources' packages, then their dependents.
            literal = self.literal_readers(path) if not owners and is_tooling(path) else []
            if not owners and not literal and not (path in checked and self.unread(path)):
                fallback.append("unowned or removed input: " + path)
            queue = deque((key, [{"from": path, "to": self.packages[key]["name"],
                                  "kind": "owner"}]) for key in owners)
            # Walk file reads transitively before traversing Cargo dependents;
            # a literal holder counts as a reader, so whatever includes it follows.
            files = deque([(path, [])])
            for reader in literal:
                trail = [{"from": path, "to": reader, "kind": "runtime literal"}]
                files.append((reader, trail))
                for key in self.owners(reader):
                    queue.append((key, trail + [{"from": reader, "to": self.packages[key]["name"],
                                                 "kind": "test/source owner"}]))
            visited = set()
            while files:
                source, chain = files.popleft()
                if source in visited:
                    continue
                visited.add(source)
                for reader, kind in sorted(self.reads.get(source, [])):
                    trail = chain + [{"from": source, "to": reader, "kind": kind}]
                    files.append((reader, trail))
                    for key in self.owners(reader):
                        queue.append((key, trail + [{"from": reader, "to": self.packages[key]["name"], "kind": "test/source owner"}]))
                for edge in self.edges:
                    if fnmatchcase(source, edge["source"]):
                        for name in edge["consumers"]:
                            consumers = sorted(p for p in self.by_name if fnmatchcase(p, name))
                            if not consumers:
                                fallback.append("missing declared source consumer " + name + " for " + source)
                            for consumer in consumers:
                                queue.append((self.by_name[consumer], chain + [{"from": source, "to": consumer,
                                              "kind": edge["kind"], "reader": edge["reader"]}]))
            seen = set()
            while queue:
                key, chain = queue.popleft()
                if key in seen:
                    continue
                seen.add(key)
                name = self.packages[key]["name"]
                reasons.setdefault(name, []).append({"input": path, "chain": chain,
                    "conservative": any(e["kind"] in ("Cargo dev", "Cargo unknown", "contract reference") for e in chain)})
                for consumer, kind, target in sorted(self.reverse[key]):
                    queue.append((consumer, chain + [{"from": name, "to": self.packages[consumer]["name"],
                                                      "kind": kind, "target": target}]))
        if broad or fallback:
            for name in self.by_name:
                reasons.setdefault(name, []).append({"input": "*", "chain": [{"kind": "conservative fallback",
                    "reason": "--all" if broad else "; ".join(sorted(set(fallback)))}], "conservative": True})
        selection = {"packages": sorted(reasons), "reasons": dict(sorted(reasons.items())),
                "fallback": sorted(set(fallback)), "workspace": bool(self.issues),
                "completeness": {"cargo": not self.issues, "source_scan": "bounded, not an execution proof",
                                 "cuda_membership": "not inspected; report mode accepts producer membership"}}
        if kernel_index is not None:
            return kernel_index.select(self, paths, selection, kernel_changes, broad)
        return selection

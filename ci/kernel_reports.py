"""Bounded, read-only consumption of kernel-build's producer contract.

Membership is build consistency evidence, never numerical or deployment
qualification. No compiler, audit tool, build script or header scanner runs here.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

SCHEMA = "chopinfer-kernel-build-v1"
FILENAMES = {"kernel-build-report.json", "kernel-build-report.profile.json"}
FLAGS = ("source_membership_complete", "source_ownership_complete", "entry_declarations_complete")
RECIPE_FIELDS = ("toolchain_version", "nvcc_sha256", "flags", "defines", "headers",
                 "include_dirs", "generated_sha256", "source_closure_sha256")
# Directory-form locks keep one gating file per source family.
FAMILY_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


@dataclass(frozen=True)
class Limits:
    # Separate from qualification receipts' 8 MB limit. Real lab reports exceed
    # 11 MB and nvcc exceeds 33 MB. Large binaries are hashed in streaming reads.
    report_bytes: int = 32 * 1024 * 1024
    source_bytes: int = 16 * 1024 * 1024
    artifact_bytes: int = 512 * 1024 * 1024
    total_bytes: int = 4 * 1024 * 1024 * 1024
    closure_bytes: int = 8 * 1024 * 1024 * 1024
    cache_bytes: int = 256 * 1024 * 1024
    reports: int = 256
    files: int = 16384
    records: int = 2000000
    entries: int = 200000
    depth: int = 24


def require(condition, message):
    if not condition:
        raise ValueError(message)


def string(value):
    require(isinstance(value, str) and bool(value) and "\0" not in value, "expected nonempty string")
    return value


def digest(value):
    return hashlib.sha256(value).hexdigest()


def hash_string(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "invalid SHA256")
    return value


def serialized(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON number: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def source_key(identity):
    return identity["kind"], identity["owner"], identity["path"]


def identity_object(identity):
    return {key: identity[key] for key in ("kind", "owner", "path")}


def file_id(file):
    return digest(serialized([identity_object(file["identity"]), file["sha256"]]))


def recipe_bytes(recipe):
    ordered = {key: recipe[key] for key in RECIPE_FIELDS}
    ordered["generated_sha256"] = dict(sorted(ordered["generated_sha256"].items()))
    return serialized(ordered)


def family_record(identity):
    """A directory-form lock record: the identity without its source provenance."""
    return {key: value for key, value in identity.items() if key != "source_closure_sha256"}


def lexical(path):
    """Normalize context roots without following report-supplied symlinks."""
    return Path(os.path.abspath(string(str(path))))


def signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class Files:
    def __init__(self, limits):
        self.limits = limits
        self.aliases = {}
        self.cache = {}
        self.cached_bytes = self.total_bytes = self.closure_bytes = 0

    def location(self, path):
        path = lexical(path)
        for alias in sorted(self.aliases, key=lambda p: (-len(p.parts), str(p))):
            if path.is_relative_to(alias):
                return self.aliases[alias] / path.relative_to(alias)
        return path

    def open(self, path, directory=False):
        path = self.location(path)
        descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        try:
            for index, part in enumerate(path.parts[1:]):
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                if index < len(path.parts) - 2 or directory:
                    flags |= os.O_DIRECTORY
                following = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = following
            info = os.fstat(descriptor)
            require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
                    "expected regular " + ("directory" if directory else "file") + ": " + str(path))
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def read(self, path, bound, *, contents=False):
        # Producer lookup files are canonical; context roots are normalized
        # separately. Do not erase a symlink/.. component before no-follow IO.
        require(".." not in Path(path).parts, "parent traversal in lookup path: " + str(path))
        path = self.location(path)
        descriptor = self.open(path)
        with os.fdopen(descriptor, "rb") as stream:
            before = signature(os.fstat(stream.fileno()))
            require(before[2] <= bound, "file byte bound: " + str(path))
            cached = self.cache.get(path)
            if cached:
                require(before == cached[0], "file changed during inspection: " + str(path))
                if not contents or cached[2] is not None:
                    return cached[1], cached[2], before[2]
            require(len(self.cache) < self.limits.files, "aggregate file count bound")
            sha, chunks, size = hashlib.sha256(), [], 0
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                self.total_bytes += len(chunk)
                require(size <= bound and self.total_bytes <= self.limits.total_bytes, "aggregate/file read byte bound")
                sha.update(chunk)
                if contents:
                    chunks.append(chunk)
            require(before == signature(os.fstat(stream.fileno())) and size == before[2],
                    "file changed during inspection: " + str(path))
            data = b"".join(chunks) if contents else None
            kept = data if data is not None and self.cached_bytes + size <= self.limits.cache_bytes else None
            if kept is not None and (not cached or cached[2] is None):
                self.cached_bytes += size
            self.cache[path] = before, sha.hexdigest(), kept
            return sha.hexdigest(), data, size

    def json(self, path, bound=None):
        return parse(self.read(path, bound or self.limits.report_bytes, contents=True)[1])


class Reader:
    def __init__(self, limits=None):
        self.limits = limits or Limits()
        self.files = Files(self.limits)
        self.gaps = []
        self.closures = {}
        self.records = self.entries = self.report_count = 0

    def enumerate(self, inputs):
        found = set()
        # Only user-supplied directory aliases may be resolved. Discover all
        # aliases before reading any report so input order cannot change results.
        roots = []
        for value in sorted({str(lexical(p)) for p in inputs}):
            path = Path(value)
            try:
                if path.is_dir():
                    canonical = path.resolve(strict=True)
                    self.files.aliases[path] = canonical
                    roots.append(canonical)
                else:
                    descriptor = self.files.open(path)
                    os.close(descriptor)
                    found.add(path)
            except (OSError, ValueError) as error:
                self.gaps.append(f"report input {path}: {error}")

        def walk(path, depth):
            require(depth <= self.limits.depth, "report traversal depth bound")
            descriptor = self.files.open(path, directory=True)
            try:
                with os.scandir(descriptor) as entries:
                    names = []
                    for entry in entries:
                        self.entries += 1
                        require(self.entries <= self.limits.entries, "report traversal entry bound")
                        names.append((entry.name, entry.is_symlink(), entry.is_dir(follow_symlinks=False)))
                for name, symlink, directory in sorted(names):
                    child = path / name
                    if symlink:
                        self.gaps.append("symlinked report descendant: " + str(child))
                    elif directory:
                        walk(child, depth + 1)
                    elif name in FILENAMES:
                        found.add(child)
                        require(len(found) <= self.limits.reports, "report count bound")
            finally:
                os.close(descriptor)

        for root in sorted(set(roots)):
            before = len(found)
            try:
                walk(root, 0)
                if len(found) == before and not any(p.is_relative_to(root) for p in found):
                    self.gaps.append("empty report directory: " + str(root))
            except (OSError, ValueError) as error:
                self.gaps.append(f"report directory {root}: {error}")
        return sorted({self.files.location(p) for p in found})

    def structure(self, report):
        require(report["schema"] == SCHEMA, "unsupported kernel build schema")
        for key in FLAGS:
            require(type(report[key]) is bool, "invalid completeness flag: " + key)
        require(report["lock_status"] in ("Verified", "Updated", "Profiling", "Skipped"), "invalid lock status")
        string(report["lock_path"])
        require(isinstance(report["compiler_path"], str), "invalid compiler path")
        if report["lock_sha256"] is not None:
            hash_string(report["lock_sha256"])
        context = report["context"]
        if context is not None:
            package = string(context["consumer"]["package"])
            require(re.fullmatch(r"[A-Za-z0-9_-]+@[^@\s/]+", package), "consumer requires package@version")
            string(context["consumer"]["source"])
            string(context["selection"])
            require(isinstance(context["sources"], list), "invalid source roots")
            ids, paths = set(), set()
            for root in context["sources"]:
                name, path = string(root["id"]), self.files.location(root["path"])
                require(name not in ids and path not in paths, "duplicate source root")
                ids.add(name)
                paths.add(path)
        require(isinstance(report["files"], dict) and isinstance(report["cubins"], list), "invalid files/cubins")
        self.records += len(report["files"]) + len(report["cubins"])
        for key, file in report["files"].items():
            identity = file["identity"]
            require(identity["kind"] in ("Project", "Generated", "Toolkit", "System"), "invalid source kind")
            require(identity["owner"] is None or isinstance(identity["owner"], str) and identity["owner"], "invalid source owner")
            path = string(identity["path"])
            require(not PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts
                    and str(PurePosixPath(path)) == path and "\\" not in path, "escaping/noncanonical source identity")
            require(identity["kind"] == "Project" or identity["owner"] is None, "nonproject source owner")
            hash_string(file["sha256"])
            require(type(file["bytes"]) is int and 0 <= file["bytes"] <= self.limits.source_bytes, "source byte bound/type")
            string(file["location"])
            require(hash_string(key) == file_id(file), "file identity digest differs")
        names = set()
        for cubin in report["cubins"]:
            name = string(cubin["name"])
            require(name not in names, "duplicate cubin name")
            names.add(name)
            string(cubin["path"])
            string(cubin["identity_path"])
            recipe, identity = cubin["recipe"], cubin["identity"]
            require(identity["name"] == name and type(identity["line_info"]) is bool, "invalid cubin identity")
            for key in ("entries", "resources", "declarations", "membership"):
                value = identity[key] if key in ("entries", "resources") else cubin[key]
                require(isinstance(value, list), "invalid " + key)
                self.records += len(value)
            for declaration in cubin["declarations"]:
                for key in ("symbol", "operation", "contract", "implementation"):
                    string(declaration[key])
                requirements = declaration["requirements"]
                require(isinstance(requirements, dict) and all(isinstance(v, str) for v in requirements.values()),
                        "invalid declaration requirements")
                if declaration["correctness_case"] is not None:
                    string(declaration["correctness_case"])
            for key in ("flags", "headers", "include_dirs"):
                require(isinstance(recipe[key], list) and all(isinstance(x, str) for x in recipe[key]), "invalid recipe " + key)
            require(isinstance(recipe["defines"], list) and all(isinstance(x, list) and len(x) == 2
                    and all(isinstance(s, str) for s in x) for x in recipe["defines"]), "invalid ordered defines")
            require(isinstance(recipe["generated_sha256"], dict), "invalid generated hashes")
            string(recipe["toolchain_version"])
            for key in ("nvcc_sha256", "source_closure_sha256"):
                hash_string(recipe[key])
            hash_string(identity["cubin_sha256"])
            for key in ("toolchain_version", "nvcc_sha256", "flags", "defines", "source_closure_sha256"):
                require(recipe[key] == identity[key], "recipe/identity differs: " + key)
            require(digest(recipe_bytes(recipe)) == hash_string(cubin["recipe_sha256"]), "recipe digest differs")
            keys, generated = set(), {}
            for member in cubin["membership"]:
                key = string(member["closure_key"])
                require(key not in keys, "duplicate source closure key")
                keys.add(key)
                require(member["file"] in report["files"], "dangling membership reference")
                file = report["files"][member["file"]]
                if key.startswith("generated/"):
                    require(source_key(file["identity"]) == ("Generated", None, key[len("generated/"):]), "generated identity differs")
                    generated[key[len("generated/"):]] = file["sha256"]
                else:
                    require(file["identity"]["kind"] != "Generated", "generated closure key missing")
            require(recipe["generated_sha256"] == generated, "generated recipe hashes differ")
        if "lock_families" in report:
            self.families(report, names)
        require(self.records <= self.limits.records, "aggregate report record bound")

    def families(self, report, names):
        families = report["lock_families"]
        require(isinstance(families, list) and families, "invalid lock families")
        require(report["lock_sha256"] is None, "directory lock has a file digest")
        seen, covered = set(), []
        for family in families:
            name = string(family["family"])
            require(FAMILY_NAME.fullmatch(name) and name not in seen, "invalid or duplicate lock family: " + name)
            seen.add(name)
            string(family["lock_path"])
            string(family["provenance_path"])
            if family["lock_sha256"] is not None:
                hash_string(family["lock_sha256"])
            cubins, changed = family["cubins"], family["provenance_changed"]
            require(isinstance(cubins, list) and cubins and all(isinstance(c, str) for c in cubins)
                    and isinstance(changed, list) and all(isinstance(c, str) for c in changed)
                    and set(changed) <= set(cubins), "invalid lock family cubins: " + name)
            self.records += len(cubins) + len(changed)
            covered += cubins
        require(len(covered) == len(set(covered)), "duplicate lock family coverage")
        require(set(covered) == names, "lock families do not cover the reported cubins")

    def verify(self, report):
        require(report["source_membership_complete"] and report["cubins"], "incomplete source membership")
        require(report["lock_status"] == "Verified", "lock is " + report["lock_status"])
        files = self.files
        if "lock_families" not in report:
            lock_hash, lock, _ = files.read(report["lock_path"], self.limits.report_bytes, contents=True)
            require(lock_hash == report["lock_sha256"], "stale lock digest")
            require(parse(lock) == [c["identity"] for c in report["cubins"]], "lock identities differ")
        else:
            # Only gating files are read; provenance never decides currency.
            records = {c["name"]: family_record(c["identity"]) for c in report["cubins"]}
            for family in report["lock_families"]:
                lock_hash, lock, _ = files.read(family["lock_path"], self.limits.report_bytes, contents=True)
                require(lock_hash == family["lock_sha256"], "stale lock digest: family " + family["family"])
                require(parse(lock) == [records[name] for name in family["cubins"]],
                        "lock identities differ: family " + family["family"])
        compiler = files.read(string(report["compiler_path"]), self.limits.artifact_bytes)[0]
        for file in report["files"].values():
            sha, _, size = files.read(file["location"], self.limits.source_bytes, contents=True)
            require(sha == file["sha256"] and size == file["bytes"], "stale source: " + file["identity"]["path"])
        for cubin in report["cubins"]:
            identity = cubin["identity"]
            require(files.json(cubin["identity_path"]) == identity, "stale cubin identity sidecar: " + cubin["name"])
            require(identity["nvcc_sha256"] == compiler, "stale compiler: " + cubin["name"])
            require(files.read(cubin["path"], self.limits.artifact_bytes)[0] == identity["cubin_sha256"], "stale cubin: " + cubin["name"])
        for cubin in report["cubins"]:
            identity = cubin["identity"]
            members = sorted(cubin["membership"], key=lambda m: m["closure_key"])
            # Local files, sidecars and cubins above are checked for every report.
            # Only the deterministic hash of identical closure bytes is reused.
            fingerprint = digest(serialized([
                [m["closure_key"], report["files"][m["file"]]["sha256"], report["files"][m["file"]]["bytes"]]
                for m in members]))
            if fingerprint not in self.closures:
                size = sum(report["files"][m["file"]]["bytes"] for m in members)
                require(files.closure_bytes + size <= self.limits.closure_bytes, "aggregate closure byte bound")
                files.closure_bytes += size
                closure = hashlib.sha256()
                for member in members:
                    file = report["files"][member["file"]]
                    raw = files.read(file["location"], self.limits.source_bytes, contents=True)[1]
                    key = member["closure_key"].encode()
                    closure.update(len(key).to_bytes(8, "little"))
                    closure.update(key)
                    closure.update(len(raw).to_bytes(8, "little"))
                    closure.update(raw)
                self.closures[fingerprint] = closure.hexdigest()
            require(self.closures[fingerprint] == identity["source_closure_sha256"],
                    "source closure differs: " + cubin["name"])

    def ownership(self, report, bindings):
        require(report["source_ownership_complete"], "incomplete source ownership")
        context = report["context"]
        require(context is not None, "missing consumer context")
        roots = {r["id"]: self.files.location(r["path"]) for r in context["sources"]}
        for file in report["files"].values():
            identity = file["identity"]
            if identity["kind"] != "Project":
                continue
            owner, path = identity["owner"], identity["path"]
            require(owner in roots and owner in bindings, "unbound source owner: " + str(owner))
            require(self.files.location(file["location"]) == roots[owner] / path, "source root/location differs: " + path)
            # Exported or previous checkout copies cannot establish freshness of
            # today's source just by declaring the same namespace.
            current = self.files.location(Path(bindings[owner]) / path)
            sha, _, size = self.files.read(current, self.limits.source_bytes)
            require(sha == file["sha256"] and size == file["bytes"], "stale bound source: " + path)

    def load(self, current=(), previous=(), bindings=None):
        variants = []
        groups = [("current", self.enumerate(current)), ("previous", self.enumerate(previous))]
        for origin, paths in groups:
            for path in paths:
                self.report_count += 1
                if self.report_count > self.limits.reports:
                    self.gaps.append("aggregate report count bound")
                    break
                row = {"path": str(path), "origin": origin, "status": "invalid", "issues": [],
                       "content_current": False, "ownership_current": False, "report": None}
                try:
                    report = self.files.json(path)
                    self.structure(report)
                    row["report"] = report
                    row["status"] = "historical" if origin == "previous" else "stale"
                    if origin == "current":
                        # Select candidates against current source ownership before
                        # spending the closure budget on obsolete build variants.
                        try:
                            self.ownership(report, bindings or {})
                            row["ownership_current"] = True
                            self.verify(report)
                            row["content_current"] = True
                            row["status"] = "current"
                        except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
                            row["issues"].append(str(error))
                except (OSError, ValueError, KeyError, TypeError, RecursionError) as error:
                    row["issues"].append(str(error))
                    self.gaps.append(f"invalid report {path}: {error}")
                variants.append(row)
        return variants

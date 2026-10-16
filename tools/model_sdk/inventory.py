#!/usr/bin/env python3
"""Read and check the model inventories, models/<m>/model.toml.

A model directory holds up to three crates, one per role: bundle/, compiler/
and contract/. The role is the directory. The inventory records what Cargo
metadata cannot: each role's package name, its exact bin and cdylib targets
(a policy assertion, not a derived fact), and the stem of the model's
checkpoint variables for CPU CI. Optional closed operational sections declare
release selections, bounded contributor recipes, CI inputs, catalog opt-in
and geometry recipe legs.

Usage: model_inventory.py checkpoint-env    print each declared stem
"""
from __future__ import annotations

import dataclasses
import pathlib
import re
import sys
import tomllib
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA = "chopinfer-model-inventory-v1"
ROLES = ("bundle", "compiler", "contract")
REQUIRED_ROLES = ("bundle", "compiler")
TOP_KEYS = {"schema", "checkpoint_env", *ROLES, "release", "contributor", "checks", "kernel_lab",
            "geometry_recipe"}
ROLE_KEYS = {"package", "bin", "cdylib"}
ENV_STEM = re.compile(r"[A-Z][A-Z0-9_]*")
NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
ARCHITECTURE = re.compile(r"[A-Z][A-Za-z0-9]*")
# A recipe stops at the contract stage while the model's bundle cannot build and
# serve an added geometry (no program yet, or one compiled for fixed geometries).
RECIPE_STAGES = ("bundle", "contract")
# Each recipe fixture directory, geometries/<fixture>/ under the model's
# fixture root (fixture_root), holds these.
RECIPE_FIXTURE_FILES = ("config.json", "geometry.json", "supported.json")


class InventoryError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class Package:
    model: str
    role: str
    name: str
    # (target name, kind) for every bin and cdylib target.
    targets: frozenset[tuple[str, str]]

    directory: str

    @property
    def manifest(self) -> str:
        return (pathlib.PurePosixPath(self.directory) / self.role / "Cargo.toml").as_posix()


@dataclasses.dataclass(frozen=True)
class Target:
    package: str
    name: str
    kind: str


@dataclasses.dataclass(frozen=True)
class Asset:
    source: str
    destination: str


@dataclasses.dataclass(frozen=True)
class Release:
    enabled: bool = False
    bins: tuple[str, ...] = ()
    cdylibs: tuple[str, ...] = ()
    assets: tuple[Asset, ...] = ()


@dataclasses.dataclass(frozen=True)
class TestCase:
    name: str
    case: str


@dataclasses.dataclass(frozen=True)
class TestSelection:
    name: str
    phase: str
    package: str
    lib: bool = False
    bins: tuple[str, ...] = ()
    suites: tuple[str, ...] = ()
    case: str | None = None
    compile: bool = False
    cases: tuple[TestCase, ...] = ()

    def expanded(self) -> tuple[TestSelection, ...]:
        if not self.cases:
            return (self,)
        return tuple(dataclasses.replace(self, name=case.name, case=case.case, cases=())
                     for case in self.cases)


@dataclasses.dataclass(frozen=True)
class Contributor:
    roots: tuple[str, ...] = ()
    fixtures: tuple[str, ...] = ()
    clear_env: tuple[str, ...] = ()
    tests: tuple[TestSelection, ...] = ()


@dataclasses.dataclass(frozen=True)
class Checks:
    reference_tests: tuple[str, ...] = ()
    compile_inputs: tuple[str, ...] = ()
    profile_targets: tuple[Target, ...] = ()


@dataclasses.dataclass(frozen=True)
class KernelLab:
    catalog: bool = False


@dataclasses.dataclass(frozen=True)
class GeometryRecipe:
    """The legs of tests/model-geometry-recipe.sh: one per fixture, at the
    model's stage (`--ci` runs bundle-stage legs, `--contract` the others)."""
    stage: str
    fixtures: tuple[str, ...]
    # The architecture alias the recipe's geometry is registered under.
    architecture: str
    # The contract crate's test suites a bundle-stage leg runs.
    contract_tests: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Model:
    name: str
    checkpoint_env: str | None
    packages: dict[str, Package]
    release: Release = dataclasses.field(default_factory=Release)
    contributor: Contributor | None = None
    checks: Checks = dataclasses.field(default_factory=Checks)
    kernel_lab: KernelLab = dataclasses.field(default_factory=KernelLab)
    geometry_recipe: GeometryRecipe | None = None
    # The bundle's kernel lock files, workspace-relative: `bundle/kernels.lock`
    # and each family file of the directory form `bundle/kernels.lock.d/`
    # (crates/kernel-build LockDir), found at load. Empty means the file form.
    kernel_locks: tuple[str, ...] = ()

    @property
    def directory(self) -> str:
        return next(iter(self.packages.values())).directory


@dataclasses.dataclass(frozen=True)
class ReleaseModel:
    name: str
    directory: str
    targets: tuple[Target, ...]
    assets: tuple[Asset, ...]


def _table(value, keys, label):
    if not isinstance(value, dict):
        raise InventoryError(f"{label}: must be a table")
    unknown = set(value) - set(keys)
    if unknown:
        raise InventoryError(f"{label}: unknown keys {sorted(unknown)}")
    return value


def _strings(value, label, pattern=None):
    if not isinstance(value, list) or not all(isinstance(x, str) and x and
            (pattern is None or pattern.fullmatch(x)) for x in value):
        raise InventoryError(f"{label}: must be a list of valid strings")
    if len(set(value)) != len(value):
        raise InventoryError(f"{label}: repeats a value")
    return tuple(value)


def _names(path, role, key, value):
    return _strings(value, f"{path}: {role}.{key}", NAME)


def _boolean(row, key, label):
    value = row.get(key, False)
    if type(value) is not bool:
        raise InventoryError(f"{label}.{key}: must be boolean")
    return value


def _text(row, key, label):
    value = row.get(key)
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise InventoryError(f"{label}.{key}: must be a nonempty printable string")
    return value


def _path(value, label, *, pattern=False):
    # Patterns allow only a bounded filename wildcard, never recursive traversal.
    component = r"[A-Za-z0-9_][A-Za-z0-9_.-]*"
    if pattern:
        component = r"[A-Za-z0-9_*?][A-Za-z0-9_.*?-]*"
    if not isinstance(value, str) or not value or not all(
            re.fullmatch(component, part) for part in value.split("/")):
        raise InventoryError(f"{label}: invalid path {value!r}")
    if pattern and "**" in value:
        raise InventoryError(f"{label}: recursive glob '**' is forbidden")
    if pattern and any("*" in part or "?" in part for part in value.split("/")[:-1]):
        raise InventoryError(f"{label}: wildcard must be in the filename")
    return value


def _rows(value, label):
    if not isinstance(value, list):
        raise InventoryError(f"{label}: must be an array of tables")
    return value


def _target(value, declared, label):
    row = _table(value, ("package", "kind", "target"), label)
    if not all(isinstance(row.get(key), str) for key in ("package", "kind", "target")):
        raise InventoryError(f"{label}: target fields must be strings")
    if row["package"] not in declared or (row["target"], row["kind"]) not in declared[row["package"]].targets:
        raise InventoryError(f"{label}: undeclared target {row['target']!r}")
    return Target(row["package"], row["target"], row["kind"])


def _release_target(model, name, kind, label):
    owners = [package.name for package in model.packages.values() if (name, kind) in package.targets]
    if len(owners) != 1:
        raise InventoryError(f"{label}: target {name!r} needs exactly one owning role")
    return Target(owners[0], name, kind)


def destination(target: Target) -> str:
    if target.kind == "bin":
        return "bin/" + target.name
    if target.kind == "cdylib":
        return "lib/lib" + target.name + ".so"
    raise InventoryError(f"unsupported release target kind {target.kind!r}")


def _claim_destination(destinations, path, row):
    for previous, owner in destinations.items():
        if path == previous or path.startswith(previous + "/") or previous.startswith(path + "/"):
            raise InventoryError(f"{row}: destination {path!r} collides with {owner} ({previous!r})")
    destinations[path] = row


LOCK_FAMILY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\.(?:lock|provenance\.json)")


def kernel_locks(model) -> tuple[str, ...]:
    """The kernel lock files a release ships for `model`: those found at load,
    or the file form `bundle/kernels.lock` when none was found, so that a
    missing lock is reported by check_files."""
    return model.kernel_locks or ((pathlib.PurePosixPath(model.directory) / "bundle/kernels.lock").as_posix(),)


def _kernel_locks(root, directory, relative) -> tuple[str, ...]:
    """The lock files of a model's bundle: the file form, then the sorted
    family files of the directory form. A family file is a regular file named
    `<family>.lock` or `<family>.provenance.json`; anything else there, or a
    linked directory or file, is refused rather than shipped or skipped."""
    bundle = pathlib.PurePosixPath(directory.relative_to(root).as_posix()) / "bundle"
    found = []
    if (directory / "bundle/kernels.lock").is_file():
        found.append((bundle / "kernels.lock").as_posix())
    lock_dir = directory / "bundle/kernels.lock.d"
    if lock_dir.is_symlink():
        raise InventoryError(f"{relative}: bundle/kernels.lock.d must not be a symlink")
    if lock_dir.is_dir():
        for path in sorted(lock_dir.iterdir()):
            if path.is_symlink() or not path.is_file() or not LOCK_FAMILY.fullmatch(path.name):
                raise InventoryError(f"{relative}: bundle/kernels.lock.d/{path.name} is not a lock family file")
            found.append((bundle / "kernels.lock.d" / path.name).as_posix())
    return tuple(found)


def release_plan(models) -> tuple[ReleaseModel, ...]:
    """Select opted-in models and refuse collisions across the complete plan.

    Eval files are discovered by the release consumer for each selected model.
    The lock destinations are stable consumer rules, not extra assets. Serving
    profiles ship inside each package, not beside the host.
    """
    planned, destinations = [], {}
    for model in models:
        if not model.release.enabled:
            continue
        label = f"{model.directory}/model.toml.release"
        targets = tuple(_release_target(model, name, kind, f"{label}.{key}[{i}]")
                        for key, kind in (("bins", "bin"), ("cdylibs", "cdylib"))
                        for i, name in enumerate(getattr(model.release, key)))
        if not targets and not model.release.assets:
            raise InventoryError(f"{label}: enabled release has nothing selected")
        bundle = pathlib.PurePosixPath(model.directory) / "bundle"
        locks = tuple(Asset(source, f"models/{model.name}/{pathlib.PurePosixPath(source).relative_to(bundle).as_posix()}")
                      for source in kernel_locks(model))
        assets = (*locks, *model.release.assets)
        entries = [(destination(target), f"{label}.{key}[{i}]")
                   for key, kind in (("bins", "bin"), ("cdylibs", "cdylib"))
                   for i, target in enumerate(target for target in targets if target.kind == kind)]
        entries += [(asset.destination, f"{label}.kernels_lock" + (f"[{i}]" if len(locks) > 1 else ""))
                    for i, asset in enumerate(locks)]
        entries += [(asset.destination, f"{label}.assets[{i}]") for i, asset in enumerate(model.release.assets)]
        for path, row in entries:
            _claim_destination(destinations, path, row)
        planned.append(ReleaseModel(model.name, model.directory, targets, tuple(assets)))
    return tuple(planned)


def _operational(data, model, label):
    declared = {package.name: package for package in model.packages.values()}
    release, contributor, checks, kernel_lab, recipe = Release(), None, Checks(), KernelLab(), None
    if "release" in data:
        row = _table(data["release"], ("enabled", "bins", "cdylibs", "assets"), label + ".release")
        assets, destinations = [], {}
        for i, asset in enumerate(_rows(row.get("assets", []), label + ".release.assets")):
            at = f"{label}.release.assets[{i}]"
            asset = _table(asset, ("source", "destination"), at)
            source = _path(asset.get("source"), at + ".source")
            dest = _path(asset.get("destination"), at + ".destination")
            prefix = f"models/{model.name}/"
            if not dest.startswith(prefix):
                raise InventoryError(f"{at}.destination: must be under {prefix}")
            if model.directory != "." and not source.startswith(model.directory + "/"):
                raise InventoryError(f"{at}.source: must be under {model.directory}/")
            _claim_destination(destinations, dest, at)
            assets.append(Asset(source, dest))
        release = Release(_boolean(row, "enabled", label + ".release"),
                          _strings(row.get("bins", []), label + ".release.bins", NAME),
                          _strings(row.get("cdylibs", []), label + ".release.cdylibs", NAME), tuple(assets))
        for key, kind in (("bins", "bin"), ("cdylibs", "cdylib")):
            for i, name in enumerate(getattr(release, key)):
                _release_target(model, name, kind, f"{label}.release.{key}[{i}]")
    if "contributor" in data:
        row = _table(data["contributor"], ("roots", "fixtures", "clear_env", "tests"), label + ".contributor")
        roots = _strings(row.get("roots", []), label + ".contributor.roots", NAME)
        for i, name in enumerate(roots):
            if name not in declared:
                raise InventoryError(f"{label}.contributor.roots[{i}]: package {name!r} is not a model role")
        fixtures = tuple(_path(path, f"{label}.contributor.fixtures[{i}]")
                         for i, path in enumerate(_strings(row.get("fixtures", []), label + ".contributor.fixtures")))
        clear_env = _strings(row.get("clear_env", []), label + ".contributor.clear_env", ENV_STEM)
        prefix = f"CHOPINFER_{model.checkpoint_env}_" if model.checkpoint_env else None
        for i, key in enumerate(clear_env):
            if prefix is None or not key.startswith(prefix) or key in (prefix + "CHECKPOINT", prefix + "TOKENIZER_DIR"):
                raise InventoryError(f"{label}.contributor.clear_env[{i}]: must be a model-owned extra with prefix {prefix!r}")
        tests, names = [], set()
        for i, test in enumerate(_rows(row.get("tests", []), label + ".contributor.tests")):
            at = f"{label}.contributor.tests[{i}]"
            test = _table(test, ("name", "phase", "package", "lib", "bins", "suites", "case", "compile", "cases"), at)
            package = _text(test, "package", at)
            if package not in declared:
                raise InventoryError(f"{at}.package: {package!r} is not a model role")
            cases = []
            for j, case in enumerate(_rows(test.get("cases", []), at + ".cases")):
                here = f"{at}.cases[{j}]"
                if isinstance(case, str):
                    case = {"name": case, "case": case}
                case = _table(case, ("name", "case"), here)
                cases.append(TestCase(_text(case, "name", here), _text(case, "case", here)))
            if "case" in test and cases:
                raise InventoryError(f"{at}: case and cases are mutually exclusive")
            exact = _text(test, "case", at) if "case" in test else None
            for case in (exact, *(case.case for case in cases)):
                if case is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)*", case):
                    raise InventoryError(f"{at}: invalid exact test case {case!r}")
            compile = _boolean(test, "compile", at)
            if compile and (exact is not None or cases):
                raise InventoryError(f"{at}: compile cannot select exact cases")
            name = _text(test, "name", at) if "name" in test else ("" if cases else _text(test, "name", at))
            selected = TestSelection(name, _text(test, "phase", at), package, _boolean(test, "lib", at),
                                     _strings(test.get("bins", []), at + ".bins", NAME),
                                     _strings(test.get("suites", []), at + ".suites", NAME), exact, compile, tuple(cases))
            if not (selected.lib or selected.bins or selected.suites):
                raise InventoryError(f"{at}: test needs a typed selection")
            for expanded in selected.expanded():
                if expanded.name in names:
                    raise InventoryError(f"{at}: duplicate test name {expanded.name!r}")
                names.add(expanded.name)
            tests.append(selected)
        if not roots or not tests:
            raise InventoryError(f"{label}.contributor: needs nonempty roots and tests")
        contributor = Contributor(roots, fixtures, clear_env, tuple(tests))
    if "checks" in data:
        row = _table(data["checks"], ("reference_tests", "compile_inputs", "profile_targets"), label + ".checks")
        references = tuple(_path(path, f"{label}.checks.reference_tests[{i}]") for i, path in
                           enumerate(_strings(row.get("reference_tests", []), label + ".checks.reference_tests")))
        patterns = tuple(_path(path, f"{label}.checks.compile_inputs[{i}]", pattern=True) for i, path in
                         enumerate(_strings(row.get("compile_inputs", []), label + ".checks.compile_inputs")))
        targets = tuple(_target(target, declared, f"{label}.checks.profile_targets[{i}]") for i, target in
                        enumerate(_rows(row.get("profile_targets", []), label + ".checks.profile_targets")))
        seen = set()
        for i, target in enumerate(targets):
            if target in seen:
                raise InventoryError(f"{label}.checks.profile_targets[{i}]: duplicate target {target.name!r}")
            seen.add(target)
        checks = Checks(references, patterns, targets)
    if "kernel_lab" in data:
        row = _table(data["kernel_lab"], ("catalog",), label + ".kernel_lab")
        kernel_lab = KernelLab(_boolean(row, "catalog", label + ".kernel_lab"))
        if kernel_lab.catalog and "contract" not in model.packages:
            raise InventoryError(f"{label}.kernel_lab.catalog: needs a contract role")
    if "geometry_recipe" in data:
        at = label + ".geometry_recipe"
        row = _table(data["geometry_recipe"], ("stage", "fixtures", "architecture", "contract_tests"), at)
        stage = _text(row, "stage", at)
        if stage not in RECIPE_STAGES:
            raise InventoryError(f"{at}.stage: must be one of {list(RECIPE_STAGES)}")
        fixtures = _strings(row.get("fixtures", []), at + ".fixtures", NAME)
        architecture = _text(row, "architecture", at)
        if not ARCHITECTURE.fullmatch(architecture):
            raise InventoryError(f"{at}.architecture: must be an architecture class name")
        tests = _strings(row.get("contract_tests", []), at + ".contract_tests", NAME)
        if not fixtures:
            raise InventoryError(f"{at}.fixtures: needs at least one fixture")
        if "contract" not in model.packages:
            raise InventoryError(f"{at}: needs a contract role")
        if stage == "bundle" and not (tests and model.checkpoint_env):
            raise InventoryError(f"{at}: a bundle-stage recipe needs contract_tests and checkpoint_env")
        if stage == "contract" and tests:
            raise InventoryError(f"{at}.contract_tests: a contract-stage recipe runs its geometry_recipe test")
        recipe = GeometryRecipe(stage, fixtures, architecture, tests)
    return release, contributor, checks, kernel_lab, recipe


def _file(root, value, label):
    path = pathlib.Path(root)
    for part in pathlib.PurePosixPath(value).parts:
        path /= part
        if path.is_symlink():
            raise InventoryError(f"{label}: symlink path {value!r}")
    if not path.is_file():
        raise InventoryError(f"{label}: missing file {value!r}")
    return value


def compile_files(root, model) -> tuple[str, ...]:
    """Expand each bounded model-relative pattern in order, sorting its matches."""
    root = pathlib.Path(root)
    files = []
    for i, pattern in enumerate(model.checks.compile_inputs):
        label = f"{model.directory}/model.toml.checks.compile_inputs[{i}]"
        directory = root / model.directory
        # Check literal parent components before globbing, including symlink dirs.
        parent = pathlib.PurePosixPath(pattern).parent
        probe = root
        for part in (pathlib.PurePosixPath(model.directory) / parent).parts:
            probe /= part
            if probe.is_symlink():
                raise InventoryError(f"{label}: symlink path {pattern!r}")
        matches = sorted(directory.glob(pattern))
        if not matches:
            raise InventoryError(f"{label}: pattern {pattern!r} matches no files")
        for path in matches:
            value = path.relative_to(root).as_posix()
            _file(root, value, label)
            if value not in files:
                files.append(value)
    return tuple(files)


def fixture_root(model) -> str:
    """The workspace-relative directory of the model's fixtures: the shared
    tree's fixtures/<model> for a model in models/<model>, else fixtures/ in
    its model root, as an exported or external model keeps them."""
    if model.directory == f"models/{model.name}":
        return f"fixtures/{model.name}"
    return (pathlib.PurePosixPath(model.directory) / "fixtures").as_posix()


def _file_rows(model):
    label = f"{model.directory}/model.toml"
    if model.release.enabled:
        for source in kernel_locks(model):
            yield source, f"{label}.release.kernels_lock"
    for i, asset in enumerate(model.release.assets):
        yield asset.source, f"{label}.release.assets[{i}].source"
    if model.contributor:
        for i, path in enumerate(model.contributor.fixtures):
            yield path, f"{label}.contributor.fixtures[{i}]"
    for i, path in enumerate(model.checks.reference_tests):
        yield (pathlib.PurePosixPath(model.directory) / path).as_posix(), f"{label}.checks.reference_tests[{i}]"
    if model.geometry_recipe:
        for i, fixture in enumerate(model.geometry_recipe.fixtures):
            for name in RECIPE_FIXTURE_FILES:
                yield f"{fixture_root(model)}/geometries/{fixture}/{name}", f"{label}.geometry_recipe.fixtures[{i}]"


def file_inputs(root, model) -> tuple[str, ...]:
    """Validated workspace-relative files needed by this model's declarations."""
    paths = [_file(root, path, label) for path, label in _file_rows(model)]
    return tuple(dict.fromkeys([*paths, *compile_files(root, model)]))


def check_files(models, root) -> None:
    for model in models:
        file_inputs(root, model)


def load_model(root: pathlib.Path, directory: pathlib.Path) -> Model:
    for label, value in (("name", directory.name),
                         ("directory", directory.relative_to(root).as_posix())):
        if any(unicodedata.category(char) in ("Cc", "Cf") for char in value):
            raise InventoryError(f"model.toml: model {label} {value!r} contains control characters")
    path = directory / "model.toml"
    relative = path.relative_to(root).as_posix()
    if not path.is_file():
        raise InventoryError(f"{relative}: is missing")
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise InventoryError(f"{relative}: {error}") from error
    unknown = set(data) - TOP_KEYS
    if unknown:
        raise InventoryError(f"{relative}: unknown keys {sorted(unknown)}")
    if data.get("schema") != SCHEMA:
        raise InventoryError(f"{relative}: schema must be {SCHEMA!r}")
    stem = data.get("checkpoint_env")
    if stem is not None and not (isinstance(stem, str) and ENV_STEM.fullmatch(stem)):
        raise InventoryError(f"{relative}: checkpoint_env must be an upper-case variable stem")
    packages = {}
    for role in ROLES:
        if role not in data:
            if role in REQUIRED_ROLES:
                raise InventoryError(f"{relative}: [{role}] is required")
            continue
        table = data[role]
        if not isinstance(table, dict):
            raise InventoryError(f"{relative}: {role} must be a table")
        unknown = set(table) - ROLE_KEYS
        if unknown:
            raise InventoryError(f"{relative}: unknown keys in [{role}] {sorted(unknown)}")
        name = table.get("package")
        if not (isinstance(name, str) and NAME.fullmatch(name)):
            raise InventoryError(f"{relative}: {role}.package must be a package name")
        targets = {(target, kind) for kind in ("bin", "cdylib")
                   for target in _names(relative, role, kind, table.get(kind, []))}
        if not (directory / role / "Cargo.toml").is_file():
            raise InventoryError(f"{relative}: [{role}] declared but {role}/Cargo.toml is missing")
        packages[role] = Package(directory.name, role, name, frozenset(targets), directory.relative_to(root).as_posix())
    for role in ROLES:
        if role not in packages and (directory / role / "Cargo.toml").is_file():
            raise InventoryError(f"{relative}: {role}/Cargo.toml has no [{role}] entry")
    model = Model(directory.name, stem, packages)
    release, contributor, checks, kernel_lab, recipe = _operational(data, model, relative)
    model = dataclasses.replace(model, release=release, contributor=contributor,
                                checks=checks, kernel_lab=kernel_lab, geometry_recipe=recipe,
                                kernel_locks=_kernel_locks(root, directory, relative) if "bundle" in packages else ())
    release_plan([model])
    return model


def load(root) -> list[Model]:
    root = pathlib.Path(root)
    models = root / "models"
    if not models.is_dir():
        return []
    loaded = [load_model(root, directory) for directory in sorted(models.iterdir()) if directory.is_dir()]
    owners = {}
    for model in loaded:
        for package in model.packages.values():
            previous = owners.setdefault(package.name, package.manifest)
            if previous != package.manifest:
                raise InventoryError(f"package {package.name} is declared by {previous} and {package.manifest}")
    release_plan(loaded)
    return loaded


def shipped(root=ROOT) -> tuple[str, ...]:
    """Every model with an inventory under root/models, by name in directory order."""
    return tuple(model.name for model in load(root))


def released(root=ROOT) -> tuple[str, ...]:
    """The shipped models whose inventory enables `[release]`, by name in
    directory order. A model declares `[release] enabled = false` until it
    qualifies; tests exempt it from the released-model facts."""
    return tuple(model.name for model in load(root) if model.release.enabled)


def packages(models) -> dict[str, Package]:
    return {package.name: package for model in models for package in model.packages.values()}


def check_metadata(models, metadata, root, *, external=False) -> None:
    """Every workspace package under models/ has exactly one declared role and
    every declared package is a workspace member at its role's path, with
    exactly the declared bin and cdylib targets."""
    root = pathlib.Path(root).resolve()
    declared = packages(models)
    members = {}
    by_id = {package["id"]: package for package in metadata["packages"]}
    for id in metadata["workspace_members"]:
        package = by_id[id]
        manifest = pathlib.Path(package["manifest_path"]).resolve().relative_to(root).as_posix()
        if package["name"] in members:
            raise InventoryError(f"ambiguous workspace package {package['name']}")
        members[package["name"]] = (manifest, package)
        if manifest.startswith("models/") or (external and any(
            kind in ("bin", "cdylib") for target in package["targets"] for kind in target["kind"])):
            expected = declared.get(package["name"])
            if expected is None or expected.manifest != manifest:
                raise InventoryError(f"{manifest}: package {package['name']} has no role in its model.toml")
    for model in models:
        if model.contributor:
            for key, names in (("roots", model.contributor.roots),
                               ("tests", tuple(row.package for row in model.contributor.tests))):
                for i, name in enumerate(names):
                    if name not in members:
                        raise InventoryError(f"{model.directory}/model.toml.contributor.{key}[{i}]: package {name!r} is not a workspace member")
    for name, expected in declared.items():
        if name not in members:
            raise InventoryError(f"{expected.manifest}: declared package {name} is not a workspace member")
        manifest, package = members[name]
        if manifest != expected.manifest:
            raise InventoryError(f"{name}: declared at {expected.manifest}, found at {manifest}")
        observed = {(target["name"], kind) for target in package["targets"]
                    for kind in target["kind"] if kind in ("bin", "cdylib")}
        if observed != expected.targets:
            raise InventoryError(f"{name} target inventory differs: declared {sorted(expected.targets)}, observed {sorted(observed)}")

    for model in models:
        if not model.contributor:
            continue
        for i, row in enumerate(model.contributor.tests):
            label = f"{model.directory}/model.toml.contributor.tests[{i}]"
            targets = members[row.package][1]["targets"]
            observed = {(target["name"], kind) for target in targets for kind in target["kind"]}
            if row.lib and not any(kind in ("lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro") for _, kind in observed):
                raise InventoryError(f"{label}.lib: package {row.package!r} has no library target")
            for key, kind in (("bins", "bin"), ("suites", "test")):
                for j, name in enumerate(getattr(row, key)):
                    if (name, kind) not in observed:
                        raise InventoryError(f"{label}.{key}[{j}]: package {row.package!r} has no {kind} target {name!r}")
    check_files(models, root)


def main(arguments) -> int:
    root = ROOT
    if arguments == ["checkpoint-env"]:
        for model in load(root):
            if model.checkpoint_env:
                print(model.checkpoint_env)
        return 0
    print(__doc__.strip().splitlines()[-1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except InventoryError as error:
        print(f"model inventory: {error}", file=sys.stderr)
        sys.exit(1)

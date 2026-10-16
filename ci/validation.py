"""Read-only validation obligations. A successful report is not validation.

The build graph, behavioral contracts and evidence applicability are separate.
No runner, device probe, server request or prerequisite preparation is called.
"""
import ast
from copy import deepcopy
from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tomllib

import change_scope
import kernel_scope
import validation_inventory

SCHEMA = "chopinfer-validation-report-v1"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def matches(path, patterns):
    return any(fnmatchcase(path, pattern) for pattern in patterns)


def gate_change_classes():
    """The change classes every model declaration maps to gate checks."""
    sys.path.insert(0, str(change_scope.ROOT / "tools/qual"))
    import declaration
    return declaration.CHANGE_CLASSES


def read_declaration(root, name, variant=None):
    """A model's qual.json, or a checkpoint variant's qual.VARIANT.json: its
    workspace path, compiled gate config and change classes, read as
    written: the fields this report reads must be there, and the rules are
    chopinfer eval's."""
    sys.path.insert(0, str(change_scope.ROOT / "tools/qual"))
    import declaration
    base = declaration.path(name, root)
    configured = variant is not None and variant in read_raw(base).get("configurations", {})
    path = base if variant is None or configured else declaration.path(name, root, variant)
    raw = read_raw(path)
    for field in ("schema", "model", "cohorts", "checks", "change_classes"):
        if field not in raw:
            raise ValueError(f"{field}: is required")
    if configured:
        gate, classes = configuration_gate(raw, variant)
        return path.relative_to(root).as_posix(), gate, classes
    return path.relative_to(root).as_posix(), declaration.read(path), raw["change_classes"]


def read_raw(path):
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("declaration: expected an object")
    return raw


def configuration_gate(raw, name):
    """A configuration's gate as this report reads it, from qual.json as
    written: its checks with every rule source the configuration adds, the
    inputs, and the base change classes kept to its checks plus its own
    (chopinfer eval declaration states the whole configuration)."""
    import declaration
    spec = raw["configurations"][name]
    checks = {check["id"]: check for check in declaration.compile_gate(raw)["checks"]}
    selected = []
    for check_id in spec["checks"]:
        check = dict(checks[check_id])
        added = [rule for rules in spec.get("append", {}).get(check_id, {}).values() for rule in rules]
        added += spec.get("rules", {}).get(check_id, [])
        if spec.get("source"):
            added.append({"source": spec["source"]})
        check["rules"] = [*check.get("rules", []), *({"source": rule["source"]} for rule in added)]
        selected.append(check)
    profiles = {cohort: {**raw["cohorts"][cohort], "checks": list(spec[key])}
                for cohort, key in (("quick", "quick"), ("full", "checks"))}
    gate = {**declaration.compile_gate(raw), "checks": selected, "variant": name, "profiles": profiles,
            "inputs": {**raw["inputs"], **spec.get("inputs", {})}}
    classes = {change: [check for check in names if check in spec["checks"]]
               + [check for check in spec.get("change_classes", {}).get(change, []) if check in spec["checks"]]
               for change, names in raw["change_classes"].items()}
    return gate, classes


def declared_variants(root, name):
    """The model's configurations (qual.json), else its variant files."""
    sys.path.insert(0, str(change_scope.ROOT / "tools/qual"))
    import declaration
    base = declaration.path(name, root)
    configured = list(read_raw(base).get("configurations", {})) if base.is_file() else []
    return configured or declaration.variants(name, root)


def gates(name, row):
    """Each release gate of a model: (gate name, its row) for qual.json and
    each checkpoint variant (MODEL.VARIANT), whose row holds its own
    declaration, gate config, change classes and readers."""
    yield name, row
    for variant, gate in sorted(row.get("variants", {}).items()):
        yield f"{name}.{variant}", gate


def gate_row(catalog, gate):
    """The catalog row a gate name (MODEL or MODEL.VARIANT) selects, with the
    model's profiles."""
    model, _, variant = gate.partition(".")
    row = catalog.models[model]
    return row if not variant else {**row, **row["variants"][variant]}


def read_json(path):
    sys.path.insert(0, str(change_scope.ROOT / "tools/qual"))
    from validation_evidence import read_json as bounded_json
    return bounded_json(path)


def file_digest(path):
    sys.path.insert(0, str(change_scope.ROOT / "tools/qual"))
    from validation_evidence import read_bytes
    return sha(read_bytes(path))


class Parameters(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def expand(value, parameters):
    if isinstance(value, str):
        return value.format_map(Parameters(parameters))
    if isinstance(value, list):
        return [expand(item, parameters) for item in value]
    if isinstance(value, dict):
        return {key: expand(item, parameters) for key, item in value.items()}
    return value


def model_parameters(name, row):
    inventory = row["inventory"]
    params = {"model": name, "checkpoint_env": inventory.checkpoint_env if inventory else "UNRESOLVED"}
    if inventory:
        params.update({role: package.name for role, package in inventory.packages.items()})
    return params


def cargo_test_packages(definition):
    packages = set()
    for command in definition["commands"]:
        if command[:2] != ["cargo", "test"]:
            continue
        arguments = iter(command[2:])
        for argument in arguments:
            if argument == "--":
                break
            if argument in ("-p", "--package"):
                packages.add(next(arguments, ""))
    return packages


def locked_identities(path):
    return [{k: identity[k] for k in
        ("name", "source_closure_sha256", "nvcc_sha256", "cubin_sha256", "flags", "defines") if k in identity}
        for identity in read_json(path)]


def lock_identities(root, paths, consumers=()):
    """A catalog lock is a lock file, a lock directory or one family's lock file."""
    def reported(lock):
        return [lock["path"], *(family["path"] for family in lock.get("families", []))]

    locks = []
    for path in paths:
        statuses = sorted({v["status"] for c in consumers for v in c["variants"]
                           if any(Path(p).as_posix().endswith("/" + path) for p in reported(v["lock"]))})
        value = {"path": path, "membership": ", ".join(statuses) if statuses else "missing report; retain consumers"}
        try:
            if (root / path).is_dir():
                # Provenance files never gate, so only family locks are identities.
                value["families"] = [{"family": file.name.removesuffix(".lock"), "path": (Path(path) / file.name).as_posix(),
                                      "sha256": file_digest(file), "identities": locked_identities(file)}
                                     for file in sorted((root / path).glob("*.lock"))]
            else:
                value["sha256"] = file_digest(root / path)
                value["identities"] = locked_identities(root / path)
        except (OSError, ValueError, TypeError):
            value["missing"] = True
        locks.append(value)
    return locks


def python_inputs(root, module, seen=None):
    """Static local imports only. Reading this graph never imports a check module."""
    seen = set() if seen is None else seen
    path = f"tools/qual/{module}.py"
    if path in seen or not (root / path).is_file():
        return seen
    seen.add(path)
    tree = ast.parse((root / path).read_text())
    for node in ast.walk(tree):
        names = ([name.name.split(".")[0] for name in node.names] if isinstance(node, ast.Import)
                 else [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module else [])
        for name in names:
            python_inputs(root, name, seen)
    return seen


class Catalog:
    def __init__(self, root, metadata):
        self.root = Path(root)
        self.data = tomllib.loads((self.root / "ci/validation.toml").read_text())
        if self.data.get("schema") != "chopinfer-validation-catalog-v1":
            raise ValueError("unsupported validation catalog")
        self.models = {}
        self.issues = []
        inventories, self.issues = validation_inventory.read(self.root, metadata)
        for inventory in inventories:
            path = self.root / "models" / inventory.name
            row = {"inventory": inventory, "data": {}, "profiles": {}, "gate": {}, "change_classes": {}, "readers": {}}
            self.models[inventory.name] = row
            try:
                data = tomllib.loads((path / "validation.toml").read_text())
                if data.get("schema") != "chopinfer-model-validation-v1":
                    raise ValueError("unsupported model validation catalog")
                if any(Path(p).is_absolute() or ".." in Path(p).parts for p in
                       [data["profiles"], *data.get("locks", [])]):
                    raise ValueError("catalog paths must stay within the workspace")
                row["data"] = data
                profiles = read_json(path / data["profiles"])
                if profiles["schema"] != "chopinfer-serve-profiles-v2" or not profiles["profiles"]:
                    raise ValueError("missing serving profiles")
                row["profiles"] = profiles["profiles"]
                row["declaration"], row["gate"], row["change_classes"] = read_declaration(self.root, inventory.name)
                row["variants"] = {}
                for variant in declared_variants(self.root, inventory.name):
                    declared, gate, classes = read_declaration(self.root, inventory.name, variant)
                    row["variants"][variant] = {"declaration": declared, "gate": gate, "change_classes": classes,
                                                "readers": {}}
                for rule in data["rules"]:
                    if rule["class"] not in self.data["classes"] or not rule["paths"]:
                        raise ValueError("invalid model source rule")
                for _, gate in gates(inventory.name, row):
                    self.gate_readers(gate)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.issues.append(f"{inventory.name}: incomplete validation metadata: {error}")
        discovered = {directory.name for directory in (self.root / "models").iterdir() if directory.is_dir()}
        for name in discovered - self.models.keys():
            self.models[name] = {"inventory": None, "data": {}, "profiles": {}, "gate": {}, "readers": {}}
        for rule in self.data["rules"]:
            if rule["class"] not in self.data["classes"]:
                raise ValueError("unknown shared change class")
        for policy in self.data["classes"].values():
            if not set(policy["checks"]) <= self.data["checks"].keys() or not set(policy["model_checks"]) <= self.data["model_checks"].keys():
                raise ValueError("class names an unknown check")
        for name in ("unknown", "validation_policy"):
            policy = self.data["classes"][name]
            if (set(policy["checks"]) != self.data["checks"].keys()
                    or set(policy["model_checks"]) != self.data["model_checks"].keys()):
                raise ValueError("incomplete broad class: " + name)
        for check in [*self.data["checks"].values(), *self.data["model_checks"].values()]:
            for field in ("claim", "kind", "commands", "observes", "targets", "prerequisites", "standing"):
                if field not in check:
                    raise ValueError("incomplete check observation metadata: " + field)

    def retain_previous(self, sources):
        """Removed declarations still impose compatibility/refusal obligations."""
        for path, text in sorted(sources.items()):
            parts = Path(path).parts
            if len(parts) != 3 or parts[0] != "models":
                continue
            name, filename = parts[1:]
            row = self.models.setdefault(name, {"inventory": None, "data": {}, "profiles": {}, "gate": {}, "readers": {}})
            if filename == "profiles.json":
                row["previous_profiles"] = json.loads(text)["profiles"]
            elif filename == "validation.toml":
                previous = tomllib.loads(text)
                row["data"].setdefault("rules", []).extend(previous.get("rules", []))
            if not row["inventory"]:
                self.issues.append(name + ": removed or incomplete model inventory; retain previous profiles")

    def gate_readers(self, row):
        config, readers = row["gate"], row["readers"]
        for check in config["checks"]:
            inputs = python_inputs(self.root, check["module"])
            for module in check.get("constants", {}):
                inputs |= python_inputs(self.root, module)
            inputs |= {row["declaration"], "tools/qual/release_gate.py"}
            rules = check["rules"] + [r for rs in check.get("input_rules", {}).values() for r in rs]
            inputs |= {r["source"].split("#")[0] for r in rules}
            for constants in check.get("constants", {}).values():
                inputs |= {entry["source"].split("#")[0] for entry in constants.values()}
            # Configured file inputs are potential readers, even if a module reads
            # them conditionally. Keep all declared readers until usage is proven.
            for value in config.get("inputs", {}).values():
                if isinstance(value, str):
                    inputs.add(value)
                elif isinstance(value, dict) and value.get("default"):
                    inputs.add(str(Path(value["default"]) / value.get("path", "")))
            readers[check["id"]] = sorted(inputs)

    def classify(self, path):
        rows = []
        for external in self.data.get("kernel_reports", {}).get("external", []):
            if path.startswith(external["directory"] + "/"):
                return [{"path": path, "class": "local_kernel" if matches(path, external["kernel_paths"]) else "frontend",
                         "models": [], "reason": "declared external model source/build domain"}]
        for name, row in self.models.items():
            for gate, declared in gates(name, row):
                for check, inputs in declared["readers"].items():
                    if any(path == p or path.startswith(p.rstrip("/") + "/") for p in inputs):
                        rows.append({"path": path, "class": "qualification", "models": [name],
                                     "gate_checks": [check], "reason": "declared policy/input/module reader: " + check,
                                     **({"gate": gate} if gate != name else {})})
        # An executable reader takes precedence over prose defaults, but does not
        # hide the source's own runtime contract.
        rule = next((r for r in self.data["rules"] if matches(path, r["paths"])), None)
        if rule and (rule["class"] != "documentation" or not rows):
            rows.append({"path": path, "class": rule["class"],
                         "models": [] if rule["class"] == "documentation" else sorted(self.models),
                         "reason": "shared rule: " + ", ".join(rule["paths"])})
        for name, row in self.models.items():
            prefix = f"models/{name}/"
            if path.startswith(prefix) and not (rule and rule["class"] == "validation_policy"):
                rule = next((r for r in row["data"].get("rules", []) if matches(path[len(prefix):], r["paths"])), None)
                rows.append({"path": path, "class": rule["class"] if rule else "model_unknown", "models": [name],
                             "reason": "owning model source/role" if rule else "unclassified code in known model"})
        if path == "fixtures/tokenizer-whitespace.json" or "/compiler/tests/support/" in path:
            names = [name for name, row in self.models.items() if row["inventory"]]
            rows.append({"path": path, "class": "frontend", "models": sorted(names),
                         "reason": "shared streaming fixture/source reaches compiler tests"})
        return rows or [{"path": path, "class": "unknown", "models": sorted(self.models),
                        "reason": "unknown ownership; widen validation"}]


def base_status():
    return {"selection": "selected", "execution": "not_run", "result": None,
            "applicability": "missing", "reasons": ["no supplied applicable evidence"]}


def obligation(identifier, definition, reasons, profiles=None):
    sys.path.insert(0, str(change_scope.ROOT / "tools/qual"))
    from validation_evidence import requirements
    row = {"id": identifier, **deepcopy(definition), "reasons": reasons, "profiles": profiles or [],
           "evidence": base_status()}
    row["required_identity"] = requirements(row)
    paths = {r["path"] for r in reasons if r["path"] != "*"}
    row["observation"] = {"inputs": sorted(p for p in paths if matches(p, row["observes"])),
                          "basis": "declared targets/features and separately executed artifacts",
                          "requires_artifact_binding": bool(row.get("artifacts")),
                          "features": row.get("features", ["default"]),
                          "platform": row.get("platform", "workspace-supported CPU platforms")}
    row["coverage_gap"] = row.get("gap")
    if paths and not row["observation"]["inputs"] and not row.get("artifacts") and not row["coverage_gap"]:
        row["coverage_gap"] = "Conservative selection only: this check's declared observation does not establish coverage of the changed input"
    row["reproduce"] = [shlex.join(command) for command in row["commands"]]
    return row


def describe_gate(config, cohort):
    # This helper is provided by the release runner; importing it performs no IO
    # beyond Python imports. Its modules are not workers until explicitly run.
    sys.path.insert(0, str(change_scope.ROOT / "tools/qual"))
    import release_gate
    return release_gate.describe(config, cohort)


def unresolved_gate(name, identifier, row, reasons, gap):
    return obligation(name + ":gate:" + identifier, {
        "claim": "Resolve release cohort coverage: " + identifier, "kind": "unknown", "commands": [],
        "observes": row["readers"].get(identifier, []), "targets": [identifier], "prerequisites": [gap],
        "standing": False, "gap": gap}, reasons,
        sorted(row["profiles"].keys() | row.get("previous_profiles", {}).keys()))


def build_report(metadata, paths, *, root=change_scope.ROOT, all_checks=False,
                 graph=None, catalog=None, context=None, receipts=(), source_identity=None,
                 kernel_reports=(), previous_kernel_reports=(), kernel_index=None):
    root = Path(root)
    paths = sorted(set(paths))
    graph = graph or change_scope.Graph(metadata)
    catalog = catalog or Catalog(root, metadata)
    changes = [row for path in paths for row in catalog.classify(path)]
    kernel_index = kernel_index or kernel_scope.Index(metadata, graph, catalog, kernel_reports, previous_kernel_reports)
    package_scope = graph.select(paths, broad=all_checks, kernel_index=kernel_index, kernel_changes=changes)
    kernel_consumers = [c for c in package_scope["kernel_consumers"] if c["selected"]]
    if all_checks or (paths and (catalog.issues or graph.issues)):
        changes.append({"path": "*", "class": "unknown", "models": sorted(catalog.models),
                        "reason": "--all" if all_checks else "incomplete dependency/inventory information"})
    required = {"cpu": [{"path": "*", "class": "standing", "reason": "standing CPU contracts"}]}
    model_required = {}
    gate_required = {}
    for change in changes:
        policy = catalog.data["classes"][change["class"]]
        for check in policy["checks"]:
            required.setdefault(check, []).append(change)
        for name in change["models"]:
            for check in policy["model_checks"]:
                model_required.setdefault((name, check), []).append(change)
            selected = {change.get("gate", name): change.get("gate_checks", [])}
            if change["class"] in ("unknown", "validation_policy", "model_unknown"):
                # Broad fallback includes declared checks outside release cohorts,
                # of every gate the model declares.
                selected = {gate: [check["id"] for check in row["gate"].get("checks", [])]
                            for gate, row in gates(name, catalog.models[name])}
            for gate, checks in selected.items():
                for check in checks:
                    gate_required.setdefault(gate, {}).setdefault(check, []).append(change)
    # Documentation can make iteration fall back to --workspace without imposing
    # model contracts. Use only contract inputs for CPU consumer supplementation.
    contract_paths = sorted({c["path"] for c in changes if c["class"] != "documentation" and c["path"] != "*"})
    if contract_paths or all_checks:
        consumers = package_scope if contract_paths == paths else graph.select(contract_paths, broad=all_checks,
            kernel_index=kernel_index, kernel_changes=[c for c in changes if c["path"] in contract_paths])
        definitions = [(None, key, definition) for key, definition in catalog.data["checks"].items()]
        definitions += [(name, key, expand(definition, model_parameters(name, row)))
                        for name, row in catalog.models.items() if row["inventory"]
                        for key, definition in catalog.data["model_checks"].items()]
        for name, key, definition in definitions:
            if definition["kind"] != "cpu_contract":
                continue
            reasons = []
            for package in sorted(cargo_test_packages(definition) & set(consumers["packages"])):
                for reason in consumers["reasons"][package]:
                    chain = "; ".join(edge.get("reason") or
                        f'{edge.get("from", "*")} -> {edge.get("to", package)} ({edge["kind"]})'
                        for edge in reason["chain"])
                    reasons.append({"path": reason["input"], "class": "cargo_consumer",
                                    "reason": package + ": " + chain, "models": [name] if name else []})
            if reasons:
                target = required if name is None else model_required
                target.setdefault(key if name is None else (name, key), []).extend(reasons)
    if kernel_consumers:
        required.setdefault("kernel_build", []).extend(
            {"path": r["input"], "class": "local_kernel", "reason": c["identity"]["package"] + ": " + r["reason"]}
            for c in kernel_consumers for r in c["reasons"])
    obligations = [obligation("shared:" + key, catalog.data["checks"][key], reasons)
                   for key, reasons in sorted(required.items())]
    for item in obligations:
        if item.get("locks"):
            item["locks"] = lock_identities(root, item["locks"], package_scope["kernel_consumers"])
    kernel_check = next((item for item in obligations if item["id"] == "shared:kernel_build"), None)
    if kernel_check:
        kernel_check["consumers"] = [c["identity"] for c in kernel_consumers]
        kernel_check["locks"] = [{**v["lock"], "consumer": c["identity"], "membership": v["status"]}
                                 for c in kernel_consumers for v in c["variants"]]
        for consumer in kernel_consumers:
            package = consumer["workspace_package"]
            if package:
                kernel_check["commands"].append(["cargo", "build", "-p", package, "--locked", "--offline"])
            elif consumer["mapping"] == "external" and consumer["build_features"] is not None:
                command = ["python3", "-B", "-m", "tools.model_sdk", "build",
                    "--workspace", "{external_model_workspace}", "--target", "{external_sdk_target}",
                    "--logs", "{external_sdk_logs}"]
                for feature in consumer["build_features"]:
                    command.extend(["--feature", feature])
                kernel_check["commands"].append(command)
            else:
                kernel_check["prerequisites"].append("Resolve build workflow and ownership for " + consumer["identity"]["package"])
        kernel_check["reproduce"] = [shlex.join(command) for command in kernel_check["commands"]]
    affected = sorted({name for change in changes for name in change["models"]} | {name for name, _ in model_required})
    profiles = []
    gaps = list(catalog.issues) + list(graph.issues) + package_scope["kernel_gaps"]
    gaps += [c["identity"]["package"] + ": external qualification metadata is unresolved; declarations cannot grant a boundary exemption"
             for c in kernel_consumers if c["mapping"] != "workspace"]
    if any(c["class"] in ("shared_kernel", "unknown", "validation_policy", "abi", "package_schema",
                           "host", "cuda_host", "packing", "frontend") and
           not c["path"].startswith("models/") for c in changes):
        gaps.append("External consumer admission beyond known inventories is unresolved: core models and the maintained "
                    "SDK example do not enumerate every publisher; reuse SDK source, inventory and graph validation "
                    "before claiming complete external coverage")
    for name in affected:
        row = catalog.models[name]
        previous = row.get("previous_profiles", {})
        for profile, settings in {**previous, **row["profiles"]}.items():
            removed = profile not in row["profiles"]
            profiles.append({"model": name, "profile": profile, "settings": settings,
                             "previous_settings": previous.get(profile), "removed": removed,
                             "qualification": "unresolved compatibility/refusal evidence" if removed else "not assessed by this report"})
            if removed:
                gaps.append(f"{name}/{profile}: removed profile compatibility/refusal evidence is unresolved; "
                            "a current serving receipt cannot cover removal")
        if not row["profiles"]:
            gaps.append(name + ": unresolved profiles; no model may be silently excluded")
            obligations.append(obligation(name + ":unresolved", {
                "claim": "Resolve external/new model inventory and validation metadata", "kind": "unknown",
                "commands": [], "observes": [], "targets": [], "prerequisites": ["SDK-validated inventory and owned validation catalog"],
                "standing": False, "gap": "Incomplete model metadata requires broad checks"},
                [c for c in changes if name in c["models"]]))
    for (name, check), reasons in sorted(model_required.items()):
        row = catalog.models[name]
        definition = expand(catalog.data["model_checks"][check], model_parameters(name, row))
        if check == "kernel_oracle":
            definition["commands"] = row["data"].get("kernel_commands", [])
            definition["locks"] = lock_identities(root, row["data"].get("locks", []), package_scope["kernel_consumers"])
        gate_key = "frontend" if check == "quality" and all(r["class"] == "frontend" for r in reasons) else check
        selected_gate = row.get("change_classes", {}).get(gate_key)
        if selected_gate is None and gate_key in gate_change_classes():
            # An unmapped class keeps every gate check rather than dropping them.
            selected_gate = [gate_check["id"] for gate_check in row["gate"].get("checks", [])]
            gaps.append(f"{name}: qual.json maps no gate checks for the {gate_key} change class; "
                        "every gate check stays selected")
        if selected_gate is not None:
            for gate_check in selected_gate:
                gate_required.setdefault(name, {}).setdefault(gate_check, []).extend(reasons)
            definition["gate_checks"] = selected_gate
            if not selected_gate:
                definition["gap"] = "Current release gate has no check for this capability; standalone evidence must be bound and reviewed"
        # Each checkpoint variant's gate, by its own change classes.
        for gate, variant in list(gates(name, row))[1:]:
            selected = variant["change_classes"].get(gate_key)
            if selected is None and gate_key in gate_change_classes():
                selected = [gate_check["id"] for gate_check in variant["gate"].get("checks", [])]
                gaps.append(f"{gate}: its declaration maps no gate checks for the {gate_key} change class; "
                            "every gate check stays selected")
            for gate_check in selected or []:
                gate_required.setdefault(gate, {}).setdefault(gate_check, []).extend(reasons)
        definition["possible_capabilities"] = row["data"].get("capabilities", [])
        item = obligation(name + ":" + check, definition, reasons, sorted(row["profiles"].keys() | row.get("previous_profiles", {}).keys()))
        capability = definition.get("capability")
        if capability:
            item["capability_scope"] = "Potential model use and profile settings; absent verified package capabilities never narrow scope"
            enabled = any(p.get("prefix_cache") if capability == "prefix" else p.get("speculation", {}).get("tokens", 0)
                          for p in [*row["profiles"].values(), *row.get("previous_profiles", {}).values()])
            # Context may expand potential use, but asserted or missing package
            # metadata can never remove declared profile/catalog requirements.
            enabled = enabled or capability in (context or {}).get("package_capabilities", {}).get(name, [])
            if capability not in definition["possible_capabilities"]:
                gap = name + ": " + capability + " absence is unverified; retain the capability obligation"
                gaps.append(gap)
                item["reasons"].append({"path": "*", "class": "capability", "reason": gap})
                if enabled:
                    gaps.append(name + ": contradictory capability metadata; keep all profiles selected")
        obligations.append(item)
    release_plans = []
    for name, checks in sorted(gate_required.items()):
        row = gate_row(catalog, name)
        config = row["gate"]
        if not config:
            for identifier, reasons in sorted(checks.items()):
                gap = f"{name}: requested check {identifier} has no cohort metadata; execution coverage is unresolved"
                gaps.append(gap)
                obligations.append(unresolved_gate(name, identifier, row, reasons, gap))
            continue
        cohort = "quick" if set(checks) <= set(config["profiles"]["quick"]["checks"]) else "full"
        description = describe_gate(config, cohort)
        command = ["python3", "-B", "tools/qual/release_gate.py", "--model", name,
                   "--profile", cohort, "--url", "{candidate_url}", "--inputs", "{external_inputs_json}",
                   "--receipts", "{external_receipts_dir}"]
        release_plans.append({"model": name, "cohort": cohort, "selected_checks": sorted(checks),
                              "additional_checks": sorted(set(description["selected_checks"]) - checks.keys()),
                              "command": command, "reproduce": shlex.join(command), "description": description,
                              "note": "Cohort is quick/full, not a serving profile. No selective execution or prerequisite preparation."})
        absent = checks.keys() - {check["id"] for check in description["checks"]}
        for identifier in sorted(absent):
            gap = f"{name}: requested check {identifier} is absent from the {cohort} cohort; execution coverage is unresolved"
            gaps.append(gap)
            obligations.append(unresolved_gate(name, identifier, row, checks[identifier], gap))
        for check in description["checks"]:
            if check["id"] not in checks:
                continue
            kind = "service_performance" if check["id"] in row["change_classes"].get("service", []) else "model_quality"
            definition = {"claim": "Release policy predicates: " + check["id"], "kind": kind,
                          "commands": [command], "observes": [p if (root / p).is_file() else p.rstrip("/") + "/*" for p in row["readers"][check["id"]]],
                          "targets": [check["module"]], "artifacts": ["observed candidate and reference deployments"],
                          "prerequisites": check["prerequisites"], "standing": False,
                          "gate": {"model": name, "id": check["id"], "rules": check["rules"],
                                   "policy_sha256": description["policy_sha256"],
                                   "implementation_sha256": sha(json.dumps({p: file_digest(root / p)
                                       for p in row["readers"][check["id"]] if p.endswith(".py") and (root / p).is_file()}, sort_keys=True).encode())}}
            obligations.append(obligation(name + ":gate:" + check["id"], definition, checks[check["id"]],
                                          sorted(row["profiles"].keys() | row.get("previous_profiles", {}).keys())))
    inputs = []
    for path in paths:
        entry = {"path": path}
        file = root / path
        if file.is_symlink():
            entry["identity"] = "symlink: unresolved"
        elif file.is_file() and file.stat().st_size <= 8_000_000:
            entry["sha256"] = file_digest(file)
        else:
            entry["identity"] = "deleted, unavailable or beyond report hash bound"
        inputs.append(entry)
    obligations.sort(key=lambda item: item["id"])
    report = {"schema": SCHEMA, "mode": "report-only", "qualified": False,
              "inputs": inputs, "change_identity_sha256": sha(json.dumps(inputs, sort_keys=True).encode()),
              "source_identity": source_identity,
              "artifacts": package_scope, "changes": changes, "profiles": profiles,
              "obligations": obligations, "release_plans": release_plans,
              "gaps": sorted(set(gaps)), "limits": [
                  "Report exit 0 means produced, never validated or qualified",
                  "CUDA membership is bounded to supplied consumers and selections; historical and incomplete evidence adds conservative edges",
                  "Kernel declarations and local build consistency do not qualify numerical behavior or deployments",
                  "Execution compatibility and checkpoint qualification are separate",
                  "Fake-host, device correctness, fixed-batch exactness and service performance are distinct",
                  "Bounded source scanning cannot prove arbitrary build scripts have no other inputs"]}
    packages = (["--workspace"] if package_scope["workspace"] else
                [arg for name in package_scope["packages"] for arg in ("-p", name)])
    report["iteration"] = {"commands": [["cargo", "test", *packages, "--locked", "--offline"]] if packages else [],
                           "features": "default", "evidence": "not_run",
                           "limit": "Potential package consumers, including conservative dev chains; this command alone does not establish every observed contract"}
    if context is not None or receipts:
        sys.path.insert(0, str(change_scope.ROOT / "tools/qual"))
        import validation_evidence
        validation_evidence.attach(report, context or {}, receipts)
    commands = {tuple(command) for item in obligations if item["evidence"]["selection"] == "selected" for command in item["commands"]}
    report["commands"] = [{"id": f"command-{index + 1}", "argv": list(command), "reproduce": shlex.join(command),
                           "obligations": [item["id"] for item in obligations if list(command) in item["commands"]],
                           "scope": "Template only: supply matching model/profile evidence contexts for each claim"}
                          for index, command in enumerate(sorted(commands))]
    report["counts"] = {"packages": len(package_scope["packages"]), "profiles": len(profiles),
                        "obligations": len(obligations), "unique_commands": len(commands),
                        "command_references": sum(len(item["commands"]) for item in obligations)}
    return report


def render_text(report):
    lines = ["Validation report (no checks executed; bash ci/cpu.sh runs the full CPU checks)",
             "Changed inputs: " + (", ".join(p["path"] for p in report["inputs"]) or "none"),
             "Packages: " + (", ".join(report["artifacts"]["packages"]) or "none"),
             "Profiles: " + (", ".join(p["model"] + "/" + p["profile"] for p in report["profiles"]) or "none")]
    lines += ["Iteration: " + shlex.join(command) for command in report["iteration"]["commands"]]
    for consumer in report["artifacts"].get("kernel_consumers", []):
        identity = consumer["identity"]
        lines.append("Kernel consumer: " + identity["source"] + "/" + identity["package"] + "/" + (identity["selection"] or "unreported selection")
                     + "; " + consumer["mapping"] + "; " + consumer["membership_decision"])
        for variant in consumer["variants"]:
            lines.append("  Report: " + variant["path"] + "; " + variant["origin"] + "; " + variant["status"]
                         + "; " + json.dumps(variant["completeness"], sort_keys=True))
            lines.append("  Lock: " + json.dumps(variant["lock"], sort_keys=True))
            lines += ["  Cubin: " + json.dumps(cubin, sort_keys=True) for cubin in variant["cubins"]]
            lines += ["  Gap: " + issue for issue in variant["issues"]]
        lines += ["  Source: " + json.dumps(source, sort_keys=True) for source in consumer["affected_sources"]]
        lines += ["  Why: " + reason["input"] + ": " + reason["kind"] + ": " + reason["reason"] for reason in consumer["reasons"]]
    for item in report["obligations"]:
        status = item["evidence"]
        lines += ["", item["id"] + ": " + item["claim"],
                  "  " + " / ".join(str(status[k]) for k in ("selection", "execution", "result", "applicability")),
                  "  Kind: " + item["kind"] + ("; covered by standing CPU gate when actually exercised" if item["standing"] else ""),
                  "  Why: " + "; ".join(sorted({r["path"] + " -> " + r["class"] + ": " + r["reason"] for r in item["reasons"]})),
                  "  Observes: " + json.dumps(item["observation"], sort_keys=True)]
        lines.append("  Required identity (" + ", ".join(item["required_identity"]["arms"]) + "): " + ", ".join(item["required_identity"]["dimensions"]))
        lines += ["  Run: " + command["id"] for command in report["commands"] if item["id"] in command["obligations"]]
        lines += ["  Needs: " + requirement for requirement in item["prerequisites"]]
        if item["coverage_gap"]:
            lines.append("  Gap: " + item["coverage_gap"])
        if item.get("limits"):
            lines.append("  Limit: " + item["limits"])
        lines += ["  Evidence: " + reason for reason in status["reasons"]]
        for lock in item.get("locks", []):
            families = lock.get("families")
            digest = lock.get("sha256") or ("directory" if families is not None and not lock.get("missing") else "missing")
            lines.append("  Lock: " + lock["path"] + " " + digest + "; " + lock["membership"])
            lines += ["    Family: " + family["family"] + " " + family["path"] + " " + (family.get("sha256") or "missing")
                      + ("; provenance differs: " + ", ".join(family["provenance_changed"])
                         if family.get("provenance_changed") else "")
                      for family in families or []]
    for plan in report["release_plans"]:
        lines.append("Cohort " + plan["model"] + "/" + plan["cohort"] + " also runs: " + (", ".join(plan["additional_checks"]) or "no extra checks"))
    lines += ["", "Commands (deduplicated templates, not interchangeable evidence contexts):"]
    lines += [command["id"] + ": " + command["reproduce"] for command in report["commands"]]
    lines += ["Gap: " + gap for gap in report["gaps"]]
    lines += ["Limit: " + limit for limit in report["limits"]]
    lines.append("Counts: " + json.dumps(report["counts"], sort_keys=True))
    return "\n".join(lines) + "\n"


def write_report(report, format="text", output=None, root=change_scope.ROOT):
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n" if format == "json" else render_text(report)
    if output is None:
        print(text, end="")
    else:
        path = Path(output).absolute()
        if (path.resolve().is_relative_to(Path(root).resolve()) or path.is_symlink()
                or any((parent / ".git").exists() for parent in path.resolve().parents)):
            raise ValueError("report output must be an explicit external location outside the repository")
        # Do not create prerequisite directories or replace an existing receipt.
        with path.open("x") as stream:
            stream.write(text)

"""Consumer policy over producer reports, Cargo and admitted SDK inventories."""
from collections import deque
from fnmatch import fnmatchcase
from pathlib import Path
import tomllib

import kernel_reports as kr
import validation_inventory


def version(manifest, workspace):
    package = tomllib.loads(Path(manifest).read_text())["package"]
    value = package["version"]
    if isinstance(value, dict) and value.get("workspace"):
        value = tomllib.loads((Path(workspace) / "Cargo.toml").read_text())["workspace"]["package"]["version"]
    return kr.string(value)


def consumer_key(context):
    return context["consumer"]["source"], context["consumer"]["package"], context["selection"]


class Index:
    def __init__(self, metadata, graph, catalog, current=(), previous=(), *, limits=None):
        self.root = catalog.root
        self.config = catalog.data.get("kernel_reports", {})
        self.source = self.config.get("source", "chopinfer")
        self.bindings = {self.source: self.root}
        self.expected, self.mapping, self.issues = {}, {}, []
        for key, package in sorted(graph.packages.items()):
            try:
                v = package.get("version") or version(package["manifest_path"], self.root)
            except (OSError, ValueError, KeyError, TypeError):
                v = "unknown"
            identity = (self.source, package["name"] + "@" + v)
            self.mapping.setdefault(identity, []).append(package["name"])
        kernel_build = {key for key, p in graph.packages.items() if p["name"] == "kernel-build"}
        for node in (metadata.get("resolve") or {}).get("nodes", []):
            if node["id"] not in graph.packages:
                continue
            deps = node.get("deps")
            if deps is None:
                deps = [{"pkg": p, "dep_kinds": [{"kind": "unknown"}]} for p in node.get("dependencies", [])]
            if any(dep["pkg"] in kernel_build and any(k.get("kind", "unknown") in ("build", "unknown")
                   for k in dep.get("dep_kinds", [{"kind": "unknown"}])) for dep in deps):
                name = graph.packages[node["id"]]["name"]
                for identity, names in self.mapping.items():
                    if name in names:
                        self.expected[identity] = {"workspace_package": name, "namespaces": [self.source],
                                                   "sdk_dependencies": [], "kernel_paths": []}
        loader = validation_inventory.loader(Path(__file__).resolve().parents[1])
        for external in self.config.get("external", []):
            directory = self.root / external["directory"]
            self.bindings[external["source"]] = directory
            try:
                inventory = loader.load_model(self.root, directory)
                package = inventory.packages[external["role"]]
                manifest = self.root / package.manifest
                identity = (external["source"], package.name + "@" + version(manifest, directory))
                actual = tomllib.loads(manifest.read_text())
                declared = set(external["sdk_dependencies"])
                surface = set(actual.get("build-dependencies", {})) | set(actual.get("dev-dependencies", {}))
                kr.require(declared == surface, "external SDK dependency surface differs from manifest")
                selections = external.get("selections")
                if selections is None:
                    selections = {external["selection"]: ["cuda-probe"]}
                kr.require(isinstance(selections, dict) and bool(selections), "external selections are required")
                for selection, features in selections.items():
                    kr.string(selection)
                    kr.require(isinstance(features, list), "selection features must be a list")
                    for feature in features:
                        kr.string(feature)
                self.expected[identity] = {"workspace_package": None, "namespaces": [self.source, external["source"]],
                    "sdk_dependencies": sorted(declared), "kernel_paths": external["kernel_paths"],
                    "directory": external["directory"], "selections": selections, "inventory": "admitted"}
            except (loader.InventoryError, OSError, ValueError, KeyError, TypeError) as error:
                self.issues.append(f"external inventory {external['source']}: {error}")
                # Even broken inventories keep their declared source domain.
                self.expected[(external["source"], "unresolved@unknown")] = {
                    "workspace_package": None, "namespaces": [self.source, external["source"]],
                    "sdk_dependencies": external["sdk_dependencies"], "kernel_paths": external["kernel_paths"],
                    "directory": external["directory"], "inventory": "unresolved"}
        self.reader = kr.Reader(limits)
        self.variants = self.reader.load(current, previous, self.bindings)
        self.issues += self.reader.gaps
        self.consumers, self.membership = {}, {}
        for variant in self.variants:
            report = variant["report"]
            if not report or not report["context"]:
                if report:
                    self.issues.append("report lacks consumer ownership: " + variant["path"])
                continue
            key = consumer_key(report["context"])
            row = self.row(key)
            if row["mapping"] == "unresolved":
                variant["issues"].append("consumer not bound to current Cargo or admitted external inventory")
                if variant["status"] == "current":
                    variant["status"] = "stale"
            row["variants"].append(variant)
            for cubin in report["cubins"]:
                for member in cubin["membership"]:
                    identity = kr.source_key(report["files"][member["file"]]["identity"])
                    self.membership.setdefault(identity, []).append((key, variant, cubin))
        for identity, expected in self.expected.items():
            matching = [key for key in self.consumers if key[:2] == identity]
            selections = expected.get("selections", {})
            for selection in selections:
                self.row((*identity, selection))
            if not matching and not selections:
                self.row((*identity, None))
        # Different current variants are additive. Conflicting audited identities
        # for the same named cubin cannot establish exclusion coverage.
        for row in self.consumers.values():
            observed = {}
            for variant in row["variants"]:
                if variant["status"] == "current":
                    for cubin in variant["report"]["cubins"]:
                        observed.setdefault(cubin["name"], set()).add(kr.serialized(cubin["identity"]))
            row["conflicts"] = sorted(name for name, identities in observed.items() if len(identities) > 1)

    def row(self, key):
        if key not in self.consumers:
            expected = self.expected.get(key[:2], {})
            names = self.mapping.get(key[:2], [])
            workspace = names[0] if len(names) == 1 else None
            mapping = "workspace" if workspace else "external" if expected.get("inventory") == "admitted" else "unresolved"
            self.consumers[key] = {"identity": {"source": key[0], "package": key[1], "selection": key[2]},
                "workspace_package": workspace, "mapping": mapping, "expected": bool(expected),
                "build_features": expected.get("selections", {}).get(key[2]),
                "namespaces": expected.get("namespaces", []), "variants": [], "conflicts": []}
        return self.consumers[key]

    def identities(self, path):
        result = []
        absolute = kr.lexical(self.root / path)
        for owner, root in self.bindings.items():
            root = kr.lexical(root)
            if absolute.is_relative_to(root):
                result.append(("Project", owner, absolute.relative_to(root).as_posix()))
        return result

    def select(self, graph, paths, baseline, changes, broad):
        rows = {key: {**row, "variants": row["variants"]} for key, row in self.consumers.items()}
        for row in rows.values():
            row.update(selected=False, reasons=[], affected_sources=[], affected_cubins=[])

        def add(key, path, kind, reason, *, chain=None, source=None, cubin=None, variant=None):
            row = rows[key]
            row["selected"] = True
            entry = {"input": path, "kind": kind, "reason": reason,
                     "chain": chain or [{"from": path, "to": key[1], "kind": kind}],
                     "conservative": kind != "current membership"}
            if variant:
                entry["report"] = variant["path"]
            if entry not in row["reasons"]:
                row["reasons"].append(entry)
            if source:
                value = dict(zip(("kind", "owner", "path"), source))
                if value not in row["affected_sources"]:
                    row["affected_sources"].append(value)
            if cubin:
                value = {"name": cubin["name"], "recipe_sha256": cubin["recipe_sha256"], "report": variant["path"]}
                if value not in row["affected_cubins"]:
                    row["affected_cubins"].append(value)

        for path in sorted(paths):
            identities = self.identities(path)
            hits = [(identity, hit) for identity in identities for hit in self.membership.get(identity, [])]
            for identity, (key, variant, cubin) in hits:
                kind = "previous membership" if variant["origin"] == "previous" else "current membership" if variant["status"] == "current" else "stale membership"
                add(key, path, kind, "logical SourceIdentity in actual nvcc dependency closure", source=identity, cubin=cubin, variant=variant)
            classes = {c["class"] for c in changes if c["path"] == path}
            models = {m for c in changes if c["path"] == path for m in c["models"]}
            for key, row in rows.items():
                expected = self.expected.get(key[:2], {})
                package = row["workspace_package"]
                real_reasons = [r for r in baseline["reasons"].get(package, []) if r["input"] == path]
                external_path = any(fnmatchcase(path, p) for p in expected.get("kernel_paths", []))
                if external_path:
                    add(key, path, "external selection input", "model-owned kernel declaration or build input")
                # Follow actual owner/reader/Cargo trails, never '*' fallbacks.
                for dependency in expected.get("sdk_dependencies", []):
                    for reason in baseline["reasons"].get(dependency, []):
                        if reason["input"] == path:
                            add(key, path, "external SDK dependency", "declared Rust/build dependency: " + dependency,
                                chain=reason["chain"] + [{"from": dependency, "to": key[1], "kind": "external SDK dependency"}])
                domain = bool(hits) or external_path or bool(classes & {"shared_kernel", "local_kernel", "unknown", "model_unknown", "validation_policy"})
                if not domain:
                    continue
                local = classes & {"local_kernel", "model_unknown"}
                directory = graph.directories.get(graph.by_name.get(package), "") or ""
                own_model = any(directory.startswith("models/" + model + "/") for model in models)
                namespace = {identity[1] for identity in identities}
                reported_namespaces = {s["id"] for v in row["variants"] for s in v["report"]["context"]["sources"]}
                relevant = bool(classes & {"unknown", "validation_policy"}) or external_path
                relevant |= bool(classes & {"shared_kernel"}) and bool(namespace & (set(row["namespaces"]) | reported_namespaces))
                relevant |= bool(local) and (own_model or bool(real_reasons))
                relevant |= any(hit[1][0] == key for hit in hits)
                relevant |= row["mapping"] == "unresolved"
                if not relevant:
                    continue
                reasons = list(self.issues)
                current = [v for v in row["variants"] if v["origin"] == "current"]
                if not current:
                    reasons.append("missing current report")
                for variant in current:
                    if variant["status"] != "current":
                        reasons.extend(variant["issues"] or ["incomplete current report"])
                if row["conflicts"]:
                    reasons.append("conflicting current cubin identities: " + ", ".join(row["conflicts"]))
                if not (self.root / path).exists() and not any(v["origin"] == "previous" for v in row["variants"]):
                    reasons.append("removed input without previous membership coverage")
                for reason in sorted(set(reasons)):
                    add(key, path, "kernel evidence fallback", reason)
        if broad:
            for key in rows:
                add(key, "*", "kernel evidence fallback", "--all")
        # Kernel edges propagate through current Cargo packages only. Old and
        # independently exported identities never become invalid Cargo -p names.
        for row in rows.values():
            if not row["selected"] or not row["workspace_package"]:
                continue
            for reason in row["reasons"]:
                queue = deque([(graph.by_name[row["workspace_package"]], reason["chain"])])
                seen = set()
                while queue:
                    key, chain = queue.popleft()
                    if key in seen:
                        continue
                    seen.add(key)
                    name = graph.packages[key]["name"]
                    baseline["reasons"].setdefault(name, []).append({"input": reason["input"], "chain": chain,
                                                                   "conservative": reason["conservative"]})
                    for target, kind, cfg in sorted(graph.reverse[key]):
                        queue.append((target, chain + [{"from": name, "to": graph.packages[target]["name"], "kind": kind, "target": cfg}]))
        output = []
        for key in sorted(rows, key=lambda key: (key[0], key[1], key[2] or "")):
            row = rows[key]
            row["current_evidence"] = any(v["status"] == "current" for v in row["variants"])
            row["package_selected"] = row["workspace_package"] in baseline["reasons"]
            row["membership_decision"] = ("affected or conservatively retained" if row["selected"] else
                "no membership hit in supplied current coverage" if row["current_evidence"] else "outside changed kernel domain")
            row["variants"] = [self.describe(v) for v in row["variants"]]
            output.append(row)
        baseline["packages"] = sorted(baseline["reasons"])
        baseline["reasons"] = dict(sorted(baseline["reasons"].items()))
        baseline["kernel_consumers"] = output
        baseline["kernel_gaps"] = sorted(set(self.issues))
        baseline["completeness"]["cuda_membership"] = "verified per report variant; missing/stale membership retained within the changed kernel domain"
        return baseline

    @staticmethod
    def describe(variant):
        report = variant["report"]
        lock = {"path": report["lock_path"], "sha256": report["lock_sha256"], "status": report["lock_status"]}
        if "lock_families" in report:
            # A directory lock has no file digest; each family file has its own.
            lock["families"] = [{"family": f["family"], "path": f["lock_path"], "sha256": f["lock_sha256"],
                                 "provenance_path": f["provenance_path"], "cubins": f["cubins"],
                                 "provenance_changed": f["provenance_changed"]} for f in report["lock_families"]]
        return {**{k: v for k, v in variant.items() if k != "report"},
                "completeness": {key: report[key] for key in kr.FLAGS},
                "sources": report["context"]["sources"],
                "lock": lock,
                "cubins": [{key: cubin[key] for key in ("name", "recipe", "recipe_sha256", "identity", "declarations")}
                           for cubin in report["cubins"]]}

"""Check the explicit Rust and Cargo catalog registration seam.

A model that sets `[kernel_lab] catalog = true` in its model.toml registers its
contract crate's catalog (`<contract>::kernels::CATALOG` or, where the catalog
lives in the contract's sm120 module, `<contract>::sm120::CATALOG`) in
crates/kernel-lab/models.rs (which build.rs and the
library both compile, so it stays a checked-in file) and names the crate in
both dependency sections of crates/kernel-lab/Cargo.toml (Cargo cannot
generate those). Each error names the exact line to add or remove."""
from collections import Counter
from pathlib import Path
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.model_sdk import inventory


def check(root=ROOT, models=None):
    models = inventory.load(root) if models is None else models
    if not models:
        return ["admission: no model inventories found"]
    selected = {model.packages["contract"].name for model in models if model.kernel_lab.catalog}
    known = {model.packages["contract"].name for model in models if "contract" in model.packages}
    rust = (root / "crates/kernel-lab/models.rs").read_text()
    rust = re.sub(r"/\*.*?\*/|//[^\n]*", "", rust, flags=re.S)
    catalogs = re.findall(r"&([A-Za-z_][A-Za-z_0-9]*)::(?:kernels|sm120)::CATALOG", rust)
    manifest = tomllib.loads((root / "crates/kernel-lab/Cargo.toml").read_text())
    workspace_path = root / "Cargo.toml"
    workspace = tomllib.loads(workspace_path.read_text()).get("workspace", {}).get("dependencies", {}) if workspace_path.is_file() else {}
    known.update(value.get("package", name) for name, value in workspace.items() if isinstance(value, dict)
                 and str(value.get("path", "")).startswith("models/")
                 and str(value.get("path", "")).endswith("/contract"))
    # The workspace key that names each contract, which a kernel-lab section
    # inherits (`<key>.workspace = true`) and Rust names the crate by.
    keys = {name:name for name in selected}
    keys.update({value.get("package", key):key for key, value in workspace.items() if isinstance(value, dict)
                 and value.get("package", key) in selected})
    sections = {}
    aliases = {name:set() for name in selected}
    for section in ("dependencies", "build-dependencies"):
        observed = set()
        for key, value in manifest.get(section, {}).items():
            value = value if isinstance(value, dict) else {}
            if value.get("workspace"):
                value = workspace.get(key, {})
                value = value if isinstance(value, dict) else {}
            package = value.get("package", key)
            path = str(value.get("path", ""))
            if package in known or (path.startswith("../../models/") and path.endswith("/contract")):
                observed.add(package)
                if package in selected:
                    aliases[package].add(key.replace("-", "_"))
        sections[section] = observed
    errors = []
    expected = {alias for name, names in aliases.items() for alias in (names or {keys[name].replace("-", "_")})}
    observed = set(catalogs)
    for name in sorted(expected - observed):
        errors.append(f"models.rs: missing catalog {name} (append `&{name}::sm120::CATALOG,` "
                      "before `&lab_shapes::CATALOG`)")
    for name in sorted(observed - expected):
        errors.append(f"models.rs: extra catalog {name} (remove its CATALOG entry)")
    for name, count in sorted(Counter(catalogs).items()):
        if count > 1:
            errors.append(f"models.rs: duplicate catalog {name}")
    for section, observed in sections.items():
        for name in sorted(selected - observed):
            errors.append(f"{section}: missing contract {name} (add `{keys[name]}.workspace = true`)")
        for name in sorted(observed - selected):
            errors.append(f"{section}: extra contract {name} (remove its entry)")
    if workspace_path.is_file():
        inherited = {value.get("package", key) for key, value in workspace.items() if isinstance(value, dict)}
        for model in models:
            name = model.packages["contract"].name if model.kernel_lab.catalog else None
            if name is not None and name not in inherited:
                errors.append(f"Cargo.toml: missing workspace dependency {name} "
                              f"(add `{name} = {{ path = \"{model.directory}/contract\" }}` to [workspace.dependencies])")
    return errors


if __name__ == "__main__":
    errors = check()
    for error in errors:
        print("kernel-lab inventory: " + error, file=sys.stderr)
    sys.exit(bool(errors))

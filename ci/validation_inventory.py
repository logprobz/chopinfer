"""Inventory adapter. The model SDK owns inventory admission and boundary policy.

Keep this seam separate from behavioral selection. External inventories need
an SDK-validated model name, role/package IDs, profile paths and capabilities;
unknown external inventories cannot grant a host-boundary exemption.
"""
import importlib.util
from pathlib import Path
import sys


def loader(root):
    path = Path(root) / "tests/model_inventory.py"
    spec = importlib.util.spec_from_file_location("validation_model_inventory", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read(root, metadata):
    module = loader(root)
    models, issues = [], []
    for directory in sorted((Path(root) / "models").iterdir()):
        if not directory.is_dir():
            continue
        try:
            models.append(module.load_model(Path(root), directory))
        except module.InventoryError as error:
            issues.append(str(error))
    try:
        module.check_metadata(models, metadata, root)
    except (module.InventoryError, KeyError, ValueError) as error:
        issues.append("inventory/Cargo mismatch: " + str(error))
    return models, issues

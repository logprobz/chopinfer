#!/usr/bin/env python3
"""Core entry point for the shared model inventory parser."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from model_sdk.inventory import (  # noqa: E402
    GeometryRecipe, InventoryError, Model, Package, check_metadata, load, load_model, main, packages,
    check_files, compile_files, destination, file_inputs, fixture_root, kernel_locks, release_plan, released, shipped,
)

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except InventoryError as error:
        print(f"model inventory: {error}", file=sys.stderr)
        sys.exit(1)

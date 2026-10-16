"""Read the encoder declarations used by qualification tools, without torch."""

import json
from pathlib import Path

from common import ROOT

SCHEMA = "chopinfer.encoder.v1"
TOWERS = {"image": "vision", "video": "vision", "audio": "audio"}


def read(path):
    path = Path(path)
    value = json.loads(path.read_text())
    if value.get("schema") != SCHEMA:
        raise ValueError(f"{path}: expected {SCHEMA}")
    kinds = {recipe["tag"].split(".")[0] for recipe in value["input"]["recipes"]}
    towers = {TOWERS.get(kind) for kind in kinds}
    if len(towers) != 1 or None in towers:
        raise ValueError(f"{path}: recipes do not identify one encoder tower")
    return {**value, "path": path, "kinds": kinds, "tower": towers.pop()}


def for_model(model):
    # Model names are directory components, never paths supplied by a run.
    if not isinstance(model, str) or not model or Path(model).name != model or model in (".", ".."):
        raise ValueError(f"invalid encoder model: {model!r}")
    paths = sorted((ROOT / "models" / model / "contract/encoders").glob("*.json"))
    values = [read(path) for path in paths if json.loads(path.read_text()).get("schema") == SCHEMA]
    if not values or any(value["model"] != model for value in values):
        raise ValueError(f"{model}: no matching encoder declarations")
    if len({value["tower"] for value in values}) != len(values):
        raise ValueError(f"{model}: duplicate encoder towers")
    return values


def parity_rules(values):
    rows, codes, every_row, direction = [], [], [], []
    for value in values:
        row = f"enc.{value['tower']}.out"
        rows.append(row)
        quantizers = [i for i, stage in enumerate(value["stages"])
                      if stage["op"].split(":")[-1].startswith("quantize.")]
        if quantizers:
            if len(quantizers) != 1:
                raise ValueError("parity requires one code stream per encoder")
            code = f"enc.{value['tower']}.x.codes"
            codes.append(code)
            group = 1
            for stage in value["stages"][quantizers[0] + 1:]:
                if stage["op"] == "merge.consecutive.v1":
                    group *= stage["rows"]
            every_row.append(f"{row}:{code}/{group}")
        elif value["tower"] == "vision":
            direction.append(row)
        else:
            every_row.append(row)
    if len(codes) > 1:
        raise ValueError("parity receipt supports one code stream")
    return {"rows": tuple(rows), "codes": next(iter(codes), None),
            "every_row": tuple(every_row), "direction_scale": tuple(direction)}


def unqualified(values, capabilities):
    """[(declaration, reason)] of each encoder whose every input kind the
    qualification declaration's `capabilities.unqualified_media` names:
    served, but not qualified in-tree, so encoder parity requires none of
    its rows. An encoder with one qualified kind stays required."""
    media = (capabilities or {}).get("unqualified_media") or {}
    return [(value, "; ".join(sorted({media[kind] for kind in value["kinds"]})))
            for value in values if value["kinds"] and value["kinds"] <= set(media)]

"""Load a model's qualification declaration, `models/<m>/qual.json`.

The declaration is the one source of a model's release gate: its cohorts,
change classes, checks with their limits, references and inputs. `load`
validates it and compiles it to the `chopinfer-release-gates/v1` structure
release_gate.py runs, field for field and in the same key order, so the
recorded `policy.json` and `policy_sha256` keep that layout and digest.
Other configs (external declarations, the mechanics fixture) pass through
unchanged.

The rules are implemented once, in Rust (crates/runtime-eval/src/
declaration.rs): `validate` sends the declaration to `chopinfer eval
declaration`, which refuses a missing, unknown or mistyped field with its
path. This module keeps the paths and names of declarations and the gate
layout. Schema evolution (tools/qual/README.md#model-declarations): v1 grows
only by optional fields whose absence keeps the behavior before them, and any
change in the meaning of a field takes a new schema version.
"""

from functools import lru_cache
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).parent
SCHEMA = "chopinfer-qual-model/v1"
GATE_SCHEMA = "chopinfer-release-gates/v1"
NAME = "qual.json"
# The change classes ci/validation.py selects gate checks for; each must be mapped.
CHANGE_CLASSES = ("frontend", "quality", "service", "speculation", "prefix")
# The checkpoint pin fields a reference's provenance can be held to.
CHECKPOINT_KEYS = ("repository", "revision", "served_model_name", "source_lock_sha256", "checkpoint_identity")

# The CPU fixture self-test config; it names no model and keeps the gate layout.
MECHANICS = Path(__file__).with_name("gates") / "mechanics.json"
# The root of external variants (`--variant-dir`): `<root>/<model>/<variant>.json`
# overlays a model's qual.json from outside the tree (tools/qual/README.md#external-variants).
VARIANTS_ENV = "CHOPINFER_QUAL_VARIANTS"


def variant_root():
    """The external variant root `CHOPINFER_QUAL_VARIANTS` names, or None."""
    import os
    value = os.environ.get(VARIANTS_ENV)
    return Path(value).resolve() if value else None


def select_variant_root(directory):
    """Select `directory` (`--variant-dir`) as the external variant root for
    this process and every tool it runs; None leaves the environment's."""
    import os
    if directory is not None:
        os.environ[VARIANTS_ENV] = str(Path(directory).resolve())


def overlay(model, variant):
    """The external variant's overlay `<variant root>/<model>/<variant>.json`,
    or None when no variant root is selected or it holds none."""
    root = variant_root()
    if root is None or not plain(model) or not plain(variant):
        return None
    file = root / model / f"{variant}.json"
    return file if file.is_file() else None


def external(config):
    """Whether `config` is an external variant's: its expansion names the
    overlay it came from (`variant_source`)."""
    return isinstance(config, dict) and "variant_source" in config


def require_variant_root(config):
    """The variant root an external variant's paths resolve under; refused
    when none is selected, since its lock and references live there."""
    root = variant_root()
    if root is None:
        raise ValueError(f"{config['model']}.{config['variant']} is an external variant: select its root with "
                         f"--variant-dir or {VARIANTS_ENV}")
    return root


def path(model, root=ROOT, variant=None):
    """The declaration of a model in the workspace, or of one of its
    variants: `qual.VARIANT.json` beside `qual.json`, which qualifies another
    checkpoint of the model against its own references, or another serving
    configuration its own gate run reaches."""
    return Path(root) / "models" / model / (NAME if variant is None else f"qual.{variant}.json")


def plain(name):
    """Whether `name` is a plain model or variant name."""
    return isinstance(name, str) and name.replace("_", "").replace("-", "").isalnum()


def split(name):
    """(model, variant) of a `--model` name: MODEL or MODEL.VARIANT."""
    model, dot, variant = name.partition(".")
    if not plain(model) or (dot and not plain(variant)):
        raise ValueError("--model must name a model declaration, as MODEL or MODEL.VARIANT")
    return model, variant or None


def require_rooted(file, root):
    """Refuse a model or checkpoint variant declaration selected from
    outside both this workspace and the reference root `root`. Its reference
    lock and input policy are found by its model name, so a declaration from
    another tree would be scored against no lock, or another declaration's
    lock. Other configs pass."""
    file = Path(file).resolve()
    if (not re.fullmatch(r"qual(?:\.(.+))?\.json", file.name) or len(file.parents) < 2
            or file.parents[1].name != "models"):
        return
    relative = Path("models", file.parent.name, file.name)
    if not any((Path(tree) / relative).is_file() and (Path(tree) / relative).resolve() == file
               for tree in (ROOT, root)):
        raise ValueError(f"{file} is not the declaration under this workspace or the reference root {root}; "
                         "select it from there, or set CHOPINFER_QUAL_MODEL_ROOT (release_gate.py "
                         "--model-root) to its tree")


def named(name):
    """The config `release_gate.py --model NAME` reads: the mechanics fixture,
    a model's declaration (MODEL) or one of its variants (MODEL.VARIANT): a
    configuration of its qual.json, a `qual.VARIANT.json` beside it, or an
    external variant under the selected variant root."""
    if name == "mechanics":
        return MECHANICS
    model, dot, variant = name.partition(".")
    if dot and plain(model) and plain(variant):
        if variant in configurations(model) or overlay(model, variant) is not None:
            return materialized(model, variant)
        return path(model, variant=variant)
    return path(name)


def configurations(model, root=ROOT):
    """The configurations a model's qual.json declares, by name, with what
    the runner reads of each (how it is served)."""
    file = path(model, root)
    if not file.is_file():
        return {}
    return json.loads(file.read_text()).get("configurations", {})


def materialized(model, config, root=ROOT):
    """A file holding the declaration of `config`, expanded by `chopinfer eval
    declaration` from the model's qual.json (with an external variant's
    overlay, under the selected variant root), for readers that take a file
    (CHOPINFER_QUAL_CONFIG), written again on each call. It lives outside
    the tree: `<cache>/<digest>/<model>/qual.<config>.json`, the digest of
    qual.json and of the variant root."""
    import hashlib
    import os
    import tempfile
    import kinds
    base = path(model, root)
    variants = variant_root()
    digest = hashlib.sha256(base.read_bytes() + str(variants or "").encode()).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / "chopinfer-qual-declarations"
    file = cache / digest / model / f"qual.{config}.json"
    done = subprocess.run([str(kinds.eval_binary()), "eval", "declaration", "--model", str(base.parent),
                           "--variant", config, *(["--variant-dir", str(variants)] if variants else [])],
                          capture_output=True, text=True, check=False)
    if done.returncode:
        raise DeclarationError(done.stderr.strip().removeprefix("eval declaration: "))
    file.parent.mkdir(parents=True, exist_ok=True)
    staging = file.with_name(f".{file.name}.{os.getpid()}")
    staging.write_text(json.dumps(json.loads(done.stdout), indent=2) + "\n")
    staging.replace(file)
    return file


def variants(model, root=ROOT):
    """The variants a model declares beside its qual.json, by name."""
    return sorted(match.group(1) for file in (Path(root) / "models" / model).glob("qual.*.json")
                  if (match := re.fullmatch(r"qual\.(.+)\.json", file.name)))


def shipped(root=ROOT):
    """Every config in the workspace: the mechanics fixture, each model's
    declaration and each variant's."""
    return [MECHANICS, *sorted(Path(root).glob("models/*/" + NAME)),
            *sorted(Path(root).glob("models/*/qual.*.json"))]


class DeclarationError(ValueError):
    """A declaration field is missing or malformed; the message names it."""


def fail(field, message):
    raise DeclarationError(f"{field}: {message}")


@lru_cache(maxsize=256)
def validated(text):
    """`chopinfer eval declaration`'s verdict on one declaration's JSON text:
    None, or the refusal naming its field."""
    import kinds
    done = subprocess.run([str(kinds.eval_binary()), "eval", "declaration", "--root", str(ROOT)], input=text,
                          capture_output=True, text=True, check=False)
    if done.returncode == 0:
        return None
    message = done.stderr.strip()
    if done.returncode != 2 or not message.startswith("eval declaration: "):
        raise RuntimeError(f"chopinfer eval declaration exited {done.returncode}: {message}")
    return message.removeprefix("eval declaration: ")


def non_finite(value, where=""):
    """The path of the first non-finite number in `value`, or None."""
    if isinstance(value, float) and not math.isfinite(value):
        return where
    items = (value.items() if isinstance(value, dict) else
             ((f"[{index}]", item) for index, item in enumerate(value)) if isinstance(value, list) else ())
    for key, item in items:
        path = f"{where}{key}" if key.startswith("[") else f"{where}.{key}" if where else key
        if found := non_finite(item, path):
            return found
    return None


def validate(declaration):
    """Refuse a malformed declaration with a message naming the field."""
    if not isinstance(declaration, dict):
        fail("declaration", "expected an object")
    if where := non_finite(declaration):
        # JSON holds no such number, so only a Python value can: name it here.
        fail(where, "must be a finite number")
    text = json.dumps(declaration)
    if refusal := validated(text):
        raise DeclarationError(refusal)


def compile_gate(declaration):
    """The release gate config the declaration states, in its legacy layout."""
    checks = [{"id": name, **{"profile_args" if key == "cohort_args" else key: value
                              for key, value in check.items()}}
              for name, check in declaration["checks"].items()]
    qualification = dict(declaration["qualification"])
    if "references" in declaration:
        qualification["references"] = declaration["references"]
    config = {"schema": GATE_SCHEMA, "model": declaration["model"], "profiles": declaration["cohorts"],
              "checks": checks}
    if "limitations" in declaration:
        config["limitations"] = declaration["limitations"]
    config.update(receipts=declaration["receipts"], inputs=declaration["inputs"], qualification=qualification)
    if "checkpoint" in declaration:
        config["checkpoint"] = declaration["checkpoint"]
    if "variant" in declaration:
        config["variant"] = declaration["variant"]
    if "variant_source" in declaration:
        config["variant_source"] = declaration["variant_source"]
    if "capabilities" in declaration:
        # Encoder parity reads the media served but not qualified in-tree.
        config["capabilities"] = declaration["capabilities"]
    if "kv_offload" in declaration:
        config["kv_offload"] = declaration["kv_offload"]
        # The offload scorer reads the speculation profile when it judges.
        if "preview" in declaration:
            config["preview"] = declaration["preview"]
    return config


def read(file):
    """The gate config a declaration file states, compiled without validating
    it: for readers that need only its check IDs, change classes and rule
    sources (ci/validation.py), not its rules."""
    value = json.loads(Path(file).read_text())
    return compile_gate(value) if isinstance(value, dict) and value.get("schema") == SCHEMA else value


def parse(value, declared=False):
    """A config from parsed JSON. A declaration (any qual.json) is validated
    and compiled; another config is returned as is."""
    if declared or (isinstance(value, dict) and value.get("schema") == SCHEMA):
        validate(value)
        return compile_gate(value)
    return value


def load(file):
    """The config a declaration or config file states. A variant's
    declaration (`qual.VARIANT.json`) must name that variant, and `qual.json`
    none. A model's or a variant's declaration must name the model its
    directory gives it, since its lock and input policy are found by that
    name."""
    file = Path(file)
    try:
        value = json.loads(file.read_text())
        named = re.fullmatch(r"qual(?:\.(.+))?\.json", file.name)
        if named is None:
            return parse(value)
        variant = named.group(1)
        located = file.resolve()
        if located.parent.parent.name == "models" and value.get("model") != located.parent.name:
            fail("model", f"must be {located.parent.name}, the model its directory names")
        # Validated as a declaration whatever it holds, so a compiled policy
        # cannot stand in for one under a declaration's name.
        validate(value)
        if value.get("variant") != variant:
            fail("variant", f"must be {variant} in {file.name}" if variant else "is not allowed in qual.json")
        return compile_gate(value)
    except DeclarationError as error:
        raise DeclarationError(f"{file}: {error}") from None


def previewed(root=ROOT):
    """The validated declaration of each model with a `preview` section, by model name in name order."""
    result = {}
    for file in sorted(Path(root).glob("models/*/" + NAME)):
        try:
            value = json.loads(file.read_text())
            validate(value)
            if value["model"] != file.parent.name:
                fail("model", f"must name its directory {file.parent.name}")
        except (DeclarationError, json.JSONDecodeError) as error:
            raise DeclarationError(f"{file}: {error}") from None
        if "preview" in value:
            result[value["model"]] = value
    return result


def renamed(model, root=ROOT):
    """A model's former check IDs, each with the ID that replaced it, so
    receipts recorded under a former ID are read under the current one. The
    mechanics fixture renamed none."""
    if model == "mechanics":
        return {}
    return declared_value(model, root).get("renamed_checks", {})


def change_classes(model, root=ROOT):
    """A model's change classes and the gate checks each selects."""
    return declared_value(model, root).get("change_classes", {})


def declared_value(name, root=ROOT):
    """The validated declaration a MODEL or MODEL.VARIANT name selects (a
    configuration as `chopinfer eval declaration` expands it)."""
    model, variant = split(name)
    if variant is not None and (variant in configurations(model, root) or overlay(model, variant) is not None):
        return json.loads(materialized(model, variant, root).read_text())
    value = json.loads(path(model, root, variant).read_text())
    validate(value)
    return value


def check_media_evidence(policy, items):
    """The declared case inventory must match authenticated reference kinds and IDs."""
    if "coverage" not in policy:
        return
    actual = {}
    for item in items:
        if item["kind"] in ("image", "audio", "video"):
            actual.setdefault(item["kind"], []).append(item["id"])
    if {kind: sorted(ids) for kind, ids in actual.items()} != {
            kind: sorted(ids) for kind, ids in policy["coverage"].items()}:
        fail("qualification.media.coverage", "reference items do not match their declared modality")

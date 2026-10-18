"""Prove that a quantized export with a new format mix is served with data alone, without a GPU.

In a detached worktree of the committed HEAD, the recipe synthesizes a sparse
ModelOpt mixed export of the Qwen3.8 Flash geometry whose format mix neither
committed layout uses: MXFP8 full-attention, indexer and shared-expert
projections beside BF16 linear-attention projections, NVFP4 routed experts
and n-gram lookup, and weight-only NVFP4 MTP experts. The nvfp4-routed-fp8-mtp
layout keeps every dense projection BF16 and mxfp8-dense-nvfp4 makes every
one MXFP8. The
shared kernels already publish BF16 and MXFP8 for every dense port at every
topology, so the mix needs no new kernel.

The recipe rebuilds the bundle with lock updates enabled and regenerates the
projection adoption ledger, admits the export through the helper (probe at
TP2 and TP4, direct preparation at TP2 with the MTP drafter), checks every
operation the package's TP2 and TP4 precision components record against the
export's rules, creates each topology on the fake device with its MTP
profile's speculation (and TP2 also without, to show the drafter's
runtime is created), and asserts that the tree, ignored files included,
differs from HEAD only in those generated ledgers with every existing
record byte-identical. Admission reads only the export's own files, so no
checkpoint data is recorded. Negative controls then show that the assertion
refuses a source edit and a rewritten record in the same tree."""
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import sys
import tempfile
import time
import zlib

from data_only_recipe import (appended_records, changed_paths, detached_worktree, kept_lock_records, probe,
                              require_build_tools, require_clean, run, tokenizer_dir)


ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen38"
LOCK = f"models/{MODEL}/bundle/kernels.lock"
ADOPTIONS = f"models/{MODEL}/bundle/projection-adoptions.json"
SCHEDULES = f"models/{MODEL}/bundle/projection-schedules.json"
# The ledgers the build (KERNELS_LOCK_UPDATE=1) and the adoption example
# regenerate: the only paths an export may change.
GENERATED = (LOCK, ADOPTIONS)
LAYERS = "model.language_model.layers"
MTP_EXPERTS = "mtp.layers.0.mlp.experts"
# ModelOpt leaves these modules unquantized, as an mxfp8-dense-nvfp4 export does: the
# MTP layer keeps BF16 dense projections, which the drafter requires.
IGNORE = ["lm_head", "model.visual.*", "mtp.fc_embedding", "mtp.fc_hidden", "mtp.pre_fc_norm_*",
          "mtp.hyper_connection_mixer*", "mtp.layers.0.self_attn*", "mtp.layers.0.attn_hyper_connection*",
          "mtp.layers.0.mlp_hyper_connection*", "mtp.layers.0.mlp.gate", "mtp.layers.0.mlp.shared_expert*"]
# Each rule's recorded weight format; an unlisted module stays BF16.
FORMATS = {"MXFP8": "Mxfp8", "NVFP4": "Nvfp4", "W4A16_NVFP4": "Nvfp4", None: "Bf16"}
WIDTHS = {"BF16": 2, "F32": 4, "I64": 8, "F8_E4M3": 1, "U8": 1}
COMPONENT_MAGIC = b"CHOPPRC1"
NARROW_ROWS = "chop.speculative-narrow-rows.v1"


def export_rules(config, table):
    """The synthesized export's `quantized_layers`, by module."""
    mxfp8 = {"quant_algo": "MXFP8", "group_size": 32}
    nvfp4 = {"quant_algo": "NVFP4", "group_size": 16}
    rules = {}
    for layer, kind in enumerate(config["text_config"]["layer_types"]):
        if kind == "full_attention":
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj"):
                rules[f"{LAYERS}.{layer}.self_attn.{projection}"] = mxfp8
        for projection in ("gate_proj", "up_proj", "down_proj"):
            rules[f"{LAYERS}.{layer}.mlp.shared_expert.{projection}"] = mxfp8
        rules[f"{LAYERS}.{layer}.mlp.experts"] = nvfp4
    rules[table] = nvfp4
    rules[MTP_EXPERTS] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    return rules


def rule_of(rules, module):
    """The quantization algorithm of `module`: its own rule or the nearest
    parent module's."""
    parts = module.split(".")
    for end in range(len(parts), 0, -1):
        rule = rules.get(".".join(parts[:end]))
        if rule is not None:
            return rule["quant_algo"]
    return None


def read_header(path):
    with path.open("rb") as source:
        (length,) = struct.unpack("<Q", source.read(8))
        return json.loads(source.read(length))


def write_sparse(path, tensors):
    """A safetensors file of `tensors` (name: (dtype, shape)) whose payload
    is a hole; returns the payload's length."""
    header, offset = {}, 0
    for name in sorted(tensors):
        dtype, shape = tensors[name]
        size = WIDTHS[dtype]
        for dimension in shape:
            size *= dimension
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        output.truncate(8 + len(encoded) + offset)
    return offset


def synthesize(reference, output):
    """Write into `output` the mixed export of the sparse reference checkpoint
    `reference` (the Flash FP8 layout the geometry generator writes) and
    return its rules. Every weight keeps its logical shape; its encoding
    follows ModelOpt's: MXFP8 E4M3 codes with one UE8M0 byte per 32 K
    values, NVFP4 E2M1 pairs with one E4M3 scale per 16 K values and an F32
    global scale (a table-wide one for the sharded lookup) and, outside
    weight-only rules, an F32 input scale."""
    config = json.loads((reference / "config.json").read_text())
    source = read_header(reference / "model.safetensors")
    tables = {name.split(".shard_")[0] for name in source if ".ngram_embedding.shard_" in name}
    assert len(tables) == 1, tables
    table = tables.pop()
    rules = export_rules(config, table)
    tensors = {}
    for name, entry in source.items():
        dtype, shape = entry["dtype"], entry["shape"]
        # The reference's FP8 block and per-tensor table companions.
        if name.endswith(".weight_scale_inv") or name == table + ".weight_scale":
            continue
        module = name.removesuffix(".weight")
        algo = rule_of(rules, module) if name.endswith(".weight") else None
        if algo is None:
            assert dtype != "F8_E4M3", f"{name}: the export has no rule for an FP8 reference weight"
            tensors[name] = (dtype, shape)
        elif algo == "MXFP8":
            rows, k = shape
            tensors[name] = ("F8_E4M3", shape)
            tensors[module + ".weight_scale"] = ("U8", [rows, k // 32])
        else:
            rows, k = shape
            tensors[name] = ("U8", [rows, k // 2])
            tensors[module + ".weight_scale"] = ("F8_E4M3", [rows, k // 16])
            if not module.startswith(table + "."):
                tensors[module + ".weight_scale_2"] = ("F32", [])
                if algo == "NVFP4":
                    tensors[module + ".input_scale"] = ("F32", [])
    tensors[table + ".weight_scale_2"] = ("F32", [1])
    output.mkdir()
    total = write_sparse(output / "model.safetensors", tensors)
    index = {"metadata": {"total_size": total}, "weight_map": dict.fromkeys(sorted(tensors), "model.safetensors")}
    (output / "model.safetensors.index.json").write_text(json.dumps(index))
    quantization = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None,
                    "ignore": IGNORE, "quantized_layers": rules}
    config["quantization_config"] = quantization
    (output / "config.json").write_text(json.dumps(config))
    (output / "hf_quant_config.json").write_text(json.dumps(quantization))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(reference / name, output / name)
    print(f"Synthesized export: {len(tensors)} tensors, {total} sparse payload bytes, {len(rules)} rules",
          flush=True)
    return rules


def decode_component(path):
    """A precision component's document (kernel-core's component envelope:
    magic, JSON length, compressed length, zlib-compressed JSON)."""
    data = path.read_bytes()
    assert data[:8] == COMPONENT_MAGIC, path
    decoded, compressed = struct.unpack("<II", data[8:16])
    assert len(data) == 16 + compressed, path
    document = zlib.decompress(data[16:])
    assert len(document) == decoded, path
    return json.loads(document)


def assert_recorded_formats(document, rules, topology):
    """Every recorded operation's weights carry the format their module's
    rule assigns, the mix's three formats all appear, and the MTP layer is
    recorded."""
    counts = {}
    for operation in document["operations"]:
        source = operation["source"]
        for weight in source["weights"]:
            module = weight["codes"]["tensor"].removesuffix(".weight")
            expected = FORMATS[rule_of(rules, module)]
            assert weight["format"] == expected, \
                f"{topology}: {source['id']} records {weight['format']}, the export makes it {expected}"
            counts[expected] = counts.get(expected, 0) + 1
    assert set(counts) == {"Bf16", "Mxfp8", "Nvfp4"}, counts
    mtp = [operation["source"]["id"] for operation in document["operations"]
           if operation["source"]["id"].startswith("mtp.")]
    assert any(name.startswith(MTP_EXPERTS + ".") for name in mtp), mtp
    print(f"{topology}: {len(document['operations'])} operations ({len(mtp)} MTP) record the export's formats; "
          f"weights by format {dict(sorted(counts.items()))}", flush=True)


def create_arguments(profile, manifest):
    """The serving options `serve` creates `profile` with: its sequences and,
    with a drafter, its draft tokens and the verify rows serve derives from
    the package's speculation declaration (runtime-host
    `speculation_config_v1`: at most 32 rows when the topology requires
    narrow speculative rows and serves at most 32 sequences)."""
    sequences = profile["max_sequences"]
    arguments = ["--max-sequences", str(sequences)]
    speculation = profile["speculation"]
    if speculation["drafter"] == "none":
        return arguments
    resources = next((resources for resources in manifest.get("topology_resources", [])
                      if resources["topology"] == profile["topology"]), manifest)
    narrow = NARROW_ROWS in resources["resource_requirements"]["required_capabilities"]
    rows = manifest["speculation"]["maximum_verify_tokens"]
    if narrow and sequences <= 32:
        rows = min(rows, 32)
    return arguments + ["--speculative-tokens", str(speculation["tokens"]), "--verify-tokens", str(rows)]


def assert_export_diff(tree, before):
    """The tree differs from HEAD only in the generated ledgers; every
    existing lock and adoption record keeps its bytes and order, and the
    schedule rows are unchanged."""
    paths = changed_paths(tree)
    assert set(paths) <= set(GENERATED), f"unexpected data-only paths: {paths}"
    old, added = kept_lock_records(before[LOCK], (tree / LOCK).read_text())
    existing, adopted = appended_records(before[ADOPTIONS], (tree / ADOPTIONS).read_text(), "symbol", "adoption record")
    assert (tree / SCHEDULES).read_text() == before[SCHEDULES], "schedule rows changed"
    print(f"Asserted data-only diff: {paths or 'no changed paths'}", flush=True)
    print(f"Lock: {len(old)} byte-identical existing records; {len(added)} added records", flush=True)
    print(f"Adoptions: {len(existing)} byte-identical existing records; "
          f"{len(adopted)} added records", flush=True)
    return added, adopted


def negative_controls(tree, before):
    """Each edit makes the diff assertion fail in this tree, which is then
    restored: a source edit, a new source file, new source files under
    ignored paths, a rewritten lock record, a rewritten adoption record and
    a changed schedule row."""
    source = tree / f"models/{MODEL}/contract/src/sm120/generated.rs"
    edits = [
        (source, lambda text: text + "// export recipe negative control\n", "unexpected data-only paths"),
        (tree / f"models/{MODEL}/bundle/src/export_recipe_control.rs", lambda _: "// control\n",
         "unexpected data-only paths"),
        (tree / "target/export_recipe_control.rs", lambda _: "// control\n", "unexpected data-only paths"),
        (tree / f"models/{MODEL}/bundle/__pycache__/build.rs", lambda _: "// control\n",
         "unexpected data-only paths"),
        (tree / LOCK, lambda text: text.replace("{\n    ", "{\n     ", 1), "pre-existing lock record"),
        (tree / ADOPTIONS, lambda text: text.replace('"entry_sha256": "', '"entry_sha256": "0', 1),
         "pre-existing adoption record"),
        (tree / SCHEDULES, lambda text: text.replace('"top_k": ', '"top_k": 1', 1), "unexpected data-only paths"),
    ]
    for path, edit, message in edits:
        original = path.read_text() if path.exists() else None
        created = [parent for parent in reversed(path.relative_to(tree).parents) if not (tree / parent).exists()]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(edit(original))
        try:
            assert_export_diff(tree, before)
        except AssertionError as error:
            assert message in str(error), f"{path.name}: {error}"
        else:
            raise AssertionError(f"the diff assertion accepted a negative control edit of {path.name}")
        finally:
            if original is None:
                path.unlink()
                for parent in reversed(created):
                    (tree / parent).rmdir()
            else:
                path.write_text(original)
        print(f"Negative control: {path.relative_to(tree)} refused ({message})", flush=True)


def recipe(target_dir=None):
    tokenizer = tokenizer_dir("QWEN38")
    require_build_tools()
    require_clean(ROOT, "export recipe")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    started = time.monotonic()
    with detached_worktree(ROOT, "chopinfer-export-", target_dir) as (scratch, tree, env):
        before = {path: (tree / path).read_text() for path in (*GENERATED, SCHEDULES)}
        locked = ("--locked", "--offline")
        print("Build the bundle, helper and loader with lock updates enabled", flush=True)
        run("cargo", "build", "-p", f"bundle-{MODEL}", "-p", f"model-{MODEL}", "-p", f"{MODEL}-load-proto", "--lib",
            "--example", "geometry-checkpoint", "--bin", f"chopinfer-model-{MODEL}", "--bin", f"{MODEL}-load-direct",
            *locked, cwd=tree, env=dict(env, KERNELS_LOCK_UPDATE="1"))
        # The adoptions read the lock as it is after the bundle build.
        print("Regenerate the projection adoption ledger", flush=True)
        ledger = run("cargo", "run", "-q", "-p", f"model-{MODEL}-contract", "--example",
                     f"{MODEL}_projection_adoptions", *locked, cwd=tree, env=env)
        (tree / ADOPTIONS).write_text(ledger)
        binaries = Path(env["CARGO_TARGET_DIR"]) / "debug"
        reference = scratch / "reference"
        print(run(str(binaries / "examples/geometry-checkpoint"),
                  str(tree / f"models/{MODEL}/compiler/fixtures/config.json"), tokenizer, str(reference),
                  env=env).splitlines()[-1], flush=True)
        export = scratch / "export"
        rules = synthesize(reference, export)
        helper = binaries / f"chopinfer-model-{MODEL}"
        for ranks in (2, 4):
            reply = probe(helper, export, ranks, env)
            assert reply == {"result": "probe", "compatible": True, "reason": "compatible metadata"}, reply
        # One package serves every topology the precision admits, so a
        # TP2 preparation also emits TP4's precision component; serve runs
        # each topology with its MTP profile, which TP2's preparation selects.
        reply = json.loads(run(str(helper), "prepare", "--direct", "--model", str(export),
                               "--tensor-parallel-size", "2", "--modalities", "all",
                               "--cache-dir", str(scratch / "cache"), env=env))
        assert reply["result"] == "prepared-direct" and reply["tensor_parallel_size"] == 2, reply
        profiles = json.loads((tree / f"models/{MODEL}/profiles.json").read_text())["profiles"]
        assert reply["profile"] == {"name": "tp2-mtp", "profile": profiles["tp2-mtp"]}, reply["profile"]
        bundle = Path(reply["bundle"])
        manifest = json.loads((bundle / "manifest.json").read_text())
        assert manifest["speculation"]["bundle_drafter"] and manifest["speculation"]["drafter_kind"] == "mtp", \
            manifest["speculation"]
        loader = binaries / f"{MODEL}-load-direct"

        def create(ranks, arguments):
            created = json.loads(run(str(loader), "create", "--checkpoint", str(export), "--bundle-dir",
                                     str(bundle), "--ranks", str(ranks), *arguments, env=env))
            assert created["created"] and created["refused_launches"] == 0, created
            return created

        for ranks in (2, 4):
            profile = profiles[f"tp{ranks}-mtp"]
            arguments = create_arguments(profile, manifest)
            assert "--speculative-tokens" in arguments, arguments
            created = create(ranks, arguments)
            topology = created["topology"]
            assert topology == profile["topology"], created
            assert_recorded_formats(decode_component(bundle / f"components/precision.{topology}.bin"),
                                    rules, topology)
            if ranks == 2:
                # Without speculation create skips the drafter: its runtime's
                # buffers are what the drafter create adds.
                plain = create(ranks, create_arguments({**profile, "speculation": {"drafter": "none"}}, manifest))
                assert created["device_buffers"] > plain["device_buffers"], (created, plain)
                print(f"TP2: the drafter adds {created['device_buffers'] - plain['device_buffers']} device buffers "
                      "to an autoregressive create", flush=True)
            print(f"TP{ranks}: probe, the package's precision and fake-device create with the MTP drafter "
                  f"({' '.join(arguments)}) passed ({created['seconds']:.1f}s create)", flush=True)
        assert_export_diff(tree, before)
        negative_controls(tree, before)
        assert_export_diff(tree, before)
    print(f"export recipe {MODEL}: {time.monotonic() - started:.3f}s (exit 0)", flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if sys.argv[1:] not in ([], ["--ci"], [MODEL]):
        raise SystemExit(f"usage: model-export-recipe.sh [--ci|{MODEL}]")
    # Remove the temporary target after the leg or SIGTERM.
    with tempfile.TemporaryDirectory(prefix="chopinfer-export-target-", dir=os.environ.get("TMPDIR")) as target:
        recipe(Path(target))

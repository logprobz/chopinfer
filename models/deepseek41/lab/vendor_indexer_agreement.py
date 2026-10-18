"""Measure the pinned vendor's BF16 selection on the GPU test's prepared operands.

The score and selection code is Indexer.forward from the supplied pinned model.py.
Projection, rotation and quantization are identity adapters here because the
fixture supplies decoded MXFP4 operands directly. This is a selection
measurement at TP1, not an end-to-end model or preparation comparison.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace


def vendor_nodes(source):
    tree = ast.parse(source)
    indexer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Indexer")
    forward = next(n for n in indexer.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    candidates = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_candidate_blocks")
    # Keep the executed method body unchanged. Only its module placement changes.
    return ast.Module(body=[candidates, forward], type_ignores=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if "CHOPINFER_KERNEL_GPU" not in os.environ:
        raise RuntimeError("the GPU queue must select this job's device")
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda", int(os.environ["CHOPINFER_KERNEL_GPU"]))
    torch.backends.cuda.matmul.allow_tf32 = False
    meta = json.loads((args.fixture / "fixture.json").read_text())
    if meta["prepared_operands"] is not True:
        raise ValueError("expected decoded MXFP4 query and key operands")
    heads, dim = meta["heads"], meta["dim"]
    width, visible = meta["width"], meta["visible"]

    def tensor(name, dtype, shape):
        data = bytearray((args.fixture / name).read_bytes())
        return torch.frombuffer(data, dtype=dtype).reshape(shape).clone().to(device)

    keys = tensor("keys.bf16", torch.bfloat16, (1, width, dim))
    own_mask = tensor("candidate-mask.u8", torch.uint8, (1, 1, width)).bool()[..., :visible]
    own_source = tensor("source-indices.i32", torch.int32, (-1,))
    own_reindex = tensor("reindex-indices.i32", torch.int32, (-1,))
    shared = SimpleNamespace(index_k=keys, candidates=None)
    namespace = {
        "torch": torch, "F": F, "world_size": 1, "shared_attn": shared,
        "fp4_block_size": 32,
        "apply_rotary_emb": lambda values, frequencies: values,
        "fp4_act_quant": lambda values, block, fake: values,
    }
    source = args.reference.read_bytes()
    exec(compile(vendor_nodes(source), "pinned-vendor-indexer", "exec"), namespace)
    x = torch.empty((1, 1, 1), dtype=torch.bfloat16, device=device)
    start = visible * meta["ratio"] - 1

    def run(prefix, candidate_source):
        q = tensor(prefix + "-q.bf16", torch.bfloat16, (1, 1, heads * dim))
        # BF16 spread mode rounds the projected weight too. The served
        # F32-TQ score keeps the fixture's original F32 weight.
        w = tensor(prefix + "-w.f32", torch.float32, (1, 1, heads)).to(torch.bfloat16)
        indexer = SimpleNamespace(
            owns_k=False, compress_ratio=meta["ratio"], rope_head_dim=64,
            n_local_heads=heads, n_heads=heads, index_head_dim=dim,
            index_topk=meta["top_k"], softmax_scale=dim**-0.5,
            is_candidate_source=candidate_source, uses_candidates=not candidate_source,
            candidate_topk_blocks=meta["candidate_blocks"], candidate_block_size=meta["block_size"],
            freqs_cis=torch.empty((0,), dtype=torch.complex64, device=device),
            wq_b=lambda _: q, weights_proj=lambda _: w,
        )
        return namespace["forward"](indexer, x, x, None, start, 0).flatten()

    vendor_source = run("source", True)
    vendor_mask = shared.candidates.clone()
    vendor_reindex = run("reindex", False)
    shared.candidates = own_mask
    vendor_same_candidates = run("reindex", False)

    def agreement(ours, vendor):
        a = set(ours[ours >= 0].cpu().tolist())
        b = set(vendor[vendor >= 0].cpu().tolist())
        common = len(a & b)
        return {"common": common, "ours": len(a), "vendor": len(b),
                "overlap_fraction": common / max(1, len(a)), "exact": a == b}

    report = {
        "reference_sha256": hashlib.sha256(source).hexdigest(),
        "torch": torch.__version__, "vendor_tp": 1,
        "scope": "BF16 vendor selection on prepared MXFP4 operands",
        "agreement_is_gated": False, "fixture": meta,
        "source": agreement(own_source, vendor_source),
        "reindex_vendor_candidates": agreement(own_reindex, vendor_reindex),
        "reindex_same_candidates": agreement(own_reindex, vendor_same_candidates),
        "candidate_position_agreement": (own_mask == vendor_mask).float().mean().item(),
        "candidate_positions_ours": own_mask.sum().item(),
        "candidate_positions_vendor": vendor_mask.sum().item(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

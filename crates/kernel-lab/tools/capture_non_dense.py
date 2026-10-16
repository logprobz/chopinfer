"""Capture bounded reference-policy operands for the lab-only chain probes.

Run only in an assigned GPU job. This wraps the pinned reference APIs without
editing their numerics or binding files. Captures qualify isolated arithmetic,
not equivalence to the served model's intermediate tensors.
"""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import sys


def digest(data):
    return hashlib.sha256(data).hexdigest()


def segment(keys):
    if not 1 <= keys <= 1_048_576:
        raise ValueError("unsupported visible key count")
    length = 256 * ((keys + 1023) // 1024)
    begin = ((keys - 1) // length) * length
    live = keys - begin
    return begin, live, (live + 31) // 32 * 32


class Capture:
    def __init__(self, root, source, limit):
        self.root = root
        root.mkdir(parents=True, exist_ok=False)
        self.source = source
        self.limit = limit
        self.used = 0
        self.cases = []
        self.seen = set()
        self.metadata = []

    def file(self, name, tensor):
        data = tensor.detach().contiguous().view(__import__("torch").uint8).cpu().numpy().tobytes()
        if self.used + len(data) > self.limit:
            raise RuntimeError("capture byte budget exceeded")
        self.used += len(data)
        (self.root / name).write_bytes(data)
        return {"path": name, "sha256": digest(data)}

    def finish(self):
        if not self.cases:
            raise RuntimeError("no matched operands captured")
        document = {"schema": "chopinfer.non-dense-accumulation.v1", "cases": self.cases}
        (self.root / "manifest.json").write_text(json.dumps(document, indent=2) + "\n")
        (self.root / "capture.json").write_text(json.dumps({
            "source_sha256": self.source, "bytes": self.used,
            "scope": "reference-policy operands; isolated accumulation only",
            "sites": self.metadata,
        }, indent=2) + "\n")


def encoder_capture(args, capture, extra):
    import torch
    from tools.refrun import encoder, encoder_ops

    original = encoder.build
    handles = []
    ks = {int(k) for k in args.ks.split(",")} if args.ks else None
    patterns = args.sites.split(",")

    def build(*values, **kwargs):
        model = original(*values, **kwargs)
        for name, module in model.named_modules():
            if not isinstance(module, encoder_ops.Gemm) or not any(fnmatch.fnmatchcase(name, p) for p in patterns):
                continue

            def hook(layer, inputs, site=name):
                if site in capture.seen or len(capture.cases) >= args.max_cases:
                    return
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
                w = layer.weight.detach()
                n, k = w.shape
                if ks is not None and k not in ks:
                    return
                m = min(args.rows, x.shape[0]) // 8 * 8
                if m == 0 or n < 128 or k % 32:
                    return
                if not torch.equal(w.float(), w.to(torch.bfloat16).float()):
                    raise RuntimeError(f"{site}: reference weight is not an exact widened BF16 operand")
                columns = torch.linspace(0, n - 1, 128, device=w.device).long()
                rows = torch.linspace(0, x.shape[0] - 1, m, device=x.device).long()
                number = len(capture.cases)
                prefix = f"encoder-{number:02}"
                capture.cases.append({
                    "id": prefix, "family": "encoder_pairs", "origin": "reference policy linear input",
                    "source_sha256": capture.source, "n": 128, "m": m, "k": k,
                    "reduction_k": k, "terms": 2,
                    "weight": capture.file(prefix + "-weight.bf16", w[columns].to(torch.bfloat16)),
                    "activation": capture.file(prefix + "-activation.f32", x[rows]),
                    "correction": None, "denominator": None,
                })
                capture.metadata.append({"id": prefix, "module": site, "original_shape": [n, k],
                                         "rows": rows.cpu().tolist(), "columns": columns.cpu().tolist(),
                                         "pre_epilogue": True})
                capture.seen.add(site)

            handles.append(module.register_forward_pre_hook(hook))
        return model

    encoder.build = build
    try:
        encoder.main(extra)
    finally:
        encoder.build = original
        for handle in handles:
            handle.remove()


def pv_capture(args, capture, extra):
    import torch
    from tools.refrun import run

    original_load = run.load_adapter
    restorations = []

    def load(path):
        adapter, adapter_path = original_load(path)
        build = adapter.build

        def wrapped_build(config, placement, *values, **kwargs):
            model = build(config, placement, *values, **kwargs)
            vendor = adapter.vendor(config.checkpoint)
            original = vendor.eager_attention_forward
            # The adapter installs its reference function here during build.
            def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, sinks=None, **kw):
                layer = getattr(module, "layer_idx", None)
                if layer == args.layer and getattr(module, "sliding_window", None) is None and layer not in capture.seen:
                    if dropout or sinks is not None or query.shape[0] != 1:
                        raise RuntimeError("capture requires one deterministic global-attention sequence without sinks")
                    qh, qlen, dim = query.shape[1:]
                    kvh, klen = key.shape[1:3]
                    group = qh // kvh
                    if group < 16 or value.shape[-1] != 128 or qlen != klen:
                        raise RuntimeError("unsupported global-attention capture geometry")
                    # Sixteen heads on one KV head at the same last position.
                    # Capture one absolute prefill segment with its own zero
                    # softmax state. V already includes any model value scale.
                    begin, live, padded = segment(klen)
                    q = query[0, :16, -1].float()
                    k = key[0, 0, begin:klen].float()
                    v = value[0, 0, begin:klen].to(torch.bfloat16)
                    score = torch.matmul(q, k.T) * float(scaling)
                    if attention_mask is not None:
                        mask = attention_mask[0, :, -1, begin:klen].float()
                        score = score + mask
                    score = score * 1.4426950408889634
                    maximum = torch.full((16,), -torch.inf, device=q.device)
                    denominator = torch.zeros(16, device=q.device)
                    probability = torch.zeros((16, padded), device=q.device)
                    correction = torch.ones((16, padded // 32), device=q.device)
                    for start in range(0, padded, 32):
                        end = min(start + 32, live)
                        tile = score[:, start:end]
                        next_max = torch.maximum(maximum, tile.max(dim=1).values)
                        c = torch.exp2(maximum - next_max)
                        p = torch.exp2(tile - next_max[:, None])
                        probability[:, start:end] = p
                        correction[:, start // 32] = c
                        denominator = denominator * c + p.sum(dim=1)
                        maximum = next_max
                    padded_v = torch.zeros((128, padded), dtype=torch.bfloat16, device=v.device)
                    padded_v[:, :live] = v.T
                    prefix = f"pv-layer{layer:02}"
                    capture.cases.append({
                        "id": prefix, "family": "pv", "origin": "reference QKV; frozen F32 score/online-softmax segment",
                        "source_sha256": capture.source, "n": 128, "m": 16, "k": padded,
                        "reduction_k": live, "terms": 1 if klen >= 32768 else 2 if klen >= 128 else 3,
                        "weight": capture.file(prefix + "-value.bf16", padded_v),
                        "activation": capture.file(prefix + "-probability.f32", probability),
                        "correction": capture.file(prefix + "-correction.f32", correction),
                        "denominator": capture.file(prefix + "-denominator.f32", denominator),
                    })
                    capture.metadata.append({"id": prefix, "query_position": qlen - 1,
                        "visible_keys": klen, "segment_begin": begin, "segment_live": live,
                        "heads": list(range(16)), "kv_head": 0, "scaling": float(scaling),
                        "value_scale": "already applied before the hook", "qkv_dtype": str(query.dtype),
                        "score_policy": "torch F32 matmul, not served QK parity",
                        "query": capture.file(prefix + "-query.bf16", query[0, :16, -1].to(torch.bfloat16)),
                        "key": capture.file(prefix + "-key.bf16", key[0, 0, begin:klen].to(torch.bfloat16)),
                    })
                    capture.seen.add(layer)
                return original(module, query, key, value, attention_mask, scaling, dropout=dropout, sinks=sinks, **kw)
            vendor.eager_attention_forward = attention
            restorations.append((vendor, original))
            return model
        adapter.build = wrapped_build
        return adapter, adapter_path

    run.load_adapter = load
    try:
        run.main(["run", *extra])
    finally:
        run.load_adapter = original_load
        for vendor, original in restorations:
            vendor.eager_attention_forward = original


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=["encoder", "pv"])
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--sites", default="*")
    parser.add_argument("--ks", default="")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--provenance", type=Path, action="append", default=[])
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if not (8 <= args.rows <= 256 and 1 <= args.max_cases <= 16):
        raise ValueError("invalid capture budget")
    if "--devices" not in extra or not all(d.startswith("cuda:") for d in extra[extra.index("--devices") + 1].split(",")):
        raise ValueError("capture requires explicit assigned CUDA devices")
    source = digest(b"".join(p.read_bytes() for p in [Path(__file__), *args.provenance]))
    capture = Capture(args.captures, source, 64 << 20)
    (encoder_capture if args.family == "encoder" else pv_capture)(args, capture, extra)
    capture.finish()
    print(f"captured {len(capture.cases)} cases in {args.captures}")


if __name__ == "__main__":
    main()

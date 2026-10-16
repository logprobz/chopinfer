"""GPU-only compact PV product captures for remaining attention families.

Encoder pair products are expanded in served k16/term/value-half order.
The matrices freeze a global maximum and normalized probability values.
They isolate accumulation, not online-softmax or complete serving parity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def expansion_shape(keys, terms, halves, tile):
    if keys <= 0 or terms not in (1, 2, 3) or halves not in (1, 2) or tile not in (16, 32, 64):
        raise ValueError("invalid product expansion")
    padded = (keys + tile - 1) // tile * tile
    return padded, padded * terms * halves, tile * terms * halves


def encoder_rows(length, mask):
    plans = [("tail", list(range(max(0, length - 16), length)))]
    if isinstance(mask, dict) and length >= 32:
        start = min(length - 16, length // 2 // 16 * 16)
        middle = list(range(start, start + 16))
        if middle != plans[0][1]:
            plans.insert(0, ("middle", middle))
    return plans


def site_hooks(active, site):
    def enter(_module, _inputs):
        active.append(site)

    def leave(_module, _inputs, _output):
        active.pop()

    return enter, leave


class ProductCapture:
    def __init__(self, root, provenance, limit):
        from capture_non_dense import Capture
        self.writer = Capture(root, provenance, limit)
        self.root, self.provenance = root, provenance
        self.cases, self.sites, self.seen = [], [], set()

    def add(self, name, family, probability, value, reductions, terms, paired, tile, metadata):
        import torch
        if name in self.seen:
            return
        if len(self.cases) >= 16:
            raise RuntimeError("PV case limit exceeded")
        # P [queries,keys], V [keys,features], both frozen reference F32.
        m, keys = probability.shape
        n = value.shape[1]
        if value.shape[0] != keys or len(reductions) != m or any(k <= 0 or k > keys for k in reductions):
            raise ValueError("PV visible-key geometry mismatch")
        if not torch.isfinite(probability).all() or not torch.isfinite(value).all():
            raise ValueError("nonfinite PV operands")
        if not (1 <= m <= 128 and 1 <= n <= 512):
            raise ValueError("PV capture geometry exceeds bounds")
        key_pad, physical, ownership = expansion_shape(keys, terms, 2 if paired else 1, tile)
        n_pad = (n + 15) // 16 * 16
        if physical > 16384:
            raise ValueError("PV physical reduction exceeds bounded probe")
        p = torch.zeros((m, key_pad), device=probability.device, dtype=torch.float32)
        p[:, :keys] = probability.float()
        v = torch.zeros((key_pad, n_pad), device=value.device, dtype=torch.float32)
        v[:keys, :n] = value.float()
        p_terms = []
        for _ in range(terms):
            part = p.to(torch.bfloat16)
            p_terms.append(part)
            p = p - part.float()
        hi = v.to(torch.bfloat16)
        values = [hi, (v - hi.float()).to(torch.bfloat16)] if paired else [hi]
        expanded_p, expanded_v = [], []
        for start in range(0, key_pad, 16):
            for part in p_terms:
                for half in values:
                    expanded_p.append(part[:, start:start + 16])
                    expanded_v.append(half[start:start + 16].T)
        x = torch.cat(expanded_p, dim=1)
        w = torch.cat(expanded_v, dim=1)
        prefix = f"pv-{len(self.cases):02}"
        offsets = list(range(0, physical + 1, ownership)) if family == "encoder_pair_pv" else [0, physical]
        self.cases.append({
            "id": prefix, "family": family, "origin": "reference policy; expanded decoded PV products; frozen global maximum",
            "source_sha256": self.provenance, "rows": m,
            "bank": {"n": n_pad, "live_n": n, "k": physical, "weight": self.writer.file(prefix + "-value.bf16", w), "scales": None, "global": 1.},
            "input": self.writer.file(prefix + "-probability.bf16", x),
            "reduction_lengths": reductions,
            "schedules": [{"id": "tile64" if family == "encoder_pair_pv" else "persistent_selected_or_window",
                           "projection": "matrix", "offsets": offsets, "baseline_k": 0}],
        })
        self.sites.append({"id": prefix, "name": name, "family": family, "queries": m, "features": n,
                           "keys_staged": keys, "keys_padded": key_pad, "physical_k": physical,
                           "terms": terms, "value_halves": len(values), "tile_keys": tile,
                           "baseline_max_mma_calls": ownership // 16 if family == "encoder_pair_pv" else physical // 16,
                           "metadata": metadata})
        self.seen.add(name)

    def finish(self):
        if not self.cases:
            raise RuntimeError("no matched PV products captured")
        (self.root / "manifest.json").write_text(json.dumps({"schema": "chopinfer.matrix-accumulation.v1", "cases": self.cases}, indent=2) + "\n")
        (self.root / "capture.json").write_text(json.dumps({"source_sha256": self.provenance, "bytes": self.writer.used,
            "scope": "stable-maximum PV products; no full serving or KV storage equivalence", "sites": self.sites}, indent=2) + "\n")


def encoder(args, capture, extra):
    import fnmatch
    import torch
    from tools.refrun import encoder as runner, encoder_ops as ops
    original_build, original_attention = runner.build, ops.attention
    active = []
    handles = []

    def build(*values, **kwargs):
        model = original_build(*values, **kwargs)
        for name, module in model.named_modules():
            if isinstance(module, ops.Block) and any(fnmatch.fnmatchcase(name, p) for p in args.sites.split(",")):
                pre, post = site_hooks(active, name)
                handles.append(module.register_forward_pre_hook(pre))
                handles.append(module.register_forward_hook(post))
        return model

    def attention(q, k, v, lengths, heads, kv_heads, head_dim, mask, sink, scale, numerics, devices):
        if active and active[-1] not in capture.seen:
            if numerics.arithmetic != "pairs":
                raise ValueError("encoder PV capture requires the declared pair policy")
            length = lengths[0]
            keys = k[:length, :head_dim].float()
            values = v[:length, :head_dim].float()
            kh, kl = ops.halves(keys)
            for label, selected in encoder_rows(length, mask):
                rows = torch.tensor(selected, device=q.device)
                query = q[rows, :head_dim].float()
                qh, ql = ops.halves(query)
                score = qh @ kh.T
                score += qh @ kl.T
                score += ql @ kh.T
                score *= scale
                if sink is not None:
                    score[:, 0] += sink[0].float()
                visible = ops.visible(mask, rows, torch.arange(length, device=q.device))
                if visible is not None:
                    score.masked_fill_(~visible, -torch.inf)
                else:
                    visible = torch.ones_like(score, dtype=torch.bool)
                score -= score.amax(dim=-1, keepdim=True)
                p = score.exp()
                denominator = p.sum(dim=-1, keepdim=True)
                reductions = visible.sum(dim=-1).cpu().tolist()
                terms = max(2, 3 if min(reductions) < 128 else 2 if min(reductions) < 32768 else 1)
                capture.add(active[-1] + ":" + label, "encoder_pair_pv", p / denominator, values, reductions, terms, True, 64,
                            {"head_dim": head_dim, "head": 0, "kv_head": 0, "rows": selected, "mask": mask,
                             "sink": "key0 score bias, masked normally" if sink is not None else None,
                             "policy": "pair QK three products, every P term times both V halves"})
            capture.seen.add(active[-1])
        return original_attention(q, k, v, lengths, heads, kv_heads, head_dim, mask, sink, scale, numerics, devices)

    runner.build, ops.attention = build, attention
    try:
        runner.main(extra)
    finally:
        runner.build, ops.attention = original_build, original_attention
        for handle in handles:
            handle.remove()


def text(args, capture, extra):
    import torch
    import torch.nn.functional as functional
    from tools.refrun import run
    original_load = run.load_adapter
    restorations = []

    def load(path):
        adapter, adapter_path = original_load(path)
        original_build = adapter.build
        def build(config, placement, *values, **kwargs):
            reference = original_build(config, placement, *values, **kwargs)
            vendor = adapter.vendor(config.checkpoint) if args.family == "swa" else adapter.hf_modules()[1]
            original = vendor.eager_attention_forward
            def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
                layer = getattr(module, "layer_idx", None)
                name = f"layer{layer}"
                wanted = layer == args.layer and name not in capture.seen and key.shape[-2] >= args.min_keys
                if args.family == "swa":
                    wanted = wanted and getattr(module, "is_swa", False)
                    if wanted:
                        length = key.shape[-2]
                        positions = [query.shape[-2] - 2, query.shape[-2] - 1]
                        absolute = [length - query.shape[-2] + p for p in positions]
                        # Absolute 16-query tile, including aligned masked staging.
                        tile0 = absolute[0] // 16 * 16
                        begin = max(0, tile0 - 127) // 32 * 32
                        end = min(length, ((tile0 + 16 + 31) // 32) * 32)
                        group = query.shape[1] // key.shape[1]
                        if group < 8:
                            raise ValueError("unsupported window head group")
                        q = query[0, :8, positions].transpose(0, 1).reshape(16, -1).float()
                        keys = key[0, 0, begin:end].float()
                        score = q @ keys.T * float(scaling)
                        mask = attention_mask[0, 0, positions, begin:end].repeat_interleave(8, dim=0)
                        score += mask.float()
                        sink = kwargs.get("sinks")
                        sink = None if sink is None else sink[:8].float().repeat(2)
                        maximum = score.amax(dim=-1)
                        if sink is not None:
                            maximum = torch.maximum(maximum, sink)
                        p = (score - maximum[:, None]).exp()
                        denominator = p.sum(dim=-1)
                        if sink is not None:
                            denominator += (sink - maximum).exp()
                        reductions = (mask == 0).sum(dim=-1).cpu().tolist()
                        terms = 3 if min(reductions) < 128 else 2
                        capture.add(name, "swa_pv", p / denominator[:, None], value[0, 0, begin:end].float(), reductions, terms, False, 32,
                                    {"positions": absolute, "heads": list(range(8)), "kv_head": 0, "begin": begin,
                                     "sink": "virtual key with zero value; probability mass not renormalized" if sink is not None else None})
                    return original(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
                if not wanted:
                    return original(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
                old_softmax = functional.softmax
                probabilities = []
                def softmax(input, *a, **kw):
                    result = old_softmax(input, *a, **kw)
                    if result.ndim == 4:
                        probabilities.append(result)
                    return result
                functional.softmax = softmax
                try:
                    result = original(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
                finally:
                    functional.softmax = old_softmax
                if len(probabilities) != 1:
                    raise RuntimeError("QSA eager softmax capture did not execute exactly once")
                ids = torch.nonzero(attention_mask[0, 0, -1] == 0, as_tuple=False).flatten()
                # Native selection writes chosen blocks in ascending index order.
                # The captured mask preserves the set; no top-k is recomputed.
                group = query.shape[1] // key.shape[1]
                p = probabilities[0][0, :group, -1].index_select(-1, ids).float()
                values = value[0, 0].index_select(0, ids).float()
                count = ids.numel()
                terms = 3 if count < 128 else 2
                capture.add(name, "qsa_pv", p, values, [count] * group, terms, False, 16,
                            {"selected_key_ids": ids.cpu().tolist(), "order": "ascending native block/key order from captured selection mask",
                             "heads": list(range(group)), "query_position": key.shape[-2] - 1, "kv_head": 0,
                             "probability": "captured F32 eager softmax before BF16 cast", "complete_blocks_plus_tail": True})
                return result
            vendor.eager_attention_forward = attention
            restorations.append((vendor, "eager_attention_forward", original))
            return reference
        adapter.build = build
        restorations.append((adapter, "build", original_build))
        return adapter, adapter_path
    run.load_adapter = load
    try:
        run.main(["run", *extra])
    finally:
        run.load_adapter = original_load
        for module, name, original in reversed(restorations):
            setattr(module, name, original)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=["encoder", "swa", "qsa"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sites", default="*.blocks.0")
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--min-keys", type=int, default=128)
    parser.add_argument("--provenance", type=Path, action="append", default=[])
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if "--devices" not in extra or not all(d.startswith("cuda:") for d in extra[extra.index("--devices") + 1].split(",")):
        raise ValueError("PV captures require assigned CUDA devices")
    paths = [Path(__file__), Path(__file__).with_name("capture_non_dense.py"), *args.provenance]
    provenance = hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()
    capture = ProductCapture(args.output, provenance, 64 << 20)
    (encoder if args.family == "encoder" else text)(args, capture, extra)
    capture.finish()
    print(f"captured {len(capture.cases)} PV product matrices")


if __name__ == "__main__":
    main()

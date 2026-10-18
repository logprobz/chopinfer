"""The executed vendor control of reference_kernel.py: the pinned tilelang
kernels of DeepSeek V4.1 Flash's `inference/kernel.py` against the BF16-mode
replacements, on one GPU.

    $CHOPINFER_DATA/venvs/tilelang-0.1.12/bin/python -B \\
        models/deepseek41/reference_vendor_control.py --checkpoint CK --output DIR

Runs only in a scratch environment with tilelang 0.1.12 (the vendor's pin),
never the runner's. Each case runs the vendor kernel and the replacement on
the same inputs and records, per output: whether the bytes are equal, how
many elements differ, the largest difference in units in the last place of
the output dtype, and, against an f64 oracle of the same math, how often
each side is the closer one. A case is classified `bitwise`, `lowering`
(differences of at most `LOWERING_ULPS` units in the last place at each
value's magnitude, or within one unit of its row's largest value where a
sum cancels, with the replacement closer to the oracle at least nine times
for every ten the vendor kernel is) or
`semantic` (anything else, which the replacement must fix; any difference in
the quantizers' codes, scales or quantize-dequantize values, whose outputs are
discrete decisions, is semantic). It exits nonzero on a semantic difference
or an error. Cases cover
accumulation cancellation, softmax tile boundaries and scale boundaries.
Writes DIR/report.json (refused if DIR exists).
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import types

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.deepseek41 import reference_kernel as ours  # noqa: E402

KERNEL = "inference/kernel.py"
KERNEL_SHA256 = "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455"
LOWERING_ULPS = 4
# One unit in the last place relative to a value: BF16 and F32.
ROW_ULP = {"torch.bfloat16": 2.0 ** -7, "torch.float32": 2.0 ** -23}


def vendor_kernels(checkpoint):
    data = (Path(checkpoint) / KERNEL).read_bytes()
    if hashlib.sha256(data).hexdigest() != KERNEL_SHA256:
        raise SystemExit(f"{KERNEL} is not the pinned {KERNEL_SHA256}")
    module = types.ModuleType("vendor_kernel")
    module.__file__ = str(Path(checkpoint) / KERNEL)
    exec(compile(data, module.__file__, "exec"), module.__dict__)
    return module


def ulps(a, b):
    """Elementwise distance in units in the last place of a's dtype, by the
    ordered integer view of each value."""
    width = {torch.bfloat16: torch.int16, torch.float32: torch.int32, torch.float8_e4m3fn: torch.int8,
             torch.float8_e8m0fnu: torch.uint8, torch.uint8: torch.uint8}[a.dtype]
    if width == torch.uint8:
        return (a.view(torch.uint8).to(torch.int64) - b.view(torch.uint8).to(torch.int64)).abs()
    ia, ib = a.view(width).to(torch.int64), b.view(width).to(torch.int64)
    sign = 1 << (torch.iinfo(width).bits - 1)
    oa = torch.where(ia < 0, -(ia + sign), ia)
    ob = torch.where(ib < 0, -(ib + sign), ib)
    return (oa - ob).abs()


def compare(name, vendor, replacement, oracle=None):
    """The record of one output."""
    same = vendor.view(torch.uint8).flatten().equal(replacement.view(torch.uint8).flatten()) \
        if vendor.dtype.itemsize == 1 else vendor.view(-1).view(
            {2: torch.int16, 4: torch.int32}[vendor.dtype.itemsize]).equal(
            replacement.view(-1).view({2: torch.int16, 4: torch.int32}[vendor.dtype.itemsize]))
    record = {"output": name, "dtype": str(vendor.dtype), "elements": vendor.numel(), "bitwise_equal": bool(same)}
    if not same:
        distance = ulps(vendor, replacement)
        record |= {"differing": int((distance != 0).sum()), "max_ulps": int(distance.max()),
                   "first_differing": (distance != 0).nonzero()[:8].tolist()}
        if vendor.is_floating_point():
            gap = (vendor.double() - replacement.double()).abs()
            larger = torch.maximum(vendor.double().abs(), replacement.double().abs())
            ulp = torch.where(larger > 0, torch.exp2(torch.floor(torch.log2(larger.clamp_min(1e-300)))), 1.0) * (
                2.0 ** -{torch.bfloat16: 7, torch.float32: 23}.get(vendor.dtype, 3))
            record |= {"max_abs_difference": float(gap.max()), "max_ulps_at_magnitude": float((gap / ulp).max()),
                       "max_relative_to_row": float((gap / larger.amax(-1, keepdim=True).clamp_min(1e-30)).max())
                       if gap.dim() > 1 else None}
        if oracle is not None and vendor.is_floating_point():
            v = (vendor.double() - oracle).abs()
            r = (replacement.double() - oracle).abs()
            differ = distance != 0
            record |= {"replacement_closer": int((r < v)[differ].sum()), "vendor_closer": int((v < r)[differ].sum()),
                       "max_abs_vendor_error": float(v.max()), "max_abs_replacement_error": float(r.max())}
    return record


def classify(records, discrete=False):
    """bitwise, lowering or semantic. A difference in a discrete decision
    (quantizer codes and scales, or values decoded from them) or one without
    an oracle to judge it is semantic."""
    records = [r for r in records if "vendor_error" not in r]
    if not records:
        return "vendor-error"
    if all(r["bitwise_equal"] for r in records):
        return "bitwise"
    if discrete:
        return "semantic"
    if all(r["bitwise_equal"] or lowering(r) for r in records):
        return "lowering"
    return "semantic"


def lowering(record):
    """A difference of reduction order: within LOWERING_ULPS at each value's
    magnitude, or (where a sum cancels) within one unit in the last place of
    its row's largest value, with the replacement closer to the oracle at
    least nine times for every ten the vendor is."""
    if "replacement_closer" not in record:
        return False
    bounded = record.get("max_ulps_at_magnitude", math.inf) <= LOWERING_ULPS \
        or (record.get("max_relative_to_row") or math.inf) <= ROW_ULP[record["dtype"]]
    return bounded and 10 * record["replacement_closer"] >= 9 * record["vendor_closer"]


def bf16_values(generator, shape, spread, device):
    x = torch.randn(shape, generator=generator, device=device) \
        * torch.exp(torch.randn(shape, generator=generator, device=device) * spread)
    return x.to(torch.bfloat16)


def quant_inputs(generator, device):
    """BF16 rows over many binades with zero blocks, blocks whose amax sits
    on a scale boundary (448 x 2^k and just above it) and values on E4M3
    rounding midpoints."""
    x = bf16_values(generator, (96, 512), 4.0, device)
    x[0] = 0.0
    x[1, :32] = 448.0 * 2.0 ** -6
    x[1, 32:64] = (448.0 * 2.0 ** -6) * (1 + 2 ** -7)
    x[2, :32] = 1e-5
    mids = torch.tensor([1.0625, 1.1875, 240.0 + 8, 2 ** -10, 3 * 2 ** -10, 0.25, 0.75, 1.25, 2.5, 5.0],
                        device=device)
    x[3, :mids.numel()] = mids.to(torch.bfloat16)
    x[3, mids.numel():32] = 448.0
    x[4, :32] = 6.0 * 2.0 ** -9
    x[5, :16] = -0.0
    return x


def case_act_quant(vk, generator, device):
    x = quant_inputs(generator, device)
    records = []
    for block, fmt, scale_dtype in ((32, "ue8m0", torch.float8_e8m0fnu), (128, None, torch.float32)):
        vy, vs = vk.act_quant(x, block, fmt, scale_dtype)
        ry, rs = ours.act_quant(x, block, fmt, scale_dtype)
        records += [compare(f"codes_b{block}", vy, ry), compare(f"scales_b{block}", vs, rs)]
        vi, ri = x.clone(), x.clone()
        vk.act_quant(vi, block, fmt, scale_dtype, True)
        ours.act_quant(ri, block, fmt, scale_dtype, True)
        records.append(compare(f"inplace_b{block}", vi, ri))
    return records


def case_fp4_act_quant(vk, generator, device):
    x = quant_inputs(generator, device)
    records = []
    for block, scale_dtype in ((32, torch.float8_e8m0fnu), (16, torch.float8_e4m3fn)):
        vy, vs = vk.fp4_act_quant(x, block, False, scale_dtype)
        ry, rs = ours.fp4_act_quant(x, block, False, scale_dtype)
        records += [compare(f"codes_b{block}", vy.view(torch.uint8), ry.view(torch.uint8)),
                    compare(f"scales_b{block}", vs, rs)]
        vi, ri = x.clone(), x.clone()
        vk.fp4_act_quant(vi, block, True, scale_dtype)
        ours.fp4_act_quant(ri, block, True, scale_dtype)
        records.append(compare(f"inplace_b{block}", vi, ri))
    return records


def weight_codes(generator, n, k, device):
    codes = torch.randint(0, 256, (n, k), generator=generator, device=device, dtype=torch.uint8)
    codes[(codes & 0x7F) == 0x7F] = 0x30
    return codes.view(torch.float8_e4m3fn)


def gemm_oracle(a, sa, b, sb, block):
    """The exact product of the dequantized operands, in f64."""
    lhs = a.double() * ours.scale_values(sa).double().repeat_interleave(block, -1)
    return lhs @ b.t()


def case_fp8_gemm(vk, generator, device):
    records = []
    for m, n, k in ((1, 256, 512), (33, 512, 5120), (100, 256, 1280)):
        a = bf16_values(generator, (m, k), 2.0, device)
        if m > 1:
            # Cancellation: row 1's blocks cancel pairwise, then a small tail.
            a[1, :k // 2] = 3.0
            a[1, k // 2:] = -3.0
            a[1, -32:] = 1e-3
        codes, scales = ours.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
        b = weight_codes(generator, n, k, device)
        bs = torch.randint(118, 132, (n // 32, k // 32), generator=generator, device=device,
                           dtype=torch.uint8).view(torch.float8_e8m0fnu)
        vc = vk.fp8_gemm(codes, scales, b, bs, torch.float8_e8m0fnu, 32)
        rc = ours.fp8_gemm(codes, scales, b, bs, torch.float8_e8m0fnu, 32)
        weight = b.double() * ours.scale_values(bs).double().repeat_interleave(32, 0).repeat_interleave(32, 1)
        records.append(compare(f"m{m}_n{n}_k{k}", vc, rc, gemm_oracle(codes, scales, weight, None, 32)))
    return records


def case_fp4_gemm(vk, generator, device):
    records = []
    for m, n, k in ((1, 256, 512), (40, 2304, 5120), (17, 5120, 2304)):
        a = bf16_values(generator, (m, k), 2.0, device)
        codes, scales = ours.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu)
        packed = torch.randint(0, 256, (n, k // 2), generator=generator, device=device, dtype=torch.uint8)
        bs = torch.randint(118, 128, (n, k // 32), generator=generator, device=device,
                           dtype=torch.uint8).view(torch.float8_e8m0fnu)
        weight = packed.view(torch.float4_e2m1fn_x2)
        vc = vk.fp4_gemm(codes, scales, weight, bs, torch.float8_e8m0fnu, 32)
        rc = ours.fp4_gemm(codes, scales, weight, bs, torch.float8_e8m0fnu, 32)
        dense = ours.unpack_e2m1(packed).double() * ours.scale_values(bs).double().repeat_interleave(32, 1)
        records.append(compare(f"m{m}_n{n}_k{k}", vc, rc, gemm_oracle(codes, scales, dense, None, 32)))
    return records


def attention_oracle(q, kv, sink, idxs, scale):
    out = torch.zeros(q.shape, dtype=torch.float64, device=q.device)
    for row in range(q.shape[1]):
        ids = idxs[0, row]
        live = ids[ids != -1].long()
        if not live.numel():
            continue
        keys = kv[0, live].double()
        scores = torch.einsum("hd,td->ht", q[0, row].double(), keys) * scale
        weights = torch.softmax(torch.cat([scores, sink.double()[:, None]], -1), -1)[:, :-1]
        out[0, row] = weights @ keys
    return out


def case_sparse_attn(vk, generator, device):
    records = []
    # 64 heads (TP1) need more shared memory than SM120 gives the vendor kernel; 16 (TP4) and
    # 8 (TP8, padded to 16 by sparse_attn) are the per-rank widths it serves.
    for heads, n, topk in ((64, 700, 640), (16, 700, 640), (8, 200, 130), (16, 200, 130)):
        q = bf16_values(generator, (1, 48, heads, 512), 0.3, device)
        kv = bf16_values(generator, (1, n, 512), 0.3, device)
        sink = torch.randn(heads, generator=generator, device=device, dtype=torch.float32)
        idxs = torch.randint(0, n, (1, 48, topk), generator=generator, device=device, dtype=torch.int32)
        idxs[0, 0] = -1
        idxs[0, 1, 64:] = -1
        idxs[0, 2, ::3] = -1
        # A large score in a late tile rescales the earlier tiles.
        kv[0, idxs[0, 3, -1].long()] = (q[0, 3, 0].float() * 4).to(torch.bfloat16)
        scale = 512 ** -0.5
        try:
            vo = vk.sparse_attn(q, kv, sink, idxs, scale)
        except Exception as error:
            records.append({"output": f"h{heads}_n{n}_topk{topk}", "bitwise_equal": False,
                            "vendor_error": f"{type(error).__name__}: {error}"})
            continue
        ro = ours.sparse_attn(q, kv, sink, idxs, scale)
        records.append(compare(f"h{heads}_n{n}_topk{topk}", vo, ro, attention_oracle(q, kv, sink, idxs, scale)))
    return records


def case_sinkhorn(vk, generator, device):
    mixes = torch.randn(1, 4096, 24, generator=generator, device=device, dtype=torch.float32) * 3
    scale = torch.tensor([0.9, 1.1, 2.3], device=device, dtype=torch.float32)
    base = torch.randn(24, generator=generator, device=device, dtype=torch.float32)
    vendor = vk.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    replacement = ours.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    oracle = ours.hc_split_sinkhorn(mixes.double(), scale, base, 4, 20, 1e-6)
    return [compare(name, v, r, o) for name, v, r, o in zip(("pre", "post", "comb"), vendor, replacement, oracle)]


# Cases whose outputs are discrete decisions: any difference is semantic.
DISCRETE = ("act_quant", "fp4_act_quant")
CASES = {"act_quant": case_act_quant, "fp4_act_quant": case_fp4_act_quant, "fp8_gemm": case_fp8_gemm,
         "fp4_gemm": case_fp4_gemm, "sparse_attn": case_sparse_attn, "hc_split_sinkhorn": case_sinkhorn}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--self-check", action="store_true",
                        help="run the replacement as the vendor too (checks this script, on any device)")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_default_dtype(torch.bfloat16)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.set_device(device)
    report = {"tilelang": None, "torch": torch.__version__,
              "device": torch.cuda.get_device_name(device) if cuda else str(device),
              "capability": list(torch.cuda.get_device_capability(device)) if cuda else None,
              "kernel_sha256": KERNEL_SHA256, "lowering_ulps": LOWERING_ULPS, "self_check": args.self_check,
              "cases": {}}
    try:
        if args.self_check:
            vk = ours
        else:
            import tilelang
            report["tilelang"] = tilelang.__version__
            vk = vendor_kernels(args.checkpoint)
    except Exception as error:  # the vendor stack does not load: record why
        report["error"] = f"{type(error).__name__}: {error}"
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1
    status = 0
    for name in args.cases.split(","):
        generator = torch.Generator(device=device).manual_seed(20261017)
        started = time.perf_counter()
        try:
            records = CASES[name](vk, generator, device)
            report["cases"][name] = {"class": classify(records, name in DISCRETE), "records": records,
                                     "seconds": round(time.perf_counter() - started, 2)}
        except Exception as error:
            report["cases"][name] = {"class": "error", "error": f"{type(error).__name__}: {error}"}
            status = 1
        print(json.dumps({name: report["cases"][name]}), flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(case["class"] == "semantic" for case in report["cases"].values()):
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())

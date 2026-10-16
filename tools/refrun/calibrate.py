"""GPU calibration of the reference runner. Runs only inside a GPU job.

    python3 -m tools.refrun.calibrate --output DIR --shard FILE [--shard FILE ...]

Measures, under the runner's own mode policies:

- SGEMM (IEEE F32, no TF32), DGEMM and BF16 GEMM rates at MiMo and Qwen
  expert, attention and head shapes, the F32 error against DGEMM (TF32 would
  show as about 1e-3), and whether a repeated GEMM is bitwise equal;
- per-call dequantization time of MiMo and Qwen expert weights;
- the f64 math SDPA path and the query-chunked attention against a direct
  f64 computation, and attention rates at vision and text shapes;
- which deterministic-mode ops raise on CUDA;
- the NVMe read rate into pinned memory through the loader (cold: the
  shards' page cache is dropped first), raw cold read rates by request size
  and thread count, and the pinned host-to-device rate;
- the tiny synthetic model through `run.py` on all four GPUs, twice, with
  byte-equal score files required.

Writes DIR/calibration.json.
"""

import argparse
import json
import os
from pathlib import Path
import time
import traceback

from tools.refrun import modes

GEMMS = {
    "mimo.expert.gate_up": (2048, 4096), "mimo.expert.down": (4096, 2048),
    "mimo.attn.qkv": (13568, 4096), "mimo.attn.o": (4096, 8192),
    "mimo.dense.gate_up": (16384, 4096), "mimo.lm_head": (152576, 4096),
    "qwen.expert.gate_up": (640, 2560), "qwen.expert.down": (2560, 640),
    "qwen.attn.q": (12288, 2560), "qwen.lm_head": (248320, 2560),
    "qwen.vision.mlp": (4304, 1152),
}
ROWS = (8, 256, 4096)


def timed(step, minimum=0.25, warmup=2, limit=50):
    """Mean seconds per call of `step` on the current device."""
    import torch
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    count, started = 0, time.perf_counter()
    while count < limit and (count < 3 or time.perf_counter() - started < minimum):
        step()
        count += 1
        torch.cuda.synchronize()
    return (time.perf_counter() - started) / count


def gemms(device):
    import torch
    results = []
    for name, (n, k) in GEMMS.items():
        for m in ROWS if "lm_head" not in name else (128,):
            for mode in ("f64", "f32", "bf16"):
                policy = modes.policy(mode)
                dtype = modes.torch_dtype(policy["dtype"])
                generator = torch.Generator(device=device).manual_seed(m * 7 + n)
                x = torch.randn((m, k), generator=generator, device=device, dtype=torch.float64)
                w = torch.randn((n, k), generator=generator, device=device, dtype=torch.float64) / k ** 0.5
                with modes.active(policy):
                    xd, wd = x.to(dtype), w.to(dtype)
                    first = torch.nn.functional.linear(xd, wd)
                    seconds = timed(lambda: torch.nn.functional.linear(xd, wd))
                    repeat = torch.equal(torch.nn.functional.linear(xd, wd), first)
                row = {"gemm": name, "mode": mode, "m": m, "n": n, "k": k, "seconds": seconds,
                       "tflops": 2 * m * n * k / seconds / 1e12, "repeat_bitwise_equal": repeat}
                if mode != "f64":
                    exact = torch.nn.functional.linear(x.to(dtype).double(), w.to(dtype).double())
                    row["max_relative_error"] = ((first.double() - exact).abs().max() / exact.abs().max()).item()
                results.append(row)
                del x, w, xd, wd, first
            torch.cuda.empty_cache()
    return results


def dequantization(device):
    import torch
    from tools.refrun import quant
    generator = torch.Generator(device=device).manual_seed(5)
    cases = {
        "mimo.expert.mxfp4": (quant.MXFP4, (2048, 4096)),
        "mimo.attn.qkv.fp8_chunked": (quant.fp8_block(128, 128, 3392), (13568, 4096)),
        "qwen.expert.fp8": (quant.fp8_block(128, 128), (640, 2560)),
    }
    results = []
    for name, (fmt, (rows, cols)) in cases.items():
        packed = cols // 2 if fmt.element == "e2m1" else cols
        codes = torch.randint(0, 256, (rows, packed), generator=generator, device=device, dtype=torch.uint8)
        if fmt.element == "e4m3":
            codes[(codes & 0x7F) == 0x7F] = 0x38
            scale = torch.rand(fmt.scale_shape(rows, cols), generator=generator, device=device).to(torch.bfloat16)
        else:
            scale = torch.randint(118, 130, fmt.scale_shape(rows, cols), generator=generator, device=device,
                                  dtype=torch.uint8)
        tensor = quant.QuantTensor(fmt, codes, scale, (rows, cols))
        for mode in ("f32", "f64", "bf16"):
            policy = modes.policy(mode)
            dtype = modes.torch_dtype(policy["dtype"])
            with modes.active(policy):
                tensor.configure(dtype, policy["dequant"])
                seconds = timed(tensor.value)
            results.append({"weight": name, "mode": mode, "rows": rows, "cols": cols, "seconds": seconds,
                            "output_gb_per_s": rows * cols * dtype.itemsize / seconds / 1e9})
        quant.release_scratch()
    return results


def attention(device):
    import torch
    import torch.nn.functional as F
    from tools.refrun import ops
    results = {"checks": [], "rates": []}
    generator = torch.Generator(device=device).manual_seed(9)
    q, k, v = (torch.randn((1, 16, 2048, 72), generator=generator, device=device, dtype=torch.float64)
               for _ in range(3))
    scale = 72 ** -0.5
    with modes.active(modes.policy("f64")):
        exact = torch.softmax((q @ k.transpose(-1, -2)) * scale, dim=-1) @ v
        sdpa = F.scaled_dot_product_attention(q, k, v, scale=scale)
        chunked = ops.attention(q, k, v, scale, chunk=512)
        causal = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)
        mask = torch.full((2048, 2048), float("-inf"), device=device, dtype=torch.float64).triu(1)
        chunked_causal = ops.attention(q, k, v, scale, mask[None, None], chunk=512)
    for name, got, want in (("sdpa_math_f64", sdpa, exact), ("chunked_f64", chunked, exact),
                            ("chunked_causal_f64_vs_sdpa", chunked_causal, causal)):
        results["checks"].append({"check": name, "dtype": str(got.dtype),
                                  "max_abs_error": (got - want).abs().max().item()})
    del q, k, v, exact, sdpa, chunked, causal, chunked_causal, mask
    for name, heads, length, dim in (("qwen.vision", 16, 8192, 72), ("mimo.text", 64, 4096, 192)):
        for mode in ("f32", "f64", "bf16"):
            policy = modes.policy(mode)
            dtype = modes.torch_dtype(policy["dtype"])
            q, k, v = (torch.randn((1, heads, length, dim), generator=generator, device=device, dtype=dtype)
                       for _ in range(3))
            with modes.active(policy):
                seconds = timed(lambda: ops.attention(q, k, v, dim ** -0.5), minimum=0.5, warmup=1, limit=5)
            results["rates"].append({"shape": name, "mode": mode, "heads": heads, "length": length, "dim": dim,
                                     "chunk": ops.CHUNK, "seconds": seconds,
                                     "tflops": 4 * heads * length * length * dim / seconds / 1e12})
            del q, k, v
            torch.cuda.empty_cache()
    return results


def deterministic_ops(device):
    import torch
    results = {}
    x = torch.randn((64, 32), device=device)
    index = torch.randint(0, 16, (64,), device=device)
    cases = {
        "index_add_": lambda: torch.zeros((16, 32), device=device).index_add_(0, index, x),
        "scatter_add_": lambda: torch.zeros((16, 32), device=device).scatter_add_(
            0, index[:, None].expand(64, 32), x),
        "index_put_accumulate": lambda: torch.zeros((16, 32), device=device).index_put_((index,), x,
                                                                                      accumulate=True),
        "cumsum": lambda: torch.cumsum(x, dim=0),
        "sort_stable": lambda: torch.sort(x, dim=-1, descending=True, stable=True),
        "topk": lambda: torch.topk(x, 4, dim=-1),
        "bincount": lambda: torch.bincount(index),
        "histc": lambda: torch.histc(x, bins=8),
        "embedding": lambda: torch.nn.functional.embedding(index, x),
        "sdpa_math": lambda: torch.nn.functional.scaled_dot_product_attention(x[None, None], x[None, None],
                                                                              x[None, None]),
    }
    with modes.active(modes.policy("f32")):
        for name, case in cases.items():
            try:
                first, second = case(), case()
                torch.cuda.synchronize()
                same = all(torch.equal(a, b) for a, b in zip(first, second)) if isinstance(first, tuple) \
                    else torch.equal(first, second)
                results[name] = {"raised": None, "repeat_bitwise_equal": same}
            except Exception as error:  # noqa: BLE001 - recorded, not hidden
                results[name] = {"raised": f"{type(error).__name__}: {str(error).splitlines()[0]}"}
    return results


class Shards:
    """The tensors of chosen safetensors shards, as place.Store reads them."""

    def __init__(self, paths):
        from tools.refrun import place
        roots = {Path(p).parent for p in paths}
        if len(roots) != 1:
            raise ValueError("calibration shards must share one directory")
        self.root = roots.pop()
        self.entries, self.files = {}, {}
        for path in paths:
            entries = place.read_header(path)
            self.entries.update(entries)
            self.files[Path(path).name] = tuple(entries)


def reads(paths, devices):
    import torch
    from tools.refrun import place
    shards = Shards(paths)
    drop_cache(paths)
    units = [place.Unit(f"shard{i}", tensors) for i, tensors in enumerate(shards.files.values())]
    placement = place.Placement(tuple(devices), tuple((u.module, devices[i % len(devices)])
                                                      for i, u in enumerate(units)))
    store = place.Store(shards, placement, units)
    result = {"files": sorted(shards.files), "bytes": store.read_bytes, "read_seconds": store.read_seconds,
              "load_seconds": store.load_seconds,
              "read_gb_per_s": store.read_bytes / store.read_seconds / 1e9,
              "load_gb_per_s": store.read_bytes / store.load_seconds / 1e9}
    del store
    torch.cuda.empty_cache()
    pinned = torch.empty(1 << 30, dtype=torch.uint8, pin_memory=True)
    rates = {}
    for device in devices:
        target = torch.empty(1 << 30, dtype=torch.uint8, device=device)
        with torch.cuda.device(device):
            seconds = timed(lambda: target.copy_(pinned, non_blocking=True), minimum=0.5, limit=10)
        rates[device] = (1 << 30) / seconds / 1e9
        del target
    result["h2d_gb_per_s"] = rates
    return result


def drop_cache(paths):
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def raw_reads(paths):
    """Cold host-only read rates by request size and thread count, to show
    what the loader's one-thread, one-tensor-per-pread reading leaves."""
    from concurrent.futures import ThreadPoolExecutor

    def read(path, size):
        buffer = memoryview(bytearray(size))
        fd = os.open(path, os.O_RDONLY)
        try:
            offset = 0
            while count := os.preadv(fd, [buffer], offset):
                offset += count
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return offset
    results = []
    for size, threads in ((4 << 20, 1), (64 << 20, 1), (64 << 20, len(paths))):
        drop_cache(paths)
        started = time.perf_counter()
        with ThreadPoolExecutor(threads) as pool:
            total = sum(pool.map(lambda path: read(path, size), paths))
        seconds = time.perf_counter() - started
        results.append({"request_bytes": size, "threads": threads, "bytes": total, "seconds": seconds,
                        "gb_per_s": total / seconds / 1e9})
    return results


def tiny_runs(output, devices):
    from tools.refrun import keys, run, tiny
    checkpoint = tiny.write_checkpoint(output / "tiny-checkpoint")
    workload = output / "tiny-workload.json"
    workload.write_text(json.dumps({"schema": "chopinfer.refrun.workload.v1", "model": "tiny", "sequences": [
        {"name": "text", "category": "prose", "token_ids": list(range(1, 90)), "score_from": 1},
        {"name": "greedy", "category": "probe", "token_ids": [3, 4, 5, 6], "score_from": 2,
         "answer": {"policy": "greedy", "max_new_tokens": 12, "stop_ids": [95]}}]}))
    provenance = []
    for attempt in ("a", "b"):
        provenance.append(run.run(run.parse_args([
            "--adapter", str(keys.ROOT / "tools/refrun/tiny.py"), "--checkpoint", str(checkpoint),
            "--lock", str(checkpoint / tiny.LOCK), "--workload", str(workload),
            "--devices", ",".join(devices), "--output", str(output / f"tiny-run-{attempt}"), "--no-cache"])))
    equal = {mode: (output / f"tiny-run-a/{mode}.json").read_bytes()
             == (output / f"tiny-run-b/{mode}.json").read_bytes() for mode in ("f32", "bf16", "f64")}
    return {"placement": provenance[0]["placement"], "modes": provenance[0]["modes"],
            "runs_byte_equal": equal}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calibrate the reference runner on the GPUs.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard", action="append", required=True, help="checkpoint shard for the read rate")
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3")
    args = parser.parse_args(argv)
    modes.prepare_environment(cuda=True)
    import torch
    from tools.refrun import keys
    args.output.mkdir(parents=True, exist_ok=False)
    devices = args.devices.split(",")
    results = {"environment": {**keys.environment("sm_%d%d" % torch.cuda.get_device_capability(devices[0])),
                               "devices": [torch.cuda.get_device_name(d) for d in devices]}}
    sections = (("deterministic_ops", lambda: deterministic_ops(devices[0])),
                ("attention", lambda: attention(devices[0])),
                ("dequantization", lambda: dequantization(devices[0])),
                ("gemm", lambda: gemms(devices[0])),
                ("reads", lambda: reads(args.shard, devices)),
                ("raw_reads", lambda: raw_reads(args.shard)),
                ("tiny_runner", lambda: tiny_runs(args.output, devices)))
    failed = []
    for name, section in sections:
        started = time.perf_counter()
        try:
            results[name] = section()
        except Exception:  # noqa: BLE001 - one failed section must not hide the others
            results[name] = {"error": traceback.format_exc()}
            failed.append(name)
        results.setdefault("seconds", {})[name] = time.perf_counter() - started
        print(f"calibrate {name}: {results['seconds'][name]:.1f}s", flush=True)
        (args.output / "calibration.json").write_text(json.dumps(results, indent=2) + "\n")
    if failed:
        raise SystemExit(f"calibration sections failed: {failed}")


if __name__ == "__main__":
    main()

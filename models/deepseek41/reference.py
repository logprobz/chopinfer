"""DeepSeek V4.1 Flash reference adapter for the GPU reference runner (tools/refrun).

It executes the checkpoint's own `inference/model.py`, `engram.py`,
`vision.py` and `image_processor.py` from the bytes whose sha256 is pinned in
VENDOR (the checkpoint's `inference/` at revision 1bae0e10, byte-identical to
the program's pinned copies), with `sys.modules["kernel"]` seeded with
reference_kernel.py instead of the tilelang `kernel.py` (tilelang is never
installed in the runner's environment). reference_kernel.LEDGER records each
replaced kernel; REPLACEMENTS records everything else the adapter changes.

Weights: every checkpoint tensor feeds the vendor parameter of the same name
(the checkpoint already uses the inference code's names), as the vendor's
`load_model` copies converted shards into the model built by `Transformer`:

- FP8 E4M3 weights with E8M0 scales per 32 x 32 block and packed E2M1 expert
  weights (I8, even K in the low nibble, `float4_e2m1fn_x2`) with an E8M0
  scale per 32 K stay as stored; the vendor's `linear` quantizes the
  activation and calls the replaced `fp8_gemm` or `fp4_gemm` on them.
- `attn.wo_a` is dequantized to BF16 as convert.py:157-173 does (code x
  scale per 32 x 32 block, in F32, then BF16), checked exact.
- Every other tensor is copied into its parameter's declared dtype; where the
  vendor declares BF16 (or takes the default dtype) the f32 and f64 modes
  widen it to the mode dtype (exact).
- The Engram tables (`layers.{1,14}.engram.embed`, 197 GB) stay in their
  shards and are gathered by row on the host (`HostEngram`, place.HostRows
  over a thread pool), then decoded by the vendor's own arithmetic.
- The MTP (DSpark) layers are not loaded; the teacher-forced forward pass
  does not run them.

Modes: bf16 runs the vendor code in its own dtype. f32 runs the same code
with F32 as the default dtype. f64 runs a widened copy of model.py and
vision.py (`WIDEN`): every `.float()` becomes `.double()` and every
`torch.float32` becomes `torch.float64`, except on the lines that build the
RoPE and vision rotary angles (and `sample`, which is not run), so the
vendor's F32 steps (norm statistics, mHC, gate scores, SwiGLU, the head)
run in f64. In every mode the quantizers read their inputs rounded to BF16
(trained quantizer inputs, reference_kernel.POLICY): F32-TQ is the gate
reference, F64-TQ and BF16 the spread.

RoPE tables (V5): the vendor's F32 angles from `precompute_freqs_cis` as
CUDA evaluates it (the program's RoPE pin, `FREQUENCY_BITS`; three pairs
differ from a CPU evaluation), with cos and sin correctly rounded to F32
from those angles, in every mode.

`forward` is `Transformer.forward` over every prompt row without sampling,
with the head over the last `keep` rows; the head runs in f64 in f64 mode.
Greedy `decode` continues one token at a time from the vendor's caches.
Media come from the workload as the vendor's BF16 patches with their
[n_vit_h, n_vit_w] grid; spans and token types are the vendor's layout.
"""

from concurrent.futures import ThreadPoolExecutor
import contextlib
import functools
import hashlib
import json
from pathlib import Path
import sys
import types
import weakref

import torch
from torch import nn
import torch.nn.functional as F

from tools.refrun import adapter, taps as tap_writer
from tools.refrun.place import HostRows, Unit
from tools.refrun.taps import Hook

MODEL = "deepseek41"
HERE = Path(__file__).resolve().parent
# The checkpoint's inference code at revision 1bae0e10 (also the program's
# pinned reference copies).
VENDOR = {
    "inference/model.py": "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d",
    "inference/kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
    "inference/engram.py": "11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897",
    "inference/vision.py": "5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c",
    "inference/image_processor.py": "482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272",
    "inference/convert.py": "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76",
}
# The official checkpoint's inference config; a synthetic checkpoint brings its own.
CONFIG = "inference/config.json"
OFFICIAL_CONFIG = "2e84f45cf1dac8c7fcbb200e96667d4b913275690668ed496f24c7747207a809"
# Helper modules the runner does not key by themselves (it keys this file).
HELPERS = {
    "reference_kernel.py": "9725729544cbc388631341a2637de165e5a6fde6aa6376c8bc9c8d6be16373b7",
    "reference_memory.py": "6e931c8ee836388b726a712ce9651da84e64041139e1136d3e71858b60c67778",
}
# Executed vendor modules: the stem, and whether f64 widens it.
STEMS = (("engram", False), ("image_processor", False), ("vision", True), ("model", True))
PACKAGE = "chopinfer_deepseek41_vendor"
# f64: lines of the widened files that keep F32 (1-based), and how many
# replacements each widening makes, so a changed source is refused.
WIDEN = {
    "model": {"keep": (376, 384, 1294), "float": 19, "float32": 12},
    "vision": {"keep": (10, 13), "float": 2, "float32": 1},
}
TOKENIZER = "tokenizer.json"
# The official tokenizer: NgramHashState derives the Engram token map from it.
OFFICIAL_TOKENIZER = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"
# Largest sequence (prompt and answer) a built model holds: the vendor sizes
# its KV caches, index keys and RoPE tables by it.
MAX_SEQ_LEN = 135168
ENGRAM_THREADS = 16
QUANTIZED = (torch.float8_e4m3fn, torch.float4_e2m1fn_x2)
REPLACEMENTS = {
    "kernel (tilelang)": "reference_kernel.py: act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn and "
                         "hc_split_sinkhorn (reference_kernel.LEDGER)",
    "precompute_freqs_cis": "the vendor's F32 angles as CUDA evaluates lines 376-388 (FREQUENCY_BITS, the RoPE "
                            "pin) with cos and sin correctly rounded to F32 from them (torch.polar's F32 sincos is "
                            "not)",
    "get_window_topk_idxs": "the vendor's indices, cached per device (its lru_cache holds one device's tensor)",
    "ParallelEngramEmbedding.forward": "HostEngram: rows gathered on the host (place.HostRows, thread pool), then "
                                       "the vendor's decode (F32 product, BF16)",
    "attn.wo_a": "dequantized to BF16 at load as convert.py:157-173 does",
    "Transformer.forward": "Plan.forward: reference_memory.prefill (row blocks, query-blocked indexer, the stream on "
                           "the host; reference_memory.REPLACEMENTS), no sampling, head over the kept rows",
    "Indexer.forward, select_candidate_blocks": "reference_memory.install: the vendor lines with ties to the lower "
                                                "index, in prefill and decode",
    "Gate.forward": "reference_memory.install: the vendor lines with ties to the lower expert index in the routing "
                    "top-k, in prefill and decode",
    "ViT.forward": "the vendor method with the vision tower's device as torch's default device, where generate.py "
                   "runs it (set_default_device): the rotary tables (vision.py:8-15) are made on that device",
    "Block.hc_mixes": "the vendor method, its pre/post/comb passed through an identity module for taps",
    "shared_attn": "share_to: before each layer, its tensors are copied to the layer's device when they live on "
                   "another (the vendor runs one device per rank)",
    "f64": "model.py and vision.py widened (WIDEN): .float() to .double(), torch.float32 to torch.float64, "
           "except the RoPE and vision rotary angle lines",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def helper(name):
    """A helper module executed from its pinned bytes."""
    path = HERE / name
    data = path.read_bytes()
    if sha256(data) != HELPERS[name]:
        raise ValueError(f"{name} is not the pinned {HELPERS[name]}; update HELPERS")
    module_name = f"{PACKAGE}_{path.stem}"
    if module_name not in sys.modules:
        module = types.ModuleType(module_name)
        module.__file__ = str(path)
        exec(compile(data, str(path), "exec"), module.__dict__)
        module.EXECUTED_SHA256 = HELPERS[name]
        sys.modules[module_name] = module
    return sys.modules[module_name]


def kernels():
    return helper("reference_kernel.py")


def memory():
    return helper("reference_memory.py")


def widen(stem, source):
    """The f64 copy of a vendor file (WIDEN)."""
    rule = WIDEN[stem]
    lines = source.decode().split("\n")
    counts = {"float": 0, "float32": 0}
    for index, line in enumerate(lines):
        if index + 1 in rule["keep"]:
            continue
        counts["float"] += line.count(".float()")
        counts["float32"] += line.count("torch.float32")
        lines[index] = line.replace(".float()", ".double()").replace("torch.float32", "torch.float64")
    if counts != {"float": rule["float"], "float32": rule["float32"]}:
        raise ValueError(f"{stem}.py widening made {counts}, not the pinned {rule}")
    return "\n".join(lines).encode()


def vendor(checkpoint, wide=False):
    """The executed vendor package (f64-widened when `wide`), loaded once per
    process. model.py imports `kernel`, `engram`, `image_processor` and
    `vision` by top-level name, so they are seeded in sys.modules for the
    exec and restored after it."""
    package = f"{PACKAGE}_wide" if wide else PACKAGE
    for file, digest in VENDOR.items():
        checkpoint.pinned(file, digest)
    if f"{package}.model" in sys.modules:
        return sys.modules[f"{package}.model"]
    root = types.ModuleType(package)
    root.__path__ = []
    sys.modules[package] = root
    seeded = {"kernel": kernels()}
    saved = {name: sys.modules.get(name) for name in ("kernel", *(stem for stem, _ in STEMS))}
    try:
        sys.modules["kernel"] = seeded["kernel"]
        for stem, widened in STEMS:
            source = checkpoint.pinned(f"inference/{stem}.py", VENDOR[f"inference/{stem}.py"])
            if wide and widened:
                source = widen(stem, source)
            module = types.ModuleType(f"{package}.{stem}")
            module.__file__ = str(checkpoint.root / "inference" / f"{stem}.py")
            exec(compile(source, module.__file__, "exec"), module.__dict__)
            sys.modules[module.__name__] = module
            sys.modules[stem] = module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    model = sys.modules[f"{package}.model"]
    install(model)
    # Lower-index selection in the vendor's Indexer.forward and
    # select_candidate_blocks, for prefill and decode alike.
    memory().install(types.SimpleNamespace(model=model))
    model.EXECUTED_SHA256 = {file: VENDOR[file] for file in VENDOR}
    model.WIDE = wide
    return model


# The vendor's F32 rotary frequencies as CUDA evaluates precompute_freqs_cis
# (model.py:376-386) under generate.py's torch.device("cuda"), the program's
# RoPE pin: three pairs differ from a CPU evaluation (CUDA's powf in window
# pairs 11 and 25, the YaRN ramp divided as a product with the reciprocal in
# compressed pair 24). Keyed by (dim, original_seq_len, base, factor,
# beta_fast, beta_slow); without YaRN only (dim, 0, base) matter.
FREQUENCY_BITS = {
    (64, 0, 10000): (
        0x3f800000, 0x3f3ff911, 0x3f0ff59a, 0x3ed7e89b, 0x3ea1e89b, 0x3e72d423, 0x3e361887, 0x3e088d77,
        0x3dcccccd, 0x3d99940d, 0x3d6655c2, 0x3d2cba16, 0x3d0186e3, 0x3cc2434f, 0x3c91ad39, 0x3c5a7bf2,
        0x3c23d70a, 0x3bf5b9b0, 0x3bb8449c, 0x3b8a2e77, 0x3b4f3e38, 0x3b1b690d, 0x3ae91528, 0x3aaec98e,
        0x3a83126f, 0x3a44948d, 0x3a136a16, 0x39dd1725, 0x39a5cb60, 0x3978a815, 0x393a7753, 0x390bd472),
    (64, 65536, 160000, 16, 32, 1): (
        0x3f800000, 0x3f300a3a, 0x3ef21c1f, 0x3ea67d01, 0x3e64f92e, 0x3e1d7475, 0x3dd88cb4, 0x3d94e963,
        0x3d4ccccd, 0x3d0cd4fb, 0x3cc1b019, 0x3c8530ce, 0x3c372dbf, 0x3bfbed88, 0x3bad3d5e, 0x3b6e4237,
        0x3b147ae1, 0x3ab714e0, 0x3a5ebdb6, 0x3a0530ce, 0x399bb3af, 0x39305978, 0x38be904d, 0x383e9b5f,
        0x37a3d708, 0x36b443d0, 0x3677eba6, 0x362a7be8, 0x35ea77ff, 0x35a13bdc, 0x355dbf30, 0x35187c4d),
}
# Where an unpinned geometry (a synthetic checkpoint's) evaluates the vendor's
# frequency lines: build sets it to the first placed device, as the vendor
# builds on its GPU.
ROPE_DEVICE = ["cpu"]


def rope_key(dim, original_seq_len, base, factor, beta_fast, beta_slow):
    if original_seq_len > 0:
        return (dim, original_seq_len, base, factor, beta_fast, beta_slow)
    return (dim, 0, base)


def rope_frequencies(dim, original_seq_len, base, factor, beta_fast, beta_slow):
    """The F32 frequency of each pair: the pin, else model.py:376-386 on ROPE_DEVICE."""
    import math
    bits = FREQUENCY_BITS.get(rope_key(dim, original_seq_len, base, factor, beta_fast, beta_slow))
    if bits is not None:
        return torch.tensor(bits, dtype=torch.int64, device="cpu").to(torch.int32).view(torch.float32)
    with torch.device(ROPE_DEVICE[0]):
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if original_seq_len > 0:
            def corrected_dim(rotations):
                return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(corrected_dim(beta_fast)), 0)
            high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
            ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
            smooth = 1 - ramp
            freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs.cpu()


def rope_angles(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """model.py:388: the F32 angle fl32(fl32(p) x f_i) of every (position,
    pair), from rope_frequencies, on the CPU (one F32 product each)."""
    freqs = rope_frequencies(dim, original_seq_len, base, factor, beta_fast, beta_slow)
    with torch.device("cpu"):
        return torch.outer(torch.arange(seqlen), freqs)


def rope_table(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """precompute_freqs_cis with cos and sin correctly rounded to F32 from the
    vendor's CUDA F32 angles (V5, the RoPE pin), complex64 on the CPU."""
    return cached_rope_table(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, ROPE_DEVICE[0])


@functools.lru_cache(4)
def cached_rope_table(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow, _device):
    angles = rope_angles(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow).double()
    return torch.complex(torch.cos(angles).float(), torch.sin(angles).float())


@functools.lru_cache(16)
def window_topk(original, window_size, bsz, seqlen, start_pos, device):
    with torch.device(device):
        return original(window_size, bsz, seqlen, start_pos)


def install(model):
    """Module-level replacements in an executed model.py."""
    original = model.get_window_topk_idxs.__wrapped__
    model.precompute_freqs_cis = rope_table

    def get_window_topk_idxs(window_size, bsz, seqlen, start_pos):
        return window_topk(original, window_size, bsz, seqlen, start_pos, str(torch.get_default_device()))
    model.get_window_topk_idxs = get_window_topk_idxs


def tokenizer_bytes(checkpoint):
    """tokenizer.json, which the Engram token map (engram.py:145-164) reads:
    with the official inference config it must be the official tokenizer; a
    synthetic checkpoint brings its own, keyed through reference_code."""
    data = (checkpoint.root / TOKENIZER).read_bytes()
    official = sha256((checkpoint.root / CONFIG).read_bytes()) == OFFICIAL_CONFIG
    if official and sha256(data) != OFFICIAL_TOKENIZER:
        raise ValueError(f"{TOKENIZER} is not the official {OFFICIAL_TOKENIZER} the official config needs")
    return data


def reference_code(checkpoint):
    """The executed vendor files, the kernel.py they replace, convert.py whose
    wo_a rule load applies, the model's inference config and the tokenizer
    the Engram token map derives from."""
    files = {}
    for file, digest in VENDOR.items():
        checkpoint.pinned(file, digest)
        files[f"deepseek41/{file}"] = checkpoint.root / file
    tokenizer_bytes(checkpoint)
    files[f"deepseek41/{CONFIG}"] = checkpoint.root / CONFIG
    files[f"deepseek41/{TOKENIZER}"] = checkpoint.root / TOKENIZER
    return files


def model_args(checkpoint, modeling, max_seq_len=MAX_SEQ_LEN):
    """ModelArgs from the checkpoint's inference config, one sequence at a time."""
    config = json.loads((checkpoint.root / CONFIG).read_text())
    return modeling.ModelArgs(**{**config, "max_batch_size": 1, "max_seq_len": max_seq_len})


def units(checkpoint):
    """Vision (with the aligner and the image span embeddings), the token
    embedding, each decoder layer (its Engram table on the host), the final
    norm, the head. MTP layers are not run and not loaded."""
    names = sorted(checkpoint.entries)
    layers = json.loads((checkpoint.root / CONFIG).read_text())["n_layers"]

    def owned(*prefixes):
        return tuple(n for n in names if n.startswith(prefixes))
    result = []
    vision = owned("vision.", "aligner.", "image_start", "image_end", "image_newline")
    if vision:
        result.append(Unit("vision", vision))
    result.append(Unit("embed", owned("embed.")))
    for layer in range(layers):
        prefix = f"layers.{layer}."
        host = owned(f"{prefix}engram.embed.")
        result.append(Unit(f"layers.{layer}", tuple(n for n in owned(prefix) if n not in host)))
        if host:
            result.append(Unit(f"layers.{layer}.engram.embed", host, host=True))
    result += [Unit("norm", owned("norm.")), Unit("head", owned("head."))]
    return result


class Tokenizer:
    """What NgramHashState reads of an HF tokenizer: the Rust backend and the
    vocabulary size with added tokens (len(AutoTokenizer) for this checkpoint)."""

    def __init__(self, source):
        import tokenizers
        data = source if isinstance(source, bytes) else Path(source).read_bytes()
        self.backend_tokenizer = tokenizers.Tokenizer.from_str(data.decode())

    def __len__(self):
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


class HostEngram(nn.Module):
    """ParallelEngramEmbedding.forward over host rows: each distinct index's
    E4M3 row and E8M0 scales read from the shard (HostRows, in parallel),
    then model.py:317-321's decode on the device: F32 product per 32-value
    block, BF16. One rank holds every row, so no index is masked."""

    def __init__(self, checkpoint, prefix, block_size, rows, device):
        super().__init__()
        self.values = HostRows(checkpoint, prefix + "weight")
        self.scales = HostRows(checkpoint, prefix + "scale")
        self.block_size, self.num_embeddings, self.device = block_size, rows, torch.device(device)
        if self.values.entry.shape[0] != rows or self.scales.entry.shape[0] != rows:
            raise ValueError(f"{prefix} holds {self.values.entry.shape[0]} rows, not {rows}")

    def gather(self, table, indices):
        chunks = [indices[i:i + 4096] for i in range(0, len(indices), 4096)]
        with ThreadPoolExecutor(ENGRAM_THREADS) as pool:
            return torch.cat(list(pool.map(table.rows, chunks)))

    def forward(self, indices):
        flat = indices.reshape(-1).cpu()
        unique, inverse = torch.unique(flat, return_inverse=True)
        rows = unique.tolist()
        values = self.gather(self.values, rows).to(self.device)
        scales = self.gather(self.scales, rows).to(self.device)
        inverse = inverse.to(self.device)
        values = F.embedding(inverse, values.view(torch.uint8)).view(torch.float8_e4m3fn)
        scales = F.embedding(inverse, scales.view(torch.uint8)).view(torch.float8_e8m0fnu)
        values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16).view(*indices.shape, -1)


class Probe(nn.Module):
    """Passes its input through, so taps can read a value no module returns."""

    def forward(self, *values):
        return values


def probe_hc_mixes(block):
    """Block.hc_mixes, its (pre, post, comb) passed through `hc_attn_probe`
    or `hc_ffn_probe` by which coefficients it was given."""
    vendor_mixes = type(block).hc_mixes
    block.hc_attn_probe, block.hc_ffn_probe = Probe(), Probe()
    # A weak reference: a closure over the block would put the model in a
    # cycle, so its device tensors would outlive the runner's release of a
    # mode until the cyclic collector ran.
    owner = weakref.ref(block)

    def hc_mixes(x, hc_fn, hc_scale, hc_base):
        block = owner()
        probe = block.hc_attn_probe if hc_fn is block.hc_attn_fn else block.hc_ffn_probe
        return probe(*vendor_mixes(block, x, hc_fn, hc_scale, hc_base))
    block.hc_mixes = hc_mixes


def on_default_device(module, device):
    """`module`'s forward with `device` as torch's default device, as generate.py
    runs every module after torch.set_default_device("cuda"), so the tensors
    the vendor makes without a device are made there."""
    vendor_forward = type(module).forward
    owner = weakref.ref(module)  # as in probe_hc_mixes: no cycle through the module

    def forward(*args, **kwargs):
        with torch.device(device):
            return vendor_forward(owner(), *args, **kwargs)
    module.forward = forward


def wo_a_bf16(weight, scale):
    """convert.py:157-173: an FP8 wo_a with E8M0 block scales as BF16, each
    value code x scale in F32, then rounded to BF16; refused unless exact."""
    rows, cols = weight.shape
    if rows % scale.shape[0] or cols % scale.shape[1]:
        raise ValueError(f"wo_a {list(weight.shape)} does not split into {list(scale.shape)} blocks")
    out_block, in_block = rows // scale.shape[0], cols // scale.shape[1]
    if (out_block, in_block) not in ((32, 32), (128, 128)):
        raise ValueError(f"wo_a blocks are {out_block} x {in_block}, not 32 x 32 or 128 x 128")
    product = (weight.unflatten(0, (-1, out_block)).unflatten(-1, (-1, in_block)).float()
               * scale[:, None, :, None].float())
    value = product.flatten(2, 3).flatten(0, 1).bfloat16()
    if not torch.equal(value.float(), product.flatten(2, 3).flatten(0, 1)):
        raise ValueError("wo_a is not exact in BF16")
    return value


def parameter_dtype(declared, mode):
    """The dtype a stored tensor takes: quantized codes and scales as stored,
    BF16 declarations widened to the mode dtype, others as declared."""
    if declared in (*QUANTIZED, torch.float8_e8m0fnu):
        return declared
    if declared == torch.bfloat16:
        return mode
    return declared


def stored_view(tensor, declared):
    """A stored tensor viewed as the vendor declares its codes."""
    if declared == torch.float4_e2m1fn_x2:
        return tensor.view(torch.uint8).view(torch.float4_e2m1fn_x2)
    return tensor


BUFFER_FILL = {"score_state": float("-inf")}


def materialize(model, device_of, dtype):
    """Every buffer the meta build left unallocated, on its module's device:
    the vendor's zero-initialized caches (score_state -inf), and the CPU RoPE
    tables moved, one copy per device. Refuses any other buffer."""
    tables = {}
    for path, module in model.named_modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is None:
                continue
            device = device_of(path)
            if name == "freqs_cis":
                key = (id(buffer), device)
                if key not in tables:
                    tables[key] = buffer.to(device)
                module._buffers[name] = tables[key]
            elif buffer.is_meta:
                if name not in ("window_kv_cache", "compress_kv_cache", "k_cache", "kv_state", "score_state"):
                    raise ValueError(f"{path}.{name}: no rule to materialize this vendor buffer")
                module._buffers[name] = torch.full(buffer.shape, BUFFER_FILL.get(name, 0.0), dtype=buffer.dtype,
                                                   device=device)


def build(config, placement, max_seq_len=MAX_SEQ_LEN, quantizer_inputs="bfloat16"):
    """The vendor Transformer for one mode around the store's tensors."""
    checkpoint, store, dtype = config.checkpoint, config.store, config.dtype
    modeling = vendor(checkpoint, wide=dtype == torch.float64)
    kernel = kernels()
    kernel.set_policy(quantizer_inputs)
    ROPE_DEVICE[0] = placement.first
    args = model_args(checkpoint, modeling, max_seq_len)
    devices = dict(placement.units)

    def device_of(path):
        while path and path not in devices:
            path = path.rpartition(".")[0]
        return torch.device(devices[path]) if path else torch.device(placement.first)
    tokenizer = Tokenizer(tokenizer_bytes(checkpoint))
    default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        # NgramHashState holds computed tables, so it is built on the CPU
        # after the meta build, which would leave them unallocated.
        hash_state, modeling.NgramHashState = modeling.NgramHashState, lambda *_args: None
        try:
            with torch.device("meta"):
                model = modeling.Transformer(args)
        finally:
            modeling.NgramHashState = hash_state
        model.engram_hash = hash_state(args, model.engram_layout, tokenizer) \
            if model.engram_layout is not None else None
    finally:
        torch.set_default_dtype(default)
    model.mtp = nn.ModuleList()
    names = {}
    modules = dict(model.named_modules())
    for name, tensor in store.tensors.items():
        parent, _, leaf = name.rpartition(".")
        if leaf == "scale" and parent.endswith(".attn.wo_a"):
            # Folded into the BF16 weight below, as convert.py folds it.
            names[name] = f"{parent}.weight"
            continue
        module = modules.get(parent)
        declared = getattr(module, leaf, None) if module is not None else None
        if not isinstance(declared, torch.Tensor):
            raise ValueError(f"{name} has no vendor parameter")
        if leaf == "weight" and parent.endswith(".attn.wo_a"):
            value = wo_a_bf16(tensor, store.get(f"{parent}.scale")).to(parameter_dtype(declared.dtype, dtype))
        else:
            value = stored_view(tensor, declared.dtype).to(parameter_dtype(declared.dtype, dtype))
        if tuple(value.shape) != tuple(declared.shape):
            raise ValueError(f"{name} is {list(value.shape)}, the vendor declares {list(declared.shape)}")
        module._parameters[leaf] = nn.Parameter(value, requires_grad=False)
        names[name] = name
    for module in model.modules():
        if isinstance(module, modeling.Linear) and module.scale is not None:
            module.weight.scale = module.scale
            if module.weight.is_meta or module.scale.is_meta:
                raise ValueError("a quantized Linear has no stored weight or scale")
    for layer, block in enumerate(model.layers):
        if block.engram is not None:
            prefix = f"layers.{layer}.engram.embed."
            embed = block.engram.embed
            block.engram.embed = HostEngram(checkpoint, prefix, embed.block_size, embed.num_embeddings,
                                            device_of(f"layers.{layer}"))
            names[prefix + "weight"] = names[prefix + "scale"] = "host:engram-rows"
        probe_hc_mixes(block)
        if block.attn.indexer is not None:
            block.attn.selection = Probe()
    materialize(model, device_of, dtype)
    left = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if left:
        raise ValueError(f"{len(left)} vendor tensors have no stored value, e.g. {left[:5]}")
    if model.vision is not None and len({device_of(u) for u in ("vision", "embed")}) != 1:
        raise ValueError("the vision tower and the token embedding must share a device")
    if model.vision is not None:
        on_default_device(model.vision, device_of("vision"))
    model.eval()
    plan = Plan(args, modeling, device_of, dtype)
    return adapter.Reference(
        model, names, layout=layout(args, dtype, quantizer_inputs, max_seq_len), replacements=REPLACEMENTS,
        media_inputs=media_inputs, taps=taps, forward=plan.forward, decode=plan.decode)


def layout(args, dtype, quantizer_inputs, max_seq_len):
    return {
        "fp8": "E4M3 weights with an E8M0 scale per 32 x 32 block; activations quantized to E4M3 per 32 with a "
               "power-of-two E8M0 scale (the vendor's linear)",
        "fp4_experts": "packed E2M1 (I8, even K in the low nibble) with an E8M0 scale per 32 K",
        "wo_a": "FP8 32 x 32 dequantized to BF16 at load (convert.py), checked exact",
        "bf16_widening": "tensors the vendor declares BF16 or in the default dtype take the mode dtype",
        "engram": "tables on the host, rows gathered per forward pass",
        "mtp": "not loaded or run",
        "quantizer_inputs": quantizer_inputs, "kernel_policy": kernels().POLICY.document(),
        "kernel_ledger": kernels().LEDGER, "widen": {stem: dict(rule, keep=list(rule["keep"]))
                                                      for stem, rule in WIDEN.items()} if dtype == torch.float64
        else None,
        "rope": "vendor F32 angles as CUDA evaluates them (pinned frequencies), cos and sin correctly rounded to "
                "F32, in every mode",
        "max_seq_len": max_seq_len, "n_layers": args.n_layers, "compress_ratios": list(args.compress_ratios),
    }


def image_spans(ids, items, args, image_processor):
    """The vendor's ImageInput of each image: its span is the next run of
    `image_token_id` of num_image_tokens(n_llm_h, n_llm_w) positions."""
    import math
    inputs, position, row = [], 0, ids.tolist()
    for patches, (n_vit_h, n_vit_w) in items:
        n_llm_h = math.ceil(n_vit_h / args.vision_downsample_ratio)
        n_llm_w = math.ceil(n_vit_w / args.vision_downsample_ratio)
        types_ = image_processor.image_token_types(n_llm_h, n_llm_w)
        while position < len(row) and row[position] != args.image_token_id:
            position += 1
        span = row[position:position + types_.numel()]
        if len(span) != types_.numel() or any(token != args.image_token_id for token in span):
            raise ValueError(f"image {len(inputs)} has no span of {types_.numel()} image tokens")
        inputs.append(image_processor.ImageInput(position, patches, n_vit_h, n_vit_w, types_))
        position += types_.numel()
    if any(token == args.image_token_id for token in row[position:]):
        raise ValueError("image tokens left after the last image's span")
    return inputs


def media_inputs(sequence, device, dtype):
    """Each image's vendor patches [n_vit_h x n_vit_w, 3, 14, 14] (BF16, as
    load_image makes them; widened to the mode dtype) and its [n_vit_h,
    n_vit_w] grid."""
    items = []
    for media in sequence.media:
        if media.kind != "image" or media.tensor.dtype != "bf16" or media.grid is None or media.grid.dtype != "i64":
            raise ValueError(f"{sequence.name}: DeepSeek V4.1 Flash takes BF16 image patches with an I64 grid")
        patches = torch.frombuffer(bytearray(media.tensor.data), dtype=torch.bfloat16).view(media.tensor.shape)
        grid = torch.frombuffer(bytearray(media.grid.data), dtype=torch.int64).tolist()
        if len(grid) != 2 or patches.shape[0] != grid[0] * grid[1]:
            raise ValueError(f"{sequence.name}: patches {list(patches.shape)} do not fill the grid {grid}")
        items.append((patches.to(device=device, dtype=dtype), tuple(grid)))
    return {"images": items}


class Plan:
    """The forward and decode schedule of one built model."""

    def __init__(self, args, modeling, device_of, dtype):
        self.args, self.modeling, self.device_of, self.dtype = args, modeling, device_of, dtype

    @contextlib.contextmanager
    def default_dtype(self):
        """The mode dtype as torch's default, as generate.py sets BF16: the
        vendor's GEMMs and caches take it."""
        previous = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(previous)

    def check_length(self, length):
        if length > self.args.max_seq_len:
            raise ValueError(f"{length} tokens exceed the built model's {self.args.max_seq_len}")

    def prefill(self, model, ids, media, keep):
        """Transformer.forward (model.py:1244-1275) at start_pos 0 over every
        row of `ids` [1, L], without sampling or the DSpark hiddens, through
        reference_memory.prefill (row blocks, the stream on the host, query
        blocks in the indexer): the F32 logits of the last `keep` rows. Each
        row block is a tap chunk; each index source's selection reaches its
        `selection` probe once the layer ran."""
        modeling = self.modeling
        images = image_spans(ids[0].cpu(), media.get("images", ()), self.args, sys.modules[
            f"{modeling.__name__.rpartition('.')[0]}.image_processor"]) if media else []
        token_types = None
        if images:
            token_types = torch.full(ids.shape, -1, dtype=torch.long)
            for image in images:
                token_types[0, image.start:image.start + image.types.numel()] = image.types

        def on_layer(index, held):
            if "selected" in held:
                model.layers[index].attn.selection(held["selected"])
        logits, _ = memory().prefill(types.SimpleNamespace(model=modeling), model, ids,
                                     keep, [images] if images else None, token_types,
                                     rows_context=tap_writer.chunk, stream_device="cpu", on_layer=on_layer)
        return logits

    def forward(self, model, ids, keep, media):
        self.check_length(ids.shape[1])
        with self.default_dtype():
            return self.prefill(model, ids, media, keep)[0]

    def decode(self, model, ids, media, max_new_tokens, stop_ids):
        """Greedy continuation: the prompt's prefill, then one token per step
        through the vendor's caches (Transformer.forward at start_pos > 0),
        argmax with ties to the lower ID."""
        self.check_length(ids.shape[1] + max_new_tokens)
        with self.default_dtype():
            row = self.prefill(model, ids, media, 1)[0, -1]
            answer, rows = [], []
            position = ids.shape[1]
            while True:
                token = int(torch.argmax(row))
                answer.append(token)
                rows.append(row)
                if token in stop_ids or len(answer) == max_new_tokens:
                    return answer, torch.stack(rows)
                row = self.step(model, token, position)[0, -1]
                position += 1

    def step(self, model, token, position):
        """One decode step at `position` (model.py:1244-1275, start_pos > 0)."""
        with self.default_dtype():
            return self.step_logits(model, token, position)

    def step_logits(self, model, token, position):
        modeling = self.modeling
        ids = torch.tensor([[token]])
        engram_hashes = model.engram_hash(ids, position) if model.engram_hash is not None else None
        device = self.device_of("embed")
        with torch.device(device):
            h = model.embed(ids.to(device))
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        pre_mix = modeling.make_identity_pre_mix(h, model.hc_mult)
        for index, layer in enumerate(model.layers):
            device = self.device_of(f"layers.{index}")
            with torch.device(device):
                h, pre_mix = h.to(device), pre_mix.to(device)
                if layer.engram is not None:
                    h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :].to(device))
                share_to(modeling, device)
                h, pre_mix = layer(h, position, pre_mix, None)
        h = layer.hc_pre(h, pre_mix)
        with torch.device(self.device_of("norm")):
            h = model.norm(h.to(self.device_of("norm")))
        with torch.device(self.device_of("head")):
            return model.head(h.to(self.device_of("head")), full_logits=True)


SHARED = ("compress_kv", "index_k", "topk_idxs", "candidates")


def share_to(modeling, device):
    """Moves the vendor's SharedAttentionRuntime tensors (model.py:1169-1183)
    to `device` before a layer there runs: a layer that reuses the compressed
    KV, index keys, top-k or candidates of a source layer on another device
    reads a copy. A source republishes its own cache whenever it runs, so a
    copy never outlives the values it copied."""
    shared = modeling.shared_attn
    for name in SHARED:
        value = getattr(shared, name)
        if isinstance(value, torch.Tensor) and value.device != device:
            setattr(shared, name, value.to(device))


def selection(value):
    """Selected key positions as u32 rows; the vendor's -1 (none) becomes 2^32 - 1."""
    return value.to(torch.int64) & 0xFFFFFFFF


def taps(model):
    """Tap names (the runner's grammar) to the vendor values carrying them.
    Streams taps hold the four mHC streams (4 x 5120) of each position.
    `x.hc.{attn,ffn}.{pre,post,comb}` are a block's mHC coefficients,
    `x.engram.out` the streams after a layer's Engram, `x.latent` the
    normalized window latent before RoPE and quantization, `x.indexer.ids`
    an index source's selected compressed positions (a set; 2^32 - 1 where
    none), `attn.core` the grouped low-rank output wo_b reads."""
    names = {"embed.out": "embed", "final.x.hc.head": Hook("norm", "input"), "final.norm": "norm"}
    if model.vision is not None:
        names["enc.vision.out"] = Hook("aligner", rows=("image",))
        for block in range(len(model.vision.blocks)):
            names[f"enc.vision.B{block:02d}.out"] = Hook(f"vision.blocks.{block}", rows=("image",))
    for layer, block in enumerate(model.layers):
        path, scope = f"layers.{layer}", f"L{layer:02d}"
        attn = f"{path}.attn"
        names |= {
            f"{scope}.in.streams": Hook(path, "input"),
            f"{scope}.x.hc.attn.pre": Hook(f"{path}.hc_attn_probe", "output", 0),
            f"{scope}.x.hc.attn.post": Hook(f"{path}.hc_attn_probe", "output", 1),
            f"{scope}.x.hc.attn.comb": Hook(f"{path}.hc_attn_probe", "output", 2),
            f"{scope}.x.hc.attn": Hook(f"{path}.attn_norm", "input"),
            f"{scope}.attn.in": f"{path}.attn_norm",
            f"{scope}.attn.q": f"{attn}.wq_b",
            f"{scope}.x.latent": f"{attn}.kv_norm",
            f"{scope}.attn.core": Hook(f"{attn}.wo_b", "input"),
            f"{scope}.attn.out": attn,
            f"{scope}.x.hc.ffn.pre": Hook(f"{path}.hc_ffn_probe", "output", 0),
            f"{scope}.x.hc.ffn.post": Hook(f"{path}.hc_ffn_probe", "output", 1),
            f"{scope}.x.hc.ffn.comb": Hook(f"{path}.hc_ffn_probe", "output", 2),
            f"{scope}.x.hc.mlp": Hook(f"{path}.ffn_norm", "input"),
            f"{scope}.mlp.in": f"{path}.ffn_norm",
            f"{scope}.router.weights": Hook(f"{path}.ffn.gate", "output", 0),
            f"{scope}.router.ids": Hook(f"{path}.ffn.gate", "output", 1),
            f"{scope}.moe.shared": f"{path}.ffn.shared_experts",
            f"{scope}.mlp.out": f"{path}.ffn",
            f"{scope}.out": Hook(path, "output", 0),
        }
        if block.engram is not None:
            names[f"{scope}.x.engram.out"] = Hook(f"{path}.engram", "output")
        if block.attn.indexer is not None:
            names[f"{scope}.x.indexer.ids"] = Hook(f"{attn}.selection", "output", 0, selection)
    return names

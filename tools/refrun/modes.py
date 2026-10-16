"""Mode policies: every numeric choice of a reference run, as canonical JSON.

- f32: F32 parameters and activations; weights dequantized exactly; IEEE F32
  matmuls (no TF32, no reduced-precision reductions, NVIDIA_TF32_OVERRIDE=0).
- f64: the same in f64 (DGEMM); attention by the math path.
- bf16: the upstream code's own dtype; weights dequantized as its BF16 load
  would (F32 product rounded to BF16); BF16 GEMMs accumulate in F32.

Every mode runs one sequence per forward pass, deterministic algorithms, a
fixed SDPA backend, no torch.compile, no CUDA graphs and none of torch's
import-time DSL overrides of eager ops. The policy document's
sha256 goes into the cache key, so any change here is a new reference.

The environment variables must be set before CUDA initializes; one process
runs every mode, so they are the same for all modes.
"""

from contextlib import contextmanager
import hashlib
import os

from tools.model_sdk.diagnostic import canonical

SCHEMA = "chopinfer.refrun.mode.v1"
ENVIRONMENT = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NVIDIA_TF32_OVERRIDE": "0"}
DEQUANT = ("exact", "rounded", "hf-bf16")
MODES = {
    "f32": {"dtype": "float32", "log_softmax": "float32", "dequant": "exact"},
    "f64": {"dtype": "float64", "log_softmax": "float64", "dequant": "exact"},
    "bf16": {"dtype": "bfloat16", "log_softmax": "float32", "dequant": "hf-bf16"},
}
FLAGS = {
    "fp32_precision": "ieee",
    "float32_matmul_precision": "highest",
    "allow_fp16_reduced_precision_reduction": False,
    "allow_bf16_reduced_precision_reduction": False,
    "allow_fp16_accumulation": False,
    "math_sdp_fp16_bf16_reduction": False,
    "deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "torch_native_overrides": False,
}


def policy(name, dequant=None):
    """The canonical policy of mode `name`. `dequant` overrides the mode's
    weight rule; `exact` refuses any weight whose mode-dtype value is not
    its exact product, `rounded` rounds it once to the mode dtype (F32 or
    f64), `hf-bf16` rounds the F32 product to BF16 first."""
    if name not in MODES:
        raise ValueError(f"unknown mode {name}; expected one of {sorted(MODES)}")
    mode = dict(MODES[name])
    if dequant is not None:
        if dequant not in DEQUANT or (mode["dtype"] == "bfloat16" and dequant != "hf-bf16"):
            raise ValueError(f"mode {name} cannot dequantize by {dequant}")
        mode["dequant"] = dequant
    return {"schema": SCHEMA, "name": name, **mode, "flags": FLAGS, "environment": ENVIRONMENT,
            "sdpa_backend": "math", "batch_plan": "one-sequence-per-forward",
            "compile": False, "cuda_graphs": False, "scores": "float64"}


def policy_sha256(document):
    return hashlib.sha256(canonical(document)).hexdigest()


def torch_dtype(name):
    import torch
    return {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16}[name]


def prepare_environment(cuda=True):
    """Sets the process environment every mode shares. Refuses when CUDA is
    already initialized with other values, because cuBLAS read them then.

    A CPU-only process (`cuda=False`, the unit tests) must not initialize the
    CUDA driver at all, but transformers asks
    torch.cuda.is_current_stream_capturing() on every forward, which does.
    Nothing can capture a stream without CUDA, so that query answers False."""
    import torch
    stale = {k: v for k, v in ENVIRONMENT.items() if os.environ.get(k) != v}
    if stale and torch.cuda.is_initialized():
        raise RuntimeError(f"CUDA initialized before the reference environment: {sorted(stale)}")
    os.environ.update(ENVIRONMENT)
    disable_native_overrides()
    if not cuda:
        if torch.cuda.is_initialized():
            raise RuntimeError("a CPU-only reference process has initialized CUDA")
        torch.cuda.is_current_stream_capturing = lambda: False


def native_overrides():
    """torch's registry of eager-op overrides, or None if this torch has none.

    torch 2.14 registers DSL kernels for some CUDA ops at import (a Triton
    kernel for outer-product bmm, which HF rotary embeddings hit, and others),
    replacing the cuBLAS or ATen kernel the upstream code would otherwise run.
    """
    try:
        from torch._native import registry
    except ImportError:
        return None
    return registry


def disable_native_overrides():
    registry = native_overrides()
    if registry is None:
        return
    names = sorted({node.dsl_name for nodes in registry._graphs.values() for node in nodes})
    if names:
        registry.deregister_op_overrides(disable_dsl_names=names)
    if registry._aten_override_libs:
        raise RuntimeError(f"torch op overrides remain: {sorted(registry._aten_override_libs)}")


@contextmanager
def active(document):
    """Applies a policy's torch flags and SDPA backend for the enclosed work."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    stale = {k for k, v in document["environment"].items() if os.environ.get(k) != v}
    if stale:
        raise RuntimeError(f"reference environment not prepared: {sorted(stale)}")
    flags = document["flags"]
    # torch refuses a mix of the legacy allow_tf32 flags and fp32_precision, so
    # only the fp32_precision API is used.
    torch.backends.fp32_precision = flags["fp32_precision"]
    torch.backends.cuda.matmul.fp32_precision = flags["fp32_precision"]
    torch.backends.cudnn.conv.fp32_precision = flags["fp32_precision"]
    torch.backends.cudnn.rnn.fp32_precision = flags["fp32_precision"]
    torch.set_float32_matmul_precision(flags["float32_matmul_precision"])
    matmul = torch.backends.cuda.matmul
    matmul.allow_fp16_reduced_precision_reduction = flags["allow_fp16_reduced_precision_reduction"]
    matmul.allow_bf16_reduced_precision_reduction = flags["allow_bf16_reduced_precision_reduction"]
    matmul.allow_fp16_accumulation = flags["allow_fp16_accumulation"]
    # The math SDPA backend has its own reduced-precision switch.
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(flags["math_sdp_fp16_bf16_reduction"])
    torch.use_deterministic_algorithms(flags["deterministic_algorithms"])
    torch.backends.cudnn.deterministic = flags["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = flags["cudnn_benchmark"]
    if torch.get_float32_matmul_precision() != flags["float32_matmul_precision"] \
            or matmul.fp32_precision != flags["fp32_precision"] \
            or matmul.allow_fp16_reduced_precision_reduction != flags["allow_fp16_reduced_precision_reduction"] \
            or matmul.allow_bf16_reduced_precision_reduction != flags["allow_bf16_reduced_precision_reduction"] \
            or matmul.allow_fp16_accumulation != flags["allow_fp16_accumulation"] \
            or torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed() != flags["math_sdp_fp16_bf16_reduction"]:
        raise RuntimeError("torch did not accept the reference matmul and attention precision")
    registry = native_overrides()
    if not flags["torch_native_overrides"] and registry is not None and registry._aten_override_libs:
        raise RuntimeError("torch op overrides are registered; call prepare_environment first")
    backend = {"math": SDPBackend.MATH}[document["sdpa_backend"]]
    with sdpa_kernel([backend]), torch.inference_mode():
        yield

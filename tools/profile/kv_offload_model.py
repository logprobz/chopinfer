#!/usr/bin/env python3
"""Host memory and PCIe model for DRAM KV offload.

Computes offload capacity estimates for MiMo V2.6 Flash and
Qwen3.8 Flash at TP4 and TP2: the host bytes one swapped request needs, how
many swapped requests fit a host budget, the copy time of a swap-out or
swap-in against the time to recompute the same tokens, and the smallest
context for which a swap beats a recompute. Inputs are bundle contract
constants, the snapshot sizes of the prefix-cache contract and Q-PREFIX, and measured
prefill rates; each names its source. Every output is computed, not
measured; copy times use the rounded M2 idle-copy rates. M3 measures
all-reduce slowdown, not an isolated contended copy rate.

Usage: kv_offload_model.py
"""

import math

PAGE_TOKENS = 64
GIB = 1 << 30

# PCIe 5.0 x16: 32 GT/s per lane, 128b/130b coding, per direction.
PCIE5_X16_RAW = 32e9 * 16 * 128 / 130 / 8
# Rounded idle-copy rates per GPU and direction from kernel-lab copy timing, 256-page
# lists on four GPUs. These are nominal and conservative idle rates, not contention
# estimates. TP2 still extrapolates TP4 copy rates.
PCIE_IDLE_MEASURED = (57e9, 53e9)
# Preserve the calculator's existing import name.
PCIE_PLANNING = PCIE_IDLE_MEASURED

# Prefill cost per token on the whole TP4 server: MiMo 11,770 tok/s at 32K,
# Qwen about 11K tok/s (TP4 prefill measurements). Recompute replays at this rate at best. TP2 is
# estimated at half the TP4 rate (half the GPUs; unmeasured).
PREFILL_SECONDS_PER_TOKEN = {"mimo": 1 / 11_770, "qwen": 1 / 11_000}

# MiMo per rank at TP4 (models/mimo26/bundle/src/contract.rs): GA pages hold
# 9 layers x 1 head x 64 tokens x (192 + 128) x element bytes; the SWA ring
# slot is 39 layers x 2 heads x 128 positions x 320 x 2 B. prefix-cache contract's
# snapshot adds the MTP hidden window (128 x 8,192 B, replicated) or the
# DFlash feature window (1,023 rows x 20,480 / ranks x 2 B).
MIMO_GA_TOKEN_BYTES_BF16_TP4 = 5_760
MIMO_RING_SLOT_TP4 = 6_389_760
MIMO_MTP_WINDOW = 128 * 8_192
MIMO_DFLASH_ROWS = 1_023
MIMO_DFLASH_ROW_VALUES = 20_480

# Qwen per rank at TP4 (models/qwen38/bundle/src/contract.rs): 12 full
# layers x (K and V of one KV head, 12,288 B BF16 per token, plus 768 B of
# BF16 QSA block keys); FP8 (FP8 KV contract's branch) halves K and V only.
# Per-slot state: 36 linear layers x (15,360 conv + 786,432 delta) at TP4,
# a 184,448-byte PLE entry and one 51,456-byte QSA key-ring copy.
QWEN_KV_TOKEN_BYTES_BF16 = 12_288
QWEN_BLOCK_KEY_TOKEN_BYTES = 768
QWEN_LINEAR_ENTRY_TP4 = 36 * (15_360 + 786_432)
QWEN_PLE_ENTRY = 184_448
QWEN_RING_COPY = 12 * 16 * (128 * 2 + 12)

# GPU KV capacity at 0.9 and 32 sequences. Qwen FP8 is scaled from BF16
# by the page ratio (13,056 / 6,912), not measured.
GPU_TOKENS = {
    ("mimo", 4, "bf16"): 6_965_184,  # AR, no media, logits today
    ("mimo", 4, "fp8"): 13_930_368,
    ("mimo", 2, "bf16"): 805_504,  # AR, no media, logits cap
    ("mimo", 2, "fp8"): 1_611_072,
    ("qwen", 4, "bf16"): 3_875_968,  # AR, n-gram table in host memory
    ("qwen", 4, "fp8"): 3_875_968 * 13_056 // 6_912,
    ("qwen", 2, "bf16"): 1_411_008,
    ("qwen", 2, "fp8"): 1_411_008 * 13_056 // 6_912,
}


def page_bytes(model, ranks, element):
    """Serving-pool page bytes per rank."""
    if model == "mimo":
        per_token = MIMO_GA_TOKEN_BYTES_BF16_TP4 * 4 // ranks
        if element == "fp8":
            per_token //= 2
        return PAGE_TOKENS * per_token
    kv = QWEN_KV_TOKEN_BYTES_BF16 // (2 if element == "fp8" else 1)
    # One KV head per rank at TP2 and TP4 (the TP4 pair duplicates it).
    return PAGE_TOKENS * (kv + QWEN_BLOCK_KEY_TOKEN_BYTES)


def snapshot_bytes(model, ranks, drafter=None):
    """Bytes of one request's bundle-state snapshot per rank."""
    if model == "mimo":
        state = MIMO_RING_SLOT_TP4 * 4 // ranks
        if drafter == "mtp":
            state += MIMO_MTP_WINDOW
        if drafter == "dflash":
            state += MIMO_DFLASH_ROWS * (MIMO_DFLASH_ROW_VALUES // ranks) * 2
        return state
    # Qwen at TP2 holds twice the linear heads per rank (estimated: the PLE
    # entry and key ring are taken as unchanged).
    return QWEN_LINEAR_ENTRY_TP4 * 4 // ranks + QWEN_PLE_ENTRY + QWEN_RING_COPY


def snapshot_pages(model, ranks, element, drafter=None):
    return math.ceil(snapshot_bytes(model, ranks, drafter) / page_bytes(model, ranks, element))


def request_pages(model, ranks, element, tokens, drafter=None):
    """Pages one swapped request copies per rank: its committed KV pages plus
    its snapshot pages."""
    return math.ceil(tokens / PAGE_TOKENS) + snapshot_pages(model, ranks, element, drafter)


def rank_bytes(model, ranks, element, tokens, drafter=None):
    return request_pages(model, ranks, element, tokens, drafter) * page_bytes(model, ranks, element)


def host_bytes(model, ranks, element, tokens, drafter=None):
    """Host bytes for one swapped request over every rank."""
    return ranks * rank_bytes(model, ranks, element, tokens, drafter)


def copy_seconds(model, ranks, element, tokens, drafter, rate):
    """One direction; every rank copies on its own link at the same time."""
    return rank_bytes(model, ranks, element, tokens, drafter) / rate


def recompute_seconds(model, ranks, tokens):
    return tokens * PREFILL_SECONDS_PER_TOKEN[model] * 4 / ranks


def break_even_tokens(model, ranks, element, drafter, rate):
    """Smallest committed length for which a swap-out plus swap-in, taken as
    fully serialized with compute, costs less than a cold recompute."""
    tokens = 1
    while 2 * copy_seconds(model, ranks, element, tokens, drafter, rate) >= recompute_seconds(model, ranks, tokens):
        tokens += 1
        if tokens > 1 << 20:
            raise ValueError("no break-even")
    return tokens


CONFIGURATIONS = (
    ("MiMo TP4 BF16, DFlash", "mimo", 4, "bf16", "dflash"),
    ("MiMo TP4 BF16, AR", "mimo", 4, "bf16", None),
    ("MiMo TP4 FP8, DFlash", "mimo", 4, "fp8", "dflash"),
    ("MiMo TP2 BF16, DFlash", "mimo", 2, "bf16", "dflash"),
    ("MiMo TP2 FP8, DFlash", "mimo", 2, "fp8", "dflash"),
    ("Qwen TP4 BF16", "qwen", 4, "bf16", None),
    ("Qwen TP4 FP8", "qwen", 4, "fp8", None),
    ("Qwen TP2 BF16", "qwen", 2, "bf16", None),
    ("Qwen TP2 FP8", "qwen", 2, "fp8", None),
)
CONTEXTS = (2_048, 8_192, 32_768, 131_072)
BUDGETS_GIB = (32, 64, 128)


def main():
    print(f"PCIe 5.0 x16 raw rate per direction: {PCIE5_X16_RAW / 1e9:.1f} GB/s")
    print()
    print("| Configuration | Page bytes per rank | Snapshot pages | Host bytes per token | "
          "Snapshot bytes, all ranks | GPU pool, GB |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, model, ranks, element, drafter in CONFIGURATIONS:
        page = page_bytes(model, ranks, element)
        pages = snapshot_pages(model, ranks, element, drafter)
        gpu = GPU_TOKENS[(model, ranks, element)] * page // PAGE_TOKENS * ranks
        print(f"| {name} | {page:,} | {pages} | {ranks * page // PAGE_TOKENS:,} | "
              f"{ranks * pages * page / 1e6:,.1f} MB | {gpu / 1e9:,.0f} |")
    print()
    print("Host MB per swapped request, and swapped requests per budget:")
    print()
    header = " | ".join(f"{c // 1024}K" for c in CONTEXTS)
    budgets = " | ".join(f"{c // 1024}K in {b} GiB" for b in BUDGETS_GIB for c in (8_192, 32_768))
    print(f"| Configuration | {header} | {budgets} |")
    print("|---|" + "---:|" * (len(CONTEXTS) + 2 * len(BUDGETS_GIB)))
    for name, model, ranks, element, drafter in CONFIGURATIONS:
        sizes = [host_bytes(model, ranks, element, c, drafter) for c in CONTEXTS]
        fits = [
            (b * GIB) // host_bytes(model, ranks, element, c, drafter)
            for b in BUDGETS_GIB
            for c in (8_192, 32_768)
        ]
        print(f"| {name} | " + " | ".join(f"{s / 1e6:,.0f}" for s in sizes) + " | "
              + " | ".join(f"{f:,}" for f in fits) + " |")
    print()
    print("One-way idle copy ms per request (57 / 53 GB/s per rank) against cold recompute ms:")
    print()
    print("| Configuration | " + " | ".join(f"{c // 1024}K copy | {c // 1024}K recompute" for c in CONTEXTS)
          + " | Break-even tokens (57 / 53) |")
    print("|---|" + "---:|" * (2 * len(CONTEXTS) + 1))
    for name, model, ranks, element, drafter in CONFIGURATIONS:
        cells = []
        for c in CONTEXTS:
            fast, slow = (copy_seconds(model, ranks, element, c, drafter, r) * 1e3 for r in PCIE_PLANNING)
            cells.append(f"{fast:.1f}-{slow:.1f} | {recompute_seconds(model, ranks, c) * 1e3:,.0f}")
        even = [break_even_tokens(model, ranks, element, drafter, r) for r in PCIE_PLANNING]
        print(f"| {name} | " + " | ".join(cells) + f" | {even[0]} / {even[1]} |")


if __name__ == "__main__":
    main()

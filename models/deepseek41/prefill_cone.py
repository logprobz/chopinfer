"""CPU dependency oracle for Flash prefill, independent of device lowering.

Run with --chunk 2048 for configured chunks or --chunk 8192 for a lone
request. JSON on stdout is an analysis artifact; save it outside the tree.
Intervals are half-open absolute token positions. This counts dataflow,
not GPU time. Sources and the snapshot-byte invariant are documented in
prefill-dependencies.md.
"""

import argparse
import json
from pathlib import Path


def union(*sets):
    result = []
    for start, end in sorted(interval for ranges in sets for interval in ranges):
        if start >= end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def count(intervals):
    return sum(end - start for start, end in intervals)


def positions(intervals):
    return (position for start, end in intervals for position in range(start, end))


def chunks(tokens, captures, limit):
    if tokens <= 0 or limit <= 0:
        raise ValueError("tokens and chunk limit must be positive")
    if sorted(set(captures)) != captures or any(not 0 < n <= tokens for n in captures):
        raise ValueError("captures must be increasing token counts within the prompt")
    end = 0
    result = []
    for boundary in sorted(set(captures + [tokens])):
        while end < boundary:
            end = min(end + limit, boundary)
            result.append(end)
    return result


def history_writers(ends, captures, window=128, headroom=6):
    """The physical raw-feature cells, including stale cells, at each capture.

    Mirrors the transfer *contract*, not its loop: a large commit retains
    its final window; a short commit retains every row. Entries never
    written by this request remain None and must keep their initial bytes.
    """
    history = [None] * (window + headroom)
    result = {}
    first = 0
    for end in ends:
        if end <= first:
            raise ValueError("chunk ends must increase")
        start = end - window if end - first > len(history) else first
        for position in range(start, end):
            history[position % len(history)] = position
        if end in captures:
            result[end] = tuple(history)
        first = end
    if set(result) != set(captures):
        raise ValueError("every capture must end a chunk")
    return result


def cone(tokens, ends, captures, dspark, logits=None, geometry=None):
    if geometry is None:
        geometry = json.loads(Path(__file__).with_name("geometries").joinpath("flash.json").read_text())["text_config"]
    if not ends or ends[-1] != tokens:
        raise ValueError("chunks must cover the prompt")
    logits = [tokens - 1] if logits is None else logits
    if any(not 0 <= p < tokens for p in logits):
        raise ValueError("logits rows must lie inside the prompt")
    window = geometry["sliding_window"]
    layers = geometry["num_hidden_layers"]
    source = max(geometry["kv_source_layer_ids"])
    if any(layer > source for layer in geometry["engram_layer_ids"]):
        raise ValueError("upper Engram dependencies need a separate rule")
    writers = history_writers(ends, captures, window)
    captured_keys = [(max(0, end - window - 8), end) for end in captures]
    # Future target queries and DSpark read the final window even without
    # a prompt-end capture, for example after an unaligned final chunk.
    keys = union(captured_keys, [(max(0, tokens - window), tokens)])
    features = union(
        [(p, p + 1) for cells in writers.values() for p in cells if p is not None],
        [(max(0, tokens - window), tokens)],
    ) if dspark else []
    needed = union([(p, p + 1) for p in logits])
    result = []
    for layer in range(layers - 1, source - 1, -1):
        queries = needed
        kv = union([(max(0, start - window + 1), end) for start, end in queries], keys)
        taps = features if layer in geometry["dspark_target_layer_ids"] else []
        needed = union(kv, taps)
        result.append({"layer": layer, "outputs": queries, "window_kv": kv,
                       "features": taps, "inputs": needed})
    result.reverse()
    upper_rows = (layers - source) * tokens
    output_rows = sum(count(row["outputs"]) for row in result)
    kv_rows = sum(count(row["window_kv"]) for row in result)
    return {"tokens": tokens, "chunk_ends": ends, "captures": captures, "dspark": dspark,
            "last_source": source, "full_input_layers": list(range(source)),
            "full_source_input_rows": tokens, "layers": result,
            "feature_roots": features, "snapshot_writers": writers,
            "upper_output_rows": output_rows, "upper_window_kv_rows": kv_rows,
            "upper_output_skip_fraction": 1 - output_rows / upper_rows,
            "upper_window_kv_skip_fraction": 1 - kv_rows / upper_rows,
            "equal_layer_row_saving": (upper_rows - output_rows) / (layers * tokens)}


def cold(tokens, chunk, dspark):
    # Scheduler capture_points uses the configured 2048-row chunk even
    # when a lone request expands its actual chunks to 8192.
    repeat = (tokens - 1) // 2048 * 2048
    captures = sorted({p for p in [repeat, tokens // 64 * 64] if p > 0})
    return cone(tokens, chunks(tokens, captures, chunk), captures, dspark)


def all_choices(tokens, dspark=True):
    """Envelope over all granule-aligned capture choices, not a forecast.

    Each possible boundary roots the copied target rings. Union commutes
    with the backward closure. Listing them together does not assert that
    a single default request captures at every granule. DSpark writer
    choices cannot enlarge this envelope: it already includes every input
    row at every upper layer.
    """
    ends = list(range(64, tokens, 64)) + [tokens]
    result = cone(tokens, ends, ends, dspark)
    result["contract"] = "all-boundary-choices-envelope"
    return result


def guaranteed(tokens, chunk, dspark, alignment=64):
    """Complete physical state after every grid chunk, as the opt-in executor.

    Previous chunks are exact external inputs. A projection touching the
    last window retains its original dispatch shape, including a partial
    overlap before the final window at unaligned prompt ends.
    """
    if tokens <= 0 or chunk <= 0 or alignment <= 0:
        raise ValueError("positive prompt, chunk and alignment required")
    plans = []
    for first in range(0, tokens, chunk):
        end = min(first + chunk, tokens)
        n = end - first
        align = lambda ranges: union(*[[(a // alignment * alignment,
            min(n, (b + alignment - 1) // alignment * alignment))] for a, b in ranges])
        captured = [(max(0, n - 136), n)]
        feature_start = 0 if n <= 134 else max(0, (end - 128) // 128 * 128 - first)
        features = [(feature_start, n)] if dspark else []
        needed = [(n - 1, n)] if end == tokens else []
        layers = []
        for layer in range(39, 19, -1):
            outputs = align(needed)
            keys = align(union([(max(0, a - 127), b) for a, b in outputs], captured))
            taps = features if layer in [37, 38, 39] else []
            needed = union(keys, taps)
            layers.append({"layer": layer, "outputs": outputs, "keys": keys, "features": taps})
        plans.append({"start": first, "end": end, "layers": list(reversed(layers))})
    outputs = sum(count(layer["outputs"]) for plan in plans for layer in plan["layers"])
    keys = sum(count(layer["keys"]) for plan in plans for layer in plan["layers"])
    return {"tokens": tokens, "chunk": chunk, "dspark": dspark, "alignment": alignment,
        "upper_output_rows": outputs, "upper_window_kv_rows": keys,
        "upper_output_skip_fraction": 1 - outputs / (20 * tokens),
        "upper_window_kv_skip_fraction": 1 - keys / (20 * tokens),
        "equal_layer_row_saving": (20 * tokens - outputs) / (40 * tokens), "chunks": plans}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", type=int, choices=[2048, 8192], default=2048)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 32768, 131072])
    parser.add_argument("--contract", choices=["fixed", "all-choices", "guaranteed"], default="fixed")
    parser.add_argument("--alignment", type=int, default=64)
    args = parser.parse_args()
    plans = [all_choices(n) for n in args.tokens] if args.contract == "all-choices" else [
        cold(n, args.chunk, dspark) for n in args.tokens for dspark in [False, True]
    ]
    if args.contract == "guaranteed":
        plans = [guaranteed(n, args.chunk, dspark, args.alignment) for n in args.tokens for dspark in [False, True]]
    print(json.dumps(plans, indent=2))


if __name__ == "__main__":
    main()

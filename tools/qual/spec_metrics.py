"""Read cumulative speculation counters from Chopinfer /metrics."""

from decimal import Decimal, InvalidOperation
import json
import os
import re
import urllib.request


FIELDS = {
    "chopinfer": {
        "verify_items": "chop_spec_verify_items_total",
        "draft_tokens": "chop_spec_draft_tokens_total",
        "accepted_tokens": "chop_spec_accepted_tokens_total",
        "emitted_tokens": "chop_spec_emitted_tokens_total",
        "rows": "chop_spec_rows_total",
        "drafted_positions": "chop_spec_drafted_position_total",
        "accepted_positions": "chop_spec_accepted_position_total",
    },
}
SCALARS = ("verify_items", "draft_tokens", "accepted_tokens", "emitted_tokens", "rows")
POSITIONS = ("drafted_positions", "accepted_positions")
SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$")
LABEL = re.compile(r'(?:^|,)([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


def metrics_text(client):
    """Fetch one raw scrape; callers choose whether a missing endpoint is fatal."""
    headers = {}
    if key := os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(client.url + "/metrics", headers=headers)
    with urllib.request.urlopen(request, timeout=min(client.timeout, 10)) as response:
        return response.read().decode("utf-8")


def _labels(raw):
    if raw is None:
        return {}
    labels = {name: json.loads('"' + value + '"') for name, value in LABEL.findall(raw)}
    if not labels or len(LABEL.sub("", raw).replace(",", "")):
        raise ValueError(f"invalid Prometheus labels: {raw}")
    return labels


def _counter(raw):
    try:
        value = Decimal(raw)
    except InvalidOperation as error:
        raise ValueError(f"invalid counter: {raw}") from error
    if not value.is_finite() or value < 0 or value != value.to_integral_value():
        raise ValueError(f"invalid counter: {raw}")
    return int(value)


def _step_rows_snapshot(text, model):
    """Exact completed decode/verify step bins; no prefill or mixed batches."""
    names = {"chop_step_rows_total", "chop_step_rows_dropped_total"}
    bins, seen = {}, set()
    dropped = None
    for line in text.splitlines():
        match = SAMPLE.match(line)
        if match is None or match[1] not in names:
            continue
        name, raw_labels, raw_value = match.groups()
        labels = _labels(raw_labels)
        if model is not None and "model_name" in labels and labels["model_name"] != model:
            continue
        series = (name, tuple(sorted(labels.items())))
        if series in seen:
            raise ValueError(f"duplicate metric series: {name}")
        seen.add(series)
        value = _counter(raw_value)
        if name == "chop_step_rows_dropped_total":
            dropped = (dropped or 0) + value
            continue
        if not {"decode_rows", "verify_rows", "row_capacity"} <= labels.keys():
            raise ValueError("missing step-row geometry labels")
        decode, verify = (_counter(labels[key]) for key in ("decode_rows", "verify_rows"))
        capacity = None if labels["row_capacity"] == "unknown" else _counter(labels["row_capacity"])
        if decode + verify == 0 or (capacity is not None and capacity < decode + verify):
            raise ValueError("invalid step-row geometry")
        key = (decode, verify, capacity)
        bins[key] = bins.get(key, 0) + value
    if bins and dropped is None:
        raise ValueError("incomplete step-row histogram counters")
    return {"available": dropped is not None, "dropped": dropped or 0,
            "bins": _step_row_bins(bins)}


def _step_row_bins(bins):
    return [{"decode_rows": decode, "verify_rows": verify, "row_capacity": capacity,
             "steps": count}
            for (decode, verify, capacity), count in sorted(
                bins.items(), key=lambda entry: (*entry[0][:2], entry[0][2] or 0))]


def _step_rows_difference(before, after):
    if not after.get("available"):
        if before.get("available"):
            raise ValueError("step-row histogram disappeared between scrapes")
        return {"available": False}
    dropped = after["dropped"] - before.get("dropped", 0)
    if dropped < 0:
        raise ValueError("step-row dropped counter reset between scrapes")
    if dropped:
        raise ValueError("step-row histogram dropped observations during the cohort")

    def counts(snapshot):
        return {(row["decode_rows"], row["verify_rows"], row["row_capacity"]): row["steps"]
                for row in snapshot.get("bins", [])}

    start, end = counts(before), counts(after)
    bins = {}
    for key in start.keys() | end.keys():
        count = end.get(key, 0) - start.get(key, 0)
        if count < 0:
            raise ValueError("step-row counter reset or disappeared between scrapes")
        if count:
            bins[key] = count
    return {"available": True, "dropped": dropped, "bins": _step_row_bins(bins),
            "steps": sum(bins.values()),
            "verify_steps": sum(count for (_, verify, _), count in bins.items() if verify),
            "unknown_capacity_steps": sum(count for (_, _, cap), count in bins.items() if cap is None)}


def snapshot(text, arm, model=None):
    """Sum engine series, retaining position counts with one-based indices."""
    names = FIELDS[arm]
    reverse = {name: field for field, name in names.items()}
    output = {field: 0 for field in SCALARS if field in names}
    output.update({field: {} for field in POSITIONS})
    seen = set()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = SAMPLE.match(line)
        if match is None or match[1] not in reverse:
            continue
        name, raw_labels, raw_value = match.groups()
        labels = _labels(raw_labels)
        if model is not None and "model_name" in labels and labels["model_name"] != model:
            continue
        field = reverse[name]
        series = (name, tuple(sorted(labels.items())))
        if series in seen:
            raise ValueError(f"duplicate metric series: {name}")
        seen.add(series)
        value = _counter(raw_value)
        if field in POSITIONS:
            if "position" not in labels:
                raise ValueError(f"missing position label: {name}")
            position = int(labels["position"])
            if position < 1 or position > 16:
                raise ValueError(f"position outside 1..16: {position}")
            output[field][position] = output[field].get(position, 0) + value
        else:
            output[field] += value
    required = {names[field] for field in ("verify_items", "draft_tokens", "accepted_tokens")}
    present = {name for name, _ in seen}
    if present and not required.issubset(present):
        raise ValueError(f"incomplete speculation counters: {sorted(required - present)}")
    output["available"] = bool(present)
    if arm == "chopinfer":
        output["step_rows"] = _step_rows_snapshot(text, model)
    return output


def difference(before, after):
    """Subtract complete scrapes and reject resets or missing series."""
    geometry = {}
    if "step_rows" in before or "step_rows" in after:
        geometry["step_rows"] = _step_rows_difference(
            before.get("step_rows", {}), after.get("step_rows", {}))
    if not after["available"]:
        if before["available"]:
            raise ValueError("speculation counters disappeared between scrapes")
        return {"available": False, **geometry}
    output = {"available": True, **geometry}
    for field in SCALARS:
        if field not in after:
            continue
        start = before.get(field, 0)
        end = after[field]
        if end < start:
            raise ValueError(f"{field} counter reset between scrapes")
        output[field] = end - start
    for field in POSITIONS:
        output[field] = {}
        for position in sorted(set(before[field]) | set(after[field])):
            start = before[field].get(position, 0)
            end = after[field].get(position, 0)
            if end < start:
                raise ValueError(f"{field} position {position} reset between scrapes")
            output[field][position] = end - start
    drafted = output["draft_tokens"]
    accepted = output["accepted_tokens"]
    steps = output["verify_items"]
    if accepted > drafted or sum(output["accepted_positions"].values()) != accepted \
            or sum(output["drafted_positions"].values()) != drafted:
        raise ValueError("speculation counters disagree with per-position counts")
    output["acceptance_rate"] = accepted / drafted if drafted else None
    output["mean_acceptance_length"] = 1 + accepted / steps if steps else None
    output["tokens_per_step"] = output["mean_acceptance_length"]
    if "emitted_tokens" in output:
        output["emitted_tokens_per_step"] = output["emitted_tokens"] / steps if steps else None
    output["per_position"] = [
        {"position": position, "drafted": count,
         "accepted": output["accepted_positions"].get(position, 0),
         "rate": output["accepted_positions"].get(position, 0) / count if count else None}
        for position, count in sorted(output["drafted_positions"].items())
    ]
    return output


def within_output(counters, requests, completion_tokens):
    """Acceptance counted only up to the requests' output limits.

    The first output token comes from prefill and each verify item emits its
    accepted drafts plus one token. Compare those totals with emitted output
    to detect counters that include drafts past the request limits.
    """
    steps = counters.get("verify_items")
    if not counters.get("available") or not steps:
        return {"accepted_past_output": None, "acceptance_length_within_output": None}
    past = max(0, requests + steps + counters["accepted_tokens"] - completion_tokens)
    return {"accepted_past_output": past,
            "acceptance_length_within_output": 1 + (counters["accepted_tokens"] - past) / steps}

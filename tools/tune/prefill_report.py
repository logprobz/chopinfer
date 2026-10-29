"""Acceptance cells for full-concurrency prefill steps without decode work."""
import math
import statistics


def decision_cells(model, selected, concurrency, change, paired):
    declared = model.get('prefill_rows', {})
    if not isinstance(declared, dict) or not set(declared) <= {str(c) for c in model['concurrencies']}:
        raise ValueError('invalid prefill target concurrencies')
    widths = declared.get(str(concurrency), [])
    if (not isinstance(widths, list) or any(type(n) is not int or n <= 0 for n in widths)
            or len(set(widths)) != len(widths)):
        raise ValueError('prefill rows must be unique positive integers')
    if not widths:
        return {}
    if len(selected) != 4:
        raise ValueError('prefill cells require all four ABBA visits')
    synthetic = [bool(row.get('synthetic')) for row in selected]
    if any(synthetic) != all(synthetic):
        raise ValueError('mixed synthetic and real prefill evidence')
    minimum, share = model.get('min_cell_steps', 20), model.get('min_cell_time_share')
    if type(minimum) is not int or minimum < 1 or (share is not None and not 0 < share <= 1):
        raise ValueError('invalid minimum prefill evidence')
    out = {}
    for width in widths:
        values = []
        for row in selected:
            samples = []
            for step in row.get('prefill_steps', []):
                if (step.get('prefill_items') != concurrency or step.get('prefill_tokens') != width
                        or step.get('decode_rows', 0) or step.get('verify_items', 0)):
                    continue
                micros = step.get('execution_us')
                if type(micros) not in (int, float) or not math.isfinite(micros) or micros <= 0:
                    raise ValueError('invalid full prefill step time')
                samples.append(micros / 1000)
            values.append(samples)
        counts = [len(v) for v in values]
        shares = [sum(v) / (row['request_seconds'] * 1000) for v, row in zip(values, selected)]
        supported = [n > 0 if all(synthetic) else n >= minimum or (share is not None and fraction >= share)
                     for n, fraction in zip(counts, shares)]
        if not all(supported):
            raise ValueError(f'prefill c{concurrency} r{width} is missing or insufficiently sampled')
        medians = [statistics.median(v) for v in values]
        name = f'prefill:c{concurrency}:r{width}'
        weight = model.get('cell_weights', {}).get(name, 1)
        cell = dict(steps=counts, time_share=shares, execution_ms_median=medians,
                    supported=supported, nominal_weight=weight, weight=weight,
                    decision=weight > 0, mechanics_only=all(synthetic),
                    **paired([1 / n for n in medians], change))
        if all(synthetic):
            cell['accepted'] = False
        out[name] = cell
    return out

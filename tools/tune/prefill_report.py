"""Acceptance cells for full-concurrency prefill steps without decode work."""
import math
import statistics


def decision_cells(model, selected, concurrency, change, paired):
    """Raw execution times and, where every step reports its Engram wait,
    adjusted ones (execution less the wait); `step_time` picks the one that
    decides."""
    primary = (model.get('step_time') or {}).get('primary', 'raw')
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
        values, adjusted = [], []
        for row in selected:
            samples, less_waits = [], []
            for step in row.get('prefill_steps', []):
                if (step.get('prefill_items') != concurrency or step.get('prefill_tokens') != width
                        or step.get('decode_rows', 0) or step.get('verify_items', 0)):
                    continue
                micros = step.get('execution_us')
                if type(micros) not in (int, float) or not math.isfinite(micros) or micros <= 0:
                    raise ValueError('invalid full prefill step time')
                samples.append(micros / 1000)
                wait = step.get('engram_wait_us')
                if less_waits is not None:
                    less_waits = None if wait is None or wait >= micros else less_waits + [(micros - wait) / 1000]
            values.append(samples)
            adjusted.append(less_waits)
        if any(v is None for v in adjusted):
            adjusted = None
        counts = [len(v) for v in values]
        shares = [sum(v) / (row['request_seconds'] * 1000) for v, row in zip(values, selected)]
        supported = [n > 0 if all(synthetic) else n >= minimum or (share is not None and fraction >= share)
                     for n, fraction in zip(counts, shares)]
        if not all(supported):
            raise ValueError(f'prefill c{concurrency} r{width} is missing or insufficiently sampled')
        name = f'prefill:c{concurrency}:r{width}'
        if primary == 'adjusted' and adjusted is None:
            raise ValueError(f'{name}: adjusted step time needs a positive Engram-adjusted time for every step')
        medians = [statistics.median(v) for v in values]
        less = None if adjusted is None else [statistics.median(v) for v in adjusted]
        weight = model.get('cell_weights', {}).get(name, 1)
        decided, other = (medians, less) if primary == 'raw' else (less, medians)
        cell = dict(steps=counts, time_share=shares, execution_ms_median=medians, adjusted_ms_median=less,
                    metric=primary, supported=supported, nominal_weight=weight, weight=weight,
                    decision=weight > 0, mechanics_only=all(synthetic),
                    **paired([1 / n for n in decided], change))
        if other is not None:
            result = paired([1 / n for n in other], change)
            other = {key: result[key] for key in ('d1', 'd2', 'mean', 'spread', 'verdict')}
        cell['adjusted' if primary == 'raw' else 'raw'] = other
        if all(synthetic):
            cell['accepted'] = False
        out[name] = cell
    return out

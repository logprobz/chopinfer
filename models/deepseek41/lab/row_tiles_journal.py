"""Prove exact decode operands, selected grids and unchanged C1 launches."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from verify_hc_journal import align_buffers, align_streams, config, launches, require

FINISH = 'chop_deepseek41_hc_round_expand_mix_finish_v1'
NORM = 'chop_norm_quantize_deepseek41_input_v1'
CACHED = 'chop_norm_quantize_deepseek41_input_cached32_v1'
OLD = 'chop_deepseek41_hc_round_expand_mix_partials_v1'
CHAIN = 'chop_deepseek41_hc_round_chain_w4_v1'
TILED = 'chop_deepseek41_hc_round_rows_r2_v1'
TAIL = 'chop_deepseek41_hc_finish_norm_w32_v1'


def project(entries, rows):
    if rows == 1:
        return entries, 0, 0
    require(rows in (8, 32), 'unqualified row bucket')
    result, pairs, widened = [], 0, 0
    at = 0
    while at < len(entries):
        p = entries[at]
        if p['symbol'] != (CHAIN if rows == 8 else OLD):
            result.append(p)
            at += 1
            continue
        require(at + 2 < len(entries), 'incomplete boundary')
        f, n = entries[at+1:at+3]
        require(f['symbol'] == FINISH and n['symbol'] == (CACHED if rows == 8 else NORM),
                'producer, finish and norm must be adjacent')
        require(p['config'] == (config(160, rows, 128) if rows == 8 else config(20, rows, 256)),
                'reference producer geometry changed')
        require(f['config'] == config(10, rows, 256), 'reference finish geometry changed')
        require(n['config'] == (config(1, rows, 1024) if rows == 8 else config(rows, 1, 256)),
                'reference norm geometry changed')
        pa, fa, na = p['args'], f['args'], n['args']
        require(len(pa) == 9 and len(fa) == len(na) == 10, 'boundary ABI changed')
        require(pa[5] == fa[0] and pa[7] == fa[8] and pa[8] == fa[9] == na[6],
                'producer workspace, expanded or live view changed')
        require(fa[4] == na[0], 'norm input changed')
        result.append(dict(p, symbol=CHAIN if rows == 8 else TILED,
                           config=config(160, rows, 128) if rows == 8 else config(160, rows//2, 256)))
        result.append(dict(symbol=TAIL, config=config(1, rows, 1024),
                           args=fa+na[1:6]+na[7:]))
        pairs += 1
        widened += na[3] != 'NullDevicePointer'
        at += 3
    return result, pairs, widened


def audit(before, after, rows, ranks):
    a, b = launches(before), launches(align_streams(before, after))
    require(a.keys() == b.keys(), 'stream set changed')
    pairs = widened = 0
    for stream, old in a.items():
        expected, count, wide = project(old, rows)
        require(expected == b[stream], f'launch operands or ordering changed on stream {stream}')
        pairs += count
        widened += wide
    require(pairs == (0 if rows == 1 else 77*ranks), 'wrong eligible boundary count')
    require(widened == (0 if rows == 1 else 47*ranks), 'wrong widening count')
    return dict(rows=rows, ranks=ranks, pairs_per_rank=pairs//ranks,
                before_per_rank=sum(map(len, a.values()))//ranks,
                after_per_rank=sum(map(len, b.values()))//ranks,
                launches_saved_per_rank=pairs//ranks, operands_and_dependencies_equal=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--before', type=Path, required=True)
    p.add_argument('--after', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    records = []
    for before in sorted(args.before.glob('decode.*.journal')):
        ranks, rows = map(int, re.search(r'\.tp([124])\.c(1|8|32)\.', before.name).groups())
        after = args.after / before.name
        read_meta = lambda path: json.loads(path.with_name(path.name+'.handles.json').read_text())
        original = before.read_text()
        actual = align_buffers(read_meta(before), read_meta(after), after.read_text())
        result = audit(original, actual, rows, ranks)
        negatives = 0
        line = next(l for l in actual.splitlines() if l.startswith('launch ') and
                    (TAIL in l if rows != 1 else CACHED in l))
        for field in ('grid_x', 'block_x', 'offset', 'length'):
            changed, count = re.subn(r'('+field+r': )(\d+)', lambda m: m[1]+str(int(m[2])+1), line, count=1)
            require(count == 1, 'negative control did not change a field')
            try: audit(original, actual.replace(line, changed, 1), rows, ranks)
            except ValueError: negatives += 1
            else: raise ValueError('changed launch accepted')
        result.update(negative_controls=negatives, journal=before.name,
                      sha256={tag:hashlib.sha256(path.read_bytes()).hexdigest()
                              for tag,path in [('before',before),('after',after)]})
        records.append(result)
    require(len(records) == 18, 'missing rank, row or KV layout')
    args.output.write_text(json.dumps(records, indent=2)+'\n')
    print('Eighteen decode journals and 72 negative controls passed; C1 launches unchanged.')


if __name__ == '__main__':
    main()

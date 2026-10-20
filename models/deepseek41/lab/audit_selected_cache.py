#!/usr/bin/env python3
"""Audit selected-cache resources and source pairs before updating journal pins."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from audit_decode_attention import digest, read

CUBIN = 'latent_decode_cache_deepseek41_sm120'
ARENA = 32 * 640 * (512 * 2 + 4)
RAW = re.compile(r'raw: (\d+)')
VIEW = re.compile(r'DeviceBuffer\(DeviceBufferSpanV1 \{ buffer: BundleDeviceBufferHandleV1 \{ raw: (\d+) \}, offset: (\d+), length: (\d+) \}\)')
PINS = {
    'tp1.native': (16260, 'dfa1ed3db03cd3e0'),
    'tp1.bf16': (16205, '923512cd9c868ac3'),
    'tp2.native': (32947, '364579e8cf5ee736'),
    'tp2.bf16': (32837, '47f778944d5fb31a'),
    'tp4.native': (70247, '9f9e8078b4ae57ab'),
    'tp4.bf16': (68991, 'b30cfe7f85e4e8cd'),
    'tp2.media': (20907, 'd28729a73500a819'),
    'tp4.media': (46079, '28012023832f4c28'),
}


def field(line, name):
    return int(re.search(r'\b' + name + r': (\d+)', line)[1])


def shape(line, grid, shared):
    expected = dict(flags=1, grid_x=grid[0], grid_y=grid[1], grid_z=grid[2],
                    block_x=128, block_y=1, block_z=1, shared_memory_bytes=shared)
    assert all(field(line, k) == v for k, v in expected.items()), ('launch shape', line)
    assert 'cluster: LaunchClusterV1 { x: 1, y: 1, z: 1 }' in line


def source_pair(producer, consumer, meta):
    name = consumer.split()[2]
    assert name.endswith('_cached_profile0_v1'), name
    stem = name.removesuffix('_cached_profile0_v1')
    rows = field(consumer, 'grid_x')
    verify = '_verify' in stem
    assert rows in ((4, 6, 32) if verify else (1, 8)), ('unadmitted row', rows, stem)
    keys = 128 if '_window_' in stem else 640
    shape(consumer, (rows, 16 if rows <= 4 else 2, 8), 73984)
    shape(producer, (keys // 4, rows, 1), 0)
    assert producer.split()[1] == consumer.split()[1], 'producer stream differs'
    assert producer.split()[2] == stem + '_cache_decode_control0_v1'
    before = producer[producer.index(' [') + 2:]
    after = consumer[consumer.index(' [') + 2:]
    assert after == before[:-1] + ', NullDevicePointer]', 'producer/source argument mismatch'
    views = list(VIEW.finditer(after))
    assert len(views) >= 2
    decoded, valid = views[-2:]
    d, v = [tuple(map(int, view.groups())) for view in (decoded, valid)]
    prepared_rows = 32 if verify else 8
    assert d[0] == v[0] and d[1:] == (0, prepared_rows * keys * 1024)
    assert v[1:] == (d[2], prepared_rows * keys * 4)
    assert meta['ranks'][str(d[0] - (1 << 32))][1] == ARENA
    assert d[0] not in [int(m[1]) for m in RAW.finditer(after[:decoded.start()])], 'cache aliases native operand'
    # Cache arguments are appended after every original scalar and span.
    args = after[:decoded.start()].removesuffix(', ')
    staging = (verify and rows == 32) or (not verify and rows == 8)
    suffix = '_stage64_h0_b0_profile0_v1' if staging else '_heads_v1'
    if staging:
        args += ', NullDevicePointer'
    line = consumer[:consumer.index(' [') + 2] + args + ']'
    line = line.replace(name, stem + suffix, 1).replace(
        'shared_memory_bytes: 73984', f'shared_memory_bytes: {75008 if staging else 83200}', 1)
    return line, name


def canonical(lines, meta):
    maps = {kind: {} for kind in ('stream', 'event', 'graph', 'function', 'buffer')}
    for category, keyword, column in [('stream', 'stream', 1), ('event', 'event', 1),
                                      ('graph', 'captured', 2), ('function', 'resolve', 1)]:
        for line in lines:
            words = line.split()
            if words[0] == keyword:
                raw = int(words[column])
                assert raw not in maps[category], ('duplicate resource', category, raw)
                maps[category][raw] = len(maps[category]) + 1
    assert len(set(meta['buffers'])) == len(meta['buffers'])
    for index, slot in enumerate(meta['buffers']):
        maps['buffer'][(1 << 32) + slot] = index + 1
    allocations = [line for line in lines if line.startswith('allocate ')]
    assert len(allocations) == len(meta['buffers'])
    for line, slot in zip(allocations, meta['buffers']):
        assert int(line.split()[1]) == meta['ranks'][str(slot)][1]
    for line in lines:
        words = line.split(' ')
        kind = words[0]
        def change(at, category):
            words[at] = str(maps[category][int(words[at])])
        if kind in ('stream', 'event'):
            change(1, kind)
        elif kind in ('capture', 'synchronize', 'zero', 'download', 'copy', 'launch'):
            change(1, 'stream')
        elif kind == 'upload':
            if words[1] != 'arena':
                change(1, 'stream')
        elif kind in ('record', 'wait'):
            change(1, 'stream'); change(2, 'event')
        elif kind in ('captured', 'replay'):
            change(1, 'stream'); change(2, 'graph')
        elif kind in ('resolve', 'configure'):
            change(1, 'function')
        elif kind == 'release':
            if len(words) > 1:
                assert words[1] in ('graph', 'stream', 'event')
                change(2, words[1])
        else:
            assert kind in ('allocate', 'state', 'serving'), kind
        yield RAW.sub(lambda m: 'raw: ' + str(maps['buffer'].get(int(m[1]), int(m[1]))), ' '.join(words))


def graph_groups(lines):
    """Validate GraphCache::pieces and atomic shared-expert units before flattening.

    Added producers increase the unit count, so the existing proportional
    splitter may move a boundary. Every launch/event remains ordered; only
    validated capture delimiters move to the end of their contiguous group.
    """
    out = []; groups = []; at = 0
    while at < len(lines):
        if not lines[at].startswith('capture '):
            out.append(lines[at]); at += 1; continue
        stream = lines[at].split()[1]; pieces = []; metadata = []; body = []; ends = []
        while at < len(lines) and lines[at] == 'capture ' + stream:
            start = lines[at]; at += 1; unit_count = 0; nodes = 0; fork = None
            while at < len(lines) and not lines[at].startswith('captured '):
                line = lines[at]; words = line.split(); body.append(line); at += 1
                if words[0] == 'launch':
                    assert words[1] == stream or fork is not None and words[1] == fork[0]
                    nodes += 1
                    if fork is None:unit_count += 1
                    else:assert '_cache_decode_control' not in words[2] and '_cached_profile' not in words[2]
                elif words[0] == 'record':
                    assert at < len(lines)
                    wait = lines[at].split()
                    assert wait[0] == 'wait' and wait[2] == words[2]
                    if fork is None:
                        assert words[1] == stream and wait[1] != stream
                        fork = (wait[1], words[2]); unit_count += 1
                    else:
                        assert words[1] == fork[0] and wait[1] == stream and words[2] != fork[1]
                        fork = None
                    body.append(lines[at]); at += 1
                else:
                    raise AssertionError(('unexpected captured operation',line))
            assert at < len(lines) and fork is None, 'graph splits an active fork'
            closed = lines[at].split(); at += 1
            assert closed[:2] == ['captured',stream] and int(closed[3]) == nodes, 'graph node count differs'
            pieces.append((unit_count,nodes)); ends.append(len(body))
            metadata += [start, ' '.join(closed[:3] + ['0'])]
        total = sum(p[0] for p in pieces)
        bounds = [0,total//32,total//6,total]
        expected = [b-a for a,b in zip(bounds,bounds[1:]) if b>a]
        mode = 'proportional'
        mapped = [i for i,line in enumerate(body) if line.startswith('launch ')
                  and line.split()[2] == 'chop_row_mxfp8_decode_v1']
        if len(mapped) > 1 and len(pieces) == len(mapped) + 1:
            # Target graph sections now join each mapped layer immediately
            # before its consumer, preserving whole shared-expert regions.
            mode = 'each-mapped-row'
            assert ends[:-1] == mapped, 'mapped-row consumer moved across its layer join'
        elif len(pieces) == 2 and len(expected) == 3:
            # DSpark captures its whole prefix and suffix separately, with
            # the mapped-row wait outside them. The cut is exactly before
            # the first EngramRows widening entry, independent of node count.
            mode = 'first-mapped-row'
            consumer = 'chop_row_mxfp8_decode_v1'
            first = next(i for i,line in enumerate(body) if line.startswith('launch ') and line.split()[2] == consumer)
            assert ends[0] == first, 'DSpark mapped-row consumer moved across the wait'
        else:
            assert [p[0] for p in pieces] == expected, ('graph piece rule',pieces,expected)
        groups.append({'stream':stream,'mode':mode,'units':total,'pieces':pieces,
                       'cache_producers':sum(line.startswith('launch ') and '_cache_decode_control0_' in line for line in body)})
        out.extend(body); out.extend(metadata)
    return out, groups


def audit(base, head, base_meta, head_meta, symbols, added):
    base, old_graphs = graph_groups(base)
    head, new_graphs = graph_groups(head)
    assert [(g['mode'],len(g['pieces'])) for g in old_graphs] == [(g['mode'],len(g['pieces'])) for g in new_graphs]
    assert len(symbols) == 24
    winners = {s for s in symbols if s.endswith('_cached_profile0_v1')}
    assert len(winners) == 6
    functions = {line.split()[1]: line.split()[2] for line in head if line.startswith('resolve ')}
    extras = [line for line in head if line.startswith('resolve ') and line.split()[2] in symbols]
    assert [line.split()[2] for line in extras] == symbols * 4
    assert Counter(head_meta['modules'].values())[CUBIN] == 4
    meta = json.loads(json.dumps(head_meta))
    arenas = [slot for slot in meta['buffers'] if meta['ranks'][str(slot)][1] == ARENA]
    assert sorted(meta['ranks'][str(slot)][0] for slot in arenas) == [0, 1, 2, 3]
    if added:
        assert CUBIN not in base_meta['modules'].values()
        meta['buffers'] = [slot for slot in meta['buffers'] if slot not in arenas]
    else:
        assert base_meta == head_meta, 'paired resources moved'
    kept = []; configured = Counter(); changed = Counter(); i = 0
    while i < len(head):
        line = head[i]; words = line.split(); i += 1
        if added and words[0] == 'resolve' and words[2] in symbols:
            continue
        if words[0] == 'configure' and functions[words[1]] in symbols:
            symbol = functions[words[1]]
            assert symbol in winners and words[2:] == ['73984']
            configured[symbol] += 1
            if added:
                continue
        if added and words == ['allocate', str(ARENA), '256']:
            continue
        if words[0] == 'launch' and '_cache_decode_control' in words[2]:
            assert words[2] in symbols and i < len(head)
            consumer = head[i]; i += 1
            assert consumer.startswith('launch ') and consumer.split()[2] in winners
            line, symbol = source_pair(line, consumer, head_meta)
            changed[symbol] += 1
        elif words[0] == 'launch':
            assert '_cached_profile' not in words[2], 'consumer without adjacent producer'
        kept.append(line)
    assert configured == Counter({symbol: 4 for symbol in winners})
    released = 4 * (len(symbols) + 2) if added else 0
    if released:
        assert kept[-released:] == ['release'] * released, 'added resources must retire at teardown'
        kept = kept[:-released]
    assert len(base) == len(kept), ('retained count', len(base), len(kept))
    assert [base_meta['ranks'][str(s)] for s in base_meta['buffers']] == [meta['ranks'][str(s)] for s in meta['buffers']]
    if added:
        pairs = zip(canonical(base, base_meta), canonical(kept, meta))
    else:
        pairs = zip(base, kept)
    for index, (old, new) in enumerate(pairs):
        assert old == new, ('retained operation changed', index, old, new)
    return {'source_substitutions':dict(changed), 'added_producers':sum(changed.values()),
            'added_resolves':96 if added else 0,'added_configurations':24 if added else 0,
            'added_cache_allocations':4 if added else 0,'bytes_each':ARENA,
            'added_graph_nodes':sum(g['cache_producers'] for g in new_graphs),
            'old_graphs':old_graphs,'new_graphs':new_graphs,
            'added_releases':released,'retained_operations':len(base)}


def controls(head, meta):
    at = next(i for i,line in enumerate(head) if line.startswith('launch ') and '_cache_decode_control0_' in line)
    producer, consumer = head[at:at+2]
    bad = []
    for which, old, new in [(0,'grid_z: 1,','grid_z: 2,'), (1,'grid_z: 8,','grid_z: 16,'),
                            (1,'73984','73983'), (1,'grid_x: 1,','grid_x: 2,'),
                            (1,'NullDevicePointer','ScalarU32(0)'), (0,'control0','control1'),
                            (1,'profile0','profile1'), (1,'flags: 1,','flags: 0,'),
                            (1,'cluster: LaunchClusterV1 { x: 1,','cluster: LaunchClusterV1 { x: 2,')]:
        pair = [producer, consumer]; pair[which] = pair[which].replace(old,new,1)
        assert pair != [producer, consumer]
        bad.append(pair)
    cache = list(VIEW.finditer(consumer))[-2]
    for part in ('raw', 'offset', 'length'):
        piece = re.sub(part + r': (\d+)', lambda m: f'{part}: {int(m[1])+1}',cache[0],count=1)
        bad.append([producer,consumer[:cache.start()]+piece+consumer[cache.end():]])
    for i,pair in enumerate(bad):
        try:
            source_pair(*pair,meta)
        except (AssertionError,KeyError,ValueError):
            continue
        raise AssertionError(f'control {i} accepted')
    return len(bad)


def main():
    parser = argparse.ArgumentParser()
    for key in ('base','head','buckets','report'):
        parser.add_argument('--'+key,type=Path,required=True)
    args = parser.parse_args()
    lock, = json.loads((Path(__file__).resolve().parents[1]/'bundle/kernels.lock.d/latent_decode_cache.lock').read_text())
    assert lock['name'] == CUBIN
    symbols = [entry['symbol'] for entry in lock['entries']]
    result = {}
    for suffix, expected in PINS.items():
        name = 'deepseek41-flash-'+suffix+'.journal'
        base, bm = read(args.base/name); head, hm = read(args.head/name)
        old = digest(base)
        assert (old['lines'],old['fnv']) == expected, ('baseline pin',name,old)
        if suffix in ('tp4.native','tp4.media'):
            row = audit(base,head,bm,hm,symbols,True)
        else:
            assert base == head and bm == hm, ('unqualified topology changed',name)
            row = {'identical':True}
        row.update(old=old,new=digest(head));result[name] = row
    stem='deepseek41-flash-tp4.Native.selected-cache.'
    base,bm=read(args.buckets/(stem+'reference.journal'))
    head,hm=read(args.buckets/(stem+'serving.journal'))
    pair=audit(base,head,bm,hm,symbols,False)
    assert set(pair['source_substitutions']) == {s for s in symbols if s.endswith('_cached_profile0_v1')}
    assert pair['added_producers']==5*40*4
    rejected=controls(head,hm)
    for kind in ('captured', 'merge'):
        changed = head.copy()
        at = next(i for i,line in enumerate(head) if line.startswith('captured ') and int(line.split()[3]) > 0) if kind == 'captured' else next(i for i,line in enumerate(head) if line.startswith('launch ') and 'chop_latent_sources_' in line and '_merge_' in line)
        if kind == 'captured':
            words = changed[at].split(); words[3] = str(int(words[3]) + 1); changed[at] = ' '.join(words)
        else:
            changed[at] = changed[at].replace('grid_z: 1,', 'grid_z: 2,', 1)
        assert changed[at] != head[at]
        try:
            audit(base,changed,bm,hm,symbols,False)
        except AssertionError:
            rejected += 1
        else:
            raise AssertionError('retained '+kind+' mutation accepted')
    output={'passed':True,'journals':result,'buckets':pair,'rejected_pair_mutations':rejected,
            'cubin_sha256':lock['cubin_sha256'],
            'release_scope':'Anonymous added releases are counted in final teardown; tests balance acquisition and release.'}
    args.report.write_text(json.dumps(output,indent=2)+'\n')
    print('SELECTED_CACHE_JOURNAL_AUDIT_OK',rejected,'negative controls')


if __name__=='__main__':
    main()

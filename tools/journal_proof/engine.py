"""Exact journal rewrites with a single create-derived identity map per fixture."""
import re
from common import Invalid, arguments, balanced, need

RAW = re.compile(r'raw:\s*(\d+)')
ALLOC = re.compile(r'^allocate (\d+) (\d+)(?: -> raw: (\d+))?$')
RESOLVE = re.compile(r'^resolve (\d+) (\S+)$')
MODULE = re.compile(r'^module (\d+) (\S+)$')
CONFIGURE = re.compile(r'^configure (\d+) (\d+)$')


class Unproven(ValueError):
    pass


def require(value, message):
    if not value:
        raise Unproven(message)


def launch(line):
    if not line.startswith('launch ') or 'LaunchConfigV1 {' not in line:
        return None
    at = line.index('LaunchConfigV1 {')
    prefix = line[:at]
    match = re.fullmatch(r'launch (.+?) (chop_\w+) (?:rows (\d+) )?', prefix)
    require(match is not None, 'unknown launch prefix')
    config, end = balanced(line, line.index('{', at))
    suffix = line[end:].strip()
    require(suffix.startswith('[') and suffix.endswith(']'), 'missing launch arguments')
    fields = dict((k, int(v)) for k, v in re.findall(r'(\w+): (\d+)', config))
    require(len(fields) == len(re.findall(r'(\w+): (\d+)', config)), 'duplicate launch config field')
    return dict(prefix=match[1], symbol=match[2], rows=int(match[3]) if match[3] else None,
                config=config, fields=fields, args=arguments(suffix[1:-1]))


def render(item):
    rows = '' if item['rows'] is None else f"rows {item['rows']} "
    return f"launch {item['prefix']} {item['symbol']} {rows}LaunchConfigV1 {item['config']} [" + ', '.join(item['args']) + ']'


def value(expr, item):
    if type(expr) is int:
        return expr
    need(isinstance(expr, dict) and len(expr) == 1, 'invalid integer expression')
    key, arg = next(iter(expr.items()))
    if key == 'config':
        need(arg in item['fields'], f'unknown config field {arg}')
        return item['fields'][arg]
    if key in ('ceil_div', 'mul', 'add'):
        need(isinstance(arg, list) and len(arg) == 2, 'binary expression needs two operands')
        a, b = (value(x, item) for x in arg)
        if key == 'ceil_div':
            need(b > 0, 'invalid divisor')
            return -(-a // b)
        return a * b if key == 'mul' else a + b
    raise Invalid(f'unsupported expression {key}')


def set_config(item, edits):
    original = dict(item['fields'])
    for key, expr in edits.items():
        need(key in original, f'unknown config key {key}')
        v = value(expr, dict(item, fields=original))
        need(type(v) is int and v >= 0, 'negative/noninteger launch config')
        item['config'], n = re.subn(r'\b' + re.escape(key) + r': \d+\b', f'{key}: {v}', item['config'])
        need(n == 1, f'ambiguous config edit {key}')
        item['fields'][key] = v


def argument(expr, originals, allocation_context=None):
    """An argument from an identified original launch, optionally an exact edit."""
    need(isinstance(expr, dict), 'argument expression must be an object')
    if 'allocation' in expr or 'peer_array' in expr:
        from allocation_args import argument as bound_argument
        return bound_argument(expr,allocation_context)
    if set(expr) == {'literal'}:
        result = expr['literal']
        need(isinstance(result, str) and re.fullmatch(r'(?:Scalar)?(?:U32|U64|I32|F32)\([-+0-9.e]+\)', result),
             'only scalar literals are allowed; handles must come from original arguments')
        return result
    need(set(expr) <= {'launch', 'arg', 'replace', 'subspan'} and 'arg' in expr, 'unknown argument expression')
    source = expr.get('launch', 0)
    need(type(source) is int and 0 <= source < len(originals), 'argument source launch is out of range')
    index = expr['arg']
    need(type(index) is int and 0 <= index < len(originals[source]['args']), 'argument index out of range')
    result = originals[source]['args'][index]
    for edit in expr.get('replace', []):
        need(set(edit) == {'from', 'to'}, 'argument edit needs from/to')
        before, after = edit['from'], edit['to']
        need(isinstance(before, str) and before and isinstance(after, str), 'invalid argument edit')
        need('raw:' not in before and 'raw:' not in after, 'argument edits cannot rename handles')
        require(result.count(before) == 1, 'argument edit does not match exactly once')
        changed = result.replace(before, after, 1)
        require(RAW.findall(changed) == RAW.findall(result), 'argument edit changed a handle')
        from allocation_args import SPAN
        require(SPAN.findall(changed)==SPAN.findall(result),
                'generic argument edits cannot resize spans; declare allocation-bound span_changes')
        result = changed
    if 'subspan' in expr:
        offset=expr['subspan']
        need(type(offset) is int and offset>=0,'subspan offset must be a nonnegative integer')
        span=re.fullmatch(r'DeviceBuffer\(DeviceBufferSpanV1 \{ buffer: BundleDeviceBufferHandleV1 \{ raw: (\d+) \}, offset: (\d+), length: (\d+) \}\)',result)
        require(span is not None and offset<=int(span[3]),'subspan leaves its source buffer')
        result=result.replace(f'offset: {span[2]}, length: {span[3]}',
                              f'offset: {int(span[2])+offset}, length: {int(span[3])-offset}')
    return result


def selected(rule, item):
    if item is None or item['symbol'] != rule['from_symbol']:
        return False
    for field, limits in rule.get('where', {}).items():
        need(field in item['fields'], f'unknown selection field {field}')
        actual = item['fields'][field]
        need(isinstance(limits, dict) and set(limits) <= {'eq', 'min', 'max'}, 'invalid selection')
        for op, expected in limits.items():
            need(type(expected) is int, 'selection needs integer')
            if (op == 'eq' and actual != expected) or (op == 'min' and actual < expected) or (op == 'max' and actual > expected):
                return False
    return True


def strip_additions(lines, policy, counts, kept=None):
    additions = {}
    for kind, pattern in (('added_resolves', RESOLVE), ('added_modules', MODULE), ('added_configures', CONFIGURE)):
        for rule in policy.get(kind, []):
            need(set(rule) == {'symbol', 'ids', 'count', 'positions'} if kind != 'added_configures'
                 else set(rule) == {'bytes', 'ids', 'count', 'positions'}, f'invalid {kind} rule')
            ids, positions = rule['ids'], rule['positions']
            need(type(rule['count']) is int and rule['count'] > 0 and len(ids) == len(positions) == rule['count'],
                 f'{kind}: count must name every exact ID and position')
            need(len(set(ids)) == len(ids), f'{kind}: repeated ID')
            payload = str(rule['bytes']) if kind == 'added_configures' else rule['symbol']
            verb = {'added_resolves': 'resolve', 'added_modules': 'module', 'added_configures': 'configure'}[kind]
            actual = [(i + 1, int(m[1])) for i, line in enumerate(lines) if (m := pattern.fullmatch(line)) and int(m[1]) in ids and m[2] == payload]
            # Count all instances of this added symbol, not just the declared IDs.
            require(actual == list(zip(positions, ids)), f'{kind}/{payload}: exact IDs/count/positions differ')
            for position, id_ in zip(positions, ids):
                need(position not in additions, 'overlapping added records')
                require(lines[position - 1] == f'{verb} {id_} {payload}', 'added record differs')
                additions[position] = kind
            counts[kind] += rule['count']
    if kept is not None:
        kept.extend(i for i in range(len(lines)) if i + 1 not in additions)
    return [line for i, line in enumerate(lines, 1) if i not in additions]


def rule_candidates(rule,original):
    if 'lines' not in rule:
        return enumerate(original)
    indexes=rule['lines']
    need(isinstance(indexes,list) and all(type(i)is int and 1<=i<=len(original) for i in indexes) and
         len(indexes)==len(set(indexes)),'invalid exact launch line selector')
    return ((i-1,original[i-1]) for i in indexes)


def transform(lines, policy, counts, allocation_context=None, trace=None):
    """The expected head lines. A `trace` list receives, for every output
    line, the tuple of base line indexes it derives from (a merge names all
    of its sources, each expansion member its one source)."""
    from allocation_args import layout_strides
    given = lines
    lines = layout_strides(lines, policy.get('layout_strides', []), allocation_context, counts)
    original = [launch(line) for line in lines]
    result = list(lines)
    used = set()
    for rule in policy.get('launch_rules', []):
        need(set(rule) <= {'from_symbol','to_symbol','where','config','args','count','expect','lines','prefix'}
             and {'from_symbol','count'} <= set(rule), 'unknown launch rule field')
        need(rule['count']=='all' or (type(rule['count']) is int and rule['count'] >= 0), 'launch rule count must be an integer or all')
        matched = 0
        for index, item in rule_candidates(rule,original):
            if not selected(rule,item) or ('lines' in rule and index+1 not in rule['lines']) or ('prefix' in rule and item['prefix']!=rule['prefix']):
                continue
            require(index not in used, 'overlapping launch rules')
            used.add(index)
            matched += 1
            changed=dict(item, fields=dict(item['fields']), args=list(item['args']))
            for field, expected in rule.get('expect',{}).items():
                require(item['fields'].get(field)==expected, f'launch {index+1}: base expectation {field} failed')
            changed['symbol']=rule.get('to_symbol',item['symbol'])
            set_config(changed,rule.get('config',{}))
            if 'args' in rule:
                changed['args']=[argument(e,[item],allocation_context) for e in rule['args']]
            result[index]=render(changed)
        require(rule['count']=='all' or matched==rule['count'],f"{rule['from_symbol']}: {matched} launches, expected {rule['count']}")
        counts['launch_changes']+=matched
    for kind, verb in (('symbol_substitutions',None),('cubin_substitutions','module')):
        for rule in policy.get(kind,[]):
            need(set(rule)=={'from','to','count'},f'invalid {kind} rule')
            old,new=rule['from'],rule['to']
            matched=0
            need(re.fullmatch(r'[A-Za-z0-9_]+',old) and re.fullmatch(r'[A-Za-z0-9_]+',new),'invalid symbol/cubin')
            for index,line in enumerate(result):
                item=launch(line)
                if kind=='symbol_substitutions' and item and item['symbol']==old:
                    require(index not in used,'overlapping symbol/launch rewrite')
                    used.add(index)
                    item['symbol']=new
                    result[index]=render(item)
                    matched+=1
                elif (m:=RESOLVE.fullmatch(line)) and kind=='symbol_substitutions' and m[2]==old:
                    result[index]=f'resolve {m[1]} {new}'
                    matched+=1
                elif (m:=MODULE.fullmatch(line)) and kind=='cubin_substitutions' and m[2]==old:
                    result[index]=f'module {m[1]} {new}'
                    matched+=1
            require(matched==rule['count'],f'{kind}/{old}: count {matched}, expected {rule["count"]}')
            counts[kind]+=matched
    for rule in policy.get('state_sizes',[]):
        need(set(rule)=={'line','from','to'},'state-size edit needs exact line/from/to')
        index=rule['line']-1
        need(0<=index<len(result),'state-size line out of range')
        require(result[index]==rule['from'],'state-size base record differs')
        before_size=re.fullmatch(r'(state |serving kv )(\d+)',rule['from'])
        after_size=re.fullmatch(r'(state |serving kv )(\d+)',rule['to'])
        before_alloc,after_alloc=ALLOC.fullmatch(rule['from']),ALLOC.fullmatch(rule['to'])
        require((before_size is not None and after_size is not None and before_size[1]==after_size[1])
                or (before_alloc is not None and after_alloc is not None
                    and before_alloc.groups()[1:]==after_alloc.groups()[1:]),
                'state-size declaration changes an identity or alignment, not just bytes')
        require(RAW.findall(rule['from'])==RAW.findall(rule['to']),'state-size change renamed handle')
        result[index]=rule['to']
        counts['state_sizes']+=1
    for rule in policy.get('upload_payloads',[]):
        need(set(rule)=={'line','from','to'},'upload payload change needs exact line/from/to')
        need(all(isinstance(rule[k],str) and re.fullmatch(r'[0-9a-f]{16}',rule[k]) for k in ('from','to')) and rule['from']!=rule['to'],
             'upload payload digests must be distinct 16-digit FNV values')
        index=rule['line']-1 if type(rule['line']) is int else -1
        need(0<=index<len(result),'upload payload line out of range')
        m=re.fullmatch(r'(upload \d+ DeviceBufferSpanV1 \{[^{}]*\{[^{}]*\}[^{}]*\}) ([0-9a-f]{16})',result[index])
        require(m is not None and m[2]==rule['from'],'upload payload base record differs')
        require(index not in used,'overlapping upload payload rewrite')
        used.add(index)
        result[index]=f"{m[1]} {rule['to']}"
        counts['upload_payloads']+=1
    removed=set()
    for rule in policy.get('removals',[]):
        need(set(rule)=={'line','symbol'},'invalid launch removal')
        index=rule['line']-1
        need(0<=index<len(lines),'removal index out of range')
        require(original[index] and original[index]['symbol']==rule['symbol'],'removal does not name base launch')
        require(index not in used and index not in removed,'overlapping removal')
        removed.add(index)
        counts['removals']+=1
    # Moves use original base line identities. Exact before/after positions
    # are declarations, never inferred by sorting or normalizing the head.
    order=[i for i in range(len(lines)) if i not in removed]
    for rule in policy.get('moves',[]):
        need(set(rule)=={'lines','symbols','before'},'invalid launch move')
        indexes=[i-1 for i in rule['lines']]
        anchor=rule['before']-1
        require(len(indexes)==len(set(indexes)) and all(i in order and original[i] for i in indexes)
                and anchor in order and anchor not in indexes,'invalid move positions')
        require([original[i]['symbol'] for i in indexes]==rule['symbols'],'moved launch symbols differ')
        order=[i for i in order if i not in indexes]
        at=order.index(anchor)
        order[at:at]=indexes
        counts['moved_launches']+=len(indexes)
    merged={}
    for rule in policy.get('merges',[]):
        need(set(rule)=={'lines','symbols','symbol','config','args','outputs'},'invalid merge fields')
        indexes=[i-1 for i in rule['lines']]
        need(indexes==sorted(set(indexes)) and len(indexes)>=2 and all(0<=i<len(lines) for i in indexes),'invalid merge positions')
        require(not (set(indexes)&(used|removed)),'overlapping merge')
        parts=[original[i] for i in indexes]
        require(all(parts) and [p['symbol'] for p in parts]==rule['symbols'],'merge base sequence differs')
        require(len({p['prefix'] for p in parts})==1,'merge crosses streams/stages')
        # Noncontiguous merging is a move. Require that its movement is
        # declared separately, rather than silently hoisting across consumers.
        positions=[order.index(i) for i in indexes]
        require(positions==list(range(positions[0],positions[-1]+1)),'noncontiguous merge requires an explicit prior move')
        changed=dict(parts[0],fields=dict(parts[0]['fields']),symbol=rule['symbol'])
        set_config(changed,rule['config'])
        changed['args']=[argument(expr,parts,allocation_context) for expr in rule['args']]
        need(isinstance(rule['outputs'],list) and rule['outputs'],'merge needs an output mapping')
        for binding in rule['outputs']:
            need({'from_launch','from_arg','to_arg'}<=set(binding)<= {'from_launch','from_arg','to_arg','byte_offset'},'invalid merge output mapping')
            source_output=argument(dict(launch=binding['from_launch'],arg=binding['from_arg'],subspan=binding.get('byte_offset',0)),parts)
            require(source_output==changed['args'][binding['to_arg']],
                    'merged output is not the declared original output')
        result[indexes[0]]=render(changed)
        removed.update(indexes[1:])
        used.update(indexes)
        merged[indexes[0]]=tuple(indexes)
        counts['merges']+=1
        counts['merged_launches']+=len(indexes)
    expanded={}
    for rule in policy.get('expansions',[]):
        need({'from_symbol','sequence','outputs','count'}<=set(rule)<=
             {'from_symbol','sequence','outputs','count','lines','where','prefix'},'invalid expansion fields')
        need(isinstance(rule['sequence'],list) and len(rule['sequence'])>=2,'expansion needs at least two launches')
        need(type(rule['count']) is int and rule['count']>0,'expansion needs a positive exact count')
        matched=0
        for index,item in rule_candidates(rule,original):
            if not selected(rule,item) or ('lines' in rule and index+1 not in rule['lines']) or ('prefix' in rule and item['prefix']!=rule['prefix']):
                continue
            require(index not in used and index not in removed,'overlapping launch expansion')
            used.add(index)
            matched+=1
            sequence=[]
            for part in rule['sequence']:
                need(set(part)=={'symbol','config','args'},'invalid expansion launch')
                changed=dict(item,fields=dict(item['fields']),symbol=part['symbol'])
                set_config(changed,part['config'])
                changed['args']=[argument(expr,[item],allocation_context) for expr in part['args']]
                sequence.append(changed)
            need(isinstance(rule['outputs'],list) and rule['outputs'],'expansion needs an output mapping')
            for output in rule['outputs']:
                need(set(output)=={'from_arg','to_launch','to_arg'},'invalid expansion output mapping')
                require(argument({'arg':output['from_arg']},[item])==sequence[output['to_launch']]['args'][output['to_arg']],
                        'expanded output differs from declared original output')
            expanded[index]=[render(part) for part in sequence]
        require(matched==rule['count'],'expansion count differs')
        counts['expansions']+=matched
        counts['expanded_launches']+=matched*len(rule['sequence'])
    order=[i for i in order if i not in removed]
    output=[line for i in order for line in expanded.get(i,[result[i]])]
    if trace is not None:
        trace.extend(merged.get(i,(i,)) for i in order for _ in expanded.get(i,[None]))
    from allocation_args import resize,validate_spans,layout_spans
    output=resize(output,policy.get('span_changes',[]),allocation_context,counts,
                  layout_spans(policy.get('layout_spans',[]),allocation_context))
    # The base line each output line keeps in place of itself: none for a
    # merge or an expansion, whose spans are all derived.
    kept=[None if i in merged or i in expanded else given[i] for i in order for _ in expanded.get(i,[None])]
    validate_spans(output,allocation_context,kept)
    return output


POLICY_FIELDS={'launch_rules','symbol_substitutions','cubin_substitutions','state_sizes','moves','merges','removals',
               'added_resolves','added_modules','added_configures','expansions','span_changes',
               'layout_spans','layout_strides','upload_payloads'}

"""Exact allocation-bound arguments and spans, including peer arrays."""
import re
from common import need
from engine import require

SPAN=re.compile(r'DeviceBufferSpanV1 \{ buffer: BundleDeviceBufferHandleV1 \{ raw: (\d+) \}, offset: (\d+), length: (\d+) \}')


def bindings(rule,a,b,buffer_map):
    aliases={}
    sizes={}
    for slot,row in a['allocations'].items():
        raw=row[1]
        new=buffer_map[raw]
        other=b['allocations'][new&0xffffffff]
        sizes[raw]=dict(base_bytes=row[3],head_bytes=other[3])
    declarations=rule.get('allocation_bindings',{})
    need(isinstance(declarations,dict),'allocation_bindings must be an object')
    for name,d in declarations.items():
        need(set(d)=={'base_raw','head_raw','rank','base_bytes','head_bytes','stream_raw'},'invalid allocation binding')
        old,new=d['base_raw'],d['head_raw']
        require(buffer_map.get(old)==new,'allocation binding does not follow the authenticated map')
        ar=a['allocations'][old&0xffffffff]
        br=b['allocations'][new&0xffffffff]
        require(ar[2]==br[2]==d['rank'] and ar[3]==d['base_bytes'] and br[3]==d['head_bytes'] and
                ar[5]==d['stream_raw'],'allocation binding rank, size or stream differs')
        aliases[name]=dict(d)
    return dict(aliases=aliases,sizes=sizes,layouts=layouts(rule,aliases))


LAYOUT_FIELDS={'planes','from_stride','to_stride','from_tail','to_tail'}


def layouts(rule,aliases):
    """Declared plane layouts of resized allocations: `planes` equal planes
    of `from_stride` bytes followed by `from_tail` bytes in the base, and of
    `to_stride` plus `to_tail` in the head. Both must tile the authenticated
    allocation sizes, and the allocation must have a declared resize."""
    declarations=rule.get('allocation_layouts',{})
    need(isinstance(declarations,dict),'allocation_layouts must be an object')
    resized={d['base_raw']:d for d in rule.get('allocation_resizes',[])}
    out={}
    for alias,d in declarations.items():
        need(alias in aliases,f'layout {alias} names no declared allocation binding')
        need(isinstance(d,dict) and set(d)==LAYOUT_FIELDS and all(type(d[k]) is int and d[k]>=0 for k in LAYOUT_FIELDS)
             and d['planes']>=1 and d['from_stride']>0 and d['to_stride']>0,'invalid allocation layout')
        b=aliases[alias]
        r=resized.get(b['base_raw'])
        require(r is not None and r['from']==b['base_bytes'] and r['to']==b['head_bytes'],
                f'layout {alias} is not tied to a declared allocation resize')
        require(d['planes']*d['from_stride']+d['from_tail']==b['base_bytes'] and
                d['planes']*d['to_stride']+d['to_tail']==b['head_bytes'],
                f'layout {alias} does not tile its base and head allocations')
        out[alias]=dict(d,raw=b['base_raw'],base_bytes=b['base_bytes'],head_bytes=b['head_bytes'])
    return out


LAYOUT_FORMS=('allocation','plane','within_plane','strided','tail')


def layout_image(d,form,offset,length):
    """The head span of a base span inside a declared layout, by its form."""
    planes,old,new=d['planes'],d['from_stride'],d['to_stride']
    if form=='allocation':
        require((offset,length)==(0,d['base_bytes']),'allocation-form span is not the whole base allocation')
        return 0,d['head_bytes']
    if form=='tail':
        start=planes*old
        require(offset>=start and offset+length<=d['base_bytes'],'tail-form span is not inside the base tail')
        require(offset-start+length<=d['to_tail'],'tail-form span exceeds the head tail')
        return planes*new+offset-start,length
    layer,within=divmod(offset,old)
    require(layer<planes,'layout span lies past the planes')
    if form=='plane':
        require(within==0 and length==old,'plane-form span is not one whole base plane')
        return layer*new,new
    if form=='within_plane':
        require(within+length<=old and within+length<=new,'within-plane span crosses a plane boundary')
        return layer*new+within,length
    # One entry at the same offset of every plane, read with the plane stride.
    entry=length-(planes-1)*old
    require(form=='strided' and planes>=2 and layer==0 and entry>0 and within+entry<=min(old,new),
            'strided-form span is not one entry in every plane')
    return offset,(planes-1)*new+entry


def layout_spans(rules,context):
    """Span changes whose head span is derived from a declared layout."""
    if not rules:
        return {}
    need(isinstance(rules,list) and context is not None and context.get('layouts'),
         'layout span changes require declared allocation layouts')
    changes={}
    for d in rules:
        need(isinstance(d,dict) and set(d)=={'allocation','form','from_offset','from_length','to_offset','to_length','count'},
             'invalid layout span change')
        need(d['allocation'] in context['layouts'],'layout span change names an undeclared layout')
        need(d['form'] in LAYOUT_FORMS,'unknown layout span form')
        for key in ('from_offset','from_length','to_offset','to_length','count'):
            need(type(d[key]) is int and d[key]>=0,'invalid layout span integer')
        need(d['count']>0,'layout span change needs a positive count')
        layout=context['layouts'][d['allocation']]
        image=layout_image(layout,d['form'],d['from_offset'],d['from_length'])
        require(image==(d['to_offset'],d['to_length']),
                f"layout span ({d['from_offset']}, {d['from_length']}) maps to {image}, not the declared head span")
        key=(layout['raw'],d['from_offset'],d['from_length'])
        need(key not in changes,'overlapping layout span changes')
        changes[key]=[d,0]
    return changes


SCALAR=re.compile(r'Scalar(U64|U32)\((\d+)\)')


def layout_strides(lines,rules,context,counts):
    """Launch stride scalars that follow a declared layout's plane stride."""
    if not rules:
        return lines
    from engine import launch,render
    need(isinstance(rules,list) and context is not None and context.get('layouts'),
         'layout stride changes require declared allocation layouts')
    seen=set()
    result=list(lines)
    for d in rules:
        need(isinstance(d,dict) and set(d)=={'symbol','span_arg','stride_arg','allocation','count'},'invalid layout stride change')
        need(d['allocation'] in context['layouts'],'layout stride change names an undeclared layout')
        need(all(type(d[k]) is int and d[k]>=0 for k in ('span_arg','stride_arg','count')) and d['span_arg']!=d['stride_arg']
             and d['count']>0,'invalid layout stride integers')
        layout=context['layouts'][d['allocation']]
        matched=0
        for index,line in enumerate(result):
            item=launch(line)
            if not item or item['symbol']!=d['symbol'] or len(item['args'])<=max(d['span_arg'],d['stride_arg']):
                continue
            m=SPAN.fullmatch(item['args'][d['span_arg']][len('DeviceBuffer('):-1]) if item['args'][d['span_arg']].startswith('DeviceBuffer(') else None
            if m is None or int(m[1])!=layout['raw']:
                continue
            need((index,d['stride_arg']) not in seen,'overlapping layout stride changes')
            seen.add((index,d['stride_arg']))
            scalar=SCALAR.fullmatch(item['args'][d['stride_arg']])
            require(scalar is not None and int(scalar[2])==layout['from_stride'],
                    f"line {index+1}: {d['symbol']} stride argument {d['stride_arg']} is not the declared base stride")
            item['args'][d['stride_arg']]=f"Scalar{scalar[1]}({layout['to_stride']})"
            result[index]=render(item)
            matched+=1
        require(matched==d['count'],f"{d['symbol']}: {matched} layout stride changes, expected {d['count']}")
        counts['layout_strides']+=matched
    return result


def span(raw,offset,length):
    return f'DeviceBufferSpanV1 {{ buffer: BundleDeviceBufferHandleV1 {{ raw: {raw} }}, offset: {offset}, length: {length} }}'


def allocation(expr,context):
    need(context is not None,'allocation arguments require authenticated auxiliary captures')
    need(set(expr)=={'allocation','offset','length'},'allocation argument needs alias, offset and length')
    need(expr['allocation'] in context['aliases'],'undeclared workspace allocation')
    d=context['aliases'][expr['allocation']]
    offset=expr['offset']
    length=expr['length']
    need(type(offset)is int and type(length)is int and offset>=0 and length>=0,'invalid allocation span')
    require(offset+length<=d['head_bytes'],'workspace argument exceeds its new allocation')
    return 'DeviceBuffer('+span(d['base_raw'],offset,length)+')'


def argument(expr,context):
    if 'allocation' in expr:
        return allocation(expr,context)
    need(set(expr)=={'peer_array'} and isinstance(expr['peer_array'],list),'invalid peer-array expression')
    need(1<=len(expr['peer_array'])<=64,'peer-array length out of bounds')
    entries=[]
    for item in expr['peer_array']:
        if item is None:
            entries.append('None')
        else:
            text=allocation(item,context)
            entries.append('Some('+text[len('DeviceBuffer('):-1]+')')
    return 'DevicePointerArray(['+', '.join(entries)+'])'


def resize(lines,rules,context,counts,layout=None):
    if not rules and not layout:
        return lines
    need(context is not None,'span changes require authenticated allocation evidence')
    changes={}
    for d in rules:
        need(set(d)=={'allocation','from_offset','from_length','to_offset','to_length','count'},'invalid span change')
        need(d['allocation'] in context['aliases'],'span change names an undeclared allocation')
        binding=context['aliases'][d['allocation']]
        raw=binding['base_raw']
        for key in ('from_offset','from_length','to_offset','to_length','count'):
            need(type(d[key])is int and d[key]>=0,'invalid span change integer')
        require(d['from_offset']+d['from_length']<=binding['base_bytes'] and
                d['to_offset']+d['to_length']<=binding['head_bytes'],'span change exceeds allocation bounds')
        key=(raw,d['from_offset'],d['from_length'])
        need(key not in changes,'overlapping span changes')
        changes[key]=[d,0]
    for key,entry in (layout or {}).items():
        need(key not in changes,'a span has both a span change and a layout span change')
        entry[0]=dict(entry[0],layout=True)
        changes[key]=entry
    def replace(m):
        key=tuple(map(int,m.groups()))
        if key not in changes:
            return m[0]
        d=changes[key][0]
        changes[key][1]+=1
        return span(key[0],d['to_offset'],d['to_length'])
    result=[SPAN.sub(replace,line) for line in lines]
    for d,count in changes.values():
        require(count==d['count'],'span change count differs' if not d.get('layout') else
                f"layout span ({d['from_offset']}, {d['from_length']}) occurs {count} times, expected {d['count']}")
        counts['layout_spans' if d.get('layout') else 'span_changes']+=count
    return result


def kept_whole(line,source):
    """The parts of `line` whose spans may keep an oversized extent from the
    base: each (part, kept) where kept says the base line `source` holds the
    identical part in the same place. A launch's parts are its arguments, each
    compared whole (its variant and every element, empty entries included) at
    the same argument index; any other record is one part, kept only whole."""
    from engine import launch
    item=launch(line)
    base=launch(source) if source is not None else None
    if item is None:
        return [(line,source is not None and base is None and line==source)]
    held=base['args'] if base is not None else []
    return [(arg,i<len(held) and held[i]==arg) for i,arg in enumerate(item['args'])]


def validate_spans(lines,context,kept=None):
    """Every span must lie inside its allocation's head size. The one
    exception is an oversized span the base already held (a fake host does not
    bound spans) in an argument kept identical: `kept[j]` is the base line
    output line j keeps (None for a merged or expanded line), the whole
    argument must equal the base line's argument at the same index, and every
    allocation that argument references must keep its size. Any difference in
    the argument bounds all its spans; another record is exempt only whole."""
    if context is None:
        return
    kept=kept or [None]*len(lines)
    need(len(kept)==len(lines),'span validation needs one source per line')
    for line,source in zip(lines,kept):
        for part,same in kept_whole(line,source):
            spans=[tuple(map(int,m.groups())) for m in SPAN.finditer(part)]
            sizes=[context['sizes'][raw] for raw,_,_ in spans if raw in context['sizes']]
            exempt=same and all(size['base_bytes']==size['head_bytes'] for size in sizes)
            for raw,offset,length in spans:
                if raw in context['sizes']:
                    require(offset+length<=context['sizes'][raw]['head_bytes'] or exempt,
                            'rewritten span exceeds its authenticated new allocation')

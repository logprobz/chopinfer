"""Compare direct-fold bindings and every retained serving journal operation."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ARG = re.compile(r'DeviceBuffer\(DeviceBufferSpanV1 \{ buffer: BundleDeviceBufferHandleV1 \{ raw: (\d+) \}, offset: (\d+), length: (\d+) \}\)|NullDevicePointer')
SHAPE = re.compile(r'_n(\d+)_k(\d+)_g(\d+)_')
DIRECT = 'chop_deepseek41_mxfp8_direct_'
WINS = {(1024,4096,8,False,1), (16384,1280,1,False,1),
        (2304,5120,1,True,1), (2304,5120,1,True,2),
        (25600,6144,1,False,32), (25600,6144,1,False,64),
        (5120,2048,1,False,1), (5120,2048,1,False,2)}


def args(line):
    return [tuple(map(int,m.group(1,2,3))) if m[1] else None for m in ARG.finditer(line)]


def shape(symbol):
    return (*map(int,SHAPE.search(symbol).groups()), '_pair' in symbol)


def normalize(lines):
    functions={l.split()[1]:l.split()[2] for l in lines if l.startswith('resolve ')}
    maps={k:{} for k in ('buffer','stream','event','graph','mapped')}
    packed={}; states={}; private=set(); initialized=[]; removed_allocations=[]
    omit=set(); workspaces={}; direct_calls=Counter()
    # Remove only the allocation records directly owned by a checked initializer.
    for i,line in enumerate(lines):
        if not line.startswith('launch ') or '_prepare_v1 ' not in line or '_mxfp8_' not in line:
            continue
        symbol=line.split()[2]; sh=shape(symbol); h=args(line)
        assert len(h)==5 and 'flags: 0,' in line
        count=4 if sh[3] else 3
        # A direct-only operation has no global partial workspace.
        preceding=0
        while i-preceding-1>=0 and lines[i-preceding-1].startswith('allocate '): preceding+=1
        assert preceding in (count,count-1), (symbol,preceding,count)
        allocations=[int(lines[j].split()[1]) for j in range(i-preceding,i)]
        expected=[h[2][2]]+([h[3][2]] if sh[3] else [])+[h[4][2]]
        assert allocations[:len(expected)]==expected
        assert re.findall(r'ScalarU32\((\d+)\)',line)==[str(h[4][2]//4)]
        assert h[0][2]==sh[0]*sh[1]*sh[2]
        for raw,fragment in [(h[0],h[2])]+([(h[1],h[3])] if sh[3] else []):
            assert raw[2]==fragment[2] and fragment[0] not in private
            private.add(fragment[0]); packed[fragment]=raw
        assert h[4][0] not in private
        assert h[4][1]==0 and h[2][1]==0 and (h[3] is None or h[3][1]==0)
        private.add(h[4][0]); states[h[4][0]]=(sh,h[4],allocations[-1] if len(allocations)>len(expected) else 0)
        initialized.append((sh,h[0],h[1],h[4][2]))
        omit.update(range(i-preceding,i+1));removed_allocations.extend(allocations)
    def handle(kind,value):
        table=maps[kind]; value=str(value)
        if value not in table: table[value]=len(table)
        return table[value]
    def span(value):
        return None if value is None else (handle('buffer',value[0]),*value[1:])
    def text(line):
        words=line.split();word=words[0]
        if word in ('stream','event'):
            x=line.split();line=f'{word} {handle(word,x[1])}'
        elif word in ('record','wait'):
            x=line.split();line=f'{word} {handle("stream",x[1])} {handle("event",x[2])}'
        elif word in ('captured','replay'):
            x=line.split();line=' '.join([word,str(handle('stream',x[1])),str(handle('graph',x[2]))]+x[3:])
        elif word in ('zero','launch','copy','download','capture','synchronize') or (word=='upload' and line.split()[1].isdigit()):
            x=line.split(' ',2);line=' '.join([x[0],str(handle('stream',x[1]))]+x[2:])
        elif word=='release':
            assert words[1] in ('graph','stream','event')
            line=f'release {words[1]} {handle(words[1],words[2])}'
        elif line.startswith('async mapped create ') or line.startswith('async mapped release '):
            x=line.split();x[3]=str(handle('mapped',x[3]));line=' '.join(x)
        return re.sub(r'raw: (\d+)',lambda m:'raw: '+str(handle('buffer',m[1])),line)
    out=[]; pending={}; capture_counts={}
    for i,line in enumerate(lines):
        if i in omit: continue
        words=line.split();word=words[0]
        if word=='resolve' or line=='release': continue
        if word=='configure':
            name=functions[words[1]]
            if name.startswith(DIRECT):
                assert int(words[2]) in (8192,16384,20480)
                continue
            out.append(f'configure {name} {words[2]}');continue
        if word=='capture': capture_counts[words[1]]=0
        if word=='captured':
            words[3]=str(capture_counts.pop(words[1]));line=' '.join(words)
        if word=='launch' and words[1] in capture_counts: capture_counts[words[1]]+=1
        if word=='launch' and re.search(r'_mxfp8_(stream|direct|ticket|prefetch)_n', words[2]):
            symbol=words[2]; sh=shape(symbol); h=args(line)
            scalars=list(map(int,re.findall(r'ScalarU32\((\d+)\)',line)))
            offset=list(map(int,re.findall(r'ScalarU64\((\d+)\)',line)))
            is_packed='_stream_' in symbol or '_direct_' in symbol
            rows=scalars[0] if is_packed or '_initialize_' not in symbol else scalars[-1]
            win=(*sh,rows) in WINS
            if '_initialize_' in symbol and win:
                assert words[1] not in pending
                pending[words[1]]=(h,offset)
                capture_counts[words[1]]-=1
                continue
            if is_packed:
                assert len(h)==12 and h[9][0] in states
                owner_shape,owner_state,workspace_bytes=states[h[9][0]]
                assert owner_shape==sh and h[9]==owner_state
                h[2]=packed[h[2]]
                if h[4] is not None: h[4]=packed[h[4]]
                if h[8] is not None:
                    assert h[8][1]==0 and 0<h[8][2]<=workspace_bytes
                    owner=workspaces.setdefault(h[8][0],h[9][0])
                    assert owner==h[9][0] and h[8][0] not in private
                if '_direct_' in symbol:
                    assert win and h[8] is None and scalars[1]==0
                    nw,kw=(4,1) if '_n32k1_' in symbol else (1,4)
                    expected={'grid_x':(sh[0]+nw*8-1)//(nw*8),'grid_y':(max(rows,1)+15)//16,'grid_z':sh[2]*(2 if sh[3] else 1),'block_x':nw*kw*32,'shared_memory_bytes':0 if kw==1 else ((sh[1]+127)//128)*16*nw*8*4,'flags':1}
                    for field,value in expected.items(): assert f'{field}: {value},' in line
                    direct_calls[(*sh,rows)]+=1
                operands=h[:8]+h[10:]
                plan=None if win else line.split('LaunchConfigV1 ',1)[1].split(' [',1)[0]
                out.append(('fold',handle('stream',words[1]),sh,rows,tuple(map(span,operands)),tuple(offset),plan))
                continue
            if win:
                assert len(h)==10 and words[1] in pending, (symbol,rows,len(h),pending)
                adm,off=pending.pop(words[1])
                assert adm[:2]==h[:2] and adm[2]==h[8] and adm[3]==h[9]
                operands=h[:8]+[h[9],adm[4]]
                out.append(('fold',handle('stream',words[1]),sh,rows,tuple(map(span,operands)),tuple(off),None))
                continue
        out.append(text(line))
    assert not pending and not capture_counts
    return out,dict(initialized=initialized,allocations=removed_allocations,direct_calls={str(k):v for k,v in direct_calls.items()})


def audit(a,b):
    old=Counter(l.split()[2] for l in a if l.startswith('resolve '));new=Counter(l.split()[2] for l in b if l.startswith('resolve '))
    assert not old-new
    extras=new-old
    assert all(n.startswith(DIRECT) or n=='chop_mxfp8_direct_publish_live_v1' for n in extras)
    left,old_meta=normalize(a);right,new_meta=normalize(b)
    assert left==right,next(((i,x,y) for i,(x,y) in enumerate(zip(left,right)) if x!=y),('length',len(left),len(right)))
    old_init={tuple((tuple(x[0]),tuple(x[1]),tuple(x[2]) if x[2] else None)):x[3] for x in old_meta['initialized']}
    new_init={tuple((tuple(x[0]),tuple(x[1]),tuple(x[2]) if x[2] else None)):x[3] for x in new_meta['initialized']}
    assert old_init.keys() <= new_init.keys()
    assert all(k[0]==(1024,4096,8,False) for k in new_init.keys()-old_init.keys())
    resized={(2304,5120,1,True):3328,(5120,2048,1,False):3584,
             (16384,1280,1,False):3072,(25600,6144,1,False):13824}
    for key,size in old_init.items(): assert new_init[key]==resized.get(key[0],size)
    ranks=sum(extras.values())//190
    assert sum(extras.values())==ranks*190
    assert b.count('release')-a.count('release')==sum(extras.values())+ranks+len(new_meta['allocations'])-len(old_meta['allocations'])
    return dict(retained_records=len(left),added_symbols=sum(extras.values()),direct_calls=new_meta['direct_calls'],old_private_allocations=len(old_meta['allocations']),new_private_allocations=len(new_meta['allocations']),bindings_equal=True)


def main(root):
    base=root/'base'; head=root/'head'; names=sorted(p.name for p in base.glob('*.journal'))
    assert len(names)==8 and names==sorted(p.name for p in head.glob('*.journal'))
    report={}
    for name in names:
        a=(base/name).read_text().splitlines();b=(head/name).read_text().splitlines()
        report[name]=audit(a,b)
    controls=0
    a=(base/'deepseek41-flash-tp4.native.journal').read_text().splitlines()
    b=(head/'deepseek41-flash-tp4.native.journal').read_text().splitlines()
    at=next(i for i,l in enumerate(b) if l.startswith('launch ') and DIRECT in l and '_prepare_v1' not in l)
    for field in ('grid_x','grid_y','grid_z','block_x','shared_memory_bytes','flags'):
        changed=b.copy();changed[at]=re.sub(field+r': (\d+)',lambda m:field+': '+str(int(m[1])+1),changed[at],count=1)
        try: audit(a,changed)
        except AssertionError: controls+=1
        else: raise AssertionError('launch geometry control escaped: '+field)
    for operand in (0,1,2,3,6,9,10,11):
        changed=b.copy();m=list(ARG.finditer(changed[at]))[operand]
        start,end=m.span();value=m[0].replace('offset: '+m[2]+',','offset: '+str(int(m[2])+16)+',')
        changed[at]=changed[at][:start]+value+changed[at][end:]
        try: audit(a,changed)
        except (AssertionError,KeyError): controls+=1
        else: raise AssertionError('operand control escaped: '+str(operand))
    at=next(i for i,l in enumerate(b) if l.startswith('launch ') and '_prepare_v1' in l and '_mxfp8_' in l)
    changed=b.copy();changed[at]=re.sub(r'ScalarU32\((\d+)\)',lambda m:'ScalarU32('+str(int(m[1])-1)+')',changed[at])
    try: audit(a,changed)
    except AssertionError: controls+=1
    else: raise AssertionError('initialization bound control escaped')
    result=dict(passed=True,journals=report,negative_controls=controls)
    (root/'journal-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main(Path(sys.argv[1]))

"""Prove that verify4 head grouping adds only its catalog and selected consumers."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import audit_selected_cache as cache
from audit_decode_attention import read,digest

CUBIN='latent_cache_groups_deepseek41_sm120'


def locks():
    root=Path(__file__).resolve().parents[2]/'bundle/kernels.lock.d'
    grouped,=json.loads((root/'latent_cache_groups.lock').read_text())
    native,=json.loads((root/'latent_decode_cache.lock').read_text())
    symbols=[entry['symbol'] for entry in grouped['entries']]
    assert len(symbols)==24 and grouped['name']==CUBIN
    return grouped,[entry['symbol'] for entry in native['entries']],symbols


def canonical_catalog(lines,meta):
    """Keep mapped operations ordered and normalize only acquired resource IDs."""
    mapped={};live=set();kept=[]
    prefix='state audited_mapped '
    for line in lines:
        assert not line.startswith(prefix),'reserved audit encoding in input'
        fields=line.split()
        if fields[0]=='async':
            assert len(fields)==4 and fields[1]=='mapped'
            operation,raw=fields[2],int(fields[3])
            if operation=='create':
                assert raw not in mapped,'duplicate mapped resource'
                mapped[raw]=len(mapped)+1;live.add(raw)
            else:
                assert operation=='release' and raw in live,'unmatched mapped release'
                live.remove(raw)
            line=prefix+operation+' '+str(mapped[raw])
        elif fields[0]=='mapped':
            assert len(fields)==4 and fields[1]=='gather' and fields[2].isdigit()
            assert live,'mapped gather without a live resource'
            # Rank and tensor name are data, so retain both byte for byte.
            line=prefix+line
        kept.append(line)
    assert not live,'mapped resource leak'
    return cache.canonical(kept,meta)


def catalog(base,head,bm,hm,symbols):
    functions={line.split()[1]:line.split()[2] for line in head if line.startswith('resolve ')}
    extra=[line for line in head if line.startswith('resolve ') and line.split()[2] in symbols]
    assert [line.split()[2] for line in extra]==symbols*4
    assert Counter(hm['modules'].values())-Counter(bm['modules'].values())==Counter({CUBIN:4})
    assert Counter(bm['modules'].values())-Counter(hm['modules'].values())==Counter()
    assert len(bm['buffers'])==len(hm['buffers'])
    assert [bm['ranks'][str(s)] for s in bm['buffers']]==[hm['ranks'][str(s)] for s in hm['buffers']]
    winners={s for s in symbols if s.endswith('_h8_profile0_v1')}
    configured=Counter();kept=[]
    for line in head:
        fields=line.split()
        if fields[0]=='resolve' and fields[2] in symbols:continue
        if fields[0]=='configure' and functions[fields[1]] in symbols:
            symbol=functions[fields[1]]
            assert symbol in winners and fields[2:]==['73984']
            configured[symbol]+=1;continue
        if fields[0]=='launch':assert fields[2] not in symbols,'lifecycle unexpectedly uses grouped verification'
        kept.append(line)
    assert configured==Counter({s:4 for s in winners})
    releases=4*(len(symbols)+1)
    assert kept[-releases:]==['release']*releases
    kept=kept[:-releases]
    assert len(base)==len(kept)
    for i,(old,new) in enumerate(zip(canonical_catalog(base,bm),canonical_catalog(kept,hm))):
        assert old==new,('retained lifecycle operation changed',i,old,new)
    return dict(old=digest(base),new=digest(head),added_resolves=96,added_configurations=12,
                added_releases=releases,added_allocations=0,retained_operations=len(base))


def paired(base,head,bm,hm,native,grouped):
    assert bm==hm,'paired catalog or allocation differs'
    changed=Counter();normalized=[]
    for line in head:
        fields=line.split()
        if fields[0]=='launch' and fields[2] in grouped:
            symbol=fields[2]
            assert '_verify' in symbol and symbol.endswith('_cached_h8_profile0_v1')
            cache.shape(line,(4,2,8),73984)
            old=symbol.removesuffix('_cached_h8_profile0_v1')+'_cached_profile0_v1'
            assert old in native
            line=line.replace(symbol,old,1).replace('grid_y: 2,','grid_y: 16,',1)
            changed[symbol]+=1
        normalized.append(line)
    expected={s:(8 if '_window_' in s else 72 if '_p32_' in s else 80)
              for s in grouped if s.endswith('_cached_h8_profile0_v1')}
    assert changed==Counter(expected),changed
    result=cache.audit(base,normalized,bm,hm,native,False)
    assert result['added_producers']==800
    return dict(grouped_consumers=dict(changed),grouped_rows=4,heads=8,**result)



def arena(base,head,bm,hm):
    """Prove the earlier capacity-independent arena allocations separately."""
    data,table=75632640,65536
    assert Counter(bm['modules'].values())==Counter(hm['modules'].values())
    allocations=[i for i,line in enumerate(head) if line.startswith('allocate ')]
    removed=set();slots=set();ranks=[]
    for index,at in enumerate(allocations):
        if head[at]!=f'allocate {data} 256':continue
        assert at+1<head.index('serving kv')
        assert head[at+1]==f'allocate {table} 256' and allocations[index+1]==at+1
        a,b=hm['buffers'][index:index+2]
        rank,size=hm['ranks'][str(a)]
        assert size==data and hm['ranks'][str(b)]==[rank,table]
        removed.update((at,at+1));slots.update((a,b));ranks.append(rank)
    assert ranks==[0,1,2,3] and len(removed)==len(slots)==8
    kept=[line for i,line in enumerate(head) if i not in removed]
    assert kept[-8:]==['release']*8
    kept=kept[:-8]
    assert len(kept)==len(base)
    meta=dict(hm,buffers=[slot for slot in hm['buffers'] if slot not in slots])
    assert [bm['ranks'][str(slot)] for slot in bm['buffers']]==[hm['ranks'][str(slot)] for slot in meta['buffers']]
    raw={(1<<32)+slot for slot in slots}
    assert not any(int(m[1]) in raw for line in kept for m in cache.RAW.finditer(line))
    for i,(old,new) in enumerate(zip(canonical_catalog(base,bm),canonical_catalog(kept,meta))):
        assert old==new,('arena retained operation changed',i,old,new)
    return dict(added_allocations=8,added_releases=8,added_bytes_per_rank=data+table,
                retained_operations=len(base))


def unchanged_widths(pre_arena_root,base_root,head_root,symbols,original_pins=None):
    expected={6:(8040,'fb6a4f817b0b2149'),8:(8992,'ea10427872fd2fbd'),
              16:(9964,'6d88a8c8d24c043f'),32:(8384,'177a8b57184fc800')}
    if original_pins is not None:
        expected={int(rows):(int(pin[0]),str(pin[1])) for rows,pin in original_pins.items()}
        assert set(expected)=={6,8,16,32}
        assert all(n>0 and len(h)==16 and all(c in '0123456789abcdef' for c in h) for n,h in expected.values())
    result={}
    for rows,pin in expected.items():
        name=f'verify-r{rows}.journal'
        before_arena,am=read(pre_arena_root/name)
        base,bm=read(base_root/name);head,hm=read(head_root/name)
        if rows==6:
            assert before_arena==base and am==bm
            arena_proof=dict(identical=True)
        else:
            arena_proof=arena(before_arena,base,am,bm)
            mutant=base.copy()
            at=mutant.index('allocate 75632640 256')
            mutant[at]='allocate 75632641 256'
            try:arena(before_arena,mutant,am,bm)
            except (AssertionError,KeyError,ValueError):pass
            else:raise AssertionError('arena allocation mutation survived')
            arena_proof['rejected_mutations']=1
        proof=catalog(base,head,bm,hm,symbols)
        phases=[]
        for root,whole in [(pre_arena_root,before_arena),(base_root,base),(head_root,head)]:
            phase=(root/f'verify-r{rows}.phase.journal').read_text().splitlines()
            assert '\n'.join(phase)+'\n' in '\n'.join(whole)+'\n','phase absent from lifetime'
            phases.append(digest(phase))
        before,intermediate,after=phases
        assert (before['lines'],before['fnv'])==pin,'baseline verify pin differs'
        assert before['lines']==intermediate['lines']==after['lines']
        rejected=0
        for prefix,replacement in [('async mapped release ','async mapped release 0'),
                                   ('mapped gather ','mapped gather 99 changed.tensor'),
                                   ('async mapped create ','async mapped create 0')]:
            mutant=head.copy()
            at=next(i for i,line in enumerate(mutant) if line.startswith(prefix))
            mutant[at]=replacement
            try:catalog(base,mutant,bm,hm,symbols)
            except (AssertionError,KeyError,ValueError):rejected+=1
            else:raise AssertionError('mapped operation mutation survived')
        result[str(rows)]=dict(old=before,after_arena=intermediate,new=after,arena=arena_proof,catalog=proof,rejected_mutations=rejected)
    return result

def main():
    parser=argparse.ArgumentParser()
    for key in ['base','base-media','head','paired','report']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--verify-original-pins',type=Path)
    parser.add_argument('--verify-pre-arena',type=Path)
    parser.add_argument('--verify-base',type=Path)
    parser.add_argument('--verify-head',type=Path)
    args=parser.parse_args();lock,native,symbols=locks();results={};rejects=0
    assert bool(args.verify_pre_arena)==bool(args.verify_base)==bool(args.verify_head)
    assert not args.verify_original_pins or args.verify_base
    for name in ['deepseek41-flash-tp'+str(rank)+'.'+layout+'.journal' for rank in [1,2,4] for layout in ['native','bf16']]+['deepseek41-flash-tp2.media.journal','deepseek41-flash-tp4.media.journal']:
        old_root=args.base_media if '.media.' in name else args.base
        base,bm=read(old_root/name);head,hm=read(args.head/name)
        if name in ['deepseek41-flash-tp4.native.journal','deepseek41-flash-tp4.media.journal']:
            results[name]=catalog(base,head,bm,hm,symbols)
            for mode in ['allocation','configure','merge']:
                changed=head.copy()
                if mode=='allocation':
                    at=next(i for i,line in enumerate(changed) if line.startswith('allocate '));parts=changed[at].split();parts[1]=str(int(parts[1])+1);changed[at]=' '.join(parts)
                elif mode=='configure':
                    funcs={line.split()[1]:line.split()[2] for line in changed if line.startswith('resolve ')}
                    at=next(i for i,line in enumerate(changed) if line.startswith('configure ') and funcs[line.split()[1]] in symbols)
                    changed[at]=changed[at].replace('73984','73983')
                else:
                    at=next(i for i,line in enumerate(changed) if line.startswith('launch ') and 'chop_latent_sources_' in line and '_merge_' in line)
                    changed[at]=changed[at].replace('grid_z: 1,','grid_z: 2,',1)
                try:catalog(base,changed,bm,hm,symbols)
                except (AssertionError,KeyError,ValueError):rejects+=1
                else:raise AssertionError(mode+' lifecycle mutation survived')
        else:
            assert base==head and bm==hm,name
            results[name]=dict(identical=True,old=digest(base),new=digest(head))
    stem='deepseek41-flash-tp4.Native.selected-cache.'
    base,bm=read(args.paired/(stem+'reference.journal'));head,hm=read(args.paired/(stem+'serving.journal'))
    pair=paired(base,head,bm,hm,native,symbols)
    at=next(i for i,line in enumerate(head) if line.startswith('launch ') and '_cached_h8_' in line)
    for old,new in [('grid_y: 2,','grid_y: 4,'),('_h8_','_h4_'),('grid_z: 8,','grid_z: 4,'),('grid_x: 4,','grid_x: 6,'),('73984','73983'),('NullDevicePointer','ScalarU32(0)')]:
        changed=head.copy();changed[at]=changed[at].replace(old,new,1);assert changed!=head
        try:paired(base,changed,bm,hm,native,symbols)
        except (AssertionError,KeyError,ValueError):rejects+=1
        else:raise AssertionError('grouped consumer mutation survived')
    assert rejects==12
    report=dict(passed=True,lifecycles=results,paired=pair,rejected_mutations=rejects,cubin_sha256=lock['cubin_sha256'])
    if args.verify_base:
        report['unchanged_verify_widths']=unchanged_widths(args.verify_pre_arena,args.verify_base,args.verify_head,symbols,
            json.loads(args.verify_original_pins.read_text()) if args.verify_original_pins else None)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    widths=report.get('unchanged_verify_widths',{})
    print(f'VERIFY_GROUPED_BINDING_AUDIT_OK lifecycles=8 grouped_consumers=160 verify_widths={len(widths)} rejected_mutations={rejects+sum(w["rejected_mutations"]+w["arena"].get("rejected_mutations",0) for w in widths.values())}')

if __name__=='__main__':main()

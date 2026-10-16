"""Authenticated host-lifetime mappings for served journal comparisons."""
import collections
import json
from pathlib import Path
from engine import POLICY_FIELDS,Unproven,require,RAW,transform,strip_additions
from common import need
from provenance import digest


def load(snapshot,index,bodies,origins):
    from pins import read_index
    from auxiliary import normalize
    payload=read_index(index)
    root=Path(index).resolve().parent
    need(payload['revision']==payload['archive_commit']==snapshot['commit'],'auxiliary archive commit mismatch')
    need(set(payload['journal_sha256'])==set(snapshot['pins']),'auxiliary index has incomplete pinned coverage')
    for name,sha in payload['journal_sha256'].items():
        need(origins[name]['sha256']==sha,f'{name}: body differs from helper output')
    for relative,sha in payload['auxiliary_files'].items():
        p=Path(relative)
        need(not p.is_absolute() and '..' not in p.parts and p.parts[0]=='aux','unsafe auxiliary file')
        path=root/p
        need(path.is_file() and not path.is_symlink() and digest(path)==sha,f'auxiliary capture changed: {relative}')
    mpath=Path(payload['auxiliary']['path'])
    if not mpath.is_absolute():
        mpath=root/mpath
    need(mpath.resolve()==root/'aux/manifest.json' and digest(mpath)==payload['auxiliary']['sha256'],'auxiliary manifest mismatch')
    manifest=normalize(json.loads(mpath.read_text()))
    need(manifest['revision']==snapshot['commit'],'auxiliary manifest revision mismatch')
    need(set(manifest['bindings'])|set(manifest['text'])|set(manifest['unrecoverable'])==set(snapshot['pins']),
         'auxiliary fixture coverage mismatch')
    hosts={}
    for name,record in manifest['hosts'].items():
        relative=record['path']
        need(relative in payload['auxiliary_files'] and payload['auxiliary_files'][relative]==record['sha256'],'unsealed host trace')
        events=[json.loads(line) for line in (root/relative).read_text().splitlines()]
        for row in events:
            if row[0]=='module' and len(row)==6:
                relative='aux/'+row[5]
                parts=Path(row[5]).parts
                need(not Path(row[5]).is_absolute() and '..' not in parts and len(parts)<=2 and relative in payload['auxiliary_files'],
                     'module payload is not helper-sealed')
                need((root/relative).stat().st_size==row[4],'module byte length differs from captured payload')
                row[5]=payload['auxiliary_files'][relative]
        notes=[e[1] for e in events if e[0]=='note']
        hosts[name]=dict(events=events,notes=notes,test=record['test'])
    for name,binding in manifest['bindings'].items():
        host=hosts[binding['host']]
        need(binding['test']==host['test'],'host/test binding mismatch')
        records=binding['records']
        need(type(records)is int and 0<records<=len(host['events']),'invalid host prefix')
        note_count=sum(e[0]=='note' for e in host['events'][:records])
        positions=binding['positions']
        need(len(positions)==len(bodies[name]) and positions==sorted(set(positions)) and positions[-1]<note_count,
             f'{name}: invalid trace witness positions')
        need(all(host['notes'][i]==line for i,line in zip(positions,bodies[name])),f'{name}: pinned lines are not from this host')
        create=binding['create']
        relative=create['path']
        need(relative in payload['auxiliary_files'] and payload['auxiliary_files'][relative]==create['sha256'],'unsealed create capture')
        lines=json.loads((root/relative).read_text())
        need(lines==host['notes'][create['start_note']:create['end_note']],'create capture differs from device history')
        need(0<=create['start_event']<=create['end_event']<=records,'invalid create event range')
        if create['kind']=='instance-create':
            marker=['created',create['start_event'],create['start_note'],create['end_event'],create['end_note']]
            need(marker in host['events'][:records],'create boundary was not observed by helper')
        if create['kind']=='whole-lifetime':
            need(not lines and create['end_event']==0 and positions==list(range(note_count))
                 and not any(e[0]=='created' for e in host['events'][:records]),
                 f'{name}: a whole-lifetime binding must cover every note of a device that ran no instance create')
        for other in binding.get('other_candidates',[]):
            # Another device that recorded the same lines: claimed by other
            # journals there, an identical copy, or not covered.
            if 'claimed_by' in other:
                need(all(manifest['bindings'].get(n,{}).get('host')==other['path'] for n in other['claimed_by']),
                     f'{name}: candidate device {other["path"]} is not held by {other["claimed_by"]}')
                continue
            need(other['path'] in hosts and hosts[other['path']]['test']==binding['test'],f'{name}: unsealed candidate device')
            prefix=hosts[other['path']]['events'][:other['records']]
            other_notes=[e[1] for e in prefix if e[0]=='note']
            from auxiliary import subsequence
            need(subsequence(bodies[name],other_notes) is not None,f'{name}: candidate device {other["path"]} did not record its lines')
            need(other['identical']==(prefix==host['events'][:records]),f'{name}: candidate device {other["path"]} identity misstated')
        binding['create_lines']=lines
    return dict(payload=payload,manifest=manifest,hosts=hosts)


def inventory(events):
    issued=[]
    typed={}
    allocations={}
    modules={}
    resolves={}
    streams={}
    for row in events:
        kind=row[0]
        if kind=='id':
            issued.append(row[1])
        elif kind=='allocation':
            raw=row[1]
            slot=raw&0xffffffff
            require(raw>>32==1,'allocated handle generation differs from 1')
            require(slot not in typed,'issued ID has two resource roles')
            typed[slot]=kind
            allocations[slot]=row
        elif kind in ('module','resolve','stream'):
            slot=row[1]
            require(slot not in typed,'issued ID has two resource roles')
            typed[slot]=kind
            {'module':modules,'resolve':resolves,'stream':streams}[kind][slot]=row
        else:
            require(kind in ('note','created'),f'unknown helper trace event {kind}')
    require(issued==list(range(1,len(issued)+1)),'device issue sequence is incomplete or reused')
    require(set(typed)<=set(issued),'resource was not issued by this device')
    # A resolve names a module this device loaded, or a module handle the
    # device never loaded (a host-provided handle, as encoder tests pass):
    # that handle is never an ID of another role here, stays literal, and its
    # payload is not authenticated.
    unloaded=sorted(i for i,row in resolves.items() if row[2] not in modules)
    for i in unloaded:
        require(typed.get(resolves[i][2]) in (None,'resolve'),'resolve names an issued non-module resource as its module')
    return dict(issued=issued,typed=typed,allocations=allocations,modules=modules,resolves=resolves,streams=streams,
                unloaded=unloaded)


def validate_rule(rule,layout=False):
    """The shape of one fixture's auxiliary declarations, before any
    evidence is read. Under a layout declaration the rule may also carry the
    layout witness (layout.WITNESS_FIELDS) and nothing that rewrites."""
    from layout import WITNESS_FIELDS
    need(isinstance(rule,dict) and {'added_resolves','added_modules','create_policy'}<=set(rule)<=
         {'added_resolves','added_modules','create_policy','removed_resolves','allocation_resizes','allocation_bindings',
          'allocation_layouts','history_policy'}|(WITNESS_FIELDS if layout else set()),'invalid auxiliary rule fields')
    if layout:
        need(not rule['create_policy'] and not rule.get('history_policy') and not rule.get('allocation_resizes')
             and not rule.get('allocation_bindings') and not rule.get('allocation_layouts'),
             'a layout fixture declares its changes in the layout declaration, not in create, history or allocation policies')
        for key in WITNESS_FIELDS:
            need(isinstance(rule.get(key,[]),list),f'{key} must be a list')
    for key in ('create_policy','history_policy'):
        policy=rule.get(key,{})
        need(isinstance(policy,dict) and set(policy)<=POLICY_FIELDS|({'graph_nodes','graph_pieces'} if key=='history_policy' else set()) and
             all(isinstance(v,list) for v in policy.values()),f'invalid auxiliary {key}')
    for key in ('added_resolves','added_modules','removed_resolves'):
        if key not in rule:
            continue
        need(isinstance(rule[key],list),'auxiliary additions must be lists')
        for d in rule[key]:
            need(isinstance(d,dict),'auxiliary addition must be an object')
            need(type(d.get('count')) is int and d['count']>0 and isinstance(d.get('ids'),list) and
                 all(type(i) is int and i>0 for i in d['ids']) and len(d['ids'])==d['count'] and
                 len(set(d['ids']))==len(d['ids']),'invalid auxiliary addition count or IDs')
    resizes=rule.get('allocation_resizes',[])
    need(isinstance(resizes,list) and all(isinstance(d,dict) and set(d)=={'base_raw','head_raw','rank','from','to','alignment'}
                                          for d in resizes),'invalid allocation resize declarations')
    for key,fields in (('allocation_bindings',{'base_raw','head_raw','rank','base_bytes','head_bytes','stream_raw'}),
                       ('allocation_layouts',{'planes','from_stride','to_stride','from_tail','to_tail'})):
        value=rule.get(key,{})
        need(isinstance(value,dict) and all(isinstance(d,dict) and set(d)==fields for d in value.values()),f'invalid {key}')


EMPTY_RULE={'added_resolves':[],'added_modules':[],'create_policy':{}}


def validate_declared_only(rule):
    """A fixture holding only declared added/removed journals compares
    nothing, so it may declare nothing."""
    validate_rule(rule)
    need({k:v for k,v in rule.items() if k!='history_policy' or v}==EMPTY_RULE,
         'a fixture of only declared added/removed journals must carry the empty auxiliary rule')


def derive(base_events,head_events,rule,replacements=(),layout=None):
    """The fixture's typed identity map. With `layout` (the parsed
    declaration and its authenticated modules) the rule's layout witness
    joins the declared additions and removals, and the result gains the
    checked layout.Witness as a seventh element."""
    a,b=inventory(base_events),inventory(head_events)
    validate_rule(rule,layout is not None)
    inserted=set()
    resolve_ids=set()
    module_ids=set()
    for declaration in rule['added_resolves']:
        need(set(declaration)=={'symbol','ids','count'},'invalid auxiliary resolve rule')
        actual=[i for i,r in b['resolves'].items() if r[3]==declaration['symbol']]
        require(actual==declaration['ids'] and len(actual)==declaration['count'],'exact auxiliary resolve IDs/count differ')
        require(not any(r[3]==declaration['symbol'] for r in a['resolves'].values()),'added symbol was already in base')
        resolve_ids.update(actual)
    for declaration in rule['added_modules']:
        need(set(declaration)=={'name','fnv1a','bytes','ids','count'},'invalid auxiliary module rule')
        ids=declaration['ids']
        require(len(ids)==declaration['count'] and len(ids)==len(set(ids)),'auxiliary module count differs')
        for i in ids:
            require(i in b['modules'] and b['modules'][i][2:5]==[declaration['name'],declaration['fnv1a'],declaration['bytes']],
                    'added module identity or payload differs')
        module_ids.update(ids)
    require(len(resolve_ids)+len(module_ids)==sum(d['count'] for d in rule['added_resolves']+rule['added_modules']),
            'added auxiliary IDs overlap')
    for i in resolve_ids:
        require(b['resolves'][i][2] in module_ids,'new resolve is not owned by its declared new module')
    for i in module_ids:
        require(any(r[2]==i for j,r in b['resolves'].items() if j in resolve_ids),'added module owns no declared resolves')
        require(all(j in resolve_ids for j,r in b['resolves'].items() if r[2]==i),'added module owns undeclared resolves')
    inserted=resolve_ids|module_ids
    witness=None
    if layout is not None:
        from layout import Witness
        witness=Witness(layout[0],layout[1],rule,a,b)
        inserted|=set(witness.added)
    deleted=set()
    replacement_by_name={r['base']['name']:r['declaration'] for r in replacements}
    for d in rule.get('removed_resolves',[]):
        need(set(d)=={'symbol','ids','count'},'invalid removed resolve declaration')
        actual=[i for i,r in a['resolves'].items() if r[3]==d['symbol']]
        require(actual==d['ids'] and len(actual)==d['count'],'removed resolve IDs/count differ')
        for i in actual:
            if witness is not None and a['resolves'][i][2] in witness.removed_ids:
                continue
            module=a['modules'][a['resolves'][i][2]][2]
            require(d['symbol'] in replacement_by_name.get(module,{}).get('removed_symbols',[]),
                    'removed resolve is not declared by a locked module replacement')
        need(not(deleted&set(actual)),'overlapping removed resolves')
        deleted.update(actual)
    if witness is not None:
        need(not(deleted&witness.removed_ids),'a removed module is also a removed resolve')
        deleted|=witness.removed_ids
    retained=[i for i in b['issued'] if i not in inserted]
    source=[i for i in a['issued'] if i not in deleted]
    require(len(source)==len(retained),'issued ID count differs beyond exact module/resolve additions/removals')
    ids=dict(zip(source,retained))
    require(len(set(ids.values()))==len(ids),'ID mapping is not bijective')
    from replacements import module_plans
    plans,replaced,permutation=module_plans(a,b,ids,replacements)
    sizes={}
    for d in rule.get('allocation_resizes',[]):
        need(set(d)=={'base_raw','head_raw','rank','from','to','alignment'},'invalid allocation resize declaration')
        old=d['base_raw']&0xffffffff
        need(old not in sizes and type(d['from'])is int and type(d['to'])is int and d['from']>0 and d['to']>0 and d['from']!=d['to'],
             'invalid or overlapping allocation resize')
        require(old in a['allocations'] and ids.get(old)==(d['head_raw']&0xffffffff),'resize is not bound to paired allocations')
        ar=a['allocations'][old]
        br=b['allocations'][ids[old]]
        require((ar[1],br[1],ar[2],br[2],ar[3],br[3],ar[4],br[4])==
                (d['base_raw'],d['head_raw'],d['rank'],d['rank'],d['from'],d['to'],d['alignment'],d['alignment']),
                'allocation resize identity/rank/size/alignment differs')
        sizes[old]=d['to']
    if witness is not None:
        for old,w in witness.resized.items():
            require(ids.get(old)==w['head_slot'],'a resized allocation is not bound to its paired head allocation')
            sizes[old]=w['to']
    # An allocation on a stream with no recorded creation keeps that stream,
    # which may not be the image of a created stream (history.Mapper).
    stream_images={ids[s] for s in a['streams']}
    def mapped(row):
        kind=row[0]
        x=list(row)
        if kind=='allocation':
            x[1]=(row[1]&~0xffffffff)|ids[row[1]&0xffffffff]
            x[3]=sizes.get(row[1]&0xffffffff,row[3])
            slot=row[5]&0xffffffff
            require(slot in a['streams'] or slot not in stream_images,
                    f'allocation {row[1]} is on stream {slot}, which has no recorded creation and is the image of a created stream')
            x[5]=(row[5]&~0xffffffff)|(ids[slot] if slot in a['streams'] else slot)
        else:
            x[1]=ids[row[1]]
            if kind=='resolve':
                x[2]=ids[row[2]] if row[2] in a['modules'] else row[2]
        return x
    for old,new in ids.items():
        require(a['typed'].get(old)==b['typed'].get(new),'issued resource role changed')
        kind=a['typed'].get(old)
        if kind=='allocation' and old!=new:
            # Under a layout declaration an unranked allocation is keyed by
            # its synthetic stream (layout.rank_key), kept fixed and compared.
            row=a['allocations'][old]
            require(row[2] is not None or (witness is not None and row[5]&0xffffffff not in a['streams']),
                    'allocation rank is unrecorded: register the synthetic stream rank before allocation; nonidentity mapping refused')
        if kind=='module' and old in replaced:
            continue
        if kind=='resolve' and a['resolves'][old][2] in replaced:
            require(permutation.get(old) in b['resolves'],'replacement lost a resolved function')
            continue
        if kind:
            key={'allocation':'allocations','module':'modules','resolve':'resolves','stream':'streams'}[kind]
            require(mapped(a[key][old])==b[key][new],f'{kind} identity, rank, size or payload changed at ID {old}')
    buffers={row[1]:b['allocations'][ids[old]][1] for old,row in a['allocations'].items()}
    streams={old:ids[old] for old in a['streams']}
    functions={old:ids[old] for old in (set(a['resolves'])|set(a['modules']))-deleted}
    functions.update(permutation)
    if witness is not None:
        return buffers,functions,streams,ids,a,b,witness
    return buffers,functions,streams,ids,a,b


def prove(base,head,fixture,policies,base_aux,head_aux,rule,replacements=(),compare_history=True,layout=None):
    """Prove one fixture: its create, every pinned journal, the rest of its
    device history through the last pinned checkpoint, and the cross-owner
    order of all of them. A fixture-level failure fails every journal.
    `compare_history=False` exists only for unit controls of the pinned
    comparison; proof.py always compares the history. With `layout` (the
    parsed declaration and its authenticated modules) every owner is
    compared under the declaration (layout.Device) instead of policies."""
    import history
    names=fixture['journals']
    declared=fixture.get('declared_journals',{})
    results={}
    counts=collections.Counter()
    try:
        for name in names:
            for side,aux in [('base',base_aux),('head',head_aux)]:
                error=aux['manifest']['unrecoverable'].get(name)
                require(error is None,f'{side}/{name}: {error}')
        ab=[base_aux['manifest']['bindings'][n] for n in names]
        hb=[head_aux['manifest']['bindings'][n] for n in names]
        require(len({b['host'] for b in ab})==len({b['host'] for b in hb})==1,'source fixture spans multiple device lifetimes')
        require({b['test'] for b in ab}=={b['test'] for b in hb},'paired fixture test origins differ')
        # Declared added/removed journals keep their own operations out of the history.
        extra={}
        for side,aux,bindings in [('base',base_aux,ab),('head',head_aux,hb)]:
            extra[side]={}
            for name in declared.get(side,[]):
                require(name not in aux['manifest']['unrecoverable'],f'{side}/{name}: declared journal has no recovered binding')
                binding=aux['manifest']['bindings'][name]
                require(binding['host']==bindings[0]['host'],f'{side}/{name}: declared journal is on another device')
                extra[side][name]=binding
        ah,bh=base_aux['hosts'][ab[0]['host']],head_aux['hosts'][hb[0]['host']]
        for side,aux,bindings in [('base',base_aux,ab),('head',head_aux,hb)]:
            host=bindings[0]['host']
            mine=set(names)|set(extra[side])
            others=sorted(n for n,v in aux['manifest']['bindings'].items() if v['host']==host and n not in mine)
            require(not others,f'{side}: device {host} also holds pinned journals of another fixture: {others[:3]}')
        ae=ah['events'][:max(b['records'] for b in ab+list(extra['base'].values()))]
        be=bh['events'][:max(b['records'] for b in hb+list(extra['head'].values()))]
        derived=derive(ae,be,rule,replacements,layout)
        buffers,functions,streams,ids,ai,bi=derived[:6]
        pairs={(a['create']['path'],b['create']['path']):(a,b) for a,b in zip(ab,hb)}
        require(len(pairs)==1,'fixture has multiple create generations; an explicit lifetime boundary is required')
        a,b=next(iter(pairs.values()))
        notes_a=[e[1] for e in ae if e[0]=='note']
        notes_b=[e[1] for e in be if e[0]=='note']
        device=None
        if layout is not None:
            import layout as declared
            witness=derived[6]
            witness.check_stream_and_lifetime(bi,be,b['create'])
            device=declared.Device(layout[0],witness,ae,be,ai,bi,b['create'])
        owner_a=history.owners(notes_a,dict(zip(names,ab),**extra['base']),a['create'])
        owner_b=history.owners(notes_b,dict(zip(names,hb),**extra['head']),b['create'])
        hist_a=[i for i,o in enumerate(owner_a) if o=='history']
        hist_b=[i for i,o in enumerate(owner_b) if o=='history']
        # Fixed handles: every raw a base record names that is not a base
        # allocation (a host buffer such as the weights arena), in any owner.
        # No allocation may map onto one, and no declared added allocation
        # may be one.
        fixed={int(v) for n in names for line in base[n] for v in RAW.findall(line)}
        fixed|={int(v) for line in notes_a for v in RAW.findall(line)}
        fixed-=set(buffers)
        require(not(set(buffers.values())&fixed),'allocation mapping redirects a fixed resource')
        if device is not None:
            added_raws={bi['allocations'][slot][1] for slot in witness.added}
            require(not(added_raws&fixed),'a declared added allocation has the handle of a fixed resource')
        from replacements import module_plans,rewrite_resolves
        from allocation_args import bindings,validate_spans
        plans,_,_=module_plans(ai,bi,ids,replacements)
        context=bindings(rule,ai,bi,buffers)

        def compare(base_lines,head_lines,base_positions,head_positions,policy,counter,mapper):
            """One owner's expected (mapped) and observed lines with their
            provenance (trace into base_lines, kept into head_lines)."""
            trace=[]
            kept=[]
            if device is None:
                expected=transform(base_lines,policy,counter,context,trace)
                expected,trace=rewrite_resolves(expected,plans,counter,trace)
                observed=strip_additions(head_lines,policy,counter,kept)
                return [mapper(l) for l in expected],observed,trace,kept
            expected,trace,originals,observed,kept,changes=declared.owner_lines(device,notes_b,base_positions,head_positions,base_lines)
            validate_spans(expected,context,originals)
            counter.update(changes)
            return [declared.resolve_placeholders(mapper(l)) for l in expected],observed,trace,kept

        # One typed mapping for create, pinned and history operations: every
        # identity of the line's form (streams, functions, events of record
        # and wait, graphs of capture, replay and release) and every buffer.
        # Journal bodies are not a closed grammar (history.Mapper).
        remapped=history.Mapper(buffers,functions,streams,ids,closed=False)
        cp=rule['create_policy']
        create_counts=collections.Counter()
        expected,observed,ctrace,ckept=compare(a['create_lines'],b['create_lines'],
                                               range(a['create']['start_note'],a['create']['end_note']),
                                               range(b['create']['start_note'],b['create']['end_note']),cp,create_counts,remapped)
        if expected!=observed:
            i=next(k for k in range(max(len(expected),len(observed))) if expected[k:k+1]!=observed[k:k+1])
            first=dict(line=i+1,expected=expected[i][:1000] if i<len(expected) else None,observed=observed[i][:1000] if i<len(observed) else None)
            raise Unproven(f'own fixture create differs beyond exact declarations and typed mapping; first difference {json.dumps(first)}')
        total_a=sum(e[0]=='note' for e in ah['events'])
        total_b=sum(e[0]=='note' for e in bh['events'])
        detail=dict(identity_source='helper-auxiliary',base_host=ab[0]['host'],head_host=hb[0]['host'],
                    issued_ids=len(ids),added_ids=len(bi['issued'])-len(ai['issued']),
                    allocations=len(buffers),buffer_map={str(k):v for k,v in buffers.items()},
                    allocation_records=[dict(base_raw=r[1],head_raw=buffers[r[1]],rank=r[2],bytes=r[3],alignment=r[4],
                                             stream_raw=r[5]) for r in ai['allocations'].values()],
                    unrecorded_allocation_ranks=sum(r[2] is None for r in ai['allocations'].values()),
                    module_replacements=[dict(base_id=p['base_id'],head_id=p['head_id'],module=p['declaration']['module']) for p in plans],
                    unloaded_module_resolves=dict(base=len(ai['unloaded']),head=len(bi['unloaded'])),
                    allocation_resizes=rule.get('allocation_resizes',[]),create_changes=dict(create_counts),
                    history=dict(status='NOT EVALUATED',
                                 base_notes=len(notes_a),head_notes=len(notes_b),
                                 owners=dict(base=dict(collections.Counter(owner_a)),head=dict(collections.Counter(owner_b))),
                                 trailing_uncompared_notes=dict(base=total_a-len(notes_a),head=total_b-len(notes_b))))
        if detail['unrecorded_allocation_ranks']:
            detail['rank_evidence_note']='The fixture uses unregistered synthetic streams. Raw allocation and stream identities remain exact; a nonidentity allocation map requires registered ranks. Minimal test-side change: register each synthetic stream slot with its rank before allocation.'
    except Unproven as error:
        return {n:dict(verdict='UNPROVEN',reasons=[str(error)],counts={}) for n in names},counts,dict(identity_source='helper-auxiliary',error=str(error))
    rank={}
    history.ranks(rank,owner_b,'create',[b['create']['start_note']+i for i in range(len(b['create_lines']))],ckept,ctrace,
                  [a['create']['start_note']+i for i in range(len(a['create_lines']))])
    for name,abind,hbind in zip(names,ab,hb):
        current=collections.Counter()
        try:
            policy=policies.get(name,{})
            expected,observed,trace,kept=compare(base[name],head[name],abind['positions'],hbind['positions'],policy,current,remapped)
            differing=[i for i in range(max(len(expected),len(observed))) if expected[i:i+1]!=observed[i:i+1]]
            if differing:
                i=differing[0]
                detail.setdefault('first_difference',{})[name]=dict(line=i+1,expected=expected[i][:1000] if i<len(expected) else None,observed=observed[i][:1000] if i<len(observed) else None)
            require(not differing,f'{len(differing)} unexplained line differences; first {[i+1 for i in differing[:5]]}')
            current=collections.Counter({k:v for k,v in current.items() if v})
            results[name]=dict(verdict='IDENTICAL' if base[name]==head[name] and not current else 'PROVEN',reasons=[],counts=dict(current))
            history.ranks(rank,owner_b,name,hbind['positions'],kept,trace,abind['positions'])
        except Unproven as error:
            results[name]=dict(verdict='UNPROVEN',reasons=[str(error)],counts=dict(current))
    record=detail['history']
    problem=None
    if not compare_history:
        record['status']='NOT EVALUATED (caller disabled history comparison)'
        for name in names:
            if results[name]['verdict']!='UNPROVEN':
                counts.update(results[name]['counts'])
        return results,counts,detail
    try:
        hp=rule.get('history_policy',{})
        hcounts=collections.Counter()
        old=[notes_a[i] for i in hist_a]
        new=[notes_b[i] for i in hist_b]
        record.update(base_operations=len(old),head_operations=len(new),operation_kinds=history.kinds(new))
        expected,observed,htrace,hkept=compare(old,new,hist_a,hist_b,hp,hcounts,history.Mapper(buffers,functions,streams,ids))
        for line in observed:
            history.classify(line)
        history.check_graph_nodes(notes_a,'base')
        history.check_graph_nodes(notes_b,'head')
        expected=history.graph_node_changes(expected,observed,hp,hcounts)
        differing=[i for i in range(max(len(expected),len(observed))) if expected[i:i+1]!=observed[i:i+1]]
        record['changes']=dict(hcounts)
        if differing:
            i=differing[0]
            record['first_difference']=dict(history_line=i+1,
                base_note=hist_a[htrace[i][0]]+1 if i<len(htrace) and htrace[i] else None,
                head_note=hist_b[hkept[i]]+1 if i<len(hkept) else None,
                expected=expected[i][:1000] if i<len(expected) else None,observed=observed[i][:1000] if i<len(observed) else None)
        require(not differing,f'auxiliary history: {len(differing)} unexplained operation differences; first history lines {[i+1 for i in differing[:5]]}')
        history.ranks(rank,owner_b,'history',hist_b,hkept,htrace,hist_a)
        correspond={hist_a[source[0]]:hist_b[hkept[k]] for k,source in enumerate(htrace) if source and len(source)==1}
        divisors=(fixture.get('piece_policy') or {}).get('divisors')
        history.regrouped_pieces(notes_a,notes_b,hp,divisors,correspond,rank,hcounts)
        record['changes']=dict(hcounts)
        failed=sorted(n for n in names if results[n]['verdict']=='UNPROVEN')
        if failed:
            raise Unproven(f'cross-owner order not checked: {failed[0]} is unproven')
        history.check_order(owner_b,notes_b,rank)
        record['status']='COMPARED'
        record['ordered_operations']=len(rank)
    except Unproven as error:
        problem=str(error)
        record['status']='FAILED'
        record['error']=problem
    for name in names:
        if results[name]['verdict']!='UNPROVEN':
            counts.update(results[name]['counts'])
    if problem:
        for name in names:
            if results[name]['verdict']!='UNPROVEN':
                results[name]=dict(verdict='UNPROVEN',reasons=[f'fixture history: {problem}'],counts=results[name]['counts'])
        counts=collections.Counter()
    return results,counts,detail

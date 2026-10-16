"""Module replacements authenticated by immutable bundle locks and captured bytes."""
import collections
import hashlib
import json
import re
from pathlib import PurePosixPath
from common import git,need
from engine import require,RESOLVE,CONFIGURE,MODULE


def lock_module(repo,revision,path,name,root):
    p=PurePosixPath(path)
    need(str(p).startswith(root+'kernels.lock.d/') and p.suffix=='.lock' and '..' not in p.parts,
         'module replacement must name a bundle lock file')
    raw=git(repo,'show',revision+':'+path)
    rows=json.loads(raw)
    need(isinstance(rows,list),'bundle lock must contain a module list')
    found=[r for r in rows if r.get('name')==name]
    need(len(found)==1,f'lock must contain exactly one module {name}')
    item=found[0]
    sha=item.get('cubin_sha256')
    need(isinstance(sha,str) and re.fullmatch('[0-9a-f]{64}',sha),'invalid locked cubin SHA-256')
    symbols=[e['symbol'] for e in item['entries']]
    need(len(symbols)==len(set(symbols)),'duplicate locked module entry')
    return dict(name=name,sha256=sha,symbols=symbols,path=path,lock_sha256=hashlib.sha256(raw).hexdigest())


def authenticate(repo,base,head,declarations,root):
    """`root` is the model's bundle directory (pins.bundle_root), whose
    kernels.lock.d holds the declared locks."""
    need(isinstance(declarations,list),'module_replacements must be a list')
    result=[]
    seen=set()
    targets=set()
    for d in declarations:
        need(isinstance(d,dict) and {'module','base','head','renames','resolve_order'}<=set(d)<=
             {'module','head_module','base','head','renames','resolve_order','configures','removed_symbols'},
             'invalid module replacement fields')
        name=d['module']
        target=d.get('head_module',name)
        need(name not in seen and target not in targets,'overlapping module replacements')
        seen.add(name)
        targets.add(target)
        need(d['resolve_order'] in ('name','locked'),'module replacement requires name or locked entry order')
        for side in ('base','head'):
            need(set(d[side])=={'lock','sha256'},'module payload declaration needs lock and sha256')
        a=lock_module(repo,base,d['base']['lock'],name,root)
        b=lock_module(repo,head,d['head']['lock'],target,root)
        need(d['base']['sha256']==a['sha256'] and d['head']['sha256']==b['sha256'],
             f'{name}: declared payload digest does not match revision lock')
        renames=d['renames']
        removed=d.get('removed_symbols',[])
        need(isinstance(renames,dict) and all(isinstance(k,str) and isinstance(v,str) for k,v in renames.items()),'invalid symbol rename map')
        need(isinstance(removed,list) and len(removed)==len(set(removed)),'invalid removed module symbols')
        need(set(renames)<=set(a['symbols']) and set(removed)<=set(a['symbols']) and not(set(renames)&set(removed)),
             'rename/removal names an absent or overlapping source entry')
        expected=[renames.get(s,s) for s in a['symbols'] if s not in removed]
        need(len(expected)==len(set(expected)) and set(expected)==set(b['symbols']),
             f'{name}: locked entries differ beyond exact renames/removals')
        configs=d.get('configures',{})
        need(isinstance(configs,dict) and set(configs)<=set(a['symbols'])-set(removed),'invalid configure declaration')
        for rule in configs.values():
            need(set(rule)=={'from','to'} and all(isinstance(rule[k],list) and all(type(v)is int and v>=0 for v in rule[k]) for k in ('from','to')),
                 'configure replacement needs exact old/new byte lists')
        result.append(dict(declaration=d,base=a,head=b))
    return result


def module_plans(a,b,ids,replacements):
    plans=[]
    matched={}
    functions={}
    by_name={r['base']['name']:r for r in replacements}
    for old,row in a['modules'].items():
        if row[2] not in by_name:
            continue
        r=by_name[row[2]]
        d=r['declaration']
        new=ids[old]
        other=b['modules'].get(new)
        require(other is not None and other[2]==r['head']['name'],'replacement module ID/name differs')
        need(len(row)==len(other)==6,'module replacement requires helper-captured cubin SHA-256 evidence')
        need(row[5]==r['base']['sha256'] and other[5]==r['head']['sha256'],
                f'{row[2]}: captured module payload does not match locked SHA-256')
        ar=[v for v in a['resolves'].values() if v[2]==old]
        br=[v for v in b['resolves'].values() if v[2]==new]
        require(all(v[3] in r['base']['symbols'] for v in ar) and all(v[3] in r['head']['symbols'] for v in br),
                'resolved symbol is absent from its locked module')
        order_a=sorted(r['base']['symbols']) if d['resolve_order']=='name' else r['base']['symbols']
        order_b=sorted(r['head']['symbols']) if d['resolve_order']=='name' else r['head']['symbols']
        require([v[3] for v in ar]==[s for s in order_a if any(v[3]==s for v in ar)] and
                [v[3] for v in br]==[s for s in order_b if any(v[3]==s for v in br)],
                'replacement resolves are not in the declared name/lock order')
        kept=[v for v in ar if v[3] not in d.get('removed_symbols',[])]
        require([ids[v[1]] for v in kept]==[v[1] for v in br],
                'replacement changed the ordered function ID set')
        renamed=sorted(((d['renames'].get(v[3],v[3]),v[1]) for v in kept),key=lambda x:order_b.index(x[0]))
        require([s for s,_ in renamed]==[v[3] for v in br],'resolved entries differ beyond declared renames')
        for (_,ident),observed in zip(renamed,br):
            functions[ident]=observed[1]
        plan=dict(base_id=old,head_id=new,declaration=d,base_rows=ar,head_rows=br,head_order=order_b)
        plans.append(plan)
        matched[old]=plan
    return plans,matched,functions


def rewrite_resolves(lines,plans,counts,trace=None):
    """Reorder replaced modules' resolve runs. With a `trace` (provenance of
    each input line), returns the output and its permuted provenance."""
    owners={r[1]:p for p in plans for r in p['base_rows']}
    source=trace if trace is not None else [None]*len(lines)
    output=[]
    provenance=[]
    i=0
    seen=collections.Counter()
    while i<len(lines):
        m=RESOLVE.fullmatch(lines[i])
        if not m or int(m[1]) not in owners:
            output.append(lines[i])
            provenance.append(source[i])
            i+=1
            continue
        plan=owners[int(m[1])]
        d=plan['declaration']
        records=[]
        while i<len(lines):
            m=RESOLVE.fullmatch(lines[i])
            if not m or owners.get(int(m[1])) is not plan:
                break
            ident=int(m[1])
            symbol=m[2]
            origin=[source[i]]
            require(any(r[1]==ident and r[3]==symbol for r in plan['base_rows']),'create resolve disagrees with authenticated module trace')
            configs=[]
            i+=1
            while i<len(lines) and (c:=CONFIGURE.fullmatch(lines[i])):
                require(int(c[1])==ident,'configure is detached from its own resolve')
                configs.append(int(c[2]))
                origin.append(source[i])
                i+=1
            if symbol in d.get('removed_symbols',[]):
                counts['removed_resolves']+=1
                continue
            new_symbol=d['renames'].get(symbol,symbol)
            if symbol in d.get('configures',{}):
                change=d['configures'][symbol]
                require(configs==change['from'],'old configure bytes differ from exact declaration')
                configs=change['to']
                seen[symbol]+=1
                counts['configure_changes']+=1
                # A changed configure list derives, as a whole, from the resolve and its old configures.
                origin=[tuple(x for o in origin if o is not None for x in o) or None]*(1+len(configs))
            records.append((new_symbol,ident,configs,symbol,origin))
        require(records,'empty retained resolve run')
        expected_ids=[r[1] for r in plan['base_rows'] if r[3] not in d.get('removed_symbols',[])]
        require([r[1] for r in records]==expected_ids,'module resolve run is incomplete or split')
        ordered=sorted(records,key=lambda x:plan['head_order'].index(x[0]))
        counts['resolve_permutations']+=sum(a[1]!=b[1] for a,b in zip(records,ordered))
        for symbol,ident,configs,old_symbol,origin in ordered:
            counts['resolve_renames']+=symbol!=old_symbol
            output.append(f'resolve {ident} {symbol}')
            output.extend(f'configure {ident} {value}' for value in configs)
            provenance.extend(origin[:1+len(configs)])
    return (output,provenance) if trace is not None else output

"""Bind pinned journal lines to helper-recorded device lifetimes and create traces."""
import hashlib
import json
from pathlib import Path
from common import need
from provenance import digest


def subsequence(want,notes):
    cursor=0
    positions=[]
    for i,line in enumerate(notes):
        if cursor<len(want) and line==want[cursor]:
            positions.append(i)
            cursor+=1
    return positions if cursor==len(want) else None


def safe(relative):
    """A trace path under aux/: a file name, or tests/<test>/<file> in
    traces written per test."""
    parts=Path(relative).parts
    return not Path(relative).is_absolute() and '..' not in parts and (len(parts)==1 or (len(parts)==3 and parts[0]=='tests'))


def collect(root,snapshot,bodies):
    root=Path(root)
    aux=root/'aux'
    checkpoints={p.name.removesuffix('.checkpoint.json'):p for p in aux.glob('*.checkpoint.json')}
    need(set(checkpoints)==set(snapshot['pins']),'auxiliary checkpoints do not cover exactly the pinned manifest')
    host_cache={}
    bindings={}
    host_exports={}
    errors={}
    texts={}
    tests={n:json.loads(p.read_text()).get('test') for n,p in checkpoints.items()}
    found={}
    for name,path in sorted(checkpoints.items()):
        cp=json.loads(path.read_text())
        need(cp.get('schema')=='chopinfer.journal-trace.checkpoint.v1' and cp['journal']==name,'invalid trace checkpoint')
        candidates=[]
        # An empty journal has no line a device could have recorded: it is a
        # text journal, its body (no lines) the only evidence.
        for descriptor in cp['devices'] if bodies[name] else []:
            relative=descriptor['path']
            need(safe(relative),'unsafe trace path')
            if relative not in host_cache:
                records=[json.loads(line) for line in (aux/relative).read_text().splitlines()]
                host_cache[relative]=records
            records=host_cache[relative][:descriptor['records']]
            notes=[row[1] for row in records if row[0]=='note']
            need(len(notes)==descriptor['notes'],'trace checkpoint note count mismatch')
            positions=subsequence(bodies[name],notes)
            if positions is not None:
                candidates.append((descriptor,records,notes,positions))
        found[name]=(cp,sorted(candidates,key=lambda c:c[0]['id']))
    # Journals one device alone recorded own their positions there. A journal
    # several devices recorded (an identical journal of another package or
    # host) cannot be bound where another journal owns those positions, so
    # those candidates drop out; of the rest the lowest device ID is bound.
    # The others are listed: an identical copy (the same records) adds
    # nothing; any other is reported as not covered. Journal order never
    # matters: only single-candidate journals claim positions.
    owned={}
    for name,(cp,candidates) in found.items():
        if len(candidates)==1:
            owned.setdefault(candidates[0][0]['path'],[]).append((set(candidates[0][3]),name))
    for name,(cp,candidates) in sorted(found.items()):
        if not candidates:
            # A journal the test builds without a fake device (a text journal):
            # no device recorded its lines, so only its body is evidence.
            texts[name]=dict(test=cp['test'],devices=len(cp['devices']))
            continue
        others=[]
        if len(candidates)>1:
            claimed=lambda c:sorted(n for held,n in owned.get(c[0]['path'],[]) if held&set(c[3]))
            free=[c for c in candidates if not claimed(c)]
            if not free:
                errors[name]=dict(reason='every device that recorded these lines holds another pinned journal there',
                                  candidate_devices=[c[0]['id'] for c in candidates],test=cp['test'])
                continue
            chosen=free[0]
            for c in candidates:
                if c is not chosen:
                    other=dict(id=c[0]['id'],path=c[0]['path'],records=c[0]['records'])
                    if claimed(c):
                        other['claimed_by']=claimed(c)
                    else:
                        other['identical']=c[1]==chosen[1]
                    others.append(other)
            candidates=[chosen]
        descriptor,records,notes,positions=candidates[0]
        created=[(i,row) for i,row in enumerate(records) if row[0]=='created']
        chosen=[(i,row) for i,row in created if row[4]<=positions[0] or (row[2]<=positions[0] and positions[-1]<row[4])]
        if chosen:
            mark_at,mark=chosen[-1]
            start_event,start_note,end_event,end_note=mark[1:]
            create_notes=notes[start_note:end_note]
            kind='instance-create'
        elif name.endswith('-create.txt'):
            start_event=start_note=0
            end_event=descriptor['records']
            end_note=positions[-1]+1
            create_notes=notes[:end_note]
            kind='pinned-create'
        else:
            first_launch=next((i for i,line in enumerate(notes) if line.startswith('launch ')),None)
            if first_launch is None or not all(line.startswith('allocate ') for line in notes[:first_launch]):
                # Encoder steps share the same device as their pinned create.
                siblings=[n for n in bodies if n.endswith('-create.txt') and tests.get(n)==cp['test']]
                matches=[n for n in siblings if notes[:len(bodies[n])]==bodies[n]]
                if len(matches)==1:
                    start_event=start_note=0
                    end_note=len(bodies[matches[0]])
                    note_count=0
                    end_event=0
                    for i,row in enumerate(records):
                        if row[0]=='note':
                            note_count+=1
                        if note_count>end_note:
                            break
                        end_event=i+1
                    create_notes=notes[:end_note]
                    kind='sibling-pinned-create'
                else:
                    # A journal of the whole device lifetime: it holds the one
                    # instance create, or the device never ran one (an encoder).
                    inside=[(i,row) for i,row in created if positions[0]<=row[2] and row[4]<=positions[-1]+1]
                    if len(inside)==1:
                        mark_at,mark=inside[0]
                        start_event,start_note,end_event,end_note=mark[1:]
                        create_notes=notes[start_note:end_note]
                        kind='instance-create'
                    elif not created and positions==list(range(len(notes))):
                        start_event=start_note=end_event=end_note=0
                        create_notes=[]
                        kind='whole-lifetime'
                    else:
                        errors[name]=dict(reason='no recoverable create boundary',test=cp['test'],device=descriptor['id'],
                            proposed_change='Mark the successful create boundary on this fixture device before its first step.')
                        continue
            else:
                start_event=start_note=0
                end_note=first_launch
                note_count=0
                end_event=0
                for i,row in enumerate(records):
                    if row[0]=='note':
                        note_count+=1
                    if note_count>end_note:
                        break
                    end_event=i+1
                create_notes=notes[:end_note]
                kind='direct-allocation-prefix'
        host_key=descriptor['path']
        create_name=str(Path(host_key).parent/f'device-{descriptor["id"]}-create-{start_event}-{end_event}.json').removeprefix('./')
        create_path=aux/create_name
        if not create_path.exists():
            create_path.write_text(json.dumps(create_notes,separators=(',',':'))+'\n')
        host_exports[host_key]=dict(path='aux/'+host_key,sha256=digest(aux/host_key),test=cp['test'])
        bindings[name]=dict(host=host_key,test=cp['test'],records=descriptor['records'],positions=positions,
                            create=dict(path='aux/'+create_name,sha256=digest(create_path),kind=kind,
                                        start_event=start_event,end_event=end_event,start_note=start_note,end_note=end_note))
        if others:
            bindings[name]['other_candidates']=others
            for other in others:
                if 'claimed_by' not in other:
                    host_exports[other['path']]=dict(path='aux/'+other['path'],sha256=digest(aux/other['path']),test=cp['test'])
    manifest=dict(schema='chopinfer.journal-auxiliary.v1',revision=snapshot['commit'],hosts=host_exports,bindings=bindings,
                  text=texts,unrecoverable=errors)
    output=aux/'manifest.json'
    output.write_text(json.dumps(manifest,indent=2)+'\n')
    return dict(path=str(output),sha256=digest(output),bound=len(bindings),text=sorted(texts),unrecoverable=errors)


LEGACY_TEXT='pinned lines do not identify exactly one device lifetime'


def normalize(manifest):
    """The manifest with every text journal under `text`. Captures sealed
    before text journals existed list them as unrecoverable with no
    candidate device; those entries move to `text` (device count unknown)."""
    value=dict(manifest,text=dict(manifest.get('text',{})),unrecoverable=dict(manifest['unrecoverable']))
    for name,error in manifest['unrecoverable'].items():
        if 'text' not in manifest and error.get('reason')==LEGACY_TEXT and error.get('candidate_devices')==[]:
            value['text'][name]=dict(test=error['test'],devices=None)
            del value['unrecoverable'][name]
    need(not(set(value['text'])&(set(value['bindings'])|set(value['unrecoverable']))),'a journal has two binding kinds')
    return value

#!/usr/bin/env python3
"""Prove a complete pinned journal transition between two sealed captures.

Usage: proof.py --repo REPO --model ID --base REV --head REV --spec SPEC
       --base-capture-index INDEX --head-capture-index INDEX --report OUT.json
       [--require identity] [--declaration-revision REV]

Exit 0: PROVEN (with --require identity, also every item of the
byte-identical checklist); 1: NOT PROVEN; 2: invalid input. See README.md.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Invalid, need, unique
from engine import POLICY_FIELDS, Unproven
from pins import snapshot
from text import compare as compare_text, validate as validate_text

SCHEMA = 'chopinfer.journal-proof.spec.v1'
EMPTY_RULE = {'added_resolves': [], 'added_modules': [], 'create_policy': {}}


def read_spec(path):
    value=json.loads(Path(path).read_text(),object_pairs_hook=unique)
    need(isinstance(value,dict) and value.get('schema')==SCHEMA,'invalid spec schema')
    need(set(value)<={'schema','fixtures','journals','added_journals','removed_journals','description','auxiliary_rules','module_replacements','graph_piece_policy','layout'},'unknown spec field')
    if 'layout' in value:
        need(isinstance(value['layout'],dict) and set(value['layout'])=={'path','sha256'},'layout names the declaration path and SHA-256')
    need(isinstance(value.get('fixtures'),list) and isinstance(value.get('journals',{}),dict),'spec needs fixtures and journal policies')
    names=[]
    for fixture in value['fixtures']:
        need(set(fixture)=={'name','create','journals'},'fixture needs name/create/journals')
        need(isinstance(fixture['name'],str) and fixture['name'] not in names,'duplicate fixture name')
        need(isinstance(fixture['journals'],list) and fixture['journals'] and len(set(fixture['journals']))==len(fixture['journals']),
             'fixture journals must be a nonempty unique list')
        need(fixture['create'] in fixture['journals'],'fixture create must be one of its pinned journals')
        names.append(fixture['name'])
    for name,policy in value.get('journals',{}).items():
        need(isinstance(name,str) and isinstance(policy,dict) and set(policy)<=POLICY_FIELDS|{'text_edits'},'unknown journal policy field')
        need(all(isinstance(v,list) for v in policy.values()),'policy fields must be lists')
    for key in ('added_journals','removed_journals'):
        entries=value.get(key,[])
        need(isinstance(entries,list) and len(entries)==len(set(entries)) and all(isinstance(n,str) for n in entries),f'invalid {key}')
    return value


def kind_of(snap,names):
    return {snap["pins"][n]["kind"] for n in names if n in snap["pins"]}


def read_layout(repo,spec,base,head,revision=None):
    """The spec's layout declaration, read with git at the head commit, or at
    `revision` for a change declared after it landed, with the locks of its
    removed and added modules. Returns (layout, report record) or (None, None)."""
    if 'layout' not in spec:
        need(revision is None,'--declaration-revision needs a spec that names a layout declaration')
        return None,None
    from common import resolve
    import layout as declared
    at=resolve(repo,revision) if revision is not None else head['commit']
    decl,data=declared.read(repo,at,spec['layout']['path'],spec['layout']['sha256'])
    modules=declared.authenticate(repo,base['commit'],head['commit'],base['root'],decl)
    need(not spec.get('module_replacements') and 'graph_piece_policy' not in spec,
         'a layout spec declares module changes in the layout declaration, not module replacements or a piece policy')
    for name,policy in spec.get('journals',{}).items():
        need(set(policy)<={'text_edits'} and name in decl['text_journals'],
             f'{name}: under a layout declaration only its declared text journals carry a policy (text_edits)')
    record=dict(path=spec['layout']['path'],sha256=spec['layout']['sha256'],revision=at,read_at_head=at==head['commit'],
                description=decl['description'],removed_modules=decl['removed_modules'],added_modules=decl['added_modules'],
                resources=sorted(decl['resources']),changes=[c['name'] for c in decl['changes']],text_journals=decl['text_journals'])
    return (decl,modules),record


def check(repo,model,base_rev,head_rev,spec,base_index,head_index,declaration_revision=None):
    base,bp,bo=snapshot(repo,base_rev,model,base_index)
    head,hp,ho=snapshot(repo,head_rev,model,head_index)
    need(base['root']==head['root'],'the bundle directory moved between the revisions')
    layout,layout_record=read_layout(repo,spec,base,head,declaration_revision)
    from replacements import authenticate as authenticate_replacements
    replacements=authenticate_replacements(repo,base['commit'],head['commit'],spec.get('module_replacements',[]),base['root'])
    piece_policy=None
    if 'graph_piece_policy' in spec:
        from history import authenticate_piece_policy
        piece_policy=authenticate_piece_policy(repo,base['commit'],head['commit'],spec['graph_piece_policy'])
    b,h=(set(s['pins']) for s in (base,head))
    added,removed=h-b,b-h
    global_problems=[]
    if added!=set(spec.get('added_journals',[])):
        global_problems.append('added journals differ from exact declaration')
    if removed!=set(spec.get('removed_journals',[])):
        global_problems.append('removed journals differ from exact declaration')
    bound=[name for f in spec['fixtures'] for name in f['journals']]
    union=b|h
    if len(bound)!=len(set(bound)):
        global_problems.append('a journal is bound to multiple fixtures')
    for fixture in spec['fixtures']:
        for snap in (base,head):
            owners={snap['pins'][n]['fixture'] for n in fixture['journals'] if n in snap['pins']}
            if len(owners)>1:
                global_problems.append(f'fixture {fixture["name"]} combines independent source fixtures')
            for owner in owners:
                expected={n for n,pin in snap['pins'].items() if pin['fixture']==owner}
                if expected!={n for n in fixture['journals'] if n in snap['pins']}:
                    global_problems.append(f'fixture {fixture["name"]} splits a source fixture across mappings')
    if set(bound)!=union:
        global_problems.append(f'fixture manifest is incomplete: missing {sorted(union-set(bound))}, extra {sorted(set(bound)-union)}')
    if set(spec.get('journals',{}))-union:
        global_problems.append('journal policy names an unpinned journal')
    for name,policy in spec.get('journals',{}).items():
        text=any(s['pins'].get(name,{}).get('kind')=='text' for s in (base,head))
        if text:
            need(set(policy)<={'text_edits'},f'{name}: a text journal takes only text_edits')
            validate_text(policy.get('text_edits',[]))
        else:
            need('text_edits' not in policy,f'{name}: text_edits apply only to text journals')
    from auxproof import load as load_aux,prove as prove_aux,validate_rule,validate_declared_only
    base_aux=load_aux(base,base_index,bp,bo)
    head_aux=load_aux(head,head_index,hp,ho)
    need(isinstance(spec.get('auxiliary_rules'),dict) and
         set(spec['auxiliary_rules'])=={f['name'] for f in spec['fixtures']},
         'auxiliary declarations must cover every fixture exactly once')
    text_fixtures=set()
    for fixture in spec['fixtures']:
        rule=spec['auxiliary_rules'][fixture['name']]
        if set(fixture['journals'])<=added|removed:
            validate_declared_only(rule)
        elif kind_of(base,fixture['journals'])|kind_of(head,fixture['journals'])=={'text'}:
            # No device lifetime: nothing for an auxiliary rule to declare.
            validate_declared_only(rule)
            text_fixtures.add(fixture['name'])
        else:
            validate_rule(rule,layout is not None)
    verdicts={}
    counts=collections.Counter()
    mappings={}
    for fixture in spec['fixtures']:
        members=fixture['journals']
        ordinary=[name for name in members if name in b&h]
        # A declared added/removed journal is still authenticated by its
        # own revision's source pin. It is explicitly not an equivalence.
        for name in set(members)&(added|removed):
            verdicts[name]=dict(verdict='DECLARED_ADD' if name in added else 'DECLARED_REMOVE',counts={},reasons=[])
            counts['added_journals' if name in added else 'removed_journals']+=1
        if not ordinary:
            continue
        kinds=dict(base=kind_of(base,ordinary),head=kind_of(head,ordinary))
        if kinds['base']!=kinds['head']:
            for name in ordinary:
                verdicts[name]=dict(verdict='UNPROVEN',counts={},
                                    reasons=[f"binding kind changed: base {'/'.join(sorted(kinds['base']))}, head {'/'.join(sorted(kinds['head']))}"])
            continue
        if len(kinds['base'])!=1:
            for name in ordinary:
                verdicts[name]=dict(verdict='UNPROVEN',counts={},
                                    reasons=[f"fixture joins journals of binding kinds {'/'.join(sorted(kinds['base']))}"])
            continue
        if fixture['name'] in text_fixtures:
            mapping=dict(kind='text',tests=sorted({base['pins'][n]['test'] for n in ordinary}),history=dict(status='TEXT'))
            for name in ordinary:
                try:
                    verdict,changes=compare_text(bp[name],hp[name],spec.get('journals',{}).get(name,{}).get('text_edits',[]))
                    verdicts[name]=dict(verdict=verdict,reasons=[],counts=changes)
                    counts.update(changes)
                except Unproven as error:
                    verdicts[name]=dict(verdict='UNPROVEN',reasons=[str(error)],counts={})
            mappings[fixture['name']]=mapping
            continue
        if fixture['create'] not in ordinary:
            for name in ordinary:
                verdicts[name]=dict(verdict='UNPROVEN',reasons=['no paired create journal'],counts={})
            continue
        rule=spec['auxiliary_rules'][fixture['name']]
        declared=dict(base=sorted(set(members)&removed),head=sorted(set(members)&added))
        result,changes,mapping=prove_aux(bp,hp,dict(fixture,journals=ordinary,declared_journals=declared,piece_policy=piece_policy),
                                         spec.get('journals',{}),base_aux,head_aux,rule,replacements,layout=layout)
        verdicts.update(result)
        counts.update(changes)
        mappings[fixture['name']]=mapping
    for name in union-set(verdicts):
        verdicts[name]=dict(verdict='UNPROVEN',reasons=['journal not covered by a unique fixture'],counts={})
    for name in union:
        verdicts[name].update(base_pin=base['pins'].get(name),head_pin=head['pins'].get(name),
                              base_body=bo.get(name),head_body=ho.get(name))
    auxiliary_counts=collections.Counter()
    history_counts=collections.Counter()
    history_status={}
    for name,mapping in mappings.items():
        auxiliary_counts.update(mapping.get('create_changes',{}))
        record=mapping.get('history')
        if record is None:
            history_status[name]='NOT EVALUATED'
        elif record:
            history_status[name]=record['status']
            if record['status']=='COMPARED':
                history_counts.update(record.get('changes',{}))
                history_counts['compared_operations']+=record['head_operations']
                history_counts['trailing_uncompared_notes']+=record['trailing_uncompared_notes']['head']
    auxiliary_provenance={}
    for side,aux,path in [('base',base_aux,base_index),('head',head_aux,head_index)]:
        auxiliary_provenance[side]=dict(archive_commit=aux['payload']['archive_commit'],schema=aux['payload']['schema'],
            producer=aux['payload']['producer'],helper_digest=aux['payload']['helper_digest'],index=str(path),
            index_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            attestation='verified local helper HMAC-SHA256')
        auxiliary_provenance[side]['call_sites']=(base if side=='base' else head)['call_sites']
        # A capture whose run ignored listed tests (--allow-ignored) names
        # the list and its digest.
        if 'allowed_ignored' in aux['payload']:
            auxiliary_provenance[side]['allowed_ignored']=aux['payload']['allowed_ignored']
    passed=not global_problems and all(v['verdict']!='UNPROVEN' for v in verdicts.values())
    compared=sorted(n for n,v in history_status.items() if v=='COMPARED')
    texts=sorted(n for n,v in history_status.items() if v=='TEXT')
    declared_only=sorted(f['name'] for f in spec['fixtures'] if set(f['journals'])<=added|removed)
    pieces=sorted(n for n,m in mappings.items() if m.get('history',{}).get('changes',{}).get('graph_pieces'))
    coverage=dict(fixtures=len(spec['fixtures']),history_compared=len(compared),
                  history_not_compared=sorted({f['name'] for f in spec['fixtures']}-set(compared)-set(texts)),
                  text_fixtures=texts,
                  declared_only_fixtures=declared_only,
                  declared_journals=dict(added=sorted(added),removed=sorted(removed)),
                  graph_piece_exception=pieces,
                  uncovered_devices=uncovered_devices(base_aux,head_aux),
                  unloaded_module_fixtures=sorted(n for n,m in mappings.items() if any((m.get('unloaded_module_resolves') or {}).values())),
                  full_history_proof=passed and len(compared)+len(texts)==len(spec['fixtures']) and not added and not removed,
                  layout_declaration=layout_record,
                  # Declared span lengths in buffers the device never
                  # allocated: exact, but bounded by no allocation.
                  unbounded_host_span_lengths=sum(c.get('host_span_lengths',0) for c in (counts,auxiliary_counts,history_counts)))
    return dict(schema='chopinfer.journal-proof.report.v1',passed=passed,status='PROVEN' if passed else 'NOT PROVEN',
                model=model,base=base['commit'],head=head['commit'],manifest_counts=dict(base=len(b),head=len(h)),
                layout_declaration=layout_record,
                sources=dict(base=base['sources'],head=head['sources']),module_replacements=replacements,graph_piece_policy=piece_policy,problems=global_problems,
                counts=dict(counts),auxiliary_create_counts=dict(auxiliary_counts),auxiliary_provenance=auxiliary_provenance,
                claim=claim(coverage),coverage=coverage,auxiliary_history_counts=dict(history_counts),
                auxiliary_history_status=dict(sorted(history_status.items())),
                journals=dict(sorted(verdicts.items())),fixtures=mappings,
                capture_mode='sealed helper captures',
                limitations=['FNV pins detect changes but are not cryptographic provenance. SHA-256 of each supplied body is reported.',
                             'A declarative spec authorizes changes; it does not prove kernel numerical equivalence.'])


def uncovered_devices(*sides):
    """Devices that recorded a bound journal's lines too, with different
    records: the proof compares only the bound device's history."""
    return sorted({f"{name}: {other['path']}" for aux in sides for name, binding in aux['manifest']['bindings'].items()
                   for other in binding.get('other_candidates', []) if other.get('identical') is False})


def claim(coverage=None):
    """What a PROVEN verdict covers, stated in every report and conditioned
    on the fixtures whose history was actually compared."""
    from history import CLAIM,NOT_COMPARED
    compared=dict(CLAIM)
    not_compared=list(NOT_COMPARED)
    if coverage is None:
        return dict(compared=compared,not_compared=not_compared)
    scope=f"Applies to the {coverage['history_compared']} of {coverage['fixtures']} fixtures whose history was COMPARED."
    if not coverage['history_compared']:
        compared={}
        not_compared.insert(0,'No fixture history was compared; this report proves no pinned, create or history equivalence.')
    else:
        compared={k:f'{v} {scope}' for k,v in compared.items()}
    if coverage['graph_piece_exception'] and 'ordering' in compared:
        compared['ordering']+=(' Exception: in fixtures '+', '.join(coverage['graph_piece_exception'])+
            ', inner boundaries of graph programs re-split under the source-authenticated piece policy are '
            'recomputed instead of ordered; those programs capture only launches and are replayed whole and in order.')
    if coverage['graph_piece_exception'] and 'ordering' not in compared:
        not_compared.insert(0,'Graph piece exception applied in fixtures '+', '.join(coverage['graph_piece_exception'])+
                            ', but no ordering was established there.')
    if coverage['history_not_compared']:
        not_compared.insert(0,'Fixtures whose history was not compared: '+', '.join(coverage['history_not_compared'])+'.')
    declared=coverage['declared_journals']
    if declared['added'] or declared['removed']:
        not_compared.insert(0,'Declared added/removed journals are authenticated against their own pins but compared to nothing; '
                            'their device histories are not compared: '+', '.join(declared['added']+declared['removed'])+'.')
    if coverage['text_fixtures']:
        compared['text_journals']=('Journals checked without a fake device (text fixtures '+', '.join(coverage['text_fixtures'])+
            '): each body byte for byte, or under its declared text_edits. They have no device history, create or handle mapping.')
    if coverage.get('unloaded_module_fixtures'):
        not_compared.insert(0,'Symbols resolved from module handles the device never loaded (fixtures '+
                            ', '.join(coverage['unloaded_module_fixtures'])+') keep their literal handle; their module bytes are not authenticated.')
    if coverage.get('uncovered_devices'):
        not_compared.insert(0,'Devices that recorded the same lines as a bound journal, with a different history, are not '
                            'covered (each journal binds one device): '+', '.join(coverage['uncovered_devices'])+'.')
    layout=coverage.get('layout_declaration')
    if layout:
        compared['layout']=(f"Changes authorized by the layout declaration {layout['path']} (SHA-256 {layout['sha256']}) read at "
            f"{layout['revision']}: added allocations against declared resources, removed and added modules against both "
            'revisions\' locks, resized allocations and their spans, collectives and kernel families selected and rewritten '
            'from the base; every other line exact under the typed mapping.')
        if not layout['read_at_head']:
            not_compared.insert(0,f"The layout declaration was read at {layout['revision']}, not at the head commit: the change "
                                  'was declared after it was made.')
    if coverage.get('unbounded_host_span_lengths'):
        not_compared.insert(0,f"{coverage['unbounded_host_span_lengths']} declared span lengths lie in buffers the device never "
                              'allocated (host-provided); they are exact but bounded by no allocation.')
    if not coverage['full_history_proof']:
        not_compared.insert(0,'This is not a full history proof (coverage.full_history_proof is false).')
    return dict(compared=compared,not_compared=not_compared)


def summary(report):
    lines=[report['status']]
    if 'journals' in report:
        for name,value in report['journals'].items():
            reason='; '.join(value['reasons'])
            lines.append(f"{value['verdict']} {name}"+(f': {reason}' if reason else '')+(f" {value['counts']}" if value['counts'] else ''))
        lines.append('Manifest: '+json.dumps(report['manifest_counts'],sort_keys=True))
        lines.append('Verdicts: '+json.dumps(dict(collections.Counter(v['verdict'] for v in report['journals'].values())),sort_keys=True))
        lines.append('Authorized pinned changes: '+json.dumps(report['counts'],sort_keys=True))
        if report.get('auxiliary_create_counts'):
            lines.append('Authorized auxiliary create changes: '+json.dumps(report['auxiliary_create_counts'],sort_keys=True))
        if report.get('auxiliary_history_status'):
            status=collections.Counter(report['auxiliary_history_status'].values())
            lines.append('Auxiliary history fixtures: '+json.dumps(dict(status),sort_keys=True))
            lines.append('Auxiliary history counts: '+json.dumps(report.get('auxiliary_history_counts',{}),sort_keys=True))
            for name,value in report['auxiliary_history_status'].items():
                if value not in ('COMPARED','TEXT'):
                    record=report['fixtures'][name].get('history',{})
                    lines.append(f"History {value} {name}: {record.get('error','')}")
        if report.get('coverage'):
            c=report['coverage']
            lines.append(f"Coverage: {c['history_compared']}/{c['fixtures']} fixture histories compared, "
                         f"{len(c.get('text_fixtures',[]))} text fixtures; "
                         f"full history proof: {str(c['full_history_proof']).lower()}")
        if 'claim' in report:
            lines.extend(f'Compared {key}: {text}' for key,text in report['claim']['compared'].items())
            lines.append('Not compared: '+' '.join(report['claim']['not_compared']))
    for item,ok in (report.get('identity_checklist') or {}).items():
        lines.append(('PASS ' if ok else 'FAIL ')+item)
    lines.extend(report.get('problems',[]))
    return '\n'.join(lines)+'\n'


def identity_checklist(report, spec):
    """The byte-identical checklist: every journal IDENTICAL, every fixture
    fully compared, and a spec that declares nothing."""
    coverage = report.get('coverage') or {}
    statuses = report.get('auxiliary_history_status') or {}
    verdicts = {name: journal.get('verdict') for name, journal in (report.get('journals') or {}).items()}
    return {
        'full history proof': coverage.get('full_history_proof') is True,
        'status PROVEN': report.get('passed') is True and report.get('status') == 'PROVEN',
        'every journal IDENTICAL': bool(verdicts) and set(verdicts.values()) == {'IDENTICAL'},
        'every auxiliary rule empty': all(rule == EMPTY_RULE for rule in spec.get('auxiliary_rules', {}).values()),
        'no journal policies': not spec.get('journals'),
        'no module_replacements': 'module_replacements' not in spec and not report.get('module_replacements'),
        'no graph_piece_policy': 'graph_piece_policy' not in spec and not report.get('graph_piece_policy'),
        'counts empty': not report.get('counts'),
        'auxiliary_create_counts empty': not report.get('auxiliary_create_counts'),
        'history counts only compared_operations and trailing_uncompared_notes':
            set(report.get('auxiliary_history_counts') or {}) <= {'compared_operations', 'trailing_uncompared_notes'},
        'graph_piece_exception empty': not coverage.get('graph_piece_exception'),
        'every fixture COMPARED or TEXT': bool(statuses) and set(statuses.values()) <= {'COMPARED', 'TEXT'}
            and coverage.get('history_compared', 0) + len(coverage.get('text_fixtures', [])) == coverage.get('fixtures') == len(statuses),
    }


def main(argv=None):
    os.umask(0o022)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for option in ('repo', 'model', 'base', 'head', 'spec', 'report', 'base-capture-index', 'head-capture-index'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--require', choices=['identity'],
                        help='identity: also require the byte-identical checklist (exit 1 when an item fails)')
    parser.add_argument('--declaration-revision',
                        help='read the spec\'s layout declaration at this revision instead of the head commit '
                             '(a change declared after it landed); the report records it')
    args = parser.parse_args(argv)
    try:
        spec = read_spec(args.spec)
        report = check(args.repo, args.model, args.base, args.head, spec, args.base_capture_index, args.head_capture_index,
                       args.declaration_revision)
        report['spec_sha256'] = hashlib.sha256(Path(args.spec).read_bytes()).hexdigest()
        code = 0 if report['passed'] else 1
        if args.require == 'identity':
            report['identity_checklist'] = identity_checklist(report, spec)
            if not all(report['identity_checklist'].values()):
                code = 1
    except (Invalid, KeyError, TypeError, IndexError, OSError, ValueError) as error:
        report = dict(schema='chopinfer.journal-proof.report.v1', status='INVALID INPUT', passed=False,
                      problems=[f'{type(error).__name__}: {error}'])
        code = 2
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    text = summary(report)
    target.with_suffix('.txt').write_text(text)
    print(text, end='')
    return code


if __name__ == '__main__':
    sys.exit(main())

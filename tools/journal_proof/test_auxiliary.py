#!/usr/bin/env python3
"""Exercise authenticated served-fixture bindings and typed host mappings."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from auxiliary import collect
from auxproof import derive
from engine import Unproven
from proof import main
from provenance import init
from testkit import CREATE, GEN, MODEL, STEP, GitFixture, checkpoint, create_events, fixtures, launch, seal_capture, write_device


def trace(bodies,head=False,rank=0,size=64):
    events=[]
    def issue(i,*row):
        events.extend([['id',i],list(row)])
    issue(1,'stream',1,0)
    events.append(['note',bodies[CREATE][0]])
    issue(2,'module',2,'cubin_old','old_hash',100)
    events.append(['note',bodies[CREATE][1]])
    issue(3,'resolve',3,2,'chop_old')
    events.append(['note',bodies[CREATE][2]])
    if head:
        issue(4,'module',4,'cubin_new','new_hash',120)
        events.append(['note',bodies[CREATE][3]])
        issue(5,'resolve',5,4,'chop_new')
        events.append(['note',bodies[CREATE][4]])
    allocation=6 if head else 4
    issue(allocation,'allocation',GEN+allocation,rank,size,256,GEN+1)
    events.extend([['note',line] for line in bodies[CREATE][-2:]])
    events.append(['created',0,0,len(events),len(bodies[CREATE])])
    create_records=len(events)
    events.extend([['note',line] for line in bodies[STEP]])
    return events,create_records


class AuxiliaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='journal-proof-aux-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        authority=patch('provenance.AUTHORITY',self.root/'private-authority')
        authority.start()
        self.addCleanup(authority.stop)
        init()
        self.repo=GitFixture(self.root/'fixture.git')
        self.base,self.head,self.spec=fixtures()
        self.spec['auxiliary_rules']={'host':dict(
            added_resolves=[dict(symbol='chop_new',ids=[5],count=1)],
            added_modules=[dict(name='cubin_new',fnv1a='new_hash',bytes=120,ids=[4],count=1)],
            create_policy=self.spec['journals'][CREATE])}

    def capture(self,label,bodies,head=False,mutate=None):
        root=self.root/label
        rev=self.repo.commit(bodies)
        events,create_records=trace(bodies,head)
        if mutate:
            mutate(events)
        write_device(root,1,events)
        for name in bodies:
            checkpoint(root,name,[(1,events,create_records if name==CREATE else len(events))])
        return rev,seal_capture(root,rev,bodies)

    def check(self,expected,head=None,mutate=None,tamper=None):
        br,bi=self.capture('base',self.base)
        hr,hi=self.capture('head',self.head if head is None else head,True,mutate)
        if tamper:
            tamper(hi)
        spec=self.root/'spec.json'
        spec.write_text(json.dumps(self.spec))
        report=self.root/'report.json'
        with contextlib.redirect_stdout(io.StringIO()):
            code=main(['--repo',str(self.repo.repo),'--model',MODEL,'--base',br,'--head',hr,'--spec',str(spec),
                       '--base-capture-index',str(bi),'--head-capture-index',str(hi),'--report',str(report)])
        value=json.loads(report.read_text())
        self.assertEqual(code,expected,value)
        return value

    def test_served_fixture_positive_control(self):
        report=self.check(0)
        self.assertEqual(report['fixtures']['host']['buffer_map'],{str(GEN+4):GEN+6})
        self.assertEqual(report['fixtures']['host']['allocation_records'][0]['rank'],0)

    def test_copied_step_rejected_with_authentic_history(self):
        self.head[STEP]=self.base[STEP][:]
        self.spec['journals'][STEP]={}
        self.check(1)

    def test_stale_handles_with_new_schedule_rejected(self):
        self.head[STEP]=[line.replace(f'raw: {GEN+6}',f'raw: {GEN+4}') for line in self.head[STEP]]
        self.check(1)

    def test_allocation_rank_change_rejected(self):
        self.check(1,mutate=lambda events:next(e for e in events if e[0]=='allocation').__setitem__(2,1))

    def test_allocation_size_change_rejected(self):
        self.check(1,mutate=lambda events:next(e for e in events if e[0]=='allocation').__setitem__(3,128))

    def test_allocation_generation_change_rejected(self):
        self.check(1,mutate=lambda events:next(e for e in events if e[0]=='allocation').__setitem__(1,2*GEN+6))

    def test_changed_module_payload_rejected(self):
        self.check(1,mutate=lambda events:next(e for e in events if e[0]=='module').__setitem__(3,'bad_hash'))

    def test_changed_new_module_payload_rejected(self):
        self.check(1,mutate=lambda events:next(e for e in events if e[0]=='module' and e[1]==4).__setitem__(3,'bad_hash'))

    def test_missing_added_resolve_rejected(self):
        self.check(1,mutate=lambda events:events.remove(next(e for e in events if e[0]=='resolve' and e[1]==5)))

    def test_wrong_resolve_owner_rejected(self):
        self.check(1,mutate=lambda events:next(e for e in events if e[0]=='resolve' and e[1]==5).__setitem__(2,2))

    def test_unrecorded_rank_cannot_authorize_an_allocation_shift(self):
        a,_=trace(self.base)
        b,_=trace(self.head,True)
        for events in (a,b):
            next(e for e in events if e[0]=='allocation')[2]=None
        with self.assertRaisesRegex(Unproven,'rank is unrecorded'):
            derive(a,b,self.spec['auxiliary_rules']['host'])

    def test_an_added_or_removed_allocation_cannot_be_declared(self):
        # No rule adds or removes a device allocation: the issued IDs must
        # pair one to one beyond declared modules and resolves. The README
        # names this refusal as the trigger for a declared allocation rule.
        a,_=trace(self.base)
        plain,_=trace(self.head,True)
        more,_=trace(self.head,True)
        more[-len(self.head[STEP]):-len(self.head[STEP])]=[['id',7],['allocation',GEN+7,0,32,256,GEN+1]]
        derive(a,plain,self.spec['auxiliary_rules']['host'])
        for base,head,rule in ((a,more,self.spec['auxiliary_rules']['host']),
                               (more,plain,dict(added_resolves=[],added_modules=[],create_policy={}))):
            with self.assertRaisesRegex(Unproven,'issued ID count differs'):
                derive(base,head,rule)

    def test_fixed_stream_cannot_follow_a_nonstream_issue_id(self):
        a=[['id',1],['allocation',GEN+1,None,64,256,GEN+3],
           ['id',2],['module',2,'old','old_hash',100],['id',3],['resolve',3,2,'old_fn']]
        b=[['id',1],['allocation',GEN+1,None,64,256,GEN+5],
           ['id',2],['module',2,'new','new_hash',120],['id',3],['resolve',3,2,'new_fn'],
           ['id',4],['module',4,'old','old_hash',100],['id',5],['resolve',5,4,'old_fn']]
        rule=dict(added_resolves=[dict(symbol='new_fn',ids=[3],count=1)],
                  added_modules=[dict(name='new',fnv1a='new_hash',bytes=120,ids=[2],count=1)],create_policy={})
        with self.assertRaisesRegex(Unproven,'identity, rank, size or payload'):
            derive(a,b,rule)

    def test_an_allocation_stream_may_not_alias_a_created_stream(self):
        # Base: created stream 1, and an allocation on unrecorded stream 3.
        # The head adds a module and a resolve first, so stream 1 becomes 3
        # and the allocation sits on created stream 3.
        a=[['id',1],['stream',1,0],['id',2],['allocation',GEN+2,0,64,256,GEN+3]]
        b=[['id',1],['module',1,'new','new_hash',120],['id',2],['resolve',2,1,'new_fn'],
           ['id',3],['stream',3,0],['id',4],['allocation',GEN+4,0,64,256,GEN+3]]
        rule=dict(added_resolves=[dict(symbol='new_fn',ids=[2],count=1)],
                  added_modules=[dict(name='new',fnv1a='new_hash',bytes=120,ids=[1],count=1)],create_policy={})
        with self.assertRaisesRegex(Unproven,'is on stream 3, which has no recorded creation and is the image of a created stream'):
            derive(a,b,rule)
        b[-1][-1]=GEN+40
        a[-1][-1]=GEN+40
        derive(a,b,rule)

    def test_unknown_auxiliary_create_policy_is_invalid(self):
        self.spec['auxiliary_rules']['host']['create_policy']['ignore_all']=[]
        self.check(2)

    def test_extra_auxiliary_fixture_is_invalid(self):
        self.spec['auxiliary_rules']['unused']={}
        self.check(2)

    def test_unsigned_auxiliary_rejected(self):
        def tamper(path):
            v=json.loads(path.read_text())
            v.pop('attestation')
            path.write_text(json.dumps(v))
        self.check(2,tamper=tamper)

    def test_forged_auxiliary_attestation_rejected(self):
        def tamper(path):
            v=json.loads(path.read_text())
            v['archive_commit']='0'*40
            path.write_text(json.dumps(v))
        self.check(2,tamper=tamper)

    def test_tampered_trace_rejected(self):
        self.check(2,tamper=lambda path:(path.parent/'aux/device-1.jsonl').write_text('[]\n'))

    def test_tampered_create_rejected(self):
        def tamper(path):
            next((path.parent/'aux').glob('device-*-create-*.json')).write_text('[]')
        self.check(2,tamper=tamper)



class BindingRuleTests(unittest.TestCase):
    """Device bindings of the shapes the first real captures of two more
    bundles showed: whole-lifetime journals holding their instance create,
    encoder devices that run no instance create, a journal byte-identical
    to another package's on its own device, and identical device copies."""

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='journal-proof-binding-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        authority=patch('provenance.AUTHORITY',self.root/'private-authority')
        authority.start()
        self.addCleanup(authority.stop)
        init()
        self.repo=GitFixture(self.root/'fixture.git')
        self.base=fixtures()[0]

    def capture(self,devices,checks):
        """`devices` {id: events}; `checks` {journal: (lines, device ids
        created by its check)}. Returns the revision, sealed index and manifest."""
        root=self.root/'capture'
        bodies={name:lines for name,(lines,_) in checks.items()}
        revision=self.repo.commit(bodies)
        for ident,events in devices.items():
            write_device(root,ident,events)
        for name,(_,ids) in checks.items():
            checkpoint(root,name,[(i,devices[i],len(devices[i])) for i in ids])
        index=seal_capture(root,revision,bodies)
        return revision,index,json.loads((root/'aux/manifest.json').read_text())

    def prove(self,revision,index):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parent/'specs'))
        from identity_spec import identity_spec
        spec=self.root/'spec.json'
        spec.write_text(json.dumps(identity_spec(self.repo.repo,MODEL,revision,index)))
        report=self.root/'report.json'
        with contextlib.redirect_stdout(io.StringIO()):
            code=main(['--repo',str(self.repo.repo),'--model',MODEL,'--base',revision,'--head',revision,'--spec',str(spec),
                       '--base-capture-index',str(index),'--head-capture-index',str(index),'--report',str(report),
                       '--require','identity'])
        value=json.loads(report.read_text())
        self.assertEqual(code,0,value.get('problems') or value.get('identity_checklist') or value.get('journals'))
        self.assertTrue(value['coverage']['full_history_proof'])
        return value

    def device(self,create,step):
        return create_events(create)+[['note',line] for line in step]

    def test_identical_device_copies_bind_the_lowest_id(self):
        events=self.device(self.base[CREATE],self.base[STEP])
        revision,index,manifest=self.capture({1:events,2:list(events)},{CREATE:(self.base[CREATE],[1,2]),STEP:(self.base[STEP],[1,2])})
        for name in (CREATE,STEP):
            self.assertEqual(manifest['bindings'][name]['host'],'device-1.jsonl')
            self.assertEqual(manifest['bindings'][name]['other_candidates'],
                             [dict(id=2,path='device-2.jsonl',records=len(events),identical=True)])
        self.assertEqual(self.prove(revision,index)['coverage']['uncovered_devices'],[])

    def test_a_device_where_another_journal_owns_the_lines_is_not_bound(self):
        # Two packages with byte-identical create journals: the second
        # package's create is checked once both devices exist. Its name sorts
        # first, so the result cannot come from journal order.
        steps=[launch('chop_keep_beta')],[launch('chop_consumer')]
        devices={1:self.device(self.base[CREATE],steps[0]),2:self.device(self.base[CREATE],steps[1])}
        checks={'toy-z-create.txt':(self.base[CREATE],[1]),'toy-z-step.txt':(steps[0],[1]),
                'toy-a-create.txt':(self.base[CREATE],[1,2]),'toy-a-step.txt':(steps[1],[1,2])}
        revision,index,manifest=self.capture(devices,checks)
        self.assertEqual(manifest['bindings']['toy-a-create.txt']['host'],'device-2.jsonl')
        self.assertEqual(manifest['bindings']['toy-a-create.txt']['other_candidates'],
                         [dict(id=1,path='device-1.jsonl',records=len(devices[1]),claimed_by=['toy-z-create.txt'])])
        report=self.prove(revision,index)
        self.assertEqual((report['coverage']['history_compared'],report['coverage']['uncovered_devices']),(2,[]))

    def test_a_different_device_with_the_same_lines_is_reported_uncovered(self):
        events=self.device(self.base[CREATE],self.base[STEP])
        other=events+[['note','synchronize 1']]
        revision,index,manifest=self.capture({1:events,2:other},{CREATE:(self.base[CREATE],[1,2]),STEP:(self.base[STEP],[1,2])})
        self.assertEqual(manifest['bindings'][STEP]['other_candidates'][0]['identical'],False)
        report=self.prove(revision,index)
        self.assertEqual(report['coverage']['uncovered_devices'],[f'{CREATE}: device-2.jsonl',f'{STEP}: device-2.jsonl'])
        self.assertTrue(any('are not covered' in line for line in report['claim']['not_compared']))

    def test_a_whole_lifetime_journal_binds_the_instance_create_inside_it(self):
        events=self.device(self.base[CREATE],self.base[STEP])
        whole=self.base[CREATE]+self.base[STEP]
        revision,index,manifest=self.capture({1:events},{'toy-whole.txt':(whole,[1])})
        create=manifest['bindings']['toy-whole.txt']['create']
        self.assertEqual((create['kind'],create['start_note'],create['end_note']),('instance-create',0,len(self.base[CREATE])))
        self.prove(revision,index)

    def test_an_encoder_device_binds_its_whole_lifetime_with_an_empty_create(self):
        # Its resolves name a module handle (1) the device never loaded.
        events=[]
        for ident,symbol in ((1,'chop_enc_a'),(2,'chop_enc_b')):
            events+=[['id',ident],['note',f'resolve {ident} {symbol}'],['resolve',ident,1,symbol]]
        events+=[['note',launch('chop_enc_a')],['note',launch('chop_enc_a')]]
        lines=[e[1] for e in events if e[0]=='note']
        revision,index,manifest=self.capture({1:events},{'toy-encoder.txt':(lines,[1])})
        create=manifest['bindings']['toy-encoder.txt']['create']
        self.assertEqual((create['kind'],create['end_note'],create['end_event']),('whole-lifetime',0,0))
        report=self.prove(revision,index)
        self.assertEqual(len(report['coverage']['unloaded_module_fixtures']),1)
        self.assertTrue(any('not authenticated' in line for line in report['claim']['not_compared']))

    def test_a_partial_journal_of_a_device_without_create_stays_unbound(self):
        events=[['id',1],['note','resolve 1 chop_enc_a'],['resolve',1,1,'chop_enc_a'],['note',launch('chop_enc_a')],['note',launch('chop_enc_a',grid=8)]]
        _,_,manifest=self.capture({1:events},{'toy-encoder.txt':([launch('chop_enc_a',grid=8)],[1])})
        self.assertEqual(manifest['unrecoverable']['toy-encoder.txt']['reason'],'no recoverable create boundary')

    def test_bindings_are_checked_against_the_traces_when_proven(self):
        from auxproof import load
        from pins import snapshot
        events=self.device(self.base[CREATE],self.base[STEP])
        other=events+[['note','synchronize 1']]
        revision,index,manifest=self.capture({1:events,2:other},{CREATE:(self.base[CREATE],[1,2]),STEP:(self.base[STEP],[1,2])})
        snap,bodies,origins=snapshot(self.repo.repo,revision,MODEL,index)
        load(snap,index,bodies,origins)
        for edit,message in ((lambda m:m['bindings'][STEP]['other_candidates'][0].update(identical=True),'identity misstated'),
                             (lambda m:m['bindings'][STEP]['create'].update(kind='whole-lifetime'),'whole-lifetime binding')):
            changed=json.loads(json.dumps(manifest))
            edit(changed)
            self.reseal(index,changed)
            with self.assertRaisesRegex(Exception,message):
                load(snap,index,bodies,origins)

    def reseal(self,index,manifest):
        from provenance import digest, seal
        root=index.parent
        (root/'aux/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        payload=json.loads(index.read_text())
        payload.pop('attestation')
        payload['auxiliary']['sha256']=payload['auxiliary_files']['aux/manifest.json']=digest(root/'aux/manifest.json')
        index.write_text(json.dumps(seal(payload)))


class SpanTests(unittest.TestCase):
    """A span the base already held past its allocation, kept unchanged in
    its own place, is the base's own record (a fake host does not bound
    spans); any other span is bounded."""
    context=dict(sizes={GEN+4:dict(base_bytes=64,head_bytes=64),GEN+5:dict(base_bytes=128,head_bytes=64)})

    def test_a_base_span_past_an_unchanged_allocation_is_kept(self):
        from allocation_args import validate_spans
        from testkit import span
        line=f'launch 1 chop_x [{span(GEN+4,0,4096)}]'
        validate_spans([line],self.context,[line])
        for kept in ([f'launch 1 chop_x [{span(GEN+4,0,64)}]'],[None],None):
            with self.assertRaisesRegex(Unproven,'rewritten span exceeds'):
                validate_spans([line],self.context,kept)
        # The same span held at another position of the kept line.
        moved=f'launch 1 chop_x [{span(GEN+4,0,64)}, {span(GEN+4,0,4096)}]'
        with self.assertRaisesRegex(Unproven,'rewritten span exceeds'):
            validate_spans([f'launch 1 chop_x [{span(GEN+4,0,4096)}, {span(GEN+4,0,4096)}]'],self.context,[moved])

    def test_a_changed_non_launch_record_is_bounded(self):
        from allocation_args import validate_spans
        from testkit import span
        record='zero 1 '+span(GEN+4,0,4096)[len('DeviceBuffer('):-1]
        validate_spans([record],self.context,[record])
        for changed in (record.replace('zero 1','zero 2'),record.replace('zero 1','download 1')):
            with self.assertRaisesRegex(Unproven,'rewritten span exceeds'):
                validate_spans([changed],self.context,[record])

    def test_a_span_of_a_resized_allocation_is_always_bounded(self):
        from allocation_args import validate_spans
        from testkit import span
        line=f'launch 1 chop_x [{span(GEN+5,0,128)}]'
        with self.assertRaisesRegex(Unproven,'rewritten span exceeds'):
            validate_spans([line],self.context,[line])


class ProofSpanCase:
    """Full proofs over a launch whose first argument spans past its 64-byte
    allocation in the base (review C2 r4): only that span, unchanged in its
    own argument slot, is exempt from the bound."""

    def base_args(self,span):
        return [span(GEN+4,0,4096),span(GEN+4,0,64),span(GEN+4,2048,2048)]

    def setUp(self):
        from engine import launch as parse, render
        from testkit import span
        self.tmp=tempfile.TemporaryDirectory(prefix='journal-proof-span-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        authority=patch('provenance.AUTHORITY',self.root/'private-authority')
        authority.start()
        self.addCleanup(authority.stop)
        init()
        self.repo=GitFixture(self.root/'fixture.git')
        self.base=fixtures()[0]
        self.item=parse(launch('chop_old'))
        self.item['args']=self.base_args(span)
        self.render=render
        self.base[STEP]=[render(self.item)]
        self.base_capture=self.capture('base',self.base)

    def capture(self,label,bodies):
        root=self.root/label
        revision=self.repo.commit(bodies)
        events=create_events(bodies[CREATE])
        create_end=len(events)
        events.extend(['note',line] for line in bodies[STEP])
        write_device(root,1,events)
        for name in bodies:
            checkpoint(root,name,[(1,events,create_end if name==CREATE else len(events))])
        return revision,seal_capture(root,revision,bodies)

    def prove(self,rule,args=None,grid=None):
        from engine import argument
        from testkit import SCHEMA
        changed=dict(self.item,args=self.item['args'] if args is None else [argument(a,[self.item]) for a in args])
        line=self.render(changed)
        if grid is not None:
            self.assertIn('grid_x: 32,',line)
            line=line.replace('grid_x: 32,',f'grid_x: {grid},')
        head=dict(self.base,**{STEP:[line]})
        label=f'head-{len(list(self.root.iterdir()))}'
        hr,hi=self.capture(label,head)
        spec=self.root/f'{label}.json'
        spec.write_text(json.dumps(dict(schema=SCHEMA,fixtures=[dict(name='host',create=CREATE,journals=[CREATE,STEP])],
            auxiliary_rules={'host':dict(added_resolves=[],added_modules=[],create_policy={})},journals={STEP:rule})))
        report=self.root/f'{label}-report.json'
        br,bi=self.base_capture
        with contextlib.redirect_stdout(io.StringIO()):
            code=main(['--repo',str(self.repo.repo),'--model',MODEL,'--base',br,'--head',hr,'--spec',str(spec),
                       '--base-capture-index',str(bi),'--head-capture-index',str(hi),'--report',str(report)])
        return code,json.loads(report.read_text())

    def refused(self,code,report):
        self.assertEqual(code,1,report['status'])
        self.assertIn('rewritten span exceeds',json.dumps(report['journals'][STEP]))

    def rule(self,args):
        return dict(launch_rules=[dict(from_symbol='chop_old',count=1,args=args)])

    def edit(self,replacements):
        """Argument expressions rewriting the first argument's text, the others kept."""
        return [dict(arg=0,replace=[{'from':a,'to':b} for a,b in replacements]),dict(arg=1),dict(arg=2)]

    def test_the_unchanged_oversized_span_is_kept(self):
        code,report=self.prove({})
        self.assertEqual(code,0,report['journals'][STEP])

    def test_the_unchanged_oversized_span_of_a_rewritten_launch_is_kept_in_place(self):
        code,report=self.prove(dict(launch_rules=[dict(from_symbol='chop_old',count=1,config={'grid_x':8})]),grid=8)
        self.assertEqual(code,0,report['journals'][STEP])



class ProofSpanTests(ProofSpanCase,unittest.TestCase):
    def test_an_oversized_span_copied_into_a_bounded_argument_is_refused(self):
        args=[dict(arg=0),dict(arg=0),dict(arg=2)]
        self.refused(*self.prove(dict(launch_rules=[dict(from_symbol='chop_old',count=1,args=args)]),args))

    def test_a_derived_span_that_another_argument_holds_is_refused(self):
        args=[dict(arg=0,subspan=2048),dict(arg=1),dict(arg=2)]
        self.refused(*self.prove(dict(launch_rules=[dict(from_symbol='chop_old',count=1,args=args)]),args))

    def test_a_new_derived_span_is_refused(self):
        args=[dict(arg=0,subspan=1024),dict(arg=1),dict(arg=2)]
        self.refused(*self.prove(dict(launch_rules=[dict(from_symbol='chop_old',count=1,args=args)]),args))


class ProofSpanScalarTests(ProofSpanCase,unittest.TestCase):
    """A scalar argument beside the oversized span (review C2 r4 re-check):
    moving the span past it, or shifting it by a new argument, changes its
    argument slot even though its ordinal among spans stays the same."""

    def base_args(self,span):
        return [span(GEN+4,0,4096),'ScalarU32(1)',span(GEN+4,0,64)]

    def test_an_oversized_span_swapped_past_a_scalar_is_refused(self):
        args=[dict(arg=1),dict(arg=0),dict(arg=2)]
        self.refused(*self.prove(self.rule(args),args))

    def test_an_oversized_span_shifted_by_a_prepended_scalar_is_refused(self):
        args=[dict(literal='ScalarU32(9)'),dict(arg=0),dict(arg=1),dict(arg=2)]
        self.refused(*self.prove(self.rule(args),args))

    def test_a_direct_buffer_wrapped_into_a_pointer_array_is_refused(self):
        # Review C2 r4 re-check 2: the argument's kind changed, its span did not.
        args=self.edit([('DeviceBuffer(','DevicePointerArray([Some('),('})','}), None])')])
        self.refused(*self.prove(self.rule(args),args))


class ProofSpanArrayTests(ProofSpanCase,unittest.TestCase):
    """The oversized span inside a pointer array, after an empty entry
    (review C2 r4 re-check 2): any change to the array, its empty entries or
    its kind bounds the span, though the span itself is unchanged."""

    def base_args(self,span):
        inner=span(GEN+4,0,4096)[len('DeviceBuffer('):-1]
        return ['DevicePointerArray([None, Some('+inner+')])','ScalarU32(1)',span(GEN+4,0,64)]

    def test_an_oversized_span_moved_past_an_empty_entry_is_refused(self):
        args=self.edit([('[None, Some(','[Some('),(')])','), None])')])
        self.refused(*self.prove(self.rule(args),args))

    def test_an_empty_entry_inserted_before_an_oversized_span_is_refused(self):
        args=self.edit([('[None, Some(','[None, None, Some(')])
        self.refused(*self.prove(self.rule(args),args))

    def test_a_pointer_array_unwrapped_to_a_direct_buffer_is_refused(self):
        args=self.edit([('DevicePointerArray([None, Some(','DeviceBuffer('),(')])',')')])
        self.refused(*self.prove(self.rule(args),args))

if __name__=='__main__':
    unittest.main()

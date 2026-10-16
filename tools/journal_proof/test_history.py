#!/usr/bin/env python3
"""Auxiliary history comparison: setup and history between pinned captures."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proof import main,SCHEMA
from provenance import init
from testkit import CREATE, GEN, MODEL, STEP, GitFixture, checkpoint, launch, seal_capture, write_device

LAYERS='chop_copy_layers_v1'
CONFIG='LaunchConfigV1 { flags: 1, grid_x: 2, grid_y: 2, grid_z: 1, block_x: 256, block_y: 1, block_z: 1, shared_memory_bytes: 0, cluster: LaunchClusterV1 { x: 1, y: 1, z: 1 } }'


def bare(raw,offset,length):
    return f'DeviceBufferSpanV1 {{ buffer: BundleDeviceBufferHandleV1 {{ raw: {raw} }}, offset: {offset}, length: {length} }}'


def layer_copy(source,destination,stride):
    return (f'launch 1 {LAYERS} {CONFIG} [DeviceBuffer({source}), DeviceBuffer({destination}), '
            f'ScalarU64({stride}), ScalarU64(64), ScalarU64(8)]')


class Lifetime:
    """One fixture device: create, history before the step, history inside
    the step (between its launches) and history after it. Base issue IDs:
    stream 1, module 2, resolve 3, allocations 4 and 5, event 6, graph 7. The
    head adds module 4 and resolve 5, so its later IDs move up by two."""

    def __init__(self,head,pool=64):
        self.head=head
        shift=2 if head else 0
        self.work,self.pool,self.event,self.graph=(GEN+4+shift,GEN+5+shift,6+shift,7+shift)
        self.pool_bytes=pool
        self.create=['stream 1','module 2 cubin_a','resolve 3 chop_a']
        if head:
            self.create+=['module 4 cubin_b','resolve 5 chop_b']
        self.create+=['allocate 64 256',f'allocate {pool} 256',f'event {self.event}']
        self.step=[launch('chop_a',self.work),launch('chop_keep',self.work)]
        stride=32 if pool==64 else pool//2
        self.before=[f'upload 1 {bare(self.work,0,64)} 00000000000000aa',
                     f'copy 1 {bare(self.pool,8,8)} {bare(self.work,0,8)}',
                     f'copy 1 {bare(self.pool,stride+8,8)} {bare(self.work,8,8)}',
                     layer_copy(bare(self.pool,0,pool),bare(self.work,0,64),stride),
                     'capture 1',launch('chop_keep',self.work),f'captured 1 {self.graph} 1',f'replay 1 {self.graph}',
                     f'record 1 {self.event}',f'wait 1 {self.event}','synchronize 1']
        self.inside=[f'record 1 {self.event}']
        self.after=[f'release graph {self.graph}',f'download 1 {bare(self.work,0,8)}','synchronize 1']

    def bodies(self):
        return {CREATE:list(self.create),STEP:list(self.step)}

    def events(self):
        ev=[]
        shift=2 if self.head else 0
        def issue(i,*row):
            ev.append(['id',i])
            if row:
                ev.append(list(row))
        issue(1,'stream',1,0)
        ev.append(['note',self.create[0]])
        issue(2,'module',2,'cubin_a','hash_a',100)
        ev.append(['note',self.create[1]])
        issue(3,'resolve',3,2,'chop_a')
        ev.append(['note',self.create[2]])
        if self.head:
            issue(4,'module',4,'cubin_b','hash_b',120)
            ev.append(['note',self.create[3]])
            issue(5,'resolve',5,4,'chop_b')
            ev.append(['note',self.create[4]])
        issue(4+shift,'allocation',self.work,0,64,256,GEN+1)
        ev.append(['note','allocate 64 256'])
        issue(5+shift,'allocation',self.pool,0,self.pool_bytes,256,GEN+1)
        ev.append(['note',f'allocate {self.pool_bytes} 256'])
        issue(6+shift)
        ev.append(['note',f'event {self.event}'])
        ev.append(['created',0,0,len(ev),len(self.create)])
        create_records=len(ev)
        for line in self.before:
            if line.startswith('captured'):
                issue(7+shift)
            ev.append(['note',line])
        ev.append(['note',self.step[0]])
        ev.extend(['note',line] for line in self.inside)
        ev.append(['note',self.step[1]])
        ev.extend(['note',line] for line in self.after)
        return ev,create_records


def rule(resized=False):
    out=dict(added_resolves=[dict(symbol='chop_b',ids=[5],count=1)],
             added_modules=[dict(name='cubin_b',fnv1a='hash_b',bytes=120,ids=[4],count=1)],
             create_policy=dict(added_modules=[dict(symbol='cubin_b',ids=[4],count=1,positions=[4])],
                                added_resolves=[dict(symbol='chop_b',ids=[5],count=1,positions=[5])]))
    if resized:
        out['allocation_resizes']=[dict(base_raw=GEN+5,head_raw=GEN+7,rank=0,**{'from':64,'to':96},alignment=256)]
        out['allocation_bindings']={'pool':dict(base_raw=GEN+5,head_raw=GEN+7,rank=0,base_bytes=64,head_bytes=96,stream_raw=GEN+1)}
        out['allocation_layouts']={'pool':dict(planes=2,from_stride=32,to_stride=48,from_tail=0,to_tail=0)}
        out['create_policy']['state_sizes']=[dict(line=5,**{'from':'allocate 64 256','to':'allocate 96 256'})]
        out['history_policy']=dict(
            layout_spans=[dict(allocation='pool',form='within_plane',from_offset=40,from_length=8,to_offset=56,to_length=8,count=1),
                          dict(allocation='pool',form='allocation',from_offset=0,from_length=64,to_offset=0,to_length=96,count=1)],
            layout_strides=[dict(symbol=LAYERS,span_arg=0,stride_arg=2,allocation='pool',count=1)])
    return out


class Harness(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='journal-proof-history-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        authority=patch('provenance.AUTHORITY',self.root/'private-authority')
        authority.start()
        self.addCleanup(authority.stop)
        init()
        self.repo=GitFixture(self.root/'fixture.git')
        self.base=Lifetime(False)
        self.head=Lifetime(True)
        self.spec=dict(schema=SCHEMA,fixtures=[dict(name='host',create=CREATE,journals=[CREATE,STEP])],
                       journals={CREATE:copy.deepcopy(rule()['create_policy'])},auxiliary_rules={'host':rule()})

    def capture(self,label,lifetime):
        root=self.root/label
        bodies=lifetime.bodies()
        rev=self.repo.commit(bodies)
        events,create_records=lifetime.events()
        write_device(root,1,events)
        for name in bodies:
            checkpoint(root,name,[(1,events,create_records if name==CREATE else len(events))])
        return rev,seal_capture(root,rev,bodies)

    def check(self,expected):
        br,bi=self.capture('base',self.base)
        hr,hi=self.capture('head',self.head)
        spec=self.root/'spec.json'
        spec.write_text(json.dumps(self.spec))
        report=self.root/'report.json'
        with contextlib.redirect_stdout(io.StringIO()):
            code=main(['--repo',str(self.repo.repo),'--model',MODEL,'--base',br,'--head',hr,'--spec',str(spec),
                       '--base-capture-index',str(bi),'--head-capture-index',str(hi),'--report',str(report)])
        value=json.loads(report.read_text())
        self.assertEqual(code,expected,json.dumps(value.get('fixtures',value),indent=1)[:4000])
        return value

    def refused(self,pattern):
        report=self.check(1)
        history=report['fixtures']['host']['history']
        self.assertEqual(history['status'],'FAILED')
        self.assertRegex(history['error'],pattern)
        self.assertTrue(all(v['verdict']=='UNPROVEN' for v in report['journals'].values()))
        return report

class HistoryTests(Harness):
    def test_unchanged_history_is_compared_under_the_mapping(self):
        report=self.check(0)
        history=report['fixtures']['host']['history']
        self.assertEqual(history['status'],'COMPARED')
        self.assertEqual(history['head_operations'],15)
        self.assertEqual(history['operation_kinds'],{'upload':1,'copy':2,'launch':2,'capture':1,'captured':1,'replay':1,
                                                     'record':2,'wait':1,'synchronize':2,'release':1,'download':1})
        self.assertEqual(history['trailing_uncompared_notes'],{'base':0,'head':0})
        self.assertGreater(history['ordered_operations'],20)
        self.assertIn('auxiliary_history',json.dumps(report['claim']))
        self.assertEqual(report['auxiliary_history_status'],{'host':'COMPARED'})

    def test_faulty_history_layer_stride_is_refused(self):
        # The review's control: a layer copy's source stride changes between
        # pinned captures; every pinned body is unchanged.
        self.head.before[3]=self.head.before[3].replace('ScalarU64(32)','ScalarU64(24)',1)
        self.refused('unexplained operation differences')

    def test_extra_history_launch_is_refused(self):
        self.head.before.insert(4,self.head.before[3])
        self.refused('unexplained operation differences')

    def test_missing_history_launch_is_refused(self):
        del self.head.before[3]
        self.refused('unexplained operation differences')

    def test_changed_upload_payload_is_refused(self):
        self.head.before[0]=self.head.before[0].replace('00000000000000aa','00000000000000ab')
        self.refused('unexplained operation differences')

    def test_declared_upload_payload_change_is_proven(self):
        self.head.before[0]=self.head.before[0].replace('00000000000000aa','00000000000000ab')
        self.spec['auxiliary_rules']['host']['history_policy']=dict(
            upload_payloads=[dict(line=1,**{'from':'00000000000000aa','to':'00000000000000ab'})])
        report=self.check(0)
        self.assertEqual(report['auxiliary_history_counts']['upload_payloads'],1)

    def test_changed_copy_size_is_refused(self):
        self.head.before[1]=f'copy 1 {bare(self.head.pool,8,16)} {bare(self.head.work,0,16)}'
        self.refused('unexplained operation differences')

    def test_history_moved_into_a_pinned_capture_is_refused(self):
        # The history alone keeps its order; only its place against the
        # pinned step launches changes.
        self.head.inside.insert(0,self.head.before.pop())
        self.refused('operation order changed')

    def test_reordered_history_is_refused(self):
        self.head.before[1],self.head.before[2]=self.head.before[2],self.head.before[1]
        self.refused('unexplained operation differences')

    def test_history_moved_across_a_step_launch_is_refused(self):
        self.head.after.insert(0,self.head.inside.pop())
        self.refused('operation order changed')

    def test_unmapped_event_id_is_refused(self):
        self.head.inside[0]=f'record 1 {self.base.event}'
        self.refused('unexplained operation differences|collides')

    def test_graph_node_count_must_match_its_capture(self):
        self.head.before[6]=f'captured 1 {self.head.graph} 2'
        self.refused('graph has 2 nodes but its capture recorded 1 launches')

    def removed_captured_launch(self,count=None):
        del self.head.before[5]
        self.head.before[5]=f'captured 1 {self.head.graph} 0'
        policy=dict(removals=[dict(line=6,symbol='chop_keep')])
        if count:
            policy['graph_nodes']=[dict(count=count)]
        self.spec['auxiliary_rules']['host']['history_policy']=policy

    def test_declared_graph_node_change_is_proven(self):
        self.removed_captured_launch(1)
        report=self.check(0)
        self.assertEqual(report['auxiliary_history_counts']['graph_nodes'],1)
        self.assertEqual(report['auxiliary_history_counts']['removals'],1)

    def test_undeclared_graph_node_change_is_refused(self):
        self.removed_captured_launch()
        self.refused('unexplained operation differences')

    def test_graph_node_change_count_is_exact(self):
        self.removed_captured_launch(2)
        self.refused('1 graph node-count changes, expected 2')

    def test_unknown_operation_form_is_refused(self):
        self.head.after.append('teleport 1 everything')
        self.refused('unrecognized auxiliary operation form')

    def test_layout_rules_explain_a_resized_pool(self):
        self.head=Lifetime(True,pool=96)
        self.spec['auxiliary_rules']['host']=rule(True)
        self.spec['journals'][CREATE]['state_sizes']=[dict(line=5,**{'from':'allocate 64 256','to':'allocate 96 256'})]
        report=self.check(0)
        self.assertEqual(report['auxiliary_history_counts']['layout_spans'],2)
        self.assertEqual(report['auxiliary_history_counts']['layout_strides'],1)

    def resized(self):
        self.head=Lifetime(True,pool=96)
        self.spec['auxiliary_rules']['host']=rule(True)
        self.spec['journals'][CREATE]['state_sizes']=[dict(line=5,**{'from':'allocate 64 256','to':'allocate 96 256'})]
        return self.spec['auxiliary_rules']['host']

    def test_layout_span_must_follow_its_layout(self):
        self.resized()['history_policy']['layout_spans'][0]['to_offset']=48
        report=self.check(1)
        self.assertRegex(report['fixtures']['host']['history']['error'],'maps to')

    def test_layout_stride_without_its_rule_is_refused(self):
        del self.resized()['history_policy']['layout_strides']
        self.refused('unexplained operation differences')

    def test_layout_stride_count_is_exact(self):
        self.resized()['history_policy']['layout_strides'][0]['count']=2
        self.refused('layout stride changes, expected 2')

    def test_layout_must_tile_the_resized_allocation(self):
        self.resized()['allocation_layouts']['pool']['to_stride']=40
        report=self.check(1)
        self.assertTrue(all(v['verdict']=='UNPROVEN' for v in report['journals'].values()))
        self.assertIn('does not tile',json.dumps(report['journals']))

    def test_layout_requires_a_declared_resize(self):
        rule_=self.resized()
        rule_['allocation_bindings']['work']=dict(base_raw=GEN+4,head_raw=GEN+6,rank=0,base_bytes=64,head_bytes=64,stream_raw=GEN+1)
        rule_['allocation_layouts']['work']=dict(planes=2,from_stride=32,to_stride=32,from_tail=0,to_tail=0)
        report=self.check(1)
        self.assertIn('not tied to a declared allocation resize',json.dumps(report['journals']))

    def test_undeclared_history_span_change_is_refused(self):
        self.resized()['history_policy']['layout_spans'].pop(0)
        self.refused('unexplained operation differences')


class ReviewTests(Harness):
    """Refusal tests reproducing two review findings of 2026-10-11."""

    def upload_stream(self,base,head):
        self.base.before[0]=self.base.before[0].replace('upload 1 ',f'upload {base} ',1)
        self.head.before[0]=self.head.before[0].replace('upload 1 ',f'upload {head} ',1)

    def test_unobserved_stream_cannot_borrow_an_event_mapping(self):
        # Finding 1: no stream creation for 6; event 6 becomes 8 in the head.
        self.upload_stream(6,8)
        self.refused('stream 6 has no recorded creation')

    def test_unobserved_stream_sharing_an_issued_number_is_refused(self):
        self.upload_stream(6,6)
        self.refused('stream 6 has no recorded creation')

    def test_unobserved_unissued_stream_stays_fixed(self):
        self.upload_stream(3000,3000)
        report=self.check(0)
        self.assertEqual(report['fixtures']['host']['history']['status'],'COMPARED')

    def test_unobserved_unissued_stream_cannot_change(self):
        self.upload_stream(3000,3001)
        self.refused('unexplained operation differences')

    def test_unissued_event_is_refused(self):
        self.base.inside[0]='record 1 900'
        self.head.inside[0]='record 1 900'
        self.refused('ID 900 was not issued')

    def test_full_proof_reports_full_coverage(self):
        report=self.check(0)
        self.assertEqual(report['coverage']['history_compared'],1)
        self.assertTrue(report['coverage']['full_history_proof'])
        self.assertIn('1 of 1 fixtures',report['claim']['compared']['auxiliary_history'])

    def declared_only(self,rule):
        self.head=Lifetime(False)
        old={'toy-old-create.txt':list(self.base.create),'toy-old-step.txt':list(self.base.step)}
        new={'toy-new-create.txt':list(self.head.create),'toy-new-step.txt':list(self.head.step)}
        self.base.bodies=lambda:old
        self.head.bodies=lambda:new
        self.spec=dict(schema=SCHEMA,fixtures=[dict(name='removed',create='toy-old-create.txt',journals=list(old)),
                                               dict(name='added',create='toy-new-create.txt',journals=list(new))],
                       journals={},added_journals=list(new),removed_journals=list(old),
                       auxiliary_rules={'removed':copy.deepcopy(rule),'added':copy.deepcopy(rule)})

    def test_declared_only_fixtures_must_carry_valid_empty_rules(self):
        # Finding 4: unvalidated rules on fixtures that compare nothing.
        self.declared_only({'unknown_and_never_validated':True})
        self.check(2)

    def test_declared_only_fixtures_cannot_declare_changes(self):
        self.declared_only(dict(added_resolves=[],added_modules=[],create_policy={'state_sizes':[]},
                                history_policy={'removals':[dict(line=1,symbol='chop_x')]}))
        self.check(2)

    def test_declared_only_fixtures_claim_no_comparison(self):
        self.declared_only(dict(added_resolves=[],added_modules=[],create_policy={}))
        report=self.check(0)
        coverage=report['coverage']
        self.assertEqual(coverage['history_compared'],0)
        self.assertFalse(coverage['full_history_proof'])
        self.assertEqual(coverage['declared_only_fixtures'],['added','removed'])
        self.assertEqual(report['claim']['compared'],{})
        self.assertIn('No fixture history was compared',' '.join(report['claim']['not_compared']))
        self.assertIn('not a full history proof',' '.join(report['claim']['not_compared']))


GRAPHS='crates/bundle-runtime/src/graphs.rs'


def graphs_source(bounds='[0, launches / 4, launches / 2, launches]',body=None,extra=''):
    """A graphs.rs in the shape of the bundle runtime's: the split
    function and the capture loop that calls it."""
    body=body or ('    let bounds = '+bounds+';\n    bounds\n        .windows(2)\n        .map(|pair| pair[0]..pair[1])\n'
                  '        .filter(|range| !range.is_empty())\n        .collect()\n')
    return ('use std::ops::Range;\n\n/// Launch ranges of a program\'s graph pieces.\n'
            'pub fn pieces(launches: usize) -> Vec<Range<usize>> {\n'+body+'}\n\n'+extra+
            'pub fn capture(launches: usize, mut record: impl FnMut(Range<usize>)) {\n'
            '    for range in pieces(launches) {\n        record(range); // one graph per piece\n    }\n}\n')


class Pieces:
    """A step journal of eight launches captured as one graph program of
    three pieces under a [0, L/4, L/2, L] split. The head's spec removes the
    fifth launch, so the policy moves the second launch into piece two."""

    def __init__(self,head,sizes=None,source='[0, launches / 4, launches / 2, launches]'):
        self.head=head
        self.source=source
        self.create=['stream 1','module 2 cubin_a','resolve 3 chop_a','allocate 64 256']
        launches=[launch(f'chop_l{i}') for i in range(1,9)]
        if head:
            launches.pop(4)
        self.step=launches
        self.sizes=sizes or ([1,2,4] if head else [2,2,4])

    def bodies(self):
        return {CREATE:list(self.create),STEP:list(self.step)}

    def events(self):
        ev=[]
        def issue(i,*row):
            ev.append(['id',i])
            ev.append(list(row))
        issue(1,'stream',1,0)
        ev.append(['note',self.create[0]])
        issue(2,'module',2,'cubin_a','hash_a',100)
        ev.append(['note',self.create[1]])
        issue(3,'resolve',3,2,'chop_a')
        ev.append(['note',self.create[2]])
        issue(4,'allocation',GEN+4,0,64,256,GEN+1)
        ev.append(['note',self.create[3]])
        ev.append(['created',0,0,len(ev),len(self.create)])
        create_records=len(ev)
        at=0
        for graph,size in enumerate(self.sizes):
            ev.append(['note','capture 1'])
            ev.extend(['note',line] for line in self.step[at:at+size])
            at+=size
            ev.append(['id',5+graph])
            ev.append(['note',f'captured 1 {5+graph} {size}'])
        for graph in range(len(self.sizes)):
            ev.append(['note',f'replay 1 {5+graph}'])
        return ev,create_records


class PieceTests(Harness):
    def setUp(self):
        super().setUp()
        self.base=Pieces(False)
        self.head=Pieces(True)
        self.spec=dict(schema=SCHEMA,fixtures=[dict(name='host',create=CREATE,journals=[CREATE,STEP])],
                       journals={STEP:dict(removals=[dict(line=5,symbol='chop_l5')])},
                       graph_piece_policy=dict(source=GRAPHS,divisors=[4,2]),
                       auxiliary_rules={'host':dict(added_resolves=[],added_modules=[],create_policy={},
                                                    history_policy=dict(graph_nodes=[dict(count=1)],graph_pieces=[dict(count=1)]))})
        commit=self.repo.commit
        self.repo.commit=lambda bodies,lifetime=None:commit(bodies,extra_files={GRAPHS:(
            getattr(self.lifetime,'graphs',None) or graphs_source(self.lifetime.source)).encode()})

    def capture(self,label,lifetime):
        self.lifetime=lifetime
        return super().capture(label,lifetime)

    def policy(self):
        return self.spec['auxiliary_rules']['host']['history_policy']

    def test_policy_regrouping_is_proven(self):
        report=self.check(0)
        self.assertEqual(report['auxiliary_history_counts']['graph_pieces'],1)
        self.assertEqual(report['auxiliary_history_counts']['graph_nodes'],1)
        self.assertEqual(report['graph_piece_policy']['divisors'],[4,2])

    def test_the_graph_source_before_the_runtime_rename_is_read(self):
        old='crates/bundle-runtime-sm120/src/graphs.rs'
        commit=GitFixture.commit
        self.repo.commit=lambda bodies,lifetime=None:commit(self.repo,bodies,extra_files={old:graphs_source(self.lifetime.source).encode()})
        self.spec['graph_piece_policy']['source']=old
        report=self.check(0)
        self.assertEqual(report['graph_piece_policy']['source'],old)
        self.assertEqual(report['auxiliary_history_counts']['graph_pieces'],1)

    def test_regrouping_without_declaration_is_refused(self):
        del self.policy()['graph_pieces']
        self.refused('operation order changed')

    def test_pieces_off_policy_are_refused(self):
        self.head.sizes=[2,1,4]
        self.refused('do not follow the split policy')

    def test_resplit_count_is_exact(self):
        self.policy()['graph_pieces'][0]['count']=2
        self.refused('1 graph programs re-split, expected 2')

    def test_policy_must_match_both_revisions_source(self):
        self.head.source='[0, launches / 4, launches / 3, launches]'
        self.check(2)

    def test_declared_policy_must_match_source(self):
        self.spec['graph_piece_policy']['divisors']=[4,3]
        self.check(2)

    def test_policy_claim_names_the_piece_exception(self):
        report=self.check(0)
        self.assertEqual(report['coverage']['graph_piece_exception'],['host'])
        self.assertIn('Exception: in fixtures host',report['claim']['compared']['ordering'])

    def replays(self,lines,both=True):
        for lifetime in ((self.base,self.head) if both else (self.head,)):
            events=lifetime.events
            def edited(events=events,head=lifetime.head):
                ev,records=events()
                ev=[e for e in ev if not(e[0]=='note' and e[1].startswith('replay '))]
                shift=0 if not head else 0
                return ev+[['note',line] for line in lines],records
            lifetime.events=edited

    def test_partial_program_replay_is_refused(self):
        # Finding 2: both sides replay only their first piece.
        self.replays(['replay 1 5'])
        self.refused('not replayed whole and in order')

    def test_repeated_piece_replay_is_refused(self):
        self.replays(['replay 1 5','replay 1 6','replay 1 6','replay 1 7'])
        self.refused('not replayed whole and in order')

    def test_reordered_piece_replay_is_refused(self):
        self.replays(['replay 1 6','replay 1 5','replay 1 7'])
        self.refused('not replayed whole and in order')

    def test_operation_between_piece_replays_is_refused(self):
        self.replays(['replay 1 5','synchronize 1','replay 1 6','replay 1 7'])
        self.refused('not replayed whole and in order')

    def test_repeated_whole_program_replay_is_proven(self):
        self.replays(['replay 1 5','replay 1 6','replay 1 7','synchronize 1','replay 1 5','replay 1 6','replay 1 7'])
        self.check(0)

    def zero_in_program(self,move):
        zero='zero 1 '+bare(GEN+4,0,8)
        self.base.step.insert(1,zero)
        self.head.step.insert(1,zero)
        self.base.sizes=[3,2,4]
        self.head.sizes=[2,2,4]
        self.spec['journals'][STEP]['removals'][0]['line']=6
        for lifetime in (self.base,self.head):
            events=lifetime.events
            def nodes_only(events=events,head=lifetime.head):
                ev,records=events()
                inside=False
                launches=0
                for e in ev:
                    if e[0]!='note':
                        continue
                    if e[1].startswith('capture '):
                        inside=True
                        launches=0
                    elif e[1].startswith('launch ') and inside:
                        launches+=1
                    elif e[1].startswith('captured '):
                        parts=e[1].split()
                        e[1]=' '.join(parts[:3]+[str(launches)])
                        inside=False
                if head and move:
                    z=next(i for i,e in enumerate(ev) if e==['note',zero])
                    item=ev.pop(z)
                    end=next(i for i,e in enumerate(ev) if e[0]=='note' and e[1].startswith('captured 1 5 '))
                    ev.insert(end+2,item)
                return ev,records
            lifetime.events=nodes_only

    def test_pinned_zero_moved_between_pieces_is_refused(self):
        # Finding 2: a pinned non-launch operation moved across an inner boundary.
        self.zero_in_program(True)
        self.refused('captures a non-launch operation')

    def test_resplit_program_with_a_non_launch_is_refused(self):
        self.zero_in_program(False)
        self.refused('captures a non-launch operation')

    def test_unrecognized_head_source_is_refused(self):
        # Finding 3 through the CLI: the head reverses the ranges.
        self.head.graphs=graphs_source().replace('.collect()','.rev()\n        .collect()')
        self.check(2)

    def test_string_policy_source_is_refused_by_the_cli(self):
        self.head.graphs=string_policy_source()
        self.check(2)

    def test_caller_modulo_assignment_is_refused_by_the_cli(self):
        self.head.graphs=caller_mutation('launches %= 4;')
        self.check(2)

    def test_caller_shift_assignment_is_refused_by_the_cli(self):
        self.head.graphs=caller_mutation('launches <<= 1;')
        self.check(2)

    def eager_after_program(self,move):
        eager=launch('chop_eager')
        for lifetime in (self.base,self.head):
            lifetime.step.append(eager)
            events=lifetime.events
            def after_program(events=events,head=lifetime.head):
                ev,records=events()
                if head and move:
                    ev.extend([['note',eager],['note','synchronize 1']])
                else:
                    ev.extend([['note','synchronize 1'],['note',eager]])
                return ev,records
            lifetime.events=after_program

    def test_failed_order_after_a_piece_exception_reports_normally(self):
        # A review finding of 2026-10-11: a genuine order failure in the only
        # fixture, after its program was re-split, was INVALID INPUT.
        self.eager_after_program(True)
        report=self.refused('operation order changed')
        self.assertEqual(report['status'],'NOT PROVEN')
        self.assertEqual(report['coverage']['history_compared'],0)
        self.assertEqual(report['claim']['compared'],{})
        self.assertIn('Graph piece exception applied in fixtures host',' '.join(report['claim']['not_compared']))

    def test_order_control_after_a_piece_exception_is_proven(self):
        self.eager_after_program(False)
        report=self.check(0)
        self.assertIn('Exception: in fixtures host',report['claim']['compared']['ordering'])

    def test_claim_with_no_comparison_and_a_piece_exception(self):
        from proof import claim
        coverage=dict(fixtures=1,history_compared=0,history_not_compared=['host'],text_fixtures=[],declared_only_fixtures=[],
                      declared_journals=dict(added=[],removed=[]),graph_piece_exception=['host'],full_history_proof=False)
        value=claim(coverage)
        self.assertEqual(value['compared'],{})
        self.assertIn('no ordering was established',' '.join(value['not_compared']))

    def test_operation_between_pieces_is_refused(self):
        events=self.head.events
        def with_sync():
            ev,records=events()
            at=next(i for i,e in enumerate(ev) if e==['note','captured 1 5 1'])
            ev.insert(at+1,['note','synchronize 1'])
            return ev,records
        self.head.events=with_sync
        self.refused('unexplained operation differences')


def string_policy_source():
    """A review case of 2026-10-11: the recognized code only inside an unused raw string;
    the exported pieces returns the whole program."""
    return ('use std::ops::Range;\nconst UNUSED_TEMPLATE: &str = r###"'+graphs_source()+'"###;\n'
            'pub fn actual_pieces(launches: usize) -> Vec<Range<usize>> { vec![0..launches] }\n'
            'pub use actual_pieces as pieces;\n'
            'pub fn capture_actual(launches: usize, mut record: impl FnMut(Range<usize>)) {\n'
            '    for range in [0..launches] { record(range); }\n}\n')


def caller_mutation(statement):
    return graphs_source().replace('pub fn capture(launches: usize','pub fn capture(mut launches: usize').replace(
        '    for range in pieces(launches) {',f'    {statement}\n    for range in pieces(launches) {{')


class SourceTests(unittest.TestCase):
    """Finding 3: the piece policy is a recognized implementation, not a line."""

    def divisors(self,text):
        from history import piece_divisors
        return piece_divisors(text)

    def refused(self,text):
        from common import Invalid
        with self.assertRaises(Invalid):
            self.divisors(text)

    def test_recognized_source(self):
        self.assertEqual(self.divisors(graphs_source()),[4,2])

    def test_unrelated_bounds_line_does_not_count(self):
        extra='fn other(launches: usize) -> usize {\n    let bounds = [0, launches / 9, launches / 3, launches];\n    bounds[1]\n}\n\n'
        self.assertEqual(self.divisors(graphs_source(extra=extra)),[4,2])

    def test_reversed_ranges_are_refused(self):
        self.refused(graphs_source().replace('.collect()','.rev()\n        .collect()'))

    def test_shadowed_launch_count_is_refused(self):
        body=('    let launches = launches + 32;\n    let bounds = [0, launches / 4, launches / 2, launches];\n'
              '    bounds\n        .windows(2)\n        .map(|pair| pair[0]..pair[1])\n        .filter(|range| !range.is_empty())\n        .collect()\n')
        self.refused(graphs_source(body=body))

    def test_shadowing_in_the_caller_is_refused(self):
        self.refused(graphs_source().replace('    for range in pieces(launches) {','    let launches = launches + 32;\n    for range in pieces(launches) {'))

    def test_bounds_only_in_a_comment_are_refused(self):
        self.refused('/*\n    let bounds = [0, launches / 4, launches / 2, launches];\n*/\n'
                     'pub fn pieces(launches: usize) -> Vec<Range<usize>> { vec![0..launches] }\n'
                     'pub fn capture(launches: usize) { for range in pieces(launches) { let _ = range; } }\n')

    def test_commented_out_implementation_is_refused(self):
        text=graphs_source()
        start=text.index('pub fn pieces')
        end=text.index('pub fn capture')
        self.refused(text[:start]+'/* '+text[start:end]+' */\npub fn pieces(launches: usize) -> Vec<Range<usize>> { vec![0..launches] }\n'+text[end:])

    def test_missing_capture_call_is_refused(self):
        self.refused(graphs_source().replace('for range in pieces(launches)','for range in [0..launches]'))

    def test_code_inside_a_string_is_refused(self):
        self.refused(string_policy_source())

    def test_code_inside_plain_strings_is_refused(self):
        quoted=graphs_source().replace('\\','\\\\').replace('"','\\"').replace('\n','\\n')
        self.refused('const A: &str = "'+quoted+'";\npub fn pieces(launches: usize) -> Vec<Range<usize>> { vec![0..launches] }\n')

    def test_strings_elsewhere_are_opaque(self):
        text=graphs_source(extra='const NOTE: &str = "for range in [0..launches] { /* not code */ }";\nconst C: char = \'"\';\n\n')
        self.assertEqual(self.divisors(text),[4,2])

    def test_caller_modulo_assignment_is_refused(self):
        self.refused(caller_mutation('launches %= 4;'))

    def test_caller_shift_assignment_is_refused(self):
        self.refused(caller_mutation('launches <<= 1;'))

    def test_every_compound_assignment_is_refused(self):
        for op in ('=','+=','-=','*=','/=','%=','<<=','>>=','&=','|=','^='):
            with self.subTest(op=op):
                self.refused(caller_mutation(f'launches {op} 1;'))

    def test_mutable_parameter_alone_is_refused(self):
        self.refused(graphs_source().replace('pub fn capture(launches: usize','pub fn capture(mut launches: usize'))

    def test_mutable_borrow_is_refused(self):
        self.refused(graphs_source().replace('    for range in pieces(launches) {','    bump(&mut launches);\n    for range in pieces(launches) {'))

    def test_alias_is_refused(self):
        self.refused(graphs_source()+'\npub use pieces as split;\npub use other as pieces;\n')

    def test_comparison_is_not_an_assignment(self):
        text=graphs_source().replace('    for range in pieces(launches) {','    assert!(launches <= 1 << 30 && launches >= 0 && launches != 7 && launches == launches);\n    for range in pieces(launches) {')
        self.assertEqual(self.divisors(text),[4,2])

    def test_capture_loop_must_take_launches(self):
        self.refused(graphs_source().replace('pub fn capture(launches: usize','pub fn capture(count: usize'))

    def test_second_definition_is_refused(self):
        self.refused(graphs_source()+'\nmod alt {\n    pub fn pieces(n: usize) -> usize { n }\n}\n')



class MapperTests(unittest.TestCase):
    """history.Mapper maps every typed identity of a line's form, for the
    history (closed) and for create and pinned bodies (not closed)."""

    def mapper(self,closed):
        import history
        return history.Mapper({4294967300:4294967310},{5:7},{2:3},{2:3,5:7,20:24,30:31},closed=closed)

    def test_record_wait_and_graph_identities_are_mapped_in_every_owner(self):
        for closed in (True,False):
            m=self.mapper(closed)
            self.assertEqual([m(l) for l in ('record 2 20','wait 2 20','captured 2 30 4','replay 2 30','release graph 30','event 20')],
                             ['record 3 24','wait 3 24','captured 3 31 4','replay 3 31','release graph 31','event 24'])

    def view(self,context,buffer):
        return ('verify view BundleLogitsViewV1 { struct_size: 80, reserved: 0, context: BundleContextHandleV1 { raw: '
                f'{context} }}, rank: 0, element_type: 1, buffer: BundleDeviceBufferViewV1 {{ struct_size: 32, reserved: 0, '
                f'buffer: BundleDeviceBufferHandleV1 {{ raw: {buffer} }}, offset: 0, length: 64 }}, row_stride_elements: 16, '
                'row_count: 1, live_width: 16, first_item_index: 0, reserved_tail: 0 }')

    def test_journal_bodies_admit_only_the_declared_annotations(self):
        from engine import Unproven
        body=self.mapper(False)
        for line in ('step prefill','step captures and runs','cancelled verify drafts OwnedDraftTokensV1 { request_id: 7, tokens: [44, 45] }'):
            self.assertEqual(body(line),line)
        # Only a view's buffer handle maps; its context handle, which has the
        # same raw form, stays.
        self.assertEqual(body(self.view(4294967300,4294967300)),self.view(4294967300,4294967310))
        for line in ('step prefill raw: 4294967300','note 5','readback 2 x 4','step {7}','prefill view BundleLogitsViewV1 { }'):
            with self.assertRaisesRegex(Unproven,'unrecognized journal line form'):
                body(line)
            with self.assertRaisesRegex(Unproven,'unrecognized auxiliary operation form'):
                self.mapper(True)(line)

    def test_readback_and_peer_copy_streams_are_typed(self):
        span='DeviceBufferSpanV1 { buffer: BundleDeviceBufferHandleV1 { raw: 4294967300 }, offset: 0, length: 4 }'
        mapped=span.replace('4294967300','4294967310')
        for closed in (True,False):
            m=self.mapper(closed)
            self.assertEqual(m(f'readback 2 {span} 4'),f'readback 3 {mapped} 4')
            self.assertEqual(m(f'peer copy 2 1 {span} 2 {span}'),f'peer copy 3 1 {mapped} 2 {mapped}')

    def test_a_fixed_stream_may_not_be_the_image_of_a_created_stream(self):
        from engine import Unproven
        # Created stream 2 maps to 3; fixed stream 3 would alias it.
        for closed in (True,False):
            with self.assertRaisesRegex(Unproven,'stream 3 has no recorded creation and is the image of a created stream'):
                self.mapper(closed)('synchronize 3')
        # Stream 5 shares a number with function 5, which is not a stream.
        with self.assertRaisesRegex(Unproven,'stream 5 has no recorded creation and shares its number'):
            self.mapper(True)('synchronize 5')
        self.assertEqual(self.mapper(False)('synchronize 5'),'synchronize 5')
        self.assertEqual(self.mapper(False)('synchronize 40'),'synchronize 40')
        with self.assertRaisesRegex(Unproven,'issued ID 21 was not issued by this device'):
            self.mapper(False)('record 2 21')

    def test_every_stream_operation_breaks_a_whole_replay(self):
        import history
        span='DeviceBufferSpanV1 { buffer: BundleDeviceBufferHandleV1 { raw: 4294967300 }, offset: 0, length: 4 }'
        self.assertIsNone(history.whole_replays(['replay 3 40','replay 3 41'],['40','41']))
        # A readback or a peer copy on the replay stream between two pieces
        # is an operation between them.
        for line in (f'readback 3 {span} 4',f'peer copy 3 1 {span} 2 {span}',f'download 3 {span}'):
            self.assertEqual(history.whole_replays(['replay 3 40',line,'replay 3 41'],['40','41']),1)

    def test_an_unmapped_function_id_is_refused(self):
        from engine import Unproven
        for closed in (True,False):
            self.assertEqual(self.mapper(closed)('configure 5 64'),'configure 7 64')
            with self.assertRaisesRegex(Unproven,'function ID 6 has no mapping'):
                self.mapper(closed)('configure 6 64')


if __name__=='__main__':
    unittest.main()

#!/usr/bin/env python3
"""Lock-authenticated replacement, expansion and allocation-bound rewrite tests."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common import fnv
from proof import SCHEMA,main
from provenance import init
from testkit import CREATE, GEN, MODEL, ROOT, STEP, GitFixture, checkpoint, launch, seal_capture, span, write_device

LOCK=ROOT+'kernels.lock.d/example.lock'
OLD=b'old cubin bytes'
NEW=b'new cubin bytes'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def line(symbol,args,grid=32):
    template=launch(symbol,GEN+5,grid=grid)
    return template[:template.index(' [')+1]+'['+', '.join(args)+']'


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='jproof-extension-test-')
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        p=patch('provenance.AUTHORITY',self.root/'authority')
        p.start()
        self.addCleanup(p.stop)
        init()
        self.git=GitFixture(self.root/'objects.git')
        self.base={CREATE:['stream 1','module 2 cube','resolve 3 chop_aa','configure 3 64',
                           'resolve 4 chop_mm','configure 4 128',f'allocate 64 256 -> raw: {GEN+5}',
                           f'allocate 128 256 -> raw: {GEN+6}'],
                   STEP:[line('chop_aa',[span(GEN+5),'ScalarU32(1)'])]}
        self.head={CREATE:['stream 1','module 2 cube','resolve 3 chop_mm','configure 3 128',
                           'resolve 4 chop_zz','configure 4 64',f'allocate 64 256 -> raw: {GEN+5}',
                           f'allocate 128 256 -> raw: {GEN+6}'],
                   STEP:[line('chop_zz',[span(GEN+5),'ScalarU32(1)'])]}
        self.spec=dict(schema=SCHEMA,fixtures=[dict(name='fixture',create=CREATE,journals=[CREATE,STEP])],
          journals={STEP:dict(symbol_substitutions=[dict(**{'from':'chop_aa','to':'chop_zz'},count=1)])},
          auxiliary_rules={'fixture':dict(added_modules=[],added_resolves=[],create_policy={})},
          module_replacements=[dict(module='cube',base=dict(lock=LOCK,sha256=sha(OLD)),head=dict(lock=LOCK,sha256=sha(NEW)),
                                    renames={'chop_aa':'chop_zz'},resolve_order='name')])
        self.head_payload=NEW
        self.head_lock_sha=sha(NEW)
        self.head_symbols=['chop_mm','chop_zz']

    def capture(self,label,bodies,payload,symbols,locked_sha):
        root=self.root/label
        lock=[dict(name='cube',cubin_sha256=locked_sha,entries=[dict(symbol=s) for s in symbols])]
        revision=self.git.commit(bodies,extra_files={LOCK:json.dumps(lock).encode()})
        events=[]
        for entry in bodies[CREATE]:
            parts=entry.split()
            if parts[0]=='stream':
                events.extend([['id',1],['stream',1,0]])
            if parts[0]=='module':
                events.extend([['id',2],['module',2,'cube',fnv(payload),len(payload),'payload.bin']])
            if parts[0]=='resolve':
                ident=int(parts[1])
                events.extend([['id',ident],['resolve',ident,2,parts[2]]])
            if parts[0]=='allocate':
                raw=int(parts[-1])
                events.extend([['id',raw&0xffffffff],['allocation',raw,0,int(parts[1]),int(parts[2]),GEN+1]])
            events.append(['note',entry])
        events.append(['created',0,0,len(events),len(bodies[CREATE])])
        boundary=len(events)
        events.extend([['note',entry] for entry in bodies[STEP]])
        write_device(root,1,events)
        (root/'aux/payload.bin').write_bytes(payload)
        for name in bodies:
            checkpoint(root,name,[(1,events,boundary if name==CREATE else len(events))])
        return revision,seal_capture(root,revision,bodies)

    def run_case(self,expected):
        b,bi=self.capture('base',self.base,OLD,['chop_aa','chop_mm'],sha(OLD))
        h,hi=self.capture('head',self.head,self.head_payload,self.head_symbols,self.head_lock_sha)
        spec=self.root/'spec.json'
        spec.write_text(json.dumps(self.spec))
        report=self.root/'report.json'
        with contextlib.redirect_stdout(io.StringIO()):
            code=main(['--repo',str(self.git.repo),'--model',MODEL,'--base',b,'--head',h,'--spec',str(spec),
                       '--base-capture-index',str(bi),'--head-capture-index',str(hi),'--report',str(report)])
        value=json.loads(report.read_text())
        self.assertEqual(code,expected,value)
        return value

    def test_module_replacement_and_resolve_permutation(self):
        self.run_case(0)

    def test_undeclared_payload_change(self):
        self.spec['module_replacements']=[]
        self.run_case(1)

    def test_declaration_digest_must_match_lock(self):
        self.spec['module_replacements'][0]['head']['sha256']='0'*64
        self.run_case(2)

    def test_captured_payload_must_match_lock(self):
        self.head_payload=b'unauthorized payload'
        self.run_case(2)

    def test_rename_with_changed_function_id(self):
        self.head[CREATE][4]='resolve 9 chop_zz'
        self.head[CREATE][5]='configure 9 64'
        self.run_case(1)

    def test_resolves_must_be_name_sorted(self):
        self.head[CREATE][2:6]=['resolve 3 chop_zz','configure 3 64','resolve 4 chop_mm','configure 4 128']
        self.run_case(1)

    def test_configure_cannot_stay_at_old_slot(self):
        self.head[CREATE][3]='configure 3 64'
        self.head[CREATE][5]='configure 4 128'
        self.run_case(1)

    def test_configure_cannot_move_away_from_resolve(self):
        value=self.head[CREATE].pop(3)
        self.head[CREATE].append(value)
        self.run_case(1)

    def test_declared_configure_change(self):
        self.head[CREATE][5]='configure 4 96'
        self.spec['module_replacements'][0]['configures']={'chop_aa':{'from':[64],'to':[96]}}
        self.run_case(0)

    def workspace(self,size=128):
        r=self.spec['auxiliary_rules']['fixture']
        r['allocation_bindings']={'workspace':dict(base_raw=GEN+6,head_raw=GEN+6,rank=0,base_bytes=128,head_bytes=size,stream_raw=GEN+1)}
        return dict(allocation='workspace',offset=0,length=size)

    def expansion(self):
        workspace=self.workspace()
        self.spec['journals'][STEP]={'expansions':[dict(from_symbol='chop_aa',count=1,
            sequence=[dict(symbol='chop_zz',config={'grid_x':8},args=[{'arg':0},workspace,{'arg':1}]),
                      dict(symbol='chop_mm',config={'grid_x':1},args=[workspace,{'arg':0},{'arg':1}])],
            outputs=[dict(from_arg=0,to_launch=1,to_arg=1)])]}
        self.head[STEP]=[line('chop_zz',[span(GEN+5),span(GEN+6,length=128),'ScalarU32(1)'],8),
                         line('chop_mm',[span(GEN+6,length=128),span(GEN+5),'ScalarU32(1)'],1)]

    def merged(self):
        workspace=self.workspace()
        self.base[STEP].append(line('chop_mm',[span(GEN+5),'ScalarU32(1)']))
        self.spec['journals'][STEP]={'merges':[dict(lines=[1,2],symbols=['chop_aa','chop_mm'],symbol='chop_zz',
            config={'grid_x':8},args=[{'launch':0,'arg':0},workspace,{'launch':0,'arg':1}],
            outputs=[dict(from_launch=1,from_arg=0,to_arg=0)])]}
        self.head[STEP]=[line('chop_zz',[span(GEN+5),span(GEN+6,length=128),'ScalarU32(1)'],8)]

    def test_merge_uses_allocation_absent_from_old_launches(self):
        self.merged()
        self.run_case(0)

    def test_merge_rejects_wrong_allocation_buffer(self):
        self.merged()
        self.head[STEP][0]=self.head[STEP][0].replace(f'raw: {GEN+6}',f'raw: {GEN+5}')
        self.run_case(1)

    def test_replacement_uses_bound_peer_array(self):
        workspace=self.workspace()
        self.spec['journals'][STEP]={'launch_rules':[dict(from_symbol='chop_aa',to_symbol='chop_zz',count=1,
            args=[{'arg':0},{'peer_array':[workspace,None]},{'arg':1}])]}
        peer='DevicePointerArray([Some('+span(GEN+6,length=128)[len('DeviceBuffer('):-1]+'), None])'
        self.head[STEP]=[line('chop_zz',[span(GEN+5),peer,'ScalarU32(1)'])]
        self.run_case(0)

    def test_locked_module_removal_preserves_remaining_identity(self):
        self.head_symbols=['chop_zz']
        self.spec['module_replacements'][0].update(removed_symbols=['chop_mm'],resolve_order='locked')
        self.spec['auxiliary_rules']['fixture']['removed_resolves']=[dict(symbol='chop_mm',ids=[4],count=1)]
        self.head[CREATE]=['stream 1','module 2 cube','resolve 3 chop_zz','configure 3 64',
                          f'allocate 64 256 -> raw: {GEN+4}',f'allocate 128 256 -> raw: {GEN+5}']
        self.head[STEP]=[line('chop_zz',[span(GEN+4),'ScalarU32(1)'])]
        self.run_case(0)

    def test_generic_argument_edit_cannot_bypass_span_binding(self):
        self.spec['journals'][STEP]={'launch_rules':[dict(from_symbol='chop_aa',to_symbol='chop_zz',count=1,
             args=[dict(arg=0,replace=[{'from':'length: 64','to':'length: 128'}]),{'arg':1}])]}
        self.head[STEP]=[line('chop_zz',[span(GEN+5,length=128),'ScalarU32(1)'])]
        self.run_case(1)

    def test_expansion_with_allocation_workspace(self):
        self.expansion()
        self.run_case(0)

    def test_expansion_wrong_buffer(self):
        self.expansion()
        self.head[STEP][1]=self.head[STEP][1].replace(f'raw: {GEN+6}',f'raw: {GEN+5}')
        self.run_case(1)

    def test_expansion_extra_launch(self):
        self.expansion()
        self.head[STEP].append(self.head[STEP][-1])
        self.run_case(1)

    def test_expansion_wrong_order(self):
        self.expansion()
        self.head[STEP].reverse()
        self.run_case(1)

    def test_expansion_exceeds_workspace(self):
        self.expansion()
        self.spec['journals'][STEP]['expansions'][0]['sequence'][0]['args'][1]['length']=129
        self.run_case(1)

    def test_expansion_missing_allocation_binding(self):
        self.expansion()
        self.spec['auxiliary_rules']['fixture']['allocation_bindings']={}
        self.run_case(2)

    def resized(self):
        self.workspace(256)
        r=self.spec['auxiliary_rules']['fixture']
        r['allocation_resizes']=[dict(base_raw=GEN+6,head_raw=GEN+6,rank=0,**{'from':128,'to':256},alignment=256)]
        old=self.base[CREATE][-1]
        new=old.replace('allocate 128','allocate 256')
        self.head[CREATE][-1]=new
        policy={'state_sizes':[dict(line=8,**{'from':old,'to':new})]}
        r['create_policy']=policy
        self.spec['journals'][CREATE]=policy
        self.base[STEP].append(line('chop_mm',[span(GEN+6,length=128)]))
        self.head[STEP].append(line('chop_mm',[span(GEN+6,length=256)]))
        self.spec['journals'][STEP]['span_changes']=[dict(allocation='workspace',from_offset=0,from_length=128,to_offset=0,to_length=256,count=1)]

    def test_allocation_and_span_resize(self):
        self.resized()
        self.run_case(0)

    def test_undeclared_resized_allocation(self):
        self.resized()
        self.spec['auxiliary_rules']['fixture']['allocation_resizes']=[]
        self.run_case(1)

    def test_span_resize_exceeds_new_allocation(self):
        self.resized()
        self.spec['journals'][STEP]['span_changes'][0]['to_length']=257
        self.run_case(1)

    def test_resize_wrong_allocation_id(self):
        self.resized()
        self.spec['auxiliary_rules']['fixture']['allocation_resizes'][0]['head_raw']=GEN+5
        self.run_case(1)

    def test_peer_array_span_resize_preserves_order(self):
        self.resized()
        self.base[STEP][-1]=line('chop_mm',['DevicePointerArray([Some('+span(GEN+6,length=128)[len('DeviceBuffer('):-1]+'), None])'])
        self.head[STEP][-1]=line('chop_mm',['DevicePointerArray([Some('+span(GEN+6,length=256)[len('DeviceBuffer('):-1]+'), None])'])
        self.run_case(0)

    def test_peer_array_permutation_is_rejected(self):
        self.resized()
        self.base[STEP][-1]=line('chop_mm',['DevicePointerArray([Some('+span(GEN+6,length=128)[len('DeviceBuffer('):-1]+'), None])'])
        self.head[STEP][-1]=line('chop_mm',['DevicePointerArray([None, Some('+span(GEN+6,length=256)[len('DeviceBuffer('):-1]+')])'])
        self.run_case(1)


if __name__=='__main__':
    unittest.main()

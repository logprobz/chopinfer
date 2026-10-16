#!/usr/bin/env python3
"""Positive and negative controls for compare.py on synthetic captures. CPU only."""
import copy
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare as strict  # noqa: E402


def sample_spec():
    def phase(kind, requests, max_tokens, lo, hi, gated=True):
        return {'kind':kind, 'gated':gated, 'requests':requests, 'max_tokens':max_tokens,
                'top':1 if kind=='decode' else 5, 'prompt_tokens':{'min':lo,'max':hi}}
    phases = {
        'prefill-3k':phase('prefill',1,1,2990,3020),
        'teacher64':phase('teacher',1,2,64,68),
        'decode-c1':phase('decode',1,4,3130,3150),
        'decode-c2':phase('decode',2,4,6260,6300,False),
    }
    phases['teacher64']['teacher']={'offset':1,'prompt_ids':'prompt_token_ids'}
    return {'version':2, 'ignored_phases':['warm'], 'phases':phases}


def choice(tokens, top, offset=0):
    scores=[-1.0-i/16 for i in range(len(tokens))]
    tops=[{f't{k}':-1.0-k/16 for k in range(top)} for _ in tokens]
    for i in range(offset):scores[i]=None;tops[i]=None
    return {'tokens':tokens, 'token_logprobs':scores, 'top_logprobs':tops,
            'text_offset':list(range(len(tokens)))}


def synthetic():
    n=66;prompt_ids=list(range(100,100+n));tokens=[f'p{i}' for i in range(n)]+['g0','g1']
    echoed=choice(tokens,5,1)
    scores=[None]
    for i in range(1,n):
        row={str(10000+k):{'decoded_token':f't{k}','logprob':-2.0-k/16,'rank':k+1} for k in range(5)}
        row[str(prompt_ids[i])]={'decoded_token':tokens[i],'logprob':echoed['token_logprobs'][i]}
        scores.append(row)
    phases=[
        {'phase':'prefill-3k','kind':'prefill','requests':1,'prompt_tokens':3009,
         'outputs':[{'text':'result','token_ids':[30],'logprobs':choice(['result'],5),'finish_reason':'length'}]},
        {'phase':'teacher64','kind':'teacher','requests':1,'prompt_tokens':n,
         'outputs':[{'text':'echo','token_ids':[20,21],'prompt_token_ids':prompt_ids,
                     'logprobs':echoed,'prompt_logprobs':scores,'finish_reason':'length'}]},
    ]
    for requests in (1,2):
        phases.append({'phase':f'decode-c{requests}','kind':'decode','requests':requests,
                       'prompt_tokens':3140*requests, 'completion_tokens':4*requests,
                       'outputs':[{'text':'abcd','token_ids':[1,2,3,4],
                                   'logprobs':[choice(['a','b'],1),choice(['c','d'],1)],
                                   'finish_reason':'length','done':True}
                                  for _ in range(requests)], 'ms_per_token':[1.0,3.0]})
    phases.append({'phase':'warm','kind':'warm','requests':1,'prompt_tokens':3081})
    return {'model':'test-model','phases':phases}


def phase(record,name):
    return next(p for p in record['phases'] if p['phase']==name)


def output(record,name='teacher64'):
    return phase(record,name)['outputs'][0]


class Strict2(unittest.TestCase):
    cli_checks=0

    @classmethod
    def tearDownClass(cls):
        print(f'CLI checks completed: {cls.cli_checks}')

    def run_capture(self,a,b,want,spec=None,raw_a=None,comparator=None,env=None):
        spec=sample_spec() if spec is None else spec
        with tempfile.TemporaryDirectory(prefix='cmp2-test-') as d:
            root=Path(d)
            for side,record in [('a',a),('b',b)]:
                (root/side).mkdir()
                (root/side/'capture.json').write_text(raw_a if side=='a' and raw_a is not None else json.dumps(record))
            (root/'spec.json').write_text(json.dumps(spec))
            result=subprocess.run([sys.executable,'-B',str(comparator or HERE/'compare.py'),
                                   str(root/'a'),str(root/'b'),str(root/'out.json'),
                                   '--spec',str(root/'spec.json')],capture_output=True,text=True,env=env)
            type(self).cli_checks+=1
            self.assertEqual(result.returncode,want,result.stdout+result.stderr)
            self.assertNotIn('Traceback',result.stderr)
            if want==1:self.assertIn('FAIL:',result.stdout)
            if want==2:self.assertIn('ERROR:',result.stderr)
            return json.loads((root/'out.json').read_text()) if (root/'out.json').exists() else None

    def reject_both(self,mutation,spec=None):
        data=synthetic();mutation(data)
        return self.run_capture(data,copy.deepcopy(data),1,spec)

    def test_complete_explicit_capture(self):
        r=self.run_capture(synthetic(),synthetic(),0)
        coverage=next(p for p in r['phases'] if p['kind']=='teacher')['teacher_coverage']
        self.assertEqual(coverage[0][0]['scored_positions'],65)
        self.assertEqual(r['phases'][-1]['ms_per_token_median'],[2.0,2.0])

    def test_a_standalone_snapshot_needs_no_qualification_modules_or_scorer(self):
        with tempfile.TemporaryDirectory() as directory:
            comparator=Path(directory)/'compare.py'
            comparator.write_bytes((HERE/'compare.py').read_bytes())
            env={key:value for key,value in os.environ.items()
                 if key not in ('PYTHONPATH','CHOPINFER_EVAL','CARGO_TARGET_DIR')}
            self.run_capture(synthetic(),synthetic(),0,comparator=comparator,env=env)

    def test_prompt_bounds_both_sides(self):
        for name,values in [('prefill-3k',[1,0,2989,3021,True,3009.0]),
                            ('teacher64',[1,63,69]),('decode-c2',[1,6259,6301])]:
            for value in values:
                with self.subTest(name=name,value=value):
                    self.reject_both(lambda r:phase(r,name).__setitem__('prompt_tokens',value))
        for value in (2990,3020):
            a=synthetic();phase(a,'prefill-3k')['prompt_tokens']=value
            self.run_capture(a,copy.deepcopy(a),0)

    def test_nan_and_infinity_every_score_form(self):
        targets=[lambda r,v:output(r,'prefill-3k')['logprobs']['token_logprobs'].__setitem__(0,v),
                 lambda r,v:output(r,'prefill-3k')['logprobs']['top_logprobs'][0].__setitem__('t0',v),
                 lambda r,v:output(r)['logprobs']['token_logprobs'].__setitem__(2,v),
                 lambda r,v:output(r)['prompt_logprobs'][2]['102'].__setitem__('logprob',v),
                 lambda r,v:output(r,'decode-c2')['logprobs'][0]['token_logprobs'].__setitem__(0,v)]
        for value in (float('nan'),float('inf'),-float('inf')):
            for index,put in enumerate(targets):
                with self.subTest(value=value,target=index):self.reject_both(lambda r:put(r,value))

    def test_explicit_nonfinite_policy_is_per_phase_and_per_kind(self):
        for value,kind in [(float('nan'),'nan'),(float('inf'),'positive_infinity'),(-float('inf'),'negative_infinity')]:
            a=synthetic();output(a,'prefill-3k')['logprobs']['token_logprobs'][0]=value
            spec=sample_spec();spec['phases']['prefill-3k']['allow_nonfinite_logprobs']=[kind]
            self.run_capture(a,copy.deepcopy(a),0,spec)
            spec['phases']['prefill-3k']['allow_nonfinite_logprobs']=[]
            spec['phases']['teacher64']['allow_nonfinite_logprobs']=[kind]
            self.run_capture(a,copy.deepcopy(a),1,spec)
        a=synthetic();b=synthetic()
        output(a,'prefill-3k')['logprobs']['token_logprobs'][0]=float('inf')
        output(b,'prefill-3k')['logprobs']['token_logprobs'][0]=-float('inf')
        spec=sample_spec();spec['phases']['prefill-3k']['allow_nonfinite_logprobs']=['positive_infinity','negative_infinity']
        self.run_capture(a,b,1,spec)

    def test_ieee_bits_and_types(self):
        for x,y in [(-0.0,0.0),(-1,-1.0),(-1.0,math.nextafter(-1.0,0.0))]:
            a=synthetic();b=synthetic()
            output(a,'prefill-3k')['logprobs']['token_logprobs'][0]=x
            output(b,'prefill-3k')['logprobs']['token_logprobs'][0]=y
            self.run_capture(a,b,1)
        a=struct.unpack('>d',bytes.fromhex('7ff8000000000001'))[0]
        b=struct.unpack('>d',bytes.fromhex('7ff8000000000002'))[0]
        self.assertFalse(strict.same(a,b));self.assertTrue(strict.same(a,a))

    def test_missing_empty_and_reported_phases_fail(self):
        for name in sample_spec()['phases']:
            with self.subTest(name=name):
                self.reject_both(lambda r:r.__setitem__('phases',[p for p in r['phases'] if p['phase']!=name]))
                self.reject_both(lambda r:phase(r,name).__setitem__('outputs',[]))
        self.run_capture({'model':'m','phases':[]},{'model':'m','phases':[]},1)
        a=synthetic();b=synthetic();b['phases'].pop(3)
        self.run_capture(a,b,1)

    def test_reported_values_can_differ_but_coverage_cannot(self):
        a=synthetic();b=synthetic();output(b,'decode-c2')['text']='different'
        report=self.run_capture(a,b,0)
        self.assertTrue(report['valid']);self.assertFalse(report['phases'][-1]['identical'])
        phase(b,'decode-c2')['outputs'].pop()
        self.run_capture(a,b,1)

    def test_required_request_completion_and_event_counts(self):
        controls=[lambda r:phase(r,'prefill-3k').__setitem__('requests',True),
                  lambda r:phase(r,'decode-c2').__setitem__('requests',1),
                  lambda r:phase(r,'decode-c1').__setitem__('completion_tokens',3),
                  lambda r:phase(r,'decode-c1').__setitem__('completion_tokens',True),
                  lambda r:output(r,'decode-c1')['logprobs'].pop(),
                  lambda r:output(r,'decode-c1')['logprobs'].append({'tokens':[]}),
                  lambda r:output(r,'prefill-3k')['logprobs']['top_logprobs'][0].pop('t0'),
                  lambda r:output(r,'prefill-3k')['logprobs']['token_logprobs'].__setitem__(0,True)]
        for i,change in enumerate(controls):
            with self.subTest(i=i):self.reject_both(change)

    def test_generated_ids_are_required_and_compared_on_every_phase(self):
        for name in ('prefill-3k','teacher64','decode-c1'):
            with self.subTest(name=name):
                # Absent on both sides is refused, not equal.
                self.reject_both(lambda r:output(r,name).pop('token_ids'))
                self.reject_both(lambda r:output(r,name).__setitem__('token_ids',output(r,name)['token_ids'][:-1]))
                self.reject_both(lambda r:output(r,name)['token_ids'].append(5))
                self.reject_both(lambda r:output(r,name).__setitem__('token_ids',None))
                # Same text and logprob bits, a different served ID: differs.
                a=synthetic();b=synthetic();output(b,name)['token_ids'][-1]+=1
                report=self.run_capture(a,b,1)
                self.assertTrue(report['valid'])
                self.assertEqual(next(p for p in report['phases'] if p['phase']==name)['differing_requests'],[0])

    def test_an_unfinished_or_errored_request_is_refused(self):
        for name in ('prefill-3k','teacher64','decode-c1'):
            for value in (None,'','tool_calls',1):
                with self.subTest(name=name,value=value):
                    self.reject_both(lambda r:output(r,name).__setitem__('finish_reason',value))
            with self.subTest(name=name):self.reject_both(lambda r:output(r,name).pop('finish_reason'))
        for value in (False,None,1):
            with self.subTest(done=value):self.reject_both(lambda r:output(r,'decode-c1').__setitem__('done',value))
        self.reject_both(lambda r:output(r,'decode-c1').pop('done'))
        self.reject_both(lambda r:output(r,'decode-c1').__setitem__('stream_errors',['{"error": {}}']))
        self.reject_both(lambda r:output(r,'prefill-3k').__setitem__('done',True))

    def test_teacher_id_and_score_truncation(self):
        for key in ('prompt_token_ids','token_ids','prompt_logprobs'):
            with self.subTest(key=key):self.reject_both(lambda r:output(r).__setitem__(key,output(r)[key][:1]))
        for key in ('tokens','token_logprobs','top_logprobs','text_offset'):
            with self.subTest(key=key):
                self.reject_both(lambda r:output(r)['logprobs'].__setitem__(key,output(r)['logprobs'][key][:-1]))
        self.reject_both(lambda r:output(r).pop('prompt_token_ids'))
        self.reject_both(lambda r:output(r)['prompt_logprobs'].__setitem__(2,None))
        self.reject_both(lambda r:output(r)['prompt_logprobs'][2].pop('102'))
        self.reject_both(lambda r:output(r)['prompt_logprobs'][2].pop('10004'))
        self.reject_both(lambda r:output(r)['prompt_logprobs'][2]['10004'].__setitem__('rank',1))
        self.reject_both(lambda r:output(r)['prompt_token_ids'].__setitem__(2,10000))
        self.reject_both(lambda r:output(r)['token_ids'].__setitem__(0,True))
        self.reject_both(lambda r:output(r)['prompt_token_ids'].__setitem__(2,-1))

    def test_scored_only_ids_and_offset(self):
        a=synthetic();o=output(a);o['scored_token_ids']=o.pop('prompt_token_ids')[1:]
        spec=sample_spec();spec['phases']['teacher64']['teacher']['prompt_ids']='scored_token_ids'
        self.run_capture(a,copy.deepcopy(a),0,spec)
        o['scored_token_ids']=o['scored_token_ids'][:1]
        self.run_capture(a,copy.deepcopy(a),1,spec)
        a=synthetic();spec=sample_spec();spec['phases']['teacher64']['teacher']['offset']=0
        self.run_capture(a,copy.deepcopy(a),1,spec)
        o=output(a);o['logprobs']['token_logprobs'][0]=-0.5
        o['logprobs']['top_logprobs'][0]={f't{i}':-1.0-i for i in range(5)}
        o['prompt_logprobs'][0]=copy.deepcopy(o['prompt_logprobs'][1]);o['prompt_logprobs'][0].pop('101')
        o['prompt_logprobs'][0]['100']={'decoded_token':'p0','logprob':-0.5}
        self.run_capture(a,copy.deepcopy(a),0,spec)

    def test_archive_mode_requires_unique_backed_prompt_ids(self):
        spec=sample_spec();spec['phases']['teacher64']['teacher']['prompt_ids']='score_map_text'
        a=synthetic();output(a).pop('prompt_token_ids')
        self.run_capture(a,copy.deepcopy(a),0,spec)
        b=copy.deepcopy(a);output(b)['prompt_logprobs'][2]['9999']=copy.deepcopy(output(b)['prompt_logprobs'][2]['102'])
        self.run_capture(b,copy.deepcopy(b),1,spec)
        b=copy.deepcopy(a);output(b)['prompt_logprobs'][2].pop('102')
        self.run_capture(b,copy.deepcopy(b),1,spec)
        b=copy.deepcopy(a);output(b)['scored_token_ids']=[101]
        self.run_capture(b,copy.deepcopy(b),1,spec)
        b=copy.deepcopy(a);output(b)['prompt_logprobs'][2]['102']['decoded_token']='wrong'
        self.run_capture(b,copy.deepcopy(b),1,spec)

    def test_malformed_spec_is_rejected(self):
        mutations=[lambda s:s.__setitem__('version',1), lambda s:s.__setitem__('phases',{}),
                   lambda s:s['phases']['prefill-3k'].pop('prompt_tokens'),
                   lambda s:s['phases']['prefill-3k'].__setitem__('min_prompt_tokens',1),
                   lambda s:s['phases']['prefill-3k'].__setitem__('gated','false'),
                   lambda s:s['phases']['prefill-3k'].__setitem__('requests',True),
                   lambda s:s['phases']['prefill-3k'].__setitem__('top',0),
                   lambda s:s['phases']['prefill-3k'].__setitem__('allow_nonfinite_logprobs',True),
                   lambda s:s['phases']['prefill-3k']['prompt_tokens'].__setitem__('max',1),
                   lambda s:s['phases']['teacher64']['teacher'].__setitem__('offset',65),
                   lambda s:s['phases']['teacher64']['teacher'].__setitem__('prompt_ids','auto'),
                   lambda s:[p.__setitem__('gated',False) for p in s['phases'].values()]]
        for i,mutate in enumerate(mutations):
            with self.subTest(i=i):
                spec=sample_spec();mutate(spec);self.run_capture(synthetic(),synthetic(),2,spec)

    def test_malformed_capture_is_rejected(self):
        changes=[lambda r:r.__setitem__('model',None),
                 lambda r:r['phases'].append(copy.deepcopy(r['phases'][0])),
                 lambda r:r['phases'].append({'phase':'unexpected'}),
                 lambda r:phase(r,'prefill-3k').__setitem__('wall_ms',float('nan')),
                 lambda r:phase(r,'decode-c2').__setitem__('ms_per_token',[]),
                 lambda r:output(r).__setitem__('text',None),
                 lambda r:output(r)['prompt_logprobs'][2].__setitem__('01',{'decoded_token':'x','logprob':-1.0})]
        for i,change in enumerate(changes):
            with self.subTest(i=i):self.reject_both(change)
        self.run_capture(synthetic(),synthetic(),1,raw_a='{"model":"a","model":"b","phases":[]}')
        self.run_capture(synthetic(),synthetic(),1,raw_a='not json')

    def test_unexpected_score_fields_fail(self):
        for name in ('prefill-3k','decode-c2'):
            for value in (float('nan'),float('inf'),-float('inf')):
                with self.subTest(name=name,value=value):
                    self.reject_both(lambda r:output(r,name).__setitem__('prompt_logprobs',
                        [{'1':{'decoded_token':'a','logprob':value}}]))
        self.reject_both(lambda r:output(r,'prefill-3k')['logprobs'].__setitem__('extra_logprob',float('nan')))

    def test_cli_keeps_intermixed_spec_and_output_order(self):
        with tempfile.TemporaryDirectory(prefix='cmp2-order-') as d:
            root=Path(d);(root/'capture.json').write_text(json.dumps(synthetic()))
            (root/'spec.json').write_text(json.dumps(sample_spec()))
            a=str(root);out=str(root/'out.json');flag=['--spec',str(root/'spec.json')]
            for args in ([a,a]+flag+[out],flag+[a,a,out],[a]+flag+[a,out]):
                result=subprocess.run([sys.executable,'-B',str(HERE/'compare.py'),*args],
                                      capture_output=True,text=True)
                type(self).cli_checks+=1
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_deep_unknown_output_metadata_fails_cleanly(self):
        nested=[]
        for _ in range(800):nested=[nested]
        a=synthetic();output(a,'prefill-3k')['extra']=nested
        self.run_capture(a,a,1)

    def test_finite_median_does_not_overflow(self):
        maximum=sys.float_info.max
        a=synthetic();phase(a,'decode-c1')['ms_per_token']=[maximum,maximum]
        r=self.run_capture(a,copy.deepcopy(a),0)
        entry=next(p for p in r['phases'] if p['phase']=='decode-c1')
        self.assertEqual(entry['ms_per_token_median'],[maximum,maximum])
        a=synthetic();phase(a,'decode-c1')['ms_per_token']=[5e-324,5e-324]
        r=self.run_capture(a,copy.deepcopy(a),0)
        entry=next(p for p in r['phases'] if p['phase']=='decode-c1')
        self.assertEqual(entry['ms_per_token_median'],[5e-324,5e-324])

    def test_output_cannot_replace_an_input(self):
        with tempfile.TemporaryDirectory(prefix='cmp2-protect-') as d:
            root=Path(d);(root/'capture.json').write_text(json.dumps(synthetic()))
            (root/'spec.json').write_text(json.dumps(sample_spec()))
            old=(root/'capture.json').read_bytes()
            result=subprocess.run([sys.executable,'-B',str(HERE/'compare.py'),str(root),str(root),
                                   str(root/'capture.json'),'--spec',str(root/'spec.json')],capture_output=True,text=True)
            type(self).cli_checks+=1
            self.assertEqual(result.returncode,2);self.assertEqual((root/'capture.json').read_bytes(),old)


if __name__=='__main__':
    unittest.main(verbosity=2)

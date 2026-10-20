"""Exercise capture coverage and every runner phase before consuming a GPU slot."""
from pathlib import Path
import collections
import copy
import hashlib
import json
import struct
import tempfile
import unittest
from unittest import mock
import corpus
import run


def pack(values):return struct.pack('<'+'I'*len(values),*values)


def synthetic(root):
    (root/'blobs').mkdir()
    def blob(data):
        sha=hashlib.sha256(data).hexdigest();file=f'blobs/{sha}.bin';(root/file).write_bytes(data)
        return dict(file=file,bytes=len(data),sha256=sha)
    real=[]
    for gi,(group,items,context,quota) in enumerate(corpus.GROUPS):
        rows=items*4;request_ids=list(range(100+gi*8,100+gi*8+items));real+=request_ids
        for ordinal in range(quota):
            directory=root/f'{group}-{ordinal:02}';directory.mkdir()
            records=[]
            batch=[i//4 for i in range(rows)]
            for layer in (0,2,20):
                page=32 if layer==2 else 64
                source='deepseek41_h16_d512_'+('window' if layer==0 else 'mixed')+'_normalized_s1_r64_verify'+('_p32' if layer==2 else '')
                scalars=dict(window_code_stride=136*512,window_scale_stride=136*16,window_slots=136,window_sequences=items,
                             window_items=items,window_key_stride=128,window_budget=128,scale_bits=1026882510)
                data={'q':blob(bytes(rows*16*512*2)),'row_batch':blob(pack(batch)),'live_rows':blob(pack([rows])),
                      'window_item_slots':blob(pack(list(range(items)))),'window_keys':blob(pack([i%128 for i in range(rows*128)])),
                      'window_codes':blob(bytes(items*136*512)),'window_scales':blob(bytes([127])*(items*136*16)),
                      'sinks':blob(bytes(16*4)),'cos_sin':blob(bytes(rows*64*4))}
                tables=[];pages=[]
                if layer:
                    scalars.update(page_stride=page*288,max_pages=2,compressed_pages=items,compressed_sequences=items,compressed_key_stride=512,compressed_budget=512)
                    tables=[x for i in range(items) for x in (i,0xffffffff)];pages=list(range(items))
                    data.update(compressed_pool=blob(bytes(items*page*288)),compressed_keys=blob(pack([0]*(rows*512))),block_tables=blob(pack(tables)))
                cached=rows==32;prefix='chop_latent_sources_'+source
                records.append(dict(source_name=source,layer=layer,rows=rows,live_rows=rows,page=page,scalars=scalars,
                                    original_scalars=scalars.copy(),buffers=data,original_window_items=data['window_item_slots'],original_tables=blob(pack(tables)),
                                    physical_window_slots=list(range(items)),physical_pages=pages,numerical_splits=8,
                                    served_first_symbol=prefix+('_cache_decode_control0_v1' if cached else '_heads_v1'),
                                    served_body_symbol=prefix+('_cached_profile0_v1' if cached else '_heads_v1'),served_merge_symbol=prefix+'_heads_merge_v1',
                                    expected_workspace=blob(bytes(rows*8*16*514*4)),expected_output=blob(bytes(rows*16*512*2))))
            value=dict(schema=corpus.SCHEMA,group=group,arm=dict(group=group,items=items,context=context,quota=quota,nonce='a'*32),
                       rows=rows,items=items,phase='verify',rank=0,ranks=4,execution='eager-current-exact',request_ids=request_ids,
                       positions=[context+i%4 for i in range(rows)],token_ids=[42]*rows,token_batch=batch,nonce='a'*32,snapshot_sha256='b'*64,
                       slot_ids=list(range(items)),q_start=list(range(0,rows,4)),q_len=[4]*items,records=records)
            (directory/'capture.json').write_text(json.dumps(value)+'\n')
    return dict(warmup=[1,2],real=real)


class CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(prefix='attention-capture-cpu-');cls.root=Path(cls.temporary.name)
        cls.origins=synthetic(cls.root)
        cls.complete=corpus.seal(cls.root,'a'*32,'b'*64,cls.origins)
    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()
    def test_complete_corpus_and_replay(self):
        self.assertEqual(self.complete['records'],24);self.assertEqual(len(self.complete['captures']),8)
        lines=[]
        for frame in self.complete['captures']:
            for record in frame['records']:
                value={k:record[k] for k in ('rows','cached','partial_sha256','output_sha256')}
                value.update(id=Path(frame['file']).parent.name+f'-L{record["layer"]:02}',poison_replays=2)
                lines.append('test prefix: ATTENTION_CAPTURE_REPLAY_OK '+json.dumps(value))
        lines+=['ATTENTION_CAPTURE_REPLAY_COMPLETE records=24 poison_replays=48','test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured;']
        text='\n'.join(lines)
        self.assertEqual(corpus.replay_result(text,self.complete),dict(records=24,poison_replays=48,bitwise=True))
        for bad in [text.replace(lines[0],'',1),text+'\n'+lines[0],text.replace('"poison_replays": 2','"poison_replays": 1',1),
                    text.replace('"rows": 4','"rows": 8',1),text.replace('1 passed','0 passed'),text.replace('0 ignored','1 ignored'),
                    text.replace('COMPLETE records=24','COMPLETE records=23'),text+'\nGPU work skipped']:
            with self.assertRaises(ValueError):corpus.replay_result(bad,self.complete)
    def test_reject_incomplete_or_wrong_provenance(self):
        path=self.root/self.complete['captures'][0]['file'];original=path.read_text();value=json.loads(original)
        for field,wrong in [('nonce','c'*32),('phase','prefill'),('rows',0),('items',8),('request_ids',[1]),('positions',[0]*4),('records',value['records'][:2])]:
            changed=copy.deepcopy(value);changed[field]=wrong;path.write_text(json.dumps(changed))
            try:
                with self.assertRaises(ValueError):corpus.seal(self.root,'a'*32,'b'*64,self.origins)
            finally:path.write_text(original)
        path.rename(path.with_suffix('.held'))
        try:
            with self.assertRaises(ValueError):corpus.seal(self.root,'a'*32,'b'*64,self.origins)
        finally:path.with_suffix('.held').rename(path)
    def test_reject_wrong_page_or_metadata_rewrite(self):
        frame=json.loads((self.root/self.complete['captures'][0]['file']).read_text())
        record=frame['records'][1]
        for change in [lambda r:r['physical_pages'].append(99),lambda r:r['scalars'].update(compressed_budget=256),
                       lambda r:r['scalars'].update(window_code_stride=7),lambda r:r.update(numerical_splits=1),
                       lambda r:r.update(served_body_symbol='wrong'),lambda r:r['buffers']['q'].update(sha256='0'*64)]:
            changed=copy.deepcopy(record);change(changed)
            with self.assertRaises((ValueError,FileNotFoundError)):corpus.validate_record(self.root,changed,frame['rows'])
    def test_origins_and_cold_responses(self):
        log='{"schema":"chopinfer.step.v1","completed_prompts":[[1,3]]}\nstartup readiness: 1s\n{"schema":"chopinfer.step.v1","completed_prompts":[[2,4]]}\n'
        self.assertEqual(corpus.origins(log),dict(warmup=[1],real=[2]))
        with self.assertRaises(ValueError):corpus.origins(log.replace('startup readiness:','none:'))
        body=dict(prompt=[[1]*8192,[2]*8192],max_tokens=16)
        response=dict(usage=dict(prompt_tokens=16384,completion_tokens=32,prompt_tokens_details=dict(cached_tokens=0)),
                      choices=[dict(index=i,finish_reason='length',token_ids=[42]*16,prompt_token_ids=p) for i,p in enumerate(body['prompt'])])
        self.assertEqual(run.response(response,body)['cached_tokens'],0)
        for mutate in [lambda r:r['usage']['prompt_tokens_details'].update(cached_tokens=1),lambda r:r['choices'].pop(),
                       lambda r:r['choices'][0].update(token_ids=[]),lambda r:r['choices'][0].update(prompt_token_ids=[2])]:
            bad=copy.deepcopy(response);mutate(bad)
            with self.assertRaises(ValueError):run.response(bad,body)
    def test_every_phase_failure_stops_before_dependent_work(self):
        class Fake:
            def __init__(self,fail=None):self.fail=fail;self.events=[];self.result=None
            def event(self,name,value=None):
                self.events.append(name)
                if self.fail==name:raise TimeoutError(name)
                return value
            def preflight(self):self.event('preflight')
            def power(self,edge):self.event('power-'+edge)
            def journal_preflight(self):self.event('journal-preflight')
            def start(self):self.event('start')
            def capture(self):self.event('capture')
            def stop(self):self.event('stop')
            def seal(self):return self.event('seal',{'records':24})
            def replay(self,complete):return self.event('replay',{'records':complete['records']})
            def journal(self):self.event('journal')
            def verdict(self,result):self.events.append('verdict');self.result=result
        ordered=['preflight','power-before','journal-preflight','start','capture','stop','seal','replay','power-after','journal','verdict']
        good=Fake();self.assertEqual(run.workflow(good)['exit_code'],0);self.assertEqual(good.events,ordered)
        for phase in ordered[:-1]:
            bad=Fake(phase)
            with self.assertRaises(TimeoutError):run.workflow(bad)
            self.assertEqual(bad.result['exit_code'],1)
            self.assertEqual(bad.events[-1],'verdict')
            if phase in ('start','capture'):self.assertIn('stop',bad.events)
            if phase in ('preflight','power-before','journal-preflight','start','capture','stop','seal'):self.assertNotIn('replay',bad.events)
    def test_actual_request_loop_arms_each_cold_group(self):
        with tempfile.TemporaryDirectory(prefix='attention-request-flow-') as directory:
            root=Path(directory);(root/'inputs').mkdir();(root/'output').mkdir();(root/'captured').mkdir()
            (root/'inputs/prompts.json').write_text(json.dumps([[i+1,42] for i in range(8)]))
            ops=run.Operations.__new__(run.Operations)
            ops.root=root/'inputs';ops.output=root/'output';ops.captures=root/'captured';ops.nonce='c'*32
            seen=[]
            def request(route,body):
                self.assertEqual(route,'/v1/completions')
                arm=json.loads((ops.captures/'armed.json').read_text())
                self.assertEqual(len(body['prompt']),arm['items'])
                self.assertTrue(all(len(p)==arm['context'] for p in body['prompt']))
                self.assertEqual(body['max_tokens'],256 if arm['items']==8 else 32)
                self.assertEqual(body['max_tokens'],body['min_tokens'])
                self.assertEqual(len({tuple(p) for p in body['prompt']}),arm['items'])
                seen.append(body['cache_salt'])
                for i in range(arm['quota']):
                    frame=ops.captures/f"{arm['group']}-{i:02}";frame.mkdir();(frame/'capture.json').write_text('{}')
                return dict(usage=dict(prompt_tokens=arm['items']*arm['context'],completion_tokens=arm['items']*body['max_tokens'],prompt_tokens_details=dict(cached_tokens=0)),
                            choices=[dict(index=i,finish_reason='length',token_ids=[42]*body['max_tokens'],prompt_token_ids=p) for i,p in enumerate(body['prompt'])])
            ops.request=request
            ops.capture()
            self.assertEqual(len(set(seen)),6)
            self.assertFalse((ops.captures/'armed.json').exists())
            self.assertEqual(len(list(ops.captures.glob('*/capture.json'))),8)
    def test_actual_start_readiness_exit_and_timeout(self):
        with tempfile.TemporaryDirectory(prefix='attention-start-flow-') as directory:
            root=Path(directory);ops=run.Operations.__new__(run.Operations)
            ops.root=root;ops.config=dict(package='package',checkpoint='checkpoint');ops.output=root;ops.captures=root
            ops.nonce='a'*32;ops.snapshot='b'*64;ops.server_log=root/'server.log';ops.remaining=lambda limit:limit
            process=mock.Mock();process.poll.return_value=None
            ops.request=mock.Mock(side_effect=[OSError('wait'),{'ready':True}])
            with mock.patch.object(run.subprocess,'Popen',return_value=process),mock.patch.object(run.time,'sleep'):
                ops.start()
            self.assertIs(ops.process,process);self.assertEqual(ops.request.call_count,2)
            process.poll.return_value=1
            with mock.patch.object(run.subprocess,'Popen',return_value=process):
                with self.assertRaises(RuntimeError):ops.start()
            process.poll.return_value=None;ops.request=mock.Mock(side_effect=OSError('wait'))
            with mock.patch.object(run.subprocess,'Popen',return_value=process),mock.patch.object(run.time,'monotonic',side_effect=[0,241]):
                with self.assertRaises(TimeoutError):ops.start()

    def test_server_cleanup_escalates_only_its_process_group(self):
        ops=run.Operations.__new__(run.Operations)
        fake=mock.Mock(pid=12345);fake.wait.side_effect=[run.subprocess.TimeoutExpired('server',15),0];ops.process=fake
        with mock.patch.object(run.os,'killpg') as kill:
            ops.stop()
        self.assertEqual(kill.call_count,2);self.assertIsNone(ops.process)


if __name__=='__main__':unittest.main()

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import prefill_cache_job as job
import prefill_selected_capture as capture

class Gates(unittest.TestCase):
    def test_pair_hashes_ignore_mutable_cache_but_reject_changed_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'run';out.mkdir()
            binary=root/'binary';binary.write_bytes(b'frozen binary')
            record=root/'model-cache.json';record.write_bytes(b'before capture')
            (root/'PAIR_SHA256SUMS').write_text(f'{capture.digest(binary)}  binary\n')
            (root/'SHA256SUMS').write_text(f'{capture.digest(record)}  model-cache.json\n')
            (root/'prompt.json').write_text(json.dumps([0]*32768))
            record.write_bytes(b'last_selected updated by serving')
            def run_step(output,name,args,timeout,**kwargs):
                if name=='hashes':
                    subprocess.run(args,check=True,capture_output=True,**kwargs)
                return '',0
            with patch.object(job,'ROOT',root),patch.object(job.logs,'run_step',side_effect=run_step),patch.object(capture,'check_space'),patch.object(job.signal,'signal'),patch.dict(os.environ,L11_CACHE_RUN_DIR=str(out)),patch.object(sys,'argv',['job','--dry-run','--samples-only']):
                job.main()
                self.assertEqual(json.loads((out/'verdict.json').read_text())['exit_code'],0)
                binary.write_bytes(b'changed binary')
                with self.assertRaises(SystemExit) as failure:job.main()
                self.assertEqual(failure.exception.code,1)

    def test_capture_manifest_validates_its_own_files_and_complete_coverage(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(capture,'ROWS',2),patch.object(capture,'OFFSETS',(0,2,4,6)):
            root=Path(directory);served=root/'capture/served';served.mkdir(parents=True)
            prompt=list(range(8));(served/'prompt.json').write_text(json.dumps(prompt))
            records=[]
            for off in capture.OFFSETS:
                step=served/f'step{off}';step.mkdir();window=step/'window.i32';window.write_bytes(bytes(2*128*4))
                base=dict(offset=off,rows=2,positions=[off,off+1],window=capture.reference(window,served))
                records.append(dict(base,layer=0,variant='window'))
                for layer in range(2,40):
                    selected=step/f'{layer}.i32';selected.write_bytes(bytes(2*512*4))
                    meta=step/f'{layer}.json';ratio=2 if layer<20 else 1
                    meta.write_text(json.dumps(dict(schema='chopinfer.prefill-selected.v1',layer=layer,offset=off,rows=2,top_k=512,positions=base['positions'],token_ids=prompt[off:off+2],rank=0,replicated=True,ratio=ratio)))
                    records.append(dict(base,layer=layer,variant='mixed32' if ratio==2 else 'mixed64',selected=capture.reference(selected,served),metadata=capture.reference(meta,served)))
            data=dict(schema='chopinfer.prefill-cache-lists.v1',samples=False,records=records,prompt=capture.reference(served/'prompt.json',served))
            manifest=served/'capture-set.json'
            def seal():
                manifest.write_text(json.dumps(data))
                (root/'capture-ready.json').write_text(json.dumps(dict(path=str(manifest),sha256=capture.digest(manifest))))
            seal()
            self.assertEqual(job.full_capture_manifest(root),manifest)
            selected=served/records[1]['selected']['path'];original=selected.read_bytes()
            selected.write_bytes(b'x'+original[1:])
            with self.assertRaisesRegex(ValueError,'digest'):job.full_capture_manifest(root)
            selected.write_bytes(original)
            saved=records.pop();seal()
            with self.assertRaisesRegex(ValueError,'coverage'):job.full_capture_manifest(root)
            records.append(saved);seal()
            manifest.write_text(manifest.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'digest'):job.full_capture_manifest(root)
            seal();selected.unlink();selected.symlink_to(served/'prompt.json')
            with self.assertRaisesRegex(ValueError,'symlink'):job.full_capture_manifest(root)
            selected.unlink();selected.write_bytes(original)
            records[1]['selected']['path']='../outside.i32';seal()
            with self.assertRaisesRegex(ValueError,'relative'):job.full_capture_manifest(root)

    def test_prefixed_pair_gates_require_both_ranks_and_exact_evidence(self):
        text='\n'.join(f'prefix PASS prefill_cache rank={r} variant=mixed32 offset=8192 rows=8192 workspace=true output=true replay=true f64=true cache_guards=true guard_control=true disjoint=true refresh=true actual_selected=true page_cases=3 selected_sha256={"a"*64} cache_bytes=42' for r in [0,1])
        self.assertEqual(len(job.gate(text,'mixed32',8192,8192,False)),2)
        for bad in [text.splitlines()[0],text+'\n'+text,text.replace('disjoint=true','disjoint=false'),text.replace('rows=8192','rows=32'),text.replace('page_cases=3','page_cases=0'),text+'\nFAILED']:
            with self.assertRaises(AssertionError):job.gate(bad,'mixed32',8192,8192,False)
    def test_timing_rejects_missing_producer_and_nonfinite_values(self):
        text='\n'.join(f'prefix TIMING prefill_cache rank={r} variant=mixed64 offset=0 rows=8192 paired_total_us=6 cached_total_us=3 producer_us=1 ratio=0.5 samples=3 cache_bytes=42 actual_selected=true' for r in [0,1])
        self.assertEqual(len(job.gate(text,'mixed64',0,8192,True)),2)
        for bad in [text.splitlines()[0],text.replace('producer_us=1','producer_us=nan'),text.replace('samples=3','samples=7')]:
            with self.assertRaises(AssertionError):job.gate(bad,'mixed64',0,8192,True)
    def test_disk_reserve_includes_capture_budget(self):
        with patch.object(capture.shutil,'disk_usage') as disk:
            disk.return_value.free=capture.RESERVE+capture.BUDGET-1
            with self.assertRaises(RuntimeError):capture.check_space(Path('.'))
            disk.return_value.free=capture.RESERVE+capture.BUDGET
            self.assertEqual(capture.check_space(Path('.')),disk.return_value.free)
    def test_full_inventory_requires_every_layer_offset_and_committed_step(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(capture,'ROWS',2),patch.object(capture,'OFFSETS',(0,2,4,6)):
            root=Path(directory);prompt=list(range(8));(root/'prompt.json').write_text(json.dumps(prompt))
            for off in capture.OFFSETS:
                step=root/'taps'/f'step{off:04}';step.mkdir(parents=True)
                (step/'capture.json').write_text(json.dumps(dict(program='Prefill',producer=dict(startup_nonce='a'*32))))
                (step/'completion.json').write_text(json.dumps(dict(completion='full',expected=1,observed=1)))
                (step/'window-keys.i32').write_bytes(bytes(2*128*4))
                for layer in range(2,40):
                    p=step/f'selected-L{layer:02}';p.mkdir()
                    (p/'selected.i32').write_bytes(bytes(2*512*4))
                    (p/'selected.json').write_text(json.dumps(dict(schema='chopinfer.prefill-selected.v1',layer=layer,offset=off,rows=2,top_k=512,positions=[off,off+1],token_ids=prompt[off:off+2],rank=0,replicated=True,ratio=2 if layer<20 else 1)))
            self.assertEqual(len(capture.inventory(root,prompt,'a'*32)['records']),156)
            bad=root/'taps/step0004/selected-L02/selected.i32';bad.write_bytes(b'')
            with self.assertRaises(AssertionError):capture.inventory(root,prompt,'a'*32)

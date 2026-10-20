import json
from pathlib import Path
import tempfile
import unittest
import indexer_select_early_job as job


def logs(memcheck=False):
    output=[]
    for shard in (0,1):
        lines=[];count=0
        for i,c in enumerate(job.cases()):
            if i%2!=shard or not job.selected(i,c,memcheck):continue
            count+=1
            timed=job.base.timed(c) and not memcheck
            row=dict(c,id=i,live=c['rows']-int(c['kind']==12),kept=31 if c['kind']==11 else 512,
                     batches=job.base.batch_count(c),comparisons=job.comparisons(c),
                     patterns=job.PATTERNS if job.crafted(c) else 0,controls=2,mismatches=0,
                     timed=timed,memcheck=memcheck)
            lines.append('test prefix ... SELECT_EARLY_CASE '+json.dumps(row))
            if timed:
                for stage in job.STAGES:
                    lines.append('SELECT_EARLY_TIMING '+json.dumps(dict(
                        id=i,stage=stage,batches=row['batches'],median_ms=[1.,.8],paired_ratio=.8,
                        exclusive=True,score_buffer_reused=True)))
        lines.append(f'SELECT_EARLY_COMPLETE shard={shard} cases={count} memcheck={str(memcheck).lower()}')
        lines.append('test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured')
        output.append('\n'.join(lines))
    return output


def receipts(root):
    paths=[]
    for i,c in enumerate(job.cases()):
        rounds=[];timed=job.base.timed(c)
        if timed:
            for measurement,stage in enumerate(job.STAGES):
                rounds.append(dict(kind='paired_summary',operation=f'select-early-{i}-{stage}',
                                   measurement=measurement,rounds=3))
                rounds.extend(dict(measurement=measurement,round=r,arm=a,elapsed_ms=1.)
                              for r in range(3) for a in (0,1))
        p=root/f'{i}.json'
        p.write_text(json.dumps(dict(id=i,timed=timed,mismatches=0,
                                    measurement_label='exclusive-device',timing_rounds=rounds)))
        paths.append(p)
    return paths


class Parser(unittest.TestCase):
    def test_inventory_and_prefixed_markers(self):
        value=job.results(logs())
        self.assertEqual((value['cases'],value['comparisons'],value['patterns'],value['controls'],value['timing_pairs']),
                         (94,394,18,188,44))
        self.assertEqual(value['summary']['select']['cases'],22)
        value=job.results(logs(True),True)
        self.assertEqual((value['cases'],value['comparisons'],value['patterns'],value['controls'],value['timing_pairs']),
                         (26,122,18,52,0))
        self.assertEqual([sum(job.selected(i,c,True) for i,c in enumerate(job.cases()) if i%2==s) for s in (0,1)],[12,14])

    def test_missing_duplicate_or_changed_case_fails(self):
        for mutation in ('missing','duplicate','bits','count','patterns','batches','kept','controls'):
            # Case 25 is layer 2's crafted case, in two dense batches.
            text=logs();line=next(l for l in text[1].splitlines() if '"id": 25,' in l)
            if mutation=='missing':text[1]=text[1].replace(line+'\n','',1)
            elif mutation=='duplicate':text[1]+='\n'+line
            else:
                before,after={'bits':('"mismatches": 0','"mismatches": 1'),
                              'count':('"comparisons": 10','"comparisons": 9'),
                              'patterns':('"patterns": 6','"patterns": 5'),
                              'batches':('"batches": 2','"batches": 1'),
                              'kept':('"kept": 512','"kept": 31'),
                              'controls':('"controls": 2','"controls": 1')}[mutation]
                self.assertIn(before,line)
                text[1]=text[1].replace(line,line.replace(before,after,1),1)
            with self.assertRaises(AssertionError,msg=mutation):job.results(text)

    def test_timing_and_memcheck_fail_closed(self):
        for before,after in [
            ('"median_ms": [1.0, 0.8]','"median_ms": [1.0, NaN]'),
            ('"exclusive": true','"exclusive": false'),
            ('"score_buffer_reused": true','"score_buffer_reused": false'),
            ('"stage": "select"','"stage": "pack"'),
            ('cases=47','cases=46'),
            ('1 passed; 0 failed; 0 ignored','0 passed; 0 failed; 1 ignored'),
        ]:
            text=logs();self.assertIn(before,text[0]);text[0]=text[0].replace(before,after,1)
            with self.assertRaises(AssertionError,msg=before):job.results(text)
        text=logs(True);text[1]=text[1].replace('cases=14','cases=13')
        with self.assertRaises(AssertionError):job.results(text,True)
        with self.assertRaises(AssertionError):job.results(logs(True))

    def test_receipts_require_every_raw_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=receipts(Path(directory))
            self.assertEqual(job.receipts(paths),dict(pairs=44,samples=264))
            path=paths[0];row=json.loads(path.read_text())
            row['timing_rounds'].pop()
            path.write_text(json.dumps(row))
            with self.assertRaises(AssertionError):job.receipts(paths)
        with tempfile.TemporaryDirectory() as directory:
            paths=receipts(Path(directory))
            row=json.loads(paths[0].read_text());row['timing_rounds'][0]['operation']='select-early-0-pack'
            paths[0].write_text(json.dumps(row))
            with self.assertRaises(AssertionError):job.receipts(paths)


if __name__=='__main__':unittest.main()

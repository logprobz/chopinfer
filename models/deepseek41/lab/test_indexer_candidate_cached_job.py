import json
from pathlib import Path
import tempfile
import unittest
import indexer_candidate_cached_job as job


def logs(memcheck=False):
    output=[]
    for shard in (0,1):
        lines=[];count=0
        for i,c in enumerate(job.cases()):
            if i%2!=shard or not job.selected(c,memcheck):continue
            count+=1
            timed=job.timed(c) and not memcheck
            row=dict(c,id=i,comparisons=4 if c['direct'] else 8,controls=2,
                     mismatches=0,timed=timed,memcheck=memcheck)
            lines.append('test prefix ... CANDIDATE_CACHED_CASE '+json.dumps(row))
            if timed:
                for stage in ('publish','all'):
                    lines.append('CANDIDATE_CACHED_TIMING '+json.dumps(dict(
                        id=i,stage=stage,median_ms=[1.,.8],paired_ratio=.8,
                        exclusive=True,score_buffer_reused=stage=='all')))
        lines.append(f'CANDIDATE_CACHED_COMPLETE shard={shard} cases={count} memcheck={str(memcheck).lower()}')
        lines.append('test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured')
        output.append('\n'.join(lines))
    return output


def receipts(root):
    paths=[]
    for i,c in enumerate(job.cases()):
        rounds=[]
        if job.timed(c):
            for measurement,stage in enumerate(('publish','all')):
                rounds.append(dict(kind='paired_summary',operation=f'candidate-cached-{i}-{stage}',
                                   measurement=measurement,rounds=3))
                rounds.extend(dict(measurement=measurement,round=r,arm=a,elapsed_ms=1.)
                              for r in range(3) for a in (0,1))
        p=root/f'{i}.json'
        p.write_text(json.dumps(dict(id=i,timed=job.timed(c),mismatches=0,
                                    measurement_label='exclusive-device',timing_rounds=rounds)))
        paths.append(p)
    return paths


class Parser(unittest.TestCase):
    def test_inventory_and_prefixed_markers(self):
        value=job.results(logs())
        self.assertEqual((value['cases'],value['comparisons'],value['controls'],value['timing_pairs']),
                         (198,1152,396,60))
        value=job.results(logs(True),True)
        self.assertEqual((value['cases'],value['comparisons'],value['controls'],value['timing_pairs']),
                         (36,252,72,0))
        self.assertIn('shard=0 cases=13 memcheck=true',logs(True)[0])
        self.assertIn('shard=1 cases=23 memcheck=true',logs(True)[1])

    def test_missing_duplicate_or_changed_case_fails(self):
        for mutation in ('missing','duplicate','bits','count','kept','controls'):
            text=logs();line=text[0].splitlines()[0]
            if mutation=='missing':text[0]=text[0].replace(line+'\n','',1)
            elif mutation=='duplicate':text[0]+='\n'+line
            else:
                before,after={'bits':('"mismatches": 0','"mismatches": 1'),
                              'count':('"comparisons": 8','"comparisons": 7'),
                              'kept':('"kept": 1','"kept": 2'),
                              'controls':('"controls": 2','"controls": 1')}[mutation]
                text[0]=text[0].replace(before,after,1)
            with self.assertRaises(AssertionError):job.results(text)

    def test_timing_and_memcheck_fail_closed(self):
        for before,after in [
            ('"median_ms": [1.0, 0.8]','"median_ms": [1.0, NaN]'),
            ('"exclusive": true','"exclusive": false'),
            ('"score_buffer_reused": true','"score_buffer_reused": false'),
            ('cases=99','cases=98'),
            ('1 passed; 0 failed; 0 ignored','0 passed; 0 failed; 1 ignored'),
        ]:
            text=logs();text[0]=text[0].replace(before,after,1)
            with self.assertRaises(AssertionError):job.results(text)
        text=logs(True);text[1]=text[1].replace('cases=23','cases=22')
        with self.assertRaises(AssertionError):job.results(text,True)

    def test_receipts_require_every_raw_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=receipts(Path(directory))
            self.assertEqual(job.receipts(paths),dict(pairs=60,samples=360))
            path=paths[0];row=json.loads(path.read_text())
            row['timing_rounds'].pop()
            path.write_text(json.dumps(row))
            with self.assertRaises(AssertionError):job.receipts(paths)


if __name__=='__main__':unittest.main()

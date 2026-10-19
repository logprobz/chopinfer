import unittest
from fold_stream_report import exclusive,records,select

class ReportTests(unittest.TestCase):
    def test_exclusive_handles_nested_overlap_and_idle_gaps(self):
        self.assertEqual(exclusive([(0,0,100,'a'),(1,5,10,'b'),(2,11,20,'c'),(3,120,140,'d')]),{'a':100,'b':0,'c':0,'d':20})
        self.assertEqual(exclusive([(0,0,10,'init'),(1,2,15,'compute')]),{'init':10,'compute':5})
    def test_libtest_prefix_does_not_hide_first_record(self):
        self.assertEqual(list(records('test foo ... {"bitwise":true}\n')), [{'bitwise':True}])
    def pair(self,ratio=.9,exclusive_ratio=.9):
        return [dict(n=512,k=5120,groups=1,paired=False,rows=1,device=i,bitwise=True,paired_ratio=ratio,exclusive_paired_ratio=exclusive_ratio,baseline_routing='ticket') for i in range(2)]
    def test_ties_regressions_and_one_device_wins_keep_baseline(self):
        for ratio in [.97,1,1.03,1.2]:
            self.assertTrue(all(r['selected']=='ticket' for r in select(self.pair(ratio))))
        rows=self.pair();rows[1]['exclusive_paired_ratio']=1.01
        self.assertTrue(all(r['selected']=='ticket' for r in select(rows)))
        self.assertTrue(all(r['selected']=='stream' for r in select(self.pair())))
    def test_served_selection_enforces_the_three_percent_gate(self):
        from fold_stream_report import check_serving,key
        rows=self.pair(1.02,1.02);enabled={key(rows[0])}
        self.assertEqual(check_serving(rows,enabled),2)
        rows[1]['paired_ratio']=1.031
        with self.assertRaises(AssertionError):check_serving(rows,enabled)
        with self.assertRaises(AssertionError):check_serving([],enabled)
    def test_missing_dram_evidence_keeps_look_ahead_unqualified(self):
        rows=self.pair()
        self.assertTrue(all(r['selected']=='ticket' for r in select(rows,ahead=True)))
    def test_wrong_bits_missing_device_or_more_dram_refuse_qualification(self):
        rows=self.pair();rows[0]['bitwise']=False
        with self.assertRaises(AssertionError):select(rows)
        with self.assertRaises(AssertionError):select(self.pair()[:1])
        rows=self.pair()
        for r in rows:r.update(dram_traffic_proven=True,baseline_dram_bytes=100,candidate_dram_bytes=101)
        self.assertTrue(all(r['selected']=='ticket' for r in select(rows,ahead=True)))
        for r in rows:r['candidate_dram_bytes']=100
        self.assertTrue(all(r['selected']=='ahead' for r in select(rows,ahead=True)))


class TraceTests(unittest.TestCase):
    def test_graph_attribution_uses_final_five_interleaved_samples(self):
        from fold_stream_report import with_trace
        baseline=['chop_mxfp8_ticket_init','chop_mxfp8_ticket_compute']
        candidate=['chop_mxfp8_stream_compute']
        a=[[(1,i*100,10+i*100,baseline[0]),(2,2+i*100,15+i*100,baseline[1])] for i in range(7)]
        b=[[(3,10000+i*100,10006+i*100,candidate[0])] for i in range(15)]
        r=dict(baseline_symbols=baseline,candidate_symbols=candidate,floor_us=.003)
        with_trace([r],[dict(symbols=sorted(baseline),replays=a),dict(symbols=candidate,replays=b)])
        self.assertAlmostEqual(r['exclusive_paired_ratio'],.4)
        self.assertAlmostEqual(r['exclusive_floor_ratio'],2)
    def test_counter_csv_requires_complete_graph_samples(self):
        import csv,tempfile
        from pathlib import Path
        from fold_stream_report import traffic
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'metrics.csv'
            with p.open('w') as f:
                w=csv.writer(f,quoting=csv.QUOTE_ALL)
                w.writerow(['ID','Kernel Name','Metric Name','Metric Unit','Metric Value'])
                for i in range(4):
                    w.writerow([i,'[CUDA Graph]','dram__bytes_read.sum','byte',100+i])
                    w.writerow([i,'[CUDA Graph]','dram__bytes_write.sum','byte',20])
            self.assertEqual(traffic(p),122)
            p.write_text(p.read_text().replace('dram__bytes_write.sum','missing'))
            with self.assertRaises(AssertionError):traffic(p)

if __name__=='__main__':unittest.main()

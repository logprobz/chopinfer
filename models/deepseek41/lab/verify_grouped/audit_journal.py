"""Audit verify4 admission using paired identical-resource fake-host journals."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import audit_selected_cache as cache
from audit_decode_attention import read,digest


def audit(root):
    stem='deepseek41-flash-tp4.Native.selected-cache.'
    base,bm=read(root/(stem+'reference.journal'));head,hm=read(root/(stem+'serving.journal'))
    lock,=json.loads((Path(__file__).resolve().parents[2]/'bundle/kernels.lock.d/latent_decode_cache.lock').read_text())
    symbols=[e['symbol'] for e in lock['entries']]
    result=cache.audit(base,head,bm,hm,symbols,False)
    assert result['added_producers']==800
    four=[line for line in head if line.startswith('launch ') and '_cached_profile0_' in line and '_verify' in line and cache.field(line,'grid_x')==4]
    assert len(four)==160
    assert set(result['source_substitutions'])=={s for s in symbols if s.endswith('_cached_profile0_v1')}
    rejected=cache.controls(head,hm)
    for mode in ['merge','order','cache_alias','consumer_stream']:
        changed=head.copy()
        at=next(i for i,line in enumerate(changed) if line.startswith('launch ') and '_cache_decode_control0_' in line)
        if mode=='merge':
            merge=next(i for i,line in enumerate(changed) if line.startswith('launch ') and 'chop_latent_sources_' in line and '_merge_' in line)
            changed[merge]=changed[merge].replace('grid_z: 1,','grid_z: 2,',1)
        elif mode=='order':changed[at],changed[at+1]=changed[at+1],changed[at]
        elif mode=='cache_alias':
            line=changed[at+1];views=list(cache.VIEW.finditer(line));native=views[0];target=views[-2]
            changed[at+1]=line[:target.start()]+target[0].replace('raw: '+target[1],'raw: '+native[1],1)+line[target.end():]
        else:
            words=changed[at+1].split();words[1]=str(int(words[1])+1);changed[at+1]=' '.join(words)
        assert changed!=head
        try:cache.audit(base,changed,bm,hm,symbols,False)
        except (AssertionError,KeyError,ValueError):rejected+=1
        else:raise AssertionError(mode+' mutation survived')
    assert rejected==16
    return dict(passed=True,reference=digest(base),candidate=digest(head),verify4_consumers=160,
                rejected_mutations=rejected,resources_identical=bm==hm,**result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--journals',type=Path,required=True);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    result=audit(args.journals);args.report.write_text(json.dumps(result,indent=2)+'\n')
    print('VERIFY4_JOURNAL_AUDIT_OK verify4_consumers=160 added_producers=800 rejected_mutations=16')

if __name__=='__main__':main()

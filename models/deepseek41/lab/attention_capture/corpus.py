"""Authenticate compact native attention captures without reading CUDA state."""
from pathlib import Path
import collections
import hashlib
import json
import re
import struct

SCHEMA = 'chopinfer.verify-attention-capture.v1'
GROUPS = [(f'c1_{n}_{letter}', 1, n, 1) for n in (8192,32768) for letter in 'ab'] + [(f'c8_{n}',8,n,2) for n in (8192,32768)]


def require(value, message):
    if not value: raise ValueError(message)


def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def words(data):
    require(len(data)%4==0,'unaligned captured words')
    return list(struct.unpack('<'+'I'*(len(data)//4),data))


def blob(root, record):
    sha=record['sha256'];require(re.fullmatch('[0-9a-f]{64}',sha),'invalid blob digest')
    require(record['file']==f'blobs/{sha}.bin','invalid blob path')
    path=root/record['file'];require(path.is_file() and not path.is_symlink(),'blob must be regular')
    data=path.read_bytes();require(len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==sha,'blob size/digest mismatch')
    return data


def origins(log):
    warmup=set();real=set();readiness=0
    for line in log.splitlines():
        if line.startswith('startup readiness:'): readiness+=1
        if not line.startswith('{'): continue
        value=json.loads(line)
        if value.get('schema')=='chopinfer.step.v1':
            for request,_ in value.get('completed_prompts',[]): (real if readiness else warmup).add(request)
    require(readiness==1 and not warmup&real,'ambiguous request provenance')
    return {'warmup':sorted(warmup),'real':sorted(real)}


def validate_record(root, record, rows):
    require(record['rows']==record['live_rows']==rows and 2<=rows<=32 and record['numerical_splits']==8,'capture row/split mismatch')
    layer=record['layer'];require(layer in (0,2,20),'unexpected layer')
    page=32 if layer==2 else 64;require(record['page']==page,'page format mismatch')
    old=record['original_scalars'];now=record['scalars']
    changed={'window_code_stride','window_scale_stride','window_sequences','page_stride','compressed_pages'}
    require(old.keys()==now.keys() and all(now[k]==v for k,v in old.items() if k not in changed),'unexpected scalar rewrite')
    require(now['window_budget']==128 and now['window_key_stride']>=128,'window selection geometry')
    data={name:blob(root,value) for name,value in record['buffers'].items()}
    require(len(data['q'])==rows*16*512*2 and len(data['row_batch'])==rows*4 and words(data['live_rows'])==[rows],'query/row extent')
    require(len(data['window_keys'])==rows*now['window_key_stride']*4,'window keys extent')
    require(len(data['sinks'])==16*4 and len(data['cos_sin'])==rows*64*4,'sink/RoPE extent')
    require(len(blob(root,record['expected_workspace']))==rows*8*16*514*4,'partial extent')
    require(len(blob(root,record['expected_output']))==rows*16*512*2,'output extent')
    batch=words(data['row_batch']);original=words(blob(root,record['original_window_items']))
    require(len(original)==old['window_items'],'original slot table extent')
    used=sorted({original[b] for b in batch if b<len(original) and original[b]<old['window_sequences']})
    require(used==record['physical_window_slots'],'window slot provenance')
    mapping={slot:i for i,slot in enumerate(used)}
    require(words(data['window_item_slots'])==[mapping.get(s,0xffffffff) for s in original],'window slot rebase')
    require(now['window_sequences']==max(1,len(used)) and now['window_code_stride']==now['window_slots']*512
            and now['window_scale_stride']==now['window_slots']*16,'window compact geometry')
    for name,stride in [('window_codes','window_code_stride'),('window_scales','window_scale_stride')]:
        require(len(data[name])==now['window_sequences']*now[stride],'window compact bytes')
    source=record['source_name']
    require('_verify' in source and '_draft_block' not in source,'not target verify attention')
    require(('_p32' in source)==(layer==2) and ('_window_' in source)==(layer==0),'wrong source layer format')
    prefix='chop_latent_sources_'+source
    cached=record['served_first_symbol']==prefix+'_cache_decode_control0_v1'
    require(record['served_first_symbol']==prefix+'_heads_v1' or cached,'unknown captured route')
    require(record['served_body_symbol']==prefix+('_cached_profile0_v1' if cached else '_heads_v1'),'body route mismatch')
    require(record['served_merge_symbol']==prefix+'_heads_merge_v1','merge route mismatch')
    if layer:
        require(now['compressed_budget']==512 and now['compressed_key_stride']>=512,'compressed selection geometry')
        keys=words(data['compressed_keys']);tables=words(blob(root,record['original_tables']))
        require(len(keys)==rows*now['compressed_key_stride'] and len(tables)==old['compressed_sequences']*old['max_pages'],'compressed metadata extent')
        pages=set()
        for row,b in enumerate(batch):
            if b>=old['compressed_sequences']:continue
            for key in keys[row*now['compressed_key_stride']:row*now['compressed_key_stride']+512]:
                if key==0xffffffff or key//page>=old['max_pages']:continue
                physical=tables[b*old['max_pages']+key//page]
                if physical<old['compressed_pages']:pages.add(physical)
        pages=sorted(pages);require(pages and pages==record['physical_pages'],'selected page provenance')
        mapping={p:i for i,p in enumerate(pages)}
        require(words(data['block_tables'])==[mapping.get(p,0xffffffff) for p in tables],'compressed page rebase')
        require(now['page_stride']==page*288 and now['compressed_pages']==len(pages)
                and len(data['compressed_pool'])==len(pages)*page*288,'compressed compact bytes')
    else:
        require(not record['physical_pages'] and not blob(root,record['original_tables']),'window has compressed pages')
    return {'layer':layer,'rows':rows,'cached':cached,'source':source,
            'partial_sha256':record['expected_workspace']['sha256'],'output_sha256':record['expected_output']['sha256']}


def seal(root, nonce, snapshot, request_origins):
    frames=[];counts=collections.Counter();records=0
    specifications={name:(items,context,quota) for name,items,context,quota in GROUPS}
    for path in sorted(root.glob('*/capture.json')):
        require(path.is_file() and not path.is_symlink(),'frame must be regular')
        value=json.loads(path.read_text());group=value['group'];require(group in specifications,'unknown capture group')
        items,context,quota=specifications[group];rows=value['rows']
        require(value['schema']==SCHEMA and value['phase']=='verify' and value['execution']=='eager-current-exact','capture schema or phase')
        require(value['rank']==0 and value['ranks']==4 and value['items']==items,'capture rank/items')
        require(value['nonce']==nonce and value['snapshot_sha256']==snapshot,'capture producer identity')
        require(value['arm']==dict(group=group,items=items,context=context,quota=quota,nonce=nonce),'capture arm mismatch')
        require(len(value['positions'])==len(value['token_ids'])==len(value['token_batch'])==rows,'row identity extent')
        require(all(context<=p<context+512 for p in value['positions']),'capture context differs')
        ids=set(value['request_ids']);require(len(ids)==items and ids<=set(request_origins['real']),'capture lacks real request provenance')
        require(set(value['token_batch'])==set(range(items)),'captured rows omit an item')
        require(len(value['records'])==3 and {r['layer'] for r in value['records']}=={0,2,20},'layer coverage')
        checked=[validate_record(root,r,rows) for r in value['records']]
        frames.append(dict(file=str(path.relative_to(root)),sha256=digest(path),group=group,rows=rows,records=checked))
        counts[group]+=1;records+=len(checked)
    require(counts==collections.Counter({name:quota for name,_,_,quota in GROUPS}),'incomplete capture groups')
    require(any(f['rows']==4 for f in frames if f['group'].startswith('c1_')),'no verify4 capture')
    require(all(any(f['rows']==32 and f['group']==name for f in frames) for name,items,_,_ in GROUPS if items==8),'no verify32 capture at each context')
    result=dict(schema='chopinfer.verify-attention-corpus.v1',complete=True,captures=frames,records=records,
                nonce=nonce,snapshot_sha256=snapshot,request_origins=request_origins)
    (root/'corpus.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def test_result(text, count):
    results=re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;',text)
    require(results==[('ok',str(count),'0','0')] and 'GPU work skipped' not in text,'test count or execution mismatch')


def replay_result(text, corpus):
    test_result(text,1)
    markers=[json.loads(line.split('ATTENTION_CAPTURE_REPLAY_OK ',1)[1]) for line in text.splitlines() if 'ATTENTION_CAPTURE_REPLAY_OK ' in line]
    wanted={Path(frame['file']).parent.name+f'-L{r["layer"]:02}':r for frame in corpus['captures'] for r in frame['records']}
    require(len(markers)==len(wanted)==corpus['records'],'replay record count')
    require({r['id'] for r in markers}==wanted.keys(),'replay identity coverage')
    for row in markers:
        expected=wanted[row['id']]
        require(row['poison_replays']==2 and all(row[k]==expected[k] for k in ('rows','cached','partial_sha256','output_sha256')),'replay bits or route proof')
    require(text.count(f'ATTENTION_CAPTURE_REPLAY_COMPLETE records={len(wanted)} poison_replays={len(wanted)*2}')==1,'replay completion marker')
    return dict(records=len(wanted),poison_replays=len(wanted)*2,bitwise=True)

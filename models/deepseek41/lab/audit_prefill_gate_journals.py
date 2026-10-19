from pathlib import Path
from collections import Counter
import json,re,sys
ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else None
ARG=re.compile(r'DeviceBuffer\(DeviceBufferSpanV1 \{ buffer: BundleDeviceBufferHandleV1 \{ raw: (\d+) \}, offset: (\d+), length: (\d+) \}\)|NullDevicePointer')
PREFIX='chop_exact_prefill_e384_before_gateup_decoded128_'
STAGES=['pack','predecode','r8','r32','r64','r128']
# Read actual scalar-free plans from the qualified family layout.
def shared(rows):
 raw=(128*32+rows*128+(128+rows)*4+15)//16*16
 return (2*raw+(128+rows)*72*2+rows*4+4+15)//16*16

def args(line):
 return [('span',*map(int,m.group(1,2,3))) if m[1] else None for m in ARG.finditer(line)]

def normalized(lines,collapse=False):
 functions={l.split()[1]:l.split()[2] for l in lines if l.startswith('resolve ')}
 maps={k:{} for k in ('buffer','stream','event','graph')}
 def handle(kind,value):
  table=maps[kind];value=str(value)
  if value not in table:table[value]=len(table)
  return table[value]
 def canonical(value):
  return ('span',handle('buffer',value[1]),*value[2:]) if isinstance(value,tuple) else value
 out=[];i=0;replaced=0
 while i<len(lines):
  line=lines[i];i+=1;word=line.split()[0]
  if word in ('resolve','release'):continue
  if word=='configure':
   _,ident,size=line.split();name=functions[ident]
   if name.startswith('chop_exact_prefill_'):continue
   out.append(f'configure {name} {size}');continue
  if collapse and line.startswith('launch ') and PREFIX in line:
   group=lines[i-1:i+5];assert len(group)==6
   stream=group[0].split()[1]
   for index,(record,stage) in enumerate(zip(group,STAGES)):
    assert record.startswith(f'launch {stream} {PREFIX}{stage}_v1 ')
    threads=32 if stage=='pack' else 256 if stage in ('predecode','r128') else 128
    grid=1 if stage=='pack' else 376
    dynamic=0 if index<2 else shared(int(stage[1:]))
    assert f'grid_x: {grid}, grid_y: 1, grid_z: 1,' in record
    assert f'block_x: {threads}, block_y: 1, block_z: 1,' in record
    assert f'shared_memory_bytes: {dynamic},' in record and 'flags: 1,' in record
   pack,pre=map(args,group[:2]);assert len(pack)==5 and len(pre)==4
   compute=[args(l) for l in group[2:]];assert all(x==compute[0] for x in compute)
   c=compute[0];assert len(c)==11 and c[-1] is None
   queue=pack[4];decoded=pre[3]
   assert queue[3]==83_923_200 and decoded==('span',queue[1],queue[2]+37_120,83_886_080)
   assert c[0]==decoded and c[7]==queue and c[3]==pre[1] and pack[3]==pre[2]==c[6]
   assert pack[0][3]==1540 and pack[2][3]==1540
   producer=lines[i-2]
   assert 'route_scatter_e384l384k6_t8192_p64_v1 ' in producer
   route=args(producer)
   assert pack[:4]==[route[1],route[2],route[6],route[5]]
   assert c[4:7]==[route[3],route[4],route[5]]
   # The six physical launches replace exactly one logical column call.
   h=[pre[0],c[1],c[2],pre[1],pack[1],c[4],c[5],c[6],c[8],c[9]]
   out.append(('gate',handle('stream',stream),tuple(map(canonical,h))))
   i+=5;replaced+=1;continue
  if collapse and line.startswith('launch ') and 'before_gateup_e384k6_n1152_k5120_r64_t8192_clamp4120000041200000_v1 ' in line:
   h=args(line);assert len(h)==10
   out.append(('gate',handle('stream',line.split()[1]),tuple(map(canonical,h))));replaced+=1;continue
  if word in ('stream','event'):
   x=line.split();line=f'{word} {handle(word,x[1])}'
  elif word in ('record','wait'):
   x=line.split();line=f'{word} {handle("stream",x[1])} {handle("event",x[2])}'
  elif word in ('captured','replay'):
   x=line.split();line=' '.join([word,str(handle('stream',x[1])),str(handle('graph',x[2]))]+x[3:])
  elif word in ('zero','launch','copy','download','capture','synchronize') or (word=='upload' and line.split()[1].isdigit()):
   x=line.split(' ',2);line=' '.join([x[0],str(handle('stream',x[1]))]+x[2:])
  line=re.sub(r'raw: (\d+)',lambda m:'raw: '+str(handle('buffer',m[1])),line)
  out.append(line)
 return out,replaced

def audit(old,new,kind,collapse=False):
 a=old.read_text().splitlines();b=new.read_text().splitlines()
 ac=Counter(l.split()[2] for l in a if l.startswith('resolve '));bc=Counter(l.split()[2] for l in b if l.startswith('resolve '))
 assert not ac-bc
 extra=bc-ac
 tp4='tp4' in old.name
 if kind=='prerequisite':
  assert all('_prefill_' in name and ('before_gateup_' in name or 'before_down_' in name) for name in extra)
  assert sum(extra.values())==(88 if tp4 else 0)
 elif kind=='serving':
  assert all(name.startswith('chop_exact_prefill_') for name in extra)
  assert sum(extra.values())==(1096 if tp4 else 0)
 else:raise AssertionError(kind)
 functions={l.split()[1]:l.split()[2] for l in b if l.startswith('resolve ')}
 configured=Counter((functions[l.split()[1]],int(l.split()[2])) for l in b if l.startswith('configure ') and functions[l.split()[1]].startswith('chop_exact_prefill_'))
 expected_config=Counter({(PREFIX+'r64_v1',54032):4,(PREFIX+'r128_v1',80400):4}) if kind=='serving' and tp4 else Counter()
 assert configured==expected_config

 full_lifecycle=old.name.startswith('deepseek41-flash-')
 expected_releases=sum(extra.values())+4 if tp4 and full_lifecycle else 0
 assert b.count('release')-a.count('release')==expected_releases
 left,n=normalized(a,collapse);right,m=normalized(b,collapse)
 assert n==m
 assert left==right,(old.name,next(((i,str(x)[:300],str(y)[:300]) for i,(x,y) in enumerate(zip(left,right)) if x!=y),('length',len(left),len(right))))
 assert sum(l.startswith('launch ') for l in b)-sum(l.startswith('launch ') for l in a)==(5*m if collapse and '8192' in old.name else 0)
 return {'journal':old.name,'stage':kind,'logical_gate_calls':n,'all_other_records_equal':True,'input_output_bindings_equal':True,'extra_resolved_functions':sum(extra.values()),'allocation_records_unchanged':True}

def run(root,serving_only=False):
 results=[]
 if not serving_only:
  for p in sorted((root/'main-journals').glob('*.journal')):
   results.append(audit(p,root/'baseline-journals'/p.name,'prerequisite'))
 for p in sorted((root/'baseline-journals').glob('*.journal')):
  results.append(audit(p,root/'serving-journals'/p.name,'serving'))
 for p in sorted((root/'baseline-prefill').glob('*.journal')):
  results.append(audit(p,root/'serving-prefill'/p.name,'serving',p.name.endswith('-8192.journal')))
 assert len(results)==(11 if serving_only else 19)
 old=(root/'baseline-prefill/tp4-native-8192.journal').read_text().splitlines()
 new=(root/'serving-prefill/tp4-native-8192.journal').read_text().splitlines()
 reference=normalized(old,True)[0]
 first=next(i for i,l in enumerate(new) if l.startswith('launch ') and PREFIX+'pack_v1 ' in l)
 controls=0
 for stage in range(6):
  for field in ['grid_x:','block_x:','shared_memory_bytes:']:
   changed=new.copy();line=changed[first+stage]
   changed[first+stage]=re.sub(re.escape(field)+r' (\d+)',lambda m:field+' '+str(int(m[1])+1),line,count=1)
   try: normalized(changed,True)
   except AssertionError: controls+=1
   else: raise AssertionError('launch geometry control escaped')
 for index in [0,1,2,3,4,5,6,8,9]:
  changed=new.copy()
  for pos in range(first+2,first+6):
   matches=list(ARG.finditer(changed[pos]));m=matches[index]
   start,end=m.span();value=m[0].replace('offset: '+m[2]+',','offset: '+str(int(m[2])+16)+',')
   changed[pos]=changed[pos][:start]+value+changed[pos][end:]
  try:
   assert normalized(changed,True)[0]==reference
  except AssertionError: controls+=1
  else: raise AssertionError('operand control escaped')
 report={'pairs':results,'negative_controls':controls,'all_passed':True}
 (root/'journal-audit.json').write_text(json.dumps(report,indent=2)+'\n')
 print('Audited',len(results),'journal pairs and',controls,'negative controls')

if __name__=='__main__':
 if ROOT is None or sys.argv[2:] not in ([],['--serving-only']):
  raise SystemExit('usage: audit_prefill_gate_journals.py EXTERNAL_ROOT [--serving-only]')
 run(ROOT,sys.argv[2:]==['--serving-only'])

"""Prove C1 launch substitutions and added private KV resources in full journals."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'tools/journal_proof'))
import history

DUAL='chop_mxfp8_dual_n1280_n512_k5120_g1_f32_v1'
Q='chop_deepseek41_mxfp8_stream_n1280_k5120_g1_f32_v1'
KV='chop_deepseek41_mxfp8_prefetch_n512_k5120_g1_f32_v3'
PACK='chop_deepseek41_mxfp8_stream_n512_k5120_g1_f32_v1_prepare_v1'
ARG=re.compile(r'DeviceBuffer\(DeviceBufferSpanV1 \{ buffer: BundleDeviceBufferHandleV1 \{ raw: (\d+) \}, offset: (\d+), length: (\d+) \}\)|NullDevicePointer|Scalar(U32|U64)\((\d+)\)')

def arguments(line):
 text=line.split(' } [',1)[1][:-1];out=[];end=0
 for match in ARG.finditer(text):
  assert text[end:match.start()] == ('' if not out else ', ')
  if match[1]:out.append(('b',int(match[1]),int(match[2]),int(match[3])))
  elif match[4]:out.append((match[4],int(match[5])))
  else:out.append(('null',))
  end=match.end()
 assert end==len(text)
 return out

def form(line):
 value=history.form(line)
 if value is not None:return value
 m=re.fullmatch(r'async mapped create (\d+)',line)
 if m:return ('mapped_create',m,('issued',))
 m=re.fullmatch(r'mapped gather [0-3] layers\.(1|14)\.engram\.embed\.(weight|scale)',line)
 if m:return ('mapped_gather',m,())
 if line=='serving kv':return ('serving',re.fullmatch('serving kv',line),())
 m=re.fullmatch(r'launch (\d+) (?:deepseek41_)?chop_\w+ LaunchConfigV1 \{.*\} \[.*\]',line)
 if m:return ('launch',m,('stream',))
 return None

def symbol(line):return line.split()[2] if line.startswith('launch ') else ''

def substitute(old,new,packed):
 skip=set();j=0;changed=0
 for i,line in enumerate(old):
  if i in skip:continue
  assert j<len(new),('missing operation',i)
  if line==new[j]:j+=1;continue
  assert symbol(line)==Q and symbol(new[j])==DUAL,(i,j,line,new[j])
  q=arguments(line);assert len(q)==15 and q[12:]==[('U64',0),('U32',1),('U32',4)]
  at=next(k for k in range(i+1,min(i+20,len(old))) if symbol(old[k])==KV+'_initialize_v3')
  assert all(symbol(l)!=Q for l in old[i+1:at])
  assert symbol(old[at+1])==KV
  k=arguments(old[at+1]);admit=arguments(old[at]);assert len(k)==12
  assert k[:2]==q[:2] and k[9]==q[10] and k[10:]==[('U32',1),('U32',2)]
  assert admit==[q[0],q[1],k[8],q[10],q[11],q[12],q[13]]
  assert old[at].split()[1]==line.split()[1]==old[at+1].split()[1]==new[j].split()[1]
  pack,state,workspace=packed[k[2]]
  expected=[*q[:4],q[6],q[8],q[9],pack,k[3],k[6],workspace,state,q[10],q[11],q[12],q[13],q[14],k[11]]
  assert arguments(new[j])==expected,('dual operands changed',i)
  header=line.split(' } [',1)[0].replace(Q,DUAL).replace('grid_x: 40,','grid_x: 720,').replace('grid_z: 10,','grid_z: 1,')
  assert new[j].split(' } [',1)[0]==header,('dual geometry changed',i)
  skip.update([at,at+1]);changed+=1;j+=1
 assert j==len(new),('extra operations',j,len(new))
 return changed

def captures(lines):
 active=None;nodes=0;graphs={};records=[]
 for line in lines:
  words=line.split()
  if line.startswith('capture '):
   assert active is None;active=int(words[1]);nodes=0
  elif line.startswith('launch ') and active is not None:nodes+=1
  elif line.startswith('captured '):
   stream,graph,count=map(int,words[1:]);assert active==stream and count==nodes
   assert graph not in graphs;graphs[graph]=count;records.append((stream,graph,count));active=None
  elif line.startswith('replay '):
   stream,graph=map(int,words[1:]);assert graph in graphs
 assert active is None
 return records

def graph_bodies(lines):
 # Partition cuts follow the unchanged graph-piece formula and move when the
 # program shrinks. Counts are checked against captured launches separately.
 return [l for l in lines if not l.startswith(('capture ','captured '))]

def proof(oldpath,newpath,completed=True):
 old=oldpath.read_text().splitlines();new=newpath.read_text().splitlines()
 ca,cb=captures(old),captures(new)
 assert len(ca)==len(cb)
 old=graph_bodies(old);new=graph_bodies(new)
 side=lambda p:json.loads(Path(str(p).removesuffix('.full')+'.handles.json').read_text())
 a,b=side(oldpath),side(newpath)
 if old==new:
  assert a==b
  return dict(identical=True,changes=0)
 extra_modules={int(i) for i,n in b['modules'].items() if n=='fold_dual_deepseek41_sm120'}
 assert len(extra_modules)==4
 ma=list(a['modules'].items());mb=[(i,n) for i,n in b['modules'].items() if int(i) not in extra_modules]
 assert [n for _,n in ma]==[n for _,n in mb]
 extra_functions={int(l.split()[1]) for l in new if l.startswith('resolve ') and l.split()[2] in (DUAL,DUAL+'_delayed_v1')}
 assert len(extra_functions)==8
 removed={i for i,l in enumerate(new) if l.startswith('resolve ') and int(l.split()[1]) in extra_functions}
 assert not any(l.startswith('configure ') and int(l.split()[1]) in extra_functions for l in new)
 alloc=[i for i,l in enumerate(new) if l.startswith('allocate ')]
 assert len(alloc)==len(b['buffers']);allocation=dict(zip(alloc,b['buffers']))
 private=set();packed={}
 for i,l in enumerate(new):
  if symbol(l)!=PACK:continue
  assert new[i-3:i]==['allocate 2621440 256','allocate 512 256','allocate 82432 256']
  v=arguments(l);assert len(v)==6 and v[1]==v[3]==('null',) and v[5]==('U32',128)
  handles=[allocation[j] for j in range(i-3,i)]
  assert v[2][1]&0xffffffff==handles[0] and v[4][1]&0xffffffff==handles[1]
  assert v[2][2:]==(0,2621440) and v[4][2:]==(0,512)
  workspace=('b',(v[2][1]&~0xffffffff)|handles[2],0,82432)
  assert v[0] not in packed;packed[v[0]]=(v[2],v[4],workspace)
  private.update(handles);removed.update(range(i-3,i+1))
 assert len(private)==480 and len(packed)==160
 stripped=[l for i,l in enumerate(new) if i not in removed]
 releases=len(extra_modules)+len(extra_functions)+len(private)
 if completed:
  assert stripped[-releases:]==['release']*releases;stripped=stripped[:-releases]
 remaining=[x for x in b['buffers'] if x not in private];assert len(remaining)==len(a['buffers'])
 buffers=dict(zip(a['buffers'],remaining));assert len(set(buffers.values()))==len(buffers)
 for x,y in buffers.items():assert a['ranks'][str(x)]==b['ranks'][str(y)]
 maps={'stream':{},'issued':dict((x[1],y[1]) for x,y in zip(ca,cb)),'function':dict((int(x),int(y)) for (x,_),(y,_) in zip(ma,mb))}
 definitions={'stream':(0,), 'event':(0,), 'resolve':(0,), 'captured':(1,)}
 left=[l for l in old if not l.startswith('launch ')];right=[l for l in stripped if not l.startswith('launch ')]
 assert len(left)==len(right),(len(left),len(right))
 for x,y in zip(left,right):
  f,g=form(x),form(y);assert f and g,(x,y)
  kind,m,roles=f;kind2,n,roles2=g;assert (kind,roles)==(kind2,roles2),(x,y)
  for at in definitions.get(kind,()):
   u,v=int(m.group(at+1)),int(n.group(at+1));role=roles[at]
   assert u not in maps[role] or maps[role][u]==v;maps[role][u]=v
 for mapping in maps.values():assert len(set(mapping.values()))==len(mapping)
 def normalize(line):
  kind,m,roles=form(line);value=line
  for at in reversed(range(len(roles))):
   raw=int(m.group(at+1));role=roles[at]
   assert raw in maps[role] or raw not in maps[role].values()
   value=value[:m.start(at+1)]+str(maps[role].get(raw,raw))+value[m.end(at+1):]
  def buffer(m):
   raw=int(m[2]);low=raw&0xffffffff
   assert low in buffers or low not in buffers.values()
   return m[1]+str((raw&~0xffffffff)|buffers.get(low,low))+m[3]
  return history.BUFFER_HANDLE.sub(buffer,value)
 for x,y in zip(ca,cb):assert maps['stream'].get(x[0],x[0])==y[0]
 normalized=[normalize(l) for l in old]
 normalized_other=[normalize(l) for l in left]
 if normalized_other!=right:
  at=next(i for i,(x,y) in enumerate(zip(normalized_other,right)) if x!=y)
  raise AssertionError((str(oldpath),at,normalized_other[at],right[at]))
 changed=substitute(normalized,stripped,packed)
 assert sum(x[2] for x in ca)-sum(x[2] for x in cb)==2*changed
 if changed:
  bad=stripped.copy();at=next(i for i,l in enumerate(bad) if symbol(l)==DUAL)
  bad[at]=bad[at].replace('length: 2048','length: 2049',1)
  try:substitute(normalized,bad,packed)
  except AssertionError:pass
  else:raise AssertionError('changed output escaped the proof')
 if oldpath.name.endswith('.full'):
  op=Path(str(oldpath).removesuffix('.full'));np=Path(str(newpath).removesuffix('.full'))
  body_changes=substitute([normalize(l) for l in graph_bodies(op.read_text().splitlines())],graph_bodies(np.read_text().splitlines()),packed)
  if oldpath.name.startswith('verify.'):assert body_changes==0
 return dict(identical=False,changes=changed,added_modules=4,added_resolves=8,added_allocations=480,added_weight_initializers=160,added_bytes=4*40*(2621440+512+82432),added_releases=releases if completed else 0,other_operations_equal=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();results=[]
 expected = {
  'lifecycle': {f'deepseek41-flash-tp{rank}.{kv}.journal' for rank in (1,2,4) for kv in ('native','bf16')}
   | {f'deepseek41-flash-tp4.{kv}.exact-experts.journal' for kv in ('native','bf16')}
   | {f'deepseek41-flash-tp{rank}.media.journal' for rank in (2,4)},
  'verify': {f'verify.tp4.r{rows}.Native.journal' for rows in (4,6,8,12,16,32)},
 }
 for kind,names in expected.items():
  for root in (a.before,a.after):
   assert {p.name for p in (root/kind).glob('*.journal')}==names,(root,kind)
  for old in sorted((a.before/kind).glob('*.journal')):
   new=a.after/kind/old.name;full=kind=='verify'
   r=proof(Path(str(old)+'.full') if full else old,Path(str(new)+'.full') if full else new,not full)
   results.append(dict(file=str(old.relative_to(a.before)),**r))
 assert len(results)==16
 for kv in ['Native','Bf16']:
  old=a.cases/f'decode.{kv}.referencetrue.journal.full';new=a.cases/f'decode.{kv}.referencefalse.journal.full'
  r=proof(old,new);assert r['changes']>=160
  results.append(dict(file=kv+' C1',**r))
 a.out.write_text(json.dumps(results,indent=2)+'\n');print('DUAL_JOURNAL_AUDIT',len(results))
if __name__=='__main__':main()

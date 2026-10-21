"""Prove that local-row admission adds only its declared module resources."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'tools/journal_proof'))
import history

CUBIN='fold_local_rows_sm120'
SYMBOLS=set()
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
 extra_modules={int(i) for i,n in b['modules'].items() if n==CUBIN}
 assert len(extra_modules)==4
 ma=list(a['modules'].items());mb=[(i,n) for i,n in b['modules'].items() if int(i) not in extra_modules]
 assert [n for _,n in ma]==[n for _,n in mb]
 extra_functions={int(l.split()[1]) for l in new if l.startswith('resolve ') and l.split()[2] in SYMBOLS}
 assert len(extra_functions)==32
 removed={i for i,l in enumerate(new) if l.startswith('resolve ') and int(l.split()[1]) in extra_functions}
 assert not any(l.startswith('configure ') and int(l.split()[1]) in extra_functions for l in new)
 private=set()
 assert not any(symbol(l) in SYMBOLS for l in new), 'unexpected local-row launch in a control history'
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
 assert normalized==stripped, 'retained launch changed'
 if oldpath.name.endswith('.full'):
  op=Path(str(oldpath).removesuffix('.full'));np=Path(str(newpath).removesuffix('.full'))
  assert [normalize(l) for l in graph_bodies(op.read_text().splitlines())]==graph_bodies(np.read_text().splitlines())
 return dict(identical=False,added_modules=4,added_resolves=32,added_releases=releases if completed else 0,allocations_equal=True,other_operations_equal=True)

def main():
 import shutil,tempfile
 p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--lock',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 [lock]=json.loads(a.lock.read_text());assert lock['name']==CUBIN
 SYMBOLS.update(e['symbol'] for e in lock['entries']);assert len(SYMBOLS)==8
 results=[];negative=0
 for kind,count in [('lifecycle',10),('verify',6)]:
  paths=sorted((a.before/kind).glob('*.journal'));assert len(paths)==count
  assert {p.name for p in paths}=={p.name for p in (a.after/kind).glob('*.journal')}
  for old in paths:
   new=a.after/kind/old.name;full=kind=='verify'
   op=Path(str(old)+'.full') if full else old;np=Path(str(new)+'.full') if full else new
   r=proof(op,np,not full);results.append(dict(file=str(old.relative_to(a.before)),**r))
   if r['identical']:continue
   with tempfile.TemporaryDirectory() as temp:
    mutant=Path(temp)/np.name;lines=np.read_text().splitlines()
    at=next(i for i,l in enumerate(lines) if l.startswith('launch '))
    lines[at]=lines[at].replace('block_x: ','block_x: 9',1);mutant.write_text('\n'.join(lines))
    shutil.copyfile(Path(str(new)+'.handles.json'),Path(temp)/(new.name+'.handles.json'))
    if full:shutil.copyfile(new,Path(temp)/new.name)
    try:proof(op,mutant,not full)
    except AssertionError:negative+=1
    else:raise AssertionError('changed launch escaped resource audit')
 assert len(results)==16 and negative==11
 a.out.write_text(json.dumps(dict(journals=results,rejected_mutations=negative),indent=2)+'\n')
 print('LOCAL_ROWS_RESOURCE_AUDIT',len(results),negative)
if __name__=='__main__':main()

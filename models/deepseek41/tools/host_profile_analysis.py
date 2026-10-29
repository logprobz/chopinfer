"""Require real host ranges and partition exposed GPU inactivity by innermost scope."""
from collections import Counter,defaultdict
import heapq,sqlite3

REQUIRED={'engine.step','host.bundle_execute','bundle.prefill.plan','bundle.prefill.upload','bundle.prefill.submit','bundle.prefill.status','bundle.verify.plan','bundle.verify.upload','bundle.verify.submit','bundle.verify.status','bundle.commit','bundle.draft.readback','bundle.route_diagnostics','host.kernel_submit','host.enqueue_mapped_gather','host.wait_mapped_gather','host.model_commit','host.readback','host.sampling'}

def merge(rows,lo,hi):
 out=[]
 for a,b in sorted(rows):
  a=max(a,lo);b=min(b,hi)
  if a>=b:continue
  if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
  else:out.append((a,b))
 return out

def idle(rows,lo,hi):
 at=lo;out=[]
 for a,b in merge(rows,lo,hi):
  if at<a:out.append((at,a))
  at=b
 if at<hi:out.append((at,hi))
 return out

def charge(gaps,segments):
 i=j=0;out=Counter()
 while i<len(gaps) and j<len(segments):
  a=max(gaps[i][0],segments[j][0]);b=min(gaps[i][1],segments[j][1])
  if a<b:out[segments[j][2]]+=b-a
  if gaps[i][1]<=segments[j][1]:i+=1
  else:j+=1
 assert sum(out.values())==sum(b-a for a,b in gaps)
 return dict(out)

def ownership(ranges,lo,hi):
 edges=defaultdict(list)
 for i,r in enumerate(ranges):
  a=max(lo,r['start']);b=min(hi,r['end'])
  if a<b:
   edges[a].append((1,i));edges[b].append((-1,i))
 edges[lo];edges[hi];active=set();heap=[];segments=[];previous=lo;route=set()
 for at,events in sorted(edges.items()):
  while heap and heap[0][1] not in active:heapq.heappop(heap)
  if at>previous:
   label='bundle.route_diagnostics' if route else (ranges[heap[0][1]]['name'] if heap else 'unannotated')
   segments.append((previous,at,label))
  for sign,i in events:
   if sign==1:
    active.add(i);heapq.heappush(heap,(-ranges[i]['start'],i))
    if ranges[i]['name']=='bundle.route_diagnostics':route.add(i)
   else:active.remove(i);route.discard(i)
  previous=at
 return segments

def osrt_evidence(rows,thread,lo,hi):
 # OSRT records only qualifying intercepted calls. An active collector does
 # not imply that the submission thread must issue one during every step.
 assert rows,'missing OS runtime collector records'
 thread_rows=[r for r in rows if r['globalTid']==thread and r['end']>r['start']]
 assert thread_rows,'missing submission-thread OS runtime coverage across capture'
 window=[r for r in rows if r['end']>lo and r['start']<hi]
 assert window,'missing OS runtime coverage across selected step'
 main=[r for r in window if r['globalTid']==thread]
 return dict(capture_calls=len(rows),capture_main_calls=len(thread_rows),step_calls=len(window),step_main_calls=len(main),main_raw_ns=dict(sum_counts(main,lo,hi)),step_calls_by_name=dict(Counter(r['name'] for r in window)))

def api_class(name):
 if 'Synchronize' in name:return 'synchronization'
 if 'Launch' in name:return 'launch'
 if 'Memcpy' in name:return 'copy'
 if 'Event' in name or 'StreamWait' in name:return 'event_or_stream'
 return 'other_cuda'

def validate(path,steps,anchors):
 from mixed_profile_analysis import clock_epoch
 db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
 tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
 assert {'NVTX_EVENTS','OSRT_API','CUPTI_ACTIVITY_KIND_RUNTIME'}<=tables,tables
 strings=dict(db.execute('select id,value from StringIds'))
 ranges=[]
 for raw in db.execute('select * from NVTX_EVENTS where end>start'):
  r=dict(raw);name=r.get('text') or strings.get(r.get('textId'))
  if name:r['name']=name;ranges.append(r)
 assert REQUIRED<={r['name'] for r in ranges},REQUIRED-{r['name'] for r in ranges}
 epoch,bracket=clock_epoch(dict(db.execute('select * from TARGET_INFO_SESSION_START_TIME').fetchone()),anchors)
 osrt_rows=[dict(r) for r in db.execute('select a.*,s.value name from OSRT_API a join StringIds s on a.nameId=s.id')]
 out=[]
 for step in steps:
  hi=step['completed_unix_ns']-epoch;lo=hi-step['execution_us']*1000
  apis=[dict(r) for r in db.execute('select a.*,s.value name from CUPTI_ACTIVITY_KIND_RUNTIME a join StringIds s on a.nameId=s.id where end>? and start<?',(lo,hi))]
  thread=Counter(r['globalTid'] for r in apis if 'LaunchKernel' in r['name'] or 'GraphLaunch' in r['name']).most_common(1)[0][0]
  selected=[r for r in ranges if r.get('globalTid')==thread and r['end']>lo and r['start']<hi]
  if step['prefill_tokens']==512:
   assert REQUIRED<={r['name'] for r in selected},REQUIRED-{r['name'] for r in selected}
   entries=step['route']['entries']
   assert entries.get('engram/prepared/prefill',{}).get('launches')==2
   assert sum(e['launches'] for e in entries.values() if e.get('symbol','').startswith('chop_comm_owned_sum_'))>=4
  segments=ownership(selected,lo,hi)
  gpu=[dict(r) for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET') for r in db.execute('select start,end,deviceId from '+table+' where end>? and start<?',(lo,hi))]
  allidle=idle([(r['start'],r['end']) for r in gpu],lo,hi)
  ranks={}
  for rank in range(4):
   gaps=idle([(r['start'],r['end']) for r in gpu if r['deviceId']==rank],lo,hi)
   ranks[rank]=dict(idle_ns=sum(b-a for a,b in gaps),scope_idle_ns=charge(gaps,segments))
  evidence=osrt_evidence(osrt_rows,thread,lo,hi)
  api_ranges=[]
  for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
   if table not in tables:continue
   for raw in db.execute('select a.*,s.value name from '+table+' a join StringIds s on a.nameId=s.id where end>? and start<? and globalTid=?',(lo,hi,thread)):
    r=dict(raw);r['name']=api_class(r['name']);api_ranges.append(r)
  api_segments=ownership(api_ranges,lo,hi)
  scope_duration=Counter()
  for a,b,name in segments:scope_duration[name]+=b-a
  out.append(dict(batch=step['batch'],mixed=bool(step['prefill_tokens']),execution_ns=hi-lo,main_global_tid=thread,range_counts=dict(Counter(r['name'] for r in selected)),all_rank_idle_ns=sum(b-a for a,b in allidle),all_rank_idle_scopes=charge(allidle,segments),all_rank_idle_cuda=charge(allidle,api_segments),ranks=ranks,osrt=evidence,scope_exclusive_ns=dict(scope_duration),scope_inclusive_ns=dict(sum_counts(selected,lo,hi))))
 db.close()
 assert len(out)==5 and sum(r['mixed'] for r in out)==1
 return dict(cpu_sampling=False,cpu_sampling_waived=True,clock_bracket_ns=bracket,steps=out,method='Innermost main-thread NVTX scope intersected with GPU-idle intervals. Route diagnostics own their nested logger time. Scope names are not proof of CPU utilization; inspect CUDA and OS runtime calls inside waits. Unannotated, engine.step, host.bundle_execute and submit remainder stay unresolved.')

def sum_counts(rows,lo,hi):
 out=Counter()
 for r in rows:out[r['name']]+=min(hi,r['end'])-max(lo,r['start'])
 return out


"""Read process fault counters without counting child totals twice."""
from pathlib import Path
import json,time

def parse_stat(text):
    pid=int(text.split(' ',1)[0]); fields=text.rsplit(')',1)[1].split()
    return {'pid':pid,'state':fields[0],'pgrp':int(fields[2]),'start_ticks':int(fields[19]),
            'minor_faults':int(fields[7]),'major_faults':int(fields[9])}

def snapshot(server):
    group=json.loads(server.pidfile.read_text())['pgid'];rows={}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            row=parse_stat(path.read_text())
            if row['pgrp']==group and row['state']!='Z':rows[str(row['pid'])]=row
        except (OSError,ValueError):pass
    if not rows:raise RuntimeError('no live server processes for fault counters')
    return {'monotonic_seconds':time.monotonic(),'process_group':group,'processes':rows}

def difference(before,after):
    if before['process_group']!=after['process_group']:raise ValueError('process group changed')
    rows=[]
    for pid,end in after['processes'].items():
        start=before['processes'].get(pid)
        if start is not None and start['start_ticks']!=end['start_ticks']:raise ValueError('PID reused within a timing window')
        row={'pid':int(pid),'start_ticks':end['start_ticks'],'born_during_window':start is None}
        for key in ('minor_faults','major_faults'):
            row[key]=end[key]-(start[key] if start else 0)
            if row[key]<0:raise ValueError('fault counter decreased')
        rows.append(row)
    missing=sorted(set(before['processes'])-set(after['processes']))
    return {'process_group':after['process_group'],'processes':rows,
            'minor_faults':sum(r['minor_faults'] for r in rows),
            'major_faults':sum(r['major_faults'] for r in rows),
            'exited_processes_without_final_counter':missing,'complete':not missing,
            'scope':'Each live process own minflt/majflt, excluding child-total fields; deltas bracket this window.'}

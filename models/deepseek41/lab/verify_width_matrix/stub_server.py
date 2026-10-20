"""CPU-only HTTP fixture. Never used by a staged GPU job."""
import collections
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import sys
import threading
import time

from run import CONFIGS, MODEL

PORT, CONFIG = int(sys.argv[1]), sys.argv[2]
FAULT = os.environ.get("X7_STUB_FAULT", "")
LOCK = threading.Lock()
PENDING = []
COUNTERS = collections.Counter()
DRAFTED = collections.Counter()
ACCEPTED = collections.Counter()
HISTOGRAM = collections.Counter()
STEP = 0


def log(row):
    print(json.dumps(row), flush=True)




def plan(c):
    cap, mode = CONFIGS[CONFIG]
    if cap == 0:
        return [0]*c
    left = 32-c
    if mode == "fixed":
        out=[]
        for _ in range(c):
            k=min(cap,left);out.append(k);left-=k
        return out
    depth = min(cap,left//c)
    out=[depth]*c
    for i in range(min(c,left-depth*c)):
        if out[i]<cap:out[i]+=1
    return out


def run_batch(rows):
    global STEP
    count = len(rows)
    tokens = rows[0][1]["max_tokens"]
    assert all(body["max_tokens"] == tokens for _,body in rows)
    shape=plan(count)
    verify=sum(k>0 for k in shape)
    decode=count-verify
    widths=sum(k+1 for k in shape if k)
    if verify and COUNTERS["verify_items"] == 0:
        for cache_id in [1, 2]:
            log(dict(schema="chopinfer.mapped-row-cache.v1", phase="ready", cache_id=cache_id,
                     charged_bytes=1 << 30, budget_bytes=2 << 30, locked_bytes=(1 << 30)-4096,
                     process_charged_bytes=cache_id*(1 << 30)))
    for handler, _ in rows:
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
    for position in range(tokens):
        with LOCK:
            STEP += 1
            step=STEP
            if not (FAULT == "zero-steps" and CONFIG == "balanced"):
                if CONFIG != "k0":
                    log(dict(schema="chopinfer.deepseek41.step-shape.v1",instance_id="1",step_id=str(step),
                             items=count,drafting=verify,bulk_passes=0,micros={"target":1000}))
                if verify and not (FAULT == "inactive" and CONFIG == "balanced"):
                    log(dict(schema="chopinfer.device-results.v1",cycle_epoch=step,kind="acceptance_packet"))
                for index,k in enumerate(shape):
                    if k:
                        log(dict(schema="chopinfer.deepseek41.drafter-confidence.v1",instance_id="1",step_id=str(step),
                                 request_id=str(index+1),completion="committed",verify_rows=k+1,emitted_positions=k,
                                 outcomes=["unknown"]*k+["not_emitted"]*(5-k)))
                capacity=decode+widths if verify else 1<<(count-1).bit_length()
                log(dict(schema="chopinfer.step.v1",batch=step,prefill_items=0,prefill_tokens=0,
                         decode_rows=decode,verify_items=verify,verify_rows=widths,live_rows=decode+widths,
                         row_capacity=capacity,execution_us=1000,completed_unix_ns=time.time_ns(),
                         host={"boundary_us":1000},
                         speculation=None if CONFIG == "k0" else dict(mode=CONFIGS[CONFIG][1], reason="drafting" if CONFIG == "adaptive" else CONFIGS[CONFIG][1], planned_drafts=sum(shape))))
                HISTOGRAM[decode,widths,capacity] += 1
                COUNTERS.update(verify_items=verify,draft_tokens=sum(shape),emitted_tokens=verify,rows=widths)
                for j in range(1,6):DRAFTED[j]+=sum(k>=j for k in shape)
        for handler, body in rows:
            token = 100+(body["seed"]+position)%64
            label = "t"+str(token)
            score = -(token%11+1)/4
            if FAULT == "logprob" and CONFIG == "balanced" and position==130:
                score=math.nextafter(score,0)
            top={str(i):-float(i+1) for i in range(5)}
            top[label]=score
            if FAULT == "token" and CONFIG == "balanced" and position==130:token+=1
            chunk={"choices":[{"token_ids":[token],"logprobs":{"tokens":[label],"token_logprobs":[score],
                    "top_logprobs":[top]},"finish_reason":None}]}
            handler.wfile.write(b'data: '+json.dumps(chunk).encode()+b'\n\n');handler.wfile.flush()
        time.sleep(.003 if CONFIG == "adaptive" else .0008)
    for handler,_ in rows:
        final={"choices":[{"finish_reason":"length","token_ids":[]}],
               "usage":{"prompt_tokens":21,"completion_tokens":tokens,"total_tokens":tokens+21}}
        handler.wfile.write(b'data: '+json.dumps(final).encode()+b'\n\ndata: [DONE]\n\n');handler.wfile.flush()
        handler.done.set()


def metrics():
    labels=f'model_name="{MODEL}",engine="0"'
    lines=[]
    if COUNTERS["verify_items"] and not (FAULT == "missing-counters" and CONFIG == "balanced"):
        for key in ("verify_items","draft_tokens","accepted_tokens","emitted_tokens","rows"):
            lines.append(f'chop_spec_{key}_total{{{labels}}} {COUNTERS[key]}')
        for position in range(1,17):
            for name,values in [("drafted",DRAFTED),("accepted",ACCEPTED)]:
                lines.append(f'chop_spec_{name}_position_total{{{labels},position="{position}"}} {values[position]}')
    for (decode,verify,capacity),n in HISTOGRAM.items():
        lines.append(f'chop_step_rows_total{{{labels},decode_rows="{decode}",verify_rows="{verify}",row_capacity="{capacity}"}} {n}')
    lines.append(f'chop_step_rows_dropped_total{{{labels}}} 0')
    return '\n'.join(lines)+'\n'


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*_args):pass
    def do_GET(self):
        with LOCK:
            data = metrics().encode() if self.path=="/metrics" else b'{"status":"ok"}'
        self.send_response(200);self.end_headers();self.wfile.write(data)
    def do_POST(self):
        global PENDING
        body=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        count=int(self.headers["X-X7-Test-Concurrency"])
        self.done=threading.Event()
        with LOCK:
            PENDING.append((self,body))
            if len(PENDING)==count:
                batch=PENDING;PENDING=[]
                threading.Thread(target=run_batch,args=(batch,),daemon=True).start()
        assert self.done.wait(30)


class Server(ThreadingHTTPServer):
    request_queue_size = 64


Server(("127.0.0.1",PORT),Handler).serve_forever()

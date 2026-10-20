"""Versioned timing payload and thread-safe episode metric aggregation."""
import csv,json,math,statistics,threading,time
from pathlib import Path
SCHEMA='bench_timing_v1'
FIELDS=('observation_encode_ms','generation_head_ms','selection_ms','selection_background_ms','server_total_ms','request_ms')


def synchronized_time(device):
    import torch
    if torch.device(device).type=='cuda':torch.cuda.synchronize(device)
    return time.perf_counter()


class EpisodeTiming:
    def __init__(self, folder):
        self.path=Path(folder)/'request_timing.jsonl'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.rows={};self.closed=set();self.lock=threading.Lock()

    def record(self, episode, payload, request_ms, accepted=True):
        timing=payload.get('bench_timing') or {}
        valid=timing.get('schema')==SCHEMA
        row=dict(episode=int(episode),wall_time=time.time(),accepted=bool(accepted),
                 timing_supported=valid,warmup=bool(timing.get('warmup',False)),
                 batch_size=timing.get('batch_size'),candidates=timing.get('candidates'))
        for key in FIELDS:
            value=request_ms if key=='request_ms' else timing.get(key) if valid else None
            row[key]=float(value) if isinstance(value,(int,float)) and math.isfinite(value) and value>=0 else None
        with self.lock:
            row['accepted']=row['accepted'] and episode not in self.closed
            with self.path.open('a') as f:f.write(json.dumps(row)+'\n')
            if row['accepted']:self.rows.setdefault(int(episode),[]).append(row)

    def summary(self, episode):
        with self.lock:
            self.closed.add(int(episode));rows=list(self.rows.get(int(episode),[]))
        result=dict(timing_requests=len(rows),timing_warmup_requests=sum(r['warmup'] for r in rows),
                    timing_supported_requests=sum(r['timing_supported'] for r in rows))
        for key in FIELDS:
            vals=sorted(r[key] for r in rows if not r['warmup'] and r[key] is not None)
            result[key+'_count']=len(vals)
            for stat in ('mean','p50','p95'):result[key+'_'+stat]=None
            result[key.removesuffix('_ms')+'_hz']=None
            if vals:
                def percentile(q):
                    pos=q*(len(vals)-1);lo=int(pos);hi=min(lo+1,len(vals)-1)
                    return vals[lo]+(pos-lo)*(vals[hi]-vals[lo])
                mean=statistics.fmean(vals)
                result.update({key+'_mean':mean,key+'_p50':percentile(.5),key+'_p95':percentile(.95),
                               key.removesuffix('_ms')+'_hz':1000/mean if mean>0 else None})
        return result

    def write_metrics(self, records, filename):
        # Write once through a temp file: no partially updated metric.csv.
        merged=[dict(r,**self.summary(int(r['episode']))) for r in records]
        if not merged:return
        fields=list(dict.fromkeys(k for r in merged for k in r))
        path=Path(filename);tmp=path.with_suffix('.csv.tmp')
        with tmp.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(merged)
        tmp.replace(path)

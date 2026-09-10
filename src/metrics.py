"""Incident metrics with label-independent aggregation of alarm episodes."""
import numpy as np

def intervals(binary):
    b=np.asarray(binary,dtype=int)
    if b.ndim!=1 or not np.isin(b,[0,1]).all(): raise ValueError('Expected binary vector')
    d=np.diff(np.r_[0,b,0])
    return [(int(a),int(b)) for a,b in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1))]

def aggregate(prediction, gap):
    if gap<0: raise ValueError('Negative merge gap')
    groups=[]
    for a,b in intervals(prediction):
        if groups and a-groups[-1]['end']<=gap:
            groups[-1]['end']=b; groups[-1]['runs'].append((a,b))
        else: groups.append(dict(start=a,end=b,runs=[(a,b)]))
    return groups

def event_metrics(labels, prediction, gap=30, sample_minutes=1):
    if len(labels)!=len(prediction) or len(labels)==0: raise ValueError('Empty or unequal vectors')
    if sample_minutes<=0: raise ValueError('Invalid sampling interval')
    truth=intervals(labels); alarms=aggregate(prediction,gap)
    # An aggregated interval is NOT filled with synthetic positive predictions.
    # Credit requires overlap with an actual threshold exceedance.
    edges=[[i for i,(a,b) in enumerate(truth)
            if any(c<b and d>a for c,d in alarm['runs'])] for alarm in alarms]
    used=set(); matched=[]
    for alarm_idx, candidates in enumerate(edges):
        for event_idx in candidates:
            if event_idx not in used:
                used.add(event_idx); matched.append((alarm_idx,event_idx)); break
    tp=len(matched); false=sum(not e for e in edges)
    duplicate=sum(bool(e) for e in edges)-tp
    precision=tp/len(alarms) if alarms else 0.
    recall=tp/len(truth) if truth else 0.
    delays=[]
    for a,e in matched:
        t0,t1=truth[e]
        hit=min(max(t0,c) for c,d in alarms[a]['runs'] if c<t1 and d>t0)
        delays.append(hit-t0)
    days=len(labels)*sample_minutes/1440
    result=dict(precision=precision,recall=recall,
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
        incidents=len(truth),alarm_episodes=len(alarms),matched=tp,missed=len(truth)-tp,
        false_alarms=false,duplicate_alarms=duplicate,false_alarms_per_day=false/days,
        unmatched_alarms_per_day=(false+duplicate)/days,exposure_days=days,
        median_delay_minutes=float(np.median(delays)*sample_minutes) if delays else None,
        raw_alarm_fraction=float(np.mean(prediction)),
        boundary_incidents=sum(a==0 or b==len(labels) for a,b in truth),
        interpretation='observed fragments at boundaries counted; no point adjustment')
    return result,alarms,truth,matched

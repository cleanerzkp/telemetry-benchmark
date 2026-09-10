"""Reproducible local benchmark with distinct prepare / fit / test phases."""
import argparse,csv,hashlib,json,platform,subprocess,time
from pathlib import Path
import joblib
import numpy as np
import sklearn,torch
from threadpoolctl import threadpool_limits
from src.metrics import event_metrics
from src.models import fit_models,score_models
ROOT=Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')
def utc(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def cfg(): return json.loads((ROOT/'config.json').read_text())
def git(*args): return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
def code_hashes():
    names=['PROTOCOL.md','config.json','run.py','src/models.py','src/metrics.py','requirements-lock.txt']
    return {n:sha(ROOT/n) for n in names}
def load(name):
    x=np.loadtxt(ROOT/'data'/name,delimiter=',')
    if not np.isfinite(x).all(): raise ValueError('Non-finite data: '+name)
    if 'labels' in name and not np.isin(x,[0,1]).all(): raise ValueError('Non-binary labels')
    return x

def prepare():
    c=cfg(); d=ROOT/'data'
    if (d/'manifest.json').exists(): raise SystemExit('Data already prepared')
    train=(d/'train.raw').read_bytes().splitlines(keepends=True)
    evaluation=(d/'evaluation.raw').read_bytes().splitlines(keepends=True)
    labels=(d/'labels.raw').read_bytes().splitlines(keepends=True)
    if len(labels)!=len(evaluation): raise ValueError('Length mismatch')
    a=int(len(train)*c['fit_fraction']); b=int(len(evaluation)*c['validation_fraction']); gap=c['purge_samples']
    slices={'fit.csv':train[:a],'calibration.csv':train[a+gap:],
            'validation.csv':evaluation[:b], 'validation_labels.csv':labels[:b],
            'holdout.csv':evaluation[b+gap:],'holdout_labels.csv':labels[b+gap:]}
    # Copy bytes only. Held-out values and labels are not parsed or inspected.
    for name,lines in slices.items(): (d/name).write_bytes(b''.join(lines))
    manifest=dict(entity=c['entity'],source_commit=c['source_commit'],prepared_at=utc(),
        source_hashes={n:sha(d/n) for n in ['train.raw','evaluation.raw','labels.raw']},
        files={n:dict(rows=len(v),sha256=sha(d/n)) for n,v in slices.items()},
        ranges={'fit':[0,a],'calibration':[a+gap,len(train)],
                'validation':[0,b],'holdout':[b+gap,len(evaluation)]},
        interval='half-open, zero-based; calibration uses train.raw, validation/holdout use evaluation.raw')
    write(d/'manifest.json',manifest); print(json.dumps(manifest['ranges'],indent=2))

def verify_data():
    m=json.loads((ROOT/'data/manifest.json').read_text())
    for n,v in m['files'].items():
        if sha(ROOT/'data'/n)!=v['sha256']: raise ValueError('Changed data '+n)

def fit():
    out=ROOT/'results'; out.mkdir(exist_ok=False)
    commit=git('rev-parse','HEAD')
    if git('status','--porcelain','--untracked-files=no'): raise ValueError('Tracked files must be committed')
    verify_data(); c=cfg(); start=time.monotonic()
    x=load('fit.csv')
    if x.ndim!=2 or x.shape[1]!=38: raise ValueError('Expected 38 channels')
    model=fit_models(x,c); joblib.dump(model,out/'models.joblib',compress=3)
    calibration,_=score_models(model,load('calibration.csv'),c)
    thresholds={n:float(np.quantile(s,c['threshold_quantile'])) for n,s in calibration.items()}
    scores,_=score_models(model,load('validation.csv'),c)
    y=load('validation_labels.csv')[c['rolling_window']:].astype(int)
    metrics={n:event_metrics(y,s>thresholds[n],c['alarm_merge_gap'],c['sample_minutes'])[0] for n,s in scores.items()}
    chosen=min(['rolling','iforest'],key=lambda n:(-metrics[n]['f1'],metrics[n]['false_alarms_per_day'],n))
    freeze=dict(protocol_commit=commit,code_hashes=code_hashes(),manifest_sha256=sha(ROOT/'data/manifest.json'),
        model_sha256=sha(out/'models.joblib'),thresholds=thresholds,selected_baseline=chosen,
        validation_metrics=metrics,frozen_at=utc(),fit_calibration_validation_seconds=time.monotonic()-start,
        plot_channel=model['plot_channel'],versions={'python':platform.python_version(),'numpy':np.__version__,
        'sklearn':sklearn.__version__,'torch':torch.__version__},training_loss=model['loss'])
    write(out/'freeze.json',freeze)
    print(json.dumps({'chosen_baseline':chosen,'thresholds':thresholds,'validation':metrics},indent=2))

def test():
    c=cfg(); out=ROOT/'results'; frozen=json.loads((out/'freeze.json').read_text())
    if git('status','--porcelain','--untracked-files=no'): raise ValueError('Freeze must be committed')
    git('ls-files','--error-unmatch','results/freeze.json')
    if code_hashes()!=frozen['code_hashes']: raise ValueError('Protocol/code changed after freeze')
    if sha(ROOT/'data/manifest.json')!=frozen['manifest_sha256']: raise ValueError('Manifest changed')
    if sha(out/'models.joblib')!=frozen['model_sha256']: raise ValueError('Model changed')
    verify_data()
    # Exclusive marker is written BEFORE accessing held-out numerical data.
    with (out/'TEST_OPENED.json').open('x') as f:
        json.dump({'opened_at':utc(),'freeze_commit':git('rev-parse','HEAD'),
                   'freeze_sha256':sha(out/'freeze.json')},f,indent=2)
    start=time.monotonic(); x=load('holdout.csv')
    # Load only model artifacts created by this local pipeline.
    model=joblib.load(out/'models.joblib'); scores,contributions=score_models(model,x,c)
    y=load('holdout_labels.csv')[c['rolling_window']:].astype(int)
    results={}; alert_rows=[]; incident_rows=[]
    for name,s in scores.items():
        pred=s>frozen['thresholds'][name]
        m,alarms,truth,pairs=event_metrics(y,pred,c['alarm_merge_gap'],c['sample_minutes'])
        results[name]=m; paired=dict(pairs)
        for i,alarm in enumerate(alarms):
            alert_rows.append([name,i,alarm['start'],alarm['end'],paired.get(i,''),json.dumps(alarm['runs'])])
        for i,(a,b) in enumerate(truth):
            match=next((j for j,k in pairs if k==i),None)
            incident_rows.append([name,i,a,b,match if match is not None else '',
                                  ';'.join(map(str,np.argsort(contributions[a:b].mean(0))[-3:][::-1]))])
    write(out/'metrics.json',dict(methods=results,selected_baseline=frozen['selected_baseline'],
        evaluated_at=utc(),test_seconds=time.monotonic()-start,freeze_sha256=sha(out/'freeze.json'),
        test_samples=len(y),positive_samples=int(y.sum()),seed=c['seed']))
    np.savez_compressed(out/'scores.npz',labels=y,telemetry=x[c['rolling_window']:],
                        **scores,ae_contributions=contributions)
    for filename,header,rows in [
        ('alerts.csv',['method','alarm_id','start','end_exclusive','matched_incident','actual_positive_runs'],alert_rows),
        ('incidents.csv',['method','incident_id','start','end_exclusive','matched_alarm','ae_top3_channels_zero_based'],incident_rows)]:
        with (out/filename).open('w',newline='') as f: w=csv.writer(f);w.writerow(header);w.writerows(rows)
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','fit','test']);a=p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1): globals()[a.command]()

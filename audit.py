"""Audit saved artifacts only. Does not load holdout input or execute models."""
from pathlib import Path
import csv,hashlib,json,subprocess
import numpy as np
from pypdf import PdfReader
from src.metrics import event_metrics
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    freeze=json.loads((ROOT/'results/freeze.json').read_text())
    marker=json.loads((ROOT/'results/TEST_OPENED.json').read_text())
    metrics=json.loads((ROOT/'results/metrics.json').read_text())
    cfg=json.loads((ROOT/'config.json').read_text())
    for name,h in freeze['code_hashes'].items():assert sha(ROOT/name)==h,name
    assert sha(ROOT/'results/models.joblib')==freeze['model_sha256']
    assert sha(ROOT/'data/manifest.json')==freeze['manifest_sha256']
    assert marker['freeze_sha256']==sha(ROOT/'results/freeze.json')==metrics['freeze_sha256']
    subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',freeze['protocol_commit'],marker['freeze_commit']],check=True)
    assert freeze['frozen_at']<=marker['opened_at']<=metrics['evaluated_at']
    frozen_in_git=subprocess.check_output(['git','-C',str(ROOT),'show',marker['freeze_commit']+':results/freeze.json'])
    assert frozen_in_git==(ROOT/'results/freeze.json').read_bytes()
    val=freeze['validation_metrics']
    selected=min(['rolling','iforest'],key=lambda n:(-val[n]['f1'],val[n]['false_alarms_per_day'],n))
    assert selected==freeze['selected_baseline']==metrics['selected_baseline']
    saved=np.load(ROOT/'results/scores.npz')
    assert len(saved['labels'])==metrics['test_samples']
    for name,recorded in metrics['methods'].items():
        # Independent reproduction of arithmetic from saved scores, not another model run.
        checked=event_metrics(saved['labels'],saved[name]>freeze['thresholds'][name],
                              cfg['alarm_merge_gap'],cfg['sample_minutes'])[0]
        assert checked==recorded,name
        assert recorded['alarm_episodes']==recorded['matched']+recorded['false_alarms']+recorded['duplicate_alarms']
    with (ROOT/'reports/metrics_table.csv').open() as stream:
        for row in csv.DictReader(stream):
            for key in ['precision','recall','f1','false_alarms_per_day']:
                assert float(row[key])==metrics['methods'][row['method']][key]
    pages=PdfReader(ROOT/'reports/RAPORT.pdf').pages;assert len(pages)==2
    print('PASS: frozen code/model, commit order, baseline selection, saved-score arithmetic, CSV and two-page PDF')
if __name__=='__main__':main()

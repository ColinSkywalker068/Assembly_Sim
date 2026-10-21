"""Offline proxy-label discrimination audit; thresholds are NOT deployment settings."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata


def ranking_metrics(y,p):
    positive=int(y.sum()); negative=len(y)-positive
    if not positive or not negative: raise ValueError('both classes required')
    auc=(rankdata(p)[y].sum()-positive*(positive+1)/2)/(positive*negative)
    order=np.argsort(-p,kind='stable'); scores=p[order]; labels=y[order]
    ends=np.r_[np.flatnonzero(np.diff(scores)!=0),len(scores)-1]
    tp=np.cumsum(labels)[ends]; recall=tp/positive; precision=tp/(ends+1)
    ap=np.sum(np.diff(np.r_[0,recall])*precision)
    return float(auc),float(ap)


def evaluate(path):
    with np.load(path,allow_pickle=False) as data:
        p=data['probability']; y=data['contact_proxy'].astype(bool)
    auc,ap=ranking_metrics(y,p)
    sweep=[]
    for threshold in (.001,.01,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9):
        pred=p>threshold; tp=int((pred&y).sum()); fp=int((pred&~y).sum()); fn=int((~pred&y).sum())
        sweep.append({'threshold':threshold,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None})
    return {'label_type':'offline geometric contact proxy','n':len(y),'positive_fraction':float(y.mean()),
        'roc_auc':auc,'average_precision':ap,
        'thresholds':sweep,'threshold_selected_for_deployment':None}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(evaluate(args.input),indent=2)+'\n')

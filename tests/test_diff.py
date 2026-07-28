import pandas as pd
import os, unittest
if not os.path.exists('data/esa/opssat/segments.csv'):
    raise unittest.SkipTest('ESA dataset not found, skipping benchmark test.')
from operational.anomaly_detector import fast_ks_2samp
from collections import deque
import numpy as np
import os, unittest
if not os.path.exists('data/esa/opssat/segments.csv'):
    raise unittest.SkipTest('ESA dataset not found, skipping benchmark test.')

df = pd.read_csv('data/esa/opssat/segments.csv')
mag = df[(df['channel'] == 'CADC0872') & (df['train'] == False)]

def run_detector(use_diff):
    tp=0; fp=0; fn=0; tn=0
    for seg_id, grp in mag.groupby('segment'):
        vals = grp['value'].values
        is_anom = grp['anomaly'].max() > 0
        
        if use_diff:
            vals = np.diff(vals)
            
        cur_q = deque(maxlen=48)
        ref_q = deque(maxlen=128)
        
        alarm = False
        for v in vals:
            if len(cur_q) >= 48 and len(ref_q) >= 20:
                d, pval = fast_ks_2samp(list(ref_q), list(cur_q))
                if pval < 1e-4:
                    alarm = True
                    break
            cur_q.append(v)
            if len(cur_q) % 2 == 0:
                ref_q.append(v)
                
        if is_anom and alarm: tp+=1
        elif not is_anom and alarm: fp+=1
        elif is_anom and not alarm: fn+=1
        else: tn+=1
        
    prec = tp/(tp+fp) if (tp+fp)>0 else 0
    rec = tp/(tp+fn) if (tp+fn)>0 else 0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    print(f"Diff={use_diff}: TP={tp}, FP={fp}, FN={fn}, TN={tn}, P={prec:.2f}, R={rec:.2f}, F1={f1:.2f}")

run_detector(False)
run_detector(True)

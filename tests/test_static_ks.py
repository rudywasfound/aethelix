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
import random
random.seed(42)
np.random.seed(42)

df = pd.read_csv('data/esa/opssat/segments.csv')

train_dist = {}
for ch in df['channel'].unique():
    train_vals = df[(df['channel'] == ch) & (df['train'] == True)]['value'].values
    if len(train_vals) > 2000:
        train_dist[ch] = np.random.choice(train_vals, 2000, replace=False).tolist()
    else:
        train_dist[ch] = train_vals.tolist()

tp=0; fp=0; fn=0; tn=0
test_df = df[df['train'] == False]
for seg_id, grp in test_df.groupby('segment'):
    ch = grp['channel'].iloc[0]
    vals = grp['value'].values
    is_anom = grp['anomaly'].max() > 0
    
    ref_q = train_dist[ch]
    cur_q = deque(maxlen=64)
    
    alarm = False
    alarm_streak = 0
    
    p_thresh = 1e-6 if 'CADC088' in ch or 'CADC089' in ch else 1e-4
    persist = 12 if 'CADC088' in ch or 'CADC089' in ch else 4
    
    for v in vals:
        cur_q.append(v)
        if len(cur_q) >= 64 and len(ref_q) >= 20:
            d, pval = fast_ks_2samp(ref_q, list(cur_q))
            if pval < p_thresh:
                alarm_streak += 1
                if alarm_streak >= persist:
                    alarm = True
                    break
            else:
                alarm_streak = 0
            
    if is_anom and alarm: tp+=1
    elif not is_anom and alarm: fp+=1
    elif is_anom and not alarm: fn+=1
    else: tn+=1
    
prec = tp/(tp+fp) if (tp+fp)>0 else 0
rec = tp/(tp+fn) if (tp+fn)>0 else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
print(f"Static KS: TP={tp}, FP={fp}, FN={fn}, TN={tn}, P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")

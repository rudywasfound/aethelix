import pandas as pd
import os, unittest
if not os.path.exists('data/esa/opssat/segments.csv'):
    raise unittest.SkipTest('ESA dataset not found, skipping benchmark test.')
import numpy as np
import os, unittest
if not os.path.exists('data/esa/opssat/segments.csv'):
    raise unittest.SkipTest('ESA dataset not found, skipping benchmark test.')

df = pd.read_csv('data/esa/opssat/segments.csv')

train_df = df[df['train'] == True]
test_df = df[df['train'] == False]

bounds = {}
for ch in df['channel'].unique():
    vals = train_df[train_df['channel'] == ch]['value'].values
    if len(vals) == 0: continue
    mean = np.mean(vals)
    std = np.std(vals)
    bounds[ch] = (mean - 3.5 * std, mean + 3.5 * std)

tp=0; fp=0; fn=0; tn=0
for seg_id, grp in test_df.groupby('segment'):
    vals = grp['value'].values
    ch = grp['channel'].iloc[0]
    is_anom = grp['anomaly'].max() > 0
    
    if ch not in bounds: continue
    
    alarm = False
    lower, upper = bounds[ch]
    for v in vals:
        if v < lower or v > upper:
            alarm = True
            break
            
    if is_anom and alarm: tp+=1
    elif not is_anom and alarm: fp+=1
    elif is_anom and not alarm: fn+=1
    else: tn+=1

prec = tp/(tp+fp) if (tp+fp)>0 else 0
rec = tp/(tp+fn) if (tp+fn)>0 else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
print(f"Static Bounds: TP={tp}, FP={fp}, FN={fn}, TN={tn}, P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")

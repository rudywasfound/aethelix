import pandas as pd
import os, unittest
if not os.path.exists('data/esa/opssat/segments.csv'):
    raise unittest.SkipTest('ESA dataset not found, skipping benchmark test.')
import numpy as np
import os, unittest
if not os.path.exists('data/esa/opssat/segments.csv'):
    raise unittest.SkipTest('ESA dataset not found, skipping benchmark test.')
from sklearn.linear_model import Ridge

df = pd.read_csv('data/esa/opssat/segments.csv')

def extract_ar(vals, lags=10):
    X, y = [], []
    for i in range(lags, len(vals)):
        X.append(vals[i-lags:i])
        y.append(vals[i])
    return np.array(X), np.array(y)

train = df[df['train'] == True]
test = df[df['train'] == False]

models = {}
residuals = {}

for ch in df['channel'].unique():
    train_vals = train[train['channel'] == ch]['value'].values
    if len(train_vals) < 20: continue
    X, y = extract_ar(train_vals, lags=10)
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    models[ch] = model
    
    preds = model.predict(X)
    res = np.abs(y - preds)
    residuals[ch] = (np.mean(res), np.std(res))

tp=0; fp=0; fn=0; tn=0
for seg_id, grp in test.groupby('segment'):
    ch = grp['channel'].iloc[0]
    vals = grp['value'].values
    is_anom = grp['anomaly'].max() > 0
    
    if ch not in models or len(vals) < 11:
        if not is_anom: tn+=1
        else: fn+=1
        continue
        
    X, y = extract_ar(vals, lags=10)
    preds = models[ch].predict(X)
    res = np.abs(y - preds)
    
    mean_res, std_res = residuals[ch]
    threshold = mean_res + 3.5 * std_res
    
    if np.any(res > threshold):
        if is_anom: tp+=1
        else: fp+=1
    else:
        if is_anom: fn+=1
        else: tn+=1

prec = tp/(tp+fp) if (tp+fp)>0 else 0
rec = tp/(tp+fn) if (tp+fn)>0 else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
print(f"AR Model: TP={tp}, FP={fp}, FN={fn}, TN={tn}, P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")

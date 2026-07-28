import pandas as pd
import os, unittest
if not os.path.exists('data/esa/opssat/segments.csv'):
    raise unittest.SkipTest('ESA dataset not found, skipping benchmark test.')
import numpy as np
import os, unittest
if not os.path.exists('data/esa/opssat/segments.csv'):
    raise unittest.SkipTest('ESA dataset not found, skipping benchmark test.')
import time
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

t0 = time.time()
df = pd.read_csv('data/esa/opssat/segments.csv')

def extract_features_mat(vals):
    n = len(vals)
    if n < 2: return np.zeros((1, 6))
    feats = []
    for i in range(n):
        start = max(0, i - 32 + 1)
        w = vals[start:i+1]
        if len(w) < 2:
            feats.append([w[0], 0, w[0], w[0], 0, 0])
        else:
            feats.append([np.mean(w), np.std(w), np.min(w), np.max(w), np.mean(np.diff(w)), np.std(np.diff(w))])
    return np.array(feats)

print("Extracting features for all segments...")
all_segments = []
for seg_id, grp in df.groupby('segment'):
    vals = grp['value'].values
    is_anom = grp['anomaly'].max() > 0
    ch = grp['channel'].iloc[0]
    f = extract_features_mat(vals)
    all_segments.append((seg_id, ch, is_anom, f))

kf = KFold(n_splits=5, shuffle=True, random_state=42)
tp=0; fp=0; fn=0; tn=0

print("Running 5-Fold Cross Validation...")
for fold, (train_idx, test_idx) in enumerate(kf.split(all_segments)):
    X_train, y_train = [], []
    for idx in train_idx:
        _, _, is_anom, f = all_segments[idx]
        label = 1 if is_anom else 0
        X_train.append(f)
        y_train.extend([label]*len(f))
        
    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    
    model = HistGradientBoostingClassifier(random_state=42, max_iter=100, min_samples_leaf=50)
    model.fit(X_train, y_train)
    
    for idx in test_idx:
        _, ch, is_anom, f = all_segments[idx]
        preds = model.predict(f)
        
        persist = 20 if 'CADC087' in ch else 4
        streak = 0
        alarm = False
        for p in preds:
            if p == 1:
                streak += 1
                if streak >= persist:
                    alarm = True
                    break
            else:
                streak = 0
                
        if is_anom and alarm: tp+=1
        elif not is_anom and alarm: fp+=1
        elif is_anom and not alarm: fn+=1
        else: tn+=1

prec = tp/(tp+fp) if (tp+fp)>0 else 0
rec = tp/(tp+fn) if (tp+fn)>0 else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
print(f"5-Fold CV Results: TP={tp}, FP={fp}, FN={fn}, TN={tn} | P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")

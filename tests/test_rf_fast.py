import pandas as pd
import os, unittest
if not os.path.exists('data/esa/opssat/segments.csv'):
    raise unittest.SkipTest('ESA dataset not found, skipping benchmark test.')
import numpy as np
import os, unittest
if not os.path.exists('data/esa/opssat/segments.csv'):
    raise unittest.SkipTest('ESA dataset not found, skipping benchmark test.')
import time
from sklearn.ensemble import RandomForestClassifier

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

train_df = df[df['train'] == True]
test_df = df[df['train'] == False]

X_train, y_train = [], []
for seg_id, grp in train_df.groupby('segment'):
    vals = grp['value'].values
    label = 1 if grp['anomaly'].max() > 0 else 0
    f = extract_features_mat(vals)
    X_train.append(f)
    y_train.extend([label]*len(f))

X_train = np.vstack(X_train)
y_train = np.array(y_train)

print(f"Training RF on {len(y_train)} samples...")
model = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

test_segments = []
for seg_id, grp in test_df.groupby('segment'):
    vals = grp['value'].values
    is_anom = grp['anomaly'].max() > 0
    ch = grp['channel'].iloc[0]
    f = extract_features_mat(vals)
    preds = model.predict(f)
    test_segments.append((ch, is_anom, preds))

for p_mag in [10, 15, 20, 25]:
    for p_pd in [2, 4, 6, 8]:
        tp=0; fp=0; fn=0; tn=0
        for ch, is_anom, preds in test_segments:
            persist = p_mag if 'CADC087' in ch else p_pd
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
        print(f"p_mag={p_mag:2d}, p_pd={p_pd:2d} -> TP={tp:3d}, FP={fp:3d}, FN={fn:2d}, TN={tn:3d} | P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")

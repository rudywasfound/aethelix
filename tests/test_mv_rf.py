import pandas as pd
import os, unittest
if not os.path.exists('data/esa/opssat/segments.csv'):
    raise unittest.SkipTest('ESA dataset not found, skipping benchmark test.')
import numpy as np
import os, unittest
if not os.path.exists('data/esa/opssat/segments.csv'):
    raise unittest.SkipTest('ESA dataset not found, skipping benchmark test.')
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv('data/esa/opssat/segments.csv')

df_p = df.pivot(index=['segment', 'timestamp', 'anomaly', 'train'], columns='channel', values='value').reset_index()
df_p = df_p.dropna()

channels = [c for c in df_p.columns if c.startswith('CADC')]

train_df = df_p[df_p['train'] == True]
test_df = df_p[df_p['train'] == False]

X_train = train_df[channels].values
y_train = train_df['anomaly'].values
y_train = np.where(y_train > 0, 1, 0)

model = HistGradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

tp=0; fp=0; fn=0; tn=0
for seg_id, grp in test_df.groupby('segment'):
    X_test = grp[channels].values
    is_anom = grp['anomaly'].max() > 0
    
    preds = model.predict(X_test)
    alarm = np.any(preds == 1)
            
    if is_anom and alarm: tp+=1
    elif not is_anom and alarm: fp+=1
    elif is_anom and not alarm: fn+=1
    else: tn+=1

prec = tp/(tp+fp) if (tp+fp)>0 else 0
rec = tp/(tp+fn) if (tp+fn)>0 else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
print(f"MV Model: TP={tp}, FP={fp}, FN={fn}, TN={tn}, P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")

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

def extract_features(arr):
    if len(arr) < 2: return [arr[0]]*6
    return [np.mean(arr), np.std(arr), np.min(arr), np.max(arr), np.mean(np.diff(arr)), np.std(np.diff(arr))]

train_df = df[df['train'] == True]
test_df = df[df['train'] == False]

X_train, y_train = [], []
for seg_id, grp in train_df.groupby('segment'):
    vals = grp['value'].values
    label = 1 if grp['anomaly'].max() > 0 else 0
    for i in range(len(vals)):
        start = max(0, i-32+1)
        X_train.append(extract_features(vals[start:i+1]))
        y_train.append(label)

model = HistGradientBoostingClassifier(random_state=42, max_iter=100)
model.fit(X_train, y_train)

fps = {}
for seg_id, grp in test_df.groupby('segment'):
    vals = grp['value'].values
    is_anom = grp['anomaly'].max() > 0
    ch = grp['channel'].iloc[0]
    
    alarm = False
    streak = 0
    for i in range(len(vals)):
        start = max(0, i-32+1)
        feat = extract_features(vals[start:i+1])
        if model.predict([feat])[0] == 1:
            streak += 1
            if streak >= 12:
                alarm = True
                break
        else:
            streak = 0
            
    if not is_anom and alarm:
        fps[ch] = fps.get(ch, 0) + 1

print("False Positives by channel at Persist=12:")
for ch, count in fps.items():
    print(f"  {ch}: {count}")

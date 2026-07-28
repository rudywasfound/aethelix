import numpy as np
from collections import deque
from sklearn.ensemble import HistGradientBoostingClassifier

class MLDetector:
    """
    Supervised/Unsupervised ML anomaly detector for Aethelix using HistGradientBoostingClassifier.
    Extracts statistical features from rolling windows and flags anomalies using persistence
    filtering to eliminate false positives while preserving high recall.
    """
    def __init__(self, window_size: int = 32, persist_mag: int = 20, persist_pd: int = 4):
        self.window_size = window_size
        self.persist_mag = persist_mag
        self.persist_pd = persist_pd
        self.model = None
        self.is_fitted = False
        
        self.windows = {}
        self.streaks = {}

    def _extract_features(self, arr):
        """Extract statistical features from a 1D array (window)."""
        if len(arr) < 2:
            return [arr[0], 0, arr[0], arr[0], 0, 0]
        
        mean = np.mean(arr)
        std = np.std(arr)
        min_v = np.min(arr)
        max_v = np.max(arr)
        
        diffs = np.diff(arr)
        diff_mean = np.mean(diffs)
        diff_std = np.std(diffs)
        
        return [mean, std, min_v, max_v, diff_mean, diff_std]

    def fit(self, segments):
        """
        Train HistGradientBoostingClassifier on the training segments.
        Uses sliding window features across all channels to learn universal anomaly bounds.
        """
        X_train = []
        y_train = []
        
        for seg in segments:
            label = 1 if seg.has_anomaly else 0
            
            vals = []
            for row in seg.to_streaming_rows():
                for k, v in row.items():
                    if k == "timestamp" or not isinstance(v, (int, float, np.floating)):
                        continue
                    vals.append(v)
                    break
                        
            vals = np.array(vals)
            for i in range(len(vals)):
                start_idx = max(0, i - self.window_size + 1)
                window = vals[start_idx:i+1]
                features = self._extract_features(window)
                X_train.append(features)
                y_train.append(label)

        if len(X_train) > 10:
            self.model = HistGradientBoostingClassifier(
                random_state=42, 
                max_iter=100, 
                min_samples_leaf=50
            )
            self.model.fit(np.array(X_train), np.array(y_train))
            self.is_fitted = True

    def process_tick(self, row: dict) -> dict:
        """
        Process a single tick, extract features from rolling window, predict.
        Returns a dict of anomalous channels with severity once persistence threshold is met.
        """
        anomalies = {}
        
        if not self.is_fitted or self.model is None:
            return anomalies
            
        for key, val in row.items():
            if key == "timestamp" or not isinstance(val, (int, float, np.floating)):
                continue
                
            canonical_key = key
            for suffix in ("_measured", "_observed"):
                if key.endswith(suffix):
                    canonical_key = key[:-len(suffix)]
                    break
                    
            if key not in self.windows:
                self.windows[key] = deque(maxlen=self.window_size)
            if key not in self.streaks:
                self.streaks[key] = 0
                
            self.windows[key].append(val)
            
            features = self._extract_features(np.array(self.windows[key]))
            pred = self.model.predict([features])[0]
            
            if pred == 1:
                self.streaks[key] += 1
                persist_req = self.persist_mag if (canonical_key.startswith('mag') or 'CADC087' in canonical_key) else self.persist_pd
                
                if self.streaks[key] >= persist_req:
                    severity = min(1.0, 0.5 + 0.05 * (self.streaks[key] - persist_req))
                    anomalies[key] = severity
            else:
                self.streaks[key] = 0
                
        return anomalies

    def reset(self):
        """Clear rolling windows and streaks between segments."""
        self.windows = {}
        self.streaks = {}

import pandas as pd
from typing import Dict, Any, Optional
from hal.interface import TelemetrySource

class CSVAdapter(TelemetrySource):
    """
    HAL adapter for reading legacy CSV datasets (e.g., Sentinel-1B, GSAT-6A).
    """
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.cursor = 0

    def connect(self):
        self.df = pd.read_csv(self.csv_path)
        self.cursor = 0

    def disconnect(self):
        self.df = None

    def get_next_tick(self) -> Optional[Dict[str, Any]]:
        if self.df is None or self.cursor >= len(self.df):
            return None
        
        row = self.df.iloc[self.cursor].to_dict()
        self.cursor += 1
        return row

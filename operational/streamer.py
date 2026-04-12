import time
import threading
import pandas as pd
from queue import Queue

class TelemetryStreamer:
    """
    Simulates real-time telemetry downlink by reading a CSV and pushing
    rows to a thread-safe Queue at a configurable speed playback.
    """
    def __init__(self, csv_path: str = None, df: pd.DataFrame = None, speed: float = 1.0, orbit_duration_s: float = 5400.0):
        """
        Args:
            csv_path: Path to telemetry CSV.
            df: Optional preexisting dataframe (preferred over csv_path if dynamically generated).
            speed: Playback speed. 1.0 = real-time, 10.0 = 10x faster. 0 = ASAP.
            orbit_duration_s: Low Earth Orbit duration (default 90 mins).
        """
        self.csv_path = csv_path
        self.speed = speed
        self.orbit_duration_s = orbit_duration_s
        self.queue = Queue()
        
        if df is not None:
            self.df = df.copy()
        elif csv_path is not None:
            self.df = pd.read_csv(csv_path, parse_dates=['timestamp'])
        else:
            raise ValueError("Must provide either csv_path or df")
        
        # Calculate synthetic orbital phase based on timestamps.
        # Assuming timestamp epoch corresponds to phase 0.
        timestamps_s = self.df['timestamp'].astype('int64') // 10**9
        epoch = timestamps_s.iloc[0] if len(timestamps_s) > 0 else 0
        self.df['orbital_phase'] = ((timestamps_s - epoch) % self.orbit_duration_s) / self.orbit_duration_s
        
        self.is_running = False

    def start(self):
        """Starts the background producer thread."""
        self.is_running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self.is_running = False
        if hasattr(self, '_thread'):
            self._thread.join()

    def _run(self):
        """Push rows into the queue respecting the playback speed."""
        if len(self.df) == 0:
            return

        # Keep track of simulation time
        start_real_time = time.time()
        start_sim_time = self.df.iloc[0]['timestamp'].timestamp()

        for idx, row in self.df.iterrows():
            if not self.is_running:
                break
                
            current_sim_time = row['timestamp'].timestamp()
            
            if self.speed > 0:
                elapsed_sim = current_sim_time - start_sim_time
                target_elapsed_real = elapsed_sim / self.speed
                actual_elapsed_real = time.time() - start_real_time
                
                sleep_time = target_elapsed_real - actual_elapsed_real
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # push as a dict
            self.queue.put(row.to_dict())

        # Sentinel to indicate end of stream
        self.queue.put(None)

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class TelemetrySource(ABC):
    """
    Hardware Abstraction Layer (HAL) interface for Aethelix.
    Ensures the core diagnostic engine is agnostic to the input transport.
    """
    
    @abstractmethod
    def connect(self):
        """Initialize Connection to hardware or data source."""
        pass

    @abstractmethod
    def disconnect(self):
        """Gracefully close the connection."""
        pass

    @abstractmethod
    def get_next_tick(self) -> Optional[Dict[str, Any]]:
        """
        Poll for the next set of telemetry readings.
        Returns a dict mapping channel names to floating point values.
        """
        pass

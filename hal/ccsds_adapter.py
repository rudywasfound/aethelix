from typing import Dict, Any, Optional
from hal.interface import TelemetrySource
from ingestion.ccsds_parser import CCSDSParser

class CCSDSAdapter(TelemetrySource):
    """
    HAL adapter for live CCSDS telemetry streams.
    Can ingest from a byte-stream (e.g., TCP socket or binary file).
    """
    def __init__(self, stream_source):
        self.stream = stream_source
        self.parser = CCSDSParser()
        self.buffer = b""

    def connect(self):
        # In a real environment, this might open a socket
        pass

    def disconnect(self):
        # Close the socket
        pass

    def get_next_tick(self) -> Optional[Dict[str, Any]]:
        """
        Extracts the next full packet from the stream and decodes it.
        This provides the mechanism for real-time agency-standard ingestion.
        """
        # 1. Read the 6-byte primary header
        header_data = self.stream.read(6)
        if not header_data:
            return None
            
        header = self.parser.parse_header(header_data)
        
        # 2. Read the data field
        payload = self.stream.read(header.data_length)
        if not payload:
            return None
            
        # 3. Decode APID mapping
        return self.parser.decode_payload(payload, header.apid)

import struct
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass

try:
    from aethelix_core import PyCCSDSParser
    RUST_CORE_AVAILABLE = True
except ImportError:
    RUST_CORE_AVAILABLE = False
    PyCCSDSParser = None

@dataclass
class CCSDSPrimaryHeader:
    version: int
    packet_type: int
    secondary_header: bool
    apid: int
    sequence_flags: int
    sequence_count: int
    data_length: int

class CCSDSParser:
    """
    Native implementation of CCSDS 133.0-B-2 Space Packet Protocol.
    Uses high-performance Rust core for bit-level ingestion if available.
    """
    
    HEADER_SIZE = 6

    def __init__(self):
        if RUST_CORE_AVAILABLE:
            self.rust_parser = PyCCSDSParser()
        else:
            self.rust_parser = None
        self.packet_buffer = b""

    def parse_header(self, raw_bytes: bytes) -> CCSDSPrimaryHeader:
        # Fallback to Python if Rust is not available
        if not RUST_CORE_AVAILABLE:
            b0, b1 = raw_bytes[0], raw_bytes[1]
            apid = ((b0 & 0x07) << 8) | b1
            b2, b3 = raw_bytes[2], raw_bytes[3]
            seq_count = ((b2 & 0x3F) << 8) | b3
            length = struct.unpack(">H", raw_bytes[4:6])[0] + 1
            return CCSDSPrimaryHeader(0, 0, False, apid, 0, seq_count, length)
        
        # In Rust mode, we don't usually call parse_header standalone,
        # but for compatibility:
        self.rust_parser.push_bytes(list(raw_bytes))
        p = self.rust_parser.next_packet()
        if not p:
            raise ValueError("Incomplete or invalid CCSDS packet")
        return CCSDSPrimaryHeader(0, 0, False, p.apid, 0, p.sequence_count, len(p.payload))

    def get_packets(self, data: bytes):
        """Streaming generator for Space Packets."""
        if self.rust_parser:
            self.rust_parser.push_bytes(list(data))
            while True:
                p = self.rust_parser.next_packet()
                if not p: break
                yield p.apid, bytes(p.payload)
        else:
            # Legacy Python streaming logic
            self.packet_buffer += data
            while len(self.packet_buffer) >= self.HEADER_SIZE:
                h = self.parse_header(self.packet_buffer)
                total_len = self.HEADER_SIZE + h.data_length
                if len(self.packet_buffer) < total_len:
                    break
                payload = self.packet_buffer[self.HEADER_SIZE : total_len]
                self.packet_buffer = self.packet_buffer[total_len:]
                yield h.apid, payload

    def decode_payload(self, payload: bytes, apid: int) -> Dict[str, float]:
        """
        Maps APIDs to telemetry fields.
        In a real ISRO mission, this would look up the Packet ID (PID)
        in a mission-specific XML/CSV database.
        """
        # Example Mapping for Aethelix Demo:
        # APID 0x100 -> Power Subsystem
        # APID 0x200 -> Thermal Subsystem
        # APID 0x300 -> ADCS Subsystem
        
        data = {}
        if apid == 0x100: # Power
            # Assuming floating point values (4 bytes each)
            vals = struct.unpack(">ffff", payload[:16])
            data = {
                "solar_input": vals[0],
                "battery_voltage": vals[1],
                "battery_charge": vals[2],
                "bus_voltage": vals[3]
            }
        elif apid == 0x300: # ADCS
            vals = struct.unpack(">ffff", payload[:16])
            data = {
                "pointing_error": vals[0],
                "wheel_speed": vals[1],
                "wheel_current": vals[2],
                "gyro_bias": vals[3]
            }
        return data

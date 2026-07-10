//! Native CCSDS 133.0-B-2 Space Packet Protocol implementation.
//! Provides high-performance, memory-safe parsing of satellite telemetry streams.

use serde::{Serialize, Deserialize};
use crate::error::{Result, Error};

/// CCSDS Primary Header (6 bytes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpacePacketHeader {
    pub version: u8,
    pub packet_type: u8,
    pub secondary_header_flag: bool,
    pub apid: u16,
    pub sequence_flags: u8,
    pub sequence_count: u16,
    pub data_length: u16, // Actual length is data_length + 1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpacePacket {
    pub header: SpacePacketHeader,
    pub payload: Vec<u8>,
}

impl SpacePacket {
    pub const HEADER_SIZE: usize = 6;

    /// Parse a Space Packet from a raw byte buffer
    pub fn parse(raw: &[u8]) -> Result<Self> {
        if raw.len() < Self::HEADER_SIZE {
            return Err(Error::StreamError("Insufficient bytes for CCSDS header".to_string()));
        }

        // Byte 0 & 1: | Ver(3) | Type(1) | Sec(1) | APID(11) |
        let b0 = raw[0];
        let b1 = raw[1];
        
        let version = (b0 >> 5) & 0x07;
        let packet_type = (b0 >> 4) & 0x01;
        let secondary_header_flag = ((b0 >> 3) & 0x01) == 1;
        let apid = (((b0 & 0x07) as u16) << 8) | (b1 as u16);

        // Byte 2 & 3: | Seq Flags(2) | Seq Count(14) |
        let b2 = raw[2];
        let b3 = raw[3];
        let sequence_flags = (b2 >> 6) & 0x03;
        let sequence_count = (((b2 & 0x3F) as u16) << 8) | (b3 as u16);

        // Byte 4 & 5: | Data Length(16) |
        let data_length = ((raw[4] as u16) << 8) | (raw[5] as u16);
        let actual_data_len = (data_length as usize) + 1;

        if raw.len() < Self::HEADER_SIZE + actual_data_len {
            return Err(Error::StreamError("Payload length mismatch".to_string()));
        }

        let header = SpacePacketHeader {
            version,
            packet_type,
            secondary_header_flag,
            apid,
            sequence_flags,
            sequence_count,
            data_length,
        };

        let payload = raw[Self::HEADER_SIZE..Self::HEADER_SIZE + actual_data_len].to_vec();

        Ok(Self { header, payload })
    }
}

pub struct CCSDSStreamParser {
    buffer: Vec<u8>,
}

impl CCSDSStreamParser {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    pub fn next_packet(&mut self) -> Result<Option<SpacePacket>> {
        if self.buffer.len() < SpacePacket::HEADER_SIZE {
            return Ok(None);
        }

        // Peek at header to get length
        let data_len = ((self.buffer[4] as u16) << 8) | (self.buffer[5] as u16);
        let total_len = SpacePacket::HEADER_SIZE + (data_len as usize) + 1;

        if self.buffer.len() < total_len {
            return Ok(None);
        }

        let packet_raw = &self.buffer[..total_len];
        match SpacePacket::parse(packet_raw) {
            Ok(packet) => {
                self.buffer.drain(..total_len);
                Ok(Some(packet))
            }
            Err(e) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ccsds_parse() {
        // Mock Packet: APID 0x123, Len 4 (5 bytes payload)
        // b0: 0x01 (Ver 0, Type 0, Sec 0, APID high 0x01)
        // b1: 0x23 (APID low 0x23)
        // b2: 0xC0 (Seq Flags 11, Seq Count 0)
        // b3: 0x00
        // b4: 0x00
        // b5: 0x04 (Len 4 -> 5 bytes)
        let mut raw = vec![0x01, 0x23, 0xC0, 0x00, 0x00, 0x04];
        raw.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00]);

        let packet = SpacePacket::parse(&raw).unwrap();
        assert_eq!(packet.header.apid, 0x123);
        assert_eq!(packet.payload.len(), 5);
        assert_eq!(packet.payload[0], 0xDE);
    }

    #[test]
    fn test_oversized_packet_overflow() {
        // data_length = 0xFFFF implies 65536 payload bytes
        let mut raw = vec![0x01, 0x23, 0xC0, 0x00, 0xFF, 0xFF];
        // Provide only 10 bytes of payload instead of 65536
        raw.extend_from_slice(&[0x00; 10]);
        let res = SpacePacket::parse(&raw);
        assert!(res.is_err());
    }

    #[test]
    fn test_stream_parser_awaits_full_oversized_packet() {
        let mut parser = CCSDSStreamParser::new();
        // data_length = 0xFFFF implies 65536 payload bytes
        let mut raw = vec![0x01, 0x23, 0xC0, 0x00, 0xFF, 0xFF];
        raw.extend_from_slice(&[0x00; 10]);
        parser.push_bytes(&raw);
        
        let res = parser.next_packet();
        assert!(res.is_ok());
        assert!(res.unwrap().is_none());
        // Buffer remains intact waiting for more bytes
        assert_eq!(parser.buffer.len(), 16);
    }
}

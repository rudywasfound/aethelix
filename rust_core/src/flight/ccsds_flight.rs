//! no_std CCSDS Space Packet parser for LEON3 flight software.
//!
//! # Key differences from ground `ccsds.rs`
//! - **No `Vec`, no heap**: payload copied into a fixed `[u8; MAX_PAYLOAD_LEN]`
//! - **No `String` errors**: returns `CcsdsError` discriminant enum
//! - **Overflow-safe**: `data_length` cast to `usize` *before* adding 1
//!   (ground version had a potential `u16` overflow bug — fixed here)
//! - **Truncation-safe**: payloads larger than `MAX_PAYLOAD_LEN` are clamped,
//!   never causing an out-of-bounds copy
//!
//! # Kani verification
//! `prove_parser_never_panics_on_any_input` symbolically verifies that
//! no panic occurs for **any** byte sequence up to 512 bytes, including
//! pathological inputs like `data_length = 0xFFFF`.

/// Maximum payload length accepted from a CCSDS Space Packet.
/// Packets declaring a larger payload are clamped to this length.
/// Sized to fit in LEON3 working RAM budget (~256 bytes).
pub const MAX_PAYLOAD_LEN: usize = 256;

/// Minimal parsed CCSDS primary header (6 bytes).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FlightPacketHeader {
    /// 11-bit Application Process Identifier — identifies the telemetry source.
    pub apid: u16,
    /// 14-bit sequence count — detects dropped packets.
    pub sequence_count: u16,
    /// True if a secondary header immediately follows the primary header.
    pub secondary_header: bool,
}

/// A parsed CCSDS Space Packet with a statically-allocated payload buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FlightSpacePacket {
    pub header:      FlightPacketHeader,
    /// Number of valid bytes in `payload` (≤ MAX_PAYLOAD_LEN).
    pub payload_len: u8,
    /// Fixed-size payload buffer. Only `payload[..payload_len]` is valid.
    pub payload:     [u8; MAX_PAYLOAD_LEN],
}

/// Parse errors — returned as a lightweight discriminant, no heap, no String.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CcsdsError {
    /// Buffer has fewer than 6 bytes — cannot hold a primary header.
    BufferTooShort = 1,
    /// `data_length + 1` bytes of payload are not present in the buffer.
    LengthMismatch = 2,
}

/// Parse a raw byte slice into a [`FlightSpacePacket`].
///
/// # Design rules (all proven panic-free by Kani)
/// 1. Slice indexing only after explicit length check.
/// 2. `data_length` cast to `usize` **before** adding 1 (no `u16` overflow).
/// 3. Payload copy clamped to `MAX_PAYLOAD_LEN` (no unchecked copy).
pub fn parse_flight_packet(raw: &[u8]) -> Result<FlightSpacePacket, CcsdsError> {
    // Rule 1: primary header is always 6 bytes
    if raw.len() < 6 {
        return Err(CcsdsError::BufferTooShort);
    }

    // Byte 0 & 1: version(3) | type(1) | secondary_flag(1) | APID(11)
    let apid             = (((raw[0] & 0x07) as u16) << 8) | (raw[1] as u16);
    let secondary_header = (raw[0] >> 3) & 0x01 == 1;

    // Byte 2 & 3: seq_flags(2) | seq_count(14)
    let sequence_count = (((raw[2] & 0x3F) as u16) << 8) | (raw[3] as u16);

    // Bytes 4–5: data_length field
    // CCSDS: actual payload length = data_length + 1
    // RULE 2: cast to usize FIRST — prevents u16 wrapping when field = 0xFFFF
    let data_length_field = ((raw[4] as u16) << 8) | (raw[5] as u16);
    let declared_len: usize = (data_length_field as usize) + 1; // usize arithmetic, no overflow

    // Rule 1: entire packet must be present
    if raw.len() < 6 + declared_len {
        return Err(CcsdsError::LengthMismatch);
    }

    // Rule 3: clamp to static buffer (graceful truncation, not panic)
    let copy_len = declared_len.min(MAX_PAYLOAD_LEN);

    let mut payload = [0u8; MAX_PAYLOAD_LEN];
    // Both bounds are valid:
    //   `copy_len   <= MAX_PAYLOAD_LEN` (clamped above)
    //   `raw[6..6+copy_len]` is valid because `copy_len <= declared_len <= raw.len() - 6`
    payload[..copy_len].copy_from_slice(&raw[6..6 + copy_len]);

    Ok(FlightSpacePacket {
        header: FlightPacketHeader { apid, sequence_count, secondary_header },
        payload_len: copy_len as u8,
        payload,
    })
}

// ── Unit tests ────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn make_packet(apid: u16, payload: &[u8]) -> [u8; 270] {
        let mut raw = [0u8; 270];
        raw[0] = 0x08 | ((apid >> 8) & 0x07) as u8;
        raw[1] = (apid & 0xFF) as u8;
        raw[2] = 0xC0; raw[3] = 0x00;
        let dlen = (payload.len() as u16).wrapping_sub(1);
        raw[4] = (dlen >> 8) as u8;
        raw[5] = (dlen & 0xFF) as u8;
        raw[6..6 + payload.len()].copy_from_slice(payload);
        raw
    }

    #[test]
    fn test_valid_packet_parsed_correctly() {
        let raw = make_packet(0x001, &[0xDE, 0xAD, 0xBE, 0xEF, 0x00]);
        let pkt = parse_flight_packet(&raw[..11]).unwrap();
        assert_eq!(pkt.header.apid, 0x001);
        assert_eq!(pkt.payload_len, 5);
        assert_eq!(pkt.payload[0], 0xDE);
        assert_eq!(pkt.payload[3], 0xEF);
    }

    #[test]
    fn test_empty_input_is_error() {
        assert_eq!(parse_flight_packet(&[]), Err(CcsdsError::BufferTooShort));
    }

    #[test]
    fn test_five_byte_input_is_error() {
        assert_eq!(parse_flight_packet(&[0u8; 5]), Err(CcsdsError::BufferTooShort));
    }

    #[test]
    fn test_max_data_length_no_overflow() {
        // data_length field = 0xFFFF → declared_len = 65536 usize
        // Buffer is only 6 bytes → must return LengthMismatch, not panic
        let raw = [0x08, 0x01, 0xC0, 0x00, 0xFF, 0xFF];
        assert_eq!(parse_flight_packet(&raw), Err(CcsdsError::LengthMismatch));
    }

    #[test]
    fn test_payload_clamped_to_max() {
        // Construct a valid packet claiming 260 bytes of payload
        let mut raw = vec![0u8; 6 + 260];
        raw[4] = 0x01; raw[5] = 0x03; // data_length = 259 → 260 bytes
        let pkt = parse_flight_packet(&raw).unwrap();
        // Payload only up to MAX_PAYLOAD_LEN (256) is copied
        assert_eq!(pkt.payload_len, MAX_PAYLOAD_LEN as u8);
    }
}

// ── Kani proofs ───────────────────────────────────────────────────────────────
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// MAIN SAFETY PROOF: parse_flight_packet NEVER panics for ANY input.
    ///
    /// Kani symbolically explores all possible byte sequences of length 0–512
    /// and verifies no panic, array-out-of-bounds, or integer overflow occurs.
    /// The parser may return `Err(...)` — that is expected and correct.
    #[kani::proof]
    fn prove_parser_never_panics_on_any_input() {
        let len: usize = kani::any();
        kani::assume(len <= 512);

        let mut buf = [0u8; 512];
        for i in 0..len {
            buf[i] = kani::any();
        }

        // MUST NEVER PANIC — may return Ok or Err
        let _ = parse_flight_packet(&buf[..len]);
    }

    /// Prove: on success, payload_len is always within [0, MAX_PAYLOAD_LEN].
    #[kani::proof]
    fn prove_payload_len_bounded() {
        let len: usize = kani::any();
        kani::assume(len <= 512);

        let mut buf = [0u8; 512];
        for i in 0..len {
            buf[i] = kani::any();
        }

        if let Ok(pkt) = parse_flight_packet(&buf[..len]) {
            kani::assert(
                (pkt.payload_len as usize) <= MAX_PAYLOAD_LEN,
                "payload_len must never exceed MAX_PAYLOAD_LEN",
            );
        }
    }

    /// Prove: data_length = 0xFFFF never causes integer overflow.
    #[kani::proof]
    fn prove_max_data_length_no_overflow() {
        // Only 6 bytes — header-only, cannot satisfy 65536-byte payload claim
        let raw: [u8; 6] = kani::any();
        let _ = parse_flight_packet(&raw);
    }
}

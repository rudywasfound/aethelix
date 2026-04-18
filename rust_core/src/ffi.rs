//! C/Ada FFI entry point for the Aethelix flight diagnostic engine.
//!
//! Exposes a single `extern "C"` function:
//!   `aethelix_process_frame(ccsds_buf, buf_len, out_alert, state) -> i32`
//!
//! This is everything a LEON3 C or Ada FDIR middleware needs to call Aethelix.
//! The caller allocates `AethelixState` (size via `aethelix_state_size()`),
//! zeroes it before first use, and passes a pointer on every frame.
//!
//! # APID-to-channel mapping
//! The function decodes the telemetry channel from the CCSDS APID field.
//! APIDs 0x001–0x008 map to the 8 primary EPS/TCS channels. Unknown APIDs
//! are silently ignored (return 0 = success, no alert written).

use crate::flight::ccsds_flight::{parse_flight_packet, CcsdsError};
use crate::flight::fdir_output::{AethelixAlert, AethelixState, AlertLevel};
use crate::flight::ks_detector::process_channel;
use crate::flight::causal_ranker::{rank_root_causes, MAX_NODES};
use crate::flight::fixed_point::normalize_q15;

// APID → channel mapping 
// Maps well-known CCSDS APIDs to Aethelix channel indices 0–7.
// Configurable for each spacecraft by modifying this table (no recompile needed
// if APID table is stored in EEPROM — add a `set_apid_map()` call to configure).
//
// Channel assignments (must match aethelix.h AETHELIX_CH_* constants):
//   0: Solar input power    (APID 0x001)
//   1: Battery voltage      (APID 0x002)
//   2: Battery SoC %        (APID 0x003)
//   3: Bus voltage          (APID 0x004)
//   4: Battery temperature  (APID 0x005)
//   5: Solar panel temp     (APID 0x006)
//   6: Payload temperature  (APID 0x007)
//   7: Bus current          (APID 0x008)
const APID_CHANNEL_MAP: [(u16, usize); 8] = [
    (0x001, 0),
    (0x002, 1),
    (0x003, 2),
    (0x004, 3),
    (0x005, 4),
    (0x006, 5),
    (0x007, 6),
    (0x008, 7),
];

// Return codes (match aethelix.h)
const RET_OK:        i32 =  0;
const RET_TOO_SHORT: i32 =  1;
const RET_BAD_LEN:   i32 =  2;
const RET_NULL_PTR:  i32 = -1;

// Global Recovery Handler 
// Bare-metal safe global callback (assumes single-threaded cooperative OS or
// interrupt-masked access if needed). 
static mut RECOVERY_HANDLER: Option<extern "C" fn(i32)> = None;

// Main API

/// Process one CCSDS Space Packet through the Aethelix diagnostic engine.
///
/// # Safety
/// All three pointer arguments must be valid, aligned, non-null, and dereferenceable.
/// `state` must have been zeroed (via `aethelix_reset_state`) before the first call.
///
/// # Returns
/// - `0`  Success (alert written to `*out_alert`; alert.level == 0 means no fault)
/// - `1`  Packet buffer too short (< 6 bytes for CCSDS primary header)
/// - `2`  Payload length field mismatch (declared length exceeds buffer)
/// - `-1` Null pointer argument
#[no_mangle]
pub unsafe extern "C" fn aethelix_process_frame(
    ccsds_buf: *const u8,
    buf_len:   u16,
    out_alert: *mut AethelixAlert,
    state:     *mut AethelixState,
) -> i32 {
    // Null-pointer guard — never dereference null on LEON3 (causes trap)
    if ccsds_buf.is_null() || out_alert.is_null() || state.is_null() {
        return RET_NULL_PTR;
    }

    let buf       = core::slice::from_raw_parts(ccsds_buf, buf_len as usize);
    let alert_ref = &mut *out_alert;
    let state_ref = &mut *state;

    // Clear output alert at entry
    *alert_ref = AethelixAlert::NO_FAULT;

    // Parse CCSDS Packets
    let pkt = match parse_flight_packet(buf) {
        Ok(p)                          => p,
        Err(CcsdsError::BufferTooShort) => return RET_TOO_SHORT,
        Err(CcsdsError::LengthMismatch) => return RET_BAD_LEN,
    };

    // Map APID to telemetry channel
    let channel = match APID_CHANNEL_MAP.iter().find(|(apid, _)| *apid == pkt.header.apid) {
        Some((_, ch)) => *ch,
        None          => return RET_OK, // Unknown APID — silently skip
    };

    // Extract measurement (first i16 in payload, big-endian CCSDS)
    if (pkt.payload_len as usize) < 2 {
        return RET_OK; // No data to process
    }
    let raw_val = ((pkt.payload[0] as i16) << 8) | (pkt.payload[1] as i16);

    // Normalize to Q15 using pre-calibrated mean/range 
    let nom_mean  = state_ref.nom_means[channel] as i32;
    let nom_range = state_ref.nom_ranges[channel];
    let q15_val   = normalize_q15(raw_val as i32, nom_mean, nom_range);

    // Run KS anomaly detection
    let severity = process_channel(state_ref, channel, q15_val);
    state_ref.frame_count = state_ref.frame_count.wrapping_add(1);

    // No anomaly — return clean
    if severity.0 == 0 {
        return RET_OK;
    }

    // Anomaly detected — populate alert
    alert_ref.evidence_mask |= 1u32 << channel;

    // Build severity array for causal ranker (only this channel is anomalous)
    let mut sev_arr = [0i16; MAX_NODES];
    if channel < MAX_NODES {
        sev_arr[channel] = severity.0;
    }

    // Run causal ranking to identify root cause
    let hits = rank_root_causes(alert_ref.evidence_mask, &sev_arr);

    if hits[0].node_id != 0xFF {
        alert_ref.root_cause_id   = hits[0].node_id;
        alert_ref.confidence_q15  = hits[0].score_q15;
        alert_ref.root_cause_2_id = hits[1].node_id;

        // Trigger active recovery callback if registered
        if let Some(handler) = RECOVERY_HANDLER {
            handler(alert_ref.root_cause_id as i32);
        }
    }

    // Set alert level from Q15 severity thresholds
    alert_ref.level = if severity.0 > 0x6000 {
        AlertLevel::Critical as u8
    } else if severity.0 > 0x3000 {
        AlertLevel::Caution as u8
    } else {
        AlertLevel::Warning as u8
    };

    alert_ref.onset_frames_ago = state_ref.alarm_streak[channel] as u16;

    RET_OK
}

/// Reset the engine state to initial zeroed condition.
/// Call before first use or to restart diagnostics after a watchdog reset.
#[no_mangle]
pub unsafe extern "C" fn aethelix_reset_state(state: *mut AethelixState) {
    if !state.is_null() {
        *state = AethelixState::zeroed();
    }
}

/// Return `sizeof(AethelixState)` in bytes.
/// Use this to dynamically allocate the state block in Ada/C without
/// hard-coding the size (which may change between firmware versions).
#[no_mangle]
pub extern "C" fn aethelix_state_size() -> u32 {
    core::mem::size_of::<AethelixState>() as u32
}

/// Register a global recovery handler to be invoked dynamically when
/// a root cause is isolated.
#[no_mangle]
pub unsafe extern "C" fn register_recovery_handler(handler: extern "C" fn(i32)) {
    RECOVERY_HANDLER = Some(handler);
}

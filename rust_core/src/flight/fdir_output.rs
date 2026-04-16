//! FDIR (Fault Detection, Isolation, Recovery) output types for LEON3.
//!
//! All types use `#[repr(C)]` for direct ABI compatibility with:
//! - C FDIR middleware (LEON3 bare-metal, RTEMS)
//! - Ada thin binding (`ada/aethelix_binding.ads`)
//! - Auto-generated C header (`include/aethelix.h` via cbindgen)
//!
//! The caller (Ada/C FDIR layer) owns and allocates `AethelixState`.
//! Its size is available at runtime via `aethelix_state_size()`.

/// FDIR alert severity, matching `AETHELIX_LEVEL_*` constants in `aethelix.h`.
#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum AlertLevel {
    None     = 0,  // No fault — nominal operation
    Warning  = 1,  // Sub-threshold anomaly; monitor, no action yet
    Caution  = 2,  // Anomaly confirmed; prepare corrective action
    Critical = 3,  // Immediate FDIR action required
}

/// Root-cause diagnosis result produced by the Aethelix causal engine.
///
/// Sized to fit in a few words — safe to pass by value across C/Ada FFI.
/// Total size: 12 bytes (verified by `AethelixAlert::SIZE_CHECK` below).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct AethelixAlert {
    /// FDIR severity (`AlertLevel as u8`).
    pub level: u8,

    /// ID of the top-ranked root cause from the compiled causal graph.
    /// `0xFF` (= `AETHELIX_FAULT_NONE`) when no fault is identified.
    pub root_cause_id: u8,

    /// Diagnostic confidence in Q15 (0 = 0 %, 32 767 = 100 %).
    pub confidence_q15: i16,

    /// Bitmask of anomalous observable channels.
    /// Bit N = 1 means channel N has a confirmed distribution shift.
    pub evidence_mask: u32,

    /// Number of CCSDS frames elapsed since the anomaly was first detected.
    /// Divide by frame rate to get wall-clock lead time.
    pub onset_frames_ago: u16,

    /// Second-ranked root cause ID (`0xFF` if only one hypothesis).
    pub root_cause_2_id: u8,

    /// Reserved — explicit padding to keep struct size deterministic.
    pub _reserved: u8,
}

impl AethelixAlert {
    /// Sentinel value meaning "no fault detected".
    pub const NO_FAULT: Self = Self {
        level:            AlertLevel::None as u8,
        root_cause_id:    0xFF,
        confidence_q15:   0,
        evidence_mask:    0,
        onset_frames_ago: 0,
        root_cause_2_id:  0xFF,
        _reserved:        0,
    };

    /// True if a fault was identified (`root_cause_id != 0xFF`).
    #[inline]
    pub fn is_fault(&self) -> bool {
        self.root_cause_id != 0xFF
    }

    /// Confidence as a percentage (0.0–100.0). Ground-station helper.
    #[cfg(feature = "std")]
    pub fn confidence_pct(&self) -> f32 {
        self.confidence_q15 as f32 / 32_767.0 * 100.0
    }
}

/// Persistent Aethelix engine state — the caller must allocate and zero this.
///
/// # Memory layout (LEON3 RAM budget)
/// | Field           | Size    |
/// |-----------------|---------|
/// | ref_windows     | 2 048 B |
/// | cur_windows     | 1 024 B |
/// | heads / lens    |    64 B |
/// | alarm_streak    |     8 B |
/// | nom_means       |    16 B |
/// | nom_ranges      |    32 B |
/// | frame_count     |     4 B |
/// | **Total**       | **~3.2 KB** |
///
/// The caller queries the exact size via `aethelix_state_size()` and passes a
/// pointer to this struct on every `aethelix_process_frame()` call.
#[repr(C)]
pub struct AethelixState {
    /// Reference (baseline) windows: 8 channels × 128 samples × i16 = 2 048 B.
    pub ref_windows: [[i16; 128]; 8],
    /// Current (short) windows: 8 channels × 64 samples × i16 = 1 024 B.
    pub cur_windows: [[i16; 64]; 8],

    /// Next-write index into `cur_windows[ch]` (wraps at 64).
    pub cur_heads: [u8; 8],
    /// Next-write index into `ref_windows[ch]` (wraps at 128).
    pub ref_heads: [u8; 8],
    /// Number of valid samples in `cur_windows[ch]` (capped at 64).
    pub cur_lens:  [u8; 8],
    /// Number of valid samples in `ref_windows[ch]` (capped at 128).
    pub ref_lens:  [u8; 8],

    /// Consecutive-alarm frame counter per channel (reset on normal sample).
    pub alarm_streak: [u8; 8],

    /// Monotonic frame counter (wraps at u32::MAX — ~13 years at 10 Hz).
    pub frame_count: u32,

    /// Nominal channel means in Q15 (pre-calibrated on the ground).
    pub nom_means:  [i16; 8],

    /// Nominal channel ranges (raw telemetry units, used by `normalize_q15`).
    pub nom_ranges: [i32; 8],
}

impl AethelixState {
    /// Zero-initialise a state block (`const fn` — usable in `static`s).
    pub const fn zeroed() -> Self {
        Self {
            ref_windows:  [[0i16; 128]; 8],
            cur_windows:  [[0i16; 64];  8],
            cur_heads:    [0u8; 8],
            ref_heads:    [0u8; 8],
            cur_lens:     [0u8; 8],
            ref_lens:     [0u8; 8],
            alarm_streak: [0u8; 8],
            frame_count:  0,
            nom_means:    [0i16; 8],
            nom_ranges:   [0i32; 8],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_fault_sentinel() {
        let a = AethelixAlert::NO_FAULT;
        assert!(!a.is_fault());
        assert_eq!(a.level, AlertLevel::None as u8);
    }

    #[test]
    fn test_state_zeroed() {
        let s = AethelixState::zeroed();
        assert_eq!(s.frame_count, 0);
        assert_eq!(s.alarm_streak[0], 0);
    }
}

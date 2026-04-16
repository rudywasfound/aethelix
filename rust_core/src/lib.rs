//! Aethelix Core — Satellite Telemetry Causal Diagnostic Framework
//!
//! Two feature-gated build targets:
//!
//! **Ground station** (`--features ground`, default):
//!   Full Python/async/Streamlit stack for operators and benchmarking.
//!
//! **Flight (LEON3 OBC)** (`--features flight --no-default-features`):
//!   `no_std` on SPARC target, no heap, no async. Compiles to a `staticlib`
//!   linked by a LEON3 C/Ada FDIR middleware via `aethelix_process_frame()`.
//!   Proven panic-free via Kani model checker.
//!
//! # Build
//! Host (x86_64) — test flight modules with std:
//!   cargo test --features flight --no-default-features
//!
//! LEON3 (SPARC) — true bare-metal no_std build:
//!   cargo +nightly build --target sparc-unknown-none-elf \
//!       --features flight --no-default-features --release

// On SPARC bare-metal targets: no standard library.
// On x86_64 (dev/test): flight modules compile against std (core ⊆ std).
// This is the standard embedded Rust pattern — test on host, deploy on target.
#![cfg_attr(target_arch = "sparc", no_std)]

// Panic handler: only needed on SPARC bare-metal (test env provides its own).
#[cfg(all(target_arch = "sparc", not(test)))]
#[panic_handler]
fn flight_panic(_info: &core::panic::PanicInfo) -> ! {
    // On LEON3 bare-metal: trigger watchdog by spinning.
    // The FDIR watchdog will reset the OBC after its timeout.
    loop {
        // SPARC NOP in inline asm prevents the loop being optimised away
        unsafe { core::arch::asm!("nop", options(nomem, nostack)); }
    }
}
// ── Flight modules (always compiled when `flight` feature is active) ──────────
#[cfg(feature = "flight")]
pub mod flight;

// C/Ada FFI entry point (enabled when flight feature is active).
// Exposes `aethelix_process_frame()` as an `extern "C"` symbol.
#[cfg(feature = "flight")]
pub mod ffi;

// ── Ground modules (require std) ─────────────────────────────────────────────
#[cfg(feature = "std")]
pub mod error;

#[cfg(feature = "std")]
pub mod measurement;

#[cfg(feature = "std")]
pub mod kalman;

#[cfg(feature = "std")]
pub mod physics;

#[cfg(feature = "std")]
pub mod state_estimate;

#[cfg(feature = "std")]
pub mod dropout_handler;

#[cfg(feature = "std")]
pub mod graph_traversal;

#[cfg(feature = "std")]
pub mod ccsds;

#[cfg(feature = "std")]
pub mod kalman_filter;

#[cfg(feature = "std")]
pub mod hidden_state_inference;

// Re-exports for ground builds
#[cfg(feature = "std")]
pub use error::{Result, Error};

#[cfg(feature = "std")]
pub use measurement::{Measurement, MeasurementValidator};

#[cfg(feature = "std")]
pub use kalman::{KalmanFilter, ExtendedKalmanFilter};

#[cfg(feature = "std")]
pub use physics::PhysicsModel;

#[cfg(feature = "std")]
pub use state_estimate::StateEstimate;

#[cfg(feature = "std")]
pub use dropout_handler::DropoutHandler;

#[cfg(feature = "std")]
pub use graph_traversal::CausalGraphState;

#[cfg(feature = "std")]
pub use ccsds::{SpacePacket, CCSDSStreamParser};

// Python bindings (ground only)
#[cfg(all(feature = "python", feature = "std"))]
pub mod python_bindings;

/// Framework version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// ── Ground: process_measurement helper ───────────────────────────────────────
#[cfg(feature = "std")]
pub fn process_measurement(
    measurement: &measurement::Measurement,
    kalman: &mut kalman::KalmanFilter,
) -> error::Result<state_estimate::StateEstimate> {
    let validator = measurement::MeasurementValidator::default();
    validator.validate(measurement)?;
    kalman.update(measurement)?;
    Ok(kalman.get_estimate())
}

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}

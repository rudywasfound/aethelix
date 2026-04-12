//! Satellite Telemetry State Estimation Framework
//!
//! High-performance Rust framework for:
//! - Real-time telemetry stream processing
//! - Physics-based state estimation (Kalman Filter, EKF)
//! - Hidden state inference during communication dropouts
//! - Multi-language bindings (Python, C, JavaScript/WASM)

pub mod error;
pub mod measurement;
pub mod kalman;
pub mod physics;
pub mod state_estimate;
pub mod dropout_handler;
pub mod graph_traversal;
pub mod ccsds;

pub use error::{Result, Error};
pub use measurement::{Measurement, MeasurementValidator};
pub use kalman::{KalmanFilter, ExtendedKalmanFilter};
pub use physics::PhysicsModel;
pub use state_estimate::StateEstimate;
pub use dropout_handler::DropoutHandler;
pub use graph_traversal::CausalGraphState;
pub use ccsds::{SpacePacket, CCSDSStreamParser};

#[cfg(feature = "python")]
pub mod python_bindings;

/// Framework version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Process a single measurement and return state estimate
pub fn process_measurement(
    measurement: &Measurement,
    kalman: &mut KalmanFilter,
) -> Result<StateEstimate> {
    // Validate measurement
    let validator = MeasurementValidator::default();
    validator.validate(measurement)?;

    // Update Kalman filter
    kalman.update(measurement)?;

    // Generate state estimate
    Ok(kalman.get_estimate())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}

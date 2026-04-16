//! LEON3 flight-grade `no_std` diagnostic engine for Aethelix.
//!
//! All sub-modules run on bare-metal LEON3 (SPARC V8) with:
//!   - Zero heap allocation (no `alloc` crate)
//!   - Q15 fixed-point arithmetic (no FPU required)
//!   - Statically-allocated ring buffers (compile-time memory budget)
//!   - Kani-verified CCSDS parser (proven panic-free on any input)
//!   - Causal graph engine loaded from a compiled ~350-byte binary
//!
//! # Build
//! ```bash
//! cargo +nightly build \
//!     --target sparc-unknown-none-elf \
//!     --features flight \
//!     --no-default-features \
//!     --release
//! ```
//!
//! # Memory budget (LEON3, 256 KB RAM)
//! | Component         | Static RAM  |
//! |-------------------|-------------|
//! | Ring buffers      | ~2.5 KB     |
//! | Causal graph bin  | ~350 bytes  |
//! | KS scratch        | ~768 bytes  |
//! | AethelixState     | ~4 KB total |

pub mod fixed_point;
pub mod ring_buffer;
pub mod ccsds_flight;
pub mod fdir_output;
pub mod ks_detector;
pub mod causal_ranker;

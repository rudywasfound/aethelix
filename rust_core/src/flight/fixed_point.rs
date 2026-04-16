//! Q15 fixed-point arithmetic for FPU-less SPARC (LEON3) and RISC-V targets.
//!
//! # What is Q15?
//! Q15 is a fixed-point representation using a signed 16-bit integer (`i16`)
//! where the value `32767` represents `+1.0` and `-32768` represents `-1.0`.
//! All arithmetic uses saturating operations — no overflow, no panic.
//!
//! # Why not `f32`?
//! LEON3 variants without a GRFPU (e.g., LEON3FT in older satellites) have
//! no hardware floating-point unit. Software-emulated `f32` costs hundreds of
//! cycles per operation. Q15 uses integer ALU instructions — single-cycle on LEON3.
//!
//! # Kani verification
//! All arithmetic operations carry `#[kani::proof]` harnesses that
//! mathematically prove no panic occurs for any possible `i16` input.

use core::ops::{Add, Mul, Neg, Sub};

/// Q15 fixed-point scalar.
///
/// Internal representation: `i16` where `32767 ≈ +1.0`, `-32768 = -1.0`.
/// All arithmetic is saturating — no overflow panics possible.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
#[repr(transparent)]
pub struct Q15(pub i16);

impl Q15 {
    /// Represents `+1.0` (approximately — true value is 32767/32767).
    pub const ONE: Self = Self(i16::MAX);

    /// Represents `0.0`.
    pub const ZERO: Self = Self(0);

    /// Represents `-1.0`.
    pub const NEG_ONE: Self = Self(i16::MIN);

    /// Represents `+0.5`.
    pub const HALF: Self = Self(0x4000);

    /// Saturating Q15 multiply: `(a × b) >> 15`.
    ///
    /// Intermediate is computed in `i32` to avoid overflow, then shifted and
    /// saturated back to `i16`. This is the standard Q15 multiply algorithm.
    #[inline]
    pub fn mul_sat(self, rhs: Self) -> Self {
        let product = (self.0 as i32) * (rhs.0 as i32);
        let shifted = product >> 15;
        Self(shifted.clamp(i16::MIN as i32, i16::MAX as i32) as i16)
    }

    /// Absolute value with saturation (`i16::MIN.abs()` saturates to `i16::MAX`).
    #[inline]
    pub fn abs(self) -> Self {
        Self(self.0.saturating_abs())
    }

    /// True if this value is strictly positive.
    #[inline]
    pub fn is_positive(self) -> bool {
        self.0 > 0
    }

    /// Clamp to `[lo, hi]`.
    #[inline]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        if self < lo { lo } else if self > hi { hi } else { self }
    }
}

// ── Arithmetic traits (all saturating) ───────────────────────────────────────

impl Add for Q15 {
    type Output = Self;
    /// Saturating addition — wraps to `i16::MAX` / `i16::MIN`, never panics.
    #[inline]
    fn add(self, rhs: Self) -> Self { Self(self.0.saturating_add(rhs.0)) }
}

impl Sub for Q15 {
    type Output = Self;
    /// Saturating subtraction.
    #[inline]
    fn sub(self, rhs: Self) -> Self { Self(self.0.saturating_sub(rhs.0)) }
}

impl Neg for Q15 {
    type Output = Self;
    /// Saturating negation (`-i16::MIN` saturates to `i16::MAX`).
    #[inline]
    fn neg(self) -> Self { Self(self.0.saturating_neg()) }
}

impl Mul for Q15 {
    type Output = Self;
    /// Q15 saturating multiply.
    #[inline]
    fn mul(self, rhs: Self) -> Self { self.mul_sat(rhs) }
}

// ── Conversion helpers ────────────────────────────────────────────────────────

/// Normalize a raw telemetry integer value into Q15 range.
///
/// `raw`      — raw sensor reading (e.g. millivolts × 100)
/// `nom_mean` — nominal mean in the same units (pre-computed on the ground)
/// `range`    — full expected swing (max − min) in the same units
///
/// Returns `(raw − nom_mean) / (range / 2)` clamped to `[-1, +1]` in Q15.
/// If `range == 0`, returns `Q15::ZERO` (never panics on division by zero).
#[inline]
pub fn normalize_q15(raw: i32, nom_mean: i32, range: i32) -> Q15 {
    if range == 0 {
        return Q15::ZERO;
    }
    let deviation = raw.saturating_sub(nom_mean);
    let half_range = (range / 2).max(1); // guard against zero
    let scaled = (deviation.saturating_mul(32767)) / half_range;
    Q15(scaled.clamp(i16::MIN as i32, i16::MAX as i32) as i16)
}

// ── Unit tests ────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_saturates_at_max() {
        let a = Q15(i16::MAX);
        assert_eq!((a + Q15(1)).0, i16::MAX);
    }

    #[test]
    fn test_sub_saturates_at_min() {
        let a = Q15(i16::MIN);
        assert_eq!((a - Q15(1)).0, i16::MIN);
    }

    #[test]
    fn test_neg_of_min_saturates() {
        assert_eq!((-Q15(i16::MIN)).0, i16::MAX);
    }

    #[test]
    fn test_mul_half_by_half() {
        // 0.5 × 0.5 = 0.25 → Q15 = 0x2000 = 8192
        let half = Q15::HALF;
        let result = half * half;
        assert!((result.0 - 0x2000i16).abs() <= 1, "Got {}", result.0);
    }

    #[test]
    fn test_normalize_at_mean() {
        // Value exactly at nominal mean → 0
        assert_eq!(normalize_q15(100, 100, 40), Q15::ZERO);
    }

    #[test]
    fn test_normalize_at_max() {
        // Value at nominal mean + range/2 → Q15::ONE
        let q = normalize_q15(120, 100, 40); // deviation = 20, half_range = 20
        assert_eq!(q, Q15::ONE);
    }

    #[test]
    fn test_normalize_zero_range_safe() {
        // range == 0 must not panic
        let q = normalize_q15(1000, 500, 0);
        assert_eq!(q, Q15::ZERO);
    }
}

// ── Kani proofs — mathematically prove no panic for ANY i16 input ─────────────
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn prove_add_never_panics() {
        let a: i16 = kani::any();
        let b: i16 = kani::any();
        let _ = Q15(a) + Q15(b);
    }

    #[kani::proof]
    fn prove_sub_never_panics() {
        let a: i16 = kani::any();
        let b: i16 = kani::any();
        let _ = Q15(a) - Q15(b);
    }

    #[kani::proof]
    fn prove_mul_never_panics() {
        let a: i16 = kani::any();
        let b: i16 = kani::any();
        let _ = Q15(a) * Q15(b);
    }

    #[kani::proof]
    fn prove_neg_never_panics() {
        let a: i16 = kani::any();
        let _ = -Q15(a);
    }

    #[kani::proof]
    fn prove_normalize_never_panics() {
        let raw:  i32 = kani::any();
        let mean: i32 = kani::any();
        let rng:  i32 = kani::any();
        let _ = normalize_q15(raw, mean, rng);
    }

    #[kani::proof]
    fn prove_abs_never_panics() {
        let a: i16 = kani::any();
        let _ = Q15(a).abs();
    }
}

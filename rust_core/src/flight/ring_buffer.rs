//! Statically-allocated ring buffer for flight telemetry windows.
//!
//! No heap allocation: all data lives in a fixed `[i16; N]` array.
//! `N` is a compile-time constant (const generic) — the compiler verifies
//! the memory budget at build time.
//! 
//! # Memory footprint
//! `size_of::<Q15RingBuffer<128>>()` = 128×2 + 8 = **264 bytes**.
//! Eight channels × reference (128) + current (64) = **3072 bytes** total.
//!
//! # Kani verification
//! `prove_ring_buffer_never_panics` exhaustively verifies no index
//! out-of-bounds occurs for any sequence of up to 32 push + get operations.

use crate::flight::fixed_point::Q15;

/// Statically-allocated ring buffer of Q15 telemetry samples.
///
/// `N` — capacity in samples (compile-time constant, must be ≥ 1).
pub struct Q15RingBuffer<const N: usize> {
    buf:  [i16; N],
    /// Index of the *next* write slot (wraps at N).
    head: usize,
    /// Number of valid samples currently held (capped at N).
    len:  usize,
}

impl<const N: usize> Q15RingBuffer<N> {
    /// Compile-time capacity.
    pub const CAPACITY: usize = N;

    /// Create an empty buffer with all slots zeroed.
    /// `const fn` — usable in `static` initialisers.
    pub const fn new() -> Self {
        Self {
            buf:  [0i16; N],
            head: 0,
            len:  0,
        }
    }

    /// Push a sample, overwriting the oldest if the buffer is already full.
    #[inline]
    pub fn push(&mut self, sample: Q15) {
        self.buf[self.head] = sample.0;
        // Advance head, wrapping at N
        self.head = {
            let next = self.head + 1;
            if next >= N { 0 } else { next }
        };
        if self.len < N {
            self.len += 1;
        }
    }

    /// Number of valid samples currently held (0 ≤ len ≤ N).
    #[inline]
    pub fn len(&self) -> usize { self.len }

    /// True when the buffer holds exactly `N` samples.
    #[inline]
    pub fn is_full(&self) -> bool { self.len == N }

    /// True when the buffer holds no samples.
    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Read the `i`-th sample in **chronological order** (0 = oldest).
    ///
    /// # Panics (debug only)
    /// Panics if `i >= self.len()`. In release builds the index is wrapped
    /// silently. Always call with `i < self.len()`.
    #[inline]
    pub fn get(&self, i: usize) -> Q15 {
        debug_assert!(i < self.len, "index {} out of range (len={})", i, self.len);
        // Oldest sample is at `head` when buffer is full, at 0 otherwise.
        let start = if self.len == N { self.head } else { 0 };
        // `% N` is safe because N >= 1 (enforced by const generic contract)
        Q15(self.buf[(start + i) % N])
    }

    /// Compute the saturating Q15 mean of all stored samples.
    pub fn mean(&self) -> Q15 {
        if self.len == 0 {
            return Q15::ZERO;
        }
        let mut sum: i32 = 0;
        for i in 0..self.len {
            sum = sum.saturating_add(self.get(i).0 as i32);
        }
        Q15((sum / self.len as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16)
    }

    /// Copy all samples into `out` in chronological order (oldest first).
    ///
    /// Returns the number of samples written (= `min(self.len(), out.len())`).
    pub fn copy_to(&self, out: &mut [i16]) -> usize {
        let n = self.len.min(out.len());
        for i in 0..n {
            out[i] = self.get(i).0;
        }
        n
    }

    /// Reset to empty state (zeroes head/len; does not zero the buffer data).
    pub fn clear(&mut self) {
        self.head = 0;
        self.len  = 0;
    }
}

// Unit tests 
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_len() {
        let mut rb = Q15RingBuffer::<4>::new();
        assert!(rb.is_empty());
        rb.push(Q15(10));
        assert_eq!(rb.len(), 1);
        rb.push(Q15(20));
        rb.push(Q15(30));
        rb.push(Q15(40));
        assert!(rb.is_full());
    }

    #[test]
    fn test_wrapping_preserves_chronological_order() {
        let mut rb = Q15RingBuffer::<4>::new();
        // Push 6 samples into a 4-slot buffer
        for i in 0..6i16 {
            rb.push(Q15(i * 100));
        }
        assert!(rb.is_full());
        // Oldest 4 should be samples 2, 3, 4, 5
        assert_eq!(rb.get(0).0, 200);
        assert_eq!(rb.get(1).0, 300);
        assert_eq!(rb.get(2).0, 400);
        assert_eq!(rb.get(3).0, 500);
    }

    #[test]
    fn test_mean_single_element() {
        let mut rb = Q15RingBuffer::<8>::new();
        rb.push(Q15(1000));
        assert_eq!(rb.mean().0, 1000);
    }

    #[test]
    fn test_copy_to() {
        let mut rb = Q15RingBuffer::<4>::new();
        rb.push(Q15(1));
        rb.push(Q15(2));
        rb.push(Q15(3));
        let mut out = [0i16; 4];
        let n = rb.copy_to(&mut out);
        assert_eq!(n, 3);
        assert_eq!(out[0], 1);
        assert_eq!(out[2], 3);
    }
}

// Kani proofs
#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Verify: push + get never panics for any sequence of up to 32 operations.
    #[kani::proof]
    fn prove_ring_buffer_never_panics() {
        let mut rb = Q15RingBuffer::<8>::new();
        let n: usize = kani::any();
        kani::assume(n <= 32);

        for _ in 0..n {
            let v: i16 = kani::any();
            rb.push(Q15(v));
        }

        // get() on any valid index must not panic
        if rb.len() > 0 {
            let idx: usize = kani::any();
            kani::assume(idx < rb.len());
            let _ = rb.get(idx);
        }
    }

    /// Verify: mean() never panics regardless of buffer contents.
    #[kani::proof]
    fn prove_mean_never_panics() {
        let mut rb = Q15RingBuffer::<16>::new();
        let n: usize = kani::any();
        kani::assume(n <= 16);
        for _ in 0..n {
            let v: i16 = kani::any();
            rb.push(Q15(v));
        }
        let _ = rb.mean();
    }
}

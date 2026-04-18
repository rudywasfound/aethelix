//! LEON3 Benchmark Binary — cycle-count profiler for Aethelix flight core.
//!
//! This `no_std` binary runs on LEON3 (SPARC V8) under TSIM3 or QEMU SPARC.
//! It exercises `aethelix_process_frame()` with synthetic CCSDS frames and
//! measures the LEON3 hardware cycle counter per call.
//!
//! # Output (via APBUART UART, visible in TSIM3/QEMU console)
//! ```
//! ==============================================
//!  AETHELIX LEON3 CYCLE-COUNT BENCHMARK
//!  Core: SPARC V8 (LEON3FT)  Freq: 50 MHz
//! ==============================================
//!  Iterations: 1000
//!  Total cycles:      3_240_000
//!  Cycles per frame:  3_240
//!  Estimated latency: 64.8 µs @ 50 MHz
//! ==============================================
//! ```
//!
//! # Build & run
//! ```bash
//! cargo +nightly build \
//!     --target sparc-unknown-none-elf \
//!     --features flight \
//!     --no-default-features \
//!     --bin leon3_bench \
//!     --release
//!
//! # QEMU
//! qemu-system-sparc -M leon3_generic -nographic \
//!     -kernel target/sparc-unknown-none-elf/release/leon3_bench
//!
//! # TSIM3 (Gaisler simulator)
//! tsim-leon3 target/sparc-unknown-none-elf/release/leon3_bench
//! ```

#![no_std]
#![no_main]

use core::panic::PanicInfo;

// Pull in flight modules
use aethelix_core::flight::fdir_output::{AethelixAlert, AethelixState};
use aethelix_core::ffi::{aethelix_process_frame, aethelix_reset_state};

// ── LEON3 APBUART (UART) base address ─────────────────────────────────────
// LEON3 generic SoC: APBUART at 0x8000_0100
const UART_DATA: *mut u32 = 0x8000_0100 as *mut u32;
const UART_STAT: *const u32 = 0x8000_0104 as *const u32;

/// Write one character to APBUART (spin-waits for TX ready).
unsafe fn uart_putchar(c: u8) {
    // Wait until transmit FIFO not full (bit 2 = TX full, wait until clear)
    while (core::ptr::read_volatile(UART_STAT) & (1 << 2)) != 0 {}
    core::ptr::write_volatile(UART_DATA, c as u32);
}

/// Write a string literal to APBUART.
unsafe fn uart_puts(s: &str) {
    for b in s.bytes() {
        uart_putchar(b);
    }
}

/// Write a u32 as decimal to UART.
unsafe fn uart_u32(mut n: u32) {
    if n == 0 {
        uart_putchar(b'0');
        return;
    }
    let mut buf = [0u8; 10];
    let mut i   = 10usize;
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    for &b in &buf[i..] {
        uart_putchar(b);
    }
}

// LEON3 hardware cycle counter (ASR16)
// LEON3 exposes an internal performance counter in ASR16 (via rdsr instruction).
// This is SPARC V8 ASR access — LEON3 specific.

#[cfg(target_arch = "sparc")]
unsafe fn read_cycle_counter() -> u32 {
    let val: u32;
    core::arch::asm!(
        "rd %asr16, {0}",
        out(reg) val,
        options(nomem, nostack, preserves_flags),
    );
    val
}

// On non-SPARC hosts (unit tests / CI), return a dummy counter.
#[cfg(not(target_arch = "sparc"))]
unsafe fn read_cycle_counter() -> u32 {
    0u32
}

// Synthetic CCSDS test frame
/// Build a minimal 8-byte CCSDS Space Packet for a given APID and i16 value.
fn make_test_frame(apid: u16, value: i16) -> [u8; 8] {
    [
        (0x08 | ((apid >> 8) & 0x07)) as u8,  // version + type + APID high
        (apid & 0xFF) as u8,                   // APID low
        0xC0, 0x00,                            // seq flags=11, count=0
        0x00, 0x01,                            // data_length = 1 → 2 bytes payload
        (value >> 8) as u8,                    // payload high byte
        (value & 0xFF) as u8,                  // payload low byte
    ]
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    const ITERS: u32 = 1000;
    const LEON3_FREQ_MHZ: u32 = 50;

    let mut state = AethelixState::zeroed();
    let mut alert = AethelixAlert::NO_FAULT;

    // Pre-compute synthetic frames (8 channels cycling through injected faults)
    let frames: [[u8; 8]; 8] = [
        make_test_frame(0x001, 16_000),  // solar_input — high value
        make_test_frame(0x002, -8_000),  // battery_voltage — depressed
        make_test_frame(0x003,  5_000),  // battery_soc
        make_test_frame(0x004, -3_000),  // bus_voltage — depressed
        make_test_frame(0x005, 12_000),  // battery_temp — elevated
        make_test_frame(0x006,  9_000),  // panel_temp
        make_test_frame(0x007,  7_000),  // payload_temp
        make_test_frame(0x008, -2_000),  // bus_current
    ];

    // Warm-up the caches and branch predictor
    for _ in 0..50 {
        for f in &frames {
            unsafe {
                aethelix_process_frame(f.as_ptr(), f.len() as u16, &mut alert, &mut state);
            }
        }
    }
    unsafe { aethelix_reset_state(&mut state); }

    // Timed benchmark 
    let t_start = unsafe { read_cycle_counter() };

    for _ in 0..ITERS {
        for f in &frames {
            unsafe {
                aethelix_process_frame(f.as_ptr(), f.len() as u16, &mut alert, &mut state);
            }
        }
    }

    let t_end   = unsafe { read_cycle_counter() };
    let total   = t_end.wrapping_sub(t_start);
    let per_frm = total / (ITERS * 8);
    // latency_ns = (per_frm * 1000) / LEON3_FREQ_MHZ  (avoids division by freq first)
    let lat_ns  = (per_frm * 1000) / LEON3_FREQ_MHZ;

    // UART output
    unsafe {
        uart_puts("\r\n===============================================\r\n");
        uart_puts(" AETHELIX LEON3 CYCLE-COUNT BENCHMARK\r\n");
        uart_puts(" Core: SPARC V8 (LEON3)   Freq: ");
        uart_u32(LEON3_FREQ_MHZ);
        uart_puts(" MHz\r\n");
        uart_puts("===============================================\r\n");
        uart_puts(" Iterations:       ");
        uart_u32(ITERS);
        uart_puts("\r\n");
        uart_puts(" Channels/frame:   8\r\n");
        uart_puts(" Total cycles:     ");
        uart_u32(total);
        uart_puts("\r\n");
        uart_puts(" Cycles/frame:     ");
        uart_u32(per_frm);
        uart_puts("\r\n");
        uart_puts(" Latency (ns):     ");
        uart_u32(lat_ns);
        uart_puts(" ns  @ 50 MHz\r\n");
        uart_puts("===============================================\r\n");
        uart_puts(" Fault detected:   ");
        uart_puts(if alert.is_fault() { "YES" } else { "NO (nominal)" });
        uart_puts("\r\n");
        uart_puts("===============================================\r\n");
        uart_puts(" BENCHMARK COMPLETE\r\n\r\n");
    }

    // Halt (LEON3 bare-metal: loop forever after output)
    loop {
        unsafe {
            // LEON3: write 0 to stop address 0x8000_0000 in simulation
            // Real hardware: spin with NOP
            core::arch::asm!("nop", options(nomem, nostack));
        }
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    unsafe {
        uart_puts("\r\n[PANIC] ");
        // In flight code, trigger watchdog reset here
    }
    loop {}
}

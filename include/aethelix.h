/**
 * aethelix.h — C / Ada interface for the Aethelix flight diagnostic engine.
 *
 * Compatible with: LEON3 (SPARC V8), RTEMS, VxWorks, bare-metal C/Ada FDIR.
 * Generated from Rust source (rust_core/src/ffi.rs) via cbindgen.
 *
 * Quick start (C) 
 *
 *   #include "aethelix.h"
 *   #include <string.h>
 *
 *   // Allocate persistent state (typically in .bss or EEPROM-backed RAM)
 *   static uint8_t aethelix_mem[4096];   // must be >= aethelix_state_size()
 *   AethelixState *state = (AethelixState*) aethelix_mem;
 *   memset(state, 0, aethelix_state_size());
 *
 *   // In the FDIR telemetry loop:
 *   AethelixAlert alert;
 *   int32_t rc = aethelix_process_frame(ccsds_buf, buf_len, &alert, state);
 *   if (rc == 0 && alert.level >= AETHELIX_LEVEL_CAUTION) {
 *       handle_fault(&alert);
 *   }
 *
 * Quick start (Ada) 
 *   See ada/aethelix_binding.ads for a type-safe Ada 2012 thin binding.
 *
 * ECSS references 
 *   ECSS-E-ST-10-03C  CCSDS Space Packet Protocol
 *   ECSS-E-ST-40C     Software Engineering (req. basis for formal verification)
 *   ECSS-Q-ST-80C     Software Product Assurance
 */

#ifndef AETHELIX_H
#define AETHELIX_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* Version*/
#define AETHELIX_VERSION_MAJOR  0U
#define AETHELIX_VERSION_MINOR  2U
#define AETHELIX_VERSION_PATCH  0U

/* Alert level severity */
#define AETHELIX_LEVEL_NONE      0U  /**< Nominal — no fault detected          */
#define AETHELIX_LEVEL_WARNING   1U  /**< Sub-threshold anomaly; monitor       */
#define AETHELIX_LEVEL_CAUTION   2U  /**< Anomaly confirmed; prepare action    */
#define AETHELIX_LEVEL_CRITICAL  3U  /**< Immediate FDIR action required       */

/* Telemetry channel indices (bit positions in evidence_mask) */
#define AETHELIX_CH_SOLAR_INPUT   0U  /**< Solar input power (W)               */
#define AETHELIX_CH_BATTERY_VOLT  1U  /**< Battery voltage (mV)                */
#define AETHELIX_CH_BATTERY_SOC   2U  /**< Battery state-of-charge (%)         */
#define AETHELIX_CH_BUS_VOLT      3U  /**< Regulated bus voltage (mV)          */
#define AETHELIX_CH_BATT_TEMP     4U  /**< Battery temperature (0.01°C)        */
#define AETHELIX_CH_PANEL_TEMP    5U  /**< Solar panel temperature (0.01°C)    */
#define AETHELIX_CH_PAYLOAD_TEMP  6U  /**< Payload temperature (0.01°C)        */
#define AETHELIX_CH_BUS_CURRENT   7U  /**< Bus current (mA)                    */

/* CCSDS APID assignments (configure to match your spacecraft) */
#define AETHELIX_APID_SOLAR_INPUT   0x001U
#define AETHELIX_APID_BATTERY_VOLT  0x002U
#define AETHELIX_APID_BATTERY_SOC   0x003U
#define AETHELIX_APID_BUS_VOLT      0x004U
#define AETHELIX_APID_BATT_TEMP     0x005U
#define AETHELIX_APID_PANEL_TEMP    0x006U
#define AETHELIX_APID_PAYLOAD_TEMP  0x007U
#define AETHELIX_APID_BUS_CURRENT   0x008U

/* ── Root-cause fault IDs (auto-generated from causal_graph.bin) */
/* Include aethelix_graph_ids.h for the full enumeration.                       */
#include "aethelix_graph_ids.h"

/* Return codes  */
#define AETHELIX_OK            0    /**< Success                               */
#define AETHELIX_ERR_TOO_SHORT 1    /**< Buffer < 6 bytes (no CCSDS header)   */
#define AETHELIX_ERR_BAD_LEN   2    /**< Payload length mismatch in header     */
#define AETHELIX_ERR_NULL_PTR  (-1) /**< Null pointer argument                 */

/* FDIR Alert (12 bytes, packed, C ABI compatible) */
typedef struct __attribute__((packed)) {
    uint8_t  level;             /**< AETHELIX_LEVEL_* severity                */
    uint8_t  root_cause_id;     /**< AETHELIX_FAULT_* or 0xFF = no fault      */
    int16_t  confidence_q15;    /**< Diagnostic confidence [0, 32767]         */
    uint32_t evidence_mask;     /**< Bit N = 1: channel N anomalous           */
    uint16_t onset_frames_ago;  /**< Frames since anomaly first detected      */
    uint8_t  root_cause_2_id;   /**< Second candidate, 0xFF if none           */
    uint8_t  _reserved;         /**< Padding — must be zero                   */
} AethelixAlert;

/* Compile-time size assertion (14 bytes with __packed__) */
typedef char _aethelix_alert_size_check[sizeof(AethelixAlert) == 12 ? 1 : -1];

/* Opaque persistent state */
/* Caller must allocate >= aethelix_state_size() bytes and zero the buffer    */
/* before the first call to aethelix_process_frame().                          */
typedef struct AethelixState AethelixState;

/* API */

/**
 * Process one CCSDS Space Packet through the Aethelix diagnostic engine.
 *
 * Thread safety: NOT thread-safe. The caller must ensure mutual exclusion
 * if multiple tasks share the same state.
 *
 * @param ccsds_buf  Pointer to raw CCSDS Space Packet bytes. Must not be NULL.
 * @param buf_len    Number of valid bytes in ccsds_buf.
 * @param out_alert  Caller-allocated alert struct (cleared on entry). Not NULL.
 * @param state      Persistent engine state (zero-init before first call). Not NULL.
 * @return           AETHELIX_OK or one of the AETHELIX_ERR_* codes.
 */
int32_t aethelix_process_frame(
    const uint8_t  *ccsds_buf,
    uint16_t        buf_len,
    AethelixAlert  *out_alert,
    AethelixState  *state
);

/**
 * Reset state to initial zeroed condition.
 * Call after a watchdog reset or before starting a new diagnostic session.
 *
 * @param state  Pointer to the state block to zero. Ignored if NULL.
 */
void aethelix_reset_state(AethelixState *state);

/**
 * Return sizeof(AethelixState) — use to allocate the correct amount of RAM.
 *
 * @return Exact byte size of AethelixState in this firmware build.
 */
uint32_t aethelix_state_size(void);

/**
 * Register an active recovery handler callback for the FDIR framework.
 * This instructs Aethelix to actively call the function pointer supplied
 * the moment a root cause fault is isolated.
 *
 * @param handler Function pointer taking an integer fault_id to execute recovery
 */
void register_recovery_handler(void (*handler)(int fault_id));

#ifdef __cplusplus
}
#endif

#endif /* AETHELIX_H */

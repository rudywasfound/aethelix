#!/usr/bin/env bash
# =============================================================================
# leon3_bench.sh — Cross-compile and benchmark Aethelix on LEON3 (SPARC V8)
#
# Steps performed:
#   1. Verify prerequisites (nightly Rust, SPARC target, cargo-binutils)
#   2. Cross-compile the Aethelix flight library for sparc-unknown-none-elf
#   3. Report binary section sizes (.text, .rodata, .bss, .data)
#   4. Optionally run under QEMU SPARC (leon3_generic machine) or TSIM3
#
# Prerequisites:
#   rustup toolchain install nightly
#   rustup +nightly target add sparc-unknown-none-elf
#   cargo install cargo-binutils
#   rustup +nightly component add llvm-tools-preview
#
# Optional (for cycle-count emulation):
#   sudo apt install qemu-system-sparc    # QEMU SPARC/LEON3
#   # or: Frontgrade TSIM3 from https://www.gaisler.com
#
# Usage:
#   bash scripts/leon3_bench.sh            # Build + size report
#   bash scripts/leon3_bench.sh --qemu     # Build + QEMU emulation
#   bash scripts/leon3_bench.sh --tsim3    # Build + TSIM3 emulation
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUST_DIR="$REPO_ROOT/rust_core"
TARGET="sparc-unknown-none-elf"
PROFILE="release"
FEATURES="flight"

# Colour helpers
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[!]${NC}  $*"; }
error() { echo -e "${RED}[✗]${NC}  $*" >&2; }
step()  { echo -e "\n${CYAN}${BOLD}── $* ${NC}"; }

RUN_QEMU=0
RUN_TSIM=0
for arg in "$@"; do
  case "$arg" in
    --qemu)  RUN_QEMU=1 ;;
    --tsim3) RUN_TSIM=1 ;;
  esac
done

# =============================================================================
# Step 1: Prerequisites
# =============================================================================
step "Checking prerequisites"

if ! command -v rustup &>/dev/null; then
  error "rustup not found. Install from https://rustup.rs"
  exit 1
fi; info "rustup found"

if ! rustup toolchain list | grep -q "nightly"; then
  warn "Nightly toolchain not found — installing..."
  rustup toolchain install nightly --profile minimal
fi; info "nightly toolchain OK"

if ! rustup +nightly target list --installed | grep -q "$TARGET"; then
  warn "Target $TARGET not installed — adding..."
  rustup +nightly target add "$TARGET"
fi; info "Target $TARGET ready"

HAS_BINUTILS=0
if cargo +nightly install --list 2>/dev/null | grep -q "cargo-binutils"; then
  HAS_BINUTILS=1; info "cargo-binutils available"
else
  warn "cargo-binutils not found (install: cargo install cargo-binutils)"
  warn "Binary size report will use 'wc' as fallback"
fi

# Also regenerate the causal graph binary if graph_compiler.py is newer
if command -v python3 &>/dev/null; then
  BIN_FILE="$REPO_ROOT/causal_graph/causal_graph.bin"
  GD_FILE="$REPO_ROOT/causal_graph/graph_definition.py"
  if [ ! -f "$BIN_FILE" ] || [ "$GD_FILE" -nt "$BIN_FILE" ]; then
    warn "causal_graph.bin is stale — regenerating..."
    cd "$REPO_ROOT"
    if [ -f "venv/bin/activate" ]; then
      source venv/bin/activate
    fi
    python3 causal_graph/graph_compiler.py
    info "causal_graph.bin regenerated"
  else
    info "causal_graph.bin is up to date ($(wc -c < "$BIN_FILE") bytes)"
  fi
fi

# =============================================================================
# Step 2: Cross-compile for LEON3
# =============================================================================
step "Cross-compiling for $TARGET (features: $FEATURES, profile: $PROFILE)"

cd "$RUST_DIR"

# Use -Zbuild-std=core to rebuild core with panic=abort (required for no_std on SPARC)
# This avoids the "unwinding panics not supported without std" linker error.
cargo +nightly build \
  --target "$TARGET" \
  --features "$FEATURES" \
  --no-default-features \
  --"$PROFILE" \
  --lib \
  -Zbuild-std=core,compiler_builtins \
  -Zbuild-std-features=compiler-builtins-mem \
  2>&1 | grep -v "^$" | tail -20

LIB_A="$RUST_DIR/target/$TARGET/$PROFILE/libaethelix_core.a"
if [ -f "$LIB_A" ]; then
  info "Static library built: $LIB_A"
  LIB_SIZE=$(wc -c < "$LIB_A")
  info "Library size: ${LIB_SIZE} bytes ($(( LIB_SIZE / 1024 )) KB)"
else
  error "Static library not found at $LIB_A"
  exit 1
fi

# =============================================================================
# Step 3: Binary size report
# =============================================================================
step "Binary section size report"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         AETHELIX LEON3 FLIGHT BINARY SIZE REPORT            ║"
echo "║         Target: sparc-unknown-none-elf (LEON3FT)            ║"
echo "╠══════════════════════════════════════════════════════════════╣"

if [ "$HAS_BINUTILS" -eq 1 ]; then
  # Use rust-size for a proper ELF section breakdown
  cargo +nightly size \
    --target "$TARGET" \
    --features "$FEATURES" \
    --no-default-features \
    --"$PROFILE" \
    --lib \
    -Zbuild-std=core,compiler_builtins \
    -- -A 2>/dev/null || \
  rust-size "$LIB_A" -A 2>/dev/null || \
  echo "  (size tool unavailable — use: cargo install cargo-binutils)"
else
  # Fallback: use nm to estimate .text size
  if command -v sparc-linux-gnu-nm &>/dev/null; then
    sparc-linux-gnu-nm --print-size --size-sort "$LIB_A" 2>/dev/null | tail -20
  elif command -v nm &>/dev/null; then
    nm --print-size --size-sort "$LIB_A" 2>/dev/null | grep " [tT] " | tail -20
  fi
fi

BUDGET_FLASH_KB=64
BUDGET_RAM_KB=8

echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Flash budget: ≤${BUDGET_FLASH_KB} KB (.text + .rodata)                  ║"
echo "║  RAM budget:   ≤${BUDGET_RAM_KB} KB (.data + .bss + stack)               ║"
echo "║  AethelixState: ~3.2 KB (static, caller-allocated)          ║"
echo "║  causal_graph.bin: 284 bytes (in .rodata = flash)           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
# Step 4: QEMU SPARC emulation (optional)
# =============================================================================
if [ "$RUN_QEMU" -eq 1 ]; then
  step "Running under QEMU SPARC (leon3_generic machine)"

  BENCH_BIN="$RUST_DIR/target/$TARGET/$PROFILE/leon3_bench"
  if [ ! -f "$BENCH_BIN" ]; then
    warn "Building leon3_bench binary..."
    cargo +nightly build \
      --target "$TARGET" \
      --features "$FEATURES" \
      --no-default-features \
      --"$PROFILE" \
      --bin leon3_bench \
      -Zbuild-std=core,compiler_builtins \
      -Zbuild-std-features=compiler-builtins-mem \
      2>&1 | tail -5
  fi

  if [ -f "$BENCH_BIN" ]; then
    info "Running: qemu-system-sparc -M leon3_generic -kernel $BENCH_BIN"
    timeout 10 qemu-system-sparc \
      -M leon3_generic \
      -nographic \
      -no-reboot \
      -kernel "$BENCH_BIN" \
      2>&1 | head -40 || true
  else
    error "leon3_bench binary not found — cannot run QEMU benchmark"
  fi
fi

# =============================================================================
# Step 5: TSIM3 emulation (optional)
# =============================================================================
if [ "$RUN_TSIM" -eq 1 ]; then
  step "Running under TSIM3 (Gaisler LEON3 simulator)"
  if ! command -v tsim-leon3 &>/dev/null; then
    error "tsim-leon3 not found. Download from https://www.gaisler.com/index.php/downloads/simulators"
    exit 1
  fi

  BENCH_BIN="$RUST_DIR/target/$TARGET/$PROFILE/leon3_bench"
  if [ -f "$BENCH_BIN" ]; then
    echo -e "load $BENCH_BIN\nrun\nperf\nquit" | tsim-leon3 -uart "$BENCH_BIN" 2>&1 | head -50
  fi
fi

# =============================================================================
# Summary
# =============================================================================
step "Benchmark complete"
echo ""
echo "  Cross-compilation: ✓  (sparc-unknown-none-elf, ${PROFILE})"
echo "  Library size:      ${LIB_SIZE} bytes"
echo ""
echo "  Next steps:"
echo "    • Install TSIM3 from Frontgrade Gaisler for cycle-count profiling"
echo "    • Run:  bash scripts/leon3_bench.sh --tsim3"
echo "    • Run:  bash scripts/leon3_bench.sh --qemu  (requires qemu-system-sparc)"
echo "    • Install cargo-binutils for detailed section breakdown:"
echo "        cargo install cargo-binutils"
echo "        rustup +nightly component add llvm-tools-preview"
echo ""

#!/usr/bin/env bash
# ============================================================================
# download_esa_datasets.sh — Fetch ESA anomaly detection datasets from Zenodo.
#
# Usage:
#   ./scripts/download_esa_datasets.sh              # download OPS-SAT only
#   ./scripts/download_esa_datasets.sh --all         # download OPS-SAT + ESA-ADB
#   ./scripts/download_esa_datasets.sh --esa-adb     # download ESA-ADB only
#
# Datasets:
#   OPSSAT-AD  — https://zenodo.org/records/10850228  (~50 MB)
#   ESA-ADB    — https://zenodo.org/records/12528696  (~31 GB, partial download)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OPSSAT_DIR="$REPO_ROOT/data/esa/opssat"
ESAADB_DIR="$REPO_ROOT/data/esa/esa_adb"

# Zenodo file download URLs
# OPSSAT-AD (Zenodo record 15108715)
OPSSAT_SEGMENTS_URL="https://zenodo.org/api/records/15108715/files/segments.csv/content"
OPSSAT_DATASET_URL="https://zenodo.org/api/records/15108715/files/dataset.csv/content"

# ESA-ADB (Zenodo record 12528696) — we download only the labels and one mission
ESAADB_BASE_URL="https://zenodo.org/records/12528696/files"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
fail()  { echo -e "\033[1;31m[FAIL]\033[0m  $*" >&2; exit 1; }

download_file() {
    local url="$1"
    local dest="$2"
    local name
    name="$(basename "$dest")"

    if [[ -f "$dest" ]]; then
        ok "$name already exists — skipping."
        return 0
    fi

    info "Downloading $name ..."
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$dest" "$url"
    else
        fail "Neither wget nor curl found. Install one and retry."
    fi
    ok "$name downloaded ($(du -h "$dest" | cut -f1))."
}

# ---------------------------------------------------------------------------
# OPSSAT-AD download
# ---------------------------------------------------------------------------
download_opssat() {
    info "=== OPSSAT-AD Dataset ==="
    mkdir -p "$OPSSAT_DIR"

    download_file "$OPSSAT_SEGMENTS_URL" "$OPSSAT_DIR/segments.csv"
    download_file "$OPSSAT_DATASET_URL"  "$OPSSAT_DIR/dataset.csv"

    # Quick sanity check
    if [[ -f "$OPSSAT_DIR/segments.csv" ]]; then
        local lines
        lines=$(wc -l < "$OPSSAT_DIR/segments.csv")
        ok "segments.csv: $lines lines"
    fi

    ok "OPSSAT-AD download complete → $OPSSAT_DIR/"
}

# ---------------------------------------------------------------------------
# ESA-ADB download (partial — labels + 1 mission)
# ---------------------------------------------------------------------------
download_esaadb() {
    info "=== ESA-ADB Dataset (partial) ==="
    warn "Full ESA-ADB is ~31 GB. Downloading labels and metadata only."
    warn "Use the ESA-ADB GitHub repo preprocessing scripts for full data."
    mkdir -p "$ESAADB_DIR"

    # The ESA-ADB Zenodo record packages data as a single large zip.
    # We download just the metadata / labels file if available separately,
    # otherwise instruct the user to use the GitHub repo.
    info "ESA-ADB data must be downloaded manually from:"
    info "  https://zenodo.org/records/12528696"
    info ""
    info "After downloading, extract to: $ESAADB_DIR/"
    info "Then run preprocessing with the ESA-ADB GitHub repo:"
    info "  https://github.com/kplabs-pl/ESA-ADB"
    info ""
    info "Aethelix will look for preprocessed data in: $ESAADB_DIR/"

    # Create a placeholder README
    cat > "$ESAADB_DIR/README.md" << 'EOF'
# ESA-ADB Dataset

Download the ESA Anomaly Detection Benchmark from Zenodo:
  https://zenodo.org/records/12528696

After downloading:
1. Extract the archive into this directory
2. Run preprocessing using the ESA-ADB GitHub tools:
   https://github.com/kplabs-pl/ESA-ADB
3. Then run: python scripts/esa_benchmark.py --dataset esa-adb
EOF

    ok "ESA-ADB placeholder created → $ESAADB_DIR/"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    local do_opssat=true
    local do_esaadb=false

    for arg in "$@"; do
        case "$arg" in
            --all)     do_opssat=true; do_esaadb=true ;;
            --esa-adb) do_opssat=false; do_esaadb=true ;;
            --opssat)  do_opssat=true; do_esaadb=false ;;
            -h|--help)
                echo "Usage: $0 [--opssat|--esa-adb|--all]"
                echo "  (default: --opssat)"
                exit 0
                ;;
            *)
                warn "Unknown argument: $arg"
                ;;
        esac
    done

    info "Aethelix ESA Dataset Downloader"
    echo ""

    if $do_opssat; then
        download_opssat
        echo ""
    fi

    if $do_esaadb; then
        download_esaadb
        echo ""
    fi

    ok "All done."
}

main "$@"

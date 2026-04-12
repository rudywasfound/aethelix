#!/usr/bin/env python3
"""
Build Aethelix documentation PDF

Generates a comprehensive PDF from all documentation files in docs/
excluding BUILD_PDF.md itself.
"""

import subprocess
import sys
from pathlib import Path


def build_pdf():
    """Build the documentation PDF."""
    docs_dir = Path("docs")

    # List of documents in order (excluding BUILD_PDF.md)
    documents = [
        "00_TABLE_OF_CONTENTS.md",
        "01_INTRODUCTION.md",
        "02_INSTALLATION.md",
        "03_QUICKSTART.md",
        "04_RUNNING_FRAMEWORK.md",
        "05_CONFIGURATION.md",
        "06_OUTPUT_INTERPRETATION.md",
        "07_REAL_EXAMPLES.md",
        "08_PHYSICS_FOUNDATION.md",
        "10_API_REFERENCE.md",
        "23_FAQ.md",
    ]

    # Verify all files exist
    doc_paths = []
    for doc in documents:
        path = docs_dir / doc
        if not path.exists():
            print(f"Missing: {path}")
            continue
        doc_paths.append(str(path))

    if not doc_paths:
        print("ERROR: No documentation files found in docs/")
        return False

    print(f"Found {len(doc_paths)} documentation files")
    print("Building PDF with:")
    for path in doc_paths:
        print(f"   ✓ {Path(path).name}")

    # Build PDF
    output_file = "aethelix_documentation.pdf"
    cmd = [
        "pandoc",
        *doc_paths,
        "-o",
        output_file,
        "--toc",
        "--toc-depth=2",
        "-V",
        "papersize=a4",
        "-V",
        "geometry:margin=1in",
        "-V",
        "fontsize=11pt",
        "-V",
        "linestretch=1.15",
        "--pdf-engine=xelatex",
    ]

    print(f"\n🔨 Building PDF: {output_file}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Check file size
        pdf_path = Path(output_file)
        if pdf_path.exists():
            size_mb = pdf_path.stat().st_size / (1024 * 1024)
            print(f"\nPDF built successfully!")
            print(f"   File: {output_file}")
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   Location: {pdf_path.absolute()}")
            return True
        else:
            print(f"ERROR: PDF file not created")
            return False

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: PDF build failed")
        if e.stderr:
            print(f"Details: {e.stderr}")
        return False
    except FileNotFoundError:
        print("\nERROR: pandoc not found")
        print("\nInstall pandoc with:")
        print("   macOS:  brew install pandoc")
        print("   Ubuntu: sudo apt-get install pandoc")
        print("   Windows: choco install pandoc")
        print("\nOr download from: https://pandoc.org/installing.html")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Aethelix Documentation PDF Builder")
    print("=" * 70 + "\n")

    success = build_pdf()
    sys.exit(0 if success else 1)

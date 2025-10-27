#!/usr/bin/env bash
set -euo pipefail

# This script runs the X-Ray analysis on a representative dataset
# and places the output report in the specified directory for deployment.

OUT_DIR=${1:-artifacts}

# Ensure the output directory exists
mkdir -p "$OUT_DIR"

# Run the analysis and generate the interactive report directly into the output directory.
# The main script is configured to name the report 'index.html'.
uv run python -m xray --input data/nacl1.csv --output "$OUT_DIR"

echo "Successfully generated interactive report in $OUT_DIR/index.html"

#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Example script demonstrating usage of parse_onednn_reorders.py tool

set -e

echo "========================================================================"
echo "oneDNN Reorder Extraction Tool - Example Usage"
echo "========================================================================"

# Create results directory
mkdir -p ./results

echo ""
echo "Example 1: Analyze sample trace and output to JSON"
echo "------------------------------------------------------------------------"
python scripts/parse_onednn_reorders.py \
    --trace scripts/examples/sample_trace.txt \
    --output-json ./results/sample_metrics.json

echo ""
echo "Example 2: Analyze sample trace and output to CSV"
echo "------------------------------------------------------------------------"
python scripts/parse_onednn_reorders.py \
    --trace scripts/examples/sample_trace.txt \
    --output-csv ./results/sample_metrics.csv

echo ""
echo "Example 3: Output both CSV and JSON"
echo "------------------------------------------------------------------------"
python scripts/parse_onednn_reorders.py \
    --trace scripts/examples/sample_trace.txt \
    --output-csv ./results/sample_both.csv \
    --output-json ./results/sample_both.json

echo ""
echo "========================================================================"
echo "Results written to ./results/"
echo "========================================================================"
echo ""
echo "View CSV output:"
echo "  cat ./results/sample_metrics.csv"
echo ""
echo "View JSON output:"
echo "  cat ./results/sample_metrics.json | python -m json.tool"
echo ""
echo "To compare baseline vs optimized traces, use:"
echo "  python scripts/parse_onednn_reorders.py \\"
echo "      --baseline ./traces/baseline.txt \\"
echo "      --optimized ./traces/optimized.txt \\"
echo "      --output-csv ./results/comparison.csv"
echo ""
echo "========================================================================"
echo "For full documentation, see:"
echo "  scripts/PARSE_ONEDNN_REORDERS_README.md"
echo "========================================================================"

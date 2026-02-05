#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Example script demonstrating baseline trace capture workflow

set -e

echo "=========================================="
echo "Baseline Trace Capture Examples"
echo "=========================================="
echo ""

# Make script executable if not already
chmod +x scripts/capture_baseline_trace.sh

# ==============================================================================
# Example 1: Basic Baseline Capture (Default Settings)
# ==============================================================================
echo "Example 1: Basic baseline capture with default settings"
echo "--------------------------------------------------------------"
echo "This will:"
echo "  - Extract layer 0 from Qwen2-1.5B"
echo "  - Capture 3 trace runs (50 iterations each)"
echo "  - Extract and validate metrics"
echo "  - Generate complete documentation"
echo ""
echo "Running..."
echo ""

./scripts/capture_baseline_trace.sh

echo ""
echo "✓ Example 1 complete"
echo ""
echo "Review the results:"
echo "  cat baseline_capture/BASELINE_TRACE_METRICS.md"
echo "  cat baseline_capture/metrics/baseline_run1_metrics.csv"
echo ""

# ==============================================================================
# Example 2: Custom Configuration
# ==============================================================================
echo "Example 2: Custom configuration (more runs, different layer)"
echo "--------------------------------------------------------------"
echo ""

./scripts/capture_baseline_trace.sh \
    --layer-index 14 \
    --num-runs 5 \
    --iterations 100 \
    --output-dir ./baseline_layer14

echo ""
echo "✓ Example 2 complete"
echo ""
echo "Review the results:"
echo "  cat baseline_layer14/BASELINE_TRACE_METRICS.md"
echo ""

# ==============================================================================
# Example 3: Quick Test (Fast, Minimal)
# ==============================================================================
echo "Example 3: Quick test (1 run, 10 iterations)"
echo "--------------------------------------------------------------"
echo ""

./scripts/capture_baseline_trace.sh \
    --num-runs 1 \
    --iterations 10 \
    --output-dir ./baseline_quick_test

echo ""
echo "✓ Example 3 complete"
echo ""

# ==============================================================================
# Example 4: Reuse Existing Model
# ==============================================================================
echo "Example 4: Reuse existing model (re-capture traces only)"
echo "--------------------------------------------------------------"
echo ""

# Assuming Example 1 already extracted the model
./scripts/capture_baseline_trace.sh \
    --skip-extraction \
    --num-runs 2 \
    --output-dir ./baseline_capture

echo ""
echo "✓ Example 4 complete"
echo ""

# ==============================================================================
# Summary
# ==============================================================================
echo "=========================================="
echo "All Examples Complete!"
echo "=========================================="
echo ""
echo "Generated baselines:"
echo "  1. baseline_capture/          - Standard baseline (3 runs, layer 0)"
echo "  2. baseline_layer14/          - Layer 14 baseline (5 runs)"
echo "  3. baseline_quick_test/       - Quick test (1 run, minimal)"
echo ""
echo "Next steps:"
echo "  1. Review documentation in each directory"
echo "  2. Analyze metrics files"
echo "  3. Compare different layers/configurations"
echo "  4. Begin optimization work"
echo ""
echo "For detailed analysis:"
echo "  cat baseline_capture/BASELINE_TRACE_METRICS.md"
echo "  cat baseline_capture/metrics/baseline_run1_metrics.csv"
echo ""

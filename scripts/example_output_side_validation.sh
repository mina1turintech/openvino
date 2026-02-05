#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Example: Output-Side Layout Optimization Validation
# This script demonstrates how to validate output-side optimizations

set -e

echo "======================================================================"
echo "Output-Side Layout Optimization Validation - Example Usage"
echo "======================================================================"
echo ""

# Example 1: Basic validation with default settings
echo "Example 1: Basic Validation"
echo "----------------------------"
echo ""
echo "Command:"
echo "  ./scripts/capture_output_side_trace.sh"
echo ""
echo "This will:"
echo "  - Extract layer 0 from Qwen2-1.5B-Instruct"
echo "  - Capture 3 trace runs (50 iterations each)"
echo "  - Parse traces and extract metrics"
echo "  - Generate validation report"
echo ""
echo "Output: ./output_side_validation/OUTPUT_SIDE_VALIDATION.md"
echo ""
read -p "Press Enter to run Example 1 (or Ctrl+C to skip)..."

./scripts/capture_output_side_trace.sh

echo ""
echo "✅ Example 1 complete!"
echo "   Review: ./output_side_validation/OUTPUT_SIDE_VALIDATION.md"
echo ""
echo "======================================================================"
echo ""

# Example 2: Validation with custom configuration
echo "Example 2: Custom Configuration"
echo "--------------------------------"
echo ""
echo "Command:"
echo "  ./scripts/capture_output_side_trace.sh \\"
echo "      --layer-index 12 \\"
echo "      --num-runs 5 \\"
echo "      --iterations 100 \\"
echo "      --output-dir ./custom_validation"
echo ""
echo "This validates a different layer with more runs for better statistics."
echo ""
read -p "Press Enter to run Example 2 (or Ctrl+C to skip)..."

./scripts/capture_output_side_trace.sh \
    --layer-index 12 \
    --num-runs 5 \
    --iterations 100 \
    --output-dir ./custom_validation

echo ""
echo "✅ Example 2 complete!"
echo "   Review: ./custom_validation/OUTPUT_SIDE_VALIDATION.md"
echo ""
echo "======================================================================"
echo ""

# Example 3: Reuse existing model
echo "Example 3: Reuse Existing Model"
echo "--------------------------------"
echo ""
echo "Command:"
echo "  ./scripts/capture_output_side_trace.sh \\"
echo "      --skip-extraction \\"
echo "      --output-dir ./output_side_validation"
echo ""
echo "This skips model extraction and reuses the existing model."
echo "Useful for re-running validation after code changes."
echo ""
read -p "Press Enter to run Example 3 (or Ctrl+C to skip)..."

./scripts/capture_output_side_trace.sh \
    --skip-extraction \
    --output-dir ./output_side_validation

echo ""
echo "✅ Example 3 complete!"
echo ""
echo "======================================================================"
echo ""

# Example 4: Python tool for existing traces
echo "Example 4: Validate Existing Traces"
echo "------------------------------------"
echo ""
echo "Command:"
echo "  python3 scripts/validate_output_side_optimizations.py \\"
echo "      --trace output_side_validation/traces/onednn_trace_output_side_run1.txt \\"
echo "      --output custom_validation_report.md \\"
echo "      --json validation_results.json"
echo ""
echo "This uses the Python tool directly on existing trace files."
echo ""
read -p "Press Enter to run Example 4 (or Ctrl+C to skip)..."

python3 scripts/validate_output_side_optimizations.py \
    --trace output_side_validation/traces/onednn_trace_output_side_run1.txt \
    --output custom_validation_report.md \
    --json validation_results.json

echo ""
echo "✅ Example 4 complete!"
echo "   Report: custom_validation_report.md"
echo "   JSON: validation_results.json"
echo ""
echo "======================================================================"
echo ""

# Example 5: Baseline comparison
echo "Example 5: Baseline Comparison"
echo "-------------------------------"
echo ""
echo "Command:"
echo "  python3 scripts/validate_output_side_optimizations.py \\"
echo "      --trace output_side_validation/traces/onednn_trace_output_side_run1.txt \\"
echo "      --baseline baseline_capture/traces/onednn_trace_baseline_run1.txt \\"
echo "      --output baseline_comparison_report.md"
echo ""
echo "This compares current state with baseline (if available)."
echo ""

if [ -f "baseline_capture/traces/onednn_trace_baseline_run1.txt" ]; then
    read -p "Baseline found! Press Enter to run Example 5 (or Ctrl+C to skip)..."
    
    python3 scripts/validate_output_side_optimizations.py \
        --trace output_side_validation/traces/onednn_trace_output_side_run1.txt \
        --baseline baseline_capture/traces/onednn_trace_baseline_run1.txt \
        --output baseline_comparison_report.md
    
    echo ""
    echo "✅ Example 5 complete!"
    echo "   Report: baseline_comparison_report.md"
else
    echo "⚠️  Baseline not found: baseline_capture/traces/onednn_trace_baseline_run1.txt"
    echo "   Skipping Example 5"
    echo ""
    echo "To create baseline, run:"
    echo "  ./scripts/capture_baseline_trace.sh"
fi

echo ""
echo "======================================================================"
echo ""

# Summary
echo "Summary of Examples"
echo "==================="
echo ""
echo "✅ Example 1: Basic validation completed"
echo "   Output: ./output_side_validation/"
echo ""
echo "✅ Example 2: Custom configuration completed"
echo "   Output: ./custom_validation/"
echo ""
echo "✅ Example 3: Model reuse demonstrated"
echo ""
echo "✅ Example 4: Python tool usage demonstrated"
echo "   Output: custom_validation_report.md"
echo ""
if [ -f "baseline_capture/traces/onednn_trace_baseline_run1.txt" ]; then
    echo "✅ Example 5: Baseline comparison completed"
    echo "   Output: baseline_comparison_report.md"
else
    echo "⚠️  Example 5: Skipped (no baseline)"
fi
echo ""
echo "======================================================================"
echo ""
echo "Next Steps:"
echo ""
echo "1. Review validation reports:"
echo "   - ./output_side_validation/OUTPUT_SIDE_VALIDATION.md"
echo "   - custom_validation_report.md"
echo ""
echo "2. Check validation status:"
echo "   - ✅ PASS: All criteria met, output-side is optimal"
echo "   - ⚠️  REVIEW: Some criteria need attention"
echo ""
echo "3. For more information:"
echo "   - README: scripts/OUTPUT_SIDE_VALIDATION_README.md"
echo "   - Full report: OUTPUT_SIDE_VALIDATION_REPORT.md"
echo ""
echo "======================================================================"

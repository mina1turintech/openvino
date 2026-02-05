#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Combined Optimization Trace Capture Script
#
# This script captures oneDNN traces from a fully optimized build with BOTH
# input-side (weight pre-reordering) and output-side (activation layout)
# optimizations applied. It validates that the optimizations compound positively.
#
# Usage:
#   ./scripts/capture_combined_trace.sh [OPTIONS]
#
# Options:
#   --model-name MODEL        HuggingFace model name (default: Qwen/Qwen2-1.5B-Instruct)
#   --layer-index INDEX       Transformer layer to extract (default: 0)
#   --num-runs NUM            Number of trace capture runs (default: 3)
#   --iterations NUM          Inference iterations per run (default: 50)
#   --output-dir DIR          Output directory (default: ./combined_validation)
#   --skip-extraction         Skip model extraction (use existing model)
#   --help                    Show this help message

set -e

# ==============================================================================
# Configuration and Default Parameters
# ==============================================================================

MODEL_NAME="Qwen/Qwen2-1.5B-Instruct"
LAYER_INDEX=0
NUM_RUNS=3
ITERATIONS=50
OUTPUT_DIR="./combined_validation"
SKIP_EXTRACTION=false

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --layer-index)
            LAYER_INDEX="$2"
            shift 2
            ;;
        --num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-extraction)
            SKIP_EXTRACTION=true
            shift
            ;;
        --help)
            grep "^#" "$0" | grep -v "^#!/" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Setup and Validation
# ==============================================================================

echo "=============================================================="
echo "Combined Optimization Trace Capture"
echo "=============================================================="
echo ""
echo "This script captures traces from a fully optimized build with:"
echo "  ✓ Input-side optimization (weight pre-reordering)"
echo "  ✓ Output-side optimization (activation layout)"
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Layer Index: $LAYER_INDEX"
echo "  Number of Runs: $NUM_RUNS"
echo "  Iterations per Run: $ITERATIONS"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Skip Extraction: $SKIP_EXTRACTION"
echo ""

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.10 or higher."
    exit 1
fi

# Check required packages
echo "Checking dependencies..."
python3 -c "import torch, transformers, openvino, numpy" 2>/dev/null || {
    echo "Error: Missing required packages."
    echo "Please install with: pip install torch transformers openvino numpy"
    exit 1
}
echo "✓ All dependencies found"
echo ""

# Verify this is a fully optimized build
echo "⚠️  Build Verification:"
echo "  This script assumes you have compiled OpenVINO with BOTH:"
echo "  1. Input-side optimization (Task 4) - Weight pre-reordering"
echo "  2. Output-side optimization (Task 5) - Activation layout"
echo ""
echo "  If either optimization is missing, the trace will not show"
echo "  the expected combined improvements."
echo ""
read -p "Continue with trace capture? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user."
    exit 1
fi
echo ""

# Create output directory structure
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/model"
mkdir -p "$OUTPUT_DIR/traces_combined"
mkdir -p "$OUTPUT_DIR/metrics_combined"
mkdir -p "$OUTPUT_DIR/logs"

EXTRACTED_MODEL_DIR="$OUTPUT_DIR/model"
TRACES_DIR="$OUTPUT_DIR/traces_combined"
METRICS_DIR="$OUTPUT_DIR/metrics_combined"
LOGS_DIR="$OUTPUT_DIR/logs"
MODEL_PATH="$EXTRACTED_MODEL_DIR/transformer_block.xml"

echo "Output structure created:"
echo "  Model: $EXTRACTED_MODEL_DIR"
echo "  Traces: $TRACES_DIR"
echo "  Metrics: $METRICS_DIR"
echo "  Logs: $LOGS_DIR"
echo ""

# ==============================================================================
# Step 1: Extract Single Transformer Block
# ==============================================================================

if [ "$SKIP_EXTRACTION" = true ] && [ -f "$MODEL_PATH" ]; then
    echo "Step 1: Skipping model extraction (using existing model)"
    echo "  Model path: $MODEL_PATH"
    echo ""
else
    echo "Step 1: Extracting single transformer block..."
    echo "--------------------------------------------------------------"
    
    EXTRACTION_LOG="$LOGS_DIR/extraction_combined.log"
    
    python3 scripts/extract_transformer_block.py \
        --model-name "$MODEL_NAME" \
        --layer-index "$LAYER_INDEX" \
        --output-dir "$EXTRACTED_MODEL_DIR" \
        --output-name transformer_block \
        --precision bf16 \
        2>&1 | tee "$EXTRACTION_LOG"
    
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Error: Model extraction failed. Expected file not found: $MODEL_PATH"
        exit 1
    fi
    
    echo ""
    echo "✓ Step 1 complete: Model extracted successfully"
    echo "  Model: $MODEL_PATH"
    echo "  Log: $EXTRACTION_LOG"
    echo ""
fi

# ==============================================================================
# Step 2: Capture Traces with Full Optimizations (Multiple Runs)
# ==============================================================================

echo "Step 2: Capturing combined optimization traces ($NUM_RUNS runs)..."
echo "--------------------------------------------------------------"
echo ""
echo "ℹ️  These traces will reflect:"
echo "  • Reduced weight reorders (input-side optimization)"
echo "  • Reduced activation reorders (output-side optimization)"
echo "  • Any interactions between the two optimizations"
echo ""

TRACE_FILES=()

for run in $(seq 1 $NUM_RUNS); do
    echo "  Run $run/$NUM_RUNS..."
    
    TAG="combined_run${run}"
    TRACE_LOG="$LOGS_DIR/capture_combined_run${run}.log"
    
    python3 scripts/capture_onednn_trace.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$TRACES_DIR" \
        --tag "$TAG" \
        --device CPU \
        --batch-size 1 \
        --seq-length 16 \
        --iterations "$ITERATIONS" \
        --verbose-level 1 \
        --seed 42 \
        --no-timestamp \
        2>&1 | tee "$TRACE_LOG"
    
    TRACE_FILE="$TRACES_DIR/onednn_trace_${TAG}.txt"
    
    if [ ! -f "$TRACE_FILE" ]; then
        echo "Error: Trace capture failed for run $run"
        exit 1
    fi
    
    TRACE_FILES+=("$TRACE_FILE")
    
    echo "  ✓ Run $run complete: $TRACE_FILE"
    echo ""
done

echo "✓ Step 2 complete: All traces captured"
echo "  Total traces: ${#TRACE_FILES[@]}"
for trace in "${TRACE_FILES[@]}"; do
    echo "    - $trace"
done
echo ""

# ==============================================================================
# Step 3: Parse Traces and Extract Reorder Metrics
# ==============================================================================

echo "Step 3: Parsing traces and extracting reorder metrics..."
echo "--------------------------------------------------------------"

METRICS_FILES=()

for i in "${!TRACE_FILES[@]}"; do
    run=$((i + 1))
    trace_file="${TRACE_FILES[$i]}"
    
    echo "  Parsing run $run..."
    
    # Extract metrics in both CSV and JSON formats
    csv_file="$METRICS_DIR/combined_run${run}_metrics.csv"
    json_file="$METRICS_DIR/combined_run${run}_metrics.json"
    parse_log="$LOGS_DIR/parse_combined_run${run}.log"
    
    python3 scripts/parse_onednn_reorders.py \
        --trace-file "$trace_file" \
        --output-csv "$csv_file" \
        --output-json "$json_file" \
        2>&1 | tee "$parse_log"
    
    if [ ! -f "$json_file" ]; then
        echo "Error: Metrics parsing failed for run $run"
        exit 1
    fi
    
    METRICS_FILES+=("$json_file")
    
    echo "  ✓ Run $run parsed: $json_file"
    echo ""
done

echo "✓ Step 3 complete: All metrics extracted"
echo "  Total metric files: ${#METRICS_FILES[@]}"
for metrics in "${METRICS_FILES[@]}"; do
    echo "    - $metrics"
done
echo ""

# ==============================================================================
# Step 4: Generate Quick Summary Report
# ==============================================================================

echo "Step 4: Generating quick summary..."
echo "--------------------------------------------------------------"

# Calculate aggregate metrics from the first JSON file
FIRST_JSON="${METRICS_FILES[0]}"

echo ""
echo "Quick Summary (from first run):"
echo "================================"

python3 - <<EOF
import json
import sys

try:
    with open('$FIRST_JSON') as f:
        metrics = json.load(f)
    
    summary = metrics.get('summary', {})
    total_count = summary.get('total_reorder_count', 0)
    total_time = summary.get('total_reorder_time_ms', 0.0)
    
    print(f"Total Reorder Count:     {total_count}")
    print(f"Total Reorder Time:      {total_time:.3f} ms")
    
    if total_count > 0:
        avg_time = total_time / total_count
        print(f"Average Reorder Time:    {avg_time:.3f} ms")
    
    print("")
    print("By Dimension (top 5 by time):")
    by_dim = metrics.get('by_dimension', {})
    sorted_dims = sorted(by_dim.items(), key=lambda x: x[1].get('time_ms', 0), reverse=True)[:5]
    
    for dim_str, dim_data in sorted_dims:
        count = dim_data.get('count', 0)
        time = dim_data.get('time_ms', 0.0)
        print(f"  {dim_str:15s} : {count:3d} ops, {time:8.3f} ms")

except Exception as e:
    print(f"Error generating summary: {e}")
    sys.exit(1)
EOF

echo ""

# ==============================================================================
# Completion Summary
# ==============================================================================

echo ""
echo "=============================================================="
echo "✓ Combined Optimization Trace Capture Complete"
echo "=============================================================="
echo ""
echo "Output Files:"
echo "  Model:   $MODEL_PATH"
echo "  Traces:  $TRACES_DIR/"
echo "  Metrics: $METRICS_DIR/"
echo "  Logs:    $LOGS_DIR/"
echo ""
echo "Next Steps:"
echo "  1. Compare with baseline using:"
echo "     python3 scripts/compare_combined_traces.py \\"
echo "       --baseline ./baseline_capture/metrics \\"
echo "       --combined $METRICS_DIR \\"
echo "       --input-side ./input_side_validation/metrics_optimized \\"
echo "       --output-side ./output_side_validation/metrics_optimized \\"
echo "       --output ./combined_validation/COMBINED_COMPARISON.md"
echo ""
echo "  2. Review the comparison report to validate:"
echo "     • Combined improvements ≈ input-side + output-side"
echo "     • No unexpected regressions"
echo "     • Optimizations compound positively"
echo ""
echo "See COMBINED_OPTIMIZATION_COMPARISON_GUIDE.md for detailed usage."
echo "=============================================================="

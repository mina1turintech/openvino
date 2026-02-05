#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Input-Side Optimization Trace Capture and Comparison Script
#
# This script captures oneDNN traces to validate input-side (weight reorder)
# optimizations. It focuses on measuring reorder operations before compute
# phases (FFN and attention) to validate that weight pre-reordering reduces
# runtime overhead.
#
# Usage:
#   ./scripts/capture_input_side_trace.sh [OPTIONS]
#
# Options:
#   --model-name MODEL        HuggingFace model name (default: Qwen/Qwen2-1.5B-Instruct)
#   --layer-index INDEX       Transformer layer to extract (default: 0)
#   --num-runs NUM            Number of trace capture runs (default: 3)
#   --iterations NUM          Inference iterations per run (default: 50)
#   --output-dir DIR          Output directory (default: ./input_side_validation)
#   --skip-extraction         Skip model extraction (use existing model)
#   --baseline                Capture baseline trace (unoptimized)
#   --optimized               Capture optimized trace (with weight pre-reordering)
#   --help                    Show this help message

set -e

# ==============================================================================
# Configuration and Default Parameters
# ==============================================================================

MODEL_NAME="Qwen/Qwen2-1.5B-Instruct"
LAYER_INDEX=0
NUM_RUNS=3
ITERATIONS=50
OUTPUT_DIR="./input_side_validation"
SKIP_EXTRACTION=false
MODE="baseline"  # baseline or optimized

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
        --baseline)
            MODE="baseline"
            shift
            ;;
        --optimized)
            MODE="optimized"
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
echo "Input-Side Optimization Trace Capture"
echo "=============================================================="
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Layer Index: $LAYER_INDEX"
echo "  Mode: $MODE"
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

# Create output directory structure
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/model"
mkdir -p "$OUTPUT_DIR/traces_${MODE}"
mkdir -p "$OUTPUT_DIR/metrics_${MODE}"
mkdir -p "$OUTPUT_DIR/logs"

EXTRACTED_MODEL_DIR="$OUTPUT_DIR/model"
TRACES_DIR="$OUTPUT_DIR/traces_${MODE}"
METRICS_DIR="$OUTPUT_DIR/metrics_${MODE}"
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
    
    EXTRACTION_LOG="$LOGS_DIR/extraction_${MODE}.log"
    
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
# Step 2: Capture Traces (Multiple Runs for Reproducibility)
# ==============================================================================

echo "Step 2: Capturing ${MODE} traces ($NUM_RUNS runs)..."
echo "--------------------------------------------------------------"

if [ "$MODE" = "optimized" ]; then
    echo "⚠️  NOTE: Optimized mode requires weight pre-reordering to be implemented."
    echo "    If not implemented, this will capture the same trace as baseline."
    echo "    See INPUT_SIDE_OPTIMIZATION_GUIDE.md for implementation details."
    echo ""
fi

TRACE_FILES=()

for run in $(seq 1 $NUM_RUNS); do
    echo "  Run $run/$NUM_RUNS..."
    
    TAG="${MODE}_run${run}"
    TRACE_LOG="$LOGS_DIR/capture_${MODE}_run${run}.log"
    
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
# Step 3: Parse Traces and Extract Weight Reorder Metrics
# ==============================================================================

echo "Step 3: Parsing traces and extracting weight reorder metrics..."
echo "--------------------------------------------------------------"

METRICS_FILES=()

for i in "${!TRACE_FILES[@]}"; do
    run=$((i + 1))
    trace_file="${TRACE_FILES[$i]}"
    
    echo "  Parsing run $run..."
    
    # Extract metrics in both CSV and JSON formats
    csv_file="$METRICS_DIR/${MODE}_run${run}_metrics.csv"
    json_file="$METRICS_DIR/${MODE}_run${run}_metrics.json"
    parse_log="$LOGS_DIR/parse_${MODE}_run${run}.log"
    
    python3 scripts/parse_onednn_reorders.py \
        --trace "$trace_file" \
        --output-csv "$csv_file" \
        --output-json "$json_file" \
        2>&1 | tee "$parse_log"
    
    if [ ! -f "$csv_file" ] || [ ! -f "$json_file" ]; then
        echo "Error: Metrics extraction failed for run $run"
        exit 1
    fi
    
    METRICS_FILES+=("$csv_file")
    
    echo "  ✓ Run $run metrics extracted"
    echo ""
done

echo "✓ Step 3 complete: All metrics extracted"
echo "  Total metric files: ${#METRICS_FILES[@]}"
for metrics in "${METRICS_FILES[@]}"; do
    echo "    - $metrics"
done
echo ""

# ==============================================================================
# Step 4: Generate Weight-Focused Analysis Report
# ==============================================================================

echo "Step 4: Generating weight reorder analysis report..."
echo "--------------------------------------------------------------"

REPORT_FILE="$OUTPUT_DIR/INPUT_SIDE_${MODE^^}_REPORT.md"

python3 << EOF
import json
import sys
from pathlib import Path

# Read first run metrics as representative sample
metrics_file = Path("$METRICS_DIR/${MODE}_run1_metrics.json")
if not metrics_file.exists():
    print(f"Error: Metrics file not found: {metrics_file}")
    sys.exit(1)

with open(metrics_file) as f:
    metrics = json.load(f)

# Generate report
report = []
report.append("# Input-Side ${MODE^^} Trace Analysis Report")
report.append("")
report.append("## Executive Summary")
report.append("")
report.append(f"**Mode**: {MODE}")
report.append(f"**Model**: $MODEL_NAME")
report.append(f"**Layer**: $LAYER_INDEX")
report.append(f"**Runs**: $NUM_RUNS")
report.append(f"**Iterations per run**: $ITERATIONS")
report.append("")

# Extract key metrics
total_reorders = metrics.get('summary', {}).get('total_reorder_count', 0)
total_time_ms = metrics.get('summary', {}).get('total_reorder_time_ms', 0)

report.append("## Key Metrics")
report.append("")
report.append(f"- **Total Reorder Operations**: {total_reorders}")
report.append(f"- **Total Reorder Time**: {total_time_ms:.3f} ms")
if total_reorders > 0:
    avg_time = total_time_ms / total_reorders
    report.append(f"- **Average Reorder Time**: {avg_time:.3f} ms")
report.append("")

# Weight-specific reorders (dimensions 1536, 8960, 256)
report.append("## Weight Reorder Analysis")
report.append("")
report.append("### Reorders by Dimension (Weight-Related)")
report.append("")
report.append("| Dimension | Count | Total Time (ms) | Avg Time (ms) | Category |")
report.append("|-----------|-------|-----------------|---------------|----------|")

by_dim = metrics.get('by_dimension', {})
weight_dims = {
    '1536x1536': 'Attention Output / FFN Intermediate',
    '256x1536': 'Q/K/V Projections',
    '1536x8960': 'FFN Expand Weights',
    '8960x1536': 'FFN Contract Weights',
    '1536x1': 'Scale/Zero-Point Vectors',
    '8960x1': 'FFN Scale/Zero-Point',
}

for dim_str, category in weight_dims.items():
    if dim_str in by_dim:
        dim_data = by_dim[dim_str]
        count = dim_data.get('count', 0)
        time_ms = dim_data.get('time_ms', 0)
        avg = time_ms / count if count > 0 else 0
        report.append(f"| {dim_str} | {count} | {time_ms:.3f} | {avg:.3f} | {category} |")

report.append("")
report.append("### Expected Optimizations")
report.append("")

if "$MODE" == "baseline":
    report.append("**Baseline Mode**: This trace captures current behavior with runtime weight reordering.")
    report.append("")
    report.append("Expected improvements with input-side optimization:")
    report.append("- ✅ Eliminate 1536x1536 reorders (~1.4ms each)")
    report.append("- ✅ Eliminate 256x1536 reorders (~0.23ms each)")
    report.append("- ✅ Eliminate 8960x1536 and 1536x8960 reorders")
    report.append("- ⚠️  Scale/zero-point reorders (1536x1, 8960x1) may remain (small overhead)")
    report.append("")
    report.append("**Implementation Strategy**:")
    report.append("1. Pre-reorder weights at model load time from u8::ab to u8::AB8b24a")
    report.append("2. Store reordered weights in global weight cache")
    report.append("3. Skip runtime reordering in \`prepareWeightsMemory()\`")
else:
    report.append("**Optimized Mode**: This trace should show reduced weight reorder overhead.")
    report.append("")
    report.append("If properly implemented, you should see:")
    report.append("- ✅ Zero or minimal 1536x1536 reorders")
    report.append("- ✅ Zero or minimal 256x1536 reorders")
    report.append("- ✅ Zero or minimal large weight reorders")
    report.append("- ⚠️  Small scale/zero-point reorders may remain")

report.append("")
report.append("## Trace Files")
report.append("")
for i, trace_file in enumerate("${TRACE_FILES[@]}".split(), 1):
    report.append(f"{i}. \`{trace_file}\`")

report.append("")
report.append("## Metrics Files")
report.append("")
for i, metrics_file in enumerate("${METRICS_FILES[@]}".split(), 1):
    report.append(f"{i}. \`{metrics_file}\`")

report.append("")
report.append("---")
report.append("")
report.append("**Generated**: $(date)")
report.append(f"**Script**: scripts/capture_input_side_trace.sh")

# Write report
with open("$REPORT_FILE", 'w') as f:
    f.write('\n'.join(report))

print(f"✓ Report generated: $REPORT_FILE")
EOF

if [ ! -f "$REPORT_FILE" ]; then
    echo "Error: Report generation failed"
    exit 1
fi

echo ""
echo "✓ Step 4 complete: Analysis report generated"
echo "  Report: $REPORT_FILE"
echo ""

# ==============================================================================
# Summary
# ==============================================================================

echo "=============================================================="
echo "Input-Side Trace Capture Complete"
echo "=============================================================="
echo ""
echo "Outputs:"
echo "  📁 Traces: $TRACES_DIR"
echo "  📊 Metrics: $METRICS_DIR"
echo "  📝 Report: $REPORT_FILE"
echo "  📋 Logs: $LOGS_DIR"
echo ""
echo "Next steps:"
if [ "$MODE" = "baseline" ]; then
    echo "  1. Run with --optimized flag to capture optimized trace"
    echo "  2. Use compare_input_side_traces.py to generate comparison"
else
    echo "  1. Review the comparison report"
    echo "  2. Validate reorder reduction against success criteria"
fi
echo ""
echo "For comparison, run:"
echo "  python3 scripts/compare_input_side_traces.py \\"
echo "      --baseline $OUTPUT_DIR/metrics_baseline \\"
echo "      --optimized $OUTPUT_DIR/metrics_optimized \\"
echo "      --output $OUTPUT_DIR/INPUT_SIDE_COMPARISON.md"
echo ""

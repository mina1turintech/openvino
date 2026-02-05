#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Multi-Scenario Trace Capture Script
#
# This script captures oneDNN traces under diverse inference conditions to validate
# the robustness of layout optimizations across different scenarios including:
# - Different batch sizes (1, 4, 8)
# - Different sequence lengths (1, 32, 128, 256)
# - Different input patterns (random, ones, small_values)
# - Multiple repeated passes (3-5 per scenario) for variance analysis
#
# Usage:
#   ./scripts/capture_multi_scenario_traces.sh [OPTIONS]
#
# Options:
#   --model-path PATH         Path to extracted model (default: auto-extract)
#   --output-dir DIR          Output directory (default: ./multi_scenario_validation)
#   --model-name MODEL        HuggingFace model name (default: Qwen/Qwen2-1.5B-Instruct)
#   --layer-index INDEX       Transformer layer to extract (default: 0)
#   --repetitions NUM         Number of repetitions per scenario (default: 3)
#   --iterations NUM          Inference iterations per capture (default: 50)
#   --skip-extraction         Skip model extraction (use existing model)
#   --quick                   Run quick test with reduced scenarios
#   --help                    Show this help message

set -e

# ==============================================================================
# Configuration and Default Parameters
# ==============================================================================

MODEL_NAME="Qwen/Qwen2-1.5B-Instruct"
LAYER_INDEX=0
OUTPUT_DIR="./multi_scenario_validation"
MODEL_PATH=""
SKIP_EXTRACTION=false
REPETITIONS=3
ITERATIONS=50
QUICK_MODE=false

# Test Matrix Configuration
BATCH_SIZES=(1 4 8)
SEQ_LENGTHS=(1 32 128 256)
INPUT_PATTERNS=("random" "ones" "small_values")

# Quick mode: reduced test matrix
QUICK_BATCH_SIZES=(1 4)
QUICK_SEQ_LENGTHS=(1 128)
QUICK_INPUT_PATTERNS=("random" "ones")

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            SKIP_EXTRACTION=true
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --layer-index)
            LAYER_INDEX="$2"
            shift 2
            ;;
        --repetitions)
            REPETITIONS="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        --skip-extraction)
            SKIP_EXTRACTION=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
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

# Use quick mode configuration if enabled
if [ "$QUICK_MODE" = true ]; then
    BATCH_SIZES=("${QUICK_BATCH_SIZES[@]}")
    SEQ_LENGTHS=("${QUICK_SEQ_LENGTHS[@]}")
    INPUT_PATTERNS=("${QUICK_INPUT_PATTERNS[@]}")
fi

# ==============================================================================
# Setup and Validation
# ==============================================================================

echo "=============================================================="
echo "Multi-Scenario Trace Capture"
echo "=============================================================="
echo ""
echo "This script validates optimization robustness across:"
echo "  • Multiple batch sizes: ${BATCH_SIZES[*]}"
echo "  • Multiple sequence lengths: ${SEQ_LENGTHS[*]}"
echo "  • Multiple input patterns: ${INPUT_PATTERNS[*]}"
echo "  • ${REPETITIONS} repetitions per scenario for variance analysis"
echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Layer Index: $LAYER_INDEX"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Iterations per Run: $ITERATIONS"
if [ "$QUICK_MODE" = true ]; then
    echo "  Mode: QUICK (reduced test matrix)"
else
    echo "  Mode: FULL (complete test matrix)"
fi
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

# Calculate total scenarios
TOTAL_SCENARIOS=$((${#BATCH_SIZES[@]} * ${#SEQ_LENGTHS[@]} * ${#INPUT_PATTERNS[@]}))
TOTAL_RUNS=$((TOTAL_SCENARIOS * REPETITIONS))

echo "Test Matrix Summary:"
echo "  Batch sizes: ${#BATCH_SIZES[@]}"
echo "  Sequence lengths: ${#SEQ_LENGTHS[@]}"
echo "  Input patterns: ${#INPUT_PATTERNS[@]}"
echo "  Repetitions per scenario: $REPETITIONS"
echo "  Total scenarios: $TOTAL_SCENARIOS"
echo "  Total trace captures: $TOTAL_RUNS"
echo ""

ESTIMATED_TIME=$((TOTAL_RUNS * 2))  # ~2 minutes per trace capture
echo "Estimated time: ~${ESTIMATED_TIME} minutes"
echo ""

read -p "Continue with multi-scenario capture? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user."
    exit 1
fi
echo ""

# Create output directory structure
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/model"
mkdir -p "$OUTPUT_DIR/traces"
mkdir -p "$OUTPUT_DIR/metrics"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/statistics"

EXTRACTED_MODEL_DIR="$OUTPUT_DIR/model"
TRACES_DIR="$OUTPUT_DIR/traces"
METRICS_DIR="$OUTPUT_DIR/metrics"
LOGS_DIR="$OUTPUT_DIR/logs"
STATS_DIR="$OUTPUT_DIR/statistics"

if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="$EXTRACTED_MODEL_DIR/transformer_block.xml"
fi

echo "Output structure created:"
echo "  Model: $EXTRACTED_MODEL_DIR"
echo "  Traces: $TRACES_DIR"
echo "  Metrics: $METRICS_DIR"
echo "  Logs: $LOGS_DIR"
echo "  Statistics: $STATS_DIR"
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
    
    EXTRACTION_LOG="$LOGS_DIR/extraction.log"
    
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
# Step 2: Capture Traces for All Scenarios
# ==============================================================================

echo "Step 2: Capturing traces for all scenarios..."
echo "=============================================================="
echo ""

# Create scenario matrix file
SCENARIO_MATRIX="$OUTPUT_DIR/scenario_matrix.txt"
echo "# Multi-Scenario Test Matrix" > "$SCENARIO_MATRIX"
echo "# Generated: $(date)" >> "$SCENARIO_MATRIX"
echo "# Format: scenario_id,batch_size,seq_length,input_pattern,repetition" >> "$SCENARIO_MATRIX"

SCENARIO_ID=0
COMPLETED=0

# Iterate through all scenario combinations
for batch_size in "${BATCH_SIZES[@]}"; do
    for seq_length in "${SEQ_LENGTHS[@]}"; do
        for input_pattern in "${INPUT_PATTERNS[@]}"; do
            SCENARIO_ID=$((SCENARIO_ID + 1))
            
            echo ""
            echo "Scenario $SCENARIO_ID/$TOTAL_SCENARIOS:"
            echo "  Batch Size: $batch_size"
            echo "  Sequence Length: $seq_length"
            echo "  Input Pattern: $input_pattern"
            echo "--------------------------------------------------------------"
            
            # Create scenario-specific directory
            SCENARIO_DIR="$TRACES_DIR/batch${batch_size}_seq${seq_length}_${input_pattern}"
            mkdir -p "$SCENARIO_DIR"
            mkdir -p "$METRICS_DIR/batch${batch_size}_seq${seq_length}_${input_pattern}"
            
            # Capture multiple repetitions for variance analysis
            for rep in $(seq 1 $REPETITIONS); do
                echo "  Repetition $rep/$REPETITIONS..."
                
                TAG="batch${batch_size}_seq${seq_length}_${input_pattern}_rep${rep}"
                TRACE_LOG="$LOGS_DIR/capture_${TAG}.log"
                
                # Capture trace with specific parameters
                python3 scripts/capture_onednn_trace.py \
                    --model-path "$MODEL_PATH" \
                    --output-dir "$SCENARIO_DIR" \
                    --tag "rep${rep}" \
                    --device CPU \
                    --batch-size "$batch_size" \
                    --seq-length "$seq_length" \
                    --input-pattern "$input_pattern" \
                    --iterations "$ITERATIONS" \
                    --verbose-level 1 \
                    --seed 42 \
                    --no-timestamp \
                    2>&1 | tee "$TRACE_LOG" > /dev/null
                
                TRACE_FILE="$SCENARIO_DIR/onednn_trace_rep${rep}.txt"
                
                if [ ! -f "$TRACE_FILE" ]; then
                    echo "  ✗ Error: Trace capture failed for repetition $rep"
                    continue
                fi
                
                # Parse trace and extract metrics
                METRICS_FILE="$METRICS_DIR/batch${batch_size}_seq${seq_length}_${input_pattern}/metrics_rep${rep}.json"
                
                python3 scripts/parse_onednn_reorders.py \
                    --trace "$TRACE_FILE" \
                    --output-json "$METRICS_FILE" \
                    > /dev/null 2>&1
                
                if [ ! -f "$METRICS_FILE" ]; then
                    echo "  ✗ Warning: Metrics extraction failed for repetition $rep"
                else
                    echo "  ✓ Repetition $rep complete"
                fi
                
                # Record in matrix
                echo "$SCENARIO_ID,$batch_size,$seq_length,$input_pattern,$rep" >> "$SCENARIO_MATRIX"
                
                COMPLETED=$((COMPLETED + 1))
            done
            
            echo "  ✓ Scenario $SCENARIO_ID complete (all repetitions)"
        done
    done
done

echo ""
echo "=============================================================="
echo "✓ Step 2 complete: All traces captured"
echo "  Total runs completed: $COMPLETED"
echo "  Scenario matrix saved: $SCENARIO_MATRIX"
echo ""

# ==============================================================================
# Step 3: Quick Statistics Summary
# ==============================================================================

echo "Step 3: Generating quick statistics summary..."
echo "--------------------------------------------------------------"

QUICK_SUMMARY="$OUTPUT_DIR/QUICK_SUMMARY.txt"

cat > "$QUICK_SUMMARY" << EOF
Multi-Scenario Trace Capture Summary
Generated: $(date)
============================================================

Configuration:
  Model: $MODEL_NAME
  Layer Index: $LAYER_INDEX
  Optimizations: Input-side + Output-side (if enabled)
  
Test Matrix:
  Batch Sizes: ${BATCH_SIZES[*]}
  Sequence Lengths: ${SEQ_LENGTHS[*]}
  Input Patterns: ${INPUT_PATTERNS[*]}
  Repetitions per Scenario: $REPETITIONS
  
Results:
  Total Scenarios: $TOTAL_SCENARIOS
  Total Runs Completed: $COMPLETED
  
Directory Structure:
  Traces: $TRACES_DIR
  Metrics: $METRICS_DIR
  Logs: $LOGS_DIR
  Statistics: $STATS_DIR
  
Next Steps:
  1. Run statistical analysis:
     python3 scripts/analyze_multi_scenario_statistics.py \\
         --metrics-dir $METRICS_DIR \\
         --output $STATS_DIR/MULTI_SCENARIO_ANALYSIS.md
  
  2. Review results:
     cat $STATS_DIR/MULTI_SCENARIO_ANALYSIS.md
  
  3. Check for variance > 5%:
     grep "High Variance" $STATS_DIR/MULTI_SCENARIO_ANALYSIS.md
  
Success Criteria:
  ✓ Reorder reduction consistent across batch sizes (> 90%)
  ✓ Reorder reduction consistent across sequence lengths (> 90%)
  ✓ Input patterns do not affect optimization (variance < 5%)
  ✓ Low standard deviation across repetitions
  ✓ No unexpected regressions in any scenario

============================================================
EOF

cat "$QUICK_SUMMARY"

echo ""
echo "✓ Quick summary saved to: $QUICK_SUMMARY"
echo ""

# ==============================================================================
# Completion
# ==============================================================================

echo "=============================================================="
echo "SUCCESS! Multi-scenario trace capture complete"
echo "=============================================================="
echo ""
echo "All traces and metrics have been captured. Next steps:"
echo ""
echo "1. Run statistical analysis:"
echo "   python3 scripts/analyze_multi_scenario_statistics.py \\"
echo "       --metrics-dir $METRICS_DIR \\"
echo "       --output $STATS_DIR/MULTI_SCENARIO_ANALYSIS.md"
echo ""
echo "2. Review the comprehensive analysis report"
echo ""
echo "3. Validate success criteria in the report"
echo ""
echo "For interactive validation:"
echo "   ./scripts/example_multi_scenario_validation.sh"
echo ""
echo "=============================================================="

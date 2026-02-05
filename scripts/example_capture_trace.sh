#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Example script demonstrating how to capture oneDNN verbose traces
# from the extracted transformer block for baseline and optimization comparison

set -e

echo "=================================================================="
echo "oneDNN Verbose Trace Capture Examples"
echo "=================================================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.10 or higher."
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import openvino, numpy" 2>/dev/null || {
    echo "Error: Missing required packages."
    echo "Please install with: pip install openvino numpy"
    exit 1
}
echo "✓ All dependencies found"
echo ""

# Check if extracted model exists
MODEL_DIR="./extracted_block"
MODEL_PATH="${MODEL_DIR}/transformer_block.xml"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Extracted model not found at $MODEL_PATH"
    echo ""
    echo "Please run extraction first:"
    echo "  python3 scripts/extract_transformer_block.py \\"
    echo "      --model-name Qwen/Qwen2-1.5B-Instruct \\"
    echo "      --layer-index 0 \\"
    echo "      --output-dir ./extracted_block"
    echo ""
    exit 1
fi

echo "✓ Found extracted model: $MODEL_PATH"
echo ""

# Create output directory for traces
TRACES_DIR="./traces"
mkdir -p "$TRACES_DIR"

echo "Output directory: $TRACES_DIR"
echo ""

# ==============================================================================
# Example 1: Basic Baseline Trace Capture
# ==============================================================================
echo "Example 1: Capturing baseline trace (basic)..."
echo "--------------------------------------------------------------"

python3 scripts/capture_onednn_trace.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$TRACES_DIR" \
    --tag baseline \
    --iterations 10

echo ""
echo "✓ Example 1 complete"
echo ""

# ==============================================================================
# Example 2: Detailed Trace with More Iterations
# ==============================================================================
echo "Example 2: Capturing detailed trace (100 iterations)..."
echo "--------------------------------------------------------------"

python3 scripts/capture_onednn_trace.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$TRACES_DIR" \
    --tag baseline_detailed \
    --iterations 100 \
    --verbose-level 2

echo ""
echo "✓ Example 2 complete"
echo ""

# ==============================================================================
# Example 3: Different Batch Sizes for Comparison
# ==============================================================================
echo "Example 3: Capturing traces with different batch sizes..."
echo "--------------------------------------------------------------"

for batch_size in 1 2 4; do
    echo "  Batch size: $batch_size"
    python3 scripts/capture_onednn_trace.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$TRACES_DIR" \
        --tag "baseline_bs${batch_size}" \
        --batch-size $batch_size \
        --iterations 20 \
        --no-timestamp
done

echo ""
echo "✓ Example 3 complete"
echo ""

# ==============================================================================
# Example 4: Different Sequence Lengths for Analysis
# ==============================================================================
echo "Example 4: Capturing traces with different sequence lengths..."
echo "--------------------------------------------------------------"

for seq_len in 16 32 64; do
    echo "  Sequence length: $seq_len"
    python3 scripts/capture_onednn_trace.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$TRACES_DIR" \
        --tag "baseline_seq${seq_len}" \
        --seq-length $seq_len \
        --iterations 20 \
        --no-timestamp
done

echo ""
echo "✓ Example 4 complete"
echo ""

# ==============================================================================
# Example 5: Reproducibility Test (Multiple Runs with Same Seed)
# ==============================================================================
echo "Example 5: Reproducibility test (3 runs with same seed)..."
echo "--------------------------------------------------------------"

for run in 1 2 3; do
    echo "  Run $run"
    python3 scripts/capture_onednn_trace.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$TRACES_DIR" \
        --tag "reproducibility_run${run}" \
        --seed 42 \
        --iterations 10 \
        --no-timestamp
done

echo ""
echo "Comparing reproducibility runs..."
TRACE1="${TRACES_DIR}/onednn_trace_reproducibility_run1.txt"
TRACE2="${TRACES_DIR}/onednn_trace_reproducibility_run2.txt"
TRACE3="${TRACES_DIR}/onednn_trace_reproducibility_run3.txt"

if cmp -s "$TRACE1" "$TRACE2" && cmp -s "$TRACE2" "$TRACE3"; then
    echo "✓ All runs produced identical traces (reproducible!)"
else
    echo "⚠ Warning: Runs produced different traces (check timestamps/system load)"
fi

echo ""
echo "✓ Example 5 complete"
echo ""

# ==============================================================================
# Summary and Analysis
# ==============================================================================
echo "=================================================================="
echo "Trace Capture Complete!"
echo "=================================================================="
echo ""
echo "Captured traces saved to: $TRACES_DIR"
echo ""
echo "Generated files:"
ls -lh "$TRACES_DIR"/*.txt 2>/dev/null | head -n 10
echo ""

# Count total traces
NUM_TRACES=$(ls -1 "$TRACES_DIR"/onednn_trace_*.txt 2>/dev/null | wc -l)
echo "Total traces captured: $NUM_TRACES"
echo ""

# Analyze baseline trace
BASELINE_TRACE=$(ls -1 "$TRACES_DIR"/onednn_trace_baseline_*.txt 2>/dev/null | head -n 1)
if [ -f "$BASELINE_TRACE" ]; then
    echo "Analyzing baseline trace: $BASELINE_TRACE"
    echo "  Total lines: $(wc -l < "$BASELINE_TRACE")"
    echo "  Reorder operations: $(grep -c "reorder" "$BASELINE_TRACE" 2>/dev/null || echo 0)"
    echo "  MatMul operations: $(grep -c "matmul" "$BASELINE_TRACE" 2>/dev/null || echo 0)"
    echo "  InnerProduct operations: $(grep -c "inner_product" "$BASELINE_TRACE" 2>/dev/null || echo 0)"
    echo ""
    
    # Check for expected dimensions
    if grep -q "1536" "$BASELINE_TRACE" && grep -q "8960" "$BASELINE_TRACE"; then
        echo "  ✓ Expected dimensions (1536, 8960) found in trace"
    else
        echo "  ⚠ Expected dimensions (1536, 8960) not found"
    fi
    echo ""
fi

# ==============================================================================
# Next Steps and Usage Instructions
# ==============================================================================
echo "Next steps:"
echo "=================================================================="
echo ""
echo "1. Extract reorder operations from baseline trace:"
echo "   grep -i reorder $TRACES_DIR/onednn_trace_baseline_*.txt > baseline_reorders.txt"
echo ""
echo "2. Analyze execution times:"
echo "   grep reorder $TRACES_DIR/onednn_trace_baseline_*.txt | awk -F',' '{print \$NF}'"
echo ""
echo "3. Compare different configurations:"
echo "   diff $TRACES_DIR/onednn_trace_baseline_bs1.txt $TRACES_DIR/onednn_trace_baseline_bs2.txt"
echo ""
echo "4. After optimization, capture optimized trace:"
echo "   python3 scripts/capture_onednn_trace.py \\"
echo "       --model-path ./optimized_block/transformer_block.xml \\"
echo "       --output-dir $TRACES_DIR \\"
echo "       --tag optimized"
echo ""
echo "5. Compare baseline vs optimized:"
echo "   grep -i reorder $TRACES_DIR/onednn_trace_baseline_*.txt > baseline_reorders.txt"
echo "   grep -i reorder $TRACES_DIR/onednn_trace_optimized_*.txt > optimized_reorders.txt"
echo "   diff baseline_reorders.txt optimized_reorders.txt"
echo ""
echo "6. Count reorder reduction:"
echo "   echo \"Baseline: \$(grep -c reorder $TRACES_DIR/onednn_trace_baseline_*.txt)\""
echo "   echo \"Optimized: \$(grep -c reorder $TRACES_DIR/onednn_trace_optimized_*.txt)\""
echo ""
echo "=================================================================="
echo "For more information, see:"
echo "  - scripts/ONEDNN_TRACE_CAPTURE_README.md"
echo "  - python3 scripts/capture_onednn_trace.py --help"
echo "=================================================================="
echo ""

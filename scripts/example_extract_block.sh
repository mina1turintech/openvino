#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Example script demonstrating how to extract transformer blocks from Qwen2-1.5B

set -e

echo "=========================================="
echo "Qwen2-1.5B Transformer Block Extraction"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.10 or higher."
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import torch, transformers, openvino" 2>/dev/null || {
    echo "Error: Missing required packages."
    echo "Please install with: pip install torch transformers openvino"
    exit 1
}
echo "✓ All dependencies found"
echo ""

# Create output directory
OUTPUT_DIR="./extracted_blocks"
mkdir -p "$OUTPUT_DIR"

# Example 1: Extract first layer (layer 0) with BFloat16 precision
echo "Example 1: Extracting first layer (layer 0)..."
python3 scripts/extract_transformer_block.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --layer-index 0 \
    --output-dir "$OUTPUT_DIR/layer_0" \
    --output-name transformer_block_0 \
    --precision bf16

echo ""
echo "✓ Example 1 complete"
echo ""

# Example 2: Extract middle layer (layer 14) for comparison
echo "Example 2: Extracting middle layer (layer 14)..."
python3 scripts/extract_transformer_block.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --layer-index 14 \
    --output-dir "$OUTPUT_DIR/layer_14" \
    --output-name transformer_block_14 \
    --precision bf16

echo ""
echo "✓ Example 2 complete"
echo ""

# Example 3: Extract last layer (layer 27) with FP32 for maximum compatibility
echo "Example 3: Extracting last layer (layer 27) in FP32..."
python3 scripts/extract_transformer_block.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --layer-index 27 \
    --output-dir "$OUTPUT_DIR/layer_27_fp32" \
    --output-name transformer_block_27 \
    --precision fp32

echo ""
echo "✓ Example 3 complete"
echo ""

# Summary
echo "=========================================="
echo "Extraction Complete!"
echo "=========================================="
echo ""
echo "Extracted models saved to: $OUTPUT_DIR"
echo ""
echo "Contents:"
ls -lh "$OUTPUT_DIR"/*/*.xml 2>/dev/null || true
echo ""
echo "Next steps:"
echo "  1. Benchmark: benchmark_app -m $OUTPUT_DIR/layer_0/transformer_block_0.xml"
echo "  2. Profile: ONEDNN_VERBOSE=1 python your_inference_script.py"
echo "  3. Analyze: Review $OUTPUT_DIR/*/transformer_block_*.txt for specifications"
echo ""

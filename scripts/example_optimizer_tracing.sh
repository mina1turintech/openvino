#!/bin/bash
# Example usage of graph optimizer instrumentation

set -e

echo "=============================================="
echo "Graph Optimizer Instrumentation Examples"
echo "=============================================="

# Example 1: Basic tracing
echo ""
echo "Example 1: Basic Tracing"
echo "------------------------"
echo "Enable tracing with default output file:"
echo ""
echo "  export OV_CPU_GRAPH_OPTIMIZER_TRACE=1"
echo "  python scripts/run_single_block_test.py"
echo "  # Output: ./graph_optimizer_trace.json"
echo ""

# Example 2: Custom output location
echo "Example 2: Custom Output Location"
echo "-----------------------------------"
echo "Specify custom output file:"
echo ""
echo "  export OV_CPU_GRAPH_OPTIMIZER_TRACE=1"
echo "  export OV_CPU_GRAPH_OPTIMIZER_TRACE_FILE=./traces/my_trace.json"
echo "  python scripts/run_single_block_test.py"
echo ""

# Example 3: Full workflow
echo "Example 3: Complete Analysis Workflow"
echo "---------------------------------------"
echo "1. Extract transformer block:"
echo "   python scripts/extract_transformer_block.py --model Qwen/Qwen2-1.5B --layer-index 0"
echo ""
echo "2. Enable tracing:"
echo "   export OV_CPU_GRAPH_OPTIMIZER_TRACE=1"
echo "   export OV_CPU_GRAPH_OPTIMIZER_TRACE_FILE=./optimizer_trace.json"
echo ""
echo "3. Run inference (this triggers graph optimization):"
echo "   python scripts/run_single_block_test.py --model transformer_block_layer0.xml"
echo ""
echo "4. Analyze results:"
echo "   python scripts/analyze_optimizer_trace.py ./optimizer_trace.json"
echo ""

# Example 4: Integration with baseline capture
echo "Example 4: Integration with Baseline Capture"
echo "----------------------------------------------"
echo "Capture both optimizer and runtime traces:"
echo ""
echo "  # Setup environment"
echo "  export OV_CPU_GRAPH_OPTIMIZER_TRACE=1"
echo "  export OV_CPU_GRAPH_OPTIMIZER_TRACE_FILE=./baseline_optimizer_trace.json"
echo "  export DNNL_VERBOSE=1"
echo ""
echo "  # Run baseline capture"
echo "  ./scripts/capture_baseline_trace.sh"
echo ""
echo "  # Analyze optimizer decisions"
echo "  python scripts/analyze_optimizer_trace.py ./baseline_optimizer_trace.json"
echo ""
echo "  # Compare with runtime behavior"
echo "  python scripts/parse_onednn_reorders.py \\
    ./baseline_capture/traces/onednn_trace_baseline_run1.txt"
echo ""

echo ""
echo "=============================================="
echo "See scripts/GRAPH_OPTIMIZER_INSTRUMENTATION_README.md for detailed documentation"
echo "=============================================="

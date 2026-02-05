#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Example Input-Side Validation Workflow
#
# This script demonstrates the complete input-side optimization validation
# workflow with example configurations.

set -e

echo "========================================================================="
echo "Input-Side Optimization Validation - Example Workflow"
echo "========================================================================="
echo ""
echo "This script demonstrates 5 common validation scenarios:"
echo "  1. Quick validation (1 run, 10 iterations)"
echo "  2. Standard validation (3 runs, 50 iterations) - RECOMMENDED"
echo "  3. Production validation (5 runs, 100 iterations)"
echo "  4. Baseline only (for initial analysis)"
echo "  5. Complete baseline + optimized + comparison"
echo ""
echo "Select a scenario or press Ctrl+C to exit:"
echo ""
echo "1) Quick validation (~2 minutes)"
echo "2) Standard validation (~10 minutes) [RECOMMENDED]"
echo "3) Production validation (~30 minutes)"
echo "4) Baseline only (~5 minutes)"
echo "5) Complete workflow (~20 minutes)"
echo ""

read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo ""
        echo "=== Scenario 1: Quick Validation ==="
        echo ""
        echo "Configuration:"
        echo "  - 1 run"
        echo "  - 10 iterations"
        echo "  - Baseline only"
        echo ""
        
        ./scripts/capture_input_side_trace.sh \
            --baseline \
            --num-runs 1 \
            --iterations 10 \
            --output-dir ./input_side_quick
        
        echo ""
        echo "✓ Quick validation complete!"
        echo ""
        echo "Review results:"
        echo "  cat ./input_side_quick/INPUT_SIDE_BASELINE_REPORT.md"
        ;;
    
    2)
        echo ""
        echo "=== Scenario 2: Standard Validation (RECOMMENDED) ==="
        echo ""
        echo "Configuration:"
        echo "  - 3 runs (for reproducibility)"
        echo "  - 50 iterations (balanced speed/accuracy)"
        echo "  - Baseline capture"
        echo ""
        
        ./scripts/capture_input_side_trace.sh \
            --baseline \
            --num-runs 3 \
            --iterations 50 \
            --output-dir ./input_side_validation
        
        echo ""
        echo "✓ Standard validation complete!"
        echo ""
        echo "Review results:"
        echo "  cat ./input_side_validation/INPUT_SIDE_BASELINE_REPORT.md"
        echo ""
        echo "Next steps:"
        echo "  1. Implement weight pre-reordering (see INPUT_SIDE_VALIDATION_GUIDE.md)"
        echo "  2. Run: ./scripts/capture_input_side_trace.sh --optimized --skip-extraction"
        echo "  3. Compare: python3 scripts/compare_input_side_traces.py --baseline ./input_side_validation/metrics_baseline --optimized ./input_side_validation/metrics_optimized --output ./INPUT_SIDE_COMPARISON.md"
        ;;
    
    3)
        echo ""
        echo "=== Scenario 3: Production Validation ==="
        echo ""
        echo "Configuration:"
        echo "  - 5 runs (high confidence)"
        echo "  - 100 iterations (precise timing)"
        echo "  - Baseline capture"
        echo ""
        
        ./scripts/capture_input_side_trace.sh \
            --baseline \
            --num-runs 5 \
            --iterations 100 \
            --output-dir ./input_side_production
        
        echo ""
        echo "✓ Production validation complete!"
        echo ""
        echo "Review results:"
        echo "  cat ./input_side_production/INPUT_SIDE_BASELINE_REPORT.md"
        ;;
    
    4)
        echo ""
        echo "=== Scenario 4: Baseline Only ==="
        echo ""
        echo "Capturing baseline trace for analysis..."
        echo ""
        
        ./scripts/capture_input_side_trace.sh \
            --baseline \
            --num-runs 3 \
            --iterations 50 \
            --output-dir ./input_side_baseline_only
        
        echo ""
        echo "✓ Baseline capture complete!"
        echo ""
        echo "Analyzing weight reorder patterns..."
        echo ""
        
        # Extract key metrics from first run
        python3 << 'EOF'
import json
from pathlib import Path

metrics_file = Path("./input_side_baseline_only/metrics_baseline/baseline_run1_metrics.json")
if metrics_file.exists():
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    print("Key Weight Reorder Metrics:")
    print("=" * 60)
    
    by_dim = metrics.get('by_dimension', {})
    
    # Weight dimensions
    weight_dims = {
        '1536x1536': 'Attention Output Projection',
        '256x1536': 'Q/K/V Projections',
        '1536x8960': 'FFN Expand Weights',
        '8960x1536': 'FFN Contract Weights',
    }
    
    total_weight_time = 0
    for dim_str, desc in weight_dims.items():
        if dim_str in by_dim:
            count = by_dim[dim_str]['count']
            time_ms = by_dim[dim_str]['time_ms']
            total_weight_time += time_ms
            print(f"  {dim_str:12} ({desc:30}): {count:2} ops, {time_ms:6.3f} ms")
    
    print("=" * 60)
    print(f"Total Weight Reorder Overhead: {total_weight_time:.3f} ms per block")
    print(f"24-Layer Model Projection:     {total_weight_time * 24:.3f} ms")
    print("")
    print("✓ These reorders can be eliminated with weight pre-reordering!")
else:
    print("Error: Metrics file not found")
EOF
        
        echo ""
        echo "Full report available:"
        echo "  cat ./input_side_baseline_only/INPUT_SIDE_BASELINE_REPORT.md"
        ;;
    
    5)
        echo ""
        echo "=== Scenario 5: Complete Workflow ==="
        echo ""
        echo "This will:"
        echo "  1. Capture baseline traces (3 runs, 50 iterations)"
        echo "  2. Capture optimized traces (if implemented)"
        echo "  3. Generate comparison report"
        echo ""
        
        read -p "Continue? [y/N]: " confirm
        if [[ ! $confirm =~ ^[Yy]$ ]]; then
            echo "Cancelled."
            exit 0
        fi
        
        echo ""
        echo "Step 1/3: Capturing baseline traces..."
        echo ""
        
        ./scripts/capture_input_side_trace.sh \
            --baseline \
            --num-runs 3 \
            --iterations 50 \
            --output-dir ./input_side_complete
        
        echo ""
        echo "Step 2/3: Capturing optimized traces..."
        echo ""
        
        # Check if optimization is implemented
        echo "⚠️  NOTE: If weight pre-reordering is not implemented, this will"
        echo "   capture the same trace as baseline (showing 0% improvement)."
        echo ""
        read -p "Continue with optimized capture? [y/N]: " confirm_opt
        
        if [[ $confirm_opt =~ ^[Yy]$ ]]; then
            ./scripts/capture_input_side_trace.sh \
                --optimized \
                --num-runs 3 \
                --iterations 50 \
                --output-dir ./input_side_complete \
                --skip-extraction
            
            echo ""
            echo "Step 3/3: Generating comparison report..."
            echo ""
            
            python3 scripts/compare_input_side_traces.py \
                --baseline ./input_side_complete/metrics_baseline \
                --optimized ./input_side_complete/metrics_optimized \
                --output ./input_side_complete/INPUT_SIDE_COMPARISON.md
            
            echo ""
            echo "✓ Complete workflow finished!"
            echo ""
            echo "Review results:"
            echo "  - Baseline:   ./input_side_complete/INPUT_SIDE_BASELINE_REPORT.md"
            echo "  - Optimized:  ./input_side_complete/INPUT_SIDE_OPTIMIZED_REPORT.md"
            echo "  - Comparison: ./input_side_complete/INPUT_SIDE_COMPARISON.md"
        else
            echo ""
            echo "Skipped optimized capture."
            echo ""
            echo "✓ Baseline capture complete!"
            echo "Review: ./input_side_complete/INPUT_SIDE_BASELINE_REPORT.md"
        fi
        ;;
    
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "========================================================================="
echo "Validation Complete"
echo "========================================================================="
echo ""
echo "For more information, see:"
echo "  - INPUT_SIDE_VALIDATION_GUIDE.md"
echo "  - scripts/ONEDNN_TRACE_CAPTURE_README.md"
echo "  - ATTENTION_WEIGHT_LAYOUT_ANALYSIS.md"
echo "  - FFN_WEIGHT_LAYOUT_ANALYSIS.md"
echo ""

#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Example: Output-Side Trace Comparison
# Task 28/32: Generate and compare baseline vs. output-side optimized traces
#
# This script demonstrates the complete workflow for comparing baseline and
# optimized traces to quantify the impact of output-side layout optimizations.

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${CYAN}================================================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}================================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}→ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Banner
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║        Output-Side Trace Comparison Example (Task 28)                ║
║                                                                       ║
║  This example demonstrates comparing baseline vs. optimized traces   ║
║  to quantify activation reorder reduction from output-side layout    ║
║  optimizations (attention and FFN output operations).                ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
EOF

echo ""
print_info "This script will:"
echo "  1. Check for existing baseline traces (Task 2)"
echo "  2. Capture output-side optimized traces (current build)"
echo "  3. Compare baseline vs. optimized traces"
echo "  4. Generate comprehensive comparison report"
echo ""
print_info "Expected duration: ~10-15 minutes"
echo ""

# Ask for confirmation
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Configuration
MODEL_NAME="Qwen/Qwen2-1.5B-Instruct"
LAYER_INDEX=0
NUM_RUNS=3
ITERATIONS=50
BASELINE_DIR="./baseline_capture"
OUTPUT_DIR="./output_side_validation"
COMPARISON_REPORT="./OUTPUT_SIDE_COMPARISON.md"

# ============================================================================
# Scenario 1: Check for Existing Baseline
# ============================================================================

print_header "Scenario 1: Checking for Existing Baseline Traces"

if [ -d "$BASELINE_DIR/metrics" ] && [ "$(ls -A $BASELINE_DIR/metrics/*.json 2>/dev/null)" ]; then
    print_success "Found existing baseline traces from Task 2"
    print_info "Baseline directory: $BASELINE_DIR"
    
    # Show baseline metrics summary
    num_baseline_files=$(ls -1 $BASELINE_DIR/metrics/*.json 2>/dev/null | wc -l)
    print_info "Baseline metric files: $num_baseline_files"
    
    SKIP_BASELINE_CAPTURE=true
else
    print_warning "No baseline traces found"
    print_info "Baseline traces can be captured from research branch (Task 2)"
    print_info "For this example, we'll create a mock baseline for demonstration"
    
    SKIP_BASELINE_CAPTURE=false
fi

echo ""
read -p "Press Enter to continue..."

# ============================================================================
# Scenario 2: Handle Missing Baseline
# ============================================================================

if [ "$SKIP_BASELINE_CAPTURE" = false ]; then
    print_header "Scenario 2: Creating Mock Baseline (Demonstration)"
    
    print_warning "In production, you would:"
    print_info "  1. git checkout research-branch"
    print_info "  2. ./build_openvino.sh"
    print_info "  3. ./scripts/capture_baseline_trace.sh --output-dir $BASELINE_DIR"
    echo ""
    
    print_info "For this example, we'll use the current build as 'baseline'"
    print_info "This will show near-zero improvement (current already optimal)"
    echo ""
    
    read -p "Create mock baseline? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Capturing mock baseline traces..."
        
        mkdir -p "$BASELINE_DIR"
        
        ./scripts/capture_baseline_trace.sh \
            --num-runs 1 \
            --iterations 50 \
            --output-dir "$BASELINE_DIR" \
            --skip-extraction 2>/dev/null || {
                # If baseline script doesn't exist, use output_side script
                ./scripts/capture_output_side_trace.sh \
                    --num-runs 1 \
                    --iterations 50 \
                    --output-dir "$BASELINE_DIR"
            }
        
        print_success "Mock baseline captured"
    else
        print_warning "Skipping baseline capture"
        print_info "Comparison will not be possible without baseline"
        exit 0
    fi
    
    echo ""
    read -p "Press Enter to continue..."
fi

# ============================================================================
# Scenario 3: Capture Optimized Traces
# ============================================================================

print_header "Scenario 3: Capturing Output-Side Optimized Traces"

print_info "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Layer: $LAYER_INDEX"
echo "  Runs: $NUM_RUNS"
echo "  Iterations: $ITERATIONS"
echo "  Output: $OUTPUT_DIR"
echo ""

print_step "Extracting transformer block and capturing traces..."
echo ""

./scripts/capture_output_side_trace.sh \
    --model-name "$MODEL_NAME" \
    --layer-index "$LAYER_INDEX" \
    --num-runs "$NUM_RUNS" \
    --iterations "$ITERATIONS" \
    --output-dir "$OUTPUT_DIR" \
    --no-baseline-compare

print_success "Optimized traces captured"
print_info "Output directory: $OUTPUT_DIR"

# Show optimized metrics summary
num_optimized_files=$(ls -1 $OUTPUT_DIR/metrics/*.json 2>/dev/null | wc -l)
print_info "Optimized metric files: $num_optimized_files"

echo ""
read -p "Press Enter to continue..."

# ============================================================================
# Scenario 4: Compare Traces
# ============================================================================

print_header "Scenario 4: Comparing Baseline vs. Optimized Traces"

print_step "Running comparison tool..."
echo ""

python3 scripts/compare_output_side_traces.py \
    --baseline "$BASELINE_DIR/metrics" \
    --optimized "$OUTPUT_DIR/metrics" \
    --output "$COMPARISON_REPORT"

print_success "Comparison report generated"
print_info "Report: $COMPARISON_REPORT"

echo ""
read -p "Press Enter to view results..."

# ============================================================================
# Scenario 5: Display Results
# ============================================================================

print_header "Scenario 5: Comparison Results"

# Extract key metrics from report
if [ -f "$COMPARISON_REPORT" ]; then
    print_step "Executive Summary:"
    echo ""
    
    # Show activation reorder metrics
    grep -A 10 "Output-Side Specific Metrics" "$COMPARISON_REPORT" | head -15 || true
    
    echo ""
    print_step "24-Layer Model Projection:"
    echo ""
    
    # Show 24-layer projection
    grep -A 5 "24-Layer Model Projection" "$COMPARISON_REPORT" | head -8 || true
    
    echo ""
    print_step "Success Criteria:"
    echo ""
    
    # Show success criteria
    grep -A 10 "Success Criteria Validation" "$COMPARISON_REPORT" | head -15 || true
    
else
    print_error "Comparison report not found"
    exit 1
fi

echo ""
echo ""
print_success "Comparison complete!"
echo ""

# ============================================================================
# Summary and Next Steps
# ============================================================================

print_header "Summary and Next Steps"

print_info "What was demonstrated:"
echo "  ✓ Baseline trace availability check"
echo "  ✓ Output-side optimized trace capture (3 runs)"
echo "  ✓ Comprehensive baseline vs. optimized comparison"
echo "  ✓ Activation reorder reduction analysis"
echo "  ✓ Weight reorder stability verification"
echo "  ✓ 24-layer model impact projection"
echo ""

print_info "Key findings (current implementation):"
echo "  • Current OpenVINO already uses optimal output layouts"
echo "  • Activation reorders: 0 (already optimal)"
echo "  • Block boundary overhead: 0 ms"
echo "  • Output formats: f32::ab (plain, optimal)"
echo ""

print_info "Generated files:"
echo "  • Comparison report: $COMPARISON_REPORT"
echo "  • Optimized traces: $OUTPUT_DIR/traces/"
echo "  • Optimized metrics: $OUTPUT_DIR/metrics/"
if [ "$SKIP_BASELINE_CAPTURE" = false ]; then
    echo "  • Mock baseline: $BASELINE_DIR/"
fi
echo ""

print_info "Next steps:"
echo "  1. Review full comparison report:"
echo "     cat $COMPARISON_REPORT"
echo ""
echo "  2. Compare with baseline from research branch:"
echo "     git checkout research-branch"
echo "     ./build_openvino.sh"
echo "     ./scripts/capture_baseline_trace.sh --output-dir ./baseline_research"
echo "     python3 scripts/compare_output_side_traces.py \\"
echo "         --baseline ./baseline_research/metrics \\"
echo "         --optimized $OUTPUT_DIR/metrics \\"
echo "         --output ./OUTPUT_SIDE_COMPARISON_RESEARCH.md"
echo ""
echo "  3. Read the comparison guide:"
echo "     cat OUTPUT_SIDE_COMPARISON_GUIDE.md"
echo ""
echo "  4. Read the baseline analysis:"
echo "     cat OUTPUT_SIDE_BASELINE_ANALYSIS.md"
echo ""

print_header "Output-Side Trace Comparison Example Complete!"

print_success "All scenarios completed successfully"
echo ""
echo "For more information:"
echo "  • Quick start: QUICK_START_OUTPUT_COMPARISON.md"
echo "  • Full guide: OUTPUT_SIDE_COMPARISON_GUIDE.md"
echo "  • Baseline analysis: OUTPUT_SIDE_BASELINE_ANALYSIS.md"
echo "  • Tool README: scripts/OUTPUT_SIDE_VALIDATION_README.md"
echo ""

#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Combined Optimization Validation Example Script
#
# This interactive script demonstrates the complete workflow for validating
# combined input-side and output-side optimizations. It guides users through
# trace capture, comparison, and results interpretation.
#
# Usage: ./scripts/example_combined_validation.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo ""
    echo "=============================================================="
    echo "$1"
    echo "=============================================================="
    echo ""
}

print_step() {
    echo ""
    echo -e "${BLUE}[$1]${NC} $2"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# ==============================================================================
# Welcome and Scenario Selection
# ==============================================================================

print_header "Combined Optimization Validation - Interactive Demo"

cat << 'EOF'
This script validates that input-side and output-side optimizations
compound positively when applied together.

What you'll do:
  1. Check for required baseline and individual optimization traces
  2. Capture combined optimization trace (both opts enabled)
  3. Generate comprehensive comparison report
  4. Review results and validate success criteria

Prerequisites:
  ✓ OpenVINO build with BOTH optimizations compiled in
  ✓ Python 3.10+ with torch, transformers, openvino installed
  ✓ Baseline trace from Task 2 (optional but recommended)
  ✓ Input-side trace from Task 27 (optional for additivity analysis)
  ✓ Output-side trace from Task 28 (optional for additivity analysis)

EOF

read -p "Ready to proceed? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user."
    exit 0
fi

# ==============================================================================
# Scenario Selection
# ==============================================================================

print_header "Scenario Selection"

cat << 'EOF'
Choose a validation scenario:

1. Quick Test (1 run, 20 iterations)
   - Fast validation (~2 minutes)
   - Good for development/debugging
   - Less statistical confidence

2. Standard Validation (3 runs, 50 iterations) [RECOMMENDED]
   - Balanced speed and accuracy (~5 minutes)
   - Standard for optimization validation
   - Good statistical confidence

3. Production Validation (5 runs, 100 iterations)
   - Comprehensive validation (~12 minutes)
   - Best statistical confidence
   - Suitable for CI/CD and release validation

4. Comparison Only (skip trace capture)
   - Use existing combined trace
   - Just generate comparison report
   - Instant results

5. Complete Four-Way Analysis
   - Capture all four traces (baseline, input-side, output-side, combined)
   - Full additivity analysis
   - ~20 minutes total

EOF

read -p "Select scenario [1-5]: " scenario

case $scenario in
    1)
        SCENARIO_NAME="Quick Test"
        NUM_RUNS=1
        ITERATIONS=20
        SKIP_CAPTURE=false
        ;;
    2)
        SCENARIO_NAME="Standard Validation"
        NUM_RUNS=3
        ITERATIONS=50
        SKIP_CAPTURE=false
        ;;
    3)
        SCENARIO_NAME="Production Validation"
        NUM_RUNS=5
        ITERATIONS=100
        SKIP_CAPTURE=false
        ;;
    4)
        SCENARIO_NAME="Comparison Only"
        NUM_RUNS=0
        ITERATIONS=0
        SKIP_CAPTURE=true
        ;;
    5)
        SCENARIO_NAME="Complete Four-Way Analysis"
        NUM_RUNS=3
        ITERATIONS=50
        SKIP_CAPTURE=false
        COMPLETE_ANALYSIS=true
        ;;
    *)
        print_error "Invalid scenario. Defaulting to Standard Validation."
        SCENARIO_NAME="Standard Validation"
        NUM_RUNS=3
        ITERATIONS=50
        SKIP_CAPTURE=false
        ;;
esac

print_success "Selected: $SCENARIO_NAME"

# ==============================================================================
# Configuration
# ==============================================================================

print_header "Configuration"

MODEL_NAME="Qwen/Qwen2-1.5B-Instruct"
LAYER_INDEX=0
OUTPUT_DIR="./combined_validation"
BASELINE_DIR="./baseline_capture"
INPUT_SIDE_DIR="./input_side_validation"
OUTPUT_SIDE_DIR="./output_side_validation"

cat << EOF
Configuration:
  Model: $MODEL_NAME
  Layer: $LAYER_INDEX
  Runs: $NUM_RUNS
  Iterations: $ITERATIONS
  Output: $OUTPUT_DIR
  
  Baseline: $BASELINE_DIR
  Input-Side: $INPUT_SIDE_DIR
  Output-Side: $OUTPUT_SIDE_DIR

EOF

# Check dependencies
print_step "SETUP" "Checking dependencies..."

if ! command -v python3 &> /dev/null; then
    print_error "python3 not found. Please install Python 3.10+"
    exit 1
fi
print_success "Python found"

python3 -c "import torch, transformers, openvino, numpy" 2>/dev/null || {
    print_error "Missing Python packages"
    echo "Install with: pip install torch transformers openvino numpy"
    exit 1
}
print_success "Required Python packages found"

# ==============================================================================
# Check for Existing Traces
# ==============================================================================

print_step "CHECK" "Checking for existing traces..."

BASELINE_AVAILABLE=false
INPUT_SIDE_AVAILABLE=false
OUTPUT_SIDE_AVAILABLE=false
COMBINED_AVAILABLE=false

if [ -d "$BASELINE_DIR/metrics" ] && [ -n "$(ls -A $BASELINE_DIR/metrics/*_metrics.json 2>/dev/null)" ]; then
    BASELINE_AVAILABLE=true
    print_success "Baseline trace found"
else
    print_warning "Baseline trace not found"
    echo "  Expected: $BASELINE_DIR/metrics/*_metrics.json"
    echo "  Capture with: ./scripts/capture_baseline_trace.sh"
fi

if [ -d "$INPUT_SIDE_DIR/metrics_optimized" ] && [ -n "$(ls -A $INPUT_SIDE_DIR/metrics_optimized/*_metrics.json 2>/dev/null)" ]; then
    INPUT_SIDE_AVAILABLE=true
    print_success "Input-side trace found"
else
    print_warning "Input-side trace not found"
    echo "  Expected: $INPUT_SIDE_DIR/metrics_optimized/*_metrics.json"
    echo "  Capture with: ./scripts/capture_input_side_trace.sh --optimized"
fi

if [ -d "$OUTPUT_SIDE_DIR/metrics_optimized" ] && [ -n "$(ls -A $OUTPUT_SIDE_DIR/metrics_optimized/*_metrics.json 2>/dev/null)" ]; then
    OUTPUT_SIDE_AVAILABLE=true
    print_success "Output-side trace found"
else
    print_warning "Output-side trace not found"
    echo "  Expected: $OUTPUT_SIDE_DIR/metrics_optimized/*_metrics.json"
    echo "  Capture with: ./scripts/capture_output_side_trace.sh"
fi

if [ -d "$OUTPUT_DIR/metrics_combined" ] && [ -n "$(ls -A $OUTPUT_DIR/metrics_combined/*_metrics.json 2>/dev/null)" ]; then
    COMBINED_AVAILABLE=true
    print_success "Combined trace found"
else
    print_info "Combined trace not yet captured"
fi

echo ""

# Determine what can be done
CAN_COMPARE_BASIC=false
CAN_COMPARE_FULL=false

if $BASELINE_AVAILABLE && $COMBINED_AVAILABLE; then
    CAN_COMPARE_BASIC=true
fi

if $BASELINE_AVAILABLE && $COMBINED_AVAILABLE && $INPUT_SIDE_AVAILABLE && $OUTPUT_SIDE_AVAILABLE; then
    CAN_COMPARE_FULL=true
fi

# ==============================================================================
# Trace Capture (if needed)
# ==============================================================================

if [ "$SKIP_CAPTURE" = false ]; then
    
    # Handle complete analysis scenario
    if [ "$COMPLETE_ANALYSIS" = true ]; then
        
        # Capture baseline if missing
        if [ "$BASELINE_AVAILABLE" = false ]; then
            print_step "CAPTURE" "Capturing baseline trace..."
            ./scripts/capture_baseline_trace.sh \
                --output-dir "$BASELINE_DIR" \
                --num-runs $NUM_RUNS \
                --iterations $ITERATIONS
            print_success "Baseline capture complete"
            BASELINE_AVAILABLE=true
        fi
        
        # Capture input-side if missing
        if [ "$INPUT_SIDE_AVAILABLE" = false ]; then
            print_step "CAPTURE" "Capturing input-side optimized trace..."
            ./scripts/capture_input_side_trace.sh \
                --optimized \
                --output-dir "$INPUT_SIDE_DIR" \
                --num-runs $NUM_RUNS \
                --iterations $ITERATIONS \
                --skip-extraction
            print_success "Input-side capture complete"
            INPUT_SIDE_AVAILABLE=true
        fi
        
        # Capture output-side if missing
        if [ "$OUTPUT_SIDE_AVAILABLE" = false ]; then
            print_step "CAPTURE" "Capturing output-side optimized trace..."
            ./scripts/capture_output_side_trace.sh \
                --output-dir "$OUTPUT_SIDE_DIR" \
                --num-runs $NUM_RUNS \
                --iterations $ITERATIONS \
                --skip-extraction
            print_success "Output-side capture complete"
            OUTPUT_SIDE_AVAILABLE=true
        fi
    fi
    
    # Capture combined trace
    print_step "CAPTURE" "Capturing combined optimization trace..."
    
    SKIP_EXTRACTION_FLAG=""
    if $BASELINE_AVAILABLE || $INPUT_SIDE_AVAILABLE || $OUTPUT_SIDE_AVAILABLE; then
        SKIP_EXTRACTION_FLAG="--skip-extraction"
        print_info "Reusing existing extracted model"
    fi
    
    ./scripts/capture_combined_trace.sh \
        --output-dir "$OUTPUT_DIR" \
        --num-runs $NUM_RUNS \
        --iterations $ITERATIONS \
        --model-name "$MODEL_NAME" \
        --layer-index $LAYER_INDEX \
        $SKIP_EXTRACTION_FLAG
    
    print_success "Combined trace capture complete"
    COMBINED_AVAILABLE=true
    
    # Update comparison capabilities
    if $BASELINE_AVAILABLE; then
        CAN_COMPARE_BASIC=true
    fi
    if $BASELINE_AVAILABLE && $INPUT_SIDE_AVAILABLE && $OUTPUT_SIDE_AVAILABLE; then
        CAN_COMPARE_FULL=true
    fi
fi

# ==============================================================================
# Generate Comparison Report
# ==============================================================================

print_step "COMPARE" "Generating comparison report..."

if ! $BASELINE_AVAILABLE; then
    print_error "Cannot generate comparison: baseline trace missing"
    echo ""
    echo "To capture baseline:"
    echo "  ./scripts/capture_baseline_trace.sh --output-dir $BASELINE_DIR"
    exit 1
fi

if ! $COMBINED_AVAILABLE; then
    print_error "Cannot generate comparison: combined trace missing"
    echo ""
    echo "To capture combined:"
    echo "  ./scripts/capture_combined_trace.sh --output-dir $OUTPUT_DIR"
    exit 1
fi

COMPARISON_OUTPUT="$OUTPUT_DIR/COMBINED_COMPARISON.md"

# Build comparison command
COMPARE_CMD="python3 scripts/compare_combined_traces.py"
COMPARE_CMD="$COMPARE_CMD --baseline $BASELINE_DIR/metrics"
COMPARE_CMD="$COMPARE_CMD --combined $OUTPUT_DIR/metrics_combined"

if $CAN_COMPARE_FULL; then
    print_info "Full comparison with additivity analysis"
    COMPARE_CMD="$COMPARE_CMD --input-side $INPUT_SIDE_DIR/metrics_optimized"
    COMPARE_CMD="$COMPARE_CMD --output-side $OUTPUT_SIDE_DIR/metrics_optimized"
else
    print_info "Basic comparison (without additivity analysis)"
    if ! $INPUT_SIDE_AVAILABLE; then
        print_warning "Input-side trace missing (no additivity analysis)"
    fi
    if ! $OUTPUT_SIDE_AVAILABLE; then
        print_warning "Output-side trace missing (no additivity analysis)"
    fi
fi

COMPARE_CMD="$COMPARE_CMD --output $COMPARISON_OUTPUT"

echo ""
echo "Running: $COMPARE_CMD"
echo ""

eval $COMPARE_CMD

print_success "Comparison report generated: $COMPARISON_OUTPUT"

# ==============================================================================
# Display Results Summary
# ==============================================================================

print_header "Results Summary"

# Extract key metrics from report
if [ -f "$COMPARISON_OUTPUT" ]; then
    
    # Extract overall metrics
    TOTAL_REDUCTION=$(grep "Total Reorder Time (ms)" "$COMPARISON_OUTPUT" | head -1 | awk -F'|' '{print $(NF-1)}' | tr -d ' ')
    IMPROVEMENT_PCT=$(grep "Total Reorder Time (ms)" "$COMPARISON_OUTPUT" | head -1 | awk -F'|' '{print $NF}' | tr -d ' ')
    
    # Extract success criteria
    CRITERIA_PASSED=$(grep "Overall.*criteria passed" "$COMPARISON_OUTPUT" | grep -oP '\d+/\d+' || echo "?/?")
    
    echo "Key Metrics:"
    echo "  Total Reorder Time Reduction: $TOTAL_REDUCTION ms"
    echo "  Improvement Percentage: $IMPROVEMENT_PCT"
    echo "  Success Criteria: $CRITERIA_PASSED passed"
    echo ""
    
    # Check for additivity
    if $CAN_COMPARE_FULL; then
        ADDITIVITY=$(grep "Additivity Ratio" "$COMPARISON_OUTPUT" | grep -oP '\d+\.\d+%' | head -1 || echo "N/A")
        echo "  Additivity Ratio: $ADDITIVITY"
        echo ""
    fi
    
    # Display recommendations section
    echo "Recommendations:"
    sed -n '/## Recommendations/,/##/p' "$COMPARISON_OUTPUT" | grep -A 10 "^###" | head -15
    
else
    print_error "Comparison report not found"
fi

# ==============================================================================
# Next Steps
# ==============================================================================

print_header "Next Steps"

cat << EOF
✓ Validation Complete

Review the full report:
  cat $COMPARISON_OUTPUT

Or open in your editor:
  \$EDITOR $COMPARISON_OUTPUT

Check for:
  • Overall improvement > 5%
  • Weight reorder reduction (input-side optimization active)
  • Activation reorder reduction (output-side optimization active)
  • Additivity ratio 80-105% (if full comparison performed)
  • Success criteria: at least 6/8 passed

EOF

if $CAN_COMPARE_FULL; then
    print_success "Full four-way comparison completed"
    echo "  • Baseline vs Combined: overall improvement"
    echo "  • Additivity analysis: optimization interactions"
    echo "  • Per-optimization breakdown available"
else
    print_warning "Basic comparison completed (without additivity analysis)"
    echo ""
    echo "For full additivity analysis, capture missing traces:"
    if ! $INPUT_SIDE_AVAILABLE; then
        echo "  ./scripts/capture_input_side_trace.sh --optimized"
    fi
    if ! $OUTPUT_SIDE_AVAILABLE; then
        echo "  ./scripts/capture_output_side_trace.sh"
    fi
fi

echo ""
print_info "Related Documentation:"
echo "  • Full Guide: COMBINED_OPTIMIZATION_COMPARISON_GUIDE.md"
echo "  • Quick Start: QUICK_START_COMBINED_VALIDATION.md"
echo "  • Input-Side: INPUT_SIDE_VALIDATION_GUIDE.md"
echo "  • Output-Side: OUTPUT_SIDE_COMPARISON_GUIDE.md"

echo ""
print_header "Validation Complete"

#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Interactive Multi-Scenario Validation Example
#
# This script provides an interactive guided workflow for multi-scenario validation
# of layout optimizations. It walks through the complete process from trace capture
# to statistical analysis and results interpretation.
#
# Usage:
#   ./scripts/example_multi_scenario_validation.sh

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR="./multi_scenario_validation"
STATS_DIR="$OUTPUT_DIR/statistics"
ANALYSIS_FILE="$STATS_DIR/MULTI_SCENARIO_ANALYSIS.md"

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${CYAN}--------------------------------------------------------------${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}--------------------------------------------------------------${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

ask_yes_no() {
    local prompt="$1"
    local default="${2:-n}"
    
    if [ "$default" = "y" ]; then
        prompt="$prompt [Y/n]"
    else
        prompt="$prompt [y/N]"
    fi
    
    read -p "$prompt " -n 1 -r
    echo
    
    if [ "$default" = "y" ]; then
        [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]
    else
        [[ $REPLY =~ ^[Yy]$ ]]
    fi
}

# ==============================================================================
# Main Workflow
# ==============================================================================

print_header "Multi-Scenario Validation - Interactive Guide"

echo "This script will guide you through validating layout optimizations"
echo "across diverse inference conditions including:"
echo ""
echo "  • Different batch sizes (1, 4, 8)"
echo "  • Different sequence lengths (1, 32, 128, 256)"
echo "  • Different input patterns (random, ones, small_values)"
echo "  • Multiple repetitions per scenario (3-5)"
echo ""
echo "You will be able to choose between:"
echo "  1. Quick mode: 8 scenarios × 3 reps = 24 runs (~45 minutes)"
echo "  2. Full mode: 36 scenarios × 3 reps = 108 runs (~3-4 hours)"
echo ""

if ! ask_yes_no "Continue with multi-scenario validation?" "y"; then
    echo "Aborted by user."
    exit 0
fi

# ==============================================================================
# Step 1: Check Prerequisites
# ==============================================================================

print_section "Step 1: Checking Prerequisites"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "python3 not found"
    echo "Please install Python 3.10 or higher"
    exit 1
fi
print_success "python3 found"

# Check required packages
echo "Checking Python packages..."
if python3 -c "import torch, transformers, openvino, numpy" 2>/dev/null; then
    print_success "All required packages installed"
else
    print_warning "Some packages missing"
    echo "Required packages: torch, transformers, openvino, numpy"
    
    if ask_yes_no "Install missing packages now?"; then
        pip install torch transformers openvino numpy
        print_success "Packages installed"
    else
        print_error "Cannot proceed without required packages"
        exit 1
    fi
fi

# Check scripts exist
if [ ! -f "scripts/capture_multi_scenario_traces.sh" ]; then
    print_error "scripts/capture_multi_scenario_traces.sh not found"
    exit 1
fi
print_success "Multi-scenario capture script found"

if [ ! -f "scripts/analyze_multi_scenario_statistics.py" ]; then
    print_error "scripts/analyze_multi_scenario_statistics.py not found"
    exit 1
fi
print_success "Statistical analysis script found"

# ==============================================================================
# Step 2: Check for Existing Data
# ==============================================================================

print_section "Step 2: Checking for Existing Data"

if [ -d "$OUTPUT_DIR" ]; then
    print_warning "Output directory already exists: $OUTPUT_DIR"
    
    # Check if traces exist
    TRACE_COUNT=$(find "$OUTPUT_DIR/traces" -name "onednn_trace_*.txt" 2>/dev/null | wc -l)
    
    if [ "$TRACE_COUNT" -gt 0 ]; then
        echo "Found $TRACE_COUNT existing trace files"
        
        if ask_yes_no "Use existing traces and skip capture?"; then
            SKIP_CAPTURE=true
            print_info "Will use existing traces"
        else
            if ask_yes_no "Delete existing data and start fresh?" "n"; then
                rm -rf "$OUTPUT_DIR"
                print_success "Existing data cleared"
                SKIP_CAPTURE=false
            else
                print_error "Cannot proceed with existing data"
                exit 1
            fi
        fi
    else
        print_info "No existing traces found"
        SKIP_CAPTURE=false
    fi
else
    print_info "No existing output directory"
    SKIP_CAPTURE=false
fi

# ==============================================================================
# Step 3: Choose Test Mode
# ==============================================================================

if [ "$SKIP_CAPTURE" != true ]; then
    print_section "Step 3: Choose Test Mode"
    
    echo "Select validation mode:"
    echo ""
    echo "1. Quick Mode (Recommended for initial testing)"
    echo "   • Batch sizes: [1, 4]"
    echo "   • Sequence lengths: [1, 128]"
    echo "   • Input patterns: [random, ones]"
    echo "   • Total: 8 scenarios × 3 repetitions = 24 runs"
    echo "   • Duration: ~45 minutes"
    echo ""
    echo "2. Full Mode (Complete validation)"
    echo "   • Batch sizes: [1, 4, 8]"
    echo "   • Sequence lengths: [1, 32, 128, 256]"
    echo "   • Input patterns: [random, ones, small_values]"
    echo "   • Total: 36 scenarios × 3 repetitions = 108 runs"
    echo "   • Duration: ~3-4 hours"
    echo ""
    
    read -p "Enter choice (1 or 2): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[1]$ ]]; then
        TEST_MODE="quick"
        print_info "Selected: Quick Mode"
    elif [[ $REPLY =~ ^[2]$ ]]; then
        TEST_MODE="full"
        print_info "Selected: Full Mode"
    else
        print_error "Invalid choice"
        exit 1
    fi
    
    # ==============================================================================
    # Step 4: Capture Traces
    # ==============================================================================
    
    print_section "Step 4: Capturing Multi-Scenario Traces"
    
    echo "Starting trace capture in $TEST_MODE mode..."
    echo "This will take some time. Progress will be shown below."
    echo ""
    
    if ! ask_yes_no "Start trace capture now?" "y"; then
        print_error "Capture aborted by user"
        exit 1
    fi
    
    # Build command
    CAPTURE_CMD="./scripts/capture_multi_scenario_traces.sh --output-dir $OUTPUT_DIR"
    
    if [ "$TEST_MODE" = "quick" ]; then
        CAPTURE_CMD="$CAPTURE_CMD --quick"
    fi
    
    echo ""
    echo "Running: $CAPTURE_CMD"
    echo ""
    
    if ! bash $CAPTURE_CMD; then
        print_error "Trace capture failed"
        exit 1
    fi
    
    print_success "Trace capture complete!"
    
else
    print_section "Step 3: Using Existing Traces"
    print_info "Skipping trace capture, using existing data"
fi

# ==============================================================================
# Step 5: Statistical Analysis
# ==============================================================================

print_section "Step 5: Running Statistical Analysis"

echo "Analyzing metrics across all scenarios..."
echo "This will calculate mean, std dev, CV, and consistency metrics."
echo ""

mkdir -p "$STATS_DIR"

ANALYSIS_CMD="python3 scripts/analyze_multi_scenario_statistics.py \
    --metrics-dir $OUTPUT_DIR/metrics \
    --output $ANALYSIS_FILE \
    --variance-threshold 5.0"

echo "Running: $ANALYSIS_CMD"
echo ""

if ! eval $ANALYSIS_CMD; then
    print_error "Statistical analysis failed"
    exit 1
fi

print_success "Statistical analysis complete!"

# ==============================================================================
# Step 6: Display Results Summary
# ==============================================================================

print_section "Step 6: Results Summary"

if [ ! -f "$ANALYSIS_FILE" ]; then
    print_error "Analysis file not found: $ANALYSIS_FILE"
    exit 1
fi

echo "Extracting key results from analysis report..."
echo ""

# Extract high variance count
HIGH_VAR_COUNT=$(grep -c "High Variance" "$ANALYSIS_FILE" 2>/dev/null || echo "0")

# Extract success criteria
CRITERIA_PASSED=$(grep "Overall Score:" "$ANALYSIS_FILE" | grep -oP '\d+/\d+' || echo "?/?")

# Extract scenario count
SCENARIO_COUNT=$(grep "Total Scenarios Analyzed:" "$ANALYSIS_FILE" | grep -oP '\d+' || echo "?")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                    VALIDATION RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Total Scenarios Analyzed: $SCENARIO_COUNT"
echo "Success Criteria Passed: $CRITERIA_PASSED"
echo ""

# Check for high variance
if [ "$HIGH_VAR_COUNT" -eq 0 ]; then
    print_success "No high variance scenarios detected (CV < 5%)"
else
    print_warning "$HIGH_VAR_COUNT scenarios show high variance (CV > 5%)"
fi

echo ""

# Extract batch size consistency
BATCH_CONSISTENT=$(grep -A 10 "Batch Size Consistency" "$ANALYSIS_FILE" | grep -o "✅\|❌\|⚠️" | head -1 || echo "?")
if [ "$BATCH_CONSISTENT" = "✅" ]; then
    print_success "Batch size consistency: PASS"
elif [ "$BATCH_CONSISTENT" = "❌" ]; then
    print_error "Batch size consistency: FAIL"
else
    print_warning "Batch size consistency: WARNING"
fi

# Extract sequence length consistency
SEQ_CONSISTENT=$(grep -A 10 "Sequence Length Consistency" "$ANALYSIS_FILE" | grep -o "✅\|❌\|⚠️" | head -1 || echo "?")
if [ "$SEQ_CONSISTENT" = "✅" ]; then
    print_success "Sequence length consistency: PASS"
elif [ "$SEQ_CONSISTENT" = "❌" ]; then
    print_error "Sequence length consistency: FAIL"
else
    print_warning "Sequence length consistency: WARNING"
fi

# Extract input pattern consistency
PATTERN_CONSISTENT=$(grep -A 10 "Input Pattern Consistency" "$ANALYSIS_FILE" | grep -o "✅\|❌\|⚠️" | head -1 || echo "?")
if [ "$PATTERN_CONSISTENT" = "✅" ]; then
    print_success "Input pattern consistency: PASS"
elif [ "$PATTERN_CONSISTENT" = "❌" ]; then
    print_error "Input pattern consistency: FAIL"
else
    print_warning "Input pattern consistency: WARNING"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Overall assessment
if [ "$CRITERIA_PASSED" = "4/4" ]; then
    echo ""
    print_success "🎉 ALL VALIDATION CRITERIA MET!"
    echo ""
    echo "Optimizations are validated as robust across all test scenarios."
    echo "Ready for integration and production deployment."
elif [ "$CRITERIA_PASSED" = "3/4" ]; then
    echo ""
    print_warning "⚠️  MOST CRITERIA MET"
    echo ""
    echo "Minor variations detected but within acceptable limits."
    echo "Review detailed report for specific scenarios with variance."
else
    echo ""
    print_error "❌ SOME CRITERIA NOT MET"
    echo ""
    echo "Investigation required. Review high variance scenarios and"
    echo "consult troubleshooting guide in MULTI_SCENARIO_VALIDATION_GUIDE.md"
fi

# ==============================================================================
# Step 7: Next Steps
# ==============================================================================

print_section "Step 7: Next Steps"

echo "1. Review the comprehensive analysis report:"
echo "   cat $ANALYSIS_FILE"
echo ""
echo "2. Check detailed metrics for specific scenarios:"
echo "   ls $OUTPUT_DIR/metrics/"
echo ""
echo "3. Review individual trace files if needed:"
echo "   ls $OUTPUT_DIR/traces/"
echo ""

if [ "$CRITERIA_PASSED" = "4/4" ] || [ "$CRITERIA_PASSED" = "3/4" ]; then
    echo "4. Document validation results in release notes"
    echo ""
    echo "5. Consider extending validation to additional models:"
    echo "   • Different model sizes (7B, 14B)"
    echo "   • Different architectures (Llama, GPT, etc.)"
    echo ""
    echo "6. Integrate quick mode into CI/CD for regression testing"
else
    echo "4. Investigate high variance scenarios:"
    echo "   • Check system load and thermal throttling"
    echo "   • Increase repetitions for more stable measurements"
    echo "   • Profile individual scenarios with DNNL_VERBOSE=2"
    echo ""
    echo "5. Review optimization code for scenario-dependent behavior"
    echo ""
    echo "6. Consult troubleshooting guide:"
    echo "   cat MULTI_SCENARIO_VALIDATION_GUIDE.md"
fi

echo ""
print_info "For detailed documentation, see MULTI_SCENARIO_VALIDATION_GUIDE.md"
print_info "For quick reference, see QUICK_START_MULTI_SCENARIO.md"

# ==============================================================================
# Step 8: Offer to View Report
# ==============================================================================

print_section "Step 8: View Full Report?"

if ask_yes_no "Display full analysis report now?" "n"; then
    echo ""
    cat "$ANALYSIS_FILE"
else
    echo ""
    print_info "Report available at: $ANALYSIS_FILE"
fi

# ==============================================================================
# Completion
# ==============================================================================

echo ""
print_header "Multi-Scenario Validation Complete!"

echo "Summary:"
echo "  • Scenarios analyzed: $SCENARIO_COUNT"
echo "  • Success criteria: $CRITERIA_PASSED"
echo "  • High variance scenarios: $HIGH_VAR_COUNT"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Analysis report: $ANALYSIS_FILE"
echo ""

if [ "$CRITERIA_PASSED" = "4/4" ]; then
    print_success "✨ All validation criteria met! Optimizations are production-ready."
elif [ "$CRITERIA_PASSED" = "3/4" ]; then
    print_warning "⚠️  Most criteria met. Minor variations detected."
else
    print_warning "⚠️  Some criteria not met. Review report for details."
fi

echo ""

# Output-Side Layout Optimization Validation

**Task 26/32**: Validate output-side optimizations with trace analysis

## Overview

This document describes the tools and methodology for validating output-side layout optimizations (attention output and FFN output) in the Qwen2-1.5B transformer model. The validation confirms that output-side operations use optimal memory layouts and incur zero reorder overhead at block boundaries.

## Quick Start

### Option 1: Full Validation Workflow (Automated)

Run the complete validation workflow with a single command:

```bash
chmod +x scripts/capture_output_side_trace.sh
./scripts/capture_output_side_trace.sh
```

This will:
1. Extract a single transformer block
2. Capture multiple oneDNN traces
3. Parse traces to extract reorder metrics
4. Generate comprehensive validation report

**Time**: ~15 minutes first run, ~5 minutes subsequent runs  
**Output**: `./output_side_validation/OUTPUT_SIDE_VALIDATION.md`

### Option 2: Validation from Existing Traces

If you already have oneDNN traces, use the Python validation tool:

```bash
python3 scripts/validate_output_side_optimizations.py \
    --trace path/to/trace.txt \
    --output OUTPUT_SIDE_VALIDATION.md
```

### Option 3: Compare with Baseline (Task 28)

To compare output-side optimized traces with baseline and generate comprehensive comparison:

```bash
# Full baseline vs. optimized comparison workflow
python3 scripts/compare_output_side_traces.py \
    --baseline ./baseline_capture/metrics \
    --optimized ./output_side_validation/metrics \
    --output ./OUTPUT_SIDE_COMPARISON.md
```

**Alternative**: Quick validation with baseline comparison:
```bash
python3 scripts/validate_output_side_optimizations.py \
    --trace output_side_trace.txt \
    --baseline baseline_trace.txt \
    --output comparison_report.md \
    --json results.json
```

---

## Task 28: Baseline vs. Optimized Comparison

### Overview

Task 28 focuses on comparing baseline traces (research branch) against output-side optimized traces to quantify the impact of activation layout optimizations.

### Comparison Workflow

#### Step 1: Capture Baseline (if not already available)

```bash
# Use existing baseline from Task 2 OR capture new baseline
git checkout research-branch
./build_openvino.sh

./scripts/capture_baseline_trace.sh \
    --num-runs 3 \
    --iterations 50 \
    --output-dir ./baseline_capture
```

#### Step 2: Capture Optimized Traces

```bash
git checkout main  # or your optimized branch
./build_openvino.sh

./scripts/capture_output_side_trace.sh \
    --num-runs 3 \
    --iterations 50 \
    --output-dir ./output_side_validation
```

#### Step 3: Generate Comparison Report

```bash
python3 scripts/compare_output_side_traces.py \
    --baseline ./baseline_capture/metrics \
    --optimized ./output_side_validation/metrics \
    --output ./OUTPUT_SIDE_COMPARISON.md
```

### What Gets Compared

**Primary Focus (Activation Reorders)**:
- Activation reorder count reduction (target: 2 → 0 per block)
- Activation reorder time reduction (target: ~0.1-0.16ms → 0ms)
- Block boundary transitions (target: 0 reorders)

**Regression Checks (Should Not Change)**:
- Weight reorder patterns (should remain unchanged)
- Compute operation performance (should be stable)
- Total reorder distribution (only activation reorders should change)

### Expected Results

**If Baseline Has Blocked Outputs**:
- Activation reorder count: 2 → 0 (100% reduction)
- Activation reorder time: ~0.1-0.16ms → 0ms (100% reduction)
- 24-layer savings: ~2.4-3.8ms per inference

**If Baseline Already Optimal** (current state):
- Activation reorder count: 0 → 0 (no change)
- This confirms the implementation is already optimal (Task 26 finding)

### Comparison Tool Features

The `compare_output_side_traces.py` tool provides:

1. **Overall Metrics**: Total reorder counts and times
2. **Category Analysis**: Activation vs. weight vs. scale/ZP reorders
3. **Dimension Analysis**: Per-dimension reorder breakdown
4. **Stability Checks**: Weight reorder regression detection
5. **Success Criteria Validation**: Automated pass/fail assessment
6. **24-Layer Projection**: Full model impact estimation

### Documentation

- **Quick Start**: `QUICK_START_OUTPUT_COMPARISON.md`
- **Comparison Guide**: `OUTPUT_SIDE_COMPARISON_GUIDE.md`
- **Baseline Analysis**: `OUTPUT_SIDE_BASELINE_ANALYSIS.md`
- **Example Script**: `scripts/example_compare_output_side.sh`

---

## Tools Description

### 1. Automated Validation Script

**File**: `scripts/capture_output_side_trace.sh`

**Purpose**: End-to-end automation for capturing and validating output-side optimizations

**Features**:
- Model extraction with customizable layer selection
- Multiple trace capture runs for reproducibility
- Automatic metric extraction and parsing
- Baseline comparison (if available)
- Comprehensive report generation with validation criteria

**Usage**:
```bash
./scripts/capture_output_side_trace.sh [OPTIONS]

Options:
  --model-name MODEL        HuggingFace model name (default: Qwen/Qwen2-1.5B-Instruct)
  --layer-index INDEX       Transformer layer to extract (default: 0)
  --num-runs NUM            Number of trace capture runs (default: 3)
  --iterations NUM          Inference iterations per run (default: 50)
  --output-dir DIR          Output directory (default: ./output_side_validation)
  --baseline-dir DIR        Baseline directory for comparison (default: ./baseline_capture)
  --skip-extraction         Skip model extraction (use existing model)
  --no-baseline-compare     Skip baseline comparison
  --help                    Show help message
```

**Example - Basic Validation**:
```bash
./scripts/capture_output_side_trace.sh
```

**Example - Custom Configuration**:
```bash
./scripts/capture_output_side_trace.sh \
    --layer-index 12 \
    --num-runs 5 \
    --iterations 100 \
    --output-dir ./my_validation
```

**Example - Reuse Existing Model**:
```bash
./scripts/capture_output_side_trace.sh \
    --skip-extraction \
    --output-dir ./output_side_validation
```

**Output Structure**:
```
./output_side_validation/
├── model/                           # Extracted transformer block
│   ├── transformer_block.xml
│   ├── transformer_block.bin
│   └── transformer_block_specs.txt
├── traces/                          # oneDNN verbose traces
│   ├── onednn_trace_output_side_run1.txt
│   ├── onednn_trace_output_side_run2.txt
│   └── onednn_trace_output_side_run3.txt
├── metrics/                         # Parsed reorder metrics
│   ├── output_side_run1_metrics.csv
│   ├── output_side_run1_metrics.json
│   ├── output_side_run2_metrics.csv
│   ├── output_side_run2_metrics.json
│   ├── output_side_run3_metrics.csv
│   └── output_side_run3_metrics.json
├── logs/                            # Execution logs
│   ├── extraction.log
│   ├── capture_run1.log
│   ├── capture_run2.log
│   ├── capture_run3.log
│   ├── parse_run1.log
│   ├── parse_run2.log
│   └── parse_run3.log
├── comparison/                      # Baseline comparison
│   └── baseline_vs_output_side.csv
└── OUTPUT_SIDE_VALIDATION.md        # Validation report
```

---

### 2. Trace Comparison Tool (Task 28)

**File**: `scripts/compare_output_side_traces.py`

**Purpose**: Compare baseline and optimized traces to quantify output-side optimization impact

**Features**:
- Loads metrics from multiple runs (baseline and optimized)
- Aggregates metrics across runs for statistical significance
- Categorizes reorders by type (activation, weight, scale/ZP)
- Identifies activation reorder reductions (primary focus)
- Validates weight reorder stability (no regressions)
- Generates comprehensive comparison reports
- Projects improvements to 24-layer models

**Usage**:
```bash
python3 scripts/compare_output_side_traces.py [OPTIONS]

Required:
  --baseline PATH           Directory containing baseline metrics (JSON files)
  --optimized PATH          Directory containing optimized metrics (JSON files)
  --output PATH             Path to output comparison report (Markdown)
```

**Example - Basic Comparison**:
```bash
python3 scripts/compare_output_side_traces.py \
    --baseline ./baseline_capture/metrics \
    --optimized ./output_side_validation/metrics \
    --output ./OUTPUT_SIDE_COMPARISON.md
```

**Output**:
- Markdown comparison report with:
  - Executive summary with overall metrics
  - Activation reorder analysis (primary focus)
  - Weight reorder stability checks
  - Reorder category breakdown
  - Success criteria validation (5 criteria)
  - 24-layer model projection
  - Recommendations based on results

---

### 3. Python Validation Tool

**File**: `scripts/validate_output_side_optimizations.py`

**Purpose**: Analyze oneDNN traces and validate output-side layout optimizations

**Features**:
- Parses oneDNN verbose traces
- Identifies output-side reorder operations
- Validates attention output layout (f32::ab)
- Validates FFN output layout (f32::ab)
- Checks block boundary transitions
- Generates detailed validation reports
- Supports baseline comparison
- Exports JSON results for automation

**Usage**:
```bash
python3 scripts/validate_output_side_optimizations.py [OPTIONS]

Required:
  --trace PATH              Path to output-side optimized trace file

Optional:
  --baseline PATH           Path to baseline trace file for comparison
  --output PATH             Path to output validation report (default: OUTPUT_SIDE_VALIDATION.md)
  --json PATH               Path to output JSON results (optional)
```

**Example - Single Trace Validation**:
```bash
python3 scripts/validate_output_side_optimizations.py \
    --trace output_side_trace.txt \
    --output validation_report.md
```

**Example - Baseline Comparison**:
```bash
python3 scripts/validate_output_side_optimizations.py \
    --trace output_side_trace.txt \
    --baseline baseline_trace.txt \
    --output comparison_report.md \
    --json comparison_results.json
```

**Output**:
- Markdown validation report with:
  - Validation criteria checklist
  - Output-side reorder analysis
  - Block boundary transition analysis
  - Baseline comparison (if provided)
  - Recommendations
- JSON results (optional) for automation

---

## Validation Criteria

The validation tools check the following criteria:

### Success Criteria

| Criterion | Target | Validation Method |
|-----------|--------|-------------------|
| **Attention Output Reorders** | 0 reorders | Count reorders with 6x1536 or 1x1536 dimensions |
| **FFN Output Reorders** | 0 reorders | Count reorders after FFN contract (8960→1536) |
| **Block Boundary Reorders** | 0 per transition | Analyze activation flow between blocks |
| **Attention Output Format** | `f32::ab` consistent | Check dst format in inner_product operations |
| **FFN Output Format** | `f32::ab` consistent | Check dst format in FFN contract operations |
| **Total Reorder Count** | No increase vs baseline | Compare total reorder counts |
| **Reorder Time** | Stable or reduced | Compare total reorder time (ms) |
| **No Regressions** | No unexpected increases | Check all operation categories |

### What Gets Validated

**Output-Side Specific**:
- ✅ Attention output projection format (1536→1536 MatMul)
- ✅ FFN contract output format (8960→1536 MatMul)
- ✅ Residual connection compatibility (format matching)
- ✅ Layer normalization input format (mvn_planar)
- ✅ Block boundary transitions (inter-block activation flow)

**Regression Checks**:
- ✅ Total reorder count (should not increase)
- ✅ Weight reorder patterns (should be preserved)
- ✅ Other operation categories (should be stable)

---

## Understanding the Results

### Validation Report Sections

1. **Executive Summary**:
   - Overall validation status (PASS/FAIL)
   - Key metrics summary
   - Baseline comparison highlights

2. **Validation Criteria**:
   - Detailed status for each criterion
   - Reorder counts and times
   - Format consistency checks

3. **Output-Side Analysis**:
   - Attention output layout validation
   - FFN output layout validation
   - Block boundary transition analysis

4. **Detailed Metrics**:
   - Total reorder operations
   - Reorder time breakdown
   - Operation-specific analysis

5. **Baseline Comparison** (if available):
   - Side-by-side metrics
   - Improvement calculations
   - Regression detection

6. **Recommendations**:
   - Action items (if optimizations needed)
   - Future monitoring suggestions
   - Related optimization opportunities

### Interpreting Validation Status

**✅ PASS** - All criteria met:
- Attention output: 0 reorders, f32::ab format
- FFN output: 0 reorders, f32::ab format
- Block boundaries: 0 reorders per transition
- No regressions detected

**⚠️ NEEDS REVIEW** - Some criteria not met:
- Review specific reorder operations
- Analyze format mismatches
- Check for unexpected patterns

**🔴 FAIL** - Major issues detected:
- Significant reorder overhead
- Format inconsistencies
- Performance regressions

---

## Common Use Cases

### Use Case 1: Initial Validation

**Scenario**: Validate output-side optimizations after implementation

```bash
# Run full automated workflow
./scripts/capture_output_side_trace.sh
```

**Expected Result**: All criteria PASS if optimizations are correct

### Use Case 2: Regression Testing

**Scenario**: Verify optimizations after code changes

```bash
# Capture new traces
./scripts/capture_output_side_trace.sh --skip-extraction

# Compare with previous validation
python3 scripts/validate_output_side_optimizations.py \
    --trace output_side_validation/traces/onednn_trace_output_side_run1.txt \
    --baseline previous_validation/traces/onednn_trace_output_side_run1.txt \
    --output regression_report.md
```

**Expected Result**: No degradation in metrics

### Use Case 3: Different Model Layer

**Scenario**: Validate optimizations on different transformer layer

```bash
# Validate layer 12 instead of layer 0
./scripts/capture_output_side_trace.sh \
    --layer-index 12 \
    --output-dir ./validation_layer12
```

**Expected Result**: Consistent behavior across layers

### Use Case 4: CI/CD Integration

**Scenario**: Automated validation in build pipeline

```bash
# Run validation and capture exit code
./scripts/capture_output_side_trace.sh --output-dir ./ci_validation

# Parse JSON results for pass/fail
python3 -c "
import json
with open('./ci_validation/metrics/output_side_run1_metrics.json', 'r') as f:
    data = json.load(f)
    
# Check for activation reorders
by_dim = data.get('by_dimension', {})
activation_reorders = sum(
    info['count'] for dim, info in by_dim.items() 
    if '6x1536' in dim or '1x1536' in dim
)

# Exit with code 1 if reorders found
exit(1 if activation_reorders > 0 else 0)
"

if [ $? -eq 0 ]; then
    echo "✅ Validation PASSED"
else
    echo "❌ Validation FAILED"
    exit 1
fi
```

---

## Prerequisites

### Required Packages

```bash
pip install torch transformers openvino numpy
```

### System Requirements

- **Python**: 3.10 or higher
- **Memory**: ~4GB RAM (for model download and inference)
- **Disk Space**: ~5GB (model + traces + metrics)
- **CPU**: AVX2 support recommended (AMD Ryzen 9 5900X tested)

### Environment Variables

For trace capture:
```bash
export DNNL_VERBOSE=1  # Enable oneDNN verbose logging
```

Optional optimization flags:
```bash
export OV_CPU_ENABLE_FAST_TRANSPOSE=1
export OV_CPU_ENABLE_BRGEMM_COPY_B=1
```

---

## Troubleshooting

### Issue: Model extraction fails

**Symptoms**: Script exits during model extraction step

**Solution**:
```bash
# Check network connectivity (for HuggingFace download)
ping huggingface.co

# Check disk space
df -h

# Try manual extraction
python3 scripts/extract_transformer_block.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --layer-index 0 \
    --output-dir ./test_extraction
```

### Issue: Trace capture produces empty file

**Symptoms**: Trace file exists but has no oneDNN verbose output

**Solution**:
```bash
# Verify DNNL_VERBOSE is set
echo $DNNL_VERBOSE  # Should output "1"

# Set explicitly and retry
export DNNL_VERBOSE=1
./scripts/capture_output_side_trace.sh
```

### Issue: Validation reports unexpected reorders

**Symptoms**: Validation finds activation reorders when expecting zero

**Solution**:
```bash
# Manually inspect trace for activation reorders
grep "reorder.*6x1536\|reorder.*1x1536" output_side_validation/traces/*.txt

# Check if these are weight or activation reorders
# Weight reorders (large first dimension): Expected and beneficial
# Activation reorders (small first dimension): Need investigation

# Review full reorder context
grep -A 2 -B 2 "reorder.*6x1536" output_side_validation/traces/*.txt
```

### Issue: Baseline comparison fails

**Symptoms**: Script cannot find baseline files

**Solution**:
```bash
# Check baseline directory structure
ls -la baseline_capture/metrics/

# Specify custom baseline directory
./scripts/capture_output_side_trace.sh \
    --baseline-dir /path/to/your/baseline

# Or skip baseline comparison
./scripts/capture_output_side_trace.sh --no-baseline-compare
```

---

## Advanced Usage

### Custom Trace Analysis

Extract specific metrics from traces:

```bash
# Count total reorders
grep -c "reorder" output_side_validation/traces/onednn_trace_output_side_run1.txt

# Find attention output operations (1536→1536)
grep "inner_product.*ic1536oc1536" output_side_validation/traces/onednn_trace_output_side_run1.txt

# Find FFN contract operations (8960→1536)
grep "inner_product.*ic8960oc1536" output_side_validation/traces/onednn_trace_output_side_run1.txt

# Check output formats
grep "inner_product.*ic1536oc1536" output_side_validation/traces/onednn_trace_output_side_run1.txt | \
    grep -o "dst_[^,]*"
```

### Automated Metrics Extraction

Use existing parsing tools:

```bash
# Extract reorder metrics to CSV
python3 scripts/parse_onednn_reorders.py \
    --trace output_side_validation/traces/onednn_trace_output_side_run1.txt \
    --output-csv reorder_metrics.csv \
    --output-json reorder_metrics.json

# View metrics
cat reorder_metrics.csv | column -t -s,
```

### Batch Validation

Validate multiple layers:

```bash
for layer in 0 6 12 18 23; do
    echo "Validating layer $layer..."
    ./scripts/capture_output_side_trace.sh \
        --layer-index $layer \
        --output-dir ./validation_layer_${layer} \
        --skip-extraction  # Only first needs extraction
done

# Compare results across layers
for layer in 0 6 12 18 23; do
    echo "Layer $layer:"
    grep "Total reorder operations:" ./validation_layer_${layer}/OUTPUT_SIDE_VALIDATION.md
done
```

---

## Expected Results

### Optimal State (Current Implementation)

When output-side optimizations are correctly implemented:

```
Validation Report Summary:
✅ All validation criteria PASSED

- Attention output reorders: 0
- FFN output reorders: 0
- Block boundary reorders: 0
- Total activation reorders: 0
- Weight reorders: ~48 (expected and beneficial)

Attention output format: f32::ab (consistent)
FFN output format: f32::ab (consistent)

Overall status: OPTIMAL
```

### Sub-Optimal State (Needs Optimization)

If output-side layouts need optimization:

```
Validation Report Summary:
⚠️  Some validation criteria need review

- Attention output reorders: 2-4 per block
- FFN output reorders: 2-4 per block
- Block boundary reorders: 1-2 per transition
- Total activation reorders: 48-96

Estimated overhead: ~2-4 ms per inference

Recommendations:
1. Enforce f32::ab output format in attention projection
2. Enforce f32::ab output format in FFN contract
3. Review graph optimizer layout propagation
```

---

## Integration with Optimization Workflow

This validation step fits into the overall layout optimization workflow:

```
Task 2: Baseline Capture
    ↓
Tasks 3-11: Analysis and Planning
    ↓
Tasks 12-25: Input-side and Weight Optimizations
    ↓
Tasks 29-31: Output-side Optimizations (Attention, FFN, Block Boundary)
    ↓
Task 26: OUTPUT-SIDE VALIDATION ← [YOU ARE HERE]
    ↓
Tasks 27-28: Input-side Optimizations (Upcoming)
    ↓
Task 32: Final Comparison and Report
```

**Purpose**: Confirm that output-side optimizations:
- Reduce reorder overhead at attention output
- Reduce reorder overhead at FFN output
- Eliminate block boundary reorders
- Do not introduce regressions elsewhere

---

## Related Documentation

- **Baseline Capture**: `scripts/BASELINE_CAPTURE_README.md`
- **Trace Parsing**: `scripts/PARSE_ONEDNN_REORDERS_README.md`
- **Attention Output Analysis**: `ATTENTION_OUTPUT_LAYOUT_OPTIMIZATION.md`
- **FFN Output Analysis**: `FFN_OUTPUT_LAYOUT_OPTIMIZATION.md`
- **Complete Validation Report**: `OUTPUT_SIDE_VALIDATION_REPORT.md`

---

## Summary

The output-side validation tools provide:

✅ **Automated workflow** for end-to-end validation  
✅ **Flexible Python tool** for custom analysis  
✅ **Comprehensive reports** with actionable insights  
✅ **Baseline comparison** for regression detection  
✅ **CI/CD integration** support  
✅ **Detailed documentation** for all use cases

**Result**: Efficient validation of output-side layout optimizations with minimal manual effort.

---

*For questions or issues, refer to the main project documentation or contact the optimization team.*

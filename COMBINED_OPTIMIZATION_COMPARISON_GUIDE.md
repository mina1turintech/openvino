# Combined Optimization Comparison Guide

**Task 29/32**: Generate and compare baseline vs. full optimization traces  
**Purpose**: Validate that input-side and output-side optimizations compound positively  
**Date**: 2025-01-21  
**Model**: Qwen2-1.5B-Instruct Transformer Block

---

## Table of Contents

1. [Overview](#overview)
2. [What Are Combined Optimizations?](#what-are-combined-optimizations)
3. [Validation Methodology](#validation-methodology)
4. [Quick Start](#quick-start)
5. [Detailed Workflow](#detailed-workflow)
6. [Interpreting Results](#interpreting-results)
7. [Success Criteria](#success-criteria)
8. [Additivity Analysis](#additivity-analysis)
9. [Troubleshooting](#troubleshooting)
10. [CI/CD Integration](#cicd-integration)

---

## Overview

Combined optimization validation measures the **cumulative impact** of applying both input-side (weight pre-reordering) and output-side (activation layout) optimizations together. This validates that:

1. **Both optimizations work independently** without conflicts
2. **Improvements are additive** (or super-additive)
3. **No unexpected regressions** from optimization interactions
4. **Total improvement ≥ sum of individual improvements**

### Key Facts

- **Input-Side Optimization**: Reduces weight reorder overhead (~5ms/block in baseline)
- **Output-Side Optimization**: Reduces activation reorder overhead (0-0.15ms/block depending on baseline)
- **Expected Combined Impact**: Sum of individual improvements with minimal interaction effects
- **24-Layer Model**: Cumulative savings scale linearly across layers

---

## What Are Combined Optimizations?

### Individual Optimizations

#### Input-Side (Weight Pre-Reordering)
```
Before Compute:
  Runtime weight reorders u8::ab → u8::AB8b24a (5ms overhead)
  
After Optimization:
  One-time reorder at compilation (~50ms once)
  Zero runtime overhead for weights
```

**Target**: Weight reorders before MatMul/InnerProduct operations

#### Output-Side (Activation Layout)
```
After Compute:
  MatMul produces blocked output → reorder to plain (0.05-0.08ms)
  
After Optimization:
  MatMul produces plain output directly
  Zero runtime overhead for activation reorders
```

**Target**: Activation reorders after MatMul/InnerProduct operations

### Combined Impact

```
Baseline (Research Branch):
  Weight reorders:     5.0 ms   } Combined: ~5.15 ms
  Activation reorders: 0.15 ms  } per block

Fully Optimized:
  Weight reorders:     0.0 ms   } Combined: ~0.0 ms
  Activation reorders: 0.0 ms   } per block

Improvement: ~5.15 ms/block → ~124 ms/24-layer model
```

---

## Validation Methodology

### Four-Way Comparison

This validation compares **four different traces**:

1. **Baseline**: No optimizations (research branch)
2. **Input-Side Only**: Weight pre-reordering enabled, activation layouts unchanged
3. **Output-Side Only**: Activation layouts optimized, weight reorders still at runtime
4. **Combined**: Both optimizations enabled

### Additivity Validation

```
Expected: Combined improvement ≈ Input-side improvement + Output-side improvement
Actual:   Combined improvement = measure from trace

Additivity Ratio = Actual / Expected × 100%

✅ 95-105%: Perfectly additive
✅ >105%:   Super-additive (synergistic effects)
⚠️ 80-95%:  Mostly additive (minor interactions)
❌ <80%:    Significant conflicts
```

### Metrics Tracked

- **Total reorder count** (baseline vs combined)
- **Total reorder time** (ms per block)
- **Weight reorder breakdown** (by dimension: 1536, 8960, 256)
- **Activation reorder breakdown** (by dimension: 6×1536, 1×1536)
- **Scale/ZP reorders** (should remain unchanged)
- **Per-implementation metrics** (jit:uni, jit_direct_copy:uni)

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install torch transformers openvino numpy

# Ensure scripts are executable
chmod +x scripts/capture_combined_trace.sh
chmod +x scripts/example_combined_validation.sh
```

### One-Command Validation

```bash
# Interactive guided validation
./scripts/example_combined_validation.sh
```

This script will:
1. Check for baseline trace availability
2. Capture combined optimization trace
3. Generate comparison report with additivity analysis
4. Display results and recommendations

### Manual Three-Step Process

#### Step 1: Capture Combined Trace

```bash
./scripts/capture_combined_trace.sh \
    --output-dir ./combined_validation \
    --num-runs 3 \
    --iterations 50
```

**Requirements**:
- OpenVINO build with **both** input-side and output-side optimizations compiled in
- ~5 minutes to complete (includes model extraction, trace capture, parsing)

#### Step 2: Generate Comparison Report

```bash
python3 scripts/compare_combined_traces.py \
    --baseline ./baseline_capture/metrics \
    --combined ./combined_validation/metrics_combined \
    --input-side ./input_side_validation/metrics_optimized \
    --output-side ./output_side_validation/metrics_optimized \
    --output ./combined_validation/COMBINED_COMPARISON.md
```

**Note**: `--input-side` and `--output-side` are optional but required for additivity analysis.

#### Step 3: Review Results

```bash
cat ./combined_validation/COMBINED_COMPARISON.md
```

Check for:
- ✅ Total improvement > individual improvements
- ✅ Additivity ratio between 80-105%
- ✅ No regressions in non-target dimensions
- ✅ Both weight and activation reorders reduced

---

## Detailed Workflow

### Phase 1: Baseline Capture

If not already done (from Task 2):

```bash
./scripts/capture_baseline_trace.sh \
    --output-dir ./baseline_capture \
    --num-runs 3 \
    --iterations 50
```

This creates the reference point with **no optimizations applied**.

### Phase 2: Individual Optimization Captures

#### Input-Side Capture

If not already done (from Task 27):

```bash
./scripts/capture_input_side_trace.sh \
    --optimized \
    --output-dir ./input_side_validation \
    --num-runs 3 \
    --iterations 50
```

#### Output-Side Capture

If not already done (from Task 28):

```bash
./scripts/capture_output_side_trace.sh \
    --output-dir ./output_side_validation \
    --num-runs 3 \
    --iterations 50
```

### Phase 3: Combined Optimization Capture

**Critical**: Ensure your OpenVINO build has **both** optimizations:

```bash
# Verify input-side optimization (weight pre-reordering)
# Check that src/plugins/intel_cpu/src/nodes/matmul.cpp includes caching logic

# Verify output-side optimization (activation layouts)
# Check that attention/FFN outputs use plain layouts (abcd)

# Then capture combined trace
./scripts/capture_combined_trace.sh \
    --output-dir ./combined_validation \
    --num-runs 3 \
    --iterations 50 \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --layer-index 0
```

**What This Does**:
1. Extracts transformer block (or reuses existing)
2. Runs inference with full oneDNN verbose logging (3 runs × 50 iterations)
3. Captures all reorder operations
4. Parses and aggregates metrics
5. Generates quick summary

**Output Structure**:
```
./combined_validation/
├── model/
│   ├── transformer_block.xml
│   └── transformer_block.bin
├── traces_combined/
│   ├── onednn_trace_combined_run1.txt
│   ├── onednn_trace_combined_run2.txt
│   └── onednn_trace_combined_run3.txt
├── metrics_combined/
│   ├── combined_run1_metrics.csv
│   ├── combined_run1_metrics.json
│   ├── combined_run2_metrics.csv
│   ├── combined_run2_metrics.json
│   ├── combined_run3_metrics.csv
│   └── combined_run3_metrics.json
└── logs/
    ├── extraction_combined.log
    ├── capture_combined_run1.log
    └── parse_combined_run1.log
```

### Phase 4: Comprehensive Comparison

Generate the full comparison report:

```bash
python3 scripts/compare_combined_traces.py \
    --baseline ./baseline_capture/metrics \
    --combined ./combined_validation/metrics_combined \
    --input-side ./input_side_validation/metrics_optimized \
    --output-side ./output_side_validation/metrics_optimized \
    --output ./combined_validation/COMBINED_COMPARISON.md
```

**Report Sections**:
1. **Executive Summary**: Overall metrics at a glance
2. **Reorder Category Analysis**: Weight vs activation vs scale/ZP breakdown
3. **Additivity Analysis**: Validates improvements compound properly
4. **Per-Dimension Breakdown**: Detailed analysis by tensor dimension
5. **Success Criteria Validation**: Automated pass/fail checks
6. **24-Layer Model Projection**: Full model impact estimation
7. **Recommendations**: Context-aware guidance
8. **Implementation Status**: Verification of optimization presence

---

## Interpreting Results

### Scenario A: Perfect Additivity

```markdown
Baseline Total Time:     5.20 ms
Input-Side Only Time:    0.15 ms  (5.05 ms reduction)
Output-Side Only Time:   5.05 ms  (0.15 ms reduction)
Combined Time:           0.00 ms  (5.20 ms reduction)

Expected Reduction: 5.05 + 0.15 = 5.20 ms
Actual Reduction:   5.20 ms
Additivity Ratio:   100%

✅ Result: Optimizations compound perfectly with no conflicts
```

### Scenario B: Super-Additive

```markdown
Baseline Total Time:     5.20 ms
Input-Side Only Time:    0.15 ms  (5.05 ms reduction)
Output-Side Only Time:   5.05 ms  (0.15 ms reduction)
Combined Time:           0.00 ms  (5.20 ms reduction)

Expected Reduction: 5.05 + 0.15 = 5.20 ms
Actual Reduction:   5.30 ms
Additivity Ratio:   102%

✅ Result: Optimizations have positive synergy!
(Possible reasons: Reduced cache pressure, better memory access patterns)
```

### Scenario C: Sub-Additive

```markdown
Baseline Total Time:     5.20 ms
Input-Side Only Time:    0.20 ms  (5.00 ms reduction)
Output-Side Only Time:   5.10 ms  (0.10 ms reduction)
Combined Time:           0.30 ms  (4.90 ms reduction)

Expected Reduction: 5.00 + 0.10 = 5.10 ms
Actual Reduction:   4.90 ms
Additivity Ratio:   96%

⚠️ Result: Minor interaction effect detected
(Still good - 96% is within acceptable range)
```

### Scenario D: Significant Conflict

```markdown
Baseline Total Time:     5.20 ms
Input-Side Only Time:    0.20 ms  (5.00 ms reduction)
Output-Side Only Time:   5.10 ms  (0.10 ms reduction)
Combined Time:           1.50 ms  (3.70 ms reduction)

Expected Reduction: 5.00 + 0.10 = 5.10 ms
Actual Reduction:   3.70 ms
Additivity Ratio:   73%

❌ Result: Significant conflict - investigate implementation
```

### Current Expected Result (Based on Tasks 27-28)

From previous analysis:
- **Baseline already has optimal output layouts** (Task 28 finding)
- **Input-side optimization expected to show ~5ms reduction** (Task 27)
- **Output-side already optimal** (0ms additional improvement)

Therefore:
```markdown
Baseline Total Time:     ~5.0 ms (mostly weight reorders)
Combined Time:           ~0.0 ms (weight reorders eliminated)
Expected Improvement:    ~5.0 ms (100% from input-side)
Additivity Ratio:        100% (output-side already optimal in baseline)

✅ Result: Input-side optimization effective, output-side already optimal
```

---

## Success Criteria

The combined validation succeeds if **all** of the following are met:

| Criterion | Target | How to Check |
|-----------|--------|--------------|
| **Combined improvement > individual** | Combined reduction ≥ max(input-side, output-side) | Compare total time reductions |
| **Weight reorder reduction** | Baseline weight time > combined weight time | Check weight_input category |
| **Activation reorder reduction** | Baseline activation time ≥ combined activation time | Check activation_output category |
| **1536-dimension improvements** | Reduced 1536×1536 and related reorders | Check per-dimension table |
| **8960-dimension improvements** | Reduced FFN weight reorders | Check per-dimension table |
| **No regressions in non-critical ops** | Scale/ZP reorders stable | Compare scale_zp category |
| **Optimizations scale proportionally** | Improvement > 5% overall | Check improvement percentage |
| **Additivity ratio (if data available)** | Ratio between 80-105% | Check additivity analysis section |

### Automated Validation

The comparison tool automatically checks all criteria:

```markdown
## Success Criteria Validation

| Criterion | Target | Status | Result |
|-----------|--------|--------|--------|
| Combined improvement > individual | ... | ✅ PASS | 5.20 ms reduction |
| Weight reorder reduction | ... | ✅ PASS | 5.05 ms reduction |
| Activation reorder reduction | ... | ✅ PASS | 0.15 ms reduction |
...

**Overall**: 8/8 criteria passed
```

---

## Additivity Analysis

### Why Additivity Matters

Additivity validates that:
1. Optimizations don't interfere with each other
2. No hidden dependencies between optimizations
3. Performance improvements are predictable
4. Optimizations can be deployed independently if needed

### Mathematical Definition

```
Let:
  B  = Baseline reorder time
  I  = Input-side optimized reorder time
  O  = Output-side optimized reorder time
  C  = Combined optimized reorder time

Reductions:
  ΔI = B - I  (input-side improvement)
  ΔO = B - O  (output-side improvement)
  ΔC = B - C  (combined improvement)

Expected Combined Time (if perfectly additive):
  C_expected = B - (ΔI + ΔO)
             = B - (B - I) - (B - O)
             = I + O - B

Additivity Ratio:
  R = ΔC / (ΔI + ΔO) × 100%
    = (B - C) / ((B - I) + (B - O)) × 100%
```

### Interpretation

| Ratio Range | Interpretation | Action |
|-------------|----------------|--------|
| **95-105%** | Perfect additivity | ✅ Deploy both optimizations |
| **>105%** | Super-additive (synergy) | ✅ Deploy both, investigate synergy |
| **80-95%** | Mostly additive | ✅ Deploy both, acceptable interaction |
| **60-80%** | Moderate conflict | ⚠️ Investigate, may still be beneficial |
| **<60%** | Significant conflict | ❌ Debug optimization implementations |

### Example Calculation

```python
# From comparison report
baseline_time = 5.20  # ms
input_side_time = 0.15  # ms (after input-side opt only)
output_side_time = 5.05  # ms (after output-side opt only)
combined_time = 0.00  # ms (after both opts)

# Calculate reductions
input_reduction = baseline_time - input_side_time  # 5.05 ms
output_reduction = baseline_time - output_side_time  # 0.15 ms
combined_reduction = baseline_time - combined_time  # 5.20 ms

# Expected combined reduction (if additive)
expected_reduction = input_reduction + output_reduction  # 5.20 ms

# Additivity ratio
additivity_ratio = (combined_reduction / expected_reduction) * 100  # 100%

print(f"Additivity: {additivity_ratio:.1f}%")  # "Additivity: 100.0%"
```

---

## Troubleshooting

### Issue: Combined improvement ≈ only one optimization

**Symptom**:
```
Combined reduction: 5.0 ms
Input-side reduction: 5.0 ms
Output-side reduction: 0.0 ms
Additivity: 100% (but only because output-side contributes nothing)
```

**Possible Causes**:
1. **Output-side optimization not compiled in**
   - Check that src/plugins/intel_cpu has output layout changes
   - Verify MatMul outputs use plain layouts

2. **Baseline already has optimal output layouts** (Expected scenario)
   - This is actually correct based on Task 28 findings
   - No action needed - input-side optimization is the primary contributor

**Solution**: Review Task 28 results. If baseline already optimal, this is expected behavior.

### Issue: Additivity ratio < 80%

**Symptom**:
```
Expected reduction: 5.20 ms
Actual reduction: 3.50 ms
Additivity: 67%
```

**Possible Causes**:
1. **Conflicting memory layout requirements**
   - Input-side wants blocked weights
   - Output-side wants plain outputs
   - Combined may force additional reorders

2. **Cache invalidation issues**
   - Combined optimizations may clear caches more frequently
   - Check cache hit rates

3. **Implementation bugs**
   - One optimization may disable the other
   - Check for conditional logic conflicts

**Solution**:
1. Review both optimization implementations
2. Check for shared state or dependencies
3. Add instrumentation to track cache behavior
4. Consider debugging with DNNL_VERBOSE=profile

### Issue: Regression in scale/ZP reorders

**Symptom**:
```
Scale/ZP baseline:  0.12 ms
Scale/ZP combined:  0.35 ms
```

**Possible Causes**:
1. Layout changes forcing additional small-vector reorders
2. Cache alignment issues

**Solution**: Scale/ZP reorders are small overhead - if total improvement still positive, may be acceptable.

### Issue: Trace capture fails on combined build

**Symptom**:
```
Error: Model compilation failed
```

**Possible Causes**:
1. Both optimizations incompatible
2. Missing kernel implementations for new layout combinations

**Solution**:
1. Test each optimization individually first
2. Check OpenVINO compilation logs
3. Verify all required kernels are available

---

## CI/CD Integration

### Automated Validation Pipeline

```yaml
# .github/workflows/combined_validation.yml
name: Combined Optimization Validation

on:
  pull_request:
    paths:
      - 'src/plugins/intel_cpu/**'
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday

jobs:
  validate-combined:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build OpenVINO (fully optimized)
        run: |
          cmake -B build -DENABLE_INPUT_SIDE_OPT=ON -DENABLE_OUTPUT_SIDE_OPT=ON
          cmake --build build -j$(nproc)
      
      - name: Capture baseline trace
        run: ./scripts/capture_baseline_trace.sh --output-dir ./baseline
      
      - name: Capture combined trace
        run: ./scripts/capture_combined_trace.sh --output-dir ./combined
      
      - name: Generate comparison
        run: |
          python3 scripts/compare_combined_traces.py \
            --baseline ./baseline/metrics \
            --combined ./combined/metrics_combined \
            --output ./COMBINED_COMPARISON.md
      
      - name: Validate results
        run: |
          # Check that report was generated
          test -f ./COMBINED_COMPARISON.md
          
          # Check for success criteria
          grep -q "8/8 criteria passed" ./COMBINED_COMPARISON.md
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: combined-validation-results
          path: |
            ./COMBINED_COMPARISON.md
            ./combined/metrics_combined/
```

### Regression Detection

```bash
#!/bin/bash
# scripts/validate_combined_regression.sh

# Compare against reference baseline
python3 scripts/compare_combined_traces.py \
    --baseline ./reference/baseline_metrics \
    --combined ./combined_validation/metrics_combined \
    --output /tmp/combined_check.md

# Check for regressions
if grep -q "❌ FAIL" /tmp/combined_check.md; then
    echo "ERROR: Combined optimization validation failed"
    cat /tmp/combined_check.md
    exit 1
fi

# Check improvement threshold
IMPROVEMENT=$(grep "Improvement %" /tmp/combined_check.md | head -1 | awk '{print $NF}' | tr -d '%')
if (( $(echo "$IMPROVEMENT < 50" | bc -l) )); then
    echo "WARNING: Combined improvement below 50% ($IMPROVEMENT%)"
    exit 1
fi

echo "✓ Combined optimization validation passed"
exit 0
```

---

## Related Documentation

- **Task 2**: [Baseline Trace Capture](scripts/BASELINE_QUICKSTART.md)
- **Task 4**: Input-Side Optimization Implementation
- **Task 5**: Output-Side Optimization Implementation
- **Task 27**: [Input-Side Validation Guide](INPUT_SIDE_VALIDATION_GUIDE.md)
- **Task 28**: [Output-Side Comparison Guide](OUTPUT_SIDE_COMPARISON_GUIDE.md)
- **Task 29**: [Quick Start Combined Validation](QUICK_START_COMBINED_VALIDATION.md) (this task)

---

## Summary

Combined optimization validation is the **final verification** that both input-side and output-side optimizations work together harmoniously. Key takeaways:

1. **Measure cumulative impact** of applying both optimizations
2. **Validate additivity** to ensure no conflicts
3. **Check per-dimension metrics** to verify targeted improvements
4. **Project to 24-layer model** for real-world impact assessment
5. **Automate in CI/CD** for regression prevention

Expected outcome for Qwen2-1.5B:
- **~5ms/block improvement** (mostly from input-side weight reorder elimination)
- **~120ms savings** in 24-layer model
- **100% additivity** (output-side already optimal in baseline)

---

**Last Updated**: 2025-01-21  
**Validation Status**: Ready for execution  
**Next Task**: Task 30 - Test with multiple trace scenarios

# Multi-Scenario Validation Guide

**Task 30/32**: Test with multiple trace scenarios  
**Purpose**: Validate robustness of layout optimizations across diverse inference conditions  
**Date**: 2025-01-21  
**Model**: Qwen2-1.5B-Instruct Transformer Block

---

## Table of Contents

1. [Overview](#overview)
2. [Why Multi-Scenario Testing?](#why-multi-scenario-testing)
3. [Test Matrix Design](#test-matrix-design)
4. [Quick Start](#quick-start)
5. [Detailed Workflow](#detailed-workflow)
6. [Metrics and Statistics](#metrics-and-statistics)
7. [Interpreting Results](#interpreting-results)
8. [Success Criteria](#success-criteria)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## Overview

Multi-scenario validation ensures that layout optimizations (both input-side weight pre-reordering and output-side activation layout) are **robust and stable** across different inference conditions. This prevents scenarios where optimizations work well in one configuration but fail or regress in others.

### Key Validation Goals

1. **Batch Size Independence**: Optimizations work consistently for batch sizes 1, 4, and 8
2. **Sequence Length Independence**: Optimizations work consistently for sequence lengths 1, 32, 128, and 256
3. **Input Pattern Independence**: Optimizations are not sensitive to input activation values
4. **Low Variance**: Multiple runs show stable, repeatable results (CV < 5%)
5. **No Regressions**: No unexpected performance degradation in any scenario

---

## Why Multi-Scenario Testing?

### Common Pitfalls Without Multi-Scenario Testing

1. **Batch-Dependent Optimizations**: Code works for batch=1 but breaks for larger batches
2. **Sequence Length Assumptions**: Optimizations assume typical sequence lengths (e.g., 128) but fail for edge cases (1 or 256)
3. **Data-Dependent Behavior**: Reorder decisions depend on activation magnitudes, causing unpredictable performance
4. **Layout Thrashing**: Certain input combinations trigger unexpected layout conversions
5. **Hidden Regressions**: Optimization helps common cases but hurts rare cases

### What We're Testing

```
Test Matrix (36 scenarios × 3 repetitions = 108 runs):

Batch Sizes: [1, 4, 8]
  ├── Sequence Lengths: [1, 32, 128, 256]
  │     ├── Input Pattern: random (standard normal distribution)
  │     ├── Input Pattern: ones (all 1.0)
  │     └── Input Pattern: small_values (centered at 0.1, low variance)
  └── 3 repetitions per scenario for variance analysis

Expected Results:
  ✓ Reorder reduction > 90% across all batch sizes
  ✓ Reorder reduction > 90% across all sequence lengths
  ✓ Input patterns show < 5% variance
  ✓ Low standard deviation across repetitions
```

---

## Test Matrix Design

### Dimension 1: Batch Size

| Batch Size | Use Case | Expected Behavior |
|------------|----------|-------------------|
| 1 | Interactive inference (chatbot) | Optimal latency, single request |
| 4 | Small batch processing | Balanced throughput and latency |
| 8 | Batch processing | Higher throughput, acceptable latency |

**Why these values?** 
- Batch=1 is most common for real-time inference
- Batch=4 tests multi-request scenarios
- Batch=8 validates batch scaling behavior

### Dimension 2: Sequence Length

| Seq Length | Use Case | Expected Behavior |
|------------|----------|-------------------|
| 1 | Token-by-token generation | Minimal activation footprint |
| 32 | Short prompts | Common chat use case |
| 128 | Medium prompts | Standard model context |
| 256 | Long prompts | Large context handling |

**Why these values?**
- Seq=1 is smallest possible (single token)
- Seq=32 represents typical chat turns
- Seq=128 is common prompt length
- Seq=256 tests larger context windows

### Dimension 3: Input Pattern

| Pattern | Description | Purpose |
|---------|-------------|---------|
| random | Standard normal (μ=0, σ=1) | Baseline realistic inputs |
| ones | All activations = 1.0 | Test extreme uniform values |
| small_values | Centered at 0.1 (σ=0.01) | Test small magnitude activations |

**Why these patterns?**
- `random`: Mimics real transformer activations
- `ones`: Tests edge case of uniform values (may trigger different quantization/reorder paths)
- `small_values`: Tests numerical stability with small magnitudes

### Dimension 4: Repetitions

- **3 repetitions** per scenario (configurable to 5)
- Allows calculation of mean, standard deviation, coefficient of variation (CV)
- Identifies measurement noise vs. real variance

---

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install torch transformers openvino numpy

# Ensure scripts are executable
chmod +x scripts/capture_multi_scenario_traces.sh
chmod +x scripts/example_multi_scenario_validation.sh
```

### One-Command Validation (Recommended)

```bash
# Interactive guided validation
./scripts/example_multi_scenario_validation.sh
```

This will:
1. Check for existing model extraction
2. Run multi-scenario trace capture (full or quick mode)
3. Generate statistical analysis report
4. Display results summary
5. Provide recommendations

### Quick Mode (Faster Testing)

```bash
# Reduced test matrix: 12 scenarios instead of 36
./scripts/capture_multi_scenario_traces.sh --quick
```

Quick mode tests:
- Batch sizes: [1, 4] (instead of [1, 4, 8])
- Sequence lengths: [1, 128] (instead of [1, 32, 128, 256])
- Input patterns: [random, ones] (instead of [random, ones, small_values])
- **Total**: 2×2×2 = 8 scenarios × 3 reps = 24 runs (~48 minutes)

---

## Detailed Workflow

### Step 1: Capture Multi-Scenario Traces

```bash
./scripts/capture_multi_scenario_traces.sh \
    --output-dir ./multi_scenario_validation \
    --repetitions 3 \
    --iterations 50 \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --layer-index 0
```

**Options**:
- `--output-dir`: Output directory (default: `./multi_scenario_validation`)
- `--repetitions`: Number of repetitions per scenario (default: 3)
- `--iterations`: Inference iterations per trace (default: 50)
- `--quick`: Use reduced test matrix
- `--skip-extraction`: Skip model extraction if already done
- `--model-path`: Use existing extracted model

**Output Structure**:
```
multi_scenario_validation/
├── model/
│   └── transformer_block.xml
├── traces/
│   ├── batch1_seq1_random/
│   │   ├── onednn_trace_rep1.txt
│   │   ├── onednn_trace_rep2.txt
│   │   └── onednn_trace_rep3.txt
│   ├── batch1_seq32_random/
│   └── ... (36 scenario directories)
├── metrics/
│   ├── batch1_seq1_random/
│   │   ├── metrics_rep1.json
│   │   ├── metrics_rep2.json
│   │   └── metrics_rep3.json
│   └── ... (36 scenario directories)
├── logs/
├── statistics/
├── scenario_matrix.txt
└── QUICK_SUMMARY.txt
```

**Duration**:
- Full mode: ~3-4 hours (36 scenarios × 3 reps)
- Quick mode: ~45 minutes (8 scenarios × 3 reps)

### Step 2: Analyze Statistics

```bash
python3 scripts/analyze_multi_scenario_statistics.py \
    --metrics-dir ./multi_scenario_validation/metrics \
    --output ./multi_scenario_validation/statistics/MULTI_SCENARIO_ANALYSIS.md \
    --variance-threshold 5.0
```

**Options**:
- `--metrics-dir`: Directory with all scenario metrics (required)
- `--baseline-dir`: Optional baseline metrics for comparison
- `--output`: Output report file (default: `MULTI_SCENARIO_ANALYSIS.md`)
- `--variance-threshold`: CV threshold for high variance detection (default: 5.0%)

**Report Contents**:
1. Executive Summary
2. High Variance Scenarios (CV > 5%)
3. Batch Size Consistency Analysis
4. Sequence Length Consistency Analysis
5. Input Pattern Consistency Analysis
6. Detailed Scenario Metrics Table
7. Success Criteria Validation
8. Recommendations

### Step 3: Review Results

```bash
cat ./multi_scenario_validation/statistics/MULTI_SCENARIO_ANALYSIS.md
```

**Key Sections to Check**:
1. **High Variance Scenarios**: Should be empty or minimal
2. **Consistency Tables**: All metrics should show ✅ (CV < 5-10%)
3. **Success Criteria**: Should pass 4/4 or 3/4 criteria
4. **Recommendations**: Should indicate "No Issues Detected"

---

## Metrics and Statistics

### Primary Metrics Tracked

| Metric | Description | Success Threshold |
|--------|-------------|-------------------|
| Total Reorder Count | Number of reorder operations per block | Low variance (CV < 5%) |
| Total Reorder Time (ms) | Time spent in reorder operations | Low variance (CV < 5%) |
| Mean Reorder Time (ms) | Average time per reorder | Consistent across scenarios |

### Statistical Measures

1. **Mean (μ)**: Average across repetitions
2. **Standard Deviation (σ)**: Measure of spread
3. **Coefficient of Variation (CV)**: σ/μ × 100% (normalized variance)
4. **Min/Max**: Range of values
5. **Range %**: (Max - Min) / Mean × 100%

### Coefficient of Variation (CV) Interpretation

| CV Range | Status | Interpretation |
|----------|--------|----------------|
| < 5% | ✅ Excellent | Very stable, low variance |
| 5-10% | ⚠️ Acceptable | Some variance, generally OK |
| 10-20% | ⚠️ High | Significant variance, investigate |
| > 20% | ❌ Critical | Unstable, requires investigation |

### Example Calculation

```
Scenario: batch1_seq128_random
Repetition 1: 42 reorders, 0.85 ms
Repetition 2: 41 reorders, 0.83 ms
Repetition 3: 43 reorders, 0.87 ms

Statistics:
  Mean: 42.0 reorders, 0.85 ms
  Std Dev: 1.0 reorders, 0.02 ms
  CV: 2.4% reorders, 2.4% time
  Status: ✅ Excellent (CV < 5%)
```

---

## Interpreting Results

### Scenario 1: All Tests Pass ✅

```
High Variance Scenarios: None
Batch Size Consistency: ✅ (CV < 5%)
Sequence Length Consistency: ✅ (CV < 5%)
Input Pattern Consistency: ✅ (CV < 5%)
Overall Score: 4/4
```

**Interpretation**: Optimizations are robust and ready for production. Proceed with integration.

**Next Steps**:
- Document validated scenarios in release notes
- Consider extending to additional model architectures
- Deploy to production with confidence

### Scenario 2: High Variance in Specific Scenarios ⚠️

```
High Variance Scenarios:
  - batch8_seq256_ones: total_reorder_time_ms (CV=7.2%)
Batch Size Consistency: ✅
Sequence Length Consistency: ✅
Input Pattern Consistency: ⚠️ (ones pattern shows variance)
Overall Score: 3/4
```

**Interpretation**: Most scenarios are stable, but the "ones" input pattern with large batch/sequence shows variance.

**Next Steps**:
1. Review trace for `batch8_seq256_ones` to identify cause
2. Check if variance is measurement noise (run more repetitions)
3. If real, investigate if "ones" pattern triggers different quantization behavior
4. Consider acceptable if variance is < 10% and not critical path

### Scenario 3: Batch Size Inconsistency ❌

```
High Variance Scenarios: Multiple
Batch Size Consistency: ❌ (CV=15% across batches)
Sequence Length Consistency: ✅
Input Pattern Consistency: ✅
Overall Score: 2/4
```

**Interpretation**: Performance varies significantly across batch sizes. Possible causes:
- Batch size affects memory layout decisions
- Different kernels selected for different batch sizes
- Batch-dependent reorder logic

**Next Steps**:
1. Profile individual batch scenarios (batch=1 vs batch=8)
2. Check if different DNNL primitives are selected
3. Review weight reorder caching logic for batch-dependent behavior
4. May require batch-specific tuning or additional optimizations

### Scenario 4: Input Pattern Sensitivity ⚠️

```
High Variance Scenarios:
  - All "small_values" scenarios show CV > 5%
Batch Size Consistency: ✅
Sequence Length Consistency: ✅
Input Pattern Consistency: ⚠️ (small_values pattern differs)
Overall Score: 3/4
```

**Interpretation**: Small magnitude activations trigger different behavior. Possible causes:
- Quantization thresholds
- Numerical precision issues
- Different reorder decisions for small values

**Next Steps**:
1. Check if small values are realistic for transformer models
2. Review quantization logic for magnitude-dependent decisions
3. Consider this acceptable if small_values pattern is unrealistic
4. Document as known limitation if necessary

---

## Success Criteria

### Must-Pass Criteria (4/4 required for full validation)

1. ✅ **No High Variance Scenarios (CV < 5%)**
   - All scenarios show stable, repeatable metrics
   - Measurement noise is minimal

2. ✅ **Batch Size Consistency (CV < 10%)**
   - Optimization improvement maintained across batch sizes 1, 4, 8
   - At least 90% of baseline improvement retained

3. ✅ **Sequence Length Consistency (CV < 10%)**
   - Optimization improvement maintained across sequence lengths 1, 32, 128, 256
   - At least 90% of baseline improvement retained

4. ✅ **Input Pattern Independence (CV < 5%)**
   - Random, ones, and small_values patterns show similar results
   - No data-dependent reorder behavior

### Nice-to-Have Criteria (not blocking)

- ⭐ **Super-Linear Scaling**: Large batch/sequence scenarios show better improvement than small ones
- ⭐ **Perfect Additivity**: Input-side + output-side improvements exactly add up in combined build
- ⭐ **Zero Variance**: CV < 1% for all scenarios (extremely stable)

---

## Troubleshooting

### Issue: High Variance in All Scenarios

**Symptoms**: CV > 10% across all scenarios

**Possible Causes**:
1. **Measurement Noise**: System load, thermal throttling, background processes
2. **DNNL Non-Determinism**: oneDNN primitives have inherent variance
3. **Insufficient Repetitions**: 3 repetitions may not be enough

**Solutions**:
- Run on dedicated system with minimal background load
- Increase repetitions to 5 or 10
- Use `--iterations 100` for more stable measurements
- Check CPU governor settings (use "performance" mode)

### Issue: Batch Size Inconsistency

**Symptoms**: Performance varies significantly across batch sizes

**Possible Causes**:
1. **Different Kernels**: oneDNN selects different primitives for different batch sizes
2. **Memory Layout**: Batch size affects memory access patterns
3. **Cache Effects**: Larger batches may exceed cache capacity

**Solutions**:
- Profile each batch size individually with `perf` or VTune
- Check DNNL primitive selection with `DNNL_VERBOSE=2`
- Review weight reorder caching logic for batch-dependent paths
- Consider batch-specific optimizations if necessary

### Issue: Input Pattern Sensitivity

**Symptoms**: Different input patterns (random, ones, small_values) show different results

**Possible Causes**:
1. **Quantization**: Different patterns may trigger different quantization behavior
2. **Numerical Precision**: Small values may hit precision limits
3. **Compiler Optimizations**: FP32 fast-math may behave differently

**Solutions**:
- Analyze which pattern deviates (usually small_values)
- Check if pattern is realistic (transformers rarely have all-ones activations)
- Review quantization logic for magnitude-dependent decisions
- Document as known limitation if pattern is unrealistic

### Issue: Traces Not Captured

**Symptoms**: Trace files are empty or missing

**Possible Causes**:
1. **DNNL_VERBOSE Not Set**: Environment variable not propagated
2. **Model Compilation Failed**: OpenVINO failed to load model
3. **Disk Space**: Insufficient disk space for trace files

**Solutions**:
- Check that `scripts/capture_onednn_trace.py` is used (handles DNNL_VERBOSE correctly)
- Verify model file exists and is valid
- Check disk space: `df -h`
- Review logs in `multi_scenario_validation/logs/`

---

## Advanced Usage

### Custom Test Matrix

Edit `scripts/capture_multi_scenario_traces.sh` to customize:

```bash
# Custom batch sizes
BATCH_SIZES=(1 2 4 8 16)

# Custom sequence lengths
SEQ_LENGTHS=(1 16 32 64 128 256 512)

# Custom input patterns (requires extending capture_onednn_trace.py)
INPUT_PATTERNS=("random" "ones" "small_values" "zeros" "custom")
```

### Running Single Scenario

```bash
# Capture single scenario
python3 scripts/capture_onednn_trace.py \
    --model-path ./model/transformer_block.xml \
    --output-dir ./single_scenario \
    --tag test \
    --batch-size 4 \
    --seq-length 128 \
    --input-pattern random \
    --iterations 50

# Extract metrics
python3 scripts/parse_onednn_reorders.py \
    --trace ./single_scenario/onednn_trace_test.txt \
    --output-json ./single_scenario/metrics.json
```

### Comparing Baseline vs Optimized

```bash
# Capture baseline (research branch)
./scripts/capture_multi_scenario_traces.sh \
    --output-dir ./multi_scenario_baseline

# Capture optimized (with optimizations)
./scripts/capture_multi_scenario_traces.sh \
    --output-dir ./multi_scenario_optimized

# Analyze with baseline comparison
python3 scripts/analyze_multi_scenario_statistics.py \
    --metrics-dir ./multi_scenario_optimized/metrics \
    --baseline-dir ./multi_scenario_baseline/metrics \
    --output ./COMPARISON_ANALYSIS.md
```

### CI/CD Integration

```yaml
# .github/workflows/multi_scenario_validation.yml
name: Multi-Scenario Validation

on:
  pull_request:
    branches: [main]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install Dependencies
        run: pip install torch transformers openvino numpy
      
      - name: Run Quick Multi-Scenario Test
        run: |
          chmod +x scripts/capture_multi_scenario_traces.sh
          ./scripts/capture_multi_scenario_traces.sh --quick --skip-extraction
      
      - name: Analyze Results
        run: |
          python3 scripts/analyze_multi_scenario_statistics.py \
            --metrics-dir ./multi_scenario_validation/metrics \
            --output ./ANALYSIS.md
      
      - name: Check Success Criteria
        run: |
          # Fail if high variance detected
          if grep -q "❌" ./ANALYSIS.md; then
            echo "Multi-scenario validation failed"
            exit 1
          fi
      
      - name: Upload Report
        uses: actions/upload-artifact@v3
        with:
          name: multi-scenario-report
          path: ./ANALYSIS.md
```

---

## Appendix: Expected Results (Qwen2-1.5B)

### Baseline (Research Branch)

```
Batch=1, Seq=128, Pattern=random:
  Total Reorder Count: ~180 per block
  Total Reorder Time: ~5.2 ms per block
  Breakdown:
    - Weight reorders: ~150 (5.0 ms)
    - Activation reorders: ~20 (0.15 ms)
    - Scale/ZP reorders: ~10 (0.05 ms)
```

### Optimized (Input + Output)

```
Batch=1, Seq=128, Pattern=random:
  Total Reorder Count: ~10 per block
  Total Reorder Time: ~0.05 ms per block
  Breakdown:
    - Weight reorders: ~0 (0.0 ms) ✅ Pre-reordered
    - Activation reorders: ~0 (0.0 ms) ✅ Plain layout
    - Scale/ZP reorders: ~10 (0.05 ms) ⚪ Unchanged
```

### Expected Variance

```
Across 3 repetitions:
  Reorder Count CV: < 2% (very stable, integer counts)
  Reorder Time CV: < 3% (stable, minimal noise)

Across batch sizes:
  Count CV: < 5% (may scale slightly with batch)
  Time CV: < 8% (some scaling effects)

Across sequence lengths:
  Count CV: < 5% (mostly independent)
  Time CV: < 10% (some scaling at extreme lengths)

Across input patterns:
  Count CV: < 3% (should be identical)
  Time CV: < 5% (minor timing differences)
```

---

## Summary

Multi-scenario validation is a critical step in ensuring that layout optimizations are **robust, stable, and production-ready**. By testing across diverse inference conditions (batch sizes, sequence lengths, input patterns) with multiple repetitions, we gain confidence that optimizations:

1. ✅ Work consistently across realistic use cases
2. ✅ Don't introduce hidden regressions
3. ✅ Show stable, repeatable behavior
4. ✅ Are not sensitive to data-dependent effects

**Recommended Flow**:
1. Run quick mode first (8 scenarios, ~45 minutes)
2. Review results, check for major issues
3. Run full mode if quick mode passes (36 scenarios, ~3 hours)
4. Analyze comprehensive report
5. Validate success criteria (4/4 passing)
6. Document validated scenarios
7. Proceed with integration

For questions or issues, refer to the [Troubleshooting](#troubleshooting) section or review individual trace files in the `traces/` directory.

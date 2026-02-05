# Quick Start: Combined Optimization Validation

**Goal**: Validate that input-side + output-side optimizations compound positively  
**Time**: ~10 minutes  
**Task**: 29/32

---

## Prerequisites

```bash
# Install dependencies
pip install torch transformers openvino numpy

# Make scripts executable
chmod +x scripts/*.sh
```

---

## One-Command Quick Start

```bash
# Interactive guided validation
./scripts/example_combined_validation.sh
```

This handles everything automatically:
- ✅ Checks for required baseline and individual optimization traces
- ✅ Captures combined optimization trace (3 runs × 50 iterations)
- ✅ Generates comprehensive comparison report with additivity analysis
- ✅ Displays results and recommendations

---

## Manual 3-Step Process

### Step 1: Capture Combined Trace

```bash
./scripts/capture_combined_trace.sh \
    --output-dir ./combined_validation \
    --num-runs 3 \
    --iterations 50
```

**What it does**: Captures oneDNN traces from fully optimized build (both input-side and output-side optimizations enabled)

**Output**: 
- `./combined_validation/traces_combined/` - Raw traces
- `./combined_validation/metrics_combined/` - Parsed metrics (CSV + JSON)

### Step 2: Generate Comparison Report

**Basic comparison** (baseline vs combined only):
```bash
python3 scripts/compare_combined_traces.py \
    --baseline ./baseline_capture/metrics \
    --combined ./combined_validation/metrics_combined \
    --output ./combined_validation/COMBINED_COMPARISON.md
```

**Full comparison** (with additivity analysis):
```bash
python3 scripts/compare_combined_traces.py \
    --baseline ./baseline_capture/metrics \
    --combined ./combined_validation/metrics_combined \
    --input-side ./input_side_validation/metrics_optimized \
    --output-side ./output_side_validation/metrics_optimized \
    --output ./combined_validation/COMBINED_COMPARISON.md
```

### Step 3: Review Results

```bash
cat ./combined_validation/COMBINED_COMPARISON.md
```

**Check for**:
- ✅ Total improvement > individual optimizations
- ✅ Additivity ratio 80-105% (if full comparison)
- ✅ Weight reorders reduced (input-side working)
- ✅ Activation reorders reduced or stable (output-side working)
- ✅ No regressions in scale/ZP dimensions
- ✅ Success criteria: 7/7 or 8/8 passed

---

## Expected Results

### Scenario A: Both Optimizations Active

```markdown
## Executive Summary

| Metric | Baseline | Combined | Reduction | Improvement % |
|--------|----------|----------|-----------|---------------|
| Total Reorder Count | 14 | 2 | 12 | 85.7% |
| Total Reorder Time (ms) | 5.20 | 0.12 | 5.08 | 97.7% |

## Additivity Analysis

| Metric | Value |
|--------|-------|
| Input-Side Reduction | 5.05 ms |
| Output-Side Reduction | 0.15 ms |
| Expected Combined Reduction | 5.20 ms |
| Actual Combined Reduction | 5.08 ms |
| Additivity Ratio | 97.7% |

✅ Result: Optimizations are perfectly additive
```

**Interpretation**: Both optimizations working as expected, no conflicts.

### Scenario B: Only Input-Side Active (Expected for Qwen2)

```markdown
## Executive Summary

| Metric | Baseline | Combined | Reduction | Improvement % |
|--------|----------|----------|-----------|---------------|
| Total Reorder Count | 12 | 2 | 10 | 83.3% |
| Total Reorder Time (ms) | 5.00 | 0.12 | 4.88 | 97.6% |

## Additivity Analysis

| Metric | Value |
|--------|-------|
| Input-Side Reduction | 4.88 ms |
| Output-Side Reduction | 0.00 ms |
| Expected Combined Reduction | 4.88 ms |
| Actual Combined Reduction | 4.88 ms |
| Additivity Ratio | 100.0% |

ℹ️  Note: Output-side shows 0.00 ms reduction
   Baseline already uses optimal output layouts (see Task 28)
```

**Interpretation**: Input-side optimization effective, output-side already optimal in baseline.

---

## Key Metrics to Check

### 1. Overall Improvement

```markdown
Total Reorder Time Reduction: 5.08 ms (97.7%)
```
**Target**: > 5% overall improvement  
**Status**: ✅ if > 5%, ❌ if < 5%

### 2. Weight Reorder Reduction (Input-Side)

```markdown
Weight Input Category: 5.00 ms → 0.10 ms (4.90 ms reduction)
```
**Target**: Baseline > Combined  
**Status**: ✅ if reduced, ❌ if increased

### 3. Activation Reorder Reduction (Output-Side)

```markdown
Activation Output Category: 0.15 ms → 0.00 ms (0.15 ms reduction)
```
**Target**: Baseline ≥ Combined  
**Status**: ✅ if reduced or stable, ❌ if increased

### 4. Additivity Ratio (if available)

```markdown
Additivity Ratio: 97.7%
```
**Target**: 80-105%  
**Status**: 
- ✅ if 95-105% (perfect)
- ✅ if 80-95% (acceptable)
- ⚠️ if 60-80% (investigate)
- ❌ if < 60% (conflicts)

### 5. Per-Dimension Improvements

```markdown
1536x1536: 1.41 ms → 0.00 ms (100% reduction)
8960x1536: 1.50 ms → 0.00 ms (100% reduction)
6x1536:    0.08 ms → 0.00 ms (100% reduction)
```
**Target**: All key dimensions show reduction  
**Status**: ✅ if all reduced, ⚠️ if some increased

### 6. Scale/ZP Stability

```markdown
Scale/ZP: 0.12 ms → 0.12 ms (stable)
```
**Target**: Should remain unchanged  
**Status**: ✅ if stable, ⚠️ if changed significantly

### 7. Success Criteria

```markdown
**Overall**: 8/8 criteria passed
```
**Target**: ≥ 6/8 passed  
**Status**: ✅ if ≥ 6, ⚠️ if 4-5, ❌ if < 4

---

## File Structure

```
./combined_validation/
├── model/
│   ├── transformer_block.xml      # Extracted model
│   └── transformer_block.bin
├── traces_combined/
│   ├── onednn_trace_combined_run1.txt   # Raw oneDNN traces
│   ├── onednn_trace_combined_run2.txt
│   └── onednn_trace_combined_run3.txt
├── metrics_combined/
│   ├── combined_run1_metrics.csv        # Parsed metrics
│   ├── combined_run1_metrics.json
│   ├── combined_run2_metrics.csv
│   ├── combined_run2_metrics.json
│   ├── combined_run3_metrics.csv
│   └── combined_run3_metrics.json
├── logs/
│   ├── extraction_combined.log          # Detailed logs
│   ├── capture_combined_run1.log
│   └── parse_combined_run1.log
└── COMBINED_COMPARISON.md               # Final report
```

---

## Troubleshooting

### Problem: "No baseline metrics found"

**Solution**:
```bash
# Capture baseline first (from Task 2)
./scripts/capture_baseline_trace.sh \
    --output-dir ./baseline_capture \
    --num-runs 3
```

### Problem: "Combined improvement ≈ 0%"

**Possible causes**:
1. Optimizations not compiled into build
2. Using wrong OpenVINO binary (baseline instead of optimized)
3. Test configuration differs from baseline

**Solution**:
```bash
# Verify build has optimizations
grep -r "PRE_REORDER" src/plugins/intel_cpu/
grep -r "plain.*layout" src/plugins/intel_cpu/

# Rebuild with optimizations
cmake -B build -DENABLE_OPTIMIZATIONS=ON
cmake --build build -j$(nproc)
```

### Problem: "Additivity ratio < 80%"

**Possible causes**:
1. Conflicting optimizations
2. Cache invalidation issues
3. Implementation bugs

**Solution**:
```bash
# Review individual optimization results
cat ./input_side_validation/INPUT_SIDE_COMPARISON.md
cat ./output_side_validation/OUTPUT_SIDE_COMPARISON.md

# Check for conflicts in combined trace
grep "unexpected" ./combined_validation/COMBINED_COMPARISON.md
```

### Problem: "Model extraction fails"

**Solution**:
```bash
# Use existing model from previous tasks
./scripts/capture_combined_trace.sh \
    --skip-extraction \
    --output-dir ./combined_validation
```

---

## Next Steps

After validating combined optimizations:

1. **If successful** (criteria passed):
   - Proceed to Task 30: Test with multiple trace scenarios
   - Consider production deployment

2. **If partially successful** (some criteria failed):
   - Review individual optimization results (Tasks 27-28)
   - Investigate specific dimension regressions
   - Check additivity analysis for interaction effects

3. **If unsuccessful** (most criteria failed):
   - Verify both optimizations are compiled in
   - Re-capture baseline to ensure consistency
   - Check for build configuration issues

---

## Related Documentation

- **Full Guide**: [COMBINED_OPTIMIZATION_COMPARISON_GUIDE.md](COMBINED_OPTIMIZATION_COMPARISON_GUIDE.md)
- **Task 27**: [INPUT_SIDE_VALIDATION_GUIDE.md](INPUT_SIDE_VALIDATION_GUIDE.md)
- **Task 28**: [OUTPUT_SIDE_COMPARISON_GUIDE.md](OUTPUT_SIDE_COMPARISON_GUIDE.md)
- **Task 2**: [scripts/BASELINE_QUICKSTART.md](scripts/BASELINE_QUICKSTART.md)

---

## Success Criteria Summary

| Criterion | Target | Weight |
|-----------|--------|--------|
| Total improvement | > 5% | Critical |
| Weight reorder reduction | Baseline > Combined | Critical |
| Activation reorder reduction | Baseline ≥ Combined | Important |
| Per-dimension improvements | All key dims reduced | Important |
| No regressions | Scale/ZP stable | Important |
| Additivity ratio | 80-105% | Optional |
| Reproducibility | Variance < 5% | Nice-to-have |

**Pass threshold**: ≥ 5/7 criteria (or 6/8 if additivity included)

---

**Last Updated**: 2025-01-21  
**Task Status**: 29/32 - Ready for execution  
**Estimated Time**: 10 minutes

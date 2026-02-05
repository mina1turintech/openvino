# Quick Start: Output-Side Trace Comparison

**Task 28/32**: Compare baseline vs. output-side optimized traces  
**Time**: ~10-15 minutes  
**Goal**: Quantify activation reorder reduction from output-side layout optimizations

---

## One-Command Quick Start

```bash
./scripts/example_compare_output_side.sh
```

**Output**: `OUTPUT_SIDE_COMPARISON.md` with full analysis

---

## 3-Step Manual Process

### Step 1: Capture Baseline (Research Branch)

```bash
# Capture baseline from research branch (if available)
git checkout research-branch
./build_openvino.sh

./scripts/capture_baseline_trace.sh \
    --num-runs 3 \
    --iterations 50 \
    --output-dir ./baseline_capture
```

**OR use existing baseline from Task 2**:
```bash
# If baseline_capture/ already exists from Task 2, skip to Step 2
ls -la ./baseline_capture/metrics/
```

---

### Step 2: Capture Optimized (Current Branch)

```bash
git checkout main
./build_openvino.sh

./scripts/capture_output_side_trace.sh \
    --num-runs 3 \
    --iterations 50 \
    --output-dir ./output_side_validation
```

---

### Step 3: Compare Traces

```bash
python3 scripts/compare_output_side_traces.py \
    --baseline ./baseline_capture/metrics \
    --optimized ./output_side_validation/metrics \
    --output ./OUTPUT_SIDE_COMPARISON.md
```

**View results**:
```bash
cat OUTPUT_SIDE_COMPARISON.md
```

---

## Expected Results

### If Baseline Has Blocked Outputs

```markdown
## Executive Summary

**Baseline Runs**: 3
**Optimized Runs**: 3

### Output-Side Specific Metrics (Activation Reorders)

| Metric | Baseline | Optimized | Reduction | Improvement % |
|--------|----------|-----------|-----------|---------------|
| Activation Reorder Count | 2 | 0 | 2 | 100% |
| Activation Reorder Time (ms) | 0.120 | 0.000 | 0.120 | 100% |

## 24-Layer Model Projection

- **Baseline 24-layer activation reorder time**: 2.880 ms
- **Optimized 24-layer activation reorder time**: 0.000 ms
- **Total activation reorder savings**: 2.880 ms (100%)
```

✅ **Optimization highly effective**: Activation reorders eliminated

---

### If Baseline Already Optimal

```markdown
## Executive Summary

**Baseline Runs**: 3
**Optimized Runs**: 3

### Output-Side Specific Metrics (Activation Reorders)

| Metric | Baseline | Optimized | Reduction | Improvement % |
|--------|----------|-----------|-----------|---------------|
| Activation Reorder Count | 0 | 0 | 0 | N/A |
| Activation Reorder Time (ms) | 0.000 | 0.000 | 0.000 | N/A |
```

✅ **Already optimal**: Baseline implementation uses plain output formats (as found in Task 26)

---

## Key Metrics to Check

### 1. Activation Reorders (Primary Focus)

Look for **6×1536 dimension** reorders in the comparison report:

```markdown
## Activation Reorder Analysis by Dimension

| Dimension | Context | Baseline Count | Optimized Count | Time Reduction (ms) |
|-----------|---------|----------------|-----------------|---------------------|
| 6x1536    | Attention/FFN output | 2 | 0 | 0.120 |
```

**Target**: Reduction from 2 → 0 per block (100%)

---

### 2. Weight Reorders (Stability Check)

Verify weight reorders are unchanged:

```markdown
## Weight Reorder Stability (Regression Check)

| Dimension | Baseline Time (ms) | Optimized Time (ms) | Status |
|-----------|-------------------|-------------------|--------|
| 1536x1536 | 1.413 | 1.408 | ✅ Stable |
| 256x1536  | 0.228 | 0.231 | ✅ Stable |
```

**Target**: All weight reorders show "✅ Stable" status

---

### 3. Success Criteria Validation

Check the success criteria table:

```markdown
## Success Criteria Validation

| Criterion | Status | Result |
|-----------|--------|--------|
| Attention output reorder reduction | ✅ PASS | 0.050 ms (100% reduction) |
| FFN output reorder reduction | ✅ PASS | Included in activation reduction above |
| Activation reorder count reduction | ✅ PASS | 2 operations eliminated |
| No weight reorder regressions | ✅ PASS | Weight reorders unchanged |
| Reproducibility across runs | ✅ PASS | 3 baseline runs, 3 optimized runs |

**Overall**: 5/5 criteria passed
```

**Target**: All criteria show "✅ PASS"

---

## Troubleshooting

### No baseline traces available

```bash
# Create new baseline from research branch
git checkout research-branch
./build_openvino.sh
./scripts/capture_baseline_trace.sh --num-runs 3 --output-dir ./baseline_capture
```

---

### Comparison tool fails

```bash
# Check metrics directories exist
ls -la ./baseline_capture/metrics/
ls -la ./output_side_validation/metrics/

# Re-parse traces if needed
python3 scripts/parse_onednn_reorders.py \
    --trace ./baseline_capture/traces/trace_run1.txt \
    --output-json ./baseline_capture/metrics/run1_metrics.json
```

---

### Results show no improvement

**If baseline already shows 0 activation reorders**:
- ✅ This is expected! The current OpenVINO implementation already uses optimal layouts.
- See `OUTPUT_SIDE_VALIDATION_REPORT.md` (Task 26) for details.

---

## What's Next?

1. **Review the full comparison report**: `OUTPUT_SIDE_COMPARISON.md`
2. **Understand the methodology**: `OUTPUT_SIDE_COMPARISON_GUIDE.md`
3. **See detailed baseline analysis**: `OUTPUT_SIDE_BASELINE_ANALYSIS.md`
4. **Next task**: Task 29 - Combined input+output optimization comparison

---

## Related Documentation

- **Comparison Guide**: `OUTPUT_SIDE_COMPARISON_GUIDE.md` - Full methodology
- **Baseline Analysis**: `OUTPUT_SIDE_BASELINE_ANALYSIS.md` - Expected patterns
- **Validation Report**: `OUTPUT_SIDE_VALIDATION_REPORT.md` (Task 26) - Current state
- **Tool README**: `scripts/OUTPUT_SIDE_VALIDATION_README.md` - Tool documentation

---

## Summary

Output-side optimizations eliminate **activation reorder overhead** at block boundaries by using plain `f32::ab` format for attention and FFN outputs. This saves ~2.4-3.8 ms per inference on a 24-layer model (assuming baseline has blocked outputs).

**Current State**: OpenVINO already uses optimal plain output formats (validated in Task 26).

# Quick Start: Output-Side Layout Optimization Validation

**Task 26/32** - Fast track guide for validating output-side optimizations

---

## TL;DR - Run This

```bash
# One command to validate everything
./scripts/capture_output_side_trace.sh
```

**Output**: `./output_side_validation/OUTPUT_SIDE_VALIDATION.md`  
**Time**: ~15 minutes (first run), ~5 minutes (subsequent)  
**Expected Result**: ✅ All validation criteria PASSED

---

## What Gets Validated

✅ Attention output uses `f32::ab` format (0 reorders)  
✅ FFN output uses `f32::ab` format (0 reorders)  
✅ Block boundaries have zero reorder overhead  
✅ No regressions in other operations  
✅ End-to-end inference executes correctly

---

## Expected Results

### ✅ Optimal State (Current)

```
Validation Summary:
✅ All criteria PASSED

- Attention output reorders: 0
- FFN output reorders: 0  
- Block boundary reorders: 0
- Total activation reorders: 0

Status: OPTIMAL
```

### ⚠️ Needs Review

```
Validation Summary:
⚠️  Some criteria need review

- Attention output reorders: 2-4
- FFN output reorders: 2-4
- Block boundary reorders: 1-2

Status: OPTIMIZATION NEEDED
```

---

## Quick Commands

### Basic Validation
```bash
./scripts/capture_output_side_trace.sh
```

### Custom Layer
```bash
./scripts/capture_output_side_trace.sh --layer-index 12
```

### Reuse Model
```bash
./scripts/capture_output_side_trace.sh --skip-extraction
```

### Python Tool
```bash
python3 scripts/validate_output_side_optimizations.py \
    --trace trace.txt \
    --output report.md
```

### With Baseline
```bash
python3 scripts/validate_output_side_optimizations.py \
    --trace output_trace.txt \
    --baseline baseline_trace.txt \
    --output comparison.md
```

---

## View Results

```bash
# Main validation report
cat ./output_side_validation/OUTPUT_SIDE_VALIDATION.md

# Metrics (JSON)
cat ./output_side_validation/metrics/output_side_run1_metrics.json | jq

# Metrics (CSV)
column -t -s, ./output_side_validation/metrics/output_side_run1_metrics.csv
```

---

## Troubleshooting

### No oneDNN output in trace?
```bash
export DNNL_VERBOSE=1
./scripts/capture_output_side_trace.sh
```

### Model extraction fails?
```bash
# Check network
ping huggingface.co

# Check disk space
df -h

# Manual extraction
python3 scripts/extract_transformer_block.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --layer-index 0 \
    --output-dir ./test_model
```

### Unexpected reorders found?
```bash
# View reorder details
grep "reorder.*6x1536" ./output_side_validation/traces/*.txt

# Check if activation or weight reorders
# Activation: Small first dimension (1-16)
# Weight: Large first dimension (>100)
```

---

## More Information

- **Full guide**: `scripts/OUTPUT_SIDE_VALIDATION_README.md`
- **Detailed report**: `OUTPUT_SIDE_VALIDATION_REPORT.md`
- **Summary**: `OUTPUT_SIDE_VALIDATION_SUMMARY.md`
- **Examples**: `scripts/example_output_side_validation.sh`

---

## Status: ✅ COMPLETE

Output-side layout optimizations are **validated and optimal**:
- Zero activation reorders at attention output
- Zero activation reorders at FFN output
- Zero reorders at block boundaries
- No regressions detected

**No further action required for output-side operations.**

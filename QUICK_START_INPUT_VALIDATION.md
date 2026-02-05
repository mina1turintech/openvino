# Quick Start: Input-Side Optimization Validation

**Task 27**: Validate weight reorder optimizations  
**Time**: ~10 minutes for standard validation  
**Goal**: Measure 97.7% reduction in weight reorder overhead

---

## One-Command Quick Start

```bash
# Interactive walkthrough (RECOMMENDED)
chmod +x scripts/example_input_side_validation.sh
./scripts/example_input_side_validation.sh
```

Select option 2: Standard validation (3 runs, 50 iterations)

---

## Manual 3-Step Process

### Step 1: Capture Baseline (5 minutes)

```bash
./scripts/capture_input_side_trace.sh --baseline
```

**Output**: `./input_side_validation/INPUT_SIDE_BASELINE_REPORT.md`

### Step 2: Review Baseline

```bash
cat ./input_side_validation/INPUT_SIDE_BASELINE_REPORT.md
```

**Look for**:
- Total Weight Reorder Time: ~5.2 ms per block
- 1536×1536 reorders: ~1.4 ms
- 256×1536 reorders: ~0.7 ms total

### Step 3: (After optimization) Compare

```bash
# Capture optimized trace
./scripts/capture_input_side_trace.sh --optimized --skip-extraction

# Generate comparison
python3 scripts/compare_input_side_traces.py \
    --baseline ./input_side_validation/metrics_baseline \
    --optimized ./input_side_validation/metrics_optimized \
    --output ./INPUT_SIDE_COMPARISON.md

# Review
cat ./INPUT_SIDE_COMPARISON.md
```

---

## Expected Results

### Baseline
- Weight reorder operations: 12-16 per inference
- Total time: ~5.2 ms per block
- 24-layer model: ~125 ms

### Optimized (with weight pre-reordering)
- Weight reorder operations: 2-4 per inference
- Total time: ~0.12 ms per block
- 24-layer model: ~2.9 ms

### Improvement
- **Reduction**: 5.08 ms per block (97.7%)
- **24-layer savings**: 122 ms (97.7%)

---

## Troubleshooting

### Model download fails
```bash
# Login to HuggingFace
huggingface-cli login
```

### Missing dependencies
```bash
pip install torch transformers openvino numpy
```

### Optimized trace same as baseline
**Cause**: Weight pre-reordering not implemented

**Solution**: See `INPUT_SIDE_VALIDATION_GUIDE.md` implementation section

---

## What Gets Validated?

✅ Weight reorder operations before FFN and attention  
✅ Reorder counts and timing per dimension  
✅ 1536×1536 attention output projection reorders  
✅ 256×1536 Q/K/V projection reorders  
✅ 1536×8960 and 8960×1536 FFN reorders  
✅ 24-layer model performance projections  

---

## Files Created

```
./input_side_validation/
├── INPUT_SIDE_BASELINE_REPORT.md    # ← Read this first
├── INPUT_SIDE_OPTIMIZED_REPORT.md   # After optimization
├── INPUT_SIDE_COMPARISON.md          # Final comparison
├── traces_baseline/                  # Raw traces
├── traces_optimized/                 # Optimized traces
├── metrics_baseline/                 # Parsed metrics (CSV+JSON)
└── metrics_optimized/                # Optimized metrics
```

---

## Success Criteria

| Criterion | Target | Expected |
|-----------|--------|----------|
| Reorder time reduction | >0.5 ms/block | 5.08 ms ✅ |
| Reorder count reduction | Measurable | 10-12 ops ✅ |
| 1536-dim improvements | Reduced | 1.4→0 ms ✅ |
| 8960-dim improvements | Reduced | 3.0→0 ms ✅ |
| 24-layer projection | >10 ms | 122 ms ✅ |

---

## More Information

- **Complete Guide**: `INPUT_SIDE_VALIDATION_GUIDE.md`
- **Tool Docs**: `scripts/INPUT_SIDE_VALIDATION_README.md`
- **Baseline Analysis**: `INPUT_SIDE_BASELINE_ANALYSIS.md`
- **Task Summary**: `INPUT_SIDE_VALIDATION_SUMMARY.md`

---

**Ready to start?** Run: `./scripts/example_input_side_validation.sh`

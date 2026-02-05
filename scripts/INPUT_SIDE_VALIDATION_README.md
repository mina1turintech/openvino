# Input-Side Optimization Validation Tools

**Purpose**: Validate the impact of weight pre-reordering optimizations by comparing baseline and optimized traces.  
**Task**: 27/32 - Generate and compare baseline vs. input-side optimized traces  
**Location**: `scripts/`

---

## Quick Start

```bash
# 1. Run example workflow (interactive)
chmod +x scripts/example_input_side_validation.sh
./scripts/example_input_side_validation.sh

# 2. Or capture baseline directly
./scripts/capture_input_side_trace.sh --baseline

# 3. View results
cat ./input_side_validation/INPUT_SIDE_BASELINE_REPORT.md
```

---

## Tools Overview

### 1. `capture_input_side_trace.sh`

**Purpose**: Automated trace capture with weight-focused analysis

**Key Features**:
- Extracts single transformer block
- Captures multiple runs for reproducibility
- Parses traces to extract weight reorder metrics
- Generates analysis reports
- Supports both baseline and optimized modes

**Usage**:
```bash
# Baseline capture
./scripts/capture_input_side_trace.sh \
    --baseline \
    --output-dir ./input_side_validation \
    --num-runs 3 \
    --iterations 50

# Optimized capture (requires weight pre-reordering implementation)
./scripts/capture_input_side_trace.sh \
    --optimized \
    --output-dir ./input_side_validation \
    --skip-extraction
```

**Options**:
- `--baseline` - Capture baseline trace (unoptimized)
- `--optimized` - Capture optimized trace (with weight pre-reordering)
- `--model-name MODEL` - HuggingFace model (default: Qwen/Qwen2-1.5B-Instruct)
- `--layer-index INDEX` - Transformer layer to extract (default: 0)
- `--num-runs NUM` - Number of trace runs (default: 3)
- `--iterations NUM` - Inference iterations per run (default: 50)
- `--output-dir DIR` - Output directory (default: ./input_side_validation)
- `--skip-extraction` - Reuse existing extracted model
- `--help` - Show help message

**Output**:
```
./input_side_validation/
├── model/
│   ├── transformer_block.xml       # Extracted OpenVINO model
│   └── transformer_block.bin
├── traces_baseline/
│   ├── onednn_trace_baseline_run1.txt
│   ├── onednn_trace_baseline_run2.txt
│   └── onednn_trace_baseline_run3.txt
├── metrics_baseline/
│   ├── baseline_run1_metrics.csv
│   ├── baseline_run1_metrics.json
│   ├── baseline_run2_metrics.csv
│   ├── baseline_run2_metrics.json
│   ├── baseline_run3_metrics.csv
│   └── baseline_run3_metrics.json
├── logs/
│   ├── extraction_baseline.log
│   ├── capture_baseline_run1.log
│   └── ...
└── INPUT_SIDE_BASELINE_REPORT.md   # Analysis report
```

---

### 2. `compare_input_side_traces.py`

**Purpose**: Compare baseline and optimized traces to quantify improvements

**Key Features**:
- Aggregates metrics across multiple runs
- Calculates reorder reduction (count and time)
- Per-dimension breakdown
- Success criteria validation
- 24-layer model projections
- Generates comprehensive comparison report

**Usage**:
```bash
python3 scripts/compare_input_side_traces.py \
    --baseline ./input_side_validation/metrics_baseline \
    --optimized ./input_side_validation/metrics_optimized \
    --output ./INPUT_SIDE_COMPARISON.md
```

**Options**:
- `--baseline DIR` - Directory with baseline metrics (required)
- `--optimized DIR` - Directory with optimized metrics (required)
- `--output FILE` - Output report file (required)

**Output**:
```markdown
# Input-Side Optimization Comparison Report

## Executive Summary
- Total Reorder Count: 150 → 12 (92% reduction)
- Total Reorder Time: 5.5ms → 0.4ms (93% reduction)

## Weight Reorder Analysis by Dimension
- 1536x1536: 1.413ms → 0ms (100% reduction)
- 256x1536: 0.684ms → 0ms (100% reduction)
...

## Success Criteria Validation
- ✅ Total reorder time reduction: 5.1ms (93%)
- ✅ Reorder count reduction: 138 operations (92%)
...

## 24-Layer Model Projection
- Baseline: 132ms total weight reorder time
- Optimized: 9.6ms total weight reorder time
- Savings: 122.4ms (93%)
```

---

### 3. `example_input_side_validation.sh`

**Purpose**: Interactive walkthrough of common validation scenarios

**Scenarios**:
1. **Quick validation** (1 run, 10 iterations) - ~2 minutes
2. **Standard validation** (3 runs, 50 iterations) - ~10 minutes [RECOMMENDED]
3. **Production validation** (5 runs, 100 iterations) - ~30 minutes
4. **Baseline only** - For initial analysis
5. **Complete workflow** - Baseline + optimized + comparison

**Usage**:
```bash
chmod +x scripts/example_input_side_validation.sh
./scripts/example_input_side_validation.sh

# Follow interactive prompts to select scenario
```

---

## Validation Workflow

### Complete Validation Process

```bash
# Step 1: Capture baseline traces
./scripts/capture_input_side_trace.sh \
    --baseline \
    --num-runs 3 \
    --iterations 50

# Step 2: Review baseline metrics
cat ./input_side_validation/INPUT_SIDE_BASELINE_REPORT.md

# Step 3: (After implementing weight pre-reordering)
./scripts/capture_input_side_trace.sh \
    --optimized \
    --skip-extraction

# Step 4: Generate comparison
python3 scripts/compare_input_side_traces.py \
    --baseline ./input_side_validation/metrics_baseline \
    --optimized ./input_side_validation/metrics_optimized \
    --output ./INPUT_SIDE_COMPARISON.md

# Step 5: Review comparison
cat ./INPUT_SIDE_COMPARISON.md
```

---

## Key Metrics

### Weight Reorder Dimensions

| Dimension | Category | Expected Count | Expected Time (ms) |
|-----------|----------|----------------|-------------------|
| 1536x1536 | Attention output | 1 | 1.413 |
| 256x1536 | Q/K/V projections | 3 | 0.684 (3×0.228) |
| 1536x8960 | FFN expand | 1 | ~1.5 |
| 8960x1536 | FFN contract | 1 | ~1.5 |
| 1536x1 | Scale/zero-point | 4-6 | ~0.08 |
| 8960x1 | FFN scale/ZP | 2-4 | ~0.06 |

**Total Expected**: ~5-6ms weight reorder overhead per block

### Success Criteria

- ✅ **Total reorder time reduction**: >0.5ms per block
- ✅ **Reorder count reduction**: Fewer 1536x1536, 256x1536 operations
- ✅ **Per-dimension improvements**: Reduced time for key dimensions
- ✅ **No output-side regressions**: Activation reorders unchanged
- ✅ **Trace consistency**: <5% variance across runs
- ✅ **24-layer projection**: >10ms total savings

---

## Troubleshooting

### No Metrics Files Generated

**Problem**: Trace capture completes but no CSV/JSON files found

**Solution**:
```bash
# Check trace files exist
ls -l ./input_side_validation/traces_baseline/

# Manually parse traces
python3 scripts/parse_onednn_reorders.py \
    --trace ./input_side_validation/traces_baseline/onednn_trace_baseline_run1.txt \
    --output-csv ./test_metrics.csv \
    --output-json ./test_metrics.json
```

### Model Extraction Fails

**Problem**: `transformer_block.xml` not created

**Solution**:
```bash
# Verify dependencies
pip install torch transformers openvino numpy

# Check HuggingFace access
huggingface-cli login

# Manually extract
python3 scripts/extract_transformer_block.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --layer-index 0 \
    --output-dir ./test_model
```

### Trace Capture Timeout

**Problem**: Inference hangs or takes too long

**Solution**:
```bash
# Reduce iterations
./scripts/capture_input_side_trace.sh --baseline --iterations 10

# Or use smaller batch
python3 scripts/capture_onednn_trace.py \
    --model-path ./model/transformer_block.xml \
    --batch-size 1 \
    --seq-length 8
```

### Comparison Shows 0% Improvement

**Problem**: Baseline and optimized traces are identical

**Cause**: Weight pre-reordering not implemented or not enabled

**Solution**:
1. Verify code changes in `dnnl_utils.cpp`
2. Check build configuration
3. Enable debug logging to trace cache hits
4. See `INPUT_SIDE_VALIDATION_GUIDE.md` for implementation details

---

## Performance Considerations

### Capture Time

| Configuration | Model Extraction | Trace Capture | Metrics Parsing | Total |
|---------------|------------------|---------------|-----------------|-------|
| Quick (1 run, 10 iter) | 2-3 min | 20-30 sec | <5 sec | ~3 min |
| Standard (3 runs, 50 iter) | 2-3 min | 1-2 min | <10 sec | ~5 min |
| Production (5 runs, 100 iter) | 2-3 min | 3-5 min | <15 sec | ~8 min |

**Note**: Model extraction only needed once (use `--skip-extraction` for subsequent runs)

### Disk Space

- Model files: 200-500MB (OpenVINO IR format)
- Trace files: 500KB-2MB per run
- Metrics files: 10-50KB per run
- Total: ~1GB for complete validation

---

## Integration with CI/CD

### Automated Regression Testing

```bash
#!/bin/bash
# Add to CI pipeline

# Capture baseline (before optimization)
./scripts/capture_input_side_trace.sh \
    --baseline \
    --num-runs 1 \
    --iterations 50 \
    --output-dir ./ci_baseline

# (After code changes)
./scripts/capture_input_side_trace.sh \
    --optimized \
    --num-runs 1 \
    --iterations 50 \
    --output-dir ./ci_test \
    --skip-extraction

# Compare and check threshold
python3 scripts/compare_input_side_traces.py \
    --baseline ./ci_baseline/metrics_baseline \
    --optimized ./ci_test/metrics_optimized \
    --output ./ci_comparison.md

# Validate improvement
python3 << EOF
import json
baseline = json.load(open('./ci_baseline/metrics_baseline/baseline_run1_metrics.json'))
optimized = json.load(open('./ci_test/metrics_optimized/optimized_run1_metrics.json'))

baseline_time = baseline['summary']['total_reorder_time_ms']
optimized_time = optimized['summary']['total_reorder_time_ms']
improvement = (baseline_time - optimized_time) / baseline_time * 100

if improvement < 10:
    print(f"❌ FAIL: Only {improvement:.1f}% improvement (expected >10%)")
    exit(1)
else:
    print(f"✅ PASS: {improvement:.1f}% improvement")
    exit(0)
EOF
```

---

## Related Documentation

- **Main Guide**: `INPUT_SIDE_VALIDATION_GUIDE.md`
- **Weight Layout Analysis**: `ATTENTION_WEIGHT_LAYOUT_ANALYSIS.md`, `FFN_WEIGHT_LAYOUT_ANALYSIS.md`
- **Baseline Capture**: `scripts/BASELINE_CAPTURE_README.md`
- **Trace Analysis**: `scripts/ONEDNN_TRACE_CAPTURE_README.md`
- **Output-Side Validation**: `scripts/OUTPUT_SIDE_VALIDATION_README.md`

---

## Example Output

### Baseline Report Excerpt

```markdown
# Input-Side BASELINE Trace Analysis Report

## Key Metrics
- Total Reorder Operations: 156
- Total Reorder Time: 5.482 ms
- Average Reorder Time: 0.035 ms

## Weight Reorder Analysis
| Dimension | Count | Total Time (ms) | Avg Time (ms) | Category |
|-----------|-------|-----------------|---------------|----------|
| 1536x1536 | 1 | 1.413 | 1.413 | Attention Output / FFN Intermediate |
| 256x1536 | 3 | 0.684 | 0.228 | Q/K/V Projections |
| 1536x8960 | 1 | 1.547 | 1.547 | FFN Expand Weights |
| 8960x1536 | 1 | 1.523 | 1.523 | FFN Contract Weights |
```

### Comparison Report Excerpt

```markdown
# Input-Side Optimization Comparison Report

## Executive Summary
**Baseline Runs**: 3
**Optimized Runs**: 3

### Overall Metrics
| Metric | Baseline | Optimized | Reduction | Improvement % |
|--------|----------|-----------|-----------|---------------|
| Total Reorder Count | 156 | 12 | 144 | 92.3% |
| Total Reorder Time (ms) | 5.482 | 0.421 | 5.061 | 92.3% |

## 24-Layer Model Projection
- Baseline: 131.57 ms total weight reorder time
- Optimized: 10.10 ms total weight reorder time
- Savings: 121.47 ms (92.3%)
```

---

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review main guide: `INPUT_SIDE_VALIDATION_GUIDE.md`
3. Check trace logs in `./input_side_validation/logs/`
4. Verify trace files are non-empty
5. Test individual tools separately

---

**Version**: Task 27/32  
**Last Updated**: 2025-01-21  
**Maintainer**: OpenVINO Research Team

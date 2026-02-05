# Quick Start: Multi-Scenario Validation

**Task 30/32**: Test layout optimizations across diverse inference conditions  
**Time Required**: 45 minutes (quick mode) or 3-4 hours (full mode)  
**Model**: Qwen2-1.5B-Instruct Transformer Block

---

## One-Command Quick Start

```bash
# Interactive guided validation (recommended)
./scripts/example_multi_scenario_validation.sh
```

This single command will:
1. ✅ Check for existing model extraction
2. ✅ Run multi-scenario trace capture (asks for quick/full mode)
3. ✅ Generate statistical analysis report
4. ✅ Display results summary
5. ✅ Provide recommendations

---

## Three-Step Manual Process

### Prerequisites

```bash
pip install torch transformers openvino numpy
chmod +x scripts/capture_multi_scenario_traces.sh
```

### Step 1: Capture Traces (Quick Mode)

```bash
./scripts/capture_multi_scenario_traces.sh --quick
```

**Quick Mode Configuration**:
- Batch sizes: [1, 4]
- Sequence lengths: [1, 128]
- Input patterns: [random, ones]
- **Total**: 8 scenarios × 3 repetitions = 24 runs (~45 minutes)

**Full Mode** (remove `--quick`):
- Batch sizes: [1, 4, 8]
- Sequence lengths: [1, 32, 128, 256]
- Input patterns: [random, ones, small_values]
- **Total**: 36 scenarios × 3 repetitions = 108 runs (~3-4 hours)

### Step 2: Analyze Statistics

```bash
python3 scripts/analyze_multi_scenario_statistics.py \
    --metrics-dir ./multi_scenario_validation/metrics \
    --output ./multi_scenario_validation/statistics/ANALYSIS.md
```

### Step 3: Review Results

```bash
cat ./multi_scenario_validation/statistics/ANALYSIS.md
```

---

## Expected Results

### Success Indicators ✅

```markdown
# Multi-Scenario Statistical Analysis Report

## Executive Summary
- Total Scenarios Analyzed: 8 (quick) or 36 (full)
- High Variance Scenarios (CV > 5%): 0 ✅

### Consistency Analysis
- Batch Size Consistency: ✅ (CV < 10%)
- Sequence Length Consistency: ✅ (CV < 10%)
- Input Pattern Consistency: ✅ (CV < 5%)

## Success Criteria Validation
✅ No high variance scenarios (CV < 5%)
✅ Batch size consistency maintained
✅ Sequence length consistency maintained
✅ Input pattern independence validated

Overall Score: 4/4 criteria passed
```

### What to Look For

| Metric | Success Threshold | Interpretation |
|--------|-------------------|----------------|
| **High Variance Scenarios** | 0 or < 10% of total | Stable measurements |
| **Batch Size CV** | < 10% | Consistent across batch sizes |
| **Sequence Length CV** | < 10% | Consistent across seq lengths |
| **Input Pattern CV** | < 5% | Data-independent behavior |
| **Repetition CV** | < 5% | Low measurement noise |

---

## Quick Validation Checklist

After running the analysis, check these items:

- [ ] ✅ **No High Variance**: Report shows 0 high variance scenarios
- [ ] ✅ **Batch Consistency**: All batch size groups show ✅ status
- [ ] ✅ **Sequence Consistency**: All sequence length groups show ✅ status
- [ ] ✅ **Pattern Consistency**: All input pattern groups show ✅ status
- [ ] ✅ **Success Criteria**: 4/4 or 3/4 criteria passed
- [ ] ✅ **Recommendations**: "No Issues Detected" or minor warnings only

**If all boxes checked**: Optimizations are validated and ready for integration! 🎉

---

## Key Metrics Reference

### Total Reorder Count

```
Baseline (research branch):
  ~180 reorders per block (batch=1, seq=128)

Optimized (with input+output optimizations):
  ~10 reorders per block (batch=1, seq=128)

Improvement: ~170 reorders eliminated (94% reduction)
```

**Expected Variance**:
- Across repetitions: CV < 2%
- Across batch sizes: CV < 5%
- Across sequence lengths: CV < 5%
- Across input patterns: CV < 3%

### Total Reorder Time (ms)

```
Baseline (research branch):
  ~5.2 ms per block (batch=1, seq=128)

Optimized (with input+output optimizations):
  ~0.05 ms per block (batch=1, seq=128)

Improvement: ~5.15 ms saved per block (99% reduction)
```

**Expected Variance**:
- Across repetitions: CV < 3%
- Across batch sizes: CV < 8%
- Across sequence lengths: CV < 10%
- Across input patterns: CV < 5%

---

## Common Scenarios

### Scenario 1: All Tests Pass ✅

```bash
$ ./scripts/example_multi_scenario_validation.sh
...
✓ Analysis complete
  High variance scenarios: 0
  Success criteria: 4/4 passed
  Recommendation: No issues detected
```

**Action**: Proceed with integration. Document validated scenarios in release notes.

---

### Scenario 2: Minor Variance in Edge Cases ⚠️

```bash
$ cat ./multi_scenario_validation/statistics/ANALYSIS.md
...
High Variance Scenarios: 2
  - batch8_seq256_ones: CV=6.2%
  - batch8_seq256_small_values: CV=5.8%

Success Criteria: 3/4 passed
```

**Action**: Review edge cases. If variance is < 10% and only affects rare scenarios (large batch + long sequence), document as acceptable. Consider re-running with more repetitions.

---

### Scenario 3: Significant Variance ❌

```bash
$ cat ./multi_scenario_validation/statistics/ANALYSIS.md
...
High Variance Scenarios: 12
Batch Size Consistency: ❌ (CV=15%)

Success Criteria: 2/4 passed
```

**Action**: Investigation required. Possible causes:
1. System load or thermal throttling
2. Batch-dependent optimization logic
3. DNNL non-determinism

**Solutions**:
- Run on dedicated system with minimal load
- Increase repetitions to 5 or 10
- Profile individual scenarios with DNNL_VERBOSE=2
- Review optimization code for batch-dependent paths

---

## File Structure

After running multi-scenario validation:

```
multi_scenario_validation/
├── model/
│   └── transformer_block.xml          # Extracted single block model
├── traces/
│   ├── batch1_seq1_random/
│   │   ├── onednn_trace_rep1.txt
│   │   ├── onednn_trace_rep2.txt
│   │   └── onednn_trace_rep3.txt
│   └── ... (8 or 36 scenario directories)
├── metrics/
│   ├── batch1_seq1_random/
│   │   ├── metrics_rep1.json
│   │   ├── metrics_rep2.json
│   │   └── metrics_rep3.json
│   └── ... (8 or 36 scenario directories)
├── logs/
│   ├── extraction.log
│   ├── capture_batch1_seq1_random_rep1.log
│   └── ... (one log per trace capture)
├── statistics/
│   └── ANALYSIS.md                    # Main analysis report
├── scenario_matrix.txt                # Test matrix metadata
└── QUICK_SUMMARY.txt                  # Quick results summary
```

---

## Advanced Options

### Custom Repetitions

```bash
# Run 5 repetitions per scenario instead of 3
./scripts/capture_multi_scenario_traces.sh --repetitions 5
```

### Custom Iterations

```bash
# Run 100 inference iterations per capture (more stable timing)
./scripts/capture_multi_scenario_traces.sh --iterations 100
```

### Skip Model Extraction

```bash
# Use existing extracted model
./scripts/capture_multi_scenario_traces.sh --skip-extraction
```

### Custom Variance Threshold

```bash
# Use 3% threshold for high variance detection (stricter)
python3 scripts/analyze_multi_scenario_statistics.py \
    --metrics-dir ./multi_scenario_validation/metrics \
    --output ./ANALYSIS.md \
    --variance-threshold 3.0
```

---

## Troubleshooting

### Issue: Trace Capture Takes Too Long

**Solution**: Use quick mode for initial testing
```bash
./scripts/capture_multi_scenario_traces.sh --quick
```

### Issue: High Variance in All Scenarios

**Possible Causes**:
- System load (background processes)
- Thermal throttling
- Insufficient repetitions

**Solutions**:
```bash
# Increase repetitions
./scripts/capture_multi_scenario_traces.sh --repetitions 5

# Use dedicated system with minimal load
# Set CPU governor to "performance" mode
sudo cpupower frequency-set -g performance
```

### Issue: Metrics Extraction Failed

**Check**:
```bash
# Verify trace files exist and are not empty
ls -lh ./multi_scenario_validation/traces/batch1_seq1_random/

# Manually parse one trace
python3 scripts/parse_onednn_reorders.py \
    --trace ./multi_scenario_validation/traces/batch1_seq1_random/onednn_trace_rep1.txt \
    --output-json ./test_metrics.json

# Check logs
cat ./multi_scenario_validation/logs/capture_batch1_seq1_random_rep1.log
```

### Issue: Report Shows Empty Scenarios

**Possible Causes**:
- Metrics directory path incorrect
- JSON files corrupted
- Directory naming mismatch

**Solutions**:
```bash
# Verify metrics directory structure
find ./multi_scenario_validation/metrics -name "*.json" | wc -l
# Should show 24 (quick) or 108 (full)

# Check directory names
ls ./multi_scenario_validation/metrics/
# Should show batch{X}_seq{Y}_{pattern} format

# Re-run with verbose output
python3 scripts/analyze_multi_scenario_statistics.py \
    --metrics-dir ./multi_scenario_validation/metrics \
    --output ./ANALYSIS.md 2>&1 | tee analysis.log
```

---

## Next Steps After Validation

### If All Tests Pass ✅

1. **Document Results**: Add multi-scenario validation results to release notes
2. **Update CI/CD**: Integrate quick mode into regression testing
3. **Extend Coverage**: Consider testing additional model architectures
4. **Production Deployment**: Optimizations validated for production use

### If Tests Fail ❌

1. **Review Analysis Report**: Identify which scenarios show high variance
2. **Profile Individual Scenarios**: Use DNNL_VERBOSE=2 for detailed tracing
3. **Check System**: Ensure dedicated system with minimal load
4. **Investigate Code**: Review optimization logic for scenario-dependent behavior
5. **Re-run with More Repetitions**: Increase to 5-10 repetitions for stability

---

## Summary Table

| Step | Command | Duration | Output |
|------|---------|----------|--------|
| **Quick Test** | `./scripts/capture_multi_scenario_traces.sh --quick` | ~45 min | 24 traces |
| **Full Test** | `./scripts/capture_multi_scenario_traces.sh` | ~3-4 hours | 108 traces |
| **Analyze** | `python3 scripts/analyze_multi_scenario_statistics.py ...` | ~1 min | ANALYSIS.md |
| **Review** | `cat ./multi_scenario_validation/statistics/ANALYSIS.md` | ~5 min | Visual review |

**Total Time**: 
- Quick mode: ~50 minutes
- Full mode: ~3-4 hours

---

## Success Criteria Summary

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| High Variance Scenarios | 0 or < 10% | ✅ Pass if 0 |
| Batch Size Consistency | CV < 10% | ✅ Pass if all groups < 10% |
| Sequence Length Consistency | CV < 10% | ✅ Pass if all groups < 10% |
| Input Pattern Consistency | CV < 5% | ✅ Pass if all groups < 5% |

**Overall Pass**: 4/4 or 3/4 criteria met

---

For comprehensive documentation, see [MULTI_SCENARIO_VALIDATION_GUIDE.md](MULTI_SCENARIO_VALIDATION_GUIDE.md).

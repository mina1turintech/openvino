# Quick Start: Baseline Trace Capture

## One-Command Baseline Capture

Capture complete baseline metrics in one command:

```bash
chmod +x scripts/capture_baseline_trace.sh
./scripts/capture_baseline_trace.sh
```

**Time:** ~15 minutes first run (includes model download), ~5 minutes subsequent runs

**Output:** Complete baseline with traces, metrics, and documentation in `./baseline_capture/`

## What You Get

After running the command, you'll have:

```
./baseline_capture/
├── BASELINE_TRACE_METRICS.md     # Complete documentation (start here!)
├── model/
│   └── transformer_block.xml     # Extracted model
├── traces/
│   ├── onednn_trace_baseline_run1.txt
│   ├── onednn_trace_baseline_run2.txt
│   └── onednn_trace_baseline_run3.txt
└── metrics/
    ├── baseline_run1_metrics.csv  # Structured metrics
    └── baseline_run1_metrics.json
```

## View Your Results

### Read the Documentation

```bash
cat baseline_capture/BASELINE_TRACE_METRICS.md
```

This includes:
- ✅ Complete baseline metrics summary
- ✅ Trace statistics and operation counts
- ✅ Reproducibility analysis
- ✅ Sample trace excerpts
- ✅ Next steps for optimization

### Analyze Metrics

```bash
# View CSV metrics (spreadsheet-friendly)
cat baseline_capture/metrics/baseline_run1_metrics.csv

# View JSON metrics (programmatic access)
cat baseline_capture/metrics/baseline_run1_metrics.json

# Quick reorder count
grep -c "reorder" baseline_capture/traces/onednn_trace_baseline_run1.txt
```

### Examine Trace Samples

```bash
# View first 20 reorder operations
grep "reorder" baseline_capture/traces/onednn_trace_baseline_run1.txt | head -n 20

# Find specific dimension reorders
grep "1536x8960" baseline_capture/traces/onednn_trace_baseline_run1.txt

# Extract timing information
grep "reorder" baseline_capture/traces/onednn_trace_baseline_run1.txt | awk -F',' '{print $NF}'
```

## Prerequisites

Install required packages:

```bash
pip install torch transformers openvino numpy
```

## Customization

### Different Layer

```bash
./scripts/capture_baseline_trace.sh --layer-index 14
```

### More Statistics

```bash
./scripts/capture_baseline_trace.sh --num-runs 5 --iterations 100
```

### Custom Output Location

```bash
./scripts/capture_baseline_trace.sh --output-dir ./my_baseline
```

## Next Steps

After capturing baseline:

1. **Review Documentation**
   ```bash
   cat baseline_capture/BASELINE_TRACE_METRICS.md
   ```

2. **Identify Optimization Opportunities**
   - High-count reorder operations
   - Large-dimension reorders
   - Redundant transformations

3. **Implement Optimizations**
   - Trace graph optimizer passes
   - Apply layout transformations

4. **Capture Optimized Trace**
   ```bash
   python3 scripts/capture_onednn_trace.py \
       --model-path ./optimized_model/transformer_block.xml \
       --output-dir baseline_capture/traces \
       --tag optimized
   ```

5. **Compare Results**
   ```bash
   python3 scripts/parse_onednn_reorders.py \
       --baseline baseline_capture/traces/onednn_trace_baseline_run1.txt \
       --optimized baseline_capture/traces/onednn_trace_optimized_*.txt \
       --output-csv baseline_capture/metrics/comparison.csv
   ```

## Troubleshooting

**Model download fails?**
```bash
huggingface-cli login
```

**Missing packages?**
```bash
pip install torch transformers openvino numpy
```

**Need help?**
```bash
./scripts/capture_baseline_trace.sh --help
```

## Full Documentation

For complete details:
- **Workflow Guide:** `scripts/BASELINE_CAPTURE_README.md`
- **Trace Capture:** `scripts/ONEDNN_TRACE_CAPTURE_README.md`
- **Metrics Parser:** `scripts/PARSE_ONEDNN_REORDERS_README.md`
- **Block Extraction:** `scripts/EXTRACT_TRANSFORMER_BLOCK_README.md`

---

**Ready?** Run the command and you'll have complete baseline metrics in ~15 minutes!

```bash
./scripts/capture_baseline_trace.sh
```

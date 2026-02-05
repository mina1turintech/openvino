# Baseline Trace Capture Workflow

## Overview

The baseline capture workflow is a comprehensive automation tool that captures and documents baseline oneDNN traces for the single transformer block. This establishes the reference metrics against which all subsequent optimizations will be measured.

The workflow automates the entire process:
1. **Extract** a single transformer block from Qwen2-1.5B
2. **Capture** multiple oneDNN verbose traces for reproducibility
3. **Parse** traces to extract reorder operation metrics
4. **Validate** metrics and check for expected patterns
5. **Generate** comprehensive documentation with all results

## Quick Start

### Basic Usage

Run the complete baseline capture workflow with default settings:

```bash
chmod +x scripts/capture_baseline_trace.sh
./scripts/capture_baseline_trace.sh
```

This will:
- Extract layer 0 from Qwen2-1.5B
- Capture 3 baseline trace runs (50 iterations each)
- Extract and validate metrics
- Generate complete documentation

**Time:** ~15 minutes first run (includes model download), ~5 minutes subsequent runs

### Output Structure

The script creates a complete output directory:

```
./baseline_capture/
├── model/                        # Extracted transformer block
│   ├── transformer_block.xml
│   ├── transformer_block.bin
│   └── transformer_block_specs.txt
├── traces/                       # oneDNN verbose traces
│   ├── onednn_trace_baseline_run1.txt
│   ├── onednn_trace_baseline_run2.txt
│   └── onednn_trace_baseline_run3.txt
├── metrics/                      # Parsed metrics
│   ├── baseline_run1_metrics.csv
│   ├── baseline_run1_metrics.json
│   ├── baseline_run2_metrics.csv
│   ├── baseline_run2_metrics.json
│   ├── baseline_run3_metrics.csv
│   └── baseline_run3_metrics.json
├── logs/                         # Execution logs
│   ├── extraction.log
│   ├── capture_run1.log
│   ├── capture_run2.log
│   ├── capture_run3.log
│   ├── parse_run1.log
│   ├── parse_run2.log
│   └── parse_run3.log
└── BASELINE_TRACE_METRICS.md     # Complete documentation
```

## Command-Line Options

### Available Options

```bash
./scripts/capture_baseline_trace.sh [OPTIONS]

Options:
  --model-name MODEL        HuggingFace model name (default: Qwen/Qwen2-1.5B-Instruct)
  --layer-index INDEX       Transformer layer to extract (default: 0)
  --num-runs NUM            Number of trace capture runs (default: 3)
  --iterations NUM          Inference iterations per run (default: 50)
  --output-dir DIR          Output directory (default: ./baseline_capture)
  --skip-extraction         Skip model extraction (use existing model)
  --help                    Show help message
```

### Example Configurations

#### Extract Different Layer

```bash
./scripts/capture_baseline_trace.sh --layer-index 14
```

#### More Runs for Better Statistics

```bash
./scripts/capture_baseline_trace.sh --num-runs 5 --iterations 100
```

#### Custom Output Location

```bash
./scripts/capture_baseline_trace.sh --output-dir ./my_baseline
```

#### Reuse Existing Model

If you've already extracted a model and want to re-capture traces:

```bash
./scripts/capture_baseline_trace.sh --skip-extraction
```

## Prerequisites

### Required Packages

Install required Python packages:

```bash
pip install torch transformers openvino numpy
```

### System Requirements

- **Python:** 3.10 or higher
- **Memory:** ~4GB RAM (for model download and inference)
- **Disk Space:** ~5GB (model + traces + metrics)
- **Time:** 15 minutes first run, 5 minutes subsequent runs

### Network Access

First run requires internet access to download Qwen2-1.5B model from HuggingFace (~3GB). Subsequent runs use cached model.

## What Gets Captured

### Baseline Configuration

The workflow captures baseline metrics with:

- **Model:** Single transformer block (layer 0 by default)
- **Precision:** BFloat16
- **Device:** CPU
- **Batch Size:** 1
- **Sequence Length:** 16
- **Iterations:** 50 per run (adjustable)
- **Random Seed:** 42 (for reproducibility)
- **DNNL_VERBOSE:** Level 1

### Metrics Extracted

For each trace capture run, the following metrics are extracted:

#### Total Metrics
- Total reorder operation count
- Total reorder time (milliseconds)
- Average time per reorder

#### By Implementation Type
- jit:uni
- jit_direct_copy:uni
- simple:any
- Others

#### By Dimension
- 1536x8960
- 8960x1536
- 1536x1536
- Others

#### By Individual Dimension Value
- 1536 (hidden size)
- 8960 (FFN intermediate)
- 256 (attention-related)
- Others

#### By Layout Transformation
- ab → ba (transpose)
- abc → acb (permutation)
- Others

### Output Formats

Metrics are provided in multiple formats:
- **CSV:** For spreadsheet analysis and visualization
- **JSON:** For programmatic processing and automation
- **Markdown:** Human-readable documentation with tables and examples

## Understanding the Documentation

### Generated Documentation File

The workflow generates `BASELINE_TRACE_METRICS.md` with:

1. **Overview** - Capture configuration and model specs
2. **Trace Statistics** - Operation counts and dimension verification
3. **Baseline Metrics** - Complete metrics breakdown
4. **Reproducibility Analysis** - Variance between runs
5. **Sample Trace Excerpts** - Representative reorder operations
6. **Interpretation Guidelines** - How to read and use the metrics
7. **Next Steps** - Optimization workflow and comparison instructions

### Key Sections to Review

#### Trace Statistics

Shows overall operation counts:
```
Total Trace Lines: 1234
Reorder Operations: 56
MatMul Operations: 12
InnerProduct Operations: 8
```

#### Total Reorder Metrics

Summary table:
```
| Metric | Value |
|--------|-------|
| Total Reorder Count | 56 |
| Total Reorder Time | 12.345 ms |
| Average Time per Reorder | 0.220 ms |
```

#### Reproducibility Analysis

Shows variance between runs:
```
Run 1: 56 reorder operations
Run 2: 56 reorder operations
Run 3: 56 reorder operations
Result: ✅ Perfect reproducibility
```

## Validation and Quality Checks

The workflow performs automatic validation:

### ✅ Success Checks

- Model extraction successful
- All traces captured without errors
- Metrics parsed successfully
- Expected dimensions (1536, 8960) present
- Reproducibility within acceptable range

### ⚠️ Warning Indicators

- Missing expected dimensions
- High variance between runs (>5%)
- Unusually low/high operation counts

### 🔴 Error Conditions

- Model extraction fails
- Trace capture fails
- Metrics parsing fails
- Missing required files

## Troubleshooting

### Model Download Fails

**Problem:** HuggingFace model download timeout or fails

**Solution:**
```bash
# Login to HuggingFace (if model requires authentication)
huggingface-cli login

# Or use local model path if already downloaded
./scripts/capture_baseline_trace.sh --model-name /path/to/local/model
```

### Missing Dependencies

**Problem:** `ImportError` for torch, transformers, or openvino

**Solution:**
```bash
pip install torch transformers openvino numpy
```

### Out of Memory

**Problem:** System runs out of memory during model extraction

**Solution:**
```bash
# Close other applications
# Or use a smaller model/layer
./scripts/capture_baseline_trace.sh --layer-index 0
```

### Trace Capture Fails

**Problem:** oneDNN trace capture returns empty or invalid trace

**Solution:**
```bash
# Check if model file exists
ls -lh baseline_capture/model/transformer_block.xml

# Verify OpenVINO installation
python3 -c "import openvino; print(openvino.__version__)"

# Try manual capture
python3 scripts/capture_onednn_trace.py \
    --model-path baseline_capture/model/transformer_block.xml \
    --output-dir ./test_traces \
    --tag test
```

### Reproducibility Issues

**Problem:** Runs produce different operation counts

**Cause:** Timing-dependent behavior or system load variations

**Solution:**
This is usually acceptable if variance < 5%. The workflow uses fixed random seed to ensure input consistency, but system factors can cause minor variations.

## Integration with Workflow

### Full Optimization Workflow

1. **Capture Baseline** (this tool)
   ```bash
   ./scripts/capture_baseline_trace.sh
   ```

2. **Analyze Baseline**
   ```bash
   cat baseline_capture/BASELINE_TRACE_METRICS.md
   cat baseline_capture/metrics/baseline_run1_metrics.csv
   ```

3. **Implement Optimizations**
   - Trace graph optimizer passes
   - Apply layout transformations
   - Rebuild optimized model

4. **Capture Optimized Trace**
   ```bash
   python3 scripts/capture_onednn_trace.py \
       --model-path ./optimized_model/transformer_block.xml \
       --output-dir baseline_capture/traces \
       --tag optimized \
       --iterations 50
   ```

5. **Compare Results**
   ```bash
   python3 scripts/parse_onednn_reorders.py \
       --baseline baseline_capture/traces/onednn_trace_baseline_run1.txt \
       --optimized baseline_capture/traces/onednn_trace_optimized_*.txt \
       --output-csv baseline_capture/metrics/comparison.csv
   ```

### Automated Regression Testing

Use the workflow in CI/CD:

```bash
# Capture baseline
./scripts/capture_baseline_trace.sh --output-dir ./ci_baseline --num-runs 1

# After changes, compare
./scripts/capture_baseline_trace.sh --output-dir ./ci_test --num-runs 1
diff ci_baseline/metrics/baseline_run1_metrics.csv ci_test/metrics/baseline_run1_metrics.csv
```

## Performance Considerations

### Capture Overhead

- **Model Download:** ~3GB, first run only (~5 minutes on fast connection)
- **Model Extraction:** ~2-3 minutes (includes PyTorch → OpenVINO conversion)
- **Trace Capture:** ~30-60 seconds per run (depends on iterations)
- **Metrics Parsing:** ~1-2 seconds per trace

### Disk Space

- **Model Files:** ~200-500MB (IR format)
- **Trace Files:** ~500KB-2MB per trace (depends on iterations)
- **Metrics Files:** ~10-50KB per run (CSV + JSON)
- **Total:** ~1-2GB for complete baseline capture

### Iteration Count Recommendations

- **Quick test:** 10 iterations (~10 seconds)
- **Standard baseline:** 50 iterations (~30 seconds) - **default**
- **Detailed analysis:** 100-500 iterations (~1-5 minutes)
- **Statistical significance:** 1000+ iterations (~10+ minutes)

More iterations = more stable timing measurements but longer capture time.

## Advanced Usage

### Capture Multiple Layers

Compare different transformer layers:

```bash
for layer in 0 14 27; do
    ./scripts/capture_baseline_trace.sh \
        --layer-index $layer \
        --output-dir baseline_layer_$layer \
        --num-runs 3
done
```

### Batch Processing

Automate multiple configurations:

```bash
#!/bin/bash
for model in "Qwen/Qwen2-1.5B-Instruct" "Qwen/Qwen2-7B-Instruct"; do
    for layer in 0 14; do
        ./scripts/capture_baseline_trace.sh \
            --model-name "$model" \
            --layer-index $layer \
            --output-dir "baseline_${model##*/}_layer${layer}" \
            --num-runs 3
    done
done
```

### Custom Analysis

After capture, perform custom analysis:

```bash
# Extract specific reorder types
grep "jit:uni" baseline_capture/traces/onednn_trace_baseline_run1.txt > jit_uni_reorders.txt

# Analyze dimension frequency
grep -o "[0-9]\+x[0-9]\+" baseline_capture/traces/onednn_trace_baseline_run1.txt | sort | uniq -c

# Calculate total reorder time
grep "reorder" baseline_capture/traces/onednn_trace_baseline_run1.txt | \
    awk -F',' '{sum += $NF} END {print sum " ms"}'
```

## Success Criteria

A successful baseline capture should have:

- ✅ Model extracted successfully (transformer_block.xml exists)
- ✅ All traces captured (3 files by default)
- ✅ Metrics extracted (6 files: 3 CSV + 3 JSON)
- ✅ Documentation generated (BASELINE_TRACE_METRICS.md)
- ✅ Expected dimensions found (1536, 8960)
- ✅ Reproducibility within acceptable range (< 5% variance)
- ✅ All validation checks passed

The generated documentation will show ✅ marks for each criterion.

## Related Documentation

- **Transformer Block Extraction:** `scripts/EXTRACT_TRANSFORMER_BLOCK_README.md`
- **oneDNN Trace Capture:** `scripts/ONEDNN_TRACE_CAPTURE_README.md`
- **Reorder Metrics Parser:** `scripts/PARSE_ONEDNN_REORDERS_README.md`
- **Quick Start Guide:** `scripts/QUICKSTART_TRANSFORMER_BLOCK.md`

## Support

### Getting Help

1. **Check logs:** Review files in `baseline_capture/logs/`
2. **Validate environment:** Run `python3 -c "import torch, transformers, openvino"`
3. **Test components individually:** Run each script separately
4. **Review documentation:** Check the generated BASELINE_TRACE_METRICS.md

### Common Issues

Most issues fall into these categories:
1. **Missing dependencies** → Install required packages
2. **Network issues** → Check HuggingFace access
3. **Memory issues** → Close other applications
4. **Permission issues** → Check write permissions for output directory

---

## Complete Example

Here's a complete example workflow from start to finish:

```bash
# 1. Install dependencies (if needed)
pip install torch transformers openvino numpy

# 2. Run baseline capture
chmod +x scripts/capture_baseline_trace.sh
./scripts/capture_baseline_trace.sh

# 3. Review documentation
cat baseline_capture/BASELINE_TRACE_METRICS.md

# 4. Analyze metrics
cat baseline_capture/metrics/baseline_run1_metrics.csv

# 5. Examine trace samples
grep reorder baseline_capture/traces/onednn_trace_baseline_run1.txt | head -n 20

# 6. Check reproducibility
diff baseline_capture/traces/onednn_trace_baseline_run1.txt \
     baseline_capture/traces/onednn_trace_baseline_run2.txt

# Done! Baseline is ready for optimization comparison.
```

**Expected time:** ~15 minutes first run, ~5 minutes subsequent runs

**Expected output:** Complete baseline capture with documentation, ready for optimization work.

---

**Ready to capture your baseline? Run the workflow and you'll have comprehensive baseline metrics in minutes!**

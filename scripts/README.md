# OpenVINO Transformer Optimization Scripts

This directory contains automation tools and utilities for transformer model optimization, specifically focused on layout optimization for Qwen2-1.5B and similar models.

## Quick Start

### Capture Baseline Metrics (Recommended Starting Point)

The fastest way to get started is to capture baseline metrics for optimization:

```bash
chmod +x scripts/capture_baseline_trace.sh
./scripts/capture_baseline_trace.sh
```

This single command will:
- Extract a transformer block from Qwen2-1.5B
- Capture oneDNN verbose traces
- Extract and analyze reorder operation metrics
- Generate comprehensive documentation

**See:** `BASELINE_QUICKSTART.md` for details

## Available Tools

### 1. Baseline Trace Capture (Start Here!)

**Purpose:** Complete workflow to capture and document baseline metrics

**Main Script:** `capture_baseline_trace.sh`

**Quick Start:**
```bash
./scripts/capture_baseline_trace.sh
```

**Documentation:**
- `BASELINE_QUICKSTART.md` - Quick start guide
- `BASELINE_CAPTURE_README.md` - Complete workflow documentation
- `example_baseline_capture.sh` - Usage examples

**Output:** Complete baseline with traces, metrics, and documentation

---

### 2. Transformer Block Extraction

**Purpose:** Extract single transformer blocks from HuggingFace models

**Main Script:** `extract_transformer_block.py`

**Quick Start:**
```bash
python3 scripts/extract_transformer_block.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --layer-index 0 \
    --output-dir ./extracted_block
```

**Documentation:**
- `EXTRACT_TRANSFORMER_BLOCK_README.md` - Complete guide
- `example_extract_block.sh` - Usage examples

**Output:** OpenVINO IR model (.xml + .bin)

---

### 3. oneDNN Trace Capture

**Purpose:** Capture oneDNN verbose traces for performance analysis

**Main Script:** `capture_onednn_trace.py`

**Quick Start:**
```bash
python3 scripts/capture_onednn_trace.py \
    --model-path ./model.xml \
    --output-dir ./traces \
    --tag baseline
```

**Documentation:**
- `ONEDNN_TRACE_CAPTURE_README.md` - Complete guide
- `example_capture_trace.sh` - Usage examples

**Output:** oneDNN verbose trace files (.txt)

---

### 4. Reorder Metrics Parser

**Purpose:** Extract and analyze reorder operation metrics from traces

**Main Script:** `parse_onednn_reorders.py`

**Quick Start:**
```bash
# Single trace analysis
python3 scripts/parse_onednn_reorders.py \
    --trace baseline_trace.txt \
    --output-csv metrics.csv

# Baseline vs optimized comparison
python3 scripts/parse_onednn_reorders.py \
    --baseline baseline_trace.txt \
    --optimized optimized_trace.txt \
    --output-csv comparison.csv
```

**Documentation:**
- `PARSE_ONEDNN_REORDERS_README.md` - Complete guide
- `example_parse_reorders.sh` - Usage examples

**Output:** Metrics in CSV and/or JSON format

---

### 5. Graph Optimizer Instrumentation

**Purpose:** Trace graph optimizer passes to understand layout optimization decisions

**Main Script:** Built into OpenVINO CPU plugin (requires recompilation)

**Quick Start:**
```bash
# Enable instrumentation
export OV_CPU_GRAPH_OPTIMIZER_TRACE=1
export OV_CPU_GRAPH_OPTIMIZER_TRACE_FILE=./optimizer_trace.json

# Run inference (this triggers graph optimization)
python3 scripts/run_single_block_test.py --model transformer_block.xml

# Analyze results
python3 scripts/analyze_optimizer_trace.py ./optimizer_trace.json
```

**Documentation:**
- `GRAPH_OPTIMIZER_INSTRUMENTATION_README.md` - Complete guide
- `example_optimizer_tracing.sh` - Usage examples

**Output:** JSON trace showing pass execution and reorder elimination

---

### 6. Block Testing

**Purpose:** Test and validate extracted transformer blocks

**Main Script:** `test_extracted_block.py`

**Quick Start:**
```bash
python3 scripts/test_extracted_block.py \
    --model-path ./extracted_block/transformer_block.xml \
    --iterations 100
```

**Output:** Validation results and performance metrics

---

## Typical Workflow

### Step 1: Capture Baseline
```bash
./scripts/capture_baseline_trace.sh
```

**Output:** `baseline_capture/` directory with complete baseline metrics

### Step 2: Analyze Baseline
```bash
cat baseline_capture/BASELINE_TRACE_METRICS.md
cat baseline_capture/metrics/baseline_run1_metrics.csv
```

### Step 3: Identify Optimization Opportunities

Review baseline metrics to find:
- High-count reorder operations
- Large-dimension reorders (1536, 8960)
- Redundant layout transformations

### Step 4: Trace Graph Optimizer Passes

Before implementing optimizations, understand how the graph optimizer currently handles layouts:

```bash
# Enable optimizer tracing
export OV_CPU_GRAPH_OPTIMIZER_TRACE=1
export OV_CPU_GRAPH_OPTIMIZER_TRACE_FILE=./baseline_capture/optimizer_trace.json

# Run with traced optimization
python3 scripts/run_single_block_test.py --model baseline_capture/model/transformer_block.xml

# Analyze optimizer decisions
python3 scripts/analyze_optimizer_trace.py ./baseline_capture/optimizer_trace.json
```

This shows which passes eliminate reorders and where opportunities remain.

### Step 5: Implement Optimizations

Apply layout optimizations to the model:
- Modify graph optimizer passes based on trace analysis
- Implement layout propagation improvements
- Add operation fusion opportunities

### Step 6: Capture Optimized Trace
```bash
python3 scripts/capture_onednn_trace.py \
    --model-path ./optimized_model/transformer_block.xml \
    --output-dir baseline_capture/traces \
    --tag optimized
```

### Step 7: Compare Results
```bash
python3 scripts/parse_onednn_reorders.py \
    --baseline baseline_capture/traces/onednn_trace_baseline_run1.txt \
    --optimized baseline_capture/traces/onednn_trace_optimized_*.txt \
    --output-csv baseline_capture/metrics/comparison.csv
```

**Output:** Comparison metrics showing optimization impact

## Prerequisites

### Required Python Packages

```bash
pip install torch transformers openvino numpy
```

### System Requirements

- **Python:** 3.10 or higher
- **Memory:** 4-8GB RAM
- **Disk Space:** 5-10GB (for models and traces)
- **Network:** Internet access for first-time model download

### Environment Setup

For some tools, you may need to source OpenVINO environment:

```bash
source /opt/intel/openvino/setupvars.sh
# Or if installed via pip, no setup needed
```

## Documentation Index

### Quick Start Guides
- `BASELINE_QUICKSTART.md` - Fastest way to capture baseline metrics
- `QUICKSTART_TRANSFORMER_BLOCK.md` - Quick guide to block extraction

### Complete Guides
- `BASELINE_CAPTURE_README.md` - Complete baseline workflow
- `EXTRACT_TRANSFORMER_BLOCK_README.md` - Block extraction guide
- `ONEDNN_TRACE_CAPTURE_README.md` - Trace capture guide
- `PARSE_ONEDNN_REORDERS_README.md` - Metrics parser guide
- `GRAPH_OPTIMIZER_INSTRUMENTATION_README.md` - Optimizer tracing guide

### Examples
- `example_baseline_capture.sh` - Baseline capture examples
- `example_extract_block.sh` - Block extraction examples
- `example_capture_trace.sh` - Trace capture examples
- `example_parse_reorders.sh` - Metrics parser examples
- `example_optimizer_tracing.sh` - Optimizer tracing examples
- `examples/` - Sample output files

## Directory Structure

```
scripts/
├── README.md                               # This file
├── BASELINE_QUICKSTART.md                  # Quick start for baseline capture
├── BASELINE_CAPTURE_README.md              # Complete baseline workflow guide
├── EXTRACT_TRANSFORMER_BLOCK_README.md     # Block extraction guide
├── ONEDNN_TRACE_CAPTURE_README.md          # Trace capture guide
├── PARSE_ONEDNN_REORDERS_README.md         # Metrics parser guide
├── GRAPH_OPTIMIZER_INSTRUMENTATION_README.md # Optimizer tracing guide
├── QUICKSTART_TRANSFORMER_BLOCK.md         # Quick block extraction guide
├── capture_baseline_trace.sh               # Main baseline workflow script
├── extract_transformer_block.py            # Block extraction tool
├── capture_onednn_trace.py                 # Trace capture tool
├── parse_onednn_reorders.py                # Metrics parser tool
├── analyze_optimizer_trace.py              # Optimizer trace analyzer
├── test_extracted_block.py                 # Block testing tool
├── example_baseline_capture.sh             # Baseline examples
├── example_extract_block.sh                # Extraction examples
├── example_capture_trace.sh                # Trace capture examples
├── example_parse_reorders.sh               # Parser examples
├── example_optimizer_tracing.sh            # Optimizer tracing examples
├── examples/                               # Sample outputs
└── utils/                                  # Utility modules
```

## Common Use Cases

### 1. Quick Baseline Capture (Recommended)

```bash
./scripts/capture_baseline_trace.sh
```

### 2. Extract Specific Layer

```bash
python3 scripts/extract_transformer_block.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --layer-index 14 \
    --output-dir ./layer_14
```

### 3. Capture Multiple Configurations

```bash
for layer in 0 14 27; do
    ./scripts/capture_baseline_trace.sh \
        --layer-index $layer \
        --output-dir baseline_layer_$layer
done
```

### 4. Compare Baseline vs Optimized

```bash
# After optimization
python3 scripts/parse_onednn_reorders.py \
    --baseline baseline_trace.txt \
    --optimized optimized_trace.txt \
    --output-csv comparison.csv \
    --output-json comparison.json
```

### 5. Batch Analysis

```bash
for trace in traces/*.txt; do
    python3 scripts/parse_onednn_reorders.py \
        --trace "$trace" \
        --output-json "${trace%.txt}_metrics.json"
done
```

## Troubleshooting

### Model Download Issues

**Problem:** HuggingFace model download fails

**Solution:**
```bash
huggingface-cli login
# Or use local model path
```

### Missing Dependencies

**Problem:** Import errors for packages

**Solution:**
```bash
pip install torch transformers openvino numpy
```

### Trace Capture Fails

**Problem:** Empty or invalid traces

**Solution:**
```bash
# Verify OpenVINO installation
python3 -c "import openvino; print(openvino.__version__)"

# Check model file
ls -lh path/to/model.xml

# Try manual test
python3 scripts/test_extracted_block.py --model-path path/to/model.xml
```

### Out of Memory

**Problem:** System runs out of memory

**Solution:**
- Close other applications
- Use smaller batch size
- Extract single layer instead of full model

## Support

### Getting Help

1. **Check documentation** - Each tool has comprehensive README
2. **Run with --help** - All Python scripts support --help flag
3. **Review examples** - Check example_*.sh scripts
4. **Check logs** - Review output logs for error details

### Reporting Issues

When reporting issues, include:
- Command executed
- Error message
- Python/OpenVINO version
- System specs (OS, RAM, CPU)

## Performance Notes

### Typical Execution Times

- **Model Extraction:** 2-5 minutes (first download: +5-10 minutes)
- **Trace Capture:** 30-60 seconds per run
- **Metrics Parsing:** 1-2 seconds per trace
- **Complete Baseline:** 15 minutes first run, 5 minutes subsequent

### Resource Usage

- **Disk Space:** 1-2GB per complete baseline
- **Memory:** 4-8GB during model extraction/inference
- **CPU:** All tools run on CPU (no GPU required)

## Contributing

When adding new tools or scripts:
1. Follow existing naming conventions
2. Include comprehensive README documentation
3. Add example usage script
4. Update this index README
5. Test on clean environment

---

## Quick Reference

### Most Common Commands

```bash
# 1. Capture complete baseline (recommended start)
./scripts/capture_baseline_trace.sh

# 2. Extract a single transformer block
python3 scripts/extract_transformer_block.py --model-name Qwen/Qwen2-1.5B-Instruct --layer-index 0 --output-dir ./block

# 3. Capture trace from existing model
python3 scripts/capture_onednn_trace.py --model-path ./block/transformer_block.xml --output-dir ./traces --tag baseline

# 4. Parse trace and extract metrics
python3 scripts/parse_onednn_reorders.py --trace ./traces/trace.txt --output-csv metrics.csv

# 5. Compare baseline vs optimized
python3 scripts/parse_onednn_reorders.py --baseline baseline.txt --optimized optimized.txt --output-csv comparison.csv
```

---

**Ready to start?** Run `./scripts/capture_baseline_trace.sh` and you'll have complete baseline metrics in ~15 minutes!

For more details on any tool, see the corresponding README file.

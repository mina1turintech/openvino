# oneDNN Verbose Trace Capture Harness

## Overview

The oneDNN Verbose Trace Capture Harness is a test tool designed to capture detailed oneDNN operation traces from the extracted transformer block model. This enables baseline and optimization comparison by providing complete, reproducible traces of all oneDNN operations executed during inference.

## Purpose

This harness enables:
- **Baseline Trace Capture**: Record initial operation traces before optimizations
- **Optimization Comparison**: Compare traces after code/layout changes
- **Memory Layout Analysis**: Identify reorder operations and their dimensions
- **Performance Profiling**: Analyze operation execution times and patterns
- **Reproducible Testing**: Consistent traces across multiple runs using fixed random seeds

## Key Features

- ✅ Automatic DNNL_VERBOSE configuration
- ✅ Complete trace capture (no truncation)
- ✅ Consistent input generation with fixed random seeds
- ✅ Descriptive trace file naming with timestamps
- ✅ Metadata preservation for reproducibility
- ✅ Trace validation (reorder operations, dimensions)
- ✅ Multiple consecutive runs support
- ✅ Error handling for missing models

## Installation & Requirements

### Prerequisites

```bash
pip install openvino numpy
```

### Minimum Versions
- Python >= 3.10
- OpenVINO >= 2024.0
- NumPy >= 1.20

## Usage

### Basic Usage

Capture a baseline trace from the extracted transformer block:

```bash
python scripts/capture_onednn_trace.py \
    --model-path ./extracted_block/transformer_block.xml \
    --output-dir ./traces \
    --tag baseline
```

### Capture Optimized Trace for Comparison

After making optimizations to the model:

```bash
python scripts/capture_onednn_trace.py \
    --model-path ./optimized_block/transformer_block.xml \
    --output-dir ./traces \
    --tag optimized
```

### Advanced Usage

#### Use Different Input Parameters

```bash
python scripts/capture_onednn_trace.py \
    --model-path ./extracted_block/transformer_block.xml \
    --output-dir ./traces \
    --tag baseline_bs2 \
    --batch-size 2 \
    --seq-length 32 \
    --iterations 50
```

#### Capture More Detailed Traces (DNNL_VERBOSE=2)

```bash
python scripts/capture_onednn_trace.py \
    --model-path ./extracted_block/transformer_block.xml \
    --output-dir ./traces \
    --tag baseline_detailed \
    --verbose-level 2
```

#### Disable Timestamp in Filename (for consistent naming)

```bash
python scripts/capture_onednn_trace.py \
    --model-path ./extracted_block/transformer_block.xml \
    --output-dir ./traces \
    --tag baseline \
    --no-timestamp
```

#### Use Custom Random Seed

```bash
python scripts/capture_onednn_trace.py \
    --model-path ./extracted_block/transformer_block.xml \
    --output-dir ./traces \
    --tag baseline \
    --seed 12345
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model-path` | str | *required* | Path to extracted transformer block model (.xml file) |
| `--output-dir` | str | `./traces` | Output directory for trace files |
| `--tag` | str | `baseline` | Tag for trace file naming (e.g., baseline, optimized) |
| `--device` | str | `CPU` | Device to run inference on |
| `--batch-size` | int | `1` | Batch size for inference |
| `--seq-length` | int | `16` | Sequence length for inference |
| `--iterations` | int | `10` | Number of inference iterations |
| `--verbose-level` | int | `1` | DNNL_VERBOSE level (1=basic, 2=detailed) |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--no-timestamp` | flag | False | Disable timestamp in trace filename |

## Output Files

After successful trace capture, the following files are created:

1. **`onednn_trace_<tag>_<timestamp>.txt`** - Complete trace output
2. **`onednn_trace_<tag>_<timestamp>_metadata.json`** - Capture metadata

### Example Output Structure

```
traces/
├── onednn_trace_baseline_20250606_143022.txt          # Baseline trace
├── onednn_trace_baseline_20250606_143022_metadata.json
├── onednn_trace_optimized_20250606_143530.txt         # Optimized trace
└── onednn_trace_optimized_20250606_143530_metadata.json
```

## Trace File Format

The trace file contains:

### Header Section
```
======================================================================
oneDNN Verbose Trace Output
======================================================================
Timestamp: 2025-06-06T14:30:22.123456
Model: ./extracted_block/transformer_block.xml
Device: CPU
Tag: baseline
Batch size: 1
Sequence length: 16
Iterations: 10
Random seed: 42
DNNL_VERBOSE level: 1
======================================================================
```

### STDOUT Section
Contains standard output from the inference run (success messages, etc.)

### STDERR Section (oneDNN Verbose Output)
Contains the actual oneDNN verbose trace with operation details:

```
dnnl_verbose,info,oneDNN v3.4.0 (commit hash)
dnnl_verbose,info,cpu,runtime:OpenMP
dnnl_verbose,info,cpu,isa:Intel AVX2
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:abcd:f0 dst_f32::blocked:acdb:f0,,,1536x8960,0.123047
dnnl_verbose,exec,cpu,inner_product,gemm:jit,forward_inference,src_f32::blocked:ab:f0 wei_f32::blocked:ba:f0 dst_f32::blocked:ab:f0,,,mb1ic1536oc8960,0.456123
...
```

## Metadata File Format

The metadata JSON file contains all parameters used for trace capture:

```json
{
  "timestamp": "2025-06-06T14:30:22.123456",
  "model_path": "./extracted_block/transformer_block.xml",
  "device": "CPU",
  "parameters": {
    "batch_size": 1,
    "seq_length": 16,
    "hidden_size": 1536,
    "num_iterations": 10,
    "random_seed": 42
  },
  "environment": {
    "dnnl_verbose_level": 1
  },
  "trace_file": "./traces/onednn_trace_baseline_20250606_143022.txt",
  "validation": {
    "has_content": true,
    "line_count": 1247,
    "has_reorder": true,
    "has_expected_dims": true,
    "reorder_count": 45,
    "convolution_count": 0,
    "matmul_count": 12,
    "inner_product_count": 24
  },
  "execution_time_seconds": 2.34
}
```

## Environment Variables

The harness automatically configures the following environment variables:

### DNNL_VERBOSE

Controls oneDNN verbose output level:

- **Level 1** (default): Basic operation information
  ```bash
  DNNL_VERBOSE=1
  ```
  Output includes: operation type, implementation, execution time
  
- **Level 2**: Detailed operation information
  ```bash
  DNNL_VERBOSE=2
  ```
  Output includes: Level 1 + memory descriptors, dimensions, strides

### Manual Environment Variable Setting (Alternative)

You can also set environment variables manually before running inference:

```bash
export DNNL_VERBOSE=1
python scripts/test_extracted_block.py --model-path ./extracted_block/transformer_block.xml
```

However, the capture harness is preferred as it handles output redirection automatically.

## Understanding oneDNN Trace Output

### Operation Format

```
dnnl_verbose,exec,cpu,<primitive>,<impl>,<prop_kind>,<src_desc> <wei_desc> <dst_desc>,<attr>,<aux>,<shapes>,<time_ms>
```

### Key Fields

- **primitive**: Operation type (reorder, inner_product, matmul, etc.)
- **impl**: Implementation used (jit:uni, gemm:jit, etc.)
- **shapes**: Tensor dimensions
- **time_ms**: Execution time in milliseconds

### Important Operations for Analysis

#### Reorder Operations

Reorder operations indicate memory layout transformations:

```
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:ab:f0 dst_f32::blocked:ba:f0,,,1536x8960,0.123
```

Key information:
- Source layout: `ab` (row-major)
- Destination layout: `ba` (column-major/transposed)
- Dimensions: `1536x8960` (matches FFN intermediate size)
- Time: `0.123ms`

#### Matrix Multiplication Operations

```
dnnl_verbose,exec,cpu,inner_product,gemm:jit,forward_inference,src_f32::blocked:ab:f0 wei_f32::blocked:ba:f0 dst_f32::blocked:ab:f0,,,mb1ic1536oc8960,0.456
```

Key information:
- Mini-batch: `mb1` (batch size 1)
- Input channels: `ic1536` (hidden size)
- Output channels: `oc8960` (FFN intermediate size)
- Time: `0.456ms`

### Expected Dimensions for Qwen2-1.5B Transformer Block

- **Hidden size**: 1536
- **FFN intermediate size**: 8960
- **Attention dimensions**: 1536 → 1536 (QKV projections)

Look for these dimensions in reorder and matmul operations.

## Validation

The harness automatically validates trace output:

### Validation Checks

1. **Content Check**: Trace is not empty
2. **Reorder Operations**: At least one reorder operation present
3. **Expected Dimensions**: Dimensions 1536 or 8960 present
4. **Operation Counts**: Number of each operation type

### Validation Output

```
Validation Results:
  Lines captured: 1247
  Has content: ✓
  Contains reorder ops: ✓
  Expected dimensions found: ✓
  Reorder operations: 45
  Convolution operations: 0
  MatMul operations: 12
  InnerProduct operations: 24
```

### Interpreting Validation Results

- ✅ **All checks pass**: Trace captured successfully
- ⚠️ **No reorder operations**: Model may already be optimized or issue with capture
- ⚠️ **Expected dimensions missing**: Check model path or configuration
- ❌ **Empty trace**: DNNL_VERBOSE not working, check OpenVINO installation

## Comparing Traces

### Using diff

Compare baseline and optimized traces:

```bash
diff traces/onednn_trace_baseline_*.txt traces/onednn_trace_optimized_*.txt
```

### Extracting Specific Operations

Extract all reorder operations:

```bash
grep -i reorder traces/onednn_trace_baseline_*.txt > baseline_reorders.txt
grep -i reorder traces/onednn_trace_optimized_*.txt > optimized_reorders.txt
diff baseline_reorders.txt optimized_reorders.txt
```

### Counting Operations

```bash
# Count reorder operations
grep -c "reorder" traces/onednn_trace_baseline_*.txt
grep -c "reorder" traces/onednn_trace_optimized_*.txt

# Count matmul operations
grep -c "matmul\|inner_product" traces/onednn_trace_baseline_*.txt
```

### Analyzing Execution Time

```bash
# Extract execution times for reorders
grep "reorder" traces/onednn_trace_baseline_*.txt | awk -F',' '{print $NF}'
```

## Reproducibility

### Ensuring Consistent Traces

The harness ensures reproducibility through:

1. **Fixed Random Seed**: Same inputs across runs
   ```bash
   --seed 42
   ```

2. **Consistent Parameters**: Documented in metadata
   ```bash
   --batch-size 1 --seq-length 16
   ```

3. **Environment Capture**: DNNL_VERBOSE level recorded

### Verifying Reproducibility

Run the same command twice and compare:

```bash
# Run 1
python scripts/capture_onednn_trace.py \
    --model-path ./model.xml \
    --output-dir ./traces \
    --tag baseline \
    --no-timestamp

# Run 2 (should be identical)
python scripts/capture_onednn_trace.py \
    --model-path ./model.xml \
    --output-dir ./traces \
    --tag baseline2 \
    --no-timestamp

# Compare
diff traces/onednn_trace_baseline.txt traces/onednn_trace_baseline2.txt
```

Traces should be identical (except timestamps in header).

## Typical Workflow

### 1. Extract Transformer Block

```bash
python scripts/extract_transformer_block.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --layer-index 0 \
    --output-dir ./extracted_block
```

### 2. Capture Baseline Trace

```bash
python scripts/capture_onednn_trace.py \
    --model-path ./extracted_block/transformer_block.xml \
    --output-dir ./traces \
    --tag baseline \
    --iterations 100
```

### 3. Apply Optimizations

(Modify model, apply layout transformations, etc.)

### 4. Capture Optimized Trace

```bash
python scripts/capture_onednn_trace.py \
    --model-path ./optimized_block/transformer_block.xml \
    --output-dir ./traces \
    --tag optimized \
    --iterations 100
```

### 5. Compare Results

```bash
# Compare reorder operations
grep -i reorder traces/onednn_trace_baseline_*.txt > baseline_reorders.txt
grep -i reorder traces/onednn_trace_optimized_*.txt > optimized_reorders.txt
diff baseline_reorders.txt optimized_reorders.txt

# Count reorders
echo "Baseline reorders: $(grep -c reorder traces/onednn_trace_baseline_*.txt)"
echo "Optimized reorders: $(grep -c reorder traces/onednn_trace_optimized_*.txt)"
```

## Troubleshooting

### Empty or Missing Trace Output

**Problem**: Trace file contains no oneDNN verbose output

**Solutions**:
1. Verify OpenVINO installation includes oneDNN support
2. Try verbose level 2: `--verbose-level 2`
3. Check that model is actually using oneDNN backend
4. Run manually with environment variable:
   ```bash
   export DNNL_VERBOSE=1
   python scripts/test_extracted_block.py --model-path ./model.xml
   ```

### No Reorder Operations Found

**Problem**: Validation shows zero reorder operations

**Possible reasons**:
1. Model already optimized (good!)
2. Model not using specific layouts that require reorders
3. Batch size/sequence length too small

**Solutions**:
1. Try larger batch sizes: `--batch-size 4`
2. Try longer sequences: `--seq-length 64`
3. Check if optimizations already applied

### Model File Not Found

**Problem**: Error: Model file not found

**Solutions**:
1. Verify model path is correct
2. Check if extraction completed successfully
3. Use absolute path instead of relative path

### Memory Issues

**Problem**: Out of memory during trace capture

**Solutions**:
1. Reduce iterations: `--iterations 5`
2. Reduce batch size: `--batch-size 1`
3. Use smaller sequence length: `--seq-length 8`

## Performance Considerations

### Trace Capture Overhead

Enabling DNNL_VERBOSE adds overhead:
- ~10-30% slowdown with level 1
- ~20-50% slowdown with level 2

This is acceptable for profiling but should not be used in production.

### File Size

Trace files can be large:
- Level 1: ~100KB to 2MB per run
- Level 2: ~500KB to 10MB per run

More iterations = larger files.

### Recommended Settings

For optimization analysis:
- Iterations: 10-100 (captures multiple runs for timing variance)
- Verbose level: 1 (sufficient for layout analysis)
- Batch size: 1 (unless analyzing batch-specific behavior)
- Sequence length: 16-32 (representative of typical use)

## Command Reference

### Quick Commands

```bash
# Capture baseline trace
python scripts/capture_onednn_trace.py --model-path ./extracted_block/transformer_block.xml --output-dir ./traces --tag baseline

# Capture optimized trace
python scripts/capture_onednn_trace.py --model-path ./optimized_block/transformer_block.xml --output-dir ./traces --tag optimized

# Compare traces
diff traces/onednn_trace_baseline_*.txt traces/onednn_trace_optimized_*.txt

# Extract reorders
grep -i reorder traces/onednn_trace_baseline_*.txt

# Count operations
grep -c "reorder" traces/onednn_trace_baseline_*.txt
grep -c "inner_product\|matmul" traces/onednn_trace_baseline_*.txt
```

### Full Example Command

```bash
python scripts/capture_onednn_trace.py \
    --model-path ./extracted_block/transformer_block.xml \
    --output-dir ./traces \
    --tag baseline_batch2_seq32 \
    --device CPU \
    --batch-size 2 \
    --seq-length 32 \
    --iterations 50 \
    --verbose-level 1 \
    --seed 42
```

## Integration with Automation

The harness is designed for easy integration into scripts:

```bash
#!/bin/bash

# Capture multiple traces with different configurations
for batch in 1 2 4; do
    for seq in 16 32 64; do
        python scripts/capture_onednn_trace.py \
            --model-path ./model.xml \
            --output-dir ./traces \
            --tag "baseline_bs${batch}_seq${seq}" \
            --batch-size $batch \
            --seq-length $seq \
            --iterations 50 \
            --no-timestamp
    done
done
```

## Related Documentation

- **Transformer Block Extraction**: `scripts/EXTRACT_TRANSFORMER_BLOCK_README.md`
- **Quick Start Guide**: `scripts/QUICKSTART_TRANSFORMER_BLOCK.md`
- **Test Script**: `scripts/test_extracted_block.py`

## Support

For issues or questions:
1. Check troubleshooting section above
2. Verify all prerequisites installed
3. Review validation output for hints
4. Check trace file for error messages

---

**Ready to capture traces? Start with the basic command and you'll have baseline traces in seconds!**

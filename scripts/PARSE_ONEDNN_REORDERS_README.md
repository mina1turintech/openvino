# oneDNN Reorder Operation Extraction and Analysis Tool

## Overview

The `parse_onednn_reorders.py` tool parses oneDNN verbose traces to extract and analyze reorder operation metrics, enabling detailed comparison between baseline and optimized builds. It identifies all reorder operations, aggregates timing and counts, breaks down metrics by dimension and operation type, and outputs results in CSV or JSON format for easy delta measurement and reporting.

## Purpose

This tool enables:
- **Reorder Operation Analysis**: Extract all reorder operations from verbose traces
- **Performance Metrics**: Aggregate execution times and operation counts
- **Dimension Breakdown**: Analyze reorders by tensor dimensions (1536, 8960, etc.)
- **Implementation Analysis**: Categorize by operation type (jit:uni, jit_direct_copy, etc.)
- **Layout Analysis**: Track memory layout transformations (ab → ba, etc.)
- **Baseline vs Optimized Comparison**: Side-by-side comparison with delta and percentage change
- **Flexible Output**: Export to CSV or JSON for further analysis and reporting

## Key Features

- ✅ Automatic reorder operation extraction from verbose traces
- ✅ Comprehensive metrics aggregation (count, time, averages)
- ✅ Multi-dimensional breakdown (by implementation, dimension, layout)
- ✅ Comparison mode with delta and percentage change calculation
- ✅ CSV and JSON output formats
- ✅ Batch processing support
- ✅ Error handling for malformed traces
- ✅ Clear command-line interface

## Installation & Requirements

### Prerequisites

```bash
# Python 3.10 or higher required
python --version

# No additional dependencies needed (uses stdlib only)
```

### Minimum Versions
- Python >= 3.10
- No external dependencies

## Usage

### Basic Usage

#### Analyze a Single Trace

Extract metrics from a single trace file and output to JSON:

```bash
python scripts/parse_onednn_reorders.py \
    --trace ./traces/onednn_trace_baseline.txt \
    --output-json ./results/baseline_metrics.json
```

Extract metrics and output to CSV:

```bash
python scripts/parse_onednn_reorders.py \
    --trace ./traces/onednn_trace_baseline.txt \
    --output-csv ./results/baseline_metrics.csv
```

#### Compare Baseline vs Optimized

Compare two traces and output comparison to CSV:

```bash
python scripts/parse_onednn_reorders.py \
    --baseline ./traces/onednn_trace_baseline.txt \
    --optimized ./traces/onednn_trace_optimized.txt \
    --output-csv ./results/comparison.csv
```

Compare and output to JSON:

```bash
python scripts/parse_onednn_reorders.py \
    --baseline ./traces/onednn_trace_baseline.txt \
    --optimized ./traces/onednn_trace_optimized.txt \
    --output-json ./results/comparison.json
```

#### Output Both CSV and JSON

```bash
python scripts/parse_onednn_reorders.py \
    --trace ./traces/onednn_trace_baseline.txt \
    --output-csv ./results/metrics.csv \
    --output-json ./results/metrics.json
```

### Command-Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--trace` | str | conditional* | Path to single trace file to analyze |
| `--baseline` | str | conditional* | Path to baseline trace file (for comparison) |
| `--optimized` | str | conditional† | Path to optimized trace file (for comparison) |
| `--output-csv` | str | conditional‡ | Output CSV file path |
| `--output-json` | str | conditional‡ | Output JSON file path |
| `--output` | str | conditional‡ | Output file path (extension determines format) |

**Notes:**
- *Either `--trace` OR `--baseline` must be specified (mutually exclusive)
- †`--optimized` requires `--baseline` (comparison mode)
- ‡At least one output format must be specified

## Output Formats

### CSV Output

The CSV output provides a tabular format suitable for spreadsheet analysis:

#### Single Trace CSV Structure

```csv
Category,Key,Count,Time_ms,Avg_Time_ms

Total,All Reorders,45,5.234567,0.116324

By Implementation Type
Implementation,Key,Count,Time_ms,Avg_Time_ms
Implementation,jit:uni,32,3.456789,0.108024
Implementation,jit_direct_copy:uni,8,1.234567,0.154321

By Dimension
Dimension,Key,Count,Time_ms,Avg_Time_ms
Dimension,1536x8960,12,1.876543,0.156379
Dimension,8960x1536,12,1.765432,0.147119
...
```

#### Comparison CSV Structure

```csv
Category,Key,Baseline_Count,Baseline_Time_ms,Optimized_Count,Optimized_Time_ms,Delta_Count,Delta_Time_ms,Change_Count_%,Change_Time_%

Total,All Reorders,45,5.234567,28,3.156789,-17,-2.077778,-37.78,-39.70

By Implementation Type
Category,Key,Baseline_Count,Baseline_Time_ms,Optimized_Count,Optimized_Time_ms,Delta_Count,Delta_Time_ms,Change_Count_%,Change_Time_%
Implementation Type,jit:uni,32,3.456789,20,2.123456,-12,-1.333333,-37.50,-38.57
...
```

### JSON Output

The JSON output provides structured data suitable for programmatic analysis:

#### Single Trace JSON Structure

```json
{
  "timestamp": "2025-01-10T14:30:22.123456",
  "metrics": {
    "total": {
      "count": 45,
      "time_ms": 5.234567
    },
    "by_implementation": {
      "jit:uni": {
        "count": 32,
        "time_ms": 3.456789
      }
    },
    "by_dimension": {
      "1536x8960": {
        "count": 12,
        "time_ms": 1.876543
      }
    },
    "by_dimension_value": {
      "1536": {
        "count": 45,
        "time_ms": 5.234567
      }
    },
    "by_layout_transformation": {
      "ab -> ba": {
        "count": 24,
        "time_ms": 3.641975
      }
    }
  },
  "parser_summary": {
    "trace_file": "./traces/onednn_trace_baseline.txt",
    "total_lines": 1247,
    "reorder_operations": 45,
    "parse_errors": 0
  }
}
```

#### Comparison JSON Structure

The comparison JSON includes both baseline and optimized metrics plus delta and percentage change:

```json
{
  "timestamp": "2025-01-10T15:45:33.789012",
  "comparison": {
    "total": {
      "baseline": {"count": 45, "time_ms": 5.234567},
      "optimized": {"count": 28, "time_ms": 3.156789},
      "delta": {"count": -17, "time_ms": -2.077778},
      "percent_change": {"count": -37.78, "time_ms": -39.70}
    }
  }
}
```

## Metrics Breakdown

### Total Metrics

- **count**: Total number of reorder operations
- **time_ms**: Cumulative execution time in milliseconds

### By Implementation Type

Categorizes reorders by oneDNN implementation:
- `jit:uni` - JIT compiled universal implementation
- `jit_direct_copy:uni` - JIT direct copy implementation
- `simple:any` - Simple reference implementation
- Others as detected in traces

### By Dimension

Groups reorders by tensor dimensions:
- `1536x8960` - Common for FFN intermediate projections
- `8960x1536` - Common for FFN output projections
- `1536x1536` - Common for attention projections
- `1x16x1536` - Common for batch/sequence operations
- Others as detected

### By Dimension Value

Aggregates reorders containing specific dimension values:
- `1536` - Hidden size dimension
- `8960` - FFN intermediate dimension
- `256` - Smaller projections
- Others as detected

### By Layout Transformation

Tracks memory layout conversions:
- `ab -> ba` - Row-major to column-major (transpose)
- `abc -> acb` - 3D tensor reorder
- `abcd -> acdb` - 4D tensor reorder
- Others as detected

## Understanding oneDNN Reorder Trace Format

### Trace Line Format

```
dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:ab:f0 dst_f32::blocked:ba:f0,,,1536x8960,0.123047
```

### Parsed Fields

| Field | Description | Example |
|-------|-------------|---------|
| **Operation Type** | Type of operation | `reorder` |
| **Implementation** | Implementation strategy | `jit:uni` |
| **Dimensions** | Tensor dimensions | `1536x8960` |
| **Execution Time** | Time in milliseconds | `0.123047` |
| **Source Layout** | Input memory layout | `ab` |
| **Destination Layout** | Output memory layout | `ba` |
| **Data Type** | Element data type | `f32` |

## Typical Workflow

### 1. Capture Baseline Trace

```bash
python scripts/capture_onednn_trace.py \
    --model-path ./extracted_block/transformer_block.xml \
    --output-dir ./traces \
    --tag baseline \
    --iterations 100
```

### 2. Extract Baseline Metrics

```bash
python scripts/parse_onednn_reorders.py \
    --trace ./traces/onednn_trace_baseline_*.txt \
    --output-json ./results/baseline_metrics.json \
    --output-csv ./results/baseline_metrics.csv
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

### 5. Compare Traces

```bash
python scripts/parse_onednn_reorders.py \
    --baseline ./traces/onednn_trace_baseline_*.txt \
    --optimized ./traces/onednn_trace_optimized_*.txt \
    --output-csv ./results/comparison.csv \
    --output-json ./results/comparison.json
```

### 6. Analyze Results

```bash
# View comparison in terminal
cat ./results/comparison.csv

# Or open in spreadsheet software
# Or analyze programmatically using the JSON output
```

## Interpreting Results

### Positive Optimization Indicators

- ✅ **Negative delta count**: Fewer reorder operations
- ✅ **Negative delta time**: Less time spent in reorders
- ✅ **Negative percentage change**: Reduction in operations/time

### Example Good Result

```
Total Reorder Operations:
  Baseline:  45 operations, 5.235 ms
  Optimized: 28 operations, 3.157 ms
  Delta:     -17 operations (-37.78%), -2.078 ms (-39.70%)
```

**Interpretation**: Optimization eliminated 17 reorder operations (37.78% reduction) and reduced reorder time by 2.078ms (39.70% reduction).

### Key Metrics to Watch

1. **Total Count**: Overall reduction in reorder operations
2. **Total Time**: Overall reduction in reorder execution time
3. **Dimension-Specific**: Which dimensions benefit most
4. **Implementation-Specific**: Which implementations are affected

### Red Flags

- ⚠️ **Positive delta**: More reorders after optimization
- ⚠️ **Increased time**: Reorders taking longer
- ⚠️ **New dimensions**: Unexpected tensor shapes appearing

## Advanced Usage

### Batch Processing Multiple Traces

Process multiple traces using shell scripting:

```bash
#!/bin/bash
# Process all traces in directory
for trace in ./traces/onednn_trace_*.txt; do
    basename=$(basename "$trace" .txt)
    python scripts/parse_onednn_reorders.py \
        --trace "$trace" \
        --output-json "./results/${basename}_metrics.json"
done
```

### Comparing Multiple Optimizations

```bash
# Compare baseline vs optimization 1
python scripts/parse_onednn_reorders.py \
    --baseline ./traces/baseline.txt \
    --optimized ./traces/opt1.txt \
    --output-csv ./results/comparison_opt1.csv

# Compare baseline vs optimization 2
python scripts/parse_onednn_reorders.py \
    --baseline ./traces/baseline.txt \
    --optimized ./traces/opt2.txt \
    --output-csv ./results/comparison_opt2.csv

# Compare the CSV files to find best optimization
```

### Integration with Spreadsheets

The CSV output can be directly imported into:
- **Excel**: File → Open → Select CSV
- **Google Sheets**: File → Import → Upload CSV
- **LibreOffice Calc**: Open the CSV file directly

Then create pivot tables, charts, and visualizations from the structured data.

### Programmatic Analysis

Use the JSON output for automated analysis:

```python
import json

# Load comparison results
with open('results/comparison.json', 'r') as f:
    data = json.load(f)

# Extract key metrics
total = data['comparison']['total']
baseline_count = total['baseline']['count']
optimized_count = total['optimized']['count']
reduction_pct = total['percent_change']['count']

print(f"Reorder count reduced by {reduction_pct:.2f}%")

# Analyze by dimension
for dim, metrics in data['comparison']['by_dimension'].items():
    if metrics['delta']['count'] < 0:
        print(f"Dimension {dim}: {metrics['delta']['count']} fewer reorders")
```

## Troubleshooting

### Empty or No Reorder Operations Found

**Problem**: Parser reports 0 reorder operations

**Solutions**:
1. Verify trace file contains oneDNN verbose output
2. Check that DNNL_VERBOSE was set during trace capture
3. Confirm trace file is not empty or corrupted
4. Try capturing trace with higher verbose level (DNNL_VERBOSE=2)

### Parse Errors Reported

**Problem**: Parser reports parse_errors > 0

**Possible Causes**:
1. Malformed trace lines (incomplete or corrupted)
2. Non-standard oneDNN verbose format
3. Mixed verbose output from different oneDNN versions

**Solutions**:
1. Check trace file integrity
2. Recapture trace if necessary
3. Review parse_errors count in output (small numbers may be acceptable)

### Unexpected Dimension Values

**Problem**: Dimensions don't match expected model dimensions

**Solutions**:
1. Verify correct model was used for trace capture
2. Check model input shapes during inference
3. Confirm batch size and sequence length parameters

### Comparison Shows Increased Reorders

**Problem**: Optimized build shows more reorders than baseline

**Possible Reasons**:
1. Optimization may have introduced additional transformations
2. Different code path triggered
3. Different oneDNN version or configuration

**Actions**:
1. Review optimization changes
2. Check by_dimension and by_implementation breakdowns
3. Verify expected optimization was actually applied
4. Consider reverting or modifying optimization strategy

## Performance Considerations

### Processing Large Traces

The tool is designed to handle large trace files efficiently:
- **Memory**: Processes line-by-line, minimal memory footprint
- **Speed**: Parses ~10,000 lines per second (typical)
- **Scalability**: Can process multi-megabyte trace files

### Batch Processing

For batch processing many traces:
1. Use shell scripts to iterate through traces
2. Output to separate files with descriptive names
3. Aggregate results post-processing if needed

## Example Outputs

Example output files are provided in `scripts/examples/`:

- `example_reorder_metrics.csv` - Single trace CSV output
- `example_reorder_metrics.json` - Single trace JSON output
- `example_comparison.csv` - Comparison CSV output
- `example_comparison.json` - Comparison JSON output

These examples demonstrate the expected format and structure of outputs.

## Success Criteria

The tool successfully meets requirements if:

- [x] Parser correctly identifies reorder operations in trace
- [x] Extracts timing and count data accurately
- [x] Segments metrics by dimension
- [x] Categorizes operations by implementation type
- [x] Outputs to CSV/JSON with clear column headers
- [x] Handles multiple trace files for batch processing
- [x] Calculates percentage change/delta for comparisons
- [x] Error handling for malformed traces
- [x] Performance: Processes traces in reasonable time
- [x] Tool successfully parses traces from capture harness
- [x] Metrics segmented correctly by dimension (1536, 8960)
- [x] CSV/JSON output is well-formatted and importable
- [x] Tool can compare two traces and report delta
- [x] Handles edge cases (empty traces, missing operations)
- [x] Runnable via command-line with clear usage instructions

## Limitations

- **Reorder Focus**: Only extracts reorder operations (not other oneDNN ops)
- **Text-Based**: Requires text-format verbose traces
- **Single-Threading**: Processes one trace at a time
- **No Visualization**: Outputs data only; visualization requires external tools

## Future Enhancements

Potential future improvements:
- Support for other operation types (matmul, convolution, etc.)
- Built-in visualization (charts, graphs)
- Multi-trace comparison (more than 2 traces)
- Statistical analysis (mean, median, std dev across runs)
- HTML report generation

## Related Tools

- **capture_onednn_trace.py**: Captures oneDNN verbose traces
- **extract_transformer_block.py**: Extracts transformer blocks for profiling

## Support

For issues or questions:
1. Check this README for common solutions
2. Review example outputs in `scripts/examples/`
3. Verify trace files are valid and properly formatted
4. Check that Python version meets requirements (≥3.10)

## License

Copyright (C) 2018-2026 Intel Corporation  
SPDX-License-Identifier: Apache-2.0

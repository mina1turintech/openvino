# Graph Optimizer Pass Instrumentation

## Overview

This document describes the instrumentation added to the OpenVINO CPU plugin's graph optimizer to trace layout optimization passes and their effects on reorder operations. This instrumentation is specifically designed to analyze how the transformer block graph is optimized, with a focus on reorder elimination and layout transformations.

## Instrumented Passes

The following graph optimizer passes have been instrumented:

1. **ShareReorders** - Identifies and shares identical reorder operations between sibling nodes
2. **DropDoubleReorders** - Eliminates consecutive reorder operations that can be replaced with a single reorder
3. **MergeTransposeAndReorder** - Merges transpose and reorder operations when they perform inverse transformations

## Environment Variables

### Enable Instrumentation

```bash
export OV_CPU_GRAPH_OPTIMIZER_TRACE=1
```

Set this to `1` to enable instrumentation. When disabled (default), there is zero overhead.

### Set Output File Path

```bash
export OV_CPU_GRAPH_OPTIMIZER_TRACE_FILE=/path/to/output.json
```

Specifies where to write the instrumentation trace. Default: `./graph_optimizer_trace.json`

## Output Format

The instrumentation generates a JSON file with the following structure:

```json
{
  "graph_optimizer_passes": [
    {
      "pass_name": "ShareReorders",
      "execution_time_us": 1234,
      "reorders_eliminated": 2,
      "before": {
        "total_nodes": 150,
        "reorder_count": 45,
        "matmul_count": 12,
        "fullyconnected_count": 8,
        "transpose_count": 10,
        "reorders": [
          {
            "name": "reorder_name",
            "parent": "parent_node",
            "child": "child_node",
            "input_dims": [1, 1536, 8960],
            "output_dims": [1, 1536, 8960],
            "is_optimized": false,
            "descriptor": "ab->ba"
          }
        ]
      },
      "after": {
        "total_nodes": 148,
        "reorder_count": 43,
        "matmul_count": 12,
        "fullyconnected_count": 8,
        "transpose_count": 10,
        "reorders": [ ... ]
      }
    },
    {
      "pass_name": "DropDoubleReorders",
      "execution_time_us": 567,
      "reorders_eliminated": 3,
      "before": { ... },
      "after": { ... }
    },
    {
      "pass_name": "MergeTransposeAndReorder",
      "execution_time_us": 890,
      "reorders_eliminated": 1,
      "before": { ... },
      "after": { ... }
    }
  ]
}
```

### Field Descriptions

- **pass_name**: Name of the optimization pass
- **execution_time_us**: Pass execution time in microseconds
- **reorders_eliminated**: Net change in reorder count (negative means reorders were added)
- **before/after**: Graph state snapshots before and after the pass
  - **total_nodes**: Total number of nodes in the graph
  - **reorder_count**: Number of reorder nodes
  - **matmul_count**: Number of MatMul operations
  - **fullyconnected_count**: Number of FullyConnected operations
  - **transpose_count**: Number of Transpose operations
  - **reorders**: Detailed information about each reorder node
    - **name**: Reorder node name
    - **parent**: Parent node name
    - **child**: Child node name
    - **input_dims**: Input tensor dimensions
    - **output_dims**: Output tensor dimensions
    - **is_optimized**: Whether this is an optimized (zero-copy) reorder
    - **descriptor**: Layout transformation description (e.g., "ab->ba")

## Usage Examples

### Basic Tracing

```bash
# Enable tracing with default output file
export OV_CPU_GRAPH_OPTIMIZER_TRACE=1
python scripts/run_single_block_test.py
# Output: ./graph_optimizer_trace.json
```

### Custom Output Location

```bash
# Specify custom output file
export OV_CPU_GRAPH_OPTIMIZER_TRACE=1
export OV_CPU_GRAPH_OPTIMIZER_TRACE_FILE=./traces/optimizer_pass_trace.json
python scripts/run_single_block_test.py
```

### Analyzing Transformer Block Optimization

```bash
# Extract single transformer block
python scripts/extract_transformer_block.py --model Qwen/Qwen2-1.5B --layer-index 0

# Enable tracing
export OV_CPU_GRAPH_OPTIMIZER_TRACE=1
export OV_CPU_GRAPH_OPTIMIZER_TRACE_FILE=./transformer_block_optimization.json

# Run inference
python scripts/run_single_block_test.py --model-dir transformer_block_layer0.xml

# Analyze results
python scripts/analyze_optimizer_trace.py ./transformer_block_optimization.json
```

## Analysis Workflow

### 1. Capture Baseline Trace

First, capture a baseline trace to understand current optimization behavior:

```bash
./scripts/capture_baseline_trace.sh
export OV_CPU_GRAPH_OPTIMIZER_TRACE=1
export OV_CPU_GRAPH_OPTIMIZER_TRACE_FILE=./baseline_optimizer_trace.json
```

### 2. Identify Optimization Opportunities

Examine the trace to identify:
- **High reorder count after optimization**: Indicates missed optimization opportunities
- **Large dimension reorders**: Focus on reorders involving dimensions like 8960, 1536
- **Reorders around MatMul/FullyConnected**: These are critical for performance
- **Zero eliminated reorders**: Passes that didn't reduce reorder count

### 3. Compare with oneDNN Trace

Correlate optimizer trace with oneDNN runtime trace:

```bash
# Compare pass-level reorder counts with runtime reorder operations
python scripts/correlate_optimizer_and_runtime_traces.py \
    --optimizer-trace ./baseline_optimizer_trace.json \
    --runtime-trace ./baseline_capture/traces/onednn_trace_baseline_run1.txt
```

### 4. Validate Optimizations

After implementing optimizations, capture a new trace and compare:

```bash
export OV_CPU_GRAPH_OPTIMIZER_TRACE=1
export OV_CPU_GRAPH_OPTIMIZER_TRACE_FILE=./optimized_trace.json

# Run with optimizations enabled
python scripts/run_single_block_test.py --with-optimizations

# Compare traces
python scripts/compare_optimizer_traces.py \
    --baseline ./baseline_optimizer_trace.json \
    --optimized ./optimized_trace.json
```

## Key Metrics to Track

### Pass-Level Metrics

1. **Reorders Eliminated**: Total reorders removed by all passes
2. **Execution Time**: Time spent in each optimization pass
3. **Pass Effectiveness**: Percentage of reorders eliminated per pass

### Graph-Level Metrics

1. **Initial vs Final Reorder Count**: Overall reduction in reorder operations
2. **Optimized Reorders**: Number of zero-copy reorders
3. **Critical Path Reorders**: Reorders on MatMul/FullyConnected inputs/outputs

### Dimension-Specific Metrics

1. **8960-dimension Reorders**: Count of reorders involving the large dimension
2. **1536-dimension Reorders**: Count of reorders involving the medium dimension
3. **Layout Patterns**: Common layout transformations (ab->ba, abc->acb)

## Integration with Baseline Workflow

The optimizer trace complements the baseline trace capture workflow:

```bash
# 1. Capture complete baseline (includes optimizer trace)
./scripts/capture_baseline_trace.sh

# 2. Review optimizer decisions
cat ./baseline_capture/optimizer_trace.json | jq '.graph_optimizer_passes[] | {pass: .pass_name, eliminated: .reorders_eliminated}'

# 3. Compare with runtime behavior
python scripts/correlate_traces.py \
    --optimizer ./baseline_capture/optimizer_trace.json \
    --runtime ./baseline_capture/traces/onednn_trace_baseline_run1.txt

# 4. Identify mismatches
# - Reorders eliminated at compile-time but still present at runtime
# - Unexpected reorder operations not captured during optimization
```

## Performance Considerations

### Overhead

- **When Disabled**: Zero overhead (checked once at startup via static variable)
- **When Enabled**: Typically < 5% overhead
  - Graph state capture: ~1-2ms per pass
  - JSON writing: ~1-3ms per pass
  - Total: ~10-15ms for typical transformer block

### Recommendations

1. **Enable only when needed**: Use for analysis, disable for production
2. **Focus on specific passes**: Can be extended to instrument only certain passes
3. **Batch analysis**: Capture traces once, analyze multiple times offline

## Troubleshooting

### No Output File Generated

- Check that `OV_CPU_GRAPH_OPTIMIZER_TRACE=1` is set
- Verify write permissions in output directory
- Check for disk space

### Incomplete JSON

- Ensure the inference completes without errors
- Check that `FinalizeInstrumentation()` is called
- Verify no early termination or exceptions

### Missing Reorder Information

- Some reorders may not have descriptor information if:
  - Reorder is still being constructed
  - Memory descriptors are not yet defined
  - Graph is in intermediate state

### High Overhead

- Reduce output detail by modifying `captureGraphState()`
- Write to faster storage (e.g., tmpfs: `/tmp`)
- Capture fewer runs

## Future Extensions

Potential enhancements to the instrumentation:

1. **Pass-specific metrics**: Custom metrics per optimization pass
2. **Diff visualization**: Before/after graph diffs
3. **Pattern detection**: Automatic detection of common optimization patterns
4. **Performance correlation**: Link optimizer decisions to runtime performance
5. **Interactive analysis**: Web-based trace viewer

## Technical Details

### Implementation

- **Location**: `src/plugins/intel_cpu/src/graph_optimizer.cpp`
- **Approach**: RAII-based instrumentation via `PassInstrumentor` class
- **Thread Safety**: Not thread-safe (assumes single-threaded graph optimization)
- **Memory**: Minimal heap allocations, uses move semantics where possible

### Key Functions

- `initInstrumentationFile()`: Initializes JSON output file
- `captureGraphState()`: Captures complete graph state snapshot
- `PassInstrumentor` constructor: Records pre-pass state and start time
- `PassInstrumentor` destructor: Records post-pass state and writes results
- `finalizeInstrumentationFile()`: Closes JSON array and file

### Design Decisions

1. **JSON Format**: Human-readable and machine-parseable
2. **Environment Variables**: Easy to enable/disable without recompilation
3. **Zero-overhead when disabled**: Static initialization check
4. **RAII Pattern**: Automatic measurement even with exceptions
5. **Complete State Capture**: All reorder details for comprehensive analysis

## Related Documentation

- [Baseline Capture README](./BASELINE_CAPTURE_README.md)
- [Graph Optimizer Source](../src/plugins/intel_cpu/src/graph_optimizer.cpp)
- [Reorder Node Implementation](../src/plugins/intel_cpu/src/nodes/reorder.cpp)

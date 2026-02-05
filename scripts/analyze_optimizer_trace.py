#!/usr/bin/env python3
"""
Analyze graph optimizer trace output to identify optimization opportunities.

Usage:
    python analyze_optimizer_trace.py <trace_file.json>
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_trace(trace_file):
    """Load and parse the optimizer trace JSON file."""
    with open(trace_file, 'r') as f:
        return json.load(f)


def analyze_pass_effectiveness(passes):
    """Analyze how effective each pass was at eliminating reorders."""
    print("\n" + "="*80)
    print("PASS EFFECTIVENESS ANALYSIS")
    print("="*80)
    
    total_eliminated = 0
    total_time_us = 0
    
    for pass_data in passes:
        pass_name = pass_data['pass_name']
        eliminated = pass_data['reorders_eliminated']
        time_us = pass_data['execution_time_us']
        
        total_eliminated += eliminated
        total_time_us += time_us
        
        before_count = pass_data['before']['reorder_count']
        after_count = pass_data['after']['reorder_count']
        
        effectiveness = (eliminated / before_count * 100) if before_count > 0 else 0
        
        print(f"\n{pass_name}:")
        print(f"  Execution time: {time_us:,} μs ({time_us/1000:.2f} ms)")
        print(f"  Reorders before: {before_count}")
        print(f"  Reorders after: {after_count}")
        print(f"  Reorders eliminated: {eliminated}")
        print(f"  Effectiveness: {effectiveness:.1f}%")
    
    print(f"\n{'TOTAL':-<40}")
    print(f"  Total time: {total_time_us:,} μs ({total_time_us/1000:.2f} ms)")
    print(f"  Total eliminated: {total_eliminated}")


def analyze_reorder_dimensions(passes):
    """Analyze reorder operations by dimension."""
    print("\n" + "="*80)
    print("DIMENSION ANALYSIS")
    print("="*80)
    
    # Track dimensions before and after optimization
    initial_dims = defaultdict(int)
    final_dims = defaultdict(int)
    
    first_pass = passes[0]
    last_pass = passes[-1]
    
    # Count initial dimensions
    for reorder in first_pass['before']['reorders']:
        dims = tuple(reorder['input_dims'])
        initial_dims[dims] += 1
    
    # Count final dimensions
    for reorder in last_pass['after']['reorders']:
        dims = tuple(reorder['input_dims'])
        final_dims[dims] += 1
    
    print("\nReorders by dimension (before optimization):")
    for dims, count in sorted(initial_dims.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {list(dims)}: {count} reorders")
    
    print("\nReorders by dimension (after optimization):")
    for dims, count in sorted(final_dims.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {list(dims)}: {count} reorders")
    
    # Check for large dimension reorders (like 8960, 1536)
    print("\nLarge dimension reorders:")
    for dims, count in final_dims.items():
        if any(d > 1000 for d in dims):
            eliminated = initial_dims.get(dims, 0) - count
            print(f"  {list(dims)}: {count} remaining (eliminated {eliminated})")


def analyze_operation_types(passes):
    """Analyze operations affected by reorders."""
    print("\n" + "="*80)
    print("OPERATION TYPE ANALYSIS")
    print("="*80)
    
    first_pass = passes[0]
    last_pass = passes[-1]
    
    print(f"\nGraph composition (before optimization):")
    print(f"  Total nodes: {first_pass['before']['total_nodes']}")
    print(f"  Reorders: {first_pass['before']['reorder_count']}")
    print(f"  MatMul: {first_pass['before']['matmul_count']}")
    print(f"  FullyConnected: {first_pass['before']['fullyconnected_count']}")
    print(f"  Transpose: {first_pass['before']['transpose_count']}")
    
    print(f"\nGraph composition (after optimization):")
    print(f"  Total nodes: {last_pass['after']['total_nodes']}")
    print(f"  Reorders: {last_pass['after']['reorder_count']}")
    print(f"  MatMul: {last_pass['after']['matmul_count']}")
    print(f"  FullyConnected: {last_pass['after']['fullyconnected_count']}")
    print(f"  Transpose: {last_pass['after']['transpose_count']}")
    
    # Calculate ratios
    final_reorders = last_pass['after']['reorder_count']
    matmul_count = last_pass['after']['matmul_count']
    fc_count = last_pass['after']['fullyconnected_count']
    compute_ops = matmul_count + fc_count
    
    if compute_ops > 0:
        ratio = final_reorders / compute_ops
        print(f"\nReorder-to-compute ratio: {ratio:.2f} reorders per compute op")


def analyze_optimized_reorders(passes):
    """Analyze optimized (zero-copy) reorders."""
    print("\n" + "="*80)
    print("OPTIMIZED REORDER ANALYSIS")
    print("="*80)
    
    last_pass = passes[-1]
    reorders = last_pass['after']['reorders']
    
    optimized_count = sum(1 for r in reorders if r['is_optimized'])
    total_count = len(reorders)
    
    print(f"\nOptimized (zero-copy) reorders: {optimized_count}/{total_count}")
    if total_count > 0:
        print(f"Percentage: {optimized_count/total_count*100:.1f}%")
    
    # Show non-optimized reorders with large dimensions
    print("\nNon-optimized reorders with large dimensions:")
    large_non_optimized = [r for r in reorders 
                           if not r['is_optimized'] 
                           and any(d > 1000 for d in r['input_dims'])]
    
    for i, reorder in enumerate(large_non_optimized[:10], 1):
        print(f"\n  {i}. {reorder['name']}")
        print(f"     Parent: {reorder['parent']}")
        print(f"     Child: {reorder['child']}")
        print(f"     Dims: {reorder['input_dims']}")
        print(f"     Layout: {reorder['descriptor']}")


def identify_optimization_opportunities(passes):
    """Identify potential optimization opportunities."""
    print("\n" + "="*80)
    print("OPTIMIZATION OPPORTUNITIES")
    print("="*80)
    
    last_pass = passes[-1]
    final_reorders = last_pass['after']['reorders']
    
    # Group reorders by parent-child pairs
    parent_child_pairs = defaultdict(list)
    for reorder in final_reorders:
        key = (reorder['parent'], reorder['child'])
        parent_child_pairs[key].append(reorder)
    
    # Find multiple reorders between same nodes
    print("\nMultiple reorders between same nodes:")
    found = False
    for (parent, child), reorders in parent_child_pairs.items():
        if len(reorders) > 1:
            found = True
            print(f"\n  {parent} -> {child}: {len(reorders)} reorders")
            for r in reorders:
                print(f"    - {r['name']}: {r['descriptor']}")
    
    if not found:
        print("  None found")
    
    # Check for reorders with identical layouts
    print("\nReorders with identical layout transformations:")
    layout_groups = defaultdict(list)
    for reorder in final_reorders:
        layout_groups[reorder['descriptor']].append(reorder)
    
    found = False
    for layout, reorders in layout_groups.items():
        if len(reorders) > 2 and layout != "unknown":
            found = True
            print(f"\n  {layout}: {len(reorders)} instances")
            for r in reorders[:3]:
                print(f"    - {r['name']} ({r['input_dims']})")
            if len(reorders) > 3:
                print(f"    ... and {len(reorders) - 3} more")
    
    if not found:
        print("  None found")


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_optimizer_trace.py <trace_file.json>")
        sys.exit(1)
    
    trace_file = sys.argv[1]
    
    if not Path(trace_file).exists():
        print(f"Error: File not found: {trace_file}")
        sys.exit(1)
    
    print(f"Analyzing optimizer trace: {trace_file}")
    
    try:
        trace = load_trace(trace_file)
        passes = trace['graph_optimizer_passes']
        
        if not passes:
            print("Error: No optimizer passes found in trace")
            sys.exit(1)
        
        analyze_pass_effectiveness(passes)
        analyze_reorder_dimensions(passes)
        analyze_operation_types(passes)
        analyze_optimized_reorders(passes)
        identify_optimization_opportunities(passes)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in trace file: {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing expected field in trace: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

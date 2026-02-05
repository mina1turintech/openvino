#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
oneDNN Reorder Operation Extraction and Analysis Tool

This script parses oneDNN verbose traces to extract reorder operation metrics,
enabling comparison between baseline and optimized builds. It identifies all
reorder operations, aggregates timing and counts, breaks down metrics by
dimension and operation type, and outputs results in CSV or JSON format.

Usage:
    # Analyze a single trace
    python parse_onednn_reorders.py --trace baseline_trace.txt --output results.json
    
    # Compare baseline vs optimized
    python parse_onednn_reorders.py --baseline baseline.txt --optimized optimized.txt --output comparison.csv
    
    # Output both CSV and JSON
    python parse_onednn_reorders.py --trace trace.txt --output-csv metrics.csv --output-json metrics.json
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ReorderOperation:
    """Represents a single reorder operation from oneDNN trace."""
    
    def __init__(self, line: str, line_number: int):
        """
        Parse a reorder operation from a trace line.
        
        Args:
            line: Raw trace line containing reorder operation
            line_number: Line number in the trace file
        """
        self.line = line.strip()
        self.line_number = line_number
        self.operation_type = None
        self.implementation = None
        self.dimensions = []
        self.dimension_str = None
        self.time_ms = None
        self.src_layout = None
        self.dst_layout = None
        self.data_type = None
        
        self._parse_line()
    
    def _parse_line(self):
        """Parse the trace line to extract reorder operation details."""
        # Expected format:
        # dnnl_verbose,exec,cpu,reorder,jit:uni,undef,src_f32::blocked:ab:f0 dst_f32::blocked:ba:f0,,,1536x8960,0.123
        
        parts = self.line.split(',')
        
        if len(parts) < 3:
            return
        
        # Extract operation type (should be 'reorder')
        if len(parts) >= 4:
            self.operation_type = parts[3].strip()
        
        # Extract implementation (e.g., jit:uni, jit_direct_copy:uni)
        if len(parts) >= 5:
            self.implementation = parts[4].strip()
        
        # Extract dimensions (usually second to last field)
        # Look for dimension patterns like "1536x8960", "mb1ic1536oc8960", etc.
        for part in reversed(parts):
            part = part.strip()
            # Pattern for dimensions: numbers separated by 'x' or embedded in mb/ic/oc notation
            if 'x' in part and part.replace('x', '').replace('.', '').isdigit():
                self.dimension_str = part
                # Parse individual dimensions
                try:
                    dims = part.split('x')
                    self.dimensions = [int(d) for d in dims if d.replace('.', '').isdigit()]
                except ValueError:
                    pass
                break
            # Alternative pattern: mb1ic1536oc8960
            elif 'mb' in part or 'ic' in part or 'oc' in part:
                self.dimension_str = part
                # Extract dimensions from mb/ic/oc notation
                ic_match = re.search(r'ic(\d+)', part)
                oc_match = re.search(r'oc(\d+)', part)
                if ic_match:
                    self.dimensions.append(int(ic_match.group(1)))
                if oc_match:
                    self.dimensions.append(int(oc_match.group(1)))
                break
        
        # Extract execution time (last field, in milliseconds)
        if parts:
            try:
                self.time_ms = float(parts[-1].strip())
            except ValueError:
                pass
        
        # Extract layout information
        # Look for src/dst layout patterns: src_f32::blocked:ab:f0 dst_f32::blocked:ba:f0
        layout_part = None
        for i, part in enumerate(parts):
            if 'src_' in part or 'dst_' in part:
                # Combine consecutive parts that contain layout info
                layout_parts = []
                for j in range(i, min(i+3, len(parts))):
                    if parts[j].strip():
                        layout_parts.append(parts[j].strip())
                layout_part = ' '.join(layout_parts)
                break
        
        if layout_part:
            # Extract data type
            dtype_match = re.search(r'src_(\w+)::', layout_part)
            if dtype_match:
                self.data_type = dtype_match.group(1)
            
            # Extract src layout
            src_match = re.search(r'src_\w+::blocked:(\w+):', layout_part)
            if src_match:
                self.src_layout = src_match.group(1)
            
            # Extract dst layout
            dst_match = re.search(r'dst_\w+::blocked:(\w+):', layout_part)
            if dst_match:
                self.dst_layout = dst_match.group(1)
    
    def is_valid(self) -> bool:
        """Check if the reorder operation was parsed successfully."""
        return (self.operation_type == 'reorder' and 
                self.time_ms is not None and 
                self.dimension_str is not None)
    
    def get_primary_dimensions(self) -> Tuple[int, ...]:
        """Get the primary dimensions as a tuple for grouping."""
        return tuple(sorted(self.dimensions)) if self.dimensions else ()
    
    def __repr__(self) -> str:
        return (f"ReorderOp(impl={self.implementation}, dims={self.dimension_str}, "
                f"time={self.time_ms}ms, src={self.src_layout}, dst={self.dst_layout})")


class TraceParser:
    """Parser for oneDNN verbose trace files."""
    
    def __init__(self, trace_file: str):
        """
        Initialize trace parser.
        
        Args:
            trace_file: Path to trace file
        """
        self.trace_file = trace_file
        self.reorder_ops: List[ReorderOperation] = []
        self.total_lines = 0
        self.parse_errors = 0
    
    def parse(self) -> List[ReorderOperation]:
        """
        Parse the trace file and extract all reorder operations.
        
        Returns:
            List of ReorderOperation objects
        """
        if not os.path.exists(self.trace_file):
            raise FileNotFoundError(f"Trace file not found: {self.trace_file}")
        
        with open(self.trace_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                self.total_lines += 1
                
                # Skip empty lines and non-verbose lines
                if not line.strip() or 'dnnl_verbose' not in line.lower():
                    continue
                
                # Only process reorder operations
                if 'reorder' not in line.lower():
                    continue
                
                try:
                    reorder_op = ReorderOperation(line, line_num)
                    if reorder_op.is_valid():
                        self.reorder_ops.append(reorder_op)
                except Exception as e:
                    self.parse_errors += 1
        
        return self.reorder_ops
    
    def get_summary(self) -> Dict:
        """Get parsing summary statistics."""
        return {
            'trace_file': self.trace_file,
            'total_lines': self.total_lines,
            'reorder_operations': len(self.reorder_ops),
            'parse_errors': self.parse_errors,
        }


class MetricsAggregator:
    """Aggregates reorder operation metrics for analysis."""
    
    def __init__(self, reorder_ops: List[ReorderOperation]):
        """
        Initialize metrics aggregator.
        
        Args:
            reorder_ops: List of reorder operations to aggregate
        """
        self.reorder_ops = reorder_ops
        self.metrics = self._compute_metrics()
    
    def _compute_metrics(self) -> Dict:
        """Compute aggregated metrics from reorder operations."""
        if not self.reorder_ops:
            return self._empty_metrics()
        
        # Total metrics
        total_count = len(self.reorder_ops)
        total_time_ms = sum(op.time_ms for op in self.reorder_ops if op.time_ms)
        
        # By implementation type
        by_implementation = defaultdict(lambda: {'count': 0, 'time_ms': 0.0})
        for op in self.reorder_ops:
            if op.implementation:
                by_implementation[op.implementation]['count'] += 1
                if op.time_ms:
                    by_implementation[op.implementation]['time_ms'] += op.time_ms
        
        # By dimension
        by_dimension = defaultdict(lambda: {'count': 0, 'time_ms': 0.0})
        for op in self.reorder_ops:
            if op.dimension_str:
                by_dimension[op.dimension_str]['count'] += 1
                if op.time_ms:
                    by_dimension[op.dimension_str]['time_ms'] += op.time_ms
        
        # By individual dimension values (e.g., 1536, 8960)
        by_dim_value = defaultdict(lambda: {'count': 0, 'time_ms': 0.0})
        for op in self.reorder_ops:
            for dim in op.dimensions:
                by_dim_value[dim]['count'] += 1
                if op.time_ms:
                    by_dim_value[dim]['time_ms'] += op.time_ms
        
        # By layout transformation
        by_layout = defaultdict(lambda: {'count': 0, 'time_ms': 0.0})
        for op in self.reorder_ops:
            if op.src_layout and op.dst_layout:
                layout_key = f"{op.src_layout} -> {op.dst_layout}"
                by_layout[layout_key]['count'] += 1
                if op.time_ms:
                    by_layout[layout_key]['time_ms'] += op.time_ms
        
        return {
            'total': {
                'count': total_count,
                'time_ms': total_time_ms,
            },
            'by_implementation': dict(by_implementation),
            'by_dimension': dict(by_dimension),
            'by_dimension_value': dict(by_dim_value),
            'by_layout_transformation': dict(by_layout),
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure."""
        return {
            'total': {'count': 0, 'time_ms': 0.0},
            'by_implementation': {},
            'by_dimension': {},
            'by_dimension_value': {},
            'by_layout_transformation': {},
        }
    
    def get_metrics(self) -> Dict:
        """Get computed metrics."""
        return self.metrics


class MetricsComparator:
    """Compares metrics between baseline and optimized traces."""
    
    def __init__(self, baseline_metrics: Dict, optimized_metrics: Dict):
        """
        Initialize comparator.
        
        Args:
            baseline_metrics: Metrics from baseline trace
            optimized_metrics: Metrics from optimized trace
        """
        self.baseline = baseline_metrics
        self.optimized = optimized_metrics
        self.comparison = self._compute_comparison()
    
    def _compute_comparison(self) -> Dict:
        """Compute comparison metrics."""
        comparison = {
            'total': self._compare_totals(),
            'by_implementation': self._compare_category('by_implementation'),
            'by_dimension': self._compare_category('by_dimension'),
            'by_dimension_value': self._compare_category('by_dimension_value'),
            'by_layout_transformation': self._compare_category('by_layout_transformation'),
        }
        return comparison
    
    def _compare_totals(self) -> Dict:
        """Compare total metrics."""
        baseline_total = self.baseline.get('total', {})
        optimized_total = self.optimized.get('total', {})
        
        baseline_count = baseline_total.get('count', 0)
        optimized_count = optimized_total.get('count', 0)
        baseline_time = baseline_total.get('time_ms', 0.0)
        optimized_time = optimized_total.get('time_ms', 0.0)
        
        count_delta = optimized_count - baseline_count
        count_pct = self._percent_change(baseline_count, optimized_count)
        
        time_delta = optimized_time - baseline_time
        time_pct = self._percent_change(baseline_time, optimized_time)
        
        return {
            'baseline': {'count': baseline_count, 'time_ms': baseline_time},
            'optimized': {'count': optimized_count, 'time_ms': optimized_time},
            'delta': {'count': count_delta, 'time_ms': time_delta},
            'percent_change': {'count': count_pct, 'time_ms': time_pct},
        }
    
    def _compare_category(self, category: str) -> Dict:
        """Compare metrics for a specific category."""
        baseline_cat = self.baseline.get(category, {})
        optimized_cat = self.optimized.get(category, {})
        
        # Get all keys from both
        all_keys = set(baseline_cat.keys()) | set(optimized_cat.keys())
        
        comparison = {}
        for key in all_keys:
            baseline_data = baseline_cat.get(key, {'count': 0, 'time_ms': 0.0})
            optimized_data = optimized_cat.get(key, {'count': 0, 'time_ms': 0.0})
            
            baseline_count = baseline_data.get('count', 0)
            optimized_count = optimized_data.get('count', 0)
            baseline_time = baseline_data.get('time_ms', 0.0)
            optimized_time = optimized_data.get('time_ms', 0.0)
            
            count_delta = optimized_count - baseline_count
            count_pct = self._percent_change(baseline_count, optimized_count)
            
            time_delta = optimized_time - baseline_time
            time_pct = self._percent_change(baseline_time, optimized_time)
            
            comparison[key] = {
                'baseline': {'count': baseline_count, 'time_ms': baseline_time},
                'optimized': {'count': optimized_count, 'time_ms': optimized_time},
                'delta': {'count': count_delta, 'time_ms': time_delta},
                'percent_change': {'count': count_pct, 'time_ms': time_pct},
            }
        
        return comparison
    
    def _percent_change(self, baseline: float, optimized: float) -> float:
        """Calculate percent change from baseline to optimized."""
        if baseline == 0:
            if optimized == 0:
                return 0.0
            return float('inf')
        return ((optimized - baseline) / baseline) * 100.0
    
    def get_comparison(self) -> Dict:
        """Get comparison results."""
        return self.comparison


class OutputFormatter:
    """Formats metrics output to CSV or JSON."""
    
    @staticmethod
    def to_csv(metrics: Dict, output_file: str, comparison: Optional[Dict] = None):
        """
        Write metrics to CSV file.
        
        Args:
            metrics: Metrics dictionary
            output_file: Output CSV file path
            comparison: Optional comparison metrics
        """
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            if comparison:
                OutputFormatter._write_comparison_csv(writer, comparison)
            else:
                OutputFormatter._write_single_csv(writer, metrics)
        
        print(f"✓ CSV output written to: {output_file}")
    
    @staticmethod
    def _write_single_csv(writer, metrics: Dict):
        """Write single trace metrics to CSV."""
        # Write header
        writer.writerow(['Category', 'Key', 'Count', 'Time_ms', 'Avg_Time_ms'])
        writer.writerow([])
        
        # Write total
        total = metrics.get('total', {})
        count = total.get('count', 0)
        time_ms = total.get('time_ms', 0.0)
        avg_time = time_ms / count if count > 0 else 0.0
        writer.writerow(['Total', 'All Reorders', count, f"{time_ms:.6f}", f"{avg_time:.6f}"])
        writer.writerow([])
        
        # Write by implementation
        writer.writerow(['By Implementation Type'])
        writer.writerow(['Implementation', 'Key', 'Count', 'Time_ms', 'Avg_Time_ms'])
        for impl, data in sorted(metrics.get('by_implementation', {}).items()):
            count = data.get('count', 0)
            time_ms = data.get('time_ms', 0.0)
            avg_time = time_ms / count if count > 0 else 0.0
            writer.writerow(['Implementation', impl, count, f"{time_ms:.6f}", f"{avg_time:.6f}"])
        writer.writerow([])
        
        # Write by dimension
        writer.writerow(['By Dimension'])
        writer.writerow(['Dimension', 'Key', 'Count', 'Time_ms', 'Avg_Time_ms'])
        for dim, data in sorted(metrics.get('by_dimension', {}).items()):
            count = data.get('count', 0)
            time_ms = data.get('time_ms', 0.0)
            avg_time = time_ms / count if count > 0 else 0.0
            writer.writerow(['Dimension', dim, count, f"{time_ms:.6f}", f"{avg_time:.6f}"])
        writer.writerow([])
        
        # Write by dimension value
        writer.writerow(['By Dimension Value'])
        writer.writerow(['Dimension Value', 'Key', 'Count', 'Time_ms', 'Avg_Time_ms'])
        for dim_val, data in sorted(metrics.get('by_dimension_value', {}).items(), 
                                     key=lambda x: int(x[0]) if isinstance(x[0], (int, str)) and str(x[0]).isdigit() else 0):
            count = data.get('count', 0)
            time_ms = data.get('time_ms', 0.0)
            avg_time = time_ms / count if count > 0 else 0.0
            writer.writerow(['Dimension Value', dim_val, count, f"{time_ms:.6f}", f"{avg_time:.6f}"])
        writer.writerow([])
        
        # Write by layout transformation
        writer.writerow(['By Layout Transformation'])
        writer.writerow(['Layout', 'Key', 'Count', 'Time_ms', 'Avg_Time_ms'])
        for layout, data in sorted(metrics.get('by_layout_transformation', {}).items()):
            count = data.get('count', 0)
            time_ms = data.get('time_ms', 0.0)
            avg_time = time_ms / count if count > 0 else 0.0
            writer.writerow(['Layout', layout, count, f"{time_ms:.6f}", f"{avg_time:.6f}"])
    
    @staticmethod
    def _write_comparison_csv(writer, comparison: Dict):
        """Write comparison metrics to CSV."""
        # Write header
        writer.writerow(['Category', 'Key', 
                        'Baseline_Count', 'Baseline_Time_ms',
                        'Optimized_Count', 'Optimized_Time_ms',
                        'Delta_Count', 'Delta_Time_ms',
                        'Change_Count_%', 'Change_Time_%'])
        writer.writerow([])
        
        # Write total comparison
        total = comparison.get('total', {})
        baseline = total.get('baseline', {})
        optimized = total.get('optimized', {})
        delta = total.get('delta', {})
        pct = total.get('percent_change', {})
        
        writer.writerow([
            'Total', 'All Reorders',
            baseline.get('count', 0), f"{baseline.get('time_ms', 0.0):.6f}",
            optimized.get('count', 0), f"{optimized.get('time_ms', 0.0):.6f}",
            delta.get('count', 0), f"{delta.get('time_ms', 0.0):.6f}",
            f"{pct.get('count', 0.0):.2f}", f"{pct.get('time_ms', 0.0):.2f}"
        ])
        writer.writerow([])
        
        # Helper function to write category comparison
        def write_category(category_name: str, category_key: str):
            writer.writerow([f'By {category_name}'])
            writer.writerow(['Category', 'Key', 
                            'Baseline_Count', 'Baseline_Time_ms',
                            'Optimized_Count', 'Optimized_Time_ms',
                            'Delta_Count', 'Delta_Time_ms',
                            'Change_Count_%', 'Change_Time_%'])
            
            for key, data in sorted(comparison.get(category_key, {}).items()):
                baseline = data.get('baseline', {})
                optimized = data.get('optimized', {})
                delta = data.get('delta', {})
                pct = data.get('percent_change', {})
                
                writer.writerow([
                    category_name, key,
                    baseline.get('count', 0), f"{baseline.get('time_ms', 0.0):.6f}",
                    optimized.get('count', 0), f"{optimized.get('time_ms', 0.0):.6f}",
                    delta.get('count', 0), f"{delta.get('time_ms', 0.0):.6f}",
                    f"{pct.get('count', 0.0):.2f}", f"{pct.get('time_ms', 0.0):.2f}"
                ])
            writer.writerow([])
        
        # Write all categories
        write_category('Implementation Type', 'by_implementation')
        write_category('Dimension', 'by_dimension')
        write_category('Dimension Value', 'by_dimension_value')
        write_category('Layout Transformation', 'by_layout_transformation')
    
    @staticmethod
    def to_json(metrics: Dict, output_file: str, comparison: Optional[Dict] = None,
                parser_summary: Optional[Dict] = None):
        """
        Write metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary
            output_file: Output JSON file path
            comparison: Optional comparison metrics
            parser_summary: Optional parser summary
        """
        output = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
        }
        
        if comparison:
            output['comparison'] = comparison
        
        if parser_summary:
            output['parser_summary'] = parser_summary
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ JSON output written to: {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Parse oneDNN verbose traces and extract reorder operation metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single trace and output JSON
  python parse_onednn_reorders.py --trace baseline_trace.txt --output-json results.json
  
  # Analyze a single trace and output CSV
  python parse_onednn_reorders.py --trace baseline_trace.txt --output-csv results.csv
  
  # Compare baseline vs optimized traces
  python parse_onednn_reorders.py --baseline baseline.txt --optimized optimized.txt --output-csv comparison.csv
  
  # Output both CSV and JSON
  python parse_onednn_reorders.py --trace trace.txt --output-csv metrics.csv --output-json metrics.json
  
  # Compare and output both formats
  python parse_onednn_reorders.py --baseline baseline.txt --optimized optimized.txt \\
      --output-csv comparison.csv --output-json comparison.json
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--trace',
        type=str,
        help='Path to single trace file to analyze'
    )
    input_group.add_argument(
        '--baseline',
        type=str,
        help='Path to baseline trace file (for comparison mode)'
    )
    
    parser.add_argument(
        '--optimized',
        type=str,
        help='Path to optimized trace file (for comparison mode, requires --baseline)'
    )
    
    # Output options
    parser.add_argument(
        '--output-csv',
        type=str,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--output-json',
        type=str,
        help='Output JSON file path'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (format determined by extension: .csv or .json)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.baseline and not args.optimized:
        parser.error("--baseline requires --optimized for comparison mode")
    
    if args.optimized and not args.baseline:
        parser.error("--optimized requires --baseline for comparison mode")
    
    if not args.output_csv and not args.output_json and not args.output:
        parser.error("At least one output format must be specified: --output-csv, --output-json, or --output")
    
    # Determine output files
    output_csv = args.output_csv
    output_json = args.output_json
    
    if args.output:
        if args.output.endswith('.csv'):
            output_csv = args.output
        elif args.output.endswith('.json'):
            output_json = args.output
        else:
            parser.error("--output file must have .csv or .json extension")
    
    print("=" * 70)
    print("oneDNN Reorder Operation Extraction and Analysis Tool")
    print("=" * 70)
    
    try:
        # Comparison mode
        if args.baseline and args.optimized:
            print(f"Mode: Comparison")
            print(f"Baseline trace: {args.baseline}")
            print(f"Optimized trace: {args.optimized}")
            print("-" * 70)
            
            # Parse baseline
            print("\nParsing baseline trace...")
            baseline_parser = TraceParser(args.baseline)
            baseline_ops = baseline_parser.parse()
            baseline_summary = baseline_parser.get_summary()
            print(f"  Found {len(baseline_ops)} reorder operations")
            
            # Parse optimized
            print("Parsing optimized trace...")
            optimized_parser = TraceParser(args.optimized)
            optimized_ops = optimized_parser.parse()
            optimized_summary = optimized_parser.get_summary()
            print(f"  Found {len(optimized_ops)} reorder operations")
            
            # Aggregate metrics
            print("\nAggregating metrics...")
            baseline_aggregator = MetricsAggregator(baseline_ops)
            baseline_metrics = baseline_aggregator.get_metrics()
            
            optimized_aggregator = MetricsAggregator(optimized_ops)
            optimized_metrics = optimized_aggregator.get_metrics()
            
            # Compare
            print("Computing comparison...")
            comparator = MetricsComparator(baseline_metrics, optimized_metrics)
            comparison = comparator.get_comparison()
            
            # Display summary
            print("\n" + "=" * 70)
            print("COMPARISON SUMMARY")
            print("=" * 70)
            total_comp = comparison.get('total', {})
            baseline_total = total_comp.get('baseline', {})
            optimized_total = total_comp.get('optimized', {})
            delta = total_comp.get('delta', {})
            pct = total_comp.get('percent_change', {})
            
            print(f"\nTotal Reorder Operations:")
            print(f"  Baseline:  {baseline_total.get('count', 0)} operations, "
                  f"{baseline_total.get('time_ms', 0.0):.3f} ms")
            print(f"  Optimized: {optimized_total.get('count', 0)} operations, "
                  f"{optimized_total.get('time_ms', 0.0):.3f} ms")
            print(f"  Delta:     {delta.get('count', 0):+d} operations ({pct.get('count', 0.0):+.2f}%), "
                  f"{delta.get('time_ms', 0.0):+.3f} ms ({pct.get('time_ms', 0.0):+.2f}%)")
            
            # Output results
            print("\n" + "=" * 70)
            if output_csv:
                OutputFormatter.to_csv(baseline_metrics, output_csv, comparison)
            
            if output_json:
                combined_summary = {
                    'baseline': baseline_summary,
                    'optimized': optimized_summary,
                }
                OutputFormatter.to_json(baseline_metrics, output_json, comparison, combined_summary)
        
        # Single trace mode
        else:
            print(f"Mode: Single trace analysis")
            print(f"Trace file: {args.trace}")
            print("-" * 70)
            
            # Parse trace
            print("\nParsing trace...")
            trace_parser = TraceParser(args.trace)
            reorder_ops = trace_parser.parse()
            summary = trace_parser.get_summary()
            print(f"  Found {len(reorder_ops)} reorder operations")
            
            # Aggregate metrics
            print("Aggregating metrics...")
            aggregator = MetricsAggregator(reorder_ops)
            metrics = aggregator.get_metrics()
            
            # Display summary
            print("\n" + "=" * 70)
            print("METRICS SUMMARY")
            print("=" * 70)
            total = metrics.get('total', {})
            print(f"\nTotal Reorder Operations: {total.get('count', 0)}")
            print(f"Total Time: {total.get('time_ms', 0.0):.3f} ms")
            
            if total.get('count', 0) > 0:
                avg_time = total.get('time_ms', 0.0) / total.get('count', 1)
                print(f"Average Time per Operation: {avg_time:.6f} ms")
            
            # Show top implementations
            by_impl = metrics.get('by_implementation', {})
            if by_impl:
                print(f"\nTop Implementation Types:")
                sorted_impl = sorted(by_impl.items(), 
                                    key=lambda x: x[1].get('count', 0), 
                                    reverse=True)[:5]
                for impl, data in sorted_impl:
                    print(f"  {impl}: {data.get('count', 0)} ops, {data.get('time_ms', 0.0):.3f} ms")
            
            # Show top dimensions
            by_dim = metrics.get('by_dimension', {})
            if by_dim:
                print(f"\nTop Dimensions:")
                sorted_dim = sorted(by_dim.items(), 
                                   key=lambda x: x[1].get('count', 0), 
                                   reverse=True)[:5]
                for dim, data in sorted_dim:
                    print(f"  {dim}: {data.get('count', 0)} ops, {data.get('time_ms', 0.0):.3f} ms")
            
            # Output results
            print("\n" + "=" * 70)
            if output_csv:
                OutputFormatter.to_csv(metrics, output_csv)
            
            if output_json:
                OutputFormatter.to_json(metrics, output_json, parser_summary=summary)
        
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

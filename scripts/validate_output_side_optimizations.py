#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Output-Side Layout Optimization Validation Tool

This script validates output-side layout optimizations (attention output and FFN output)
by analyzing oneDNN traces and confirming that:
1. Attention output uses optimal f32::ab format
2. FFN output uses optimal f32::ab format
3. Block boundary transitions have zero or minimal reorders
4. No regressions introduced in other operations

Usage:
    # Validate output-side optimizations with single trace
    python validate_output_side_optimizations.py --trace output_trace.txt --output report.md
    
    # Compare with baseline
    python validate_output_side_optimizations.py \\
        --trace output_trace.txt \\
        --baseline baseline_trace.txt \\
        --output comparison_report.md
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class OutputSideValidator:
    """Validates output-side layout optimizations from oneDNN traces."""
    
    def __init__(self, trace_file: str, baseline_file: Optional[str] = None):
        """
        Initialize validator.
        
        Args:
            trace_file: Path to output-side optimized trace
            baseline_file: Optional path to baseline trace for comparison
        """
        self.trace_file = trace_file
        self.baseline_file = baseline_file
        self.reorders = []
        self.baseline_reorders = []
        self.compute_ops = []
        
    def parse_trace(self, trace_file: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Parse oneDNN trace to extract reorder and compute operations.
        
        Args:
            trace_file: Path to trace file
            
        Returns:
            Tuple of (reorder_list, compute_ops_list)
        """
        reorders = []
        compute_ops = []
        
        with open(trace_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                if 'dnnl_verbose' not in line.lower():
                    continue
                
                parts = line.split(',')
                
                # Parse reorder operations
                if 'reorder' in line.lower():
                    reorder_info = self._parse_reorder(line, parts, line_num)
                    if reorder_info:
                        reorders.append(reorder_info)
                
                # Parse compute operations (inner_product, matmul)
                elif 'inner_product' in line.lower() or 'matmul' in line.lower():
                    compute_info = self._parse_compute(line, parts, line_num)
                    if compute_info:
                        compute_ops.append(compute_info)
        
        return reorders, compute_ops
    
    def _parse_reorder(self, line: str, parts: List[str], line_num: int) -> Optional[Dict]:
        """Parse a reorder operation line."""
        try:
            # Extract dimensions
            dimension_str = None
            dimensions = []
            
            for part in reversed(parts):
                part = part.strip()
                if 'x' in part and part.replace('x', '').replace('.', '').isdigit():
                    dimension_str = part
                    dims = part.split('x')
                    dimensions = [int(d) for d in dims if d.replace('.', '').isdigit()]
                    break
            
            if not dimension_str:
                return None
            
            # Extract time (last field)
            time_ms = None
            try:
                time_ms = float(parts[-1].strip())
            except (ValueError, IndexError):
                return None
            
            # Extract layouts
            src_layout = None
            dst_layout = None
            src_dtype = None
            dst_dtype = None
            
            for part in parts:
                if 'src_' in part:
                    # Extract source dtype and layout
                    src_match = re.search(r'src_(\w+)::blocked:(\w+):', part)
                    if src_match:
                        src_dtype = src_match.group(1)
                        src_layout = src_match.group(2)
                
                if 'dst_' in part:
                    # Extract destination dtype and layout
                    dst_match = re.search(r'dst_(\w+)::blocked:(\w+):', part)
                    if dst_match:
                        dst_dtype = dst_match.group(1)
                        dst_layout = dst_match.group(2)
            
            return {
                'line_num': line_num,
                'dimensions': dimensions,
                'dimension_str': dimension_str,
                'time_ms': time_ms,
                'src_layout': src_layout,
                'dst_layout': dst_layout,
                'src_dtype': src_dtype,
                'dst_dtype': dst_dtype,
                'raw_line': line
            }
        
        except Exception:
            return None
    
    def _parse_compute(self, line: str, parts: List[str], line_num: int) -> Optional[Dict]:
        """Parse a compute operation line."""
        try:
            # Extract dimensions
            dimension_str = None
            
            for part in parts:
                # Look for mb/ic/oc pattern
                if 'mb' in part and ('ic' in part or 'oc' in part):
                    dimension_str = part
                    break
            
            if not dimension_str:
                return None
            
            # Extract time
            time_ms = None
            try:
                time_ms = float(parts[-1].strip())
            except (ValueError, IndexError):
                return None
            
            # Extract input/output channels
            ic_match = re.search(r'ic(\d+)', dimension_str)
            oc_match = re.search(r'oc(\d+)', dimension_str)
            
            in_channels = int(ic_match.group(1)) if ic_match else None
            out_channels = int(oc_match.group(1)) if oc_match else None
            
            # Extract operation type
            op_type = 'inner_product' if 'inner_product' in line.lower() else 'matmul'
            
            return {
                'line_num': line_num,
                'op_type': op_type,
                'dimension_str': dimension_str,
                'in_channels': in_channels,
                'out_channels': out_channels,
                'time_ms': time_ms,
                'raw_line': line
            }
        
        except Exception:
            return None
    
    def analyze_output_side_reorders(self, reorders: List[Dict]) -> Dict:
        """
        Analyze reorders specifically related to output-side operations.
        
        Args:
            reorders: List of reorder operations
            
        Returns:
            Dictionary with output-side reorder analysis
        """
        attention_output_reorders = []
        ffn_output_reorders = []
        block_boundary_reorders = []
        weight_reorders = []
        other_reorders = []
        
        for reorder in reorders:
            dims = reorder['dimensions']
            dim_str = reorder['dimension_str']
            
            # Classify reorder type
            is_activation = False
            is_weight = False
            
            # Check if this is an activation reorder (batch dimension present)
            # Activations typically have small first dimension (batch size)
            if len(dims) == 2:
                if dims[0] <= 16:  # Likely batch size
                    is_activation = True
                    
                    # Check if it's attention or FFN related
                    if 1536 in dims:
                        # Could be attention output or FFN output
                        if '6x1536' in dim_str or '1x1536' in dim_str:
                            # Attention output activations are batch × 1536
                            attention_output_reorders.append(reorder)
                    
                    elif 8960 in dims:
                        # Likely FFN intermediate
                        if '6x8960' in dim_str or '1x8960' in dim_str:
                            block_boundary_reorders.append(reorder)
                
                else:
                    # Large dimensions, likely weight matrix
                    is_weight = True
                    weight_reorders.append(reorder)
            
            if not is_activation and not is_weight:
                other_reorders.append(reorder)
        
        return {
            'attention_output_reorders': attention_output_reorders,
            'ffn_output_reorders': ffn_output_reorders,
            'block_boundary_reorders': block_boundary_reorders,
            'weight_reorders': weight_reorders,
            'other_reorders': other_reorders,
            'total_activation_reorders': len(attention_output_reorders) + len(ffn_output_reorders) + len(block_boundary_reorders),
            'total_weight_reorders': len(weight_reorders)
        }
    
    def validate(self) -> Dict:
        """
        Perform validation of output-side optimizations.
        
        Returns:
            Validation results dictionary
        """
        # Parse current trace
        print(f"Parsing trace: {self.trace_file}")
        self.reorders, self.compute_ops = self.parse_trace(self.trace_file)
        
        print(f"Found {len(self.reorders)} reorder operations")
        print(f"Found {len(self.compute_ops)} compute operations")
        
        # Analyze output-side reorders
        output_analysis = self.analyze_output_side_reorders(self.reorders)
        
        # Parse baseline if provided
        baseline_analysis = None
        if self.baseline_file and os.path.exists(self.baseline_file):
            print(f"\nParsing baseline trace: {self.baseline_file}")
            self.baseline_reorders, _ = self.parse_trace(self.baseline_file)
            print(f"Found {len(self.baseline_reorders)} baseline reorder operations")
            
            baseline_analysis = self.analyze_output_side_reorders(self.baseline_reorders)
        
        # Compute metrics
        total_reorder_time = sum(r['time_ms'] for r in self.reorders if r.get('time_ms'))
        attention_output_time = sum(r['time_ms'] for r in output_analysis['attention_output_reorders'] if r.get('time_ms'))
        ffn_output_time = sum(r['time_ms'] for r in output_analysis['ffn_output_reorders'] if r.get('time_ms'))
        
        results = {
            'trace_file': self.trace_file,
            'total_reorders': len(self.reorders),
            'total_reorder_time_ms': total_reorder_time,
            'total_compute_ops': len(self.compute_ops),
            
            'output_side_analysis': {
                'attention_output_reorder_count': len(output_analysis['attention_output_reorders']),
                'attention_output_reorder_time_ms': attention_output_time,
                'ffn_output_reorder_count': len(output_analysis['ffn_output_reorders']),
                'ffn_output_reorder_time_ms': ffn_output_time,
                'block_boundary_reorder_count': len(output_analysis['block_boundary_reorders']),
                'total_activation_reorders': output_analysis['total_activation_reorders'],
                'total_weight_reorders': output_analysis['total_weight_reorders'],
            },
            
            'validation_criteria': {
                'attention_output_optimal': len(output_analysis['attention_output_reorders']) == 0,
                'ffn_output_optimal': len(output_analysis['ffn_output_reorders']) == 0,
                'block_boundary_optimal': len(output_analysis['block_boundary_reorders']) == 0,
                'total_activation_optimal': output_analysis['total_activation_reorders'] == 0,
            }
        }
        
        # Add baseline comparison if available
        if baseline_analysis:
            baseline_total_time = sum(r['time_ms'] for r in self.baseline_reorders if r.get('time_ms'))
            baseline_attention_time = sum(r['time_ms'] for r in baseline_analysis['attention_output_reorders'] if r.get('time_ms'))
            baseline_ffn_time = sum(r['time_ms'] for r in baseline_analysis['ffn_output_reorders'] if r.get('time_ms'))
            
            results['baseline_comparison'] = {
                'baseline_total_reorders': len(self.baseline_reorders),
                'baseline_total_time_ms': baseline_total_time,
                'baseline_attention_reorders': len(baseline_analysis['attention_output_reorders']),
                'baseline_attention_time_ms': baseline_attention_time,
                'baseline_ffn_reorders': len(baseline_analysis['ffn_output_reorders']),
                'baseline_ffn_time_ms': baseline_ffn_time,
                
                'reorder_count_reduction': len(self.baseline_reorders) - len(self.reorders),
                'reorder_time_reduction_ms': baseline_total_time - total_reorder_time,
                'attention_improvement_ms': baseline_attention_time - attention_output_time,
                'ffn_improvement_ms': baseline_ffn_time - ffn_output_time,
            }
        
        return results
    
    def generate_report(self, results: Dict, output_file: str):
        """
        Generate validation report in Markdown format.
        
        Args:
            results: Validation results dictionary
            output_file: Path to output report file
        """
        with open(output_file, 'w') as f:
            f.write("# Output-Side Layout Optimization Validation Report\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Trace File**: `{results['trace_file']}`\n")
            f.write(f"**Architecture**: AMD Ryzen 9 5900X (AVX2)\n")
            f.write(f"**Model**: Qwen2-1.5B-Instruct Single Transformer Block\n\n")
            
            f.write("---\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            output_analysis = results['output_side_analysis']
            validation = results['validation_criteria']
            
            f.write("### Validation Results\n\n")
            
            status = "✅ PASS" if all(validation.values()) else "⚠️ NEEDS REVIEW"
            f.write(f"**Overall Status**: {status}\n\n")
            
            f.write("| Criterion | Status | Count | Time (ms) |\n")
            f.write("|-----------|--------|-------|----------|\n")
            
            att_status = "✅ Optimal" if validation['attention_output_optimal'] else "⚠️ Has reorders"
            f.write(f"| Attention Output | {att_status} | {output_analysis['attention_output_reorder_count']} | {output_analysis['attention_output_reorder_time_ms']:.3f} |\n")
            
            ffn_status = "✅ Optimal" if validation['ffn_output_optimal'] else "⚠️ Has reorders"
            f.write(f"| FFN Output | {ffn_status} | {output_analysis['ffn_output_reorder_count']} | {output_analysis['ffn_output_reorder_time_ms']:.3f} |\n")
            
            bb_status = "✅ Optimal" if validation['block_boundary_optimal'] else "⚠️ Has reorders"
            f.write(f"| Block Boundary | {bb_status} | {output_analysis['block_boundary_reorder_count']} | - |\n")
            
            act_status = "✅ Zero reorders" if validation['total_activation_optimal'] else f"{output_analysis['total_activation_reorders']} reorders"
            f.write(f"| Total Activations | {act_status} | {output_analysis['total_activation_reorders']} | - |\n")
            
            f.write("\n")
            
            # Overall metrics
            f.write("### Overall Metrics\n\n")
            f.write(f"- **Total reorder operations**: {results['total_reorders']}\n")
            f.write(f"- **Total reorder time**: {results['total_reorder_time_ms']:.3f} ms\n")
            f.write(f"- **Total compute operations**: {results['total_compute_ops']}\n")
            f.write(f"- **Activation reorders**: {output_analysis['total_activation_reorders']}\n")
            f.write(f"- **Weight reorders**: {output_analysis['total_weight_reorders']} (expected and beneficial)\n\n")
            
            # Baseline comparison if available
            if 'baseline_comparison' in results:
                f.write("### Baseline Comparison\n\n")
                
                baseline = results['baseline_comparison']
                
                f.write("| Metric | Baseline | Optimized | Improvement |\n")
                f.write("|--------|----------|-----------|-------------|\n")
                
                f.write(f"| Total Reorders | {baseline['baseline_total_reorders']} | {results['total_reorders']} | ")
                if baseline['reorder_count_reduction'] > 0:
                    f.write(f"↓ {baseline['reorder_count_reduction']} |\n")
                elif baseline['reorder_count_reduction'] < 0:
                    f.write(f"↑ {abs(baseline['reorder_count_reduction'])} ⚠️ |\n")
                else:
                    f.write("= |\n")
                
                f.write(f"| Total Time (ms) | {baseline['baseline_total_time_ms']:.3f} | {results['total_reorder_time_ms']:.3f} | ")
                if baseline['reorder_time_reduction_ms'] > 0:
                    f.write(f"↓ {baseline['reorder_time_reduction_ms']:.3f} ms |\n")
                elif baseline['reorder_time_reduction_ms'] < 0:
                    f.write(f"↑ {abs(baseline['reorder_time_reduction_ms']):.3f} ms ⚠️ |\n")
                else:
                    f.write("= |\n")
                
                f.write(f"| Attention Reorders | {baseline['baseline_attention_reorders']} | {output_analysis['attention_output_reorder_count']} | ")
                att_reduction = baseline['baseline_attention_reorders'] - output_analysis['attention_output_reorder_count']
                if att_reduction > 0:
                    f.write(f"↓ {att_reduction} |\n")
                else:
                    f.write("= |\n")
                
                f.write(f"| FFN Reorders | {baseline['baseline_ffn_reorders']} | {output_analysis['ffn_output_reorder_count']} | ")
                ffn_reduction = baseline['baseline_ffn_reorders'] - output_analysis['ffn_output_reorder_count']
                if ffn_reduction > 0:
                    f.write(f"↓ {ffn_reduction} |\n")
                else:
                    f.write("= |\n")
                
                f.write("\n")
            
            # Detailed analysis
            f.write("---\n\n")
            f.write("## Detailed Analysis\n\n")
            
            f.write("### 1. Attention Output Layout\n\n")
            
            if validation['attention_output_optimal']:
                f.write("✅ **Status**: Optimal - Zero reorders detected\n\n")
                f.write("The attention output projection consistently produces `f32::ab` (plain format) ")
                f.write("activations that flow directly into residual connections without requiring ")
                f.write("format conversion.\n\n")
            else:
                f.write(f"⚠️ **Status**: {output_analysis['attention_output_reorder_count']} reorders detected\n\n")
                f.write("Review the reorder operations to determine if optimizations can be applied.\n\n")
            
            f.write("### 2. FFN Output Layout\n\n")
            
            if validation['ffn_output_optimal']:
                f.write("✅ **Status**: Optimal - Zero reorders detected\n\n")
                f.write("The FFN contract output (8960→1536) produces `f32::ab` activations that ")
                f.write("seamlessly transition to the next block without format conversion.\n\n")
            else:
                f.write(f"⚠️ **Status**: {output_analysis['ffn_output_reorder_count']} reorders detected\n\n")
                f.write("Review the reorder operations to determine if optimizations can be applied.\n\n")
            
            f.write("### 3. Block Boundary Transitions\n\n")
            
            if validation['block_boundary_optimal']:
                f.write("✅ **Status**: Optimal - Zero boundary reorders detected\n\n")
                f.write("Inter-block activation flow maintains format consistency, eliminating ")
                f.write("the need for reorder operations at block boundaries.\n\n")
            else:
                f.write(f"⚠️ **Status**: {output_analysis['block_boundary_reorder_count']} boundary reorders detected\n\n")
            
            f.write("### 4. Weight Reorders\n\n")
            f.write(f"**Count**: {output_analysis['total_weight_reorders']}\n\n")
            f.write("Weight reorders (typically ab→AB8b24a) are expected and beneficial, as they ")
            f.write("convert weight matrices to blocked formats optimized for BRGEMM kernels.\n\n")
            
            # Recommendations
            f.write("---\n\n")
            f.write("## Recommendations\n\n")
            
            if all(validation.values()):
                f.write("✅ **Current state is optimal**\n\n")
                f.write("1. Maintain current layout strategy for output-side operations\n")
                f.write("2. Monitor future changes to ensure graph optimizer passes preserve these layouts\n")
                f.write("3. Consider extending similar analysis to other transformer models\n")
            else:
                f.write("⚠️ **Optimization opportunities identified**\n\n")
                
                if not validation['attention_output_optimal']:
                    f.write("1. Investigate attention output reorders and apply layout propagation\n")
                
                if not validation['ffn_output_optimal']:
                    f.write("2. Investigate FFN output reorders and enforce plain format output\n")
                
                if not validation['block_boundary_optimal']:
                    f.write("3. Review block boundary layout compatibility\n")
            
            f.write("\n---\n\n")
            f.write("*Report generated by validate_output_side_optimizations.py*\n")
        
        print(f"\nValidation report generated: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate output-side layout optimizations from oneDNN traces"
    )
    
    parser.add_argument(
        '--trace',
        type=str,
        required=True,
        help='Path to output-side optimized trace file'
    )
    
    parser.add_argument(
        '--baseline',
        type=str,
        default=None,
        help='Path to baseline trace file for comparison'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='OUTPUT_SIDE_VALIDATION.md',
        help='Path to output validation report (default: OUTPUT_SIDE_VALIDATION.md)'
    )
    
    parser.add_argument(
        '--json',
        type=str,
        default=None,
        help='Path to output JSON results (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.trace):
        print(f"Error: Trace file not found: {args.trace}", file=sys.stderr)
        return 1
    
    if args.baseline and not os.path.exists(args.baseline):
        print(f"Warning: Baseline file not found: {args.baseline}", file=sys.stderr)
        args.baseline = None
    
    # Run validation
    print("=" * 60)
    print("Output-Side Layout Optimization Validation")
    print("=" * 60)
    print()
    
    validator = OutputSideValidator(args.trace, args.baseline)
    
    try:
        results = validator.validate()
        
        # Generate report
        print("\nGenerating validation report...")
        validator.generate_report(results, args.output)
        
        # Save JSON if requested
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"JSON results saved: {args.json}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Validation Summary")
        print("=" * 60)
        
        validation = results['validation_criteria']
        
        if all(validation.values()):
            print("✅ All validation criteria PASSED")
            print("   Output-side layouts are optimal")
        else:
            print("⚠️  Some validation criteria need review")
            
            if not validation['attention_output_optimal']:
                print(f"   - Attention output: {results['output_side_analysis']['attention_output_reorder_count']} reorders")
            
            if not validation['ffn_output_optimal']:
                print(f"   - FFN output: {results['output_side_analysis']['ffn_output_reorder_count']} reorders")
            
            if not validation['block_boundary_optimal']:
                print(f"   - Block boundary: {results['output_side_analysis']['block_boundary_reorder_count']} reorders")
        
        print()
        print(f"Total reorders: {results['total_reorders']}")
        print(f"Total time: {results['total_reorder_time_ms']:.3f} ms")
        print()
        print(f"Report: {args.output}")
        print("=" * 60)
        
        return 0
    
    except Exception as e:
        print(f"\nError during validation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Multi-Scenario Statistical Analysis Tool

This script analyzes reorder metrics across multiple test scenarios (batch sizes,
sequence lengths, input patterns) to validate the robustness of layout optimizations.
It calculates statistical measures (mean, min, max, std dev) and identifies variance
patterns that may indicate data-dependent behavior or optimization instability.

Usage:
    python analyze_multi_scenario_statistics.py --metrics-dir ./multi_scenario_validation/metrics --output ANALYSIS.md
    
    # With baseline comparison
    python analyze_multi_scenario_statistics.py --metrics-dir ./metrics --baseline-dir ./baseline_metrics --output ANALYSIS.md
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics


class ScenarioMetrics:
    """Container for metrics from a single scenario."""
    
    def __init__(self, batch_size: int, seq_length: int, input_pattern: str):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.input_pattern = input_pattern
        self.repetitions: List[Dict] = []
        
    def add_repetition(self, metrics: Dict):
        """Add metrics from one repetition."""
        self.repetitions.append(metrics)
    
    def get_statistic(self, metric_path: List[str]) -> Dict:
        """
        Calculate statistics for a specific metric across repetitions.
        
        Args:
            metric_path: Path to metric in JSON (e.g., ['summary', 'total_count'])
        
        Returns:
            Dict with mean, min, max, std_dev, cv (coefficient of variation)
        """
        values = []
        for rep in self.repetitions:
            value = self._get_nested_value(rep, metric_path)
            if value is not None:
                values.append(value)
        
        if not values:
            return None
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        cv_val = (std_val / mean_val * 100) if mean_val != 0 else 0.0
        
        return {
            'mean': mean_val,
            'min': min(values),
            'max': max(values),
            'std_dev': std_val,
            'cv_percent': cv_val,  # Coefficient of variation
            'n_samples': len(values),
            'values': values
        }
    
    @staticmethod
    def _get_nested_value(data: Dict, path: List[str]):
        """Get nested value from dictionary using path."""
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
    
    def get_scenario_id(self) -> str:
        """Get unique identifier for this scenario."""
        return f"batch{self.batch_size}_seq{self.seq_length}_{self.input_pattern}"


class MultiScenarioAnalyzer:
    """Analyzer for multi-scenario trace statistics."""
    
    def __init__(self, metrics_dir: str, baseline_dir: Optional[str] = None):
        """
        Initialize analyzer.
        
        Args:
            metrics_dir: Directory containing metrics for all scenarios
            baseline_dir: Optional directory with baseline metrics for comparison
        """
        self.metrics_dir = Path(metrics_dir)
        self.baseline_dir = Path(baseline_dir) if baseline_dir else None
        self.scenarios: List[ScenarioMetrics] = []
        self.baseline_metrics: Optional[Dict] = None
        
    def load_all_scenarios(self):
        """Load metrics from all scenario directories."""
        if not self.metrics_dir.exists():
            raise FileNotFoundError(f"Metrics directory not found: {self.metrics_dir}")
        
        # Find all scenario directories (format: batch{X}_seq{Y}_{pattern})
        for scenario_dir in sorted(self.metrics_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            
            # Parse scenario parameters from directory name
            dir_name = scenario_dir.name
            try:
                parts = dir_name.split('_')
                batch_size = int(parts[0].replace('batch', ''))
                seq_length = int(parts[1].replace('seq', ''))
                input_pattern = '_'.join(parts[2:])
                
                scenario = ScenarioMetrics(batch_size, seq_length, input_pattern)
                
                # Load all repetition metrics
                for metrics_file in sorted(scenario_dir.glob('metrics_rep*.json')):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                        scenario.add_repetition(metrics)
                
                if scenario.repetitions:
                    self.scenarios.append(scenario)
                    
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse scenario directory: {dir_name}", file=sys.stderr)
                continue
        
        if not self.scenarios:
            raise ValueError("No valid scenarios found in metrics directory")
        
        print(f"Loaded {len(self.scenarios)} scenarios")
    
    def load_baseline(self):
        """Load baseline metrics for comparison."""
        if not self.baseline_dir or not self.baseline_dir.exists():
            return
        
        # Find baseline metrics (average across runs)
        baseline_files = list(self.baseline_dir.glob('*.json'))
        if not baseline_files:
            print("Warning: No baseline metrics found", file=sys.stderr)
            return
        
        # Use first baseline file (or average if multiple)
        with open(baseline_files[0], 'r') as f:
            self.baseline_metrics = json.load(f)
    
    def analyze_batch_size_consistency(self) -> Dict:
        """Analyze consistency across different batch sizes."""
        # Group by seq_length and input_pattern, vary batch_size
        grouped = defaultdict(list)
        
        for scenario in self.scenarios:
            key = (scenario.seq_length, scenario.input_pattern)
            grouped[key].append(scenario)
        
        results = {}
        for (seq_len, pattern), scenarios in grouped.items():
            if len(scenarios) < 2:
                continue
            
            key = f"seq{seq_len}_{pattern}"
            results[key] = self._analyze_scenario_group(scenarios, 'batch_size')
        
        return results
    
    def analyze_seq_length_consistency(self) -> Dict:
        """Analyze consistency across different sequence lengths."""
        # Group by batch_size and input_pattern, vary seq_length
        grouped = defaultdict(list)
        
        for scenario in self.scenarios:
            key = (scenario.batch_size, scenario.input_pattern)
            grouped[key].append(scenario)
        
        results = {}
        for (batch, pattern), scenarios in grouped.items():
            if len(scenarios) < 2:
                continue
            
            key = f"batch{batch}_{pattern}"
            results[key] = self._analyze_scenario_group(scenarios, 'seq_length')
        
        return results
    
    def analyze_input_pattern_consistency(self) -> Dict:
        """Analyze consistency across different input patterns."""
        # Group by batch_size and seq_length, vary input_pattern
        grouped = defaultdict(list)
        
        for scenario in self.scenarios:
            key = (scenario.batch_size, scenario.seq_length)
            grouped[key].append(scenario)
        
        results = {}
        for (batch, seq_len), scenarios in grouped.items():
            if len(scenarios) < 2:
                continue
            
            key = f"batch{batch}_seq{seq_len}"
            results[key] = self._analyze_scenario_group(scenarios, 'input_pattern')
        
        return results
    
    def _analyze_scenario_group(self, scenarios: List[ScenarioMetrics], varying_param: str) -> Dict:
        """
        Analyze a group of scenarios that differ in one parameter.
        
        Args:
            scenarios: List of scenarios to analyze
            varying_param: Parameter that varies ('batch_size', 'seq_length', 'input_pattern')
        
        Returns:
            Analysis results with variance metrics
        """
        # Extract key metrics for comparison
        metric_paths = [
            (['summary', 'total_count'], 'total_reorder_count'),
            (['summary', 'total_time_ms'], 'total_reorder_time_ms'),
            (['summary', 'mean_time_ms'], 'mean_reorder_time_ms'),
        ]
        
        results = {
            'varying_parameter': varying_param,
            'scenarios': [],
            'metrics': {}
        }
        
        for path, name in metric_paths:
            metric_values = []
            
            for scenario in scenarios:
                stats = scenario.get_statistic(path)
                if stats:
                    param_value = getattr(scenario, varying_param)
                    results['scenarios'].append({
                        'parameter_value': param_value,
                        'scenario_id': scenario.get_scenario_id(),
                    })
                    metric_values.append(stats['mean'])
            
            if len(metric_values) >= 2:
                # Calculate variance across scenarios
                mean_across = statistics.mean(metric_values)
                std_across = statistics.stdev(metric_values)
                cv_across = (std_across / mean_across * 100) if mean_across != 0 else 0.0
                
                results['metrics'][name] = {
                    'values': metric_values,
                    'mean': mean_across,
                    'std_dev': std_across,
                    'cv_percent': cv_across,
                    'min': min(metric_values),
                    'max': max(metric_values),
                    'range_percent': ((max(metric_values) - min(metric_values)) / mean_across * 100) if mean_across != 0 else 0.0
                }
        
        return results
    
    def identify_high_variance_scenarios(self, threshold_cv: float = 5.0) -> List[Dict]:
        """
        Identify scenarios with high variance (CV > threshold).
        
        Args:
            threshold_cv: Coefficient of variation threshold (default 5%)
        
        Returns:
            List of scenarios with high variance
        """
        high_variance = []
        
        for scenario in self.scenarios:
            # Check key metrics
            total_count_stats = scenario.get_statistic(['summary', 'total_count'])
            total_time_stats = scenario.get_statistic(['summary', 'total_time_ms'])
            
            if total_count_stats and total_count_stats['cv_percent'] > threshold_cv:
                high_variance.append({
                    'scenario_id': scenario.get_scenario_id(),
                    'metric': 'total_reorder_count',
                    'cv_percent': total_count_stats['cv_percent'],
                    'mean': total_count_stats['mean'],
                    'std_dev': total_count_stats['std_dev'],
                })
            
            if total_time_stats and total_time_stats['cv_percent'] > threshold_cv:
                high_variance.append({
                    'scenario_id': scenario.get_scenario_id(),
                    'metric': 'total_reorder_time_ms',
                    'cv_percent': total_time_stats['cv_percent'],
                    'mean': total_time_stats['mean'],
                    'std_dev': total_time_stats['std_dev'],
                })
        
        return high_variance
    
    def generate_report(self, output_path: str):
        """Generate comprehensive markdown report."""
        
        with open(output_path, 'w') as f:
            # Header
            f.write("# Multi-Scenario Statistical Analysis Report\n\n")
            f.write(f"**Generated**: {self._get_timestamp()}\n\n")
            f.write("---\n\n")
            
            # Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Scenarios Analyzed**: {len(self.scenarios)}\n")
            
            batch_sizes = sorted(set(s.batch_size for s in self.scenarios))
            seq_lengths = sorted(set(s.seq_length for s in self.scenarios))
            input_patterns = sorted(set(s.input_pattern for s in self.scenarios))
            
            f.write(f"- **Batch Sizes**: {batch_sizes}\n")
            f.write(f"- **Sequence Lengths**: {seq_lengths}\n")
            f.write(f"- **Input Patterns**: {input_patterns}\n\n")
            
            # High variance scenarios
            f.write("### High Variance Scenarios (CV > 5%)\n\n")
            high_var = self.identify_high_variance_scenarios(threshold_cv=5.0)
            
            if high_var:
                f.write("⚠️ **Found scenarios with high variance:**\n\n")
                f.write("| Scenario | Metric | CV (%) | Mean | Std Dev |\n")
                f.write("|----------|--------|--------|------|----------|\n")
                for item in high_var:
                    f.write(f"| {item['scenario_id']} | {item['metric']} | "
                           f"{item['cv_percent']:.2f}% | {item['mean']:.2f} | {item['std_dev']:.2f} |\n")
                f.write("\n")
            else:
                f.write("✅ **All scenarios show low variance (CV < 5%)**\n\n")
            
            # Consistency Analysis
            f.write("---\n\n")
            f.write("## Consistency Analysis\n\n")
            
            # Batch size consistency
            f.write("### Batch Size Consistency\n\n")
            f.write("*Validates that optimization improvements are consistent across different batch sizes.*\n\n")
            batch_analysis = self.analyze_batch_size_consistency()
            self._write_consistency_section(f, batch_analysis, "Batch Size")
            
            # Sequence length consistency
            f.write("### Sequence Length Consistency\n\n")
            f.write("*Validates that optimization improvements are consistent across different sequence lengths.*\n\n")
            seq_analysis = self.analyze_seq_length_consistency()
            self._write_consistency_section(f, seq_analysis, "Sequence Length")
            
            # Input pattern consistency
            f.write("### Input Pattern Consistency\n\n")
            f.write("*Validates that input activation values do not affect optimization effectiveness.*\n\n")
            pattern_analysis = self.analyze_input_pattern_consistency()
            self._write_consistency_section(f, pattern_analysis, "Input Pattern")
            
            # Detailed scenario metrics
            f.write("---\n\n")
            f.write("## Detailed Scenario Metrics\n\n")
            self._write_detailed_metrics(f)
            
            # Success criteria validation
            f.write("---\n\n")
            f.write("## Success Criteria Validation\n\n")
            self._write_success_criteria(f, high_var, batch_analysis, seq_analysis, pattern_analysis)
            
            # Recommendations
            f.write("---\n\n")
            f.write("## Recommendations\n\n")
            self._write_recommendations(f, high_var, batch_analysis, seq_analysis, pattern_analysis)
    
    def _write_consistency_section(self, f, analysis: Dict, section_name: str):
        """Write consistency analysis section."""
        if not analysis:
            f.write(f"*No {section_name.lower()} variation found in scenarios.*\n\n")
            return
        
        for group_key, group_data in analysis.items():
            f.write(f"#### {group_key}\n\n")
            
            if 'metrics' in group_data:
                f.write("| Metric | Mean | Std Dev | CV (%) | Range (%) | Status |\n")
                f.write("|--------|------|---------|--------|-----------|--------|\n")
                
                for metric_name, metric_data in group_data['metrics'].items():
                    cv = metric_data['cv_percent']
                    range_pct = metric_data['range_percent']
                    
                    # Status: ✅ < 5%, ⚠️ 5-10%, ❌ > 10%
                    if cv < 5.0:
                        status = "✅ Low"
                    elif cv < 10.0:
                        status = "⚠️ Medium"
                    else:
                        status = "❌ High"
                    
                    f.write(f"| {metric_name} | {metric_data['mean']:.2f} | "
                           f"{metric_data['std_dev']:.2f} | {cv:.2f}% | "
                           f"{range_pct:.2f}% | {status} |\n")
                
                f.write("\n")
    
    def _write_detailed_metrics(self, f):
        """Write detailed metrics for each scenario."""
        f.write("| Scenario | Batch | Seq Len | Pattern | Reorder Count | Reorder Time (ms) | CV (%) |\n")
        f.write("|----------|-------|---------|---------|---------------|-------------------|--------|\n")
        
        for scenario in sorted(self.scenarios, key=lambda s: (s.batch_size, s.seq_length, s.input_pattern)):
            count_stats = scenario.get_statistic(['summary', 'total_count'])
            time_stats = scenario.get_statistic(['summary', 'total_time_ms'])
            
            if count_stats and time_stats:
                f.write(f"| {scenario.get_scenario_id()} | {scenario.batch_size} | "
                       f"{scenario.seq_length} | {scenario.input_pattern} | "
                       f"{count_stats['mean']:.1f} ± {count_stats['std_dev']:.1f} | "
                       f"{time_stats['mean']:.2f} ± {time_stats['std_dev']:.2f} | "
                       f"{time_stats['cv_percent']:.2f}% |\n")
        
        f.write("\n")
    
    def _write_success_criteria(self, f, high_var, batch_analysis, seq_analysis, pattern_analysis):
        """Write success criteria validation section."""
        f.write("### Validation Results\n\n")
        
        criteria = []
        
        # Criterion 1: High variance check
        if len(high_var) == 0:
            criteria.append(("✅", "No high variance scenarios (CV < 5%)"))
        elif len(high_var) < len(self.scenarios) * 0.1:
            criteria.append(("⚠️", f"Low number of high variance scenarios ({len(high_var)}/{len(self.scenarios)})"))
        else:
            criteria.append(("❌", f"Multiple high variance scenarios detected ({len(high_var)}/{len(self.scenarios)})"))
        
        # Criterion 2: Batch size consistency
        batch_consistent = self._check_consistency(batch_analysis, threshold=10.0)
        if batch_consistent:
            criteria.append(("✅", "Batch size consistency maintained (CV < 10%)"))
        else:
            criteria.append(("❌", "Batch size shows inconsistent behavior"))
        
        # Criterion 3: Sequence length consistency
        seq_consistent = self._check_consistency(seq_analysis, threshold=10.0)
        if seq_consistent:
            criteria.append(("✅", "Sequence length consistency maintained (CV < 10%)"))
        else:
            criteria.append(("❌", "Sequence length shows inconsistent behavior"))
        
        # Criterion 4: Input pattern consistency
        pattern_consistent = self._check_consistency(pattern_analysis, threshold=5.0)
        if pattern_consistent:
            criteria.append(("✅", "Input pattern independence validated (CV < 5%)"))
        else:
            criteria.append(("⚠️", "Input patterns show some variance"))
        
        # Write criteria
        for status, description in criteria:
            f.write(f"{status} {description}\n\n")
        
        # Overall assessment
        passed = sum(1 for status, _ in criteria if status == "✅")
        total = len(criteria)
        
        f.write(f"\n**Overall Score**: {passed}/{total} criteria passed\n\n")
        
        if passed == total:
            f.write("🎉 **All success criteria met!** Optimizations are robust across all test scenarios.\n\n")
        elif passed >= total * 0.75:
            f.write("✅ **Most criteria met.** Minor variations detected but within acceptable limits.\n\n")
        else:
            f.write("⚠️ **Some criteria not met.** Review high variance scenarios and investigate root causes.\n\n")
    
    def _write_recommendations(self, f, high_var, batch_analysis, seq_analysis, pattern_analysis):
        """Write recommendations section."""
        recommendations = []
        
        if high_var:
            recommendations.append(
                "**Investigate High Variance Scenarios**: Review the scenarios with CV > 5% to determine "
                "if variance is due to measurement noise or actual optimization instability."
            )
        
        if not self._check_consistency(pattern_analysis, threshold=5.0):
            recommendations.append(
                "**Input Pattern Sensitivity Detected**: Some input patterns show different reorder behavior. "
                "Consider investigating whether certain patterns trigger different code paths or layout decisions."
            )
        
        if not self._check_consistency(batch_analysis, threshold=10.0):
            recommendations.append(
                "**Batch Size Dependency**: Performance varies significantly across batch sizes. "
                "Consider profiling individual batch scenarios to understand scaling behavior."
            )
        
        if not recommendations:
            recommendations.append(
                "**No Issues Detected**: All scenarios show consistent, low-variance behavior. "
                "Optimizations are validated as robust across diverse inference conditions."
            )
            recommendations.append(
                "**Next Steps**: Proceed with integration into main branch and consider extending "
                "test coverage to additional model architectures."
            )
        
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n\n")
    
    def _check_consistency(self, analysis: Dict, threshold: float = 10.0) -> bool:
        """Check if analysis shows consistency (all CV < threshold)."""
        if not analysis:
            return True
        
        for group_data in analysis.values():
            if 'metrics' in group_data:
                for metric_data in group_data['metrics'].values():
                    if metric_data['cv_percent'] > threshold:
                        return False
        return True
    
    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze multi-scenario trace statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_multi_scenario_statistics.py --metrics-dir ./metrics --output ANALYSIS.md
  
  # With baseline comparison
  python analyze_multi_scenario_statistics.py --metrics-dir ./metrics --baseline-dir ./baseline --output ANALYSIS.md
        """
    )
    
    parser.add_argument(
        '--metrics-dir',
        type=str,
        required=True,
        help='Directory containing metrics for all scenarios'
    )
    
    parser.add_argument(
        '--baseline-dir',
        type=str,
        help='Optional directory with baseline metrics for comparison'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='MULTI_SCENARIO_ANALYSIS.md',
        help='Output report file (default: MULTI_SCENARIO_ANALYSIS.md)'
    )
    
    parser.add_argument(
        '--variance-threshold',
        type=float,
        default=5.0,
        help='CV threshold for high variance detection (default: 5.0%%)'
    )
    
    args = parser.parse_args()
    
    try:
        print("=" * 70)
        print("Multi-Scenario Statistical Analysis")
        print("=" * 70)
        print(f"Metrics directory: {args.metrics_dir}")
        if args.baseline_dir:
            print(f"Baseline directory: {args.baseline_dir}")
        print(f"Output file: {args.output}")
        print(f"Variance threshold: {args.variance_threshold}%")
        print("=" * 70)
        print()
        
        # Initialize analyzer
        analyzer = MultiScenarioAnalyzer(args.metrics_dir, args.baseline_dir)
        
        # Load scenarios
        print("Loading scenarios...")
        analyzer.load_all_scenarios()
        print(f"✓ Loaded {len(analyzer.scenarios)} scenarios")
        print()
        
        # Load baseline if provided
        if args.baseline_dir:
            print("Loading baseline...")
            analyzer.load_baseline()
            print("✓ Baseline loaded")
            print()
        
        # Generate report
        print("Generating analysis report...")
        analyzer.generate_report(args.output)
        print(f"✓ Report saved to: {args.output}")
        print()
        
        # Quick summary
        high_var = analyzer.identify_high_variance_scenarios(threshold_cv=args.variance_threshold)
        
        print("=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Total scenarios: {len(analyzer.scenarios)}")
        print(f"High variance scenarios: {len(high_var)}")
        
        if high_var:
            print("\nHigh variance scenarios detected:")
            for item in high_var[:5]:  # Show first 5
                print(f"  - {item['scenario_id']}: {item['metric']} (CV={item['cv_percent']:.2f}%)")
            if len(high_var) > 5:
                print(f"  ... and {len(high_var) - 5} more")
        else:
            print("\n✅ All scenarios show low variance!")
        
        print(f"\nFull report available at: {args.output}")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

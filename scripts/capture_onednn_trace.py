#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
oneDNN Verbose Trace Capture Harness

This script captures oneDNN verbose traces from the extracted transformer block
model for baseline and optimization comparison. It configures DNNL_VERBOSE logging,
runs inference with consistent input parameters, and saves the complete trace output.

Usage:
    python capture_onednn_trace.py --model-path ./extracted_block/transformer_block.xml --output-dir ./traces --tag baseline
    
    # For optimized model comparison
    python capture_onednn_trace.py --model-path ./optimized_block/transformer_block.xml --output-dir ./traces --tag optimized
"""

import argparse
import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path
import numpy as np


def create_consistent_inputs(batch_size=1, seq_length=16, hidden_size=1536, seed=42, input_pattern='random'):
    """
    Create consistent test inputs with fixed random seed for reproducibility.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        hidden_size: Hidden dimension size
        seed: Random seed for reproducibility
        input_pattern: Input pattern type ('random', 'ones', 'small_values', 'zeros')
    
    Returns:
        Dictionary of input tensors
    """
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Create hidden states based on pattern
    if input_pattern == 'random':
        # Standard normal distribution
        hidden_states = np.random.randn(batch_size, seq_length, hidden_size).astype(np.float32)
    elif input_pattern == 'ones':
        # All ones
        hidden_states = np.ones((batch_size, seq_length, hidden_size), dtype=np.float32)
    elif input_pattern == 'small_values':
        # Small values centered around 0.1 with small variance
        hidden_states = (np.random.randn(batch_size, seq_length, hidden_size) * 0.01 + 0.1).astype(np.float32)
    elif input_pattern == 'zeros':
        # All zeros
        hidden_states = np.zeros((batch_size, seq_length, hidden_size), dtype=np.float32)
    else:
        raise ValueError(f"Unknown input pattern: {input_pattern}. Use 'random', 'ones', 'small_values', or 'zeros'.")
    
    # Create attention mask (all ones = attend to all tokens)
    attention_mask = np.ones((batch_size, seq_length), dtype=np.int64)
    
    # Create position IDs
    position_ids = np.arange(seq_length, dtype=np.int64).reshape(1, -1)
    if batch_size > 1:
        position_ids = np.repeat(position_ids, batch_size, axis=0)
    
    return {
        'hidden_states': hidden_states,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
    }


def generate_trace_filename(output_dir, tag, timestamp=True):
    """
    Generate descriptive trace filename.
    
    Args:
        output_dir: Output directory path
        tag: Tag for the trace (e.g., 'baseline', 'optimized')
        timestamp: Whether to include timestamp
    
    Returns:
        Full path to trace file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if timestamp:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"onednn_trace_{tag}_{time_str}.txt"
    else:
        filename = f"onednn_trace_{tag}.txt"
    
    return os.path.join(output_dir, filename)


def capture_trace_subprocess(model_path, device, batch_size, seq_length, 
                             num_iterations, verbose_level, seed, input_pattern='random'):
    """
    Capture oneDNN trace by running inference in a subprocess with DNNL_VERBOSE set.
    
    This approach ensures that the DNNL_VERBOSE environment variable is properly
    set before OpenVINO is loaded.
    
    Args:
        model_path: Path to OpenVINO model
        device: Device to run on
        batch_size: Batch size
        seq_length: Sequence length
        num_iterations: Number of inference iterations
        verbose_level: DNNL_VERBOSE level (1 or 2)
        seed: Random seed
        input_pattern: Input pattern type ('random', 'ones', 'small_values', 'zeros')
    
    Returns:
        Tuple of (stdout, stderr) containing trace output
    """
    # Create a subprocess script that runs inference with DNNL_VERBOSE
    inference_script = f"""
import os
os.environ['DNNL_VERBOSE'] = '{verbose_level}'

import openvino as ov
import numpy as np

# Set seed for reproducibility
np.random.seed({seed})

# Load model
core = ov.Core()
model = core.read_model('{model_path}')
compiled_model = core.compile_model(model, '{device}')

# Create inputs based on pattern
if '{input_pattern}' == 'random':
    hidden_states = np.random.randn({batch_size}, {seq_length}, 1536).astype(np.float32)
elif '{input_pattern}' == 'ones':
    hidden_states = np.ones(({batch_size}, {seq_length}, 1536), dtype=np.float32)
elif '{input_pattern}' == 'small_values':
    hidden_states = (np.random.randn({batch_size}, {seq_length}, 1536) * 0.01 + 0.1).astype(np.float32)
elif '{input_pattern}' == 'zeros':
    hidden_states = np.zeros(({batch_size}, {seq_length}, 1536), dtype=np.float32)
else:
    hidden_states = np.random.randn({batch_size}, {seq_length}, 1536).astype(np.float32)

attention_mask = np.ones(({batch_size}, {seq_length}), dtype=np.int64)
position_ids = np.arange({seq_length}, dtype=np.int64).reshape(1, -1)
if {batch_size} > 1:
    position_ids = np.repeat(position_ids, {batch_size}, axis=0)

inputs = {{
    'hidden_states': hidden_states,
    'attention_mask': attention_mask,
    'position_ids': position_ids,
}}

# Run inference
for i in range({num_iterations}):
    result = compiled_model(inputs)
    
print("Inference completed successfully")
"""
    
    # Run the subprocess
    result = subprocess.run(
        [sys.executable, '-c', inference_script],
        capture_output=True,
        text=True,
        env={**os.environ, 'DNNL_VERBOSE': str(verbose_level)}
    )
    
    return result.stdout, result.stderr


def validate_trace_output(trace_content, expected_dimensions=(1536, 8960)):
    """
    Validate that trace output contains expected operations and dimensions.
    
    Args:
        trace_content: Trace file content as string
        expected_dimensions: Tuple of expected dimensions to look for
    
    Returns:
        Dictionary with validation results
    """
    validation = {
        'has_content': len(trace_content.strip()) > 0,
        'line_count': len(trace_content.strip().split('\n')),
        'has_reorder': 'reorder' in trace_content.lower(),
        'has_expected_dims': False,
        'reorder_count': trace_content.lower().count('reorder'),
        'convolution_count': trace_content.lower().count('convolution'),
        'matmul_count': trace_content.lower().count('matmul'),
        'inner_product_count': trace_content.lower().count('inner_product'),
    }
    
    # Check for expected dimensions
    for dim in expected_dimensions:
        if str(dim) in trace_content:
            validation['has_expected_dims'] = True
            break
    
    return validation


def save_trace_metadata(metadata_path, model_path, device, batch_size, seq_length,
                        num_iterations, verbose_level, seed, input_pattern, validation_results,
                        trace_file, execution_time):
    """
    Save trace capture metadata for reproducibility.
    
    Args:
        metadata_path: Path to save metadata JSON file
        model_path: Model path used
        device: Device used
        batch_size: Batch size used
        seq_length: Sequence length used
        num_iterations: Number of iterations
        verbose_level: DNNL_VERBOSE level
        seed: Random seed
        input_pattern: Input pattern type
        validation_results: Dictionary with validation results
        trace_file: Path to trace file
        execution_time: Execution time in seconds
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_path': model_path,
        'device': device,
        'parameters': {
            'batch_size': batch_size,
            'seq_length': seq_length,
            'hidden_size': 1536,
            'num_iterations': num_iterations,
            'random_seed': seed,
            'input_pattern': input_pattern,
        },
        'environment': {
            'dnnl_verbose_level': verbose_level,
        },
        'trace_file': trace_file,
        'validation': validation_results,
        'execution_time_seconds': execution_time,
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Capture oneDNN verbose traces from transformer block model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture baseline trace
  python capture_onednn_trace.py --model-path ./extracted_block/transformer_block.xml --output-dir ./traces --tag baseline
  
  # Capture optimized trace for comparison
  python capture_onednn_trace.py --model-path ./optimized_block/transformer_block.xml --output-dir ./traces --tag optimized
  
  # Use different parameters
  python capture_onednn_trace.py --model-path ./model.xml --output-dir ./traces --tag test --batch-size 2 --seq-length 32 --iterations 50
  
  # More verbose oneDNN output (level 2)
  python capture_onednn_trace.py --model-path ./model.xml --output-dir ./traces --tag baseline --verbose-level 2
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to extracted transformer block model (.xml file)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./traces',
        help='Output directory for trace files (default: ./traces)'
    )
    
    parser.add_argument(
        '--tag',
        type=str,
        default='baseline',
        help='Tag for trace file naming (e.g., baseline, optimized, test) (default: baseline)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='CPU',
        help='Device to run inference on (default: CPU)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for inference (default: 1)'
    )
    
    parser.add_argument(
        '--seq-length',
        type=int,
        default=16,
        help='Sequence length for inference (default: 16)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of inference iterations (default: 10)'
    )
    
    parser.add_argument(
        '--verbose-level',
        type=int,
        choices=[1, 2],
        default=1,
        help='DNNL_VERBOSE level: 1=basic, 2=detailed (default: 1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--input-pattern',
        type=str,
        choices=['random', 'ones', 'small_values', 'zeros'],
        default='random',
        help='Input pattern type: random (default), ones, small_values, or zeros'
    )
    
    parser.add_argument(
        '--no-timestamp',
        action='store_true',
        help='Disable timestamp in trace filename'
    )
    
    args = parser.parse_args()
    
    # Validate model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}", file=sys.stderr)
        return 1
    
    print("=" * 70)
    print("oneDNN Verbose Trace Capture Harness")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Tag: {args.tag}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parameters:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Random seed: {args.seed}")
    print(f"  Input pattern: {args.input_pattern}")
    print(f"  DNNL_VERBOSE level: {args.verbose_level}")
    print("=" * 70)
    
    try:
        # Generate trace filename
        trace_file = generate_trace_filename(
            args.output_dir, 
            args.tag, 
            timestamp=not args.no_timestamp
        )
        metadata_file = trace_file.replace('.txt', '_metadata.json')
        
        print(f"\nCapturing oneDNN trace...")
        print(f"This may take a moment...\n")
        
        # Capture trace
        start_time = time.time()
        stdout, stderr = capture_trace_subprocess(
            args.model_path,
            args.device,
            args.batch_size,
            args.seq_length,
            args.iterations,
            args.verbose_level,
            args.seed,
            args.input_pattern
        )
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Check for errors
        if "Error" in stderr or "error" in stderr.lower():
            if "dnnl" not in stderr.lower():  # Ignore DNNL verbose output
                print(f"Warning: Potential errors detected in output", file=sys.stderr)
        
        # Save trace to file
        with open(trace_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("oneDNN Verbose Trace Output\n")
            f.write("=" * 70 + "\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Device: {args.device}\n")
            f.write(f"Tag: {args.tag}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Sequence length: {args.seq_length}\n")
            f.write(f"Iterations: {args.iterations}\n")
            f.write(f"Random seed: {args.seed}\n")
            f.write(f"Input pattern: {args.input_pattern}\n")
            f.write(f"DNNL_VERBOSE level: {args.verbose_level}\n")
            f.write("=" * 70 + "\n\n")
            f.write("STDOUT:\n")
            f.write("-" * 70 + "\n")
            f.write(stdout)
            f.write("\n" + "-" * 70 + "\n\n")
            f.write("STDERR (oneDNN verbose output):\n")
            f.write("-" * 70 + "\n")
            f.write(stderr)
            f.write("\n" + "-" * 70 + "\n")
        
        print(f"✓ Trace saved to: {trace_file}")
        
        # Validate trace output
        print(f"\nValidating trace output...")
        validation_results = validate_trace_output(stderr)
        
        print(f"\nValidation Results:")
        print(f"  Lines captured: {validation_results['line_count']}")
        print(f"  Has content: {'✓' if validation_results['has_content'] else '✗'}")
        print(f"  Contains reorder ops: {'✓' if validation_results['has_reorder'] else '✗'}")
        print(f"  Expected dimensions found: {'✓' if validation_results['has_expected_dims'] else '✗'}")
        print(f"  Reorder operations: {validation_results['reorder_count']}")
        print(f"  Convolution operations: {validation_results['convolution_count']}")
        print(f"  MatMul operations: {validation_results['matmul_count']}")
        print(f"  InnerProduct operations: {validation_results['inner_product_count']}")
        
        # Save metadata
        save_trace_metadata(
            metadata_file,
            args.model_path,
            args.device,
            args.batch_size,
            args.seq_length,
            args.iterations,
            args.verbose_level,
            args.seed,
            args.input_pattern,
            validation_results,
            trace_file,
            execution_time
        )
        
        print(f"\n✓ Metadata saved to: {metadata_file}")
        print(f"\nExecution time: {execution_time:.2f} seconds")
        
        # Summary
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"Trace file: {trace_file}")
        print(f"Metadata file: {metadata_file}")
        print(f"Lines captured: {validation_results['line_count']}")
        
        if not validation_results['has_content']:
            print("\nWARNING: Trace appears to be empty. Check DNNL_VERBOSE configuration.")
            return 1
        
        if not validation_results['has_reorder']:
            print("\nWARNING: No reorder operations found in trace.")
        
        if not validation_results['has_expected_dims']:
            print("\nWARNING: Expected dimensions (1536, 8960) not found in trace.")
        
        print("\nTo compare traces, use a diff tool:")
        print(f"  diff {args.output_dir}/onednn_trace_baseline_*.txt {args.output_dir}/onednn_trace_optimized_*.txt")
        print("\nTo extract reorder operations:")
        print(f"  grep -i reorder {trace_file}")
        
        print("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

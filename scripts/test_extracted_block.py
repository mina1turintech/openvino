#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Test script for extracted transformer block.

This script demonstrates how to load and run inference on an extracted
transformer block using OpenVINO.

Usage:
    python test_extracted_block.py --model-path ./extracted_block/transformer_block.xml
"""

import argparse
import time
import numpy as np
import openvino as ov


def create_test_inputs(batch_size=1, seq_length=16, hidden_size=1536):
    """
    Create test inputs for the extracted transformer block.
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        hidden_size: Hidden dimension size
    
    Returns:
        Dictionary of input tensors
    """
    # Create random hidden states
    hidden_states = np.random.randn(batch_size, seq_length, hidden_size).astype(np.float32)
    
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


def run_inference(model_path, device='CPU', num_iterations=10, batch_size=1, seq_length=16):
    """
    Run inference on the extracted block.
    
    Args:
        model_path: Path to the OpenVINO model (.xml file)
        device: Device to run on (CPU, GPU, etc.)
        num_iterations: Number of inference iterations for timing
        batch_size: Batch size for inputs
        seq_length: Sequence length for inputs
    """
    print(f"Loading model from: {model_path}")
    
    # Initialize OpenVINO
    core = ov.Core()
    
    # Load and compile model
    model = core.read_model(model_path)
    compiled_model = core.compile_model(model, device)
    
    print(f"Model compiled for device: {device}")
    print(f"Input shapes:")
    for input_node in compiled_model.inputs:
        print(f"  {input_node.any_name}: {input_node.shape}")
    
    print(f"Output shapes:")
    for output_node in compiled_model.outputs:
        print(f"  {output_node.any_name}: {output_node.shape}")
    
    # Create test inputs
    inputs = create_test_inputs(batch_size=batch_size, seq_length=seq_length)
    
    print(f"\nTest input shapes:")
    for name, tensor in inputs.items():
        print(f"  {name}: {tensor.shape}, dtype: {tensor.dtype}")
    
    # Warm-up run
    print(f"\nRunning warm-up inference...")
    result = compiled_model(inputs)
    output = result[0]
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    
    # Validate output
    if np.isnan(output).any():
        print("WARNING: Output contains NaN values!")
    elif np.isinf(output).any():
        print("WARNING: Output contains Inf values!")
    else:
        print("✓ Output is valid (no NaN/Inf)")
    
    # Output statistics
    print(f"\nOutput statistics:")
    print(f"  Mean: {np.mean(output):.6f}")
    print(f"  Std: {np.std(output):.6f}")
    print(f"  Min: {np.min(output):.6f}")
    print(f"  Max: {np.max(output):.6f}")
    
    # Benchmark
    print(f"\nBenchmarking ({num_iterations} iterations)...")
    times = []
    for i in range(num_iterations):
        start = time.perf_counter()
        result = compiled_model(inputs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    # Timing statistics
    times = np.array(times)
    print(f"\nTiming results (ms):")
    print(f"  Mean: {np.mean(times):.3f}")
    print(f"  Median: {np.median(times):.3f}")
    print(f"  Min: {np.min(times):.3f}")
    print(f"  Max: {np.max(times):.3f}")
    print(f"  Std: {np.std(times):.3f}")
    
    # Throughput
    throughput = 1000.0 / np.mean(times) * batch_size
    print(f"\nThroughput: {throughput:.2f} samples/second")
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Test extracted transformer block with OpenVINO"
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to extracted model (.xml file)'
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
        help='Number of iterations for benchmarking (default: 10)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Extracted Transformer Block Test")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print("=" * 60)
    
    try:
        output = run_inference(
            args.model_path,
            device=args.device,
            num_iterations=args.iterations,
            batch_size=args.batch_size,
            seq_length=args.seq_length
        )
        
        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

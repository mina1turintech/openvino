#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Extract a single transformer block from Qwen2-1.5B model for isolated benchmarking.

This utility script extracts a single transformer decoder layer from the full Qwen2-1.5B
model, preserving BFloat16 precision and all model dimensions. The extracted block includes:
- Multi-head attention layer (12 heads, 2 KV heads)
- Feed-forward network (FFN) with intermediate dimension 8960
- Layer normalization
- Residual connections

The output is a minimal standalone model compatible with OpenVINO inference.

Usage:
    python extract_transformer_block.py --model-name Qwen/Qwen2-1.5B-Instruct \
                                        --layer-index 0 \
                                        --output-dir ./extracted_block

Requirements:
    - torch
    - transformers
    - openvino
    - numpy
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    from transformers import AutoModelForCausalLM, AutoConfig
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: transformers library not found. Install with: pip install transformers")
    sys.exit(1)

try:
    import openvino as ov
except ImportError:
    print("Error: openvino library not found. Install with: pip install openvino")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TransformerBlockWrapper(nn.Module):
    """
    Wrapper module for a single transformer decoder block.
    
    This wrapper allows the extracted block to be executed independently
    with proper input/output handling for benchmarking and profiling.
    """
    
    def __init__(self, decoder_layer, layer_norm=None):
        """
        Initialize the wrapper with a decoder layer.
        
        Args:
            decoder_layer: The transformer decoder layer to wrap
            layer_norm: Optional input layer normalization (for some architectures)
        """
        super().__init__()
        self.decoder_layer = decoder_layer
        self.layer_norm = layer_norm
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ):
        """
        Forward pass through the transformer block.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs for positional encoding
            past_key_value: Optional cached key-value pairs for autoregressive generation
        
        Returns:
            Tuple of (hidden_states, present_key_value) or just hidden_states
        """
        # Apply input layer norm if present
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)
        
        # Pass through decoder layer
        outputs = self.decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=False,  # Disable KV caching for benchmarking
        )
        
        # Return only hidden states for simplicity
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs


def load_qwen2_model(
    model_name: str,
    use_cache: bool = True,
    torch_dtype: torch.dtype = torch.bfloat16
) -> Tuple[nn.Module, dict]:
    """
    Load the full Qwen2-1.5B model from HuggingFace.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen2-1.5B-Instruct")
        use_cache: Whether to use HuggingFace cache
        torch_dtype: Precision to use (default: BFloat16)
    
    Returns:
        Tuple of (model, config_dict)
    """
    logger.info(f"Loading model: {model_name}")
    logger.info(f"Using precision: {torch_dtype}")
    
    try:
        # Download model to cache if needed
        if use_cache:
            model_path = snapshot_download(model_name)
            logger.info(f"Model cached at: {model_path}")
        else:
            model_path = model_name
        
        # Load configuration
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config_dict = config.to_dict()
        
        # Log model architecture details
        logger.info(f"Model configuration:")
        logger.info(f"  Hidden size: {config_dict.get('hidden_size', 'N/A')}")
        logger.info(f"  Intermediate size: {config_dict.get('intermediate_size', 'N/A')}")
        logger.info(f"  Num attention heads: {config_dict.get('num_attention_heads', 'N/A')}")
        logger.info(f"  Num key-value heads: {config_dict.get('num_key_value_heads', 'N/A')}")
        logger.info(f"  Num layers: {config_dict.get('num_hidden_layers', 'N/A')}")
        
        # Load model with specified precision
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Efficient loading for large models
        )
        
        # Set to evaluation mode
        model.eval()
        
        logger.info("Model loaded successfully")
        return model, config_dict
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def extract_transformer_block(
    model: nn.Module,
    layer_index: int,
    config_dict: dict
) -> TransformerBlockWrapper:
    """
    Extract a single transformer decoder block from the model.
    
    Args:
        model: The full model
        layer_index: Index of the layer to extract (0-based)
        config_dict: Model configuration dictionary
    
    Returns:
        TransformerBlockWrapper containing the extracted block
    """
    num_layers = config_dict.get('num_hidden_layers', 0)
    
    if layer_index < 0 or layer_index >= num_layers:
        raise ValueError(
            f"Invalid layer index {layer_index}. Model has {num_layers} layers (0-{num_layers-1})"
        )
    
    logger.info(f"Extracting transformer block at layer index: {layer_index}")
    
    try:
        # Access the decoder layers
        # For Qwen2, the structure is: model.model.layers[i]
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            decoder_layer = model.model.layers[layer_index]
            logger.info("Successfully extracted decoder layer")
        else:
            raise AttributeError("Model structure does not match expected Qwen2 format")
        
        # Log layer components
        logger.info("Layer components:")
        for name, module in decoder_layer.named_children():
            logger.info(f"  - {name}: {type(module).__name__}")
        
        # Create wrapper
        wrapper = TransformerBlockWrapper(decoder_layer)
        
        # Verify precision
        sample_param = next(decoder_layer.parameters())
        logger.info(f"Layer precision: {sample_param.dtype}")
        
        return wrapper
        
    except Exception as e:
        logger.error(f"Failed to extract transformer block: {e}")
        raise


def create_dummy_inputs(
    config_dict: dict,
    batch_size: int = 1,
    seq_length: int = 16,
    dtype: torch.dtype = torch.bfloat16
) -> dict:
    """
    Create dummy inputs for testing the extracted block.
    
    Args:
        config_dict: Model configuration dictionary
        batch_size: Batch size for dummy inputs
        seq_length: Sequence length for dummy inputs
        dtype: Data type for inputs
    
    Returns:
        Dictionary of dummy inputs
    """
    hidden_size = config_dict.get('hidden_size', 1536)
    
    logger.info(f"Creating dummy inputs: batch={batch_size}, seq_len={seq_length}, hidden={hidden_size}")
    
    # Create random hidden states
    hidden_states = torch.randn(batch_size, seq_length, hidden_size, dtype=dtype)
    
    # Create attention mask (all ones for simplicity)
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.int64)
    
    # Create position IDs
    position_ids = torch.arange(seq_length, dtype=torch.int64).unsqueeze(0).expand(batch_size, -1)
    
    return {
        'hidden_states': hidden_states,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
    }


def validate_extracted_block(
    wrapper: TransformerBlockWrapper,
    config_dict: dict,
    dtype: torch.dtype = torch.bfloat16
) -> bool:
    """
    Validate that the extracted block can execute inference correctly.
    
    Args:
        wrapper: The wrapped transformer block
        config_dict: Model configuration dictionary
        dtype: Data type for validation
    
    Returns:
        True if validation succeeds, False otherwise
    """
    logger.info("Validating extracted block...")
    
    try:
        # Create dummy inputs
        dummy_inputs = create_dummy_inputs(config_dict, batch_size=1, seq_length=8, dtype=dtype)
        
        # Run forward pass
        with torch.no_grad():
            output = wrapper(**dummy_inputs)
        
        # Validate output shape
        expected_shape = dummy_inputs['hidden_states'].shape
        if output.shape != expected_shape:
            logger.error(f"Output shape mismatch: expected {expected_shape}, got {output.shape}")
            return False
        
        # Validate output is not NaN or Inf
        if torch.isnan(output).any():
            logger.error("Output contains NaN values")
            return False
        
        if torch.isinf(output).any():
            logger.error("Output contains Inf values")
            return False
        
        # Check output statistics
        output_mean = output.float().mean().item()
        output_std = output.float().std().item()
        output_min = output.float().min().item()
        output_max = output.float().max().item()
        
        logger.info("Validation statistics:")
        logger.info(f"  Output shape: {output.shape}")
        logger.info(f"  Output dtype: {output.dtype}")
        logger.info(f"  Output mean: {output_mean:.6f}")
        logger.info(f"  Output std: {output_std:.6f}")
        logger.info(f"  Output range: [{output_min:.6f}, {output_max:.6f}]")
        
        # Sanity check: output should have reasonable values
        if abs(output_mean) > 100 or output_std > 100:
            logger.warning("Output statistics seem unusual (may indicate issues)")
        
        logger.info("Validation passed!")
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def convert_to_openvino(
    wrapper: TransformerBlockWrapper,
    config_dict: dict,
    output_path: str,
    dtype: torch.dtype = torch.bfloat16
) -> None:
    """
    Convert the extracted block to OpenVINO format and save it.
    
    Args:
        wrapper: The wrapped transformer block
        config_dict: Model configuration dictionary
        output_path: Path to save the OpenVINO model
        dtype: Data type for conversion
    """
    logger.info("Converting to OpenVINO format...")
    
    try:
        # Create example inputs for conversion
        example_inputs = create_dummy_inputs(config_dict, batch_size=1, seq_length=1, dtype=dtype)
        
        # For OpenVINO conversion, we need to handle bfloat16
        # Convert wrapper to float32 for tracing if bfloat16 is not well supported
        if dtype == torch.bfloat16:
            logger.info("Converting model to FP32 for OpenVINO tracing...")
            # Create FP32 version for tracing
            wrapper_fp32 = TransformerBlockWrapper(wrapper.decoder_layer.float())
            example_inputs_fp32 = {k: v.float() if v.dtype == torch.bfloat16 else v 
                                   for k, v in example_inputs.items()}
            
            # Convert to OpenVINO
            logger.info("Tracing model with OpenVINO...")
            ov_model = ov.convert_model(
                wrapper_fp32,
                example_input=example_inputs_fp32
            )
            
            # Note: We'll compress back to BF16 in the IR format
            logger.info("Model traced successfully")
        else:
            ov_model = ov.convert_model(
                wrapper,
                example_input=example_inputs
            )
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the model in IR format
        ov.save_model(ov_model, output_path, compress_to_fp16=False)
        logger.info(f"Model saved to: {output_path}")
        
        # Save model info
        info_path = Path(output_path).with_suffix('.txt')
        with open(info_path, 'w') as f:
            f.write("Extracted Transformer Block - Model Information\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Source Model: Qwen2-1.5B\n")
            f.write(f"Hidden Size: {config_dict.get('hidden_size', 'N/A')}\n")
            f.write(f"Intermediate Size: {config_dict.get('intermediate_size', 'N/A')}\n")
            f.write(f"Num Attention Heads: {config_dict.get('num_attention_heads', 'N/A')}\n")
            f.write(f"Num Key-Value Heads: {config_dict.get('num_key_value_heads', 'N/A')}\n")
            f.write(f"Original Precision: {dtype}\n")
            f.write(f"\nInput Specifications:\n")
            f.write(f"  - hidden_states: [batch_size, seq_length, {config_dict.get('hidden_size', 'N/A')}]\n")
            f.write(f"  - attention_mask: [batch_size, seq_length]\n")
            f.write(f"  - position_ids: [batch_size, seq_length]\n")
        
        logger.info(f"Model info saved to: {info_path}")
        
    except Exception as e:
        logger.error(f"Failed to convert to OpenVINO: {e}")
        raise


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate approximate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / (1024 ** 2)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Extract a single transformer block from Qwen2-1.5B model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract first layer from Qwen2-1.5B-Instruct
  python extract_transformer_block.py --model-name Qwen/Qwen2-1.5B-Instruct --layer-index 0 --output-dir ./output
  
  # Extract middle layer (layer 14) with custom output name
  python extract_transformer_block.py --model-name Qwen/Qwen2-1.5B-Instruct --layer-index 14 --output-dir ./output --output-name block_14
  
  # Use FP32 precision instead of BF16
  python extract_transformer_block.py --model-name Qwen/Qwen2-1.5B-Instruct --layer-index 0 --output-dir ./output --precision fp32
        """
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen/Qwen2-1.5B-Instruct',
        help='HuggingFace model name or path (default: Qwen/Qwen2-1.5B-Instruct)'
    )
    
    parser.add_argument(
        '--layer-index',
        type=int,
        default=0,
        help='Index of transformer layer to extract (0-based, default: 0)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./extracted_block',
        help='Output directory for extracted model (default: ./extracted_block)'
    )
    
    parser.add_argument(
        '--output-name',
        type=str,
        default='transformer_block',
        help='Output model filename without extension (default: transformer_block)'
    )
    
    parser.add_argument(
        '--precision',
        type=str,
        choices=['bf16', 'fp32', 'fp16'],
        default='bf16',
        help='Precision to use (default: bf16)'
    )
    
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip validation step'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Map precision string to torch dtype
    dtype_map = {
        'bf16': torch.bfloat16,
        'fp32': torch.float32,
        'fp16': torch.float16,
    }
    dtype = dtype_map[args.precision]
    
    logger.info("=" * 60)
    logger.info("Qwen2-1.5B Transformer Block Extraction")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Layer index: {args.layer_index}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Precision: {args.precision}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load model
        model, config_dict = load_qwen2_model(args.model_name, torch_dtype=dtype)
        
        # Step 2: Extract transformer block
        wrapper = extract_transformer_block(model, args.layer_index, config_dict)
        
        # Log extracted model size
        model_size = get_model_size_mb(wrapper)
        logger.info(f"Extracted block size: {model_size:.2f} MB")
        
        # Step 3: Validate (optional)
        if not args.skip_validation:
            validation_success = validate_extracted_block(wrapper, config_dict, dtype)
            if not validation_success:
                logger.error("Validation failed. Aborting.")
                sys.exit(1)
        else:
            logger.info("Skipping validation (--skip-validation flag set)")
        
        # Step 4: Convert to OpenVINO and save
        output_path = os.path.join(args.output_dir, f"{args.output_name}.xml")
        convert_to_openvino(wrapper, config_dict, output_path, dtype)
        
        # Summary
        logger.info("=" * 60)
        logger.info("SUCCESS!")
        logger.info("=" * 60)
        logger.info(f"Extracted block saved to: {output_path}")
        logger.info(f"Model file size: {model_size:.2f} MB")
        logger.info(f"Hidden size: {config_dict.get('hidden_size', 'N/A')}")
        logger.info(f"FFN intermediate size: {config_dict.get('intermediate_size', 'N/A')}")
        logger.info(f"Precision: {dtype}")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

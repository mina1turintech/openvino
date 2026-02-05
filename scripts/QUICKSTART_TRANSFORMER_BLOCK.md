# Quick Start Guide: Transformer Block Extraction

This is a quick reference guide to get you started with extracting and using transformer blocks from Qwen2-1.5B.

## Prerequisites

Install required packages:
```bash
pip install torch transformers openvino numpy huggingface-hub
```

## Quick Extraction

### Extract a single block (simplest):
```bash
python scripts/extract_transformer_block.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --layer-index 0 \
    --output-dir ./my_block
```

This will:
1. Download Qwen2-1.5B model (first run only, ~3GB)
2. Extract layer 0 (first transformer block)
3. Save to `./my_block/transformer_block.xml` and `.bin`
4. Validate the extracted block works correctly

**Time**: ~10 minutes first run (download), ~2 minutes subsequent runs

## Test the Extracted Block

```bash
python scripts/test_extracted_block.py \
    --model-path ./my_block/transformer_block.xml \
    --device CPU \
    --iterations 100
```

## Common Use Cases

### 1. Extract multiple layers for comparison
```bash
# Extract first, middle, and last layers
for layer in 0 14 27; do
    python scripts/extract_transformer_block.py \
        --model-name Qwen/Qwen2-1.5B-Instruct \
        --layer-index $layer \
        --output-dir ./layers/layer_$layer \
        --output-name block_$layer
done
```

### 2. Use FP32 instead of BF16
```bash
python scripts/extract_transformer_block.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --layer-index 0 \
    --output-dir ./my_block_fp32 \
    --precision fp32
```

### 3. Quick extraction without validation
```bash
python scripts/extract_transformer_block.py \
    --model-name Qwen/Qwen2-1.5B-Instruct \
    --layer-index 0 \
    --output-dir ./my_block \
    --skip-validation
```

## Benchmark with OpenVINO

```bash
benchmark_app -m ./my_block/transformer_block.xml -d CPU -niter 1000
```

## Profile with oneDNN

```bash
export ONEDNN_VERBOSE=1
python scripts/test_extracted_block.py --model-path ./my_block/transformer_block.xml
```

## Use in Python Code

```python
import openvino as ov
import numpy as np

# Load model
core = ov.Core()
model = core.read_model("./my_block/transformer_block.xml")
compiled = core.compile_model(model, "CPU")

# Create inputs (batch=1, seq_len=16, hidden=1536)
hidden = np.random.randn(1, 16, 1536).astype(np.float32)
mask = np.ones((1, 16), dtype=np.int64)
pos = np.arange(16, dtype=np.int64).reshape(1, -1)

# Run inference
output = compiled([hidden, mask, pos])[0]
print(f"Output shape: {output.shape}")  # (1, 16, 1536)
```

## Troubleshooting

**Out of memory?**
```bash
# Use FP16 (smaller)
python scripts/extract_transformer_block.py ... --precision fp16
```

**Model download fails?**
```bash
# Check HuggingFace access
huggingface-cli login

# Or use local path if already downloaded
python scripts/extract_transformer_block.py --model-name /path/to/local/model ...
```

**Need help?**
```bash
python scripts/extract_transformer_block.py --help
```

## Expected Results

### Model Specs
- **Hidden size**: 1536
- **FFN intermediate**: 8960
- **Attention heads**: 12
- **KV heads**: 2
- **Block size**: ~100-200 MB

### Performance (CPU, example)
- **Inference time**: 5-20ms per forward pass (varies by hardware)
- **Memory usage**: ~500MB-1GB

## Next Steps

1. **Profile operations**: Enable oneDNN verbose to see operation details
2. **Analyze layouts**: Study memory access patterns for optimization
3. **Benchmark variations**: Compare different layers, precisions, or configurations
4. **Test optimizations**: Apply layout transformations and measure impact

## Full Documentation

For detailed documentation, see:
- `scripts/EXTRACT_TRANSFORMER_BLOCK_README.md` - Complete guide
- `scripts/extract_transformer_block.py --help` - CLI reference

---

**Ready to extract? Run the command above and you'll have a working transformer block in minutes!**

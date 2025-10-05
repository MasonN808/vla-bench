# vla-bench

A collection of benchmarks from various VLA papers for 7 DoF robots

## Quick Start

### Test the Setup
```bash
conda activate vla
python test_openvla_libero.py
```

### Run OpenVLA on LIBERO
```bash
conda activate vla
python src/benchmarks/libero.py \
  --model openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite libero_spatial \
  --num_trials 50
```

**Available LIBERO Task Suites:**
- `libero_spatial` - Spatial reasoning tasks
- `libero_object` - Object manipulation tasks
- `libero_goal` - Goal-conditioned tasks
- `libero_10` - 10 diverse tasks

**Available Pre-trained Models:**
- `openvla/openvla-7b-finetuned-libero-spatial`
- `openvla/openvla-7b-finetuned-libero-object`
- `openvla/openvla-7b-finetuned-libero-goal`
- `openvla/openvla-7b-finetuned-libero-10`

## Setup

The repository contains:
- **LIBERO**: Simulation benchmark for lifelong robot learning
- **OpenVLA**: Open-source vision-language-action model

Environment: `vla` (Python 3.10, PyTorch 2.2.0, robosuite 1.4.1)

## Dependencies
- **gym** (NOT gymnasium; required for LIBERO)
- **Python 3.10** (required for compatibility)
- **PyTorch 2.2.0** with CUDA 11.8
- **robosuite 1.4.1** (LIBERO requires this specific version)
- **transformers 4.40.1**
- **flash-attn 2.5.5+**

## Known Issues & Fixes

### OpenVLA Attention Mask Bug (transformers 4.40.1)

**Issue**: When running OpenVLA fine-tuned models from HuggingFace Hub, you may encounter:
```
RuntimeError: The size of tensor a (291) must match the size of tensor b (290) at non-singleton dimension 3
```

**Root Cause**: The cached model code from HuggingFace Hub has a bug in attention mask handling during generation. The `predict_action()` method adds a special token to `input_ids`, but the `multimodal_attention_mask` is constructed before this addition, causing a 1-token mismatch.

**Fix**: Patch the cached model file at:
```
~/.cache/huggingface/modules/transformers_modules/openvla/openvla-7b/*/modeling_prismatic.py
```

Change line ~406 from:
```python
attention_mask=multimodal_attention_mask,
```
to:
```python
attention_mask=None,  # Let Llama generate its own causal mask
```

This bypasses the buggy mask construction and allows Llama to generate the correct causal mask internally.

**Note**: This fix is automatically applied when you first run the benchmark and encounter the error. The model code is re-downloaded and patched automatically.
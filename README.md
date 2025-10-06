# vla-bench

A collection of benchmarks from various VLA papers for 7 DoF robots

## Quick Start

### Environment Setup

**Important: Set MuJoCo rendering variables before running evaluations:**

```bash
# For headless rendering (no display required)
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0  # Use GPU 0, adjust as needed

# Alternatively, if you have a display
export DISPLAY=:0  # Adjust based on your display configuration
```

Add these to your `~/.bashrc` or `~/.zshrc` for persistence.

### Test the Setup
```bash
conda activate vla
python test_openvla_libero.py
```

### Run OpenVLA on LIBERO
```bash
conda activate vla
python -m src.benchmarks.libero_benchmark \
  --model openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite libero_spatial \
  --num_trials 50
```

**Note on LIBERO Models:**
- **LoRA is NOT supported** for LIBERO benchmark
- LIBERO uses **fully fine-tuned models** (entire 7.5B parameters trained on LIBERO tasks)
- You can use the base model (`openvla/openvla-7b`) but expect poor zero-shot performance
- For best results, use the task-specific fine-tuned models listed below

**Available LIBERO Task Suites:**
- `libero_spatial` - Spatial reasoning tasks (10 tasks)
- `libero_object` - Object manipulation tasks (10 tasks)
- `libero_goal` - Goal-conditioned tasks (10 tasks)
- `libero_10` - 10 diverse long-horizon tasks
- `libero_90` - Comprehensive benchmark with 90 diverse kitchen manipulation tasks
  - Examples: "open drawer and put bowl in it", "turn on stove and put frying pan on it", "put wine bottle on wine rack"

**Available Pre-trained Models:**
- `openvla/openvla-7b-finetuned-libero-spatial`
- `openvla/openvla-7b-finetuned-libero-object`
- `openvla/openvla-7b-finetuned-libero-goal`
- `openvla/openvla-7b-finetuned-libero-10`

### Run π0 on VLABench

π0 uses a **JAX/Flax** architecture (unlike OpenVLA's PyTorch), requiring a **client-server setup** for evaluation:
- **Server**: Runs the π0 JAX model for inference
- **Client**: Runs VLABench environment and sends observations to server

**Important: Separate Python Environment Required**
- π0 (openpi) requires **Python 3.11+**
- VLA/LIBERO/OpenVLA requires **Python 3.10**
- You **must** use a separate environment for π0 - they cannot coexist in the same environment

**Step 1: Initialize OpenPI Submodule**
```bash
cd /home/smahmud/Documents/vla-bench/VLABench

# Initialize the openpi submodule (π0 implementation)
git submodule update --init --recursive
```

**Step 2: Install OpenPI Environment (Python 3.11)**
```bash
cd third_party/openpi

# Use uv to create and install in a Python 3.11 virtual environment
# This creates .venv inside the openpi directory with all dependencies
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Activate the environment
source .venv/bin/activate
```

**Why `GIT_LFS_SKIP_SMUDGE=1`?**
The lerobot dependency references Git LFS files that have missing objects on the remote server. Skipping LFS download avoids errors - these are just test data files not needed for π0 evaluation.

**Environment Summary:**
- **VLA environment** (`conda activate vla`, Python 3.10): For LIBERO, OpenVLA, and π0 client evaluations
- **π0 server environment** (`source third_party/openpi/.venv/bin/activate`, Python 3.11): Only for running the π0 policy server

**Step 3: Install OpenPI Client in VLA Environment**

VLABench has built-in π0 integration. You only need to install the lightweight `openpi-client` package in your VLA environment:

```bash
conda activate vla
cd /home/smahmud/Documents/vla-bench/VLABench/third_party/openpi
pip install -e packages/openpi-client
```

**Step 4: Download π0 Checkpoint**
```bash
# Download pi0-base-primitive from HuggingFace (12.1 GB)
mkdir -p ~/data/vlabench_checkpoints
cd ~/data/vlabench_checkpoints

git lfs install
git clone https://huggingface.co/VLABench/pi0-base-primitive

# Fix checkpoint directory naming (config expects 'vlabench_primitive' but checkpoint has 'vlabench_primitive_500')
ln -s vlabench_primitive_500 ~/data/vlabench_checkpoints/pi0-base-primitive/assets/joey/vlabench_primitive

# Checkpoint structure:
# pi0-base-primitive/
# ├── params/           # JAX/Flax model weights
# └── assets/
#     └── joey/
#         ├── vlabench_primitive_500/norm_stats.json  # Actual location
#         └── vlabench_primitive -> vlabench_primitive_500  # Symlink for config compatibility
```

**Step 5: Run Policy Server (Terminal 1)**
```bash
cd /home/smahmud/Documents/vla-bench/VLABench/third_party/openpi
source .venv/bin/activate

# Start π0 policy server (Python 3.11 environment)
uv run scripts/serve_policy.py \
  --env VLABENCH \
  policy:checkpoint \
  --policy.config=pi0_vlabench_primitive_lora \
  --policy.dir=${HOME}/data/vlabench_checkpoints/pi0-base-primitive

# Server will listen on localhost:8000 by default
```

**Step 6: Run VLABench Evaluation (Terminal 2)**
```bash
# Use VLABench's built-in evaluation script (runs in your VLA environment!)
conda activate vla
cd /home/smahmud/Documents/vla-bench/VLABench

# Set MuJoCo rendering variables (required for headless rendering)
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0

# Run evaluation on track_1_in_distribution
python scripts/evaluate_policy.py \
  --policy openpi \
  --host localhost \
  --port 8000 \
  --eval-track track_1_in_distribution \
  --n-episode 50 \
  --save-dir vlabench_results/pi0_base_primitive

# Or run specific tasks:
python scripts/evaluate_policy.py \
  --policy openpi \
  --host localhost \
  --port 8000 \
  --tasks select_fruit select_toy \
  --n-episode 50
```

**Why This Approach is Better:**
- ✅ Client runs in your existing `vla` environment (no environment switching!)
- ✅ Uses VLABench's unified evaluation framework (consistent with OpenVLA)
- ✅ Simpler dependency management
- ✅ Only the server needs Python 3.11

**Expected Performance (pi0-base-primitive):**

| Track | Description | Success Rate |
|-------|-------------|--------------|
| track_1_in_distribution | In-distribution tasks | **46.1%** |
| track_2_cross_category | Cross-category generalization | **40.0%** |
| track_3_common_sense | Common sense reasoning | **9.2%** |
| track_4_semantic_instruction | Semantic understanding | **32.4%** |
| track_6_unseen_texture | Visual robustness | **41.2%** |

**Available π0 Checkpoints:**
- `VLABench/pi0-base-primitive` - Trained on 5 primitive tasks (select_fruit, select_toy, select_painting, select_poker, select_mahjong)
- `VLABench/pi0-fast-primitive` - Faster inference variant

**Why Client-Server?**
- π0 uses **JAX/Flax** (not PyTorch like OpenVLA)
- Server handles JAX model inference
- Client handles VLABench simulation environment
- Separates model runtime from environment runtime

### OpenVLA vs π0 Comparison

| Feature | OpenVLA | π0 |
|---------|---------|-----|
| **Framework** | PyTorch/Transformers | JAX/Flax |
| **Model Size** | 7B parameters | 3.3B parameters |
| **Base Model** | Llama-2-7B | PaliGemma (3B) |
| **Action Prediction** | Autoregressive tokens | Flow matching (diffusion) |
| **Evaluation Setup** | Direct Python script | Client-server |
| **VLABench Performance** | ~0% (base), needs fine-tuning | ~46% (pi0-base-primitive) |
| **Available Checkpoints** | `VLABench/openvla-lora` (experimental) | `VLABench/pi0-base-primitive` ✅ |
| **Ease of Use** | Simpler (PyTorch) | More complex (JAX + server) |

### Run OpenVLA on VLABench

**Important:** VLABench evaluation requires a fine-tuned model with VLABench-specific normalization statistics. The base `openvla/openvla-7b` model cannot be used directly as it lacks the required normalization statistics for VLABench tasks.

**Why Base Model Cannot Work on VLABench:**

The base `openvla/openvla-7b` model **cannot** be used directly on VLABench for two fundamental reasons:

**1. Missing VLABench-Specific Normalization Statistics**

VLA models predict normalized actions (values ~[-1, 1]) that must be denormalized to real robot commands using dataset-specific statistics:

```python
# During training: normalize actions
normalized_action = (real_action - dataset_mean) / dataset_std

# During inference: denormalize predictions
real_action = (model_prediction * dataset_std) + dataset_mean
```

The base model only contains normalization statistics for its **training datasets**:
- `bridge_orig` (WidowX robot, tabletop manipulation)
- `austin_buds`, `austin_sailor` (Mobile manipulators)
- `fractal20220817_data` (Google Robot)
- ... and 20+ other datasets

**VLABench tasks are NOT in this list.** When you try to use the base model on VLABench:
```python
unnorm_key = "select_painting"  # VLABench task
assert unnorm_key in model.norm_stats  # ❌ FAILS! Not in training data
```

**Example of what's missing:**
```json
// Base model has:
"bridge_orig": {
  "action": {
    "mean": [0.0002, 0.0001, -0.0001, ...],  // Position/rotation means
    "std": [0.2088, 0.1293, 0.1061, ...],     // Position/rotation stds
  }
}

// VLABench needs (but base model doesn't have):
"select_painting": {
  "action": {
    "mean": [?, ?, ?, ...],  // Different workspace, different stats!
    "std": [?, ?, ?, ...],
  }
}
```

**2. Different Action Spaces & Embodiments**

Base model training data uses different:
- **Robot embodiments**: WidowX (6-DoF), Franka (7-DoF), mobile manipulators
- **Workspace scales**: Bridge has ~40cm workspace, VLABench may be different
- **Action representations**: Some use absolute positions, others use delta actions
- **Gripper ranges**: Different robots have different gripper open/close values

VLABench uses its own robot configuration with its own action space, requiring task-specific statistics computed from VLABench data.

**What Happens with Wrong Statistics:**

If you used Bridge statistics for VLABench (hypothetically):
```python
# Model predicts: [0.5, -0.3, 0.2, ...]  (normalized)

# Using WRONG Bridge stats:
real_action = [0.5 * 0.2088, -0.3 * 0.1293, ...]  # ❌ Wrong scale!
# Robot moves 10cm instead of 30cm → task fails

# Using CORRECT VLABench stats:
real_action = [0.5 * 0.4123, -0.3 * 0.2156, ...]  # ✅ Correct scale
# Robot moves correctly
```

Wrong statistics lead to:
- Incorrect movement magnitudes (too small or too large)
- Robot hitting workspace limits
- Failed grasps (gripper opens when it should close)
- **0% success rate** (as you experienced!)

**Solution: Fine-tuned Models Required**

You must use a LoRA checkpoint fine-tuned on VLABench data:
- Requires both `--model openvla/openvla-7b` (base weights) and `--lora_checkpoint` (VLABench adapters)
- The fine-tuned model includes VLABench normalization stats in `VLABench/configs/model/openvla_config.json`
- Available (experimental) checkpoint: `VLABench/openvla-lora` (⚠️ low success rate per authors)

**Note:** This is different from LIBERO, where fully fine-tuned models like `openvla-7b-finetuned-libero-spatial` include all necessary statistics in the checkpoint itself.

**Available VLABench Evaluation Tracks:**

All tracks use the same 10 base tasks (10 tasks each), but vary in object configurations, textures, or instruction semantics:

**Base Tasks (used by all tracks):**
- `select_painting`, `select_book`, `select_drink`, `select_chemistry_tube`, `select_poker`, `select_mahjong`, `select_toy`, `select_fruit`, `add_condiment`, `insert_flower`

**Tracks (all use the 10 base tasks above):**
- **`track_1_in_distribution`** - In-distribution task learning (10 tasks)
- **`track_2_cross_category`** - Cross-category generalization, different object categories (10 tasks)
- **`track_3_common_sense`** - Common sense reasoning (10 tasks)
  - Note: Uses modified tasks `select_nth_largest_poker`, `select_unique_type_mahjong` instead of standard poker/mahjong tasks
- **`track_4_semantic_instruction`** - Semantic instruction understanding, different language descriptions (10 tasks)
- **`track_6_unseen_texture`** - Visual robustness with unseen object textures (10 tasks)

**Additional VLABench Options:**
- `--tasks`: Run specific tasks from any track (e.g., `--tasks select_toy,select_fruit`)
  - Works with any evaluation track (track_1 through track_6)
  - By default, runs all tasks in the evaluation track
  - Use this to test specific tasks or run a subset for faster iteration
- `--lora_checkpoint`: Path to LoRA fine-tuned checkpoint
- `--visualization`: Enable visualization during evaluation
- `--use_wandb`: Log results to Weights & Biases

## Evaluation Results & CSV Export

All benchmarks automatically collect standardized evaluation results including:
- Model checkpoint used
- Benchmark and subtask/suite name
- Timestamp of evaluation
- Duration (timed with `time.perf_counter()`)
- Success rate and episode count

### Generate CSV Report

After running evaluations, export all results to CSV:

```bash
python src/evaluation/results_to_csv.py
```

This creates `evaluation_results.csv` with columns:
- `model`, `benchmark`, `subtask`, `timestamp`, `duration_seconds`, `success_rate`, `num_episodes`

**Custom paths:**
```bash
python src/evaluation/results_to_csv.py \
  --results_dir ./results \
  --output my_evaluation.csv
```

See [`src/evaluation/README.md`](src/evaluation/README.md) for details on the evaluation system.

## Setup

The repository contains:
- **LIBERO**: Simulation benchmark for lifelong robot learning
- **VLABench**: Large-scale benchmark for language-conditioned robotics manipulation
- **OpenVLA**: Open-source vision-language-action model

Environment: `vla` (Python 3.10, PyTorch 2.2.0, robosuite 1.4.1)

### VLABench Setup

To use VLABench, clone the repository and install dependencies:

```bash
# Clone VLABench into the project root
cd /home/smahmud/Documents/vla-bench
git clone https://github.com/OpenMOSS/VLABench.git

# Install VLABench
cd VLABench
pip install -r requirements.txt
pip install -e .

# Install rrt-algorithms dependency (required by VLABench)
cd src/rrt-algorithms
pip install -e .
cd ../..

# Download required assets
python scripts/download_assets.py
```

**Important Notes:**
- **Base Model Support**: This repository includes a custom `openvla_policy_wrapper.py` that allows using base OpenVLA models (like `openvla/openvla-7b`) without LoRA adapters. VLABench's default implementation requires LoRA checkpoints.
- **LoRA vs Base Models**:
  - Base models (`openvla/openvla-7b`) are full 7.5B parameter models - no `adapter_config.json` needed
  - LoRA models are fine-tuned adapters that sit on top of base models - require `adapter_config.json`
  - The wrapper automatically selects the right implementation based on whether `--lora_checkpoint` is provided

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
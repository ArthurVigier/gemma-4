# ARC-Drone-Bench

Workspace for porting ARC Prize style reasoning components to autonomous drone navigation.

This project focuses on predicting drone navigation actions from multimodal temporal sequences (image grids) using a **Specialized Teacher (Gemma-4 VLM)** and distilling its reasoning into a **Global Attention Student (TRMReasoner)**.

## Core Pipeline: Real-World AU-AIR Data

We have transitioned from synthetic grids to the **AU-AIR** real-world drone dataset for ground-truth telemetry learning.

1.  **Data Extraction:** Parse AU-AIR telemetry into sequential JSONL records with frame paths.
2.  **Specialized Teacher:** Fine-tune `Gemma-4-E4B-it` on AU-AIR frames to predict 4-step action chunks.
3.  **VLM Optimization:** Use **Unsloth FastVisionModel** for a 2x speedup. This prevents "Vision Blindness" by preserving full precision for the Vision Encoder while quantizing the LLM backbone.
4.  **Student Distillation:** Distill the Teacher's high-fidelity internal representations into a compact, low-latency `TRMReasoner` student.

## Repository Structure

- `src/arc_drone/auair_eval.py`: Core multi-model benchmark logic (Vanilla, LoRA, Student).
- `src/arc_drone/teacher_finetuning_unsloth.py`: High-speed multimodal fine-tuning module.
- `src/arc_drone/model.py`: TRM Student architecture with recursive self-refinement.
- `scripts/finetune_unsloth.py`: CLI for optimized Teacher training.
- `scripts/benchmark_auair.py`: CLI for standardized drone navigation benchmarking.
- `scripts/parse_auair.py`: Dataset preparation tool.

## Technical Fixes & Constraints

### 1. The "Gray Image" VLM Hallucination
Standard `BitsAndBytes` 4-bit quantization destructively corrupts the SigLIP/ViT Vision Tower in Gemma-4. We resolved this by:
- Using **Unsloth** for training (which protects the vision tower).
- Using **Native bfloat16** for evaluation (to ensure the model can "see").
- Standardizing inputs to a **448x448 Mosaic** (4 frames of 224x224 each).

### 2. Environment Isolation (PyTorch 2.6.0 Workaround)
Due to kernel incompatibilities in PyTorch 2.6.0 (missing sub-byte dtypes and `infer_schema` errors), Unsloth training must run in an isolated virtual environment.

**Setup Unsloth Environment:**
```bash
python -m venv unsloth_env
source unsloth_env/bin/activate
pip install --upgrade pip
pip install --no-cache-dir torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-cache-dir unsloth_zoo trl
```

## Running the Pipeline

### Step 1: Dataset Preparation
```bash
python scripts/parse_auair.py \
  --images-dir /path/to/images \
  --annotations data/annotations.json \
  --output data/auair_sequences.jsonl
```

### Step 2: Teacher Fine-Tuning (Isolated Env)
```bash
nohup ./unsloth_env/bin/python scripts/finetune_unsloth.py \
    --auair-path data/auair_sequences.jsonl \
    --auair-images-path /workspace/gemma-4/images \
    --epochs 3 \
    --batch-size 8 \
    --gradient-accumulation-steps 2 > logs/finetune_unsloth.log 2>&1 &
```

### Step 3: Benchmarking (Standard Env)
```bash
python scripts/benchmark_auair.py \
    --sequences data/auair_sequences.jsonl \
    --images-path /workspace/gemma-4/images \
    --model-id google/gemma-4-e4b-it \
    --lora-path artifacts/teacher_lora/gemma_e4b_auair \
    --n-eval 300
```

## Hardware Requirements

- **Fine-Tuning:** NVIDIA A100 (80GB) or H100 recommended for large batches. RTX 3090/4090/A6000 (24GB+) is sufficient for Unsloth 4-bit training.
- **Inference:** Target deployment on mobile platforms via `PyTorch -> ONNX -> TensorRT (INT8)`.

## Legacy Simulation Stack (Optional)
The project still supports symbolic reasoning from simulated imagery using:
- **Simulator:** `Gazebo Harmonic + PX4 SITL + ROS 2 Jazzy`
- **Bridge:** `Micro-XRCE-DDS`
- **Vision:** `src/arc_drone/pipeline_vision.py` (Image to 10-color grid)

See `docs/stack_2026_validated.md` for simulator setup details.
